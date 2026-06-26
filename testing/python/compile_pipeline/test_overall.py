import os

import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.layout import make_zz_layout

from compile_pipeline import compile_test, target
from formal_verify_funcs import *


@target("Sunmmio")
def kernel_overall(M, N, K, block_M, block_N, block_K, dtype="bfloat16", accum_dtype="float32"):
    shard_policy = T.MeshShardingPolicy(y=0, x=1)
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    A_shape = (M, K)
    B_shape = (K, N)
    C_shape = (M, N)
    A_layout = make_zz_layout(A_shape, [0, 1], (32, 32))
    B_layout = make_zz_layout(B_shape, [0, 1], (32, 32))
    C_layout = make_zz_layout(C_shape, [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor(A_shape, shard_policy, device_mesh_config, dtype, layout=A_layout),
        B: T.MeshTensor(B_shape, shard_policy, device_mesh_config, dtype, layout=B_layout),
        Bias: T.MeshTensor(C_shape, shard_policy, device_mesh_config, accum_dtype, layout=C_layout),
        C: T.MeshTensor(C_shape, shard_policy, device_mesh_config, accum_dtype, layout=C_layout),
    ):
        # Initialize Kernel Context
        with T.Kernel(ncores) as _cid:
            sharded_M, sharded_K = A.shape
            _, sharded_N = B.shape

            # [wanghz18] Automatic SRAM Scope Inference
            # We declare generic 'shared' scope, expecting InferSramScope pass to
            # refine them to 'shared.asram', 'shared.wsram', 'shared.rsram'
            A_shared = T.alloc_shared((block_M, block_K), dtype=dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype=dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            Bias_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            for bx in T.serial(T.ceildiv(sharded_N, block_N)):
                for by in T.serial(T.ceildiv(sharded_M, block_M)):
                    T.clear(C_shared)  # Avoid Fill op unsupported scope error

                    # [wanghz18] GEMM Lowering to mma_sunmmio intrinsic
                    for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=2):
                        T.copy(A[by * block_M, k * block_K], A_shared)
                        T.copy(B[k * block_K, bx * block_N], B_shared)
                        T.gemm(A_shared, B_shared, C_shared)

                    # Load Bias
                    T.copy(Bias[by * block_M, bx * block_N], Bias_shared)

                    # [weizzh] Tiles Loop for Element-wise operation
                    # This loop should be legalized and vectorized by LegalizeTilesLoop/TilesLoop passes
                    for i, j in T.Tiles(C_shared, parallel=True):
                        C_shared[i, j] = C_shared[i, j] + Bias_shared[i, j]

                    # [xiaoyao-NKU] Inter-core Communication (Broadcast)
                    C_remote = T.alloc_shared((block_M, block_N), accum_dtype)
                    T.comm.broadcast(C_shared, C_remote, (0, 0), direction="h")

                    # Store result
                    T.copy(C_remote, C[by * block_M, bx * block_N])

    return main


def test_overall(is_log=False):
    func = kernel_overall(128, 128, 128, 64, 64, 32)
    script_device_mode = """
        with T.launch_thread("blockIdx.x", 16) as bx:
            T.barrier_init(T.int64(15))
            with T.decl_buffer((2, 64, 32), "bfloat16", data=A_shared.data, scope="shared.asram") as A_shared:
                B_shared = T.decl_buffer((2, 32, 64), "bfloat16", data=B_shared.data, scope="shared.wsram")
                C_shared = T.decl_buffer((64, 64), data=C_shared.data, scope="shared.rsram")
                Bias_shared = T.decl_buffer((64, 64), data=Bias_shared.data, scope="shared.rsram")
                C_remote = T.decl_buffer((64, 64), data=C_remote.data, scope="shared.rsram")
                A_rsram_stage = T.decl_buffer((2, 64, 32), "bfloat16", data=A_rsram_stage.data, scope="shared.rsram")
                Bias_layout_stage = T.decl_buffer((64, 64), data=Bias_layout_stage.data, scope="shared.rsram")
                C_layout_stage = T.decl_buffer((64, 64), data=C_layout_stage.data, scope="shared.rsram")
                for i0 in T.serial(16, annotations={"tile.domain": [64, 64], "tile.execution_axis": 0, "tile.execution_domain_axes": [0, 1], "tile.scope_entry": 1, "tile.tile_size": [4, 32]}):
                    for i1 in T.serial(2, annotations={"tile.execution_axis": 1}):
                        for ki in T.serial(4, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                            for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                C_shared[i0 * 4 + ki, i1 * 32 + kj] = T.float32(0.0)
                T.dma_copy(T.region(A_1[0, 0], 1, 64, 32), T.region(A_rsram_stage[0, 0, 0], 2, 1, 64, 32), 0, T.sync_token_id(0))
                T.wait_token(0)
                T.dma_copy(T.region(A_rsram_stage[0, 0, 0], 1, 1, 64, 32), T.region(A_shared[0, 0, 0], 2, 1, 64, 32), 0, T.sync_token_id(1))
                T.dma_copy(T.region(B_1[0, 0], 1, 32, 64), T.region(B_shared[0, 0, 0], 2, 1, 32, 64), 0, T.sync_token_id(2))
                T.wait_token(1)
                T.wait_token(2)
                T.mma_sunmmio(T.region(A_shared[0, 0, 0], 1, 1, 64, 32), T.region(B_shared[0, 0, 0], 1, 1, 32, 64), T.region(C_shared[0, 0], 3, 64, 64), T.bool(False), T.bool(False), T.bool(False), 0, T.sync_token_id(3))
                T.wait_token(3)
                T.dma_copy(T.region(A_rsram_stage[0, 0, 0], 1, 1, 64, 32), T.region(A_shared[0, 0, 0], 2, 1, 64, 32), 1024, T.sync_token_id(4))
                T.wait_token(4)
                T.mma_sunmmio(T.region(A_shared[0, 0, 0], 1, 1, 64, 32), T.region(B_shared[0, 0, 0], 1, 1, 32, 64), T.region(C_shared[0, 0], 3, 64, 64), T.bool(False), T.bool(False), T.bool(False), 2048, T.sync_token_id(5))
                T.dma_copy(T.region(Bias_1[0, 0], 1, 64, 64), T.region(Bias_layout_stage[0, 0], 2, 64, 64), 0, T.sync_token_id(6))
                T.wait_token(6)
                T.sunmmio_layout_transform(T.region(Bias_layout_stage[0, 0], 1, 64, 64), T.region(Bias_shared[0, 0], 2, 64, 64), T.sync_token_id(7))
                T.wait_token(5)
                T.wait_token(7)
                for i in T.serial(16, annotations={"tile.domain": [64, 64], "tile.execution_axis": 0, "tile.execution_domain_axes": [0, 1], "tile.scope_entry": 1, "tile.tile_size": [4, 32]}):
                    for j in T.serial(2, annotations={"tile.execution_axis": 1}):
                        for ki in T.serial(4, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                            for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                C_shared[i * 4 + ki, j * 32 + kj] = C_shared[i * 4 + ki, j * 32 + kj] + Bias_shared[i * 4 + ki, j * 32 + kj]
                T.barrier_arrive_and_wait(T.int64(15))
                T.broadcast_(T.region(C_shared[0, 0], 1, 64, 64), T.region(C_remote[0, 0], 2, 64, 64), 0, 15, 0, 0, T.sync_token_id(8))
                T.wait_token(8)
                T.barrier_arrive_and_wait(T.int64(15))
                T.sunmmio_layout_transform(T.region(C_remote[0, 0], 1, 64, 64), T.region(C_layout_stage[0, 0], 2, 64, 64), T.sync_token_id(9))
                T.wait_token(9)
                T.dma_copy(T.region(C_layout_stage[0, 0], 1, 64, 64), T.region(C_1[0, 0], 2, 64, 64), 0, T.sync_token_id(10))
            T.wait_token(10)
        return 0
    """

    script_lower_tile_op = [
        'A = T.match_buffer(A_handle, (32, 32), "bfloat16", strides=(32, 1))',
        'B = T.match_buffer(B_handle, (32, 32), "bfloat16", strides=(32, 1))',
        "Bias = T.match_buffer(Bias_handle, (32, 32), strides=(32, 1))",
        "C = T.match_buffer(C_handle, (32, 32), strides=(32, 1))",
        'bx = T.launch_thread("blockIdx.x", 16)',
        "for bx_1, by in T.grid(1, 1):",
        "T.dma_copy(T.region(A[0, 0], 1, 64, 32), T.region(A_rsram_stage[0, 0], 2, 64, 32), 0)",
        "T.dma_copy(T.region(B[0, 0], 1, 32, 64), T.region(B_shared[0, 0], 2, 32, 64), 0)",
        "T.broadcast_(T.region(C_shared[0, 0], 1, 64, 64), T.region(C_remote[0, 0], 2, 64, 64), 0, T.int64(15), 0, 0)",
        "T.sunmmio_layout_transform(T.region(C_remote[0, 0], 1, 64, 64), T.region(C_layout_stage[0, 0], 2, 64, 64))",
        "T.dma_copy(T.region(C_layout_stage[0, 0], 1, 64, 64), T.region(C[0, 0], 2, 64, 64), 0)",
    ]

    script_InjectSunmmioSync = [
        'with T.launch_thread("blockIdx.x", 16) as bx:',
        "T.dma_copy(T.region(A_1[0, 0], 1, 64, 32), T.region(A_rsram_stage[0, 0, 0], 2, 1, 64, 32), 0, T.sync_token_id(0))",
        "T.dma_copy(T.region(A_rsram_stage[0, 0, 0], 1, 1, 64, 32), T.region(A_shared[0, 0, 0], 2, 1, 64, 32), 0, T.sync_token_id(1))",
        "T.dma_copy(T.region(B_1[0, 0], 1, 32, 64), T.region(B_shared[0, 0, 0], 2, 1, 32, 64), 0, T.sync_token_id(2))",
        "T.mma_sunmmio(T.region(A_shared[0, 0, 0], 1, 1, 64, 32), T.region(B_shared[0, 0, 0], 1, 1, 32, 64), T.region(C_shared[0, 0], 3, 64, 64), T.bool(False), T.bool(False), T.bool(False), 0, T.sync_token_id(3))",
        "T.dma_copy(T.region(Bias_1[0, 0], 1, 64, 64), T.region(Bias_layout_stage[0, 0], 2, 64, 64), 0, T.sync_token_id(6))",
        "T.sunmmio_layout_transform(T.region(Bias_layout_stage[0, 0], 1, 64, 64), T.region(Bias_shared[0, 0], 2, 64, 64), T.sync_token_id(7))",
        "T.barrier_init(T.int64(15))",
        "T.barrier_arrive_and_wait(T.int64(15))",
        "T.broadcast_(T.region(C_shared[0, 0], 1, 64, 64), T.region(C_remote[0, 0], 2, 64, 64), 0, 15, 0, 0, T.sync_token_id(8))",
        "T.barrier_arrive_and_wait(T.int64(15))",
        "T.sunmmio_layout_transform(T.region(C_remote[0, 0], 1, 64, 64), T.region(C_layout_stage[0, 0], 2, 64, 64), T.sync_token_id(9))",
        "T.dma_copy(T.region(C_layout_stage[0, 0], 1, 64, 64), T.region(C_1[0, 0], 2, 64, 64), 0, T.sync_token_id(10))",
        "T.wait_token(10)",
    ]

    test_config = {
        "LowerTileOp": {
            "script_expected": script_lower_tile_op,
        },
        "InjectSunmmioSync": {
            "script_expected": script_InjectSunmmioSync,
        },
        "DeviceMod": {
            "script_expected": script_device_mode,
        },
    }
    test_config = get_or_add_default_verify(func, test_config)
    if not is_log:
        compile_test(func, out_idx=[2], target="Sunmmio", test_config=test_config)
    else:
        compile_test(
            func,
            out_idx=[2],
            target="Sunmmio",
            log_pass_output=True,
            log_dir=os.path.join(os.path.dirname(__file__), "_debug", "overall"),
            remove_header=True,
        )


if __name__ == "__main__":
    test_overall()
    # test_overall(is_log=True)
