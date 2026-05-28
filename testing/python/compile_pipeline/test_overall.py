import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.layout import make_zz_layout

from compile_pipeline import compile_test, target
from formal_verify_funcs import *


@target("Sunmmio")
def kernel_overall(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
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

            for bx, by in T.Persistent(
                [T.ceildiv(sharded_N, block_N), T.ceildiv(sharded_M, block_M)],
                ncores,
                _cid,
            ):
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


def test_overall():
    func = kernel_overall(128, 128, 128, 64, 64, 32)
    script_device_mode = """
        with T.launch_thread("blockIdx.x", 2) as bx:
            buf_shmem = T.allocate([32768], "uint8", "shared.rsram")
            A_shared = T.allocate([4096], "float16", "shared.asram")
            B_shared = T.allocate([4096], "float16", "shared.wsram")
            T.barrier_init(T.int64(15))
            by = T.launch_thread("blockIdx.y", 2)
            tx = T.launch_thread("threadIdx.x", 128)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            C_shared = T.Buffer((4096,), data=buf_shmem, scope="shared.rsram")
            for i0 in T.serial(16, annotations={"tile.domain": [64, 64], "tile.execution_axis": 0, "tile.execution_domain_axes": [0, 1], "tile.scope_entry": 1, "tile.tile_size": [4, 32]}):
                for i1 in T.serial(2, annotations={"tile.execution_axis": 1}):
                    for ki in T.serial(4, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                        C_shared[i0 * 256 + ki * 64 + i1 * 32:i0 * 256 + ki * 64 + i1 * 32 + 32] = T.Broadcast(T.float32(0.0), 32)
            A_1 = T.Buffer((1024,), "float16", data=A)
            A_shared_1 = T.Buffer((4096,), "float16", data=A_shared, scope="shared.asram")
            T.dma_copy(T.region(A_1[by * 2048], 1, 2048), T.region(A_shared_1[0], 2, 2048), T.sync_token_id(0))
            B_1 = T.Buffer((1024,), "float16", data=B)
            B_shared_1 = T.Buffer((4096,), "float16", data=B_shared, scope="shared.wsram")
            T.dma_copy(T.region(B_1[bx * 64], 1, 1056), T.region(B_shared_1[0], 2, 2048), T.sync_token_id(1))
            T.wait_token(0)
            T.wait_token(1)
            T.mma_sunmmio(T.region(A_shared_1[0], 1, 2048), T.region(B_shared_1[0], 1, 2048), T.region(C_shared[0], 3, 4096), T.bool(False), T.bool(False), T.bool(False), T.sync_token_id(2))
            T.dma_copy(T.region(A_1[by * 2048 + 32], 1, 2048), T.region(A_shared_1[2048], 2, 2048), T.sync_token_id(3))
            T.dma_copy(T.region(B_1[bx * 64 + 1024], 1, 1056), T.region(B_shared_1[2048], 2, 2048), T.sync_token_id(4))
            T.wait_token(3)
            T.wait_token(4)
            T.wait_token(2)
            T.mma_sunmmio(T.region(A_shared_1[2048], 1, 2048), T.region(B_shared_1[2048], 1, 2048), T.region(C_shared[0], 3, 4096), T.bool(False), T.bool(False), T.bool(False), T.sync_token_id(5))
            T.dma_copy(T.region(A_1[by * 2048 + 64], 1, 2048), T.region(A_shared_1[0], 2, 2048), T.sync_token_id(6))
            T.dma_copy(T.region(B_1[bx * 64 + 2048], 1, 1056), T.region(B_shared_1[0], 2, 2048), T.sync_token_id(7))
            T.wait_token(6)
            T.wait_token(7)
            T.wait_token(5)
            T.mma_sunmmio(T.region(A_shared_1[0], 1, 2048), T.region(B_shared_1[0], 1, 2048), T.region(C_shared[0], 3, 4096), T.bool(False), T.bool(False), T.bool(False), T.sync_token_id(8))
            T.dma_copy(T.region(A_1[by * 2048 + 96], 1, 2048), T.region(A_shared_1[2048], 2, 2048), T.sync_token_id(9))
            T.dma_copy(T.region(B_1[bx * 64 + 3072], 1, 1056), T.region(B_shared_1[2048], 2, 2048), T.sync_token_id(10))
            T.wait_token(9)
            T.wait_token(10)
            T.wait_token(8)
            T.mma_sunmmio(T.region(A_shared_1[2048], 1, 2048), T.region(B_shared_1[2048], 1, 2048), T.region(C_shared[0], 3, 4096), T.bool(False), T.bool(False), T.bool(False), T.sync_token_id(11))
            Bias_1 = T.Buffer((1024,), data=Bias)
            Bias_shared = T.Buffer((4096,), data=buf_shmem, scope="shared.rsram")
            T.dma_copy(T.region(Bias_1[by * 2048 + bx * 64], 1, 2080), T.region(Bias_shared[4096], 2, 4096), T.sync_token_id(12))
            for i in T.serial(16, annotations={"tile.domain": [64, 64], "tile.execution_axis": 0, "tile.execution_domain_axes": [0, 1], "tile.scope_entry": 1, "tile.tile_size": [4, 32]}):
                for j in T.serial(2, annotations={"tile.execution_axis": 1}):
                    for ki in T.serial(4, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                        T.wait_token(11)
                        T.wait_token(12)
                        C_shared[i * 256 + ki * 64 + j * 32:i * 256 + ki * 64 + j * 32 + 32] = C_shared[i * 256 + ki * 64 + j * 32:i * 256 + ki * 64 + j * 32 + 32] + Bias_shared[i * 256 + ki * 64 + j * 32 + 4096:i * 256 + ki * 64 + j * 32 + 4096 + 32]
            C_remote = T.Buffer((4096,), data=buf_shmem, scope="shared.rsram")
            T.barrier_arrive_and_wait(T.int64(15))
            T.broadcast_(T.region(C_shared[0], 1, 4096), T.region(C_remote[4096], 2, 4096), 4096, 0, 0, T.sync_token_id(13))
            T.wait_token(13)
            T.barrier_arrive_and_wait(T.int64(15))
            C_1 = T.Buffer((1024,), data=C)
            T.dma_copy(T.region(C_remote[4096], 1, 4096), T.region(C_1[by * 2048 + bx * 64], 2, 2080), T.sync_token_id(14))
            T.wait_token(14)
    """

    script_lower_tile_op = [
        'A = T.match_buffer(A_handle, (32, 32), "float16", strides=(32, 1))',
        'B = T.match_buffer(B_handle, (32, 32), "float16", strides=(32, 1))',
        "Bias = T.match_buffer(Bias_handle, (32, 32), strides=(32, 1))",
        "C = T.match_buffer(C_handle, (32, 32), strides=(32, 1))",
        'bx = T.launch_thread("blockIdx.x", 16)',
        "for w in range(1):",
        "T.dma_copy(T.region(A[bx * 64, 0], 1, 64, 32), T.region(A_rsram_stage[0, 0], 2, 64, 32), 0)",
        "T.dma_copy(T.region(B[0, 0], 1, 32, 64), T.region(B_shared[0, 0], 2, 32, 64), 0)",
        "T.broadcast_(T.region(C_shared[0, 0], 1, 64, 64), T.region(C_remote[0, 0], 2, 64, 64), 0, T.int64(15), 0, 0)",
        "T.dma_copy(T.region(C_remote[0, 0], 1, 64, 64), T.region(C[bx * 64, 0], 2, 64, 64), 0)",
    ]

    script_InjectSunmmioSync = [
        'with T.launch_thread("blockIdx.x", 16) as bx:',
        "T.dma_copy(T.region(A_1[bx * 64, 0], 1, 64, 32), T.region(A_rsram_stage[0, 0, 0], 2, 1, 64, 32), 0, T.sync_token_id(0))",
        "T.dma_copy(T.region(B_1[0, 0], 1, 32, 64), T.region(B_shared[0, 0, 0], 2, 1, 32, 64), 0, T.sync_token_id(1))",
        "T.mma_sunmmio(T.region(A_shared[0, 0, 0], 1, 1, 64, 32), T.region(B_shared[0, 0, 0], 1, 1, 32, 64), T.region(C_shared[0, 0], 3, 64, 64), T.bool(False), T.bool(False), T.bool(False), 0, T.sync_token_id(3))",
        "T.dma_copy(T.region(Bias_1[bx * 64, 0], 1, 64, 64), T.region(Bias_shared[0, 0], 2, 64, 64), 0, T.sync_token_id(4))",
        "T.barrier_init(T.int64(15))",
        "T.barrier_arrive_and_wait(T.int64(15))",
        "T.broadcast_(T.region(C_shared[0, 0], 1, 64, 64), T.region(C_remote[0, 0], 2, 64, 64), 0, 15, 0, 0, T.sync_token_id(5))",
        "T.barrier_arrive_and_wait(T.int64(15))",
        "T.dma_copy(T.region(C_remote[0, 0], 1, 64, 64), T.region(C_1[bx * 64, 0], 2, 64, 64), 0, T.sync_token_id(6))",
        "T.wait_token(6)",
    ]

    test_config = {
        "LowerTileOp": {
            "script_expected": script_lower_tile_op,
        },
        "InjectSunmmioSync": {
            "script_expected": script_InjectSunmmioSync,
        },
        "DeviceMode": {
            "script_expected": script_device_mode,
        },
    }
    test_config = get_or_add_default_verify(func, test_config)
    compile_test(func, out_idx=[2], target="Sunmmio", test_config=test_config)


if __name__ == "__main__":
    test_overall()
