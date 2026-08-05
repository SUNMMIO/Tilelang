import os

import tilelang.language as T
from tilelang.layout import make_zz_layout

from testing.python.sunmmio.common.compile_pipeline import compile_test, target
from testing.python.sunmmio.common.formal_verify import *


@target("Sunmmio")
def kernel_sync(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    shard_policy = T.MeshShardingPolicy(y=0, x=1)

    A_shape = (M, K)
    B_shape = (M, K)
    C_shape = (M, N)
    A_layout = make_zz_layout(A_shape, [0, 1], (32, 32))
    B_layout = make_zz_layout(B_shape, [0, 1], (32, 32))
    C_layout = make_zz_layout(C_shape, [0, 1], (32, 32))

    @T.prim_func
    def kernel(
        A: T.MeshTensor(A_shape, shard_policy, dtype, layout=A_layout),
        B: T.MeshTensor(B_shape, shard_policy, dtype, layout=B_layout),
        C: T.MeshTensor(C_shape, shard_policy, dtype, layout=C_layout),
    ):
        # Initialize Kernel Context
        with T.Kernel() as _cid:
            sharded_M, _ = A.local_shape
            _, sharded_N = C.local_shape

            A_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.asram")
            B_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.wsram")
            C_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.rsram")
            D_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.rsram")
            E_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.rsram")

            for bx in T.serial(T.ceildiv(sharded_N, block_N)):
                for _by in T.serial(T.ceildiv(sharded_M, block_M)):
                    T.gemm(A_shared, B_shared, C_shared)
                    if bx <= 2:
                        T.clear(D_shared)

                    for i in range(5):
                        C_shared[i, 0] = C_shared[i, 0] + 1.0

                    for _i in range(10):
                        T.comm.broadcast(D_shared, E_shared, (0, 0), direction="h")
                        E_shared[0, 0] = E_shared[0, 0] + 1.0
                        T.comm.broadcast(E_shared, D_shared, (0, 0), direction="h")

    return kernel


def test_sync(is_log=False):
    func = kernel_sync(1024 * 16, 1024 * 16, 1024 * 16, 1024, 1024, 1024)

    script_device_mode = """
        with T.launch_thread("blockIdx.x", 16) as bx:
            T.barrier_init(T.int64(15))
            with T.decl_buffer((1024, 1024), "float16", data=A_shared.data, scope="shared.asram") as A_shared:
                B_shared = T.decl_buffer((1024, 1024), "float16", data=B_shared.data, scope="shared.wsram")
                C_shared = T.decl_buffer((1024, 1024), "float16", data=C_shared.data, scope="shared.rsram")
                D_shared = T.decl_buffer((1024, 1024), "float16", data=D_shared.data, scope="shared.rsram")
                E_shared = T.decl_buffer((1024, 1024), "float16", data=E_shared.data, scope="shared.rsram")
                T.sync_null_token(2)
                for bx_1 in range(4):
                    T.sync_null_token(2)
                    for _by in range(4):
                        T.mma_sunmmio(T.region(A_shared[0, 0], 1, 1024, 1024), T.region(B_shared[0, 0], 1, 1024, 1024), T.region(C_shared[0, 0], 3, 1024, 1024), T.bool(False), T.bool(False), T.bool(False), 0, T.sync_token_id(0))
                        if bx_1 <= 2:
                            T.wait_token(2)
                            for i0 in T.serial(128, annotations={"tile.domain": [1024, 1024], "tile.execution_axis": 0, "tile.execution_domain_axes": [0, 1], "tile.scope_entry": 1, "tile.tile_size": [8, 32]}):
                                for i1 in T.serial(32, annotations={"tile.execution_axis": 1}):
                                    for ki in T.serial(8, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                        for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                            D_shared[i0 * 8 + ki, i1 * 32 + kj] = T.float16(0.0)
                        T.wait_token(0)
                        for i in range(5):
                            C_shared[i, 0] = T.Cast("float16", T.Cast("float32", C_shared[i, 0]) + T.float32(1.0))
                        T.sync_null_token(2)
                        for _i in range(10):
                            T.wait_token(2)
                            T.barrier_arrive_and_wait(T.int64(15))
                            T.broadcast_(T.region(D_shared[0, 0], 1, 1024, 1024), T.region(E_shared[0, 0], 2, 1024, 1024), 0, 15, 0, 0, T.sync_token_id(1))
                            T.wait_token(1)
                            E_shared[0, 0] = T.Cast("float16", T.Cast("float32", E_shared[0, 0]) + T.float32(1.0))
                            T.barrier_arrive_and_wait(T.int64(15))
                            T.broadcast_(T.region(E_shared[0, 0], 1, 1024, 1024), T.region(D_shared[0, 0], 2, 1024, 1024), 0, 15, 0, 0, T.sync_token_id(2))
            T.wait_token(2)
        return 0
    """

    script_lower_tile_op = [
        'A = T.match_buffer(A_handle, (4096, 4096), "float16", strides=(4096, 1))',
        'B = T.match_buffer(B_handle, (4096, 4096), "float16", strides=(4096, 1))',
        'C = T.match_buffer(C_handle, (4096, 4096), "float16", strides=(4096, 1))',
        'bx = T.launch_thread("blockIdx.x", 16)',
        "for bx_1, _by in T.grid(4, 4):",
        "T.mma_sunmmio(T.region(A_shared[0, 0], 1, 1024, 1024), T.region(B_shared[0, 0], 1, 1024, 1024), T.region(C_shared[0, 0], 3, 1024, 1024), T.bool(False), T.bool(False), T.bool(False), 0)",
        "T.broadcast_(T.region(D_shared[0, 0], 1, 1024, 1024), T.region(E_shared[0, 0], 2, 1024, 1024), 0, T.int64(15), 0, 0)",
        "T.broadcast_(T.region(E_shared[0, 0], 1, 1024, 1024), T.region(D_shared[0, 0], 2, 1024, 1024), 0, T.int64(15), 0, 0)",
    ]

    script_InjectSunmmioSync = [
        'with T.launch_thread("blockIdx.x", 16) as bx:',
        "T.mma_sunmmio(T.region(A_shared[0, 0], 1, 1024, 1024), T.region(B_shared[0, 0], 1, 1024, 1024), T.region(C_shared[0, 0], 3, 1024, 1024), T.bool(False), T.bool(False), T.bool(False), 0, T.sync_token_id(0))",
        "T.sync_null_token(2)",
        "T.barrier_init(T.int64(15))",
        "T.barrier_arrive_and_wait(T.int64(15))",
        "T.broadcast_(T.region(D_shared[0, 0], 1, 1024, 1024), T.region(E_shared[0, 0], 2, 1024, 1024), 0, 15, 0, 0, T.sync_token_id(1))",
        "T.barrier_arrive_and_wait(T.int64(15))",
        "T.broadcast_(T.region(E_shared[0, 0], 1, 1024, 1024), T.region(D_shared[0, 0], 2, 1024, 1024), 0, 15, 0, 0, T.sync_token_id(2))",
        "T.wait_token(2)",
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
            log_dir=os.path.join(os.path.dirname(__file__), "_debug", "sync"),
            remove_header=True,
        )


if __name__ == "__main__":
    test_sync()
    # test_sync(is_log=True)
