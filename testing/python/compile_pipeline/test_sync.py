import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.layout import make_zz_layout

from compile_pipeline import compile_test, target
from formal_verify_funcs import *


@target("Sunmmio")
def kernel_sync(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    shard_policy = T.MeshShardingPolicy(y=0, x=1)
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    A_shape = (M, K)
    B_shape = (M, K)
    C_shape = (M, N)
    A_layout = make_zz_layout(A_shape, [0, 1], (32, 32))
    B_layout = make_zz_layout(B_shape, [0, 1], (32, 32))
    C_layout = make_zz_layout(C_shape, [0, 1], (32, 32))

    @T.prim_func
    def kernel(
        A: T.MeshTensor(A_shape, shard_policy, device_mesh_config, dtype, layout=A_layout),
        B: T.MeshTensor(B_shape, shard_policy, device_mesh_config, dtype, layout=B_layout),
        C: T.MeshTensor(C_shape, shard_policy, device_mesh_config, dtype, layout=C_layout),
    ):
        # Initialize Kernel Context
        with T.Kernel(ncores) as _cid:
            sharded_M, _ = A.shape
            _, sharded_N = C.shape

            A_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.asram")
            B_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.wsram")
            C_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.rsram")
            D_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.rsram")
            E_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.rsram")

            for bx, _by in T.Persistent(
                [T.ceildiv(sharded_N, block_N), T.ceildiv(sharded_M, block_M)],
                ncores,
                _cid,
            ):
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


def test_sync():
    func = kernel_sync(1024 * 16, 1024 * 16, 1024 * 16, 1024, 1024, 1024)

    script_device_mode = """
    @T.prim_func(private=True)
    def kernel_kernel() -> T.int32:
        T.func_attr({"target": T.target({"keys": ["cpu"], "kind": "llvm", "mattr": ["device_mesh_nrow_4", "device_mesh_ncol_4"], "mcpu": "sunmmio-a4e", "tag": ""}), "tir.is_global_func": True, "tir.noalias": True, "tl.non_restrict_params": []})
        with T.launch_thread("blockIdx.x", 16) as bx:
            by = T.launch_thread("blockIdx.y", 16)
            tx = T.launch_thread("threadIdx.x", 1)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            with T.allocate([1048576], "float16", "shared.rsram") as C_shared:
                D_shared = T.allocate([1048576], "float16", "shared.rsram")
                C_shared_1 = T.Buffer((1048576,), "float16", data=C_shared, scope="shared.rsram")
                with T.allocate([1048576], "float16", "shared.asram") as A_shared:
                    B_shared = T.allocate([1048576], "float16", "shared.wsram")
                    A_shared_1 = T.Buffer((1048576,), "float16", data=A_shared, scope="shared.asram")
                    B_shared_1 = T.Buffer((1048576,), "float16", data=B_shared, scope="shared.wsram")
                    T.mma_sunmmio(T.region(A_shared_1[0], 1, 1048576), T.region(B_shared_1[0], 1, 1048576), T.region(C_shared_1[0], 3, 1048576), T.bool(False), T.bool(False), T.bool(False), T.sync_token_id(0))
                D_shared_1 = T.Buffer((1048576,), "float16", data=D_shared, scope="shared.rsram")
                if bx <= 2:
                    for i in T.unroll(16384):
                        D_shared_1[i * 64:i * 64 + 64] = T.Broadcast(T.float16(0.0), 64)
                for i in range(5):
                    T.wait_token(0)
                    C_shared_1[i * 1024] = T.Cast("float16", T.Cast("float32", C_shared_1[i * 1024]) + T.float32(1.0))
                T.sync_null_token(2)
                T.barrier_init(1, T.int64(1), T.int64(15))
                for _i in range(10):
                    E_shared = T.allocate([1048576], "float16", "shared.rsram")
                    T.wait_token(2)
                    T.barrier_arrive_and_wait(1)
                    E_shared_1 = T.Buffer((1048576,), "float16", data=E_shared, scope="shared.rsram")
                    T.broadcast_(T.region(D_shared_1[0], 1, 1048576), T.region(E_shared_1[0], 2, 1048576), 1048576, 0, 0, T.sync_token_id(1))
                    T.barrier_init(0, T.int64(1), T.int64(15))
                    T.wait_token(1)
                    T.barrier_arrive_and_wait(0)
                    E_shared_1[0] = T.Cast("float16", T.Cast("float32", E_shared_1[0]) + T.float32(1.0))
                    T.broadcast_(T.region(E_shared_1[0], 1, 1048576), T.region(D_shared_1[0], 2, 1048576), 1048576, 0, 0, T.sync_token_id(2))
                    T.barrier_init(1, T.int64(1), T.int64(15))
            T.wait_token(2)
            T.barrier_arrive_and_wait(1)
        return 0
    """

    script_lower_tile_op = [
        'A = T.match_buffer(A_handle, (4096, 4096), "float16", strides=(4096, 1))',
        'B = T.match_buffer(B_handle, (4096, 4096), "float16", strides=(4096, 1))',
        'C = T.match_buffer(C_handle, (4096, 4096), "float16", strides=(4096, 1))',
        'bx = T.launch_thread("blockIdx.x", 16)',
        "for w in range(1):",
        "T.mma_sunmmio(T.region(A_shared[0, 0], 1, 1024, 1024), T.region(B_shared[0, 0], 1, 1024, 1024), T.region(C_shared[0, 0], 3, 1024, 1024), T.bool(False), T.bool(False), T.bool(False), 0)",
        "T.broadcast_(T.region(D_shared[0, 0], 1, 1024, 1024), T.region(E_shared[0, 0], 2, 1024, 1024), 0, T.int64(15), 0, 0)",
        "T.broadcast_(T.region(E_shared[0, 0], 1, 1024, 1024), T.region(D_shared[0, 0], 2, 1024, 1024), 0, T.int64(15), 0, 0)",
    ]

    script_InjectSunmmioSync = [
        'with T.launch_thread("blockIdx.x", 16) as bx:',
        "T.mma_sunmmio(T.region(A_shared[0, 0], 1, 1024, 1024), T.region(B_shared[0, 0], 1, 1024, 1024), T.region(C_shared[0, 0], 3, 1024, 1024), T.bool(False), T.bool(False), T.bool(False), 0, T.sync_token_id(0))",
        "T.sync_null_token(2)",
        "T.barrier_init(1, T.int64(1), T.int64(15))",
        "T.broadcast_(T.region(D_shared[0, 0], 1, 1024, 1024), T.region(E_shared[0, 0], 2, 1024, 1024), 0, 15, 0, 0, T.sync_token_id(1))",
        "T.barrier_init(0, T.int64(1), T.int64(15))",
        "T.broadcast_(T.region(E_shared[0, 0], 1, 1024, 1024), T.region(D_shared[0, 0], 2, 1024, 1024), 0, 15, 0, 0, T.sync_token_id(2))",
        "T.barrier_init(1, T.int64(1), T.int64(15))",
        "T.wait_token(2)",
        "T.barrier_arrive_and_wait(1)",
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
    test_sync()
