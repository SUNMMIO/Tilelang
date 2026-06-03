import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.layout import make_zz_layout

from compile_pipeline import compile_test, target
from formal_verify_funcs import *


_ROW_MASK_BX = "T.int64(15)"

_COL_MASK_BX = "T.int64(15)"


@target("Sunmmio")
def kernel_comm(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
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
        C: T.MeshTensor(C_shape, shard_policy, device_mesh_config, accum_dtype, layout=C_layout),
    ):
        # Initialize Kernel Context
        with T.Kernel(ncores) as _cid:
            sharded_M, _ = A.shape
            _, sharded_N = C.shape

            A_shared = T.alloc_shared((block_M, block_K), dtype=dtype)
            A_remote_1 = T.alloc_shared((block_M, block_K), dtype=dtype)
            A_remote_2 = T.alloc_shared((block_M, block_K), dtype=dtype)
            A_remote_3 = T.alloc_shared((block_M, block_K), dtype=dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype=dtype)
            B_remote_1 = T.alloc_shared((block_K, block_N), dtype=dtype)
            B_remote_2 = T.alloc_shared((block_K, block_N), dtype=dtype)
            B_remote_3 = T.alloc_shared((block_K, block_N), dtype=dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype=accum_dtype)
            C_allgather_1 = T.alloc_shared((16, block_M, block_N), dtype=accum_dtype)
            C_allgather_2 = T.alloc_shared((4, block_M, block_N), dtype=accum_dtype)
            C_allgather_3 = T.alloc_shared((4, block_M, block_N), dtype=accum_dtype)

            for _bx, _by in T.Persistent(
                [T.ceildiv(sharded_N, block_N), T.ceildiv(sharded_M, block_M)],
                ncores,
                _cid,
            ):
                T.clear(A_shared)
                T.clear(B_shared)
                T.clear(C_shared)  # Avoid Fill op unsupported scope error
                T.comm.broadcast(A_shared, A_remote_1, (0, 0), direction="all")
                T.comm.broadcast(A_shared, A_remote_2, (0, 0), direction="h")
                T.comm.broadcast(A_shared, A_remote_3, (0, 0), direction="v")
                T.comm.put(B_shared, B_remote_1, (1, 2), (2, 3))
                T.comm.put(B_shared, B_remote_2, (1, 2), (1, 3))
                T.comm.put(B_shared, B_remote_3, (1, 2), (3, 2))
                T.comm.all_gather(C_shared, C_allgather_1, direction="all")
                T.comm.all_gather(C_shared, C_allgather_2, direction="h")
                T.comm.all_gather(C_shared, C_allgather_3, direction="v")

    return main


def test_comm():
    func = kernel_comm(1024 * 16, 1024 * 16, 1024 * 16, 1024, 1024, 1024)
    script_comm = [
        'A = T.match_buffer(A_handle, (4096, 4096), "float16", strides=(4096, 1))',
        'B = T.match_buffer(B_handle, (4096, 4096), "float16", strides=(4096, 1))',
        "C = T.match_buffer(C_handle, (4096, 4096), strides=(4096, 1))",
        'bx = T.launch_thread("blockIdx.x", 16)',
        "for w in range(1):",
        "T.broadcast_(T.region(A_shared[0, 0], 1, 1024, 1024), T.region(A_remote_1[0, 0], 2, 1024, 1024), 1, T.int64(15), 0, 0)",
        "T.broadcast_(T.region(A_remote_1[0, 0], 1, 1024, 1024), T.region(A_remote_1[0, 0], 2, 1024, 1024), 0, T.int64(15), 0, 12)",
        "T.broadcast_(T.region(B_shared[0, 0], 1, 1024, 1024), T.region(B_remote_1[0, 0], 2, 1024, 1024), 1, T.int64(4), 0, 6)",
        f"T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_1[bx, 0, 0], 2, 1, 1024, 1024), 0, {_ROW_MASK_BX}, 0)",
        f"T.broadcast_(T.region(C_allgather_1[bx // 4 * 4, 0, 0], 1, 4, 1024, 1024), T.region(C_allgather_1[bx // 4 * 4, 0, 0], 2, 4, 1024, 1024), 1, {_COL_MASK_BX}, 0)",
        f"T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_2[bx % 4, 0, 0], 2, 1, 1024, 1024), 0, {_ROW_MASK_BX}, 0)",
        f"T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_3[bx // 4, 0, 0], 2, 1, 1024, 1024), 1, {_COL_MASK_BX}, 0)",
    ]
    test_config = {
        "LowerTileOp": {
            "script_expected": script_comm,
        },
    }
    test_config = get_or_add_default_verify(func, test_config)
    compile_test(func, out_idx=[2], target="Sunmmio", test_config=test_config)


if __name__ == "__main__":
    test_comm()
