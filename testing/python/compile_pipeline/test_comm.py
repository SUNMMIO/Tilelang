import tilelang.language as T
from compile_pipeline import compile_test
from formal_verify_funcs import *


def kernel_comm(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    mesh_device_config = (4, 4)

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.MeshShardingPolicy(x=1, y=0), mesh_device_config, dtype),
        B: T.MeshTensor((K, N), T.MeshShardingPolicy(x=1, y=0), mesh_device_config, dtype),
        C: T.MeshTensor((M, N), T.MeshShardingPolicy(x=1, y=0), mesh_device_config, accum_dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
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
        """
                T.broadcast_(T.region(A_shared[0, 0], 1, 1024, 1024), T.region(A_remote_1[0, 0], 2, 1024, 1024), 1048576, 0, 1)
                T.broadcast_(T.region(A_remote_1[0, 0], 1, 1024, 1024), T.region(A_remote_1[0, 0], 2, 1024, 1024), 1048576, 0, 0)
                T.broadcast_(T.region(A_remote_1[0, 0], 1, 1024, 1024), T.region(A_remote_1[0, 0], 2, 1024, 1024), 1048576, 4, 0)
                T.broadcast_(T.region(A_remote_1[0, 0], 1, 1024, 1024), T.region(A_remote_1[0, 0], 2, 1024, 1024), 1048576, 8, 0)
                T.broadcast_(T.region(A_remote_1[0, 0], 1, 1024, 1024), T.region(A_remote_1[0, 0], 2, 1024, 1024), 1048576, 12, 0)
        """,
        """
                T.broadcast_(T.region(A_shared[0, 0], 1, 1024, 1024), T.region(A_remote_2[0, 0], 2, 1024, 1024), 1048576, 0, 0)
        """,
        """
                T.broadcast_(T.region(A_shared[0, 0], 1, 1024, 1024), T.region(A_remote_3[0, 0], 2, 1024, 1024), 1048576, 0, 1)
        """,
        """
                T.broadcast_(T.region(B_shared[0, 0], 1, 1024, 1024), T.region(B_remote_1[0, 0], 2, 1024, 1024), 1048576, 6, 1, 0, 1, 3)
                T.broadcast_(T.region(B_remote_1[0, 0], 1, 1024, 1024), T.region(B_remote_1[0, 0], 2, 1024, 1024), 1048576, 10, 0, 0, 1, 2)
        """,
        """
                T.broadcast_(T.region(B_shared[0, 0], 1, 1024, 1024), T.region(B_remote_2[0, 0], 2, 1024, 1024), 1048576, 6, 0, 0, 1, 2)
        """,
        """
                T.broadcast_(T.region(B_shared[0, 0], 1, 1024, 1024), T.region(B_remote_3[0, 0], 2, 1024, 1024), 1048576, 6, 1, 0, 1, 2)
        """,
        """
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_1[0, 0, 0], 2, 16, 1024, 1024), 1048576, 0, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_1[0, 0, 0], 2, 16, 1024, 1024), 1048576, 1, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_1[0, 0, 0], 2, 16, 1024, 1024), 1048576, 2, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_1[0, 0, 0], 2, 16, 1024, 1024), 1048576, 3, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_1[0, 0, 0], 2, 16, 1024, 1024), 1048576, 4, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_1[0, 0, 0], 2, 16, 1024, 1024), 1048576, 5, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_1[0, 0, 0], 2, 16, 1024, 1024), 1048576, 6, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_1[0, 0, 0], 2, 16, 1024, 1024), 1048576, 7, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_1[0, 0, 0], 2, 16, 1024, 1024), 1048576, 8, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_1[0, 0, 0], 2, 16, 1024, 1024), 1048576, 9, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_1[0, 0, 0], 2, 16, 1024, 1024), 1048576, 10, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_1[0, 0, 0], 2, 16, 1024, 1024), 1048576, 11, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_1[0, 0, 0], 2, 16, 1024, 1024), 1048576, 12, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_1[0, 0, 0], 2, 16, 1024, 1024), 1048576, 13, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_1[0, 0, 0], 2, 16, 1024, 1024), 1048576, 14, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_1[0, 0, 0], 2, 16, 1024, 1024), 1048576, 15, 0)
                T.broadcast_(T.region(C_allgather_1[0, 0, 0], 1, 4, 1024, 1024), T.region(C_allgather_1[0, 0, 0], 2, 4, 1024, 1024), 4194304, 0, 1)
                T.broadcast_(T.region(C_allgather_1[4, 0, 0], 1, 4, 1024, 1024), T.region(C_allgather_1[4, 0, 0], 2, 4, 1024, 1024), 4194304, 4, 1)
                T.broadcast_(T.region(C_allgather_1[8, 0, 0], 1, 4, 1024, 1024), T.region(C_allgather_1[8, 0, 0], 2, 4, 1024, 1024), 4194304, 8, 1)
                T.broadcast_(T.region(C_allgather_1[12, 0, 0], 1, 4, 1024, 1024), T.region(C_allgather_1[12, 0, 0], 2, 4, 1024, 1024), 4194304, 12, 1)
                T.broadcast_(T.region(C_allgather_1[0, 0, 0], 1, 4, 1024, 1024), T.region(C_allgather_1[0, 0, 0], 2, 4, 1024, 1024), 4194304, 1, 1)
                T.broadcast_(T.region(C_allgather_1[4, 0, 0], 1, 4, 1024, 1024), T.region(C_allgather_1[4, 0, 0], 2, 4, 1024, 1024), 4194304, 5, 1)
                T.broadcast_(T.region(C_allgather_1[8, 0, 0], 1, 4, 1024, 1024), T.region(C_allgather_1[8, 0, 0], 2, 4, 1024, 1024), 4194304, 9, 1)
                T.broadcast_(T.region(C_allgather_1[12, 0, 0], 1, 4, 1024, 1024), T.region(C_allgather_1[12, 0, 0], 2, 4, 1024, 1024), 4194304, 13, 1)
                T.broadcast_(T.region(C_allgather_1[0, 0, 0], 1, 4, 1024, 1024), T.region(C_allgather_1[0, 0, 0], 2, 4, 1024, 1024), 4194304, 2, 1)
                T.broadcast_(T.region(C_allgather_1[4, 0, 0], 1, 4, 1024, 1024), T.region(C_allgather_1[4, 0, 0], 2, 4, 1024, 1024), 4194304, 6, 1)
                T.broadcast_(T.region(C_allgather_1[8, 0, 0], 1, 4, 1024, 1024), T.region(C_allgather_1[8, 0, 0], 2, 4, 1024, 1024), 4194304, 10, 1)
                T.broadcast_(T.region(C_allgather_1[12, 0, 0], 1, 4, 1024, 1024), T.region(C_allgather_1[12, 0, 0], 2, 4, 1024, 1024), 4194304, 14, 1)
                T.broadcast_(T.region(C_allgather_1[0, 0, 0], 1, 4, 1024, 1024), T.region(C_allgather_1[0, 0, 0], 2, 4, 1024, 1024), 4194304, 3, 1)
                T.broadcast_(T.region(C_allgather_1[4, 0, 0], 1, 4, 1024, 1024), T.region(C_allgather_1[4, 0, 0], 2, 4, 1024, 1024), 4194304, 7, 1)
                T.broadcast_(T.region(C_allgather_1[8, 0, 0], 1, 4, 1024, 1024), T.region(C_allgather_1[8, 0, 0], 2, 4, 1024, 1024), 4194304, 11, 1)
                T.broadcast_(T.region(C_allgather_1[12, 0, 0], 1, 4, 1024, 1024), T.region(C_allgather_1[12, 0, 0], 2, 4, 1024, 1024), 4194304, 15, 1)
        """,
        """
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_2[0, 0, 0], 2, 4, 1024, 1024), 1048576, 0, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_2[0, 0, 0], 2, 4, 1024, 1024), 1048576, 1, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_2[0, 0, 0], 2, 4, 1024, 1024), 1048576, 2, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_2[0, 0, 0], 2, 4, 1024, 1024), 1048576, 3, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_2[0, 0, 0], 2, 4, 1024, 1024), 1048576, 4, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_2[0, 0, 0], 2, 4, 1024, 1024), 1048576, 5, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_2[0, 0, 0], 2, 4, 1024, 1024), 1048576, 6, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_2[0, 0, 0], 2, 4, 1024, 1024), 1048576, 7, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_2[0, 0, 0], 2, 4, 1024, 1024), 1048576, 8, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_2[0, 0, 0], 2, 4, 1024, 1024), 1048576, 9, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_2[0, 0, 0], 2, 4, 1024, 1024), 1048576, 10, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_2[0, 0, 0], 2, 4, 1024, 1024), 1048576, 11, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_2[0, 0, 0], 2, 4, 1024, 1024), 1048576, 12, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_2[0, 0, 0], 2, 4, 1024, 1024), 1048576, 13, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_2[0, 0, 0], 2, 4, 1024, 1024), 1048576, 14, 0)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_2[0, 0, 0], 2, 4, 1024, 1024), 1048576, 15, 0)
        """,
        """
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_3[0, 0, 0], 2, 4, 1024, 1024), 1048576, 0, 1)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_3[0, 0, 0], 2, 4, 1024, 1024), 1048576, 4, 1)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_3[0, 0, 0], 2, 4, 1024, 1024), 1048576, 8, 1)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_3[0, 0, 0], 2, 4, 1024, 1024), 1048576, 12, 1)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_3[0, 0, 0], 2, 4, 1024, 1024), 1048576, 1, 1)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_3[0, 0, 0], 2, 4, 1024, 1024), 1048576, 5, 1)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_3[0, 0, 0], 2, 4, 1024, 1024), 1048576, 9, 1)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_3[0, 0, 0], 2, 4, 1024, 1024), 1048576, 13, 1)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_3[0, 0, 0], 2, 4, 1024, 1024), 1048576, 2, 1)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_3[0, 0, 0], 2, 4, 1024, 1024), 1048576, 6, 1)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_3[0, 0, 0], 2, 4, 1024, 1024), 1048576, 10, 1)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_3[0, 0, 0], 2, 4, 1024, 1024), 1048576, 14, 1)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_3[0, 0, 0], 2, 4, 1024, 1024), 1048576, 3, 1)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_3[0, 0, 0], 2, 4, 1024, 1024), 1048576, 7, 1)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_3[0, 0, 0], 2, 4, 1024, 1024), 1048576, 11, 1)
                T.broadcast_(T.region(C_shared[0, 0], 1, 1024, 1024), T.region(C_allgather_3[0, 0, 0], 2, 4, 1024, 1024), 1048576, 15, 1)
        """,
    ]
    script_mere_allocate = [
        """
        buf_shmem = T.allocate([121634816], "uint8", "shared.rsram")
        """
    ]

    def get_verify_merge_allocate():
        """Merge test for multiple scopes with mixed constant and non-constant sizes"""
        kernel_name = "main_kernel"
        # 8 float16 buffers of 1024*1024 + 4 float32 buffers, no reuse (many wait at the end)
        cnt_r = 1024 * 1024 * 8 * 2 + 1024 * 1024 * 4 * (1 + 16 + 4 + 4)
        return build_verify_merge_allocate(kernel_name=kernel_name, cnt_r=cnt_r)

    test_config = {
        "LowerTileOp": {
            "script_expected": script_comm,
        },
        "MergeSharedMemoryAllocationsSunmmio": {
            "script_expected": script_mere_allocate,
            "formal_verify": get_verify_merge_allocate(),
        },
    }
    test_config = get_or_add_default_verify(func, test_config)
    compile_test(func, out_idx=[2], target="Sunmmio", test_config=test_config)


if __name__ == "__main__":
    test_comm()
