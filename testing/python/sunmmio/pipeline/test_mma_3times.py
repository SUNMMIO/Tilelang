import os

import tilelang.language as T
from tilelang.layout import make_zz_layout

from testing.python.sunmmio.common.compile_pipeline import compile_test, target
from testing.python.sunmmio.common.formal_verify import *


@target("Sunmmio")
def kernel_mma_3times_single_thread(M=16, N=16, K=16, block_M=128, block_N=128, block_K=32, dtype="float16"):
    shard_policy = T.MeshShardingPolicy(y=0, x=1)

    A_shape = (M, K)
    B_shape = (K, N)
    C_shape = (M, N)
    A_layout = make_zz_layout(A_shape, [0, 1], (32, 32))
    B_layout = make_zz_layout(B_shape, [0, 1], (32, 32))
    C_layout = make_zz_layout(C_shape, [0, 1], (32, 32))

    @T.prim_func
    def mma_3times_kernel(
        A: T.MeshTensor(A_shape, shard_policy, dtype, layout=A_layout),
        B: T.MeshTensor(B_shape, shard_policy, dtype, layout=B_layout),
        C: T.MeshTensor(C_shape, shard_policy, dtype, layout=C_layout),
    ):
        # Initialize single-thread Kernel context
        with T.Kernel() as _cid:
            sharded_M, _ = A.local_shape
            _, sharded_N = C.local_shape

            # [Key modification] Split multiple shared memory allocations to test merge_shared_memory_allocations
            # Allocate multiple slice memories related to A (simulate A data storage in different stages)
            A_shared1 = T.alloc_shared((block_M, block_K), dtype)
            A_shared2 = T.alloc_shared((block_M, block_K), dtype)
            A_shared3 = T.alloc_shared((block_M, block_K), dtype)
            # Allocate multiple slice memories related to B (simulate B data storage in different stages)
            B_shared1 = T.alloc_shared((block_K, block_N), dtype)
            B_shared2 = T.alloc_shared((block_K, block_N), dtype)
            B_shared3 = T.alloc_shared((block_K, block_N), dtype)
            # Allocate multiple accumulation memories related to C (simulate intermediate results in different MMA stages)
            C_shared1 = T.alloc_shared((block_M, block_N), dtype)

            for _bx in T.serial(T.ceildiv(sharded_N, block_N)):
                for _by in T.serial(T.ceildiv(sharded_M, block_M)):
                    # 1st MMA: copy data to stage1 memory -> compute -> save result to acc1

                    T.copy(A[block_M * 0, block_K * 0], A_shared1)
                    T.copy(B[block_K * 0, block_N * 0], B_shared1)
                    T.clear(C_shared1)
                    T.gemm(A_shared1, B_shared1, C_shared1)

                    # 2nd MMA: copy data to stage2 memory -> accumulate based on acc1 -> save result to acc2
                    T.copy(A[block_M * 1, block_K * 0], A_shared2)
                    T.copy(B[block_K * 1, block_N * 0], B_shared2)
                    T.gemm(A_shared2, B_shared2, C_shared1)

                    # 3rd MMA: copy data to stage3 memory -> accumulate based on acc2 -> save result to final
                    T.copy(A[block_M * 2, block_K * 0], A_shared3)
                    T.copy(B[block_K * 2, block_N * 0], B_shared3)
                    T.gemm(A_shared3, B_shared3, C_shared1)

                    # Write the final result back to global memory
                    T.copy(C_shared1, C[0, 0])

    return mma_3times_kernel


def test_mma_3times(is_log=False):
    func = kernel_mma_3times_single_thread(1024, 1024, 1024)

    test_config = {}
    test_config = get_or_add_default_verify(func, test_config)
    if not is_log:
        compile_test(func, out_idx=[2], target="Sunmmio", test_config=test_config)
    else:
        compile_test(
            func,
            out_idx=[2],
            target="Sunmmio",
            log_pass_output=True,
            log_dir=os.path.join(os.path.dirname(__file__), "_debug", "mma_3times"),
            remove_header=True,
        )


if __name__ == "__main__":
    test_mma_3times()
    # test_mma_3times(is_log=True)
