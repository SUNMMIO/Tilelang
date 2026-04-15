import tilelang.language as T
from compile_pipeline import compile_test
from formal_verify_funcs import *


def kernel_mma_3times_single_thread(M=16, N=16, K=16, block_M=128, block_N=128, block_K=32, dtype="float16"):
    @T.prim_func
    def mma_3times_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Initialize single-thread Kernel context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=1) as (
            bx,
            by,
        ):
            # with T.Kernel(1, 1, threads=1) as (bx, by):
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
            # C_shared2 = T.alloc_shared((block_M, block_N), dtype)
            # C_shared3 = T.alloc_shared((block_M, block_N), dtype)

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


def test_mma_3times():
    func = kernel_mma_3times_single_thread(1024, 1024, 1024)

    script_mere_allocate = """
        with T.launch_thread("blockIdx.x", 8) as bx:
            buf_shmem = T.allocate([16384], "uint8", "shared.wsram")
            buf_shmem_1 = T.allocate([16384], "uint8", "shared.asram")
            C_shared1 = T.allocate([16384], "float16", "shared.rsram")
    """

    def get_verify_merge_allocate():
        kernel_name = "mma_3times_kernel_kernel"
        # block 128*32; float16->uint * 2; a,c reuse, so only 2 buffer spaces are needed * 2
        cnt_a = 128 * 32 * 2 * 2
        cnt_w = 128 * 32 * 2 * 2
        # c_shared only has one, does not participate in merge, so it remains the original size, type unchanged
        cnt_r = 128 * 128
        return build_verify_merge_allocate(kernel_name=kernel_name, cnt_a=cnt_a, cnt_w=cnt_w, cnt_r=cnt_r)

    test_config = {
        "MergeSharedMemoryAllocationsSunmmio": {
            "script_expected": script_mere_allocate,
            "formal_verify": get_verify_merge_allocate(),
        },
    }
    test_config = get_or_add_default_verify(func, test_config)
    compile_test(func, target="Sunmmio", test_config=test_config)


if __name__ == "__main__":
    test_mma_3times()
