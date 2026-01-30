"""test of using DMA operations in Tilelang Mesh Language to perform matrix multiplication with ReLU activation."""
import tilelang.language as T


def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):

    @T.prim_func
    def matmul_relu_kernel(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Clear local accumulation
            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # Copy tile of A through DMA

                T.dma_load(A[by * block_M, ko * block_K], A_shared)

                # Copy tile of B
                T.dma_load(B[ko * block_K, bx * block_N], B_shared)

                # Perform a tile-level GEMM on the shared buffers
                T.gemm(A_shared, B_shared, C_local)

            # relu
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)

            # Copy result back to global memory through DMA
            T.dma_store(C_local, C[by * block_M, bx * block_N])

    return matmul_relu_kernel


M = 1024  # M = T.dynamic("m") if you want to use dynamic shape
N = 1024
K = 1024
block_M = 128
block_N = 128
block_K = 32

mul_primfunc = matmul(M, N, K, block_M, block_N, block_K)

expected_result = """# from tvm.script import tir as T

@T.prim_func
def matmul_relu_kernel(A_handle: T.handle, B_handle: T.handle, C_handle: T.handle):
    A = T.match_buffer(A_handle, (1024, 1024), "float16", strides=(1024, 1))
    B = T.match_buffer(B_handle, (1024, 1024), "float16", strides=(1024, 1))
    C = T.match_buffer(C_handle, (1024, 1024), "float16", strides=(1024, 1))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", 8)
    by = T.launch_thread("blockIdx.y", 8)
    tx = T.launch_thread("threadIdx.x", 128)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(A[by * 128, 0:993], B[0:993, bx * 128], C[by * 128, bx * 128])
        T.writes()
        A_shared = T.alloc_buffer((128, 32), "float16", scope="shared.dyn")
        B_shared = T.alloc_buffer((32, 128), "float16", scope="shared.dyn")
        C_local = T.alloc_buffer((128, 128), scope="local.fragment")
        T.fill(T.region(C_local[0, 0], 2, 128, 128), 0)
        for ko in T.serial(32, annotations={"num_stages": 3}):
            T.dma_load(T.region(A[by * 128, ko * 32], 1, 128, 32), T.region(A_shared[0, 0], 2, 128, 32), 0)
            T.dma_load(T.region(B[ko * 32, bx * 128], 1, 32, 128), T.region(B_shared[0, 0], 2, 32, 128), 0)
            T.gemm_py(T.region(A_shared[0, 0], 1, 128, 32), T.region(B_shared[0, 0], 1, 32, 128), T.region(C_local[0, 0], 3, 128, 128), T.bool(False), T.bool(False), 128, 128, 32, 0, T.bool(False), 32, 128, 0, 0, 1, 0, T.uint32(0), 0, 0)
        for i in T.parallel(128):
            for j in T.parallel(128):
                C_local[i, j] = T.max(C_local[i, j], T.float32(0.0))
        T.dma_store(T.region(C_local[0, 0], 1, 128, 128), T.region(C[by * 128, bx * 128], 2, 128, 128), 0)"""

# mul_primfunc.show() # Uncomment to see the generated primfunc
assert mul_primfunc.script(
) == expected_result, "The output is not as expected. Use mul_primfunc.show() to view the results."
