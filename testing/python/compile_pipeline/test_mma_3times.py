import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.layout import make_zz_layout

from compile_pipeline import compile_test, target
from formal_verify_funcs import *


@target("Sunmmio")
def kernel_mma_3times_single_thread(M=16, N=16, K=16, block_M=128, block_N=128, block_K=32, dtype="float16"):
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
    def mma_3times_kernel(
        A: T.MeshTensor(A_shape, shard_policy, device_mesh_config, dtype, layout=A_layout),
        B: T.MeshTensor(B_shape, shard_policy, device_mesh_config, dtype, layout=B_layout),
        C: T.MeshTensor(C_shape, shard_policy, device_mesh_config, dtype, layout=C_layout),
    ):
        # Initialize single-thread Kernel context
        with T.Kernel(ncores) as _cid:
            sharded_M, _ = A.shape
            _, sharded_N = C.shape

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

            for _bx, _by in T.Persistent(
                [T.ceildiv(sharded_N, block_N), T.ceildiv(sharded_M, block_M)],
                ncores,
                _cid,
            ):
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

    test_config = {}
    test_config = get_or_add_default_verify(func, test_config)
    compile_test(func, target="Sunmmio", test_config=test_config)


if __name__ == "__main__":
    test_mma_3times()
