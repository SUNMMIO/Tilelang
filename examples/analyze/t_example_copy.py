import tilelang.testing
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target
import tilelang.language as T

# 替换成pipeline文件的compile_test
# from testing.my_test.python.comile_sunmmio.compile_with_log import compile_with_log
from compile_pipeline import compile_test


def mma_3times_single_thread(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32, dtype="float16"):

    @T.prim_func
    def mma_3times_kernel(
        A: T.Tensor((M, N, K), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_like_share = T.alloc_shared((M, N, K), dtype)
            A_shared1 = T.alloc_shared((block_M, block_N, block_K), dtype)
            A_shared2 = T.alloc_shared((block_M, block_N, block_K), dtype)
            A_shared3 = T.alloc_shared((block_N, block_K), dtype)

            # 【base_rule】 type: buffer , buffer_load, buffer_region
            # 【base_rule】extent: shape, None(补1)， region_extend
            # 【rule】buffer copy: 校验shape,不一致，报错,一致，extend = shape
            # T.copy(A, A_shared1)
            T.copy(A, A_like_share)

            # 【rule】Scalar BufferLoad -> scalar BufferLoad, lowered as direct store.
            T.copy(A[1, 2, 3], A_shared1[0, 0, 0])

            # 【rule】 extent(!=1)的数量相同的，不校验
            # BufferLoad -> Buffer.
            T.copy(A[1, 2, 3], A_shared1)

            # BufferRegion -> buffer
            T.copy(A[0:16, 0:16, 0:16], A_shared1)

            # BufferRegion -> Bufferload
            T.copy(A[0:16, 0:16, 0:16], A_shared1[0, 0, 0])

            # BufferRegion -> BufferRegion.
            T.copy(A[0:16, 0:16, 0:16], A_shared1[0:16, 0:16, 0:16])
            T.copy(A[0:32, 0:32, 0:32], A_shared1[0:16, 0:16, 0:16])

            # 特殊的BufferRegion -> BufferRegion.，实际上还是extend = 1,1,64
            # 1D region -> 3D region with matching tail extent.
            T.copy(A[0, 0, 0:64], A_shared1[0:1, 0:1, 0:64])

            # 【rule】 extent(!=1)的数量不同
            # bufferreion 扩写后都要min原始extent，所以结果就是基本不变
            T.copy(A[0:32, 0:16, 0:64], A_shared1[0:32, 0:1, 0:64])
            T.copy(A[0:32, 0:1, 0:64], A_shared1[0:32, 0:16, 0:64])

            T.copy(A[0:32, 0:16, 0:64], A_shared1[0:16, 0:1, 0:16])

            T.copy(A[0:128, 0:128, 0:128], A_shared1[0:16, 0:1, 0:16])

            T.copy(A_shared3[:16, :16], A[1, :16, :16])

            T.copy(A[0:16, 0:16, 0:16], A_shared3)

            # 【rule】rsam,转为tile操作
            T.copy(A_shared1[0:16, 0:32, 0:32], A_shared2[0:16, 0:32, 0:32])
            T.copy(A_shared1, A_shared2)

    return mma_3times_kernel


mma_3times_kernel = mma_3times_single_thread()


def codegen_sunmmio_suvm_from_kernel(kernel):
    # sunmmio的处理pass，可以替换成piepeline的 compile_test
    _, device_mod = compile_test(
        kernel,
        target="Sunmmio",
        log_file="t_codegen_tmp",
        split_flag=False,
        origin_flag=False,
    )
    print("device_mod:")
    print(device_mod.show())
    # target = determine_target("Sunmmio", return_object=True)
    # builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    # return builder(device_mod, target, "suvm").inspect_source()


def t1():
    codegen_sunmmio_suvm_from_kernel(mma_3times_kernel)
    # print("codegen_str:")
    # print(codegen_str)


if __name__ == "__main__":
    t1()
