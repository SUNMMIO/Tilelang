import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.utils.target import SUNMMIO_TARGET_DESC

from compile_pipeline import compile_test

# os.environ["SUNMMIO_TEST_LOG_IR"] = "1"
tilelang.env.disable_cache()


def gemm_matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float16):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def build_sunmmio_source_without_compile(mod):
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    return builder(mod, target, "suvm").inspect_source()


def test_sunmmio_codegen_lowers_region_dma_copy_and_mma():
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)

    with tvm.target.Target(target):
        kernel = gemm_matmul(1024, 1024, 1024, 128, 128, 32)
        mod = tvm.IRModule({"main": kernel})
        _, device_mod = compile_test(mod, target=target, remove_header=True)

    src = build_sunmmio_source_without_compile(device_mod)
    print(src)

    assert "suvm.get_partitioned_tile_view" in src
    assert "suvm.copy_async" in src
    assert "suvm.tc.mma" in src
