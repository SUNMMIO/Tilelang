import tilelang
import pytest
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target
import tilelang as tl
import tilelang.language as T

tilelang.env.disable_cache()


def layout_func(i, j, continuous):
    return (i // 32 * (continuous // 32) + j // 32) * 32 * 32 + i % 32 * 32 + j % 32


def matmul(M, N, K, block_M, block_N, block_K, version, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            # T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                if version == 1:
                    T.gemm_v1(A_shared[0:8, 16:32], B_shared[0:16, 8:16], C_shared[8:24, 16:32], transpose_A=True, transpose_B=True)
                elif version == 2:
                    T.gemm_v2(A_shared[0:8, 16:32], B_shared[0:16, 8:16], C_shared[8:24, 16:32], transpose_A=True, transpose_B=True)
                else:
                    raise ValueError(f"unsupported gemm version: {version}")

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return tvm.IRModule({"main": main})


stmts = [
    "T.mma_sunmmio(T.region(A_shared[0, 16], 1, 8, 16), T.region(B_shared[0, 8], 1, 16, 8), T.region(C_shared[8, 16], 3, 16, 16), T.bool(True), T.bool(True), T.bool(False))",
    "T.mma_sunmmio(T.region(A_shared[0, 16], 1, 8, 16), T.region(B_shared[0, 8], 1, 16, 8), T.region(C_shared[8, 16], 3, 16, 16), T.bool(True), T.bool(True), T.bool(False))",
]
TEST_CASES = [
    # gemm v1
    (128, 128, 128, 32, 32, 32, 1, stmts[0]),
    (128, 128, 128, 64, 32, 64, 1, stmts[1]),
    # # gemm v2
    (128, 128, 128, 32, 32, 32, 2, stmts[0]),
    (128, 128, 128, 64, 32, 64, 2, stmts[1]),
]


@pytest.mark.parametrize(
    "M, N, K, block_M, block_N, block_K, version, lower_stmt",
    TEST_CASES,
)
def test_tilelang_gemm_sunmmio_layout(M, N, K, block_M, block_N, block_K, version, lower_stmt):
    target_name = "Sunmmio"
    target = determine_target(target_name, return_object=True)
    with tvm.target.Target(target):
        mod = matmul(M, N, K, block_M, block_N, block_K, version)
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.InferSramScope()(mod)
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)
        texts = mod.script().split("\n")
        text = texts[-2].lstrip()
        assert text == lower_stmt
