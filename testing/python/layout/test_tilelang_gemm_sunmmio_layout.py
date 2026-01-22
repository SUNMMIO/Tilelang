import tilelang
import pytest
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target
import tilelang as tl
import tilelang.language as T
from tvm import tir
from tvm.tir import PyStmtExprVisitor
from tvm.tir.transform import prim_func_pass

tilelang.env.disable_cache()

collected_result = {}


def layout_func(i, j, continuous):
    return (i // 32 * (continuous // 32) + j // 32) * 32 * 32 + i % 32 * 32 + j % 32


def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):

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

            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return tvm.IRModule({'main': main})


@tir.functor.visitor
class _LayoutVisualVisitor(PyStmtExprVisitor):

    def __init__(self):
        super().__init__()

    def visit_block_(self, op: tir.Block) -> None:
        if "layout_map" in op.annotations:
            layout_map = op.annotations["layout_map"]
            collected_result.clear()
            for key, layout in layout_map.items():
                collected_result[key.name] = layout


def LayoutVisual():

    def pass_fn(func: tir.PrimFunc, mod, ctx):
        _LayoutVisualVisitor().visit_stmt(func.body)
        return func

    return prim_func_pass(pass_fn, opt_level=0)


TEST_CASES = [
    # (M, N, K, block_M, block_N, block_K, version)
    (128, 128, 128, 32, 32, 32),
    (128, 128, 128, 64, 64, 64),
    (128, 128, 128, 64, 32, 64),
    (128, 128, 128, 32, 64, 64),
    (128, 128, 128, 64, 64, 32),
    (128, 128, 128, 64, 32, 32),
    (128, 128, 128, 32, 64, 32),
    (128, 128, 128, 32, 32, 64),
]


@pytest.mark.parametrize(
    "M, N, K, block_M, block_N, block_K",
    TEST_CASES,
)
def test_tilelang_gemm_sunmmio_layout(M, N, K, block_M, block_N, block_K):
    target_name = "Sunmmio"
    target = determine_target(target_name, return_object=True)
    with tvm.target.Target(target):
        mod = matmul(M, N, K, block_M, block_N, block_K)
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tl.transform.LayoutInference()(mod)
        LayoutVisual()(mod)

        for i in range(block_M):
            for j in range(block_K):
                index = layout_func(i, j, block_K)
                assert index == collected_result['A_shared'].map_forward_index([i, j])[0]

        for i in range(block_K):
            for j in range(block_N):
                index = layout_func(i, j, block_N)
                assert index == collected_result['B_shared'].map_forward_index([i, j])[0]

        for i in range(block_M):
            for j in range(block_N):
                index = layout_func(i, j, block_N)
                assert index == collected_result['C_shared'].map_forward_index([i, j])[0]


test_tilelang_gemm_sunmmio_layout(128, 128, 128, 32, 32, 64)
