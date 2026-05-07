import tilelang
import pytest
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target
import tilelang as tl
import tilelang.language as T
from tvm import tir
from tvm.tir import PyStmtExprVisitor
from tvm.tir.transform import prim_func_pass
import tilelang.env as env
from tilelang.layout import make_zz_layout, make_zn_layout
from tilelang.layout.cute_layout import is_same_layout

tilelang.env.disable_cache()

collected_result = {}


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

            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                if version == 1:
                    T.gemm_v1(A_shared, B_shared, C_shared)
                elif version == 2:
                    T.gemm_v2(A_shared, B_shared, C_shared)
                else:
                    raise ValueError(f"unsupported gemm version: {version}")

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return tvm.IRModule({"main": main})


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
    # gemm v1
    (128, 128, 128, 32, 32, 32, 1),
    (128, 128, 128, 64, 64, 64, 1),
    (128, 128, 128, 64, 32, 64, 1),
    (128, 128, 128, 32, 64, 64, 1),
    (128, 128, 128, 64, 64, 32, 1),
    (128, 128, 128, 64, 32, 32, 1),
    (128, 128, 128, 32, 64, 32, 1),
    (128, 128, 128, 32, 32, 64, 1),
    # gemm v2
    (128, 128, 128, 32, 32, 32, 2),
    (128, 128, 128, 64, 64, 64, 2),
    (128, 128, 128, 64, 32, 64, 2),
    (128, 128, 128, 32, 64, 64, 2),
    (128, 128, 128, 64, 64, 32, 2),
    (128, 128, 128, 64, 32, 32, 2),
    (128, 128, 128, 32, 64, 32, 2),
    (128, 128, 128, 32, 32, 64, 2),
]


@pytest.mark.parametrize(
    "M, N, K, block_M, block_N, block_K, version",
    TEST_CASES,
)
def test_tilelang_gemm_sunmmio_layout(M, N, K, block_M, block_N, block_K, version):
    # Enable v2
    env.TILELANG_USE_GEMM_V1 = 0
    assert not env.use_gemm_v1()
    target_name = "Sunmmio"
    target = determine_target(target_name, return_object=True)
    with tvm.target.Target(target):
        mod = matmul(M, N, K, block_M, block_N, block_K, version)
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tl.transform.InferSramScope()(mod)
        mod = tl.transform.LegalizeSunmmioCopyPath()(mod)
        mod = tl.transform.LayoutReducer()(mod)
        mod = tl.transform.SunmmioLayoutInference()(mod)
        LayoutVisual()(mod)

        # A (ASRAM): ZZ layout with block_shape (32, 32)
        expected_a = make_zz_layout((block_M, block_K), block_shape=(32, 32))
        assert is_same_layout(collected_result["A_shared"], expected_a), "A_shared layout mismatch: expected ZZ(32,32)"

        # B (WSRAM): ZN layout with block_shape (32, 32) for transB=False
        expected_b = make_zn_layout((block_K, block_N), axes=[0, 1], block_shape=[32, 32])
        assert is_same_layout(collected_result["B_shared"], expected_b), "B_shared layout mismatch: expected ZN(32,32)"

        # C (RSRAM): ZZ layout with block_shape (32, 32)
        expected_c = make_zz_layout((block_M, block_N), block_shape=(32, 32))
        assert is_same_layout(collected_result["C_shared"], expected_c), "C_shared layout mismatch: expected ZZ(32,32)"
