import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.layout import make_zz_layout

from testing.python.sunmmio.common.codegen_validation import (
    validate_sunmmio_codegen_with_npuir_opt,
)
from testing.python.sunmmio.common.compile_pipeline import target


tilelang.env.disable_cache()

SUPPORTED_UNROLL_FACTORS = (1, 2, 4, 8, 16)
SUVM_PIPELINE_OPT_ARGS = ("--verify-each", "--suvm-to-llvm-pipeline")


def _make_frontend_kernel(unroll):
    @T.prim_func
    def main(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (64, 64), dtype=T.float32)

        for i, j in T.Tiles((64, 64), parallel=True, unroll=unroll):
            A[i, j] = A[i, j] + T.float32(1)

    return main


def _make_frontend_kernel_without_unroll():
    @T.prim_func
    def main(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (64, 64), dtype=T.float32)

        for i, j in T.Tiles((64, 64), parallel=True):
            A[i, j] = A[i, j] + T.float32(1)

    return main


def _collect_tiles_loops(func):
    loops = []

    def visit(node):
        if isinstance(node, tvm.tir.For) and "tile.loop_parallel" in node.annotations:
            loops.append(node)

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return loops


@pytest.mark.parametrize("factor", SUPPORTED_UNROLL_FACTORS)
def test_tiles_frontend_attaches_unroll_to_scope_root_only(factor):
    loops = _collect_tiles_loops(_make_frontend_kernel(factor))
    roots = [loop for loop in loops if "tile.domain" in loop.annotations]

    assert len(roots) == 1
    assert int(roots[0].annotations["tile.loop_unroll"]) == factor
    assert all("tile.loop_unroll" not in loop.annotations for loop in loops if loop is not roots[0])


def test_tiles_frontend_omits_unroll_annotation_by_default():
    for kernel in (_make_frontend_kernel_without_unroll(), _make_frontend_kernel(None)):
        loops = _collect_tiles_loops(kernel)
        assert loops
        assert all("tile.loop_unroll" not in loop.annotations for loop in loops)


@pytest.mark.parametrize("factor", (True, False, 0, -1, 3, 32, 2.0, "4"))
def test_tiles_frontend_rejects_unsupported_unroll_factors(factor):
    with pytest.raises((TypeError, ValueError), match="unroll"):
        _make_frontend_kernel(factor)


@target("Sunmmio")
def _tiles_unroll_kernel(unroll=None):
    shape = (64, 64)
    domain = (63, 63)
    layout = make_zz_layout(shape, [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, T.placement.replicated(), T.float32, layout=layout),  # type: ignore
        B: T.MeshTensor(shape, T.placement.replicated(), T.float32, layout=layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, T.float32)
            B_shared = T.alloc_shared(shape, T.float32)

            T.copy(A, A_shared)
            T.clear(B_shared)
            for i, j in T.Tiles(domain, parallel=True, unroll=unroll):
                B_shared[i, j] = A_shared[i, j] + T.float32(1)
            T.copy(B_shared, B)

    return main


@target("Sunmmio")
def _fused_tiles_unroll_kernel(first_unroll, second_unroll):
    shape = (64, 64)
    layout = make_zz_layout(shape, [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, T.placement.replicated(), T.float32, layout=layout),  # type: ignore
        B: T.MeshTensor(shape, T.placement.replicated(), T.float32, layout=layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, T.float32)
            intermediate = T.alloc_shared(shape, T.float32)
            B_shared = T.alloc_shared(shape, T.float32)

            T.copy(A, A_shared)
            for i, j in T.Tiles(shape, parallel=True, unroll=first_unroll):
                intermediate[i, j] = A_shared[i, j] + T.float32(1)
            for i, j in T.Tiles(shape, parallel=True, unroll=second_unroll):
                B_shared[i, j] = intermediate[i, j] * T.float32(2)
            T.copy(B_shared, B)

    return main


@pytest.mark.parametrize("factor", SUPPORTED_UNROLL_FACTORS)
def test_tiles_unroll_reaches_sunmmio_scf_loop(factor, tmp_path, monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT", "1")
    src = validate_sunmmio_codegen_with_npuir_opt(
        _tiles_unroll_kernel(factor),
        tmp_path,
        mlir_filename=f"tiles_unroll_{factor}.mlir",
        expected_tokens=("suvm.tile.cmpi",),
        opt_args=SUVM_PIPELINE_OPT_ARGS,
    )

    assert src.count(f"#loop_unroll = #llvm.loop_unroll<count = {factor} : i32>") == 1
    assert src.count("#loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll>") == 1
    assert src.count("llvm.loop_annotation = #loop_annotation") == 1


def test_tiles_unroll_omission_preserves_unannotated_codegen(tmp_path, monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT", "1")
    src = validate_sunmmio_codegen_with_npuir_opt(
        _tiles_unroll_kernel(),
        tmp_path,
        mlir_filename="tiles_unroll_omitted.mlir",
        expected_tokens=("suvm.tile.cmpi",),
        opt_args=SUVM_PIPELINE_OPT_ARGS,
    )

    assert "llvm.loop_annotation" not in src


def test_fused_tiles_unroll_uses_smallest_explicit_factor(tmp_path, monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT", "1")
    src = validate_sunmmio_codegen_with_npuir_opt(
        _fused_tiles_unroll_kernel(4, 2),
        tmp_path,
        mlir_filename="fused_tiles_unroll.mlir",
        opt_args=SUVM_PIPELINE_OPT_ARGS,
    )

    assert src.count("#loop_unroll = #llvm.loop_unroll<count = 2 : i32>") == 1
    assert "#llvm.loop_unroll<count = 4 : i32>" not in src
    assert src.count("llvm.loop_annotation = #loop_annotation") == 1


if __name__ == "__main__":
    tilelang.testing.main()
