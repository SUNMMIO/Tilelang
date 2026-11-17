import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.layout import make_zz_layout
from tilelang.layout import make_aligned_row_major

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
def _fused_tiles_unroll_kernel(first_unroll, second_unroll, materialize_intermediate=False):
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
            if materialize_intermediate:
                T.copy(intermediate, B)
            for i, j in T.Tiles(shape, parallel=True, unroll=second_unroll):
                B_shared[i, j] = intermediate[i, j] * T.float32(2)
            T.copy(B_shared, B)

    return main


@target("Sunmmio")
def _scalar_fallback_unroll_kernel():
    shape = (16, 16)
    layout = make_aligned_row_major(shape, T.float32, align_bytes=64)

    @T.prim_func
    def main():
        with T.Kernel():
            src = T.alloc_shared(shape, T.float32)
            dst = T.alloc_shared(shape, T.float32)
            T.annotate_layout({src: layout, dst: layout})

            T.fill(src, 1)
            T.fill(dst, 0)
            for i, j in T.Tiles((4, 4), parallel=True, unroll=4):
                dst[i, j] = src[i, 2 * j]

    return main


@target("Sunmmio")
def _rank3_tiles_unroll_kernel(unroll):
    shape = (2, 64, 64)
    layout = make_zz_layout(shape, [1, 2], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, T.placement.replicated(), T.float32, layout=layout),  # type: ignore
        B: T.MeshTensor(shape, T.placement.replicated(), T.float32, layout=layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, T.float32)
            B_shared = T.alloc_shared(shape, T.float32)

            T.copy(A, A_shared)
            for _repeat in T.serial(2):
                for b, i, j in T.Tiles((2, 63, 63), parallel=True, unroll=unroll):
                    B_shared[b, i, j] = A_shared[b, i, j] + T.float32(1)
            T.copy(B_shared, B)

    return main


@pytest.mark.parametrize("factor", SUPPORTED_UNROLL_FACTORS)
def test_tiles_unroll_reaches_all_sunmmio_scf_domain_loops(factor, tmp_path, monkeypatch):
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
    assert src.count("llvm.loop_annotation = #loop_annotation") == 2


@pytest.mark.parametrize("factor", (None, 4))
def test_rank3_tiles_unroll_includes_serial_domain_axis_only(factor, tmp_path, monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT", "1")
    src = validate_sunmmio_codegen_with_npuir_opt(
        _rank3_tiles_unroll_kernel(factor),
        tmp_path,
        mlir_filename=f"rank3_tiles_unroll_{factor}.mlir",
        expected_tokens=("suvm.tile.cmpi",),
        opt_args=SUVM_PIPELINE_OPT_ARGS,
    )

    # The surrounding repeat loop is unannotated; b, i and j share the hint.
    assert src.count("scf.for ") == 4
    assert src.count("llvm.loop_annotation = #loop_annotation") == (0 if factor is None else 3)


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


def test_tiles_unroll_is_dropped_on_scalar_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT", "1")
    src = validate_sunmmio_codegen_with_npuir_opt(
        _scalar_fallback_unroll_kernel(),
        tmp_path,
        mlir_filename="tiles_unroll_scalar_fallback.mlir",
        expected_tokens=("suvm.tile.pick", "suvm.tile.set"),
        opt_args=SUVM_PIPELINE_OPT_ARGS,
    )

    assert "llvm.loop_annotation" not in src


@pytest.mark.parametrize("factors, expected", (((4, 2), 2), ((None, 4), 4), ((4, None), 4), ((None, None), None)))
def test_fused_tiles_unroll_uses_smallest_explicit_factor_on_each_loop(factors, expected, tmp_path, monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT", "1")
    src = validate_sunmmio_codegen_with_npuir_opt(
        _fused_tiles_unroll_kernel(*factors),
        tmp_path,
        mlir_filename="fused_tiles_unroll.mlir",
        opt_args=SUVM_PIPELINE_OPT_ARGS,
    )

    if expected is None:
        assert "llvm.loop_annotation" not in src
    else:
        assert src.count(f"#loop_unroll = #llvm.loop_unroll<count = {expected} : i32>") == 1
        assert src.count("#llvm.loop_unroll<count = ") == 1
        assert src.count("llvm.loop_annotation = #loop_annotation") == 2


def test_tiles_unroll_does_not_leak_to_next_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT", "1")
    src = validate_sunmmio_codegen_with_npuir_opt(
        _fused_tiles_unroll_kernel(4, None, materialize_intermediate=True),
        tmp_path,
        mlir_filename="separate_tiles_unroll.mlir",
        opt_args=SUVM_PIPELINE_OPT_ARGS,
    )

    assert src.count("scf.for ") == 4
    assert src.count("llvm.loop_annotation = #loop_annotation") == 2
    # The DMA store separates the hinted scope from the unhinted scope.
    first_scope, second_scope = src.split("suvm.copy_async", 2)[1:]
    assert first_scope.count("llvm.loop_annotation = #loop_annotation") == 2
    assert "llvm.loop_annotation" not in second_scope


if __name__ == "__main__":
    tilelang.testing.main()
