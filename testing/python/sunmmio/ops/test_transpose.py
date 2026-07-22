"""Compilation coverage for asynchronous Sunmmio RSRAM transpose."""

import pytest

import tilelang
import tilelang.language as T
from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import (
    validate_sunmmio_codegen_with_npuir_opt,
)
from tilelang.layout import make_zn_layout, make_zz_layout


tilelang.env.disable_cache()


TRANSPOSE_CONFIGS = [
    pytest.param(64, 64, "bfloat16", "zz", id="bfloat16-zz-64x64"),
    pytest.param(64, 128, "bfloat16", "zn", id="bfloat16-zn-64x128"),
    pytest.param(64, 128, "float32", "zz", id="float32-zz-64x128"),
]


@target("Sunmmio")
def mesh_transpose_kernel(
    m,
    n,
    dtype,
    layout_family,
    control_flow="plain",
    expect_transposed=True,
):
    """Build a replicated transpose with matching DRAM and RSRAM layouts."""
    placement = T.MeshShardingPolicy(replicate=T.MeshReplicationType.ALL)
    src_layout = make_zz_layout((m, n)) if layout_family == "zz" else make_zn_layout((m, n), [0, 1], (32, 32))
    transposed_layout = make_zz_layout((n, m)) if layout_family == "zz" else make_zn_layout((n, m), [0, 1], (32, 32))
    output_shape = (n, m) if expect_transposed else (m, n)
    output_layout = transposed_layout if expect_transposed else src_layout

    @T.prim_func
    def main(
        a: T.MeshTensor((m, n), placement, dtype, layout=src_layout),
        b: T.MeshTensor(output_shape, placement, dtype, layout=output_layout),
    ):
        with T.Kernel():
            src = T.alloc_shared((m, n), dtype, scope="shared.rsram")
            dst = T.alloc_shared((n, m), dtype, scope="shared.rsram")
            if layout_family == "zn":
                T.annotate_layout({src: make_zn_layout((m, n), [0, 1], (32, 32))})
            T.copy(a, src)

            if control_flow == "loop":
                # Each loop iteration performs a round trip. The odd case adds
                # one final transpose, while the even case leaves src unchanged.
                for _ in T.serial(2):
                    T.transpose(src, dst)
                    T.transpose(dst, src)
                if expect_transposed:
                    T.transpose(src, dst)
            else:
                T.transpose(src, dst)

            if expect_transposed:
                T.copy(dst, b)
            else:
                T.copy(src, b)

    return main


@pytest.mark.parametrize(("m", "n", "dtype", "layout_family"), TRANSPOSE_CONFIGS)
def test_transpose_codegen_matrix(tmp_path, m, n, dtype, layout_family):
    src = validate_sunmmio_codegen_with_npuir_opt(
        mesh_transpose_kernel(m, n, dtype, layout_family),
        tmp_path,
        mlir_filename=f"transpose_{dtype}_{layout_family}_{m}x{n}.mlir",
        expected_tokens=(
            "suvm.copy_async",
            "suvm.transpose_async",
            "suvm.wait_token",
        ),
    )

    assert src.count("suvm.transpose_async") == 1


def test_transpose_loop_codegen(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        mesh_transpose_kernel(
            64,
            128,
            "float32",
            "zn",
            control_flow="loop",
            expect_transposed=False,
        ),
        tmp_path,
        mlir_filename="transpose_loop.mlir",
        expected_tokens=(
            "scf.for",
            "suvm.copy_async",
            "suvm.transpose_async",
            "suvm.wait_token",
        ),
    )

    assert src.count("suvm.transpose_async") == 2
