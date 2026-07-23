from pathlib import Path

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing

from testing.python.sunmmio.common.codegen_validation import validate_sunmmio_codegen_with_npuir_opt
from testing.python.sunmmio.common.compile_pipeline import target


tilelang.env.disable_cache()


def _line_of(marker: str) -> int:
    lines = Path(__file__).read_text(encoding="utf-8").splitlines()
    for line_no, line in enumerate(lines, start=1):
        if marker in line:
            return line_no
    raise AssertionError(f"cannot find marker {marker} in {__file__}")


def _run_expected_validate_failure(kernel, tmp_path, *, mlir_filename: str) -> str:
    with pytest.raises(Exception) as exc_info:
        validate_sunmmio_codegen_with_npuir_opt(
            kernel,
            tmp_path,
            mlir_filename=mlir_filename,
            opt_args=("--verify-each",),
        )
    return str(exc_info.value)


def _assert_dsl_span_diagnostic(message: str, marker: str, expected_substrings: tuple[str, ...]):
    expected_line = _line_of(marker)
    expected_file = Path(__file__).name
    expected_loc = f"{expected_file}:{expected_line}:"

    missing = [substring for substring in expected_substrings if substring not in message]
    assert not missing, f"missing expected diagnostic substrings {missing}\n{message}"
    assert "at TileLang DSL:" in message, message
    assert expected_loc in message, message


@target("Sunmmio")
def debug_dsl_realistic_kernel(
    m=32,
    n=32,
    k=32,
    dtype="bfloat16",
    accum_dtype="float32",
):
    @T.prim_func
    def main(
        A: T.Tensor((m, k), dtype),
        B: T.Tensor((k, n), dtype),
        C: T.Tensor((m, n), accum_dtype),
    ):
        with T.Kernel():
            A_shared = T.alloc_shared((m, k), dtype, scope="shared.asram")
            B_shared = T.alloc_shared((k, n), dtype, scope="shared.wsram")
            C_shared = T.alloc_shared((m, n), accum_dtype, scope="shared.rsram")

            T.copy(A[0, 0], A_shared)
            T.copy(B[0, 0], B_shared)
            T.clear(C_shared)
            T.gemm(A_shared, B_shared, C_shared)

            for i, j in T.Tiles([m, n], parallel=True):
                C_shared[i, j] = T.sin(C_shared[i, j])  # SPAN_CASE_REALISTIC_TILES

            T.copy(C_shared, C[0, 0])

    return main


def test_debug_codegen_reports_dsl_line_for_realistic_kernel(tmp_path):
    kernel = debug_dsl_realistic_kernel()

    message = _run_expected_validate_failure(
        kernel,
        tmp_path,
        mlir_filename="debug_dsl_realistic_kernel_suvm.mlir",
    )
    _assert_dsl_span_diagnostic(
        message,
        "SPAN_CASE_REALISTIC_TILES",
        ("unsupported expr", "tir.Call", "selected unary math calls"),
    )


def test_debug_codegen_default_span_does_not_emit_marker_in_script():
    kernel = debug_dsl_realistic_kernel()
    assert "tilelang.dsl_span" not in kernel.script()


if __name__ == "__main__":
    tilelang.testing.main()
