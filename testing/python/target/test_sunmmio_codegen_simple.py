import os

import tilelang
import tilelang.language as T
import tilelang.testing

from compile_pipeline import target
from sunmmio_codegen_validation_utils import (
    validate_sunmmio_codegen_with_npuir_opt,
)


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
# os.environ["SUNMMIO_TEST_LOG_IR"] = "1"

LOOSE_OPT_ARGS = ("--verify-each",)


@target("Sunmmio")
def simple_global_copy_gemm_kernel(
    M=32,
    N=32,
    K=32,
    dtype="float16",
    accum_dtype="float32",
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(1):
            A_shared = T.alloc_shared((M, K), dtype)
            B_shared = T.alloc_shared((K, N), dtype)
            C_shared = T.alloc_shared((M, N), accum_dtype)

            T.copy(A[0, 0], A_shared)
            T.copy(B[0, 0], B_shared)
            T.gemm(A_shared, B_shared, C_shared)
            T.copy(C_shared, C[0, 0])

    return main


def test_simple_global_copy_gemm_codegen_validates_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        simple_global_copy_gemm_kernel(),
        tmp_path,
        mlir_filename="simple_global_copy_gemm_suvm.mlir",
        expected_tokens=(
            "suvm.copy_async",
            "suvm.tc.mma",
        ),
        opt_args=LOOSE_OPT_ARGS,
    )

    assert "sunmmio.fake" not in src
    assert src.count("suvm.copy_async") >= 3
    assert src.count("suvm.tc.mma") == 1


if __name__ == "__main__":
    tilelang.testing.main()
