"""Preserve product dtype and intermediate rounding when matching mixed mul."""

import pytest
import tilelang.language as T
from testing.python.sunmmio.common.codegen_validation import validate_sunmmio_codegen_with_npuir_opt


def kernel(mode):
    a_dtype = T.float32 if mode == "narrow_before" else T.bfloat16
    b_dtype = T.bfloat16 if mode in ("both_promoted", "bf16_product") else T.float32
    placement = T.placement.replicated()

    @T.prim_func
    def main(
        A: T.MeshTensor((32, 32), placement, a_dtype),
        B: T.MeshTensor((32, 32), placement, b_dtype),
        C: T.MeshTensor((32, 32), placement, T.float32),
    ):
        with T.Kernel(T.mesh_ncores()):
            a = T.alloc_shared((32, 32), a_dtype)
            b = T.alloc_shared((32, 32), b_dtype)
            c = T.alloc_shared((32, 32), T.float32)
            T.copy(A, a)
            T.copy(B, b)
            for i, j in T.Tiles(a):
                if mode == "both_promoted":
                    c[i, j] = T.Cast(T.float32, a[i, j]) * T.Cast(T.float32, b[i, j])
                elif mode == "rounded_sum":
                    c[i, j] = T.Cast(T.float32, a[i, j] + T.bfloat16(0.125)) * b[i, j]
                elif mode == "bf16_product":
                    c[i, j] = T.Cast(T.float32, a[i, j] * b[i, j])
                elif mode == "narrow_before":
                    c[i, j] = T.Cast(T.float32, T.Cast(T.bfloat16, a[i, j])) * b[i, j]
                elif mode == "add_control":
                    c[i, j] = T.Cast(T.float32, a[i, j]) + b[i, j]
                else:
                    c[i, j] = T.Cast(T.float32, a[i, j]) * b[i, j]
            T.copy(c, C)

    return main


@pytest.mark.parametrize("mode", ["promoted", "both_promoted", "bf16_product", "narrow_before", "add_control", "rounded_sum"])
def test_rounding_boundaries_and_operand_types(tmp_path, mode, monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT", "1")
    src = validate_sunmmio_codegen_with_npuir_opt(kernel(mode), tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))
    opcode = "suvm.tile.addf" if mode == "add_control" else "suvm.tile.mulf"
    (operation,) = [line for line in src.splitlines() if opcode in line]
    operand_types = (
        ("f32", "f32") if mode == "add_control" else (("bf16", "bf16") if mode in ("both_promoted", "bf16_product") else ("bf16", "f32"))
    )
    result = "bf16" if mode == "bf16_product" else "f32"
    assert f": !suvm.tile<32x32x{operand_types[0]}>, !suvm.tile<32x32x{operand_types[1]}> -> !suvm.tile<32x32x{result}>" in operation
    if mode == "rounded_sum":
        (addition,) = [line for line in src.splitlines() if "suvm.tile.addf" in line]
        assert addition.endswith("-> !suvm.tile<32x32xbf16>")
    if mode == "narrow_before":
        # FP32 -> BF16 quantization remains before the mixed product.
        assert "tile<32x32xf32> -> !suvm.tile<32x32xbf16>" in src
    if mode == "bf16_product":
        # A BF16 product rounded before promotion is NOT an FP32 product.
        assert "tile<32x32xbf16> -> !suvm.tile<32x32xf32>" in src
