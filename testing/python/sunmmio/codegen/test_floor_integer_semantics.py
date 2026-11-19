"""Floor division/remainder must not inherit truncation for signed operands."""

import pytest

import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.common.codegen_validation import (
    codegen_sunmmio_suvm_mlir,
    validate_sunmmio_codegen_with_npuir_opt,
    validate_suvm_mlir_with_npuir_opt,
)


@pytest.mark.parametrize("dtype", ["int32", "int64", "uint32"])
@pytest.mark.parametrize("operation", ["floor_div", "floor_mod", "trunc_div", "trunc_mod"])
def test_integer_division_semantics(dtype, operation, tmp_path):
    a, b = tvm.tir.Var("a", dtype), tvm.tir.Var("b", dtype)
    expression = {
        "floor_div": tvm.tir.FloorDiv,
        "floor_mod": tvm.tir.FloorMod,
        "trunc_div": tvm.tir.Div,
        "trunc_mod": tvm.tir.Mod,
    }[operation](a, b)
    func = tvm.tir.PrimFunc([a, b], tvm.tir.Evaluate(expression)).with_attr("global_symbol", "main")
    source = codegen_sunmmio_suvm_mlir(tvm.IRModule({"main": func}))
    unsigned = dtype.startswith("uint")
    if operation.startswith("floor") and not unsigned:
        assert "arith.floordivsi" in source
        assert "arith.divsi " not in source
        assert "arith.remsi " not in source
        if operation == "floor_mod":
            assert "arith.muli" in source and "arith.subi" in source
    else:
        opcode = "div" if operation.endswith("div") else "rem"
        assert f"arith.{opcode}{'ui' if unsigned else 'si'} " in source
        assert "arith.floordivsi" not in source
    validate_suvm_mlir_with_npuir_opt(source, tmp_path)


@pytest.mark.parametrize(
    "dtype,operation,opcode",
    [
        (T.int32, T.floordiv, "floordivsi"),
        (T.int32, T.floormod, "floordivsi"),
        (T.int32, T.truncdiv, "divsi"),
        (T.int32, T.truncmod, "remsi"),
        (T.uint32, T.floordiv, "divui"),
        (T.uint32, T.floormod, "remui"),
    ],
)
def test_uniform_division_in_tiles(tmp_path, dtype, operation, opcode):
    placement = T.placement.replicated()

    @T.prim_func
    def main(A: T.MeshTensor((32, 32), placement, dtype), B: T.MeshTensor((32, 32), placement, dtype), x: dtype, y: dtype):
        with T.Kernel(T.mesh_ncores()):
            a = T.alloc_shared((32, 32), dtype)
            b = T.alloc_shared((32, 32), dtype)
            T.copy(A, a)
            for i, j in T.Tiles(a):
                b[i, j] = a[i, j] + operation(x, y)
            T.copy(b, B)

    src = validate_sunmmio_codegen_with_npuir_opt(main, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))
    assert any(f"arith.{opcode} " in line and line.endswith(": i32") for line in src.splitlines())
    if operation is T.floormod and dtype == T.int32:
        assert "arith.muli" in src and "arith.subi" in src


def test_uniform_floor_division_keeps_width_in_bf16_select(tmp_path):
    placement = T.placement.replicated()

    @T.prim_func
    def main(A: T.MeshTensor((32, 32), placement, T.bfloat16), B: T.MeshTensor((32, 32), placement, T.bfloat16), x: T.int32, y: T.int32):
        with T.Kernel(T.mesh_ncores()):
            a = T.alloc_shared((32, 32), T.bfloat16)
            b = T.alloc_shared((32, 32), T.bfloat16)
            T.copy(A, a)
            for i, j in T.Tiles(a):
                b[i, j] = a[i, j] + T.if_then_else(T.floordiv(x, y) == 4, T.bfloat16(1), T.bfloat16(0))
            T.copy(b, B)

    src = validate_sunmmio_codegen_with_npuir_opt(main, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))
    (division,) = (line for line in src.splitlines() if "arith.floordivsi" in line)
    assert division.endswith(": i32")
    quotient = division.split("=")[0].strip()
    assert any(f"arith.cmpi eq, {quotient}," in line and line.endswith(": i32") for line in src.splitlines())
