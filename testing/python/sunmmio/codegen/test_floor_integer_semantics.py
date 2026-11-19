"""Floor division/remainder must not inherit truncation for signed operands."""

import pytest

from tilelang import tvm
from testing.python.sunmmio.common.codegen_validation import (
    codegen_sunmmio_suvm_mlir,
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
