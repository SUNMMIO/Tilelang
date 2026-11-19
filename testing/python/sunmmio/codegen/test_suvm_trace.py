"""Independent scalar integer semantics for the SUVM test interpreter."""

from types import SimpleNamespace

import pytest
from tilelang import tvm

from testing.python.sunmmio.common.suvm_trace import trace_suvm_control_flow


def scalar_program(operator, a, b, expected, width):
    rank = tvm.tir.Var("rank", "int32")
    func = tvm.tir.PrimFunc([rank], tvm.tir.Evaluate(0)).with_attr("tl.dist.rank_id_param_index", 0)
    result = SimpleNamespace(device_mod=tvm.IRModule({"check": func}))
    dtype = "index" if width == "index" else f"i{width}"
    source = f"""module {{
  func.func @check(%arg0: i32) {{
    %a = arith.constant {a} : {dtype}
    %b = arith.constant {b} : {dtype}
    %actual = arith.{operator} %a, %b : {dtype}
    %expected = arith.constant {expected} : {dtype}
    %ok = arith.cmpi eq, %actual, %expected : {dtype}
    cf.assert %ok, "integer result"
    return
  }}
}}
"""
    return result, source


@pytest.mark.parametrize("width", [8, 16, 32, 64, "index"])
@pytest.mark.parametrize(
    "a,b,q,r,floor",
    [(-3, 2, -1, -1, -2), (3, -2, -1, 1, -2), (-3, -2, 1, -1, 1), (3, 2, 1, 1, 1), (-4, 2, -2, 0, -2)],
)
@pytest.mark.parametrize("operator,column", [("divsi", 0), ("remsi", 1), ("floordivsi", 2)])
def test_signed_integer_division_and_remainder(width, a, b, q, r, floor, operator, column):
    result, source = scalar_program(operator, a, b, (q, r, floor)[column], width)
    trace_suvm_control_flow(result, source, rank_id=0)


@pytest.mark.parametrize("operator,expected", [("divsi", 3074457345618258602), ("remsi", 1), ("floordivsi", 3074457345618258602)])
def test_i64_division_uses_integer_precision(operator, expected):
    result, source = scalar_program(operator, 2**63 - 1, 3, expected, 64)
    trace_suvm_control_flow(result, source, rank_id=0)


@pytest.mark.parametrize("width", [8, 16, 32, 64])
@pytest.mark.parametrize("operator", ["divui", "remui"])
def test_unsigned_division_uses_bit_patterns(width, operator):
    expected = ((1 << width) - 1) // 2 if operator == "divui" else 1
    result, source = scalar_program(operator, -1, 2, expected, width)
    trace_suvm_control_flow(result, source, rank_id=0)


@pytest.mark.parametrize("operator", ["divsi", "remsi", "floordivsi", "divui", "remui"])
def test_division_by_zero_fails_explicitly(operator):
    result, source = scalar_program(operator, 3, 0, 0, 32)
    with pytest.raises(AssertionError, match="division by zero"):
        trace_suvm_control_flow(result, source, rank_id=0)


@pytest.mark.parametrize("operator", ["divsi", "floordivsi"])
def test_signed_division_overflow_fails_explicitly(operator):
    result, source = scalar_program(operator, -(2**31), -1, 0, 32)
    with pytest.raises(AssertionError, match="division overflow"):
        trace_suvm_control_flow(result, source, rank_id=0)


@pytest.mark.parametrize("width", [32, 64])
def test_minimum_signed_remainder_is_zero(width):
    result, source = scalar_program("remsi", -(1 << (width - 1)), -1, 0, width)
    trace_suvm_control_flow(result, source, rank_id=0)
