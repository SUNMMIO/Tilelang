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


def index_loop_program(lower, upper, step, expected, operator=None, operand=0):
    rank = tvm.tir.Var("rank", "int32")
    func = tvm.tir.PrimFunc([rank], tvm.tir.Evaluate(0)).with_attr("tl.dist.rank_id_param_index", 0)
    result = SimpleNamespace(device_mod=tvm.IRModule({"check": func}))
    upper_expr = f"%upper = arith.constant {upper} : index"
    if operator:
        upper_expr = f"""%a = arith.constant {upper} : index
    %b = arith.constant {operand} : index
    %upper = arith.{operator} %a, %b : index"""
    source = f"""module {{
  func.func @check(%arg0: i32) {{
    %lower = arith.constant {lower} : index
    {upper_expr}
    %step = arith.constant {step} : index
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %expected = arith.constant {expected} : i32
    %count = scf.for %iv = %lower to %upper step %step iter_args(%acc = %zero) -> (i32) {{
      %next = arith.addi %acc, %one : i32
      %within = arith.cmpi sle, %next, %expected : i32
      cf.assert %within, "too many iterations"
      scf.yield %next : i32
    }}
    %ok = arith.cmpi eq, %count, %expected : i32
    cf.assert %ok, "iteration count"
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


@pytest.mark.parametrize(
    "lower,upper,expected,operator,operand",
    [
        ((1 << 63) - 1, (1 << 63) + 1, 0, None, 0),
        ((1 << 63) - 1, (1 << 63) - 1, 0, "addi", 2),
        ((1 << 63) - 3, -(1 << 63), 2, "subi", 1),
        (0, 1 << 62, 0, "muli", 2),
        (-2, 0, 2, None, 0),
    ],
)
def test_index_arithmetic_and_for_bounds_use_signed_64_bits(lower, upper, expected, operator, operand):
    result, source = index_loop_program(lower, upper, 1, expected, operator, operand)
    trace_suvm_control_flow(result, source, rank_id=0)


def test_for_increment_overflow_cannot_silently_pass_trace():
    result, source = index_loop_program((1 << 63) - 2, (1 << 63) - 1, 2, 1)
    with pytest.raises(AssertionError, match="index increment overflow"):
        trace_suvm_control_flow(result, source, rank_id=0)


@pytest.mark.parametrize("width", [32, 64, "index"])
@pytest.mark.parametrize("value,expected", [(-1, 0), (3, 3)])
def test_signed_maximum_interprets_operand_bit_patterns(width, value, expected):
    result, source = scalar_program("maxsi", value, 0, expected, width)
    trace_suvm_control_flow(result, source, rank_id=0)
