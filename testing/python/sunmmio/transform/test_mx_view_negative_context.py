"""Negative MX copy coordinates retain their lexical proof context."""

import pytest
import tilelang
from tilelang import tvm
from tvm import tir
from tilelang.language.copy_op import _encode_mx_copy_operand


def _copy(source, peer, strict=False):
    return tir.Evaluate(
        tir.call_intrin(
            "handle", tvm.ir.Op.get("tl.mx_deferred_copy"), _encode_mx_copy_operand(source), _encode_mx_copy_operand(peer), int(strict)
        )
    )


def _loads(stmt):
    found = []

    def visit(node):
        if isinstance(node, tir.Call) and node.op.same_as(tvm.ir.Op.get("tl.mx_deferred_copy")):
            found.append(node.args[0].args[1])

    tir.stmt_functor.post_order_visit(stmt, visit)
    return found


def _legalize(body):
    mod = tvm.IRModule({"main": tir.PrimFunc([], body)})
    return tilelang.transform.LegalizeNegativeIndex()(mod)["main"].body


@pytest.mark.parametrize("context", ["then", "else", "nested", "thread", "virtual", "loop", "let", "assume", "assert", "lazy"])
def test_deferred_copy_uses_lexical_bounds(context):
    rows = 16 if context in ("thread", "virtual") else 4
    field = tir.decl_buffer((rows, 32), "float8_e8m0fnu", scope="shared.rsram")
    peer = tir.decl_buffer((1, 32), "float8_e8m0fnu", scope="shared.rsram")
    var = tir.Var("index", "int32")
    index = var - rows if context in ("thread", "virtual", "loop") else var
    body = _copy(tir.BufferLoad(field, [index, 0]), peer)
    if context == "then":
        body = tir.IfThenElse(var < 0, body, tir.Evaluate(0))
    elif context == "else":
        body = tir.IfThenElse(var >= 0, tir.Evaluate(0), body)
    elif context == "nested":
        body = tir.IfThenElse(var >= -4, tir.IfThenElse(tir.Not(var >= 0), body, tir.Evaluate(0)), tir.Evaluate(0))
    elif context in ("thread", "virtual"):
        iv = tir.IterVar(None, var, tir.IterVar.ThreadIndex, "blockIdx.x" if context == "thread" else "vthread")
        body = tir.AttrStmt(iv, "thread_extent" if context == "thread" else "virtual_thread", rows, body)
    elif context == "loop":
        body = tir.For(var, 0, rows, tir.ForKind.SERIAL, body)
    elif context == "let":
        body = tir.LetStmt(var, -1, body)
    elif context == "assume":
        body = tir.AttrStmt(var < 0, "tl.assume", tir.StringImm("negative"), body)
    elif context == "assert":
        body = tir.AssertStmt(var < 0, tir.StringImm("negative"), body)
    else:
        body = tir.Evaluate(tir.if_then_else(var < 0, body.value, tir.call_intrin("handle", "tir.reinterpret", tir.IntImm("int64", 0))))
    result = _legalize(body)
    actual = _loads(result)[0].indices[0]
    expected = index + rows
    if context == "let":
        expected = tir.IntImm("int32", rows - 1)
    analyzer = tvm.arith.Analyzer()
    assert analyzer.can_prove_equal(actual, expected), (context, actual, expected)


def test_deferred_copy_context_does_not_leak_or_wrap_twice():
    field = tir.decl_buffer((4, 32), "float8_e8m0fnu", scope="shared.rsram")
    peer = tir.decl_buffer((1, 32), "float8_e8m0fnu", scope="shared.rsram")
    index = tir.Var("index", "int32")

    def access(row):
        return _copy(tir.BufferLoad(field, [row, 0]), peer)

    body = tir.SeqStmt([tir.IfThenElse(index < 0, access(index), access(index)), access(index), access(-5)])
    once = _legalize(body)
    twice = _legalize(once)
    actual = [load.indices[0] for load in _loads(twice)]
    expected = [index + 4, index, index, -1]
    assert len(actual) == len(expected)
    for value, reference in zip(actual, expected):
        assert tvm.arith.Analyzer().can_prove_equal(value, reference), (value, reference)


@pytest.mark.parametrize("strict", [False, True])
def test_deferred_slice_bounds_normalize_once_and_keep_strict_policy(strict):
    field = tir.decl_buffer((4, 32), "float8_e8m0fnu", scope="shared.rsram")
    peer = tir.decl_buffer((2, 32), "float8_e8m0fnu", scope="shared.rsram")
    region = tir.BufferRegion(field, [tvm.ir.Range.from_min_extent(-3, 2), tvm.ir.Range.from_min_extent(0, 32)])
    body = _legalize(_legalize(_copy(region, peer, strict)))
    operand = body.value.args[0]
    lo, hi = operand.args[1].indices[0], operand.args[2]
    assert int(lo) == (-3 if strict else 1)
    assert int(hi) == (-1 if strict else 3)


def test_select_operands_do_not_inherit_lazy_branch_constraints():
    field = tir.decl_buffer((4, 32), "float8_e8m0fnu", scope="shared.rsram")
    peer = tir.decl_buffer((1, 32), "float8_e8m0fnu", scope="shared.rsram")
    index = tir.Var("index", "int32")
    yes = _copy(tir.BufferLoad(field, [index, 0]), peer).value
    no = _copy(tir.BufferLoad(field, [-1, 0]), peer).value
    body = _legalize(tir.Evaluate(tir.Select(index < 0, yes, no)))
    loads = _loads(body)
    assert loads[0].indices[0].same_as(index)
    assert int(loads[1].indices[0]) == 3


def test_impossible_context_is_not_used_to_normalize_neighbouring_access():
    field = tir.decl_buffer((4, 32), "float8_e8m0fnu", scope="shared.rsram")
    peer = tir.decl_buffer((1, 32), "float8_e8m0fnu", scope="shared.rsram")
    index = tir.Var("index", "int32")
    access = _copy(tir.BufferLoad(field, [index, 0]), peer)
    impossible = tir.IfThenElse(index >= 0, tir.IfThenElse(index < 0, access, tir.Evaluate(0)), tir.Evaluate(0))
    body = _legalize(tir.SeqStmt([impossible, access]))
    loads = _loads(body)
    assert len(loads) == 1
    assert loads[0].indices[0].same_as(index)


def test_shape_dependent_index_resolves_after_early_normalization():
    extent = tir.Var("extent", "int32")
    index = tir.Var("index", "int32")
    field = tir.decl_buffer((extent, 32), "float8_e8m0fnu", scope="shared.rsram")
    peer = tir.decl_buffer((1, 32), "float8_e8m0fnu", scope="shared.rsram")
    body = tir.For(index, 0, 4, tir.ForKind.SERIAL, _copy(tir.BufferLoad(field, [index - extent, 0]), peer))
    early = _legalize(body)
    late = _legalize(tir.stmt_functor.substitute(early, {extent: tir.IntImm("int32", 4)}))
    assert tvm.arith.Analyzer().can_prove_equal(_loads(late)[0].indices[0], index)


@pytest.mark.parametrize("context", ["likely", "compound", "let_expr", "block", "empty_loop"])
def test_deferred_copy_context_boundary_forms(context):
    field = tir.decl_buffer((4, 32), "float8_e8m0fnu", scope="shared.rsram")
    peer = tir.decl_buffer((1, 32), "float8_e8m0fnu", scope="shared.rsram")
    index = tir.Var("index", "int32")
    coordinate = tir.Let(index, -1, index) if context == "let_expr" else index
    if context == "block":
        coordinate = index - 4
    body = _copy(tir.BufferLoad(field, [coordinate, 0]), peer)
    if context == "likely":
        body = tir.IfThenElse(tir.likely(index < 0), body, tir.Evaluate(0))
    elif context == "compound":
        body = tir.IfThenElse(tir.And(index >= -4, tir.Not(index >= 0)), body, tir.Evaluate(0))
    elif context == "block":
        axis = tir.IterVar(tvm.ir.Range(0, 4), index, tir.IterVar.DataPar)
        body = tir.BlockRealize([tir.IntImm("int32", 0)], True, tir.Block([axis], [], [], "context", body))
    elif context == "empty_loop":
        body = tir.For(index, 0, 0, tir.ForKind.SERIAL, body)
    result = _legalize(body)
    if context == "empty_loop":
        assert not _loads(result)
    else:
        expected = 3 if context == "let_expr" else index if context == "block" else index + 4
        assert tvm.arith.Analyzer().can_prove_equal(_loads(result)[0].indices[0], expected)


@pytest.mark.parametrize("coordinate,expected", [(-5, -1), (-4, 0), (-1, 3), (0, 0), (3, 3), (4, 4)])
def test_deferred_copy_keeps_out_of_bounds_distinct_from_negative_wrapping(coordinate, expected):
    field = tir.decl_buffer((4, 32), "float8_e8m0fnu", scope="shared.rsram")
    peer = tir.decl_buffer((1, 32), "float8_e8m0fnu", scope="shared.rsram")
    body = _legalize(_legalize(_copy(tir.BufferLoad(field, [coordinate, 0]), peer)))
    assert int(_loads(body)[0].indices[0]) == expected
