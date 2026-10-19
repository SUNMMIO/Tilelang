"""Conditional fan-out must preserve the original evaluation and lexical scope."""

import numpy as np
import pytest

import tilelang
from tilelang import tvm
from tvm import tir


def _module(buffer, body, *extra_params):
    func = tir.PrimFunc([buffer.data, *extra_params], body, buffer_map={buffer.data: buffer})
    return tvm.IRModule({"main": func.with_attr("global_symbol", "main").with_attr("tir.noalias", True)})


def _execute(mod, initial, *extra_args, addons=None):
    executable = tvm.compile(mod["main"], target="c").jit(addons=addons, options=["-std=c++17"])
    values = tvm.runtime.tensor(np.array(initial, dtype="int32"))
    executable(values, *extra_args)
    return values.numpy()


@pytest.mark.parametrize("merge", [False, True])
def test_if_stmt_binding_preserves_single_evaluation(merge):
    buffer = tir.decl_buffer((2,), "int32", name="A")
    body = tir.IfThenElse(
        buffer[0] > 0,
        tir.SeqStmt([tir.BufferStore(buffer, 0, [0]), tir.BufferStore(buffer, 1, [1])]),
        None,
    )
    transformed = tilelang.transform.IfStmtBinding()(_module(buffer, body))
    transformed = tilelang.transform.Simplify()(transformed)
    if merge:
        transformed = tilelang.transform.MergeIfStmt()(transformed)
    np.testing.assert_array_equal(_execute(transformed, [1, 7]), [0, 1])
    np.testing.assert_array_equal(_execute(transformed, [0, 7]), [0, 7])


def test_if_stmt_binding_evaluates_side_effecting_condition_once(tmp_path):
    buffer = tir.decl_buffer((2,), "int32", name="A")
    addon = tmp_path / "condition.cc"
    addon.write_text(
        '#include <stdint.h>\nextern "C" int32_t next_condition(int32_t* value) { return (*value)++; }\n',
        encoding="utf-8",
    )
    condition = tir.call_extern("int32", "next_condition", tir.address_of(buffer[0])) > 0
    body = tir.IfThenElse(
        condition,
        tir.SeqStmt([tir.BufferStore(buffer, buffer[1] + 1, [1]), tir.BufferStore(buffer, buffer[1] + 2, [1])]),
        None,
    )
    transformed = tilelang.transform.IfStmtBinding()(_module(buffer, body))
    np.testing.assert_array_equal(_execute(transformed, [1, 0], addons=[str(addon)]), [2, 3])


def test_if_stmt_binding_does_not_snapshot_pure_condition():
    buffer = tir.decl_buffer((2,), "int32", name="A")
    enabled = tir.Var("enabled", "int32")
    condition = enabled > 0
    stores = [tir.BufferStore(buffer, 0, [0]), tir.BufferStore(buffer, 1, [1])]
    before = _module(buffer, tir.IfThenElse(condition, tir.SeqStmt(stores), None), enabled)
    expected = _module(buffer, tir.SeqStmt([tir.IfThenElse(condition, store, None) for store in stores]), enabled)
    transformed = tilelang.transform.IfStmtBinding()(before)
    tvm.ir.assert_structural_equal(transformed, expected)


def test_if_stmt_binding_keeps_single_guard_without_snapshot():
    buffer = tir.decl_buffer((1,), "int32", name="A")
    before = _module(buffer, tir.IfThenElse(buffer[0] > 0, tir.BufferStore(buffer, 0, [0]), None))
    tvm.ir.assert_structural_equal(tilelang.transform.IfStmtBinding()(before), before)


def test_if_stmt_binding_keeps_existing_let_scope():
    buffer = tir.decl_buffer((3,), "int32", name="A")
    value = tir.Var("value", "int32")
    scoped = tir.LetStmt(
        value,
        buffer[1] + 1,
        tir.SeqStmt([tir.BufferStore(buffer, value, [1]), tir.BufferStore(buffer, value + 2, [2])]),
    )
    body = tir.IfThenElse(buffer[0] > 0, tir.SeqStmt([tir.BufferStore(buffer, 0, [0]), scoped]), None)
    transformed = tilelang.transform.IfStmtBinding()(_module(buffer, body))
    np.testing.assert_array_equal(_execute(transformed, [1, 7, 0]), [0, 8, 10])


@pytest.mark.parametrize("enabled", [0, 1])
def test_if_stmt_binding_keeps_nested_condition_at_its_original_position(enabled):
    buffer = tir.decl_buffer((3,), "int32", name="A")
    nested = tir.IfThenElse(
        buffer[0] > 0,
        tir.SeqStmt([tir.BufferStore(buffer, 1, [1]), tir.BufferStore(buffer, 99, [2])]),
        None,
    )
    body = tir.IfThenElse(
        buffer[0] > 0,
        tir.SeqStmt([tir.BufferStore(buffer, 0, [0]), nested, tir.BufferStore(buffer, buffer[2] + 1, [2])]),
        None,
    )
    transformed = tilelang.transform.IfStmtBinding()(_module(buffer, body))
    np.testing.assert_array_equal(_execute(transformed, [enabled, 7, 0]), [0, 7, enabled])


def test_if_stmt_binding_keeps_independent_condition_snapshots():
    buffer = tir.decl_buffer((3,), "int32", name="A")
    first = tir.IfThenElse(
        buffer[0] > 0,
        tir.SeqStmt([tir.BufferStore(buffer, 0, [0]), tir.BufferStore(buffer, 1, [1])]),
        None,
    )
    second = tir.IfThenElse(
        buffer[0] > 0,
        tir.SeqStmt([tir.BufferStore(buffer, 99, [1]), tir.BufferStore(buffer, 1, [2])]),
        None,
    )
    transformed = tilelang.transform.IfStmtBinding()(_module(buffer, tir.SeqStmt([first, second])))
    transformed = tilelang.transform.MergeIfStmt()(transformed)
    np.testing.assert_array_equal(_execute(transformed, [1, 7, 0]), [0, 1, 0])


@pytest.mark.parametrize("indexed_condition", [False, True])
def test_if_stmt_binding_re_evaluates_condition_each_iteration(indexed_condition):
    buffer = tir.decl_buffer((5,), "int32", name="A")
    step = tir.Var("step", "int32")
    index = step if indexed_condition else tir.IntImm("int32", 0)
    body = tir.IfThenElse(
        buffer[index] > 0,
        tir.SeqStmt([tir.BufferStore(buffer, 0, [index]), tir.BufferStore(buffer, buffer[4] + 1, [4])]),
        None,
    )
    loop = tir.For(step, 0, 4, tir.ForKind.SERIAL, body)
    transformed = tilelang.transform.IfStmtBinding()(_module(buffer, loop))
    expected = [0, 0, 0, 0, 2] if indexed_condition else [0, 0, 1, 0, 1]
    np.testing.assert_array_equal(_execute(transformed, [1, 0, 1, 0, 0]), expected)
