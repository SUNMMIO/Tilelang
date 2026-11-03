"""Residency accepts pure arithmetic, never unknown stateful boundary calls."""

import importlib.util
from pathlib import Path

import pytest
from tilelang import tvm
from tvm import tir

_spec = importlib.util.spec_from_file_location(
    "residency_pure_helpers", Path(__file__).with_name("test_generic_sunmmio_residency.py"))
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
_clean_residency_env = _helpers._clean_residency_env


def _epilogue_module(kind):
    mod = _helpers._make_module()
    acc = next(buf for buf in mod["main"].body.block.alloc_buffers if "_acc0" in buf.name)
    value = tir.Cast("float32", tir.BufferLoad(acc, [0, 0]))
    if kind == "opaque":
        value = tir.call_extern("float32", "unknown_math", value)
    elif kind == "nested_opaque":
        value = tir.exp(tir.call_extern("float32", "unknown_math", value))
    elif kind == "pointer_argument":
        value = tir.call_pure_extern("float32", "declared_pure_but_pointer", acc.data)
    else:
        value = {"exp": tir.exp, "sqrt": tir.sqrt, "tanh": tir.tanh,
                 "let_exp": tir.exp}[kind](value)
    body = tir.BufferStore(acc, tir.Cast("bfloat16", value), [0, 0])
    if kind == "let_exp":
        local = tir.Var("activation_value", "float32")
        body = tir.LetStmt(local, value, tir.BufferStore(acc, tir.Cast("bfloat16", local), [0, 0]))
    def rewrite(node):
        if isinstance(node, tir.For) and node.loop_var.name == "natural_column":
            seq = list(node.body.seq)
            core = next(i for i, stmt in enumerate(seq) if isinstance(stmt, tir.For) and "num_stages" in stmt.annotations)
            seq.insert(core + 1, body)
            return tir.For(node.loop_var, node.min, node.extent, node.kind,
                           tir.SeqStmt(seq), annotations=node.annotations, step=node.step)
        return None
    return tvm.IRModule.from_expr(mod["main"].with_body(tir.stmt_functor.ir_transform(mod["main"].body, None, rewrite)))


@pytest.mark.parametrize("kind", ["exp", "sqrt", "tanh", "let_exp"])
def test_pure_math_epilogue_is_eligible(kind, tmp_path, monkeypatch):
    _, report = _helpers._analyze(_epilogue_module(kind), tmp_path, monkeypatch)
    assert _helpers._accepted(report), report


@pytest.mark.parametrize("kind", ["opaque", "nested_opaque", "pointer_argument"])
def test_effectful_or_pointer_boundary_is_rejected(kind, tmp_path, monkeypatch):
    _, report = _helpers._analyze(_epilogue_module(kind), tmp_path, monkeypatch)
    assert not _helpers._accepted(report)
    assert any(item["reason"] == "prefix_or_suffix_side_effect" for item in report["candidates"])
