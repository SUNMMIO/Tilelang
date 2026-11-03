"""Bounded dynamic addresses and hole-free tiled initialization.

These are IR legality tests, not a claim about runtime speed or SRAM capacity.
"""

import importlib.util
from pathlib import Path

import pytest
import tilelang as tl
from tilelang import tvm
from tvm import tir

_spec = importlib.util.spec_from_file_location(
    "residency_test_helpers", Path(__file__).with_name("test_generic_sunmmio_residency.py"))
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
APPLIED = _helpers.APPLIED
_accepted, _analyze = _helpers._accepted, _helpers._analyze
_clean_residency_env = _helpers._clean_residency_env
_make_module, _marked = _helpers._make_module, _helpers._marked
_op, _region, _same_statements = _helpers._op, _helpers._region, _helpers._same_statements


def _natural_dynamic_address(fault=None, *, tiled=False):
    module = _make_module(reuse_extent=2, fault="full_float_clear")
    scope = "global" if fault == "global_metadata" else "local.var"
    shape = (2,) if fault == "not_singleton" else (1,)
    pointer = tir.Var("task_metadata", tvm.ir.PointerType(tvm.ir.PrimType("int32"), scope))
    scalar = tir.decl_buffer(shape, "int32", name="task_metadata", data=pointer, scope=scope)
    alias = tir.decl_buffer(shape, "int32", name="task_metadata_alias", data=pointer, scope=scope)
    address = tir.BufferLoad(scalar, [1 if fault == "out_of_bounds_scalar" else 0])

    def rewrite(node):
        if isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.tileop.region":
            load = node.args[0]
            if isinstance(load, tir.BufferLoad) and load.buffer.scope() == "global":
                indices = [address, *load.indices[1:]]
                return tir.Call(node.dtype, node.op,
                                [tir.BufferLoad(load.buffer, indices), *node.args[1:]])
        if isinstance(node, tir.For) and node.loop_var.name == "natural_column":
            statements = list(node.body.seq)
            if fault == "suffix_write":
                statements.append(tir.BufferStore(scalar, 1, [0]))
            if fault == "prefix_write":
                statements.insert(0, tir.BufferStore(scalar, 1, [0]))
            if fault == "alias_write":
                statements.append(tir.BufferStore(alias, 1, [0]))
            if fault == "async_write":
                statements.append(_op("tl.dma_copy", _region(scalar, 1, [0], [1]),
                                      _region(scalar, 2, [0], [1]), 0))
            return tir.For(node.loop_var, node.min, node.extent, node.kind,
                           tir.SeqStmt(statements), node.thread_binding, node.annotations, node.step)
        if isinstance(node, tir.For) and node.loop_var.name == "natural_row":
            initial = tir.BufferStore(scalar, node.loop_var, [0])
            if fault == "conditional_initialization":
                initial = tir.IfThenElse(node.loop_var == 0, initial, None)
            if fault == "zero_trip_initialization":
                initial = tir.For(tir.Var("empty_init", "int32"), 0, 0, tir.ForKind.SERIAL, initial)
            prefix = [] if fault in {"uninitialized", "global_metadata"} else [initial]
            if fault == "pointer_escape":
                prefix.append(tir.Evaluate(tir.call_extern("int32", "opaque_write", scalar.data)))
            if fault == "address_of_escape":
                prefix.append(tir.Evaluate(tir.call_extern(
                    "int32", "opaque_write", tir.call_intrin("handle", "tir.address_of", address))))
            if fault == "preceding_async_write":
                prefix.append(_op("tl.dma_copy", _region(scalar, 1, [0], [1]),
                                  _region(scalar, 2, [0], [1]), 0))
            body = tir.SeqStmt([*prefix, node.body]) if prefix else node.body
            if fault == "shared_candidate_first_uninitialized":
                # Both placements intentionally share exactly the same TIR
                # node. The second definition cannot justify the first use.
                body = tir.SeqStmt([node.body, initial, node.body])
            return tir.For(node.loop_var, node.min, node.extent, node.kind, body,
                           node.thread_binding, node.annotations, node.step)
        if isinstance(node, tir.Block) and node.name_hint == "root":
            matches = list(node.match_buffers)
            if fault == "match_buffer_alias":
                alias_pointer = tir.Var("alias_pointer", tvm.ir.PointerType(tvm.ir.PrimType("int32"), scope))
                alias_view = tir.decl_buffer((1,), "int32", name="alias_view", data=alias_pointer, scope=scope)
                matches.append(tir.MatchBufferRegion(alias_view, tir.BufferRegion(scalar, [tvm.ir.Range(0, 1)])))
            return tir.Block(node.iter_vars, node.reads, node.writes, node.name_hint,
                             node.body, node.init,
                             [*node.alloc_buffers, scalar] if scope != "global" else node.alloc_buffers,
                             matches, node.annotations)
        return None

    func = module["main"]
    body = tir.stmt_functor.ir_transform(func.body, None, rewrite)
    if scope == "global":
        func = tir.PrimFunc([*func.params, scalar.data], body, func.ret_type,
                           {**dict(func.buffer_map), scalar.data: scalar}, func.attrs)
    else:
        func = func.with_body(body)
    module = tvm.IRModule.from_expr(func)
    return _tiled_clear(module) if tiled else module


def _tiled_clear(module, fault=None):
    """Turn 32x32 initialization into a mixed-radix map, independently of names."""
    def rewrite(node):
        if not isinstance(node, tir.For) or not node.loop_var.name.startswith("clear_col_"):
            return None
        outer = tir.Var("column_tile", "int32")
        inner = tir.Var("column_lane", "int32")
        stride = 9 if fault == "holes" else 8
        index = outer * stride + inner
        if fault == "offset":
            index = index + 1
        if fault == "shared_axis":
            # Same induction variable controls both dimensions: diagonal, not a full tile.
            index = node.body.indices[0]
        body = tir.stmt_functor.substitute(node.body, {node.loop_var: index})
        if fault == "conditional":
            body = tir.IfThenElse(inner < 7, body, None)
        body = tir.For(inner, 0, 7 if fault == "short" else 8,
                       tir.ForKind.VECTORIZED, body, step=2 if fault == "step" else 1)
        return tir.For(outer, 0, 4, tir.ForKind.SERIAL, body)
    return tvm.IRModule.from_expr(module["main"].with_body(
        tir.stmt_functor.ir_transform(module["main"].body, None, rewrite, ["tir.For"])))


@pytest.mark.parametrize("tiled", [False, True])
def test_readonly_local_scalar_address_proves_and_rewrites(tmp_path, monkeypatch, tiled):
    before = _natural_dynamic_address(tiled=tiled)
    analyzed, report = _analyze(before, tmp_path, monkeypatch, policy="forced")
    assert _accepted(report), report
    normalized = tl.transform.SunmmioMultidimRegion("auto")(analyzed)
    after = tl.transform.ApplySunmmioMultidimInterchange("auto")(normalized)
    cores = _marked(after, APPLIED)
    assert len(cores) == 1
    assert int(cores[0].extent) == 2
    calls = []
    tir.stmt_functor.post_order_visit(cores[0].body, lambda node: calls.append(node)
        if isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.Op)
        and node.op.name == "tl.mma_sunmmio" else None)
    assert len(calls) == 2
    assert len({call.args[2].args[0].buffer.data for call in calls}) == 2


@pytest.mark.parametrize("fault", [
    "suffix_write", "prefix_write", "alias_write", "async_write",
    "pointer_escape", "address_of_escape", "uninitialized",
    "conditional_initialization", "zero_trip_initialization",
    "global_metadata", "not_singleton", "out_of_bounds_scalar",
    "preceding_async_write", "match_buffer_alias",
    "shared_candidate_first_uninitialized",
])
def test_mutable_or_unproved_scalar_addresses_reject(tmp_path, monkeypatch, fault):
    before = _natural_dynamic_address(fault)
    after, report = _analyze(before, tmp_path, monkeypatch)
    assert not _accepted(report), (fault, report)
    assert _same_statements(before, after)


def test_mixed_radix_clear_is_complete(tmp_path, monkeypatch):
    before = _tiled_clear(_make_module(reuse_extent=2, fault="full_float_clear"))
    _, report = _analyze(before, tmp_path, monkeypatch, policy="forced")
    assert _accepted(report), report


@pytest.mark.parametrize("fault", ["holes", "offset", "short", "step", "conditional", "shared_axis"])
def test_partial_or_noninjective_tiled_clear_rejects(tmp_path, monkeypatch, fault):
    before = _tiled_clear(_make_module(reuse_extent=2, fault="full_float_clear"), fault)
    after, report = _analyze(before, tmp_path, monkeypatch)
    assert not _accepted(report), (fault, report)
    assert _same_statements(before, after)
