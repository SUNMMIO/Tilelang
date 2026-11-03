"""Whole-storage safety tests for automatic independent ODMA loop-entry carry."""

import importlib.util
from pathlib import Path

import pytest

from tilelang import tvm

_spec = importlib.util.spec_from_file_location("tile_entry_sync_helpers", Path(__file__).with_name("test_inject_sync.py"))
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
_apply, _dma = _helpers._apply, _helpers._dma
_pointer_var, _target = _helpers._pointer_var, _helpers._target


SETTING = "TL_SUNMMIO_SYNC_TILE_ENTRY_OVERLAP"


def _module(case="independent", *, extent=4, step=None, annotated=True, physical_guard=False):
    buffers = {}
    for name in ("input", "prefetch", "vector", "sink", "other", "acc"):
        data = _pointer_var(name)
        buffers[name] = tvm.tir.decl_buffer((32, 32), "bfloat16", name=name, data=data, scope="shared.rsram")
    q, v = buffers["prefetch"], buffers["vector"]
    tile = tvm.tir.Var("tile", "int32")
    zero = tvm.tir.IntImm("int32", 0)
    value = tvm.tir.BufferLoad(v, [tile, zero]) + tvm.tir.const(1, "bfloat16")
    target = v
    predicate = None
    before = [_dma(buffers["input"], q)]
    if case == "raw":
        value = tvm.tir.BufferLoad(q, [tile, zero])
    elif case == "waw":
        target = q
    elif case == "war":
        before = [_dma(v, q)]
    elif case == "view_alias":
        alias = tvm.tir.decl_buffer(
            (32, 32),
            "bfloat16",
            name="alias",
            data=q.data,
            scope="shared.rsram",
            elem_offset=32,
        )
        buffers["alias"] = alias
        value = tvm.tir.BufferLoad(alias, [tile, zero])
    elif case == "predicate_alias":
        predicate = tvm.tir.BufferLoad(q, [tile, zero]) != 0
    elif case == "load_predicate_alias":
        value = tvm.tir.BufferLoad(v, [tile, zero], predicate=tvm.tir.BufferLoad(q, [tile, zero]) != 0)
    elif case == "metadata_load":
        alias = tvm.tir.decl_buffer(
            (32, 32),
            "bfloat16",
            name="metadata_view",
            data=v.data,
            scope="shared.rsram",
            elem_offset=tvm.tir.Cast("int32", tvm.tir.BufferLoad(q, [zero, zero])),
        )
        buffers["alias"] = alias
        value = tvm.tir.BufferLoad(alias, [tile, zero])
    elif case == "outer_handle_alias":
        alias = tvm.tir.decl_buffer(
            (32, 32),
            "bfloat16",
            name="outer_view",
            data=_pointer_var("alias_pointer"),
            scope="shared.rsram",
        )
        buffers["alias"] = alias
        value = tvm.tir.BufferLoad(alias, [tile, zero])
    elif case == "index_alias":
        value = tvm.tir.BufferLoad(v, [tvm.tir.Cast("int32", tvm.tir.BufferLoad(q, [zero, zero])), zero])
    elif case == "same_unit_other_endpoint":
        before.append(_dma(buffers["other"], v))
    elif case == "different_units":
        before.append(_dma(buffers["other"], v, unit="odma1"))
    elif case == "header_load":
        extent = tvm.tir.Cast("int32", tvm.tir.BufferLoad(v, [zero, zero]))
    elif case == "pure_math":
        value = tvm.tir.exp(value)
    elif case in ("unknown_unit", "unknown_unit_conflict"):
        call = before[0].value
        before = [tvm.tir.Evaluate(tvm.tir.Call(call.dtype, call.op, call.args[:3]))]
        if case == "unknown_unit_conflict":
            before.append(_dma(buffers["other"], v, unit="odma0"))
    elif case == "independent_tc":
        before.append(
            _helpers._mma_regions(
                _helpers._region(buffers["other"], 1), _helpers._region(buffers["sink"], 1), _helpers._region(buffers["acc"], 3)
            )
        )
    elif case == "independent_vector":
        before.append(tvm.tir.BufferStore(buffers["acc"], tvm.tir.const(0, "bfloat16"), [zero, zero]))

    store = tvm.tir.BufferStore(target, value, [tile, zero], predicate=predicate)
    body = store
    if case == "unknown_call":
        body = tvm.tir.SeqStmt([tvm.tir.Evaluate(tvm.tir.call_extern("int32", "unknown_effect")), store])
    elif case == "pointer_escape":
        body = tvm.tir.SeqStmt([tvm.tir.Evaluate(q.data), store])
    elif case == "branch":
        body = tvm.tir.IfThenElse(tile < 2, store, None)
    elif case == "async_body":
        body = tvm.tir.SeqStmt([_dma(buffers["other"], q), store])
    elif case == "explicit_sync":
        body = tvm.tir.SeqStmt([tvm.tir.Evaluate(tvm.tir.call_intrin("handle", tvm.ir.Op.get("tl.sunmmio_sync"), 1)), store])
    elif case == "nested_plain_loop":
        inner = tvm.tir.Var("inner", "int32")
        body = tvm.tir.For(inner, 0, 1, tvm.tir.ForKind.SERIAL, store)
    elif case == "nested_tile_loop":
        inner = tvm.tir.Var("inner", "int32")
        body = tvm.tir.For(inner, 0, 1, tvm.tir.ForKind.VECTORIZED, store, annotations={"tile.interior": 1})
    loop = tvm.tir.For(
        tile,
        0,
        extent,
        tvm.tir.ForKind.SERIAL,
        body,
        annotations={"tile.execution_axis": 0} if annotated else {},
        step=tvm.tir.IntImm("int32", step) if step is not None else None,
    )
    after = [_dma(q, buffers["sink"], unit="odma1")]
    if case == "same_unit_reissue":
        after = [_dma(buffers["other"], buffers["sink"], unit="odma0")]
    sequence = tvm.tir.SeqStmt(before + [loop] + after)
    for buffer in reversed(list(buffers.values())):
        sequence = tvm.tir.DeclBuffer(buffer, sequence)
    if case == "outer_handle_alias":
        sequence = tvm.tir.LetStmt(buffers["alias"].data, q.data, sequence)
    func = tvm.tir.PrimFunc([buffers[name].data for name in buffers if name != "alias"], sequence)
    func = func.with_attr("global_symbol", "main").with_attr("tir.is_global_func", True)
    if case != "no_alias_contract":
        func = func.with_attr("tir.noalias", True)
    if case == "nonrestrict":
        func = func.with_attr("tl.non_restrict_params", [q.data])
    if physical_guard:
        func = func.with_attr("tl.sunmmio.multidim.applied", True)
    return tvm.tir.transform.BindTarget(_target())(tvm.IRModule({"main": func}))


def _events(module):
    result = []

    def visit(node):
        if isinstance(node, tvm.tir.For) and node.loop_var.name == "tile":
            result.append(("loop", 0))
        elif isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op):
            if node.op.name == "tl.sunmmio_sync":
                result.append(("sync", int(node.args[0])))
            elif node.op.name == "tl.dma_copy":
                result.append(("dma", 0))
        return True

    tvm.tir.stmt_functor.pre_order_visit(module["main"].body, visit)
    return result


@pytest.mark.parametrize("setting", ["0", "garbage"])
def test_explicit_disabled_preserves_original_entry_wait(monkeypatch, setting):
    monkeypatch.setenv(SETTING, setting)
    events = _events(_apply(_module()))
    assert events.index(("sync", 1)) < events.index(("loop", 0))


@pytest.mark.parametrize("setting", [None, "1"])
def test_default_and_explicit_enabled_defer_entry_wait(monkeypatch, setting):
    if setting is None:
        monkeypatch.delenv(SETTING, raising=False)
    else:
        monkeypatch.setenv(SETTING, setting)
    events = _events(_apply(_module()))
    assert events.index(("loop", 0)) < events.index(("sync", 1))


@pytest.mark.parametrize("case", ["independent", "pure_math", "nested_tile_loop", "unknown_unit"])
@pytest.mark.parametrize("extent", [0, 1, 4])
def test_independent_dma_is_waited_at_later_consumer(monkeypatch, case, extent):
    monkeypatch.setenv(SETTING, "1")
    events = _events(_apply(_module(case, extent=extent)))
    expected = 3 if case == "unknown_unit" else 1
    assert events.index(("loop", 0)) < events.index(("sync", expected))
    assert events[-2:] == [("sync", expected), ("dma", 0)]
    assert events.count(("sync", expected)) == 1


@pytest.mark.parametrize(
    "case",
    [
        "raw",
        "war",
        "waw",
        "view_alias",
        "predicate_alias",
        "load_predicate_alias",
        "metadata_load",
        "index_alias",
        "outer_handle_alias",
        "no_alias_contract",
        "nonrestrict",
        "same_unit_other_endpoint",
        "header_load",
        "unknown_call",
        "pointer_escape",
        "branch",
        "async_body",
        "explicit_sync",
        "nested_plain_loop",
    ],
)
def test_unproven_or_dependent_loop_keeps_entry_wait(monkeypatch, case):
    monkeypatch.setenv(SETTING, "1")
    events = _events(_apply(_module(case)))
    assert events.index(("sync", 1)) < events.index(("loop", 0))


def test_unannotated_loop_keeps_entry_wait(monkeypatch):
    monkeypatch.setenv(SETTING, "1")
    events = _events(_apply(_module(annotated=False)))
    assert events.index(("sync", 1)) < events.index(("loop", 0))


def test_units_are_partitioned_not_individual_accesses(monkeypatch):
    monkeypatch.setenv(SETTING, "1")
    events = _events(_apply(_module("different_units")))
    loop = events.index(("loop", 0))
    assert events.index(("sync", 2)) < loop < events.index(("sync", 1))


def test_later_same_engine_reissue_guard_is_preserved(monkeypatch):
    monkeypatch.setenv(SETTING, "1")
    events = _events(_apply(_module("same_unit_reissue", physical_guard=True)))
    assert events.index(("loop", 0)) < events.index(("sync", 1))
    assert events[-1] == ("dma", 0)


def test_loop_step_and_body_are_unchanged(monkeypatch):
    monkeypatch.setenv(SETTING, "1")
    before = _module(step=2)
    after = _apply(before)
    loops = []
    tvm.tir.stmt_functor.post_order_visit(after["main"].body, lambda node: loops.append(node) if isinstance(node, tvm.tir.For) else None)
    assert len(loops) == 1
    assert int(loops[0].step) == 2
    assert int(loops[0].extent) == 4


@pytest.mark.parametrize("case,mask", [("independent_tc", 4), ("independent_vector", 96)])
def test_non_odma_units_still_drain_at_entry(monkeypatch, case, mask):
    monkeypatch.setenv(SETTING, "1")
    events = _events(_apply(_module(case)))
    loop = events.index(("loop", 0))
    assert events.index(("sync", mask)) < loop < events.index(("sync", 1))


def test_unknown_unit_cannot_be_split_from_conflicting_endpoint(monkeypatch):
    monkeypatch.setenv(SETTING, "1")
    events = _events(_apply(_module("unknown_unit_conflict")))
    assert events.index(("sync", 3)) < events.index(("loop", 0))


@pytest.mark.parametrize("kind", ["let_expression", "match_buffer"])
def test_function_scope_alias_constructs_disable_carry(monkeypatch, kind):
    monkeypatch.setenv(SETTING, "1")
    module = _module()
    func = module["main"]
    source = tvm.tir.decl_buffer((32, 32), "bfloat16", name="source_view", data=func.params[1], scope="shared.rsram")
    if kind == "let_expression":
        pointer = _pointer_var("internal_pointer")
        expression = tvm.tir.Let(pointer, source.data, tvm.tir.const(0, "int32"))
        body = tvm.tir.SeqStmt([tvm.tir.Evaluate(expression), func.body])
    else:
        alias = tvm.tir.decl_buffer((32, 32), "bfloat16", name="alias_view", scope="shared.rsram")
        region = tvm.tir.BufferRegion(source, [tvm.ir.Range(0, 32), tvm.ir.Range(0, 32)])
        block = tvm.tir.Block([], [], [], "alias_scope", func.body, match_buffers=[tvm.tir.MatchBufferRegion(alias, region)])
        body = tvm.tir.BlockRealize([], True, block)
    module = tvm.IRModule({"main": func.with_body(body)})
    events = _events(_apply(module))
    assert events.index(("sync", 1)) < events.index(("loop", 0))
