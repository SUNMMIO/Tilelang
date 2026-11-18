"""Structural positive/negative tests for the isolated WSRAM DMA experiment."""

import pytest
import tilelang
from tilelang import tvm
from testing.python.sunmmio.transform.test_inject_sync import (
    _target,
    _pointer_var,
    _dma,
)


def module(mode="safe"):
    t = tvm.tir

    def buffer(name, scope):
        return t.decl_buffer((32, 32), "bfloat16", name=name, data=_pointer_var(name, scope=scope), scope=scope)

    src = buffer("dram", "global.dram")
    w = buffer("w", "shared.wsram")
    r = buffer("r", "shared.rsram")
    out = buffer("out", "global.dram")
    i, j = t.Var("i", "int32"), t.Var("j", "int32")
    value = r[i, j] + t.const(1, "bfloat16")
    if mode == "if_then_else":
        value = t.if_then_else(j < 16, value, t.const(0, "bfloat16"))
    if mode == "read_wsram":
        value = w[i, j]
    if mode == "alias_wsram":
        alias = t.decl_buffer((32, 32), "bfloat16", data=w.data, scope="shared.wsram", elem_offset=0)
        value = alias[i, j]
    if mode == "read_dram":
        value = src[i, j]
    if mode == "unknown_call":
        value = t.call_pure_extern("bfloat16", "unknown", value)
    if mode == "pointer_escape":
        value = t.call_pure_extern("bfloat16", "escape", w.data)
    store = t.BufferStore(w if mode == "write_wsram" else src if mode == "write_dram" else r, value, [i, j])
    if mode in ("sync", "barrier", "nested_dma"):
        effect = (
            _dma(src, w)
            if mode == "nested_dma"
            else t.Evaluate(
                t.Call(
                    "handle", tvm.ir.Op.get("tl.sunmmio_sync" if mode == "sync" else "tl.barrier_arrive_and_wait"), [t.IntImm("int32", 1)]
                )
            )
        )
        store = t.SeqStmt([store, effect])
    if mode == "branch":
        store = t.IfThenElse(i < 16, store, None)
    if mode in ("fused_reduce", "reduce_wsram", "alloc_wsram"):
        temp = buffer("temp", "shared.wsram" if mode == "alloc_wsram" else "shared.rsram")
        reduced = t.decl_buffer((32,), "bfloat16", data=r.data, scope="shared.rsram")
        from testing.python.sunmmio.transform.test_inject_sync import _region

        dst_region = t.Call("handle", tvm.ir.Op.get("tl.tileop.region"), [reduced[0], t.IntImm("int32", 2), t.IntImm("int32", 32)])
        reduce = t.Evaluate(
            t.Call(
                "handle",
                tvm.ir.Op.get("tl.vector_core_in_tile_reduce"),
                [t.StringImm("max"), dst_region, _region(w if mode == "reduce_wsram" else temp, 1), t.IntImm("int32", 1)],
            )
        )
        store = t.SeqStmt(
            [
                store,
                t.Allocate(
                    temp.data,
                    temp.dtype,
                    temp.shape,
                    t.const(True),
                    t.DeclBuffer(temp, t.IfThenElse(j == 31, reduce, None)),
                    annotations={"tile.sunmmio_reduce_register_temp": 1},
                ),
            ]
        )
    inner = t.For(j, 0, 32, t.ForKind.SERIAL, store)
    annotations = {"tile.domain": [32, 32]}
    if mode == "pipeline":
        annotations["software_pipeline_stage"] = [0]
    if mode == "unknown_annotation":
        annotations["unknown.schedule"] = 1
    if mode == "ordinary_loop":
        annotations = {}
    minimum = r[0, 0].astype("int32") if mode == "header_load" else 0
    loop = t.For(i, minimum, 0 if mode in ("zero_extent", "terminal_zero") else 32, t.ForKind.SERIAL, inner, annotations=annotations)
    before = [_dma(src, r if mode == "rsram_dma" else w, unit="odma1" if mode == "odma1" else "odma0")]
    other_src = buffer("other_dram", "global.dram")
    if mode == "mixed_unit_accesses":
        before.append(_dma(other_src, r))
    after = [] if mode in ("terminal", "terminal_zero") else [_dma(src, w) if mode == "overwrite" else _dma(w, out, unit="odma1")]
    body = t.SeqStmt([*before, loop, *after])
    if mode != "external_rsram":
        body = t.Allocate(r.data, r.dtype, r.shape, t.const(True), body)
    if mode != "external_wsram":
        body = t.Allocate(w.data, w.dtype, w.shape, t.const(True), body)
    params = [src.data, out.data]
    if mode == "mixed_unit_accesses":
        params.append(other_src.data)
    if mode == "external_rsram":
        params.append(r.data)
    if mode == "external_wsram":
        params.append(w.data)
    func = t.PrimFunc(params, body).with_attr("global_symbol", "main").with_attr("tir.is_global_func", True)
    return t.transform.BindTarget(_target())(tvm.IRModule({"main": func}))


def top_level(mode):
    mod = tilelang.transform.InjectSunmmioSync()(module(mode))
    body = mod["main"].body
    while isinstance(body, tvm.tir.Allocate):
        body = body.body
    if isinstance(body, tvm.tir.SeqStmt) and any(isinstance(s, tvm.tir.Allocate) for s in body.seq):
        body = next(s for s in body.seq if isinstance(s, tvm.tir.Allocate))
        while isinstance(body, tvm.tir.Allocate):
            body = body.body
    seq = list(body.seq)
    index = next(i for i, s in enumerate(seq) if isinstance(s, tvm.tir.For))

    def mask(stmts):
        total = 0
        for s in stmts:
            if isinstance(s, tvm.tir.Evaluate) and isinstance(s.value, tvm.tir.Call) and s.value.op == tvm.ir.Op.get("tl.sunmmio_sync"):
                total |= s.value.args[0].value
        return total

    return mask(seq[:index]), mask(seq[index + 1 :]), seq[index]


@pytest.mark.parametrize(
    "mode", ["safe", "if_then_else", "zero_extent", "branch", "fused_reduce", "terminal", "overwrite", "terminal_zero"]
)
def test_dma_wait_crosses_loop_but_precedes_consumer(mode):
    before, after, loop = top_level(mode)
    assert not before & 1
    assert after & 1
    calls = []
    tvm.tir.stmt_functor.post_order_visit(
        loop, lambda n: calls.append(n) if isinstance(n, tvm.tir.Call) and n.op == tvm.ir.Op.get("tl.sunmmio_sync") else None
    )
    assert calls  # Existing vector/RSRAM backedge waits remain.
    assert all(not n.args[0].value & 1 for n in calls)


@pytest.mark.parametrize(
    "mode",
    [
        "read_wsram",
        "write_wsram",
        "read_dram",
        "write_dram",
        "unknown_call",
        "pointer_escape",
        "sync",
        "barrier",
        "nested_dma",
        "pipeline",
        "unknown_annotation",
        "ordinary_loop",
        "header_load",
        "rsram_dma",
        "mixed_unit_accesses",
        "external_rsram",
        "external_wsram",
        "reduce_wsram",
        "alloc_wsram",
        "alias_wsram",
    ],
)
def test_unproved_loop_retains_entry_wait(mode):
    before, _, _ = top_level(mode)
    assert before & 1


def test_other_dma_unit_is_not_changed():
    before, _, _ = top_level("odma1")
    assert before & 2
