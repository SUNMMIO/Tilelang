"""Move drains only for proven independent maps over distinct local allocations."""

import pytest
import tilelang
from tilelang import tvm
from testing.python.sunmmio.transform.test_inject_sync import _target, _pointer_var, _dma_regions


def module(mode="independent"):
    tir = tvm.tir
    a = tir.decl_buffer((1024,), "bfloat16", data=_pointer_var("a"), scope="shared.rsram", name="a")
    b = tir.decl_buffer(
        (1024,),
        "bfloat16",
        data=a.data if mode == "alias" else _pointer_var("b"),
        scope="shared.rsram",
        name="b",
        strides=[1] if mode == "explicit_stride" else None,
        elem_offset=1 if mode == "offset_view" else 0,
    )
    group, lane = tir.Var("group", "int32"), tir.Var("lane", "int32")
    stride = 128 if mode == "overlap" else 256
    index = group * stride + lane
    value = a[index]
    if mode == "unknown_call":
        value = tir.call_pure_extern("bfloat16", "unknown", value)
    if mode == "read_output":
        value = value + b[index]
    store = tir.BufferStore(b, value, [index])
    if mode == "branch":
        store = tir.IfThenElse(group < 2, store, None)
    tile_annotations = {"tile.domain": [256]}
    if mode == "inner_pipeline":
        tile_annotations["software_pipeline_stage"] = [0]
    tile = tir.For(lane, 0, 256, tir.ForKind.PARALLEL, store, annotations=tile_annotations)
    annotations = {}
    if mode == "outer_pipeline":
        annotations["software_pipeline_stage"] = [0]
    if mode == "unknown_annotation":
        annotations["unknown.schedule"] = 1
    extent = 5 if mode == "out_of_bounds" else 0 if mode == "zero_extent" else 3 if mode == "nonzero_min" else 4
    minimum = -1 if mode == "negative_min" else 1 if mode == "nonzero_min" else 0
    loop = tir.For(group, minimum, extent, tir.ForKind.SERIAL, tile, annotations=annotations)
    body = loop
    params = [a.data, b.data] if mode == "external" else []
    if mode == "dma":
        source = tir.decl_buffer((1024,), "bfloat16", data=_pointer_var("source", scope="global.dram"), name="source")
        destination = tir.decl_buffer((1024,), "bfloat16", data=_pointer_var("destination", scope="global.dram"), name="destination")

        def region(buffer, access):
            return tir.Call(
                "handle", tvm.ir.Op.get("tl.tileop.region"), [buffer[0], tir.IntImm("int32", access), tir.IntImm("int32", 1024)]
            )

        body = tir.SeqStmt([_dma_regions(region(source, 1), region(a, 2)), loop, _dma_regions(region(b, 1), region(destination, 2))])
        params = [source.data, destination.data]
    if mode != "external":
        if mode != "alias":
            body = tir.Allocate(b.data, b.dtype, b.shape, tir.const(True), body)
        body = tir.Allocate(a.data, a.dtype, a.shape, tir.const(True), body)
    function = tir.PrimFunc(params, body).with_attr("global_symbol", "main").with_attr("tir.is_global_func", True)
    return tir.transform.BindTarget(_target())(tvm.IRModule({"main": function}))


def syncs_in_loop(mod, loop_name):
    loops, syncs = [], []

    def visit(node):
        if isinstance(node, tvm.tir.For) and node.loop_var.name == loop_name:
            loops.append(node)

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, visit)
    assert len(loops) == 1

    def find(node):
        if isinstance(node, tvm.tir.Call) and node.op == tvm.ir.Op.get("tl.sunmmio_sync"):
            syncs.append(node)

    tvm.tir.stmt_functor.post_order_visit(loops[0].body, find)
    return syncs


@pytest.mark.parametrize("mode", ["independent", "nonzero_min"])
def test_independent_local_map_has_one_exit_drain(mode):
    result = tilelang.transform.InjectSunmmioSync()(module(mode))
    assert not syncs_in_loop(result, "group")
    assert result.script().count("T.sunmmio_sync(") == 1
    body = result["main"].body
    while isinstance(body, tvm.tir.Allocate):
        body = body.body
    assert isinstance(body, tvm.tir.SeqStmt)
    assert isinstance(body.seq[0], tvm.tir.For)
    assert isinstance(body.seq[-1], tvm.tir.Evaluate)
    assert body.seq[-1].value.op == tvm.ir.Op.get("tl.sunmmio_sync")


def test_dma_producer_and_consumer_still_have_boundary_waits():
    result = tilelang.transform.InjectSunmmioSync()(module("dma"))
    assert not syncs_in_loop(result, "group")
    body = result["main"].body
    while isinstance(body, tvm.tir.Allocate):
        body = body.body
    assert isinstance(body, tvm.tir.SeqStmt)
    operations = [stmt.value.op.name if isinstance(stmt, tvm.tir.Evaluate) else "loop" for stmt in body.seq]
    assert operations == ["tl.dma_copy", "tl.sunmmio_sync", "loop", "tl.sunmmio_sync", "tl.dma_copy"]
    assert body.seq[1].value.args[0].value == 1  # ODMA0 before the first tile load.
    assert body.seq[3].value.args[0].value != 1  # Vector/RSRAM before output DMA.


@pytest.mark.parametrize(
    "mode",
    [
        "alias",
        "overlap",
        "read_output",
        "branch",
        "external",
        "out_of_bounds",
        "outer_pipeline",
        "inner_pipeline",
        "unknown_annotation",
        "unknown_call",
        "zero_extent",
        "negative_min",
        "explicit_stride",
        "offset_view",
    ],
)
def test_unproved_maps_keep_backedge_drains(mode):
    result = tilelang.transform.InjectSunmmioSync()(module(mode))
    assert syncs_in_loop(result, "group")
