import tilelang as tl
from tilelang import tvm
from tvm import arith, tir


OUTER = "tl.sunmmio.multidim.workitem_outer"
INNER = "tl.sunmmio.multidim.workitem_inner"


def _make_workitem_module(outer_extent=2, inner_extent=3):
    output = tir.decl_buffer((64,), "int32", name="output")
    expert = tir.Var("expert", "int32")
    bn = tir.Var("bn", "int32")
    bk = tir.Var("bk", "int32")
    index = expert * inner_extent + bn
    store = tir.BufferStore(output, expert * 100 + bn * 10 + bk, [index])
    core = tir.For(
        bk,
        0,
        2,
        tir.ForKind.SERIAL,
        store,
        annotations={"num_stages": tir.IntImm("int32", 2)},
    )
    inner = tir.For(
        bn,
        0,
        inner_extent,
        tir.ForKind.SERIAL,
        core,
        annotations={INNER: tir.IntImm("int32", 1)},
    )
    outer = tir.For(
        expert,
        0,
        outer_extent,
        tir.ForKind.SERIAL,
        inner,
        annotations={OUTER: tir.IntImm("int32", 1)},
    )
    func = tir.PrimFunc(
        [output.data], outer, buffer_map={output.data: output}
    ).with_attr("global_symbol", "main")
    return tvm.IRModule.from_expr(func)


def _tile_region(buffer, access, indices, extents):
    args = [tir.BufferLoad(buffer, indices), tir.IntImm("int32", access)]
    args.extend(tir.IntImm("int32", extent) for extent in extents)
    return tir.call_intrin("handle", tvm.ir.Op.get("tl.tileop.region"), *args)


def _make_a_residency_module(a_depends_on_bn=False, marked=True):
    expert = tir.Var("expert", "int32")
    bn = tir.Var("bn", "int32")
    bk = tir.Var("bk", "int32")

    def buffer(name, shape, scope):
        data = tir.Var(
            name,
            tvm.ir.PointerType(tvm.ir.PrimType("bfloat16"), scope),
        )
        return tir.decl_buffer(shape, "bfloat16", name=name, data=data, scope=scope)

    a = buffer("a", (2, 2, 32, 32), "global")
    b = buffer("b", (2, 2, 2, 32, 32), "global")
    a_stage = buffer("a_stage", (32, 32), "shared.rsram")
    a_shared = buffer("a_shared", (32, 32), "shared.asram")
    b_shared = buffer("b_shared", (32, 32), "shared.wsram")
    accumulator = buffer("accumulator", (2, 32, 32), "shared.rsram")
    output = buffer("output", (2, 32, 32), "shared.rsram")
    projection = buffer("projection", (4, 32, 32), "global")

    a_bn = bn if a_depends_on_bn else tir.IntImm("int32", 0)
    dma = tir.Evaluate(
        tir.Call(
            "handle",
            tvm.ir.Op.get("tl.dma_copy"),
            [
                _tile_region(a, 1, [expert, a_bn, 0, 0], [1, 1, 32, 32]),
                _tile_region(a_stage, 2, [0, 0], [32, 32]),
                tir.IntImm("int32", 0),
            ],
        )
    )
    a_broadcast = tir.Evaluate(
        tir.Call(
            "handle",
            tvm.ir.Op.get("tl.broadcast_"),
            [
                _tile_region(a_stage, 1, [0, 0], [32, 32]),
                _tile_region(a_shared, 2, [0, 0], [32, 32]),
                tir.IntImm("int32", 0),
                tir.IntImm("int64", 15),
                tir.IntImm("int32", 0),
            ],
        )
    )
    b_broadcast = tir.Evaluate(
        tir.Call(
            "handle",
            tvm.ir.Op.get("tl.broadcast_"),
            [
                _tile_region(b, 1, [expert, bk, bn, 0, 0], [1, 1, 1, 32, 32]),
                _tile_region(b_shared, 2, [0, 0], [32, 32]),
                tir.IntImm("int32", 1),
                tir.IntImm("int64", 15),
                tir.IntImm("int32", 0),
            ],
        )
    )
    mma = tir.Evaluate(
        tir.Call(
            "handle",
            tvm.ir.Op.get("tl.mma_sunmmio"),
            [
                _tile_region(a_shared, 1, [0, 0], [32, 32]),
                _tile_region(b_shared, 1, [0, 0], [32, 32]),
                _tile_region(accumulator, 3, [bn, 0, 0], [1, 32, 32]),
                tir.IntImm("bool", 0),
                tir.IntImm("bool", 0),
                bk == 0,
                tir.IntImm("int32", 0),
                _tile_region(output, 2, [bn, 0, 0], [1, 32, 32]),
                bk == 1,
            ],
        )
    )
    core = tir.For(
        bk,
        0,
        2,
        tir.ForKind.SERIAL,
        tir.SeqStmt([dma, a_broadcast, b_broadcast, mma]),
        annotations={"num_stages": tir.IntImm("int32", 2)},
    )
    suffix = tir.BufferStore(
        projection,
        tir.BufferLoad(output, [bn, 0, 0]),
        [expert * 2 + bn, 0, 0],
    )
    inner_annotations = {INNER: tir.IntImm("int32", 1)} if marked else {}
    outer_annotations = {OUTER: tir.IntImm("int32", 1)} if marked else {}
    inner = tir.For(
        bn,
        0,
        2,
        tir.ForKind.SERIAL,
        tir.SeqStmt([core, suffix]),
        annotations=inner_annotations,
    )
    outer = tir.For(
        expert,
        0,
        2,
        tir.ForKind.SERIAL,
        inner,
        annotations=outer_annotations,
    )
    # The independent input/output allocations are part of this legal fixture.
    # Generic loop interchange must not infer non-aliasing from variable names.
    buffers = [a, b, a_stage, a_shared, b_shared, accumulator, output, projection]
    func = tir.PrimFunc(
        [item.data for item in buffers],
        outer,
        buffer_map={item.data: item for item in buffers},
    ).with_attr("global_symbol", "main").with_attr("tir.noalias", True)
    return tvm.IRModule.from_expr(func)


def test_static_workitems_flatten_and_preserve_coordinates():
    result = tl.transform.FlattenSunmmioStaticWorkItems("forced")(
        _make_workitem_module()
    )["main"]
    assert bool(result.attrs["tl.sunmmio.multidim.workitem_flatten.applied"])
    device_keys = {str(key) for key in result.attrs["tl.device_func_attr_keys"]}
    assert "tl.sunmmio.multidim.workitem_flatten.applied" in device_keys

    flattened = result.body
    assert isinstance(flattened, tir.For)
    assert int(flattened.extent) == 6
    assert bool(flattened.annotations["tl.sunmmio.multidim.workitem.applied"])
    assert int(flattened.annotations["tl.sunmmio.multidim.workitem.outer_extent"]) == 2
    assert int(flattened.annotations["tl.sunmmio.multidim.workitem.inner_extent"]) == 3
    assert int(flattened.annotations["tl.sunmmio.multidim.workitem.total_extent"]) == 6
    assert OUTER not in flattened.annotations

    stores = []
    tir.stmt_functor.post_order_visit(
        flattened.body,
        lambda node: stores.append(node) if isinstance(node, tir.BufferStore) else None,
    )
    assert len(stores) == 1
    analyzer = arith.Analyzer()
    for logical in range(6):
        concrete = tir.stmt_functor.substitute(
            stores[0].indices[0],
            {flattened.loop_var: tir.IntImm("int32", logical)},
        )
        assert int(analyzer.simplify(concrete)) == logical


def test_dynamic_workitem_range_records_auto_fallback():
    extent = tir.Var("experts", "int32")
    result = tl.transform.FlattenSunmmioStaticWorkItems("auto")(
        _make_workitem_module(outer_extent=extent)
    )["main"]
    outer = result.body
    assert isinstance(outer, tir.For)
    assert not bool(outer.annotations["tl.sunmmio.multidim.workitem.applied"])
    assert str(outer.annotations["tl.sunmmio.multidim.workitem.fallback_reason"]) == "dynamic_workitem_range"


def test_unmarked_loop_is_unchanged_in_auto_mode():
    before = _make_workitem_module()
    outer = before["main"].body
    annotations = dict(outer.annotations)
    annotations.pop(OUTER)
    unmarked = tir.For(
        outer.loop_var,
        outer.min,
        outer.extent,
        outer.kind,
        outer.body,
        annotations=annotations,
    )
    before = tvm.IRModule.from_expr(before["main"].with_body(unmarked))
    after = tl.transform.FlattenSunmmioStaticWorkItems("auto")(before)
    assert tvm.ir.structural_equal(before, after, map_free_vars=True)


def test_a_residency_interchanges_static_bn_and_pipeline_core():
    result = tl.transform.FlattenSunmmioStaticWorkItems("auto")(
        _make_a_residency_module()
    )["main"]
    assert bool(result.attrs["tl.sunmmio.multidim.a_residency.applied"])
    assert "tl.sunmmio.multidim.workitem_flatten.applied" not in result.attrs
    device_keys = {str(key) for key in result.attrs["tl.device_func_attr_keys"]}
    assert "tl.sunmmio.multidim.a_residency.applied" in device_keys

    outer = result.body
    assert isinstance(outer, tir.For)
    assert int(outer.extent) == 2
    assert bool(outer.annotations["tl.sunmmio.multidim.a_residency.applied"])
    core = outer.body.seq[0]
    assert isinstance(core, tir.For)
    assert int(core.extent) == 2
    broadcasts = []
    tir.stmt_functor.post_order_visit(
        core.body,
        lambda node: broadcasts.append(node)
        if isinstance(node, tir.Call) and node.op.same_as(tvm.ir.Op.get("tl.broadcast_"))
        else None,
    )
    assert len(broadcasts) == 3


def test_a_residency_rejects_a_region_that_depends_on_bn():
    result = tl.transform.FlattenSunmmioStaticWorkItems("auto")(
        _make_a_residency_module(a_depends_on_bn=True)
    )["main"]
    assert "tl.sunmmio.multidim.a_residency.applied" not in result.attrs
    assert bool(result.attrs["tl.sunmmio.multidim.workitem_flatten.applied"])
    assert isinstance(result.body, tir.For)
    assert int(result.body.extent) == 4


def test_auto_discovers_unmarked_static_workitems():
    result = tl.transform.FlattenSunmmioStaticWorkItems("auto")(
        _make_a_residency_module(marked=False)
    )["main"]
    assert bool(result.attrs["tl.sunmmio.multidim.a_residency.applied"])
    assert int(result.attrs["tl.sunmmio.multidim.workitem.auto_discovered"]) == 1
    outer = result.body
    assert isinstance(outer, tir.For)
    assert bool(outer.annotations["tl.sunmmio.multidim.workitem.auto_discovered"])
    core = outer.body.seq[0]
    assert isinstance(core, tir.For)
    broadcasts = []
    tir.stmt_functor.post_order_visit(
        core.body,
        lambda node: broadcasts.append(node)
        if isinstance(node, tir.Call) and node.op.same_as(tvm.ir.Op.get("tl.broadcast_"))
        else None,
    )
    assert len(broadcasts) == 3


def test_auto_discovery_rejects_variant_shared_transfer():
    before = _make_a_residency_module(a_depends_on_bn=True, marked=False)
    after = tl.transform.FlattenSunmmioStaticWorkItems("auto")(before)
    assert tvm.ir.structural_equal(before, after, map_free_vars=True)


def test_forced_mode_still_requires_explicit_workitem_annotations():
    try:
        tl.transform.FlattenSunmmioStaticWorkItems("forced")(
            _make_a_residency_module(marked=False)
        )
    except tvm.TVMError as error:
        assert "no_marked_workitem_nest" in str(error)
    else:
        raise AssertionError("forced mode must require explicit WorkItem annotations")
