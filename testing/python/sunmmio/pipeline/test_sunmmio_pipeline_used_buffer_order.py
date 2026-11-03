"""Stable ILP used_buffers export; TIR-only, never backend or simulator.

The internal read-only hook invokes the production exporter so duplicate names
and storage views are tested independently of the solver's existing model.
The final cases also exercise the real planner and dynamic-short-loop injector.
"""

import itertools

import pytest
import tilelang as tl
from tilelang import tvm
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from tvm import tir


def _buffer(name, *, data=None, shape=(32,), strides=None, offset=0, scope="shared.rsram"):
    return tir.decl_buffer(shape, "float32", name=name, data=data, strides=strides,
                           elem_offset=offset, scope=scope)


def _function(body, *, allocations=(), params=(), buffer_map=None):
    root = tir.Block([], [], [], "root", body, alloc_buffers=list(allocations))
    return tir.PrimFunc(list(params), tir.BlockRealize([], True, root),
                        buffer_map=buffer_map or {})


def _ordered(func, required):
    before = tvm.ir.save_json(func)
    result = tvm.get_global_func("tl.testing.SunmmioILPStableUsedBuffers")(func, list(required))
    assert tvm.ir.save_json(func) == before, "the read-only exporter changed IR"
    return result


def _assert_identities(actual, expected):
    assert actual is not None
    assert len(actual) == len(expected)
    assert all(a.same_as(b) for a, b in zip(actual, expected))


@pytest.mark.parametrize("creation_order", list(itertools.permutations(range(3))))
@pytest.mark.parametrize("duplicate_names", [False, True])
def test_allocation_order_not_heap_order_or_name(creation_order, duplicate_names):
    buffers = {}
    for index in creation_order:
        buffers[index] = _buffer("same" if duplicate_names else ["z", "a", "m"][index])
    semantic_order = [buffers[2], buffers[0], buffers[1]]
    func = _function(tir.Evaluate(0), allocations=semantic_order)
    # The request enumeration is deliberately unrelated to definition order.
    _assert_identities(_ordered(func, [buffers[1], buffers[2], buffers[0]]), semantic_order)


def test_parameter_order_not_buffer_map_iteration():
    a, b = _buffer("z", scope="global"), _buffer("a", scope="global")
    func = _function(tir.Evaluate(0), params=[a.data, b.data],
                     buffer_map={b.data: b, a.data: a})
    _assert_identities(_ordered(func, [b, a]), [a, b])


def test_same_data_different_views_are_not_deduplicated_or_rebuilt():
    base = _buffer("storage", shape=(128,))
    a = _buffer("same", data=base.data, strides=[2], offset=1)
    b = _buffer("same", data=base.data, strides=[3], offset=2)
    statement = tir.BufferStore(a, tir.BufferLoad(b, [0]), [0])
    func = _function(statement, allocations=[base])
    _assert_identities(_ordered(func, [b, a, base]), [base, a, b])
    assert a.data.same_as(b.data) and not a.same_as(b)
    assert int(a.strides[0]) == 2 and int(b.elem_offset) == 2


def test_required_subset_and_repeated_accesses_preserve_exact_inventory():
    a, b, unused = _buffer("a"), _buffer("b"), _buffer("unused")
    func = _function(tir.SeqStmt([tir.Evaluate(a[0]), tir.Evaluate(b[0]),
                                 tir.Evaluate(a[1])]), allocations=[unused, b, a])
    _assert_identities(_ordered(func, [a, b, a]), [b, a])
    _assert_identities(_ordered(func, []), [])


def test_nested_definitions_are_lexical_not_annotation_order():
    outer, first, second = _buffer("same"), _buffer("same"), _buffer("same")
    k = tir.Var("k", "int32")
    nested = tir.Block([], [], [], "nested", tir.Evaluate(first[0]),
                       alloc_buffers=[first, second],
                       annotations={"used_buffers": [second, outer, first]})
    body = tir.For(k, 0, 2, tir.ForKind.SERIAL, tir.BlockRealize([], True, nested),
                    annotations={"used_buffers": [second, first, outer]})
    func = _function(body, allocations=[outer])
    _assert_identities(_ordered(func, [second, first, outer]), [outer, first, second])


@pytest.mark.parametrize("access", ["load", "store"])
def test_predicate_only_buffer_access_is_visited(access):
    value, predicate_buffer = _buffer("value"), _buffer("predicate")
    predicate = tir.BufferLoad(predicate_buffer, [0]) > tir.const(0, "float32")
    if access == "load":
        body = tir.Evaluate(tir.BufferLoad(value, [0], predicate=predicate))
    else:
        body = tir.BufferStore(value, tir.const(0, "float32"), [0], predicate=predicate)
    func = _function(body)
    _assert_identities(_ordered(func, [predicate_buffer, value]), [value, predicate_buffer])


def test_call_arguments_and_index_expressions_are_visited_in_ast_order():
    a, b, index = _buffer("same"), _buffer("same"), tir.decl_buffer((1,), "int32")
    expr = tir.call_extern("int32", "not_executed", a[index[0]], b[0])
    func = _function(tir.Evaluate(expr))
    _assert_identities(_ordered(func, [b, index, a]), [a, index, b])


@pytest.mark.parametrize("declaration", ["decl", "realize", "block_regions", "match"])
def test_ordered_buffer_declaration_and_region_fields(declaration):
    a, b = _buffer("same"), _buffer("same")
    region = tir.BufferRegion(b, [tvm.ir.Range.from_min_extent(0, 32)])
    if declaration == "decl":
        body = tir.DeclBuffer(a, tir.Evaluate(b[0]))
    elif declaration == "realize":
        body = tir.BufferRealize(a, [tvm.ir.Range.from_min_extent(0, 32)], True,
                                 tir.Evaluate(b[0]))
    elif declaration == "block_regions":
        read = tir.BufferRegion(a, [tvm.ir.Range.from_min_extent(0, 32)])
        body = tir.BlockRealize([], True, tir.Block([], [read], [region], "regions", tir.Evaluate(0)))
    else:
        body = tir.BlockRealize([], True, tir.Block([], [], [], "match", tir.Evaluate(0),
            match_buffers=[tir.MatchBufferRegion(a, region)]))
    _assert_identities(_ordered(_function(body), [b, a]), [a, b])


@pytest.mark.parametrize("location", ["absent", "function", "block", "loop", "call", "attr"])
def test_unknown_metadata_only_identity_is_refused(location):
    visible, ghost = _buffer("same"), _buffer("same")
    body = tir.Evaluate(visible[0])
    if location == "block":
        body = tir.BlockRealize([], True, tir.Block([], [], [], "meta", body,
                                                  annotations={"hidden": ghost}))
    elif location == "loop":
        body = tir.For(tir.Var("i", "int32"), 0, 2, tir.ForKind.SERIAL, body,
                       annotations={"used_buffers": [ghost, visible]})
    elif location == "call":
        body = tir.Evaluate(tir.Call("int32", tvm.ir.Op.get("tir.call_extern"),
            [tir.StringImm("not_executed"), visible[0]], annotations={"hidden": ghost[0]}))
    elif location == "attr":
        body = tir.AttrStmt(ghost, "hidden", 1, body)
    func = _function(body)
    if location == "function":
        func = func.with_attr("hidden", ghost)
    assert _ordered(func, [visible, ghost]) is None
    # A known subset still succeeds; the unknown identity is never appended.
    _assert_identities(_ordered(func, [visible]), [visible])


def test_same_data_unknown_view_is_not_replaced_by_known_definition():
    base = _buffer("same")
    ghost = _buffer("same", data=base.data, strides=[2], offset=1)
    func = _function(tir.Evaluate(base[0]), allocations=[base])
    assert _ordered(func, [base, ghost]) is None


def _region(buffer, indices, access, extents):
    return tir.Call("handle", tvm.ir.Op.get("tl.tileop.region"),
                    [tir.BufferLoad(buffer, indices), access, *extents])


def _dma(source, destination):
    return tir.Evaluate(tir.Call("handle", tvm.ir.Op.get("tl.dma_copy"),
                                 [source, destination, 0]))


def _dynamic_pipeline(reverse_creation):
    source = _buffer("input", shape=(8, 256), scope="global")
    output = _buffer("output", shape=(2, 8, 256), scope="global")
    local = {}
    for name in (["a_second", "z_first"] if reverse_creation else ["z_first", "a_second"]):
        local[name] = _buffer(name, shape=(256,))
    a, b = local["z_first"], local["a_second"]
    n, k = tir.Var("n", "int32"), tir.Var("k", "int32")
    body = tir.SeqStmt([
        _dma(_region(source, [k, 0], 1, [1, 256]), _region(b, [0], 2, [256])),
        _dma(_region(source, [k, 0], 1, [1, 256]), _region(a, [0], 2, [256])),
        _dma(_region(a, [0], 1, [256]), _region(output, [0, k, 0], 2, [1, 1, 256])),
        _dma(_region(b, [0], 1, [256]), _region(output, [1, k, 0], 2, [1, 1, 256])),
    ])
    loop = tir.For(k, 0, n, tir.ForKind.SERIAL, body,
                   annotations={"num_stages": tir.IntImm("int32", 2)})
    func = _function(loop, allocations=[a, b], params=[source.data, output.data, n],
                     buffer_map={source.data: source, output.data: output})
    return func.with_attr("global_symbol", "main"), [a, b]


@pytest.mark.parametrize("reverse_creation", [False, True])
def test_real_dynamic_planner_and_injector_keep_stable_complete_array(monkeypatch, reverse_creation):
    # No backend/gem5. The dynamic fallback deliberately keeps its metadata;
    # this test must pass without deleting it or weakening an equality gate.
    for key in ("TL_SUNMMIO_ILP_PROBLEM_JSON", "TL_SUNMMIO_ILP_SOLUTION_JSON"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TL_SUNMMIO_FASTER", "20")
    monkeypatch.setenv("HIGHS_THREADS", "1")
    func, required = _dynamic_pipeline(reverse_creation)
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = tir.transform.BindTarget(target)(tvm.IRModule.from_expr(func))
    with target, tvm.transform.PassContext(config={"tl.sunmmio_ilp_stage_shrink": False}):
        mod = tl.transform.ResolveSunmmioUnit()(mod)
        planned = tl.transform.SunmmioPipelinePlanningILP(debug=False)(mod)
        planned_loops = []
        tir.stmt_functor.post_order_visit(planned["main"].body,
            lambda node: planned_loops.append(node) if isinstance(node, tir.For)
            and "used_buffers" in node.annotations else None)
        assert len(planned_loops) == 1
        assert bool(planned_loops[0].annotations["tl.sunmmio.pipeline.applied"])
        _assert_identities(planned_loops[0].annotations["used_buffers"], required)
        injected = tl.transform.InjectSunmmioPipelineILP()(planned)
    fallbacks = []
    tir.stmt_functor.post_order_visit(injected["main"].body,
        lambda node: fallbacks.append(node) if isinstance(node, tir.For)
        and str(node.annotations.get("tl.sunmmio.pipeline.fallback_reason", ""))
        == "runtime_short_extent" else None)
    assert len(fallbacks) == 1
    exported = fallbacks[0].annotations["used_buffers"]
    assert len(exported) == 2
    _assert_identities(exported, _ordered(injected["main"], exported))
    assert "iterations" not in fallbacks[0].annotations
    assert str(fallbacks[0].annotations["tl.sunmmio.pipeline.fallback_reason"]) == "runtime_short_extent"
