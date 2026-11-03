"""Analysis-only fallback metadata must not create execution loop boundaries."""

import importlib.util
from pathlib import Path

import pytest
import tilelang
from tilelang import tvm


DIAGNOSTICS = {
    "tl.sunmmio.multidim.selected_mode": "ilp",
    "tl.sunmmio.multidim.fallback_stage": "generic_residency",
    "tl.sunmmio.multidim.fallback_reason": "reuse_extent_too_short",
    "tl.sunmmio.residency.reject_reason": "reuse_extent_too_short",
}


def lower(annotations, *, extent=1, minimum=0, thread=False):
    buf = tvm.tir.decl_buffer((16,), "int32", name="output")
    index = tvm.tir.Var("index", "int32")
    binding = tvm.tir.IterVar(tvm.ir.Range(0, 1), index,
                            tvm.tir.IterVar.ThreadIndex, "threadIdx.x") if thread else None
    loop = tvm.tir.For(index, minimum, extent,
                      tvm.tir.ForKind.THREAD_BINDING if thread else tvm.tir.ForKind.SERIAL,
                      tvm.tir.BufferStore(buf, index + 1, [index]),
                      thread_binding=binding, annotations=annotations)
    params = [buf.data] + ([extent] if isinstance(extent, tvm.tir.Var) else [])
    func = tvm.tir.PrimFunc(params, loop, buffer_map={buf.data: buf})
    return tilelang.transform.LowerOpaqueBlock()(tvm.IRModule({"main": func}))["main"]


@pytest.mark.parametrize("annotations", [
    {}, {"pragma_unroll_explicit": 1}, DIAGNOSTICS,
    {**DIAGNOSTICS, "pragma_unroll_explicit": 1},
    *[{key: value} for key, value in DIAGNOSTICS.items()],
])
@pytest.mark.parametrize("minimum", [0, 3])
def test_diagnostic_unit_loop_is_erased(annotations, minimum):
    func = lower(annotations, minimum=minimum)
    assert isinstance(func.body, tvm.tir.BufferStore)
    assert int(func.body.indices[0]) == minimum
    assert tvm.arith.Analyzer().simplify(func.body.value).value == minimum + 1
    assert tvm.ir.structural_equal(func, lower({}, minimum=minimum), map_free_vars=True)


@pytest.mark.parametrize("key,value", [
    ("tile.execution_axis", 0), ("tile.interior", 1), ("num_stages", 2),
    ("software_pipeline_stage", [0]), ("tl.sunmmio.multidim.region", 1),
    ("tl.sunmmio.multidim.inner_stage_count", 2),
    ("tl.sunmmio.collective_order", 1), ("tl.sunmmio.bank", 1),
    ("tl.sunmmio.multidim.unknown_semantics", 1), ("unknown_attribute", 1),
])
def test_real_or_unknown_annotation_preserves_loop(key, value):
    func = lower({**DIAGNOSTICS, key: value})
    assert isinstance(func.body, tvm.tir.For)
    assert key in func.body.annotations
    for diagnostic in DIAGNOSTICS:
        assert diagnostic in func.body.annotations


@pytest.mark.parametrize("extent", [0, 2, tvm.tir.Var("extent", "int32")])
def test_nonunit_loop_and_diagnostics_preserved(extent):
    func = lower(DIAGNOSTICS, extent=extent)
    assert isinstance(func.body, tvm.tir.For)
    if isinstance(extent, tvm.tir.Var):
        assert func.body.extent.same_as(extent)
    else:
        assert int(func.body.extent) == extent
    assert all(key in func.body.annotations for key in DIAGNOSTICS)


def test_thread_launch_is_not_erased():
    func = lower(DIAGNOSTICS, thread=True)
    assert isinstance(func.body, tvm.tir.AttrStmt)
    assert func.body.attr_key == "thread_extent"


def test_fallback_wrapper_does_not_prevent_safe_odma_carry(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "fallback_tile_helpers", Path(__file__).with_name("test_sunmmio_tile_entry_overlap.py"))
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    monkeypatch.setenv(helpers.SETTING, "1")
    baseline = helpers._module()
    index = tvm.tir.Var("diagnostic_only", "int32")
    wrapped = baseline["main"].with_body(tvm.tir.For(
        index, 0, 1, tvm.tir.ForKind.SERIAL, baseline["main"].body,
        annotations=DIAGNOSTICS))
    lowered = tilelang.transform.LowerOpaqueBlock()(tvm.IRModule({"main": wrapped}))
    actual = helpers._apply(lowered)
    expected = helpers._apply(tilelang.transform.LowerOpaqueBlock()(baseline))
    assert tvm.ir.structural_equal(actual, expected)
    events = helpers._events(actual)
    assert events.index(("loop", 0)) < events.index(("sync", 1))
    assert events[-2:] == [("sync", 1), ("dma", 0)]
