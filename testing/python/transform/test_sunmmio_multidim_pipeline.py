"""Focused region-discovery tests for the SunMMIO multidimensional ILP mode."""


import pytest
import tilelang as tl
from tilelang import tvm
from tvm import tir


def _make_nested_pipeline(
    *,
    outer_min=0,
    outer_extent=2,
    inner_min=0,
    inner_extent=3,
    inner_extra=None,
):
    outer = tir.Var("outer", "int32")
    inner = tir.Var("inner", "int32")
    inner_body = [tir.Evaluate(outer * 100 + inner)]
    if inner_extra is not None:
        inner_body.insert(0, inner_extra)
    inner_body = inner_body[0] if len(inner_body) == 1 else tir.SeqStmt(inner_body)
    inner_loop = tir.For(
        inner,
        inner_min,
        inner_extent,
        tir.ForKind.SERIAL,
        inner_body,
        annotations={"num_stages": 2},
    )
    outer_loop = tir.For(
        outer,
        outer_min,
        outer_extent,
        tir.ForKind.SERIAL,
        inner_loop,
    )
    params = [value for value in (outer_extent, inner_extent) if isinstance(value, tir.Var)]
    func = tir.PrimFunc(params, outer_loop).with_attr("global_symbol", "main")
    return tvm.IRModule.from_expr(func)


def _tile_region(buffer, access, indices, extents):
    return tir.Call(
        "handle",
        tvm.ir.Op.get("tl.tileop.region"),
        [tir.BufferLoad(buffer, indices), tir.IntImm("int32", access), *extents],
    )




def _normalized_outer(mod):
    loop = mod["main"].body
    assert isinstance(loop, tir.For)
    return loop


def _inner_coordinates(stmt):
    coordinates = []

    def visit(node):
        if isinstance(node, tir.AttrStmt) and node.attr_key == "tl.sunmmio.multidim.inner_coordinate":
            coordinates.append(int(node.value))

    tir.stmt_functor.post_order_visit(stmt, visit)
    return coordinates


def _with_outer_annotations(mod, updates):
    loop = _normalized_outer(mod)
    annotations = dict(loop.annotations)
    annotations.update(updates)
    replacement = tir.For(
        loop.loop_var,
        loop.min,
        loop.extent,
        loop.kind,
        loop.body,
        loop.thread_binding,
        annotations,
        loop.step,
    )
    return tvm.IRModule.from_expr(mod["main"].with_body(replacement))


def _make_selector_input(
    *, multidim_ii, multidim_makespan, one_dim_ii, one_dim_makespan
):
    normalized = tl.transform.SunmmioMultidimRegion("auto")(
        _make_nested_pipeline(outer_extent=4, inner_extent=8)
    )
    multidim = _normalized_outer(normalized)
    multidim_annotations = dict(multidim.annotations)
    multidim_annotations.update(
        {
            "tl.sunmmio.pipeline.requested": True,
            "tl.sunmmio.pipeline.applied": True,
            "tl.sunmmio.pipeline.mode": "multidim_ilp",
            "ii": multidim_ii,
            "makespan": multidim_makespan,
            "stage_count": 2,
            "steady_state_max_iter_offset": 1,
        }
    )
    multidim = tir.For(
        multidim.loop_var,
        multidim.min,
        multidim.extent,
        multidim.kind,
        multidim.body,
        multidim.thread_binding,
        multidim_annotations,
        multidim.step,
    )

    original = _normalized_outer(normalized).annotations[
        "tl.sunmmio.multidim.original_loop"
    ]
    inner = original.body
    inner_annotations = dict(inner.annotations)
    inner_annotations.update(
        {
            "tl.sunmmio.pipeline.suppress_artifacts": True,
            "tl.sunmmio.pipeline.requested": True,
            "tl.sunmmio.pipeline.applied": True,
            "tl.sunmmio.pipeline.mode": "ilp",
            "ii": one_dim_ii,
            "makespan": one_dim_makespan,
        }
    )
    one_dim = tir.For(
        inner.loop_var,
        inner.min,
        inner.extent,
        inner.kind,
        inner.body,
        inner.thread_binding,
        inner_annotations,
        inner.step,
    )
    outer_annotations = dict(original.annotations)
    outer_annotations["tl.sunmmio.multidim.candidate_loop"] = multidim
    outer = tir.For(
        original.loop_var,
        original.min,
        original.extent,
        original.kind,
        one_dim,
        original.thread_binding,
        outer_annotations,
        original.step,
    )
    return tvm.IRModule.from_expr(normalized["main"].with_body(outer))


def _make_selector_input_without_one_dim_cost():
    module = _make_selector_input(
        multidim_ii=5,
        multidim_makespan=10,
        one_dim_ii=2,
        one_dim_makespan=4,
    )
    outer = _normalized_outer(module)
    inner = outer.body
    inner_annotations = dict(inner.annotations)
    applied_keys = {
        "tl.sunmmio.pipeline.suppress_artifacts",
        "tl.sunmmio.pipeline.requested",
        "tl.sunmmio.pipeline.applied",
        "tl.sunmmio.pipeline.mode",
        "ii",
        "makespan",
    }
    for key in list(inner_annotations):
        if str(key) in applied_keys:
            inner_annotations.pop(key)
    unplanned_inner = tir.For(
        inner.loop_var,
        inner.min,
        inner.extent,
        inner.kind,
        inner.body,
        inner.thread_binding,
        inner_annotations,
        inner.step,
    )
    unplanned_outer = tir.For(
        outer.loop_var,
        outer.min,
        outer.extent,
        outer.kind,
        unplanned_inner,
        outer.thread_binding,
        outer.annotations,
        outer.step,
    )
    return tvm.IRModule.from_expr(module["main"].with_body(unplanned_outer))


@pytest.mark.parametrize(
    ("inner_min", "inner_extent", "expected"),
    [
        (0, 3, [0, 1, 2]),
        (5, 3, [5, 6, 7]),
        (7, 1, [7]),
    ],
)
def test_multidim_region_expands_inner_axis_and_preserves_coordinates(
    inner_min, inner_extent, expected
):
    normalized = tl.transform.SunmmioMultidimRegion("forced")(
        _make_nested_pipeline(inner_min=inner_min, inner_extent=inner_extent)
    )
    outer = _normalized_outer(normalized)

    assert bool(outer.annotations["tl.sunmmio.multidim.region"])
    assert int(outer.annotations["tl.sunmmio.multidim.rank"]) == 2
    assert int(outer.annotations["tl.sunmmio.multidim.inner_min"]) == inner_min
    assert int(outer.annotations["tl.sunmmio.multidim.inner_extent"]) == inner_extent
    assert int(outer.annotations["num_stages"]) == 2
    assert _inner_coordinates(outer.body) == expected
    assert not any(
        isinstance(node, tir.For) and "num_stages" in (node.annotations or {})
        for node in _post_order_nodes(outer.body)
    )


def _post_order_nodes(stmt):
    nodes = []
    tir.stmt_functor.post_order_visit(stmt, nodes.append)
    return nodes


def test_multidim_region_auto_records_dynamic_extent_fallback():
    extent = tir.Var("extent", "int32")
    normalized = tl.transform.SunmmioMultidimRegion("auto")(
        _make_nested_pipeline(outer_extent=extent)
    )
    outer = _normalized_outer(normalized)

    # The candidate owns the diagnostic, while its concrete core owns selection.
    assert "tl.sunmmio.multidim.selected_mode" not in outer.annotations
    assert str(outer.annotations["tl.sunmmio.multidim.fallback_stage"]) == "region_discovery"
    assert str(outer.annotations["tl.sunmmio.multidim.fallback_reason"]) == "dynamic_or_non_affine_extent"
    assert isinstance(outer.body, tir.For)
    assert int(outer.body.annotations["num_stages"]) == 2
    assert str(outer.body.annotations["tl.sunmmio.multidim.selected_mode"]) == "ilp"
    assert str(outer.body.annotations["tl.sunmmio.multidim.fallback_reason"]) == "dynamic_or_non_affine_extent"


def test_multidim_region_forced_rejects_dynamic_extent():
    extent = tir.Var("extent", "int32")
    with pytest.raises(tvm.TVMError, match="dynamic_or_non_affine_extent"):
        tl.transform.SunmmioMultidimRegion("forced")(
            _make_nested_pipeline(outer_extent=extent)
        )


def test_multidim_region_rejects_barrier():
    barrier = tir.Evaluate(
        tir.Call(
            "handle",
            tvm.ir.Op.get("tl.barrier_arrive_and_wait"),
            [tir.IntImm("int64", 1)],
        )
    )
    normalized = tl.transform.SunmmioMultidimRegion("auto")(
        _make_nested_pipeline(inner_extra=barrier)
    )
    outer = _normalized_outer(normalized)
    assert str(outer.annotations["tl.sunmmio.multidim.fallback_reason"]) == "barrier_region_boundary"
    assert int(outer.body.annotations["num_stages"]) == 2


def test_multidim_region_rejects_multiple_inner_pipelines():
    outer = tir.Var("outer", "int32")

    def make_inner(name):
        inner = tir.Var(name, "int32")
        return tir.For(
            inner,
            0,
            2,
            tir.ForKind.SERIAL,
            tir.Evaluate(inner),
            annotations={"num_stages": 2},
        )

    body = tir.For(
        outer,
        0,
        2,
        tir.ForKind.SERIAL,
        tir.SeqStmt([make_inner("inner_a"), make_inner("inner_b")]),
    )
    mod = tvm.IRModule.from_expr(
        tir.PrimFunc([], body).with_attr("global_symbol", "main")
    )
    normalized = tl.transform.SunmmioMultidimRegion("auto")(mod)
    loop = _normalized_outer(normalized)
    assert str(loop.annotations["tl.sunmmio.multidim.fallback_reason"]) == "multiple_inner_pipeline_loops"


def test_multidim_region_rejects_single_outer_iteration():
    normalized = tl.transform.SunmmioMultidimRegion("auto")(
        _make_nested_pipeline(outer_extent=1)
    )
    outer = _normalized_outer(normalized)
    assert str(outer.annotations["tl.sunmmio.multidim.fallback_reason"]) == "outer_extent_too_short"
    assert int(outer.body.annotations["num_stages"]) == 2




def test_multidim_auto_restores_original_loop_after_late_failure():
    normalized = tl.transform.SunmmioMultidimRegion("auto")(
        _make_nested_pipeline()
    )
    planned_like = _with_outer_annotations(
        normalized,
        {
            "tl.sunmmio.pipeline.requested": True,
            "tl.sunmmio.pipeline.applied": True,
            "tl.sunmmio.pipeline.mode": "multidim_ilp",
            "stage_count": 2,
            "steady_state_max_iter_offset": 1,
        },
    )

    failed_like = _with_outer_annotations(
        planned_like,
        {
            "tl.sunmmio.pipeline.applied": False,
            "tl.sunmmio.pipeline.fallback_reason": "injected_test_failure",
        },
    )
    restored = tl.transform.ValidateSunmmioMultidimPipeline("auto", "inject")(
        failed_like
    )
    outer = _normalized_outer(restored)
    assert str(outer.annotations["tl.sunmmio.multidim.selected_mode"]) == "ilp"
    assert str(outer.annotations["tl.sunmmio.multidim.fallback_reason"]) == (
        "injected_test_failure"
    )
    assert "tl.sunmmio.multidim.original_loop" not in outer.annotations
    assert _inner_coordinates(outer.body) == []
    assert isinstance(outer.body, tir.For)
    assert int(outer.body.annotations["num_stages"]) == 2


def test_multidim_auto_selects_1d_when_full_loop_prediction_is_slower():
    selected = tl.transform.SelectSunmmioMultidimPipeline("auto")(
        _make_selector_input(
            multidim_ii=23,
            multidim_makespan=29,
            one_dim_ii=2,
            one_dim_makespan=4,
        )
    )
    outer = _normalized_outer(selected)
    inner = outer.body

    assert str(outer.annotations["tl.sunmmio.multidim.selected_mode"]) == "ilp"
    assert int(outer.annotations["tl.sunmmio.multidim.predicted_score"]) == 98
    assert int(outer.annotations["tl.sunmmio.multidim.one_dim_predicted_score"]) == 72
    assert int(outer.annotations["tl.sunmmio.multidim.min_gain_percent"]) == 2
    assert str(outer.annotations["tl.sunmmio.multidim.fallback_reason"]) == (
        "predicted_not_better_than_1d"
    )
    assert "tl.sunmmio.multidim.candidate_loop" not in outer.annotations
    assert "tl.sunmmio.pipeline.suppress_artifacts" not in inner.annotations
    assert str(inner.annotations["tl.sunmmio.pipeline.mode"]) == "ilp"




def test_multidim_auto_preserves_1d_when_cost_is_unavailable():
    selected = tl.transform.SelectSunmmioMultidimPipeline("auto")(
        _make_selector_input_without_one_dim_cost()
    )
    outer = _normalized_outer(selected)

    assert str(outer.annotations["tl.sunmmio.multidim.selected_mode"]) == "ilp"
    assert int(outer.annotations["tl.sunmmio.multidim.predicted_score"]) == -1
    assert int(
        outer.annotations["tl.sunmmio.multidim.one_dim_predicted_score"]
    ) == -1
    assert str(outer.annotations["tl.sunmmio.multidim.fallback_reason"]) == (
        "one_dim_cost_unavailable"
    )
    assert "tl.sunmmio.multidim.candidate_loop" not in outer.annotations
    assert "tl.sunmmio.pipeline.applied" not in outer.body.annotations
def test_multidim_auto_selects_multidim_when_prediction_clears_threshold():
    selected = tl.transform.SelectSunmmioMultidimPipeline("auto")(
        _make_selector_input(
            multidim_ii=5,
            multidim_makespan=10,
            one_dim_ii=2,
            one_dim_makespan=4,
        )
    )
    outer = _normalized_outer(selected)

    assert str(outer.annotations["tl.sunmmio.multidim.selected_mode"]) == (
        "multidim_ilp"
    )
    assert int(outer.annotations["tl.sunmmio.multidim.predicted_score"]) == 25
    assert int(outer.annotations["tl.sunmmio.multidim.one_dim_predicted_score"]) == 72
    assert "tl.sunmmio.multidim.fallback_reason" not in outer.annotations
    assert "tl.sunmmio.multidim.candidate_loop" not in outer.annotations
    assert str(outer.annotations["tl.sunmmio.pipeline.mode"]) == "multidim_ilp"


def test_multidim_auto_falls_back_when_candidate_score_overflows():
    selected = tl.transform.SelectSunmmioMultidimPipeline("auto")(
        _make_selector_input(
            multidim_ii=tir.IntImm("int64", (1 << 63) - 1),
            multidim_makespan=10,
            one_dim_ii=2,
            one_dim_makespan=4,
        )
    )
    outer = _normalized_outer(selected)

    assert str(outer.annotations["tl.sunmmio.multidim.selected_mode"]) == "ilp"
    assert int(outer.annotations["tl.sunmmio.multidim.predicted_score"]) == -1
    assert int(
        outer.annotations["tl.sunmmio.multidim.one_dim_predicted_score"]
    ) == -1
    assert str(outer.annotations["tl.sunmmio.multidim.fallback_reason"]) == (
        "candidate_cost_overflow"
    )


def test_multidim_auto_falls_back_to_1d_without_cross_outer_overlap():
    normalized = tl.transform.SunmmioMultidimRegion("auto")(
        _make_nested_pipeline()
    )
    planned_like = _with_outer_annotations(
        normalized,
        {
            "tl.sunmmio.pipeline.requested": True,
            "tl.sunmmio.pipeline.applied": True,
            "tl.sunmmio.pipeline.mode": "multidim_ilp",
            "stage_count": 1,
            "steady_state_max_iter_offset": 0,
        },
    )

    restored = tl.transform.ValidateSunmmioMultidimPipeline(
        "auto", "planning"
    )(planned_like)
    outer = _normalized_outer(restored)
    assert str(outer.annotations["tl.sunmmio.multidim.selected_mode"]) == "ilp"
    assert str(outer.annotations["tl.sunmmio.multidim.fallback_reason"]) == (
        "no_cross_outer_overlap"
    )
    assert isinstance(outer.body, tir.For)
    assert int(outer.body.annotations["num_stages"]) == 2


def test_multidim_forced_rejects_schedule_without_cross_outer_overlap():
    normalized = tl.transform.SunmmioMultidimRegion("forced")(
        _make_nested_pipeline()
    )
    planned_like = _with_outer_annotations(
        normalized,
        {
            "tl.sunmmio.pipeline.requested": True,
            "tl.sunmmio.pipeline.applied": True,
            "tl.sunmmio.pipeline.mode": "multidim_ilp",
            "stage_count": 1,
            "steady_state_max_iter_offset": 0,
        },
    )

    with pytest.raises(tvm.TVMError, match="no_cross_outer_overlap"):
        tl.transform.ValidateSunmmioMultidimPipeline("forced", "planning")(
            planned_like
        )

def test_hierarchical_region_preserves_inner_pipeline(monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_MULTIDIM_HIERARCHICAL", "1")
    normalized = tl.transform.SunmmioMultidimRegion("forced")(
        _make_nested_pipeline(outer_extent=4, inner_extent=8)
    )
    outer = _normalized_outer(normalized)
    inner = outer.body

    assert str(outer.annotations["tl.sunmmio.multidim.strategy"]) == "hierarchical_v2"
    assert bool(outer.annotations["tl.sunmmio.multidim.region"])
    assert isinstance(inner, tir.For)
    assert int(inner.annotations["num_stages"]) == 2
    assert bool(inner.annotations["tl.sunmmio.multidim.hierarchical_inner"])
    assert _inner_coordinates(outer.body) == []


def test_hierarchical_summary_records_inner_1d_schedule(monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_MULTIDIM_HIERARCHICAL", "1")
    normalized = tl.transform.SunmmioMultidimRegion("forced")(
        _make_nested_pipeline(outer_extent=4, inner_extent=8)
    )
    outer = _normalized_outer(normalized)
    inner = outer.body
    inner_annotations = dict(inner.annotations)
    inner_annotations.pop("num_stages")
    inner_annotations.update(
        {
            "tl.sunmmio.pipeline.requested": True,
            "tl.sunmmio.pipeline.applied": True,
            "tl.sunmmio.pipeline.mode": "ilp",
            "ii": 2,
            "makespan": 4,
            "stage_count": 2,
        }
    )
    planned_inner = tir.For(
        inner.loop_var,
        inner.min,
        inner.extent,
        inner.kind,
        inner.body,
        inner.thread_binding,
        inner_annotations,
        inner.step,
    )
    planned_outer = tir.For(
        outer.loop_var,
        outer.min,
        outer.extent,
        outer.kind,
        planned_inner,
        outer.thread_binding,
        outer.annotations,
        outer.step,
    )
    mod = tvm.IRModule.from_expr(normalized["main"].with_body(planned_outer))
    summarized = tl.transform.BuildSunmmioHierarchicalSummary("forced")(mod)
    result = _normalized_outer(summarized)

    assert bool(result.annotations["tl.sunmmio.multidim.inner_summary_valid"])
    assert int(result.annotations["tl.sunmmio.multidim.inner_ii"]) == 2
    assert int(result.annotations["tl.sunmmio.multidim.inner_makespan"]) == 4
    assert int(result.annotations["tl.sunmmio.multidim.inner_stage_count"]) == 2
    assert str(result.annotations["tl.sunmmio.pipeline.mode"]) == "multidim_ilp"
    assert isinstance(result.body, tir.For)


@pytest.mark.parametrize("outer_versions", [None, 3])
def test_hierarchical_outer_builder_constructs_three_stages(
    monkeypatch, outer_versions
):
    monkeypatch.setenv("TL_SUNMMIO_MULTIDIM_PREFETCH_BEFORE_PREFIX", "1")
    if outer_versions is not None:
        monkeypatch.setenv(
            "TL_SUNMMIO_MULTIDIM_OUTER_VERSIONS", str(outer_versions)
        )
    outer_var = tir.Var("tile", "int32")
    scratch = tir.decl_buffer((2,), "float32", name="scratch", scope="shared.rsram")
    prefix = tir.BufferStore(scratch, tir.FloatImm("float32", 0.0), [0])
    inner_prefetch = tir.AttrStmt(
        outer_var,
        "tl.sunmmio.multidim.inner_phase",
        0,
        tir.BufferStore(scratch, tir.Cast("float32", outer_var), [0]),
    )
    inner_compute = tir.AttrStmt(
        outer_var,
        "tl.sunmmio.multidim.inner_phase",
        1,
        tir.BufferStore(
            scratch,
            tir.BufferLoad(scratch, [0]) + tir.FloatImm("float32", 1.0),
            [1],
        ),
    )
    suffix = tir.Evaluate(tir.BufferLoad(scratch, [1]))
    annotations = {
        "tl.sunmmio.multidim.region": True,
        "tl.sunmmio.multidim.strategy": "hierarchical_v2",
        "tl.sunmmio.multidim.inner_summary_valid": True,
        "tl.sunmmio.multidim.inner_ii": 2,
        "tl.sunmmio.multidim.inner_makespan": 4,
        "tl.sunmmio.multidim.inner_extent": 8,
        "tl.sunmmio.pipeline.requested": True,
        "tl.sunmmio.pipeline.applied": True,
        "tl.sunmmio.pipeline.mode": "multidim_ilp",
    }
    outer = tir.For(
        outer_var,
        0,
        4,
        tir.ForKind.SERIAL,
        tir.SeqStmt([prefix, inner_prefetch, inner_compute, suffix]),
        annotations=annotations,
    )
    mod = tvm.IRModule.from_expr(
        tir.PrimFunc([], outer).with_attr("global_symbol", "main")
    )

    built = tl.transform.BuildSunmmioHierarchicalOuterPipeline("forced")(mod)
    result = _normalized_outer(built)

    assert bool(result.annotations["tl.sunmmio.multidim.outer_pipeline_built"])
    assert not bool(
        result.annotations["tl.sunmmio.multidim.prefetch_before_prefix"]
    )
    expected_versions = 2 if outer_versions is None else outer_versions
    assert int(result.annotations["iterations"]) == expected_versions
    assert int(
        result.annotations["tl.sunmmio.multidim.outer_versions"]
    ) == expected_versions
    assert list(map(str, result.annotations["prologue_orders"])) == ["0-0"]
    assert list(map(str, result.annotations["body_orders"])) == [
        "1-0",
        "0-1",
        "0-2",
    ]
    assert list(map(str, result.annotations["epilogue_orders"])) == [
        "1-1",
        "1-2",
    ]
    assert len(result.annotations["runtime_multiversion_buffers"]) == 1
    assert isinstance(result.body, tir.SeqStmt)
    assert [
        int(stage.value)
        for stage in result.body.seq
        if isinstance(stage, tir.AttrStmt)
    ] == [0, 1, 2]


def test_hierarchical_outer_builder_moves_disjoint_prefetch_before_prefix(
    monkeypatch,
):
    monkeypatch.setenv("TL_SUNMMIO_MULTIDIM_PREFETCH_BEFORE_PREFIX", "1")
    outer_var = tir.Var("tile", "int32")
    accumulator = tir.decl_buffer(
        (2,), "float32", name="accumulator", scope="shared.rsram"
    )
    staging = tir.decl_buffer(
        (1,), "float32", name="staging", scope="shared.asram"
    )
    prefix = tir.BufferStore(
        accumulator, tir.FloatImm("float32", 0.0), [0]
    )
    inner_prefetch = tir.AttrStmt(
        outer_var,
        "tl.sunmmio.multidim.inner_phase",
        0,
        tir.BufferStore(staging, tir.Cast("float32", outer_var), [0]),
    )
    inner_compute = tir.AttrStmt(
        outer_var,
        "tl.sunmmio.multidim.inner_phase",
        1,
        tir.BufferStore(accumulator, tir.BufferLoad(staging, [0]), [1]),
    )
    suffix = tir.Evaluate(tir.BufferLoad(accumulator, [1]))
    annotations = {
        "tl.sunmmio.multidim.region": True,
        "tl.sunmmio.multidim.strategy": "hierarchical_v2",
        "tl.sunmmio.multidim.inner_summary_valid": True,
        "tl.sunmmio.multidim.inner_ii": 2,
        "tl.sunmmio.multidim.inner_makespan": 4,
        "tl.sunmmio.multidim.inner_extent": 8,
        "tl.sunmmio.pipeline.requested": True,
        "tl.sunmmio.pipeline.applied": True,
        "tl.sunmmio.pipeline.mode": "multidim_ilp",
    }
    outer = tir.For(
        outer_var,
        0,
        4,
        tir.ForKind.SERIAL,
        tir.SeqStmt([prefix, inner_prefetch, inner_compute, suffix]),
        annotations=annotations,
    )
    mod = tvm.IRModule.from_expr(
        tir.PrimFunc([], outer).with_attr("global_symbol", "main")
    )

    built = tl.transform.BuildSunmmioHierarchicalOuterPipeline("forced")(mod)
    result = _normalized_outer(built)

    assert bool(
        result.annotations["tl.sunmmio.multidim.prefetch_before_prefix"]
    )
    phase_zero = result.body.seq[0]
    assert isinstance(phase_zero, tir.AttrStmt)
    assert isinstance(phase_zero.body, tir.SeqStmt)
    first, second = phase_zero.body.seq
    assert isinstance(first, tir.BufferStore)
    assert first.buffer.same_as(staging)
    assert isinstance(second, tir.BufferStore)
    assert second.buffer.same_as(accumulator)


def test_hierarchical_outer_builder_keeps_prefetch_after_same_storage_prefix(
    monkeypatch,
):
    monkeypatch.setenv("TL_SUNMMIO_MULTIDIM_PREFETCH_BEFORE_PREFIX", "1")
    outer_var = tir.Var("tile", "int32")
    shared_data = tir.Var(
        "shared_data",
        tvm.ir.PointerType(tvm.ir.PrimType("float32"), "shared.rsram"),
    )
    prefix_view = tir.decl_buffer(
        (2,),
        "float32",
        name="prefix_view",
        data=shared_data,
        scope="shared.rsram",
    )
    prefetch_view = tir.decl_buffer(
        (2,),
        "float32",
        name="prefetch_view",
        data=shared_data,
        scope="shared.rsram",
    )
    output = tir.decl_buffer(
        (1,), "float32", name="output", scope="shared.asram"
    )
    prefix = tir.BufferStore(
        prefix_view, tir.FloatImm("float32", 0.0), [0]
    )
    inner_prefetch = tir.AttrStmt(
        outer_var,
        "tl.sunmmio.multidim.inner_phase",
        0,
        tir.BufferStore(prefetch_view, tir.Cast("float32", outer_var), [1]),
    )
    inner_compute = tir.AttrStmt(
        outer_var,
        "tl.sunmmio.multidim.inner_phase",
        1,
        tir.BufferStore(output, tir.BufferLoad(prefetch_view, [1]), [0]),
    )
    suffix = tir.Evaluate(tir.BufferLoad(output, [0]))
    annotations = {
        "tl.sunmmio.multidim.region": True,
        "tl.sunmmio.multidim.strategy": "hierarchical_v2",
        "tl.sunmmio.multidim.inner_summary_valid": True,
        "tl.sunmmio.multidim.inner_ii": 2,
        "tl.sunmmio.multidim.inner_makespan": 4,
        "tl.sunmmio.multidim.inner_extent": 8,
        "tl.sunmmio.pipeline.requested": True,
        "tl.sunmmio.pipeline.applied": True,
        "tl.sunmmio.pipeline.mode": "multidim_ilp",
    }
    outer = tir.For(
        outer_var,
        0,
        4,
        tir.ForKind.SERIAL,
        tir.SeqStmt([prefix, inner_prefetch, inner_compute, suffix]),
        annotations=annotations,
    )
    mod = tvm.IRModule.from_expr(
        tir.PrimFunc([], outer).with_attr("global_symbol", "main")
    )

    built = tl.transform.BuildSunmmioHierarchicalOuterPipeline("forced")(mod)
    result = _normalized_outer(built)

    assert not bool(
        result.annotations["tl.sunmmio.multidim.prefetch_before_prefix"]
    )
    phase_zero = result.body.seq[0]
    assert isinstance(phase_zero, tir.AttrStmt)
    assert isinstance(phase_zero.body, tir.SeqStmt)
    first, second = phase_zero.body.seq
    assert isinstance(first, tir.BufferStore)
    assert first.buffer.same_as(prefix_view)
    assert isinstance(second, tir.BufferStore)
    assert second.buffer.same_as(prefetch_view)


def _make_interchange_candidate():
    outer = tir.Var("outer", "int32")
    inner = tir.Var("inner", "int32")
    accumulator = tir.decl_buffer(
        (1,), "float32", name="accumulator", scope="shared.rsram"
    )
    staging = tir.decl_buffer(
        (1,), "float32", name="staging", scope="shared.rsram"
    )
    banked = tir.decl_buffer(
        (1,), "float32", name="banked", scope="shared.asram"
    )
    prefix = tir.BufferStore(accumulator, tir.FloatImm("float32", 0.0), [0])
    shared_prefix = tir.BufferStore(
        staging, tir.Cast("float32", inner), [0]
    )
    banked_prefix = tir.BufferStore(
        banked, tir.BufferLoad(staging, [0]), [0]
    )
    outer_dependent = tir.BufferStore(
        accumulator,
        tir.BufferLoad(accumulator, [0])
        + tir.BufferLoad(banked, [0])
        + tir.Cast("float32", outer),
        [0],
    )
    banked_rewrite = tir.BufferStore(
        banked, tir.BufferLoad(staging, [0]) + tir.FloatImm("float32", 1.0), [0]
    )
    inner_loop = tir.For(
        inner,
        0,
        8,
        tir.ForKind.SERIAL,
        tir.SeqStmt(
            [shared_prefix, banked_prefix, outer_dependent, banked_rewrite]
        ),
        annotations={"num_stages": 2},
    )
    suffix = tir.Evaluate(tir.BufferLoad(accumulator, [0]))
    outer_loop = tir.For(
        outer,
        0,
        4,
        tir.ForKind.SERIAL,
        tir.SeqStmt([prefix, inner_loop, suffix]),
    )
    root = tir.Block(
        [],
        [],
        [],
        "root",
        outer_loop,
        alloc_buffers=[accumulator, staging, banked],
    )
    func = tir.PrimFunc(
        [], tir.BlockRealize([], tir.const(True, "bool"), root)
    ).with_attr("global_symbol", "main")
    return tvm.IRModule.from_expr(func)


def test_interchange_versions_accumulator_and_reuses_shared_prefix(monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_MULTIDIM_INTERCHANGE", "1")
    normalized = tl.transform.SunmmioMultidimRegion("forced")(
        _make_interchange_candidate()
    )
    outer = normalized["main"].body.block.body
    assert isinstance(outer, tir.For)
    assert str(outer.annotations["tl.sunmmio.multidim.strategy"]) == (
        "interchange_reuse_v1"
    )

    prepared = tl.transform.PrepareSunmmioMultidimInterchangeBuffers()(
        normalized
    )
    assert tvm.ir.structural_equal(normalized, prepared)

    applied = tl.transform.ApplySunmmioMultidimInterchange("forced")(
        prepared
    )
    root = applied["main"].body.block
    versions = [
        buffer
        for buffer in root.alloc_buffers
        if str(buffer.name).startswith("accumulator_v")
    ]
    assert [str(buffer.name) for buffer in versions] == [
        "accumulator_v0",
        "accumulator_v1",
        "accumulator_v2",
        "accumulator_v3",
    ]
    assert all(list(map(int, buffer.shape)) == [1] for buffer in versions)

    body = root.body
    assert isinstance(body, tir.SeqStmt)
    pipeline_loops = [
        stmt
        for stmt in body.seq
        if isinstance(stmt, tir.For) and "num_stages" in stmt.annotations
    ]
    assert len(pipeline_loops) == 1
    pipeline = pipeline_loops[0]
    annotated_versions = pipeline.annotations[
        "tl.sunmmio.multidim.interchange.accumulator_versions"
    ]
    assert [str(buffer.name) for buffer in annotated_versions] == [
        "accumulator_v0",
        "accumulator_v1",
        "accumulator_v2",
        "accumulator_v3",
    ]
    assert bool(
        pipeline.annotations["tl.sunmmio.multidim.interchange.applied"]
    )
    assert int(
        pipeline.annotations[
            "tl.sunmmio.multidim.interchange.shared_prefix_count"
        ]
    ) == 1
    assert int(
        pipeline.annotations["tl.sunmmio.multidim.interchange.outer_versions"]
    ) == 4

    coordinates = [
        int(stmt.value)
        for stmt in pipeline.body.seq
        if isinstance(stmt, tir.AttrStmt)
        and stmt.attr_key == "tl.sunmmio.multidim.inner_coordinate"
    ]
    # Only the stable staging write is shared. The ASRAM write is repeated
    # for every outer coordinate because the same buffer is written again
    # later in the logical inner body.
    assert coordinates == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_ilp_injector_consumes_interchange_inner_coordinate_markers(monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_MULTIDIM_INTERCHANGE", "1")
    normalized = tl.transform.SunmmioMultidimRegion("forced")(
        _make_interchange_candidate()
    )
    applied = tl.transform.ApplySunmmioMultidimInterchange("forced")(
        normalized
    )

    assert _inner_coordinates(applied["main"].body)
    injected = tl.transform.InjectSunmmioPipelineILP()(applied)
    assert _inner_coordinates(injected["main"].body) == []


def test_interchange_is_noop_for_auto_policy(monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_MULTIDIM_INTERCHANGE", "1")
    before = _make_interchange_candidate()
    after = tl.transform.ApplySunmmioMultidimInterchange("auto")(before)
    assert tvm.ir.structural_equal(before, after)



if __name__ == "__main__":
    pytest.main([__file__])
