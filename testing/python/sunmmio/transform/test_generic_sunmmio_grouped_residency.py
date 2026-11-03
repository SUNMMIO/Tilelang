"""Bounded residency groups preserve the complete natural loop's semantics.

These are compiler-IR checks.  The operation trace checks coordinates and state
ownership, not floating-point results, backend capacity, or measured speedup.
"""

import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import tilelang as tl
from tilelang import tvm
from tvm import arith, tir


_spec = importlib.util.spec_from_file_location(
    "grouped_residency_test_helpers",
    Path(__file__).with_name("test_generic_sunmmio_residency.py"),
)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
_clean_residency_env = _helpers._clean_residency_env
_accepted, _analyze = _helpers._accepted, _helpers._analyze
_marked, _same_statements = _helpers._marked, _helpers._same_statements
GROUP_WIDTH = "tl.sunmmio.residency.group_width"
ORIGINAL = "tl.sunmmio.residency.grouping_original_func"


@pytest.fixture(autouse=True)
def _clean_grouping_env(monkeypatch):
    monkeypatch.delenv("TL_SUNMMIO_RESIDENCY_GROUP_WIDTH", raising=False)
    monkeypatch.delenv("TL_SUNMMIO_MULTIDIM_REGION_JSON", raising=False)


def _make_module(extent, *, minimum=0, fault=None, consumers=1):
    """Resize the existing natural fixture so large columns stay in bounds."""
    mod = _helpers._make_module(
        reuse_extent=extent,
        fault=fault,
        consumers=consumers,
    )
    func = mod["main"]
    resized = {}

    def resize(buffer):
        if buffer.scope() != "global" or len(buffer.shape) != 5:
            return buffer
        if buffer not in resized:
            shape = list(buffer.shape)
            axis = 1 if buffer.name.endswith("_result") else 2
            shape[axis] = max(int(shape[axis]), minimum + extent)
            # Preserve distinct alias views: key by Buffer, not data Var.
            resized[buffer] = tir.decl_buffer(
                shape,
                buffer.dtype,
                name=buffer.name,
                data=buffer.data,
                strides=buffer.strides,
                elem_offset=buffer.elem_offset,
                scope=buffer.scope(),
                data_alignment=buffer.data_alignment,
                offset_factor=buffer.offset_factor,
                buffer_type="auto_broadcast" if int(buffer.buffer_type) == 2 else "",
                axis_separators=buffer.axis_separators,
            )
        return resized[buffer]

    def rewrite(node):
        if isinstance(node, tir.BufferLoad):
            return tir.BufferLoad(resize(node.buffer), node.indices, node.predicate)
        if isinstance(node, tir.BufferStore):
            return tir.BufferStore(
                resize(node.buffer),
                node.value,
                node.indices,
                node.predicate,
            )
        if isinstance(node, tir.For) and node.loop_var.name == "natural_column":
            return tir.For(
                node.loop_var,
                minimum,
                node.extent,
                node.kind,
                node.body,
                node.thread_binding,
                node.annotations,
                node.step,
            )
        return None

    body = tir.stmt_functor.ir_transform(func.body, None, rewrite)
    buffer_map = {param: resize(buffer) for param, buffer in func.buffer_map.items()}
    func = tir.PrimFunc(func.params, body, func.ret_type, buffer_map, func.attrs, func.span)
    return tvm.IRModule.from_expr(func)


def _apply(analyzed):
    normalized = tl.transform.SunmmioMultidimRegion("auto")(analyzed)
    prepared = tl.transform.PrepareSunmmioMultidimInterchangeBuffers()(normalized)
    return tl.transform.ApplySunmmioMultidimInterchange("auto")(prepared)


def _calls(stmt, name):
    calls = []
    tir.stmt_functor.post_order_visit(
        stmt,
        lambda node: calls.append(node) if isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == name else None,
    )
    return calls


def _constant(expr, bindings):
    replacements = {var: tir.const(value, var.dtype) for var, value in bindings.items()}
    value = arith.Analyzer().simplify(tir.stmt_functor.substitute(expr, replacements))
    assert isinstance(value, tir.IntImm), f"unbound coordinate: {value}"
    return int(value)


def _events(stmt, bindings=None):
    """Expand only control flow; do not execute DMA or matrix arithmetic."""
    bindings = {} if bindings is None else bindings
    if isinstance(stmt, tir.BlockRealize):
        assert _constant(stmt.predicate, bindings)
        yield from _events(stmt.block.body, bindings)
    elif isinstance(stmt, tir.SeqStmt):
        for child in stmt.seq:
            yield from _events(child, bindings)
    elif isinstance(stmt, tir.For):
        start = _constant(stmt.min, bindings)
        extent = _constant(stmt.extent, bindings)
        assert stmt.step is None or _constant(stmt.step, bindings) == 1
        for coordinate in range(start, start + extent):
            yield from _events(stmt.body, {**bindings, stmt.loop_var: coordinate})
    elif isinstance(stmt, tir.AttrStmt):
        yield from _events(stmt.body, bindings)
    elif isinstance(stmt, tir.Evaluate) and isinstance(stmt.value, tir.Call):
        if isinstance(stmt.value.op, tvm.ir.Op):
            yield stmt.value, bindings
    elif isinstance(stmt, tir.BufferStore):
        # The fixture's explicit initialization precedes clear-on-first MMA.
        return
    else:
        raise AssertionError(f"unexpected control flow in grouped fixture: {type(stmt)}")


def _assert_coordinate_and_state_trace(mod, extent, minimum=0, consumers=1):
    weights, state, outputs = {}, {}, {}
    reductions = defaultdict(list)
    stores = Counter()
    for call, bindings in _events(mod["main"].body):
        name = call.op.name
        if name in {"tl.dma_copy", "tl.broadcast_"}:
            source, destination = call.args[0].args[0], call.args[1].args[0]
            if source.buffer.name == "natural_right":
                weights[destination.buffer.data] = tuple(_constant(index, bindings) for index in source.indices[:3])
            if destination.buffer.name == "natural_result":
                row, column, consumer = (_constant(index, bindings) for index in destination.indices[:3])
                assert outputs[source.buffer.data] == (row, column), (
                    "tail or group reused an output before its own store",
                    row,
                    column,
                )
                stores[row, column, consumer] += 1
        elif name == "tl.mma_sunmmio":
            row, reduction, column = weights[call.args[1].args[0].buffer.data]
            accumulator = call.args[2].args[0].buffer.data
            output = call.args[7].args[0].buffer.data
            clear = _constant(call.args[5], bindings)
            final = _constant(call.args[8], bindings)
            assert bool(clear) == (reduction == 0)
            assert bool(final) == (reduction == 1)
            if clear:
                state[accumulator] = (row, column)
            assert state[accumulator] == (row, column), "accumulator versions alias columns"
            reductions[row, column, accumulator].append(reduction)
            if final:
                outputs[output] = (row, column)

    expected = {(row, column, consumer) for row in range(2) for column in range(minimum, minimum + extent) for consumer in range(consumers)}
    assert set(stores) == expected
    assert set(stores.values()) == {1}, "every output tile must be stored exactly once"
    assert len(reductions) == 2 * extent * consumers
    assert all(values == [0, 1] for values in reductions.values())


def _assert_bounded_versions_and_bindings(mod, group_width, consumers=1):
    func = mod["main"]
    # UndefinedVars handles Allocate, but does not bind Block.alloc_buffers.
    # Supply only the fixture's actual root allocations, never variables
    # discovered in accesses: missing buffers and loop binders must still fail.
    assert isinstance(func.body, tir.BlockRealize)
    definitions = [*func.params, *(buffer.data for buffer in func.body.block.alloc_buffers)]
    assert not tir.analysis.undefined_vars(func.body, definitions)
    assert tir.analysis.verify_ssa(func), "full groups and tails need distinct loop binders"
    cores = []
    tir.stmt_functor.post_order_visit(
        func.body,
        lambda node: cores.append(node) if isinstance(node, tir.For) and "num_stages" in node.annotations else None,
    )
    assert 1 <= len(cores) <= 2, "IR size must not grow with the number of groups"
    for core in cores:
        calls = _calls(core.body, "tl.mma_sunmmio")
        assert 1 <= len(calls) <= group_width * consumers
        assert len({call.args[2].args[0].buffer.data for call in calls}) == len(calls)
        assert len({call.args[7].args[0].buffer.data for call in calls}) == len(calls)
    buffers = []
    tir.stmt_functor.post_order_visit(
        func.body,
        lambda node: buffers.extend(node.alloc_buffers) if isinstance(node, tir.Block) else None,
    )
    states = [buffer for buffer in buffers if buffer.name.startswith(("natural_acc", "natural_out"))]
    assert len(states) == 2 * consumers * group_width
    assert len({buffer.data for buffer in states}) == len(states)


@pytest.mark.parametrize("extent", [10, 28])
def test_large_natural_nests_use_default_bounded_groups(tmp_path, monkeypatch, extent):
    before = _make_module(extent, consumers=2)
    analyzed, report = _analyze(before, tmp_path, monkeypatch)
    assert _accepted(report), report
    candidates = _marked(analyzed, _helpers.ELIGIBLE)
    assert len(candidates) == 1
    assert int(candidates[0].annotations[GROUP_WIDTH]) == 2
    assert not _marked(analyzed, _helpers.APPLIED)
    after = _apply(analyzed)
    assert _marked(after, _helpers.APPLIED)
    assert ORIGINAL in after["main"].attrs
    _check_snapshot_envelope(after["main"])
    _assert_bounded_versions_and_bindings(after, 2, consumers=2)
    _assert_coordinate_and_state_trace(after, extent, consumers=2)


@pytest.mark.parametrize("extent", [11, 13])
@pytest.mark.parametrize("minimum", [0, 3])
def test_grouped_tail_and_nonzero_min_preserve_every_column(tmp_path, monkeypatch, extent, minimum):
    monkeypatch.setenv("TL_SUNMMIO_RESIDENCY_GROUP_WIDTH", "4")
    before = _make_module(extent, minimum=minimum)
    analyzed, report = _analyze(before, tmp_path, monkeypatch)
    assert _accepted(report), report
    after = _apply(analyzed)
    _assert_bounded_versions_and_bindings(after, 4)
    _assert_coordinate_and_state_trace(after, extent, minimum)


@pytest.mark.parametrize("width", [0, 1])
def test_group_width_optout_preserves_oversized_natural_region(tmp_path, monkeypatch, width):
    monkeypatch.setenv("TL_SUNMMIO_RESIDENCY_GROUP_WIDTH", str(width))
    before = _make_module(10)
    analyzed, report = _analyze(before, tmp_path, monkeypatch)
    assert not _accepted(report)
    assert any(item["reason"] == "expansion_budget" for item in report["candidates"])
    after = _apply(analyzed)
    assert _same_statements(before, after)
    assert not _marked(after, _helpers.APPLIED)


@pytest.mark.parametrize("policy", ["off", "shadow"])
@pytest.mark.parametrize("extent", [10, 28])
def test_non_mutating_policies_preserve_large_loop_ir(tmp_path, monkeypatch, policy, extent):
    before = _make_module(extent)
    after, report = _analyze(before, tmp_path, monkeypatch, policy)
    assert tvm.ir.structural_equal(before, after, map_free_vars=True)
    if policy == "shadow":
        assert _accepted(report), report


@pytest.mark.parametrize(
    "fault",
    [
        "source_depends_on_reuse",
        "suffix_writes_source",
        "suffix_alias_writes_source",
        "source_modified_in_core",
        "unknown_core_effect",
        "nested_core_effect",
        "unknown_suffix_effect",
        "prefix_writes_source",
        "conditional_mma",
        "conditional_collective",
        "dependent_core_extent",
        "aliased_state_views",
        "nonunit_core_step",
        "nonunit_reuse_step",
        "state_escapes",
        "hidden_index_read_modified",
        "partial_collective",
        "nonfinal_suffix_writes_source",
        "nonfinal_prefix_writes_source",
        "partial_float_clear",
    ],
)
def test_grouping_does_not_bypass_complete_region_proof(tmp_path, monkeypatch, fault):
    before = _make_module(10, fault=fault)
    analyzed, report = _analyze(before, tmp_path, monkeypatch)
    assert not _accepted(report), (fault, report)
    assert report["candidates"]
    assert all(item["reason"] != "expansion_budget" for item in report["candidates"]), (
        "the semantic proof must inspect oversized candidates",
        fault,
        report,
    )
    after = _apply(analyzed)
    assert not _marked(after, _helpers.APPLIED)
    assert _same_statements(before, after)


def test_grouped_allocation_budget_still_rejects(tmp_path, monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_RESIDENCY_MAX_ADDED_BYTES", "0")
    before = _make_module(28)
    analyzed, report = _analyze(before, tmp_path, monkeypatch)
    assert not _accepted(report)
    assert any(item["reason"] == "allocation_budget" for item in report["candidates"])
    assert _same_statements(before, analyzed)


def test_grouping_requires_global_noalias_contract(tmp_path, monkeypatch):
    before = _make_module(10)
    before = tvm.IRModule.from_expr(before["main"].without_attr("tir.noalias"))
    analyzed, report = _analyze(before, tmp_path, monkeypatch)
    assert not _accepted(report)
    assert any(item["reason"] == "global_alias_unproven" for item in report["candidates"])
    assert _same_statements(before, analyzed)


GROUPED_CORE = "tl.sunmmio.residency.grouped_core"
CORE_COUNT = "tl.sunmmio.residency.grouping_core_count"
GROUP_FALLBACK = "tl.sunmmio.residency.grouping_fallback"
PIPELINE_APPLIED = "tl.sunmmio.pipeline.applied"


def _transaction_fixture(tmp_path, monkeypatch):
    from tilelang.layout.cute_layout import CuteLayout

    monkeypatch.setenv("TL_SUNMMIO_RESIDENCY_GROUP_WIDTH", "4")
    plain = _make_module(11)
    func = plain["main"]
    root = func.body.block
    layouts = {buffer: CuteLayout(buffer.shape, buffer.shape, [buffer.shape[1], 1], [1, 1])._inner for buffer in root.alloc_buffers}
    root_with_layouts = tir.Block(
        root.iter_vars,
        root.reads,
        root.writes,
        root.name_hint,
        root.body,
        root.init,
        root.alloc_buffers,
        root.match_buffers,
        {**dict(root.annotations), "layout_map": layouts},
    )
    func = func.with_body(tir.BlockRealize([], func.body.predicate, root_with_layouts))
    func = func.with_attr("layout_map", layouts).with_attr("grouping_test_user_attr", "keep")
    before = tvm.IRModule.from_expr(func)
    analyzed, report = _analyze(before, tmp_path, monkeypatch)
    assert _accepted(report), report
    assert int(analyzed["main"].attrs[CORE_COUNT]) == 2
    applied = _apply(analyzed)
    assert len(_marked(applied, GROUPED_CORE)) == 2
    assert len(applied["main"].attrs["layout_map"]) > len(layouts)
    return before, analyzed, applied


def _rewrite_group_cores(mod, change):
    index = 0

    def rewrite(node):
        nonlocal index
        if not isinstance(node, tir.For) or not bool(node.annotations.get(GROUPED_CORE, False)):
            return None
        annotations = dict(node.annotations)
        change(index, annotations)
        index += 1
        return tir.For(node.loop_var, node.min, node.extent, node.kind, node.body, node.thread_binding, annotations, node.step)

    body = tir.stmt_functor.ir_transform(mod["main"].body, None, rewrite, ["tir.For"])
    assert index == 2
    return tvm.IRModule.from_expr(mod["main"].with_body(body))


def _fake_successful_plan(mod):
    def mark_success(_index, annotations):
        annotations.update(
            {
                "tl.sunmmio.pipeline.requested": True,
                PIPELINE_APPLIED: True,
                "tl.sunmmio.pipeline.mode": "multidim_ilp",
                "stage_count": 2,
                "steady_state_max_iter_offset": 1,
            }
        )

    return _rewrite_group_cores(mod, mark_success)


def _assert_complete_original_restored(before, restored):
    func = restored["main"]
    assert GROUP_FALLBACK in func.attrs
    reason = func.attrs[GROUP_FALLBACK]
    assert str(reason)
    expected = before["main"].with_attr(GROUP_FALLBACK, reason)
    _assert_layout_forwards_match(before["main"], func)
    # Includes body, root allocations, both complete layout maps, buffer_map
    # and unrelated attributes.  A core-only rollback cannot pass.
    _assert_complete_function_with_bound_layout_maps(expected, func)
    assert ORIGINAL not in func.attrs and CORE_COUNT not in func.attrs
    assert not _marked(restored, GROUPED_CORE)
    _assert_layout_and_buffer_aliases(func)


def test_unplanned_grouping_restores_entire_function_and_layouts(tmp_path, monkeypatch):
    before, _, applied = _transaction_fixture(tmp_path, monkeypatch)
    restored = tl.transform.ValidateSunmmioMultidimPipeline("auto", "planning")(applied)
    _assert_complete_original_restored(before, restored)


@pytest.mark.parametrize("fault", ["one_core_failed", "one_marker_missing", "all_markers_missing"])
def test_incomplete_group_plan_restores_entire_function(tmp_path, monkeypatch, fault):
    before, _, applied = _transaction_fixture(tmp_path, monkeypatch)
    planned = _fake_successful_plan(applied)

    def damage(index, annotations):
        if fault == "one_core_failed" and index == 1:
            annotations[PIPELINE_APPLIED] = False
        if fault == "all_markers_missing" or (fault == "one_marker_missing" and index == 1):
            annotations.pop(GROUPED_CORE)

    damaged = _rewrite_group_cores(planned, damage)
    restored = tl.transform.ValidateSunmmioMultidimPipeline("auto", "planning")(damaged)
    _assert_complete_original_restored(before, restored)


@pytest.mark.parametrize("fault", ["pipeline_fallback", "missing_function_applied"])
def test_injection_failure_restores_entire_function(tmp_path, monkeypatch, fault):
    before, _, applied = _transaction_fixture(tmp_path, monkeypatch)
    planned = tl.transform.ValidateSunmmioMultidimPipeline("auto", "planning")(_fake_successful_plan(applied))
    func = planned["main"]
    if fault == "pipeline_fallback":
        func = func.with_attr("tl.sunmmio.multidim.applied", True)
        inserted = False

        def add_fallback(node):
            nonlocal inserted
            if not isinstance(node, tir.For) or inserted:
                return None
            if bool(node.annotations.get(GROUPED_CORE, False)):
                return None
            inserted = True
            annotations = dict(node.annotations)
            annotations["tl.sunmmio.pipeline.fallback_reason"] = "test_injection_failure"
            return tir.For(node.loop_var, node.min, node.extent, node.kind, node.body, node.thread_binding, annotations, node.step)

        func = func.with_body(tir.stmt_functor.ir_transform(func.body, None, add_fallback, ["tir.For"]))
        assert inserted
    else:
        func = func.without_attr("tl.sunmmio.multidim.applied")
    restored = tl.transform.ValidateSunmmioMultidimPipeline("auto", "inject")(tvm.IRModule.from_expr(func))
    _assert_complete_original_restored(before, restored)


@pytest.mark.parametrize("stage", ["planning", "inject"])
def test_forced_group_validation_raises_instead_of_atomic_fallback(tmp_path, monkeypatch, stage):
    _, _, applied = _transaction_fixture(tmp_path, monkeypatch)
    with pytest.raises(tvm.TVMError):
        tl.transform.ValidateSunmmioMultidimPipeline("forced", stage)(applied)


def test_backup_lives_through_planning_and_clears_only_after_injection_success(tmp_path, monkeypatch):
    before, _, applied = _transaction_fixture(tmp_path, monkeypatch)
    planned = tl.transform.ValidateSunmmioMultidimPipeline("auto", "planning")(_fake_successful_plan(applied))
    assert ORIGINAL in planned["main"].attrs and CORE_COUNT in planned["main"].attrs
    assert str(planned["main"].attrs[ORIGINAL]) == str(applied["main"].attrs[ORIGINAL])
    _check_snapshot_envelope(planned["main"])
    injected = tvm.IRModule.from_expr(planned["main"].with_attr("tl.sunmmio.multidim.applied", True))
    validated = tl.transform.ValidateSunmmioMultidimPipeline("auto", "inject")(injected)
    assert ORIGINAL not in validated["main"].attrs and CORE_COUNT not in validated["main"].attrs
    assert GROUP_FALLBACK not in validated["main"].attrs
    expected = injected["main"].without_attr(ORIGINAL).without_attr(CORE_COUNT)
    tvm.ir.assert_structural_equal(expected, validated["main"], map_free_vars=True)


@pytest.mark.parametrize("policy", ["auto", "forced"])
def test_construction_exception_is_atomic_or_explicit(tmp_path, monkeypatch, policy):
    before, analyzed, _ = _transaction_fixture(tmp_path, monkeypatch)
    normalized = tl.transform.SunmmioMultidimRegion("auto")(analyzed)

    def corrupt(node):
        if not isinstance(node, tir.For) or not bool(node.annotations.get(GROUP_WIDTH, False)):
            return None
        annotations = dict(node.annotations)
        annotations["runtime_multiversion_buffers"] = []
        return tir.For(node.loop_var, node.min, node.extent, node.kind, node.body, node.thread_binding, annotations, node.step)

    body = tir.stmt_functor.ir_transform(normalized["main"].body, None, corrupt, ["tir.For"])
    damaged = tvm.IRModule.from_expr(normalized["main"].with_body(body))
    if policy == "forced":
        with pytest.raises(tvm.TVMError):
            tl.transform.ApplySunmmioMultidimInterchange(policy)(damaged)
    else:
        restored = tl.transform.ApplySunmmioMultidimInterchange(policy)(damaged)
        _assert_complete_original_restored(before, restored)


def _assert_layout_and_buffer_aliases(func):
    """JSON must retain identity across allocations, layouts and accesses."""
    root = func.body.block
    allocations = list(root.alloc_buffers)
    function_layout_buffers = list(func.attrs["layout_map"].keys())
    block_layout_buffers = list(root.annotations["layout_map"].keys())
    assert len(function_layout_buffers) == len(allocations)
    assert len(block_layout_buffers) == len(allocations)
    for buffer in allocations:
        assert any(key.same_as(buffer) for key in function_layout_buffers)
        assert any(key.same_as(buffer) for key in block_layout_buffers)
    known_buffers = [*allocations, *func.buffer_map.values()]

    def inspect(node):
        if isinstance(node, (tir.BufferLoad, tir.BufferStore)):
            assert any(buffer.same_as(node.buffer) for buffer in known_buffers), (
                "deserialization detached a buffer access from its allocation/map",
                node.buffer.name,
            )

    tir.stmt_functor.post_order_visit(func.body, inspect)


def _assert_complete_function_with_bound_layout_maps(expected, actual):
    """Bind Buffer-key metadata to declarations before comparing all fields.

    Native StructuralEqual cannot match a reloaded Buffer used as a Map key,
    even when comparing the same pair of Buffers directly returns true.  This
    follows boundary_artifacts.exact_injected_equal's allocation-position
    approach without making repository tests depend on an external run tree.
    """

    def without_layout_maps(func):
        root = func.body.block
        annotations = {key: value for key, value in root.annotations.items() if str(key) != "layout_map"}
        block = tir.Block(
            root.iter_vars,
            root.reads,
            root.writes,
            root.name_hint,
            root.body,
            root.init,
            root.alloc_buffers,
            root.match_buffers,
            annotations,
        )
        body = tir.BlockRealize(func.body.iter_values, func.body.predicate, block)
        return func.with_body(body).without_attr("layout_map")

    tvm.ir.assert_structural_equal(
        without_layout_maps(expected),
        without_layout_maps(actual),
        map_free_vars=True,
    )

    def maps_by_binding(func):
        declared = [(("parameter", index), func.buffer_map[param]) for index, param in enumerate(func.params) if param in func.buffer_map]
        declared.extend((("allocation", index), buffer) for index, buffer in enumerate(func.body.block.alloc_buffers))
        maps = []
        for source in (func.attrs["layout_map"], func.body.block.annotations["layout_map"]):
            mapped = {}
            for buffer, layout in source.items():
                positions = [position for position, declared_buffer in declared if declared_buffer.same_as(buffer)]
                assert len(positions) == 1, "layout key must identify an actual declared Buffer"
                assert positions[0] not in mapped
                mapped[positions[0]] = (buffer, layout)
            assert len(mapped) == len(source)
            maps.append(mapped)
        return maps

    for left, right in zip(maps_by_binding(expected), maps_by_binding(actual)):
        assert left.keys() == right.keys()
        for position in left:
            lhs_buffer, lhs_layout = left[position]
            rhs_buffer, rhs_layout = right[position]
            tvm.ir.assert_structural_equal(lhs_buffer, rhs_buffer, map_free_vars=True)
            assert type(lhs_layout).__name__ == type(rhs_layout).__name__ == "CuteLayout"
            for field in ("logical_shape", "mode_shape", "mode_stride", "dim_levels", "input_size"):
                tvm.ir.assert_structural_equal(getattr(lhs_layout, field), getattr(rhs_layout, field))
            # Unlike a sampled Forward check, this checks the complete mapping
            # expression and exact canonical placeholder identity.
            tvm.ir.assert_structural_equal(
                tvm.get_global_func("tl.Layout_index")(lhs_layout),
                tvm.get_global_func("tl.Layout_index")(rhs_layout),
                map_free_vars=False,
            )


def test_snapshot_backup_prints_and_validation_restores_executable_layouts(tmp_path, monkeypatch):
    before, analyzed, applied = _transaction_fixture(tmp_path, monkeypatch)
    planned = tl.transform.ValidateSunmmioMultidimPipeline("auto", "planning")(_fake_successful_plan(applied))
    backup_json = str(analyzed["main"].attrs[ORIGINAL])
    for index, mod in enumerate((analyzed, applied, planned)):
        func = mod["main"]
        assert str(func.attrs[ORIGINAL]) == backup_json
        assert func.script(show_meta=True), "the backup must not break TVMScript printing"
        _check_snapshot_envelope(func)
        candidate = mod
        if index == 2:

            def remove_marker(core_index, annotations):
                if core_index == 1:
                    annotations.pop(GROUPED_CORE)

            candidate = _rewrite_group_cores(mod, remove_marker)
        # Only the production restoration path canonicalizes placeholders and
        # rebuilds layouts.  Raw JSON deserialization is not an executable backup.
        restored = tl.transform.ValidateSunmmioMultidimPipeline("auto", "planning")(candidate)
        _assert_complete_original_restored(before, restored)


def test_artifact_validation_keeps_planned_backup_without_injection_marker(tmp_path, monkeypatch):
    _, _, applied = _transaction_fixture(tmp_path, monkeypatch)
    planned = _fake_successful_plan(applied)
    assert "tl.sunmmio.multidim.applied" not in planned["main"].attrs
    checked = tl.transform.ValidateSunmmioMultidimPipeline("auto", "artifact")(planned)
    assert ORIGINAL in checked["main"].attrs and CORE_COUNT in checked["main"].attrs
    assert GROUP_FALLBACK not in checked["main"].attrs
    assert str(checked["main"].attrs[ORIGINAL]) == str(planned["main"].attrs[ORIGINAL])
    tvm.ir.assert_structural_equal(planned["main"], checked["main"], map_free_vars=True)


@pytest.mark.parametrize("fault", ["unplanned", "missing_marker"])
def test_artifact_validation_rejects_incomplete_planning(tmp_path, monkeypatch, fault):
    before, _, applied = _transaction_fixture(tmp_path, monkeypatch)
    candidate = applied
    if fault == "missing_marker":

        def remove_marker(index, annotations):
            if index == 1:
                annotations.pop(GROUPED_CORE)

        candidate = _rewrite_group_cores(_fake_successful_plan(applied), remove_marker)
    restored = tl.transform.ValidateSunmmioMultidimPipeline("auto", "artifact")(candidate)
    _assert_complete_original_restored(before, restored)


def _constant_layout_forward(layout, coordinates=(1, 2)):
    # These APIs expose the exact placeholders/index expressions used by
    # LayoutNode::Forward.  Substitution must eliminate every placeholder.
    variables = tvm.get_global_func("tl.Layout_forward_vars")(layout)
    expressions = tvm.get_global_func("tl.Layout_index")(layout)
    assert len(variables) == len(coordinates)
    substitution = {var: tir.const(value, var.dtype) for var, value in zip(variables, coordinates)}
    analyzer = arith.Analyzer()
    values = [analyzer.simplify(tir.stmt_functor.substitute(expr, substitution)) for expr in expressions]
    assert all(isinstance(value, tir.IntImm) for value in values), (
        "restored layout Forward left detached input placeholders",
        values,
    )
    return [int(value) for value in values]


def _assert_layout_forwards_match(before, restored):
    for original_map, restored_map in (
        (before.attrs["layout_map"], restored.attrs["layout_map"]),
        (before.body.block.annotations["layout_map"], restored.body.block.annotations["layout_map"]),
    ):
        originals = {str(buffer.name): layout for buffer, layout in original_map.items()}
        recovered = {str(buffer.name): layout for buffer, layout in restored_map.items()}
        assert originals.keys() == recovered.keys()
        for name, layout in originals.items():
            assert _constant_layout_forward(recovered[name]) == _constant_layout_forward(layout)


def _check_snapshot_envelope(func):
    snapshot = tvm.ir.load_json(str(func.attrs[ORIGINAL]))
    assert len(snapshot) == 2
    assert isinstance(snapshot[0], tir.PrimFunc)
    assert all(isinstance(variable, tir.Var) for variable in snapshot[1])


def test_raw_cute_layout_json_roundtrip_detaches_input_placeholders():
    from tilelang.layout.cute_layout import CuteLayout

    original = CuteLayout([32, 32], [32, 32], [32, 1], [1, 1])._inner
    assert _constant_layout_forward(original) == [34]
    restored = tvm.ir.load_json(tvm.ir.save_json(original))
    # This records the unmodified upstream serialization limitation.  Actual
    # grouping recovery must use its snapshot restorer, which the tests above
    # require to recover both exact structure and executable layout mappings.
    with pytest.raises(AssertionError, match="detached input placeholders"):
        _constant_layout_forward(restored)


def test_complete_layout_comparison_rejects_a_matching_sample_with_different_mapping(tmp_path, monkeypatch):
    from tilelang.layout.cute_layout import CuteLayout

    before, _, _ = _transaction_fixture(tmp_path, monkeypatch)
    expected = before["main"]
    layouts = dict(expected.attrs["layout_map"])
    buffer = expected.body.block.alloc_buffers[0]
    original = layouts[buffer]
    changed = CuteLayout([32, 32], [32, 32], [30, 2], [1, 1])._inner
    assert _constant_layout_forward(original) == _constant_layout_forward(changed) == [34]
    layouts[buffer] = changed
    corrupted = expected.with_attr("layout_map", layouts)
    with pytest.raises(ValueError):
        _assert_complete_function_with_bound_layout_maps(expected, corrupted)


@pytest.mark.parametrize("extent", [11, 13])
@pytest.mark.parametrize("minimum", [0, 3])
def test_region_report_matches_actual_group_graphs_and_group_base(tmp_path, monkeypatch, extent, minimum):
    width = 4
    monkeypatch.setenv("TL_SUNMMIO_RESIDENCY_GROUP_WIDTH", str(width))
    analyzed, proof_report = _analyze(_make_module(extent, minimum=minimum), tmp_path, monkeypatch)
    assert _accepted(proof_report), proof_report
    report_path = tmp_path / "grouped_regions.json"
    monkeypatch.setenv("TL_SUNMMIO_MULTIDIM_REGION_JSON", str(report_path))
    applied = _apply(analyzed)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    regions = report["regions"]
    cores = _marked(applied, GROUPED_CORE)
    assert len(regions) == len(cores) == int(applied["main"].attrs[CORE_COUNT]) == 2
    assert all(region["accepted"] for region in regions)
    main_region, tail_region = sorted(regions, key=lambda region: region["tail"])
    assert main_region["tail"] is False and tail_region["tail"] is True
    assert main_region["group_count"] == extent // width
    assert tail_region["group_count"] == 1
    assert main_region["outer_extent"] == main_region["inner_extent"] == width
    assert tail_region["outer_extent"] == tail_region["inner_extent"] == extent % width
    assert sum(region["outer_extent"] * region["group_count"] for region in regions) == extent

    loops = []
    tir.stmt_functor.post_order_visit(
        applied["main"].body,
        lambda node: loops.append(node) if isinstance(node, tir.For) else None,
    )
    for region in regions:
        assert region["source_outer_extent"] == extent
        lane_count = region["outer_extent"]
        matching = [core for core in cores if int(core.annotations["tl.sunmmio.multidim.outer_extent"]) == lane_count]
        assert len(matching) == 1
        core = matching[0]
        assert int(core.annotations["tl.sunmmio.multidim.inner_extent"]) == lane_count
        assert int(core.annotations["tl.sunmmio.residency.source_outer_extent"]) == extent
        assert int(core.annotations[GROUP_WIDTH]) == width
        assert len(_calls(core.body, "tl.mma_sunmmio")) == lane_count
        base = core.annotations["tl.sunmmio.residency.group_base"]
        if region["tail"]:
            assert _constant(base, {}) == minimum + (extent // width) * width
        else:
            free = list(tir.analysis.undefined_vars(base))
            assert len(free) == 1, "the main group must retain a live group-base expression"
            owner = [loop for loop in loops if loop.loop_var.same_as(free[0])]
            assert len(owner) == 1, "group_base must reference the surviving SSA group binder"
            assert int(owner[0].min) == 0 and int(owner[0].extent) == region["group_count"]
            for group in range(region["group_count"]):
                assert _constant(base, {free[0]: group}) == minimum + group * width
