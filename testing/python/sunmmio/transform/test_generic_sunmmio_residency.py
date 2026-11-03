"""Legality and selection tests for automatic tile-residency analysis.

These are compiler-IR tests.  They do not establish backend capacity, numerical
correctness, or measured speedup; those require the run's end-to-end matrix.
"""

import json

import pytest
import tilelang as tl
from tilelang import tvm
from tvm import arith, tir


OUTER = "tl.sunmmio.multidim.workitem_outer"
INNER = "tl.sunmmio.multidim.workitem_inner"
ELIGIBLE = "tl.sunmmio.residency.generic_eligible"
APPLIED = "tl.sunmmio.residency.generic_applied"


def _region(buffer, access, indices, extents):
    return tir.Call(
        "handle",
        tvm.ir.Op.get("tl.tileop.region"),
        [tir.BufferLoad(buffer, indices), tir.IntImm("int32", access), *extents],
    )


def _op(name, *args):
    return tir.Evaluate(tir.Call("handle", tvm.ir.Op.get(name), list(args)))


def _buffer(name, shape, scope):
    data = tir.Var(name, tvm.ir.PointerType(tvm.ir.PrimType("bfloat16"), scope))
    return tir.decl_buffer(shape, "bfloat16", name=name, data=data, scope=scope)


def _make_module(
    *,
    marked=False,
    slot_state=False,
    reuse_extent=3,
    name_prefix="natural",
    fault=None,
    consumers=1,
    grouped=False,
    batch_wrapper=False,
    extra_readback=False,
):
    """Natural GEMM nest; the compiler, not the kernel, versions local state."""
    expert = tir.Var(name_prefix + "_row", "int32")
    column_dtype = "int64" if isinstance(reuse_extent, int) and reuse_extent > (1 << 31) - 1 else "int32"
    column = tir.Var(name_prefix + "_column", column_dtype)
    reduction = tir.Var(name_prefix + "_reduction", "int32")
    a = _buffer(name_prefix + "_left", (2, 2, 8, 32, 32), "global")
    b = _buffer(name_prefix + "_right", (2, 2, 8, 32, 32), "global")
    dst = _buffer(name_prefix + "_result", (2, 8, consumers, 32, 32), "global")
    index_buffer = tir.decl_buffer((1,), "int32", name="runtime_index")
    staging = _buffer(name_prefix + "_staging", (32, 32), "shared.rsram")
    resident_shape = (64, 32) if fault == "partial_collective" else (32, 32)
    resident = _buffer(name_prefix + "_resident", resident_shape, "shared.asram")
    weight = _buffer(name_prefix + "_weight", (32, 32), "shared.wsram")
    state_shape = (8, 32, 32) if slot_state else (32, 32)
    if fault == "insufficient_slots":
        state_shape = (1, 32, 32)
    accumulators = [_buffer(name_prefix + "_acc" + str(i), state_shape, "shared.rsram") for i in range(consumers)]
    outputs = [_buffer(name_prefix + "_out" + str(i), state_shape, "shared.rsram") for i in range(consumers)]
    if fault == "aliased_state_views":
        outputs[0] = tir.decl_buffer(
            state_shape,
            "bfloat16",
            name="state_alias",
            data=accumulators[0].data,
            scope="shared.rsram",
        )
    state_indices = [column, 0, 0] if slot_state else [0, 0]
    state_extents = [1, 32, 32] if slot_state else [32, 32]
    a_column = column if fault == "source_depends_on_reuse" else 0
    a_row = tir.call_extern("int32", "unknown_index_effect") if fault == "nested_core_effect" else expert
    if fault == "hidden_index_read_modified":
        a_row = tir.BufferLoad(index_buffer, [0])
    load = _op(
        "tl.dma_copy",
        _region(a, 1, [a_row, reduction, a_column, 0, 0], [1, 1, 1, 32, 32]),
        _region(staging, 2, [0, 0], [32, 32]),
        0,
    )
    share = _op(
        "tl.broadcast_",
        _region(staging, 1, [0, 0], [32, 32]),
        _region(resident, 2, [0, 0], [32, 32]),
        0,
        tir.IntImm("int64", 15),
        0,
    )
    weight_load = _op(
        "tl.broadcast_",
        _region(b, 1, [expert, reduction, column, 0, 0], [1, 1, 1, 32, 32]),
        _region(weight, 2, [0, 0], [32, 32]),
        1,
        tir.IntImm("int64", 15),
        0,
    )
    computes = []
    for accumulator, output in zip(accumulators, outputs):
        computes.append(
            _op(
                "tl.mma_sunmmio",
                _region(resident, 1, [0, 0], [32, 32]),
                _region(weight, 1, [0, 0], [32, 32]),
                _region(accumulator, 3, state_indices, state_extents),
                tir.IntImm("bool", 0),
                tir.IntImm("bool", 0),
                tir.IntImm("bool", 0) if fault in {"full_float_clear", "partial_float_clear"} else reduction == 0,
                0,
                _region(output, 2, state_indices, state_extents),
                reduction == 1,
            )
        )
    core_stmts = [load, share, weight_load, *computes]
    if fault == "unknown_core_effect":
        core_stmts.insert(2, tir.Evaluate(tir.call_extern("int32", "unknown_effect")))
    if fault == "resident_overwritten":
        core_stmts.append(_op("tl.dma_copy", _region(weight, 1, [0, 0], [32, 32]), _region(resident, 2, [0, 0], [32, 32]), 0))
    if fault == "conditional_mma":
        core_stmts[-1] = tir.IfThenElse(column < 2, computes[-1], computes[-1])
    if fault == "conditional_collective":
        core_stmts[1] = tir.IfThenElse(expert == 0, share, tir.Evaluate(0))
    if fault == "source_modified_in_core":
        core_stmts.append(tir.BufferStore(a, tir.const(0, "bfloat16"), [expert, 0, 0, 0, 0]))
    if grouped:
        core_stmts = [tir.SeqStmt(core_stmts[:2]), tir.SeqStmt(core_stmts[2:])]
    core_extent = column + 1 if fault == "dependent_core_extent" else 2
    core = tir.For(
        reduction,
        0,
        core_extent,
        tir.ForKind.SERIAL,
        tir.SeqStmt(core_stmts),
        annotations={"num_stages": 2},
        step=2 if fault == "nonunit_core_step" else 1,
    )
    prefix = []
    if not slot_state:
        prefix = [tir.BufferStore(acc, tir.const(0, "bfloat16"), state_indices) for acc in accumulators]
    if fault == "full_float_clear":
        prefix = []
        for index, acc in enumerate(accumulators):
            row = tir.Var("clear_row_" + str(index), "int32")
            col = tir.Var("clear_col_" + str(index), "int32")
            prefix.append(
                tir.For(
                    row,
                    0,
                    32,
                    tir.ForKind.SERIAL,
                    tir.For(col, 0, 32, tir.ForKind.SERIAL, tir.BufferStore(acc, tir.FloatImm("bfloat16", 0.0), [row, col])),
                )
            )
    suffix = [
        _op(
            "tl.dma_copy",
            _region(output, 1, state_indices, state_extents),
            _region(dst, 2, [expert, column, i, 0, 0], [1, 1, 1, 32, 32]),
            0,
        )
        for i, output in enumerate(outputs)
    ]
    if extra_readback:
        suffix.append(tir.Evaluate(tir.BufferLoad(accumulators[0], state_indices)))
    escape_load = None
    if fault == "state_escapes":
        escape_load = tir.BufferLoad(accumulators[0], state_indices)
        suffix.append(tir.Evaluate(escape_load))
    if fault in {"suffix_writes_source", "suffix_alias_writes_source", "nonfinal_suffix_writes_source"}:
        target = a
        if fault == "suffix_alias_writes_source":
            target = tir.decl_buffer(a.shape, a.dtype, name="different_view", data=a.data)
        suffix.append(tir.BufferStore(target, tir.const(0, "bfloat16"), [expert, 0, 0, 0, 0]))
        if fault == "nonfinal_suffix_writes_source":
            suffix.append(tir.Evaluate(tir.BufferLoad(accumulators[0], state_indices)))
    if fault == "unknown_suffix_effect":
        suffix.append(tir.Evaluate(tir.call_extern("int32", "unknown_effect")))
    if fault == "prefix_writes_source":
        prefix.append(tir.BufferStore(a, tir.const(0, "bfloat16"), [expert, 0, 0, 0, 0]))
    if fault == "nonfinal_prefix_writes_source":
        prefix.insert(0, tir.BufferStore(a, tir.const(0, "bfloat16"), [expert, 0, 0, 0, 0]))
    if fault == "hidden_index_read_modified":
        suffix.append(tir.BufferStore(index_buffer, column % 2, [0]))
    if fault == "uniform_boundary_condition":
        prefix = [tir.IfThenElse(expert == 0, prefix[0] if len(prefix) == 1 else tir.SeqStmt(prefix), tir.Evaluate(0))]
        suffix = [tir.IfThenElse(expert == 0, suffix[0] if len(suffix) == 1 else tir.SeqStmt(suffix), tir.Evaluate(0))]
    inner_body = tir.SeqStmt([*prefix, core, *suffix])
    if grouped:
        inner_body = tir.SeqStmt([tir.SeqStmt([*prefix, core]), *suffix])
    inner = tir.For(
        column,
        0,
        reuse_extent,
        tir.ForKind.SERIAL,
        inner_body,
        annotations={INNER: 1} if marked else {},
        step=2 if fault == "nonunit_reuse_step" else 1,
    )
    outer = tir.For(
        expert,
        0,
        2,
        tir.ForKind.SERIAL,
        inner,
        annotations={OUTER: 1} if marked else {},
    )
    if batch_wrapper:
        batch = tir.Var("independent_batch", "int32")
        outer = tir.For(batch, 0, 2, tir.ForKind.SERIAL, outer)
    if escape_load is not None:
        outer = tir.SeqStmt([outer, tir.Evaluate(escape_load)])
    locals_ = [staging, resident, weight, *accumulators, *outputs]
    if fault == "aliased_state_views":
        locals_ = [item for item in locals_ if not item.same_as(outputs[0])]
    root = tir.Block([], [], [], "root", outer, alloc_buffers=locals_)
    params = [a.data, b.data, dst.data, index_buffer.data]
    if isinstance(reuse_extent, tir.Var):
        params.append(reuse_extent)
    func = (
        tir.PrimFunc(
            params,
            tir.BlockRealize([], tir.const(True, "bool"), root),
            buffer_map={a.data: a, b.data: b, dst.data: dst, index_buffer.data: index_buffer},
        )
        .with_attr("global_symbol", "main")
        .with_attr("tir.noalias", True)
    )
    return tvm.IRModule.from_expr(func)


@pytest.fixture(autouse=True)
def _clean_residency_env(monkeypatch):
    for key in (
        "TL_SUNMMIO_RESIDENCY_POLICY",
        "TL_SUNMMIO_RESIDENCY_REPORT",
        "TL_SUNMMIO_RESIDENCY_MAX_ADDED_BYTES",
        "TL_SUNMMIO_RESIDENCY_MAX_REUSE_EXTENT",
        "TL_SUNMMIO_MULTIDIM_INTERCHANGE",
        "TL_SUNMMIO_ILP_ENABLE_LATENCY_GCD",
        "FINAL_ILP_ENABLE_LATENCY_GCD",
    ):
        monkeypatch.delenv(key, raising=False)


def _analyze(mod, tmp_path, monkeypatch, policy="auto_strict"):
    path = tmp_path / "residency.json"
    monkeypatch.setenv("TL_SUNMMIO_RESIDENCY_REPORT", str(path))
    result = tl.transform.AnalyzeSunmmioTileResidency(policy)(mod)
    if policy == "off" and not path.exists():
        return result, {"candidates": []}
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 2
    assert report["policy"] == policy
    return result, report


def _accepted(report):
    return [candidate for candidate in report["candidates"] if candidate["accepted"]]


def _marked(mod, key):
    found = []
    tir.stmt_functor.post_order_visit(
        mod["main"].body,
        lambda node: found.append(node) if isinstance(node, tir.For) and bool(node.annotations.get(key, False)) else None,
    )
    return found


def _without_diagnostic_annotations(mod):
    def rewrite(node):
        if not isinstance(node, tir.For):
            return None
        annotations = {
            str(key): value
            for key, value in node.annotations.items()
            if not str(key).startswith("tl.sunmmio.residency.") and not str(key).startswith("tl.sunmmio.multidim.")
        }
        return tir.For(node.loop_var, node.min, node.extent, node.kind, node.body, node.thread_binding, annotations, node.step)

    body = tir.stmt_functor.ir_transform(mod["main"].body, None, rewrite, ["tir.For"])
    return tvm.IRModule.from_expr(mod["main"].with_body(body))


def _same_statements(before, after):
    return tvm.ir.structural_equal(
        _without_diagnostic_annotations(before),
        _without_diagnostic_annotations(after),
        map_free_vars=True,
    )


@pytest.mark.parametrize("policy", ["off", "shadow"])
def test_non_mutating_policies_preserve_complete_ir(tmp_path, monkeypatch, policy):
    before = _make_module()
    after, report = _analyze(before, tmp_path, monkeypatch, policy)
    assert tvm.ir.structural_equal(before, after, map_free_vars=True)
    if policy == "shadow":
        assert _accepted(report)


@pytest.mark.parametrize("extent", [3, 4])
def test_marker_free_natural_nest_discovers_resident_tile(tmp_path, monkeypatch, extent):
    before = _make_module(reuse_extent=extent)
    after, report = _analyze(before, tmp_path, monkeypatch)
    candidates = _accepted(report)
    assert candidates
    assert any(int(item["reuse_extent"]) == extent for item in candidates)
    assert _marked(after, ELIGIBLE)
    assert not _marked(after, APPLIED), "analysis is not evidence of an executed rewrite"


def test_auto_rejects_residency_when_reuse_cannot_amortize_shared_prefix(tmp_path, monkeypatch):
    before = _make_module(reuse_extent=2)
    after, report = _analyze(before, tmp_path, monkeypatch)
    assert not _accepted(report)
    assert any(item["reason"] == "insufficient_reuse_amortization" for item in report["candidates"])
    assert _same_statements(before, after)


def test_forced_can_profile_low_amortization_candidate(tmp_path, monkeypatch):
    _, report = _analyze(_make_module(reuse_extent=2), tmp_path, monkeypatch, "forced")
    assert _accepted(report)


def test_multiple_consumers_amortize_short_reuse(tmp_path, monkeypatch):
    _, report = _analyze(_make_module(reuse_extent=2, consumers=2), tmp_path, monkeypatch)
    assert _accepted(report)


def test_renaming_buffers_and_coordinates_does_not_change_acceptance(tmp_path, monkeypatch):
    _, first = _analyze(_make_module(name_prefix="first"), tmp_path, monkeypatch)
    _, second = _analyze(_make_module(name_prefix="completely_different"), tmp_path, monkeypatch)
    fields = ("reuse_extent", "shared_prefix_count", "consumer_count", "added_bytes")
    assert [[item[key] for key in fields] for item in _accepted(first)] == [[item[key] for key in fields] for item in _accepted(second)]
    assert _accepted(first)


def test_nested_statement_groups_and_three_plus_loops(tmp_path, monkeypatch):
    _, report = _analyze(_make_module(grouped=True, batch_wrapper=True), tmp_path, monkeypatch)
    assert _accepted(report)


def test_multiple_consumers_are_not_a_four_statement_template(tmp_path, monkeypatch):
    _, report = _analyze(_make_module(consumers=2), tmp_path, monkeypatch)
    candidates = _accepted(report)
    assert candidates
    assert any(int(item["consumer_count"]) >= 2 for item in candidates)


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
def test_unsafe_or_unproved_nest_is_not_silently_rewritten(tmp_path, monkeypatch, fault):
    before = _make_module(fault=fault)
    after, report = _analyze(before, tmp_path, monkeypatch)
    assert not _accepted(report)
    assert report["candidates"]
    assert all(item["reason"] for item in report["candidates"])
    assert _same_statements(before, after)


@pytest.mark.parametrize("extent", [0, tir.Var("runtime_columns", "int32"), 1 << 40])
def test_unproved_or_excessive_extent_rejects_safely(tmp_path, monkeypatch, extent):
    before = _make_module(reuse_extent=extent)
    after, report = _analyze(before, tmp_path, monkeypatch)
    assert not _accepted(report)
    assert _same_statements(before, after)


def test_added_allocation_budget_is_not_claimed_as_hardware_capacity(tmp_path, monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_RESIDENCY_MAX_ADDED_BYTES", "0")
    before = _make_module()
    after, report = _analyze(before, tmp_path, monkeypatch)
    assert not _accepted(report)
    assert any(item["reason"] == "allocation_budget" for item in report["candidates"])
    assert _same_statements(before, after)


def test_forced_rejection_is_explicit(tmp_path, monkeypatch):
    with pytest.raises(tvm.TVMError):
        _analyze(_make_module(fault="unknown_core_effect"), tmp_path, monkeypatch, "forced")


def test_marked_independent_slots_remain_valid(tmp_path, monkeypatch):
    _, report = _analyze(_make_module(marked=True, slot_state=True), tmp_path, monkeypatch)
    assert _accepted(report)


def test_accumulator_slot_capacity_must_be_proved(tmp_path, monkeypatch):
    before = _make_module(marked=True, slot_state=True, fault="insufficient_slots")
    after, report = _analyze(before, tmp_path, monkeypatch)
    assert not _accepted(report)
    assert _same_statements(before, after)


def test_reduction_order_and_output_predicates_survive_analysis(tmp_path, monkeypatch):
    before = _make_module()
    after, report = _analyze(before, tmp_path, monkeypatch)
    assert _accepted(report)
    calls = []
    tir.stmt_functor.post_order_visit(
        after["main"].body,
        lambda node: calls.append(node)
        if isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.mma_sunmmio"
        else None,
    )
    assert len(calls) == 1
    original_calls = []
    tir.stmt_functor.post_order_visit(
        before["main"].body,
        lambda node: original_calls.append(node)
        if isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.mma_sunmmio"
        else None,
    )
    assert tvm.ir.structural_equal(calls[0], original_calls[0], map_free_vars=True)
    reduction = None
    for value in (calls[0].args[5], calls[0].args[8]):
        assert isinstance(value, tir.EQ)
        reduction = value.a
    analyzer = arith.Analyzer()
    for k in (0, 1):
        replace = {reduction: tir.IntImm("int32", k)}
        assert bool(analyzer.simplify(tir.stmt_functor.substitute(calls[0].args[5], replace))) == (k == 0)
        assert bool(analyzer.simplify(tir.stmt_functor.substitute(calls[0].args[8], replace))) == (k == 1)


def test_global_alias_contract_must_be_explicit(tmp_path, monkeypatch):
    original = _make_module()
    before = tvm.IRModule.from_expr(original["main"].without_attr("tir.noalias"))
    after, report = _analyze(before, tmp_path, monkeypatch)
    assert not _accepted(report)
    assert any(item["reason"] == "global_alias_unproven" for item in report["candidates"])
    assert _same_statements(before, after)


def test_overwritten_asram_does_not_prevent_safe_staging_only_reuse(tmp_path, monkeypatch):
    _, report = _analyze(_make_module(fault="resident_overwritten"), tmp_path, monkeypatch)
    candidates = _accepted(report)
    assert candidates
    assert all(int(item["shared_prefix_count"]) == 1 for item in candidates)


def test_pure_boundary_if_else_is_preserved_and_supported(tmp_path, monkeypatch):
    before = _make_module(fault="uniform_boundary_condition")
    after, report = _analyze(before, tmp_path, monkeypatch)
    assert _accepted(report)
    conditions = []
    tir.stmt_functor.post_order_visit(
        after["main"].body,
        lambda node: conditions.append(node) if isinstance(node, tir.IfThenElse) else None,
    )
    assert len(conditions) == 2


@pytest.mark.parametrize("fault", ["suffix_writes_source", "unknown_core_effect", "missing_noalias"])
def test_rejected_analysis_cannot_force_residency_interchange(tmp_path, monkeypatch, fault):
    before = _make_module(fault=None if fault == "missing_noalias" else fault, extra_readback=True)
    if fault == "missing_noalias":
        before = tvm.IRModule.from_expr(before["main"].without_attr("tir.noalias"))
    monkeypatch.setenv("TL_SUNMMIO_MULTIDIM_INTERCHANGE", "1")
    monkeypatch.setenv("TL_SUNMMIO_RESIDENCY_POLICY", "auto_strict")
    analyzed, report = _analyze(before, tmp_path, monkeypatch)
    assert not _accepted(report)
    normalized = tl.transform.SunmmioMultidimRegion("forced")(analyzed)
    after = tl.transform.ApplySunmmioMultidimInterchange("forced")(normalized)
    assert not _marked(after, APPLIED)
    assert not _marked(after, "tl.sunmmio.multidim.interchange.applied")
    assert _same_statements(before, after)


@pytest.mark.parametrize("grouped", [False, True])
def test_natural_candidate_really_versions_state_and_rewrites_in_auto(tmp_path, monkeypatch, grouped):
    analyzed, report = _analyze(_make_module(grouped=grouped), tmp_path, monkeypatch)
    assert _accepted(report)
    normalized = tl.transform.SunmmioMultidimRegion("auto")(analyzed)
    after = tl.transform.ApplySunmmioMultidimInterchange("auto")(normalized)
    pipelines = _marked(after, APPLIED)
    assert len(pipelines) == 1
    core = pipelines[0]
    assert int(core.min) == 0 and int(core.extent) == 2
    calls = []
    tir.stmt_functor.post_order_visit(
        core.body,
        lambda node: calls.append(node)
        if isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.mma_sunmmio"
        else None,
    )
    assert len(calls) == 3
    accumulator_data = [call.args[2].args[0].buffer.data for call in calls]
    output_data = [call.args[7].args[0].buffer.data for call in calls]
    assert len(set(accumulator_data)) == 3
    assert len(set(output_data)) == 3
    analyzer = arith.Analyzer()
    for call in calls:
        for k in (0, 1):
            subst = {core.loop_var: tir.IntImm("int32", k)}
            assert bool(analyzer.simplify(tir.stmt_functor.substitute(call.args[5], subst))) == (k == 0)
            assert bool(analyzer.simplify(tir.stmt_functor.substitute(call.args[8], subst))) == (k == 1)


def test_slotted_candidate_uses_only_generic_residency(tmp_path, monkeypatch):
    path = tmp_path / "selection.json"
    monkeypatch.setenv("TL_SUNMMIO_RESIDENCY_REPORT", str(path))
    monkeypatch.setenv("TL_SUNMMIO_RESIDENCY_POLICY", "auto_strict")
    after = tl.transform.FlattenSunmmioStaticWorkItems("auto")(_make_module(marked=True, slot_state=True))
    report = json.loads(path.read_text(encoding="utf-8"))
    assert _marked(after, APPLIED)
    assert any(item["selected_strategy"] == "generic_residency" for item in report["candidates"])
    assert all("legacy_equivalent" not in item for item in report["candidates"])


def test_full_float_zero_initialization_supports_accumulate_only_mma(tmp_path, monkeypatch):
    _, report = _analyze(_make_module(fault="full_float_clear"), tmp_path, monkeypatch)
    assert _accepted(report)


@pytest.mark.parametrize("step", [2, tir.Var("runtime_clear_step", "int32")])
def test_nonunit_or_symbolic_initialization_step_cannot_prove_full_tile(tmp_path, monkeypatch, step):
    original = _make_module(fault="full_float_clear")

    def rewrite(node):
        if isinstance(node, tir.For) and node.loop_var.name.startswith("clear_row_"):
            return tir.For(node.loop_var, node.min, node.extent, node.kind, node.body, node.thread_binding, node.annotations, step)
        return None

    body = tir.stmt_functor.ir_transform(original["main"].body, None, rewrite, ["tir.For"])
    func = original["main"].with_body(body)
    if isinstance(step, tir.Var):
        func = tir.PrimFunc([*func.params, step], func.body, func.ret_type, func.buffer_map, func.attrs, func.span)
    before = tvm.IRModule.from_expr(func)
    after, report = _analyze(before, tmp_path, monkeypatch)
    assert not _accepted(report), "a strided clear does not initialize every accumulator element"
    assert any(item["reason"] == "accumulator_not_fully_initialized" for item in report["candidates"])
    assert _same_statements(before, after)
