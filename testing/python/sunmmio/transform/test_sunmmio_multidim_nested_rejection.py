"""Rejected outer candidates must not hide or misidentify independent cores."""

import json

import pytest
import tilelang as tl
from tilelang import tvm
from tvm import tir


PREFIX = "tl.sunmmio.multidim."
DECISION = PREFIX + "core_decision_id"
CANDIDATE = PREFIX + "candidate_id"
RESIDENCY = "tl.sunmmio.residency.reject_reason"


def _core(var, minimum, extent):
    return tir.For(var, minimum, extent, tir.ForKind.SERIAL,
                   tir.Evaluate(var + minimum), annotations={"num_stages": 2})


def _loops(stmt):
    found = []
    tir.stmt_functor.pre_order_visit(stmt, lambda n: (found.append(n)
        if isinstance(n, tir.For) else None) is None)
    return found


def _normalize(body, tmp_path, monkeypatch, policy="auto"):
    report = tmp_path / "regions.json"
    monkeypatch.setenv("TL_SUNMMIO_MULTIDIM_REGION_JSON", str(report))
    monkeypatch.setenv("TL_SUNMMIO_MULTIDIM_INTERCHANGE", "0")
    monkeypatch.setenv("TL_SUNMMIO_MULTIDIM_BOUNDARY_OVERLAP", "1")
    original = tvm.IRModule({"main": tir.PrimFunc([], body)})
    transformed = tl.transform.SunmmioMultidimRegion(policy)(original)
    return original, transformed, json.loads(report.read_text())["regions"]


@pytest.mark.parametrize("tail_first", [False, True])
def test_outer_rejection_does_not_hide_nested_same_named_core(tmp_path, monkeypatch, tail_first):
    by, bx = tir.Var("by", "int32"), tir.Var("bx", "int32")
    main_var, tail_var = tir.Var("k", "int32"), tir.Var("k", "int32")
    main = _core(main_var, 0, bx + 1)
    tail = _core(tail_var, 0, tir.IntImm("int32", 4))
    nested = tir.For(bx, 0, 3, tir.ForKind.SERIAL, main,
                     annotations={RESIDENCY: "dynamic_core_range"})
    original_body = tir.For(by, 0, 1, tir.ForKind.SERIAL,
                            tir.SeqStmt([tail, nested] if tail_first else [nested, tail]))
    _, transformed, regions = _normalize(original_body, tmp_path, monkeypatch)
    loops = _loops(transformed["main"].body)
    selected = [loop for loop in loops if PREFIX + "selected_mode" in loop.annotations]
    assert len(selected) == 2 and len(regions) == 2
    assert all("num_stages" in loop.annotations for loop in selected)
    assert len({int(loop.annotations[DECISION]) for loop in selected}) == 2
    events = {entry["core_decision_id"]: entry for entry in regions}
    for var, reason, stage, extent in (
        (main_var, "dynamic_core_range", "generic_residency", None),
        (tail_var, "outer_extent_too_short", "region_discovery", 4),
    ):
        loop = next(loop for loop in selected if loop.loop_var.same_as(var))
        assert str(loop.annotations[PREFIX + "selected_mode"]) == "ilp"
        assert str(loop.annotations[PREFIX + "fallback_reason"]) == reason
        assert str(loop.annotations[PREFIX + "fallback_stage"]) == stage
        event = events[int(loop.annotations[DECISION])]
        assert event["accepted"] is False and event["reason"] == reason
        assert event["fallback_stage"] == stage and event["inner_extent"] == extent
        assert event["candidate_id"] == int(loop.annotations[CANDIDATE])
        assert event["outer_extent"] == (3 if var.same_as(main_var) else 1)
        if extent is None:
            assert "bx" in event["inner_extent_expr"]
        expected = main if var.same_as(main_var) else tail
        assert tvm.ir.structural_equal(loop.body, expected.body)
        assert tvm.ir.structural_equal(loop.extent, expected.extent)
    assert not any(entry["reason"] == "missing_static_outer_region" for entry in regions)
    assert tir.analysis.verify_ssa(transformed["main"])
    assert not tir.analysis.undefined_vars(transformed["main"].body, [])


def test_multiple_direct_cores_have_distinct_decisions_and_one_candidate(tmp_path, monkeypatch):
    outer = tir.Var("outer", "int32")
    a, b = tir.Var("k", "int32"), tir.Var("k", "int32")
    body = tir.For(outer, 0, 3, tir.ForKind.SERIAL, tir.SeqStmt([_core(a, 0, 4), _core(b, 0, 5)]))
    _, transformed, regions = _normalize(body, tmp_path, monkeypatch)
    selected = [loop for loop in _loops(transformed["main"].body) if DECISION in loop.annotations]
    assert len(selected) == 2 and len(regions) == 2
    assert {int(loop.annotations[DECISION]) for loop in selected} == {entry["core_decision_id"] for entry in regions}
    assert len({entry["core_decision_id"] for entry in regions}) == 2
    assert len({entry["candidate_id"] for entry in regions}) == 1
    assert all(entry["reason"] == "multiple_inner_pipeline_loops" for entry in regions)
    assert [entry["inner_extent"] for entry in regions] == [4, 5]


def test_rejected_nonzero_ranges_are_not_reported_as_zero(tmp_path, monkeypatch):
    outer, inner = tir.Var("outer", "int32"), tir.Var("inner", "int32")
    _, _, regions = _normalize(tir.For(outer, 7, 1, tir.ForKind.SERIAL, _core(inner, 5, 4)), tmp_path, monkeypatch)
    assert len(regions) == 1
    assert {name: regions[0][name] for name in ("outer_min", "outer_extent", "inner_min", "inner_extent")} == {
        "outer_min": 7, "outer_extent": 1, "inner_min": 5, "inner_extent": 4}


def test_top_level_core_gets_one_self_decision(tmp_path, monkeypatch):
    core = _core(tir.Var("k", "int32"), 0, 4)
    _, transformed, regions = _normalize(core, tmp_path, monkeypatch)
    assert len(regions) == 1 and regions[0]["reason"] == "missing_static_outer_region"
    assert len(_loops(transformed["main"].body)) == 1
    assert transformed["main"].body.loop_var.same_as(core.loop_var)
    assert DECISION in transformed["main"].body.annotations


def test_boundary_eligibility_priority_is_unchanged(tmp_path, monkeypatch):
    outer, inner = tir.Var("outer", "int32"), tir.Var("inner", "int32")
    body = tir.For(outer, 0, 1, tir.ForKind.SERIAL, _core(inner, 0, 4), annotations={
        "tl.sunmmio.boundary_overlap.eligible": 1, RESIDENCY: "no_invariant_transfer_prefix"})
    original, transformed, regions = _normalize(body, tmp_path, monkeypatch)
    assert regions == [] and tvm.ir.structural_equal(original, transformed)


def test_forced_rejection_still_raises(tmp_path, monkeypatch):
    outer, inner = tir.Var("outer", "int32"), tir.Var("inner", "int32")
    with pytest.raises(tvm.error.TVMError, match="outer_extent_too_short"):
        _normalize(tir.For(outer, 0, 1, tir.ForKind.SERIAL, _core(inner, 0, 4)), tmp_path, monkeypatch, "forced")


@pytest.mark.parametrize("policy", ["auto", "forced"])
def test_generic_proof_rejection_is_authoritative_in_every_policy(tmp_path, monkeypatch, policy):
    outer, inner = tir.Var("outer", "int32"), tir.Var("k", "int32")
    core = _core(inner, 0, outer + 1)
    body = tir.For(outer, 0, 3, tir.ForKind.SERIAL, core,
                   annotations={RESIDENCY: "dynamic_core_range"})
    _, transformed, regions = _normalize(body, tmp_path, monkeypatch, policy)
    result = transformed["main"].body
    assert len(regions) == 1
    assert regions[0]["reason"] == "dynamic_core_range"
    assert regions[0]["fallback_stage"] == "generic_residency"
    assert regions[0]["accepted"] is False
    assert PREFIX + "selected_mode" not in result.annotations
    assert result.body.loop_var.same_as(inner)
    assert str(result.body.annotations[PREFIX + "selected_mode"]) == "ilp"
    assert str(result.body.annotations[PREFIX + "fallback_stage"]) == "generic_residency"
    assert str(result.body.annotations[PREFIX + "fallback_reason"]) == "dynamic_core_range"
    assert int(result.body.annotations[DECISION]) == regions[0]["core_decision_id"]
    assert tvm.ir.structural_equal(result.body.body, core.body)
    assert tvm.ir.structural_equal(result.body.extent, core.extent)
