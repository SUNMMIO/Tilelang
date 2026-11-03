"""A proved boundary without a solved candidate must not claim Multidim."""
import importlib.util
from pathlib import Path

import tilelang as tl
from tilelang import tvm
from tilelang.utils.target import SUNMMIO_TARGET_DESC

_spec = importlib.util.spec_from_file_location(
    "boundary_proof_helpers", Path(__file__).with_name("test_sunmmio_multidim_boundary_overlap.py"))
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)


def test_boundary_proof_is_default_enabled_but_has_explicit_opt_out(monkeypatch):
    monkeypatch.delenv("TL_SUNMMIO_MULTIDIM_BOUNDARY_OVERLAP", raising=False)
    result = _helpers._analyze(_helpers.make_boundary_module())
    assert bool(_helpers._outer(result).annotations[_helpers.ELIGIBLE])
    monkeypatch.setenv("TL_SUNMMIO_MULTIDIM_BOUNDARY_OVERLAP", "0")
    original = _helpers.make_boundary_module()
    result = _helpers._analyze(original)
    assert tvm.ir.structural_equal(original, result)


def test_boundary_planning_rejection_records_retained_inner_pipeline(monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_MULTIDIM_BOUNDARY_OVERLAP", "1")
    mod = _helpers._analyze(_helpers.make_boundary_module())
    mod = tvm.tir.transform.BindTarget(tvm.target.Target(SUNMMIO_TARGET_DESC))(mod)
    mod = tl.transform.ResolveSunmmioUnit()(mod)
    before = _helpers._outer(mod)
    # Deliberately omit the core planning pass. A legality proof is not a
    # solved schedule, and rejection must not masquerade as Multidim success.
    result = tl.transform.SunmmioRegionCrossOuterPipelinePlanning(
        policy="auto", strategy="multidim_boundary")(mod)
    after = _helpers._outer(result)
    assert str(after.annotations["tl.sunmmio.region_cross_outer.fallback_reason"]) == "inner_ilp_schedule_missing"
    assert str(after.annotations["tl.sunmmio.multidim.selected_mode"]) == "ilp"
    assert str(after.annotations["tl.sunmmio.multidim.fallback_stage"]) == "boundary_overlap"
    assert str(after.annotations["tl.sunmmio.multidim.fallback_reason"]) == "inner_ilp_schedule_missing"
    assert tvm.ir.structural_equal(before.body, after.body)
    assert not result["main"].attrs.get("tl.sunmmio.boundary_overlap.planned", False)
