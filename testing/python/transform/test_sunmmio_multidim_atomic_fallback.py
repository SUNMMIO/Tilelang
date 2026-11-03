import json
import os
from pathlib import Path

import pytest

from tilelang import tvm
from tvm import tir
from tilelang.engine import phase
from tilelang.engine.sunmmio_candidate_selection import compare_pipeline_candidate_work


def _append_pass(stage):
    return lambda mod: mod + (stage,)

@pytest.fixture(autouse=True)
def _clean_multidim_env(monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_STATIC_WORKITEM_POLICY", "off")


def test_multidim_auto_retry_uses_complete_original_module(monkeypatch, tmp_path):
    diagnostic = tmp_path / "fallback.json"
    monkeypatch.setenv("TL_SUNMMIO_MULTIDIM_ATOMIC_FALLBACK_JSON", str(diagnostic))

    monkeypatch.setattr(
        phase.tilelang.transform,
        "AnalyzeSunmmioTileResidency",
        lambda policy: _append_pass("residency"),
    )

    def fail_after_residency(debug, policy):
        def run(mod):
            assert mod == ("original", "residency")
            raise RuntimeError("planned failure")

        return run

    monkeypatch.setattr(
        phase.tilelang.transform,
        "SunmmioMultidimPipelinePlanningILP",
        fail_after_residency,
    )
    monkeypatch.setattr(
        phase.tilelang.transform,
        "SunmmioPipelinePlanningILP",
        lambda debug: _append_pass("1d_plan"),
    )
    monkeypatch.setattr(
        phase.tilelang.transform,
        "InjectSunmmioPipelineILP",
        lambda: _append_pass("1d_inject"),
    )

    result = phase._apply_sunmmio_multidim_atomic(("original",), "auto")

    assert result == ("original", "1d_plan", "1d_inject")
    payload = json.loads(diagnostic.read_text(encoding="utf-8"))
    assert payload["fallback"] == "complete_module_to_1d_ilp"
    assert payload["stage"] == "analysis_planning_or_injection"


def test_multidim_forced_propagates_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(
        phase.tilelang.transform,
        "AnalyzeSunmmioTileResidency",
        lambda policy: _append_pass("residency"),
    )
    monkeypatch.setattr(
        phase.tilelang.transform,
        "SunmmioMultidimPipelinePlanningILP",
        lambda debug, policy: lambda mod: (_ for _ in ()).throw(RuntimeError("forced failure")),
    )

    with pytest.raises(RuntimeError, match="forced failure"):
        phase._apply_sunmmio_multidim_atomic(("original",), "forced")


def test_static_and_regular_candidates_compete_and_ordinary_can_win(monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_STATIC_WORKITEM_POLICY", "auto")
    monkeypatch.setattr(
        phase.tilelang.transform,
        "FlattenSunmmioStaticWorkItems",
        lambda policy: _append_pass("workitem_flatten:" + policy),
    )
    monkeypatch.setattr(
        phase,
        "_module_has_static_workitem_transform",
        lambda mod: mod[-1] == "workitem_flatten:auto",
    )
    monkeypatch.setattr(
        phase.tilelang.transform,
        "SunmmioPipelinePlanningILP",
        lambda debug: _append_pass("inner_plan"),
    )
    monkeypatch.setattr(
        phase.tilelang.transform,
        "SunmmioRegionCrossOuterPipelinePlanning",
        lambda debug, policy, strategy: _append_pass(
            f"outer_plan:{policy}:{strategy}"
        ),
    )
    monkeypatch.setattr(
        phase.tilelang.transform,
        "InjectSunmmioPipelineILP",
        lambda: _append_pass("hierarchical_inject"),
    )
    monkeypatch.setattr(
        phase.tilelang.transform,
        "AnalyzeSunmmioTileResidency",
        lambda policy: _append_pass("residency:" + policy),
    )
    monkeypatch.setattr(
        phase.tilelang.transform,
        "SunmmioMultidimPipelinePlanningILP",
        lambda debug, policy: _append_pass("multidim_plan:" + policy),
    )
    monkeypatch.setattr(
        phase.tilelang.transform,
        "InjectSunmmioMultidimPipelineILP",
        lambda policy: _append_pass("multidim_inject:" + policy),
    )
    monkeypatch.setattr(
        phase,
        "compare_pipeline_candidate_work",
        lambda static, ordinary: 1,
    )

    result = phase._apply_sunmmio_multidim_atomic(("original",), "auto")

    assert result == (
        "original",
        "residency:auto_strict",
        "multidim_plan:auto",
        "multidim_inject:auto",
    )


def test_static_candidate_survives_ordinary_candidate_failure(monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_STATIC_WORKITEM_POLICY", "auto")
    monkeypatch.setattr(
        phase,
        "_apply_sunmmio_static_workitem_candidate",
        lambda original, policy: (("static_plan",), ("static_selected",)),
    )

    def fail_ordinary(original, policy):
        raise RuntimeError("ordinary candidate failure")

    monkeypatch.setattr(
        phase, "_apply_sunmmio_ordinary_multidim_candidate", fail_ordinary
    )

    result = phase._apply_sunmmio_multidim_atomic(("original",), "auto")

    assert result == ("static_selected",)


def test_unapplied_static_candidate_keeps_regular_multidim(monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_STATIC_WORKITEM_POLICY", "auto")
    monkeypatch.setattr(
        phase.tilelang.transform,
        "FlattenSunmmioStaticWorkItems",
        lambda policy: _append_pass("workitem_analysis:" + policy),
    )
    monkeypatch.setattr(
        phase,
        "_module_has_static_workitem_transform",
        lambda mod: False,
    )
    monkeypatch.setattr(
        phase.tilelang.transform,
        "AnalyzeSunmmioTileResidency",
        lambda policy: _append_pass("residency:" + policy),
    )
    monkeypatch.setattr(
        phase.tilelang.transform,
        "SunmmioMultidimPipelinePlanningILP",
        lambda debug, policy: _append_pass("multidim_plan:" + policy),
    )
    monkeypatch.setattr(
        phase.tilelang.transform,
        "InjectSunmmioMultidimPipelineILP",
        lambda policy: _append_pass("multidim_inject:" + policy),
    )

    result = phase._apply_sunmmio_multidim_atomic(("original",), "auto")

    assert result == (
        "original",
        "residency:auto_strict",
        "multidim_plan:auto",
        "multidim_inject:auto",
    )


def test_static_candidate_auto_failure_keeps_regular_multidim(monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_STATIC_WORKITEM_POLICY", "auto")
    monkeypatch.setattr(
        phase.tilelang.transform,
        "FlattenSunmmioStaticWorkItems",
        lambda policy: lambda mod: (_ for _ in ()).throw(
            RuntimeError("candidate failure")
        ),
    )
    monkeypatch.setattr(
        phase.tilelang.transform,
        "AnalyzeSunmmioTileResidency",
        lambda policy: _append_pass("residency:" + policy),
    )
    monkeypatch.setattr(
        phase.tilelang.transform,
        "SunmmioMultidimPipelinePlanningILP",
        lambda debug, policy: _append_pass("multidim_plan:" + policy),
    )
    monkeypatch.setattr(
        phase.tilelang.transform,
        "InjectSunmmioMultidimPipelineILP",
        lambda policy: _append_pass("multidim_inject:" + policy),
    )

    result = phase._apply_sunmmio_multidim_atomic(("original",), "auto")

    assert result == (
        "original",
        "residency:auto_strict",
        "multidim_plan:auto",
        "multidim_inject:auto",
    )


def test_post_sync_failure_retries_original_module(monkeypatch, tmp_path):
    diagnostic = tmp_path / "post_fallback.json"
    monkeypatch.setenv("TL_SUNMMIO_MULTIDIM_ATOMIC_FALLBACK_JSON", str(diagnostic))
    monkeypatch.setattr(phase, "should_enable_sunmmio_pipeline", lambda pass_ctx: True)
    monkeypatch.setattr(phase, "get_sunmmio_pipeline_mode", lambda pass_ctx: "multidim_ilp")
    monkeypatch.setattr(phase, "get_sunmmio_multidim_policy", lambda pass_ctx: "auto")
    monkeypatch.setattr(phase, "_infer_sunmmio_noalias", lambda mod: mod)
    monkeypatch.setattr(
        phase.tilelang.transform,
        "IfStmtBinding",
        lambda: _append_pass("if_binding"),
    )
    monkeypatch.setattr(
        phase,
        "_apply_sunmmio_multidim_atomic",
        lambda mod, policy: mod + ("multidim",),
    )
    monkeypatch.setattr(
        phase,
        "_apply_sunmmio_one_dimensional_ilp",
        lambda mod: mod + ("1d",),
    )

    lowered = []

    def lower(mod):
        lowered.append(mod)
        if mod[-1] == "multidim":
            raise RuntimeError("sync failure")
        return mod + ("lowered",)

    monkeypatch.setattr(phase, "_lower_sunmmio_after_pipeline", lower)

    result = phase.OptimizeForSunmmio(("original",), target=None)

    assert lowered == [
        ("original", "if_binding", "multidim"),
        ("original", "if_binding", "1d"),
    ]
    assert result == ("original", "if_binding", "1d", "lowered")
    payload = json.loads(diagnostic.read_text(encoding="utf-8"))
    assert payload["stage"] == "post_pipeline_or_sync"


def _make_planned_module(
    ii,
    makespan,
    extent,
    command_slots,
    versions,
    outer_extent=None,
    outer_pipeline=None,
):
    data = tir.Var(
        "state",
        tvm.ir.PointerType(tvm.ir.PrimType("float32"), "shared.rsram"),
    )
    buffer = tir.decl_buffer(
        (4,), "float32", name="state", data=data, scope="shared.rsram"
    )
    loop_var = tir.Var("k", "int32")
    body = tir.BufferStore(
        buffer,
        tir.BufferLoad(buffer, [0]) + tir.FloatImm("float32", 1.0),
        [0],
    )
    loop = tir.For(
        loop_var,
        0,
        extent,
        tir.ForKind.SERIAL,
        body,
        annotations={
            "tl.sunmmio.pipeline.applied": tir.IntImm("bool", 1),
            "tl.sunmmio.pipeline.mode": "ilp",
            "ii": ii,
            "makespan": makespan,
            "iterations": 2,
            "tl.sunmmio.pipeline.command_start_times": list(
                range(command_slots)
            ),
            "versioned_buffers": [buffer] * versions,
            "used_buffers": [buffer],
        },
    )
    if outer_extent is not None:
        outer_var = tir.Var("outer", "int32")
        annotations = {}
        if outer_pipeline is not None:
            outer_ii, outer_makespan = outer_pipeline
            annotations = {
                "tl.sunmmio.pipeline.applied": tir.IntImm("bool", 1),
                "tl.sunmmio.pipeline.mode": "hierarchical_ilp",
                "ii": outer_ii,
                "makespan": outer_makespan,
                "iterations": 2,
            }
        loop = tir.For(
            outer_var,
            0,
            outer_extent,
            tir.ForKind.SERIAL,
            loop,
            annotations=annotations,
        )
    root = tir.Block([], [], [], "root", loop, alloc_buffers=[buffer])
    func = tir.PrimFunc(
        [],
        tir.BlockRealize([], tir.const(True, "bool"), root),
    ).with_attr("global_symbol", "main")
    return tvm.IRModule.from_expr(func)


def test_compare_pipeline_candidate_work_uses_normalized_total_work():
    static = _make_planned_module(
        10, 16, 3, 4, 1, outer_extent=4
    )
    ordinary = _make_planned_module(17, 23, 3, 6, 2)

    assert compare_pipeline_candidate_work(static, ordinary) == 1
    assert compare_pipeline_candidate_work(ordinary, static) == -1
    assert compare_pipeline_candidate_work(static, static) == 0


def test_work_normalization_does_not_double_count_nested_pipeline():
    nested = _make_planned_module(
        10,
        16,
        3,
        4,
        1,
        outer_extent=3,
        outer_pipeline=(7, 11),
    )
    same_outer_with_different_inner = _make_planned_module(
        1000,
        2000,
        3,
        8,
        4,
        outer_extent=3,
        outer_pipeline=(7, 11),
    )

    assert compare_pipeline_candidate_work(
        nested, same_outer_with_different_inner
    ) == 0



def test_shadow_scope_redirects_diagnostics_and_restores(monkeypatch, tmp_path):
    official = tmp_path / "official.json"
    env_name = "TL_SUNMMIO_ILP_PROBLEM_JSON"
    monkeypatch.setenv(env_name, str(official))

    assert not phase.sunmmio_shadow_planning_active()
    with phase._sunmmio_shadow_planning_scope():
        redirected = Path(os.environ[env_name])
        assert phase.sunmmio_shadow_planning_active()
        assert redirected != official
        assert redirected.parent != official.parent

    assert not phase.sunmmio_shadow_planning_active()
    assert os.environ[env_name] == str(official)
