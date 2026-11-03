import json
import os
from contextlib import contextmanager

import tilelang as tl
from tilelang import tvm
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from tvm import tir


ODMA0 = 2
ODMA1 = 3
ASRAM_IN = 8
ASRAM_OUT = 9


def _region(buffer, indices, access_mask, extents):
    return tir.Call(
        "handle",
        tvm.ir.Op.get("tl.tileop.region"),
        [tir.BufferLoad(buffer, indices), access_mask, *extents],
    )


def _async_call(op_name, *args):
    return tir.Evaluate(tir.Call("handle", tvm.ir.Op.get(op_name), list(args)))


@contextmanager
def _scoped_env(updates):
    old = {key: os.environ.get(key) for key in updates}
    for key, value in updates.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_ilp(func, output_path, faster=None):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with (
        target,
        _scoped_env(
            {
                "TL_SUNMMIO_FASTER": faster,
                "TL_SUNMMIO_ILP_PROBLEM_JSON": output_path,
            }
        ),
    ):
        tl.transform.SunmmioPipelinePlanningILP(debug=False)(mod)
    return json.loads(output_path.read_text(encoding="utf-8"))


def _run_greedy(func, output_path):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with (
        target,
        _scoped_env(
            {
                "TL_SUNMMIO_FASTER": None,
                "FINAL_ILP_FASTER": None,
                "FINAL_FORK_JOIN_GEMM_ILP_FASTER": None,
                "FINAL_QKV_SUMMA_GEMM_ILP_FASTER": None,
                "TL_SUNMMIO_PIPELINE_GRAPH_JSON": output_path,
            }
        ),
    ):
        tl.transform.SunmmioPipelinePlanning(debug=False)(mod)
    return json.loads(output_path.read_text(encoding="utf-8"))


def _make_transpose_pipeline():
    a = tir.decl_buffer((2, 32, 32), "bfloat16", name="a")
    b = tir.decl_buffer((2, 32, 32), "bfloat16", name="b")
    src = tir.decl_buffer((32, 32), "bfloat16", name="src", scope="shared.rsram")
    dst = tir.decl_buffer((32, 32), "bfloat16", name="dst", scope="shared.rsram")
    k = tir.Var("k", "int32")
    body = tir.SeqStmt(
        [
            _async_call(
                "tl.dma_copy",
                _region(a, [k, 0, 0], 1, [1, 32, 32]),
                _region(src, [0, 0], 2, [32, 32]),
                0,
            ),
            _async_call(
                "tl.sunmmio_transpose",
                _region(src, [0, 0], 1, [32, 32]),
                _region(dst, [0, 0], 2, [32, 32]),
            ),
            _async_call(
                "tl.dma_copy",
                _region(dst, [0, 0], 1, [32, 32]),
                _region(b, [k, 0, 0], 2, [1, 32, 32]),
                0,
            ),
        ]
    )
    loop = tir.For(
        k,
        0,
        2,
        tir.ForKind.SERIAL,
        body,
        annotations={"num_stages": tir.IntImm("int32", 2)},
    )
    root = tir.Block([], [], [], "root", loop, alloc_buffers=[src, dst])
    return tir.PrimFunc(
        [a.data, b.data],
        tir.BlockRealize([], True, root),
        buffer_map={a.data: a, b.data: b},
    )


def test_transpose_parser_access_and_resource_are_consistent(tmp_path):
    problem = _run_ilp(_make_transpose_pipeline(), tmp_path / "transpose_problem.json")
    assert problem["N"] == 3
    assert problem["commands"]["1"]["resources"] == [ODMA1]
    assert [0, 1] in problem["dep_edges"]
    assert [1, 2] in problem["dep_edges"]
    assert problem["delta"]["0,1"] == 0
    assert problem["delta"]["1,2"] == 0


def _make_dram_asram_pipeline():
    a = tir.decl_buffer((2, 32, 32), "bfloat16", name="a")
    b = tir.decl_buffer((2, 32, 32), "bfloat16", name="b")
    stage = tir.decl_buffer((32, 32), "bfloat16", name="stage", scope="shared.asram")
    k = tir.Var("k", "int32")
    body = tir.SeqStmt(
        [
            _async_call(
                "tl.dma_copy",
                _region(a, [k, 0, 0], 1, [1, 32, 32]),
                _region(stage, [0, 0], 2, [32, 32]),
                0,
            ),
            _async_call(
                "tl.dma_copy",
                _region(stage, [0, 0], 1, [32, 32]),
                _region(b, [k, 0, 0], 2, [1, 32, 32]),
                0,
            ),
        ]
    )
    loop = tir.For(
        k,
        0,
        2,
        tir.ForKind.SERIAL,
        body,
        annotations={"num_stages": tir.IntImm("int32", 2)},
    )
    root = tir.Block([], [], [], "root", loop, alloc_buffers=[stage])
    return tir.PrimFunc(
        [a.data, b.data],
        tir.BlockRealize([], True, root),
        buffer_map={a.data: a, b.data: b},
    )


def test_dram_to_asram_resource_no_longer_fatals(tmp_path):
    problem = _run_ilp(_make_dram_asram_pipeline(), tmp_path / "dram_asram_problem.json")
    load_resources = problem["commands"]["0"]["resources"]
    store_resources = problem["commands"]["1"]["resources"]
    assert ODMA1 in load_resources
    assert ASRAM_IN in load_resources
    assert ODMA0 in store_resources
    assert ASRAM_OUT in store_resources
    assert [0, 1] in problem["dep_edges"]


def test_default_profile_equals_explicit_heuristic(tmp_path):
    func = _make_transpose_pipeline()
    default_problem = _run_ilp(func, tmp_path / "default_heuristic.json")
    with tvm.transform.PassContext(config={"tl.sunmmio_cost_model": "heuristic"}):
        explicit_problem = _run_ilp(func, tmp_path / "explicit_heuristic.json")
    assert default_problem == explicit_problem


def test_gem5_ilp_preserves_absolute_cycles_without_explicit_scale(tmp_path):
    with tvm.transform.PassContext(config={"tl.sunmmio_cost_model": "gem5_a4e_2026"}):
        problem = _run_ilp(
            _make_transpose_pipeline(),
            tmp_path / "gem5_absolute.json",
        )
    assert [problem["commands"][str(i)]["latency"] for i in range(3)] == [72, 75, 11]
    assert problem["cost_model"] == {
        "profile": "gem5_a4e_2026",
        "preserve_absolute_cycles": True,
        "solver_scale_factor": 1,
        "solver_scaled": False,
        "latency_gcd_enabled": False,
        "multidim": False,
        "latency_gcd": 1,
        "auto_faster_selected": False,
        "planner_fallback": False,
        "fallback_used": False,
    }
    for command in problem["commands"].values():
        estimate = command["cost_estimate"]
        assert estimate["profile"] == "gem5_a4e_2026"
        assert estimate["latency_cycles"] == command["latency"]
        assert estimate["solver_latency_cycles"] == command["latency"]
        assert estimate["solver_scale_factor"] == 1
        assert estimate["solver_scaled"] is False


def test_gem5_greedy_exports_absolute_cycles_without_scale(tmp_path):
    with tvm.transform.PassContext(config={"tl.sunmmio_cost_model": "gem5_a4e_2026"}):
        graph = _run_greedy(
            _make_transpose_pipeline(),
            tmp_path / "gem5_greedy_absolute.json",
        )
    assert [command["id"] for command in graph["commands"]] == [0, 1, 2]
    assert [command["static_model_cycles"] for command in graph["commands"]] == [
        72,
        75,
        11,
    ]
    assert [command["planned_duration_cycles"] for command in graph["commands"]] == [
        72,
        75,
        11,
    ]
    assert graph["cost_model"] == {
        "profile": "gem5_a4e_2026",
        "preserve_absolute_cycles": True,
        "solver_scale_factor": 1,
        "solver_scaled": False,
        "latency_gcd": 1,
        "auto_faster_selected": False,
        "planner_fallback": False,
        "fallback_used": False,
    }
    for command in graph["commands"]:
        estimate = command["cost_estimate"]
        assert estimate["profile"] == "gem5_a4e_2026"
        assert estimate["latency_cycles"] == command["planned_duration_cycles"]
        assert estimate["solver_latency_cycles"] == command["planned_duration_cycles"]
        assert estimate["solver_scale_factor"] == 1
        assert estimate["solver_scaled"] is False


def test_gem5_ilp_explicit_scale_changes_only_solver_latency(tmp_path):
    with tvm.transform.PassContext(
        config={
            "tl.sunmmio_cost_model": "gem5_a4e_2026",
            "tl.sunmmio_faster": 10,
        }
    ):
        problem = _run_ilp(
            _make_transpose_pipeline(),
            tmp_path / "gem5_scaled.json",
        )
    assert [problem["commands"][str(i)]["latency"] for i in range(3)] == [8, 8, 2]
