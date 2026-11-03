"""Collective-order dependencies use semantic broadcast args, not metadata."""

import json

import pytest
import tilelang as tl
from tilelang import tvm
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from tvm import tir


def _region(buffer, access):
    return tir.Call(
        "handle",
        tvm.ir.Op.get("tl.tileop.region"),
        [tir.BufferLoad(buffer, [0, 0]), access, 32, 32],
    )


def _broadcast(src, dst, source_core, with_unit):
    args = [_region(src, 1), _region(dst, 2), 0, tir.IntImm("int64", 15), 0]
    if source_core is not None:
        args.append(tir.IntImm("int32", source_core))
    if with_unit:
        args.append(tir.Call("handle", tvm.ir.Op.get("tl.odma_unit"), [tir.StringImm("odma1")]))
    return tir.Evaluate(tir.Call("handle", tvm.ir.Op.get("tl.broadcast_"), args))


def _make_pipeline(source_cores, with_unit):
    sources = [tir.decl_buffer((32, 32), "bfloat16", name=f"source_{i}") for i in range(len(source_cores))]
    destinations = [
        tir.decl_buffer((32, 32), "bfloat16", name=f"destination_{i}", scope="shared.asram")
        for i in range(len(source_cores))
    ]
    body = tir.SeqStmt([
        _broadcast(src, dst, source_core, with_unit)
        for src, dst, source_core in zip(sources, destinations, source_cores)
    ])
    loop = tir.For(
        tir.Var("k", "int32"), 0, 4, tir.ForKind.SERIAL, body,
        annotations={"num_stages": tir.IntImm("int32", 2)},
    )
    root = tir.Block([], [], [], "root", loop, alloc_buffers=destinations)
    return tir.PrimFunc(
        [src.data for src in sources],
        tir.BlockRealize([], True, root),
        buffer_map={src.data: src for src in sources},
    ).with_attr("global_symbol", "main")


@pytest.mark.parametrize("with_unit", [False, True], ids=["legacy_args", "odma_metadata"])
@pytest.mark.parametrize(
    "source_cores, expected",
    [
        ((None, None), {(0, 1, 0), (1, 0, 1)}),
        ((0, 0), set()),
        ((None, 0, None), {(0, 2, 0), (2, 0, 1)}),
    ],
    ids=["all_gather", "rooted_broadcast", "mixed"],
)
def test_ilp_collective_order_ignores_odma_metadata(tmp_path, monkeypatch, with_unit, source_cores, expected):
    problem_path = tmp_path / "ilp_problem.json"
    monkeypatch.setenv("TL_SUNMMIO_ILP_PROBLEM_JSON", str(problem_path))
    monkeypatch.setenv("TL_SUNMMIO_FASTER", "100")
    monkeypatch.setenv("TL_SUNMMIO_ILP_ENABLE_LATENCY_GCD", "1")
    monkeypatch.setenv("TL_SUNMMIO_ILP_FAST_VERIFIED_SCHEDULER", "1")
    monkeypatch.setenv("TL_SUNMMIO_ILP_EXPORT_ONLY", "1")
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = tvm.IRModule.from_expr(_make_pipeline(source_cores, with_unit))
    mod = tir.transform.BindTarget(target)(mod)
    # Deliberately do not resolve units: test both accepted legacy intrinsic
    # layouts and resolved metadata without changing the classifier input.
    with target:
        tl.transform.SunmmioPipelinePlanningILP(debug=False)(mod)

    problem = json.loads(problem_path.read_text(encoding="utf-8"))
    assert problem["N"] == len(source_cores)
    edges = [edge for edge in problem["dependency_constraints"] if edge["hazard"] == "collective_order"]
    assert {(edge["src"], edge["dst"], edge["delta"]) for edge in edges} == expected
    for edge in edges:
        assert edge["lag"] == problem["commands"][str(edge["src"])]["latency"]
