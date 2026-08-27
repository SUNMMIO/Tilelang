"""Focused planning and injection coverage for the SunMMIO ILP pipeline."""

import json
import os
from contextlib import contextmanager
from pathlib import Path

import pytest
import tilelang as tl
from tilelang import tvm
from tilelang.engine.phase import should_force_let_inline
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from tvm import tir

from testing.python.transform.sunmmio_mesh_kernel_new_syntax_reference import (
    mesh_ffn_new,
    mesh_flashattn_new,
    mesh_matmul_new,
)

CASES = {
    "gemm": lambda num_stages: mesh_matmul_new(1024, 1024, 1024, 128, 128, 32, num_stages=num_stages),
    "flashattn": lambda num_stages: mesh_flashattn_new(num_stages=num_stages),
    "ffn": lambda num_stages: mesh_ffn_new(num_stages=num_stages),
}


def _lower_and_legalize(mod, target):
    mod = tir.transform.BindTarget(target)(mod)
    mod = tl.transform.ResolveSunmmioMeshSymbols()(mod)
    if should_force_let_inline():
        mod = tl.transform.LetInline()(mod)
    mod = tl.transform.LegalizeNegativeIndex()(mod)
    mod = tl.transform.InjectAssumes()(mod)
    mod = tl.transform.Simplify()(mod)
    mod = tl.transform.InferSramScope()(mod)
    mod = tl.transform.LegalizeSunmmioDataPath()(mod)
    mod = tl.transform.SunmmioLayoutInference()(mod)
    mod = tl.transform.LegalizeSunmmioGemm()(mod)
    mod = tl.transform.LowerTileOp()(mod)
    mod = tl.transform.LegalizeTilesLoop()(mod)
    mod = tl.transform.TilesLoop()(mod)
    mod = tl.transform.LegalizeVectorizedLoop()(mod)
    mod = tl.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tl.transform.LowerAccessPtr()(mod)
    mod = tl.transform.Simplify()(mod)
    mod = tl.transform.HoistNonRestrictParams()(mod)
    return tl.transform.HoistBlockAnnotationsToFuncAttrs()(mod)


@contextmanager
def _scoped_env(updates):
    old = {key: os.environ.get(key) for key in updates}
    os.environ.update({key: str(value) for key, value in updates.items()})
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _pipeline_loops(stmt):
    loops = []

    def visit(node):
        if node is None:
            return
        if isinstance(node, tir.For):
            if node.annotations and "tl.sunmmio.pipeline.requested" in node.annotations:
                loops.append(node)
            visit(node.body)
        elif isinstance(node, tir.BlockRealize):
            visit(node.block.body)
        elif isinstance(node, tir.Block):
            visit(node.body)
        elif isinstance(node, tir.SeqStmt):
            for child in node.seq:
                visit(child)
        elif isinstance(node, tir.IfThenElse):
            visit(node.then_case)
            visit(node.else_case)
        elif isinstance(node, (tir.AttrStmt, tir.LetStmt)):
            visit(node.body)

    visit(stmt)
    return loops


def _lower(case_name, num_stages):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        func = CASES[case_name](num_stages).with_attr("global_symbol", "main")
        mod = tvm.IRModule.from_expr(func)
        mod = _lower_and_legalize(mod, target)
        return tl.transform.IfStmtBinding()(mod)


def _output_dir(tmp_path, case_name, num_stages, shrink):
    configured_root = os.environ.get("SUNMMIO_ILP_PASS_TEST_OUTPUT")
    root = Path(configured_root) if configured_root else tmp_path
    return root / case_name / f"stage{num_stages}_shrink_{'on' if shrink else 'off'}"


def _write_ir(path, mod):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mod.script(show_meta=True).strip() + "\n", encoding="utf-8")


@pytest.mark.parametrize("case_name", CASES, ids=CASES)
@pytest.mark.parametrize("num_stages", (2, 3), ids=lambda value: f"stage{value}")
@pytest.mark.parametrize("shrink", (False, True), ids=("no_shrink", "shrink"))
def test_sunmmio_pipeline_ilp_planning_matrix(tmp_path, case_name, num_stages, shrink):
    """Plan 3 kernels at stages 2/3 with stage shrinking both disabled/enabled."""
    output_dir = _output_dir(tmp_path, case_name, num_stages, shrink)
    output_dir.mkdir(parents=True, exist_ok=True)
    problem_path = output_dir / "ilp_problem.json"
    solution_path = output_dir / "ilp_solution.json"

    mod = _lower(case_name, num_stages)
    _write_ir(output_dir / "00_before_planning.py", mod)
    with (
        tl.transform.PassContext(config={tl.PassConfigKey.TL_SUNMMIO_ILP_STAGE_SHRINK: shrink}),
        _scoped_env(
            {
                "TL_SUNMMIO_FASTER": "200",
                "TL_SUNMMIO_ILP_PROBLEM_JSON": problem_path,
                "TL_SUNMMIO_ILP_SOLUTION_JSON": solution_path,
            }
        ),
    ):
        planned = tl.transform.SunmmioPipelinePlanningILP(debug=False)(mod)
    _write_ir(output_dir / "01_after_planning.py", planned)

    loops = _pipeline_loops(planned["main"].body)
    assert loops
    for loop in loops:
        annotations = loop.annotations
        assert bool(annotations["tl.sunmmio.pipeline.requested"])
        assert bool(annotations["tl.sunmmio.pipeline.applied"])
        assert str(annotations["tl.sunmmio.pipeline.mode"]) == "ilp"
        iterations = int(annotations["iterations"])
        assert 1 <= iterations <= num_stages
        if not shrink:
            assert iterations == num_stages
        assert annotations["body_orders"]
        assert "runtime_bank_flip_modes" in annotations

    problem_paths = sorted(output_dir.glob("ilp_problem*.json"))
    assert problem_paths
    assert solution_path.is_file()
    solution = json.loads(solution_path.read_text(encoding="utf-8"))
    assert int(solution["ii"]) > 0
    assert solution["nodes"]
    assert solution["flows"]


def test_sunmmio_pipeline_ilp_inject_ffn_stage2():
    """Exercise the injector separately on FFN's two collective pipelines."""
    mod = _lower("ffn", 2)
    with tl.transform.PassContext(config={tl.PassConfigKey.TL_SUNMMIO_ILP_STAGE_SHRINK: False}):
        planned = tl.transform.SunmmioPipelinePlanningILP(debug=False)(mod)
        injected = tl.transform.InjectSunmmioPipelineILP()(planned)

    assert len(_pipeline_loops(planned["main"].body)) == 2
    script = injected.script(show_meta=True)
    assert '"tl.sunmmio.pipeline.fallback_reason"' not in script
    assert "_ping" in script
    assert "_pong" in script
    assert script.count("T.mma_sunmmio(") >= 2
    broadcasts = []
    tir.stmt_functor.post_order_visit(
        injected["main"].body,
        lambda node: broadcasts.append(node)
        if isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.broadcast_"
        else None,
    )
    assert len(broadcasts) >= 4


def _make_shifted_multiversion_pipeline():
    k = tir.Var("k", "int32")
    output = tir.decl_buffer((16,), "int32", name="output")
    scratch = tir.decl_buffer((16,), "int32", name="scratch", scope="shared")

    producer = tir.BufferStore(scratch, tir.IntImm("int32", 7), [k + 1])
    consumer = tir.BufferStore(output, tir.BufferLoad(scratch, [k]), [k])
    annotations = {
        "iterations": 3,
        "ii": 1,
        "makespan": 2,
        "steady_state_max_iter_offset": 0,
        "used_buffers": [scratch],
        "versioned_buffers": [scratch],
        "runtime_multiversion_buffers": [scratch],
        "runtime_banked_buffers": [],
        "runtime_resident_banked_buffers": [],
        "runtime_bank_start_phases": {},
        "runtime_bank_read_delta_parities": {},
        "runtime_bank_writer_phases": {},
        "runtime_bank_reader_phases": {},
        "runtime_bank_flip_modes": {},
        "runtime_bank_peer_buffers": {},
        "prologue_orders": ["0-0"],
        "body_orders": ["1-1", "1-0"],
        "epilogue_orders": ["8-1"],
        "tl.sunmmio.pipeline.requested": True,
        "tl.sunmmio.pipeline.applied": True,
        "tl.sunmmio.pipeline.mode": "ilp",
    }
    loop = tir.For(
        k,
        0,
        8,
        tir.ForKind.SERIAL,
        tir.SeqStmt([producer, consumer]),
        annotations=annotations,
    )
    root = tir.Block([], [], [], "root", loop, alloc_buffers=[scratch])
    body = tir.BlockRealize([], tir.const(True, "bool"), root)
    func = tir.PrimFunc([output.data], body, buffer_map={output.data: output}).with_attr("global_symbol", "main")
    return tvm.IRModule.from_expr(func)


def test_sunmmio_pipeline_ilp_inject_shifted_access_version():
    """A producer at k writing value k+1 must match its consumer at k+1."""
    injected = tl.transform.InjectSunmmioPipelineILP()(_make_shifted_multiversion_pipeline())
    script = injected.script()

    # Value 1 is produced in command iteration 0 and consumed in iteration 1.
    # Both accesses must select slot 1 of the three-version buffer.
    assert "scratch[1, T.Add(0, 1)] = 7" in script
    assert "scratch[(k + 1) % 3, k + 1]" in script

    # The next producer writes value k+2 and therefore advances to slot k+2.
    assert "scratch[(k + 1 + 1) % 3, k + 1 + 1] = 7" in script
