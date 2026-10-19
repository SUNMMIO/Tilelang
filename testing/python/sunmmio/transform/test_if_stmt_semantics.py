"""Conditional evaluation through the production SunMMIO optimization phase."""

import numpy as np
import pytest

from tilelang import tvm
import tilelang.language as T
from tilelang.engine.phase import OptimizeForSunmmio
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from tvm import tir

from testing.python.sunmmio.common.codegen_validation import codegen_sunmmio_suvm_mlir, validate_suvm_mlir_with_npuir_opt
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.transform.sunmmio_mesh_kernel_new_syntax_reference import mesh_matmul_new


@tvm.instrument.pass_instrument
class _CapturePasses:
    def __init__(self):
        self.names = []
        self.modules = {}

    def run_after_pass(self, mod, info):
        self.names.append(info.name)
        if "Pipeline" in info.name or info.name == "tl.IfStmtBinding":
            self.modules[info.name] = mod


def _nodes(stmt, node_type):
    nodes = []
    tir.stmt_functor.post_order_visit(stmt, lambda node: nodes.append(node) if isinstance(node, node_type) else None)
    return nodes


def _conditional_loop(independent, indexed_condition):
    buffer = tir.decl_buffer((5,), "int32", name="A")
    step = tir.Var("step", "int32")
    index = step if indexed_condition else tir.IntImm("int32", 0)
    condition = buffer[index] > 0
    reset = tir.BufferStore(buffer, 0, [index])
    increment = tir.BufferStore(buffer, buffer[4] + 1, [4])
    if independent:
        body = tir.SeqStmt([tir.IfThenElse(condition, reset, None), tir.IfThenElse(condition, increment, None)])
    else:
        body = tir.IfThenElse(condition, tir.SeqStmt([reset, increment]), None)
    loop = tir.For(step, 0, 4, tir.ForKind.SERIAL, body, annotations={"num_stages": 2})
    core = tvm.te.thread_axis("blockIdx.x")
    root = tir.Block([], [], [], "root", tir.AttrStmt(core, "thread_extent", 16, loop))
    func = tir.PrimFunc([buffer.data], tir.BlockRealize([], tir.const(True, "bool"), root), buffer_map={buffer.data: buffer})
    return func.with_attr("global_symbol", "main").with_attr("tir.noalias", True)


def _execute_single_core(func, initial):
    # Keep the optimized loop unchanged; remove the launch wrapper for CPU execution.
    def strip_thread(stmt):
        if isinstance(stmt, tir.AttrStmt) and stmt.attr_key == "thread_extent":
            return stmt.body
        return None

    body = tir.stmt_functor.ir_transform(func.body, None, strip_thread, ["tir.AttrStmt"])
    func = func.with_body(body).without_attr("target")
    executable = tvm.compile(func, target="c").jit(options=["-std=c++17"])
    values = tvm.runtime.tensor(np.array(initial, dtype="int32"))
    executable(values)
    return values.numpy()


@pytest.mark.parametrize("mode", ["greedy", "ilp", "disabled"])
@pytest.mark.parametrize("independent", [False, True], ids=["single_if", "independent_ifs"])
@pytest.mark.parametrize("indexed_condition", [False, True], ids=["fixed_index", "loop_index"])
def test_sunmmio_preserves_conditional_evaluation(mode, independent, indexed_condition):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    func = _conditional_loop(independent, indexed_condition).with_attr("target", target)
    capture = _CapturePasses()
    config = {
        "tl.sunmmio_pipeline_mode": "greedy" if mode == "disabled" else mode,
        "tl.disable_sunmmio_pipeline": mode == "disabled",
    }
    with tvm.transform.PassContext(config=config, instruments=[capture]), target:
        optimized = OptimizeForSunmmio(tvm.IRModule({"main": func}), target)

    pipeline_passes = [name for name in capture.names if "Pipeline" in name]
    if mode == "disabled":
        assert pipeline_passes == []
    else:
        suffix = "ILP" if mode == "ilp" else ""
        assert pipeline_passes == [f"tl.SunmmioPipelinePlanning{suffix}", f"tl.InjectSunmmioPipeline{suffix}"]
        injected = capture.modules[pipeline_passes[-1]]["main"]
        loops = _nodes(injected.body, tir.For)
        assert len(loops) == 1
        assert not bool(loops[0].annotations["tl.sunmmio.pipeline.applied"])
        assert str(loops[0].annotations["tl.sunmmio.pipeline.fallback_detail"]) == "unsupported_statement"

    final = optimized["main"]
    loops = _nodes(final.body, tir.For)
    assert len(loops) == 1
    snapshots = _nodes(loops[0].body, tir.LetStmt)
    assert len(snapshots) == (0 if independent else 1)
    if snapshots:
        assert _nodes(snapshots[0].value, tir.BufferLoad)

    count = 0 if independent else (2 if indexed_condition else 1)
    expected = [0, 0, 0, 0, count] if indexed_condition else [0, 0, 1, 0, count]
    np.testing.assert_array_equal(_execute_single_core(final, [1, 0, 1, 0, 0]), expected)


@pytest.mark.parametrize("mode", ["greedy", "ilp"])
def test_sunmmio_normal_pipeline_remains_applied(mode):
    suffix = "ILP" if mode == "ilp" else ""
    planner = f"tl.SunmmioPipelinePlanning{suffix}"
    injector = f"tl.InjectSunmmioPipeline{suffix}"
    kernel = mesh_matmul_new(128, 128, 128, block_M=32, block_N=32, block_K=32, num_stages=2)
    result = lower_to_device_tir(kernel, capture_passes=(planner, injector), pass_configs={"tl.sunmmio_pipeline_mode": mode})
    planned = result.pass_snapshot(planner).mod["main"]
    loops = [loop for loop in _nodes(planned.body, tir.For) if "tl.sunmmio.pipeline.applied" in loop.annotations]
    assert len(loops) == 1
    assert bool(loops[0].annotations["tl.sunmmio.pipeline.applied"])
    injected = result.pass_snapshot(injector).mod["main"]
    assert all("tl.sunmmio.pipeline.fallback_reason" not in loop.annotations for loop in _nodes(injected.body, tir.For))


def test_sunmmio_condition_snapshot_codegen_and_device_validation(tmp_path):
    @T.prim_func
    def main(A: T.Tensor((32, 32), T.bfloat16), B: T.Tensor((32, 32), T.bfloat16), C: T.Tensor((32, 32), T.bfloat16)):
        with T.Kernel():
            state = T.alloc_shared((32, 32), T.bfloat16, scope="shared.rsram")
            T.copy(A, state)
            if state[0, 0] > T.bfloat16(0):
                T.copy(B, state)
                T.copy(state, C)

    result = lower_to_device_tir(main)
    device_func = next(iter(result.device_mod.functions.values()))
    snapshots = [node for node in _nodes(device_func.body, tir.LetStmt) if str(node.var.dtype) == "bool"]
    assert len(snapshots) == 1
    src = codegen_sunmmio_suvm_mlir(result.device_mod)
    assert src.count("suvm.tile.pick") == 1
    assert src.count("scf.if") == 1
    validate_suvm_mlir_with_npuir_opt(src, tmp_path)
