"""Automatic pipelines reject dist communication, including calls without signal declarations."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.utils.target import determine_target
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir


@tilelang.jit(target="sunmmio")
def pipeline_dist_kernel_factory(op="put", stages=2, nested="none", world_size: int = 1):
    @T.macro
    def operation(src, dst, gathered, exchange_dst, counts, deltas, signal, signals, completed, rank_id):
        if op == "put":
            T.dist.put(src, dst, (rank_id + 1) % world_size, signal=signal, submit=True)
        if op == "routed_put":
            T.dist.routed_put(src, dst, routes=[[0, (rank_id + 1) % world_size, 1]], signal=signal, submit=True)
        if op == "put_signal":
            T.dist.put_signal(signal, (rank_id + 1) % world_size, submit=True)
        if op == "barrier":
            T.dist.barrier(signal=signal)
        if op == "wait_signal":
            T.dist.wait_signal(signal)
        if op == "wait_all":
            T.dist.wait_all(signals)
        if op == "wait_delta":
            T.dist.wait_signal(signals[0], expected_delta=1)
        if op == "wait_signals":
            T.dist.wait_signals(signals, expected_deltas=deltas)
        if op == "completion":
            local_completion = T.dist.completion(signals, expected_deltas=deltas)
            T.dist.wait_all(local_completion)
        if op == "wait_any":
            index = T.dist.wait_any(completed)
            dst[index] = 0
        if op == "has_pending":
            while T.dist.has_pending(completed):
                T.dist.wait_any(completed)
        if op == "wait_completion":
            T.dist.wait_all(completed)
        if op == "all_gather":
            ready = T.dist.all_gather(src, gathered, signals=signals, submit=True)
            T.dist.wait_all(ready)
        if op == "all_to_allv":
            ready = T.dist.all_to_allv(gathered, exchange_dst, send_counts=counts, recv_counts=counts, signals=signals, submit=True)
            T.dist.wait_all(ready)

    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            gathered = T.alloc_shared((world_size, 32), T.bfloat16)
            exchange_dst = T.alloc_shared((world_size, 32), T.bfloat16)
            counts = T.alloc_shared((world_size,), T.int32)
            deltas = T.alloc_shared((world_size,), T.uint32)
            signal = T.dist.signal()
            signals = T.dist.signals(
                world_size,
                expect="manual" if op in ("wait_delta", "wait_signals", "completion") else None,
            )
            completion_signals = T.dist.signals(world_size, expect="manual")
            completed = T.dist.completion(completion_signals, expected_deltas=deltas)
            step = T.alloc_var(T.int32, init=0)
            for _ in T.Pipelined(4, num_stages=stages):
                if nested == "serial":
                    for _ in T.serial(2):
                        operation(src, dst, gathered, exchange_dst, counts, deltas, signal, signals, completed, rank_id)
                elif nested == "disabled_pipeline":
                    for _ in T.Pipelined(2, num_stages=0):
                        operation(src, dst, gathered, exchange_dst, counts, deltas, signal, signals, completed, rank_id)
                elif nested == "while":
                    while step < 2:
                        operation(src, dst, gathered, exchange_dst, counts, deltas, signal, signals, completed, rank_id)
                        step += 1
                else:
                    operation(src, dst, gathered, exchange_dst, counts, deltas, signal, signals, completed, rank_id)

    return main


@tilelang.jit(target="sunmmio")
def pipeline_queue_only_kernel_factory(op="wait", stages=2, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            for _ in T.Pipelined(4, num_stages=stages):
                if op == "wait":
                    T.dist.wait()
                else:
                    T.dist.submit()

    return main


@tilelang.jit(target="sunmmio")
def local_pipeline_after_dist_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(
        output: T.MeshTensor((128,), T.placement.replicated(), T.bfloat16),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            stage = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            T.dist.put(src, dst, (rank_id + 1) % world_size, signal=signal, submit=True)
            T.dist.wait_signal(signal)
            for chunk in T.Pipelined(4, num_stages=2):
                T.copy(dst, stage)
                T.copy(stage, output[chunk * 32 : (chunk + 1) * 32])
            T.dist.wait()

    return main


def _plan_signals(func):
    target = tvm.target.Target(determine_target("sunmmio", return_object=True))
    mod = tvm.IRModule({"main": func.with_attr("target", target)})
    mod = tilelang.transform.ResolveSunmmioMeshSymbols()(mod)
    mod = tilelang.transform.LowerDistCollectives()(mod)
    mod = tilelang.transform.InferSramScope()(mod)
    return tilelang.transform.PlanDistSignals()(mod)


@pytest.mark.parametrize(
    "op",
    [
        "put",
        "routed_put",
        "put_signal",
        "barrier",
        "wait_signal",
        "wait_all",
        "wait_delta",
        "wait_signals",
        "completion",
        "wait_any",
        "has_pending",
        "wait_completion",
        "all_gather",
        "all_to_allv",
    ],
)
def test_pipeline_rejects_dist_calls_and_completion_expressions(op):
    func = pipeline_dist_kernel_factory.get_tir(op=op, world_size=4)
    with pytest.raises(tvm.error.InternalError, match="move communication, submit and waits outside"):
        _plan_signals(func)


@pytest.mark.parametrize("nested", ["serial", "disabled_pipeline", "while"])
def test_nested_loops_cannot_escape_pipeline_guard(nested):
    func = pipeline_dist_kernel_factory.get_tir(nested=nested, world_size=4)
    with pytest.raises(tvm.error.InternalError, match="T.Pipelined"):
        _plan_signals(func)


@pytest.mark.parametrize("op", ["wait", "submit"])
@pytest.mark.parametrize("stages", [1, 2])
def test_pipeline_queue_ops_are_rejected_without_signal_declarations(op, stages):
    func = pipeline_queue_only_kernel_factory.get_tir(op=op, stages=stages, world_size=4)
    assert "dist_signal_decl" not in func.script()
    with pytest.raises(tvm.error.InternalError, match="T.Pipelined"):
        lower_to_device_tir(func)


def test_num_stages_zero_preserves_existing_dist_loop_support():
    result = lower_to_device_tir(pipeline_dist_kernel_factory.get_tir(stages=0, world_size=4))
    assert "T.dist_put_(" in result.device_mod.script()


def test_local_pipeline_allows_communication_and_waits_outside():
    result = lower_to_device_tir(local_pipeline_after_dist_kernel_factory.get_tir(world_size=4))
    assert "T.dist_put_(" in result.device_mod.script()
    assert "T.dist_wait_signal_(" in result.device_mod.script()
