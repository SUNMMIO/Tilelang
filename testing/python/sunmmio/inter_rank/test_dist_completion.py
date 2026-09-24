"""Completion snapshots, pending state, wait_any/wait_all, and resource reuse."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import (
    collect_op_names,
)


@tilelang.jit(target="sunmmio")
def manual_completion_kernel_factory(signal_kind, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            expected_deltas = T.alloc_shared((2,), T.uint32)
            signals = T.dist.signals(
                2,
                kind=signal_kind,
                expect="manual",
            )
            completion = T.dist.completion(
                signals,
                expected_deltas=expected_deltas,
            )
            while T.dist.has_pending(completion):
                member = T.dist.wait_any(completion)
                expected_deltas[member] = 0

    return main


@tilelang.jit(target="sunmmio")
def auto_completion_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signals = T.dist.signals(2)
            T.dist.put(
                src,
                dst,
                dst_rank=(rank_id + 1) % world_size,
                signal=signals[0],
                submit=True,
            )
            completion = T.dist.completion(signals)
            T.dist.wait_all(completion)

    return main


@tilelang.jit(target="sunmmio")
def auto_completion_after_loop_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signals = T.dist.signals(2)
            for iteration in T.serial(2):
                if iteration == 0:
                    T.dist.put(
                        src,
                        dst,
                        (rank_id + 1) % world_size,
                        signal=signals[0],
                    )
                    T.dist.submit()
            completion = T.dist.completion(signals)
            T.dist.wait_all(completion)

    return main


@tilelang.jit(target="sunmmio")
def repeated_manual_completion_kernel_factory(drain_first=False, observe_first=False, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            expected_deltas = T.alloc_shared((2,), T.uint32)
            signals = T.dist.signals(2, kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE, expect="manual")
            first = T.dist.completion(signals, expected_deltas=expected_deltas)
            if observe_first:  # noqa: SIM102 - Keep static selection separate from the device condition.
                if T.dist.has_pending(first):
                    T.dist.wait_any(first)
            if drain_first:
                T.dist.wait_all(first)
            second = T.dist.completion(signals, expected_deltas=expected_deltas)
            T.dist.wait_all(second)

    return main


@tilelang.jit(target="sunmmio")
def loop_manual_completion_kernel_factory(drain_each=False, observe_each=False, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            expected_deltas = T.alloc_shared((2,), T.uint32)
            signals = T.dist.signals(2, kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE, expect="manual")
            for _ in T.serial(2):
                completion = T.dist.completion(signals, expected_deltas=expected_deltas)
                if observe_each:  # noqa: SIM102 - Keep static selection separate from the device condition.
                    if T.dist.has_pending(completion):
                        T.dist.wait_any(completion)
                if drain_each:
                    T.dist.wait_all(completion)

    return main


@pytest.mark.parametrize("signal_kind", list(T.dist.SignalKind))
def test_manual_completion_wait_any_supports_every_signal_kind(signal_kind):
    func = manual_completion_kernel_factory.get_tir(
        signal_kind=signal_kind,
        world_size=4,
    )
    result = lower_to_device_tir(
        func,
        capture_passes=(
            "tl.PlanDistSignals",
            "tl.LowerDistCommunication",
            "tl.InjectDistSync",
        ),
    )

    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert int(planned.attrs["tl.dist.signal_counts"][signal_kind.value]) == 2
    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod
    assert collect_op_names(lowered).count("tl.dist_completion_init_") == 1
    assert collect_op_names(lowered).count("tl.dist_wait_any_") == 1
    injected = result.pass_snapshot("tl.InjectDistSync").mod
    assert "tl.dist_completion_init_" not in collect_op_names(injected)
    assert collect_op_names(result.device_mod).count("tl.dist_wait_any_") == 1


def test_auto_completion_captures_only_current_put_members():
    func = auto_completion_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(
        func,
        capture_passes=("tl.LowerDistCommunication", "tl.InjectDistSync"),
    )

    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod
    lowered_script = lowered.script()
    assert collect_op_names(lowered).count("tl.dist_completion_init_") == 1
    assert "T.bool(False), v_completion_deltas_1[0], v_completion_deltas_1[1]" in lowered_script
    assert 'v_completion_deltas_1[0] = v_completion_deltas_1[0] + T.Cast("int32", 1)' in lowered_script

    injected = result.pass_snapshot("tl.InjectDistSync").mod
    injected_script = injected.script()
    assert "tl.dist_completion_init_" not in collect_op_names(injected)
    assert 'v_pending_1[0] = T.Cast("uint8", v_completion_deltas_1[0] > 0)' in injected_script
    assert 'v_pending_1[1] = T.Cast("uint8", v_completion_deltas_1[1] > 0)' in injected_script


def test_auto_completion_after_loop_uses_runtime_delta_without_free_loop_var():
    result = lower_to_device_tir(
        auto_completion_after_loop_kernel_factory.get_tir(world_size=4),
        capture_passes=("tl.LowerDistCommunication", "tl.InjectDistSync"),
    )
    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod.script()
    assert "completion_deltas" in lowered
    assert "iteration == 0" in lowered
    assert "T.dist_completion_init_" in lowered
    injected_mod = result.pass_snapshot("tl.InjectDistSync").mod
    injected = next(
        func for func in injected_mod.functions.values() if isinstance(func, tvm.tir.PrimFunc) and "completion_deltas" in func.script()
    )
    # SplitHostDevice retains a symbolic mesh extent in the thread binding.
    # Normalize that unrelated metadata to the 4x4 test mesh before checking scope.
    core = injected.body.seq[0]
    core_iter = tvm.tir.IterVar(tvm.ir.Range(0, 16), core.node.var, core.node.iter_type, core.node.thread_tag)
    body = tvm.tir.SeqStmt([tvm.tir.AttrStmt(core_iter, core.attr_key, core.value, core.body), *injected.body.seq[1:]])
    normalized = tvm.tir.PrimFunc(injected.params, body, injected.ret_type, injected.buffer_map)
    assert tvm.tir.analysis.verify_well_formed(normalized)
    pending_checks = [line for line in injected.script().splitlines() if "pending" in line and "> 0" in line]
    assert pending_checks and all("iteration" not in line for line in pending_checks)


def test_manual_completion_only_requires_drain_when_used_for_progress():
    with pytest.raises(tvm.error.InternalError, match="previous completion must be drained"):
        lower_to_device_tir(repeated_manual_completion_kernel_factory.get_tir(observe_first=True, world_size=4))
    aggregate = lower_to_device_tir(repeated_manual_completion_kernel_factory.get_tir(world_size=4))
    assert collect_op_names(aggregate.device_mod).count("tl.dist_wait_any_") == 1
    result = lower_to_device_tir(repeated_manual_completion_kernel_factory.get_tir(drain_first=True, world_size=4))
    assert collect_op_names(result.device_mod).count("tl.dist_wait_any_") == 2


def test_loop_manual_completion_only_requires_drain_for_progress():
    with pytest.raises(tvm.error.InternalError, match="drained in that iteration"):
        lower_to_device_tir(loop_manual_completion_kernel_factory.get_tir(observe_each=True, world_size=4))
    lower_to_device_tir(loop_manual_completion_kernel_factory.get_tir(world_size=4))
    result = lower_to_device_tir(loop_manual_completion_kernel_factory.get_tir(drain_each=True, world_size=4))
    assert collect_op_names(result.device_mod).count("tl.dist_wait_any_") == 1
