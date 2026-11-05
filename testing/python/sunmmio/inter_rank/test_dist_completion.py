"""Completion snapshots, pending state, wait_any/wait_all, and resource reuse."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.language.utils import region
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import (
    collect_calls,
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
            "tl.LowerDistSync",
        ),
    )

    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert int(planned.attrs["tl.dist.signal_counts"][signal_kind.value]) == 2
    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod
    initializers = collect_calls(lowered, "tl.dist_completion_init_")
    assert len(initializers) == 1
    initializer = initializers[0]
    state_dtype = "uint64" if signal_kind.value.endswith("memory") else "uint8"
    # The group target and this completion's snapshot follow the physical
    # counter width; pending bookkeeping is independent of that width.
    for state_region in initializer.args[:2]:
        assert state_region.args[0].buffer.dtype == state_dtype
    assert initializer.args[2].args[0].buffer.dtype == "uint8"
    assert initializer.args[3].buffer.dtype == "int32"
    assert collect_op_names(lowered).count("tl.dist_wait_any_") == 1
    injected = result.pass_snapshot("tl.LowerDistSync").mod
    assert "tl.dist_completion_init_" not in collect_op_names(injected)
    assert collect_op_names(result.device_mod).count("tl.dist_wait_any_") == 1


@pytest.mark.parametrize("constant_deltas", [False, True], ids=["runtime-delta", "constant-delta"])
@pytest.mark.parametrize(
    "state_dtype, initial, deltas, expected",
    [
        pytest.param("uint8", [250, 17], [256, 0], [250, 17], id="byte-full-wrap-stays-pending"),
        pytest.param("uint8", [255, 17], [1, 0], [0, 17], id="byte-boundary-wrap"),
        pytest.param("uint64", [2**32 - 1, 2**32 + 17], [2, 0], [2**32 + 1, 2**32 + 17], id="memory-crosses-u32"),
    ],
)
def test_completion_initialization_preserves_counter_width_and_pending(state_dtype, initial, deltas, expected, constant_deltas):
    """Evaluate the real expanded stores with boundary inputs, without a NIC."""
    group_expect = tvm.tir.decl_buffer((2,), state_dtype, name="group_expect", scope="local")
    snapshot = tvm.tir.decl_buffer((2,), state_dtype, name="snapshot", scope="local")
    pending = tvm.tir.decl_buffer((2,), "uint8", name="pending", scope="local")
    pending_count = tvm.tir.decl_buffer((1,), "int32", name="pending_count", scope="local.var")
    delta_buffer = tvm.tir.decl_buffer((2,), "uint32", name="deltas", scope="local")
    delta_values = [tvm.tir.const(value, "uint32") if constant_deltas else delta_buffer[i] for i, value in enumerate(deltas)]
    initializer = tvm.tir.call_intrin(
        "handle",
        tvm.ir.Op.get("tl.dist_completion_init_"),
        region(group_expect[0], "rw", 2),
        region(snapshot[0], "w", 2),
        region(pending[0], "w", 2),
        pending_count[0],
        tvm.tir.const(True, "bool"),
        *delta_values,
    )
    buffers = (group_expect, snapshot, pending, pending_count, delta_buffer)
    func = tvm.tir.PrimFunc(
        [buffer.data for buffer in buffers],
        tvm.tir.Evaluate(initializer),
        buffer_map={buffer.data: buffer for buffer in buffers},
    ).with_attr("tl.dist.world_size", 4)
    lowered = tilelang.transform.LowerDistSync()(tvm.IRModule({"main": func}))
    assert "tl.dist_completion_init_" not in collect_op_names(lowered)
    # Exercise the production simplifier before substituting runtime inputs:
    # a literal delta of 256 must not leave an invalid uint8(256) cast.
    if constant_deltas:
        lowered = tilelang.transform.Simplify()(lowered)
    statements = lowered["main"].body.seq
    assert all(isinstance(stmt, tvm.tir.BufferStore) for stmt in statements)

    memory = {
        group_expect.data: list(initial),
        snapshot.data: [0, 0],
        pending.data: [0, 0],
        pending_count.data: [0],
        delta_buffer.data: list(deltas),
    }
    analyzer = tvm.arith.Analyzer()

    def evaluate_constants(node):
        if isinstance(node, tvm.tir.BufferLoad):
            index = int(analyzer.simplify(node.indices[0]))
            return tvm.tir.const(memory[node.buffer.data][index], node.dtype)
        if isinstance(node, tvm.tir.Call) and node.op.name == "tir.bitwise_and":
            # Analyzer does not fold unsigned bitwise operations. Evaluate the
            # mask emitted by the pass; casts themselves must simplify normally.
            lhs, rhs = [analyzer.simplify(arg) for arg in node.args]
            if isinstance(lhs, tvm.tir.IntImm) and isinstance(rhs, tvm.tir.IntImm):
                return tvm.tir.const(int(lhs) & int(rhs), node.dtype)
        return None

    for store in statements:
        value = tvm.tir.stmt_functor.ir_transform(
            tvm.tir.Evaluate(store.value), None, evaluate_constants, ["tir.BufferLoad", "tir.Call"]
        ).value
        value = analyzer.simplify(value)
        assert isinstance(value, tvm.tir.IntImm)
        memory[store.buffer.data][int(store.indices[0])] = int(value)

    assert memory[group_expect.data] == expected
    assert memory[snapshot.data] == expected
    assert memory[pending.data] == [1, 0]
    assert memory[pending_count.data] == [1]


def test_auto_completion_captures_only_current_put_members():
    func = auto_completion_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(
        func,
        capture_passes=("tl.LowerDistCommunication", "tl.LowerDistSync"),
    )

    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod
    lowered_script = lowered.script()
    assert collect_op_names(lowered).count("tl.dist_completion_init_") == 1
    assert "T.bool(False), v_completion_deltas_1[0], v_completion_deltas_1[1]" in lowered_script
    assert 'v_completion_deltas_1[0] = v_completion_deltas_1[0] + T.Cast("int32", 1)' in lowered_script

    injected = result.pass_snapshot("tl.LowerDistSync").mod
    injected_script = injected.script()
    assert "tl.dist_completion_init_" not in collect_op_names(injected)
    assert 'v_pending_1[0] = T.Cast("uint8", v_completion_deltas_1[0] > 0)' in injected_script
    assert 'v_pending_1[1] = T.Cast("uint8", v_completion_deltas_1[1] > 0)' in injected_script


def test_auto_completion_after_loop_uses_runtime_delta_without_free_loop_var():
    result = lower_to_device_tir(
        auto_completion_after_loop_kernel_factory.get_tir(world_size=4),
        capture_passes=("tl.LowerDistCommunication", "tl.LowerDistSync"),
    )
    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod.script()
    assert "completion_deltas" in lowered
    assert "iteration == 0" in lowered
    assert "T.dist_completion_init_" in lowered
    injected_mod = result.pass_snapshot("tl.LowerDistSync").mod
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
