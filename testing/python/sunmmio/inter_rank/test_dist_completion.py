"""Completion snapshots, pending state, wait_any/wait_all, and resource reuse."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.language.utils import region
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import (
    collect_calls,
    collect_nodes,
    collect_op_names,
)


@tilelang.jit(target="sunmmio")
def manual_completion_kernel_factory(signal_kind, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            expected_deltas = T.alloc_shared((2,), T.uint32)
            T.fill(expected_deltas, 1)
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
            T.fill(expected_deltas, 1)
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
            T.fill(expected_deltas, 1)
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
    state_dtype = "uint32" if signal_kind.value.endswith("memory") else "int32"
    # Ordinary software arrays use i32 independently of the signal width.
    for state_region in initializer.args[:2]:
        assert state_region.args[0].buffer.dtype == state_dtype
    assert initializer.args[2].args[0].buffer.dtype == "int32"
    assert initializer.args[3].buffer.dtype == "int32"
    assert collect_op_names(lowered).count("tl.dist_wait_any_") == 1
    injected = result.pass_snapshot("tl.LowerDistSync").mod
    assert "tl.dist_completion_init_" not in collect_op_names(injected)
    assert collect_op_names(result.device_mod).count("tl.dist_wait_any_") == 1


@pytest.mark.parametrize("constant_deltas", [False, True], ids=["runtime-delta", "constant-delta"])
@pytest.mark.parametrize(
    "state_dtype, initial, deltas, expected",
    [
        pytest.param("uint8", [250, 17], [5, 0], [255, 17], id="byte-safe-cumulative-boundary"),
        pytest.param("uint8", [254, 17], [1, 0], [255, 17], id="byte-safe-increment"),
        pytest.param("uint32", [2**31 - 1, 2**31 + 17], [2, 0], [2**31 + 1, 2**31 + 17], id="memory-unsigned-high-bit"),
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
    # This isolated expansion consumes inputs already admitted by the
    # communication pass; overflow rejection is tested through that pass.
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
        return None

    for store in statements:
        value = tvm.tir.stmt_functor.ir_transform(tvm.tir.Evaluate(store.value), None, evaluate_constants, ["tir.BufferLoad"]).value
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
    assert collect_op_names(lowered).count("tl.dist_completion_init_") == 1
    initializer = collect_calls(lowered, "tl.dist_completion_init_")[0]
    assert not bool(initializer.args[4])
    deltas = initializer.args[5:]
    assert [int(delta.indices[0]) for delta in deltas] == [0, 1]
    assert deltas[0].buffer.data.same_as(deltas[1].buffer.data)
    updates = collect_nodes(
        lowered,
        tvm.tir.BufferStore,
        lambda store: store.buffer.data.same_as(deltas[0].buffer.data) and isinstance(store.value, tvm.tir.Add),
    )
    assert len(updates) == 1
    assert int(updates[0].indices[0]) == 0

    injected = result.pass_snapshot("tl.LowerDistSync").mod
    assert "tl.dist_completion_init_" not in collect_op_names(injected)
    pending = collect_calls(injected, "tl.dist_wait_any_")[0].args[4].args[0].buffer
    assert pending.dtype == "int32"


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
        func for func in injected_mod.functions.values() if isinstance(func, tvm.tir.PrimFunc) and collect_calls(func, "tl.dist_wait_any_")
    )

    # SplitHostDevice retains a symbolic mesh extent in the thread binding.
    # Normalize that unrelated metadata to the 4x4 test mesh before checking scope.
    def normalize_core(node):
        if isinstance(node, tvm.tir.AttrStmt) and node.attr_key == "thread_extent":
            axis = node.node
            bound = tvm.tir.IterVar(tvm.ir.Range(0, node.value), axis.var, axis.iter_type, axis.thread_tag)
            return tvm.tir.AttrStmt(bound, node.attr_key, node.value, node.body)

    body = tvm.tir.stmt_functor.ir_transform(injected.body, None, normalize_core, ["tir.AttrStmt"])
    normalized = injected.with_body(body)
    assert tvm.tir.analysis.verify_well_formed(normalized)
    wait_any = collect_calls(injected, "tl.dist_wait_any_")[0]
    pending_count = wait_any.args[5]
    waits = collect_nodes(injected, tvm.tir.While)
    assert waits
    analyzer = tvm.arith.Analyzer()
    assert any(analyzer.can_prove_equal(loop.condition, pending_count > 0) for loop in waits)


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


@tilelang.jit(target="sunmmio")
def completion_range_kernel_factory(auto=True, rounds=255, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            deltas = T.alloc_shared((1,), T.uint32)
            signals = T.dist.signals(1, expect="auto" if auto else "manual")
            T.fill(deltas, 1)
            for _ in T.serial(rounds):
                if auto:
                    T.dist.put(src, dst, (rank_id + 1) % world_size, signal=signals[0], submit=True)
                else:
                    done = T.dist.completion(signals, expected_deltas=deltas)
                    T.dist.wait_all(done)
            if auto:
                done = T.dist.completion(signals)
                T.dist.wait_all(done)

    return main


@pytest.mark.parametrize("auto", [False, True])
@pytest.mark.parametrize("rounds", [255, 256])
def test_completion_expected_accumulates_once_and_drain_does_not_reset(auto, rounds):
    func = completion_range_kernel_factory.get_tir(auto, rounds, world_size=2)
    if rounds == 255:
        lower_to_device_tir(func)
    else:
        with pytest.raises(tvm.error.InternalError, match="cumulative expected"):
            lower_to_device_tir(func)
