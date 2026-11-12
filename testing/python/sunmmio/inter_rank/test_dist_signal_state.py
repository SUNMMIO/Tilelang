"""Signal expected/generation updates, repeated waits, and reuse across loop iterations."""

from collections import Counter

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import (
    base_mod,
    collect_calls,
    collect_nodes,
    collect_op_names,
    signal_counts,
    single_prim_func,
    strip_casts,
)


@tilelang.jit(target="sunmmio")
def explicit_memory_signal_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY)
            peer_rank = (rank_id + 1) % world_size
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal)
            T.dist.submit()
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def reused_signal_kernel_factory(
    signal_kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE,
    expect_mode="auto",
    world_size: int = 1,
):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32, 32), T.bfloat16)
            dst = T.alloc_shared((32, 32), T.bfloat16)
            signal = T.dist.signal(kind=signal_kind, expect=expect_mode)
            peer_rank = (rank_id + 1) % world_size
            for _step in T.serial(2):
                T.dist.put(src, dst, dst_rank=peer_rank, signal=signal)
                T.dist.submit()
                if expect_mode == "manual":
                    T.dist.wait_signal(signal, expected_delta=1)
                else:
                    T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def signal_only_dtype_kernel_factory(signal_kind, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            signal = T.dist.signal(kind=signal_kind)
            T.dist.put_signal(signal, (rank_id + 1) % world_size, submit=True)
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def manual_delta_dtype_kernel_factory(signal_kind, fixed_delta=None, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, delta: T.uint32):
        with T.Kernel():
            signal = T.dist.signal(kind=signal_kind, expect="manual")
            T.dist.wait_signal(signal, expected_delta=delta if fixed_delta is None else fixed_delta)

    return main


@tilelang.jit(target="sunmmio")
def two_puts_two_waits_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32, 32), T.bfloat16)
            dst = T.alloc_shared((32, 32), T.bfloat16)
            signal = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE)
            peer_rank = (rank_id + 1) % world_size
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal)
            T.dist.submit()
            T.dist.wait_signal(signal)
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def multi_signal_kernel_factory(M, N, world_size: int = 1):
    placement = T.placement.full_shard(0, 1)

    @T.prim_func
    def main(
        A: T.MeshTensor((M, N), placement, T.bfloat16),  # type: ignore
        B: T.MeshTensor((M, N), placement, T.bfloat16),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            local_M, local_N = A.local_shape
            src = T.alloc_shared((local_M, local_N), T.bfloat16)
            sram_dst_0 = T.alloc_shared((local_M, local_N), T.bfloat16)
            sram_dst_1 = T.alloc_shared((local_M, local_N), T.bfloat16)
            s0 = T.dist.signal()
            s1 = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE)
            d0 = T.dist.signal()
            m0 = T.dist.signal(kind=T.dist.SignalKind.DRAM_MEMORY)

            T.copy(A, src)
            peer_rank = (rank_id + 1) % world_size
            T.dist.put(src, sram_dst_0, dst_rank=peer_rank, signal=s0)
            T.dist.put(src, sram_dst_1, dst_rank=peer_rank, signal=s1)
            T.dist.put(src, sram_dst_0, dst_rank=peer_rank, signal=s0)
            T.dist.put(src, B, dst_rank=peer_rank, signal=d0)
            T.dist.put(src, B, dst_rank=peer_rank, signal=m0)
            T.dist.submit()
            T.dist.wait_signal(s1)
            T.dist.wait_signal(s0)
            T.dist.wait_signal(d0)
            T.dist.wait_signal(m0)

    return main


@tilelang.jit(target="sunmmio")
def multi_destination_value_kernel_factory(
    signal_kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE,
    world_size: int = 1,
):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal(kind=signal_kind)
            if rank_id == 0:
                T.dist.put(src, dst, dst_rank=1, signal=signal)
                T.dist.put(src, dst, dst_rank=2, signal=signal)
                T.dist.submit()
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def loop_destination_generation_kernel_factory(
    signal_kind,
    expect_mode,
    group_member=False,
    world_size: int = 3,
):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            # Separate rounds cannot overwrite a send or an unconsumed receive.
            src = T.alloc_shared((2, 32), T.bfloat16)
            dst = T.alloc_shared((2, 32), T.bfloat16)
            if group_member:
                signals = T.dist.signals(1, kind=signal_kind, expect=expect_mode)
            else:
                signal = T.dist.signal(kind=signal_kind, expect=expect_mode)
            for step in T.serial(2):
                for peer in T.serial(1, world_size):
                    if rank_id == 0:
                        T.dist.put(
                            src[step : step + 1, 0:32],
                            dst[step : step + 1, 0:32],
                            dst_rank=peer,
                            signal=signals[0] if group_member else signal,
                            submit=True,
                        )
                if expect_mode == "manual":
                    T.dist.wait_signal(
                        signals[0] if group_member else signal,
                        expected_delta=T.if_then_else(rank_id == 0, 0, 1),
                    )
                else:
                    T.dist.wait_signal(signals[0] if group_member else signal)

    return main


def _collect_leaf_signals(func, op_name, kind_index_positions, expected_position):
    result = []

    def visit(node):
        if not isinstance(node, tvm.tir.Call) or not isinstance(node.op, tvm.ir.Op):
            return
        if node.op.name != op_name:
            return
        expected = strip_casts(node.args[expected_position])
        if isinstance(expected, tvm.tir.BufferLoad):
            state_name = expected.buffer.name
        else:
            assert isinstance(expected, tvm.tir.IntImm)
            assert int(expected) == 0
            state_name = None
        kind_position, index_position = kind_index_positions
        result.append(
            (
                str(node.args[kind_position].value),
                int(node.args[index_position]),
                state_name,
            )
        )

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return result


def _collect_generation_advances(func):
    advances = Counter()

    def visit(node):
        if isinstance(node, tvm.tir.BufferStore) and node.buffer.scope() == "local.var":
            advances[node.buffer.name] += 1

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return advances


def test_explicit_memory_signal_reaches_stable_leaf_tir():
    func = explicit_memory_signal_kernel_factory.get_tir(world_size=4)
    device_func = single_prim_func(lower_to_device_tir(func).device_mod)
    put = collect_calls(device_func, "tl.dist_put_")[0]
    wait = collect_calls(device_func, "tl.dist_wait_signal_")[0]
    assert put.args[3].value == wait.args[0].value == "sram_memory"
    assert put.args[5].dtype == wait.args[2].dtype == "uint64"
    assert strip_casts(put.args[5]).buffer.dtype == strip_casts(wait.args[2]).buffer.dtype == "uint32"


@pytest.mark.parametrize("signal_kind", [T.dist.SignalKind.SRAM_FLAGREG_VALUE, T.dist.SignalKind.SRAM_MEMORY])
@pytest.mark.parametrize("expect_mode", ["auto", "manual"])
def test_signal_generation_is_reused_across_serial_loop(signal_kind, expect_mode):
    func = reused_signal_kernel_factory.get_tir(signal_kind=signal_kind, expect_mode=expect_mode, world_size=4)
    result = lower_to_device_tir(func)
    device_func = single_prim_func(result.device_mod)
    op_names = collect_op_names(device_func)
    script = device_func.script()

    assert op_names.count("tl.dist_put_") == 1
    assert op_names.count("tl.dist_wait_signal_") == 1
    assert op_names.count("tl.dist_wait_send") == 2
    assert "for _step in range(2):" in script
    state_dtype = "uint32" if signal_kind.value.endswith("memory") else "uint8"
    assert f"signal_expect_1[0] = signal_expect_1[0] + T.{state_dtype}(1)" in script
    assert f"signal_generation_1[0] = signal_generation_1[0] + T.{state_dtype}(1)" in script
    generation = strip_casts(collect_calls(device_func, "tl.dist_put_")[0].args[5])
    assert isinstance(generation, tvm.tir.BufferLoad)
    assert tuple(int(extent) for extent in generation.buffer.shape) == (1,)
    assert int(generation.indices[0]) == 0
    assert collect_calls(device_func, "tl.dist_wait_signal_")[0].args[0].value == signal_kind.value
    initial, expected = (2**32 - 2, 2**32 - 1) if state_dtype == "uint32" else (254, 255)
    receiver_state = collect_calls(device_func, "tl.dist_wait_signal_")[0].args[2]
    for state in (generation, receiver_state):
        assert _evaluate_state_update(device_func, state, initial) == expected


def test_put_advances_generation_and_wait_only_reads_it():
    func = two_puts_two_waits_kernel_factory.get_tir(world_size=4)
    device_func = single_prim_func(lower_to_device_tir(func).device_mod)
    script = device_func.script()

    assert script.count("signal_expect_1[0] = signal_expect_1[0] + T.uint8(1)") == 2
    assert script.count("signal_generation_1[0] = signal_generation_1[0] + T.uint8(1)") == 2
    assert script.count("T.dist_put_(") == 2
    assert script.count('T.dist_wait_signal_("sram_flagreg_value", 0, signal_expect_1[0]') == 2

    lines = [line.strip() for line in script.splitlines()]
    advance_indices = [index for index, line in enumerate(lines) if line.startswith("signal_generation_1[0] =")]
    put_indices = [index for index, line in enumerate(lines) if line.startswith("T.dist_put_(")]
    wait_indices = [index for index, line in enumerate(lines) if line.startswith("T.dist_wait_signal_(")]
    assert advance_indices[0] < put_indices[0] < advance_indices[1] < put_indices[1]
    assert put_indices[1] < wait_indices[0] < wait_indices[1]


def test_signals_preserve_independent_expected_and_generation():
    func = multi_signal_kernel_factory.get_tir(32, 32, world_size=4)
    result = lower_to_device_tir(
        func,
        capture_before_passes="tl.PlanDistSignals",
        capture_passes=("tl.PlanDistSignals", "tl.LowerDistSync"),
    )

    before_plan = result.pass_snapshot("tl.PlanDistSignals", when="before").mod.script()
    assert before_plan.count('T.dist_signal_decl("auto"') == 2
    assert "T.dist_signal(" not in before_plan

    after_plan_func = single_prim_func(result.pass_snapshot("tl.PlanDistSignals").mod)
    assert signal_counts(after_plan_func) == {
        "sram_flagreg_inc": 1,
        "dram_flagreg_inc": 1,
        "sram_flagreg_value": 1,
        "dram_flagreg_value": 0,
        "sram_memory": 0,
        "dram_memory": 1,
    }
    after_plan_script = after_plan_func.script()
    assert "T.dist_signal_decl" not in after_plan_script
    assert 's0: T.handle = T.dist_signal("sram_flagreg_inc", 0, "auto")' in after_plan_script
    assert 's1: T.handle = T.dist_signal("sram_flagreg_value", 0, "auto")' in after_plan_script
    assert 'd0: T.handle = T.dist_signal("dram_flagreg_inc", 0, "auto")' in after_plan_script
    assert 'm0: T.handle = T.dist_signal("dram_memory", 0, "auto")' in after_plan_script

    device_func = single_prim_func(result.device_mod)
    assert signal_counts(device_func) == signal_counts(after_plan_func)
    puts = _collect_leaf_signals(device_func, "tl.dist_put_", (3, 4), 5)
    waits = _collect_leaf_signals(device_func, "tl.dist_wait_signal_", (0, 1), 2)
    assert [(kind, index) for kind, index, _ in puts] == [
        ("sram_flagreg_inc", 0),
        ("sram_flagreg_value", 0),
        ("sram_flagreg_inc", 0),
        ("dram_flagreg_inc", 0),
        ("dram_memory", 0),
    ]
    assert [(kind, index) for kind, index, _ in waits] == [
        ("sram_flagreg_value", 0),
        ("sram_flagreg_inc", 0),
        ("dram_flagreg_inc", 0),
        ("dram_memory", 0),
    ]

    expected_by_signal = {(kind, index): name for kind, index, name in waits}
    generation_by_signal = {(kind, index): name for kind, index, name in puts if name is not None}
    advances = _collect_generation_advances(device_func)
    assert advances[generation_by_signal[("sram_flagreg_value", 0)]] == 1
    assert advances[generation_by_signal[("dram_memory", 0)]] == 1
    assert advances[expected_by_signal[("sram_flagreg_inc", 0)]] == 2
    assert advances[expected_by_signal[("sram_flagreg_value", 0)]] == 1
    assert advances[expected_by_signal[("dram_flagreg_inc", 0)]] == 1
    assert advances[expected_by_signal[("dram_memory", 0)]] == 1


@pytest.mark.parametrize(
    "signal_kind",
    [T.dist.SignalKind.SRAM_FLAGREG_VALUE, T.dist.SignalKind.SRAM_MEMORY],
)
def test_value_and_memory_signals_use_generation_per_destination(signal_kind):
    func = multi_destination_value_kernel_factory.get_tir(
        signal_kind=signal_kind,
        world_size=4,
    )
    result = lower_to_device_tir(
        func,
        capture_passes=("tl.LowerDistCommunication", "tl.LowerDistSync"),
    )
    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod
    allocations = []

    def collect_allocation(node):
        if isinstance(node, tvm.tir.Allocate) and "generation" in node.buffer_var.name:
            allocations.append(tuple(int(extent) for extent in node.extents))

    tvm.tir.stmt_functor.post_order_visit(lowered["main"].body, collect_allocation)
    assert allocations == [(4,)]

    generation_indices = []

    def visit(node):
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.dist_put_":
            generation = node.args[5]
            if isinstance(generation, tvm.tir.Cast):
                assert generation.dtype == ("uint64" if signal_kind.value.endswith("memory") else "uint8")
                generation = generation.value
            assert isinstance(generation, tvm.tir.BufferLoad)
            generation_indices.append(int(generation.indices[0]))

    for candidate in result.pass_snapshot("tl.LowerDistSync").mod.functions.values():
        if isinstance(candidate, tvm.tir.PrimFunc):
            tvm.tir.stmt_functor.post_order_visit(candidate.body, visit)
    assert generation_indices == [1, 2]


@pytest.mark.parametrize("signal_kind", [T.dist.SignalKind.SRAM_FLAGREG_VALUE, T.dist.SignalKind.SRAM_MEMORY])
@pytest.mark.parametrize("expect_mode", ["auto", "manual"])
@pytest.mark.parametrize("group_member", [False, True])
def test_loop_destinations_keep_independent_signal_generations(signal_kind, expect_mode, group_member):
    world_size = 3
    func = loop_destination_generation_kernel_factory.get_tir(
        signal_kind=signal_kind,
        expect_mode=expect_mode,
        group_member=group_member,
        world_size=world_size,
    )
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistCommunication")
    analyzer = tvm.arith.Analyzer()
    for mod in (result.pass_snapshot("tl.LowerDistCommunication").mod, result.device_mod):
        puts = collect_calls(mod, "tl.dist_put_")
        assert len(puts) == 1  # The payload peer loop remains a loop.
        put = puts[0]
        generation = put.args[5]
        if isinstance(generation, tvm.tir.Cast):
            assert generation.dtype == ("uint64" if signal_kind.value.endswith("memory") else "uint8")
            generation = generation.value
        assert isinstance(generation, tvm.tir.BufferLoad)
        assert tuple(int(extent) for extent in generation.buffer.shape) == (world_size,)
        assert analyzer.can_prove_equal(generation.indices[0], put.args[2])
    advances = collect_nodes(
        result.device_mod,
        tvm.tir.BufferStore,
        lambda store: store.buffer.data.same_as(generation.buffer.data) and not isinstance(store.value, tvm.tir.IntImm),
    )
    assert len(advances) == 1
    assert analyzer.can_prove_equal(advances[0].indices[0], generation.indices[0])
    assert analyzer.can_prove_equal(advances[0].value, generation + 1)


def _evaluate_state_update(mod, state, initial, bindings=None):
    """Evaluate the actual pass-generated state update at a numeric boundary."""
    state = strip_casts(state)
    stores = collect_nodes(
        mod,
        tvm.tir.BufferStore,
        lambda store: store.buffer.data.same_as(state.buffer.data) and not isinstance(store.value, tvm.tir.IntImm),
    )
    assert len(stores) == 1

    def evaluate_constants(node):
        if isinstance(node, tvm.tir.BufferLoad) and node.buffer.data.same_as(state.buffer.data):
            return tvm.tir.IntImm(node.dtype, initial)

    value = tvm.tir.stmt_functor.substitute(stores[0].value, bindings or {})
    statement = tvm.tir.stmt_functor.ir_transform(tvm.tir.Evaluate(value), None, evaluate_constants, ["tir.BufferLoad"])
    return int(tvm.arith.Analyzer().simplify(statement.value))


@pytest.mark.parametrize("signal_kind", list(T.dist.SignalKind))
def test_signal_only_state_uses_suvm_width_and_preserves_boundary_increment(signal_kind):
    result = lower_to_device_tir(signal_only_dtype_kernel_factory.get_tir(signal_kind=signal_kind, world_size=2))
    put = collect_calls(result.device_mod, "tl.dist_signal_put_")[0]
    wait = collect_calls(result.device_mod, "tl.dist_wait_signal_")[0]
    state_dtype = "uint64" if signal_kind.value.endswith("memory") else "uint8"
    assert put.args[3].dtype == wait.args[2].dtype == state_dtype
    initial, expected = (2**32 - 2, 2**32 - 1) if state_dtype == "uint64" else (254, 255)
    assert _evaluate_state_update(result.device_mod, wait.args[2], initial) == expected
    if signal_kind.value.endswith("inc"):
        assert isinstance(put.args[3], tvm.tir.IntImm) and int(put.args[3]) == 0
        assert not collect_nodes(result.device_mod, tvm.tir.Allocate, lambda node: "generation" in node.buffer_var.name)
    else:
        assert _evaluate_state_update(result.device_mod, put.args[3], initial) == expected


@pytest.mark.parametrize("signal_kind", list(T.dist.SignalKind))
def test_manual_delta_uses_signal_width_without_truncating_memory_state(signal_kind):
    memory_kind = signal_kind.value.endswith("memory")
    if not memory_kind:
        with pytest.raises(tvm.error.InternalError, match="cumulative expected"):
            lower_to_device_tir(manual_delta_dtype_kernel_factory.get_tir(signal_kind=signal_kind, world_size=2))
        return
    result = lower_to_device_tir(manual_delta_dtype_kernel_factory.get_tir(signal_kind=signal_kind, world_size=2))
    device = single_prim_func(result.device_mod)
    state = collect_calls(device, "tl.dist_wait_signal_")[0].args[2]
    delta = next(param for param in device.params if param.name == "delta")
    assert state.dtype == "uint64"
    for increment in (0, 255, 256, 2**32 - 1):
        assert _evaluate_state_update(device, state, 0, {delta: tvm.tir.IntImm("uint32", increment)}) == increment


@pytest.mark.parametrize("signal_kind", list(T.dist.SignalKind))
@pytest.mark.parametrize("increment", [256, 257])
def test_constant_manual_delta_reaches_device_tir(signal_kind, increment):
    if not signal_kind.value.endswith("memory"):
        with pytest.raises(tvm.error.InternalError, match="cumulative expected"):
            lower_to_device_tir(manual_delta_dtype_kernel_factory.get_tir(signal_kind=signal_kind, fixed_delta=increment, world_size=2))
        return
    result = lower_to_device_tir(
        manual_delta_dtype_kernel_factory.get_tir(signal_kind=signal_kind, fixed_delta=increment, world_size=2),
        capture_passes="tl.LowerDistSync",
    )
    wait = collect_calls(result.device_mod, "tl.dist_wait_signal_")[0]
    state_dtype = "uint64" if signal_kind.value.endswith("memory") else "uint8"
    assert wait.args[2].dtype == state_dtype
    lowered = result.pass_snapshot("tl.LowerDistSync").mod
    state = collect_calls(lowered, "tl.dist_wait_signal_")[0].args[2]
    assert _evaluate_state_update(lowered, state, 0) == (increment if state_dtype == "uint64" else increment % 256)


@tilelang.jit(target="sunmmio")
def expected_range_kernel_factory(kind, delta=1, rounds=1, group=False, world_size: int = 1):
    delta_expr = tvm.tir.const(delta, "uint64" if delta >= 0 else "int64")

    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            if group:
                signals = T.dist.signals(2, kind=kind, expect="manual")
            else:
                signal = T.dist.signal(kind=kind, expect="manual")
            for _ in T.serial(rounds):
                T.dist.wait_signal(signals[0] if group else signal, expected_delta=delta_expr)
            if group:
                T.dist.wait_signal(signals[1], expected_delta=delta_expr)
            T.dist.submit()
            T.dist.wait_signal(signals[0] if group else signal, expected_delta=0)

    return main


@pytest.mark.parametrize("kind", list(T.dist.SignalKind))
@pytest.mark.parametrize("delta,rounds", [(0, 3), (255, 1), (1, 255), (200, 1)])
def test_expected_range_accepts_safe_cumulative_values(kind, delta, rounds):
    result = lower_to_device_tir(expected_range_kernel_factory.get_tir(kind, delta, rounds, group=True, world_size=2))
    assert collect_calls(result.device_mod, "tl.dist_wait_signal_")


@pytest.mark.parametrize("kind", [kind for kind in T.dist.SignalKind if not kind.value.endswith("memory")])
@pytest.mark.parametrize("delta,rounds", [(256, 1), (128, 2), (1, 256), (-1, 1)])
def test_expected_range_rejects_flag_overflow_and_negative_delta(kind, delta, rounds):
    with pytest.raises(tvm.error.InternalError, match="T.dist .*expected"):
        lower_to_device_tir(expected_range_kernel_factory.get_tir(kind, delta, rounds, world_size=2))


@pytest.mark.parametrize("kind", [T.dist.SignalKind.SRAM_MEMORY, T.dist.SignalKind.DRAM_MEMORY])
@pytest.mark.parametrize("delta,rounds", [(256, 2), (2**32 - 1, 1), (2**31 - 1, 2)])
def test_expected_range_preserves_wide_memory_values(kind, delta, rounds):
    result = lower_to_device_tir(expected_range_kernel_factory.get_tir(kind, delta, rounds, world_size=2))
    assert collect_calls(result.device_mod, "tl.dist_wait_signal_")[0].args[2].dtype == "uint64"


@tilelang.jit(target="sunmmio")
def guarded_expected_range_kernel_factory(mode="branch", world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, delta: T.int32):
        with T.Kernel():
            signal = T.dist.signal(expect="manual")
            if mode == "branch":
                if delta > 0:
                    T.dist.wait_signal(signal, expected_delta=200)
                else:
                    T.dist.wait_signal(signal, expected_delta=100)
                T.dist.wait_signal(signal, expected_delta=55)
            elif mode == "guard":
                if delta >= 0 and delta <= 5:
                    T.dist.wait_signal(signal, expected_delta=250)
                    T.dist.wait_signal(signal, expected_delta=delta)
            elif mode == "bounded_loop":
                for _ in T.serial(T.min(T.max(delta, 0), 5)):
                    T.dist.wait_signal(signal, expected_delta=50)
            else:
                T.dist.wait_signal(signal, expected_delta=delta)

    return main


@pytest.mark.parametrize("mode", ["branch", "guard", "bounded_loop"])
def test_expected_range_accepts_bounded_dynamic_updates(mode):
    lower_to_device_tir(guarded_expected_range_kernel_factory.get_tir(mode, world_size=2))


def test_expected_range_rejects_unknown_manual_delta():
    with pytest.raises(tvm.error.InternalError, match="cannot prove.*expected"):
        lower_to_device_tir(guarded_expected_range_kernel_factory.get_tir("unknown", world_size=2))


@tilelang.jit(target="sunmmio")
def expected_sequence_kernel_factory(first=250, second=5, memory=False, world_size: int = 1):
    first_value = tvm.tir.const(first, "uint64")
    second_value = tvm.tir.const(second, "uint64")

    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            signal = T.dist.signal(
                kind=T.dist.SignalKind.SRAM_MEMORY if memory else T.dist.SignalKind.SRAM_FLAGREG_INC,
                expect="manual",
            )
            T.dist.wait_signal(signal, expected_delta=first_value)
            T.dist.submit()
            T.dist.wait_signal(signal, expected_delta=0)
            T.dist.wait_signal(signal, expected_delta=second_value)

    return main


@pytest.mark.parametrize(
    "first,second,memory,valid",
    [
        (250, 5, False, True),
        (250, 10, False, False),
        (2**31, 1, True, True),
        (2**32 - 1, 0, True, True),
        (2**32 - 1, 1, True, False),
        (2**32, 0, True, False),
        (2**64 - 1, 0, True, False),
    ],
)
def test_expected_range_checks_original_cumulative_value(first, second, memory, valid):
    func = expected_sequence_kernel_factory.get_tir(first, second, memory, world_size=2)
    if valid:
        lower_to_device_tir(func)
    else:
        with pytest.raises(tvm.error.InternalError, match="cumulative expected"):
            lower_to_device_tir(func)


@tilelang.jit(target="sunmmio")
def expected_array_range_kernel_factory(mode="safe", world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, delta: T.int32):
        with T.Kernel():
            deltas = T.alloc_shared((2,), T.int32)
            signals = T.dist.signals(2, expect="manual")
            T.fill(deltas, 200)
            T.dist.wait_signals(signals, expected_deltas=deltas)
            deltas[0] = -1 if mode == "negative" else 55
            deltas[1] = 56 if mode == "overflow" else 55
            if mode == "unknown_write":
                deltas[1] = delta
            T.dist.wait_signals(signals, expected_deltas=deltas)

    return main


@pytest.mark.parametrize("mode", ["safe", "negative", "overflow", "unknown_write"])
def test_expected_range_tracks_delta_array_updates(mode):
    func = expected_array_range_kernel_factory.get_tir(mode, world_size=2)
    if mode == "safe":
        lower_to_device_tir(func)
    else:
        with pytest.raises(tvm.error.InternalError, match="T.dist .*expected"):
            lower_to_device_tir(func)


@pytest.mark.parametrize("rounds", [5000, 2**30])
def test_expected_range_summarizes_long_invariant_memory_loop(rounds):
    # Keep this range-check test independent of the downstream loop unroller.
    mod = base_mod(expected_range_kernel_factory.get_tir(T.dist.SignalKind.SRAM_MEMORY, 1, rounds, world_size=2))
    for transform in (
        tilelang.transform.InferSramScope(),
        tilelang.transform.PlanDistSignals(),
        tilelang.transform.LowerDistRouting(),
        tilelang.transform.LowerDistCommunication(),
    ):
        mod = transform(mod)
    assert any(int(loop.extent) == rounds for loop in collect_nodes(mod, tvm.tir.For))
    assert collect_calls(mod, "tl.dist_expect_")


@tilelang.jit(target="sunmmio")
def expected_overwritten_delta_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(inputs: T.MeshTensor((32,), T.placement.replicated(), T.int32), rank_id: T.dist.RankId):
        with T.Kernel():
            deltas = T.alloc_shared((32,), T.int32)
            signal = T.dist.signal(expect="manual")
            T.fill(deltas, 1)
            T.copy(inputs, deltas)
            T.dist.wait_signal(signal, expected_delta=deltas[0])

    return main


def test_expected_range_invalidates_delta_after_copy():
    with pytest.raises(tvm.error.InternalError, match="cannot prove.*expected"):
        lower_to_device_tir(expected_overwritten_delta_kernel_factory.get_tir(world_size=2))


@tilelang.jit(target="sunmmio")
def memory_dynamic_delta_kernel_factory(repeat=False, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, delta: T.uint64):
        with T.Kernel():
            signal = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            T.dist.wait_signal(signal, expected_delta=delta)
            if repeat:
                T.dist.wait_signal(signal, expected_delta=delta)

    return main


@pytest.mark.parametrize("repeat", [False, True])
def test_memory_expected_range_checks_full_width_dynamic_accumulation(repeat):
    func = memory_dynamic_delta_kernel_factory.get_tir(repeat, world_size=2)
    with pytest.raises(tvm.error.InternalError, match="cumulative expected"):
        lower_to_device_tir(func)


@tilelang.jit(target="sunmmio")
def dynamic_member_expected_range_kernel_factory(kind, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, member: T.int32):
        with T.Kernel():
            signals = T.dist.signals(2, kind=kind, expect="manual")
            T.dist.wait_signal(signals[member], expected_delta=1)
            T.dist.wait_signal(signals[member], expected_delta=1)

    return main


@pytest.mark.parametrize("kind", list(T.dist.SignalKind))
def test_expected_range_preserves_repeated_dynamic_member_updates(kind):
    lower_to_device_tir(dynamic_member_expected_range_kernel_factory.get_tir(kind, world_size=2))


@tilelang.jit(target="sunmmio")
def refilled_delta_kernel_factory(replacement=256, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, member: T.int32):
        with T.Kernel():
            deltas = T.alloc_shared((2,), T.int32)
            signal = T.dist.signal(expect="manual")
            T.fill(deltas, 1)
            T.dist.wait_signal(signal, expected_delta=deltas[member % 2])
            T.fill(deltas, replacement)
            T.dist.wait_signal(signal, expected_delta=deltas[member % 2])

    return main


@pytest.mark.parametrize("replacement", [254, 255, 256])
def test_expected_range_uses_refilled_dynamic_delta(replacement):
    func = refilled_delta_kernel_factory.get_tir(replacement, world_size=2)
    if replacement == 254:
        lower_to_device_tir(func)
    else:
        with pytest.raises(tvm.error.InternalError, match="cumulative expected"):
            lower_to_device_tir(func)


@tilelang.jit(target="sunmmio")
def alias_fill_delta_kernel_factory(view="partial", world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            backing = T.alloc_shared((64,), T.int32)
            deltas = T.decl_buffer((32,), T.int32, data=backing.data, scope="shared.rsram")
            alias = T.decl_buffer(
                (16 if view == "partial" else 32,),
                T.int32,
                data=backing.data,
                elem_offset=0 if view == "equivalent" else 16,
                scope="shared.rsram",
            )
            signal = T.dist.signal(expect="manual")
            T.fill(backing, 256)
            T.fill(alias, 1)
            T.dist.wait_signal(signal, expected_delta=deltas[0])

    return main


@pytest.mark.parametrize("view", ["partial", "offset", "equivalent"])
def test_expected_range_fill_requires_equivalent_tracked_buffer(view):
    func = alias_fill_delta_kernel_factory.get_tir(view, world_size=2)
    if view == "equivalent":
        lower_to_device_tir(func)
    else:
        with pytest.raises(tvm.error.InternalError, match="T.dist cannot prove.*expected"):
            lower_to_device_tir(func)
