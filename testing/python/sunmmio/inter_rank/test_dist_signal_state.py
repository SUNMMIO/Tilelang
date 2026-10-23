"""Signal expected/generation updates, repeated waits, and reuse across loop iterations."""

from collections import Counter

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import (
    collect_calls,
    collect_nodes,
    collect_op_names,
    signal_counts,
    single_prim_func,
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
        expected = node.args[expected_position]
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
    script = device_func.script()

    assert "T.dist_put_(" in script
    assert ', "sram_memory", 0, signal_generation_1[0])' in script
    assert 'T.dist_wait_signal_("sram_memory", 0, signal_expect_1[0]' in script


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
    assert "signal_expect_1[0] = signal_expect_1[0] + T.uint32(1)" in script
    assert "signal_generation_1[0] = signal_generation_1[0] + T.uint32(1)" in script
    generation = collect_calls(device_func, "tl.dist_put_")[0].args[5]
    assert isinstance(generation, tvm.tir.BufferLoad)
    assert tuple(int(extent) for extent in generation.buffer.shape) == (1,)
    assert int(generation.indices[0]) == 0
    assert collect_calls(device_func, "tl.dist_wait_signal_")[0].args[0].value == signal_kind.value


def test_put_advances_generation_and_wait_only_reads_it():
    func = two_puts_two_waits_kernel_factory.get_tir(world_size=4)
    device_func = single_prim_func(lower_to_device_tir(func).device_mod)
    script = device_func.script()

    assert script.count("signal_expect_1[0] = signal_expect_1[0] + T.uint32(1)") == 2
    assert script.count("signal_generation_1[0] = signal_generation_1[0] + T.uint32(1)") == 2
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
