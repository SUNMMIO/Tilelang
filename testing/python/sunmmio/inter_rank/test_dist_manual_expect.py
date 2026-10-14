"""Manual expected deltas, SignalList indexing, and collective protocol integration."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import (
    collect_nodes,
    collect_op_names,
)


@tilelang.jit(target="sunmmio")
def routed_expect_mode_kernel_factory(
    expect_mode="manual",
    group_member=False,
    dst_row=0,
    signal_kind=None,
    repeat=False,
    world_size: int = 1,
):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel() as core_id:
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            if group_member:
                signals = T.dist.signals(2, kind=signal_kind, expect=expect_mode)
            else:
                signal = T.dist.signal(kind=signal_kind, expect=expect_mode)
            T.dist.routed_put(
                src,
                dst,
                routes=[[0, (rank_id + 1) % world_size, dst_row]],
                signal=signals[1] if group_member else signal,
                submit=True,
            )
            if repeat:
                T.dist.routed_put(
                    src,
                    dst,
                    routes=[[0, (rank_id + 1) % world_size, dst_row]],
                    signal=signals[1] if group_member else signal,
                    submit=True,
                )
            if expect_mode == "manual":
                if core_id // T.mesh_ncols() == dst_row:
                    T.dist.wait_signal(
                        signals[1] if group_member else signal,
                        expected_delta=2 if repeat else 1,
                    )
            else:
                T.dist.wait_signal(signals[1] if group_member else signal)

    return main


@tilelang.jit(target="sunmmio")
def manual_sender_signals_kernel_factory(
    expect_mode="manual",
    cross_row=False,
    world_size: int = 1,
):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel() as core_id:
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            send_counts = T.alloc_shared((world_size,), T.int32)
            expected_deltas = T.alloc_shared((world_size,), T.uint32)
            sender_signals = T.dist.signals(
                world_size,
                kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE,
                expect=expect_mode,
            )
            peer_rank = (rank_id + 1) % world_size
            current_row = core_id // T.mesh_ncols()
            dst_row = (current_row + 1) % T.mesh_nrows() if cross_row else current_row

            if send_counts[peer_rank] > 0:
                T.dist.put(
                    src,
                    dst,
                    dst_rank=peer_rank,
                    dst_row=dst_row,
                    signal=sender_signals[rank_id],
                )
            T.dist.submit()
            T.dist.wait_signals(
                sender_signals,
                expected_deltas=expected_deltas,
            )
            T.dist.wait()

    return main


@tilelang.jit(target="sunmmio")
def manual_signal_wave_kernel_factory(
    total_signals=20,
    wave_size=8,
    world_size: int = 1,
):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            expected_deltas = T.alloc_shared((total_signals,), T.uint32)
            signals = T.dist.signals(
                total_signals,
                kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE,
                expect="manual",
            )
            peer_rank = (rank_id + 1) % world_size

            for wave in T.serial(T.ceildiv(total_signals, wave_size)):
                for lane in T.serial(wave_size):
                    index = wave * wave_size + lane
                    if index < total_signals:
                        T.dist.put(
                            src,
                            dst,
                            dst_rank=peer_rank,
                            signal=signals[index],
                        )
                T.dist.submit()
                for member in T.serial(total_signals):
                    expected_deltas[member] = T.if_then_else(
                        T.And(
                            wave * wave_size <= member,
                            member < T.min((wave + 1) * wave_size, total_signals),
                        ),
                        1,
                        0,
                    )
                T.dist.wait_signals(
                    signals,
                    expected_deltas=expected_deltas,
                )
                T.dist.wait()

    return main


@tilelang.jit(target="sunmmio")
def manual_destination_loop_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            expected_deltas = T.alloc_shared((world_size,), T.uint32)
            sender_signals = T.dist.signals(
                world_size,
                kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE,
                expect="manual",
            )

            for peer_rank in T.serial(world_size):
                if peer_rank != rank_id:
                    T.dist.put(
                        src,
                        dst,
                        dst_rank=peer_rank,
                        signal=sender_signals[rank_id],
                    )
            T.dist.submit()
            for source_rank in T.serial(world_size):
                expected_deltas[source_rank] = T.if_then_else(
                    source_rank != rank_id,
                    1,
                    0,
                )
            T.dist.wait_signals(
                sender_signals,
                expected_deltas=expected_deltas,
            )
            T.dist.wait()

    return main


@tilelang.jit(target="sunmmio")
def manual_dynamic_member_wait_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            expected_deltas = T.alloc_shared((7,), T.uint32)
            signals = T.dist.signals(
                7,
                kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE,
                expect="manual",
            )
            peer_rank = (rank_id + 1) % world_size

            for expert in T.serial(7):
                T.dist.put(
                    src,
                    dst,
                    dst_rank=peer_rank,
                    signal=signals[expert],
                )
                T.dist.submit()
                T.dist.wait_signal(
                    signals[expert],
                    expected_delta=expected_deltas[expert],
                )

    return main


@tilelang.jit(target="sunmmio")
def protocol_dynamic_member_kernel_factory(
    expect_mode=None,
    world_size: int = 1,
):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            send_counts = T.alloc_shared((world_size,), T.int32)
            recv_counts = T.alloc_shared((world_size,), T.int32)
            signals = T.dist.signals(
                2,
                kind=T.dist.SignalKind.SRAM_FLAGREG_INC,
                expect=expect_mode,
            )

            for expert in T.serial(2):
                T.dist.all_to_allv(
                    src,
                    dst,
                    send_counts=send_counts,
                    recv_counts=recv_counts,
                    signal=signals[expert],
                    submit=True,
                )
                T.dist.wait_signal(signals[expert])

    return main


@pytest.mark.parametrize("group_member", [False, True])
@pytest.mark.parametrize("dst_row", [0, 1])
@pytest.mark.parametrize(
    "signal_kind",
    [
        None,
        T.dist.SignalKind.SRAM_FLAGREG_VALUE,
        T.dist.SignalKind.SRAM_MEMORY,
    ],
)
def test_manual_routed_put_only_advances_expected_at_explicit_wait(
    group_member,
    dst_row,
    signal_kind,
):
    result = lower_to_device_tir(
        routed_expect_mode_kernel_factory.get_tir(
            group_member=group_member,
            dst_row=dst_row,
            signal_kind=signal_kind,
            repeat=True,
            world_size=4,
        ),
        capture_passes=("tl.LowerDistCommunication", "tl.InjectDistSync"),
    )
    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod
    markers = collect_nodes(
        lowered,
        tvm.tir.Call,
        lambda call: isinstance(call.op, tvm.ir.Op) and call.op.name == "tl.dist_expect_",
    )
    assert len(markers) == 1
    assert int(markers[0].args[1]) == 2
    script = lowered.script()
    assert script.rindex("T.dist_put_(") < script.index("T.dist_expect_(")

    injected = result.pass_snapshot("tl.InjectDistSync").mod
    waits = collect_nodes(
        injected,
        tvm.tir.Call,
        lambda call: isinstance(call.op, tvm.ir.Op) and call.op.name == "tl.dist_wait_signal_",
    )
    assert len(waits) == 1
    expected = waits[0].args[2]
    stores = collect_nodes(
        injected,
        tvm.tir.BufferStore,
        lambda store: store.buffer.data.same_as(expected.buffer.data) and not isinstance(store.value, tvm.tir.IntImm),
    )
    assert len(stores) == 1
    if signal_kind in (T.dist.SignalKind.SRAM_FLAGREG_VALUE, T.dist.SignalKind.SRAM_MEMORY):
        puts = collect_nodes(
            injected,
            tvm.tir.Call,
            lambda call: isinstance(call.op, tvm.ir.Op) and call.op.name == "tl.dist_put_",
        )
        assert len(puts) == 2
        assert all(isinstance(call.args[5], tvm.tir.BufferLoad) for call in puts)
        generation = puts[0].args[5].buffer.data
        increments = collect_nodes(
            injected,
            tvm.tir.BufferStore,
            lambda store: store.buffer.data.same_as(generation) and not isinstance(store.value, tvm.tir.IntImm),
        )
        assert len(increments) == 2


@pytest.mark.parametrize("expect_mode", [None, "auto"])
@pytest.mark.parametrize("group_member", [False, True])
def test_auto_routed_put_still_infers_expected(expect_mode, group_member):
    result = lower_to_device_tir(
        routed_expect_mode_kernel_factory.get_tir(
            expect_mode=expect_mode,
            group_member=group_member,
            dst_row=1,
            world_size=4,
        ),
        capture_passes="tl.LowerDistCommunication",
    )
    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod
    assert collect_op_names(lowered).count("tl.dist_expect_") == 1
    assert lowered.script().index("T.dist_expect_(") < lowered.script().index("T.dist_put_(")


@pytest.mark.parametrize("expect_mode", ["manual", T.dist.ExpectMode.MANUAL])
def test_manual_expect_accepts_string_and_enum(expect_mode):
    func = manual_sender_signals_kernel_factory.get_tir(
        expect_mode=expect_mode,
        world_size=4,
    )
    names = collect_op_names(func)
    assert names.count("tl.dist_signal_group_decl") == 1
    assert names.count("tl.dist_signal_group_route") == 1
    assert names.count("tl.dist_wait_signals") == 1


def test_manual_sender_signals_lower_dynamic_if_and_expected_deltas():
    func = manual_sender_signals_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(
        func,
        capture_passes=(
            "tl.PlanDistSignals",
            "tl.LowerDistRouting",
            "tl.LowerDistCommunication",
            "tl.InjectDistSync",
        ),
    )

    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_value"]) == 4

    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    routed_script = routed.script()
    assert "tl.dist_wait_signals" in collect_op_names(routed)
    assert "send_counts" in routed_script
    assert "T.dist_signal_group_route" in routed_script

    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod
    lowered_names = collect_op_names(lowered)
    assert "tl.dist_wait_signals" not in lowered_names
    assert lowered_names.count("tl.dist_expect_") == 4
    assert lowered_names.count("tl.dist_wait_signal_") == 4
    lowered_script = lowered.script()
    first_wait = lowered_script.index("T.dist_wait_signal_")
    assert lowered_script.rindex("T.dist_expect_") < first_wait

    injected = result.pass_snapshot("tl.InjectDistSync").mod
    injected_names = collect_op_names(injected)
    injected_script = injected.script()
    assert "tl.dist_expect_" not in injected_names
    assert injected_names.count("tl.dist_wait_signal_") == 4
    assert "expected_deltas[0]" in injected_script
    assert "expected_deltas[3]" in injected_script
    assert "send_counts" in result.device_mod.script()


def test_manual_sender_signals_support_cross_row_dynamic_if():
    func = manual_sender_signals_kernel_factory.get_tir(
        cross_row=True,
        world_size=4,
    )
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistRouting")
    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    names = collect_op_names(routed)
    script = routed.script()

    assert names.count("tl.tileop.dist_routed_peer_put") == 4
    assert "dist_route_active" in script
    assert "send_counts" in script


def test_manual_sender_signals_reject_auto_mode():
    with pytest.raises(TypeError, match="expect='manual'"):
        manual_sender_signals_kernel_factory.get_tir(
            expect_mode="auto",
            world_size=4,
        )


def test_manual_sender_signals_reject_invalid_expect_mode():
    with pytest.raises(ValueError, match="expect must be"):
        manual_sender_signals_kernel_factory.get_tir(
            expect_mode="invalid",
            world_size=4,
        )


def test_manual_signal_list_supports_static_member():
    @tilelang.jit(target="sunmmio")
    def static_member_kernel_factory(world_size: int = 1):
        @T.prim_func
        def main(rank_id: T.dist.RankId):
            with T.Kernel():
                src = T.alloc_shared((32,), T.bfloat16)
                dst = T.alloc_shared((32,), T.bfloat16)
                sender_signals = T.dist.signals(
                    world_size,
                    kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE,
                    expect="manual",
                )
                T.dist.put(
                    src,
                    dst,
                    dst_rank=(rank_id + 1) % world_size,
                    signal=sender_signals[0],
                )
                T.dist.submit()

        return main

    lower_to_device_tir(static_member_kernel_factory.get_tir(world_size=4))


def test_put_rejects_whole_signal_list():
    @tilelang.jit(target="sunmmio")
    def invalid_kernel_factory(world_size: int = 1):
        @T.prim_func
        def main(rank_id: T.dist.RankId):
            with T.Kernel():
                src = T.alloc_shared((32,), T.bfloat16)
                dst = T.alloc_shared((32,), T.bfloat16)
                sender_signals = T.dist.signals(world_size, expect="manual")
                T.dist.put(
                    src,
                    dst,
                    dst_rank=(rank_id + 1) % world_size,
                    signal=sender_signals,
                )

        return main

    with pytest.raises(TypeError, match="T.dist.signal"):
        invalid_kernel_factory.get_tir(world_size=4)


def test_dynamic_signal_index_requires_manual_and_static_analysis():
    @tilelang.jit(target="sunmmio")
    def invalid_auto_kernel_factory(world_size: int = 1):
        @T.prim_func
        def main(rank_id: T.dist.RankId):
            with T.Kernel():
                signals = T.dist.signals(world_size, expect="auto")
                T.evaluate(signals[rank_id].handle)

        return main

    @tilelang.jit(target="sunmmio")
    def rank_expression_kernel_factory(world_size: int = 1):
        @T.prim_func
        def main(rank_id: T.dist.RankId):
            with T.Kernel():
                src = T.alloc_shared((32,), T.bfloat16)
                dst = T.alloc_shared((32,), T.bfloat16)
                signals = T.dist.signals(
                    world_size,
                    kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE,
                    expect="manual",
                )
                T.dist.put(
                    src,
                    dst,
                    dst_rank=(rank_id + 1) % world_size,
                    signal=signals[(rank_id + 1) % world_size],
                )
                T.dist.submit()

        return main

    with pytest.raises(TypeError, match="expect='manual'"):
        invalid_auto_kernel_factory.get_tir(world_size=4)
    lower_to_device_tir(rank_expression_kernel_factory.get_tir(world_size=4))


def test_manual_signal_index_supports_static_waves_with_tail():
    func = manual_signal_wave_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(
        func,
        capture_passes=(
            "tl.PlanDistSignals",
            "tl.LowerDistCommunication",
        ),
    )

    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_value"]) == 20
    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod
    names = collect_op_names(lowered)
    script = lowered.script()
    assert names.count("tl.dist_put_") == 1
    assert names.count("tl.dist_expect_") == 20
    assert names.count("tl.dist_wait_signal_") == 20
    assert "wave * 8 + lane" in script


def test_manual_expected_supports_static_loop_destination_rank():
    func = manual_destination_loop_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(
        func,
        capture_passes=(
            "tl.PlanDistSignals",
            "tl.LowerDistRouting",
            "tl.LowerDistCommunication",
        ),
    )

    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    planned_script = planned.script()
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_value"]) == 4
    assert 'T.dist_signal_group("sram_flagreg_value", 0, 4, "manual")' in planned_script

    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    routed_script = routed.script()
    assert "for peer_rank in range(4)" in routed_script
    assert "peer_rank" in routed_script

    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod
    names = collect_op_names(lowered)
    assert names.count("tl.dist_put_") == 1
    assert names.count("tl.dist_expect_") == 4
    assert names.count("tl.dist_wait_signal_") == 4


def test_manual_dynamic_member_wait_lowers_to_one_physical_wait():
    func = manual_dynamic_member_wait_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(
        func,
        capture_passes=("tl.LowerDistCommunication", "tl.InjectDistSync"),
    )

    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod
    names = collect_op_names(lowered)
    script = lowered.script()
    assert names.count("tl.dist_expect_") == 1
    assert names.count("tl.dist_wait_signal_") == 1
    assert "expert" in script
    assert "expected_deltas[expert]" in script


@pytest.mark.parametrize("expect_mode", [None, "manual"])
def test_alltoallv_dynamic_member_resolves_protocol_manual(expect_mode):
    func = protocol_dynamic_member_kernel_factory.get_tir(
        expect_mode=expect_mode,
        world_size=4,
    )
    result = lower_to_device_tir(func, capture_passes="tl.PlanDistSignals")
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert 'T.dist_signal_group("sram_flagreg_inc", 0, 2, "manual")' in planned.script()


def test_alltoallv_dynamic_member_rejects_explicit_auto():
    with pytest.raises(TypeError, match="cannot use expect='auto'"):
        protocol_dynamic_member_kernel_factory.get_tir(
            expect_mode="auto",
            world_size=4,
        )
