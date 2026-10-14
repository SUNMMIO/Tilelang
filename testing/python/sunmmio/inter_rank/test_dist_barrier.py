"""Rank barrier frontend and lowering tests."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import (
    collect_op_names,
    lower_collectives,
)


@tilelang.jit(target="sunmmio")
def barrier_kernel_factory(signal_kind=None, explicit=False, group=None, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            if explicit:
                signal = T.dist.signal(kind=signal_kind)
                T.dist.barrier(group=group, signal=signal)
            else:
                T.dist.barrier(group=group)

    return main


@tilelang.jit(target="sunmmio")
def repeated_barrier_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            signal = T.dist.signal()
            T.dist.barrier(signal=signal)
            T.dist.barrier(signal=signal)

    return main


@tilelang.jit(target="sunmmio")
def split_barrier_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            signal = T.dist.signal()
            T.dist.barrier_arrive(signal, submit=True)
            T.evaluate(rank_id + 1)
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def put_signal_ring_kernel_factory(signal_kind=None, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            signal = T.dist.signal(kind=signal_kind)
            T.dist.put_signal(signal, (rank_id + 1) % world_size, submit=True)
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def invalid_local_put_signal_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            signal = T.dist.signal()
            T.dist.put_signal(signal, rank_id)
            T.dist.wait_signal(signal)

    return main


def test_barrier_frontend_creates_internal_increment_signal():
    func = barrier_kernel_factory.get_tir(world_size=4)
    names = collect_op_names(func)

    assert names.count("tl.dist_barrier") == 1
    assert names.count("tl.dist_signal_decl") == 1
    assert 'T.dist_signal_decl("sram_flagreg_inc"' in func.script()


def test_lower_barrier_emits_all_peer_arrivals_and_wait():
    func = barrier_kernel_factory.get_tir(world_size=4)
    lowered = lower_collectives(func)
    names = collect_op_names(lowered)
    script = lowered.script()

    assert "tl.dist_barrier" not in names
    assert names.count("tl.dist_submit") == 1
    assert "tl.dist_wait_send" not in names
    assert names.count("tl.dist_signal_put") == 3
    assert names.count("tl.dist_wait_signal") == 1
    assert script.rindex("T.dist_signal_put") < script.index("T.dist_submit()")
    assert script.index("T.dist_submit()") < script.index("T.dist_wait_signal")


def test_barrier_reaches_stable_device_leaves():
    func = barrier_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(
        func,
        capture_passes=(
            "tl.LowerDistCollectives",
            "tl.PlanDistSignals",
            "tl.LowerDistCommunication",
            "tl.InjectDistSync",
        ),
    )

    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 1

    lower_comm = result.pass_snapshot("tl.LowerDistCommunication").mod
    lower_names = collect_op_names(lower_comm)
    assert lower_names.count("tl.dist_signal_put_") == 3
    assert lower_names.count("tl.dist_submit_") == 1
    assert lower_names.count("tl.dist_wait_signal_") == 1
    assert lower_names.count("tl.dist_expect_") == 3
    assert "tl.dist_signal_put" not in lower_names
    assert "tl.dist_wait_signal" not in lower_names

    injected = result.pass_snapshot("tl.InjectDistSync").mod
    injected_names = collect_op_names(injected)
    injected_script = injected.script()
    assert "tl.dist_expect_" not in injected_names
    assert injected_names.count("tl.dist_signal_put_") == 3
    assert injected_names.count("tl.dist_submit_") == 1
    assert injected_names.count("tl.dist_wait_send") == 2
    assert injected_names.count("tl.dist_wait_signal_") == 1
    assert "_generation" not in injected_script
    assert injected_script.count('"sram_flagreg_inc", 0, T.uint8(0)') == 3
    assert "_expect_1[0] =" in injected_script
    assert collect_op_names(result.device_mod).count("tl.dist_wait_signal_") == 1


def test_barrier_accepts_explicit_dram_increment_signal():
    func = barrier_kernel_factory.get_tir(
        explicit=True,
        signal_kind=T.dist.SignalKind.DRAM_FLAGREG_INC,
        world_size=4,
    )
    result = lower_to_device_tir(func, capture_passes="tl.PlanDistSignals")
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]

    assert int(planned.attrs["tl.dist.signal_counts"]["dram_flagreg_inc"]) == 1
    assert '"dram_flagreg_inc"' in result.device_mod.script()


def test_repeated_barrier_accumulates_expected():
    func = repeated_barrier_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.InjectDistSync")
    injected = result.pass_snapshot("tl.InjectDistSync").mod
    names = collect_op_names(injected)

    assert names.count("tl.dist_signal_put_") == 6
    assert names.count("tl.dist_wait_signal_") == 2
    assert injected.script().count("_expect_1[0] =") == 6


def test_barrier_arrive_is_split_from_wait_signal():
    func = split_barrier_kernel_factory.get_tir(world_size=4)
    names = collect_op_names(func)
    assert names.count("tl.dist_barrier_arrive") == 1
    assert names.count("tl.dist_submit") == 1
    assert names.count("tl.dist_wait_signal") == 1

    lowered = lower_collectives(func)
    script = lowered.script()
    assert collect_op_names(lowered).count("tl.dist_signal_put") == 3
    assert collect_op_names(lowered).count("tl.dist_submit") == 1
    assert script.rindex("T.dist_signal_put") < script.index("T.evaluate(rank_id + 1)")
    assert script.index("T.evaluate(rank_id + 1)") < script.index("T.dist_wait_signal")


@pytest.mark.parametrize(
    "signal_kind, expected_kind",
    [
        (None, "sram_flagreg_inc"),
        (T.dist.SignalKind.SRAM_FLAGREG_VALUE, "sram_flagreg_value"),
    ],
)
def test_public_put_signal_infers_ring_expected(signal_kind, expected_kind):
    func = put_signal_ring_kernel_factory.get_tir(signal_kind=signal_kind, world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.InjectDistSync")
    injected = result.pass_snapshot("tl.InjectDistSync").mod
    names = collect_op_names(injected)

    assert names.count("tl.dist_signal_put_") == 1
    assert names.count("tl.dist_wait_signal_") == 1
    assert f'"{expected_kind}"' in injected.script()
    assert "_expect_1[0] =" in injected.script()


def test_put_signal_rejects_local_rank_destination():
    func = invalid_local_put_signal_kernel_factory.get_tir(world_size=4)
    with pytest.raises(tvm.error.InternalError, match="remote peer Rank"):
        lower_to_device_tir(func)


def test_barrier_rejects_world_size_one():
    func = barrier_kernel_factory.get_tir(world_size=1)
    with pytest.raises(tvm.error.InternalError, match="world_size > 1"):
        lower_collectives(func)


def test_barrier_rejects_nonempty_group():
    with pytest.raises(NotImplementedError, match="group is reserved"):
        barrier_kernel_factory.get_tir(group=(0, 1), world_size=4)


def test_barrier_rejects_non_increment_signal():
    with pytest.raises(ValueError, match="INC flagreg"):
        barrier_kernel_factory.get_tir(
            explicit=True,
            signal_kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE,
            world_size=4,
        )
