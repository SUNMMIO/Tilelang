"""An aggregate collective signal must belong to a single invocation."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir


@tilelang.jit(target="sunmmio")
def collective_signal_kernel_factory(
    op="all_gather",
    scenario="repeat",
    extra="put",
    extra_first=False,
    member=False,
    kind=None,
    world_size: int = 1,
):
    @T.macro
    def launch(src, dst, gathered, send_counts, recv_counts, signal):
        if op == "all_gather":
            T.dist.all_gather(src, gathered, signal=signal, submit=True)
        elif op == "all_to_all":
            T.dist.all_to_all(src, dst, signal=signal, submit=True)
        elif op == "all_reduce":
            T.dist.all_reduce(src, dst, signal=signal)
        else:
            T.dist.all_to_allv(src, dst, send_counts=send_counts, recv_counts=recv_counts, signal=signal, submit=True)

    @T.macro
    def other_write(src, dst, rank_id, signal):
        if extra == "put":
            T.dist.put(src, dst, (rank_id + 1) % world_size, signal=signal, submit=True)
        elif extra == "routed_put":
            T.dist.routed_put(src, dst, routes=[[0, (rank_id + 1) % world_size, 0]], signal=signal, submit=True)
        elif extra == "rank_routed_put":
            T.dist.routed_put(src, dst, src_rank=0, routes=[[0, (rank_id + 1) % world_size, 0]], signal=signal, submit=True)
        elif extra == "put_signal":
            T.dist.put_signal(signal, (rank_id + 1) % world_size, submit=True)
        elif extra == "barrier_arrive":
            T.dist.barrier_arrive(signal, submit=True)
        else:
            T.dist.barrier(signal=signal)

    @T.prim_func
    def main(rank_id: T.dist.RankId, rounds: T.int32):
        with T.Kernel():
            src = T.alloc_shared((world_size, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 8), T.bfloat16)
            gathered = T.alloc_shared((world_size, world_size, 8), T.bfloat16)
            send_counts = T.alloc_shared((world_size,), T.int32)
            recv_counts = T.alloc_shared((world_size,), T.int32)
            signal = T.dist.signal(kind=kind)
            second = T.dist.signal(kind=kind)
            signals = T.dist.signals(world_size, kind=kind)
            iteration = T.alloc_var(T.int32, init=0)
            if scenario == "serial":
                for _ in T.serial(2):
                    launch(src, dst, gathered, send_counts, recv_counts, signals[0] if member else signal)
                    T.dist.wait_signal(signals[0] if member else signal)
            elif scenario == "unroll":
                for _ in T.unroll(2):
                    launch(src, dst, gathered, send_counts, recv_counts, signals[0] if member else signal)
                    T.dist.wait_signal(signals[0] if member else signal)
            elif scenario == "while":
                while iteration < 2:
                    launch(src, dst, gathered, send_counts, recv_counts, signals[0] if member else signal)
                    T.dist.wait_signal(signals[0] if member else signal)
                    iteration += 1
            elif scenario == "dynamic_for":
                for _ in T.serial(rounds):
                    launch(src, dst, gathered, send_counts, recv_counts, signal)
                    T.dist.wait_signal(signal)
            elif scenario == "unknown_member":
                launch(src, dst, gathered, send_counts, recv_counts, signals[rank_id])
                T.dist.wait_signal(signals[rank_id])
            else:
                if scenario == "mixed" and extra_first:
                    other_write(src, dst, rank_id, signals[0] if member else signal)
                if scenario == "group_overlap" and extra_first:
                    first = T.dist.all_gather(src, gathered, signals=signals, submit=True)
                    T.dist.wait_all(first)
                launch(src, dst, gathered, send_counts, recv_counts, signals[0] if member else signal)
                T.dist.wait_signal(signals[0] if member else signal)
                if scenario == "repeat":
                    launch(src, dst, gathered, send_counts, recv_counts, signals[0] if member else signal)
                elif scenario == "independent":
                    launch(src, dst, gathered, send_counts, recv_counts, signals[1] if member else second)
                    T.dist.wait_signal(signals[1] if member else second)
                elif scenario == "mixed" and not extra_first:
                    other_write(src, dst, rank_id, signals[0] if member else signal)
                elif scenario == "group_overlap" and not extra_first:
                    last = T.dist.all_gather(src, gathered, signals=signals, submit=True)
                    T.dist.wait_all(last)
                T.dist.wait_signal(signals[0] if member else signal)

    return main


@pytest.mark.parametrize("op", ["all_gather", "all_to_all", "all_reduce", "all_to_allv"])
@pytest.mark.parametrize("scenario", ["repeat", "serial", "while"])
def test_aggregate_signal_cannot_be_reused(op, scenario):
    message = (
        "all_reduce inside For/While"
        if op == "all_reduce" and scenario != "repeat"
        else "(dedicated to one collective|exclusive collective signal)"
    )
    with pytest.raises(tvm.error.InternalError, match=message):
        lower_to_device_tir(
            collective_signal_kernel_factory.get_tir(
                op=op,
                scenario=scenario,
                world_size=4,
            )
        )


@pytest.mark.parametrize("kind", [None, T.dist.SignalKind.SRAM_FLAGREG_INC])
@pytest.mark.parametrize("scenario", ["repeat", "unroll"])
def test_aggregate_member_and_two_rank_case_follow_same_rule(kind, scenario):
    with pytest.raises(tvm.error.InternalError, match="dedicated to one collective"):
        lower_to_device_tir(
            collective_signal_kernel_factory.get_tir(
                scenario=scenario,
                member=True,
                kind=kind,
                world_size=2,
            )
        )


@pytest.mark.parametrize(
    "extra",
    [
        "put",
        "routed_put",
        "rank_routed_put",
        "put_signal",
        "barrier_arrive",
        "barrier",
    ],
)
@pytest.mark.parametrize("extra_first", [False, True])
def test_collective_signal_cannot_mix_with_other_updates(extra, extra_first):
    with pytest.raises(tvm.error.InternalError, match="dedicated to one collective"):
        lower_to_device_tir(
            collective_signal_kernel_factory.get_tir(
                scenario="mixed",
                extra=extra,
                extra_first=extra_first,
                world_size=4,
            )
        )


@pytest.mark.parametrize("extra_first", [False, True])
def test_aggregate_member_cannot_overlap_whole_group_collective(extra_first):
    with pytest.raises(tvm.error.InternalError, match="dedicated to one collective"):
        lower_to_device_tir(
            collective_signal_kernel_factory.get_tir(
                scenario="group_overlap",
                member=True,
                extra_first=extra_first,
                world_size=4,
            )
        )


@pytest.mark.parametrize("op", ["all_gather", "all_to_all", "all_reduce", "all_to_allv"])
def test_one_collective_can_expand_multiple_sends_and_repeat_wait(op):
    result = lower_to_device_tir(
        collective_signal_kernel_factory.get_tir(
            op=op,
            scenario="wait_again",
            world_size=4,
        )
    )
    script = result.device_mod.script()
    assert script.count("T.dist_put_(") == 3
    assert script.count("T.dist_wait_signal_(") >= 2


@pytest.mark.parametrize("member", [False, True])
def test_distinct_collective_resources_remain_usable(member):
    result = lower_to_device_tir(
        collective_signal_kernel_factory.get_tir(
            scenario="independent",
            member=member,
            world_size=4,
        )
    )
    assert result.device_mod.script().count("T.dist_put_(") == 6


@pytest.mark.parametrize("scenario", ["dynamic_for", "unknown_member"])
def test_unproven_aggregate_resource_uniqueness_is_rejected(scenario):
    with pytest.raises(tvm.error.InternalError, match="exclusive collective signal"):
        lower_to_device_tir(
            collective_signal_kernel_factory.get_tir(
                op="all_to_allv",
                scenario=scenario,
                world_size=4,
            )
        )
