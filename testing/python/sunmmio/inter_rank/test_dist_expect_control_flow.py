"""Expected-value inference, auto/manual boundaries, and phase ordering in branches and loops."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import (
    collect_calls,
)


@tilelang.jit(target="sunmmio")
def branch_exchange_kernel_factory(
    sender_rank=0,
    send_count=1,
    resource="single",
    world_size: int = 1,
):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            signals = T.dist.signals(2)
            if rank_id == 0:
                if sender_rank == 0:
                    T.dist.put(src, dst, 1, signal=signal if resource == "single" else signals[1], submit=True)
                    if send_count == 2:
                        T.dist.put(src, dst, 1, signal=signal if resource == "single" else signals[1], submit=True)
                else:
                    if resource == "all":
                        T.dist.wait_all(signals)
                    else:
                        T.dist.wait_signal(signal if resource == "single" else signals[1])
            else:
                if sender_rank == 1:
                    T.dist.put(src, dst, 0, signal=signal if resource == "single" else signals[1], submit=True)
                    if send_count == 2:
                        T.dist.put(src, dst, 0, signal=signal if resource == "single" else signals[1], submit=True)
                else:
                    if resource == "all":
                        T.dist.wait_all(signals)
                    else:
                        T.dist.wait_signal(signal if resource == "single" else signals[1])

    return main


@tilelang.jit(target="sunmmio")
def independent_phase_kernel_factory(group_members=False, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            data_ready = T.dist.signal()
            round_barrier = T.dist.signal()
            signals = T.dist.signals(2)
            for step in T.serial(2):
                T.dist.put(src, dst, (rank_id + 1) % world_size, signal=signals[0] if group_members else data_ready, submit=True)
                if step == 0:
                    T.dist.wait_signal(signals[0] if group_members else data_ready)
                else:
                    T.dist.wait_signal(signals[0] if group_members else data_ready)
                    # Use local data and retain the compute-only loop in the else branch.
                    for i in T.serial(32):
                        src[i] = dst[i]
                    if group_members:
                        T.dist.put(src, dst, (rank_id + 1) % world_size, signal=signals[1], submit=True)
                        T.dist.wait_signal(signals[1])
                    else:
                        T.dist.barrier(signal=round_barrier)

    return main


@tilelang.jit(target="sunmmio")
def unsafe_phase_kernel_factory(
    pattern="reuse",
    send_in_else=True,
    world_size: int = 1,
):
    @T.prim_func
    def main(rank_id: T.dist.RankId, choice: T.int32):
        with T.Kernel() as core_id:
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            signals = T.dist.signals(2)
            if pattern == "runtime_condition":
                condition = choice != 0
            else:
                condition = (core_id % T.mesh_ncols() == 0) != send_in_else
            if condition:
                if not send_in_else:
                    T.dist.wait_signal(signal)
                    T.dist.put(src, dst, (rank_id + 1) % world_size, signal=signal, submit=True)
                else:
                    T.copy(src, dst)
            else:
                if send_in_else:
                    if pattern == "completion":
                        completed = T.dist.completion(signals)
                        T.dist.wait_all(completed)
                    elif pattern == "nested":
                        if rank_id == 0:
                            T.dist.wait_signal(signal)
                    elif pattern == "early_return":
                        if rank_id == 0:
                            T.evaluate(tvm.tir.ret(1))
                    elif pattern == "group_all":
                        T.dist.wait_all(signals)
                    elif pattern == "group_member":
                        T.dist.wait_signal(signals[1])
                    elif pattern != "runtime_condition":
                        if pattern == "reuse":
                            T.dist.put(src, dst, (rank_id + 1) % world_size, signal=signal, submit=True)
                        T.dist.wait_signal(signal)
                    T.dist.put(
                        src,
                        dst,
                        (rank_id + 1) % world_size,
                        signal=signals[1] if pattern in ("group_all", "group_member", "completion") else signal,
                        submit=True,
                    )
                    if pattern == "reuse":
                        T.dist.wait_signal(signal)
                else:
                    T.copy(src, dst)

    return main


@tilelang.jit(target="sunmmio")
def sender_local_condition_kernel_factory(
    structure="if",
    op="put",
    manual=False,
    world_size: int = 1,
):
    @T.macro
    def send(src, dst, signal, signals, rank_id):
        if op == "put_signal":
            T.dist.put_signal(signal, (rank_id + 1) % world_size)
        else:
            T.dist.put(src, dst, (rank_id + 1) % world_size, signal=signals[0] if manual else signal)

    @T.prim_func
    def main(rank_id: T.dist.RankId, enabled: T.int32, recv_count: T.int32):
        with T.Kernel() as core_id:
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            signals = T.dist.signals(1, expect="manual" if manual else None)
            if structure == "if":
                if enabled > 0:
                    send(src, dst, signal, signals, rank_id)
            elif structure == "if_else":
                if enabled > 0:
                    send(src, dst, signal, signals, rank_id)
                else:
                    T.copy(src, dst)
            elif structure == "nested":
                if core_id % T.mesh_ncols() == 0:  # noqa: SIM102 - Exercise nested device branches.
                    if enabled > 0:
                        send(src, dst, signal, signals, rank_id)
            else:
                if core_id % T.mesh_ncols() == 0:
                    if enabled > 0:
                        send(src, dst, signal, signals, rank_id)
                    else:
                        T.copy(src, dst)
            T.dist.submit()
            if manual:
                T.dist.wait_signal(signals[0], expected_delta=recv_count)
            else:
                T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def conditional_two_phase_signal_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel() as core_id:
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            if core_id % T.mesh_ncols() == 0:
                T.dist.put(
                    src,
                    dst,
                    (rank_id + 1) % world_size,
                    signal=signal,
                    submit=True,
                )
                T.dist.wait_signal(signal)
                T.dist.put(
                    src,
                    dst,
                    (rank_id + 1) % world_size,
                    signal=signal,
                    submit=True,
                )
                T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def conditional_while_put_kernel_factory(iterations: int, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel() as core_id:
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            step = T.alloc_var(T.int32, init=0)
            if core_id % T.mesh_ncols() == 0:
                while step < iterations:
                    T.dist.put(
                        src,
                        dst,
                        (rank_id + 1) % world_size,
                        signal=signal,
                        submit=True,
                    )
                    step += 1
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def nested_conditional_wait_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel() as core_id:
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            if core_id % T.mesh_ncols() == 0:
                T.dist.put(src, dst, (rank_id + 1) % world_size, signal=signal)
                T.dist.submit()
                if rank_id == 0:
                    T.dist.wait_signal(signal)
                T.dist.put(src, dst, (rank_id + 1) % world_size, signal=signal)
                T.dist.submit()

    return main


@tilelang.jit(target="sunmmio")
def loop_expect_kernel_factory(
    bound="rank",
    form="for",
    expect_mode=None,
    group_member=False,
    signal_only=False,
    world_size: int = 1,
):
    @T.prim_func
    def main(rank_id: T.dist.RankId, rounds: T.int32, recv_count: T.int32):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal(expect=expect_mode)
            signals = T.dist.signals(2, expect=expect_mode)
            local_count = T.alloc_var(T.int32, init=rounds)
            step = T.alloc_var(T.int32, init=0)
            if bound == "rank":
                limit = rank_id + 1
            elif bound == "param":
                limit = rounds
            elif bound == "buffer":
                limit = local_count
            else:
                limit = 2

            if form == "while":
                while step < limit:
                    T.dist.put(src, dst, (rank_id + 1) % world_size, signal=signals[0] if group_member else signal, submit=True)
                    step += 1
            elif form == "nested":
                for outer in T.serial(limit):
                    for _ in T.serial(outer + 1):
                        T.dist.put(src, dst, (rank_id + 1) % world_size, signal=signals[0] if group_member else signal, submit=True)
            else:
                for _ in T.serial(limit):
                    if form == "exit":  # noqa: SIM102 - Keep static selection separate from the device condition.
                        if rank_id == 0:
                            T.evaluate(tvm.tir.ret(1))
                    if signal_only:
                        T.dist.put_signal(signal, (rank_id + 1) % world_size, submit=True)
                    else:
                        T.dist.put(src, dst, (rank_id + 1) % world_size, signal=signals[0] if group_member else signal, submit=True)

            if expect_mode == "manual":
                T.dist.wait_signal(signals[0] if group_member else signal, expected_delta=recv_count)
            else:
                T.dist.wait_signal(signals[0] if group_member else signal)

    return main


@tilelang.jit(target="sunmmio")
def collective_loop_kernel_factory(variable_counts=False, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, rounds: T.int32):
        with T.Kernel():
            src = T.alloc_shared((world_size, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, world_size, 8), T.bfloat16)
            variable_dst = T.alloc_shared((world_size, 8), T.bfloat16)
            send_counts = T.alloc_shared((world_size,), T.int32)
            recv_counts = T.alloc_shared((world_size,), T.int32)
            signals = T.dist.signals(world_size)
            for _ in T.serial(rounds):
                if variable_counts:
                    completion = T.dist.all_to_allv(
                        src,
                        variable_dst,
                        send_counts=send_counts,
                        recv_counts=recv_counts,
                        signals=signals,
                        submit=True,
                    )
                else:
                    completion = T.dist.all_gather(src, dst, signals=signals, submit=True)
                T.dist.wait_all(completion)

    return main


@pytest.mark.parametrize("sender_rank", [0, 1])
@pytest.mark.parametrize("send_count", [1, 2])
@pytest.mark.parametrize("resource", ["single", "member", "all"])
def test_one_way_if_else_expectation_only_counts_receiver(sender_rank, send_count, resource):
    result = lower_to_device_tir(
        branch_exchange_kernel_factory.get_tir(
            sender_rank=sender_rank,
            send_count=send_count,
            resource=resource,
            world_size=2,
        ),
        capture_passes="tl.LowerDistCommunication",
    )
    mod = result.pass_snapshot("tl.LowerDistCommunication").mod
    markers = collect_calls(mod, "tl.dist_expect_")
    assert len(markers) == 1
    delta = markers[0].args[1]
    rank = next(param for func in mod.functions.values() for param in func.params if param.name == "rank_id")
    analyzer = tvm.arith.Analyzer()
    for value in (0, 1):
        resolved = analyzer.simplify(
            tvm.tir.stmt_functor.substitute(
                delta,
                {
                    rank: tvm.tir.const(value, "int32"),
                },
            )
        )
        assert int(resolved) == (send_count if value != sender_rank else 0)
    script = mod.script()
    assert script.index("T.dist_expect_(") < script.index("if rank_id")
    assert len(collect_calls(result.device_mod, "tl.dist_put_")) == send_count
    assert collect_calls(result.device_mod, "tl.dist_wait_signal_")


@pytest.mark.parametrize("group_members", [False, True])
def test_independent_signal_wait_then_send_keeps_each_expectation(group_members):
    result = lower_to_device_tir(
        independent_phase_kernel_factory.get_tir(
            group_members=group_members,
            world_size=2,
        ),
        capture_passes="tl.LowerDistCommunication",
    )
    mod = result.pass_snapshot("tl.LowerDistCommunication").mod
    markers = collect_calls(mod, "tl.dist_expect_")
    assert len(markers) == 2
    # The first receive target advances by one; the later independent send does not affect it.
    assert isinstance(markers[0].args[1], tvm.tir.IntImm)
    assert int(markers[0].args[1]) == 1
    first, second = (marker.args[0] for marker in markers)
    if group_members:
        assert first.buffer.same_as(second.buffer)
        assert int(first.indices[0]) == 0 and int(second.indices[0]) == 1
    else:
        assert not first.buffer.same_as(second.buffer)
    assert len(collect_calls(result.device_mod, "tl.dist_wait_signal_")) >= 2


@pytest.mark.parametrize(
    "pattern, send_in_else",
    [
        ("reuse", True),
        ("wait_first", True),
        ("wait_first", False),
        ("group_all", True),
        ("group_member", True),
        ("nested", True),
        ("completion", True),
        ("early_return", True),
        ("runtime_condition", True),
    ],
)
def test_unproven_if_else_expectation_is_rejected(pattern, send_in_else):
    with pytest.raises(tvm.error.InternalError, match="(Cannot.*T.dist|sent after a wait)"):
        lower_to_device_tir(
            unsafe_phase_kernel_factory.get_tir(
                pattern=pattern,
                send_in_else=send_in_else,
                world_size=2,
            )
        )


@pytest.mark.parametrize("structure", ["if", "if_else", "nested", "nested_else"])
@pytest.mark.parametrize("op", ["put", "put_signal"])
def test_auto_expect_rejects_sender_local_conditions_on_every_path(structure, op):
    with pytest.raises(tvm.error.InternalError, match="sender-local condition.*expect='manual'"):
        lower_to_device_tir(
            sender_local_condition_kernel_factory.get_tir(
                structure=structure,
                op=op,
                world_size=4,
            )
        )


@pytest.mark.parametrize("structure", ["if", "nested"])
def test_manual_expect_uses_receiver_delta_for_conditional_sends(structure):
    result = lower_to_device_tir(
        sender_local_condition_kernel_factory.get_tir(
            structure=structure,
            manual=True,
            world_size=4,
        ),
        capture_passes="tl.LowerDistCommunication",
    )
    markers = collect_calls(result.pass_snapshot("tl.LowerDistCommunication").mod, "tl.dist_expect_")
    assert len(markers) == 1
    assert isinstance(markers[0].args[1], tvm.tir.Var)
    assert markers[0].args[1].name == "recv_count"
    assert collect_calls(result.device_mod, "tl.dist_put_")


def test_conditional_signal_expected_does_not_cross_receiver_wait():
    result = lower_to_device_tir(
        conditional_two_phase_signal_kernel_factory.get_tir(world_size=4),
        capture_passes="tl.LowerDistCommunication",
    )
    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod.script()
    first_put = lowered.index("T.dist_put_(")
    first_wait = lowered.index("T.dist_wait_signal_(")
    second_put = lowered.index("T.dist_put_(", first_put + 1)
    second_wait = lowered.index("T.dist_wait_signal_(", first_wait + 1)
    first_delta = lowered.index("T.dist_expect_(")
    second_delta = lowered.index("T.dist_expect_(", first_delta + 1)
    assert first_delta < first_put < first_wait < second_delta < second_put < second_wait
    assert lowered.count("T.dist_expect_(") == 2


def test_nested_conditional_wait_requires_explicit_phase_split():
    func = nested_conditional_wait_kernel_factory.get_tir(world_size=4)
    with pytest.raises(tvm.error.InternalError, match="split the condition into phases"):
        lower_to_device_tir(func)


@pytest.mark.parametrize("iterations", [0, 2])
def test_conditional_while_rejects_hoisting_auto_expectation(iterations):
    func = conditional_while_put_kernel_factory.get_tir(iterations, world_size=4)
    with pytest.raises(tvm.error.InternalError, match="cannot prove matching.*loop"):
        lower_to_device_tir(func)


@pytest.mark.parametrize(
    "bound, form",
    [
        ("rank", "for"),
        ("param", "for"),
        ("buffer", "for"),
        ("rank", "while"),
        ("param", "while"),
        ("static", "while"),
        ("param", "nested"),
        ("static", "exit"),
    ],
)
def test_auto_rejects_unproven_loop_iterations(bound, form):
    with pytest.raises(tvm.error.InternalError, match="cannot prove matching.*loop"):
        lower_to_device_tir(
            loop_expect_kernel_factory.get_tir(
                bound=bound,
                form=form,
                world_size=4,
            )
        )


@pytest.mark.parametrize(
    "group_member, signal_only, expect_mode",
    [
        (True, False, None),
        (False, True, None),
        (False, False, "auto"),
    ],
)
def test_auto_loop_check_covers_members_and_signal_only_sends(
    group_member,
    signal_only,
    expect_mode,
):
    with pytest.raises(tvm.error.InternalError, match="use expect='manual'"):
        lower_to_device_tir(
            loop_expect_kernel_factory.get_tir(
                group_member=group_member,
                signal_only=signal_only,
                expect_mode=expect_mode,
                world_size=4,
            )
        )


@pytest.mark.parametrize("form", ["for", "nested"])
def test_auto_keeps_proven_uniform_loops(form):
    result = lower_to_device_tir(
        loop_expect_kernel_factory.get_tir(
            bound="static",
            form=form,
            world_size=4,
        ),
        capture_passes="tl.LowerDistCommunication",
    )
    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod
    assert len(collect_calls(lowered, "tl.dist_expect_")) == 1
    assert "for " in lowered.script()


@pytest.mark.parametrize(
    "bound, form",
    [
        ("rank", "for"),
        ("param", "for"),
        ("buffer", "for"),
        ("rank", "while"),
        ("param", "while"),
        ("param", "nested"),
    ],
)
def test_manual_dynamic_loops_only_use_receiver_supplied_delta(bound, form):
    result = lower_to_device_tir(
        loop_expect_kernel_factory.get_tir(
            bound=bound,
            form=form,
            group_member=True,
            expect_mode="manual",
            world_size=4,
        ),
        capture_passes="tl.LowerDistCommunication",
    )
    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod
    markers = collect_calls(lowered, "tl.dist_expect_")
    assert len(markers) == 1
    assert isinstance(markers[0].args[1], tvm.tir.Var)
    assert markers[0].args[1].name == "recv_count"
    assert lowered.script().index("T.dist_put_(") < lowered.script().index("T.dist_expect_(")
    assert collect_calls(result.device_mod, "tl.dist_wait_signal_")


def test_allgather_auto_completion_does_not_bypass_loop_check():
    with pytest.raises(tvm.error.InternalError, match="use expect='manual'"):
        lower_to_device_tir(collective_loop_kernel_factory.get_tir(world_size=4))


def test_alltoallv_protocol_manual_allows_dynamic_loop():
    result = lower_to_device_tir(
        collective_loop_kernel_factory.get_tir(
            variable_counts=True,
            world_size=4,
        ),
        capture_passes="tl.PlanDistSignals",
    )
    declarations = collect_calls(result.pass_snapshot("tl.PlanDistSignals").mod, "tl.dist_signal_group")
    assert len(declarations) == 1
    assert declarations[0].args[3].value == "manual"
    assert collect_calls(result.device_mod, "tl.dist_put_")
