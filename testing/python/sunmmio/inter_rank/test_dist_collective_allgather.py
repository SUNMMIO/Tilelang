"""Static Rank all-gather frontend and lowering tests."""

from collections import Counter

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import (
    collect_calls,
    collect_dist_put_kinds,
    collect_nodes,
    collect_op_names,
    lower_collectives,
)


def _dst_shape(world_size, axis):
    if axis is None:
        return world_size, 4, 8
    if axis == 0:
        return world_size * 4, 8
    return 4, world_size * 8


@tilelang.jit(target="sunmmio")
def allgather_kernel_factory(axis=None, signal_kind=None, group=None, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.bfloat16)
            dst = T.alloc_shared(_dst_shape(world_size, axis), T.bfloat16)
            signal = T.dist.signal(kind=signal_kind)
            T.dist.all_gather(src, dst, signal=signal, axis=axis, group=group, submit=True)
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def invalid_allgather_shape_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 7), T.bfloat16)
            signal = T.dist.signal()
            T.dist.all_gather(src, dst, signal=signal, submit=True)

    return main


@tilelang.jit(target="sunmmio")
def mismatched_allgather_dtype_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 8), T.float32)
            signal = T.dist.signal()
            T.dist.all_gather(src, dst, signal=signal, submit=True)

    return main


@tilelang.jit(target="sunmmio")
def overlapping_allgather_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            storage = T.alloc_shared((world_size * 4, 8), T.bfloat16)
            signal = T.dist.signal()
            T.dist.all_gather(storage[0:4, 0:8], storage, signal=signal, axis=0, submit=True)

    return main


@tilelang.jit(target="sunmmio")
def dynamic_offset_allgather_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.bfloat16)
            storage = T.alloc_shared((world_size * 8, 8), T.bfloat16)
            signal = T.dist.signal()
            base = rank_id * 4
            T.dist.all_gather(
                src,
                storage[base : base + world_size * 4, 0:8],
                signal=signal,
                axis=0,
                submit=True,
            )

    return main


@tilelang.jit(target="sunmmio")
def constant_offset_allgather_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src_storage = T.alloc_shared((6, 8), T.bfloat16)
            dst_storage = T.alloc_shared((world_size * 4 + 3, 8), T.bfloat16)
            signal = T.dist.signal()
            T.dist.all_gather(
                src_storage[1:5, 0:8],
                dst_storage[2 : 2 + world_size * 4, 0:8],
                signal=signal,
                axis=0,
                submit=True,
            )

    return main


@tilelang.jit(target="sunmmio")
def allgather_scope_combinations_kernel_factory(world_size: int = 1):
    placement = T.placement.replicated()

    @T.prim_func
    def main(
        A: T.MeshTensor((32, 32), placement, T.bfloat16),  # type: ignore
        B: T.MeshTensor((world_size, 32, 32), placement, T.bfloat16),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            src = T.alloc_shared((32, 32), T.bfloat16)
            dst = T.alloc_shared((world_size, 32, 32), T.bfloat16)
            signal0 = T.dist.signal()
            signal1 = T.dist.signal()
            signal2 = T.dist.signal()
            signal3 = T.dist.signal()
            T.dist.all_gather(src, dst, signal=signal0, submit=True)
            T.dist.wait_signal(signal0)
            T.dist.all_gather(src, B, signal=signal1, submit=True)
            T.dist.wait_signal(signal1)
            T.dist.all_gather(A, dst, signal=signal2, submit=True)
            T.dist.wait_signal(signal2)
            T.dist.all_gather(A, B, signal=signal3, submit=True)
            T.dist.wait_signal(signal3)

    return main


@tilelang.jit(target="sunmmio")
def allgather_signal_group_member_kernel_factory(
    signal_kind=None,
    world_size: int = 1,
):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            signals = T.dist.signals(1, kind=signal_kind)
            T.dist.all_gather(src, dst, signal=signals[0], submit=True)
            T.dist.wait_signal(signals[0])

    return main


@tilelang.jit(target="sunmmio")
def allgather_signal_list_kernel_factory(
    signal_kind=None,
    world_size: int = 1,
):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            signals = T.dist.signals(world_size, kind=signal_kind)
            completion = T.dist.all_gather(src, dst, signals=signals, submit=True)
            while T.dist.has_pending(completion):
                source = T.dist.wait_any(completion)
                dst[source, 0, 0] = dst[source, 0, 0]

    return main


@tilelang.jit(target="sunmmio")
def repeated_allgather_signal_list_kernel_factory(drain_first=False, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            signals = T.dist.signals(world_size, kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE)
            first = T.dist.all_gather(src, dst, signals=signals, submit=True)
            if drain_first:
                while T.dist.has_pending(first):
                    T.dist.wait_any(first)
            second = T.dist.all_gather(src, dst, signals=signals, submit=True)
            if drain_first:
                T.dist.wait_all(second)
            else:
                T.dist.wait_all(signals)

    return main


@tilelang.jit(target="sunmmio")
def allgather_dram_signal_list_kernel_factory(
    signal_kind=None,
    world_size: int = 1,
):
    placement = T.placement.replicated()

    @T.prim_func
    def main(
        src: T.MeshTensor((4, 8), placement, T.bfloat16),  # type: ignore
        dst: T.MeshTensor((world_size, 4, 8), placement, T.bfloat16),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            signals = T.dist.signals(world_size, kind=signal_kind)
            completion = T.dist.all_gather(src, dst, signals=signals, submit=True)
            T.dist.wait_all(completion)

    return main


@pytest.mark.parametrize("axis", [None, 0, -1])
def test_allgather_frontend_emits_one_high_level_tileop(axis):
    func = allgather_kernel_factory.get_tir(axis=axis, world_size=4)
    names = collect_op_names(func)

    assert names.count("tl.tileop.dist_allgather") == 1
    assert "tl.tileop.dist_put" not in names
    assert names.count("tl.dist_wait_signal") == 1
    assert "tl.dist_wait_send" not in names
    assert names.count("tl.dist_submit") == 1


@pytest.mark.parametrize(
    "axis, slot",
    [
        (None, "dst[rank_id, 0, 0]"),
        (0, "dst[rank_id * 4, 0]"),
        (-1, "dst[0, rank_id * 8]"),
    ],
)
def test_lower_dist_collectives_expands_direct_allgather(axis, slot):
    func = allgather_kernel_factory.get_tir(axis=axis, world_size=4)
    lowered = lower_collectives(func)
    names = collect_op_names(lowered)
    script = lowered.script()

    assert "tl.tileop.dist_allgather" not in names
    assert names.count("tl.tileop.copy") == 1
    assert names.count("tl.tileop.dist_put") == 3
    assert names.count("tl.dist_wait_signal") == 1
    assert names.count("tl.dist_submit") == 1
    assert "tl.dist_wait_send" not in names
    assert script.count(slot) == 4
    for offset in range(1, 4):
        assert f"(rank_id + {offset}) % 4" in script

    lowered_again = tilelang.transform.LowerDistCollectives()(lowered)
    assert tvm.ir.structural_equal(lowered_again, lowered)


def test_allgather_collective_does_not_insert_receiver_wait():
    func = allgather_kernel_factory.get_tir(world_size=4)
    before = collect_op_names(func).count("tl.dist_wait_signal")
    lowered = lower_collectives(func)
    assert collect_op_names(lowered).count("tl.dist_wait_signal") == before


@pytest.mark.parametrize("axis", [None, 0, -1])
def test_allgather_all_axes_reach_device_tir(axis):
    func = allgather_kernel_factory.get_tir(axis=axis, world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistCollectives")

    collective_names = collect_op_names(result.pass_snapshot("tl.LowerDistCollectives").mod)
    device_names = collect_op_names(result.device_mod)
    assert "tl.tileop.dist_allgather" not in collective_names
    assert device_names.count("tl.dist_put_") == 3
    assert device_names.count("tl.dist_wait_signal_") == 1
    assert device_names.count("tl.dist_wait_send") == 2
    assert device_names.count("tl.dist_submit_") == 1


def test_allgather_supports_all_p2p_scope_combinations():
    func = allgather_scope_combinations_kernel_factory.get_tir(world_size=2)
    result = lower_to_device_tir(func, capture_passes="tl.PlanDistSignals")
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]

    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 2
    assert int(planned.attrs["tl.dist.signal_counts"]["dram_flagreg_inc"]) == 2
    assert Counter(collect_dist_put_kinds(result.device_mod)) == {
        "sram_flagreg_inc": 2,
        "dram_flagreg_inc": 2,
    }


def test_allgather_accepts_explicit_inc_signal_group_member():
    func = allgather_signal_group_member_kernel_factory.get_tir(
        signal_kind=T.dist.SignalKind.SRAM_FLAGREG_INC,
        world_size=4,
    )
    result = lower_to_device_tir(func, capture_passes="tl.PlanDistSignals")
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]

    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 1
    assert 'T.dist_signal_group("sram_flagreg_inc", 0, 1, "auto")' in planned.script()
    assert collect_op_names(result.device_mod).count("tl.dist_put_") == 3


def test_allgather_auto_signal_group_member_plans_whole_group_as_increment():
    func = allgather_signal_group_member_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.PlanDistSignals")
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]

    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 1
    assert 'T.dist_signal_group("sram_flagreg_inc", 0, 1, "auto")' in planned.script()


def test_allgather_signal_list_returns_per_source_completion():
    func = allgather_signal_list_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(
        func,
        capture_passes=(
            "tl.LowerDistCollectives",
            "tl.PlanDistSignals",
            "tl.LowerDistCommunication",
            "tl.LowerDistSync",
        ),
    )

    collective = result.pass_snapshot("tl.LowerDistCollectives").mod
    assert collect_op_names(collective).count("tl.dist_completion") == 1
    assert '"all_gather"' in collective.script()

    lower_comm = result.pass_snapshot("tl.LowerDistCommunication").mod
    assert collect_op_names(lower_comm).count("tl.dist_wait_any_") == 1
    assert "tl.dist_completion_init_" in collect_op_names(lower_comm)

    injected = result.pass_snapshot("tl.LowerDistSync").mod
    injected_script = injected.script()
    assert "tl.dist_completion_init_" not in collect_op_names(injected)
    assert collect_op_names(injected).count("tl.dist_wait_any_") == 1
    assert "rank_id != 0" in injected_script
    assert "rank_id != 3" in injected_script


def test_allgather_signal_list_supports_cumulative_group_wait_or_progress_drain():
    aggregate = lower_to_device_tir(repeated_allgather_signal_list_kernel_factory.get_tir(world_size=4))
    assert collect_op_names(aggregate.device_mod).count("tl.dist_wait_signal_") == 4
    result = lower_to_device_tir(repeated_allgather_signal_list_kernel_factory.get_tir(drain_first=True, world_size=4))
    assert collect_op_names(result.device_mod).count("tl.dist_wait_any_") == 2


@pytest.mark.parametrize(
    "signal_kind",
    [
        T.dist.SignalKind.SRAM_FLAGREG_INC,
        T.dist.SignalKind.SRAM_FLAGREG_VALUE,
        T.dist.SignalKind.SRAM_MEMORY,
    ],
)
def test_allgather_signal_list_supports_all_sram_signal_kinds(signal_kind):
    func = allgather_signal_list_kernel_factory.get_tir(
        signal_kind=signal_kind,
        world_size=2,
    )
    result = lower_to_device_tir(func, capture_passes="tl.PlanDistSignals")
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]

    assert int(planned.attrs["tl.dist.signal_counts"][signal_kind.value]) == 2
    assert collect_op_names(result.device_mod).count("tl.dist_wait_any_") == 1


@pytest.mark.parametrize(
    "signal_kind",
    [
        T.dist.SignalKind.DRAM_FLAGREG_INC,
        T.dist.SignalKind.DRAM_FLAGREG_VALUE,
        T.dist.SignalKind.DRAM_MEMORY,
    ],
)
def test_allgather_signal_list_supports_all_dram_signal_kinds(signal_kind):
    func = allgather_dram_signal_list_kernel_factory.get_tir(
        signal_kind=signal_kind,
        world_size=2,
    )
    result = lower_to_device_tir(func, capture_passes="tl.PlanDistSignals")
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]

    assert int(planned.attrs["tl.dist.signal_counts"][signal_kind.value]) == 2
    assert collect_op_names(result.device_mod).count("tl.dist_wait_any_") == 1


def test_allgather_world_size_one_is_rejected_by_collective_pass():
    func = allgather_kernel_factory.get_tir(world_size=1)
    with pytest.raises(tvm.error.InternalError, match="world_size > 1"):
        lower_collectives(func)


def test_allgather_rejects_nonempty_group():
    with pytest.raises(NotImplementedError, match="group is reserved"):
        allgather_kernel_factory.get_tir(group=(0, 1), world_size=4)


def test_allgather_rejects_unsupported_axis():
    with pytest.raises(ValueError, match="only supports axis=None, axis=0, or axis=-1"):
        allgather_kernel_factory.get_tir(axis=1, world_size=4)


def test_allgather_rejects_value_signal():
    with pytest.raises(ValueError, match="automatic or INC flagreg signal"):
        allgather_kernel_factory.get_tir(
            signal_kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE,
            world_size=4,
        )


def test_allgather_rejects_invalid_destination_shape():
    with pytest.raises((AssertionError, ValueError), match="shape|extents"):
        invalid_allgather_shape_kernel_factory.get_tir(world_size=4)


def test_allgather_rejects_mismatched_dtype():
    with pytest.raises(TypeError, match="dtypes must match"):
        mismatched_allgather_dtype_kernel_factory.get_tir(world_size=4)


def test_allgather_rejects_overlapping_storage():
    with pytest.raises(ValueError, match="overlapping source and destination"):
        overlapping_allgather_kernel_factory.get_tir(world_size=4)


def test_allgather_rejects_endpoint_dependent_region_offset():
    func = dynamic_offset_allgather_kernel_factory.get_tir(world_size=4)
    with pytest.raises(tvm.error.InternalError, match="region offsets must be compile-time constants"):
        lower_collectives(func)


def test_allgather_accepts_constant_nonzero_region_offsets():
    func = constant_offset_allgather_kernel_factory.get_tir(world_size=4)
    lowered = lower_collectives(func)
    assert collect_op_names(lowered).count("tl.tileop.dist_put") == 3


@tilelang.jit(target="sunmmio")
def loop_offset_allgather_kernel_factory(axis=None, error=None, wait_mode="all", world_size: int = 1):
    rounds, block, width = 3, 32, 128
    source_shape = ((rounds + (1 if error == "unaligned" else 0)) * block, width)
    if axis is None:
        destination_shape = (world_size, rounds * block, width)
    elif axis == 0:
        destination_shape = (world_size * rounds * block, width)
    else:
        destination_shape = (rounds * block, world_size * width)

    @T.prim_func
    def main(rank_id: T.dist.RankId, runtime: T.int32):
        with T.Kernel() as core:
            source = T.alloc_shared(source_shape, T.bfloat16)
            destination = T.alloc_shared(destination_shape, T.bfloat16)
            offsets = T.alloc_shared((rounds,), T.int32)
            ready = T.dist.signals(world_size)
            for b in T.serial(runtime if error == "dynamic_loop" else rounds):
                begin = b * block
                if error == "rank":
                    begin = b * block + rank_id * block
                elif error == "core":
                    begin = b * block + core * block
                elif error == "data":
                    begin = offsets[b]
                elif error == "nonaffine":
                    begin = b * b * block
                elif error == "unaligned":
                    begin = b * block + 1
                elif error == "bounds":
                    begin = (b + 1) * block
                dst_begin = 0 if error == "reuse" else b * block
                if axis is None:
                    done = T.dist.all_gather(
                        source[begin : begin + block, :], destination[:, dst_begin : dst_begin + block, :], signals=ready, submit=True
                    )
                elif axis == 0:
                    done = T.dist.all_gather(
                        source[begin : begin + block, :],
                        destination[dst_begin * world_size : (dst_begin + block) * world_size, :],
                        axis=0,
                        signals=ready,
                        submit=True,
                    )
                else:
                    done = T.dist.all_gather(
                        source[begin : begin + block, :], destination[dst_begin : dst_begin + block, :], axis=-1, signals=ready, submit=True
                    )
                if wait_mode in ("all", "no_send_wait"):
                    T.dist.wait_all(done)
                elif wait_mode == "any":
                    while T.dist.has_pending(done):
                        T.dist.wait_any(done)
                elif wait_mode == "conditional":  # noqa: SIM102 - retain a runtime TIR branch
                    if runtime > 0:
                        T.dist.wait_all(done)
                if wait_mode != "no_send_wait":
                    T.dist.wait()

    return main


@pytest.mark.parametrize("axis", [None, 0, -1])
@pytest.mark.parametrize("wait_mode", ["all", "any"])
def test_allgather_loop_offsets_preserve_source_and_destination_slots(axis, wait_mode):
    func = loop_offset_allgather_kernel_factory.get_tir(axis=axis, wait_mode=wait_mode, world_size=2)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistCollectives")
    expanded = result.pass_snapshot("tl.LowerDistCollectives").mod
    puts = collect_calls(expanded, "tl.tileop.dist_put")
    assert len(puts) == 1
    loop = collect_nodes(expanded, tvm.tir.For)[0]
    rank = func.params[int(func.attrs["tl.dist.rank_id_param_index"])]
    # Resolve frontend Let aliases before checking the generated addresses.
    bindings = {node.var: node.value for node in collect_nodes(expanded, tvm.tir.LetStmt) if str(node.var.dtype) == "int32"}
    analyzer = tvm.arith.Analyzer()
    for iteration in range(3):
        for source_rank in range(2):
            values = {loop.loop_var: tvm.tir.IntImm("int32", iteration), rank: tvm.tir.IntImm("int32", source_rank)}
            for var, value in bindings.items():
                values[var] = analyzer.simplify(tvm.tir.stmt_functor.substitute(value, values))
            src = [int(analyzer.simplify(tvm.tir.stmt_functor.substitute(x, values))) for x in puts[0].args[0].args[0].indices]
            dst = [int(analyzer.simplify(tvm.tir.stmt_functor.substitute(x, values))) for x in puts[0].args[1].args[0].indices]
            assert src == [iteration * 32, 0]
            expected = (
                [source_rank, iteration * 32, 0]
                if axis is None
                else [iteration * 64 + source_rank * 32, 0]
                if axis == 0
                else [iteration * 32, source_rank * 128]
            )
            assert dst == expected
    assert collect_op_names(result.device_mod).count("tl.dist_wait_any_") == 1


@pytest.mark.parametrize(
    "error,message",
    [
        ("rank", "Rank/core-uniform"),
        ("core", "Rank/core-uniform"),
        ("data", "Rank/core-uniform"),
        ("nonaffine", "must be affine"),
        ("unaligned", "align to per-peer"),
        ("bounds", "in bounds"),
        ("reuse", "independent destination slots"),
        ("dynamic_loop", "statically bounded serial loops"),
    ],
)
def test_allgather_loop_offsets_reject_unsupported_addresses(error, message):
    func = loop_offset_allgather_kernel_factory.get_tir(error=error, world_size=2)
    with pytest.raises(tvm.error.InternalError, match=message):
        lower_collectives(func)


@pytest.mark.parametrize("wait_mode", ["no_wait", "conditional", "no_send_wait"])
def test_allgather_loop_offsets_require_iteration_completion(wait_mode):
    func = loop_offset_allgather_kernel_factory.get_tir(wait_mode=wait_mode, world_size=2)
    with pytest.raises(tvm.error.InternalError, match="same iteration"):
        lower_collectives(func)


def test_allgather_loop_offsets_reject_conditional_block_wait():
    func = loop_offset_allgather_kernel_factory.get_tir(world_size=2)
    rank = func.params[int(func.attrs["tl.dist.rank_id_param_index"])]

    def guard_wait(node):
        call = node.value
        if isinstance(call, tvm.tir.Call) and call.op.name == "tl.dist_wait_completion_all":
            return tvm.tir.BlockRealize([], rank == 0, tvm.tir.Block([], [], [], "guard_wait", node))
        return None

    body = tvm.tir.stmt_functor.ir_transform(func.body, None, guard_wait, ["tir.Evaluate"])
    with pytest.raises(tvm.error.InternalError, match="same iteration"):
        lower_collectives(func.with_body(body))
