"""RANK/ROW all_to_allv lowering: fixed-capacity sends, zero-count skips, and recv_counts-based waits."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import (
    collect_op_arg_counts,
    collect_op_names,
    lower_collectives,
)


@tilelang.jit(target="sunmmio")
def alltoallv_rank_kernel_factory(
    wait_mode="all",
    signal_kind=None,
    world_size: int = 1,
):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            send_counts = T.alloc_shared((world_size,), T.int32)
            recv_counts = T.alloc_shared((world_size,), T.int32)
            signals = T.dist.signals(world_size, kind=signal_kind)
            completion = T.dist.all_to_allv(
                src,
                dst,
                send_counts=send_counts,
                recv_counts=recv_counts,
                signals=signals,
                submit=True,
            )
            if wait_mode == "all":
                T.dist.wait_all(completion)
            else:
                while T.dist.has_pending(completion):
                    source = T.dist.wait_any(completion)
                    recv_counts[source] = 0

    return main


@tilelang.jit(target="sunmmio")
def alltoallv_row_kernel_factory(
    wait_mode="all",
    signal_kind=None,
    world_size: int = 1,
):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            rows = T.mesh_nrows()
            src = T.alloc_shared((world_size, rows, 4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, rows, 4, 8), T.bfloat16)
            send_counts = T.alloc_shared((world_size, rows), T.int32)
            recv_counts = T.alloc_shared((world_size, rows), T.int32)
            signals = T.dist.signals(world_size * rows, kind=signal_kind)
            completion = T.dist.all_to_allv(
                src,
                dst,
                send_counts=send_counts,
                recv_counts=recv_counts,
                signals=signals,
                domain=T.dist.CollectiveDomain.ROW,
                submit=True,
            )
            if wait_mode == "all":
                T.dist.wait_all(completion)
            else:
                while T.dist.has_pending(completion):
                    source = T.dist.wait_any(completion)
                    recv_counts[source // rows, source % rows] = 0

    return main


@tilelang.jit(target="sunmmio")
def alltoallv_single_signal_rank_kernel_factory(signal_kind=None, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            send_counts = T.alloc_shared((world_size,), T.int32)
            recv_counts = T.alloc_shared((world_size,), T.int32)
            signal = T.dist.signal(kind=signal_kind)
            T.dist.all_to_allv(
                src,
                dst,
                send_counts=send_counts,
                recv_counts=recv_counts,
                signal=signal,
                submit=True,
            )
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def alltoallv_single_signal_row_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            rows = T.mesh_nrows()
            src = T.alloc_shared((world_size, rows, 4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, rows, 4, 8), T.bfloat16)
            send_counts = T.alloc_shared((world_size, rows), T.int32)
            recv_counts = T.alloc_shared((world_size, rows), T.int32)
            signal = T.dist.signal()
            T.dist.all_to_allv(
                src,
                dst,
                send_counts=send_counts,
                recv_counts=recv_counts,
                signal=signal,
                domain=T.dist.CollectiveDomain.ROW,
                submit=True,
            )
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def alltoallv_single_signal_dram_kernel_factory(world_size: int = 1):
    placement = T.placement.replicated()

    @T.prim_func
    def main(
        src: T.MeshTensor((world_size, 4, 8), placement, T.bfloat16),  # type: ignore
        dst: T.MeshTensor((world_size, 4, 8), placement, T.bfloat16),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            send_counts = T.alloc_shared((world_size,), T.int32)
            recv_counts = T.alloc_shared((world_size,), T.int32)
            signal = T.dist.signal()
            T.dist.all_to_allv(
                src,
                dst,
                send_counts=send_counts,
                recv_counts=recv_counts,
                signal=signal,
                submit=True,
            )
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def alltoallv_inc_group_member_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            send_counts = T.alloc_shared((world_size,), T.int32)
            recv_counts = T.alloc_shared((world_size,), T.int32)
            signals = T.dist.signals(2, kind=T.dist.SignalKind.SRAM_FLAGREG_INC)
            T.dist.all_to_allv(
                src,
                dst,
                send_counts=send_counts,
                recv_counts=recv_counts,
                signal=signals[1],
                submit=True,
            )
            T.dist.wait_signal(signals[1])

    return main


@tilelang.jit(target="sunmmio")
def invalid_alltoallv_count_shape_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            send_counts = T.alloc_shared((world_size, 1), T.int32)
            recv_counts = T.alloc_shared((world_size,), T.int32)
            signals = T.dist.signals(world_size)
            completion = T.dist.all_to_allv(
                src,
                dst,
                send_counts=send_counts,
                recv_counts=recv_counts,
                signals=signals,
                submit=True,
            )
            T.dist.wait_all(completion)

    return main


@tilelang.jit(target="sunmmio")
def repeated_alltoallv_kernel_factory(drain_first="all", world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            send_counts = T.alloc_shared((world_size,), T.int32)
            recv_counts = T.alloc_shared((world_size,), T.int32)
            signals = T.dist.signals(world_size, kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE)
            first = T.dist.all_to_allv(
                src,
                dst,
                send_counts=send_counts,
                recv_counts=recv_counts,
                signals=signals,
                submit=True,
            )
            if drain_first == "all":
                T.dist.wait_all(first)
            elif drain_first == "any":
                while T.dist.has_pending(first):
                    T.dist.wait_any(first)
            elif drain_first == "partial":
                if T.dist.has_pending(first):
                    T.dist.wait_any(first)
            elif drain_first == "branch":
                if rank_id == 0:
                    T.dist.wait_all(first)
            elif drain_first == "group_before":
                if T.dist.has_pending(first):
                    T.dist.wait_any(first)
                T.dist.wait_all(signals)
            second = T.dist.all_to_allv(
                src,
                dst,
                send_counts=send_counts,
                recv_counts=recv_counts,
                signals=signals,
                submit=True,
            )
            if drain_first == "late_any":
                while T.dist.has_pending(first):
                    T.dist.wait_any(first)
            if drain_first in ("skip", "branch"):
                T.dist.wait_all(signals)
            else:
                T.dist.wait_all(second)

    return main


@tilelang.jit(target="sunmmio")
def alltoallv_scope_combinations_kernel_factory(world_size: int = 1):
    placement = T.placement.replicated()

    @T.prim_func
    def main(
        A: T.MeshTensor((world_size, 4, 8), placement, T.bfloat16),  # type: ignore
        B: T.MeshTensor((world_size, 4, 8), placement, T.bfloat16),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            src = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            send_counts = T.alloc_shared((world_size,), T.int32)
            recv_counts = T.alloc_shared((world_size,), T.int32)
            signals0 = T.dist.signals(world_size)
            signals1 = T.dist.signals(world_size)
            signals2 = T.dist.signals(world_size)
            signals3 = T.dist.signals(world_size)
            completion0 = T.dist.all_to_allv(
                src,
                dst,
                send_counts=send_counts,
                recv_counts=recv_counts,
                signals=signals0,
                submit=True,
            )
            T.dist.wait_all(completion0)
            completion1 = T.dist.all_to_allv(
                src,
                B,
                send_counts=send_counts,
                recv_counts=recv_counts,
                signals=signals1,
                submit=True,
            )
            T.dist.wait_all(completion1)
            completion2 = T.dist.all_to_allv(
                A,
                dst,
                send_counts=send_counts,
                recv_counts=recv_counts,
                signals=signals2,
                submit=True,
            )
            T.dist.wait_all(completion2)
            completion3 = T.dist.all_to_allv(
                A,
                B,
                send_counts=send_counts,
                recv_counts=recv_counts,
                signals=signals3,
                submit=True,
            )
            T.dist.wait_all(completion3)

    return main


def test_alltoallv_rank_frontend_builds_completion():
    func = alltoallv_rank_kernel_factory.get_tir(world_size=4)
    names = collect_op_names(func)

    assert names.count("tl.tileop.dist_alltoallv") == 1
    assert names.count("tl.dist_signal_group_decl") == 1
    assert "tl.dist_signal_decl" not in names
    assert names.count("tl.dist_wait_completion_all") == 1
    assert "tl.tileop.dist_put" not in names


def test_alltoallv_rank_frontend_builds_wait_any_loop():
    func = alltoallv_rank_kernel_factory.get_tir(wait_mode="any", world_size=4)
    names = collect_op_names(func)

    assert names.count("tl.dist_completion_has_pending") == 1
    assert names.count("tl.dist_wait_any") == 1
    assert "while" in func.script()


def test_alltoallv_row_frontend_uses_rank_major_signal_group():
    func = alltoallv_row_kernel_factory.get_tir(world_size=2)
    names = collect_op_names(func)

    assert names.count("tl.tileop.dist_alltoallv") == 1
    assert names.count("tl.dist_signal_group_decl") == 1
    assert "tl.dist_signal_decl" not in names
    assert '"row"' in func.script()


def test_alltoallv_single_signal_frontend_uses_regular_wait_signal():
    func = alltoallv_single_signal_rank_kernel_factory.get_tir(world_size=4)
    names = collect_op_names(func)

    assert names.count("tl.tileop.dist_alltoallv") == 1
    assert names.count("tl.dist_signal_decl") == 1
    assert "tl.dist_signal_group_decl" not in names
    assert names.count("tl.dist_wait_signal") == 1
    assert "tl.dist_completion" not in names
    assert "tl.dist_wait_completion_all" not in names


def test_lower_dist_collectives_emits_single_signal_active_routes_and_expect():
    func = alltoallv_single_signal_rank_kernel_factory.get_tir(world_size=4)
    lowered = lower_collectives(func)
    names = collect_op_names(lowered)
    script = lowered.script()

    assert "tl.tileop.dist_alltoallv" not in names
    assert names.count("tl.tileop.dist_put") == 4
    assert names.count("tl.dist_signal_route") == 4
    assert names.count("tl.dist_expect") == 1
    assert "tl.dist_completion" not in names
    assert "recv_counts[0]" in script
    assert "recv_counts[3]" in script
    assert "rank_id" in script


def test_lower_dist_collectives_emits_count_guarded_fixed_rank_routes():
    func = alltoallv_rank_kernel_factory.get_tir(world_size=4)
    lowered = lower_collectives(func)
    names = collect_op_names(lowered)
    script = lowered.script()

    assert "tl.tileop.dist_alltoallv" not in names
    assert names.count("tl.tileop.dist_put") == 4
    assert names.count("tl.dist_signal_group_route") == 4
    assert names.count("tl.dist_completion") == 1
    assert names.count("tl.dist_submit") == 1
    assert "tl.dist_wait_send" not in names
    assert script.count("> 0") >= 4
    src_region_lines = [line for line in script.splitlines() if "T.region(src[" in line]
    assert len(src_region_lines) == 4
    assert all("send_counts" not in line.split("T.dist_signal_group_route", 1)[0] for line in src_region_lines)


def test_lower_dist_collectives_keeps_row_payload_regions_static():
    func = alltoallv_row_kernel_factory.get_tir(world_size=2)
    lowered = lower_collectives(func)
    src_region_lines = [line for line in lowered.script().splitlines() if "T.region(src[" in line]

    assert src_region_lines
    assert all("send_counts" not in line.split("T.dist_signal_group_route", 1)[0] for line in src_region_lines)


def test_alltoallv_auto_group_prefers_increment_flagreg_when_it_fits():
    func = alltoallv_rank_kernel_factory.get_tir(world_size=4)
    mod = lower_collectives(func)
    mod = tilelang.transform.InferSramScope()(mod)
    planned_mod = tilelang.transform.PlanDistSignals()(mod)
    planned = planned_mod["main"]

    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 4
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_value"]) == 0
    script = planned.script()
    assert 'T.dist_signal_group("sram_flagreg_inc", 0, 4, "manual")' in script
    assert "T.dist_signal(" not in script
    planned_again = tilelang.transform.PlanDistSignals()(planned_mod)
    assert tvm.ir.structural_equal(planned_again, planned_mod)


def test_alltoallv_auto_group_spills_whole_group_to_value_flagreg():
    func = alltoallv_rank_kernel_factory.get_tir(world_size=9)
    mod = lower_collectives(func)
    mod = tilelang.transform.InferSramScope()(mod)
    planned = tilelang.transform.PlanDistSignals()(mod)["main"]

    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 0
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_value"]) == 9
    assert 'T.dist_signal_group("sram_flagreg_value", 0, 9, "manual")' in planned.script()


def test_alltoallv_auto_group_falls_back_to_memory():
    func = alltoallv_rank_kernel_factory.get_tir(world_size=33)
    mod = lower_collectives(func)
    mod = tilelang.transform.InferSramScope()(mod)
    mod = tilelang.transform.PlanDistSignals()(mod)
    planned = mod["main"]

    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_value"]) == 0
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_memory"]) == 33
    mod = tilelang.transform.LowerDistRouting()(mod)
    mod = tilelang.transform.LowerDistCommunication()(mod)
    assert collect_op_names(mod).count("tl.dist_put_") == 32
    assert "uint32" in mod.script()


def test_alltoallv_single_signal_uses_inc_and_reaches_regular_wait_leaf():
    func = alltoallv_single_signal_rank_kernel_factory.get_tir(world_size=4)
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
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_value"]) == 0
    assert 'T.dist_signal("sram_flagreg_inc", 0, "manual")' in planned.script()

    lower_comm = result.pass_snapshot("tl.LowerDistCommunication").mod
    lower_names = collect_op_names(lower_comm)
    assert lower_names.count("tl.dist_put_") == 3
    assert lower_names.count("tl.dist_wait_signal_") == 1
    assert lower_names.count("tl.dist_expect_") == 1
    assert "tl.dist_signal_route" not in lower_names
    assert "tl.dist_completion_init_" not in lower_names
    assert "tl.dist_wait_any_" not in lower_names

    injected = result.pass_snapshot("tl.InjectDistSync").mod
    injected_names = collect_op_names(injected)
    injected_script = injected.script()
    assert "tl.dist_expect_" not in injected_names
    assert injected_names.count("tl.dist_wait_signal_") == 1
    assert "recv_counts[0]" in injected_script
    assert "recv_counts[3]" in injected_script
    assert '"sram_flagreg_inc"' in injected_script
    assert "pending_count" not in injected_script


def test_alltoallv_single_signal_row_routes_active_and_reuses_wait_leaf():
    func = alltoallv_single_signal_row_kernel_factory.get_tir(world_size=2)
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
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 1

    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    routed_names = collect_op_names(routed)
    assert routed_names.count("tl.tileop.dist_routed_peer_put") == 16
    assert "dist_route_active" in routed.script()

    lower_names = collect_op_names(result.pass_snapshot("tl.LowerDistCommunication").mod)
    assert lower_names.count("tl.dist_put_") == 16
    assert lower_names.count("tl.dist_wait_signal_") == 1
    assert "tl.dist_wait_any_" not in lower_names
    injected_script = result.pass_snapshot("tl.InjectDistSync").mod.script()
    assert "recv_counts[1, 3]" in injected_script
    assert "pending_count" not in injected_script


def test_alltoallv_single_signal_infers_dram_increment_kind():
    func = alltoallv_single_signal_dram_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.PlanDistSignals")
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]

    assert int(planned.attrs["tl.dist.signal_counts"]["dram_flagreg_inc"]) == 1
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 0
    device_script = result.device_mod.script()
    assert '"dram_flagreg_inc"' in device_script
    assert collect_op_names(result.device_mod).count("tl.dist_wait_signal_") == 1


def test_alltoallv_accepts_explicit_inc_signal_group_member():
    func = alltoallv_inc_group_member_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.PlanDistSignals")
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]

    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 2
    device_script = result.device_mod.script()
    assert 'T.dist_wait_signal_("sram_flagreg_inc", 1' in device_script
    assert "tl.dist_wait_any_" not in collect_op_names(result.device_mod)


def test_alltoallv_rank_reaches_dynamic_completion_leaves():
    func = alltoallv_rank_kernel_factory.get_tir(wait_mode="any", world_size=4)
    result = lower_to_device_tir(
        func,
        capture_passes=(
            "tl.LowerDistCollectives",
            "tl.PlanDistSignals",
            "tl.LowerDistCommunication",
            "tl.InjectDistSync",
        ),
    )

    lower_comm = result.pass_snapshot("tl.LowerDistCommunication").mod
    lower_names = collect_op_names(lower_comm)
    assert lower_names.count("tl.dist_put_") == 3
    assert lower_names.count("tl.dist_completion_init_") == 1
    assert lower_names.count("tl.dist_wait_any_") == 1
    assert collect_op_arg_counts(lower_comm, "tl.dist_wait_any_") == [6]
    assert "tl.dist_completion_has_pending" not in lower_names

    injected = result.pass_snapshot("tl.InjectDistSync").mod
    injected_names = collect_op_names(injected)
    injected_script = injected.script()
    assert "tl.dist_completion_init_" not in injected_names
    assert injected_names.count("tl.dist_wait_any_") == 1
    assert collect_op_arg_counts(injected, "tl.dist_wait_any_") == [6]
    assert collect_op_arg_counts(result.device_mod, "tl.dist_wait_any_") == [6]
    assert "pending_count" in injected_script
    assert "sram_flagreg_inc" in injected_script
    assert '"sram_flagreg_inc", rank_id' in injected_script
    assert "v_pending_1[source]" in injected_script


def test_alltoallv_wait_all_drains_pending_completion():
    func = alltoallv_rank_kernel_factory.get_tir(wait_mode="all", world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.InjectDistSync")
    injected = result.pass_snapshot("tl.InjectDistSync").mod
    names = collect_op_names(injected)
    script = injected.script()

    assert names.count("tl.dist_wait_any_") == 1
    assert "while 0 <" in script
    assert "dist_wait_any_index" in script
    assert "pending_count" in script


def test_alltoallv_reuses_group_state_with_independent_completions():
    func = repeated_alltoallv_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(
        func,
        capture_passes=("tl.LowerDistCommunication", "tl.InjectDistSync"),
    )

    lower_script = result.pass_snapshot("tl.LowerDistCommunication").mod.script()
    injected = result.pass_snapshot("tl.InjectDistSync").mod
    assert lower_script.count("_generation = T.allocate([16]") == 1
    assert "rank_id * 4" in lower_script
    assert collect_op_names(injected).count("tl.dist_wait_any_") == 2
    assert injected.script().count("pending_count") >= 2


def test_alltoallv_reuses_group_after_wait_any_drains_every_member():
    func = repeated_alltoallv_kernel_factory.get_tir(drain_first="any", world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.PlanDistSignals")
    assert collect_op_names(result.device_mod).count("tl.dist_wait_any_") == 2


@pytest.mark.parametrize("drain_first", ["partial", "group_before", "late_any"])
def test_alltoallv_rejects_reuse_before_first_completion_is_drained(drain_first):
    func = repeated_alltoallv_kernel_factory.get_tir(drain_first=drain_first, world_size=4)
    with pytest.raises(tvm.error.InternalError, match="previous completion must be drained"):
        lower_to_device_tir(func)


@pytest.mark.parametrize("drain_first", ["skip", "branch"])
def test_alltoallv_allows_cumulative_group_wait_without_per_member_progress(drain_first):
    func = repeated_alltoallv_kernel_factory.get_tir(drain_first=drain_first, world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistCommunication")
    assert collect_op_names(result.device_mod).count("tl.dist_wait_signal_") == 4
    assert collect_op_names(result.device_mod).count("tl.dist_put_") == 6


def test_alltoallv_supports_all_p2p_scope_combinations():
    func = alltoallv_scope_combinations_kernel_factory.get_tir(world_size=2)
    result = lower_to_device_tir(func, capture_passes="tl.PlanDistSignals")
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]

    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 4
    assert int(planned.attrs["tl.dist.signal_counts"]["dram_flagreg_inc"]) == 4
    device_script = result.device_mod.script()
    assert device_script.count('"sram_flagreg_inc"') >= 2
    assert device_script.count('"dram_flagreg_inc"') >= 2


def test_alltoallv_row_routes_active_state_and_reaches_completion_leaves():
    func = alltoallv_row_kernel_factory.get_tir(wait_mode="any", world_size=2)
    result = lower_to_device_tir(
        func,
        capture_passes=(
            "tl.LowerDistCollectives",
            "tl.PlanDistSignals",
            "tl.LowerDistRouting",
            "tl.LowerDistCommunication",
            "tl.InjectDistSync",
        ),
    )

    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 8

    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    routed_names = collect_op_names(routed)
    routed_script = routed.script()
    assert routed_names.count("tl.tileop.dist_routed_peer_put") == 16
    assert routed_names.count("tl.tileop.comm_put") == 36
    assert "dist_route_active" in routed_script

    lower_comm = result.pass_snapshot("tl.LowerDistCommunication").mod
    lower_names = collect_op_names(lower_comm)
    assert lower_names.count("tl.dist_put_") == 16
    assert lower_names.count("tl.dist_wait_any_") == 1

    injected_script = result.pass_snapshot("tl.InjectDistSync").mod.script()
    assert "rank_id * 4" in injected_script
    assert "recv_counts[1, 3]" in injected_script
    assert "pending_count" in injected_script
    assert "_generation" not in injected_script
    assert "T.uint8(0)" in injected_script


def test_alltoallv_row_value_group_flattens_member_and_destination_generation():
    func = alltoallv_row_kernel_factory.get_tir(
        signal_kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE,
        world_size=2,
    )
    result = lower_to_device_tir(
        func,
        capture_passes=("tl.PlanDistSignals", "tl.LowerDistCommunication"),
    )
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_value"]) == 8

    allocations = []

    def visit(node):
        if isinstance(node, tvm.tir.Allocate) and "generation" in node.buffer_var.name:
            allocations.append(tuple(int(extent) for extent in node.extents))

    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod["main"]
    tvm.tir.stmt_functor.post_order_visit(lowered.body, visit)
    assert allocations == [(64,)]


def test_alltoallv_rejects_count_shape_mismatch():
    with pytest.raises(ValueError, match="send_counts shape"):
        invalid_alltoallv_count_shape_kernel_factory.get_tir(world_size=4)


def test_alltoallv_world_size_one_is_rejected_by_collective_pass():
    func = alltoallv_rank_kernel_factory.get_tir(world_size=1)
    with pytest.raises(tvm.error.InternalError, match="world_size > 1"):
        lower_collectives(func)


def test_alltoallv_requires_exactly_one_signal_form():
    @tilelang.jit(target="sunmmio")
    def invalid_kernel_factory(pass_both: bool, world_size: int = 1):
        @T.prim_func
        def main(rank_id: T.dist.RankId):
            with T.Kernel():
                src = T.alloc_shared((world_size, 4, 8), T.bfloat16)
                dst = T.alloc_shared((world_size, 4, 8), T.bfloat16)
                send_counts = T.alloc_shared((world_size,), T.int32)
                recv_counts = T.alloc_shared((world_size,), T.int32)
                signal = T.dist.signal()
                if pass_both:
                    signals = T.dist.signals(world_size)
                    T.dist.all_to_allv(
                        src,
                        dst,
                        send_counts=send_counts,
                        recv_counts=recv_counts,
                        signal=signal,
                        signals=signals,
                    )
                else:
                    T.dist.all_to_allv(
                        src,
                        dst,
                        send_counts=send_counts,
                        recv_counts=recv_counts,
                    )

        return main

    with pytest.raises(TypeError, match="exactly one"):
        invalid_kernel_factory.get_tir(pass_both=False, world_size=4)
    with pytest.raises(TypeError, match="exactly one"):
        invalid_kernel_factory.get_tir(pass_both=True, world_size=4)


def test_alltoallv_single_signal_rejects_non_increment_kind():
    with pytest.raises(ValueError, match="INC flagreg"):
        alltoallv_single_signal_rank_kernel_factory.get_tir(
            signal_kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE,
            world_size=4,
        )
