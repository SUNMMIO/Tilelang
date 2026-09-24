"""Signal/SignalList declarations, kind/scope inference, allocation, and sender constraints."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.utils.target import determine_target
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import (
    collect_op_names,
    signal_counts,
    single_prim_func,
)


@tilelang.jit(target="sunmmio")
def mixed_scope_signal_kernel_factory(M, N, world_size: int = 1):
    placement = T.placement.full_shard(0, 1)

    @T.prim_func
    def main(
        B: T.MeshTensor((M, N), placement, T.bfloat16),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            local_M, local_N = B.local_shape
            src = T.alloc_shared((local_M, local_N), T.bfloat16)
            sram_dst = T.alloc_shared((local_M, local_N), T.bfloat16)
            signal = T.dist.signal()
            peer_rank = (rank_id + 1) % world_size
            T.dist.put(src, sram_dst, dst_rank=peer_rank, signal=signal)
            T.dist.put(src, B, dst_rank=peer_rank, signal=signal)
            T.dist.submit()
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def explicit_sram_signal_on_dram_kernel_factory(M, N, world_size: int = 1):
    placement = T.placement.full_shard(0, 1)

    @T.prim_func
    def main(
        B: T.MeshTensor((M, N), placement, T.bfloat16),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            local_M, local_N = B.local_shape
            src = T.alloc_shared((local_M, local_N), T.bfloat16)
            signal = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_INC)
            peer_rank = (rank_id + 1) % world_size
            T.dist.put(src, B, dst_rank=peer_rank, signal=signal)
            T.dist.submit()
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def too_many_explicit_sram_inc_flagregs_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal_0 = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_INC)
            signal_1 = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_INC)
            signal_2 = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_INC)
            signal_3 = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_INC)
            signal_4 = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_INC)
            signal_5 = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_INC)
            signal_6 = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_INC)
            signal_7 = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_INC)
            signal_8 = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_INC)
            peer_rank = (rank_id + 1) % world_size
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_0)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_1)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_2)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_3)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_4)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_5)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_6)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_7)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_8)
            T.dist.submit()

    return main


@tilelang.jit(target="sunmmio")
def signal_list_kernel_factory(signal_kind=None, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src0 = T.alloc_shared((32,), T.bfloat16)
            src1 = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((64,), T.bfloat16)
            signal_list = T.dist.signals(2, kind=signal_kind)
            peer_rank = (rank_id + 1) % world_size

            T.dist.put(src0, dst[0:32], dst_rank=peer_rank, signal=signal_list[0])
            T.dist.put(src1, dst[32:64], dst_rank=peer_rank, signal=signal_list[1])
            T.dist.submit()
            T.dist.wait_all(signal_list)
            T.dist.wait()

    return main


@tilelang.jit(target="sunmmio")
def mixed_scope_signal_list_kernel_factory(world_size: int = 1):
    placement = T.placement.replicated()

    @T.prim_func
    def main(
        B: T.MeshTensor((32,), placement, T.bfloat16),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal_list = T.dist.signals(2)
            peer_rank = (rank_id + 1) % world_size
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_list[0])
            T.dist.put(src, B, dst_rank=peer_rank, signal=signal_list[1])
            T.dist.submit()
            T.dist.wait()

    return main


@tilelang.jit(target="sunmmio")
def multiple_sender_one_signal_kernel_factory(signal_kind=None, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal(kind=signal_kind)
            T.dist.put(src, dst, dst_rank=0, signal=signal)
            T.dist.submit()
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def too_many_signals_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal_0 = T.dist.signal()
            signal_1 = T.dist.signal()
            signal_2 = T.dist.signal()
            signal_3 = T.dist.signal()
            signal_4 = T.dist.signal()
            signal_5 = T.dist.signal()
            signal_6 = T.dist.signal()
            signal_7 = T.dist.signal()
            signal_8 = T.dist.signal()
            peer_rank = (rank_id + 1) % world_size
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_0)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_1)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_2)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_3)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_4)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_5)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_6)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_7)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_8)
            T.dist.submit()

    return main


@tilelang.jit(target="sunmmio")
def invalid_payload_scope_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32, 32), T.bfloat16, scope="shared.asram")
            dst = T.alloc_shared((32, 32), T.bfloat16, scope="shared.asram")
            signal = T.dist.signal()
            peer_rank = (rank_id + 1) % T.dist.world_size()
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal)
            T.dist.submit()
            T.dist.wait_signal(signal)
            T.dist.wait()

    return main


@tilelang.jit(target="sunmmio")
def signal_in_loop_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            for _step in T.serial(2):
                T.dist.signal()

    return main


@tilelang.jit(target="sunmmio")
def all_signal_kinds_kernel_factory(M, N, world_size: int = 1):
    placement = T.placement.full_shard(0, 1)

    @T.prim_func
    def main(
        B: T.MeshTensor((M, N), placement, T.bfloat16),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            local_M, local_N = B.local_shape
            src = T.alloc_shared((local_M, local_N), T.bfloat16)
            sram_dst = T.alloc_shared((local_M, local_N), T.bfloat16)
            sram_inc = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_INC)
            dram_inc = T.dist.signal(kind=T.dist.SignalKind.DRAM_FLAGREG_INC)
            sram_value = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE)
            dram_value = T.dist.signal(kind=T.dist.SignalKind.DRAM_FLAGREG_VALUE)
            sram_memory = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY)
            dram_memory = T.dist.signal(kind=T.dist.SignalKind.DRAM_MEMORY)
            peer_rank = (rank_id + 1) % world_size

            T.dist.put(src, sram_dst, dst_rank=peer_rank, signal=sram_inc)
            T.dist.put(src, B, dst_rank=peer_rank, signal=dram_inc)
            T.dist.put(src, sram_dst, dst_rank=peer_rank, signal=sram_value)
            T.dist.put(src, B, dst_rank=peer_rank, signal=dram_value)
            T.dist.put(src, sram_dst, dst_rank=peer_rank, signal=sram_memory)
            T.dist.put(src, B, dst_rank=peer_rank, signal=dram_memory)
            T.dist.submit()

    return main


@tilelang.jit(target="sunmmio")
def generation_cost_planning_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal_0 = T.dist.signal()
            signal_1 = T.dist.signal()
            signal_2 = T.dist.signal()
            signal_3 = T.dist.signal()
            signal_4 = T.dist.signal()
            signal_5 = T.dist.signal()
            signal_6 = T.dist.signal()
            signal_7 = T.dist.signal()
            signal_8 = T.dist.signal()
            peer_rank = (rank_id + 1) % world_size
            if rank_id == 0:
                T.dist.put(src, dst, dst_rank=1, signal=signal_0)
                T.dist.put(src, dst, dst_rank=2, signal=signal_0)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_1)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_2)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_3)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_4)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_5)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_6)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_7)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_8)
            T.dist.submit()

    return main


def test_plan_dist_signals_rejects_mixed_destination_scopes():
    func = mixed_scope_signal_kernel_factory.get_tir(32, 32, world_size=4)
    with pytest.raises(tvm.error.InternalError, match="inconsistent destination scopes"):
        lower_to_device_tir(func)


def test_plan_dist_signals_rejects_sram_flagreg_for_dram_destination():
    func = explicit_sram_signal_on_dram_kernel_factory.get_tir(32, 32, world_size=4)
    with pytest.raises(tvm.error.InternalError, match="explicitly requests sram_flagreg_inc"):
        lower_to_device_tir(func)


def test_plan_dist_signals_rejects_explicit_flagreg_capacity_overflow():
    func = too_many_explicit_sram_inc_flagregs_kernel_factory.get_tir(world_size=4)
    with pytest.raises(tvm.error.InternalError, match="sram_flagreg_inc signal capacity exceeded"):
        lower_to_device_tir(func)


def test_plan_dist_signals_resolves_all_six_explicit_kinds():
    func = all_signal_kinds_kernel_factory.get_tir(32, 32, world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.PlanDistSignals")
    planned = single_prim_func(result.pass_snapshot("tl.PlanDistSignals").mod)
    assert signal_counts(planned) == {
        "sram_flagreg_inc": 1,
        "dram_flagreg_inc": 1,
        "sram_flagreg_value": 1,
        "dram_flagreg_value": 1,
        "sram_memory": 1,
        "dram_memory": 1,
    }
    script = planned.script()
    for kind in signal_counts(planned):
        assert f'T.dist_signal("{kind}", 0, "auto")' in script


def test_plan_dist_signals_is_idempotent_after_resources_are_resolved():
    func = all_signal_kinds_kernel_factory.get_tir(32, 32, world_size=4)
    target = tvm.target.Target(determine_target("sunmmio", return_object=True))
    mod = tvm.IRModule({"main": func.with_attr("target", target)})
    mod = tilelang.transform.ResolveSunmmioMeshSymbols()(mod)
    mod = tilelang.transform.InferSramScope()(mod)
    planned = tilelang.transform.PlanDistSignals()(mod)
    planned_again = tilelang.transform.PlanDistSignals()(planned)
    assert tvm.ir.structural_equal(planned_again, planned)


def test_global_planning_keeps_high_generation_cost_signal_in_inc_capacity():
    func = generation_cost_planning_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.PlanDistSignals")
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    counts = signal_counts(planned)
    assert counts["sram_flagreg_inc"] == 8
    assert counts["sram_flagreg_value"] == 1
    script = planned.script()
    assert 'signal_0: T.handle = T.dist_signal("sram_flagreg_inc", 0, "auto")' in script
    assert 'signal_8: T.handle = T.dist_signal("sram_flagreg_value", 0, "auto")' in script


@pytest.mark.parametrize("signal_kind", [None, T.dist.SignalKind.SRAM_FLAGREG_INC])
def test_signals_and_wait_all_expand_to_independent_waits(signal_kind):
    func = signal_list_kernel_factory.get_tir(signal_kind=signal_kind, world_size=4)
    frontend_names = collect_op_names(tvm.IRModule({"main": func}))
    assert frontend_names.count("tl.dist_signal_group_decl") == 1
    assert "tl.dist_signal_decl" not in frontend_names
    assert frontend_names.count("tl.dist_signal_ref") == 2
    assert frontend_names.count("tl.dist_wait_all") == 1

    result = lower_to_device_tir(func, capture_passes=("tl.PlanDistSignals", "tl.LowerDistCommunication"))
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 2
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_value"]) == 0
    assert 'T.dist_signal_group("sram_flagreg_inc", 0, 2, "auto")' in planned.script()

    lowered_names = collect_op_names(result.pass_snapshot("tl.LowerDistCommunication").mod)
    assert "tl.dist_wait_all" not in lowered_names
    assert lowered_names.count("tl.dist_wait_signal_") == 2


def test_signal_list_rejects_mixed_destination_scopes():
    func = mixed_scope_signal_list_kernel_factory.get_tir(world_size=4)
    with pytest.raises(tvm.error.InternalError, match="inconsistent destination scopes"):
        lower_to_device_tir(func)


def test_signal_list_rejects_out_of_range_static_index():
    with pytest.raises(IndexError, match="index out of range"):

        @tilelang.jit(target="sunmmio")
        def invalid_kernel_factory(world_size: int = 1):
            @T.prim_func
            def main(rank_id: T.dist.RankId):
                with T.Kernel():
                    signal_list = T.dist.signals(2)
                    T.evaluate(signal_list[2])

            return main

        invalid_kernel_factory.get_tir(world_size=4)


def test_increment_signal_allows_multiple_physical_senders():
    func = multiple_sender_one_signal_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.InjectDistSync")
    script = result.pass_snapshot("tl.InjectDistSync").mod.script()
    assert "signal_expect" in script
    assert "T.Select(rank_id == 0, 3, 0)" in script


@pytest.mark.parametrize(
    "signal_kind",
    [T.dist.SignalKind.SRAM_FLAGREG_VALUE, T.dist.SignalKind.SRAM_MEMORY],
)
def test_value_and_memory_signals_reject_multiple_physical_senders(signal_kind):
    func = multiple_sender_one_signal_kernel_factory.get_tir(
        signal_kind=signal_kind,
        world_size=4,
    )
    with pytest.raises(tvm.error.InternalError, match="multiple physical senders"):
        lower_to_device_tir(func)


def test_signal_planning_spills_ninth_auto_signal_to_value_flagreg():
    func = too_many_signals_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.PlanDistSignals")
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    counts = planned.attrs["tl.dist.signal_counts"]
    assert int(counts["sram_flagreg_inc"]) == 8
    assert int(counts["sram_flagreg_value"]) == 1


def test_signal_planning_rejects_unsupported_destination_scope():
    func = invalid_payload_scope_kernel_factory.get_tir(world_size=4)
    target = tvm.target.Target(determine_target("sunmmio", return_object=True))
    mod = tvm.IRModule({"main": func})
    mod = tvm.tir.transform.BindTarget(target)(mod)
    with pytest.raises(tvm.error.InternalError, match="destination must use shared.rsram or global/DRAM"):
        tilelang.transform.PlanDistSignals()(mod)


def test_signal_rejects_loop_local_declaration():
    with pytest.raises(RuntimeError, match="outside loops and conditionals"):
        signal_in_loop_kernel_factory.get_tir(world_size=4)
