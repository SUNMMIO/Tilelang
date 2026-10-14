"""P2P frontend, SRAM/DRAM payload scopes, and complete device TIR lowering."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import (
    collect_op_arg_counts,
    collect_op_names,
    single_prim_func,
)


@tilelang.jit(target="sunmmio")
def minimal_put_kernel_factory(M, N, world_size: int = 1):
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
            dst = T.alloc_shared((local_M, local_N), T.bfloat16)
            signal = T.dist.signal()

            T.copy(A, src)
            peer_rank = (rank_id + 1) % world_size
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal)
            T.dist.submit()
            T.dist.wait_signal(signal)
            T.dist.wait()
            T.copy(dst, B)

    return main


@tilelang.jit(target="sunmmio")
def wait_signal_global_acquire_kernel_factory(M, N, world_size: int = 1):
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
            dst = T.alloc_shared((local_M, local_N), T.bfloat16)
            unrelated = T.alloc_shared((local_M, local_N), T.bfloat16)
            signal = T.dist.signal()

            T.copy(A, src)
            T.dist.put(src, dst, dst_rank=(rank_id + 1) % world_size, signal=signal)
            T.dist.submit()
            T.copy(A, unrelated)
            T.dist.wait_signal(signal)
            T.copy(dst, B)

    return main


@tilelang.jit(target="sunmmio")
def dram_to_dram_kernel_factory(world_size: int = 1):
    placement = T.placement.full_shard(0, 1)

    @T.prim_func
    def main(
        A: T.MeshTensor((32, 32), placement, T.bfloat16),  # type: ignore
        B: T.MeshTensor((32, 32), placement, T.bfloat16),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            signal = T.dist.signal()
            T.dist.put(
                A,
                B,
                dst_rank=(rank_id + 1) % world_size,
                signal=signal,
            )
            T.dist.submit()
            T.dist.wait_signal(signal)
            T.dist.wait()

    return main


@tilelang.jit(target="sunmmio")
def dram_to_rsram_kernel_factory(world_size: int = 1):
    placement = T.placement.full_shard(0, 1)

    @T.prim_func
    def main(
        A: T.MeshTensor((32, 32), placement, T.bfloat16),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            local_M, local_N = A.local_shape
            dst = T.alloc_shared((local_M, local_N), T.bfloat16)
            signal = T.dist.signal()
            T.dist.put(A, dst, dst_rank=(rank_id + 1) % world_size, signal=signal)
            T.dist.submit()
            T.dist.wait_signal(signal)

    return main


def test_p2p_frontend_emits_high_level_ops_and_signal_metadata():
    func = minimal_put_kernel_factory.get_tir(32, 32, world_size=4)
    op_names = collect_op_names(func)

    assert "tl.dist.sram_signal_count" not in func.attrs
    assert op_names.count("tl.dist_signal_decl") == 1
    assert "tl.dist_signal" not in op_names
    assert op_names.count("tl.tileop.dist_put") == 1
    assert op_names.count("tl.dist_submit") == 1
    assert op_names.count("tl.dist_wait_signal") == 1
    assert op_names.count("tl.dist_wait_send") == 1
    assert "tl.dist_put_" not in op_names
    frontend_script = func.script()
    assert 'T.dist_signal_decl("auto", 0, "infer")' in frontend_script
    assert "T.dist_put" in frontend_script
    assert "local.var" not in frontend_script


def test_p2p_reaches_device_leaf_ops():
    func = minimal_put_kernel_factory.get_tir(32, 32, world_size=4)
    result = lower_to_device_tir(
        func,
        capture_passes=(
            "tl.PlanDistSignals",
            "tl.LowerDistRouting",
            "tl.LowerDistCommunication",
            "tl.LowerTileOp",
            "tl.InjectDistSync",
        ),
    )

    planned_signal = result.pass_snapshot("tl.PlanDistSignals").mod
    assert 'T.dist_signal("sram_flagreg_inc", 0, "auto")' in planned_signal.script()
    assert int(planned_signal["main"].attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 1

    routed_names = collect_op_names(result.pass_snapshot("tl.LowerDistRouting").mod)
    assert "tl.tileop.dist_put" not in routed_names
    assert routed_names.count("tl.tileop.dist_peer_put") == 1
    assert "tl.tileop.comm_put" not in routed_names

    planned_names = collect_op_names(result.pass_snapshot("tl.LowerDistCommunication").mod)
    assert "tl.tileop.dist_put" not in planned_names
    assert "tl.dist_signal_decl" not in planned_names
    assert "tl.dist_put_" in planned_names
    assert "tl.dist_wait_signal_" in planned_names
    assert "tl.dist_expect_" in planned_names
    planned_script = result.pass_snapshot("tl.LowerDistCommunication").mod.script()
    assert "tl.dist_signal" not in planned_names
    assert "signal_expect" in planned_script
    assert "signal_generation" not in planned_script
    assert "signal_expect_1[0] =" not in planned_script
    assert '"sram_flagreg_inc", 0, T.uint8(0)' in planned_script
    assert collect_op_arg_counts(result.pass_snapshot("tl.LowerDistCommunication").mod, "tl.dist_put_") == [6]
    assert collect_op_arg_counts(result.pass_snapshot("tl.LowerDistCommunication").mod, "tl.dist_wait_signal_") == [3]

    lowered_names = collect_op_names(result.pass_snapshot("tl.LowerTileOp").mod)
    assert "tl.tileop.dist_put" not in lowered_names
    assert "tl.dist_signal_decl" not in lowered_names
    assert "tl.dist_signal" not in lowered_names
    assert "tl.dist_put_" in lowered_names
    assert "tl.dist_wait_signal_" in lowered_names
    assert "tl.dist_wait_send" in lowered_names
    lowered_script = result.pass_snapshot("tl.LowerTileOp").mod.script()
    assert "signal_expect" in lowered_script
    assert "signal_generation" not in lowered_script
    assert "signal_expect_1[0] =" not in lowered_script
    assert collect_op_arg_counts(result.pass_snapshot("tl.LowerTileOp").mod, "tl.dist_put_") == [6]
    assert collect_op_arg_counts(result.pass_snapshot("tl.LowerTileOp").mod, "tl.dist_wait_signal_") == [3]

    injected_names = collect_op_names(result.pass_snapshot("tl.InjectDistSync").mod)
    assert "tl.dist_put_" in injected_names
    assert "tl.dist_wait_signal_" in injected_names
    assert "tl.dist_wait_send" in injected_names
    injected_script = result.pass_snapshot("tl.InjectDistSync").mod.script()
    assert "signal_expect" in injected_script
    assert "T.dist_signal_decl" not in injected_script
    assert "T.dist_signal(" not in injected_script
    assert "signal_expect_1[0] =" in injected_script
    assert "signal_generation_1[0] =" not in injected_script
    assert collect_op_arg_counts(result.pass_snapshot("tl.InjectDistSync").mod, "tl.dist_put_") == [6]
    assert collect_op_arg_counts(result.pass_snapshot("tl.InjectDistSync").mod, "tl.dist_wait_signal_") == [3]

    device_func = single_prim_func(result.device_mod)
    device_names = collect_op_names(device_func)
    assert int(device_func.attrs["tl.dist.world_size"]) == 4
    assert "tl.dist.sram_signal_count" not in device_func.attrs
    assert int(device_func.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 1
    assert device_names.count("tl.dist_signal_decl") == 0
    assert device_names.count("tl.dist_signal") == 0
    assert device_names.count("tl.dist_put_") == 1
    assert device_names.count("tl.dist_wait_signal_") == 1
    assert device_names.count("tl.dist_wait_send") == 2
    assert device_names.count("tl.dist_submit_") == 1

    script = device_func.script()
    assert "T.dist_signal_decl(" not in script
    assert "T.dist_signal(" not in script
    assert "T.dist_put_(" in script
    assert "T.dist_wait_signal_(" in script
    assert "T.dist_wait_send()" in script
    assert "signal_expect" in script
    assert "signal_generation" not in script
    assert "signal_expect_1[0] = signal_expect_1[0] + T.uint8(1)" in script
    assert '"sram_flagreg_inc", 0, T.uint8(0)' in script
    assert "uint8" in script
    assert "tl.wait_token" not in device_names
    assert "tl.sync_token_id" not in device_names


def test_wait_signal_is_a_conservative_global_acquire():
    func = wait_signal_global_acquire_kernel_factory.get_tir(32, 32, world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.InjectSunmmioSync")
    op_names = collect_op_names(result.pass_snapshot("tl.InjectSunmmioSync").mod)

    wait_signal_index = op_names.index("tl.dist_wait_signal_")
    copy_indices = [index for index, name in enumerate(op_names) if name == "tl.dma_copy"]
    assert len(copy_indices) == 3
    assert copy_indices[1] < wait_signal_index < copy_indices[2]
    assert "tl.sunmmio_sync" in op_names[copy_indices[1] + 1 : wait_signal_index]
    assert "tl.wait_token" not in op_names
    assert "tl.sync_token_id" not in op_names


def test_validate_dist_reports_communication_op_in_single_rank_kernel():
    func = minimal_put_kernel_factory.get_tir(32, 32, world_size=1)
    with pytest.raises(tvm.error.InternalError, match="world_size=1 disables Rank communication"):
        lower_to_device_tir(func)


def test_existing_non_dist_kernel_is_unchanged_by_dist_passes():
    @T.prim_func
    def main(A: T.Tensor((32,), T.float32)):
        with T.Kernel():
            A[0] = A[0] + 1

    target = tvm.target.Target("llvm")
    mod = tvm.IRModule({"main": main.with_attr("target", target)})
    for transform in (
        tilelang.transform.PlanDistSignals(),
        tilelang.transform.LowerDistRouting(),
        tilelang.transform.LowerDistCommunication(),
        tilelang.transform.InjectDistSync(),
    ):
        after = transform(mod)
        assert tvm.ir.structural_equal(after, mod)


def test_dram_to_dram_reaches_dist_leaf_and_uses_dram_signal():
    func = dram_to_dram_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes=("tl.PlanDistSignals", "tl.LowerDistCommunication"))
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert int(planned.attrs["tl.dist.signal_counts"]["dram_flagreg_inc"]) == 1

    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod
    assert collect_op_names(lowered).count("tl.dist_put_") == 1
    assert "A[" in lowered.script()
    assert "B[" in lowered.script()
    assert "rsram_stage" not in lowered.script()


def test_dram_source_to_rsram_destination_reaches_peer_leaf():
    func = dram_to_rsram_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes=("tl.PlanDistSignals", "tl.LowerDistCommunication"))
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 1
    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod
    assert collect_op_names(lowered).count("tl.dist_put_") == 1
    assert "A[" in lowered.script()
    assert "dst[" in lowered.script()
