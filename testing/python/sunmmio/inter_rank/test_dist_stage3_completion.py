"""阶段三静态 P2P 能力收口测试。"""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir


@tilelang.jit(target="sunmmio")
def signal_list_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src0 = T.alloc_shared((32,), T.bfloat16)
            src1 = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((64,), T.bfloat16)
            signal_list = T.dist.signals(2)
            peer_rank = (rank_id + 1) % world_size

            T.dist.put(src0, dst[0:32], dst_rank=peer_rank, signal=signal_list[0])
            T.dist.put(src1, dst[32:64], dst_rank=peer_rank, signal=signal_list[1])
            T.dist.wait_all(signal_list, dst=dst)
            T.dist.wait()

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
            T.dist.wait_signal(signal, dst=B)
            T.dist.wait()

    return main


@tilelang.jit(target="sunmmio")
def dram_cross_row_kernel_factory(world_size: int = 1):
    placement = T.placement.full_shard(0, 1)

    @T.prim_func
    def main(
        A: T.MeshTensor((32, 32), placement, T.bfloat16),  # type: ignore
        B: T.MeshTensor((32, 32), placement, T.bfloat16),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            signal = T.dist.signal()
            T.dist.routed_put(
                A,
                B,
                routes=[[0, (rank_id + 1) % world_size, 1]],
                signal=signal,
            )
            T.dist.wait_signal(signal, dst=B)
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
            T.dist.wait_signal(signal, dst=dst)

    return main


@tilelang.jit(target="sunmmio")
def dram_to_rsram_cross_row_kernel_factory(world_size: int = 1):
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
            T.dist.routed_put(
                A,
                dst,
                routes=[[0, (rank_id + 1) % world_size, 1]],
                signal=signal,
            )
            T.dist.wait_signal(signal, dst=dst)

    return main


@tilelang.jit(target="sunmmio")
def explicit_source_rank_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            T.dist.routed_put(
                src,
                dst,
                routes=[[0, 2, 1]],
                signal=signal,
                src_rank=1,
            )
            T.dist.wait_signal(signal, dst=dst)

    return main


@tilelang.jit(target="sunmmio")
def local_peer_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            T.dist.put(src, dst, dst_rank=rank_id, signal=signal)
            T.dist.wait_signal(signal, dst=dst)

    return main


@tilelang.jit(target="sunmmio")
def local_cross_row_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel() as core_id:
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            current_row = core_id // T.mesh_ncols()
            T.dist.put(
                src,
                dst,
                dst_rank=rank_id,
                dst_row=(current_row + 1) % T.mesh_nrows(),
                signal=signal,
            )
            T.dist.wait_signal(signal, dst=dst)

    return main


@tilelang.jit(target="sunmmio")
def local_dram_cross_row_kernel_factory(world_size: int = 1):
    placement = T.placement.full_shard(0, 1)

    @T.prim_func
    def main(
        A: T.MeshTensor((32, 32), placement, T.bfloat16),  # type: ignore
        B: T.MeshTensor((32, 32), placement, T.bfloat16),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel() as core_id:
            signal = T.dist.signal()
            current_row = core_id // T.mesh_ncols()
            T.dist.put(
                A,
                B,
                dst_rank=rank_id,
                dst_row=(current_row + 1) % T.mesh_nrows(),
                signal=signal,
            )
            T.dist.wait_signal(signal, dst=B)

    return main


@tilelang.jit(target="sunmmio")
def multiple_routed_put_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((64,), T.bfloat16)
            dst = T.alloc_shared((128,), T.bfloat16)
            signals = T.dist.signals(2)
            peer_rank = (rank_id + 1) % world_size

            T.dist.routed_put(
                src[8:40],
                dst[16:48],
                routes=[[0, peer_rank, 1]],
                signal=signals[0],
            )
            T.dist.routed_put(
                src[32:64],
                dst[80:112],
                routes=[[2, peer_rank, 3]],
                signal=signals[1],
            )
            T.dist.wait_all(signals, dst=dst)
            T.dist.wait()

    return main


@tilelang.jit(target="sunmmio")
def invalid_route_kernel_factory(routes, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            T.dist.routed_put(src, dst, routes=routes, signal=signal)

    return main


@tilelang.jit(target="sunmmio")
def multiple_sender_one_signal_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            T.dist.put(src, dst, dst_rank=0, signal=signal)
            T.dist.wait_signal(signal, dst=dst)

    return main


def _op_names(mod):
    names = []

    def visit(node):
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op):
            names.append(node.op.name)

    for func in mod.functions.values():
        if isinstance(func, tvm.tir.PrimFunc):
            tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return names


def test_signals_and_wait_all_expand_to_independent_waits():
    func = signal_list_kernel_factory.get_tir(world_size=4)
    assert _op_names(tvm.IRModule({"main": func})).count("tl.dist_wait_all") == 1

    result = lower_to_device_tir(func, capture_passes=("tl.PlanDistSignals", "tl.LowerDistCommunication"))
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 2

    lowered_names = _op_names(result.pass_snapshot("tl.LowerDistCommunication").mod)
    assert "tl.dist_wait_all" not in lowered_names
    assert lowered_names.count("tl.dist_wait_signal_") == 2


def test_signal_list_rejects_dynamic_indexing():
    with pytest.raises(TypeError, match="compile-time integer indexing"):

        @tilelang.jit(target="sunmmio")
        def invalid_kernel_factory(world_size: int = 1):
            @T.prim_func
            def main(rank_id: T.dist.RankId):
                with T.Kernel():
                    signal_list = T.dist.signals(2)
                    T.evaluate(signal_list[rank_id])

            return main

        invalid_kernel_factory.get_tir(world_size=4)


def test_dram_to_dram_reaches_dist_leaf_and_uses_dram_signal():
    func = dram_to_dram_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes=("tl.PlanDistSignals", "tl.LowerDistCommunication"))
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert int(planned.attrs["tl.dist.signal_counts"]["dram_flagreg_inc"]) == 1

    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod
    assert _op_names(lowered).count("tl.dist_put_") == 1
    assert "A[" in lowered.script()
    assert "B[" in lowered.script()
    assert "rsram_stage" not in lowered.script()


def test_dram_cross_row_forwards_source_to_egress_staging():
    func = dram_cross_row_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistRouting")
    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    names = _op_names(routed)
    assert names.count("tl.tileop.dist_routed_peer_put") == 1
    assert names.count("tl.tileop.comm_put") == 1
    assert "dist_route_stage" in routed.script()
    assert "dist_local_stage" not in routed.script()
    assert _op_names(result.device_mod).count("tl.dist_put_") == 1


def test_dram_source_to_rsram_destination_reaches_peer_leaf():
    func = dram_to_rsram_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes=("tl.PlanDistSignals", "tl.LowerDistCommunication"))
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 1
    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod
    assert _op_names(lowered).count("tl.dist_put_") == 1
    assert "A[" in lowered.script()
    assert "dst[" in lowered.script()


def test_cross_row_dram_source_to_rsram_uses_egress_staging():
    func = dram_to_rsram_cross_row_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistRouting")
    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    names = _op_names(routed)
    assert names.count("tl.tileop.comm_put") == 1
    assert names.count("tl.tileop.dist_routed_peer_put") == 1
    assert "dist_route_stage" in routed.script()
    assert _op_names(result.device_mod).count("tl.dist_put_") == 1


def test_explicit_source_rank_becomes_outer_rank_guard():
    func = explicit_source_rank_kernel_factory.get_tir(world_size=4)
    frontend_names = _op_names(tvm.IRModule({"main": func}))
    assert frontend_names.count("tl.dist_rank_routed_put") == 1

    result = lower_to_device_tir(func, capture_passes="tl.LowerDistRouting")
    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    names = _op_names(routed)
    assert "tl.dist_rank_routed_put" not in names
    assert "tl.dist_routed_put" not in names
    assert "rank_id == 1" in routed.script()
    assert _op_names(result.device_mod).count("tl.dist_put_") == 1


def test_same_rank_peer_route_lowers_to_copy_without_dist_expectation():
    func = local_peer_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(
        func,
        capture_passes=("tl.LowerDistRouting", "tl.LowerDistCommunication"),
    )
    local_names = _op_names(result.pass_snapshot("tl.LowerDistRouting").mod)
    assert "tl.tileop.dist_put" not in local_names
    assert local_names.count("tl.tileop.copy") == 1
    assert "tl.tileop.comm_put" not in local_names
    assert "tl.dist_expect_" not in _op_names(result.pass_snapshot("tl.LowerDistCommunication").mod)
    assert "tl.dist_put_" not in _op_names(result.device_mod)


def test_same_rank_cross_row_route_lowers_to_comm_put():
    func = local_cross_row_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistRouting")
    names = _op_names(result.pass_snapshot("tl.LowerDistRouting").mod)
    assert "tl.tileop.dist_put" not in names
    assert names.count("tl.tileop.comm_put") == 4
    assert "tl.dist_put_" not in _op_names(result.device_mod)


def test_same_rank_dram_cross_row_uses_destination_staging_and_writeback():
    func = local_dram_cross_row_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistRouting")
    local = result.pass_snapshot("tl.LowerDistRouting").mod
    names = _op_names(local)
    assert "tl.tileop.dist_put" not in names
    assert names.count("tl.tileop.comm_put") == 4
    assert names.count("tl.tileop.copy") == 4
    assert "dist_local_stage" in local.script()
    assert "tl.dist_put_" not in _op_names(result.device_mod)


def test_multiple_routed_puts_keep_offsets_and_use_independent_staging():
    func = multiple_routed_put_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistRouting")
    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    script = routed.script()
    assert _op_names(routed).count("tl.tileop.comm_put") == 2
    assert script.count("dist_route_stage") > 0
    assert "src[8:40]" in script
    assert "src[32:64]" in script
    assert "dst[16:48]" in script
    assert "dst[80:112]" in script


@pytest.mark.parametrize(
    "routes, message",
    [
        (((4, 1, 0),), "source row is outside"),
        (((0, 4, 0),), "cannot prove dst_rank is in"),
        (((0, 1, 0), (0, 1, 0)), "duplicate static route"),
    ],
)
def test_explicit_route_rejects_invalid_endpoints_and_duplicates(routes, message):
    func = invalid_route_kernel_factory.get_tir(routes=routes, world_size=4)
    with pytest.raises(tvm.error.InternalError, match=message):
        lower_to_device_tir(func)


def test_one_signal_rejects_multiple_physical_senders():
    func = multiple_sender_one_signal_kernel_factory.get_tir(world_size=4)
    with pytest.raises(tvm.error.InternalError, match="multiple physical senders"):
        lower_to_device_tir(func)
