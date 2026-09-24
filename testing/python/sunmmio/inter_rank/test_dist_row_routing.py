"""Static same-column cross-row Rank put routing tests."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import (
    collect_op_names,
    single_prim_func,
)


@tilelang.jit(target="sunmmio")
def row_shift_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel() as core_id:
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            current_row = core_id // T.mesh_ncols()
            target_row = (current_row + 1) % T.mesh_nrows()
            T.dist.put(
                src,
                dst,
                dst_rank=(rank_id + 1) % world_size,
                dst_row=target_row,
                signal=signal,
            )
            T.dist.submit()
            T.dist.wait_signal(signal)
            T.dist.wait()

    return main


@tilelang.jit(target="sunmmio")
def explicit_peer_row_kernel_factory(world_size: int = 1):
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
                dst_rank=(rank_id + 1) % world_size,
                dst_row=current_row,
                signal=signal,
            )
            T.dist.submit()
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def dynamic_row_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            route = T.alloc_shared((1,), T.int32)
            signal = T.dist.signal()
            T.dist.put(
                src,
                dst,
                dst_rank=(rank_id + 1) % world_size,
                dst_row=route[0],
                signal=signal,
            )
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def fixed_destination_row_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel() as core_id:
            src = T.alloc_shared((1, 32), T.bfloat16)
            dst = T.alloc_shared((4, 32), T.bfloat16)
            signal = T.dist.signal()
            current_row = core_id // T.mesh_ncols()
            T.dist.put(
                src,
                dst[current_row : current_row + 1, :],
                dst_rank=(rank_id + 1) % world_size,
                dst_row=0,
                signal=signal,
            )
            T.dist.submit()
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def invalid_static_row_kernel_factory(dst_row: int, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            T.dist.put(
                src,
                dst,
                dst_rank=(rank_id + 1) % world_size,
                dst_row=dst_row,
                signal=signal,
            )
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def cross_row_loop_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel() as core_id:
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            current_row = core_id // T.mesh_ncols()
            for _step in T.serial(2):
                T.dist.put(
                    src,
                    dst,
                    dst_rank=(rank_id + 1) % world_size,
                    dst_row=(current_row + 1) % T.mesh_nrows(),
                    signal=signal,
                )
                T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def cross_row_while_kernel_factory(explicit_routes=False, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel() as core_id:
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signals = T.dist.signals(1, expect="manual")
            step = T.alloc_var(T.int32, init=0)
            while step < 2:
                if explicit_routes:
                    T.dist.routed_put(
                        src,
                        dst,
                        routes=[[0, (rank_id + 1) % world_size, 1]],
                        signal=signals[0],
                        submit=True,
                    )
                else:
                    T.dist.put(
                        src,
                        dst,
                        dst_rank=(rank_id + 1) % world_size,
                        dst_row=(core_id // T.mesh_ncols() + 1) % T.mesh_nrows(),
                        signal=signals[0],
                        submit=True,
                    )
                T.dist.wait_signal(signals[0], expected_delta=1)
                step += 1

    return main


@tilelang.jit(target="sunmmio")
def public_routed_put_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            T.dist.routed_put(
                src,
                dst,
                routes=[[1, (rank_id + 1) % world_size, 2]],
                signal=signal,
            )
            T.dist.submit()
            T.dist.wait_signal(signal)
            T.dist.wait()

    return main


@tilelang.jit(target="sunmmio")
def dynamic_offset_put_kernel_factory(cross_row=False, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel() as core_id:
            src = T.alloc_shared((world_size * 32,), T.bfloat16)
            dst = T.alloc_shared((world_size * 32,), T.bfloat16)
            signal = T.dist.signal()
            offset = rank_id * 32
            current_row = core_id // T.mesh_ncols()
            T.dist.put(
                src[offset : offset + 32],
                dst[offset : offset + 32],
                dst_rank=(rank_id + 1) % world_size,
                dst_row=(current_row + 1) % T.mesh_nrows() if cross_row else current_row,
                signal=signal,
                submit=True,
            )
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def rank_guarded_routed_put_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            if rank_id == 0:
                T.dist.routed_put(
                    src,
                    dst,
                    routes=[[1, 1, 2]],
                    signal=signal,
                )
                T.dist.submit()
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def rank_guarded_peer_put_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            if rank_id == 0:
                T.dist.put(src, dst, dst_rank=1, signal=signal)
                T.dist.submit()
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def column_guarded_routed_put_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel() as core_id:
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            if core_id % T.mesh_ncols() == 1:
                T.dist.routed_put(
                    src,
                    dst,
                    routes=[[1, (rank_id + 1) % world_size, 2]],
                    signal=signal,
                )
                T.dist.submit()
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def cid_guarded_put_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel() as core_id:
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            if core_id == 5:
                T.dist.put(
                    src,
                    dst,
                    dst_rank=(rank_id + 1) % world_size,
                    signal=signal,
                )
                T.dist.submit()

    return main


@tilelang.jit(target="sunmmio")
def row_guarded_routed_put_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel() as core_id:
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            if core_id // T.mesh_ncols() == 1:
                T.dist.routed_put(
                    src,
                    dst,
                    routes=[[1, (rank_id + 1) % world_size, 2]],
                    signal=signal,
                )
                T.dist.submit()

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
            T.dist.submit()
            T.dist.wait_signal(signal)
            T.dist.wait()

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
            T.dist.submit()
            T.dist.wait_signal(signal)

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
            T.dist.submit()
            T.dist.wait_signal(signal)

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
            T.dist.submit()
            T.dist.wait_signal(signal)

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
            T.dist.submit()
            T.dist.wait_signal(signal)

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
            T.dist.submit()
            T.dist.wait_signal(signal)

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
            T.dist.submit()
            T.dist.wait_all(signals)
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


def test_explicit_current_row_uses_peer_fast_path():
    func = explicit_peer_row_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistRouting")
    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    names = collect_op_names(routed)

    assert names.count("tl.tileop.dist_peer_put") == 1
    assert "tl.tileop.comm_put" not in names
    assert "dist_route_stage" not in routed.script()


def test_row_shift_expands_local_forwarding_and_peer_puts():
    func = row_shift_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(
        func,
        capture_passes=(
            "tl.LowerDistRouting",
            "tl.LowerDistCommunication",
            "tl.LowerTileOp",
            "tl.InjectDistSync",
        ),
    )

    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    routed_names = collect_op_names(routed)
    assert routed_names.count("tl.tileop.comm_put") == 4
    assert routed_names.count("tl.tileop.dist_routed_peer_put") == 1
    assert routed_names.count("tl.dist_peer_route") == 4
    assert "tl.dist_expect" not in routed_names
    assert routed.script().count("dist_route_stage") > 0

    lowered_names = collect_op_names(result.pass_snapshot("tl.LowerDistCommunication").mod)
    assert "tl.tileop.dist_peer_put" not in lowered_names
    assert "tl.tileop.dist_routed_peer_put" not in lowered_names
    assert lowered_names.count("tl.dist_put_") == 4
    assert lowered_names.count("tl.dist_expect_") == 1

    tile_lowered_names = collect_op_names(result.pass_snapshot("tl.LowerTileOp").mod)
    assert "tl.tileop.comm_put" not in tile_lowered_names
    assert "tl.broadcast_" in tile_lowered_names

    device_func = single_prim_func(result.device_mod)
    device_names = []

    def collect_device_op(node):
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op):
            device_names.append(node.op.name)

    tvm.tir.stmt_functor.post_order_visit(device_func.body, collect_device_op)
    assert "tl.dist_routed_put" not in device_names
    assert "tl.tileop.dist_routed_peer_put" not in device_names
    assert "tl.dist_route_table" not in device_names
    assert "tl.dist_peer_route_table" not in device_names
    device_script = device_func.script()
    assert "T.dist_expect_" not in device_script
    assert "signal_expect" in device_script
    assert "signal_generation" not in device_script
    assert device_script.count('"sram_flagreg_inc", 0, T.uint8(0)') == 4
    assert device_script.count("T.dist_put_(") == 4
    assert "T.wait_token(" not in device_script
    assert "T.sync_token_id(" not in device_script


def test_fixed_destination_row_rejects_source_row_dependent_offset():
    func = fixed_destination_row_kernel_factory.get_tir(world_size=4)
    with pytest.raises(
        tvm.error.InternalError,
        match="region offsets must be compile-time constants for cross-row",
    ):
        lower_to_device_tir(func)


def test_cross_row_rejects_data_dependent_destination_row():
    func = dynamic_row_kernel_factory.get_tir(world_size=4)
    with pytest.raises(tvm.error.InternalError, match="cannot depend on BufferLoad"):
        lower_to_device_tir(func)


@pytest.mark.parametrize("dst_row", [-1, 4])
def test_cross_row_rejects_out_of_range_static_row(dst_row):
    func = invalid_static_row_kernel_factory.get_tir(dst_row=dst_row, world_size=4)
    with pytest.raises(tvm.error.InternalError, match="cannot prove dst_row is in"):
        lower_to_device_tir(func)


def test_cross_row_rejects_loop_carried_staging_reuse():
    func = cross_row_loop_kernel_factory.get_tir(world_size=4)
    with pytest.raises(tvm.error.InternalError, match="Cross-row T.dist.put inside loops"):
        lower_to_device_tir(func)


@pytest.mark.parametrize(
    "explicit_routes, message",
    [
        (False, "Cross-row T.dist.put inside loops"),
        (True, "T.dist.routed_put inside loops"),
    ],
)
def test_cross_row_while_rejects_staging_reuse(explicit_routes, message):
    func = cross_row_while_kernel_factory.get_tir(explicit_routes=explicit_routes, world_size=4)
    with pytest.raises(tvm.error.InternalError, match=message):
        lower_to_device_tir(func)


def test_dst_row_rejects_non_integer_value():
    with pytest.raises(TypeError, match="dst_row must be an integer or TIR PrimExpr"):

        @tilelang.jit(target="sunmmio")
        def invalid_kernel_factory(world_size: int = 1):
            @T.prim_func
            def main(rank_id: T.dist.RankId):
                with T.Kernel():
                    src = T.alloc_shared((32,), T.bfloat16)
                    dst = T.alloc_shared((32,), T.bfloat16)
                    signal = T.dist.signal()
                    T.dist.put(
                        src,
                        dst,
                        dst_rank=(rank_id + 1) % world_size,
                        dst_row="row",
                        signal=signal,
                    )

            return main

        invalid_kernel_factory.get_tir(world_size=4)


def test_public_routed_put_enters_the_explicit_route_pipeline():
    func = public_routed_put_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistRouting")
    routed_names = collect_op_names(result.pass_snapshot("tl.LowerDistRouting").mod)
    assert routed_names.count("tl.tileop.comm_put") == 1
    assert routed_names.count("tl.tileop.dist_routed_peer_put") == 1


def test_direct_peer_put_allows_sender_evaluated_dynamic_offsets():
    func = dynamic_offset_put_kernel_factory.get_tir(cross_row=False, world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistRouting")
    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    assert collect_op_names(routed).count("tl.tileop.dist_peer_put") == 1
    assert "rank_id * 32" in routed.script()


def test_cross_row_put_rejects_dynamic_region_offsets():
    func = dynamic_offset_put_kernel_factory.get_tir(cross_row=True, world_size=4)
    with pytest.raises(
        tvm.error.InternalError,
        match="region offsets must be compile-time constants for cross-row",
    ):
        lower_to_device_tir(func)


def test_rank_guard_is_preserved_for_send_and_lifted_for_expectation():
    func = rank_guarded_routed_put_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistCommunication")
    script = result.pass_snapshot("tl.LowerDistCommunication").mod.script()
    marker = "T.Select(rank_id == 1 and bx // 4 == 2, 1, 0)"
    assert "T.dist_expect_(" in script
    assert marker in script
    assert "if rank_id == 0:" in script
    assert script.index(marker) < script.index("if rank_id == 0:")


def test_rank_guarded_peer_put_uses_the_same_expectation_lifting():
    func = rank_guarded_peer_put_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistCommunication")
    script = result.pass_snapshot("tl.LowerDistCommunication").mod.script()
    marker = "T.Select(rank_id == 1, 1, 0)"
    assert "T.dist_expect_(" in script
    assert marker in script
    assert script.index(marker) < script.index("if rank_id == 0:")


def test_column_guard_remains_visible_to_receiver_expectation():
    func = column_guarded_routed_put_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistCommunication")
    script = result.pass_snapshot("tl.LowerDistCommunication").mod.script()
    assert "bx % 4 == 1" in script
    assert "T.dist_expect_(" in script


def test_cid_guarded_put_requires_explicit_routed_put():
    func = cid_guarded_put_kernel_factory.get_tir(world_size=4)
    with pytest.raises(tvm.error.InternalError, match="Use T.dist.routed_put"):
        lower_to_device_tir(func)


def test_row_guard_cannot_wrap_routed_put():
    func = row_guarded_routed_put_kernel_factory.get_tir(world_size=4)
    with pytest.raises(tvm.error.InternalError, match="uniform across rows"):
        lower_to_device_tir(func)


def test_dram_cross_row_forwards_source_to_egress_staging():
    func = dram_cross_row_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistRouting")
    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    names = collect_op_names(routed)
    assert names.count("tl.tileop.dist_routed_peer_put") == 1
    assert names.count("tl.tileop.comm_put") == 1
    assert "dist_route_stage" in routed.script()
    assert "dist_local_stage" not in routed.script()
    assert collect_op_names(result.device_mod).count("tl.dist_put_") == 1


def test_cross_row_dram_source_to_rsram_uses_egress_staging():
    func = dram_to_rsram_cross_row_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistRouting")
    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    names = collect_op_names(routed)
    assert names.count("tl.tileop.comm_put") == 1
    assert names.count("tl.tileop.dist_routed_peer_put") == 1
    assert "dist_route_stage" in routed.script()
    assert collect_op_names(result.device_mod).count("tl.dist_put_") == 1


def test_explicit_source_rank_becomes_outer_rank_guard():
    func = explicit_source_rank_kernel_factory.get_tir(world_size=4)
    frontend_names = collect_op_names(tvm.IRModule({"main": func}))
    assert frontend_names.count("tl.dist_rank_routed_put") == 1

    result = lower_to_device_tir(func, capture_passes="tl.LowerDistRouting")
    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    names = collect_op_names(routed)
    assert "tl.dist_rank_routed_put" not in names
    assert "tl.dist_routed_put" not in names
    assert "rank_id == 1" in routed.script()
    assert collect_op_names(result.device_mod).count("tl.dist_put_") == 1


def test_same_rank_peer_route_lowers_to_copy_without_dist_expectation():
    func = local_peer_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(
        func,
        capture_passes=("tl.LowerDistRouting", "tl.LowerDistCommunication"),
    )
    local_names = collect_op_names(result.pass_snapshot("tl.LowerDistRouting").mod)
    assert "tl.tileop.dist_put" not in local_names
    assert local_names.count("tl.tileop.copy") == 1
    assert "tl.tileop.comm_put" not in local_names
    assert "tl.dist_expect_" not in collect_op_names(result.pass_snapshot("tl.LowerDistCommunication").mod)
    assert "tl.dist_put_" not in collect_op_names(result.device_mod)


def test_same_rank_cross_row_route_lowers_to_comm_put():
    func = local_cross_row_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistRouting")
    names = collect_op_names(result.pass_snapshot("tl.LowerDistRouting").mod)
    assert "tl.tileop.dist_put" not in names
    assert names.count("tl.tileop.comm_put") == 4
    assert "tl.dist_put_" not in collect_op_names(result.device_mod)


def test_same_rank_dram_cross_row_uses_destination_staging_and_writeback():
    func = local_dram_cross_row_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistRouting")
    local = result.pass_snapshot("tl.LowerDistRouting").mod
    names = collect_op_names(local)
    assert "tl.tileop.dist_put" not in names
    assert names.count("tl.tileop.comm_put") == 4
    assert names.count("tl.tileop.copy") == 4
    assert "dist_local_stage" in local.script()
    assert "tl.dist_put_" not in collect_op_names(result.device_mod)


def test_multiple_routed_puts_keep_offsets_and_use_independent_staging():
    func = multiple_routed_put_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistRouting")
    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    script = routed.script()
    assert collect_op_names(routed).count("tl.tileop.comm_put") == 2
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
