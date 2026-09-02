"""Static same-column cross-row Rank put routing tests."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir


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
            T.dist.wait_signal(signal, dst=dst)
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
            T.dist.wait_signal(signal, dst=dst)

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
            T.dist.wait_signal(signal, dst=dst)

    return main


@tilelang.jit(target="sunmmio")
def memory_signal_cross_row_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel() as core_id:
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY)
            current_row = core_id // T.mesh_ncols()
            T.dist.put(
                src,
                dst,
                dst_rank=(rank_id + 1) % world_size,
                dst_row=(current_row + 1) % T.mesh_nrows(),
                signal=signal,
            )
            T.dist.wait_signal(signal, dst=dst)

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
            T.dist.wait_signal(signal, dst=dst)

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
            T.dist.wait_signal(signal, dst=dst)

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
                T.dist.wait_signal(signal, dst=dst)

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
            T.dist.wait_signal(signal, dst=dst)
            T.dist.wait()

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
            T.dist.wait_signal(signal, dst=dst)

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
            T.dist.wait_signal(signal, dst=dst)

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
            T.dist.wait_signal(signal, dst=dst)

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


def _single_device_func(mod):
    funcs = [func for func in mod.functions.values() if isinstance(func, tvm.tir.PrimFunc)]
    assert len(funcs) == 1
    return funcs[0]


def test_explicit_current_row_uses_peer_fast_path():
    func = explicit_peer_row_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistRouting")
    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    names = _op_names(routed)

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
    routed_names = _op_names(routed)
    assert routed_names.count("tl.tileop.comm_put") == 4
    assert routed_names.count("tl.tileop.dist_routed_peer_put") == 1
    assert routed_names.count("tl.dist_peer_route") == 4
    assert "tl.dist_expect" not in routed_names
    assert routed.script().count("dist_route_stage") > 0

    lowered_names = _op_names(result.pass_snapshot("tl.LowerDistCommunication").mod)
    assert "tl.tileop.dist_peer_put" not in lowered_names
    assert "tl.tileop.dist_routed_peer_put" not in lowered_names
    assert lowered_names.count("tl.dist_put_") == 4
    assert lowered_names.count("tl.dist_expect_") == 1

    tile_lowered_names = _op_names(result.pass_snapshot("tl.LowerTileOp").mod)
    assert "tl.tileop.comm_put" not in tile_lowered_names
    assert "tl.broadcast_" in tile_lowered_names

    device_func = _single_device_func(result.device_mod)
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
    assert "signal_generation" in device_script
    assert device_script.count("T.dist_put_(") == 4
    assert device_script.count("T.wait_token(") >= 4


def test_fixed_destination_row_keeps_peer_route_and_expands_other_rows():
    func = fixed_destination_row_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(
        func,
        capture_passes=("tl.LowerDistRouting", "tl.LowerDistCommunication"),
    )
    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    names = _op_names(routed)

    assert names.count("tl.tileop.comm_put") == 3
    assert names.count("tl.tileop.dist_routed_peer_put") == 1
    assert names.count("tl.dist_peer_route") == 4
    assert "tl.dist_expect" not in names
    script = routed.script()
    assert "dst[0, 0:32]" in script
    assert "dst[1, 0:32]" in script
    assert "dst[2, 0:32]" in script
    assert "dst[3, 0:32]" in script
    expected_script = result.pass_snapshot("tl.LowerDistCommunication").mod.script()
    assert "T.dist_expect_(" in expected_script
    assert "T.Select(bx // 4 == 0, 4, 0)" in expected_script


def test_cross_row_rejects_data_dependent_destination_row():
    func = dynamic_row_kernel_factory.get_tir(world_size=4)
    with pytest.raises(tvm.error.InternalError, match="cannot depend on BufferLoad"):
        lower_to_device_tir(func)


def test_cross_row_rejects_dram_memory_signal():
    func = memory_signal_cross_row_kernel_factory.get_tir(world_size=4)
    with pytest.raises(tvm.error.InternalError, match="sram_memory signal does not support cross-row"):
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
    routed_names = _op_names(result.pass_snapshot("tl.LowerDistRouting").mod)
    assert routed_names.count("tl.tileop.comm_put") == 1
    assert routed_names.count("tl.tileop.dist_routed_peer_put") == 1


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
