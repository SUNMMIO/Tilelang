"""Print TIR before and after key passes for core inter-rank communication features.

Run explicitly with ``python -m testing.python.sunmmio.inter_rank.show_dist_tir``.
Examples retain assertions but are not collected by default pytest discovery.
Use --case to select examples and --output to choose display or log output.
"""

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys

import tilelang
import tilelang.language as T
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import (
    collect_op_names,
)


# Options: "show" displays TIR with mod.show(); "log" writes to the log directory below.
TIR_OUTPUT_MODE = "show"
_TIR_LOG_DIR = Path(__file__).resolve().parent / "log"


SIGNAL_PASSES = (
    "tl.PlanDistSignals",
    "tl.LowerDistCommunication",
    "tl.InjectDistSync",
)

ROUTING_PASSES = (
    "tl.PlanDistSignals",
    "tl.LowerDistRouting",
    "tl.LowerDistCommunication",
)

DRAM_ROUTING_PASSES = (
    "tl.PlanDistSignals",
    "tl.LowerDistRouting",
    "tl.LowerDistCommunication",
)

LOCAL_RANK_PASSES = (
    "tl.LowerDistRouting",
    "tl.LowerDistCommunication",
    "tl.LowerTileOp",
)

MANUAL_EXPECT_PASSES = (
    "tl.PlanDistSignals",
    "tl.LowerDistRouting",
    "tl.LowerDistCommunication",
    "tl.InjectDistSync",
)

COLLECTIVE_PASSES = (
    "tl.LowerDistCollectives",
    "tl.PlanDistSignals",
    "tl.LowerDistRouting",
    "tl.LowerDistCommunication",
    "tl.InjectDistSync",
)


@tilelang.jit(target="sunmmio")
def rsram_cross_row_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel() as core_id:
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            current_row = core_id // T.mesh_ncols()
            dst_row = (current_row + 1) % T.mesh_nrows()
            T.dist.put(
                src,
                dst,
                dst_rank=(rank_id + 1) % world_size,
                dst_row=dst_row,
                signal=signal,
            )
            T.dist.submit()
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def signal_list_wait_all_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src0 = T.alloc_shared((32,), T.bfloat16)
            src1 = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((64,), T.bfloat16)
            signals = T.dist.signals(2)
            peer_rank = (rank_id + 1) % world_size

            T.dist.put(src0, dst[0:32], dst_rank=peer_rank, signal=signals[0])
            T.dist.put(src1, dst[32:64], dst_rank=peer_rank, signal=signals[1])
            T.dist.submit()
            # wait_signals
            T.dist.wait_all(signals)
            # T.dist.wait()

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
def local_rank_routes_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel() as core_id:
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signals = T.dist.signals(2)
            current_row = core_id // T.mesh_ncols()

            T.dist.put(src, dst, dst_rank=rank_id, signal=signals[0])
            T.dist.put(
                src,
                dst,
                dst_rank=rank_id,
                dst_row=(current_row + 1) % T.mesh_nrows(),
                signal=signals[1],
            )
            T.dist.wait_all(signals)

    return main


@tilelang.jit(target="sunmmio")
def signal_kind_generation_kernel_factory(world_size: int = 1):
    placement = T.dist.placement.replicated()

    @T.prim_func
    def main(
        B: T.MeshTensor((32,), T.placement.replicated(), T.bfloat16, rank_placement=placement),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            sram_inc = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_INC)
            sram_value = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE)
            sram_memory = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY)
            dram_inc = T.dist.signal(kind=T.dist.SignalKind.DRAM_FLAGREG_INC)
            dram_value = T.dist.signal(kind=T.dist.SignalKind.DRAM_FLAGREG_VALUE)
            dram_memory = T.dist.signal(kind=T.dist.SignalKind.DRAM_MEMORY)
            peer_rank = (rank_id + 1) % world_size

            T.dist.put(src, dst, dst_rank=peer_rank, signal=sram_inc)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=sram_value)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=sram_memory)
            T.dist.put(src, B, dst_rank=peer_rank, signal=dram_inc)
            T.dist.put(src, B, dst_rank=peer_rank, signal=dram_value)
            T.dist.put(src, B, dst_rank=peer_rank, signal=dram_memory)
            T.dist.submit()
            T.dist.wait_signal(sram_inc)
            T.dist.wait_signal(sram_value)
            T.dist.wait_signal(sram_memory)
            T.dist.wait_signal(dram_inc)
            T.dist.wait_signal(dram_value)
            T.dist.wait_signal(dram_memory)

    return main


@tilelang.jit(target="sunmmio")
def p2p_scope_matrix_kernel_factory(world_size: int = 1):
    rank_placement = T.dist.placement.replicated()
    placement = T.placement.replicated()

    @T.prim_func
    def main(
        A: T.MeshTensor((64,), placement, T.bfloat16, rank_placement=rank_placement),  # type: ignore
        B: T.MeshTensor((64,), placement, T.bfloat16, rank_placement=rank_placement),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            shared_src = T.alloc_shared((32,), T.bfloat16)
            shared_dst0 = T.alloc_shared((32,), T.bfloat16)
            shared_dst1 = T.alloc_shared((32,), T.bfloat16)
            sram_signals = T.dist.signals(2)
            dram_signals = T.dist.signals(2)
            peer_rank = (rank_id + 1) % world_size

            T.dist.put(shared_src, shared_dst0, dst_rank=peer_rank, signal=sram_signals[0])
            T.dist.put(A[0:32], shared_dst1, dst_rank=peer_rank, signal=sram_signals[1])
            T.dist.put(shared_src, B[0:32], dst_rank=peer_rank, signal=dram_signals[0])
            T.dist.put(A[32:64], B[32:64], dst_rank=peer_rank, signal=dram_signals[1])
            T.dist.submit()
            T.dist.wait_all(sram_signals)
            T.dist.wait_all(dram_signals)

    return main


@tilelang.jit(target="sunmmio")
def rank_guarded_routed_put_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            T.dist.routed_put(
                src,
                dst,
                routes=[[0, (rank_id + 1) % world_size, 1]],
                signal=signal,
                src_rank=1,
            )
            T.dist.submit()
            T.dist.wait_signal(signal)
            T.dist.wait()

    return main


@tilelang.jit(target="sunmmio")
def signal_only_sync_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            notify = T.dist.signal()
            arrival = T.dist.signal()
            T.dist.put_signal(notify, (rank_id + 1) % world_size, submit=True)
            T.dist.wait_signal(notify)
            T.dist.barrier_arrive(arrival, submit=True)
            T.dist.wait_signal(arrival)

    return main


@tilelang.jit(target="sunmmio")
def manual_sender_wait_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            send_counts = T.alloc_shared((world_size,), T.int32)
            recv_counts = T.alloc_shared((world_size,), T.int32)
            expected_deltas = T.alloc_shared((world_size,), T.uint32)
            sender_signals = T.dist.signals(
                world_size,
                kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE,
                expect="manual",
            )

            for peer_rank in T.serial(world_size):
                if T.And(peer_rank != rank_id, send_counts[peer_rank] > 0):
                    T.dist.put(
                        src,
                        dst,
                        dst_rank=peer_rank,
                        signal=sender_signals[rank_id],
                    )
            T.dist.submit()
            for source_rank in T.serial(world_size):
                expected_deltas[source_rank] = T.if_then_else(
                    T.And(source_rank != rank_id, recv_counts[source_rank] > 0),
                    1,
                    0,
                )
            T.dist.wait_signals(
                sender_signals,
                expected_deltas=expected_deltas,
            )
            T.dist.wait()

    return main


@tilelang.jit(target="sunmmio")
def manual_signal_wave_kernel_factory(world_size: int = 1):
    total_signals = 20
    wave_size = 8

    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            expected_deltas = T.alloc_shared((total_signals,), T.uint32)
            signals = T.dist.signals(
                total_signals,
                kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE,
                expect=T.dist.ExpectMode.MANUAL,
            )
            peer_rank = (rank_id + 1) % world_size

            for wave in T.serial(T.ceildiv(total_signals, wave_size)):
                for lane in T.serial(wave_size):
                    index = wave * wave_size + lane
                    if index < total_signals:
                        T.dist.put(src, dst, dst_rank=peer_rank, signal=signals[index])
                T.dist.submit()
                for member in T.serial(total_signals):
                    expected_deltas[member] = T.if_then_else(
                        T.And(
                            wave * wave_size <= member,
                            member < T.min((wave + 1) * wave_size, total_signals),
                        ),
                        1,
                        0,
                    )
                T.dist.wait_signals(signals, expected_deltas=expected_deltas)
                T.dist.wait()

    return main


@tilelang.jit(target="sunmmio")
def collective_all_gather_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.bfloat16)
            dst_new_axis = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            dst_axis0 = T.alloc_shared((world_size * 4, 8), T.bfloat16)
            dst_last_axis = T.alloc_shared((4, world_size * 8), T.bfloat16)
            signals = T.dist.signals(3)
            T.dist.all_gather(src, dst_new_axis, signal=signals[0])
            T.dist.all_gather(src, dst_axis0, signal=signals[1], axis=0)
            T.dist.all_gather(src, dst_last_axis, signal=signals[2], axis=-1)
            T.dist.submit()
            T.dist.wait_all(signals)

    return main


@tilelang.jit(target="sunmmio")
def collective_all_gather_wait_any_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            signals = T.dist.signals(
                world_size,
                kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE,
            )
            completion = T.dist.all_gather(
                src,
                dst,
                signals=signals,
                submit=True,
            )
            while T.dist.has_pending(completion):
                source_rank = T.dist.wait_any(completion)
                dst[source_rank, 0, 0] = dst[source_rank, 0, 0]

    return main


@tilelang.jit(target="sunmmio")
def collective_all_to_all_rank_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            signal = T.dist.signal()
            T.dist.all_to_all(src, dst, signal=signal, submit=True)
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def collective_all_to_all_row_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            shape = (world_size, T.mesh_nrows(), 4, 8)
            src = T.alloc_shared(shape, T.bfloat16)
            dst = T.alloc_shared(shape, T.bfloat16)
            signal = T.dist.signal()
            T.dist.all_to_all(
                src,
                dst,
                signal=signal,
                domain=T.dist.CollectiveDomain.ROW,
                submit=True,
            )
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def collective_all_reduce_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.bfloat16)
            dst = T.alloc_shared((4, 8), T.bfloat16)
            signal = T.dist.signal()
            T.dist.all_reduce(src, dst, "sum", signal=signal)

    return main


@tilelang.jit(target="sunmmio")
def collective_all_to_allv_rank_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            send_counts = T.alloc_shared((world_size,), T.int32)
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
            while T.dist.has_pending(completion):
                source = T.dist.wait_any(completion)
                T.evaluate(dst[source, 0, 0])

    return main


@tilelang.jit(target="sunmmio")
def collective_all_to_allv_wait_all_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            send_counts = T.alloc_shared((world_size,), T.int32)
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
def collective_all_to_allv_single_signal_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 8), T.bfloat16)
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
def collective_all_to_allv_row_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            rows = T.mesh_nrows()
            src = T.alloc_shared((world_size, rows, 4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, rows, 4, 8), T.bfloat16)
            send_counts = T.alloc_shared((world_size, rows), T.int32)
            recv_counts = T.alloc_shared((world_size, rows), T.int32)
            signals = T.dist.signals(world_size * rows)
            completion = T.dist.all_to_allv(
                src,
                dst,
                send_counts=send_counts,
                recv_counts=recv_counts,
                signals=signals,
                domain=T.dist.CollectiveDomain.ROW,
                submit=True,
            )
            while T.dist.has_pending(completion):
                source = T.dist.wait_any(completion)
                T.evaluate(dst[source // rows, source % rows, 0, 0])

    return main


@tilelang.jit(target="sunmmio")
def dist_barrier_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            T.dist.barrier()

    return main


def _kernel_name(kernel_factory):
    return kernel_factory.func.__name__.removesuffix("_factory")


@contextmanager
def _tir_output(kernel_name):
    if TIR_OUTPUT_MODE == "show":
        yield None
        return
    if TIR_OUTPUT_MODE != "log":
        raise ValueError(f"Unsupported TIR_OUTPUT_MODE {TIR_OUTPUT_MODE!r}; expected 'show' or 'log'")

    _TIR_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _TIR_LOG_DIR / f"{kernel_name}.log"
    with log_path.open("w", encoding="utf-8") as stream:
        yield stream
    print(f"TIR log written to {log_path}")


def _lower_and_print_case(title, kernel_factory, passes, *, world_size=4):
    kernel_name = _kernel_name(kernel_factory)
    func = kernel_factory.get_tir(world_size=world_size)
    result = lower_to_device_tir(
        func,
        capture_before_passes=(passes[0],),
        capture_passes=passes,
    )

    with _tir_output(kernel_name) as stream:
        output = stream or sys.stdout
        print(f"\n========== {title} ({kernel_name}) ==========", file=output)
        result.print_pass_tir(passes[0], when="before", file=stream)
        for selector in passes:
            result.print_pass_tir(selector, file=stream)
        result.print_device_tir(file=stream)
    return result


def show_signal_list_and_wait_all_passes():
    result = _lower_and_print_case(
        "多 signal、BufferRegion offset 与 wait-all",
        signal_list_wait_all_kernel_factory,
        SIGNAL_PASSES,
    )

    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod
    names = collect_op_names(lowered)
    assert "tl.dist_wait_all" not in names
    assert names.count("tl.dist_wait_signal_") == 2


def show_cross_row_routing_passes():
    result = _lower_and_print_case(
        "RSRAM 自动 cross-row 路由",
        rsram_cross_row_kernel_factory,
        ROUTING_PASSES,
    )

    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    names = collect_op_names(routed)
    assert names.count("tl.tileop.comm_put") == 4
    assert names.count("tl.tileop.dist_routed_peer_put") == 1


def show_dram_cross_row_routing_passes():
    result = _lower_and_print_case(
        "DRAM 到 DRAM 的显式 cross-row 路由",
        dram_cross_row_kernel_factory,
        DRAM_ROUTING_PASSES,
    )

    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert int(planned.attrs["tl.dist.signal_counts"]["dram_flagreg_inc"]) == 1
    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    assert "dist_route_stage" in routed.script()
    assert collect_op_names(routed).count("tl.tileop.comm_put") == 1


def show_same_rank_copy_and_comm_passes():
    result = _lower_and_print_case(
        "本 Rank 同 row copy 与 cross-row T.comm.put",
        local_rank_routes_kernel_factory,
        LOCAL_RANK_PASSES,
    )

    local = result.pass_snapshot("tl.LowerDistRouting").mod
    names = collect_op_names(local)
    assert "tl.tileop.dist_put" not in names
    assert names.count("tl.tileop.copy") == 1
    assert names.count("tl.tileop.comm_put") == 4
    assert "tl.dist_put_" not in collect_op_names(result.device_mod)


def show_signal_kinds_and_generation_passes():
    result = _lower_and_print_case(
        "六种 signal kind 与 sender generation",
        signal_kind_generation_kernel_factory,
        SIGNAL_PASSES,
    )

    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    counts = planned.attrs["tl.dist.signal_counts"]
    for kind in (
        "sram_flagreg_inc",
        "sram_flagreg_value",
        "sram_memory",
        "dram_flagreg_inc",
        "dram_flagreg_value",
        "dram_memory",
    ):
        assert int(counts[kind]) == 1
    lower_names = collect_op_names(result.pass_snapshot("tl.LowerDistCommunication").mod)
    assert lower_names.count("tl.dist_put_") == 6
    assert lower_names.count("tl.dist_wait_signal_") == 6


def show_p2p_scope_matrix_passes():
    result = _lower_and_print_case(
        "RSRAM/DRAM 四种 P2P scope组合",
        p2p_scope_matrix_kernel_factory,
        ROUTING_PASSES,
    )

    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    counts = planned.attrs["tl.dist.signal_counts"]
    assert int(counts["sram_flagreg_inc"]) == 2
    assert int(counts["dram_flagreg_inc"]) == 2
    assert collect_op_names(result.device_mod).count("tl.dist_put_") == 4


def show_rank_guarded_routed_put_passes():
    result = _lower_and_print_case(
        "显式 source Rank 的 routed-put",
        rank_guarded_routed_put_kernel_factory,
        ROUTING_PASSES,
    )

    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    script = routed.script()
    assert "rank_id == 1" in script
    assert collect_op_names(routed).count("tl.tileop.comm_put") == 1
    assert collect_op_names(routed).count("tl.tileop.dist_routed_peer_put") == 1


def show_signal_only_and_split_barrier_passes():
    result = _lower_and_print_case(
        "put-signal 与 barrier-arrive/wait拆分",
        signal_only_sync_kernel_factory,
        COLLECTIVE_PASSES,
    )

    lowered = result.pass_snapshot("tl.LowerDistCollectives").mod
    assert collect_op_names(lowered).count("tl.dist_signal_put") == 4
    device_names = collect_op_names(result.device_mod)
    assert device_names.count("tl.dist_signal_put_") == 4
    assert device_names.count("tl.dist_wait_signal_") == 2


def show_manual_sender_wait_passes():
    result = _lower_and_print_case(
        "Manual expected、sender signal与循环 destination Rank",
        manual_sender_wait_kernel_factory,
        MANUAL_EXPECT_PASSES,
    )

    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert 'T.dist_signal_group("sram_flagreg_value", 0, 4, "manual")' in planned.script()
    routed_script = result.pass_snapshot("tl.LowerDistRouting").mod.script()
    assert "send_counts" in routed_script
    assert "peer_rank" in routed_script
    lowered_names = collect_op_names(result.pass_snapshot("tl.LowerDistCommunication").mod)
    assert lowered_names.count("tl.dist_expect_") == 4
    assert lowered_names.count("tl.dist_wait_signal_") == 4


def show_manual_signal_wave_passes():
    result = _lower_and_print_case(
        "20个 manual signal按8个一轮处理",
        manual_signal_wave_kernel_factory,
        MANUAL_EXPECT_PASSES,
    )

    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_value"]) == 20
    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod
    names = collect_op_names(lowered)
    assert names.count("tl.dist_put_") == 1
    assert names.count("tl.dist_expect_") == 20
    assert names.count("tl.dist_wait_signal_") == 20
    assert "wave * 8 + lane" in lowered.script()


def show_collective_all_gather_passes():
    result = _lower_and_print_case(
        "Collective all-gather 的 axis=None/0/-1",
        collective_all_gather_kernel_factory,
        COLLECTIVE_PASSES,
        world_size=4,
    )

    collective_names = collect_op_names(result.pass_snapshot("tl.LowerDistCollectives").mod)
    assert "tl.tileop.dist_allgather" not in collective_names
    assert collective_names.count("tl.tileop.copy") == 3
    assert collective_names.count("tl.tileop.dist_put") == 9
    routed_names = collect_op_names(result.pass_snapshot("tl.LowerDistRouting").mod)
    assert routed_names.count("tl.tileop.dist_peer_put") == 9
    assert collect_op_names(result.device_mod).count("tl.dist_put_") == 9
    assert collect_op_names(result.device_mod).count("tl.dist_wait_signal_") == 3


def show_collective_all_gather_wait_any_passes():
    result = _lower_and_print_case(
        "Collective all-gather 的逐 source wait-any",
        collective_all_gather_wait_any_kernel_factory,
        COLLECTIVE_PASSES,
        world_size=4,
    )

    collective = result.pass_snapshot("tl.LowerDistCollectives").mod
    assert collect_op_names(collective).count("tl.dist_completion") == 1
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_value"]) == 4
    lower_names = collect_op_names(result.pass_snapshot("tl.LowerDistCommunication").mod)
    assert lower_names.count("tl.dist_completion_init_") == 1
    assert lower_names.count("tl.dist_wait_any_") == 1


def show_collective_all_to_all_rank_passes():
    result = _lower_and_print_case(
        "Collective all-to-all 的 RANK domain",
        collective_all_to_all_rank_kernel_factory,
        COLLECTIVE_PASSES,
        world_size=4,
    )

    collective_names = collect_op_names(result.pass_snapshot("tl.LowerDistCollectives").mod)
    assert "tl.tileop.dist_alltoall" not in collective_names
    assert collective_names.count("tl.tileop.dist_put") == 4
    routed_names = collect_op_names(result.pass_snapshot("tl.LowerDistRouting").mod)
    assert routed_names.count("tl.tileop.copy") == 1
    assert routed_names.count("tl.tileop.dist_peer_put") == 3
    assert collect_op_names(result.device_mod).count("tl.dist_put_") == 3


def show_collective_all_to_all_row_passes():
    result = _lower_and_print_case(
        "Collective all-to-all 的完整 (rank, row) domain",
        collective_all_to_all_row_kernel_factory,
        COLLECTIVE_PASSES,
        world_size=2,
    )

    collective_names = collect_op_names(result.pass_snapshot("tl.LowerDistCollectives").mod)
    assert "tl.tileop.dist_alltoall" not in collective_names
    assert collective_names.count("tl.tileop.dist_put") == 8
    routed_names = collect_op_names(result.pass_snapshot("tl.LowerDistRouting").mod)
    assert routed_names.count("tl.tileop.copy") == 4
    assert routed_names.count("tl.tileop.comm_put") == 24
    assert routed_names.count("tl.tileop.dist_routed_peer_put") == 4
    assert routed_names.count("tl.dist_peer_route") == 16
    device_names = collect_op_names(result.device_mod)
    assert device_names.count("tl.dist_put_") == 16
    assert device_names.count("tl.dist_wait_signal_") == 1


def show_collective_all_reduce_passes():
    result = _lower_and_print_case(
        "Collective all-reduce 的 gather + local reduce",
        collective_all_reduce_kernel_factory,
        COLLECTIVE_PASSES,
        world_size=4,
    )

    lowered = result.pass_snapshot("tl.LowerDistCollectives").mod
    lowered_names = collect_op_names(lowered)
    lowered_script = lowered.script()
    assert "tl.tileop.dist_allreduce" not in lowered_names
    assert lowered_names.count("tl.tileop.dist_put") == 3
    assert lowered_names.count("tl.dist_wait_signal") == 1
    assert lowered_names.count("tl.tileop.reduce") == 1
    assert lowered_names.count("tl.dist_wait_send") == 1
    assert "dist_allreduce_gather_0" in lowered_script
    assert lowered_script.index("T.dist_wait_signal") < lowered_script.index("T.reduce(")
    assert lowered_script.index("T.reduce(") < lowered_script.index("T.dist_wait_send()")
    device_names = collect_op_names(result.device_mod)
    assert device_names.count("tl.dist_put_") == 3
    assert device_names.count("tl.dist_wait_signal_") == 1


def show_collective_all_to_allv_rank_passes():
    result = _lower_and_print_case(
        "Collective all-to-allv 的 RANK domain 与 wait-any",
        collective_all_to_allv_rank_kernel_factory,
        COLLECTIVE_PASSES,
        world_size=4,
    )

    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 4
    lower_names = collect_op_names(result.pass_snapshot("tl.LowerDistCommunication").mod)
    assert lower_names.count("tl.dist_put_") == 3
    assert lower_names.count("tl.dist_completion_init_") == 1
    assert lower_names.count("tl.dist_wait_any_") == 1


def show_collective_all_to_allv_completion_wait_all_passes():
    result = _lower_and_print_case(
        "Collective all-to-allv completion的wait-all",
        collective_all_to_allv_wait_all_kernel_factory,
        COLLECTIVE_PASSES,
        world_size=4,
    )

    lower_comm = result.pass_snapshot("tl.LowerDistCommunication").mod
    lower_names = collect_op_names(lower_comm)
    assert lower_names.count("tl.dist_completion_init_") == 1
    assert lower_names.count("tl.dist_wait_any_") == 1
    injected_script = result.pass_snapshot("tl.InjectDistSync").mod.script()
    assert "while 0 <" in injected_script
    assert "pending_count" in injected_script


def show_collective_all_to_allv_single_signal_passes():
    result = _lower_and_print_case(
        "Collective all-to-allv 的单 signal 聚合完成",
        collective_all_to_allv_single_signal_kernel_factory,
        COLLECTIVE_PASSES,
        world_size=4,
    )

    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 1
    lower_names = collect_op_names(result.pass_snapshot("tl.LowerDistCommunication").mod)
    assert lower_names.count("tl.dist_put_") == 3
    assert lower_names.count("tl.dist_expect_") == 1
    assert lower_names.count("tl.dist_wait_signal_") == 1
    assert "tl.dist_wait_any_" not in lower_names


def show_collective_all_to_allv_row_passes():
    result = _lower_and_print_case(
        "Collective all-to-allv 的 ROW domain 与 active route",
        collective_all_to_allv_row_kernel_factory,
        COLLECTIVE_PASSES,
        world_size=2,
    )

    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    assert "dist_route_active" in routed.script()
    assert collect_op_names(routed).count("tl.tileop.dist_routed_peer_put") == 16
    device_names = collect_op_names(result.device_mod)
    assert device_names.count("tl.dist_put_") == 16
    assert device_names.count("tl.dist_wait_any_") == 1


def show_dist_barrier_passes():
    result = _lower_and_print_case(
        "全 Rank 对等 barrier 与纯 signal 通知",
        dist_barrier_kernel_factory,
        COLLECTIVE_PASSES,
        world_size=4,
    )

    lowered = result.pass_snapshot("tl.LowerDistCollectives").mod
    assert collect_op_names(lowered).count("tl.dist_signal_put") == 3
    device_names = collect_op_names(result.device_mod)
    assert device_names.count("tl.dist_signal_put_") == 3
    assert device_names.count("tl.dist_wait_signal_") == 1


CASES = {
    "signal_list_and_wait_all": show_signal_list_and_wait_all_passes,
    "cross_row_routing": show_cross_row_routing_passes,
    "dram_cross_row_routing": show_dram_cross_row_routing_passes,
    "same_rank_copy_and_comm": show_same_rank_copy_and_comm_passes,
    "signal_kinds_and_generation": show_signal_kinds_and_generation_passes,
    "p2p_scope_matrix": show_p2p_scope_matrix_passes,
    "rank_guarded_routed_put": show_rank_guarded_routed_put_passes,
    "signal_only_and_split_barrier": show_signal_only_and_split_barrier_passes,
    "manual_sender_wait": show_manual_sender_wait_passes,
    "manual_signal_wave": show_manual_signal_wave_passes,
    "collective_all_gather": show_collective_all_gather_passes,
    "collective_all_gather_wait_any": show_collective_all_gather_wait_any_passes,
    "collective_all_to_all_rank": show_collective_all_to_all_rank_passes,
    "collective_all_to_all_row": show_collective_all_to_all_row_passes,
    "collective_all_reduce": show_collective_all_reduce_passes,
    "collective_all_to_allv_rank": show_collective_all_to_allv_rank_passes,
    "collective_all_to_allv_completion_wait_all": show_collective_all_to_allv_completion_wait_all_passes,
    "collective_all_to_allv_single_signal": show_collective_all_to_allv_single_signal_passes,
    "collective_all_to_allv_row": show_collective_all_to_allv_row_passes,
    "dist_barrier": show_dist_barrier_passes,
}


def main(argv=None):
    global TIR_OUTPUT_MODE
    parser = argparse.ArgumentParser(description="显示 dist 示例的关键 pass 和最终 device TIR。")
    parser.add_argument("--case", nargs="+", choices=tuple(CASES), help="只运行指定示例；默认运行全部。")
    parser.add_argument(
        "--output", choices=("show", "log"), default=TIR_OUTPUT_MODE, help="show 使用 mod.show()；log 写入 log/<kernel 名>.log。"
    )
    parser.add_argument("--list", action="store_true", help="列出示例名称，不执行编译。")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(CASES))
        return
    TIR_OUTPUT_MODE = args.output
    for name in args.case or CASES:
        CASES[name]()


if __name__ == "__main__":
    main()
