"""打印核心 Rank 间通信能力在关键 pass 前后的 TIR。

直接运行本文件，或使用 ``pytest -s``，可以查看 P2P、routing 和 collective 正向路径。
"""

from contextlib import contextmanager
from pathlib import Path
import sys

import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir


# 可选值："show" 通过 mod.show() 显示；"log" 写入下方日志目录。
# TIR_OUTPUT_MODE = "show"
TIR_OUTPUT_MODE = "log"
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
            T.dist.wait_signal(signal, dst=dst)

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
            T.dist.wait_all(signals, dst=dst)
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
            T.dist.wait_all(signals, dst=dst)

    return main


@tilelang.jit(target="sunmmio")
def collective_all_gather_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            signal = T.dist.signal()
            T.dist.all_gather(src, dst, signal=signal)
            T.dist.wait_signal(signal, dst=dst)

    return main


@tilelang.jit(target="sunmmio")
def collective_all_to_all_rank_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            signal = T.dist.signal()
            T.dist.all_to_all(src, dst, signal=signal)
            T.dist.wait_signal(signal, dst=dst)

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
            )
            T.dist.wait_signal(signal, dst=dst)

    return main


@tilelang.jit(target="sunmmio")
def collective_all_reduce_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.bfloat16)
            dst = T.alloc_shared((4, 8), T.bfloat16)
            T.dist.all_reduce(src, dst, "sum")

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
            )
            while T.dist.has_pending(completion):
                source = T.dist.wait_any(completion)
                T.evaluate(dst[source, 0, 0])

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
            )
            T.dist.wait_signal(signal, dst=dst)

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


def _collect_op_names(mod):
    names = []

    def visit(node):
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op):
            names.append(node.op.name)

    for func in mod.functions.values():
        if isinstance(func, tvm.tir.PrimFunc):
            tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return names


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


def test_print_signal_list_and_wait_all_passes():
    result = _lower_and_print_case(
        "多 signal、BufferRegion offset 与 wait-all",
        signal_list_wait_all_kernel_factory,
        SIGNAL_PASSES,
    )

    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod
    names = _collect_op_names(lowered)
    assert "tl.dist_wait_all" not in names
    assert names.count("tl.dist_wait_signal_") == 2


def test_print_cross_row_routing_passes():
    result = _lower_and_print_case(
        "RSRAM 自动 cross-row 路由",
        rsram_cross_row_kernel_factory,
        ROUTING_PASSES,
    )

    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    names = _collect_op_names(routed)
    assert names.count("tl.tileop.comm_put") == 4
    assert names.count("tl.tileop.dist_routed_peer_put") == 1


def test_print_dram_cross_row_routing_passes():
    result = _lower_and_print_case(
        "DRAM 到 DRAM 的显式 cross-row 路由",
        dram_cross_row_kernel_factory,
        DRAM_ROUTING_PASSES,
    )

    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert int(planned.attrs["tl.dist.signal_counts"]["dram_flagreg_inc"]) == 1
    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    assert "dist_route_stage" in routed.script()
    assert _collect_op_names(routed).count("tl.tileop.comm_put") == 1


def test_print_same_rank_copy_and_comm_passes():
    result = _lower_and_print_case(
        "本 Rank 同 row copy 与 cross-row T.comm.put",
        local_rank_routes_kernel_factory,
        LOCAL_RANK_PASSES,
    )

    local = result.pass_snapshot("tl.LowerDistRouting").mod
    names = _collect_op_names(local)
    assert "tl.tileop.dist_put" not in names
    assert names.count("tl.tileop.copy") == 1
    assert names.count("tl.tileop.comm_put") == 4
    assert "tl.dist_put_" not in _collect_op_names(result.device_mod)


def test_print_collective_all_gather_passes():
    result = _lower_and_print_case(
        "Collective all-gather 的 direct Rank 通信",
        collective_all_gather_kernel_factory,
        COLLECTIVE_PASSES,
        world_size=4,
    )

    collective_names = _collect_op_names(result.pass_snapshot("tl.LowerDistCollectives").mod)
    assert "tl.tileop.dist_allgather" not in collective_names
    assert collective_names.count("tl.tileop.copy") == 1
    assert collective_names.count("tl.tileop.dist_put") == 3
    routed_names = _collect_op_names(result.pass_snapshot("tl.LowerDistRouting").mod)
    assert routed_names.count("tl.tileop.dist_peer_put") == 3
    assert _collect_op_names(result.device_mod).count("tl.dist_put_") == 3


def test_print_collective_all_to_all_rank_passes():
    result = _lower_and_print_case(
        "Collective all-to-all 的 RANK domain",
        collective_all_to_all_rank_kernel_factory,
        COLLECTIVE_PASSES,
        world_size=4,
    )

    collective_names = _collect_op_names(result.pass_snapshot("tl.LowerDistCollectives").mod)
    assert "tl.tileop.dist_alltoall" not in collective_names
    assert collective_names.count("tl.tileop.dist_put") == 4
    routed_names = _collect_op_names(result.pass_snapshot("tl.LowerDistRouting").mod)
    assert routed_names.count("tl.tileop.copy") == 1
    assert routed_names.count("tl.tileop.dist_peer_put") == 3
    assert _collect_op_names(result.device_mod).count("tl.dist_put_") == 3


def test_print_collective_all_to_all_row_passes():
    result = _lower_and_print_case(
        "Collective all-to-all 的完整 (rank, row) domain",
        collective_all_to_all_row_kernel_factory,
        COLLECTIVE_PASSES,
        world_size=2,
    )

    collective_names = _collect_op_names(result.pass_snapshot("tl.LowerDistCollectives").mod)
    assert "tl.tileop.dist_alltoall" not in collective_names
    assert collective_names.count("tl.tileop.dist_put") == 8
    routed_names = _collect_op_names(result.pass_snapshot("tl.LowerDistRouting").mod)
    assert routed_names.count("tl.tileop.copy") == 4
    assert routed_names.count("tl.tileop.comm_put") == 24
    assert routed_names.count("tl.tileop.dist_routed_peer_put") == 4
    assert routed_names.count("tl.dist_peer_route") == 16
    device_names = _collect_op_names(result.device_mod)
    assert device_names.count("tl.dist_put_") == 16
    assert device_names.count("tl.dist_wait_signal_") == 1


def test_print_collective_all_reduce_passes():
    result = _lower_and_print_case(
        "Collective all-reduce 的 gather + local reduce",
        collective_all_reduce_kernel_factory,
        COLLECTIVE_PASSES,
        world_size=4,
    )

    lowered = result.pass_snapshot("tl.LowerDistCollectives").mod
    lowered_names = _collect_op_names(lowered)
    lowered_script = lowered.script()
    assert "tl.tileop.dist_allreduce" not in lowered_names
    assert lowered_names.count("tl.tileop.dist_put") == 3
    assert lowered_names.count("tl.tileop.dist_wait_signal") == 1
    assert lowered_names.count("tl.tileop.reduce") == 1
    assert "dist_allreduce_gather_0" in lowered_script
    assert lowered_script.index("T.dist_wait_signal") < lowered_script.index("T.reduce(")
    device_names = _collect_op_names(result.device_mod)
    assert device_names.count("tl.dist_put_") == 3
    assert device_names.count("tl.dist_wait_signal_") == 1


def test_print_collective_all_to_allv_rank_passes():
    result = _lower_and_print_case(
        "Collective all-to-allv 的 RANK domain 与 wait-any",
        collective_all_to_allv_rank_kernel_factory,
        COLLECTIVE_PASSES,
        world_size=4,
    )

    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_value"]) == 4
    lower_names = _collect_op_names(result.pass_snapshot("tl.LowerDistCommunication").mod)
    assert lower_names.count("tl.dist_put_") == 3
    assert lower_names.count("tl.dist_completion_init_") == 1
    assert lower_names.count("tl.dist_wait_any_") == 1


def test_print_collective_all_to_allv_single_signal_passes():
    result = _lower_and_print_case(
        "Collective all-to-allv 的单 signal 聚合完成",
        collective_all_to_allv_single_signal_kernel_factory,
        COLLECTIVE_PASSES,
        world_size=4,
    )

    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 1
    lower_names = _collect_op_names(result.pass_snapshot("tl.LowerDistCommunication").mod)
    assert lower_names.count("tl.dist_put_") == 3
    assert lower_names.count("tl.dist_expect_") == 1
    assert lower_names.count("tl.dist_wait_signal_") == 1
    assert "tl.dist_wait_any_" not in lower_names


def test_print_collective_all_to_allv_row_passes():
    result = _lower_and_print_case(
        "Collective all-to-allv 的 ROW domain 与 active route",
        collective_all_to_allv_row_kernel_factory,
        COLLECTIVE_PASSES,
        world_size=2,
    )

    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    assert "dist_route_active" in routed.script()
    assert _collect_op_names(routed).count("tl.tileop.dist_routed_peer_put") == 16
    device_names = _collect_op_names(result.device_mod)
    assert device_names.count("tl.dist_put_") == 16
    assert device_names.count("tl.dist_wait_any_") == 1


def test_print_dist_barrier_passes():
    result = _lower_and_print_case(
        "全 Rank 对等 barrier 与纯 signal 通知",
        dist_barrier_kernel_factory,
        COLLECTIVE_PASSES,
        world_size=4,
    )

    lowered = result.pass_snapshot("tl.LowerDistCollectives").mod
    assert _collect_op_names(lowered).count("tl.dist_signal_put") == 3
    device_names = _collect_op_names(result.device_mod)
    assert device_names.count("tl.dist_signal_put_") == 3
    assert device_names.count("tl.dist_wait_barrier_") == 1


if __name__ == "__main__":
    test_print_signal_list_and_wait_all_passes()
    test_print_cross_row_routing_passes()
    test_print_dram_cross_row_routing_passes()
    test_print_same_rank_copy_and_comm_passes()
    test_print_collective_all_gather_passes()
    test_print_collective_all_to_all_rank_passes()
    test_print_collective_all_to_all_row_passes()
    test_print_collective_all_reduce_passes()
    test_print_collective_all_to_allv_rank_passes()
    test_print_collective_all_to_allv_single_signal_passes()
    test_print_collective_all_to_allv_row_passes()
    test_print_dist_barrier_passes()
