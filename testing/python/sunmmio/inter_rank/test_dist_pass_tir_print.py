"""打印阶段三核心 Rank 间通信能力在关键 pass 前后的 TIR。

直接运行本文件，或使用 ``pytest -s``，可以查看四类正向路径：多 signal/wait-all、
RSRAM cross-row、DRAM cross-row，以及本 Rank copy/comm 降级。
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


def _lower_and_print_case(title, kernel_factory, passes):
    kernel_name = _kernel_name(kernel_factory)
    func = kernel_factory.get_tir(world_size=4)
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


if __name__ == "__main__":
    test_print_signal_list_and_wait_all_passes()
    test_print_cross_row_routing_passes()
    test_print_dram_cross_row_routing_passes()
    test_print_same_rank_copy_and_comm_passes()
