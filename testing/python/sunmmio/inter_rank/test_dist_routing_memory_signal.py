"""Cross-row routing with MEMORY signals, automatic fallback, and sender/scope constraints."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import (
    collect_calls,
    collect_nodes,
)


def _enqueue_peer_signals(src, dst, peer, count):
    # Emit independent declarations during DSL construction, without a device loop or SignalList.
    for _ in range(count):
        signal = T.dist.signal()
        T.evaluate(T.dist.put(src, dst, peer, signal=signal))


@tilelang.jit(target="sunmmio")
def memory_route_kernel_factory(
    kind=T.dist.SignalKind.SRAM_MEMORY,
    dram=False,
    route="implicit",
    peer_signal_count=0,
    multi_sender=False,
    world_size: int = 1,
):
    @T.prim_func
    def main(
        output: T.MeshTensor((32,), T.placement.replicated(), T.bfloat16),  # type: ignore
        rank_id: T.dist.RankId,
        enabled: T.int32,
        recv_count: T.int32,
    ):
        with T.Kernel() as core_id:
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            active = T.alloc_shared((1,), T.int32)
            active[0] = enabled
            peer = 0 if multi_sender else (rank_id + 1) % world_size
            _enqueue_peer_signals(src, output if dram else dst, peer, peer_signal_count)
            if route in ("member", "active"):
                signals = T.dist.signals(2, kind=kind, expect="manual" if route == "active" else None)
            else:
                signal = T.dist.signal(kind=kind)
            target_row = (core_id // T.mesh_ncols() + 1) % T.mesh_nrows()
            if route == "explicit":
                T.dist.routed_put(src, output if dram else dst, routes=[[0, peer, 1]], signal=signal)
            elif route == "active":
                if active[0] > 0:
                    T.dist.put(src, output if dram else dst, peer, dst_row=target_row, signal=signals[1])
            else:
                T.dist.put(src, output if dram else dst, peer, dst_row=target_row, signal=signals[1] if route == "member" else signal)
            T.dist.submit()
            if route == "active":
                T.dist.wait_signal(signals[1], expected_delta=recv_count)
            else:
                T.dist.wait_signal(signals[1] if route == "member" else signal)

    return main


@pytest.mark.parametrize(
    "dram, kind",
    [
        (False, T.dist.SignalKind.SRAM_MEMORY),
        (True, T.dist.SignalKind.DRAM_MEMORY),
    ],
)
@pytest.mark.parametrize("route", ["implicit", "explicit", "member", "active"])
def test_memory_cross_row_preserves_generation_and_receiver_kind(dram, kind, route):
    result = lower_to_device_tir(
        memory_route_kernel_factory.get_tir(
            kind=kind,
            dram=dram,
            route=route,
            world_size=4,
        ),
        capture_passes=("tl.LowerDistRouting", "tl.InjectDistSync"),
    )
    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    assert collect_calls(routed, "tl.tileop.comm_put")
    injected = result.pass_snapshot("tl.InjectDistSync").mod
    puts = collect_calls(injected, "tl.dist_put_")
    assert len(puts) == (1 if route == "explicit" else 4)
    for put in puts:
        assert put.args[3].value == kind.value
        assert int(put.args[4]) == (1 if route in ("member", "active") else 0)
        generation = put.args[5]
        assert isinstance(generation, tvm.tir.BufferLoad)
        assert str(generation.dtype) == "uint32"
        advances = collect_nodes(
            injected,
            tvm.tir.BufferStore,
            lambda node, generation=generation: node.buffer.data.same_as(generation.buffer.data)
            and not isinstance(node.value, tvm.tir.IntImm),
        )
        assert len(advances) == len(puts)
        analyzer = tvm.arith.Analyzer()
        assert all(
            analyzer.can_prove(advance.value == tvm.tir.BufferLoad(advance.buffer, advance.indices) + tvm.tir.const(1, "uint32"))
            for advance in advances
        )
        assert "dist_route_stage" in str(put.args[0])
    waits = collect_calls(result.device_mod, "tl.dist_wait_signal_")
    assert len(waits) == 1 and waits[0].args[0].value == kind.value
    assert str(waits[0].args[2].dtype) == "uint32"
    assert not collect_calls(result.device_mod, "tl.tileop.dist_routed_peer_put")
    if route == "active":
        assert "dist_route_active" in injected.script()
    if route == "explicit":
        # Advance generation in the row branch that sends, not unconditionally at the origin.
        assert "if " in injected.script()
        seqs = collect_nodes(injected, tvm.tir.SeqStmt)
        assert any(
            isinstance(a, tvm.tir.BufferStore)
            and a.buffer.data.same_as(puts[0].args[5].buffer.data)
            and isinstance(b, tvm.tir.Evaluate)
            and isinstance(b.value, tvm.tir.Call)
            and b.value.same_as(puts[0])
            for seq in seqs
            for a, b in zip(seq.seq, seq.seq[1:])
        )


@pytest.mark.parametrize("dram", [False, True])
def test_auto_cross_row_spills_after_forty_peer_signals(dram):
    result = lower_to_device_tir(
        memory_route_kernel_factory.get_tir(
            kind=None,
            dram=dram,
            peer_signal_count=40,
            world_size=4,
        ),
        capture_passes="tl.PlanDistSignals",
    )
    planned = next(iter(result.pass_snapshot("tl.PlanDistSignals").mod.functions.values()))
    prefix = "dram" if dram else "sram"
    counts = planned.attrs["tl.dist.signal_counts"]
    assert int(counts[prefix + "_flagreg_inc"]) == 8
    assert int(counts[prefix + "_flagreg_value"]) == 32
    assert int(counts[prefix + "_memory"]) == 1
    memory_puts = [call for call in collect_calls(result.device_mod, "tl.dist_put_") if call.args[3].value == prefix + "_memory"]
    assert len(memory_puts) == 4
    assert all("dist_route_stage" in str(call.args[0]) for call in memory_puts)


@pytest.mark.parametrize(
    "dram, kind",
    [
        (False, T.dist.SignalKind.DRAM_MEMORY),
        (True, T.dist.SignalKind.SRAM_MEMORY),
    ],
)
def test_memory_cross_row_rejects_scope_mismatch(dram, kind):
    with pytest.raises(tvm.error.InternalError, match="explicitly requests.*destination scope"):
        lower_to_device_tir(
            memory_route_kernel_factory.get_tir(
                kind=kind,
                dram=dram,
                world_size=4,
            )
        )


@pytest.mark.parametrize(
    "dram, kind",
    [
        (False, T.dist.SignalKind.SRAM_MEMORY),
        (True, T.dist.SignalKind.DRAM_MEMORY),
        (False, T.dist.SignalKind.SRAM_FLAGREG_VALUE),
        (True, T.dist.SignalKind.DRAM_FLAGREG_VALUE),
    ],
)
def test_cross_row_value_and_memory_still_require_unique_sender(dram, kind):
    with pytest.raises(tvm.error.InternalError, match="multiple physical senders"):
        lower_to_device_tir(
            memory_route_kernel_factory.get_tir(
                kind=kind,
                dram=dram,
                multi_sender=True,
                world_size=4,
            )
        )
