"""Signal kind planning and independent multi-signal generation tests."""

from collections import Counter

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.utils.target import determine_target
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir


@tilelang.jit(target="sunmmio")
def multi_signal_kernel_factory(M, N, world_size: int = 1):
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
            sram_dst_0 = T.alloc_shared((local_M, local_N), T.bfloat16)
            sram_dst_1 = T.alloc_shared((local_M, local_N), T.bfloat16)
            s0 = T.dist.signal()
            s1 = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE)
            d0 = T.dist.signal()
            m0 = T.dist.signal(kind=T.dist.SignalKind.DRAM_MEMORY)

            T.copy(A, src)
            peer_rank = (rank_id + 1) % world_size
            T.dist.put(src, sram_dst_0, dst_rank=peer_rank, signal=s0)
            T.dist.put(src, sram_dst_1, dst_rank=peer_rank, signal=s1)
            T.dist.put(src, sram_dst_0, dst_rank=peer_rank, signal=s0)
            T.dist.put(src, B, dst_rank=peer_rank, signal=d0)
            T.dist.put(src, B, dst_rank=peer_rank, signal=m0)
            T.dist.wait_signal(s1, dst=sram_dst_1)
            T.dist.wait_signal(s0, dst=sram_dst_0)
            T.dist.wait_signal(d0, dst=B)
            T.dist.wait_signal(m0, dst=B)
            T.dist.wait()

    return main


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
            T.dist.wait_signal(signal, dst=B)

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
            T.dist.wait_signal(signal, dst=B)

    return main


@tilelang.jit(target="sunmmio")
def too_many_explicit_dram_flagregs_kernel_factory(world_size: int = 1):
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

    return main


def _single_prim_func(mod):
    funcs = [func for func in mod.functions.values() if isinstance(func, tvm.tir.PrimFunc)]
    assert len(funcs) == 1
    return funcs[0]


def _signal_counts(func):
    return {str(kind): int(count) for kind, count in func.attrs["tl.dist.signal_counts"].items()}


def _collect_leaf_signals(func, op_name, kind_index_positions, expected_position):
    result = []

    def visit(node):
        if not isinstance(node, tvm.tir.Call) or not isinstance(node.op, tvm.ir.Op):
            return
        if node.op.name != op_name:
            return
        expected = node.args[expected_position]
        assert isinstance(expected, tvm.tir.BufferLoad)
        kind_position, index_position = kind_index_positions
        result.append(
            (
                str(node.args[kind_position].value),
                int(node.args[index_position]),
                expected.buffer.name,
            )
        )

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return result


def _collect_generation_advances(func):
    advances = Counter()

    def visit(node):
        if isinstance(node, tvm.tir.BufferStore) and node.buffer.scope() == "local.var":
            advances[node.buffer.name] += 1

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return advances


def test_plan_dist_signals_infers_kinds_and_preserves_independent_state():
    func = multi_signal_kernel_factory.get_tir(32, 32, world_size=4)
    result = lower_to_device_tir(
        func,
        capture_before_passes="tl.PlanDistSignals",
        capture_passes=("tl.PlanDistSignals", "tl.InjectDistSync"),
    )

    before_plan = result.pass_snapshot("tl.PlanDistSignals", when="before").mod.script()
    assert before_plan.count('T.dist_signal_decl("auto"') == 2
    assert "T.dist_signal(" not in before_plan

    after_plan_func = _single_prim_func(result.pass_snapshot("tl.PlanDistSignals").mod)
    assert _signal_counts(after_plan_func) == {
        "sram_flagreg_inc": 1,
        "dram_flagreg_inc": 1,
        "sram_flagreg_value": 1,
        "dram_flagreg_value": 0,
        "sram_memory": 0,
        "dram_memory": 1,
    }
    after_plan_script = after_plan_func.script()
    assert "T.dist_signal_decl" not in after_plan_script
    assert 's0: T.handle = T.dist_signal("sram_flagreg_inc", 0)' in after_plan_script
    assert 's1: T.handle = T.dist_signal("sram_flagreg_value", 0)' in after_plan_script
    assert 'd0: T.handle = T.dist_signal("dram_flagreg_inc", 0)' in after_plan_script
    assert 'm0: T.handle = T.dist_signal("dram_memory", 0)' in after_plan_script

    device_func = _single_prim_func(result.device_mod)
    assert _signal_counts(device_func) == _signal_counts(after_plan_func)
    puts = _collect_leaf_signals(device_func, "tl.dist_put_", (3, 4), 5)
    waits = _collect_leaf_signals(device_func, "tl.dist_wait_signal_", (0, 1), 2)
    assert [(kind, index) for kind, index, _ in puts] == [
        ("sram_flagreg_inc", 0),
        ("sram_flagreg_value", 0),
        ("sram_flagreg_inc", 0),
        ("dram_flagreg_inc", 0),
        ("dram_memory", 0),
    ]
    assert [(kind, index) for kind, index, _ in waits] == [
        ("sram_flagreg_value", 0),
        ("sram_flagreg_inc", 0),
        ("dram_flagreg_inc", 0),
        ("dram_memory", 0),
    ]

    expected_by_signal = {(kind, index): name for kind, index, name in waits}
    generation_by_signal = {(kind, index): name for kind, index, name in puts}
    advances = _collect_generation_advances(device_func)
    assert advances[generation_by_signal[("sram_flagreg_inc", 0)]] == 2
    assert advances[generation_by_signal[("sram_flagreg_value", 0)]] == 1
    assert advances[generation_by_signal[("dram_flagreg_inc", 0)]] == 1
    assert advances[generation_by_signal[("dram_memory", 0)]] == 1
    assert advances[expected_by_signal[("sram_flagreg_inc", 0)]] == 2
    assert advances[expected_by_signal[("sram_flagreg_value", 0)]] == 1
    assert advances[expected_by_signal[("dram_flagreg_inc", 0)]] == 1
    assert advances[expected_by_signal[("dram_memory", 0)]] == 1


def test_plan_dist_signals_rejects_mixed_destination_scopes():
    func = mixed_scope_signal_kernel_factory.get_tir(32, 32, world_size=4)
    with pytest.raises(tvm.error.InternalError, match="inconsistent destination scopes"):
        lower_to_device_tir(func)


def test_plan_dist_signals_rejects_sram_flagreg_for_dram_destination():
    func = explicit_sram_signal_on_dram_kernel_factory.get_tir(32, 32, world_size=4)
    with pytest.raises(tvm.error.InternalError, match="explicitly requests sram_flagreg_inc"):
        lower_to_device_tir(func)


def test_plan_dist_signals_rejects_explicit_flagreg_capacity_overflow():
    func = too_many_explicit_dram_flagregs_kernel_factory.get_tir(world_size=4)
    with pytest.raises(tvm.error.InternalError, match="sram_flagreg_inc signal capacity exceeded"):
        lower_to_device_tir(func)


def test_plan_dist_signals_resolves_all_six_explicit_kinds():
    func = all_signal_kinds_kernel_factory.get_tir(32, 32, world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.PlanDistSignals")
    planned = _single_prim_func(result.pass_snapshot("tl.PlanDistSignals").mod)
    assert _signal_counts(planned) == {
        "sram_flagreg_inc": 1,
        "dram_flagreg_inc": 1,
        "sram_flagreg_value": 1,
        "dram_flagreg_value": 1,
        "sram_memory": 1,
        "dram_memory": 1,
    }
    script = planned.script()
    for kind in _signal_counts(planned):
        assert f'T.dist_signal("{kind}", 0)' in script


def test_plan_dist_signals_is_idempotent_after_resources_are_resolved():
    func = multi_signal_kernel_factory.get_tir(32, 32, world_size=4)
    target = tvm.target.Target(determine_target("sunmmio", return_object=True))
    mod = tvm.IRModule({"main": func.with_attr("target", target)})
    mod = tilelang.transform.ResolveSunmmioMeshSymbols()(mod)
    mod = tilelang.transform.InferSramScope()(mod)
    planned = tilelang.transform.PlanDistSignals()(mod)
    planned_again = tilelang.transform.PlanDistSignals()(planned)
    assert tvm.ir.structural_equal(planned_again, planned)
