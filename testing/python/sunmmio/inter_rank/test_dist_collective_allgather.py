"""Static Rank all-gather frontend and lowering tests."""

from collections import Counter

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.utils.target import determine_target
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir


def _dst_shape(world_size, axis):
    if axis is None:
        return world_size, 4, 8
    if axis == 0:
        return world_size * 4, 8
    return 4, world_size * 8


@tilelang.jit(target="sunmmio")
def allgather_kernel_factory(axis=None, signal_kind=None, group=None, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.bfloat16)
            dst = T.alloc_shared(_dst_shape(world_size, axis), T.bfloat16)
            signal = T.dist.signal(kind=signal_kind)
            T.dist.all_gather(src, dst, signal=signal, axis=axis, group=group)
            T.dist.wait_signal(signal, dst=dst)

    return main


@tilelang.jit(target="sunmmio")
def invalid_allgather_shape_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 7), T.bfloat16)
            signal = T.dist.signal()
            T.dist.all_gather(src, dst, signal=signal)

    return main


@tilelang.jit(target="sunmmio")
def mismatched_allgather_dtype_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 8), T.float32)
            signal = T.dist.signal()
            T.dist.all_gather(src, dst, signal=signal)

    return main


@tilelang.jit(target="sunmmio")
def overlapping_allgather_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            storage = T.alloc_shared((world_size * 4, 8), T.bfloat16)
            signal = T.dist.signal()
            T.dist.all_gather(storage[0:4, 0:8], storage, signal=signal, axis=0)

    return main


@tilelang.jit(target="sunmmio")
def allgather_scope_combinations_kernel_factory(world_size: int = 1):
    placement = T.placement.replicated()

    @T.prim_func
    def main(
        A: T.MeshTensor((32, 32), placement, T.bfloat16),  # type: ignore
        B: T.MeshTensor((world_size, 32, 32), placement, T.bfloat16),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            src = T.alloc_shared((32, 32), T.bfloat16)
            dst = T.alloc_shared((world_size, 32, 32), T.bfloat16)
            signal0 = T.dist.signal()
            signal1 = T.dist.signal()
            signal2 = T.dist.signal()
            signal3 = T.dist.signal()
            T.dist.all_gather(src, dst, signal=signal0)
            T.dist.wait_signal(signal0, dst=dst)
            T.dist.all_gather(src, B, signal=signal1)
            T.dist.wait_signal(signal1, dst=B)
            T.dist.all_gather(A, dst, signal=signal2)
            T.dist.wait_signal(signal2, dst=dst)
            T.dist.all_gather(A, B, signal=signal3)
            T.dist.wait_signal(signal3, dst=B)

    return main


@tilelang.jit(target="sunmmio")
def repeated_allgather_signal_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32, 32), T.bfloat16)
            dst = T.alloc_shared((world_size, 32, 32), T.bfloat16)
            signal = T.dist.signal()
            T.dist.all_gather(src, dst, signal=signal)
            T.dist.wait_signal(signal, dst=dst)
            T.dist.all_gather(src, dst, signal=signal)
            T.dist.wait_signal(signal, dst=dst)

    return main


@tilelang.jit(target="sunmmio")
def allgather_signal_group_member_kernel_factory(
    signal_kind=None,
    world_size: int = 1,
):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            signals = T.dist.signals(1, kind=signal_kind)
            T.dist.all_gather(src, dst, signal=signals[0])
            T.dist.wait_signal(signals[0], dst=dst)

    return main


def _collect_op_names(func_or_mod):
    names = []
    funcs = func_or_mod.functions.values() if isinstance(func_or_mod, tvm.IRModule) else (func_or_mod,)

    def visit(node):
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op):
            names.append(node.op.name)

    for func in funcs:
        if isinstance(func, tvm.tir.PrimFunc):
            tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return names


def _collect_dist_put_kinds(mod):
    kinds = []

    def visit(node):
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.dist_put_":
            kinds.append(str(node.args[3].value))

    for func in mod.functions.values():
        if isinstance(func, tvm.tir.PrimFunc):
            tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return kinds


def _lower_collectives(func):
    target = tvm.target.Target(determine_target("sunmmio", return_object=True))
    mod = tvm.IRModule({"main": func.with_attr("target", target)})
    mod = tilelang.transform.ResolveSunmmioMeshSymbols()(mod)
    return tilelang.transform.LowerDistCollectives()(mod)


@pytest.mark.parametrize("axis", [None, 0, -1])
def test_allgather_frontend_emits_one_high_level_tileop(axis):
    func = allgather_kernel_factory.get_tir(axis=axis, world_size=4)
    names = _collect_op_names(func)

    assert names.count("tl.tileop.dist_allgather") == 1
    assert "tl.tileop.dist_put" not in names
    assert names.count("tl.tileop.dist_wait_signal") == 1
    assert "tl.dist_wait_send" not in names


@pytest.mark.parametrize(
    "axis, slot",
    [
        (None, "dst[rank_id, 0, 0]"),
        (0, "dst[rank_id * 4, 0]"),
        (-1, "dst[0, rank_id * 8]"),
    ],
)
def test_lower_dist_collectives_expands_direct_allgather(axis, slot):
    func = allgather_kernel_factory.get_tir(axis=axis, world_size=4)
    lowered = _lower_collectives(func)
    names = _collect_op_names(lowered)
    script = lowered.script()

    assert "tl.tileop.dist_allgather" not in names
    assert names.count("tl.tileop.copy") == 1
    assert names.count("tl.tileop.dist_put") == 3
    assert names.count("tl.tileop.dist_wait_signal") == 1
    assert names.count("tl.dist_wait_send") == 1
    assert script.count(slot) == 4
    for offset in range(1, 4):
        assert f"(rank_id + {offset}) % 4" in script

    lowered_again = tilelang.transform.LowerDistCollectives()(lowered)
    assert tvm.ir.structural_equal(lowered_again, lowered)


def test_allgather_collective_does_not_insert_receiver_wait():
    func = allgather_kernel_factory.get_tir(world_size=4)
    before = _collect_op_names(func).count("tl.tileop.dist_wait_signal")
    lowered = _lower_collectives(func)
    assert _collect_op_names(lowered).count("tl.tileop.dist_wait_signal") == before


@pytest.mark.parametrize("axis", [None, 0, -1])
def test_allgather_all_axes_reach_device_tir(axis):
    func = allgather_kernel_factory.get_tir(axis=axis, world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistCollectives")

    collective_names = _collect_op_names(result.pass_snapshot("tl.LowerDistCollectives").mod)
    device_names = _collect_op_names(result.device_mod)
    assert "tl.tileop.dist_allgather" not in collective_names
    assert device_names.count("tl.dist_put_") == 3
    assert device_names.count("tl.dist_wait_signal_") == 1
    assert device_names.count("tl.dist_wait_send") == 1


def test_allgather_supports_all_p2p_scope_combinations():
    func = allgather_scope_combinations_kernel_factory.get_tir(world_size=2)
    result = lower_to_device_tir(func, capture_passes="tl.PlanDistSignals")
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]

    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 2
    assert int(planned.attrs["tl.dist.signal_counts"]["dram_flagreg_inc"]) == 2
    assert Counter(_collect_dist_put_kinds(result.device_mod)) == {
        "sram_flagreg_inc": 2,
        "dram_flagreg_inc": 2,
    }


def test_allgather_reuses_one_signal_across_collectives():
    func = repeated_allgather_signal_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func)
    device_names = _collect_op_names(result.device_mod)

    assert device_names.count("tl.dist_put_") == 6
    assert device_names.count("tl.dist_wait_signal_") == 2
    assert device_names.count("tl.dist_wait_send") == 2


def test_allgather_accepts_explicit_inc_signal_group_member():
    func = allgather_signal_group_member_kernel_factory.get_tir(
        signal_kind=T.dist.SignalKind.SRAM_FLAGREG_INC,
        world_size=4,
    )
    result = lower_to_device_tir(func, capture_passes="tl.PlanDistSignals")
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]

    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 1
    assert 'T.dist_signal_group("sram_flagreg_inc", 0, 1)' in planned.script()
    assert _collect_op_names(result.device_mod).count("tl.dist_put_") == 3


def test_allgather_rejects_automatic_signal_group_member():
    with pytest.raises(ValueError, match="single automatic signal"):
        allgather_signal_group_member_kernel_factory.get_tir(world_size=4)


def test_allgather_world_size_one_is_rejected_by_collective_pass():
    func = allgather_kernel_factory.get_tir(world_size=1)
    with pytest.raises(tvm.error.InternalError, match="world_size > 1"):
        _lower_collectives(func)


def test_allgather_rejects_nonempty_group():
    with pytest.raises(NotImplementedError, match="group is reserved"):
        allgather_kernel_factory.get_tir(group=(0, 1), world_size=4)


def test_allgather_rejects_unsupported_axis():
    with pytest.raises(ValueError, match="only supports axis=None, axis=0, or axis=-1"):
        allgather_kernel_factory.get_tir(axis=1, world_size=4)


def test_allgather_rejects_value_signal():
    with pytest.raises(ValueError, match="automatic or INC flagreg signal"):
        allgather_kernel_factory.get_tir(
            signal_kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE,
            world_size=4,
        )


def test_allgather_rejects_invalid_destination_shape():
    with pytest.raises((AssertionError, ValueError), match="shape|extents"):
        invalid_allgather_shape_kernel_factory.get_tir(world_size=4)


def test_allgather_rejects_mismatched_dtype():
    with pytest.raises(TypeError, match="dtypes must match"):
        mismatched_allgather_dtype_kernel_factory.get_tir(world_size=4)


def test_allgather_rejects_overlapping_storage():
    with pytest.raises(ValueError, match="overlapping source and destination"):
        overlapping_allgather_kernel_factory.get_tir(world_size=4)
