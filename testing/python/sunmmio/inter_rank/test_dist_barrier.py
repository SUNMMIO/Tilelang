"""Rank barrier frontend and lowering tests."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.utils.target import determine_target
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir


@tilelang.jit(target="sunmmio")
def barrier_kernel_factory(signal_kind=None, explicit=False, group=None, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            if explicit:
                signal = T.dist.signal(kind=signal_kind)
                T.dist.barrier(group=group, signal=signal)
            else:
                T.dist.barrier(group=group)

    return main


@tilelang.jit(target="sunmmio")
def repeated_barrier_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            signal = T.dist.signal()
            T.dist.barrier(signal=signal)
            T.dist.barrier(signal=signal)

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


def _lower_collectives(func):
    target = tvm.target.Target(determine_target("sunmmio", return_object=True))
    mod = tvm.IRModule({"main": func.with_attr("target", target)})
    mod = tilelang.transform.ResolveSunmmioMeshSymbols()(mod)
    return tilelang.transform.LowerDistCollectives()(mod)


def test_barrier_frontend_creates_internal_increment_signal():
    func = barrier_kernel_factory.get_tir(world_size=4)
    names = _collect_op_names(func)

    assert names.count("tl.dist_barrier") == 1
    assert names.count("tl.dist_signal_decl") == 1
    assert 'T.dist_signal_decl("sram_flagreg_inc"' in func.script()


def test_lower_barrier_emits_all_peer_arrivals_and_wait():
    func = barrier_kernel_factory.get_tir(world_size=4)
    lowered = _lower_collectives(func)
    names = _collect_op_names(lowered)
    script = lowered.script()

    assert "tl.dist_barrier" not in names
    assert names.count("tl.dist_wait_send") == 1
    assert names.count("tl.dist_expect") == 1
    assert names.count("tl.dist_signal_put") == 3
    assert names.count("tl.dist_wait_barrier") == 1
    assert "T.dist_expect(" in script and ", 3)" in script
    assert script.index("T.dist_wait_send()") < script.index("T.dist_signal_put")
    assert script.rindex("T.dist_signal_put") < script.index("T.dist_wait_barrier")


def test_barrier_reaches_stable_device_leaves():
    func = barrier_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(
        func,
        capture_passes=(
            "tl.LowerDistCollectives",
            "tl.PlanDistSignals",
            "tl.LowerDistCommunication",
            "tl.InjectDistSync",
        ),
    )

    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 1

    lower_comm = result.pass_snapshot("tl.LowerDistCommunication").mod
    lower_names = _collect_op_names(lower_comm)
    assert lower_names.count("tl.dist_signal_put_") == 3
    assert lower_names.count("tl.dist_wait_barrier_") == 1
    assert lower_names.count("tl.dist_expect_") == 1
    assert "tl.dist_signal_put" not in lower_names
    assert "tl.dist_wait_barrier" not in lower_names

    injected = result.pass_snapshot("tl.InjectDistSync").mod
    injected_names = _collect_op_names(injected)
    injected_script = injected.script()
    assert "tl.dist_expect_" not in injected_names
    assert injected_names.count("tl.dist_signal_put_") == 3
    assert injected_names.count("tl.dist_wait_barrier_") == 1
    assert injected_script.count("_generation_1[0] =") == 3
    assert "_expect_1[0] =" in injected_script
    assert _collect_op_names(result.device_mod).count("tl.dist_wait_barrier_") == 1


def test_barrier_accepts_explicit_dram_increment_signal():
    func = barrier_kernel_factory.get_tir(
        explicit=True,
        signal_kind=T.dist.SignalKind.DRAM_FLAGREG_INC,
        world_size=4,
    )
    result = lower_to_device_tir(func, capture_passes="tl.PlanDistSignals")
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]

    assert int(planned.attrs["tl.dist.signal_counts"]["dram_flagreg_inc"]) == 1
    assert '"dram_flagreg_inc"' in result.device_mod.script()


def test_repeated_barrier_accumulates_two_epochs():
    func = repeated_barrier_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.InjectDistSync")
    injected = result.pass_snapshot("tl.InjectDistSync").mod
    names = _collect_op_names(injected)

    assert names.count("tl.dist_signal_put_") == 6
    assert names.count("tl.dist_wait_barrier_") == 2
    assert injected.script().count("_expect_1[0] =") == 2


def test_barrier_rejects_world_size_one():
    func = barrier_kernel_factory.get_tir(world_size=1)
    with pytest.raises(tvm.error.InternalError, match="world_size > 1"):
        _lower_collectives(func)


def test_barrier_rejects_nonempty_group():
    with pytest.raises(NotImplementedError, match="group is reserved"):
        barrier_kernel_factory.get_tir(group=(0, 1), world_size=4)


def test_barrier_rejects_non_increment_signal():
    with pytest.raises(ValueError, match="INC flagreg"):
        barrier_kernel_factory.get_tir(
            explicit=True,
            signal_kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE,
            world_size=4,
        )
