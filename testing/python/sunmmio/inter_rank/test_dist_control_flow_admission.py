"""Temporary front-end admission boundary for explicit SunMMIO jumps."""

import pytest
import tilelang
import tilelang.language as T
from tilelang import tvm

from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import base_mod


@tilelang.jit(target="sunmmio")
def jump_kernel(operation="return", form="for", use_dist=False, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, enabled: T.int32):
        with T.Kernel():
            if use_dist:
                signal = T.dist.signal()
                T.dist.put_signal(signal, (rank_id + 1) % world_size, submit=True)
                T.dist.wait_signal(signal)
            index = T.alloc_var(T.int32, init=0)
            if form == "top":
                T.evaluate(tvm.tir.ret(0))
            elif form == "branch":
                if enabled != 0:
                    T.evaluate(tvm.tir.ret(0))
            elif form == "while":
                while index < 2:
                    index += 1
                    if operation == "return":
                        T.evaluate(tvm.tir.ret(0))
                    elif operation == "break":
                        break
                    else:
                        continue
            else:
                for _ in T.serial(2):
                    if enabled != 0:
                        if operation == "return":
                            T.evaluate(tvm.tir.ret(0))
                        elif operation == "break":
                            break
                        else:
                            continue

    return main


@pytest.mark.parametrize(
    "operation,form",
    [
        ("return", "top"),
        ("return", "branch"),
        ("return", "for"),
        ("return", "while"),
        ("break", "for"),
        ("break", "while"),
        ("continue", "for"),
        ("continue", "while"),
    ],
)
@pytest.mark.parametrize("use_dist", [False, True])
def test_sunmmio_explicit_jumps_are_rejected_before_dist_lowering(operation, form, use_dist):
    func = jump_kernel.get_tir(operation, form, use_dist, world_size=2 if use_dist else 1)
    with pytest.raises(tvm.error.InternalError, match="explicit return/break/continue.*supported yet"):
        lower_to_device_tir(func)


def _jump_mod(operation, *, marked=True, thread_scope=False, unreachable=False, host=False, target=None):
    op = tvm.ir.Op.get("tir." + operation)
    jump = tvm.tir.Evaluate(tvm.tir.Call("int32", op, [tvm.tir.const(0)] if operation == "ret" else []))
    body = tvm.tir.IfThenElse(tvm.tir.const(False, "bool"), jump, None) if unreachable else jump
    block = tvm.tir.Block([], [], [], "kernel", body, annotations={"tilelang.is_sunmmio_kernel_frame": True} if marked else {})
    body = tvm.tir.BlockRealize([], True, block)
    if thread_scope:
        axis = tvm.tir.IterVar(tvm.ir.Range(0, 16), tvm.tir.Var("core", "int32"), tvm.tir.IterVar.ThreadIndex, "blockIdx.x")
        body = tvm.tir.AttrStmt(axis, "thread_extent", 16, body)
    if host:
        # A host-side jump followed by a kernel with a normal body.
        body = tvm.tir.SeqStmt(
            [
                jump,
                tvm.tir.BlockRealize(
                    [],
                    True,
                    tvm.tir.Block([], [], [], "kernel", tvm.tir.Evaluate(0), annotations={"tilelang.is_sunmmio_kernel_frame": True}),
                ),
            ]
        )
    func = tvm.tir.PrimFunc([], body)
    if target is None:
        return base_mod(func)
    return tvm.IRModule({"main": func.with_attr("target", tvm.target.Target(target))})


@pytest.mark.parametrize("operation", ["ret", "break_loop", "continue_loop"])
@pytest.mark.parametrize("thread_scope", [False, True])
def test_explicit_jump_admission_does_not_analyze_reachability(operation, thread_scope):
    mod = _jump_mod(operation, marked=not thread_scope, thread_scope=thread_scope, unreachable=True)
    with pytest.raises(tvm.error.InternalError, match="explicit return/break/continue"):
        tilelang.transform.LowerDistCollectives()(mod)


@pytest.mark.parametrize("operation", ["ret", "break_loop", "continue_loop"])
def test_explicit_jump_admission_preserves_host_scope_and_other_targets(operation):
    for mod in (
        _jump_mod(operation, host=True),
        _jump_mod(operation, marked=False),
        _jump_mod(operation, target="llvm"),
        _jump_mod(operation, target="cuda"),
    ):
        after = tilelang.transform.LowerDistCollectives()(mod)
        assert tvm.ir.structural_equal(after, mod)


@tilelang.jit(target="sunmmio")
def structured_kernel(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, enabled: T.int32):
        with T.Kernel():
            index = T.alloc_var(T.int32, init=0)
            for _ in T.serial(2):
                if enabled != 0:
                    index += 1
            while index < 3:
                index += 1

    return main


def test_structured_control_flow_and_factory_return_remain_supported():
    result = lower_to_device_tir(structured_kernel.get_tir())
    assert result.device_mod.get_global_vars()
