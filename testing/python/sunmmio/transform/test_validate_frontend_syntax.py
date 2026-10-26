import pytest

import tilelang.language as T
import tilelang.transform as tl_transform
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize
from tilelang.utils.target import determine_target
from tvm.script import tir as TVM_T


def _bind(func, target):
    return tvm.tir.transform.BindTarget(target)(tvm.IRModule({"main": func}))


def _validate(func, target_name="Sunmmio"):
    target = determine_target(target_name, return_object=True)
    with tvm.target.Target(target):
        return tl_transform.ValidateSunmmioFrontendSyntax()(_bind(func, target))


def explicit_block_where():
    @T.prim_func
    def main():
        with T.Kernel(T.mesh_ncores()) as cid, T.block("conditional"):
            T.where(cid < 2)
            T.evaluate(0)

    return main


def root_block_where():
    @T.prim_func
    def main():
        with T.Kernel(T.mesh_ncores()) as cid:
            T.where(cid < 2)
            T.evaluate(0)

    return main


def constant_true_where():
    @T.prim_func
    def main():
        with T.Kernel(T.mesh_ncores()), T.block("conditional"):
            T.where(True)
            T.evaluate(0)

    return main


def unmarked_tvm_where():
    @T.prim_func
    def main():
        with T.Kernel(T.mesh_ncores()) as cid:
            send = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            recv = T.alloc_shared((32, 32), "bfloat16")
            with T.block("sender_only"):
                TVM_T.where(cid == 0)
                T.comm.put(send, recv, (0, 0), (0, 1))

    return main


def block_without_where():
    @T.prim_func
    def main():
        with T.Kernel(T.mesh_ncores()), T.block("work"):
            T.evaluate(0)

    return main


def explicit_if():
    @T.prim_func
    def main():
        with T.Kernel(T.mesh_ncores()) as cid:
            if cid < 2:
                T.evaluate(0)

    return main


@pytest.mark.parametrize(
    "factory,block_name",
    [
        (explicit_block_where, "conditional"),
        (root_block_where, "tilelang_root"),
        (constant_true_where, "conditional"),
        (unmarked_tvm_where, "sender_only"),
    ],
)
def test_sunmmio_rejects_explicit_where(factory, block_name):
    with pytest.raises(
        tvm.error.InternalError,
        match=rf"Sunmmio frontend does not support T.where.*Block '{block_name}'",
    ):
        _validate(factory())


@pytest.mark.parametrize("factory", [block_without_where, explicit_if])
def test_sunmmio_accepts_supported_control_flow(factory):
    _validate(factory())


@pytest.mark.parametrize("factory", [explicit_block_where, unmarked_tvm_where])
def test_non_sunmmio_target_keeps_where(factory):
    target = tvm.target.Target("llvm")
    func = factory()
    tl_transform.ValidateSunmmioFrontendSyntax()(_bind(func, target))


@pytest.mark.parametrize("factory", [explicit_block_where, unmarked_tvm_where])
def test_sunmmio_lowering_rejects_where_before_legalization(factory):
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": factory()})
    with (
        tvm.target.Target(target),
        pytest.raises(
            tvm.error.InternalError,
            match="Sunmmio frontend does not support T.where",
        ),
    ):
        LowerAndLegalize(mod, target)
