import pytest

import tilelang.language as T
import tilelang.transform as tl_transform
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize
from tilelang.utils.target import determine_target


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


def test_non_sunmmio_target_keeps_explicit_where():
    target = tvm.target.Target("llvm")
    func = explicit_block_where()
    tl_transform.ValidateSunmmioFrontendSyntax()(_bind(func, target))


def test_sunmmio_lowering_rejects_explicit_where_before_legalization():
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": explicit_block_where()})
    with (
        tvm.target.Target(target),
        pytest.raises(
            tvm.error.InternalError,
            match="Sunmmio frontend does not support T.where",
        ),
    ):
        LowerAndLegalize(mod, target)
