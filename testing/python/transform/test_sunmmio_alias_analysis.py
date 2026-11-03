"""Tests for the bounded pre-pipeline SunMMIO alias proof."""

from tilelang import tvm
from tilelang.engine import phase
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from tvm import tir


def _buffer_param(name):
    handle = tir.Var(name + "_handle", "handle")
    data = tir.Var(
        name,
        tvm.ir.PointerType(tvm.ir.PrimType("float32"), "global"),
    )
    buffer = tir.decl_buffer(
        (16,), "float32", name=name, data=data, scope="global"
    )
    return handle, buffer


def _make_module(*, target=SUNMMIO_TARGET_DESC, non_restrict=False):
    a_handle, a = _buffer_param("a")
    b_handle, b = _buffer_param("b")
    func = tir.PrimFunc(
        [a_handle, b_handle],
        tir.Evaluate(0),
        buffer_map={a_handle: a, b_handle: b},
    ).with_attr("target", tvm.target.Target(target))
    if non_restrict:
        func = func.with_attr("tl.non_restrict_params", [a_handle])
    return tvm.IRModule.from_expr(func)


def _main(mod):
    return next(
        func for func in mod.functions.values() if isinstance(func, tir.PrimFunc)
    )


def test_distinct_sunmmio_parameter_roots_prove_noalias():
    after = phase._infer_sunmmio_noalias(_make_module())
    func = _main(after)

    assert bool(func.attrs["tir.noalias"])
    assert bool(func.attrs[phase._SUNMMIO_ALIAS_PROOF_ATTR])
    assert str(func.attrs[phase._SUNMMIO_ALIAS_PROOF_REASON_ATTR]) == (
        "sunmmio_device_abi_distinct_parameter_roots"
    )


def test_non_restrict_parameter_fails_closed():
    after = phase._infer_sunmmio_noalias(
        _make_module(non_restrict=True)
    )
    assert "tir.noalias" not in _main(after).attrs


def test_hoisted_block_non_restrict_parameter_fails_closed():
    a_handle, a = _buffer_param("a")
    block = tir.Block(
        [],
        [],
        [],
        "root",
        tir.Evaluate(0),
        annotations={"tl.non_restrict_params": [a_handle]},
    )
    func = tir.PrimFunc(
        [a_handle],
        tir.BlockRealize([], tir.const(True, "bool"), block),
        buffer_map={a_handle: a},
    ).with_attr("target", tvm.target.Target(SUNMMIO_TARGET_DESC))
    mod = tvm.IRModule.from_expr(func)

    mod = phase.tilelang.transform.HoistNonRestrictParams()(mod)
    after = phase._infer_sunmmio_noalias(mod)
    assert "tir.noalias" not in _main(after).attrs


def test_opaque_handle_parameter_fails_closed():
    opaque = tir.Var("opaque", "handle")
    func = tir.PrimFunc([opaque], tir.Evaluate(0)).with_attr(
        "target", tvm.target.Target(SUNMMIO_TARGET_DESC)
    )

    after = phase._infer_sunmmio_noalias(tvm.IRModule.from_expr(func))
    assert "tir.noalias" not in _main(after).attrs


def test_shared_parameter_root_fails_closed():
    a_handle, a = _buffer_param("a")
    b_handle = tir.Var("b_handle", "handle")
    b = tir.decl_buffer(
        (8,), "float32", name="b", data=a.data, scope="global"
    )
    func = tir.PrimFunc(
        [a_handle, b_handle],
        tir.Evaluate(0),
        buffer_map={a_handle: a, b_handle: b},
    ).with_attr("target", tvm.target.Target(SUNMMIO_TARGET_DESC))

    after = phase._infer_sunmmio_noalias(tvm.IRModule.from_expr(func))
    assert "tir.noalias" not in _main(after).attrs


def test_non_sunmmio_target_is_unchanged():
    before = _make_module(target="llvm")
    after = phase._infer_sunmmio_noalias(before)
    assert tvm.ir.structural_equal(before, after)


def test_explicit_noalias_setting_is_preserved():
    before = _make_module()
    func = _main(before).with_attr("tir.noalias", False)
    before = tvm.IRModule.from_expr(func)

    after = phase._infer_sunmmio_noalias(before)
    assert not bool(_main(after).attrs["tir.noalias"])
    assert phase._SUNMMIO_ALIAS_PROOF_ATTR not in _main(after).attrs
