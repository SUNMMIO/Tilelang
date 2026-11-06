import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tvm import tir
from testing.python.sunmmio.common.mx_view_validation import assert_mx_view_resolved


def test_mx_view_binding_tracks_cloned_allocation_identity():
    parent = tir.decl_buffer((32, 32), T.mxfp8, name="parent", scope="shared.rsram")
    field = tir.decl_buffer((32, 32), T.float8_e4m3fn, name="field", scope="shared.rsram")
    value = tir.call_intrin(
        "handle",
        tvm.ir.Op.get("tl.mx_field_handle"),
        parent.data,
        0,
        annotations={"tl.mx_parent": parent, "tl.mx_declaration": tir.Var("declaration", "int32")},
    )
    write = tir.BufferStore(field, tir.Cast(T.float8_e4m3fn, tir.FloatImm("float32", 1)), [0, 0])
    body = tir.Allocate(parent.data, parent.dtype, [32, 32], True, tir.LetStmt(field.data, value, tir.DeclBuffer(field, write)))
    loop = tir.For(tir.Var("i", "int32"), 0, 2, tir.ForKind.UNROLLED, body)
    mod = tvm.IRModule({"main": tir.PrimFunc([], loop).with_attr("global_symbol", "main")})
    expanded = tir.transform.UnrollLoop()(mod)
    normalized = tilelang.transform.NormalizeMXViews()(expanded)
    allocations = []
    bindings = []

    def collect(node):
        if isinstance(node, tir.Allocate):
            allocations.append(node.buffer_var)
        if isinstance(node, tir.Call) and node.op.same_as(tvm.ir.Op.get("tl.mx_field_handle")):
            bindings.append(node)

    tir.stmt_functor.post_order_visit(normalized["main"].body, collect)
    assert len(allocations) == len(bindings) == 2
    assert not allocations[0].same_as(allocations[1])
    for binding in bindings:
        assert binding.args[0].same_as(binding.annotations["tl.mx_parent"].data)
        assert any(binding.args[0].same_as(var) for var in allocations)
    assert tir.analysis.verify_ssa(normalized["main"])


@pytest.mark.parametrize("location", ["assume", "call", "loop", "function", "buffer"])
def test_mx_view_exit_contract_checks_metadata(location):
    parent = tir.decl_buffer((64, 64), T.mxfp8, scope="shared.rsram")
    shape = tir.call_intrin("int32", tvm.ir.Op.get("tl.mx_scale_extent"), parent.data, 0)
    body = tir.Evaluate(0)
    if location == "assume":
        body = tir.AttrStmt(shape > 0, "tl.assume", tir.StringImm("positive"), body)
    elif location == "call":
        body = tir.Evaluate(
            tir.Call("int32", tvm.ir.Op.get("tir.call_extern"), [tir.StringImm("example")], annotations={"nested": {"shape": [shape]}})
        )
    elif location == "loop":
        body = tir.For(tir.Var("i", "int32"), 0, 1, tir.ForKind.SERIAL, body, annotations={"nested": [shape]})
    elif location == "buffer":
        field = tir.decl_buffer((shape, 32), T.float8_e8m0fnu, scope="shared.rsram")
        body = tir.DeclBuffer(field, body)
    func = tir.PrimFunc([], body)
    if location == "function":
        func = func.with_attr("nested", {"shape": shape})
    with pytest.raises(AssertionError, match="Unresolved MX"):
        assert_mx_view_resolved(tvm.IRModule({"main": func}))
    with pytest.raises(tvm.error.InternalError, match="Unresolved MX"):
        tvm.ffi.get_global_func("tl.sunmmio.CheckMXViewResolved")(func)


@pytest.mark.parametrize("force_inline", [False, True])
def test_deferred_copy_metadata_follows_let_substitution(force_inline):
    from tilelang.engine.lower import lower
    from testing.python.sunmmio.common.compile_pipeline import target

    @target("sunmmio")
    def kernel():
        @T.prim_func
        def main():
            with T.Kernel() as cid:
                mx = T.alloc_shared((64, 64), T.mxfp8, scope="shared.rsram")
                data, scale = T.mx_view(mx)
                peer = T.alloc_shared((4, 32), T.float8_e8m0fnu)
                row = cid % 4
                alias = row + 1
                T.copy(scale, peer, annotations={"probe": {"index": alias, "shape": [scale.shape[0]]}})

        return main

    checked = []

    @tvm.instrument.pass_instrument
    class Check:
        def run_after_pass(self, mod, info):
            if info.name != "tl.SunmmioLayoutInference":
                return
            for func in mod.functions.values():
                bound = set(func.params)
                expressions = []

                def visit(node, bound=bound, expressions=expressions):
                    if isinstance(node, tir.LetStmt):
                        bound.add(node.var)
                    if isinstance(node, tir.For):
                        bound.add(node.loop_var)
                    if isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent":
                        bound.add(node.node.var)
                    if isinstance(node, tir.Call) and "probe" in node.annotations:
                        expressions.append(node.annotations["probe"]["index"])
                        assert int(node.annotations["probe"]["shape"][0]) == 4

                tir.stmt_functor.post_order_visit(func.body, visit)
                for expr in expressions:
                    used = set()
                    tir.stmt_functor.post_order_visit(expr, lambda node, used=used: used.add(node) if isinstance(node, tir.Var) else None)
                    assert used <= bound
                checked.extend(expressions)

    with tilelang.transform.PassContext(config={"tl.force_let_inline": force_inline}, instruments=[Check()]):
        lower(kernel(), target="sunmmio")
    assert checked


def test_mx_metadata_rewriting_preserves_folded_inline_calls():
    condition = tir.Var("condition", "bool")
    expr = tir.if_then_else(condition, tir.IntImm("int32", 17), tir.IntImm("int32", 23))
    body = tir.LetStmt(condition, tir.IntImm("bool", 1), tir.Evaluate(expr))
    mod = tvm.IRModule({"main": tir.PrimFunc([], body)})
    result = tilelang.transform.LetInline()(mod)
    assert isinstance(result["main"].body, tir.Evaluate)
    assert int(result["main"].body.value) == 17
