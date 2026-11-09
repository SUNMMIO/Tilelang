import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tvm import tir
from testing.python.sunmmio.common.mx_view_validation import assert_mx_view_resolved


def parent_descriptor_module(bits, case, storage):
    dtype = T.mxfp8 if bits == 8 else T.mxfp4
    owner = tir.decl_buffer((64, 64), dtype, name="owner", scope="shared.rsram")
    shape, alias_dtype = (64, 64), dtype
    kwargs = {"data": owner.data, "name": "rebuilt_parent", "elem_offset": 0}
    if case == "strided":
        kwargs["strides"] = (1, 64)
    elif case == "compact":
        kwargs["strides"] = (64, 1)
    elif case == "constant_expr":
        shape = (tir.Add(tir.IntImm("int32", 60), tir.IntImm("int32", 4)), 64)
        kwargs["strides"] = (tir.Mul(tir.IntImm("int32", 8), tir.IntImm("int32", 8)), 1)
        kwargs["elem_offset"] = tir.Sub(tir.IntImm("int32", 1), tir.IntImm("int32", 1))
    elif case == "reshape":
        shape = (32, 128)
    elif case == "offset":
        kwargs["elem_offset"] = 1
    elif case == "axis_separators":
        kwargs["axis_separators"] = [1]
    elif case == "alignment":
        kwargs["data_alignment"] = 128
    elif case == "offset_factor":
        kwargs["offset_factor"] = 8
    elif case == "buffer_type":
        kwargs.update(strides=(64, 1), buffer_type="auto_broadcast")
    elif case == "dtype":
        alias_dtype = T.mxfp4 if bits == 8 else T.mxfp8
    elif case == "pointer_type":
        kwargs["data"] = tir.Var("old_storage", tvm.ir.PointerType(tvm.ir.PrimType("uint8"), "shared.rsram"))
    elif case in ("scope_stale", "scope_inference", "dynamic_scope", "cloned_var"):
        scope = {"scope_stale": "shared.asram", "scope_inference": "shared", "dynamic_scope": "shared.dyn", "cloned_var": "shared.rsram"}[
            case
        ]
        kwargs["data"] = tir.Var("old_owner", tvm.ir.PointerType(tvm.ir.PrimType(dtype), scope))
    alias = tir.decl_buffer(shape, alias_dtype, **kwargs)
    declaration = tir.Var("declaration", "int32")
    body = tir.Evaluate(0)
    for role, field_dtype in ((1, T.float8_e8m0fnu), (0, T.float8_e4m3fn if bits == 8 else T.float4_e2m1fn)):
        field = tir.decl_buffer(shape, field_dtype, name=f"field_{role}", scope="shared.rsram")
        binding = tir.Call(
            "handle",
            tvm.ir.Op.get("tl.mx_field_handle"),
            [owner.data, role],
            annotations={"tl.mx_parent": alias, "tl.mx_declaration": declaration},
        )
        body = tir.LetStmt(field.data, binding, tir.DeclBuffer(field, body))
    if storage == "block":
        body = tir.BlockRealize([], True, tir.Block([], [], [], "root", tir.DeclBuffer(alias, body), alloc_buffers=[owner]))
    else:
        if storage == "allocate":
            body = tir.DeclBuffer(owner, tir.DeclBuffer(alias, body))
        elif storage == "outer_alias":
            body = tir.DeclBuffer(alias, tir.DeclBuffer(owner, body))
        body = tir.Allocate(owner.data, owner.dtype, [64, 64], True, body)
    return tvm.IRModule({"main": tir.PrimFunc([], body)}), owner


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("storage", ["block", "allocate", "outer_alias", "no_owner_decl"])
@pytest.mark.parametrize("case", ["strided", "reshape", "offset", "axis_separators", "buffer_type", "dtype", "scope_stale", "pointer_type"])
def test_mx_parent_descriptor_mismatch_rejected_before_rebinding(bits, storage, case):
    mod, _ = parent_descriptor_module(bits, case, storage)
    original = tvm.ir.save_json(mod)
    # Also exercise Normalize without declaration validation: repair passes
    # must not erase an invalid descriptor before a later validation pass.
    for validate in (False, True):
        with pytest.raises(tvm.error.InternalError, match="T.mx_view.*(parent|allocation|offset)"):
            tilelang.transform.NormalizeMXViews(validate=validate)(mod)
        assert tvm.ir.save_json(mod) == original


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("storage", ["block", "allocate", "outer_alias"])
@pytest.mark.parametrize("case", ["alignment", "offset_factor"])
def test_mx_parent_rebinding_preserves_storage_contract(bits, storage, case):
    mod, _ = parent_descriptor_module(bits, case, storage)
    with pytest.raises(tvm.error.InternalError, match="T.mx_view.*parent"):
        tilelang.transform.NormalizeMXViews(validate=True)(mod)


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("case,attribute,value", [("alignment", "data_alignment", 128), ("offset_factor", "offset_factor", 8)])
def test_mx_parent_allocate_without_descriptor_retains_contract(bits, case, attribute, value):
    # Allocate itself has no alignment/offset-factor fields to compare against;
    # normalization must preserve this metadata, not replace it with defaults.
    mod, _ = parent_descriptor_module(bits, case, "no_owner_decl")
    result = tilelang.transform.NormalizeMXViews(validate=True)(mod)
    parents = []

    def collect(node):
        if isinstance(node, tir.Call) and node.op.same_as(tvm.ir.Op.get("tl.mx_field_handle")):
            parents.append(node.annotations["tl.mx_parent"])

    tir.stmt_functor.post_order_visit(result["main"].body, collect)
    assert len(parents) == 2
    assert all(getattr(parent, attribute) == value for parent in parents)


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("storage", ["block", "allocate", "no_owner_decl"])
@pytest.mark.parametrize("case", ["equivalent", "compact", "constant_expr", "cloned_var", "scope_inference", "dynamic_scope"])
def test_mx_parent_equivalent_descriptor_rebinding(bits, storage, case):
    mod, owner = parent_descriptor_module(bits, case, storage)
    result = tilelang.transform.NormalizeMXViews(validate=True)(mod)
    bindings = []

    def collect(node):
        if isinstance(node, tir.Call) and node.op.same_as(tvm.ir.Op.get("tl.mx_field_handle")):
            bindings.append(node)

    tir.stmt_functor.post_order_visit(result["main"].body, collect)
    assert len(bindings) == 2
    for call in bindings:
        parent = call.annotations["tl.mx_parent"]
        assert call.args[0].same_as(owner.data)
        assert parent.data.same_as(owner.data)
        assert parent.scope() == "shared.rsram"
        analyzer = tvm.arith.Analyzer()
        assert all(analyzer.can_prove_equal(extent, 64) for extent in parent.shape)
        assert parent.dtype == owner.dtype
        assert not parent.strides or all(analyzer.can_prove_equal(a, b) for a, b in zip(parent.strides, (64, 1)))
    assert bindings[0].annotations["tl.mx_parent"].same_as(bindings[1].annotations["tl.mx_parent"])
    assert tvm.ir.structural_equal(result, tilelang.transform.NormalizeMXViews(validate=True)(result))


def test_mx_parent_metadata_cannot_rebind_another_live_allocation():
    mod, _ = parent_descriptor_module(8, "cloned_var", "block")
    bindings = []
    tir.stmt_functor.post_order_visit(
        mod["main"].body,
        lambda node: bindings.append(node) if isinstance(node, tir.Call) and node.op.same_as(tvm.ir.Op.get("tl.mx_field_handle")) else None,
    )
    stale = bindings[0].annotations["tl.mx_parent"]
    body = tir.Allocate(stale.data, stale.dtype, [64, 64], True, mod["main"].body)
    mod = tvm.IRModule({"main": mod["main"].with_body(body)})
    with pytest.raises(tvm.error.InternalError, match="different live allocation"):
        tilelang.transform.NormalizeMXViews()(mod)


@pytest.mark.parametrize("with_mx_view", [False, True])
def test_normalize_mx_views_keeps_unrelated_strided_aliases(with_mx_view):
    owner = tir.decl_buffer((64, 64), "bfloat16", scope="shared.rsram")
    alias = tir.decl_buffer((64, 64), "bfloat16", data=owner.data, strides=(1, 64))
    body = tir.Evaluate(tir.BufferLoad(alias, [0, 1]))
    if with_mx_view:
        mx_mod, _ = parent_descriptor_module(8, "equivalent", "block")
        body = tir.SeqStmt([body, mx_mod["main"].body])
    body = tir.DeclBuffer(alias, body)
    body = tir.BlockRealize([], True, tir.Block([], [], [], "root", body, alloc_buffers=[owner]))
    mod = tvm.IRModule({"main": tir.PrimFunc([], body)})
    normalized = tilelang.transform.NormalizeMXViews(validate=True)(mod)
    aliases = []
    tir.stmt_functor.post_order_visit(
        normalized["main"].body, lambda node: aliases.append(node.buffer) if isinstance(node, tir.BufferLoad) else None
    )
    assert len(aliases) == 1 and aliases[0].same_as(alias)


def test_mx_parent_validation_keeps_unrelated_untyped_allocation():
    mod, _ = parent_descriptor_module(8, "equivalent", "block")
    # Generic TIR Allocate permits handle Vars without a pointer annotation.
    # Only the MX parent needs a Buffer descriptor for this validation.
    scratch = tir.Var("scratch", "handle")
    body = tir.Allocate(scratch, "float32", [16], True, mod["main"].body)
    result = tilelang.transform.NormalizeMXViews(validate=True)(tvm.IRModule({"main": mod["main"].with_body(body)}))
    assert result["main"].body.buffer_var.same_as(scratch)


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
