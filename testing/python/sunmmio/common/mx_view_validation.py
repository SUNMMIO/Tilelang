"""Structural checks for compiler-only MX placeholders, including metadata."""

from tilelang import tvm


def assert_mx_view_resolved(module):
    forbidden = {"tl.mx_scale_extent", "tl.mx_deferred_copy", "tl.mx_copy_operand"}
    seen = set()

    def inspect(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return
        if isinstance(value, tvm.ir.container.Array):
            for item in value:
                inspect(item)
            return
        if isinstance(value, tvm.ir.container.Map):
            for key, item in value.items():
                inspect(key)
                inspect(item)
            return
        if not isinstance(value, tvm.runtime.Object):
            return
        identity = tvm.runtime.Object.__hash__(value)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(value, tvm.tir.Call) and isinstance(value.op, tvm.ir.Op):
            assert value.op.name not in forbidden, f"Unresolved MX operation: {value.op.name}"
        # Reflection exposes AttrStmt.node, Buffer shapes and nested annotations
        # that the ordinary statement visitor does not walk.
        info = type(value).__tvm_ffi_type_info__
        while info is not None:
            for field in info.fields:
                inspect(field.getter(value))
            info = info.parent_type_info

    inspect(module)
