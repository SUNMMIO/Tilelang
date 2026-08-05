from types import SimpleNamespace

import pytest

import tilelang.testing
from tilelang import tvm as tvm
from tilelang import language as T
from tilelang.language.allocate import _resolve_allocation_shape
import tilelang.language.frame as frame


def test_resolve_let_bound_expr_expands_transitive_bindings():
    a = tvm.tir.Var("a", "int32")
    b = tvm.tir.Var("b", "int32")
    stack = frame._get_let_stack()
    initial_depth = len(stack)

    try:
        stack.push(SimpleNamespace(var=b, value=tvm.tir.IntImm("int32", 5)))
        stack.push(SimpleNamespace(var=a, value=b))
        resolved = frame.resolve_let_bound_expr(a + b)
    finally:
        while len(stack) > initial_depth:
            stack.pop()

    tvm.ir.assert_structural_equal(resolved, tvm.tir.IntImm("int32", 10))


def test_resolve_let_bound_expr_preserves_memory_backed_binding():
    shape = tvm.tir.decl_buffer((1,), "int32", name="shape")
    extent = tvm.tir.Var("extent", "int32")
    stack = frame._get_let_stack()
    initial_depth = len(stack)

    try:
        stack.push(SimpleNamespace(var=extent, value=shape[0]))
        resolved = frame.resolve_let_bound_expr(extent)
        assert resolved.same_as(extent)
        with pytest.raises(ValueError, match="non-invariant let binding"):
            _resolve_allocation_shape((extent,))
    finally:
        while len(stack) > initial_depth:
            stack.pop()


@tilelang.testing.requires_cuda
def test_let_vectorize_load():
    @T.prim_func
    def main(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (16, 16), dtype=T.float32, align=16)

        for _blockIdx in T.thread_binding(1, thread="blockIdx.x"):
            for _threadIdx in T.thread_binding(128, thread="threadIdx.x"):
                b = A[0, 0:4]
                A[0, 4:8] = b

    mod = tvm.IRModule({"main": main})
    mod = tvm.compile(mod, target="cuda")
    assert "float4 b" in mod.mod.imports[0].inspect_source()


if __name__ == "__main__":
    tilelang.testing.main()
