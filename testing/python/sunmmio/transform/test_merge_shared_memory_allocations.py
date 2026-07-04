from tilelang import tvm as tvm
import tilelang as tl
from tvm import te
from tvm.tir import (
    BufferStore,
    PrimFunc,
    Allocate,
    AttrStmt,
    SeqStmt,
    stmt_functor,
)
from tvm.ir import PrimType, PointerType
from tilelang.utils.target import determine_target


target = determine_target("Sunmmio", return_object=True)


def show_module(mod, desc="module:"):
    print(desc)
    print(mod.script())


def _count_alloc_by_scope(body, scope):
    cnt = 0

    def visitor(n):
        nonlocal cnt
        if isinstance(n, Allocate):
            ta = n.buffer_var.type_annotation
            if getattr(ta, "storage_scope", "") == scope:
                cnt += 1

    stmt_functor.post_order_visit(body, visitor)
    return cnt


def _get_single_alloc_extent(body, scope):
    extent = None

    def visitor(n):
        nonlocal extent
        if isinstance(n, Allocate):
            ta = n.buffer_var.type_annotation
            if getattr(ta, "storage_scope", "") == scope and extent is None:
                extent = n.extents[0]

    stmt_functor.post_order_visit(body, visitor)
    return extent


def build_multi_scope_mixed_allocs():
    # uint8 pointer types per scope
    u8 = PrimType("uint8")
    A1 = tvm.tir.Var("A1", PointerType(u8, "shared.asram"))
    A2 = tvm.tir.Var("A2", PointerType(u8, "shared.asram"))
    W1 = tvm.tir.Var("W1", PointerType(u8, "shared.wsram"))
    W2 = tvm.tir.Var("W2", PointerType(u8, "shared.wsram"))
    R1 = tvm.tir.Var("R1", PointerType(u8, "shared.rsram"))
    R2 = tvm.tir.Var("R2", PointerType(u8, "shared.rsram"))

    # dynamic extents
    n_as = tvm.tir.Var("n_as", "int32")
    n_ws = tvm.tir.Var("n_ws", "int32")
    n_rs = tvm.tir.Var("n_rs", "int32")

    # Buffers (mix of const and dynamic sizes)
    bufA1 = tvm.tir.decl_buffer([tvm.tir.IntImm("int32", 64)], "uint8", name="A1", data=A1)
    bufA2 = tvm.tir.decl_buffer([n_as], "uint8", name="A2", data=A2)
    bufW1 = tvm.tir.decl_buffer([tvm.tir.IntImm("int32", 128)], "uint8", name="W1", data=W1)
    bufW2 = tvm.tir.decl_buffer([n_ws], "uint8", name="W2", data=W2)
    bufR1 = tvm.tir.decl_buffer([n_rs], "uint8", name="R1", data=R1)
    bufR2 = tvm.tir.decl_buffer([tvm.tir.IntImm("int32", 16)], "uint8", name="R2", data=R2)

    # Some simple stores to create uses
    stores = [
        BufferStore(bufA1, tvm.tir.IntImm("uint8", 1), [tvm.tir.IntImm("int32", 0)]),
        BufferStore(bufA2, tvm.tir.IntImm("uint8", 2), [tvm.tir.IntImm("int32", 1)]),
        BufferStore(bufW1, tvm.tir.IntImm("uint8", 3), [tvm.tir.IntImm("int32", 2)]),
        BufferStore(bufW2, tvm.tir.IntImm("uint8", 4), [tvm.tir.IntImm("int32", 3)]),
        BufferStore(bufR1, tvm.tir.IntImm("uint8", 5), [tvm.tir.IntImm("int32", 4)]),
        BufferStore(bufR2, tvm.tir.IntImm("uint8", 6), [tvm.tir.IntImm("int32", 5)]),
    ]
    body = SeqStmt(stores)

    # Allocate in nested order; each scope has two allocs (const + dynamic)
    body = Allocate(R2, "uint8", [tvm.tir.IntImm("int32", 16)], tvm.tir.const(True, "bool"), body)
    body = Allocate(R1, "uint8", [n_rs], tvm.tir.const(True, "bool"), body)
    body = Allocate(W2, "uint8", [n_ws], tvm.tir.const(True, "bool"), body)
    body = Allocate(W1, "uint8", [tvm.tir.IntImm("int32", 128)], tvm.tir.const(True, "bool"), body)
    body = Allocate(A2, "uint8", [n_as], tvm.tir.const(True, "bool"), body)
    body = Allocate(A1, "uint8", [tvm.tir.IntImm("int32", 64)], tvm.tir.const(True, "bool"), body)

    # Thread scope
    tx = te.thread_axis("threadIdx.x")
    body = AttrStmt(tx, "thread_extent", tvm.tir.IntImm("int32", 1), body)
    return PrimFunc(params=[], body=body)


def _build_two_if_seq(scope):
    u8 = PrimType("uint8")
    A = tvm.tir.Var("A", PointerType(u8, scope))
    B = tvm.tir.Var("B", PointerType(u8, scope))
    bufA = tvm.tir.decl_buffer([tvm.tir.IntImm("int32", 2049)], "uint8", name="A", data=A)
    bufB = tvm.tir.decl_buffer([tvm.tir.IntImm("int32", 2049)], "uint8", name="B", data=B)

    then_a = BufferStore(bufA, tvm.tir.IntImm("uint8", 1), [tvm.tir.IntImm("int32", 0)])
    then_b = BufferStore(bufB, tvm.tir.IntImm("uint8", 2), [tvm.tir.IntImm("int32", 0)])
    body = SeqStmt([then_a, then_b])

    body = tvm.tir.IfThenElse(tvm.tir.const(True, "bool"), body, None)

    body = Allocate(B, "uint8", [tvm.tir.IntImm("int32", 2049)], tvm.tir.const(True, "bool"), body)
    body = Allocate(A, "uint8", [tvm.tir.IntImm("int32", 2049)], tvm.tir.const(True, "bool"), body)

    # tx = te.thread_axis("threadIdx.x")
    tx = te.thread_axis("blockIdx.x")
    body = AttrStmt(tx, "thread_extent", tvm.tir.IntImm("int32", 1), body)
    return PrimFunc(params=[], body=body)


def test_merge_multi_scope_mixed_sizes():
    """多scope,常量和非常量大小混合的合并测试"""
    f = build_multi_scope_mixed_allocs()
    mod = tvm.IRModule({"main": f})
    mod = tvm.tir.transform.BindTarget(target)(mod)
    show_module(mod, "# before merge_shared_memory_allocations")
    with tvm.transform.PassContext(config={"tir.merge_static_smem": True, "tl.debug_merge_shared_memory_allocations": True}):
        out = tl.transform.MergeSharedMemoryAllocationsSunmmio(True)(mod)
    show_module(out, "# after merge_shared_memory_allocations")
    f_out = out["main"]
    # Expect exactly one Allocate per scope after merge
    assert _count_alloc_by_scope(f_out.body, "shared.asram") == 1, "shared.asram not 1"
    assert _count_alloc_by_scope(f_out.body, "shared.wsram") == 1, "shared.wsram not 1"
    assert _count_alloc_by_scope(f_out.body, "shared.rsram") == 1, "shared.rsram not 1"


def aggressive_reuse_test(scope):
    """对比测试,主要是aggressive策略下的内存重用对比"""
    f = _build_two_if_seq(scope)
    mod = tvm.IRModule({"main": f})
    mod = tvm.tir.transform.BindTarget(target)(mod)
    show_module(mod, "# before merge_shared_memory_allocations")
    with tvm.transform.PassContext(config={"tir.merge_static_smem": True}):
        out_conservative = tl.transform.MergeSharedMemoryAllocationsSunmmio()(mod)
        out_aggressive = tl.transform.MergeSharedMemoryAllocationsSunmmio(True)(mod)
    show_module(out_conservative, "# after merge_shared_memory_allocations")
    show_module(out_aggressive, "# after merge_shared_memory_allocations")
    f0 = out_conservative["main"]
    f1 = out_aggressive["main"]

    assert _count_alloc_by_scope(f0.body, scope) == 1
    assert _count_alloc_by_scope(f1.body, scope) == 1

    e0 = _get_single_alloc_extent(f0.body, scope)
    e1 = _get_single_alloc_extent(f1.body, scope)
    assert e0 is not None and e1 is not None
    assert int(e1) < int(e0), f"e0={e0},e1={e1}"


def test_aggressive_reuse_diff():
    test_scopes = ["shared.asram", "shared.wsram", "shared.rsram"]
    for scope in test_scopes:
        aggressive_reuse_test(scope)


if __name__ == "__main__":
    import tilelang.testing

    tilelang.testing.main()
    # test_aggressive_reuse_diff()
    # test_merge_multi_scope_mixed_sizes()
