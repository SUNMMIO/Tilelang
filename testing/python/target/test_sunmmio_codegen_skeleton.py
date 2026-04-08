import pytest
import re
import numpy as np
import tilelang.language as T
import tilelang.testing
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target

# SunMMIO codegen traversal contract:
# Supported now:
# - core scalar arithmetic/control flow
# - For, IfThenElse
# - Allocate, AllocateConst
# - DeclBuffer, BufferRealize, BufferLoad, BufferStore
# - Block, BlockRealize
# - Ramp, Broadcast
# - TileLang/SunMMIO intrinsic Call
#
# Intentionally unsupported:
# - While
# - Shuffle
# - legacy Load
# - Any


def simple_add_kernel(n: int = 16):
    @T.prim_func
    def main(
        A: T.Tensor((n,), dtype=T.float32),
        B: T.Tensor((n,), dtype=T.float32),
    ):
        with T.Kernel(1, 1) as (bx, by):
            for i in T.serial(n):
                B[i] = A[i] + T.float32(1.0)

    return main


def build_sunmmio_module_without_compile(func):
    target = determine_target("Sunmmio", return_object=True)
    func = func.with_attr("global_symbol", "main")
    func = func.with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
    mod = tvm.IRModule({"main": func})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    return builder(mod, target)


def build_sunmmio_source_without_compile(func):
    return build_sunmmio_module_without_compile(func).inspect_source()


def build_sunmmio_source_from_stmt(stmt):
    target = determine_target("Sunmmio", return_object=True)
    func = tvm.tir.PrimFunc([], stmt)
    func = func.with_attr("global_symbol", "main")
    func = func.with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
    mod = tvm.IRModule({"main": func})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    return builder(mod, target).inspect_source()


def test_sunmmio_codegen_without_compile_emits_mlir_source():
    src = build_sunmmio_source_without_compile(simple_add_kernel())
    print(src)
    assert "module {" in src
    assert "func.func @main" in src
    assert "scf.for" in src
    assert "memref.load" in src
    assert "arith.addf" in src
    assert "memref.store" in src
    assert "return" in src


def test_sunmmio_codegen_no_placeholder_summary_text():
    src = build_sunmmio_source_without_compile(simple_add_kernel())
    print(src)
    assert "sunmmio.traversal_summary" not in src
    assert "status: traversal_only_no_emission" not in src


def test_sunmmio_codegen_emits_multidim_store_indices():
    @T.prim_func
    def main(A: T.Tensor((4, 4), dtype=T.float32), B: T.Tensor((4, 4), dtype=T.float32)):
        with T.attr(0, "sunmmio.test_attr", 7):
            for i, j in T.grid(2, 3):
                with T.block("B0"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    T.reads(A[vi, vj])
                    T.writes(B[vi, vj])
                    B[vi, vj] = A[vi, vj] + T.float32(1.0)

    src = build_sunmmio_source_without_compile(main)
    print(src)
    assert "scf.for" in src
    assert "memref.store" in src
    assert re.search(r"memref\.store .*?\[[^,\]]+,\s*[^,\]]+\]", src), src


def test_sunmmio_codegen_classifies_sunmmio_intrinsic_calls():
    dma_call = tvm.tir.Call("handle", tvm.ir.Op.get("tl.dma_copy"), [])
    mma_call = tvm.tir.Call("handle", tvm.ir.Op.get("tl.mma_sunmmio"), [])
    body = tvm.tir.SeqStmt([tvm.tir.Evaluate(dma_call), tvm.tir.Evaluate(mma_call)])
    src = build_sunmmio_source_from_stmt(body)
    print(src)
    assert 'sunmmio.call @"tl.dma_copy"(' in src
    assert 'sunmmio.call @"tl.mma_sunmmio"(' in src
    assert 'category = "sunmmio_intrinsic"' in src


def test_sunmmio_codegen_block_predicate_emits_control_flow():
    @T.prim_func
    def main(A: T.Tensor((8,), dtype=T.float32), B: T.Tensor((8,), dtype=T.float32)):
        for i in T.serial(8):
            with T.block("blk"):
                vi = T.axis.spatial(8, i)
                T.where(vi < 4)
                T.reads(A[vi])
                T.writes(B[vi])
                B[vi] = A[vi] + T.float32(1.0)

    src = build_sunmmio_source_without_compile(main)
    print(src)
    assert "scf.if" in src
    assert "memref.store" in src


def test_sunmmio_codegen_block_annotations_are_traversed():
    @T.prim_func
    def main(A: T.Tensor((8,), dtype=T.float32), B: T.Tensor((8,), dtype=T.float32)):
        for i in T.serial(8):
            with T.block("blk"):
                vi = T.axis.spatial(8, i)
                T.block_attr({"sunmmio.anno_expr": vi + 1, "sunmmio.anno_const": 7})
                T.reads(A[vi])
                T.writes(B[vi])
                B[vi] = A[vi] + T.float32(1.0)

    src = build_sunmmio_source_without_compile(main)
    print(src)
    assert "func.func @main" in src
    assert "arith.addi" in src
    assert "memref.store" in src


def test_sunmmio_codegen_allocate_const_is_handled():
    dev = tvm.cpu(0)
    dtype = "float32"
    shape = (4,)
    cbuf = tvm.tir.decl_buffer(shape, dtype, name="C")
    data = tvm.runtime.tensor(np.array([1, 2, 3, 4], dtype=dtype), device=dev)
    body = tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0))
    stmt = tvm.tir.AllocateConst(cbuf.data, dtype, shape, data, body)
    src = build_sunmmio_source_from_stmt(stmt)
    print(src)
    assert "memref.alloc" in src


def test_sunmmio_codegen_block_alloc_buffers_are_handled():
    @T.prim_func
    def main(A: T.Tensor((8,), dtype=T.float32), B: T.Tensor((8,), dtype=T.float32)):
        for i in T.serial(8):
            with T.block("blk"):
                vi = T.axis.spatial(8, i)
                tmp = T.alloc_buffer((8,), dtype=T.float32, scope="local")
                T.reads(A[vi])
                T.writes(B[vi], tmp[vi])
                tmp[vi] = A[vi]
                B[vi] = tmp[vi] + T.float32(1.0)

    src = build_sunmmio_source_without_compile(main)
    print(src)
    assert "memref.alloc" in src
    assert "memref.store" in src


def test_sunmmio_codegen_unsupported_stmt_fails_loudly():
    cond = tvm.tir.LT(tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 1))
    body = tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0))
    stmt = tvm.tir.While(cond, body)
    target = determine_target("Sunmmio", return_object=True)
    func = tvm.tir.PrimFunc([], stmt)
    func = func.with_attr("global_symbol", "main")
    func = func.with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
    mod = tvm.IRModule({"main": func})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(Exception, match="CodeGenTileLangSunMMIO unsupported stmt: tir.While"):
        builder(mod, target)


def test_sunmmio_codegen_shuffle_fails_loudly():
    shuffle = tvm.tir.Shuffle(
        [tvm.tir.Broadcast(tvm.tir.IntImm("int32", 7), 4)],
        [tvm.tir.IntImm("int32", 0)],
    )
    stmt = tvm.tir.Evaluate(shuffle)
    target = determine_target("Sunmmio", return_object=True)
    func = tvm.tir.PrimFunc([], stmt)
    func = func.with_attr("global_symbol", "main")
    func = func.with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
    mod = tvm.IRModule({"main": func})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(Exception, match="CodeGenTileLangSunMMIO unsupported expr: tir.Shuffle"):
        builder(mod, target)


def test_sunmmio_codegen_ramp_is_supported():
    ramp = tvm.tir.Ramp(tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 1), 4)
    stmt = tvm.tir.Evaluate(ramp)
    src = build_sunmmio_source_from_stmt(stmt)
    print(src)
    assert "sunmmio.ramp" in src
    assert "vector<4xi32>" in src


def test_sunmmio_codegen_broadcast_is_supported():
    bcast = tvm.tir.Broadcast(tvm.tir.FloatImm("float32", 1.5), 4)
    stmt = tvm.tir.Evaluate(bcast)
    src = build_sunmmio_source_from_stmt(stmt)
    print(src)
    assert "vector.broadcast" in src
    assert "vector<4xf32>" in src


def test_sunmmio_codegen_legacy_loadnode_fails_loudly():
    if not hasattr(tvm.tir, "Load"):
        pytest.skip("legacy tir.Load is unavailable in this Python binding")
    buf = tvm.tir.decl_buffer((4,), "float32", name="A")
    pred = tvm.tir.IntImm("bool", 1)
    load = tvm.tir.Load("float32", buf.data, tvm.tir.IntImm("int32", 0), pred)
    stmt = tvm.tir.Evaluate(load)
    target = determine_target("Sunmmio", return_object=True)
    func = tvm.tir.PrimFunc([], stmt)
    func = func.with_attr("global_symbol", "main")
    func = func.with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
    mod = tvm.IRModule({"main": func})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(Exception, match="CodeGenTileLangSunMMIO unsupported expr: tir.Load"):
        builder(mod, target)


def test_sunmmio_codegen_anynode_fails_loudly_or_skips():
    if not hasattr(tvm.tir, "Any"):
        pytest.skip("tir.Any is unavailable in this Python binding")
    try:
        any_expr = tvm.tir.Any()
    except Exception:
        pytest.skip("tir.Any cannot be constructed in this Python binding")
    stmt = tvm.tir.Evaluate(any_expr)
    target = determine_target("Sunmmio", return_object=True)
    func = tvm.tir.PrimFunc([], stmt)
    func = func.with_attr("global_symbol", "main")
    func = func.with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
    mod = tvm.IRModule({"main": func})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(Exception, match="CodeGenTileLangSunMMIO unsupported expr: tir.Any"):
        builder(mod, target)


def test_sunmmio_codegen_compile_path_not_implemented():
    target = determine_target("Sunmmio", return_object=True)
    func = simple_add_kernel().with_attr("global_symbol", "main")
    func = func.with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
    mod = tvm.IRModule({"main": func})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio")
    with pytest.raises(Exception, match="not implemented yet"):
        builder(mod, target)


if __name__ == "__main__":
    tilelang.testing.main()
