import json
import os

import pytest
import tilelang.testing
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target

PRINT = True


def _maybe_print_kernel_and_mlir(func, src: str):
    if not PRINT:
        return
    print("=== TVM Kernel ===")
    if hasattr(func, "script"):
        print(func.script())
    else:
        print(func)
    print("=== SunMMIO SUVM MLIR ===")
    print(src)


def _to_device_kernel_func(func):
    return func.with_attr("global_symbol", "main").with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))


def _primfunc_from_stmt(stmt):
    return _to_device_kernel_func(tvm.tir.PrimFunc([], stmt))


def build_sunmmio_module_without_compile(func):
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": _to_device_kernel_func(func)})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    return builder(mod, target, "suvm")


def build_sunmmio_source_without_compile(func):
    src = build_sunmmio_module_without_compile(func).inspect_source()
    _maybe_print_kernel_and_mlir(func, src)
    return src


def build_sunmmio_source_from_stmt(stmt):
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": _primfunc_from_stmt(stmt)})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    src = builder(mod, target, "suvm").inspect_source()
    _maybe_print_kernel_and_mlir(mod["main"], src)
    return src


def make_scalar_control_kernel():
    i = tvm.tir.Var("i", "int32")
    j = tvm.tir.Var("j", "int32")
    v = tvm.tir.Var("v", "int32")

    expr_then = tvm.tir.Select(
        tvm.tir.LT(j, tvm.tir.IntImm("int32", 8)),
        tvm.tir.Add(i, j),
        tvm.tir.Sub(i, j),
    )
    let_stmt = tvm.tir.LetStmt(
        v,
        tvm.tir.Add(i, j),
        tvm.tir.Evaluate(tvm.tir.Max(v, tvm.tir.IntImm("int32", 0))),
    )
    then_stmt = tvm.tir.SeqStmt([tvm.tir.Evaluate(expr_then), let_stmt])
    else_stmt = tvm.tir.Evaluate(tvm.tir.Min(i, j))
    inner_if = tvm.tir.IfThenElse(tvm.tir.LE(j, tvm.tir.IntImm("int32", 12)), then_stmt, else_stmt)
    inner_for = tvm.tir.For(j, 0, 16, tvm.tir.ForKind.SERIAL, inner_if)
    outer_for = tvm.tir.For(i, 0, 8, tvm.tir.ForKind.SERIAL, inner_for)
    return _primfunc_from_stmt(outer_for)


def make_alloc_scope_kernel():
    f16 = tvm.ir.PrimType("float16")
    one = tvm.tir.IntImm("bool", 1)
    body = tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0))

    rsram = tvm.tir.Var("rsram_buf", tvm.ir.PointerType(f16, "shared.rsram"))
    wsram = tvm.tir.Var("wsram_buf", tvm.ir.PointerType(f16, "shared.wsram"))
    asram = tvm.tir.Var("asram_buf", tvm.ir.PointerType(f16, "shared.asram"))

    stmt = tvm.tir.Allocate(rsram, "float16", [16, 16], one, body)
    stmt = tvm.tir.Allocate(wsram, "float16", [16, 16], one, stmt)
    stmt = tvm.tir.Allocate(asram, "float16", [16, 16], one, stmt)
    return _primfunc_from_stmt(stmt)


def make_intrinsic_sync_kernel():
    f16 = tvm.ir.PrimType("float16")
    a_data = tvm.tir.Var("a_data", tvm.ir.PointerType(f16, "shared.asram"))
    b_data = tvm.tir.Var("b_data", tvm.ir.PointerType(f16, "shared.wsram"))
    c_data = tvm.tir.Var("c_data", tvm.ir.PointerType(f16, "shared.rsram"))
    a_buf = tvm.tir.decl_buffer((32, 32), "float16", name="A", data=a_data, scope="shared.asram")
    b_buf = tvm.tir.decl_buffer((32, 32), "float16", name="B", data=b_data, scope="shared.wsram")
    c_buf = tvm.tir.decl_buffer((32, 32), "float16", name="C", data=c_data, scope="shared.rsram")

    def region(buf, access):
        return tvm.tir.call_intrin(
            "handle",
            tvm.ir.Op.get("tl.tileop.region"),
            tvm.tir.BufferLoad(
                buf,
                [tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 0)],
            ),
            tvm.tir.IntImm("int32", access),
            tvm.tir.IntImm("int32", 32),
            tvm.tir.IntImm("int32", 32),
        )

    def sync_token(token_id):
        return tvm.tir.call_intrin(
            "handle",
            tvm.ir.Op.get("tl.sync_token_id"),
            tvm.tir.IntImm("int32", token_id),
        )

    dma = tvm.tir.Call(
        "handle",
        tvm.ir.Op.get("tl.dma_copy"),
        [region(a_buf, 1), region(c_buf, 2), tvm.tir.IntImm("int32", 0), sync_token(0)],
    )
    mma = tvm.tir.Call(
        "handle",
        tvm.ir.Op.get("tl.mma_sunmmio"),
        [
            region(a_buf, 1),
            region(b_buf, 1),
            region(c_buf, 3),
            tvm.tir.IntImm("bool", 0),
            tvm.tir.IntImm("bool", 0),
            tvm.tir.IntImm("bool", 0),
            tvm.tir.IntImm("int32", 0),
            sync_token(1),
        ],
    )
    sync = tvm.tir.Call(
        "handle",
        tvm.ir.Op.get("tir.tvm_storage_sync"),
        [tvm.tir.StringImm("shared")],
    )
    ramp = tvm.tir.Ramp(tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 1), 4)
    bcast = tvm.tir.Broadcast(tvm.tir.FloatImm("float32", 1.25), 4)

    i = tvm.tir.Var("i", "int32")
    loop_body = tvm.tir.SeqStmt(
        [
            tvm.tir.Evaluate(dma),
            tvm.tir.Evaluate(sync),
            tvm.tir.Evaluate(mma),
            tvm.tir.Evaluate(ramp),
            tvm.tir.Evaluate(bcast),
        ]
    )
    pred = tvm.tir.LT(i, tvm.tir.IntImm("int32", 2))
    if_stmt = tvm.tir.IfThenElse(pred, loop_body, tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0)))
    stmt = tvm.tir.For(i, 0, 4, tvm.tir.ForKind.SERIAL, if_stmt)
    stmt = tvm.tir.DeclBuffer(a_buf, tvm.tir.DeclBuffer(b_buf, tvm.tir.DeclBuffer(c_buf, stmt)))
    return _to_device_kernel_func(tvm.tir.PrimFunc([a_data, b_data, c_data], stmt))


def make_block_realize_kernel():
    body = tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0))
    block = tvm.tir.Block([], [], [], "B", body)
    stmt = tvm.tir.BlockRealize([], tvm.tir.IntImm("bool", 1), block)
    return _primfunc_from_stmt(stmt)


def make_decl_buffer_kernel():
    body = tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0))
    buf = tvm.tir.decl_buffer((16, 16), "float16", name="A")
    stmt = tvm.tir.DeclBuffer(buf, body)
    return _primfunc_from_stmt(stmt)


def make_buffer_realize_kernel():
    body = tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0))
    buf = tvm.tir.decl_buffer((16, 16), "float16", name="A")
    bounds = [
        tvm.ir.Range.from_min_extent(0, 16),
        tvm.ir.Range.from_min_extent(0, 16),
    ]
    stmt = tvm.tir.BufferRealize(buf, bounds, tvm.tir.IntImm("bool", 1), body)
    return _primfunc_from_stmt(stmt)


def make_buffer_load_kernel():
    buf = tvm.tir.decl_buffer((16, 16), "float16", name="A")
    stmt = tvm.tir.Evaluate(
        tvm.tir.BufferLoad(
            buf,
            [tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 0)],
        )
    )
    return _primfunc_from_stmt(stmt)


def make_buffer_store_kernel():
    buf = tvm.tir.decl_buffer((16, 16), "float16", name="A")
    stmt = tvm.tir.BufferStore(
        buf,
        tvm.tir.FloatImm("float16", 1.0),
        [tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 0)],
    )
    return _primfunc_from_stmt(stmt)


def make_real_tilelang_frontend_kernel():
    @T.prim_func
    def main():
        with T.Kernel(1, 1, threads=1) as (bx, by):
            for i in T.serial(0, 8):
                T.evaluate(i + 1)

    return main


def _assert_coverage_report_complete(report_path):
    assert report_path.exists(), f"coverage report not generated: {report_path}"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required_keys = [
        "expected_node_types",
        "visited_node_types",
        "missing_node_types",
        "expected_call_ops",
        "visited_call_ops",
        "missing_call_ops",
    ]
    for key in required_keys:
        assert key in report, f"missing coverage key: {key}"
    assert report["missing_node_types"] == []
    assert report["missing_call_ops"] == []


def test_sunmmio_codegen_without_compile_emits_nonempty_suvm_source():
    src = build_sunmmio_source_without_compile(make_scalar_control_kernel())
    assert src.strip()
    assert "module" in src
    assert "func.func @main" in src


def test_sunmmio_codegen_while_emits_scf_while():
    cond = tvm.tir.LT(tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 1))
    body = tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0))
    stmt = tvm.tir.While(cond, body)
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": _primfunc_from_stmt(stmt)})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    src = builder(mod, target, "suvm").inspect_source()
    assert "scf.while" in src
    assert "scf.condition" in src
    assert "scf.yield" in src


def test_sunmmio_codegen_shuffle_fails_loudly():
    shuffle = tvm.tir.Shuffle(
        [tvm.tir.Broadcast(tvm.tir.IntImm("int32", 7), 4)],
        [tvm.tir.IntImm("int32", 0)],
    )
    stmt = tvm.tir.Evaluate(shuffle)
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": _primfunc_from_stmt(stmt)})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(Exception, match="CodeGenTileLangSunMMIO unsupported expr: tir.Shuffle"):
        builder(mod, target, "suvm")


def test_sunmmio_codegen_compile_path_not_implemented():
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": make_scalar_control_kernel()})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio")
    with pytest.raises(Exception, match="not implemented yet"):
        builder(mod, target)


def test_sunmmio_codegen_block_realize_fails_loudly():
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": make_block_realize_kernel()})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(
        Exception,
        match="BlockRealizeNode should be eliminated by LowerOpaqueBlock before SunMMIO codegen",
    ):
        builder(mod, target, "suvm")


def test_sunmmio_codegen_decl_buffer_is_benign_wrapper():
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": make_decl_buffer_kernel()})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    src = builder(mod, target, "suvm").inspect_source()
    assert src.strip()
    assert "module" in src
    assert "func.func @main" in src


def test_sunmmio_codegen_buffer_realize_fails_loudly():
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": make_buffer_realize_kernel()})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(
        Exception,
        match="BufferRealizeNode should be lowered into a concrete view/alias representation before SunMMIO codegen",
    ):
        builder(mod, target, "suvm")


def test_sunmmio_codegen_buffer_load_fails_loudly():
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": make_buffer_load_kernel()})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(
        Exception,
        match="generic BufferLoadNode should not reach SunMMIO codegen; tiled buffer accesses must be lowered through tile-aware paths",
    ):
        builder(mod, target, "suvm")


def test_sunmmio_codegen_buffer_store_fails_loudly():
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": make_buffer_store_kernel()})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(
        Exception,
        match="generic BufferStoreNode should not reach SunMMIO codegen; tiled buffer stores must be lowered through tile-aware paths",
    ):
        builder(mod, target, "suvm")


@pytest.mark.parametrize(
    "kernel_name,kernel_factory",
    [
        ("scalar_control", make_scalar_control_kernel),
        ("alloc_scope", make_alloc_scope_kernel),
        ("intrinsic_sync", make_intrinsic_sync_kernel),
        pytest.param(
            "real_tilelang_frontend",
            make_real_tilelang_frontend_kernel,
            marks=pytest.mark.xfail(
                reason=(
                    "generic SunMMIO codegen now requires pre-lowered backend IR; "
                    "this real frontend case still bypasses the full SunMMIO "
                    "lowering pipeline. Revisit when the full pass pipeline lands."
                ),
                strict=True,
            ),
        ),
    ],
)
def test_sunmmio_codegen_coverage_report_has_no_missing_entries(tmp_path, kernel_name, kernel_factory):
    report_path = tmp_path / f"codegen_coverage_{kernel_name}.json"
    old_path = os.environ.get("TL_SUNMMIO_CODEGEN_COVERAGE_PATH")
    old_strict = os.environ.get("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT")
    os.environ["TL_SUNMMIO_CODEGEN_COVERAGE_PATH"] = str(report_path)
    os.environ["TL_SUNMMIO_CODEGEN_COVERAGE_STRICT"] = "1"
    try:
        _ = build_sunmmio_source_without_compile(kernel_factory())
    finally:
        if old_path is None:
            os.environ.pop("TL_SUNMMIO_CODEGEN_COVERAGE_PATH", None)
        else:
            os.environ["TL_SUNMMIO_CODEGEN_COVERAGE_PATH"] = old_path
        if old_strict is None:
            os.environ.pop("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT", None)
        else:
            os.environ["TL_SUNMMIO_CODEGEN_COVERAGE_STRICT"] = old_strict

    _assert_coverage_report_complete(report_path)


if __name__ == "__main__":
    tilelang.testing.main()
