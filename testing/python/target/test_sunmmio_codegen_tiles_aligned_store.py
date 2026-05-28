from tilelang import tvm
from tilelang.utils.target import determine_target


def _to_device_kernel_func(func):
    return func.with_attr("global_symbol", "main").with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))


def _build_sunmmio_source_from_stmt(stmt):
    target = determine_target("Sunmmio", return_object=True)
    func = _to_device_kernel_func(tvm.tir.PrimFunc([], stmt))
    mod = tvm.IRModule({"main": func})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    return builder(mod, target, "suvm").inspect_source()


def _make_tile_valued_aligned_store_stmt():
    bf16 = tvm.ir.PrimType("bfloat16")
    one = tvm.tir.IntImm("bool", 1)

    a_data = tvm.tir.Var("A_shared_data", tvm.ir.PointerType(bf16, "shared.rsram"))
    b_data = tvm.tir.Var("B_shared_data", tvm.ir.PointerType(bf16, "shared.rsram"))
    out_data = tvm.tir.Var("Out_shared_data", tvm.ir.PointerType(bf16, "shared.rsram"))

    a_buf = tvm.tir.decl_buffer((64,), "bfloat16", name="A_shared", data=a_data, scope="shared.rsram")
    b_buf = tvm.tir.decl_buffer((64,), "bfloat16", name="B_shared", data=b_data, scope="shared.rsram")
    out_buf = tvm.tir.decl_buffer((64,), "bfloat16", name="Out_shared", data=out_data, scope="shared.rsram")

    tile_i = tvm.tir.Var("tile_i", "int32")
    ki = tvm.tir.Var("ki", "int32")
    tile_base = tvm.tir.Mul(tile_i, tvm.tir.IntImm("int32", 8))
    elem_index = tvm.tir.Add(tile_base, ki)

    # These 1D tiled loads are lowered through the regular tile access path,
    # so the add result is a logical tile value rather than a scalar.
    value = tvm.tir.Add(
        tvm.tir.BufferLoad(a_buf, [elem_index]),
        tvm.tir.BufferLoad(b_buf, [elem_index]),
    )
    store = tvm.tir.BufferStore(out_buf, value, [elem_index])

    inner = tvm.tir.For(
        ki,
        0,
        8,
        tvm.tir.ForKind.SERIAL,
        store,
        annotations={
            "tile.interior": tvm.tir.IntImm("int32", 1),
            "tile.interior_axis": tvm.tir.IntImm("int32", 0),
        },
    )

    outer = tvm.tir.For(
        tile_i,
        0,
        8,
        tvm.tir.ForKind.SERIAL,
        inner,
        annotations={
            "tile.domain": [tvm.tir.IntImm("int32", 64)],
            "tile.execution_axis": tvm.tir.IntImm("int32", 0),
            "tile.execution_domain_axes": [tvm.tir.IntImm("int32", 0)],
            "tile.scope_entry": tvm.tir.IntImm("int32", 1),
            "tile.tile_size": [tvm.tir.IntImm("int32", 8)],
        },
    )

    return tvm.tir.DeclBuffer(
        a_buf,
        tvm.tir.DeclBuffer(
            b_buf,
            tvm.tir.DeclBuffer(
                out_buf,
                tvm.tir.Allocate(
                    a_data,
                    "bfloat16",
                    [64],
                    one,
                    tvm.tir.Allocate(
                        b_data,
                        "bfloat16",
                        [64],
                        one,
                        tvm.tir.Allocate(out_data, "bfloat16", [64], one, outer),
                    ),
                ),
            ),
        ),
    )


def test_sunmmio_codegen_aligned_1d_store_uses_insert_slice_bridge():
    src = _build_sunmmio_source_from_stmt(_make_tile_valued_aligned_store_stmt())
    assert "fake_tile_insert_slice" in src
    assert "fake_tile_load" in src
    assert "fake_tile_unsqueeze" in src
