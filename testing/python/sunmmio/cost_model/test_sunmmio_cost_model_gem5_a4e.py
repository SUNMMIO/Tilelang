import pytest

from tilelang import tvm
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from tvm import tir


ODMA = 0
TENSOR_CORE = 1
VECTOR_CORE = 2


def _pointer_var(name, dtype, scope):
    return tir.Var(name, tvm.ir.PointerType(tvm.ir.PrimType(dtype), scope))


def _buffer(name, shape, dtype, scope):
    return tir.decl_buffer(
        shape,
        dtype,
        name=name,
        data=_pointer_var(name + "_data", dtype, scope),
        scope=scope,
    )


def _region(buffer, access_mask, extents=None):
    extents = tuple(buffer.shape) if extents is None else tuple(extents)
    return tir.Call(
        "handle",
        tvm.ir.Op.get("tl.tileop.region"),
        [
            tir.BufferLoad(buffer, [tir.IntImm("int32", 0)] * len(buffer.shape)),
            tir.IntImm("int32", access_mask),
            *extents,
        ],
    )


def _call_stmt(op_name, *args):
    return tir.Evaluate(tir.Call("handle", tvm.ir.Op.get(op_name), list(args)))


def _estimate(device, stmt, profile="gem5_a4e_2026", fallback=True):
    func = tvm.get_global_func("tl.sunmmio.EstimateCostForTesting")
    return {str(k): str(v) for k, v in func(device, stmt, profile, fallback).items()}


def _integer(result, key):
    return int(result[key])


def _make_mma(
    row,
    col,
    acc,
    input_dtype="bfloat16",
    output_dtype="float32",
    clear=True,
    rhs_shape=None,
    trans_b=False,
):
    a = _buffer("a", (row, acc), input_dtype, "shared.asram")
    b = _buffer("b", rhs_shape or (acc, col), input_dtype, "shared.wsram")
    c = _buffer("c", (row, col), output_dtype, "shared.rsram")
    evaluate = _call_stmt(
        "tl.mma_sunmmio",
        _region(a, 1),
        _region(b, 1),
        _region(c, 3),
        tir.IntImm("bool", 0),
        tir.IntImm("bool", int(trans_b)),
        tir.IntImm("bool", int(clear)),
        tir.IntImm("int32", 0),
    )
    block = tir.Block([], [], [], "mma", evaluate)
    return tir.BlockRealize([], tir.IntImm("bool", 1), block)


@pytest.mark.parametrize(
    "stmt, expected",
    [
        (_make_mma(16, 32, 32, clear=True), 38),
        (_make_mma(32, 32, 32, output_dtype="bfloat16", clear=True), 38),
        (_make_mma(32, 32, 32, output_dtype="float32", clear=True), 38),
        (_make_mma(32, 32, 64, input_dtype="int8", output_dtype="int8", clear=True), 39),
        (_make_mma(256, 256, 128, output_dtype="float32", clear=False), 457),
    ],
)
def test_tensor_formula_vectors(stmt, expected):
    result = _estimate(TENSOR_CORE, stmt)
    assert _integer(result, "latency_cycles") == expected
    assert result["latency_source"] == "TensorCore::calculateOpDelayCycles"
    assert _integer(result, "exact_static_submodel") == 1
    assert _integer(result, "used_fallback") == 0


@pytest.mark.parametrize(
    "stmt, expected, expected_row, expected_col, expected_acc",
    [
        # fork_join_gemm: logical 32x32x32 becomes CSR row=16,acc=32,col=32.
        (_make_mma(32, 32, 32, clear=False), 38, 16, 32, 32),
        # qkv_summa_gemm: transposed B is 32x128, but TC_COL_SIZE comes from C.
        (
            _make_mma(
                32,
                32,
                128,
                clear=False,
                rhs_shape=(32, 128),
                trans_b=True,
            ),
            41,
            16,
            32,
            128,
        ),
    ],
)
def test_tensor_csr_shape_vectors(stmt, expected, expected_row, expected_col, expected_acc):
    result = _estimate(TENSOR_CORE, stmt)
    assert _integer(result, "latency_cycles") == expected
    assert _integer(result, "row_size") == expected_row
    assert _integer(result, "col_size") == expected_col
    assert _integer(result, "reduction_size") == expected_acc
    assert _integer(result, "exact_static_submodel") == 1
    assert _integer(result, "used_fallback") == 0


def test_tensor_unsupported_dtype_strict():
    stmt = _make_mma(16, 32, 32, input_dtype="float16", output_dtype="float32")
    with pytest.raises(tvm.error.TVMError, match="unsupported_tensor_dtype"):
        _estimate(TENSOR_CORE, stmt, fallback=False)


def _make_copy(src_scope, dst_scope, shape, dtype="bfloat16"):
    src = _buffer("src", shape, dtype, src_scope)
    dst = _buffer("dst", shape, dtype, dst_scope)
    return _call_stmt("tl.dma_copy", _region(src, 1), _region(dst, 2), 0)


@pytest.mark.parametrize(
    "stmt, expected, path",
    [
        (_make_copy("global", "shared.rsram", (32, 16)), 71, "dram_to_local_sram"),
        (_make_copy("global", "shared.rsram", (32, 32)), 72, "dram_to_local_sram"),
        (_make_copy("global", "shared.wsram", (32, 32)), 71, "dram_to_local_sram"),
        (
            _make_copy("shared.rsram", "shared.asram", (32, 32), "float32"),
            14,
            "rsram_to_asram_non_bf16_local",
        ),
        (
            _make_copy("shared.rsram", "shared.asram", (32, 32)),
            14,
            "rsram_to_asram_bf16_local",
        ),
        (
            _make_copy("shared.rsram", "shared.wsram", (32, 32)),
            14,
            "rsram_to_wsram_local",
        ),
        (
            _make_copy("shared.rsram", "shared.rsram", (32, 32)),
            11,
            "rsram_to_rsram_local",
        ),
    ],
)
def test_odma_local_copy_vectors(stmt, expected, path):
    result = _estimate(ODMA, stmt)
    assert _integer(result, "latency_cycles") == expected
    assert result["transfer_path"] == path
    assert result["src_scope"]
    assert result["dst_scope"]
    assert _integer(result, "endpoint_timing_modeled") == 1


def _make_broadcast(src_scope, dst_scope, shape, direction, dtype="bfloat16", token=False):
    src = _buffer("broadcast_src", shape, dtype, src_scope)
    dst = _buffer("broadcast_dst", shape, dtype, dst_scope)
    args = [
        _region(src, 1),
        _region(dst, 2),
        tir.IntImm("int32", direction),
        tir.IntImm("int64", 15),
        tir.IntImm("int32", 0),
    ]
    if token:
        args.append(
            tir.Call(
                "handle",
                tvm.ir.Op.get("tl.sync_token_id"),
                [tir.IntImm("int32", 7)],
            )
        )
    return _call_stmt("tl.broadcast_", *args)


def test_broadcast_link_path_and_token_parser():
    result = _estimate(
        ODMA,
        _make_broadcast("global", "shared.rsram", (32, 32), 0, token=True),
    )
    assert _integer(result, "latency_cycles") == 113
    assert result["transfer_path"] == "dram_to_link_rsram"
    assert _integer(result, "endpoint_timing_modeled") == 1

    result = _estimate(
        ODMA,
        _make_broadcast("global", "shared.wsram", (32, 32), 0),
    )
    assert _integer(result, "latency_cycles") == 102
    assert result["transfer_path"] == "dram_to_link_asram_wsram"
    assert _integer(result, "endpoint_timing_modeled") == 1

    result = _estimate(
        ODMA,
        _make_broadcast("shared.rsram", "shared.asram", (32, 32), 1),
    )
    assert _integer(result, "latency_cycles") == 24
    assert result["transfer_path"] == "rsram_to_link_asram_bf16"
    assert _integer(result, "endpoint_timing_modeled") == 1


@pytest.mark.parametrize(
    "dtype, expected_packets, expected_first, expected_last",
    [("bfloat16", 64, 12, 75), ("float32", 256, 12, 267)],
)
def test_transpose_descriptor_formula(dtype, expected_packets, expected_first, expected_last):
    src = _buffer("transpose_src", (32, 32), dtype, "shared.rsram")
    dst = _buffer("transpose_dst", (32, 32), dtype, "shared.rsram")
    result = _estimate(
        ODMA,
        _call_stmt("tl.sunmmio_transpose", _region(src, 1), _region(dst, 2)),
    )
    assert _integer(result, "packet_count") == expected_packets
    assert _integer(result, "issue_start_cycles") == expected_first
    assert _integer(result, "latency_cycles") == expected_last


def test_layout_transform_missing_context_falls_back():
    src = _buffer("layout_src", (32, 32), "bfloat16", "shared.rsram")
    dst = _buffer("layout_dst", (32, 32), "bfloat16", "shared.asram")
    result = _estimate(
        ODMA,
        _call_stmt("tl.sunmmio_layout_transform", _region(src, 1), _region(dst, 2)),
    )
    assert _integer(result, "used_fallback") == 1
    assert result["fallback_reason"] == "missing_layout_context"


def _make_vector_stmt(value_builder, extent=128):
    src = _buffer("vector_src", (extent,), "float32", "shared.rsram")
    dst = _buffer("vector_dst", (extent,), "float32", "shared.rsram")
    i = tir.Var("i", "int32")
    value = value_builder(tir.BufferLoad(src, [i]))
    return tir.For(
        i,
        0,
        extent,
        tir.ForKind.SERIAL,
        tir.BufferStore(dst, value, [i]),
    )


@pytest.mark.parametrize(
    "stmt, expected",
    [
        (_make_vector_stmt(lambda value: value + tir.FloatImm("float32", 1.0)), 18),
        (
            _make_vector_stmt(lambda value: tir.Call("float32", tvm.ir.Op.get("tir.exp2"), [value])),
            25,
        ),
    ],
)
def test_vector_conservative_fu_sum(stmt, expected):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with target:
        result = _estimate(VECTOR_CORE, stmt)
    assert _integer(result, "latency_cycles") == expected
    assert result["latency_source"].endswith("conservative_sum")
    assert _integer(result, "used_fallback") == 0
    assert _integer(result, "resource_overlap_modeled") == 0


def test_vector_dynamic_loop_strict():
    extent = tir.Var("extent", "int32")
    src = _buffer("dynamic_src", (128,), "float32", "shared.rsram")
    dst = _buffer("dynamic_dst", (128,), "float32", "shared.rsram")
    i = tir.Var("i", "int32")
    stmt = tir.For(
        i,
        0,
        extent,
        tir.ForKind.SERIAL,
        tir.BufferStore(dst, tir.BufferLoad(src, [i]), [i]),
    )
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with target, pytest.raises(tvm.error.TVMError, match="dynamic_vector_loop_extent"):
        _estimate(VECTOR_CORE, stmt, fallback=False)


def test_default_solver_cycles_preserve_absolute_cycles():
    stmt = _make_copy("global", "shared.rsram", (32, 16))
    result = _estimate(ODMA, stmt)
    assert _integer(result, "latency_cycles") == 71
    assert _integer(result, "solver_latency_cycles") == 71
    assert _integer(result, "solver_scale_factor") == 1
    assert _integer(result, "solver_scaled") == 0
