"""Source-span diagnostics for SunMMIO codegen checks and fatal paths."""

from pathlib import Path

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.utils.target import determine_target


def _line_of(marker: str) -> int:
    for line_number, line in enumerate(Path(__file__).read_text(encoding="utf-8").splitlines(), start=1):
        if marker in line:
            return line_number
    raise AssertionError(f"marker {marker!r} is missing from {__file__}")


def _span(marker: str):
    line = _line_of(marker)
    return tvm.ir.Span(tvm.ir.SourceName(Path(__file__).name), line, line, 1, 80)


def _device_kernel(stmt, params=()):
    return (
        tvm.tir.PrimFunc(list(params), stmt)
        .with_attr("global_symbol", "main")
        .with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
    )


def _build(kernel):
    target = determine_target("Sunmmio", return_object=True)
    module = tvm.IRModule({"main": kernel})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    return builder(module, target, "suvm")


def _assert_diagnostic(kernel, marker: str, expected: str) -> None:
    with pytest.raises(Exception) as exc_info:
        _build(kernel)

    message = str(exc_info.value)
    assert expected in message, message
    assert "at TileLang DSL:" in message, message
    location = f"{Path(__file__).name}:{_line_of(marker)}:"
    assert location in message, message


def _region(buffer, access: int):
    return tvm.tir.call_intrin(
        "handle",
        tvm.ir.Op.get("tl.tileop.region"),
        tvm.tir.BufferLoad(buffer, [0, 0]),
        tvm.tir.IntImm("int32", access),
        tvm.tir.IntImm("int32", 32),
        tvm.tir.IntImm("int32", 32),
    )


def _sync_token():
    return tvm.tir.call_intrin(
        "handle",
        tvm.ir.Op.get("tl.sync_token_id"),
        tvm.tir.IntImm("int32", 0),
    )


def _make_buffers(scopes, dtypes):
    buffers = []
    data_vars = []
    for index, (scope, dtype) in enumerate(zip(scopes, dtypes)):
        element_type = tvm.ir.PrimType(dtype)
        data = tvm.tir.Var(
            f"data_{index}",
            tvm.ir.PointerType(element_type, scope),
        )
        buffer = tvm.tir.decl_buffer(
            (32, 32),
            dtype,
            name=f"Buffer{index}",
            data=data,
            scope=scope,
        )
        data_vars.append(data)
        buffers.append(buffer)
    return buffers, data_vars


def _wrap_buffers(stmt, buffers):
    for buffer in reversed(buffers):
        stmt = tvm.tir.DeclBuffer(buffer, stmt)
    return stmt


def dma_wrong_argument_count_kernel():
    call = tvm.tir.Call(  # DMA_WRONG_ARG_COUNT
        "handle",
        tvm.ir.Op.get("tl.dma_copy"),
        [tvm.tir.IntImm("int32", 0)] * 3,
        span=_span("DMA_WRONG_ARG_COUNT"),
    )
    return _device_kernel(tvm.tir.Evaluate(call))


def test_dma_copy_rejects_wrong_argument_count():
    _assert_diagnostic(
        dma_wrong_argument_count_kernel(),
        "DMA_WRONG_ARG_COUNT",
        "tl.dma_copy expects src region, dst region, src_offset_byte, and sync_token_id",
    )


def dma_nonconstant_offset_kernel():
    buffers, data_vars = _make_buffers(
        ("shared.rsram", "shared.rsram"),
        ("bfloat16", "bfloat16"),
    )
    offset = tvm.tir.Var("runtime_offset", "int32")
    call = tvm.tir.Call(  # DMA_NONCONSTANT_OFFSET
        "handle",
        tvm.ir.Op.get("tl.dma_copy"),
        [_region(buffers[0], 1), _region(buffers[1], 2), offset, _sync_token()],
        span=_span("DMA_NONCONSTANT_OFFSET"),
    )
    stmt = _wrap_buffers(tvm.tir.Evaluate(call), buffers)
    return _device_kernel(stmt, [*data_vars, offset])


def test_dma_copy_rejects_nonconstant_offset():
    _assert_diagnostic(
        dma_nonconstant_offset_kernel(),
        "DMA_NONCONSTANT_OFFSET",
        "tl.dma_copy src_offset_byte must be a constant IntImm",
    )


def dma_negative_offset_kernel():
    buffers, data_vars = _make_buffers(
        ("shared.rsram", "shared.rsram"),
        ("bfloat16", "bfloat16"),
    )
    call = tvm.tir.Call(  # DMA_NEGATIVE_OFFSET
        "handle",
        tvm.ir.Op.get("tl.dma_copy"),
        [_region(buffers[0], 1), _region(buffers[1], 2), -1, _sync_token()],
        span=_span("DMA_NEGATIVE_OFFSET"),
    )
    stmt = _wrap_buffers(tvm.tir.Evaluate(call), buffers)
    return _device_kernel(stmt, data_vars)


def test_dma_copy_rejects_negative_offset():
    _assert_diagnostic(
        dma_negative_offset_kernel(),
        "DMA_NEGATIVE_OFFSET",
        "tl.dma_copy src_offset_byte must be non-negative",
    )


def dma_invalid_token_kernel():
    buffers, data_vars = _make_buffers(
        ("shared.rsram", "shared.rsram"),
        ("bfloat16", "bfloat16"),
    )
    call = tvm.tir.Call(  # DMA_INVALID_TOKEN
        "handle",
        tvm.ir.Op.get("tl.dma_copy"),
        [_region(buffers[0], 1), _region(buffers[1], 2), 0, 7],
        span=_span("DMA_INVALID_TOKEN"),
    )
    stmt = _wrap_buffers(tvm.tir.Evaluate(call), buffers)
    return _device_kernel(stmt, data_vars)


def test_dma_copy_rejects_invalid_sync_token():
    _assert_diagnostic(
        dma_invalid_token_kernel(),
        "DMA_INVALID_TOKEN",
        "tl.dma_copy expects fourth argument to be tl.sync_token_id",
    )


def _mma_buffers():
    return _make_buffers(
        ("shared.asram", "shared.wsram", "shared.rsram"),
        ("bfloat16", "bfloat16", "float32"),
    )


def _mma_args(buffers):
    return [
        _region(buffers[0], 1),
        _region(buffers[1], 1),
        _region(buffers[2], 3),
        tvm.tir.IntImm("bool", 0),
        tvm.tir.IntImm("bool", 0),
        tvm.tir.IntImm("bool", 0),
        tvm.tir.IntImm("int32", 0),
        _sync_token(),
    ]


def mma_wrong_argument_count_kernel():
    call = tvm.tir.Call(  # MMA_WRONG_ARG_COUNT
        "handle",
        tvm.ir.Op.get("tl.mma_sunmmio"),
        [tvm.tir.IntImm("int32", 0)] * 7,
        span=_span("MMA_WRONG_ARG_COUNT"),
    )
    return _device_kernel(tvm.tir.Evaluate(call))


def test_mma_rejects_wrong_argument_count():
    _assert_diagnostic(
        mma_wrong_argument_count_kernel(),
        "MMA_WRONG_ARG_COUNT",
        "tl.mma_sunmmio expects A/B/C regions, three flag operands, acc_offset_byte, and sync_token_id",
    )


def mma_nonconstant_flag_kernel():
    buffers, data_vars = _mma_buffers()
    runtime_flag = tvm.tir.Var("runtime_transpose", "bool")
    args = _mma_args(buffers)
    args[3] = runtime_flag
    call = tvm.tir.Call(  # MMA_NONCONSTANT_FLAG
        "handle",
        tvm.ir.Op.get("tl.mma_sunmmio"),
        args,
        span=_span("MMA_NONCONSTANT_FLAG"),
    )
    stmt = _wrap_buffers(tvm.tir.Evaluate(call), buffers)
    return _device_kernel(stmt, [*data_vars, runtime_flag])


def test_mma_rejects_nonconstant_flag():
    _assert_diagnostic(
        mma_nonconstant_flag_kernel(),
        "MMA_NONCONSTANT_FLAG",
        "tl.mma_sunmmio transA must be a constant bool",
    )


def mma_wrong_flag_dtype_kernel():
    buffers, data_vars = _mma_buffers()
    args = _mma_args(buffers)
    args[4] = tvm.tir.IntImm("int32", 0)
    call = tvm.tir.Call(  # MMA_WRONG_FLAG_DTYPE
        "handle",
        tvm.ir.Op.get("tl.mma_sunmmio"),
        args,
        span=_span("MMA_WRONG_FLAG_DTYPE"),
    )
    stmt = _wrap_buffers(tvm.tir.Evaluate(call), buffers)
    return _device_kernel(stmt, data_vars)


def test_mma_rejects_wrong_flag_dtype():
    _assert_diagnostic(
        mma_wrong_flag_dtype_kernel(),
        "MMA_WRONG_FLAG_DTYPE",
        "tl.mma_sunmmio transB must have bool dtype",
    )


def mma_nonconstant_offset_kernel():
    buffers, data_vars = _mma_buffers()
    runtime_offset = tvm.tir.Var("runtime_acc_offset", "int32")
    args = _mma_args(buffers)
    args[6] = runtime_offset
    call = tvm.tir.Call(  # MMA_NONCONSTANT_OFFSET
        "handle",
        tvm.ir.Op.get("tl.mma_sunmmio"),
        args,
        span=_span("MMA_NONCONSTANT_OFFSET"),
    )
    stmt = _wrap_buffers(tvm.tir.Evaluate(call), buffers)
    return _device_kernel(stmt, [*data_vars, runtime_offset])


def test_mma_rejects_nonconstant_offset():
    _assert_diagnostic(
        mma_nonconstant_offset_kernel(),
        "MMA_NONCONSTANT_OFFSET",
        "tl.mma_sunmmio acc_offset_byte must be a constant IntImm",
    )


def mma_negative_offset_kernel():
    buffers, data_vars = _mma_buffers()
    args = _mma_args(buffers)
    args[6] = tvm.tir.IntImm("int32", -1)
    call = tvm.tir.Call(  # MMA_NEGATIVE_OFFSET
        "handle",
        tvm.ir.Op.get("tl.mma_sunmmio"),
        args,
        span=_span("MMA_NEGATIVE_OFFSET"),
    )
    stmt = _wrap_buffers(tvm.tir.Evaluate(call), buffers)
    return _device_kernel(stmt, data_vars)


def test_mma_rejects_negative_offset():
    _assert_diagnostic(
        mma_negative_offset_kernel(),
        "MMA_NEGATIVE_OFFSET",
        "tl.mma_sunmmio acc_offset_byte must be non-negative",
    )


def mma_invalid_token_kernel():
    buffers, data_vars = _mma_buffers()
    args = _mma_args(buffers)
    args[7] = tvm.tir.IntImm("int32", 9)
    call = tvm.tir.Call(  # MMA_INVALID_TOKEN
        "handle",
        tvm.ir.Op.get("tl.mma_sunmmio"),
        args,
        span=_span("MMA_INVALID_TOKEN"),
    )
    stmt = _wrap_buffers(tvm.tir.Evaluate(call), buffers)
    return _device_kernel(stmt, data_vars)


def test_mma_rejects_invalid_sync_token():
    _assert_diagnostic(
        mma_invalid_token_kernel(),
        "MMA_INVALID_TOKEN",
        "tl.mma_sunmmio expects last argument to be tl.sync_token_id",
    )


def unbound_var_kernel():
    missing = tvm.tir.Var(  # FATAL_UNBOUND_VAR
        "missing_runtime_var",
        "int32",
        span=_span("FATAL_UNBOUND_VAR"),
    )
    stmt = tvm.tir.Evaluate(missing + 1)
    return _device_kernel(stmt)


def test_fatal_reports_unbound_var_span():
    _assert_diagnostic(
        unbound_var_kernel(),
        "FATAL_UNBOUND_VAR",
        "unbound TIR var `missing_runtime_var`",
    )


def allocate_without_buffer_kernel():
    element_type = tvm.ir.PrimType("bfloat16")
    data = tvm.tir.Var(
        "orphan_asram",
        tvm.ir.PointerType(element_type, "shared.asram"),
    )
    body = tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0))
    stmt = tvm.tir.Allocate(  # FATAL_ALLOCATE_WITHOUT_BUFFER
        data,
        "bfloat16",
        [32, 32],
        tvm.tir.IntImm("bool", 1),
        body,
        span=_span("FATAL_ALLOCATE_WITHOUT_BUFFER"),
    )
    return _device_kernel(stmt)


def test_fatal_reports_allocate_without_buffer_span():
    _assert_diagnostic(
        allocate_without_buffer_kernel(),
        "FATAL_ALLOCATE_WITHOUT_BUFFER",
        "SunMMIO SUVM allocate cannot find buffer for variable orphan_asram",
    )


@tilelang.jit(target="sunmmio", execution_backend="sunmmio_sunsim")
def unsupported_tiles_expr_jit_kernel():
    @T.prim_func
    def main(
        source: T.Tensor((32, 32), "float32"),
        output: T.Tensor((32, 32), "float32"),
    ):
        with T.Kernel():
            source_shared = T.alloc_shared(
                (32, 32),
                "float32",
                scope="shared.rsram",
            )
            output_shared = T.alloc_shared(
                (32, 32),
                "float32",
                scope="shared.rsram",
            )
            T.copy(source[0, 0], source_shared)
            for i, j in T.Tiles([32, 32]):
                output_shared[i, j] = T.sin(source_shared[i, j])  # FATAL_UNSUPPORTED_EXPR
            T.copy(output_shared, output[0, 0])

    return main


def test_fatal_reports_jit_unsupported_expr_span():
    with pytest.raises(Exception) as exc_info:
        unsupported_tiles_expr_jit_kernel()

    message = str(exc_info.value)
    assert "CodeGenTileLangSunMMIO unsupported expr: tir.Call" in message
    assert "selected unary math calls" in message
    assert "at TileLang DSL:" in message
    location = f"{Path(__file__).name}:{_line_of('FATAL_UNSUPPORTED_EXPR')}:"
    assert location in message, message


if __name__ == "__main__":
    tilelang.testing.main()
