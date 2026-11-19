"""Sunmmio device library operations on rank-two RSRAM regions."""

from __future__ import annotations

from tvm import tir
from tilelang._typing import BufferLikeType
from tilelang.language.mesh_tensor import _unwrap_mesh_tensor
from tilelang.language.utils import get_extent
from tilelang.utils.language import to_buffer_region


def _device_lib(name, *operands):
    """Frontend intrinsic constructor used by the TIR script printer."""
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.device_lib"), name, *operands)


def _device_lib_call(name, *operands):
    """Lowered intrinsic constructor used by the TIR script printer."""
    return tir.call_intrin("handle", tir.op.Op.get("tl.device_lib"), name, *operands)


def _region(buffer: BufferLikeType, access: str):
    buffer = _unwrap_mesh_tensor(buffer)
    extents = list(buffer.shape) if isinstance(buffer, tir.Buffer) else get_extent(buffer)
    return to_buffer_region(buffer, access_type=access, extents=extents)


def topk(input: BufferLikeType, values: BufferLikeType, indices: BufferLikeType, k: int | tir.PrimExpr):
    """Select each row's k largest values, modifying input in place.

    Input and values are BF16 or FP32; indices are int32, relative to the
    input region. Equal values select the smaller column first. All operands
    are rank-two RSRAM regions with equal row counts and non-overlapping
    storage. Input may be ZZ or row-major; outputs are dense row-major.
    ZZ regions must contain whole hardware blocks (32x32 on A4E). Region
    starts must be multiples of their extents; unsupported slices are rejected.
    Views retain the requested extents rather than including layout padding.

    Selected input elements become -inf. Padding must contain -inf, and each
    row must have at least k elements greater than -inf. Output widths must
    hold k elements and have vector-aligned explicit widths; unused values/indices are
    filled with -inf/-1. This is a local operation, with no cross-core merge.
    """
    return _device_lib(
        "topk",
        _region(input, "rw"),
        _region(values, "w"),
        _region(indices, "w"),
        tir.Cast("int32", k),
    )


def _unary(name, input, output):
    return _device_lib(name, _region(input, "r"), _region(output, "w"))


def exp_refined(input: BufferLikeType, output: BufferLikeType):
    """FP32 exp with one PWL correction; normal-result relative error <1e-5.

    Rank-two RSRAM regions must have equal shapes and matching row-major or ZZ
    layouts. Exact in-place operation is allowed; partial overlap is not.
    Results are approximate, not correctly rounded; subnormals may flush early.
    The vector PWL coefficient tables must have their default contents.
    ZZ regions require whole hardware blocks. Region starts must be multiples
    of their extents; implicit padding and rounded-down slice starts are not used.
    """
    return _unary("exp_fast", input, output)


def log_refined(input: BufferLikeType, output: BufferLikeType):
    """FP32 natural log with one PWL refinement, not correctly rounded.

    Outside [0.5, 2], relative error <3.5e-6; inside, absolute error <2e-6.
    Operand layout, aliasing and coefficient requirements match exp_refined.
    """
    return _unary("ln_fast", input, output)


def reciprocal_refined(input: BufferLikeType, output: BufferLikeType):
    """FP32 reciprocal with one Newton refinement, not correctly rounded.

    Error is at most 5 ULP except early saturation for abs(x) < 2**-127.
    Operand layout, aliasing and coefficient requirements match exp_refined.
    """
    return _unary("recip_fast", input, output)


def rsqrt_refined(input: BufferLikeType, output: BufferLikeType):
    """FP32 reciprocal square root with one Newton refinement (up to 4 ULP).

    Operand layout, aliasing and coefficient requirements match exp_refined.
    Results are approximate, not correctly rounded.
    """
    return _unary("rsqrt_fast", input, output)
