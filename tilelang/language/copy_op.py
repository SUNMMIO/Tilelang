"""Copy operations exposed on the TileLang language surface."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Any
import warnings
from tilelang._typing import BufferLikeType
from tilelang.utils.language import (
    to_buffer_region,
    prim_expr_equal,
    legalize_pairwise_extents,
)
from tilelang.language.utils import get_extent
from tilelang.utils.target import target_is_sunmmio
from tvm import ir, tir
from tvm.target import Target


_OperandKind = Literal["buffer", "region", "load"]


@dataclass(frozen=True)
class _CopyRegionSpec:
    original: Any
    kind: _OperandKind
    buffer: tir.Buffer
    mins: list[tir.PrimExpr]
    extents: list[tir.PrimExpr] | None
    explicit_extents: bool


@dataclass(frozen=True)
class _NormalizedCopyRegion:
    spec: _CopyRegionSpec
    mins: list[tir.PrimExpr]
    extents: list[tir.PrimExpr]


def _resolve_let_value(obj: Any) -> Any:
    from tilelang.language.frame import has_let_value, get_let_value

    if isinstance(obj, tir.Var) and has_let_value(obj):
        return get_let_value(obj)
    return obj


def _extract_copy_region_spec(obj: BufferLikeType) -> _CopyRegionSpec:
    obj = _resolve_let_value(obj)
    if isinstance(obj, tir.Buffer):
        mins = [tir.IntImm("int32", 0) for _ in obj.shape]
        return _CopyRegionSpec(obj, "buffer", obj, mins, list(obj.shape), False)
    if isinstance(obj, tir.BufferRegion):
        mins = [r.min for r in obj.region]
        extents = [r.extent for r in obj.region]
        return _CopyRegionSpec(obj, "region", obj.buffer, mins, extents, True)
    if isinstance(obj, tir.BufferLoad):
        return _CopyRegionSpec(obj, "load", obj.buffer, list(obj.indices), None, False)
    raise TypeError(f"Unsupported argument type for T.copy: {type(obj)}")


def _as_static_int(expr: Any) -> int | None:
    if isinstance(expr, int):
        return expr
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    return None


def _expr_is_one(expr: tir.PrimExpr) -> bool:
    return prim_expr_equal(expr, 1)


def _expr_is_squeezable_one(expr: tir.PrimExpr) -> bool:
    if _expr_is_one(expr):
        return True
    if isinstance(expr, tir.Min):
        return _expr_is_one(expr.a) or _expr_is_one(expr.b)
    return False


def _expr_equal(lhs: tir.PrimExpr, rhs: tir.PrimExpr) -> bool:
    return prim_expr_equal(lhs, rhs)


def _expr_gt(lhs: tir.PrimExpr, rhs: tir.PrimExpr) -> bool:
    lhs_int = _as_static_int(lhs)
    rhs_int = _as_static_int(rhs)
    return lhs_int is not None and rhs_int is not None and lhs_int > rhs_int


# Warn when an explicit BufferRegion exceeds the buffer shape.
def _warn_explicit_oob(buffer: tir.Buffer, dim: int, min_value: tir.PrimExpr, extent: tir.PrimExpr, shape: tir.PrimExpr) -> None:
    warnings.warn(
        "T.copy explicit BufferRegion exceeds buffer shape and will be clipped: "
        f"{buffer.name}[dim={dim}], min={min_value}, extent={extent}, shape={shape}",
        stacklevel=3,
    )


# Clip extents so the region stays within the buffer shape.
def _clip_extent_to_shape(
    buffer: tir.Buffer,
    dim: int,
    min_value: tir.PrimExpr,
    extent: tir.PrimExpr,
    shape: tir.PrimExpr,
    *,
    warn_if_clipped: bool,
) -> tir.PrimExpr:
    min_int = _as_static_int(min_value)
    extent_int = _as_static_int(extent)
    shape_int = _as_static_int(shape)

    if min_int is not None and shape_int is not None and (min_int < 0 or min_int >= shape_int):
        raise ValueError(
            f"T.copy region starts outside buffer shape: {buffer.name}[dim={dim}], min={min_value}, extent={extent}, shape={shape}"
        )

    if min_int is not None and extent_int is not None and shape_int is not None:
        available = shape_int - min_int
        clipped = min(extent_int, available)
        if clipped < extent_int and warn_if_clipped:
            _warn_explicit_oob(buffer, dim, min_value, extent, shape)
        return tir.IntImm(extent.dtype if hasattr(extent, "dtype") else "int32", clipped)

    return extent


def _int_one() -> tir.IntImm:
    return tir.IntImm("int32", 1)


def _infer_load_extents_from_peer(load: _CopyRegionSpec, peer_extents: list[tir.PrimExpr]) -> list[tir.PrimExpr]:
    rank = len(load.mins)
    extents = list(peer_extents)
    if len(extents) < rank:
        return [_int_one() for _ in range(rank - len(extents))] + extents
    return extents[-rank:]


def _clip_region_to_shape(spec: _CopyRegionSpec, mins: list[tir.PrimExpr], extents: list[tir.PrimExpr]) -> _NormalizedCopyRegion:
    if len(mins) != len(extents) or len(extents) != len(spec.buffer.shape):
        raise ValueError(
            "T.copy region rank does not match buffer rank before clipping: "
            f"{spec.buffer.name}, mins={len(mins)}, extents={len(extents)}, shape={len(spec.buffer.shape)}"
        )
    clipped_extents = [
        _clip_extent_to_shape(
            spec.buffer,
            dim,
            min_value,
            extent,
            spec.buffer.shape[dim],
            warn_if_clipped=spec.explicit_extents,
        )
        for dim, (min_value, extent) in enumerate(zip(mins, extents))
    ]
    return _NormalizedCopyRegion(spec, list(mins), clipped_extents)


def _normalize_copy_regions(src: _CopyRegionSpec, dst: _CopyRegionSpec) -> tuple[_NormalizedCopyRegion, _NormalizedCopyRegion]:
    if src.kind == "load" and dst.kind == "load":
        raise AssertionError("BufferLoad-to-BufferLoad copy should be handled by the scalar BufferStore fast path")

    if src.kind == "load" and dst.kind == "region":
        assert dst.extents is not None
        dst_region = _clip_region_to_shape(dst, dst.mins, list(dst.extents))
        src_extents = _infer_load_extents_from_peer(src, dst_region.extents)
        src_region = _clip_region_to_shape(src, src.mins, src_extents)
        return src_region, dst_region

    if src.kind == "region" and dst.kind == "load":
        assert src.extents is not None
        src_region = _clip_region_to_shape(src, src.mins, list(src.extents))
        dst_extents = _infer_load_extents_from_peer(dst, src_region.extents)
        dst_region = _clip_region_to_shape(dst, dst.mins, dst_extents)
        return src_region, dst_region

    src_extents = src.extents
    dst_extents = dst.extents

    if src.kind == "load":
        assert dst_extents is not None
        src_extents = _infer_load_extents_from_peer(src, dst_extents)
    if dst.kind == "load":
        assert src_extents is not None
        dst_extents = _infer_load_extents_from_peer(dst, src_extents)

    assert src_extents is not None and dst_extents is not None
    src_region = _clip_region_to_shape(src, src.mins, list(src_extents))
    dst_region = _clip_region_to_shape(dst, dst.mins, list(dst_extents))
    return src_region, dst_region


def _format_extents(extents: list[tir.PrimExpr]) -> str:
    return "[" + ", ".join(str(extent) for extent in extents) + "]"


def _suffix_axis_map(src: _NormalizedCopyRegion, dst: _NormalizedCopyRegion) -> list[tuple[int, int]]:
    src_rank = len(src.extents)
    dst_rank = len(dst.extents)
    matched_rank = min(src_rank, dst_rank)

    if src_rank > dst_rank:
        extra_extents = src.extents[: src_rank - dst_rank]
        for dim, extent in enumerate(extra_extents):
            if not _expr_is_one(extent):
                raise ValueError(
                    "T.copy rank mismatch: src has non-1 extra leading dimension "
                    f"at dim {dim}, extent={extent}; src={_format_extents(src.extents)}, "
                    f"dst={_format_extents(dst.extents)}"
                )
        return [(src_rank - matched_rank + i, i) for i in range(matched_rank)]

    if dst_rank > src_rank:
        extra_extents = dst.extents[: dst_rank - src_rank]
        for dim, extent in enumerate(extra_extents):
            if not _expr_is_one(extent):
                raise ValueError(
                    "T.copy rank mismatch: dst has non-1 extra leading dimension "
                    f"at dim {dim}, extent={extent}; src={_format_extents(src.extents)}, "
                    f"dst={_format_extents(dst.extents)}"
                )
        return [(i, dst_rank - matched_rank + i) for i in range(matched_rank)]

    return [(i, i) for i in range(matched_rank)]


def _squeezed_axis_map(src: _NormalizedCopyRegion, dst: _NormalizedCopyRegion) -> list[tuple[int, int]]:
    src_axes = [(dim, extent) for dim, extent in enumerate(src.extents) if not _expr_is_squeezable_one(extent)]
    dst_axes = [(dim, extent) for dim, extent in enumerate(dst.extents) if not _expr_is_squeezable_one(extent)]

    if len(src_axes) != len(dst_axes):
        raise ValueError(
            "T.copy rank mismatch: mixed-region copy requires the same number of "
            "non-1 extents after squeezing unit dimensions; "
            f"src={_format_extents(src.extents)}, dst={_format_extents(dst.extents)}"
        )

    return [(src_dim, dst_dim) for (src_dim, _), (dst_dim, _) in zip(src_axes, dst_axes)]


def _validate_and_adjust_copy_regions(
    src: _NormalizedCopyRegion,
    dst: _NormalizedCopyRegion,
    *,
    require_exact_match: bool,
) -> tuple[_NormalizedCopyRegion, _NormalizedCopyRegion]:
    axis_map = _suffix_axis_map(src, dst) if require_exact_match else _squeezed_axis_map(src, dst)
    dst_extents = list(dst.extents)

    for src_dim, dst_dim in axis_map:
        src_extent = src.extents[src_dim]
        dst_extent = dst.extents[dst_dim]

        if require_exact_match:
            if not _expr_equal(src_extent, dst_extent):
                raise ValueError(
                    "T.copy extent mismatch: exact match is required for Buffer-to-Buffer copy; "
                    f"src dim {src_dim} extent={src_extent}, dst dim {dst_dim} extent={dst_extent}; "
                    f"src={_format_extents(src.extents)}, dst={_format_extents(dst.extents)}"
                )
            continue

        if _expr_gt(src_extent, dst_extent):
            raise ValueError(
                "T.copy extent mismatch: src extent is larger than dst extent at matched axis; "
                f"src dim {src_dim} extent={src_extent}, dst dim {dst_dim} extent={dst_extent}; "
                f"src={_format_extents(src.extents)}, dst={_format_extents(dst.extents)}"
            )
        if not _expr_equal(src_extent, dst_extent):
            dst_extents[dst_dim] = src_extent

    return src, _NormalizedCopyRegion(dst.spec, dst.mins, dst_extents)


def _encode_normalized_region(region: _NormalizedCopyRegion, access_type: str) -> tir.PrimExpr:
    return to_buffer_region(region.spec.original, access_type=access_type, extents=region.extents)


def copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    coalesced_width: int | None = None,
    disable_tma: bool = False,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    annotations: dict | None = None,
    loop_layout: Any | None = None,
) -> tir.PrimExpr | tir.Stmt:
    """Copy data between memory regions.

    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source memory region
        dst (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Destination memory region
        coalesced_width (Optional[int], keyword-only): Width for coalesced memory access. Defaults to None.
        disable_tma (bool, keyword-only): Whether to disable TMA acceleration. Defaults to False.
        eviction_policy (Optional[str], keyword-only): Cache eviction policy. Defaults to None.
        annotations (Optional[dict], keyword-only): Additional annotations dict. If provided,
            coalesced_width, disable_tma, and eviction_policy can also be specified here.
            Values in annotations take precedence over individual arguments.
        loop_layout (Optional[Fragment], keyword-only): A parallel loop layout hint for the SIMT copy
            (only valid for normal SIMT copy; incompatible with TMA/LDSM/STSM/TMem). When provided,
            it is attached to the outermost parallel loop generated by this copy.

    Raises:
        TypeError: If copy extents cannot be deduced from arguments

    Returns:
        tir.Call: A handle to the copy operation

    Range handling notes:
    - `Buffer` copies the full shape, `BufferRegion` uses explicit ranges, and
      `BufferLoad` starts from its indices with extents inferred from the peer.
    - Scalar `BufferLoad -> BufferLoad` keeps the legacy fast path and lowers to
      a direct `BufferStore`.
    - Rank-mismatched copies use suffix matching only: extra leading dimensions
      on the higher-rank side must have extent 1.
    - For mixed region copies, `dst` may be shrunk to the matched `src` extents
      when `src <= dst`; static `src > dst` is rejected.
    """
    src = _resolve_let_value(src)
    dst = _resolve_let_value(dst)
    src_extent = get_extent(src)
    dst_extent = get_extent(dst)
    # Combine the nested if statements into a single if statement as suggested by SIM102
    if src_extent is None and dst_extent is None and isinstance(src, tir.BufferLoad) and isinstance(dst, tir.BufferLoad):
        # FIXME(sunmmio): For Sunmmio an invalid D<->D copy operation will enter here
        # check if the case is like this:
        # copy(buffer_a[i], buffer_b[i]) where both are BufferLoad nodes
        # In this case, lower it to a simple BufferStore: buffer_b[i] = buffer_a[i]
        return tir.BufferStore(dst.buffer, src, dst.indices)

    target = Target.current(allow_none=True)
    if target and target_is_sunmmio(target):
        # get original info. e.g. T.copy(A, C) -> src_spec = {kind: "buffer", buffer = A, mins = [0,0,0], extents = A.shape, explicit_extents = False}
        src_spec = _extract_copy_region_spec(src)
        dst_spec = _extract_copy_region_spec(dst)
        # transfer spec into normalized region: buffer + mins + extents
        src_region, dst_region = _normalize_copy_regions(src_spec, dst_spec)
        # check rank and extents
        src_region, dst_region = _validate_and_adjust_copy_regions(
            src_region,
            dst_region,
            require_exact_match=src_spec.kind == "buffer" and dst_spec.kind == "buffer",
        )
        src = _encode_normalized_region(src_region, access_type="r")
        dst = _encode_normalized_region(dst_region, access_type="w")
    else:
        if isinstance(src, tir.Buffer) and isinstance(dst, tir.Buffer):
            ir.assert_structural_equal(src.shape, dst.shape)
        assert src_extent or dst_extent, "Can't deduce copy extents from args"
        src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
        dst_extent = list(dst_extent) if dst_extent else [1] * len(src_extent)
        src_extent, dst_extent = legalize_pairwise_extents(src_extent, dst_extent)
        src = to_buffer_region(src, access_type="r", extents=src_extent)
        dst = to_buffer_region(dst, access_type="w", extents=dst_extent)

    # Build annotations dict
    ann = annotations.copy() if annotations else {}

    # Individual arguments take lower precedence than annotations
    if "coalesced_width" not in ann and coalesced_width is not None:
        ann["coalesced_width"] = coalesced_width
    if "disable_tma" not in ann and disable_tma:
        ann["disable_tma"] = disable_tma
    if "eviction_policy" not in ann and eviction_policy is not None:
        eviction_policy_map = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}
        ann["eviction_policy"] = eviction_policy_map[eviction_policy]

    # Parallel loop layout hint (Fragment). Mirrors T.Parallel(loop_layout=...)
    if loop_layout is not None and "parallel_loop_layout" not in ann:
        ann["parallel_loop_layout"] = loop_layout

    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.copy"), src, dst, annotations=ann if ann else None)


def c2d_im2col(
    img: BufferLikeType,
    col: BufferLikeType,
    nhw_step: tir.PrimExpr,
    c_step: tir.PrimExpr,
    kernel: int,
    stride: int,
    dilation: int,
    pad: int,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
) -> tir.PrimExpr:
    """Perform im2col transformation for 2D convolution.

    Args:
        img (tir.Buffer): Input image buffer
        col (tir.Buffer): Output column buffer
        nhw_step (tir.PrimExpr): Step size for batch and spatial dimensions
        c_step (tir.PrimExpr): Step size for channel dimension
        kernel (int): Kernel size
        stride (int): Stride of the convolution
        dilation (int): Dilation rate
        pad (int): Padding size

    Returns:
        tir.Call: A handle to the im2col operation
    """
    if eviction_policy is None:
        eviction_policy = 0
    else:
        eviction_policy = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}[eviction_policy]
    img_region = to_buffer_region(img, access_type="r")
    col_region = to_buffer_region(col, access_type="w")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.c2d_im2col"),
        img_region,
        col_region,
        nhw_step,
        c_step,
        kernel,
        stride,
        dilation,
        pad,
        eviction_policy,
    )
