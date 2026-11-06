"""MX physical pack/unpack operations."""

from __future__ import annotations

from functools import partial

from tilelang._typing import BufferLikeType
from tilelang.language.mesh_tensor import _unwrap_mesh_tensor
from tilelang.language.utils import get_extent
from tilelang.utils.language import to_buffer_region
from tvm import tir
from tvm.script.ir_builder import tir as ir_builder
from tvm.target import Target
from tilelang.utils.target import target_is_sunmmio


def mx_view(mx: tir.Buffer) -> tuple[tir.Buffer, tir.Buffer]:
    """Return writable payload and scale aliases of a local SunMMIO MX buffer.

    The parent owns the storage. Creating views neither allocates nor copies
    data. Scale dimensions are resolved from the parent's final layout during
    compilation and can be used in TIR loops and regions, but not as Python
    integers. Use the fields in ``T.Tiles`` or legal RSRAM region copies;
    standalone scalar access is unsupported.

    Supports static rank-2 MXFP4/MXFP8 parents with MXZZ, MXZNZ, or MX
    row-major layout. Declare views once per parent allocation. Automatic
    pipelining involving the parent or its fields is unsupported.
    """
    if not isinstance(mx, tir.Buffer):
        raise TypeError("T.mx_view expects a complete allocated MX Buffer")
    target = Target.current(allow_none=True)
    if target is not None and not target_is_sunmmio(target):
        raise ValueError("T.mx_view supports only SunMMIO")
    if str(mx.dtype) not in {"mxfp8", "mxfp4", "custom[mxfp8]", "custom[mxfp4]", "custom[mxfp8]8", "custom[mxfp4]4"}:
        raise ValueError("T.mx_view expects MXFP8 or MXFP4")
    if len(mx.shape) != 2 or not all(isinstance(x, tir.IntImm) and int(x) > 0 for x in mx.shape):
        raise ValueError("T.mx_view expects a static rank-2 MX buffer")
    if mx.scope() not in {"shared.rsram", "shared", "shared.dyn"}:
        raise ValueError("T.mx_view expects a local shared.rsram allocation")
    data_dtype = "float8_e4m3fn" if str(mx.dtype) in {"mxfp8", "custom[mxfp8]", "custom[mxfp8]8"} else "float4_e2m1fn"
    scale_shape = [tir.call_intrin("int32", tir.op.Op.get("tl.mx_scale_extent"), mx.data, i) for i in range(2)]
    declaration = tir.Var("mx_view_declaration", "int32")
    fields = []
    for kind, dtype, shape in [(0, data_dtype, mx.shape), (1, "float8_e8m0fnu", scale_shape)]:
        buffer = tir.decl_buffer(shape, dtype, name="mx_data" if kind == 0 else "mx_scale", scope="shared.rsram")
        value = tir.call_intrin(
            "handle",
            tir.op.Op.get("tl.mx_field_handle"),
            mx.data,
            kind,
            annotations={"tl.mx_parent": mx, "tl.mx_declaration": declaration},
        )
        frame = ir_builder.LetStmt(value, var=buffer.data)
        frame.add_callback(partial(frame.__exit__, None, None, None))
        frame.__enter__()
        decl = ir_builder.decl_buffer(shape, dtype, data=buffer.data, scope="shared.rsram")
        decl.add_callback(partial(decl.__exit__, None, None, None))
        fields.append(decl.__enter__())
    return tuple(fields)


def _full_region_arg(buffer: BufferLikeType, access_type: str):
    buffer = _unwrap_mesh_tensor(buffer)
    extents = get_extent(buffer)
    if extents is None:
        if isinstance(buffer, tir.Buffer):
            extents = list(buffer.shape)
        else:
            raise TypeError(f"Cannot deduce full region extent from {type(buffer)}")
    return to_buffer_region(buffer, access_type=access_type, extents=list(extents))


def mx_pack(data: BufferLikeType, scale: BufferLikeType, mx: BufferLikeType) -> tir.PrimExpr:
    """Copy prepared data/scale buffers into an MX container buffer."""
    data_region = _full_region_arg(data, "r")
    scale_region = _full_region_arg(scale, "r")
    mx_region = _full_region_arg(mx, "w")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.mx_pack"),
        data_region,
        scale_region,
        mx_region,
    )


def mx_unpack(mx: BufferLikeType, data: BufferLikeType, scale: BufferLikeType) -> tir.PrimExpr:
    """Copy data/scale fields out of an MX container buffer."""
    mx_region = _full_region_arg(mx, "r")
    data_region = _full_region_arg(data, "w")
    scale_region = _full_region_arg(scale, "w")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.mx_unpack"),
        mx_region,
        data_region,
        scale_region,
    )
