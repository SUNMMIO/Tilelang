"""Shared MX kernels for opt validation and external numerical tests.

This module contains kernel definitions only; it has no simulator dependency.
"""

import tilelang.language as T
from tilelang.layout import (
    make_aligned_row_major,
    make_mx_row_major_layout,
    make_mxznz_layout,
    make_mxzz_layout,
    make_row_major,
    make_zz_layout,
    get_mx_scale_shape,
)
from testing.python.sunmmio.common.compile_pipeline import target


@target("sunmmio")
def mx_row_major_numeric_kernel(k=64, bits=8, operation="pack", use_view=False):
    rows = 32
    dtype = T.float8_e4m3fn if bits == 8 else T.float4_e2m1fn
    mx_dtype = T.mxfp8 if bits == 8 else T.mxfp4
    data_cols = ((k * bits + 511) // 512) * 512 // bits
    scale_cols = ((k // 32 + 63) // 64) * 64
    padded_dram_rows = data_cols != k and (data_cols * bits // 8) % 1024 == 0
    global_data_cols = k if padded_dram_rows else data_cols
    global_data_layout = make_aligned_row_major((rows, global_data_cols), dtype, 64)
    shape = (rows, k)
    scale_shape = (rows, k // 32)
    mx_layout = make_mx_row_major_layout(shape, dtype=mx_dtype)
    data_layout = make_aligned_row_major(shape, dtype, 64)
    scale_layout = make_aligned_row_major(scale_shape, T.float8_e8m0fnu, 64)
    replicated = T.placement.replicated()

    @T.prim_func
    def main(
        Data: T.MeshTensor((rows, global_data_cols), replicated, dtype, layout=global_data_layout),
        Scale: T.MeshTensor((rows, scale_cols), replicated, T.float8_e8m0fnu, layout=make_row_major((rows, scale_cols))),
        MX: T.MeshTensor(shape, replicated, mx_dtype, layout=mx_layout),
        DataAfter: T.MeshTensor((rows, global_data_cols), replicated, dtype, layout=global_data_layout),
        ScaleAfter: T.MeshTensor((rows, scale_cols), replicated, T.float8_e8m0fnu, layout=make_row_major((rows, scale_cols))),
    ):
        with T.Kernel():
            data = T.alloc_shared(shape, dtype, scope="shared.rsram")
            scale = T.alloc_shared(scale_shape, T.float8_e8m0fnu, scope="shared.rsram")
            mx = T.alloc_shared(shape, mx_dtype, scope="shared.rsram")
            data_after = T.alloc_shared(shape, dtype, scope="shared.rsram")
            scale_after = T.alloc_shared(scale_shape, T.float8_e8m0fnu, scope="shared.rsram")
            T.annotate_layout({data: data_layout, scale: scale_layout, mx: mx_layout, data_after: data_layout, scale_after: scale_layout})
            # Explicit physical regions seed and export the existing padding.
            # mx_pack/unpack themselves always receive full logical buffers.
            # For 1024B-padded rows, the existing DMA carrier expands matching
            # full logical rows to their physical widths automatically.
            if padded_dram_rows:
                T.copy(Data, data)
                T.copy(DataAfter, data_after)
            else:
                T.copy(Data, data[:, :data_cols])
                T.copy(DataAfter, data_after[:, :data_cols])
            T.copy(Scale, scale[:, :scale_cols])
            T.copy(MX, mx)
            T.copy(ScaleAfter, scale_after[:, :scale_cols])
            if use_view:
                mx_data, mx_scale = T.mx_view(mx)
            if operation == "pack":
                if use_view:
                    T.copy(data, mx_data)
                    T.copy(scale, mx_scale)
                else:
                    T.mx_pack(data, scale, mx)
            elif operation == "unpack":
                if use_view:
                    T.copy(mx_data, data)
                    T.copy(mx_scale, scale)
                else:
                    T.mx_unpack(mx, data, scale)
            else:
                if use_view:
                    T.copy(data, mx_data)
                    T.copy(scale, mx_scale)
                    T.copy(mx_data, data_after)
                    T.copy(mx_scale, scale_after)
                else:
                    T.mx_pack(data, scale, mx)
                    T.mx_unpack(mx, data_after, scale_after)
            if padded_dram_rows:
                T.copy(data, Data)
                T.copy(data_after, DataAfter)
            else:
                T.copy(data[:, :data_cols], Data)
                T.copy(data_after[:, :data_cols], DataAfter)
            T.copy(scale[:, :scale_cols], Scale)
            T.copy(mx, MX)
            T.copy(scale_after[:, :scale_cols], ScaleAfter)

    return main


@target("sunmmio")
def mx_view_quant_numeric_kernel(bits=8, use_view=True):
    shape = (32, 32)
    mx_dtype = T.mxfp8 if bits == 8 else T.mxfp4
    data_dtype = T.float8_e4m3fn if bits == 8 else T.float4_e2m1fn
    maximum = 448.0 if bits == 8 else 6.0
    layout = make_zz_layout(shape)
    mx_layout = make_mxzz_layout(shape, dtype=mx_dtype)
    scale_shape = tuple(int(x) for x in get_mx_scale_shape(mx_layout, mx_dtype))
    replicated = T.placement.replicated()

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, replicated, T.bfloat16, layout=layout),
        Y: T.MeshTensor(shape, replicated, T.bfloat16, layout=layout),
        Payload: T.MeshTensor(shape, replicated, T.bfloat16, layout=layout),
        Scales: T.MeshTensor(shape, replicated, T.bfloat16, layout=layout),
    ):
        with T.Kernel():
            a = T.alloc_shared(shape, T.bfloat16, scope="shared.rsram")
            amax = T.alloc_shared((32,), T.float32, scope="shared.rsram")
            mx = T.alloc_shared(shape, mx_dtype, scope="shared.rsram")
            T.annotate_layout({mx: mx_layout})
            if use_view:
                data, scale = T.mx_view(mx)
                restored_data = data
                restored_scale = scale
            else:
                data = T.alloc_shared(shape, data_dtype, scope="shared.rsram")
                scale = T.alloc_shared(scale_shape, T.float8_e8m0fnu, scope="shared.rsram")
                restored_data = T.alloc_shared(shape, data_dtype, scope="shared.rsram")
                restored_scale = T.alloc_shared(scale_shape, T.float8_e8m0fnu, scope="shared.rsram")
            y = T.alloc_shared(shape, T.bfloat16, scope="shared.rsram")
            payload = T.alloc_shared(shape, T.bfloat16, scope="shared.rsram")
            scales = T.alloc_shared(shape, T.bfloat16, scope="shared.rsram")

            T.copy(A, a)
            T.reduce_absmax(a, amax, dim=1, clear=True)
            for row in T.Tiles(amax):
                scale[0, row] = T.Cast(
                    T.float8_e8m0fnu,
                    T.Cast(T.bfloat16, T.max(amax[row], T.float32(1e-4)) * T.float32(1.0 / maximum)),
                )
            for row, col in T.Tiles(data):
                scale_fp32 = T.Cast(T.float32, T.Cast(T.bfloat16, scale[0, row]))
                normalized = T.Cast(T.float32, a[row, col]) / scale_fp32
                data[row, col] = T.Cast(data_dtype, T.Cast(T.bfloat16, T.clamp(normalized, -maximum, maximum)))
            if not use_view:
                T.mx_pack(data, scale, mx)
                T.mx_unpack(mx, restored_data, restored_scale)
            for row, col in T.Tiles(y):
                payload[row, col] = T.Cast(T.bfloat16, restored_data[row, col])
                scales[row, col] = T.Cast(T.bfloat16, restored_scale[0, row])
                y[row, col] = T.Cast(T.bfloat16, T.Cast(T.float32, payload[row, col]) * T.Cast(T.float32, scales[row, col]))
            T.copy(y, Y)
            T.copy(payload, Payload)
            T.copy(scales, Scales)

    return main


@target("sunmmio")
def mx_view_region_numeric_kernel(case, reverse=False, bits=8):
    count = 2 if case in ("negative_bounds", "negative_end", "both_negative") else 1
    if case in ("auto_shape", "auto_static"):
        count = 4
    scope = "shared.dyn" if case in ("auto_shape", "auto_static") else "shared.rsram"
    mx_dtype = T.mxfp8 if bits == 8 else T.mxfp4

    @T.prim_func
    def main(
        Input: T.MeshTensor((32, 32), T.placement.replicated(), T.bfloat16, layout=make_zz_layout((32, 32))),
        Output: T.MeshTensor((32, 32), T.placement.replicated(), T.bfloat16, layout=make_zz_layout((32, 32))),
    ):
        with T.Kernel() as cid:
            mx = T.alloc_shared((64, 64), mx_dtype, scope="shared.rsram")
            data, scale = T.mx_view(mx)
            if case == "auto_shape":
                peer = T.alloc_shared(scale.shape, T.float8_e8m0fnu)
            else:
                peer = T.alloc_shared((count, 32), T.float8_e8m0fnu, scope=scope)
            out = T.alloc_shared((32, 32), T.bfloat16, scope="shared.rsram")
            seed = T.alloc_shared((32, 32), T.bfloat16, scope="shared.rsram")
            T.copy(Input, seed)
            for row, j in T.Tiles(scale.shape):
                scale[row, j] = T.Cast(T.float8_e8m0fnu, seed[row, j])
            for row, j in T.Tiles(peer.shape):
                peer[row, j] = T.Cast(T.float8_e8m0fnu, seed[row + 4, j])
            if case == "negative_last":
                if reverse:
                    T.copy(peer, scale[-1:, :])
                else:
                    T.copy(scale[-1:, :], peer)
            elif case == "negative_bounds":
                if reverse:
                    T.copy(peer, scale[-3:-1, :])
                else:
                    T.copy(scale[-3:-1, :], peer)
            elif case == "negative_end":
                if reverse:
                    T.copy(peer, scale[1:-1, :])
                else:
                    T.copy(scale[1:-1, :], peer)
            elif case == "both_negative":
                if reverse:
                    T.copy(peer[-2:, :], scale[-3:-1, :])
                else:
                    T.copy(scale[-3:-1, :], peer[-2:, :])
            elif case == "let":
                row = cid % 4
                alias = row + 1
                index = alias - 1
                if reverse:
                    T.copy(peer, scale[index : index + 1, :])
                else:
                    T.copy(scale[index : index + 1, :], peer)
            elif case == "loop":
                for i in T.serial(4):
                    negative_row = i - 4
                    if reverse:
                        T.copy(peer, scale[negative_row, 0])
                    else:
                        T.copy(scale[negative_row, 0], peer)
            else:
                if reverse:
                    T.copy(peer, scale)
                else:
                    T.copy(scale, peer)
            if reverse:
                for row in T.serial(4):
                    for i, j in T.Tiles((8, 32)):
                        out[row * 8 + i, j] = T.Cast(T.bfloat16, scale[row, j])
            else:
                for row in T.serial(count):
                    for i, j in T.Tiles((32 // count, 32)):
                        out[row * (32 // count) + i, j] = T.Cast(T.bfloat16, peer[row, j])
            T.copy(out, Output)

    return main


@target("sunmmio")
def mx_view_gemm_numeric_kernel(bits):
    mx_dtype = T.mxfp8 if bits == 8 else T.mxfp4
    payload_dtype = T.float8_e4m3fn if bits == 8 else T.float4_e2m1fn

    @T.prim_func
    def main(
        Input: T.MeshTensor((64, 64), T.placement.replicated(), T.bfloat16, layout=make_zz_layout((64, 64))),
        Output: T.MeshTensor((64, 64), T.placement.replicated(), T.bfloat16, layout=make_zz_layout((64, 64))),
    ):
        with T.Kernel():
            ap = T.alloc_shared((64, 64), mx_dtype, scope="shared.rsram")
            bp = T.alloc_shared((64, 64), mx_dtype, scope="shared.rsram")
            a = T.alloc_shared((64, 64), mx_dtype, scope="shared.asram")
            b = T.alloc_shared((64, 64), mx_dtype, scope="shared.wsram")
            c = T.alloc_shared((64, 64), T.bfloat16, scope="shared.rsram")
            x = T.alloc_shared((64, 64), T.bfloat16, scope="shared.rsram")
            T.copy(Input, x)
            ad, asc = T.mx_view(ap)
            bd, bsc = T.mx_view(bp)
            for i, j in T.Tiles(ad.shape):
                ad[i, j] = T.Cast(payload_dtype, x[i, j])
            for i, j in T.Tiles(asc.shape):
                asc[i, j] = T.Cast(T.float8_e8m0fnu, T.bfloat16(1))
            for i, j in T.Tiles(bd.shape):
                bd[i, j] = T.Cast(payload_dtype, x[i, j])
            for i, j in T.Tiles(bsc.shape):
                bsc[i, j] = T.Cast(T.float8_e8m0fnu, T.bfloat16(1))
            T.copy(ap, a)
            T.copy(bp, b)
            T.gemm(a, b, c, clear_accum=True)
            T.copy(c, Output)

    return main


def mesh_target(mesh):
    return f"llvm -mcpu=sunmmio-a4e -mattr=device_mesh_nrow_{mesh[0]},device_mesh_ncol_{mesh[1]}"


@target("sunmmio")
def context_kernel(case, bits=8, reverse=False, layout_kind="mxzz", positive=False):
    mx_dtype = T.mxfp8 if bits == 8 else T.mxfp4
    shape = (16, 1024) if layout_kind == "row_major" else (256, 64)
    layout = {"mxzz": make_mxzz_layout, "mxznz": make_mxznz_layout, "row_major": make_mx_row_major_layout}[layout_kind](
        shape, dtype=mx_dtype
    )
    tensor_layout = make_zz_layout((32, 32))

    def transfer(scale, peer, index):
        if reverse:
            return T.copy(peer, scale[index, 0])
        return T.copy(scale[index, 0], peer)

    @T.prim_func
    def main(
        Input: T.MeshTensor((32, 32), T.placement.replicated(), T.bfloat16, layout=tensor_layout),
        Output: T.MeshTensor((32, 32), T.placement.replicated(), T.bfloat16, layout=tensor_layout),
    ):
        with T.Kernel() as cid:
            mx = T.alloc_shared(shape, mx_dtype, scope="shared.rsram")
            T.annotate_layout({mx: layout})
            data, scale = T.mx_view(mx)
            peer = T.alloc_shared((1, 32), T.float8_e8m0fnu, scope="shared.rsram")
            seed = T.alloc_shared((32, 32), T.bfloat16, scope="shared.rsram")
            out = T.alloc_shared((32, 32), T.bfloat16, scope="shared.rsram")
            T.copy(Input, seed)
            for i, j in T.Tiles(scale.shape):
                scale[i, j] = T.Cast(T.float8_e8m0fnu, seed[i, j])
            for i, j in T.Tiles(peer.shape):
                peer[i, j] = T.Cast(T.float8_e8m0fnu, seed[16 + i, j])
            if case == "thread":
                transfer(scale, peer, cid if positive else cid - 16)
            elif case == "late_shape":
                transfer(scale, peer, cid if positive else cid - scale.shape[0])
            elif case == "slice":
                row = cid % 15 - 16
                if reverse:
                    if positive:
                        T.copy(peer, scale[row + 16 : row + 17, :])
                    else:
                        T.copy(peer, scale[row : row + 1, :])
                else:
                    if positive:
                        T.copy(scale[row + 16 : row + 17, :], peer)
                    else:
                        T.copy(scale[row : row + 1, :], peer)
            elif case == "loop":
                for i in T.serial(2):
                    row = cid % 8 - 4 - i
                    if row < 0:
                        transfer(scale, peer, row + 16 if positive else row)
                    else:
                        transfer(scale, peer, 3)
            else:
                row = cid % 8 - 4
                if case == "then":
                    if row < 0:
                        transfer(scale, peer, row + 16 if positive else row)
                    else:
                        transfer(scale, peer, 3)
                elif case == "else":
                    if row >= 0:
                        transfer(scale, peer, 3)
                    else:
                        transfer(scale, peer, row + 16 if positive else row)
                else:
                    if cid % 2 == 0:
                        if T.Not(row >= 0):
                            transfer(scale, peer, row + 16 if positive else row)
                        else:
                            transfer(scale, peer, 3)
                    else:
                        transfer(scale, peer, 3)
            if reverse:
                for row in T.serial(16):
                    for i, j in T.Tiles((2, 32)):
                        out[row * 2 + i, j] = T.Cast(T.bfloat16, scale[row, j])
            else:
                for i, j in T.Tiles(out.shape):
                    out[i, j] = T.Cast(T.bfloat16, peer[0, j])
            T.copy(out, Output)

    return main
