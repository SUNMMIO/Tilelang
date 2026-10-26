"""Bit-exact MX row-major field copies through the production compiler and gem5."""

import numpy as np
import pytest

import tilelang
import tilelang.language as T
from tilelang.layout import make_aligned_row_major, make_mx_row_major_layout, make_row_major
from tilelang.jit.adapter.sunmmio.libgen import SunmmioToolchain
from testing.python.sunmmio.common.compile_pipeline import target


tilelang.env.disable_cache()
ROW_MAJOR_WIDTHS = (32, 64, 96, 128, 160, 992, 1024, 1056, 2016, 2048, 2080, 2112, 2144, 4096, 4128)


@target("sunmmio")
def mx_row_major_numeric_kernel(k=64, bits=8, operation="pack"):
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
            if operation == "pack":
                T.mx_pack(data, scale, mx)
            elif operation == "unpack":
                T.mx_unpack(mx, data, scale)
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


@pytest.fixture(scope="module")
def mx_sunsim():
    sunsim = pytest.importorskip("sunsim")
    from sunsim._paths import resolve_gem5_paths

    try:
        resolve_gem5_paths().validate()
        SunmmioToolchain.resolve()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    return sunsim


def _raw_pattern(rows, cols, seed):
    # Includes every byte pattern (also NaNs, signed zero and all FP4 codes),
    # with distinct rows so a wrong physical pitch cannot pass unnoticed.
    row = np.arange(rows, dtype=np.uint32)[:, None]
    col = np.arange(cols, dtype=np.uint32)[None, :]
    return ((row * 37 + col * 13 + seed) % 256).astype(np.uint8)


@pytest.mark.parametrize("bits", [pytest.param(8, id="mxfp8"), pytest.param(4, id="mxfp4")])
@pytest.mark.parametrize("k", ROW_MAJOR_WIDTHS)
@pytest.mark.parametrize("operation", ["pack", "unpack", "roundtrip"])
def test_mx_row_major_copy_bits_and_padding(mx_sunsim, tmp_path, bits, k, operation):
    rows = 32
    payload_bytes = k * bits // 8
    data_bytes = (payload_bytes + 63) // 64 * 64
    scale_count = k // 32
    scale_bytes = (scale_count + 63) // 64 * 64
    mx_bytes = data_bytes + scale_bytes
    arrays = [
        _raw_pattern(rows, data_bytes, 0x11),
        _raw_pattern(rows, scale_bytes, 0x47),
        _raw_pattern(rows, mx_bytes, 0xA3),
        _raw_pattern(rows, data_bytes, 0x5A),
        _raw_pattern(rows, scale_bytes, 0xD9),
    ]
    expected = [x.copy() for x in arrays]
    if operation in ("pack", "roundtrip"):
        expected[2][:, :payload_bytes] = arrays[0][:, :payload_bytes]
        expected[2][:, data_bytes : data_bytes + scale_count] = arrays[1][:, :scale_count]
    if operation == "unpack":
        expected[0][:, :payload_bytes] = arrays[2][:, :payload_bytes]
        expected[1][:, :scale_count] = arrays[2][:, data_bytes : data_bytes + scale_count]
    if operation == "roundtrip":
        expected[3][:, :payload_bytes] = arrays[0][:, :payload_bytes]
        expected[4][:, :scale_count] = arrays[1][:, :scale_count]

    kernel = tilelang.compile(mx_row_major_numeric_kernel(k, bits, operation), target="sunmmio", execution_backend="sunmmio_sunsim")
    # Raw-byte markers avoid host quantization/reordering. Each kernel parameter
    # describes the same storage with its actual data/scale/MX element type.
    markers = [mx_sunsim.Inout(array.copy(), placement=[mx_sunsim.R(), mx_sunsim.R()]) for array in arrays]
    result = kernel(*markers, mesh=(4, 4), timeout=180, working_dir=tmp_path)
    assert result.exit_code == 0
    for name, marker, reference in zip(("data", "scale", "mx", "data_after", "scale_after"), markers, expected):
        np.testing.assert_array_equal(marker.data, reference, err_msg=f"{operation}: {name}, bits={bits}, K={k}")
