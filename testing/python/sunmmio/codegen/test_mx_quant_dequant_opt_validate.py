import os

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.language.mesh_tensor import MeshReplicationType
from tilelang.tileview import make_tileview
from tilelang.layout import (
    get_mx_scale_shape,
    # make_aligned_row_major,
    make_mxznz_layout,
    make_mxzz_layout,
    make_zz_layout,
)

from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import validate_sunmmio_codegen_with_npuir_opt


tilelang.env.disable_cache()

# Debug logs from this file:
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
os.environ["SUNMMIO_TEST_LOG_IR"] = "1"

MX_COPY_ALIGNED_SHAPE = (128, 32)
MX_GENERIC_SHAPE = (64, 64)


def _int_shape(shape):
    return tuple(int(x) for x in shape)


def _assert_no_e8m0_tile_select(src):
    for line in src.splitlines():
        assert not ("suvm.tile.select" in line and "f8E8M0FNU" in line)


def _assert_e8m0_scale_casts(src):
    for line in src.splitlines():
        if "suvm.tile.cast" in line and "f8E8M0FNU" in line:
            assert "xf32" not in line
    for token in ("suvm.tile.ln", "suvm.tile.ceil", "suvm.tile.exp"):
        assert token not in src


# A4E converts E8M0 through BF16; the stored E8M0 value is the scale source of truth.
def _to_e8m0_scale(x):
    return T.Cast("float8_e8m0fnu", x)


def _e8m0_scale_to_fp32(x):
    return T.Cast("float32", T.Cast("bfloat16", x))


MX_FULL_CHAIN_CASES = (
    pytest.param(T.mxfp8, T.float8_e4m3fn, "float8_e4m3fn", "!suvm.mxfp8", id="mxfp8"),
    pytest.param(T.mxfp4, T.float4_e2m1fn, "float4_e2m1fn", "!suvm.mxfp4", id="mxfp4"),
)

MX_OCP_QUANT_CASES = (
    pytest.param(
        T.mxfp8,
        T.float8_e4m3fn,
        "float8_e4m3fn",
        448.0,
        "!suvm.mxfp8",
        id="mxfp8",
    ),
    pytest.param(
        T.mxfp4,
        T.float4_e2m1fn,
        "float4_e2m1fn",
        6.0,
        "!suvm.mxfp4",
        id="mxfp4",
    ),
)


@target("Sunmmio")
def mx_ocp_quant_dequant_full_chain_kernel_for_debug(mx_dtype, data_dtype, data_dtype_name, data_max):
    shape = (32, 32)
    shard_policy = T.MeshShardingPolicy(replicate=MeshReplicationType.ALL)
    tensor_layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))
    mx_layout = make_mxzz_layout(shape, dtype=mx_dtype)
    scale_shape = _int_shape(get_mx_scale_shape(mx_layout, mx_dtype))
    data_max_inv = 1.0 / data_max

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, shard_policy, T.bfloat16, layout=tensor_layout),  # type: ignore
        Y: T.MeshTensor(shape, shard_policy, T.bfloat16, layout=tensor_layout),  # type: ignore
    ):
        with T.Kernel():
            a_rsram = T.alloc_shared(shape, T.bfloat16)
            amax = T.alloc_shared((64,), T.float32)
            data = T.alloc_shared(shape, data_dtype)
            scale = T.alloc_shared(scale_shape, T.float8_e8m0fnu)
            scale_bf16_vec = T.alloc_shared((64,), T.bfloat16)
            scale_fp32_vec = T.alloc_shared((64,), T.float32)
            mx = T.alloc_shared(shape, mx_dtype)
            unpacked_data = T.alloc_shared(shape, data_dtype)
            unpacked_scale = T.alloc_shared(scale_shape, T.float8_e8m0fnu)
            unpacked_scale_fp32_vec = T.alloc_shared((64,), T.float32)
            y_rsram = T.alloc_shared(shape, T.bfloat16)

            T.copy(A, a_rsram)

            # OCP MX quantization: each logical row is one 32-element scale
            # group in this 32x32 test block.
            T.fill(amax, T.float32(1e-4))
            T.reduce_absmax(a_rsram, amax[0:32], dim=1, clear=True)
            for j in T.Tiles([scale_shape[1]]):
                safe_amax = T.max(amax[j], T.float32(1e-4))
                scale_bf16_vec[j] = T.Cast("bfloat16", safe_amax * T.float32(data_max_inv))
            for j in T.Tiles([scale_shape[1]]):
                scale[0, j] = _to_e8m0_scale(scale_bf16_vec[j])
            for j in T.Tiles([scale_shape[1]]):
                scale_fp32_vec[j] = _e8m0_scale_to_fp32(scale[0, j])

            T.annotate_tileview(
                {
                    a_rsram: make_tileview(a_rsram, (8, 32), (-2, -1)),
                    data: make_tileview(data, (8, 32), (-2, -1)),
                }
            )
            for row, col in T.Tiles(data):
                raw = T.Cast("float32", a_rsram[row, col]) / scale_fp32_vec[row]
                clamped = T.min(T.max(raw, T.float32(-data_max)), T.float32(data_max))
                data[row, col] = T.Cast(data_dtype_name, clamped)

            T.mx_pack(data, scale, mx)
            T.mx_unpack(mx, unpacked_data, unpacked_scale)

            for j in T.Tiles([scale_shape[1]]):
                unpacked_scale_fp32_vec[j] = _e8m0_scale_to_fp32(unpacked_scale[0, j])

            T.annotate_tileview(
                {
                    unpacked_data: make_tileview(unpacked_data, (8, 32), (-2, -1)),
                    y_rsram: make_tileview(y_rsram, (8, 32), (-2, -1)),
                }
            )
            for row, col in T.Tiles(y_rsram):
                q = T.Cast("float32", unpacked_data[row, col])
                y_rsram[row, col] = T.Cast("bfloat16", q * unpacked_scale_fp32_vec[row])

            T.copy(y_rsram, Y)

    return main


@target("Sunmmio")
def mx_ocp_quant_dequant_full_chain_kernel_original(mx_dtype, data_dtype, data_dtype_name, data_max):
    shape = (32, 32)
    shard_policy = T.MeshShardingPolicy(replicate=MeshReplicationType.ALL)
    tensor_layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))
    mx_layout = make_mxzz_layout(shape, dtype=mx_dtype)
    scale_shape = _int_shape(get_mx_scale_shape(mx_layout, mx_dtype))
    data_max_inv = 1.0 / data_max

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, shard_policy, T.bfloat16, layout=tensor_layout),  # type: ignore
        Y: T.MeshTensor(shape, shard_policy, T.bfloat16, layout=tensor_layout),  # type: ignore
    ):
        with T.Kernel():
            a_rsram = T.alloc_shared(shape, T.bfloat16)
            amax = T.alloc_shared((64,), T.float32)
            data = T.alloc_shared(shape, data_dtype)
            scale = T.alloc_shared(scale_shape, T.float8_e8m0fnu)
            scale_bf16_vec = T.alloc_shared((64,), T.bfloat16)
            scale_fp32_vec = T.alloc_shared((64,), T.float32)
            mx = T.alloc_shared(shape, mx_dtype)
            unpacked_data = T.alloc_shared(shape, data_dtype)
            unpacked_scale = T.alloc_shared(scale_shape, T.float8_e8m0fnu)
            unpacked_scale_fp32_vec = T.alloc_shared((64,), T.float32)
            y_rsram = T.alloc_shared(shape, T.bfloat16)

            T.copy(A, a_rsram)

            # OCP MX quantization: each logical row is one 32-element scale
            # group in this 32x32 test block.
            T.fill(amax, T.float32(1e-4))
            T.reduce_absmax(a_rsram, amax[0:32], dim=1, clear=True)
            for j in T.Tiles([scale_shape[1]]):
                safe_amax = T.max(amax[j], T.float32(1e-4))
                scale_bf16_vec[j] = T.Cast("bfloat16", safe_amax * T.float32(data_max_inv))

            for j in T.Tiles([scale_shape[1]]):
                scale[0, j] = _to_e8m0_scale(scale_bf16_vec[j])

            for j in T.Tiles([scale_shape[1]]):
                scale_fp32_vec[j] = _e8m0_scale_to_fp32(scale[0, j])

            for row, col in T.Tiles(data):
                raw = T.Cast("float32", a_rsram[row, col]) / scale_fp32_vec[row]
                clamped = T.min(T.max(raw, T.float32(-data_max)), T.float32(data_max))
                data[row, col] = T.Cast(data_dtype_name, clamped)

            T.mx_pack(data, scale, mx)
            T.mx_unpack(mx, unpacked_data, unpacked_scale)

            for j in T.Tiles([scale_shape[1]]):
                unpacked_scale_fp32_vec[j] = _e8m0_scale_to_fp32(unpacked_scale[0, j])

            for row, col in T.Tiles(y_rsram):
                q = T.Cast("float32", unpacked_data[row, col])
                y_rsram[row, col] = T.Cast("bfloat16", q * unpacked_scale_fp32_vec[row])

            T.copy(y_rsram, Y)

    return main


@target("Sunmmio")
def mx_ocp_quant_kernel_for_debug(mx_dtype, data_dtype, data_dtype_name, data_max):
    shape = MX_COPY_ALIGNED_SHAPE
    shard_policy = T.MeshShardingPolicy(replicate=MeshReplicationType.ALL)
    tensor_layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))
    mx_layout = make_mxzz_layout(shape, dtype=mx_dtype)
    scale_shape = _int_shape(get_mx_scale_shape(mx_layout, mx_dtype))
    data_max_inv = 1.0 / data_max

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, shard_policy, T.bfloat16, layout=tensor_layout),  # type: ignore
        MX: T.MeshTensor(shape, shard_policy, mx_dtype, layout=mx_layout),  # type: ignore
    ):
        with T.Kernel():
            a_rsram = T.alloc_shared(shape, T.bfloat16)
            amax = T.alloc_shared((shape[0],), T.float32)
            data = T.alloc_shared(shape, data_dtype)
            scale = T.alloc_shared(scale_shape, T.float8_e8m0fnu)
            scale_bf16_vec = T.alloc_shared((scale_shape[0] * 64,), T.bfloat16)
            scale_fp32_vec = T.alloc_shared((scale_shape[0] * 64,), T.float32)
            mx = T.alloc_shared(shape, mx_dtype)

            T.copy(A, a_rsram)

            T.fill(amax, T.float32(1e-4))
            T.reduce_absmax(a_rsram, amax, dim=1, clear=True)
            for block in T.serial(scale_shape[0]):
                amax_base = block * scale_shape[1]
                scale_base = block * 64
                for j in T.Tiles([scale_shape[1]]):
                    safe_amax = T.max(amax[amax_base + j], T.float32(1e-4))
                    scale_bf16_vec[scale_base + j] = T.Cast("bfloat16", safe_amax * T.float32(data_max_inv))

            for block in T.serial(scale_shape[0]):
                scale_base = block * 64
                for j in T.Tiles([scale_shape[1]]):
                    scale[block, j] = _to_e8m0_scale(scale_bf16_vec[scale_base + j])

            for block in T.serial(scale_shape[0]):
                scale_base = block * 64
                for j in T.Tiles([scale_shape[1]]):
                    scale_fp32_vec[scale_base + j] = _e8m0_scale_to_fp32(scale[block, j])

            T.annotate_tileview(
                {
                    a_rsram: make_tileview(a_rsram, (8, 32), (-2, -1)),
                    data: make_tileview(data, (8, 32), (-2, -1)),
                }
            )
            for block in T.serial(scale_shape[0]):
                row_base = block * scale_shape[1]
                scale_base = block * 64
                for row, col in T.Tiles([32, 32]):
                    raw = T.Cast("float32", a_rsram[row_base + row, col]) / scale_fp32_vec[scale_base + row]
                    clamped = T.min(T.max(raw, T.float32(-data_max)), T.float32(data_max))
                    data[row_base + row, col] = T.Cast(data_dtype_name, clamped)

            T.mx_pack(data, scale, mx)
            T.copy(mx, MX)

    return main


@target("Sunmmio")
def mx_ocp_dequant_kernel_for_debug(mx_dtype, data_dtype, data_dtype_name, data_max):
    shape = MX_COPY_ALIGNED_SHAPE
    shard_policy = T.MeshShardingPolicy(replicate=MeshReplicationType.ALL)
    tensor_layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))
    mx_layout = make_mxzz_layout(shape, dtype=mx_dtype)
    scale_shape = _int_shape(get_mx_scale_shape(mx_layout, mx_dtype))

    @T.prim_func
    def main(
        MX: T.MeshTensor(shape, shard_policy, mx_dtype, layout=mx_layout),  # type: ignore
        Y: T.MeshTensor(shape, shard_policy, T.bfloat16, layout=tensor_layout),  # type: ignore
    ):
        with T.Kernel():
            mx = T.alloc_shared(shape, mx_dtype)
            data = T.alloc_shared(shape, data_dtype)
            scale = T.alloc_shared(scale_shape, T.float8_e8m0fnu)
            scale_fp32_vec = T.alloc_shared((scale_shape[0] * 64,), T.float32)
            y_rsram = T.alloc_shared(shape, T.bfloat16)

            T.copy(MX, mx)
            T.mx_unpack(mx, data, scale)

            for block in T.serial(scale_shape[0]):
                scale_base = block * 64
                for j in T.Tiles([scale_shape[1]]):
                    scale_fp32_vec[scale_base + j] = _e8m0_scale_to_fp32(scale[block, j])

            T.annotate_tileview(
                {
                    data: make_tileview(data, (8, 32), (-2, -1)),
                    y_rsram: make_tileview(y_rsram, (8, 32), (-2, -1)),
                }
            )
            for block in T.serial(scale_shape[0]):
                row_base = block * scale_shape[1]
                scale_base = block * 64
                for row, col in T.Tiles([32, 32]):
                    q = T.Cast("float32", data[row_base + row, col])
                    y_rsram[row_base + row, col] = T.Cast("bfloat16", q * scale_fp32_vec[scale_base + row])

            T.copy(y_rsram, Y)

    return main


@target("Sunmmio")
def mx_ocp_quant_generic_shape_kernel_for_debug(mx_dtype, data_dtype, data_dtype_name, data_max):
    shape = MX_GENERIC_SHAPE
    num_m_blocks = shape[0] // 32
    num_n_blocks = shape[1] // 32
    shard_policy = T.MeshShardingPolicy(replicate=MeshReplicationType.ALL)
    tensor_layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))
    mx_layout = make_mxzz_layout(shape, dtype=mx_dtype)
    scale_shape = _int_shape(get_mx_scale_shape(mx_layout, mx_dtype))
    data_max_inv = 1.0 / data_max
    assert shape[0] % 32 == 0
    assert shape[1] % 32 == 0
    assert scale_shape == (num_m_blocks * num_n_blocks, 32)

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, shard_policy, T.bfloat16, layout=tensor_layout),  # type: ignore
        MX: T.MeshTensor(shape, shard_policy, mx_dtype, layout=mx_layout),  # type: ignore
    ):
        with T.Kernel():
            a_rsram = T.alloc_shared(shape, T.bfloat16)
            tile_rsram = T.alloc_shared((32, 32), T.bfloat16)
            amax = T.alloc_shared((64,), T.float32)
            data = T.alloc_shared(shape, data_dtype)
            scale = T.alloc_shared(scale_shape, T.float8_e8m0fnu)
            scale_bf16_vec = T.alloc_shared((scale_shape[0] * 64,), T.bfloat16)
            scale_fp32_vec = T.alloc_shared((scale_shape[0] * 64,), T.float32)
            mx = T.alloc_shared(shape, mx_dtype)

            T.copy(A, a_rsram)

            T.annotate_tileview(
                {
                    a_rsram: make_tileview(a_rsram, (8, 32), (-2, -1)),
                    tile_rsram: make_tileview(tile_rsram, (8, 32), (-2, -1)),
                    data: make_tileview(data, (8, 32), (-2, -1)),
                }
            )
            for block_m in T.serial(num_m_blocks):
                for block_n in T.serial(num_n_blocks):
                    block = block_m * num_n_blocks + block_n
                    row_base = block_m * 32
                    col_base = block_n * 32
                    scale_base = block * 64

                    for row, col in T.Tiles([32, 32]):
                        tile_rsram[row, col] = a_rsram[row_base + row, col_base + col]

                    T.fill(amax, T.float32(1e-4))
                    T.reduce_absmax(tile_rsram, amax[0:32], dim=1, clear=True)
                    for row in T.Tiles([scale_shape[1]]):
                        safe_amax = T.max(amax[row], T.float32(1e-4))
                        scale_bf16_vec[scale_base + row] = T.Cast("bfloat16", safe_amax * T.float32(data_max_inv))

                    for row in T.Tiles([scale_shape[1]]):
                        scale[block, row] = _to_e8m0_scale(scale_bf16_vec[scale_base + row])

                    for row in T.Tiles([scale_shape[1]]):
                        scale_fp32_vec[scale_base + row] = _e8m0_scale_to_fp32(scale[block, row])

                    for row, col in T.Tiles([32, 32]):
                        raw = T.Cast("float32", tile_rsram[row, col]) / scale_fp32_vec[scale_base + row]
                        clamped = T.min(T.max(raw, T.float32(-data_max)), T.float32(data_max))
                        data[row_base + row, col_base + col] = T.Cast(data_dtype_name, clamped)

            T.mx_pack(data, scale, mx)
            T.copy(mx, MX)

    return main


@target("Sunmmio")
def mx_ocp_dequant_generic_shape_kernel_for_debug(mx_dtype, data_dtype, data_dtype_name, data_max):
    shape = MX_GENERIC_SHAPE
    num_m_blocks = shape[0] // 32
    num_n_blocks = shape[1] // 32
    shard_policy = T.MeshShardingPolicy(replicate=MeshReplicationType.ALL)
    tensor_layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))
    mx_layout = make_mxzz_layout(shape, dtype=mx_dtype)
    scale_shape = _int_shape(get_mx_scale_shape(mx_layout, mx_dtype))
    assert shape[0] % 32 == 0
    assert shape[1] % 32 == 0
    assert scale_shape == (num_m_blocks * num_n_blocks, 32)

    @T.prim_func
    def main(
        MX: T.MeshTensor(shape, shard_policy, mx_dtype, layout=mx_layout),  # type: ignore
        Y: T.MeshTensor(shape, shard_policy, T.bfloat16, layout=tensor_layout),  # type: ignore
    ):
        with T.Kernel():
            mx = T.alloc_shared(shape, mx_dtype)
            data = T.alloc_shared(shape, data_dtype)
            scale = T.alloc_shared(scale_shape, T.float8_e8m0fnu)
            scale_fp32_vec = T.alloc_shared((scale_shape[0] * 64,), T.float32)
            y_rsram = T.alloc_shared(shape, T.bfloat16)

            T.copy(MX, mx)
            T.mx_unpack(mx, data, scale)

            T.annotate_tileview(
                {
                    data: make_tileview(data, (8, 32), (-2, -1)),
                    y_rsram: make_tileview(y_rsram, (8, 32), (-2, -1)),
                }
            )
            for block_m in T.serial(num_m_blocks):
                for block_n in T.serial(num_n_blocks):
                    block = block_m * num_n_blocks + block_n
                    row_base = block_m * 32
                    col_base = block_n * 32
                    scale_base = block * 64

                    for row in T.Tiles([scale_shape[1]]):
                        scale_fp32_vec[scale_base + row] = _e8m0_scale_to_fp32(scale[block, row])

                    for row, col in T.Tiles([32, 32]):
                        q = T.Cast("float32", data[row_base + row, col_base + col])
                        y_rsram[row_base + row, col_base + col] = T.Cast("bfloat16", q * scale_fp32_vec[scale_base + row])

            T.copy(y_rsram, Y)

    return main


@target("Sunmmio")
def mx_ocp_quantized_mma_kernel_for_debug(mx_dtype, data_dtype, data_dtype_name, data_max):
    a_shape = (32, 64)
    b_shape = (32, 64)
    c_shape = (32, 32)
    a_k_blocks = a_shape[1] // 32
    b_k_blocks = b_shape[1] // 32
    shard_policy = T.MeshShardingPolicy(replicate=MeshReplicationType.ALL)
    a_tensor_layout = make_zz_layout(a_shape, axes=[0, 1], block_shape=(32, 32))
    b_tensor_layout = make_zz_layout(b_shape, axes=[0, 1], block_shape=(32, 32))
    c_tensor_layout = make_zz_layout(c_shape, axes=[0, 1], block_shape=(32, 32))
    a_mx_layout = make_mxzz_layout(a_shape, dtype=mx_dtype)
    b_mx_layout = make_mxzz_layout(b_shape, dtype=mx_dtype)
    a_scale_shape = _int_shape(get_mx_scale_shape(a_mx_layout, mx_dtype))
    b_scale_shape = _int_shape(get_mx_scale_shape(b_mx_layout, mx_dtype))
    data_max_inv = 1.0 / data_max
    assert a_shape[0] % 32 == 0
    assert a_shape[1] % 32 == 0
    assert b_shape[0] % 32 == 0
    assert b_shape[1] % 32 == 0
    assert a_scale_shape == (a_k_blocks, 32)
    assert b_scale_shape == (b_k_blocks, 32)

    @T.prim_func
    def main(
        A: T.MeshTensor(a_shape, shard_policy, T.bfloat16, layout=a_tensor_layout),  # type: ignore
        B: T.MeshTensor(b_shape, shard_policy, T.bfloat16, layout=b_tensor_layout),  # type: ignore
        C: T.MeshTensor(c_shape, shard_policy, T.bfloat16, layout=c_tensor_layout),  # type: ignore
    ):
        with T.Kernel():
            a_rsram = T.alloc_shared(a_shape, T.bfloat16)
            b_rsram = T.alloc_shared(b_shape, T.bfloat16)
            a_tile = T.alloc_shared((32, 32), T.bfloat16)
            b_tile = T.alloc_shared((32, 32), T.bfloat16)
            amax = T.alloc_shared((64,), T.float32)
            a_data = T.alloc_shared(a_shape, data_dtype)
            b_data = T.alloc_shared(b_shape, data_dtype)
            a_scale = T.alloc_shared(a_scale_shape, T.float8_e8m0fnu)
            b_scale = T.alloc_shared(b_scale_shape, T.float8_e8m0fnu)
            a_scale_bf16_vec = T.alloc_shared((a_scale_shape[0] * 64,), T.bfloat16)
            b_scale_bf16_vec = T.alloc_shared((b_scale_shape[0] * 64,), T.bfloat16)
            a_scale_fp32_vec = T.alloc_shared((a_scale_shape[0] * 64,), T.float32)
            b_scale_fp32_vec = T.alloc_shared((b_scale_shape[0] * 64,), T.float32)
            a_mx = T.alloc_shared(a_shape, mx_dtype)
            b_mx = T.alloc_shared(b_shape, mx_dtype)
            a_mx_asram = T.alloc_shared(a_shape, mx_dtype, scope="shared.asram")
            b_mx_wsram = T.alloc_shared(b_shape, mx_dtype, scope="shared.wsram")
            c_rsram = T.alloc_shared(c_shape, T.bfloat16)

            T.annotate_layout(
                {
                    a_data: a_tensor_layout,
                    a_mx: a_mx_layout,
                    a_mx_asram: a_mx_layout,
                    b_rsram: b_tensor_layout,
                    b_data: b_tensor_layout,
                    b_mx: b_mx_layout,
                    b_mx_wsram: b_mx_layout,
                }
            )

            T.copy(A, a_rsram)
            T.copy(B, b_rsram)

            T.annotate_tileview(
                {
                    a_rsram: make_tileview(a_rsram, (8, 32), (-2, -1)),
                    b_rsram: make_tileview(b_rsram, (8, 32), (-2, -1)),
                    a_tile: make_tileview(a_tile, (8, 32), (-2, -1)),
                    b_tile: make_tileview(b_tile, (8, 32), (-2, -1)),
                    a_data: make_tileview(a_data, (8, 32), (-2, -1)),
                    b_data: make_tileview(b_data, (8, 32), (-2, -1)),
                }
            )

            for block_k in T.serial(a_k_blocks):
                col_base = block_k * 32
                scale_base = block_k * 64

                for row, col in T.Tiles([32, 32]):
                    a_tile[row, col] = a_rsram[row, col_base + col]

                T.fill(amax, T.float32(1e-4))
                T.reduce_absmax(a_tile, amax[0:32], dim=1, clear=True)
                for row in T.Tiles([a_scale_shape[1]]):
                    safe_amax = T.max(amax[row], T.float32(1e-4))
                    a_scale_bf16_vec[scale_base + row] = T.Cast("bfloat16", safe_amax * T.float32(data_max_inv))

                for row in T.Tiles([a_scale_shape[1]]):
                    a_scale[block_k, row] = _to_e8m0_scale(a_scale_bf16_vec[scale_base + row])

                for row in T.Tiles([a_scale_shape[1]]):
                    a_scale_fp32_vec[scale_base + row] = _e8m0_scale_to_fp32(a_scale[block_k, row])

                for row, col in T.Tiles([32, 32]):
                    raw = T.Cast("float32", a_tile[row, col]) / a_scale_fp32_vec[scale_base + row]
                    clamped = T.min(T.max(raw, T.float32(-data_max)), T.float32(data_max))
                    a_data[row, col_base + col] = T.Cast(data_dtype_name, clamped)

            for block_k in T.serial(b_k_blocks):
                col_base = block_k * 32
                scale_base = block_k * 64

                for row, col in T.Tiles([32, 32]):
                    b_tile[row, col] = b_rsram[row, col_base + col]

                T.fill(amax, T.float32(1e-4))
                T.reduce_absmax(b_tile, amax[0:32], dim=1, clear=True)
                for row in T.Tiles([b_scale_shape[1]]):
                    safe_amax = T.max(amax[row], T.float32(1e-4))
                    b_scale_bf16_vec[scale_base + row] = T.Cast("bfloat16", safe_amax * T.float32(data_max_inv))

                for row in T.Tiles([b_scale_shape[1]]):
                    b_scale[block_k, row] = _to_e8m0_scale(b_scale_bf16_vec[scale_base + row])

                for row in T.Tiles([b_scale_shape[1]]):
                    b_scale_fp32_vec[scale_base + row] = _e8m0_scale_to_fp32(b_scale[block_k, row])

                for row, col in T.Tiles([32, 32]):
                    raw = T.Cast("float32", b_tile[row, col]) / b_scale_fp32_vec[scale_base + row]
                    clamped = T.min(T.max(raw, T.float32(-data_max)), T.float32(data_max))
                    b_data[row, col_base + col] = T.Cast(data_dtype_name, clamped)

            T.mx_pack(a_data, a_scale, a_mx)
            T.mx_pack(b_data, b_scale, b_mx)
            T.copy(a_mx, a_mx_asram)
            T.copy(b_mx, b_mx_wsram)
            T.clear(c_rsram)
            T.gemm(a_mx_asram, b_mx_wsram, c_rsram, transpose_B=True)
            T.copy(c_rsram, C)

    return main


@target("Sunmmio")
def mx_ocp_quantized_mma_mxznz_weight_kernel_for_debug(mx_dtype, data_dtype, data_dtype_name, data_max):
    a_shape = (32, 64)
    b_shape = (64, 32)
    c_shape = (32, 32)
    a_k_blocks = a_shape[1] // 32
    b_k_blocks = b_shape[0] // 32
    shard_policy = T.MeshShardingPolicy(replicate=MeshReplicationType.ALL)
    a_tensor_layout = make_zz_layout(a_shape, axes=[0, 1], block_shape=(32, 32))
    b_tensor_layout = make_zz_layout(b_shape, axes=[0, 1], block_shape=(32, 32))
    b_tile_layout = make_zz_layout((32, 32), axes=[0, 1], block_shape=(32, 32))
    b_data_layout = make_zz_layout(b_shape, axes=[0, 1], block_shape=(32, 32))
    c_tensor_layout = make_zz_layout(c_shape, axes=[0, 1], block_shape=(32, 32))
    a_tensor = T.MeshTensor(a_shape, shard_policy, T.bfloat16, layout=a_tensor_layout)
    b_tensor = T.MeshTensor(b_shape, shard_policy, T.bfloat16, layout=b_tensor_layout)
    c_tensor = T.MeshTensor(c_shape, shard_policy, T.bfloat16, layout=c_tensor_layout)
    b_sharded_layout = b_tensor.meta_data["sharded_layout"]
    a_mx_layout = make_mxzz_layout(a_shape, dtype=mx_dtype)
    b_mx_layout = make_mxznz_layout(b_shape, dtype=mx_dtype)
    a_scale_shape = _int_shape(get_mx_scale_shape(a_mx_layout, mx_dtype))
    b_scale_shape = _int_shape(get_mx_scale_shape(b_mx_layout, mx_dtype))
    data_max_inv = 1.0 / data_max
    assert a_scale_shape == (a_k_blocks, 32)
    assert b_scale_shape == (b_k_blocks, 32)

    @T.prim_func
    def main(
        A: a_tensor,  # type: ignore
        B: b_tensor,  # type: ignore
        C: c_tensor,  # type: ignore
    ):
        with T.Kernel():
            a_rsram = T.alloc_shared(a_shape, T.bfloat16)
            b_rsram = T.alloc_shared(b_shape, T.bfloat16)
            a_tile = T.alloc_shared((32, 32), T.bfloat16)
            b_tile = T.alloc_shared((32, 32), T.bfloat16)
            amax = T.alloc_shared((64,), T.float32)
            a_data = T.alloc_shared(a_shape, data_dtype)
            b_data = T.alloc_shared(b_shape, data_dtype)
            a_scale = T.alloc_shared(a_scale_shape, T.float8_e8m0fnu)
            b_scale = T.alloc_shared(b_scale_shape, T.float8_e8m0fnu)
            a_scale_bf16_vec = T.alloc_shared((a_scale_shape[0] * 64,), T.bfloat16)
            b_scale_bf16_vec = T.alloc_shared((b_scale_shape[0] * 64,), T.bfloat16)
            a_scale_fp32_vec = T.alloc_shared((a_scale_shape[0] * 64,), T.float32)
            b_scale_fp32_vec = T.alloc_shared((b_scale_shape[0] * 64,), T.float32)
            a_mx = T.alloc_shared(a_shape, mx_dtype)
            b_mx = T.alloc_shared(b_shape, mx_dtype)
            a_mx_asram = T.alloc_shared(a_shape, mx_dtype, scope="shared.asram")
            b_mx_wsram = T.alloc_shared(b_shape, mx_dtype, scope="shared.wsram")
            c_rsram = T.alloc_shared(c_shape, T.bfloat16)

            T.annotate_layout(
                {
                    a_data: a_tensor_layout,
                    a_mx: a_mx_layout,
                    a_mx_asram: a_mx_layout,
                    b_rsram: b_sharded_layout,
                    b_tile: b_tile_layout,
                    b_data: b_data_layout,
                    b_mx: b_mx_layout,
                }
            )

            T.copy(A, a_rsram)
            T.copy(B, b_rsram)

            T.annotate_tileview(
                {
                    a_rsram: make_tileview(a_rsram, (8, 32), (-2, -1)),
                    b_rsram: make_tileview(b_rsram, (8, 32), (-2, -1)),
                    a_tile: make_tileview(a_tile, (8, 32), (-2, -1)),
                    b_tile: make_tileview(b_tile, (8, 32), (-2, -1)),
                    a_data: make_tileview(a_data, (8, 32), (-2, -1)),
                    b_data: make_tileview(b_data, (8, 32), (-2, -1)),
                }
            )

            for block_k in T.serial(a_k_blocks):
                col_base = block_k * 32
                scale_base = block_k * 64

                for row, col in T.Tiles([32, 32]):
                    a_tile[row, col] = a_rsram[row, col_base + col]

                T.fill(amax, T.float32(1e-4))
                T.reduce_absmax(a_tile, amax[0:32], dim=1, clear=True)
                for row in T.Tiles([a_scale_shape[1]]):
                    safe_amax = T.max(amax[row], T.float32(1e-4))
                    a_scale_bf16_vec[scale_base + row] = T.Cast("bfloat16", safe_amax * T.float32(data_max_inv))

                for row in T.Tiles([a_scale_shape[1]]):
                    a_scale[block_k, row] = _to_e8m0_scale(a_scale_bf16_vec[scale_base + row])

                for row in T.Tiles([a_scale_shape[1]]):
                    a_scale_fp32_vec[scale_base + row] = _e8m0_scale_to_fp32(a_scale[block_k, row])

                for row, col in T.Tiles([32, 32]):
                    raw = T.Cast("float32", a_tile[row, col]) / a_scale_fp32_vec[scale_base + row]
                    clamped = T.min(T.max(raw, T.float32(-data_max)), T.float32(data_max))
                    a_data[row, col_base + col] = T.Cast(data_dtype_name, clamped)

            for block_k in T.serial(b_k_blocks):
                row_base = block_k * 32
                scale_base = block_k * 64

                for row, col in T.Tiles([32, 32]):
                    b_tile[row, col] = b_rsram[row_base + row, col]

                T.fill(amax, T.float32(1e-4))
                T.reduce_absmax(b_tile, amax[0:32], dim=0, clear=True)
                for col in T.Tiles([b_scale_shape[1]]):
                    safe_amax = T.max(amax[col], T.float32(1e-4))
                    b_scale_bf16_vec[scale_base + col] = T.Cast("bfloat16", safe_amax * T.float32(data_max_inv))

                for col in T.Tiles([b_scale_shape[1]]):
                    b_scale[block_k, col] = _to_e8m0_scale(b_scale_bf16_vec[scale_base + col])

                for col in T.Tiles([b_scale_shape[1]]):
                    b_scale_fp32_vec[scale_base + col] = _e8m0_scale_to_fp32(b_scale[block_k, col])

                for row, col in T.Tiles([32, 32]):
                    raw = T.Cast("float32", b_tile[row, col]) / b_scale_fp32_vec[scale_base + col]
                    clamped = T.min(T.max(raw, T.float32(-data_max)), T.float32(data_max))
                    b_data[row_base + row, col] = T.Cast(data_dtype_name, clamped)

            T.mx_pack(a_data, a_scale, a_mx)
            T.mx_pack(b_data, b_scale, b_mx)
            T.copy(a_mx, a_mx_asram)
            T.copy(b_mx, b_mx_wsram)
            T.clear(c_rsram)
            T.gemm(a_mx_asram, b_mx_wsram, c_rsram)
            T.copy(c_rsram, C)

    return main


@pytest.mark.parametrize("mx_dtype,data_dtype,data_dtype_name,data_max,mx_token", MX_OCP_QUANT_CASES)
def test_mx_ocp_quant_dequant_full_chain_codegen_logs_mlir(
    tmp_path,
    mx_dtype,
    data_dtype,
    data_dtype_name,
    data_max,
    mx_token,
):
    src = validate_sunmmio_codegen_with_npuir_opt(
        mx_ocp_quant_dequant_full_chain_kernel_for_debug(mx_dtype, data_dtype, data_dtype_name, data_max),
        tmp_path,
        mlir_filename=f"mx_ocp_quant_dequant_full_chain_{data_dtype_name}_suvm.mlir",
        expected_tokens=(
            mx_token,
            "suvm.unpack",
            "suvm.tile.load",
            "suvm.tile.store",
            "suvm.tile.cast",
        ),
    )
    _assert_no_e8m0_tile_select(src)
    _assert_e8m0_scale_casts(src)


@pytest.mark.parametrize("mx_dtype,data_dtype,data_dtype_name,data_max,mx_token", MX_OCP_QUANT_CASES)
def test_mx_ocp_quant_kernel_codegen_logs_mlir(
    tmp_path,
    mx_dtype,
    data_dtype,
    data_dtype_name,
    data_max,
    mx_token,
):
    src = validate_sunmmio_codegen_with_npuir_opt(
        mx_ocp_quant_kernel_for_debug(mx_dtype, data_dtype, data_dtype_name, data_max),
        tmp_path,
        mlir_filename=f"mx_ocp_quant_{data_dtype_name}_suvm.mlir",
        expected_tokens=(
            mx_token,
            "suvm.unpack",
            "suvm.tile.load",
            "suvm.tile.store",
            "suvm.tile.cast",
        ),
    )
    _assert_no_e8m0_tile_select(src)
    _assert_e8m0_scale_casts(src)


@pytest.mark.parametrize("mx_dtype,data_dtype,data_dtype_name,data_max,mx_token", MX_OCP_QUANT_CASES)
def test_mx_ocp_dequant_kernel_codegen_logs_mlir(
    tmp_path,
    mx_dtype,
    data_dtype,
    data_dtype_name,
    data_max,
    mx_token,
):
    src = validate_sunmmio_codegen_with_npuir_opt(
        mx_ocp_dequant_kernel_for_debug(mx_dtype, data_dtype, data_dtype_name, data_max),
        tmp_path,
        mlir_filename=f"mx_ocp_dequant_{data_dtype_name}_suvm.mlir",
        expected_tokens=(
            mx_token,
            "suvm.unpack",
            "suvm.tile.load",
            "suvm.tile.store",
            "suvm.tile.cast",
        ),
    )
    _assert_e8m0_scale_casts(src)


@pytest.mark.parametrize("mx_dtype,data_dtype,data_dtype_name,data_max,mx_token", MX_OCP_QUANT_CASES)
def test_mx_ocp_quant_generic_shape_kernel_codegen_logs_mlir(
    tmp_path,
    mx_dtype,
    data_dtype,
    data_dtype_name,
    data_max,
    mx_token,
):
    src = validate_sunmmio_codegen_with_npuir_opt(
        mx_ocp_quant_generic_shape_kernel_for_debug(mx_dtype, data_dtype, data_dtype_name, data_max),
        tmp_path,
        mlir_filename=f"mx_ocp_quant_generic_{data_dtype_name}_suvm.mlir",
        expected_tokens=(
            mx_token,
            "suvm.unpack",
            "suvm.tile.load",
            "suvm.tile.store",
            "suvm.tile.cast",
        ),
    )
    _assert_no_e8m0_tile_select(src)
    _assert_e8m0_scale_casts(src)


@pytest.mark.parametrize("mx_dtype,data_dtype,data_dtype_name,data_max,mx_token", MX_OCP_QUANT_CASES)
def test_mx_ocp_dequant_generic_shape_kernel_codegen_logs_mlir(
    tmp_path,
    mx_dtype,
    data_dtype,
    data_dtype_name,
    data_max,
    mx_token,
):
    src = validate_sunmmio_codegen_with_npuir_opt(
        mx_ocp_dequant_generic_shape_kernel_for_debug(mx_dtype, data_dtype, data_dtype_name, data_max),
        tmp_path,
        mlir_filename=f"mx_ocp_dequant_generic_{data_dtype_name}_suvm.mlir",
        expected_tokens=(
            mx_token,
            "suvm.unpack",
            "suvm.tile.load",
            "suvm.tile.store",
            "suvm.tile.cast",
        ),
    )
    _assert_e8m0_scale_casts(src)


@pytest.mark.parametrize("mx_dtype,data_dtype,data_dtype_name,data_max,mx_token", MX_OCP_QUANT_CASES)
def test_mx_ocp_quantized_mma_kernel_codegen_logs_mlir(
    tmp_path,
    mx_dtype,
    data_dtype,
    data_dtype_name,
    data_max,
    mx_token,
):
    src = validate_sunmmio_codegen_with_npuir_opt(
        mx_ocp_quantized_mma_kernel_for_debug(mx_dtype, data_dtype, data_dtype_name, data_max),
        tmp_path,
        mlir_filename=f"mx_ocp_quantized_mma_{data_dtype_name}_suvm.mlir",
        expected_tokens=(
            mx_token,
            "suvm.unpack",
            "suvm.copy_async",
            "suvm.tc.mma",
            "suvm.tile.load",
            "suvm.tile.store",
            "suvm.tile.cast",
        ),
    )
    _assert_no_e8m0_tile_select(src)
    _assert_e8m0_scale_casts(src)


@pytest.mark.parametrize("mx_dtype,data_dtype,data_dtype_name,data_max,mx_token", MX_OCP_QUANT_CASES)
def test_mx_ocp_quantized_mma_mxznz_weight_kernel_codegen_logs_mlir_strict(
    tmp_path,
    mx_dtype,
    data_dtype,
    data_dtype_name,
    data_max,
    mx_token,
):
    src = validate_sunmmio_codegen_with_npuir_opt(
        mx_ocp_quantized_mma_mxznz_weight_kernel_for_debug(mx_dtype, data_dtype, data_dtype_name, data_max),
        tmp_path,
        mlir_filename=f"mx_ocp_quantized_mma_mxznz_weight_{data_dtype_name}_suvm.mlir",
        expected_tokens=(
            mx_token,
            "suvm.unpack",
            "suvm.copy_async",
            "suvm.tc.mma",
            "suvm.tile.load",
            "suvm.tile.store",
            "suvm.tile.cast",
        ),
    )
    _assert_no_e8m0_tile_select(src)
    _assert_e8m0_scale_casts(src)


if __name__ == "__main__":
    tilelang.testing.main()
