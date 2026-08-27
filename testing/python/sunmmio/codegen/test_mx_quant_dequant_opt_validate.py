import os

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.language.mesh_tensor import MeshReplicationType
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
            a = T.alloc_shared(shape, T.bfloat16)
            amax = T.alloc_shared((shape[0],), T.float32)
            data = T.alloc_shared(shape, data_dtype)
            scale = T.alloc_shared(scale_shape, T.float8_e8m0fnu)
            mx = T.alloc_shared(shape, mx_dtype)
            unpacked_data = T.alloc_shared(shape, data_dtype)
            unpacked_scale = T.alloc_shared(scale_shape, T.float8_e8m0fnu)
            y = T.alloc_shared(shape, T.bfloat16)

            T.copy(A, a)

            # OCP MX quantization: each logical row is one 32-element scale
            # group in this 32x32 test block.
            T.reduce_absmax(a, amax, dim=1, clear=True)
            for row in T.Tiles(amax):
                safe_amax = T.max(amax[row], T.float32(1e-4))
                bf16_scale = T.Cast("bfloat16", safe_amax * T.float32(data_max_inv))
                scale[0, row] = _to_e8m0_scale(bf16_scale)

            for row, col in T.Tiles(data):
                scale_fp32 = _e8m0_scale_to_fp32(scale[0, row])
                value = T.Cast("float32", a[row, col]) / scale_fp32
                data[row, col] = T.Cast(
                    data_dtype_name,
                    T.clamp(value, T.float32(-data_max), T.float32(data_max)),
                )

            T.mx_pack(data, scale, mx)
            T.mx_unpack(mx, unpacked_data, unpacked_scale)

            for row, col in T.Tiles(y):
                value = T.Cast("float32", unpacked_data[row, col])
                scale_fp32 = _e8m0_scale_to_fp32(unpacked_scale[0, row])
                y[row, col] = T.Cast("bfloat16", value * scale_fp32)

            T.copy(y, Y)

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
            mx = T.alloc_shared(shape, mx_dtype)

            T.copy(A, a_rsram)

            T.reduce_absmax(a_rsram, amax, dim=1, clear=True)
            for block in T.serial(scale_shape[0]):
                row_base = block * scale_shape[1]
                for row in T.Tiles([scale_shape[1]]):
                    safe_amax = T.max(amax[row_base + row], T.float32(1e-4))
                    bf16_scale = T.Cast("bfloat16", safe_amax * T.float32(data_max_inv))
                    scale[block, row] = _to_e8m0_scale(bf16_scale)

                for row, col in T.Tiles([scale_shape[1], shape[1]]):
                    scale_fp32 = _e8m0_scale_to_fp32(scale[block, row])
                    value = T.Cast("float32", a_rsram[row_base + row, col]) / scale_fp32
                    data[row_base + row, col] = T.Cast(
                        data_dtype_name,
                        T.clamp(value, T.float32(-data_max), T.float32(data_max)),
                    )

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
            y_rsram = T.alloc_shared(shape, T.bfloat16)

            T.copy(MX, mx)
            T.mx_unpack(mx, data, scale)

            for block in T.serial(scale_shape[0]):
                row_base = block * scale_shape[1]
                for row, col in T.Tiles([scale_shape[1], shape[1]]):
                    value = T.Cast("float32", data[row_base + row, col])
                    scale_fp32 = _e8m0_scale_to_fp32(scale[block, row])
                    y_rsram[row_base + row, col] = T.Cast("bfloat16", value * scale_fp32)

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
            amax = T.alloc_shared((32,), T.float32)
            data = T.alloc_shared(shape, data_dtype)
            scale = T.alloc_shared(scale_shape, T.float8_e8m0fnu)
            mx = T.alloc_shared(shape, mx_dtype)

            T.copy(A, a_rsram)

            for block_m in T.serial(num_m_blocks):
                for block_n in T.serial(num_n_blocks):
                    block = block_m * num_n_blocks + block_n
                    row_base = block_m * 32
                    col_base = block_n * 32

                    T.reduce_absmax(
                        a_rsram[row_base : row_base + 32, col_base : col_base + 32],
                        amax,
                        dim=1,
                        clear=True,
                    )
                    for row in T.Tiles([scale_shape[1]]):
                        safe_amax = T.max(amax[row], T.float32(1e-4))
                        bf16_scale = T.Cast("bfloat16", safe_amax * T.float32(data_max_inv))
                        scale[block, row] = _to_e8m0_scale(bf16_scale)

                    for row, col in T.Tiles([32, 32]):
                        scale_fp32 = _e8m0_scale_to_fp32(scale[block, row])
                        value = T.Cast("float32", a_rsram[row_base + row, col_base + col]) / scale_fp32
                        data[row_base + row, col_base + col] = T.Cast(
                            data_dtype_name,
                            T.clamp(value, T.float32(-data_max), T.float32(data_max)),
                        )

            T.mx_pack(data, scale, mx)
            T.copy(mx, MX)

    return main


@target("Sunmmio")
def mx_ocp_quant_sharded_kernel_for_debug(mx_dtype, data_dtype, data_dtype_name, data_max):
    global_shape = (256, 256)
    shard_policy = T.MeshShardingPolicy(y=0, x=1)
    tensor_layout = make_zz_layout(global_shape, axes=[0, 1], block_shape=(32, 32))
    mx_layout = make_mxzz_layout(global_shape, dtype=mx_dtype)
    data_max_inv = 1.0 / data_max
    a_tensor = T.MeshTensor(global_shape, shard_policy, T.bfloat16, layout=tensor_layout)
    mx_tensor = T.MeshTensor(global_shape, shard_policy, mx_dtype, layout=mx_layout)
    local_m, local_n = a_tensor.local_shape
    num_m_blocks = T.ceildiv(local_m, 32)
    num_n_blocks = T.ceildiv(local_n, 32)
    scale_shape = (num_m_blocks * num_n_blocks, 32)

    @T.prim_func
    def main(
        A: a_tensor,  # type: ignore
        MX: mx_tensor,  # type: ignore
    ):
        with T.Kernel():
            # local_m, local_n = A.local_shape
            # num_m_blocks = T.ceildiv(local_m, 32)
            # num_n_blocks = T.ceildiv(local_n, 32)
            # scale_shape = (num_m_blocks * num_n_blocks, 32)

            a_rsram = T.alloc_shared((local_m, local_n), T.bfloat16)
            amax = T.alloc_shared((32,), T.float32)
            data = T.alloc_shared((local_m, local_n), data_dtype)
            scale = T.alloc_shared(scale_shape, T.float8_e8m0fnu)
            mx = T.alloc_shared((local_m, local_n), mx_dtype)

            T.copy(A, a_rsram)

            for block_m in T.serial(num_m_blocks):
                for block_n in T.serial(num_n_blocks):
                    block = block_m * num_n_blocks + block_n
                    row_base = block_m * 32
                    col_base = block_n * 32

                    T.reduce_absmax(
                        a_rsram[row_base : row_base + 32, col_base : col_base + 32],
                        amax,
                        dim=1,
                        clear=True,
                    )
                    for row in T.Tiles([32]):
                        safe_amax = T.max(amax[row], T.float32(1e-4))
                        bf16_scale = T.Cast("bfloat16", safe_amax * T.float32(data_max_inv))
                        scale[block, row] = _to_e8m0_scale(bf16_scale)

                    for row, col in T.Tiles([32, 32]):
                        scale_fp32 = _e8m0_scale_to_fp32(scale[block, row])
                        value = T.Cast("float32", a_rsram[row_base + row, col_base + col]) / scale_fp32
                        data[row_base + row, col_base + col] = T.Cast(
                            data_dtype_name,
                            T.clamp(value, T.float32(-data_max), T.float32(data_max)),
                        )

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
            y_rsram = T.alloc_shared(shape, T.bfloat16)

            T.copy(MX, mx)
            T.mx_unpack(mx, data, scale)

            for block_m in T.serial(num_m_blocks):
                for block_n in T.serial(num_n_blocks):
                    block = block_m * num_n_blocks + block_n
                    row_base = block_m * 32
                    col_base = block_n * 32

                    for row, col in T.Tiles([32, 32]):
                        value = T.Cast("float32", data[row_base + row, col_base + col])
                        scale_fp32 = _e8m0_scale_to_fp32(scale[block, row])
                        y_rsram[row_base + row, col_base + col] = T.Cast("bfloat16", value * scale_fp32)

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
            amax = T.alloc_shared((32,), T.float32)
            a_data = T.alloc_shared(a_shape, data_dtype)
            b_data = T.alloc_shared(b_shape, data_dtype)
            a_scale = T.alloc_shared(a_scale_shape, T.float8_e8m0fnu)
            b_scale = T.alloc_shared(b_scale_shape, T.float8_e8m0fnu)
            a_mx = T.alloc_shared(a_shape, mx_dtype)
            b_mx = T.alloc_shared(b_shape, mx_dtype)
            a_mx_asram = T.alloc_shared(a_shape, mx_dtype)
            b_mx_wsram = T.alloc_shared(b_shape, mx_dtype)
            c_rsram = T.alloc_shared(c_shape, T.bfloat16)

            T.copy(A, a_rsram)
            T.copy(B, b_rsram)

            for block_k in T.serial(a_k_blocks):
                col_base = block_k * 32

                T.reduce_absmax(a_rsram[0:32, col_base : col_base + 32], amax, dim=1, clear=True)
                for row in T.Tiles([a_scale_shape[1]]):
                    safe_amax = T.max(amax[row], T.float32(1e-4))
                    bf16_scale = T.Cast("bfloat16", safe_amax * T.float32(data_max_inv))
                    a_scale[block_k, row] = _to_e8m0_scale(bf16_scale)

                for row, col in T.Tiles([32, 32]):
                    scale_fp32 = _e8m0_scale_to_fp32(a_scale[block_k, row])
                    value = T.Cast("float32", a_rsram[row, col_base + col]) / scale_fp32
                    a_data[row, col_base + col] = T.Cast(
                        data_dtype_name,
                        T.clamp(value, T.float32(-data_max), T.float32(data_max)),
                    )

            for block_k in T.serial(b_k_blocks):
                col_base = block_k * 32

                T.reduce_absmax(b_rsram[0:32, col_base : col_base + 32], amax, dim=1, clear=True)
                for row in T.Tiles([b_scale_shape[1]]):
                    safe_amax = T.max(amax[row], T.float32(1e-4))
                    bf16_scale = T.Cast("bfloat16", safe_amax * T.float32(data_max_inv))
                    b_scale[block_k, row] = _to_e8m0_scale(bf16_scale)

                for row, col in T.Tiles([32, 32]):
                    scale_fp32 = _e8m0_scale_to_fp32(b_scale[block_k, row])
                    value = T.Cast("float32", b_rsram[row, col_base + col]) / scale_fp32
                    b_data[row, col_base + col] = T.Cast(
                        data_dtype_name,
                        T.clamp(value, T.float32(-data_max), T.float32(data_max)),
                    )

            T.mx_pack(a_data, a_scale, a_mx)
            T.mx_pack(b_data, b_scale, b_mx)
            T.copy(a_mx, a_mx_asram)
            T.copy(b_mx, b_mx_wsram)
            T.clear(c_rsram)
            T.gemm(a_mx_asram, b_mx_wsram, c_rsram, transpose_B=True)
            T.copy(c_rsram, C)

    return main


@target("Sunmmio")
def mx_ocp_quantized_mma_generic_shape_kernel_for_debug(
    M,
    N,
    K,
    mx_dtype,
    data_dtype,
    data_dtype_name,
    data_max,
):
    assert M % 32 == 0
    assert N % 32 == 0
    assert K % 32 == 0

    a_shape = (M, K)
    b_shape = (N, K)
    c_shape = (M, N)
    num_m_blocks = M // 32
    num_n_blocks = N // 32
    num_k_blocks = K // 32
    shard_policy = T.MeshShardingPolicy(replicate=MeshReplicationType.ALL)
    a_tensor_layout = make_zz_layout(a_shape, axes=[0, 1], block_shape=(32, 32))
    b_tensor_layout = make_zz_layout(b_shape, axes=[0, 1], block_shape=(32, 32))
    c_tensor_layout = make_zz_layout(c_shape, axes=[0, 1], block_shape=(32, 32))
    a_mx_layout = make_mxzz_layout(a_shape, dtype=mx_dtype)
    b_mx_layout = make_mxzz_layout(b_shape, dtype=mx_dtype)
    a_scale_shape = _int_shape(get_mx_scale_shape(a_mx_layout, mx_dtype))
    b_scale_shape = _int_shape(get_mx_scale_shape(b_mx_layout, mx_dtype))
    data_max_inv = 1.0 / data_max
    assert a_scale_shape == (num_m_blocks * num_k_blocks, 32)
    assert b_scale_shape == (num_n_blocks * num_k_blocks, 32)

    @T.prim_func
    def main(
        A: T.MeshTensor(a_shape, shard_policy, T.bfloat16, layout=a_tensor_layout),  # type: ignore
        B: T.MeshTensor(b_shape, shard_policy, T.bfloat16, layout=b_tensor_layout),  # type: ignore
        C: T.MeshTensor(c_shape, shard_policy, T.bfloat16, layout=c_tensor_layout),  # type: ignore
    ):
        with T.Kernel():
            a_rsram = T.alloc_shared(a_shape, T.bfloat16)
            b_rsram = T.alloc_shared(b_shape, T.bfloat16)
            amax = T.alloc_shared((32,), T.float32)
            a_data = T.alloc_shared(a_shape, data_dtype)
            b_data = T.alloc_shared(b_shape, data_dtype)
            a_scale = T.alloc_shared(a_scale_shape, T.float8_e8m0fnu)
            b_scale = T.alloc_shared(b_scale_shape, T.float8_e8m0fnu)
            a_mx = T.alloc_shared(a_shape, mx_dtype)
            b_mx = T.alloc_shared(b_shape, mx_dtype)
            a_mx_asram = T.alloc_shared(a_shape, mx_dtype)
            b_mx_wsram = T.alloc_shared(b_shape, mx_dtype)
            c_rsram = T.alloc_shared(c_shape, T.bfloat16)

            T.copy(A, a_rsram)
            T.copy(B, b_rsram)

            for block_m in T.serial(num_m_blocks):
                for block_k in T.serial(num_k_blocks):
                    block = block_m * num_k_blocks + block_k
                    row_base = block_m * 32
                    col_base = block_k * 32

                    T.reduce_absmax(
                        a_rsram[row_base : row_base + 32, col_base : col_base + 32],
                        amax,
                        dim=1,
                        clear=True,
                    )
                    for row in T.Tiles([a_scale_shape[1]]):
                        safe_amax = T.max(amax[row], T.float32(1e-4))
                        bf16_scale = T.Cast("bfloat16", safe_amax * T.float32(data_max_inv))
                        a_scale[block, row] = _to_e8m0_scale(bf16_scale)

                    for row, col in T.Tiles([32, 32]):
                        scale_fp32 = _e8m0_scale_to_fp32(a_scale[block, row])
                        value = T.Cast("float32", a_rsram[row_base + row, col_base + col]) / scale_fp32
                        a_data[row_base + row, col_base + col] = T.Cast(
                            data_dtype_name,
                            T.clamp(value, T.float32(-data_max), T.float32(data_max)),
                        )

            for block_n in T.serial(num_n_blocks):
                for block_k in T.serial(num_k_blocks):
                    block = block_n * num_k_blocks + block_k
                    row_base = block_n * 32
                    col_base = block_k * 32

                    T.reduce_absmax(
                        b_rsram[row_base : row_base + 32, col_base : col_base + 32],
                        amax,
                        dim=1,
                        clear=True,
                    )
                    for row in T.Tiles([b_scale_shape[1]]):
                        safe_amax = T.max(amax[row], T.float32(1e-4))
                        bf16_scale = T.Cast("bfloat16", safe_amax * T.float32(data_max_inv))
                        b_scale[block, row] = _to_e8m0_scale(bf16_scale)

                    for row, col in T.Tiles([32, 32]):
                        scale_fp32 = _e8m0_scale_to_fp32(b_scale[block, row])
                        value = T.Cast("float32", b_rsram[row_base + row, col_base + col]) / scale_fp32
                        b_data[row_base + row, col_base + col] = T.Cast(
                            data_dtype_name,
                            T.clamp(value, T.float32(-data_max), T.float32(data_max)),
                        )

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
    c_tensor_layout = make_zz_layout(c_shape, axes=[0, 1], block_shape=(32, 32))
    a_tensor = T.MeshTensor(a_shape, shard_policy, T.bfloat16, layout=a_tensor_layout)
    b_tensor = T.MeshTensor(b_shape, shard_policy, T.bfloat16, layout=b_tensor_layout)
    c_tensor = T.MeshTensor(c_shape, shard_policy, T.bfloat16, layout=c_tensor_layout)
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
            amax = T.alloc_shared((32,), T.float32)
            a_data = T.alloc_shared(a_shape, data_dtype)
            b_data = T.alloc_shared(b_shape, data_dtype)
            a_scale = T.alloc_shared(a_scale_shape, T.float8_e8m0fnu)
            b_scale = T.alloc_shared(b_scale_shape, T.float8_e8m0fnu)
            a_mx = T.alloc_shared(a_shape, mx_dtype)
            b_mx = T.alloc_shared(b_shape, mx_dtype)
            a_mx_asram = T.alloc_shared(a_shape, mx_dtype)
            b_mx_wsram = T.alloc_shared(b_shape, mx_dtype)
            c_rsram = T.alloc_shared(c_shape, T.bfloat16)

            T.copy(A, a_rsram)
            T.copy(B, b_rsram)

            for block_k in T.serial(a_k_blocks):
                col_base = block_k * 32

                T.reduce_absmax(a_rsram[0:32, col_base : col_base + 32], amax, dim=1, clear=True)
                for row in T.Tiles([a_scale_shape[1]]):
                    safe_amax = T.max(amax[row], T.float32(1e-4))
                    bf16_scale = T.Cast("bfloat16", safe_amax * T.float32(data_max_inv))
                    a_scale[block_k, row] = _to_e8m0_scale(bf16_scale)

                for row, col in T.Tiles([32, 32]):
                    scale_fp32 = _e8m0_scale_to_fp32(a_scale[block_k, row])
                    value = T.Cast("float32", a_rsram[row, col_base + col]) / scale_fp32
                    a_data[row, col_base + col] = T.Cast(
                        data_dtype_name,
                        T.clamp(value, T.float32(-data_max), T.float32(data_max)),
                    )

            for block_k in T.serial(b_k_blocks):
                row_base = block_k * 32

                T.reduce_absmax(b_rsram[row_base : row_base + 32, 0:32], amax, dim=0, clear=True)
                for col in T.Tiles([b_scale_shape[1]]):
                    safe_amax = T.max(amax[col], T.float32(1e-4))
                    bf16_scale = T.Cast("bfloat16", safe_amax * T.float32(data_max_inv))
                    b_scale[block_k, col] = _to_e8m0_scale(bf16_scale)

                for row, col in T.Tiles([32, 32]):
                    scale_fp32 = _e8m0_scale_to_fp32(b_scale[block_k, col])
                    value = T.Cast("float32", b_rsram[row_base + row, col]) / scale_fp32
                    b_data[row_base + row, col] = T.Cast(
                        data_dtype_name,
                        T.clamp(value, T.float32(-data_max), T.float32(data_max)),
                    )

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
def test_mx_ocp_quant_sharded_kernel_codegen_logs_mlir(
    tmp_path,
    mx_dtype,
    data_dtype,
    data_dtype_name,
    data_max,
    mx_token,
):
    src = validate_sunmmio_codegen_with_npuir_opt(
        mx_ocp_quant_sharded_kernel_for_debug(mx_dtype, data_dtype, data_dtype_name, data_max),
        tmp_path,
        mlir_filename=f"mx_ocp_quant_sharded_{data_dtype_name}_suvm.mlir",
        expected_tokens=(
            mx_token,
            "!suvm.memtensor<64x64xbf16",
            f"!suvm.memtensor<64x64x{mx_token}",
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


@pytest.mark.parametrize(
    "M,N,K",
    (
        pytest.param(32, 32, 64, id="m32_n32_k64"),
        pytest.param(64, 64, 64, id="m64_n64_k64"),
        pytest.param(64, 32, 128, id="m64_n32_k128"),
    ),
)
@pytest.mark.parametrize("mx_dtype,data_dtype,data_dtype_name,data_max,mx_token", MX_OCP_QUANT_CASES)
def test_mx_ocp_quantized_mma_generic_shape_kernel_codegen_logs_mlir(
    tmp_path,
    M,
    N,
    K,
    mx_dtype,
    data_dtype,
    data_dtype_name,
    data_max,
    mx_token,
):
    src = validate_sunmmio_codegen_with_npuir_opt(
        mx_ocp_quantized_mma_generic_shape_kernel_for_debug(
            M,
            N,
            K,
            mx_dtype,
            data_dtype,
            data_dtype_name,
            data_max,
        ),
        tmp_path,
        mlir_filename=f"mx_ocp_quantized_mma_m{M}_n{N}_k{K}_{data_dtype_name}_suvm.mlir",
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
