import os

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.carver.arch import driver
from tilelang.layout import make_row_major, make_zz_layout

from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import (
    assert_source_contains,
    validate_sunmmio_codegen_with_npuir_opt,
)


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
os.environ["SUNMMIO_TEST_LOG_IR"] = "1"

LOOSE_OPT_ARGS = ("--verify-each",)


@target("Sunmmio")
def gqa_decode_pr223(
    batch=1,
    heads=4,
    kv_heads=1,
    seqlen_kv=128,
    dim=64,
    block_N=128,
    enable_mask=True,
):
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    scale = (1.0 / dim) ** 0.5 * 1.44269504
    shape_q = [batch, heads, dim]
    shape_kv = [batch, seqlen_kv, kv_heads, dim]
    shape_o = [batch, heads, dim]
    dtype = T.float16
    accum_dtype = T.float32
    kv_group_num = heads // kv_heads
    block_H = kv_group_num

    @T.prim_func
    def main(
        Q: T.MeshTensor(shape_q, T.MeshShardingPolicy(y=0, x=1), device_mesh_config, dtype, make_zz_layout(shape_q)),  # type: ignore
        K: T.MeshTensor(shape_kv, T.MeshShardingPolicy(y=0, x=2), device_mesh_config, dtype, make_zz_layout(shape_kv, axes=(1, 3))),  # type: ignore
        V: T.MeshTensor(shape_kv, T.MeshShardingPolicy(y=0, x=2), device_mesh_config, dtype, make_zz_layout(shape_kv, axes=(1, 3))),  # type: ignore
        mask: T.MeshTensor(
            [batch, seqlen_kv],
            T.MeshShardingPolicy(y=0, replicate=T.MeshReplicationType.ROW),
            device_mesh_config,
            "uint16",
            make_row_major([batch, seqlen_kv]),
        ),  # type: ignore
        Output: T.MeshTensor(shape_o, T.MeshShardingPolicy(y=0, x=1), device_mesh_config, dtype, make_zz_layout(shape_o)),  # type: ignore
    ):
        with T.Kernel(ncores):
            sharded_batch, sharded_heads, _ = Q.shape

            Q_shared = T.alloc_shared([block_H, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            acc_s = T.alloc_shared([block_H, block_N], accum_dtype)
            acc_s_cast = T.alloc_shared([block_H, block_N], dtype)
            mask_local = T.alloc_shared([block_N], "uint16")
            acc_o = T.alloc_shared([block_H, dim], accum_dtype)
            acc_o_cast = T.alloc_shared([block_H, dim], dtype)
            scores_max = T.alloc_shared([block_H], accum_dtype)
            scores_max_prev = T.alloc_shared([block_H], accum_dtype)
            scores_scale = T.alloc_shared([block_H], accum_dtype)
            scores_sum = T.alloc_shared([block_H], accum_dtype)
            logsum = T.alloc_shared([block_H], accum_dtype)

            for bid in T.serial(sharded_batch):
                for hid in T.serial(T.ceildiv(sharded_heads, block_H)):
                    cur_kv_head = hid
                    T.copy(Q[bid, hid * block_H : (hid + 1) * block_H, :], Q_shared)
                    T.fill(acc_o, 0)
                    T.fill(logsum, 0)
                    T.fill(scores_max, -T.infinity(accum_dtype))

                    loop_range = T.ceildiv(seqlen_kv, block_N)
                    for k in T.serial(loop_range):
                        T.copy(K[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], K_shared)
                        T.copy(mask[bid, k * block_N : (k + 1) * block_N], mask_local)
                        T.gemm(Q_shared, K_shared, acc_s, clear_accum=True, transpose_B=True)
                        if enable_mask:
                            for i, j in T.Tiles([block_H, block_N]):
                                acc_s[i, j] = T.if_then_else(mask_local[j] != 0, acc_s[i, j], -T.infinity(accum_dtype))
                        T.copy(scores_max, scores_max_prev)
                        T.fill(scores_max, -T.infinity(accum_dtype))
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Tiles([block_H]):
                            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                        for i in T.Tiles([block_H]):
                            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                        for i, j in T.Tiles([block_H, block_N]):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Tiles([block_H]):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        T.copy(acc_s, acc_s_cast)
                        for i, j in T.Tiles([block_H, dim]):
                            acc_o[i, j] *= scores_scale[i]
                        T.copy(V[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], V_shared)
                        T.gemm(acc_s_cast, V_shared, acc_o)

                    for i, j in T.Tiles([block_H, dim]):
                        acc_o_cast[i, j] = acc_o[i, j] / logsum[i]
                    T.copy(acc_o_cast, Output[bid, hid * block_H : (hid + 1) * block_H, :])

    return main


@pytest.mark.parametrize("enable_mask", [False, True])
def test_pr223_gqa_decode_h4_codegen_validates_with_npuir_opt(tmp_path, enable_mask):
    case_name = f"pr223_gqa_decode_{'with' if enable_mask else 'without'}_mask_h4_d64"
    src = validate_sunmmio_codegen_with_npuir_opt(
        gqa_decode_pr223(heads=4, kv_heads=1, dim=64, enable_mask=enable_mask),
        tmp_path,
        mlir_filename=f"{case_name}.mlir",
        expected_tokens=("suvm.tc.mma", "suvm.tile.reduce"),
        opt_args=LOOSE_OPT_ARGS,
    )
    assert_source_contains(src, ("suvm.tc.mma", "suvm.tile.reduce"))
    assert "fake_missing" not in src


if __name__ == "__main__":
    tilelang.testing.main()
