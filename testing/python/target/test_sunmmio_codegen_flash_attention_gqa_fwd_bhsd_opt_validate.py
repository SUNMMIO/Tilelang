import os

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.carver.arch import driver
from tilelang.layout import make_zz_layout

from sunmmio_codegen_validation_utils import (
    assert_source_contains,
    validate_sunmmio_codegen_with_npuir_opt,
)


tilelang.env.disable_cache()
os.environ["SUNMMIO_TEST_PRINT"] = "1"
# os.environ["SUNMMIO_TEST_LOG_IR"] = "1"


def flashattn_gqa_fwd_bhsd(
    batch=1,
    heads=4,
    seq_len=128,
    dim=64,
    groups=1,
    block_M=64,
    block_N=64,
    num_stages=0,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    shard_policy = T.MeshShardingPolicy(y=0, x=2)
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    Q_layout = make_zz_layout(q_shape, [1, 3], (32, 32))
    K_layout = make_zz_layout(kv_shape, [1, 3], (32, 32))
    V_layout = make_zz_layout(kv_shape, [1, 3], (32, 32))
    O_layout = make_zz_layout(q_shape, [1, 3], (32, 32))

    @T.prim_func
    def main(
        Q: T.MeshTensor(q_shape, shard_policy, device_mesh_config, dtype, layout=Q_layout),  # type: ignore
        K: T.MeshTensor(kv_shape, shard_policy, device_mesh_config, dtype, layout=K_layout),  # type: ignore
        V: T.MeshTensor(kv_shape, shard_policy, device_mesh_config, dtype, layout=V_layout),  # type: ignore
        Output: T.MeshTensor(q_shape, shard_policy, device_mesh_config, dtype, layout=O_layout),  # type: ignore
    ):
        with T.Kernel(ncores) as _cid:
            sharded_batch = Q.shape[0]
            sharded_heads = Q.shape[2]

            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_shared([block_M, block_N], accum_dtype, scope="shared.rsram")
            acc_s_cast_local = T.alloc_shared([block_M, block_N], dtype)
            acc_s_cast = T.alloc_shared([block_M, block_N], dtype)
            acc_o = T.alloc_shared([block_M, dim], accum_dtype)
            scores_max = T.alloc_shared([block_M], accum_dtype)
            scores_max_prev = T.alloc_shared([block_M], accum_dtype)
            scores_scale = T.alloc_shared([block_M], accum_dtype)
            scores_sum = T.alloc_shared([block_M], accum_dtype)
            logsum = T.alloc_shared([block_M], accum_dtype)

            for bz in T.serial(sharded_batch):
                for by in T.serial(sharded_heads):
                    for bx in T.serial(T.ceildiv(seq_len, block_M)):
                        T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
                        T.fill(acc_o, 0)
                        T.fill(logsum, 0)
                        T.fill(scores_max, -T.infinity(accum_dtype))

                        loop_range = T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N))
                        for k in T.Pipelined(loop_range, num_stages=num_stages):
                            T.copy(K[bz, k * block_N : (k + 1) * block_N, by // groups, :], K_shared)
                            for i, j in T.Tiles([block_M, block_N]):
                                acc_s[i, j] = T.if_then_else(
                                    bx * block_M + i >= k * block_N + j,
                                    0,
                                    -T.infinity(acc_s.dtype),
                                )
                            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)

                            T.copy(scores_max, scores_max_prev)
                            T.fill(scores_max, -T.infinity(accum_dtype))
                            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                            for i in T.Tiles([block_M]):
                                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                            for i in T.Tiles([block_M]):
                                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                            for i, j in T.Tiles([block_M, block_N]):
                                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                            T.reduce_sum(acc_s, scores_sum, dim=1)
                            for i in T.Tiles([block_M]):
                                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                            for i, j in T.Tiles([block_M, block_N]):
                                acc_s_cast_local[i, j] = acc_s[i, j]
                            T.copy(acc_s_cast_local, acc_s_cast)

                            for i, j in T.Tiles([block_M, dim]):
                                acc_o[i, j] *= scores_scale[i]

                            T.copy(V[bz, k * block_N : (k + 1) * block_N, by // groups, :], V_shared)
                            T.gemm(acc_s_cast, V_shared, acc_o)

                        for i, j in T.Tiles([block_M, dim]):
                            acc_o[i, j] /= logsum[i]
                        T.copy(acc_o, O_shared)
                        T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main


def test_flashattn_gqa_fwd_bhsd_codegen_passes_loose_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        flashattn_gqa_fwd_bhsd(),
        tmp_path,
        mlir_filename="flashattn_gqa_fwd_bhsd_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.tc.mma", "suvm.tile.reduce"),
        opt_args=("--verify-each",),
    )
    assert_source_contains(src, ("suvm.tc.mma", "suvm.tile.reduce"))


if __name__ == "__main__":
    tilelang.testing.main()
