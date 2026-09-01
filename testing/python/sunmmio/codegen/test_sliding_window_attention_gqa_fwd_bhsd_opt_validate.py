import os

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import make_zz_layout

from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import (
    assert_source_contains,
    validate_sunmmio_codegen_with_npuir_opt,
)


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
os.environ["SUNMMIO_TEST_LOG_IR"] = "1"


SLIDING_WINDOW_ATTENTION_GQA_FWD_BHSD_CASES = [
    (1, 32, 8, 128, 128, 4, 64),
    (1, 48, 8, 128, 128, 4, 96),
]


@target("Sunmmio")
def sliding_window_attention_gqa_fwd_bhsd(
    batch=1,
    q_heads=32,
    kv_heads=8,
    seq_len=128,
    dim=128,
    global_window=4,
    local_window=1024,
    block_M=64,
    block_N=64,
    num_stages=0,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    groups = q_heads // kv_heads
    q_shape = [batch, seq_len, q_heads, dim]
    kv_shape = [batch, seq_len, kv_heads, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    assert q_heads % kv_heads == 0
    assert global_window <= seq_len
    assert global_window <= block_N

    shard_policy = T.placement.full_shard(0, 2)

    Q_layout = make_zz_layout(q_shape, [1, 3], (32, 32))
    K_layout = make_zz_layout(kv_shape, [1, 3], (32, 32))
    V_layout = make_zz_layout(kv_shape, [1, 3], (32, 32))
    O_layout = make_zz_layout(q_shape, [1, 3], (32, 32))

    @T.prim_func
    def main(
        Q: T.MeshTensor(q_shape, shard_policy, dtype, layout=Q_layout),  # type: ignore
        K: T.MeshTensor(kv_shape, shard_policy, dtype, layout=K_layout),  # type: ignore
        V: T.MeshTensor(kv_shape, shard_policy, dtype, layout=V_layout),  # type: ignore
        Output: T.MeshTensor(q_shape, shard_policy, dtype, layout=O_layout),  # type: ignore
    ):
        with T.Kernel() as _cid:
            sharded_batch = Q.local_shape[0]
            sharded_heads = Q.local_shape[2]

            Q_shared = T.alloc_shared([block_M, dim], dtype)
            Q_shared_stage = T.alloc_shared([block_M, dim], dtype, scope="shared.rsram")
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
                        T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared_stage)
                        T.copy(Q_shared_stage, Q_shared)
                        T.fill(acc_o, 0)
                        T.fill(logsum, 0)
                        T.fill(scores_max, -T.infinity(accum_dtype))

                        loop_range = T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N))
                        # for k in T.Pipelined(loop_range, num_stages=num_stages):
                        for k in T.serial(loop_range):
                            T.copy(K[bz, k * block_N : (k + 1) * block_N, by // groups, :], K_shared)
                            for i, j in T.Tiles([block_M, block_N]):
                                q_pos = bx * block_M + i
                                k_pos = k * block_N + j
                                acc_s[i, j] = T.if_then_else(
                                    q_pos >= k_pos,
                                    T.if_then_else(
                                        k_pos > q_pos - local_window,
                                        0,
                                        -T.infinity(acc_s.dtype),
                                    ),
                                    -T.infinity(acc_s.dtype),
                                )
                            if k == 0 and bx > 0:
                                for i, j in T.Tiles([block_M, block_N]):
                                    acc_s[i, j] = T.if_then_else(
                                        j + i < global_window + i,
                                        0,
                                        acc_s[i, j],
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


@pytest.mark.parametrize(
    "batch,q_heads,kv_heads,seq_len,dim,global_window,local_window",
    SLIDING_WINDOW_ATTENTION_GQA_FWD_BHSD_CASES,
)
def test_sliding_window_attention_gqa_fwd_bhsd_codegen_passes_loose_npuir_opt(
    tmp_path,
    batch,
    q_heads,
    kv_heads,
    seq_len,
    dim,
    global_window,
    local_window,
):
    src = validate_sunmmio_codegen_with_npuir_opt(
        sliding_window_attention_gqa_fwd_bhsd(
            batch=batch,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_len=seq_len,
            dim=dim,
            global_window=global_window,
            local_window=local_window,
        ),
        tmp_path,
        mlir_filename=(
            f"sliding_window_attention_gqa_fwd_bhsd_b{batch}_qh{q_heads}_kvh{kv_heads}"
            f"_s{seq_len}_d{dim}_gw{global_window}_lw{local_window}_suvm.mlir"
        ),
        expected_tokens=("suvm.copy_async", "suvm.tc.mma", "suvm.tile.reduce"),
    )
    assert_source_contains(
        src,
        ("suvm.tc.mma", "suvm.tile.reduce", "suvm.tile.range", "suvm.tile.cmpi", "suvm.tile.select"),
    )
    assert "fake_missing_binary" not in src


if __name__ == "__main__":
    tilelang.testing.main()
