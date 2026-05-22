import pytest
from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
from tilelang.engine.phase import *
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from tilelang.language.mesh_tensor import MeshShardingPolicy


def matmul(M, N, K, block_M, block_N, block_K, num_stages, dtype="float16", accum_dtype="float"):
    @T.prim_func
    def gemm(
        A: T.MeshTensor(
            (M, K),
            sharding_policy=MeshShardingPolicy(cross_mesh_dim=0),
            device_mesh_config=(2, 2),
            hierarchical_dims=(4, 32, 128),
            hierarchical_groups=((0, 2), (2, 3)),
            hierarchical_strides=(32, 1, 4096),
            dtype=dtype,
        ),
        B: T.MeshTensor(
            (K, N),
            sharding_policy=MeshShardingPolicy(cross_mesh_dim=0),
            device_mesh_config=(2, 2),
            hierarchical_dims=(4, 32, 128),
            hierarchical_groups=((0, 2), (2, 3)),
            hierarchical_strides=(32, 1, 4096),
            dtype=dtype,
        ),
        C: T.MeshTensor(
            (M, N),
            sharding_policy=MeshShardingPolicy(cross_mesh_dim=0),
            device_mesh_config=(2, 2),
            hierarchical_dims=(4, 32, 128),
            hierarchical_groups=((0, 2), (2, 3)),
            hierarchical_strides=(32, 1, 4096),
            dtype=accum_dtype,
        ),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return gemm


# flashattention gpa prefill
def flashattn(batch=1, heads=64, seq_len=4096, dim=128, is_causal=False, groups=16, block_M=64, block_N=64, num_stages=3, threads=128):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = T.float16
    accum_dtype = T.float16

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_shared([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_shared([block_M, block_N], dtype)
            acc_o = T.alloc_shared([block_M, dim], accum_dtype)
            scores_max = T.alloc_shared([block_M], accum_dtype)
            scores_max_prev = T.alloc_shared([block_M], accum_dtype)
            scores_scale = T.alloc_shared([block_M], accum_dtype)
            scores_sum = T.alloc_shared([block_M], accum_dtype)
            logsum = T.alloc_shared([block_M], accum_dtype)

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N)) if is_causal else T.ceildiv(seq_len, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by // groups, :], K_shared)
                if is_causal:
                    for i, j in T.Tiles(acc_s, parallel=True):
                        acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype))
                else:
                    for i, j in T.Tiles(acc_s, parallel=True):
                        acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_len, -T.infinity(acc_s.dtype), 0)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Tiles(scores_max, parallel=True):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Tiles(scores_scale, parallel=True):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Tiles(acc_s, parallel=True):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Tiles(logsum, parallel=True):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Tiles(acc_o, parallel=True):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[bz, k * block_N : (k + 1) * block_N, by // groups, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Tiles(acc_o, parallel=True):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main


# flash decoding gqa decode
def flashdecoding(batch=1, heads=256, groups=8, seqlen_kv=8192, dim=128, block_N=128, block_H=64, num_split=1, num_stages=2, threads=1):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape_q = [batch, heads, dim]
    shape_k = [batch, seqlen_kv, groups, dim]
    shape_v = [batch, seqlen_kv, groups, dim]
    shape_o = [batch, heads, dim]
    dtype = T.float16
    accum_dtype = T.float16
    kv_group_num = heads // groups

    part_shape = [batch, heads, num_split, dim]
    valid_block_H = min(block_H, kv_group_num)
    valid_block_N = min(block_N, seqlen_kv // num_split)

    @T.prim_func
    def flashattn_gqa_decode_split(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_k, dtype),
        V: T.Tensor(shape_v, dtype),
        mask: T.Tensor([batch, seqlen_kv, groups], dtype),
        glse: T.Tensor([batch, heads, num_split], dtype),
        Output_partial: T.Tensor(part_shape, dtype),
        Output: T.Tensor(shape_o, dtype),
    ):
        # split
        with T.Kernel(batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([valid_block_H, dim], dtype)
            acc_s = T.alloc_shared([block_H, block_N], accum_dtype)
            acc_s_cast = T.alloc_shared([block_H, block_N], dtype)
            mask_local = T.alloc_shared([block_N], dtype)
            acc_o = T.alloc_shared([block_H, dim], accum_dtype)
            scores_max = T.alloc_shared([block_H], accum_dtype)
            scores_max_prev = T.alloc_shared([block_H], accum_dtype)
            scores_scale = T.alloc_shared([block_H], accum_dtype)
            scores_sum = T.alloc_shared([block_H], accum_dtype)
            logsum = T.alloc_shared([block_H], accum_dtype)

            bid = bx
            hid = by
            sid = bz
            cur_kv_head = hid // (kv_group_num // valid_block_H)

            T.copy(Q[bid, hid * valid_block_H : hid * valid_block_H + block_H, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv((seqlen_kv // num_split), block_N)

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(
                    K[
                        bid,
                        (seqlen_kv // num_split) * sid + k * valid_block_N : (seqlen_kv // num_split) * sid + (k + 1) * valid_block_N,
                        cur_kv_head,
                        :,
                    ],
                    K_shared,
                )
                T.copy(
                    mask[
                        bid,
                        (seqlen_kv // num_split) * sid + k * valid_block_N : (seqlen_kv // num_split) * sid + (k + 1) * valid_block_N,
                        cur_kv_head,
                    ],
                    mask_local,
                )
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Tiles(acc_s, parallel=True):
                    acc_s[i, j] = T.if_then_else((mask_local[j] != 0) & (j < seqlen_kv // num_split), acc_s[i, j], -T.infinity(accum_dtype))
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Tiles(scores_max, parallel=True):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Tiles(scores_scale, parallel=True):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Tiles(acc_s, parallel=True):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Tiles(logsum, parallel=True):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)
                for i, j in T.Tiles(acc_o, parallel=True):
                    acc_o[i, j] *= scores_scale[i]
                T.copy(
                    V[
                        bid,
                        (seqlen_kv // num_split) * sid + k * valid_block_N : (seqlen_kv // num_split) * sid + (k + 1) * valid_block_N,
                        cur_kv_head,
                        :,
                    ],
                    V_shared,
                )
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Tiles(acc_o, parallel=True):
                acc_o[i, j] /= logsum[i]
            for i in T.Tiles(logsum, parallel=True):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

            # for i in T.Tiles(glse, parallel=True):
            #     if i < valid_block_H:
            #         glse[bid, hid * valid_block_H + i, sid] = logsum[i]
            # # T.copy(acc_o[:valid_block_H, :], O_shared)
            # T.copy(acc_o, O_shared)
            T.copy(O_shared, Output_partial[bid, hid * valid_block_H : (hid + 1) * valid_block_H, sid, :])

        # combine
        # with T.Kernel(heads, batch, threads=128) as (by, bz):
        #     po_local = T.alloc_shared([dim], dtype)
        #     o_accum_local = T.alloc_shared([dim], accum_dtype)
        #     lse_local = T.alloc_shared([num_split, 128], dtype)
        #     lse_logsum_local = T.alloc_shared([128], accum_dtype)
        #     lse_max_local = T.alloc_shared([128], accum_dtype)
        #     scale_local = T.alloc_shared([128], accum_dtype)

        #     T.clear(lse_logsum_local)
        #     T.clear(o_accum_local)
        #     for k, j in T.Tiles(lse_local, parallel=True):
        #         lse_local[k, j] = glse[bz, by, k]
        #     T.reduce_max(lse_local, lse_max_local, dim=0, clear=True)
        #     for k in T.serial(num_split):
        #         for j in T.Tiles(128):
        #             lse_logsum_local[j] += T.exp2(lse_local[k, j] - lse_max_local[j])
        #     for j in T.Tiles(128):
        #         lse_logsum_local[j] = T.log2(lse_logsum_local[j]) + lse_max_local[j]
        #     for k in T.serial(num_split):
        #         for i in T.Tiles(dim):
        #             po_local[i] = Output_partial[bz, by, k, i]
        #         for j in T.Tiles(128):
        #             scale_local[j] = T.exp2(lse_local[k, j] - lse_logsum_local[j])
        #         # Note: Pay attention to dim and the number of threads in Tiles
        #         for i in T.Tiles(dim):
        #             o_accum_local[i] += po_local[i] * scale_local[i]
        #     for i in T.Tiles(dim):
        #         Output[bz, by, i] = o_accum_local[i]

    @T.prim_func
    def flashattn_gqa_decode_no_split(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_k, dtype),
        V: T.Tensor(shape_v, dtype),
        mask: T.Tensor([batch, seqlen_kv, groups], dtype),
        glse: T.Tensor([batch, heads, num_split], dtype),
        Output_partial: T.Tensor(part_shape, dtype),
        Output: T.Tensor(shape_o, dtype),
    ):
        with T.Kernel(batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([valid_block_H, dim], dtype)
            acc_s = T.alloc_shared([block_H, block_N], accum_dtype)
            acc_s_cast = T.alloc_shared([block_H, block_N], dtype)
            mask_local = T.alloc_shared([block_N], dtype)
            acc_o = T.alloc_shared([block_H, dim], accum_dtype)
            scores_max = T.alloc_shared([block_H], accum_dtype)
            scores_max_prev = T.alloc_shared([block_H], accum_dtype)
            scores_scale = T.alloc_shared([block_H], accum_dtype)
            scores_sum = T.alloc_shared([block_H], accum_dtype)
            logsum = T.alloc_shared([block_H], accum_dtype)

            bid = bx
            hid = by
            cur_kv_head = hid // (kv_group_num // valid_block_H)

            T.copy(Q[bid, hid * valid_block_H : hid * valid_block_H + block_H, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv((seqlen_kv // num_split), block_N)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], K_shared)
                T.copy(mask[bid, k * block_N : (k + 1) * block_N, cur_kv_head], mask_local)
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Tiles(acc_s, parallel=True):
                    acc_s[i, j] = T.if_then_else(mask_local[j] != 0, acc_s[i, j], -T.infinity(accum_dtype))
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Tiles(scores_max, parallel=True):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Tiles(scores_scale, parallel=True):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Tiles(acc_s, parallel=True):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Tiles(logsum, parallel=True):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)
                for i, j in T.Tiles(acc_o, parallel=True):
                    acc_o[i, j] *= scores_scale[i]
                T.copy(V[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Tiles(acc_o, parallel=True):
                acc_o[i, j] /= logsum[i]
            for i in T.Tiles(logsum, parallel=True):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            # T.copy(acc_o[:valid_block_H, :], O_shared)
            T.copy(O_shared, Output[bid, hid * valid_block_H : (hid + 1) * valid_block_H, :])

    if num_split > 1:
        return flashattn_gqa_decode_split
    else:
        return flashattn_gqa_decode_no_split


# flash mla decode
def flashmladecode(
    batch=1,
    heads=128,
    kv_head_num=1,
    seqlen_kv=8192,
    dim=512,
    pe_dim=64,
    block_N=64,
    block_H=64,
    num_split=1,
    softmax_scale=1 / 24,
    num_stages=1,
):
    scale = float(softmax_scale * 1.44269504)  # log2(e)
    dtype = T.float16
    accum_dtype = T.float16
    kv_group_num = heads // kv_head_num
    VALID_BLOCK_H = min(block_H, kv_group_num)
    assert kv_head_num == 1, "kv_head_num must be 1"

    @T.prim_func
    def main_split(
        Q: T.Tensor([batch, heads, dim], dtype),
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
        glse: T.Tensor([batch, heads, num_split], dtype),
        Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        # flash_attn_split
        with T.Kernel(batch, heads // min(block_H, kv_group_num), num_split, threads=256) as (bid, hid, bz):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
            KV_shared = T.alloc_shared([block_N, dim], dtype)
            K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
            O_shared = T.alloc_shared([block_H, dim], dtype)
            acc_s = T.alloc_shared([block_H, block_N], accum_dtype)
            acc_s_cast = T.alloc_shared([block_H, block_N], dtype)
            acc_o = T.alloc_shared([block_H, dim], accum_dtype)
            scores_max = T.alloc_shared([block_H], accum_dtype)
            scores_max_prev = T.alloc_shared([block_H], accum_dtype)
            scores_scale = T.alloc_shared([block_H], accum_dtype)
            scores_sum = T.alloc_shared([block_H], accum_dtype)
            logsum = T.alloc_shared([block_H], accum_dtype)

            cur_kv_head = hid // (kv_group_num // block_H)
            T.use_swizzle(10)

            T.copy(Q[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :], Q_shared)
            T.copy(Q_pe[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :], Q_pe_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv((seqlen_kv // num_split), block_N)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                kv_start = (seqlen_kv // num_split) * bz + k * block_N
                kv_end = (seqlen_kv // num_split) * bz + (k + 1) * block_N
                T.copy(KV[bid, kv_start:kv_end, cur_kv_head, :], KV_shared)
                T.copy(K_pe[bid, kv_start:kv_end, cur_kv_head, :], K_pe_shared)
                T.clear(acc_s)
                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Tiles(scores_max, parallel=True):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Tiles(scores_scale, parallel=True):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Tiles(acc_s, parallel=True):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, S_shared)
                T.copy(S_shared, acc_s_cast)
                for i in T.Tiles(logsum, parallel=True):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Tiles(acc_o, parallel=True):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(acc_s_cast, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
            for i, j in T.Tiles(acc_o, parallel=True):
                acc_o[i, j] /= logsum[i]
            for i in T.Tiles(logsum, parallel=True):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(logsum, glse[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, bz])
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output_partial[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, bz, :])

        # combine
        # with T.Kernel(heads, batch, threads=128) as (hid, bz):
        #     po_local = T.alloc_shared([dim], dtype)
        #     o_accum_local = T.alloc_shared([dim], accum_dtype)
        #     lse_local_split = T.alloc_var(accum_dtype)
        #     lse_logsum_local = T.alloc_var(accum_dtype)
        #     lse_max_local = T.alloc_var(accum_dtype)
        #     scale_local = T.alloc_var(accum_dtype)

        #     T.clear(lse_logsum_local)
        #     T.clear(o_accum_local)
        #     lse_max_local = -T.infinity(accum_dtype)
        #     for k in T.Tiles(glse, parallel=True):
        #         lse_max_local = T.max(lse_max_local, glse[bz, hid, k])
        #     for k in T.Pipelined(num_split, num_stages=num_stages):
        #         lse_local_split = glse[bz, hid, k]
        #         lse_logsum_local += T.exp2(lse_local_split - lse_max_local)
        #     lse_logsum_local = T.log2(lse_logsum_local) + lse_max_local
        #     for k in T.serial(num_split):
        #         for i in T.Tiles(po_local, parallel=True):
        #             po_local[i] = Output_partial[bz, hid, k, i]
        #         lse_local_split = glse[bz, hid, k]
        #         scale_local = T.exp2(lse_local_split - lse_logsum_local)
        #         for i in T.Tiles(o_accum_local, parallel=True):
        #             o_accum_local[i] += po_local[i] * scale_local
        #     for i in T.Tiles(Output, parallel=True):
        #         Output[bz, hid, i] = o_accum_local[i]

    @T.prim_func
    def main_no_split(
        Q: T.Tensor([batch, heads, dim], dtype),
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
        glse: T.Tensor([batch, heads, num_split], dtype),
        Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        with T.Kernel(heads // min(block_H, kv_group_num), batch, threads=256) as (hid, bid):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
            KV_shared = T.alloc_shared([block_N, dim], dtype)
            K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
            O_shared = T.alloc_shared([block_H, dim], dtype)
            acc_s = T.alloc_shared([block_H, block_N], accum_dtype)
            acc_o = T.alloc_shared([block_H, dim], accum_dtype)
            scores_max = T.alloc_shared([block_H], accum_dtype)
            scores_max_prev = T.alloc_shared([block_H], accum_dtype)
            scores_scale = T.alloc_shared([block_H], accum_dtype)
            scores_sum = T.alloc_shared([block_H], accum_dtype)
            logsum = T.alloc_shared([block_H], accum_dtype)

            cur_kv_head = hid // (kv_group_num // block_H)

            T.copy(Q[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :], Q_shared)
            T.copy(Q_pe[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :], Q_pe_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(seqlen_kv, block_N)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(KV[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], KV_shared)
                T.copy(K_pe[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], K_pe_shared)
                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)
                T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Tiles(scores_max, parallel=True):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Tiles(scores_scale, parallel=True):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Tiles(acc_s, parallel=True):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, S_shared)
                for i in T.Tiles(logsum, parallel=True):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Tiles(acc_o, parallel=True):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
            for i, j in T.Tiles(acc_o, parallel=True):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :])

    if num_split > 1:
        return main_split
    else:
        return main_no_split


def native_sparse_attention(
    batch=2, heads=32, seq_len=64, dim=32, is_causal=True, scale=0.1, block_size=32, groups=32, selected_blocks=32, num_stages=3
):
    if scale is None:
        scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    else:
        scale = scale * 1.44269504  # log2(e)

    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    block_indices_shape = [batch, seq_len, head_kv, selected_blocks]
    block_indices_dtype = T.int32
    dtype = T.float16
    accum_dtype = T.float16
    block_S = block_size
    block_T = min(128, tilelang.math.next_power_of_2(dim))

    NK = tilelang.cdiv(dim, block_T)
    NV = tilelang.cdiv(dim, block_T)
    assert NK == 1, "The key dimension can not be larger than 256"

    S = selected_blocks
    G = groups
    BS = block_S
    BK = BV = block_T
    threads = 32
    print(G, BS, BK, BV)

    @T.prim_func
    def native_sparse_attention(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(seq_len, NV, batch * head_kv, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([G, BK], dtype)
            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            O_shared = T.alloc_shared([G, BV], dtype)

            acc_s = T.alloc_shared([G, BS], accum_dtype)
            acc_s_cast = T.alloc_shared([G, BS], dtype)
            acc_o = T.alloc_shared([G, BV], accum_dtype)
            scores_max = T.alloc_shared([G], accum_dtype)
            scores_max_prev = T.alloc_shared([G], accum_dtype)
            scores_scale = T.alloc_shared([G], accum_dtype)
            scores_sum = T.alloc_shared([G], accum_dtype)
            logsum = T.alloc_shared([G], accum_dtype)

            i_t, i_v, i_bh = bx, by, bz
            i_b, i_h = i_bh // head_kv, i_bh % head_kv

            NS = S
            T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Q_shared)

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            for i in T.Pipelined(NS, num_stages=num_stages):
                i_s = BlockIndices[i_b, i_t, i_h, i] * BS
                if i_s <= i_t and i_s >= 0:
                    # [BS, BK]
                    T.copy(K[i_b, i_s : i_s + BS, i_h, :], K_shared)

                    if is_causal:
                        for i, j in T.Tiles(acc_s, parallel=True):
                            acc_s[i, j] = T.if_then_else(i_t >= (i_s + j), 0, -T.infinity(acc_s.dtype))
                    else:
                        T.clear(acc_s)

                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    # Softmax
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=True)
                    for i in T.Tiles(scores_scale, parallel=True):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Tiles(acc_s, parallel=True):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Tiles(logsum, parallel=True):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)

                    # Rescale
                    for i, j in T.Tiles(acc_o, parallel=True):
                        acc_o[i, j] *= scores_scale[i]

                    # V * softmax(Q * K)
                    T.copy(V[i_b, i_s : i_s + BS, i_h, i_v * BV : (i_v + 1) * BV], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Tiles(acc_o, parallel=True):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[i_b, i_t, i_h * G : (i_h + 1) * G, i_v * BV : (i_v + 1) * BV])

    return native_sparse_attention


CASES = [
    # matmul(1024, 1024, 1024, 128, 128, 32, num_stages=2),
    # matmul(1024, 1024, 1024, 128, 128, 32, num_stages=3),
    # matmul(1024, 1024, 1024, 128, 128, 32, num_stages=4),
    # lambda: matmul(1024, 1024, 1024, 128, 128, 32, num_stages=5),
    # lambda: flashattn(num_stages=3),
    lambda: flashdecoding(num_stages=3),
    # lambda: flashmladecode(num_stages=5),
    # lambda: native_sparse_attention(num_stages=5),
]


@pytest.mark.parametrize(
    "kernel",
    CASES,
)
def test_tilelang_transform_sunmmio_pipeline(kernel):
    name = SUNMMIO_TARGET_DESC
    target = tvm.target.Target(name)

    with tvm.target.Target(target):
        mod = tvm.IRModule.from_expr(kernel().with_attr("global_symbol", "main"))
        mod = LowerAndLegalize(mod, target)
        mod = tl.transform.SunmmioPipelinePlanning(debug=True)(mod)
        mod = tl.transform.InjectSunmmioPipeline()(mod)
        mod.show()
