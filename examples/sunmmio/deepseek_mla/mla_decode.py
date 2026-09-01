"""DeepSeek MLA decode on Sunmmio.

Based on examples/deepseek_mla/example_mla_decode_persistent.py, but with a mesh
sharding/tiling design instead of the CUDA persistent split-K:

  * The 4 mesh ROWS shard the batch dimension.
  * Q / Q_pe (and Output) are sharded on the HEAD dimension across a ROW of 4
    cores (each core stores heads/4) -- one identical policy for the query side
    and the output.  At the start of each batch the row all-gathers the head
    slices so every core holds the full query (all heads).
  * The latent KV cache (KV, K_pe), which has a single head, is split across the
    ROW on the SEQUENCE axis (split-K): each core owns seqlen_kv/4.

So each core, after the head gather, runs a full-heads MLA attention over its
own 1/4 of the KV sequence, producing a per-head partial output and log-sum-exp.
The four seqlen partials are then merged across the row with an LSE combine, and
-- since Output is head-sharded the same way as Q (the reverse of the input
gather) -- each column writes back only the head blocks it owns.

MLA specifics: kv_head_num == 1 (all query heads share one latent KV).  Score is
  S = Q.KV^T + Q_pe.K_pe^T  (two GEMMs over dim and pe_dim); value projection is
  O = softmax(S).KV  (output dimension is the latent dim).

Lower-and-legalize example (Sunmmio codegen is still WIP).
"""

import argparse
from typing import Callable

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.engine.phase import LowerAndLegalize
from tilelang.utils.target import determine_target
from tilelang.layout import make_zz_layout


def mla_decode(batch, heads, kv_heads, seqlen_kv, dim, pe_dim, block_N=64, block_H=16) -> "Callable":
    ncols = T.ncols()

    assert kv_heads == 1, "MLA shares a single latent KV across all query heads"
    assert heads % block_H == 0, "heads must be divisible by block_H"
    scale = (1.0 / (dim + pe_dim)) ** 0.5 * 1.44269504  # log2(e)

    shape_q = [batch, heads, dim]
    shape_qpe = [batch, heads, pe_dim]
    shape_kv = [batch, seqlen_kv, kv_heads, dim]
    shape_kpe = [batch, seqlen_kv, kv_heads, pe_dim]
    shape_o = [batch, heads, dim]
    dtype = T.float16
    accum_dtype = T.float32

    heads_per_col = heads // ncols

    # Q, Q_pe and Output all share the same policy: batch on the rows (y=0), the
    # head axis sharded across the columns (x=1). KV/K_pe split the sequence
    # axis across the columns (x=1) -> split-K.
    head_policy = T.placement.full_shard(0, 1)
    kv_policy = T.placement.full_shard(0, 1)

    @T.prim_func
    def main(
        Q: T.MeshTensor(shape_q, head_policy, dtype, layout=make_zz_layout(shape_q)),
        Q_pe: T.MeshTensor(shape_qpe, head_policy, dtype, layout=make_zz_layout(shape_qpe)),
        KV: T.MeshTensor(shape_kv, kv_policy, dtype, layout=make_zz_layout(shape_kv, axes=(1, 3))),
        K_pe: T.MeshTensor(shape_kpe, kv_policy, dtype, layout=make_zz_layout(shape_kpe, axes=(1, 3))),
        Output: T.MeshTensor(shape_o, head_policy, dtype, layout=make_zz_layout(shape_o)),
    ):
        with T.Kernel() as (cid):
            sharded_batch = Q.local_shape[0]
            _, sharded_seqlen, _, _ = KV.local_shape
            # This core's column owns the global head blocks
            # [col*blocks_per_col, (col+1)*blocks_per_col).
            col = cid % T.ncols()
            blocks_per_col = heads_per_col // block_H

            # Head gather (per batch): load this core's head slice, all-gather
            # across the row into the full head set.
            Q_local = T.alloc_shared([heads_per_col, dim], dtype)
            Q_pe_local = T.alloc_shared([heads_per_col, pe_dim], dtype)
            Q_full = T.alloc_shared([heads, dim], dtype)
            Q_pe_full = T.alloc_shared([heads, pe_dim], dtype)

            # Per head-block operands / accumulators.
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
            # The single latent KV feeds two GEMMs with different contraction
            # axes (QK: transposed B over dim; PV: plain B over block_N), which
            # need different WSRAM operand layouts, so it is staged twice.
            KV_shared = T.alloc_shared([block_N, dim], dtype)
            KV_v_shared = T.alloc_shared([block_N, dim], dtype)
            K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
            acc_s = T.alloc_shared([block_H, block_N], accum_dtype)
            acc_s_cast = T.alloc_shared([block_H, block_N], dtype)
            acc_o = T.alloc_shared([block_H, dim], accum_dtype)
            scores_max = T.alloc_shared([block_H], accum_dtype)
            scores_max_prev = T.alloc_shared([block_H], accum_dtype)
            scores_scale = T.alloc_shared([block_H], accum_dtype)
            scores_sum = T.alloc_shared([block_H], accum_dtype)
            logsum = T.alloc_shared([block_H], accum_dtype)
            lse = T.alloc_shared([block_H], accum_dtype)

            # Cross-row LSE-combine scratch.
            lse_dist = T.alloc_shared([T.ncols(), block_H], accum_dtype)
            lse_max = T.alloc_shared([block_H], accum_dtype)
            lse_denom = T.alloc_shared([block_H], accum_dtype)
            o_scaled = T.alloc_shared([block_H, dim], accum_dtype)
            o_dist = T.alloc_shared([T.ncols(), block_H, dim], accum_dtype)
            o_final = T.alloc_shared([block_H, dim], accum_dtype)
            o_cast = T.alloc_shared([block_H, dim], dtype)

            for bid in T.serial(sharded_batch):
                # --- Gather the full head set across the row. ---
                T.copy(Q[bid, :, :], Q_local)
                T.comm.all_gather(Q_local, Q_full, direction="h", axis=0)
                T.copy(Q_pe[bid, :, :], Q_pe_local)
                T.comm.all_gather(Q_pe_local, Q_pe_full, direction="h", axis=0)

                for hid in T.serial(heads // block_H):
                    T.copy(Q_full[hid * block_H : (hid + 1) * block_H, :], Q_shared)
                    T.copy(Q_pe_full[hid * block_H : (hid + 1) * block_H, :], Q_pe_shared)
                    T.fill(acc_o, 0)
                    T.fill(logsum, 0)
                    T.fill(scores_max, -T.infinity(accum_dtype))

                    # --- Local MLA attention over this core's seqlen chunk. ---
                    for k in T.serial(T.ceildiv(sharded_seqlen, block_N)):
                        T.copy(KV[bid, k * block_N : (k + 1) * block_N, 0, :], KV_shared)
                        T.copy(KV[bid, k * block_N : (k + 1) * block_N, 0, :], KV_v_shared)
                        T.copy(K_pe[bid, k * block_N : (k + 1) * block_N, 0, :], K_pe_shared)
                        # S = Q.KV^T + Q_pe.K_pe^T
                        T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, clear_accum=True)
                        T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True)
                        # online softmax
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
                        # O += softmax(S) . KV
                        T.gemm(acc_s_cast, KV_v_shared, acc_o)

                    # Normalize the local partial and compute its log-sum-exp.
                    for i, j in T.Tiles([block_H, dim]):
                        acc_o[i, j] = acc_o[i, j] / logsum[i]
                    for i in T.Tiles([block_H]):
                        lse[i] = T.log2(logsum[i]) + scores_max[i] * scale

                    # --- LSE combine across the row (the seqlen-split axis). ---
                    T.comm.all_gather(lse, lse_dist, direction="h")
                    T.reduce_max(lse_dist, lse_max, dim=0, clear=True)
                    for c, i in T.Tiles([T.ncols(), block_H]):
                        lse_dist[c, i] = T.exp2(lse_dist[c, i] - lse_max[i])
                    T.reduce_sum(lse_dist, lse_denom, dim=0, clear=True)
                    for i, j in T.Tiles([block_H, dim]):
                        o_scaled[i, j] = acc_o[i, j] * T.exp2(lse[i] - lse_max[i]) / lse_denom[i]
                    T.comm.all_gather(o_scaled, o_dist, direction="h")
                    T.reduce_sum(o_dist, o_final, dim=0, clear=True)

                    T.copy(o_final, o_cast)
                    # Output is head-sharded like Q (the reverse of the input
                    # head-gather): every core combined the full head set, but
                    # only the owning column writes each global head block into
                    # its local head slice.
                    if (col * blocks_per_col <= hid) and (hid < (col + 1) * blocks_per_col):
                        lhid = hid - col * blocks_per_col
                        T.copy(o_cast, Output[bid, lhid * block_H : (lhid + 1) * block_H, :])

    return main


def main(batch, heads, kv_heads, kv_seqlen, dim, pe_dim) -> None:
    target = determine_target("Sunmmio", return_object=True)

    pass_configs = {tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True}
    with tvm.target.Target(target), tvm.transform.PassContext(config=pass_configs):
        kernel = mla_decode(batch, heads, kv_heads, kv_seqlen, dim, pe_dim)
        mod = LowerAndLegalize(tvm.IRModule({"main": kernel}), target)
        print(mod)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--heads", type=int, default=128, help="q heads")
    parser.add_argument("--kv_heads", type=int, default=1, help="kv heads (MLA: 1)")
    parser.add_argument("--kv_seqlen", type=int, default=8192, help="kv sequence length")
    parser.add_argument("--dim", type=int, default=512, help="latent head dim")
    parser.add_argument("--pe_dim", type=int, default=64, help="pe head dim")
    args = parser.parse_args()
    main(args.batch, args.heads, args.kv_heads, args.kv_seqlen, args.dim, args.pe_dim)
