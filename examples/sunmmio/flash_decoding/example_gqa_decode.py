import argparse

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.engine.phase import LowerAndLegalize
from tilelang.utils.target import determine_target
from tilelang.layout import make_zz_layout, make_row_major


def gqa_flashattn(batch, heads, kv_heads, seqlen_kv, dim, block_N=128):
    # Device configurations

    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape_q = [batch, heads, dim]
    shape_k = [batch, seqlen_kv, kv_heads, dim]
    shape_v = [batch, seqlen_kv, kv_heads, dim]
    shape_o = [batch, heads, dim]
    dtype = T.float16
    accum_dtype = T.float32
    kv_group_num = heads // kv_heads
    assert heads % kv_heads == 0, "GQA requires kv_heads is the multiple of heads"

    block_H = kv_group_num

    @T.prim_func
    def main(
        Q: T.MeshTensor(shape_q, T.MeshShardingPolicy(y=0, x=1), dtype, layout=make_zz_layout(shape_q)),
        K: T.MeshTensor(shape_k, T.MeshShardingPolicy(y=0, x=2), dtype, layout=make_zz_layout(shape_k, axes=(1, 3))),
        V: T.MeshTensor(shape_v, T.MeshShardingPolicy(y=0, x=2), dtype, layout=make_zz_layout(shape_k, axes=(1, 3))),
        mask: T.MeshTensor(
            [batch, seqlen_kv],
            T.MeshShardingPolicy(y=0, replicate=T.MeshReplicationType.ROW),
            "uint16",
            layout=make_row_major([batch, seqlen_kv]),
        ),
        Output: T.MeshTensor(shape_o, T.MeshShardingPolicy(y=0, x=1), dtype, layout=make_zz_layout(shape_o)),
    ):
        with T.Kernel() as (_cid):
            sharded_batch, sharded_heads, _ = Q.local_shape

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
                        T.gemm(Q_shared, K_shared, acc_s, clear_accum=True, transpose_B=True)  # Not accmulate
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


def main(batch, heads, kv_heads, kv_seqlen, dim) -> None:
    target = determine_target("Sunmmio", return_object=True)

    pass_configs = {tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True}
    with tvm.target.Target(target), tvm.transform.PassContext(config=pass_configs):
        kernel = gqa_flashattn(batch, heads, kv_heads, kv_seqlen, dim)
        mod = LowerAndLegalize(tvm.IRModule({"main": kernel}), target)
        print(mod)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--heads", type=int, default=64, help="heads")
    parser.add_argument("--kv_heads", type=int, default=8, help="groups")
    parser.add_argument("--kv_seqlen", type=int, default=8192, help="kv sequence length")
    parser.add_argument("--dim", type=int, default=128, help="dim")
    args = parser.parse_args()
    main(args.batch, args.heads, args.kv_heads, args.kv_seqlen, args.dim)
