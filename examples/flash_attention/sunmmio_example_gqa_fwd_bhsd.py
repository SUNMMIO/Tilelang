import tilelang.language as T
from tilelang.layout import make_zz_layout
from tilelang.carver.arch import driver


def flashattn(batch, heads, seq_len, dim, groups=1, block_M=64, block_N=64, num_stages=0):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = T.float16
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
        Q: T.MeshTensor(q_shape, shard_policy, device_mesh_config, dtype, layout=Q_layout),
        K: T.MeshTensor(kv_shape, shard_policy, device_mesh_config, dtype, layout=K_layout),
        V: T.MeshTensor(kv_shape, shard_policy, device_mesh_config, dtype, layout=V_layout),
        Output: T.MeshTensor(q_shape, shard_policy, device_mesh_config, dtype, layout=O_layout),
    ):
        with T.Kernel(ncores) as _cid:
            # Get sharded logical shapes
            sharded_batch = Q.shape[0]
            sharded_heads = Q.shape[2]

            # Declare shared memory buffers
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

            # Persistent loop
            for bz, by, bx in T.Persistent(sharded_batch, sharded_heads, T.ceildiv(seq_len, block_M)):
                T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N))

                # loop over K/V blocks
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(K[bz, k * block_N : (k + 1) * block_N, by // groups, :], K_shared)
                    for i, j in T.Tiles([block_M, block_N]):
                        acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype))
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Tiles(block_M):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    for i in T.Tiles(block_M):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Tiles([block_M, block_N]):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Tiles([block_M]):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)

                    for i, j in T.Tiles([block_M, dim]):
                        acc_o[i, j] *= scores_scale[i]

                    T.copy(V[bz, k * block_N : (k + 1) * block_N, by // groups, :], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o)

                for i, j in T.Tiles([block_M, dim]):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main
