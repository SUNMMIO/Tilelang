"""
Reference implementations showing how to rewrite the four core SunMMIO ILP
kernels with the newer TileLang-Mesh style.

This file is documentation-oriented:
- It is intentionally separate from the existing tests.
- It focuses on the three decisions that matter most:
  1. Mesh sharding policy
  2. Global ZZ layout choice
  3. MeshTensor parameter declaration

Important note
--------------
The user guide describes a newer API shape:

    A: T.MeshTensor((M, K), shard_policy, dtype, layout=A_layout)
    with T.Kernel() as cid:
        sharded_M, sharded_K = A.local_shape
        valid_M, valid_K = A.get_local_extent(cid)

At the time of writing, parts of that API are still ahead of the Python DSL
implementation in this local checkout. So the examples below do two things:

- They follow the *design intent* of the new Mesh style:
  explicit `MeshShardingPolicy`, explicit `make_zz_layout`, explicit
  `T.MeshTensor(...)`.
- They also stay close to constructs that already exist in the repository, so
  the examples remain grounded in the current codebase.

Read this file as "how these kernels should be structured", not as an ABI
contract for the exact parser behavior of the current checkout.
"""

import tilelang.language as T
from tilelang.language.mesh_tensor import MeshReplicationType
from tilelang.layout import make_zz_layout, make_row_major
from testing.python.sunmmio.common.compile_pipeline import target


@target("Sunmmio")
def mesh_matmul_new(
    M,
    N,
    K,
    block_M=128,
    block_N=128,
    block_K=32,
    num_stages=2,
    dtype="bfloat16",
    accum_dtype="float",
):
    # 1) Sharding policy
    # GEMM matrices are the easiest case:
    # - row mesh dimension shards tensor dim 0
    # - col mesh dimension shards tensor dim 1
    # Use the same effective partitioning as the executable strict test:
    # A is row-sharded on M and replicated across mesh rows for K traversal,
    # B is col-sharded on N and replicated across mesh cols for K traversal,
    # C is sharded on both output axes.
    a_policy = T.MeshShardingPolicy(y=0, replicate=MeshReplicationType.ROW)
    b_policy = T.MeshShardingPolicy(x=1, replicate=MeshReplicationType.COLUMN)
    c_policy = T.MeshShardingPolicy(y=0, x=1)

    # 2) Layout
    # For GEMM, the compute-critical dimensions are exactly the matrix axes.
    # So we block both dimensions with a 32x32 ZZ layout.
    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        # 3) MeshTensor declaration
        # New-style intent:
        #   T.MeshTensor(shape, shard_policy, dtype, layout=...)
        # This says:
        # - logical global shape is `(M, K)`
        # - sharding is controlled by `shard_policy`
        # - layout is explicit
        A: T.MeshTensor((M, K), a_policy, (4, 4), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), b_policy, (4, 4), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), c_policy, (4, 4), accum_dtype, layout=C_layout),
    ):
        with T.Kernel() as _cid:
            sharded_M, sharded_K = A.local_shape
            _, sharded_N = B.local_shape

            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            for bx in T.serial(T.ceildiv(sharded_M, block_M)):
                for by in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.clear(C_shared)
                    for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=num_stages):
                        T.copy(
                            A[
                                bx * block_M : (bx + 1) * block_M,
                                k * block_K : (k + 1) * block_K,
                            ],
                            A_shared,
                        )
                        T.copy(
                            B[
                                k * block_K : (k + 1) * block_K,
                                by * block_N : (by + 1) * block_N,
                            ],
                            B_shared,
                        )
                        T.gemm(A_shared, B_shared, C_shared)
                    T.copy(C_shared, C[bx * block_M, by * block_N])

    return main


@target("Sunmmio")
def mesh_flashattn_new(
    batch=1,
    heads=64,
    seq_len=4096,
    dim=128,
    groups=16,
    is_causal=False,
    block_M=64,
    block_N=64,
    num_stages=2,
    dtype=T.bfloat16,
    accum_dtype=T.bfloat16,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]

    # 1) Sharding policy
    # Attention tensors are not matrices. The usual choice is:
    # - shard batch on mesh rows
    # - shard heads / kv-heads on mesh cols
    q_policy = T.MeshShardingPolicy(y=0, x=2)
    kv_policy = T.MeshShardingPolicy(y=0, x=2)

    # 2) Layout
    # For Q/K/V/O the compute-critical axes are typically sequence and dim.
    Q_layout = make_zz_layout(q_shape, [1, 3], (32, 32))
    K_layout = make_zz_layout(kv_shape, [1, 3], (32, 32))
    V_layout = make_zz_layout(kv_shape, [1, 3], (32, 32))
    O_layout = make_zz_layout(q_shape, [1, 3], (32, 32))

    @T.prim_func
    def main(
        Q: T.MeshTensor(q_shape, q_policy, (4, 4), dtype, layout=Q_layout),
        K: T.MeshTensor(kv_shape, kv_policy, (4, 4), dtype, layout=K_layout),
        V: T.MeshTensor(kv_shape, kv_policy, (4, 4), dtype, layout=V_layout),
        Output: T.MeshTensor(q_shape, q_policy, (4, 4), dtype, layout=O_layout),
    ):
        with T.Kernel() as _cid:
            sharded_batch = Q.local_shape[0]
            sharded_heads = Q.local_shape[2]
            local_q_tiles = T.ceildiv(Q.local_shape[1], block_M)

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

            for bz in T.serial(sharded_batch):
                for by in T.serial(sharded_heads):
                    for bx in T.serial(local_q_tiles):
                        T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
                        T.fill(acc_o, 0)
                        T.fill(logsum, 0)
                        T.fill(scores_max, -T.infinity(accum_dtype))

                        loop_range = (
                            T.min(T.ceildiv(K.local_shape[1], block_N), T.ceildiv((bx + 1) * block_M, block_N))
                            if is_causal
                            else T.ceildiv(K.local_shape[1], block_N)
                        )

                        for k in T.Pipelined(loop_range, num_stages=num_stages):
                            T.copy(K[bz, k * block_N : (k + 1) * block_N, by // groups, :], K_shared)
                            if is_causal:
                                for i, j in T.Tiles(acc_s, parallel=True):
                                    acc_s[i, j] = T.if_then_else(
                                        bx * block_M + i >= k * block_N + j,
                                        0,
                                        -T.infinity(acc_s.dtype),
                                    )
                            else:
                                for i, j in T.Tiles(acc_s, parallel=True):
                                    acc_s[i, j] = T.if_then_else(
                                        k * block_N + j >= seq_len,
                                        -T.infinity(acc_s.dtype),
                                        0,
                                    )

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


@target("Sunmmio")
def mesh_flashdecoding_new(
    batch=1,
    heads=256,
    kv_heads=8,
    seqlen_kv=8192,
    dim=128,
    block_N=128,
    block_H=64,
    num_split=1,
    num_stages=2,
    dtype=T.bfloat16,
    accum_dtype=T.bfloat16,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    shape_q = [batch, heads, dim]
    shape_k = [batch, seqlen_kv, kv_heads, dim]
    shape_v = [batch, seqlen_kv, kv_heads, dim]
    shape_o = [batch, heads, dim]
    kv_group_num = heads // kv_heads
    assert heads % kv_heads == 0, "GQA requires kv_heads to divide heads"

    @T.prim_func
    def main(
        Q: T.MeshTensor(shape_q, T.MeshShardingPolicy(y=0, x=1), dtype, layout=make_zz_layout(shape_q)),
        K: T.MeshTensor(shape_k, T.MeshShardingPolicy(y=0, x=2), dtype, layout=make_zz_layout(shape_k, axes=(1, 3))),
        V: T.MeshTensor(shape_v, T.MeshShardingPolicy(y=0, x=2), dtype, layout=make_zz_layout(shape_k, axes=(1, 3))),
        mask: T.MeshTensor(
            [batch, seqlen_kv],
            T.MeshShardingPolicy(y=0, replicate=T.MeshReplicationType.ROW),
            dtype,
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
            mask_local = T.alloc_shared([block_N], dtype)
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
                    for k in T.Pipelined(loop_range, num_stages=num_stages):
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


@target("Sunmmio")
def mesh_flashmladecode_new(
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
    num_stages=2,
    dtype=T.bfloat16,
    accum_dtype=T.bfloat16,
):
    scale = float(softmax_scale * 1.44269504)
    kv_group_num = heads // kv_head_num
    valid_block_H = min(block_H, kv_group_num)

    q_shape = [batch, heads, dim]
    qpe_shape = [batch, heads, pe_dim]
    kv_shape = [batch, seqlen_kv, kv_head_num, dim]
    kpe_shape = [batch, seqlen_kv, kv_head_num, pe_dim]
    glse_shape = [batch, heads, num_split]
    part_shape = [batch, heads, num_split, dim]
    out_shape = [batch, heads, dim]

    # 1) Sharding policy
    # Q / Q_pe / Output are head-oriented tensors.
    # KV / K_pe are kv-stream tensors.
    q_policy = T.MeshShardingPolicy(y=0, x=1)
    kv_policy = T.MeshShardingPolicy(y=0, x=2)

    # 2) Layout
    # Q and Q_pe use head-oriented layouts.
    # KV and K_pe use seq + dim style layouts.
    Q_layout = make_zz_layout(q_shape, [1, 2], (32, 32))
    Qpe_layout = make_zz_layout(qpe_shape, [1, 2], (32, 32))
    KV_layout = make_zz_layout(kv_shape, [1, 3], (32, 32))
    Kpe_layout = make_zz_layout(kpe_shape, [1, 3], (32, 32))
    O_layout = make_zz_layout(out_shape, [1, 2], (32, 32))

    @T.prim_func
    def main(
        Q: T.MeshTensor(q_shape, q_policy, (4, 4), dtype, layout=Q_layout),
        Q_pe: T.MeshTensor(qpe_shape, q_policy, (4, 4), dtype, layout=Qpe_layout),
        KV: T.MeshTensor(kv_shape, kv_policy, (4, 4), dtype, layout=KV_layout),
        K_pe: T.MeshTensor(kpe_shape, kv_policy, (4, 4), dtype, layout=Kpe_layout),
        glse: T.MeshTensor(glse_shape, q_policy, (4, 4), dtype),
        Output_partial: T.MeshTensor(part_shape, q_policy, (4, 4), dtype),
        Output: T.MeshTensor(out_shape, q_policy, (4, 4), dtype, layout=O_layout),
    ):
        with T.Kernel() as _cid:
            # This kernel has two score-producing paths:
            #   Q_shared   @ KV_shared^T
            #   Q_pe_shared @ K_pe_shared^T
            #
            # That is why the buffer roles matter more here than in plain flashattn.
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
            KV_shared = T.alloc_shared([block_N, dim], dtype)
            KV_shared2 = T.alloc_shared([block_N, dim], dtype)
            K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            O_shared = T.alloc_shared([block_H, dim], dtype)
            acc_s = T.alloc_shared([block_H, block_N], accum_dtype)
            acc_o = T.alloc_shared([block_H, dim], accum_dtype)
            scores_max = T.alloc_shared([block_H], accum_dtype)
            scores_max_prev = T.alloc_shared([block_H], accum_dtype)
            scores_scale = T.alloc_shared([block_H], accum_dtype)
            scores_sum = T.alloc_shared([block_H], accum_dtype)
            logsum = T.alloc_shared([block_H], accum_dtype)

            sharded_batch = Q.local_shape[0]
            sharded_heads = Q.local_shape[1]

            for bid in T.serial(sharded_batch):
                for hid in T.serial(T.ceildiv(sharded_heads, valid_block_H)):
                    for sid in T.serial(num_split):
                        cur_kv_head = hid // (kv_group_num // block_H)

                        T.copy(Q[bid, hid * valid_block_H : (hid + 1) * valid_block_H, :], Q_shared)
                        T.copy(Q_pe[bid, hid * valid_block_H : (hid + 1) * valid_block_H, :], Q_pe_shared)
                        T.fill(acc_o, 0)
                        T.fill(logsum, 0)
                        T.fill(scores_max, -T.infinity(accum_dtype))

                        loop_range = T.ceildiv((seqlen_kv // num_split), block_N)
                        for k in T.Pipelined(loop_range, num_stages=num_stages):
                            kv_start = (seqlen_kv // num_split) * sid + k * block_N
                            kv_end = (seqlen_kv // num_split) * sid + (k + 1) * block_N
                            T.copy(KV[bid, kv_start:kv_end, cur_kv_head, :], KV_shared)
                            T.copy(KV[bid, kv_start:kv_end, cur_kv_head, :], KV_shared2)
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
                            for i in T.Tiles(logsum, parallel=True):
                                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                            for i, j in T.Tiles(acc_o, parallel=True):
                                acc_o[i, j] *= scores_scale[i]
                            T.gemm(S_shared, KV_shared2, acc_o, policy=T.GemmWarpPolicy.FullCol)

                        for i, j in T.Tiles(acc_o, parallel=True):
                            acc_o[i, j] /= logsum[i]
                        for i in T.Tiles(logsum, parallel=True):
                            logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                        T.copy(logsum, glse[bid, hid * valid_block_H : (hid + 1) * valid_block_H, sid])
                        T.copy(acc_o, O_shared)
                        T.copy(O_shared, Output_partial[bid, hid * valid_block_H : (hid + 1) * valid_block_H, sid, :])

    return main
