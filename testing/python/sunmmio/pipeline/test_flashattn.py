import os
import tilelang.language as T
from tilelang.layout import make_zz_layout

from testing.python.sunmmio.common.compile_pipeline import compile_test, target
from testing.python.sunmmio.common.formal_verify import *


@target("Sunmmio")
def kernel_flashattn(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    block_M=64,
    block_N=64,
    num_stages=1,
    threads=1,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, heads, dim]
    # Different precisions will cause different number of allocates. The default allocate is allocated according to uint8, so when the data type is bfloat16, the number of allocates will be doubled.
    dtype = T.bfloat16
    # accum_dtype = T.float32
    accum_dtype = T.bfloat16
    shard_policy = T.MeshShardingPolicy(y=0, x=2)

    Q_layout = make_zz_layout(q_shape, [1, 3], (32, 32))
    K_layout = make_zz_layout(kv_shape, [1, 3], (32, 32))
    V_layout = make_zz_layout(kv_shape, [1, 3], (32, 32))
    O_layout = make_zz_layout(q_shape, [1, 3], (32, 32))

    @T.prim_func
    def main(
        Q: T.MeshTensor(q_shape, shard_policy, dtype, layout=Q_layout),
        K: T.MeshTensor(kv_shape, shard_policy, dtype, layout=K_layout),
        V: T.MeshTensor(kv_shape, shard_policy, dtype, layout=V_layout),
        Output: T.MeshTensor(q_shape, shard_policy, dtype, layout=O_layout),
    ):
        with T.Kernel() as _cid:
            sharded_batch = Q.local_shape[0]
            sharded_heads = Q.local_shape[2]

            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_shared([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_shared([block_M, block_N], accum_dtype, scope="shared.asram")
            acc_o = T.alloc_shared([block_M, dim], accum_dtype, scope="shared.rsram")
            scores_max = T.alloc_shared([block_M], accum_dtype)
            scores_max_prev = T.alloc_shared([block_M], accum_dtype)
            scores_scale = T.alloc_shared([block_M], accum_dtype)
            scores_sum = T.alloc_shared([block_M], accum_dtype)
            logsum = T.alloc_shared([block_M], accum_dtype)

            q_tiles = T.ceildiv(seq_len, block_M)
            for bz in T.serial(sharded_batch):
                for by in T.serial(sharded_heads):
                    for bx in T.serial(q_tiles):
                        T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
                        T.fill(acc_o, 0)
                        T.fill(logsum, 0)
                        T.fill(scores_max, -T.infinity(accum_dtype))

                        loop_range = (
                            T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N))
                            if is_causal
                            else T.ceildiv(seq_len, block_N)
                        )

                        for k in T.Pipelined(loop_range, num_stages=num_stages):
                            T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)
                            if is_causal:
                                for i, j in T.Tiles([block_M, block_N]):
                                    acc_s[i, j] = T.if_then_else(
                                        bx * block_M + i >= k * block_N + j,
                                        0,
                                        -T.infinity(acc_s.dtype),
                                    )
                                # T.fill(acc_s, T.if_then_else(bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype)))
                            else:
                                for i, j in T.Tiles([block_M, block_N]):
                                    acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_len, -T.infinity(acc_s.dtype), 0)

                            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)

                            for i in T.Tiles([block_M]):
                                scores_max_prev[i] = scores_max[i]
                            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                            for i in T.Tiles([block_M]):
                                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                            for i in T.Tiles([block_M]):
                                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

                            for i, j in T.Tiles([block_M, block_N]):
                                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                            T.reduce_sum(acc_s, scores_sum, dim=1, clear=True)

                            for i in T.Tiles([block_M]):
                                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                            T.copy(acc_s, acc_s_cast)

                            for i, j in T.Tiles([block_M, dim]):
                                acc_o[i, j] *= scores_scale[i]

                            T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                            T.gemm(acc_s_cast, V_shared, acc_o)

                        for i, j in T.Tiles([block_M, dim]):
                            acc_o[i, j] /= logsum[i]
                        T.copy(acc_o, O_shared)
                        T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main


def test_flashattn(is_log=False):
    func = kernel_flashattn(8, 32, 4096, 128, False, block_M=128, block_N=128, num_stages=1, threads=1)

    script_device_mode = """
        with T.launch_thread("blockIdx.x", 16) as bx:
            with T.decl_buffer((128, 128), "bfloat16", data=Q_shared.data, scope="shared.asram") as Q_shared:
                K_shared = T.decl_buffer((1, 128, 128), "bfloat16", data=K_shared.data, scope="shared.wsram")
                V_shared = T.decl_buffer((1, 128, 128), "bfloat16", data=V_shared.data, scope="shared.wsram")
                O_shared = T.decl_buffer((128, 128), "bfloat16", data=O_shared.data, scope="shared.rsram")
                acc_s = T.decl_buffer((1, 128, 128), "bfloat16", data=acc_s.data, scope="shared.rsram")
                acc_s_cast = T.decl_buffer((1, 128, 128), "bfloat16", data=acc_s_cast.data, scope="shared.asram")
                acc_o = T.decl_buffer((128, 128), "bfloat16", data=acc_o.data, scope="shared.rsram")
                scores_max = T.decl_buffer((128,), "bfloat16", data=scores_max.data, scope="shared.rsram")
                scores_max_prev = T.decl_buffer((128,), "bfloat16", data=scores_max_prev.data, scope="shared.rsram")
                scores_scale = T.decl_buffer((128,), "bfloat16", data=scores_scale.data, scope="shared.rsram")
                scores_sum = T.decl_buffer((128,), "bfloat16", data=scores_sum.data, scope="shared.rsram")
                logsum = T.decl_buffer((128,), "bfloat16", data=logsum.data, scope="shared.rsram")
                Q_rsram_stage = T.decl_buffer((1, 128, 1, 128), "bfloat16", data=Q_rsram_stage.data, scope="shared.rsram")
                Q_layout_stage = T.decl_buffer((1, 128, 1, 128), "bfloat16", data=Q_layout_stage.data, scope="shared.rsram")
                Output_layout_stage = T.decl_buffer((128, 128), "bfloat16", data=Output_layout_stage.data, scope="shared.rsram")
                T.sync_null_token(21)
                T.sync_null_token(23)
                for bz in range(2):
                    T.sync_null_token(21)
                    T.sync_null_token(23)
                    for by in range(8):
                        T.sync_null_token(21)
                        T.sync_null_token(23)
                        for bx_1 in range(32):
                            T.dma_copy(T.region(Q_1[bz, bx_1 * 128, by, 0], 1, 1, 128, 1, 128), T.region(Q_layout_stage[0, 0, 0, 0], 2, 1, 128, 1, 128), 0, T.sync_token_id(0))
                            T.wait_token(0)
                            T.sunmmio_layout_transform(T.region(Q_layout_stage[0, 0, 0, 0], 1, 1, 128, 1, 128), T.region(Q_rsram_stage[0, 0, 0, 0], 2, 1, 128, 1, 128), T.sync_token_id(1))
                            for i0 in T.serial(16, annotations={"tile.domain": [128, 128], "tile.execution_axis": 0, "tile.execution_domain_axes": [0, 1], "tile.scope_entry": 1, "tile.tile_size": [8, 32]}):
                                for i1 in T.serial(4, annotations={"tile.execution_axis": 1}):
                                    for ki in T.serial(8, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                        for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                            acc_o[i0 * 8 + ki, i1 * 32 + kj] = T.bfloat16(0.0)
                            for i0 in T.serial(1, annotations={"tile.domain": [128], "tile.execution_axis": 0, "tile.execution_domain_axes": [0], "tile.scope_entry": 1, "tile.tile_size": [128]}):
                                for ki in T.vectorized(128, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                    logsum[ki] = T.bfloat16(0.0)
                            for i0 in T.serial(1, annotations={"tile.domain": [128], "tile.execution_axis": 0, "tile.execution_domain_axes": [0], "tile.scope_entry": 1, "tile.tile_size": [128]}):
                                for ki in T.vectorized(128, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                    scores_max[ki] = T.infinity("bfloat16") * T.bfloat16(-1.0)
                            T.dma_copy(T.region(K_1[bz, 0, by, 0], 1, 1, 128, 1, 128), T.region(K_shared[0, 0, 0], 2, 1, 128, 128), 0, T.sync_token_id(2))
                            for i in T.serial(16, annotations={"tile.domain": [128, 128], "tile.execution_axis": 0, "tile.execution_domain_axes": [0, 1], "tile.scope_entry": 1, "tile.tile_size": [8, 32]}):
                                for j in T.serial(4, annotations={"tile.execution_axis": 1}):
                                    for ki in T.serial(8, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                        for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                            acc_s[0, i * 8 + ki, j * 32 + kj] = T.bfloat16(0.0)
                            T.wait_token(21)
                            T.dma_copy(T.region(V_1[bz, 0, by, 0], 1, 1, 128, 1, 128), T.region(V_shared[0, 0, 0], 2, 1, 128, 128), 0, T.sync_token_id(3))
                            T.sync_null_token(8)
                            T.sync_null_token(13)
                            T.wait_token(1)
                            T.wait_token(2)
                            T.wait_token(3)
                            for k in range(31):
                                T.dma_copy(T.region(Q_rsram_stage[0, 0, 0, 0], 1, 1, 128, 1, 128), T.region(Q_shared[0, 0], 2, 128, 128), 0, T.sync_token_id(4))
                                for i in T.serial(1, annotations={"tile.domain": [128], "tile.execution_axis": 0, "tile.execution_domain_axes": [0], "tile.scope_entry": 1, "tile.tile_size": [128]}):
                                    for ki in T.vectorized(128, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                        scores_max_prev[ki] = scores_max[ki]
                                T.wait_token(8)
                                T.wait_token(4)
                                T.mma_sunmmio(T.region(Q_shared[0, 0], 1, 128, 128), T.region(K_shared[0, 0, 0], 1, 1, 128, 128), T.region(acc_s[0, 0, 0], 3, 1, 128, 128), T.bool(False), T.bool(True), T.bool(False), 0, T.sync_token_id(5))
                                T.wait_token(5)
                                T.dma_copy(T.region(Q_rsram_stage[0, 0, 0, 0], 1, 1, 128, 1, 128), T.region(Q_shared[0, 0], 2, 128, 128), 1024, T.sync_token_id(6))
                                T.wait_token(6)
                                T.mma_sunmmio(T.region(Q_shared[0, 0], 1, 128, 128), T.region(K_shared[0, 0, 0], 1, 1, 128, 128), T.region(acc_s[0, 0, 0], 3, 1, 128, 128), T.bool(False), T.bool(True), T.bool(False), 1024, T.sync_token_id(7))
                                with T.decl_buffer((8, 32), "bfloat16", scope="shared.rsram") as scores_max_acc:
                                    scores_max_res = T.decl_buffer((8,), "bfloat16", scope="shared.rsram")
                                    T.wait_token(7)
                                    for i0 in T.serial(16, annotations={"tile.domain": [128, 128], "tile.execution_axis": 0, "tile.execution_domain_axes": [0, 1], "tile.scope_entry": 1, "tile.tile_size": [8, 32]}):
                                        for i1 in T.serial(4, annotations={"tile.execution_axis": 1}):
                                            if i1 == 0:
                                                for ki in T.serial(8, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                                    for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                                        scores_max_acc[ki, kj] = T.bfloat16("-inf")
                                            for ki in T.serial(8, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                                for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                                    scores_max_acc[ki, kj] = T.max(scores_max_acc[ki, kj], acc_s[0, i0 * 8 + ki, i1 * 32 + kj])
                                            if i1 == 3:
                                                T.vector_core_in_tile_reduce("max", T.region(scores_max_res[0], 1, 8), T.region(scores_max_acc[0, 0], 1, 8, 32), 1)
                                                for ki in T.vectorized(8, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                                    scores_max[i0 * 8 + ki] = T.max(scores_max[i0 * 8 + ki], scores_max_res[ki])
                                T.dma_copy(T.region(K_1[bz, k * 128 + 128, by, 0], 1, 1, 128, 1, 128), T.region(K_shared[0, 0, 0], 2, 1, 128, 128), 0, T.sync_token_id(8))
                                for i in T.serial(1, annotations={"tile.domain": [128], "tile.execution_axis": 0, "tile.execution_domain_axes": [0], "tile.scope_entry": 1, "tile.tile_size": [128]}):
                                    for ki in T.vectorized(128, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                        scores_max[ki] = T.max(scores_max[ki], scores_max_prev[ki])
                                    for ki in T.vectorized(128, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                        scores_scale[ki] = T.Cast("bfloat16", T.exp2(T.Cast("float32", scores_max_prev[ki]) * T.float32(0.1275174307460247) - T.Cast("float32", scores_max[ki]) * T.float32(0.1275174307460247)))
                                for i in T.serial(16, annotations={"tile.domain": [128, 128], "tile.execution_axis": 0, "tile.execution_domain_axes": [0, 1], "tile.scope_entry": 1, "tile.tile_size": [8, 32]}):
                                    for j in T.serial(4, annotations={"tile.execution_axis": 1}):
                                        for ki in T.serial(8, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                            for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                                acc_s[0, i * 8 + ki, j * 32 + kj] = T.Cast("bfloat16", T.exp2(T.Cast("float32", acc_s[0, i * 8 + ki, j * 32 + kj]) * T.float32(0.1275174307460247) - T.Cast("float32", scores_max[i * 8 + ki]) * T.float32(0.1275174307460247)))
                                        scores_sum_acc = T.decl_buffer((8, 32), "bfloat16", scope="shared.rsram")
                                        if j == 0:
                                            for ki in T.serial(8, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                                for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                                    scores_sum_acc[ki, kj] = T.bfloat16(0.0)
                                        for ki in T.serial(8, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                            for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                                scores_sum_acc[ki, kj] = scores_sum_acc[ki, kj] + acc_s[0, i * 8 + ki, j * 32 + kj]
                                        if j == 3:
                                            T.vector_core_in_tile_reduce("sum", T.region(scores_sum[i * 8], 2, 8), T.region(scores_sum_acc[0, 0], 1, 8, 32), 1)
                                T.dma_copy(T.region(acc_s[0, 0, 0], 1, 1, 128, 128), T.region(acc_s_cast[0, 0, 0], 2, 1, 128, 128), 0, T.sync_token_id(9))
                                for i in T.serial(16, annotations={"tile.domain": [128, 128], "tile.execution_axis": 0, "tile.execution_domain_axes": [0, 1], "tile.scope_entry": 1, "tile.tile_size": [8, 32]}):
                                    for j in T.serial(4, annotations={"tile.execution_axis": 1}):
                                        for ki in T.serial(8, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                            for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                                acc_o[i * 8 + ki, j * 32 + kj] = acc_o[i * 8 + ki, j * 32 + kj] * scores_scale[i * 8 + ki]
                                for i in T.serial(1, annotations={"tile.domain": [128], "tile.execution_axis": 0, "tile.execution_domain_axes": [0], "tile.scope_entry": 1, "tile.tile_size": [128]}):
                                    for ki in T.vectorized(128, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                        logsum[ki] = logsum[ki] * scores_scale[ki] + scores_sum[ki]
                                T.wait_token(13)
                                T.wait_token(9)
                                T.mma_sunmmio(T.region(acc_s_cast[0, 0, 0], 1, 1, 128, 128), T.region(V_shared[0, 0, 0], 1, 1, 128, 128), T.region(acc_o[0, 0], 3, 128, 128), T.bool(False), T.bool(False), T.bool(False), 0, T.sync_token_id(10))
                                T.wait_token(10)
                                T.dma_copy(T.region(acc_s[0, 0, 0], 1, 1, 128, 128), T.region(acc_s_cast[0, 0, 0], 2, 1, 128, 128), 1024, T.sync_token_id(11))
                                T.wait_token(11)
                                T.mma_sunmmio(T.region(acc_s_cast[0, 0, 0], 1, 1, 128, 128), T.region(V_shared[0, 0, 0], 1, 1, 128, 128), T.region(acc_o[0, 0], 3, 128, 128), T.bool(False), T.bool(False), T.bool(False), 1024, T.sync_token_id(12))
                                for i in T.serial(16, annotations={"tile.domain": [128, 128], "tile.execution_axis": 0, "tile.execution_domain_axes": [0, 1], "tile.scope_entry": 1, "tile.tile_size": [8, 32]}):
                                    for j in T.serial(4, annotations={"tile.execution_axis": 1}):
                                        for ki in T.serial(8, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                            for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                                acc_s[0, i * 8 + ki, j * 32 + kj] = T.bfloat16(0.0)
                                T.wait_token(12)
                                T.dma_copy(T.region(V_1[bz, k * 128 + 128, by, 0], 1, 1, 128, 1, 128), T.region(V_shared[0, 0, 0], 2, 1, 128, 128), 0, T.sync_token_id(13))
                            T.dma_copy(T.region(Q_rsram_stage[0, 0, 0, 0], 1, 1, 128, 1, 128), T.region(Q_shared[0, 0], 2, 128, 128), 0, T.sync_token_id(14))
                            for i in T.serial(1, annotations={"tile.domain": [128], "tile.execution_axis": 0, "tile.execution_domain_axes": [0], "tile.scope_entry": 1, "tile.tile_size": [128]}):
                                for ki in T.vectorized(128, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                    scores_max_prev[ki] = scores_max[ki]
                            T.wait_token(14)
                            T.wait_token(8)
                            T.mma_sunmmio(T.region(Q_shared[0, 0], 1, 128, 128), T.region(K_shared[0, 0, 0], 1, 1, 128, 128), T.region(acc_s[0, 0, 0], 3, 1, 128, 128), T.bool(False), T.bool(True), T.bool(False), 0, T.sync_token_id(15))
                            T.wait_token(15)
                            T.dma_copy(T.region(Q_rsram_stage[0, 0, 0, 0], 1, 1, 128, 1, 128), T.region(Q_shared[0, 0], 2, 128, 128), 1024, T.sync_token_id(16))
                            T.wait_token(16)
                            T.mma_sunmmio(T.region(Q_shared[0, 0], 1, 128, 128), T.region(K_shared[0, 0, 0], 1, 1, 128, 128), T.region(acc_s[0, 0, 0], 3, 1, 128, 128), T.bool(False), T.bool(True), T.bool(False), 1024, T.sync_token_id(17))
                            with T.decl_buffer((8, 32), "bfloat16", scope="shared.rsram") as scores_max_acc:
                                scores_max_res = T.decl_buffer((8,), "bfloat16", scope="shared.rsram")
                                T.wait_token(17)
                                for i0 in T.serial(16, annotations={"tile.domain": [128, 128], "tile.execution_axis": 0, "tile.execution_domain_axes": [0, 1], "tile.scope_entry": 1, "tile.tile_size": [8, 32]}):
                                    for i1 in T.serial(4, annotations={"tile.execution_axis": 1}):
                                        if i1 == 0:
                                            for ki in T.serial(8, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                                for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                                    scores_max_acc[ki, kj] = T.bfloat16("-inf")
                                        for ki in T.serial(8, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                            for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                                scores_max_acc[ki, kj] = T.max(scores_max_acc[ki, kj], acc_s[0, i0 * 8 + ki, i1 * 32 + kj])
                                        if i1 == 3:
                                            T.vector_core_in_tile_reduce("max", T.region(scores_max_res[0], 1, 8), T.region(scores_max_acc[0, 0], 1, 8, 32), 1)
                                            for ki in T.vectorized(8, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                                scores_max[i0 * 8 + ki] = T.max(scores_max[i0 * 8 + ki], scores_max_res[ki])
                            for i in T.serial(1, annotations={"tile.domain": [128], "tile.execution_axis": 0, "tile.execution_domain_axes": [0], "tile.scope_entry": 1, "tile.tile_size": [128]}):
                                for ki in T.vectorized(128, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                    scores_max[ki] = T.max(scores_max[ki], scores_max_prev[ki])
                                for ki in T.vectorized(128, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                    scores_scale[ki] = T.Cast("bfloat16", T.exp2(T.Cast("float32", scores_max_prev[ki]) * T.float32(0.1275174307460247) - T.Cast("float32", scores_max[ki]) * T.float32(0.1275174307460247)))
                            for i in T.serial(16, annotations={"tile.domain": [128, 128], "tile.execution_axis": 0, "tile.execution_domain_axes": [0, 1], "tile.scope_entry": 1, "tile.tile_size": [8, 32]}):
                                for j in T.serial(4, annotations={"tile.execution_axis": 1}):
                                    for ki in T.serial(8, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                        for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                            acc_s[0, i * 8 + ki, j * 32 + kj] = T.Cast("bfloat16", T.exp2(T.Cast("float32", acc_s[0, i * 8 + ki, j * 32 + kj]) * T.float32(0.1275174307460247) - T.Cast("float32", scores_max[i * 8 + ki]) * T.float32(0.1275174307460247)))
                                    scores_sum_acc = T.decl_buffer((8, 32), "bfloat16", scope="shared.rsram")
                                    if j == 0:
                                        for ki in T.serial(8, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                            for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                                scores_sum_acc[ki, kj] = T.bfloat16(0.0)
                                    for ki in T.serial(8, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                        for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                            scores_sum_acc[ki, kj] = scores_sum_acc[ki, kj] + acc_s[0, i * 8 + ki, j * 32 + kj]
                                    if j == 3:
                                        T.vector_core_in_tile_reduce("sum", T.region(scores_sum[i * 8], 2, 8), T.region(scores_sum_acc[0, 0], 1, 8, 32), 1)
                            T.dma_copy(T.region(acc_s[0, 0, 0], 1, 1, 128, 128), T.region(acc_s_cast[0, 0, 0], 2, 1, 128, 128), 0, T.sync_token_id(18))
                            for i in T.serial(16, annotations={"tile.domain": [128, 128], "tile.execution_axis": 0, "tile.execution_domain_axes": [0, 1], "tile.scope_entry": 1, "tile.tile_size": [8, 32]}):
                                for j in T.serial(4, annotations={"tile.execution_axis": 1}):
                                    for ki in T.serial(8, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                        for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                            acc_o[i * 8 + ki, j * 32 + kj] = acc_o[i * 8 + ki, j * 32 + kj] * scores_scale[i * 8 + ki]
                            for i in T.serial(1, annotations={"tile.domain": [128], "tile.execution_axis": 0, "tile.execution_domain_axes": [0], "tile.scope_entry": 1, "tile.tile_size": [128]}):
                                for ki in T.vectorized(128, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                    logsum[ki] = logsum[ki] * scores_scale[ki] + scores_sum[ki]
                            T.wait_token(18)
                            T.wait_token(13)
                            T.mma_sunmmio(T.region(acc_s_cast[0, 0, 0], 1, 1, 128, 128), T.region(V_shared[0, 0, 0], 1, 1, 128, 128), T.region(acc_o[0, 0], 3, 128, 128), T.bool(False), T.bool(False), T.bool(False), 0, T.sync_token_id(19))
                            T.wait_token(19)
                            T.dma_copy(T.region(acc_s[0, 0, 0], 1, 1, 128, 128), T.region(acc_s_cast[0, 0, 0], 2, 1, 128, 128), 1024, T.sync_token_id(20))
                            T.wait_token(20)
                            T.mma_sunmmio(T.region(acc_s_cast[0, 0, 0], 1, 1, 128, 128), T.region(V_shared[0, 0, 0], 1, 1, 128, 128), T.region(acc_o[0, 0], 3, 128, 128), T.bool(False), T.bool(False), T.bool(False), 1024, T.sync_token_id(21))
                            for i in T.serial(16, annotations={"tile.domain": [128, 128], "tile.execution_axis": 0, "tile.execution_domain_axes": [0, 1], "tile.scope_entry": 1, "tile.tile_size": [8, 32]}):
                                for j in T.serial(4, annotations={"tile.execution_axis": 1}):
                                    for ki in T.serial(8, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                        for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                            acc_o[i * 8 + ki, j * 32 + kj] = acc_o[i * 8 + ki, j * 32 + kj] / logsum[i * 8 + ki]
                                    for ki in T.serial(8, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                        for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                            O_shared[i * 8 + ki, j * 32 + kj] = acc_o[i * 8 + ki, j * 32 + kj]
                            T.wait_token(23)
                            T.sunmmio_layout_transform(T.region(O_shared[0, 0], 1, 128, 128), T.region(Output_layout_stage[0, 0], 2, 128, 128), T.sync_token_id(22))
                            T.wait_token(22)
                            T.dma_copy(T.region(Output_layout_stage[0, 0], 1, 128, 128), T.region(Output_1[bz, bx_1 * 128, by, 0], 2, 1, 128, 1, 128), 0, T.sync_token_id(23))
            T.wait_token(23)
        return 0
    """

    script_lower_tile_op = [
        """
        Q = T.match_buffer(Q_handle, (2, 4096, 8, 128), "bfloat16", strides=(4194304, 1024, 128, 1))
        K = T.match_buffer(K_handle, (2, 4096, 8, 128), "bfloat16", strides=(4194304, 1024, 128, 1))
        V = T.match_buffer(V_handle, (2, 4096, 8, 128), "bfloat16", strides=(4194304, 1024, 128, 1))
        Output = T.match_buffer(Output_handle, (2, 4096, 8, 128), "bfloat16", strides=(4194304, 1024, 128, 1))
        """,
        """
            bx = T.launch_thread("blockIdx.x", 16)
            with T.block("tilelang_root"):
        """,
        "for bz, by, bx_1 in T.grid(2, 8, 32):",
        "T.dma_copy(T.region(Q[bz, bx_1 * 128, by, 0], 1, 1, 128, 1, 128), T.region(Q_layout_stage[0, 0, 0, 0], 2, 1, 128, 1, 128), 0)",
        "T.sunmmio_layout_transform(T.region(Q_layout_stage[0, 0, 0, 0], 1, 1, 128, 1, 128), T.region(Q_rsram_stage[0, 0, 0, 0], 2, 1, 128, 1, 128))",
        "T.dma_copy(T.region(Q_rsram_stage[0, 0, 0, 0], 1, 1, 128, 1, 128), T.region(Q_shared[0, 0], 2, 128, 128), 0)",
    ]

    script_inject_sunmmio_sync = [
        """
        Output_1 = T.decl_buffer((2, 4096, 8, 128), "bfloat16", data=Output, strides=(4194304, 1024, 128, 1))
        V_1 = T.decl_buffer((2, 4096, 8, 128), "bfloat16", data=V, strides=(4194304, 1024, 128, 1))
        K_1 = T.decl_buffer((2, 4096, 8, 128), "bfloat16", data=K, strides=(4194304, 1024, 128, 1))
        Q_1 = T.decl_buffer((2, 4096, 8, 128), "bfloat16", data=Q, strides=(4194304, 1024, 128, 1))
        with T.launch_thread("blockIdx.x", 16) as bx:
        """,
        "for bz in range(2):",
        "for by in range(8):",
        "for bx_1 in range(32):",
        "T.dma_copy(T.region(Q_1[bz, bx_1 * 128, by, 0], 1, 1, 128, 1, 128), T.region(Q_layout_stage[0, 0, 0, 0], 2, 1, 128, 1, 128), 0, T.sync_token_id(0))",
        "T.sunmmio_layout_transform(T.region(Q_layout_stage[0, 0, 0, 0], 1, 1, 128, 1, 128), T.region(Q_rsram_stage[0, 0, 0, 0], 2, 1, 128, 1, 128), T.sync_token_id(1))",
        "T.dma_copy(T.region(Q_rsram_stage[0, 0, 0, 0], 1, 1, 128, 1, 128), T.region(Q_shared[0, 0], 2, 128, 128), 0, T.sync_token_id(",
        """
        Q = T.match_buffer(Q_handle, (2, 4096, 8, 128), "bfloat16", data=Q.data, strides=(4194304, 1024, 128, 1))
        K = T.match_buffer(K_handle, (2, 4096, 8, 128), "bfloat16", data=K.data, strides=(4194304, 1024, 128, 1))
        V = T.match_buffer(V_handle, (2, 4096, 8, 128), "bfloat16", data=V.data, strides=(4194304, 1024, 128, 1))
        Output = T.match_buffer(Output_handle, (2, 4096, 8, 128), "bfloat16", data=Output.data, strides=(4194304, 1024, 128, 1))
        """,
    ]

    test_config = {
        "LowerTileOp": {
            "script_expected": script_lower_tile_op,
        },
        "InjectSunmmioSync": {
            "script_expected": script_inject_sunmmio_sync,
        },
        "DeviceMod": {
            "script_expected": script_device_mode,
            "show_generated_script": True,
        },
    }

    test_config = get_or_add_default_verify(func, test_config)
    if not is_log:
        compile_test(func, target="Sunmmio", test_config=test_config)
    else:
        compile_test(
            func,
            out_idx=[2],
            target="Sunmmio",
            log_pass_output=True,
            log_dir=os.path.join(os.path.dirname(__file__), "_debug", "flashattn"),
            remove_header=True,
        )


if __name__ == "__main__":
    test_flashattn()
    # test_flashattn(is_log=True)
