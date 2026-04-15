import tilelang.language as T
from compile_pipeline import compile_test
from formal_verify_funcs import *


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
    shape = [batch, seq_len, heads, dim]
    # Different precisions will cause different number of allocates. The default allocate is allocated according to uint8, so when the data type is float16, the number of allocates will be doubled.
    dtype = T.float16
    # accum_dtype = T.float32
    accum_dtype = T.float16

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (
            bx,
            by,
            bz,
        ):
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

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N)) if is_causal else T.ceildiv(seq_len, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            bx * block_M + i >= k * block_N + j,
                            0,
                            -T.infinity(acc_s.dtype),
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_len, -T.infinity(acc_s.dtype), 0)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)

                for i in T.serial(0, block_M):
                    scores_max_prev[i] = scores_max[i]
                    scores_max[i] = -T.infinity(accum_dtype)
                    for j in T.serial(0, block_N):
                        scores_max[i] = T.max(scores_max[i], acc_s[i, j])
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

                for i in T.serial(0, block_M):
                    scores_sum[i] = T.cast(0, accum_dtype)
                    for j in T.serial(0, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        scores_sum[i] = scores_sum[i] + acc_s[i, j]

                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main


def test_flashattn():
    func = kernel_flashattn(8, 32, 4096, 128, False, block_M=128, block_N=128, num_stages=1, threads=1)
    script_lower_tile_op = """
            with T.block("tilelang_root"):
                T.reads(Q[bz, bx * 128, by, 0], K[bz, 0:3969, by, 0], V[bz, 0:3969, by, 0], Output[bz, bx * 128, by, 0])
                T.writes()
                T.block_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Layout"][2], acc_s_cast: metadata["tl.Layout"][3], V_shared: metadata["tl.Layout"][4], acc_o: metadata["tl.Layout"][5], O_shared: metadata["tl.Layout"][6]}})
                Q_shared = T.alloc_buffer((128, 128), "float16", data=Q_shared.data, scope="shared.asram")
                K_shared = T.alloc_buffer((128, 128), "float16", data=K_shared.data, scope="shared.wsram")
                V_shared = T.alloc_buffer((128, 128), "float16", data=V_shared.data, scope="shared.wsram")
                O_shared = T.alloc_buffer((128, 128), "float16", data=O_shared.data, scope="shared.rsram")
                acc_s = T.alloc_buffer((128, 128), "float16", data=acc_s.data, scope="shared.rsram")
                acc_s_cast = T.alloc_buffer((128, 128), "float16", data=acc_s_cast.data, scope="shared.asram")
                acc_o = T.alloc_buffer((128, 128), "float16", data=acc_o.data, scope="shared.rsram")
                scores_max = T.alloc_buffer((128,), "float16", scope="shared.rsram")
                scores_max_prev = T.alloc_buffer((128,), "float16", scope="shared.rsram")
                scores_scale = T.alloc_buffer((128,), "float16", scope="shared.rsram")
                scores_sum = T.alloc_buffer((128,), "float16", scope="shared.rsram")
                logsum = T.alloc_buffer((128,), "float16", scope="shared.rsram")
                T.dma_copy(T.region(Q[bz, bx * 128, by, 0], 1, 1, 128, 1, 128), T.region(Q_shared[0, 0], 2, 128, 128))
                for i0 in T.serial(128, annotations={"tile.domain": [128, 128], "tile.loop_parallel": 1, "tile.loop_stage": 0}):
                    for i1 in T.serial(128, annotations={"tile.loop_parallel": 1, "tile.loop_stage": 0}):
                        acc_o[i0, i1] = T.Cast("float16", 0)
                for i0 in T.serial(128, annotations={"tile.domain": [128], "tile.loop_parallel": 1, "tile.loop_stage": 0}):
                    logsum[i0] = T.Cast("float16", 0)
                for i0 in T.serial(128, annotations={"tile.domain": [128], "tile.loop_parallel": 1, "tile.loop_stage": 0}):
                    scores_max[i0] = T.infinity("float16") * T.float16(-1.0)
                for k in T.serial(32, annotations={"num_stages": 1}):
                    T.dma_copy(T.region(K[bz, k * 128, by, 0], 1, 1, 128, 1, 128), T.region(K_shared[0, 0], 2, 128, 128))
                    for i in T.unroll(512, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        for vec in T.vectorized(32):
                            acc_s[(i * 32 + vec) // 128, (i * 32 + vec) % 128] = T.float16(0.0)
                    with T.block("_gemm_sss"):
                        T.reads()
                        T.writes()
                        T.mma_sunmmio(T.region(Q_shared[0, 0], 1, 128, 128), T.region(K_shared[0, 0], 1, 128, 128), T.region(acc_s[0, 0], 3, 128, 128), T.bool(False), T.bool(True), T.bool(False))
                    for i in range(128):
                        scores_max_prev[i] = scores_max[i]
                        scores_max[i] = T.infinity("float16") * T.float16(-1.0)
                        for j in range(128):
                            scores_max[i] = T.max(scores_max[i], acc_s[i, j])
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        for vec in T.vectorized(32):
                            scores_scale[i * 32 + vec] = T.Cast("float16", T.exp2(T.Cast("float32", scores_max_prev[i * 32 + vec]) * T.float32(0.1275174307460247) - T.Cast("float32", scores_max[i * 32 + vec]) * T.float32(0.1275174307460247)))
                    for i in range(128):
                        scores_sum[i] = T.float16(0.0)
                        for j in range(128):
                            acc_s[i, j] = T.Cast("float16", T.exp2(T.Cast("float32", acc_s[i, j]) * T.float32(0.1275174307460247) - T.Cast("float32", scores_max[i]) * T.float32(0.1275174307460247)))
                            scores_sum[i] = scores_sum[i] + acc_s[i, j]
                    for i in T.unroll(2, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        for vec in T.vectorized(64):
                            logsum[i * 64 + vec] = logsum[i * 64 + vec] * scores_scale[i * 64 + vec] + scores_sum[i * 64 + vec]
                    T.dma_copy(T.region(acc_s[0, 0], 1, 128, 128), T.region(acc_s_cast[0, 0], 2, 128, 128))
                    for i in T.unroll(512, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        for vec in T.vectorized(32):
                            acc_o[(i * 32 + vec) // 128, (i * 32 + vec) % 128] = acc_o[(i * 32 + vec) // 128, (i * 32 + vec) % 128] * scores_scale[(i * 32 + vec) // 128]
                    T.dma_copy(T.region(V[bz, k * 128, by, 0], 1, 1, 128, 1, 128), T.region(V_shared[0, 0], 2, 128, 128))
                    with T.block("_gemm_sss"):
                        T.reads()
                        T.writes()
                        T.mma_sunmmio(T.region(acc_s_cast[0, 0], 1, 128, 128), T.region(V_shared[0, 0], 1, 128, 128), T.region(acc_o[0, 0], 3, 128, 128), T.bool(False), T.bool(False), T.bool(False))
                for i in T.unroll(512, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    for vec in T.vectorized(32):
                        acc_o[(i * 32 + vec) // 128, (i * 32 + vec) % 128] = acc_o[(i * 32 + vec) // 128, (i * 32 + vec) % 128] / logsum[(i * 32 + vec) // 128]
                for i in T.serial(128, annotations={"tile.domain": [128, 128], "tile.loop_parallel": 1, "tile.loop_stage": 0}):
                    for j in T.serial(128, annotations={"tile.loop_parallel": 1, "tile.loop_stage": 0}):
                        O_shared[i, j] = acc_o[i, j]
                T.dma_copy(T.region(O_shared[0, 0], 1, 128, 128), T.region(Output[bz, bx * 128, by, 0], 2, 1, 128, 1, 128))
        """

    script_inject_sunmmio_sync = """
        with T.launch_thread("blockIdx.x", 32) as bx:
            by = T.launch_thread("blockIdx.y", 32)
            bz = T.launch_thread("blockIdx.z", 8)
            tx = T.launch_thread("threadIdx.x", 1)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            with T.decl_buffer((128, 128), "float16", scope="shared.asram") as Q_shared:
                K_shared = T.decl_buffer((1, 128, 128), "float16", scope="shared.wsram")
                V_shared = T.decl_buffer((1, 128, 128), "float16", scope="shared.wsram")
                O_shared = T.decl_buffer((128, 128), "float16", scope="shared.rsram")
                acc_s = T.decl_buffer((1, 128, 128), "float16", scope="shared.rsram")
                acc_s_cast = T.decl_buffer((1, 128, 128), "float16", scope="shared.asram")
                acc_o = T.decl_buffer((128, 128), "float16", scope="shared.rsram")
                scores_max = T.decl_buffer((128,), "float16", scope="shared.rsram")
                scores_max_prev = T.decl_buffer((128,), "float16", scope="shared.rsram")
                scores_scale = T.decl_buffer((128,), "float16", scope="shared.rsram")
                scores_sum = T.decl_buffer((128,), "float16", scope="shared.rsram")
                logsum = T.decl_buffer((128,), "float16", scope="shared.rsram")
                Q_2 = T.Buffer((8, 4096, 32, 128), "float16", data=Q, strides=(16777216, 4096, 128, 1))
                T.dma_copy(T.region(Q_2[bz, bx * 128, by, 0], 1, 1, 128, 1, 128), T.region(Q_shared[0, 0], 2, 128, 128), T.sync_token_id(0))
                for i0 in T.serial(16, annotations={"tile.domain": [128, 128], "tile.execution_axis": 0, "tile.execution_domain_axes": [0, 1], "tile.scope_entry": 1, "tile.tile_size": [8, 32]}):
                    for i1 in T.serial(4, annotations={"tile.execution_axis": 1}):
                        for ki in T.serial(8, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                            for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                acc_o[i0 * 8 + ki, i1 * 32 + kj] = T.float16(0.0)
                for i0 in T.serial(1, annotations={"tile.domain": [128], "tile.execution_axis": 0, "tile.execution_domain_axes": [0], "tile.scope_entry": 1, "tile.tile_size": [128]}):
                    for ki in T.serial(2, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                        for vec in T.vectorized(64):
                            logsum[ki * 64 + vec] = T.float16(0.0)
                for i0 in T.serial(1, annotations={"tile.domain": [128], "tile.execution_axis": 0, "tile.execution_domain_axes": [0], "tile.scope_entry": 1, "tile.tile_size": [128]}):
                    for ki in T.serial(2, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                        for vec in T.vectorized(64):
                            scores_max[ki * 64 + vec] = T.infinity("float16") * T.float16(-1.0)
                K_2 = T.Buffer((8, 4096, 32, 128), "float16", data=K, strides=(16777216, 4096, 128, 1))
                T.dma_copy(T.region(K_2[bz, 0, by, 0], 1, 1, 128, 1, 128), T.region(K_shared[0, 0, 0], 2, 1, 128, 128), T.sync_token_id(1))
                for i in T.unroll(512):
                    for vec in T.vectorized(32):
                        acc_s[0, i // 4, i % 4 * 32 + vec] = T.float16(0.0)
                V_2 = T.Buffer((8, 4096, 32, 128), "float16", data=V, strides=(16777216, 4096, 128, 1))
                T.dma_copy(T.region(V_2[bz, 0, by, 0], 1, 1, 128, 1, 128), T.region(V_shared[0, 0, 0], 2, 1, 128, 128), T.sync_token_id(2))
                T.sync_null_token(4)
                T.sync_null_token(5)
                T.sync_null_token(6)
                T.sync_null_token(7)
                for k in range(31):
                    T.wait_token(0)
                    T.wait_token(1)
                    T.wait_token(4)
                    T.wait_token(5)
                    T.mma_sunmmio(T.region(Q_shared[0, 0], 1, 128, 128), T.region(K_shared[0, 0, 0], 1, 1, 128, 128), T.region(acc_s[0, 0, 0], 3, 1, 128, 128), T.bool(False), T.bool(True), T.bool(False), T.sync_token_id(3))
                    for i in range(128):
                        scores_max_prev[i] = scores_max[i]
                        scores_max[i] = T.infinity("float16") * T.float16(-1.0)
                        for j in range(128):
                            T.wait_token(3)
                            scores_max[i] = T.max(scores_max[i], acc_s[0, i, j])
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    T.dma_copy(T.region(K_2[bz, k * 128 + 128, by, 0], 1, 1, 128, 1, 128), T.region(K_shared[0, 0, 0], 2, 1, 128, 128), T.sync_token_id(4))
                    for i in range(128):
                        scores_sum[i] = T.float16(0.0)
                        for j in range(128):
                            acc_s[0, i, j] = T.Cast("float16", T.exp2(T.Cast("float32", acc_s[0, i, j]) * T.float32(0.1275174307460247) - T.Cast("float32", scores_max[i]) * T.float32(0.1275174307460247)))
                            scores_sum[i] = scores_sum[i] + acc_s[0, i, j]
                    for i in T.unroll(4):
                        for vec in T.vectorized(32):
                            scores_scale[i * 32 + vec] = T.Cast("float16", T.exp2(T.Cast("float32", scores_max_prev[i * 32 + vec]) * T.float32(0.1275174307460247) - T.Cast("float32", scores_max[i * 32 + vec]) * T.float32(0.1275174307460247)))
                    T.wait_token(6)
                    T.dma_copy(T.region(acc_s[0, 0, 0], 1, 1, 128, 128), T.region(acc_s_cast[0, 0, 0], 2, 1, 128, 128), T.sync_token_id(5))
                    for i in T.unroll(512):
                        for vec in T.vectorized(32):
                            acc_o[i // 4, i % 4 * 32 + vec] = acc_o[i // 4, i % 4 * 32 + vec] * scores_scale[i // 4]
                    T.wait_token(2)
                    T.wait_token(7)
                    T.mma_sunmmio(T.region(acc_s_cast[0, 0, 0], 1, 1, 128, 128), T.region(V_shared[0, 0, 0], 1, 1, 128, 128), T.region(acc_o[0, 0], 3, 128, 128), T.bool(False), T.bool(False), T.bool(False), T.sync_token_id(6))
                    for i in T.unroll(2):
                        for vec in T.vectorized(64):
                            logsum[i * 64 + vec] = logsum[i * 64 + vec] * scores_scale[i * 64 + vec] + scores_sum[i * 64 + vec]
                    for i in T.unroll(512):
                        for vec in T.vectorized(32):
                            acc_s[0, i // 4, i % 4 * 32 + vec] = T.float16(0.0)
                    T.dma_copy(T.region(V_2[bz, k * 128 + 128, by, 0], 1, 1, 128, 1, 128), T.region(V_shared[0, 0, 0], 2, 1, 128, 128), T.sync_token_id(7))
                T.wait_token(4)
                T.mma_sunmmio(T.region(Q_shared[0, 0], 1, 128, 128), T.region(K_shared[0, 0, 0], 1, 1, 128, 128), T.region(acc_s[0, 0, 0], 3, 1, 128, 128), T.bool(False), T.bool(True), T.bool(False), T.sync_token_id(8))
                for i in range(128):
                    scores_max_prev[i] = scores_max[i]
                    scores_max[i] = T.infinity("float16") * T.float16(-1.0)
                    for j in range(128):
                        T.wait_token(8)
                        scores_max[i] = T.max(scores_max[i], acc_s[0, i, j])
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in range(128):
                    scores_sum[i] = T.float16(0.0)
                    for j in range(128):
                        acc_s[0, i, j] = T.Cast("float16", T.exp2(T.Cast("float32", acc_s[0, i, j]) * T.float32(0.1275174307460247) - T.Cast("float32", scores_max[i]) * T.float32(0.1275174307460247)))
                        scores_sum[i] = scores_sum[i] + acc_s[0, i, j]
                for i in T.unroll(4):
                    for vec in T.vectorized(32):
                        scores_scale[i * 32 + vec] = T.Cast("float16", T.exp2(T.Cast("float32", scores_max_prev[i * 32 + vec]) * T.float32(0.1275174307460247) - T.Cast("float32", scores_max[i * 32 + vec]) * T.float32(0.1275174307460247)))
                T.dma_copy(T.region(acc_s[0, 0, 0], 1, 1, 128, 128), T.region(acc_s_cast[0, 0, 0], 2, 1, 128, 128), T.sync_token_id(9))
                for i in T.unroll(512):
                    for vec in T.vectorized(32):
                        acc_o[i // 4, i % 4 * 32 + vec] = acc_o[i // 4, i % 4 * 32 + vec] * scores_scale[i // 4]
                T.wait_token(9)
                T.wait_token(7)
                T.mma_sunmmio(T.region(acc_s_cast[0, 0, 0], 1, 1, 128, 128), T.region(V_shared[0, 0, 0], 1, 1, 128, 128), T.region(acc_o[0, 0], 3, 128, 128), T.bool(False), T.bool(False), T.bool(False), T.sync_token_id(10))
                for i in T.unroll(2):
                    for vec in T.vectorized(64):
                        logsum[i * 64 + vec] = logsum[i * 64 + vec] * scores_scale[i * 64 + vec] + scores_sum[i * 64 + vec]
                for i in T.unroll(512):
                    for vec in T.vectorized(32):
                        T.wait_token(10)
                        acc_o[i // 4, i % 4 * 32 + vec] = acc_o[i // 4, i % 4 * 32 + vec] / logsum[i // 4]
                for i in T.serial(16, annotations={"tile.domain": [128, 128], "tile.execution_axis": 0, "tile.execution_domain_axes": [0, 1], "tile.scope_entry": 1, "tile.tile_size": [8, 32]}):
                    for j in T.serial(4, annotations={"tile.execution_axis": 1}):
                        for ki in T.serial(8, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                            for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                O_shared[i * 8 + ki, j * 32 + kj] = acc_o[i * 8 + ki, j * 32 + kj]
                Output_2 = T.Buffer((8, 4096, 32, 128), "float16", data=Output, strides=(16777216, 4096, 128, 1))
                T.dma_copy(T.region(O_shared[0, 0], 1, 128, 128), T.region(Output_2[bz, bx * 128, by, 0], 2, 1, 128, 1, 128), T.sync_token_id(11))
            T.wait_token(11)
    """

    script_device_mode = """
    def main_kernel(K: T.handle("float16", "global"), Output: T.handle("float16", "global"), Q: T.handle("float16", "global"), V: T.handle("float16", "global")) -> T.int32:
        T.func_attr({"target": T.target({"keys": ["cpu"], "kind": "llvm", "mattr": ["device_mesh_nrow_4", "device_mesh_ncol_4"], "mcpu": "sunmmio-a4e", "tag": ""}), "thread_extent": {"blockIdx.x": 32, "blockIdx.y": 32, "blockIdx.z": 8, "threadIdx.x": 1, "threadIdx.y": 1, "threadIdx.z": 1}, "tir.is_global_func": T.bool(True), "tir.noalias": True, "tl.non_restrict_params": [], "tl.readonly_param_indices": [0, 1, 2, 3]})
        with T.launch_thread("blockIdx.x", 32) as bx:
            buf_shmem = T.allocate([100352], "uint8", "shared.rsram")
            buf_shmem_1 = T.allocate([65536], "uint8", "shared.wsram")
            buf_shmem_2 = T.allocate([65536], "uint8", "shared.asram")
            by = T.launch_thread("blockIdx.y", 32)
            bz = T.launch_thread("blockIdx.z", 8)
            tx = T.launch_thread("threadIdx.x", 1)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            Q_1 = T.Buffer((134217728,), "float16", data=Q)
            Q_shared = T.Buffer((16384,), "float16", data=buf_shmem_2, scope="shared.asram")
            T.dma_copy(T.region(Q_1[bz * 16777216 + bx * 524288 + by * 128], 1, 520320), T.region(Q_shared[16384], 2, 16384), T.sync_token_id(0))
            acc_o = T.Buffer((16384,), "float16", data=buf_shmem, scope="shared.rsram")
            for i in T.unroll(512):
                acc_o[i * 32:i * 32 + 32] = T.Broadcast(T.float16(0.0), 32)
            logsum = T.Buffer((128,), "float16", data=buf_shmem, scope="shared.rsram")
            for i in T.unroll(2):
                logsum[i * 64 + 32768:i * 64 + 32768 + 64] = T.Broadcast(T.float16(0.0), 64)
            scores_max = T.Buffer((128,), "float16", data=buf_shmem, scope="shared.rsram")
            for i in T.unroll(2):
                scores_max[i * 64 + 32896:i * 64 + 32896 + 64] = T.Broadcast(T.infinity("float16") * T.float16(-1.0), 64)
            K_1 = T.Buffer((134217728,), "float16", data=K)
            K_shared = T.Buffer((16384,), "float16", data=buf_shmem_1, scope="shared.wsram")
            T.dma_copy(T.region(K_1[bz * 16777216 + by * 128], 1, 520320), T.region(K_shared[16384], 2, 16384), T.sync_token_id(1))
            acc_s = T.Buffer((16384,), "float16", data=buf_shmem, scope="shared.rsram")
            for i in T.unroll(512):
                acc_s[i * 32 + 16384:i * 32 + 16384 + 32] = T.Broadcast(T.float16(0.0), 32)
            V_1 = T.Buffer((134217728,), "float16", data=V)
            V_shared = T.Buffer((16384,), "float16", data=buf_shmem_1, scope="shared.wsram")
            T.dma_copy(T.region(V_1[bz * 16777216 + by * 128], 1, 520320), T.region(V_shared[0], 2, 16384), T.sync_token_id(2))
            T.sync_null_token(4)
            T.sync_null_token(5)
            T.sync_null_token(6)
            T.sync_null_token(7)
            scores_max_prev = T.Buffer((128,), "float16", data=buf_shmem, scope="shared.rsram")
            scores_sum = T.Buffer((128,), "float16", data=buf_shmem, scope="shared.rsram")
            scores_scale = T.Buffer((128,), "float16", data=buf_shmem, scope="shared.rsram")
            acc_s_cast = T.Buffer((16384,), "float16", data=buf_shmem_2, scope="shared.asram")
            for k in range(31):
                T.wait_token(0)
                T.wait_token(1)
                T.wait_token(4)
                T.wait_token(5)
                T.mma_sunmmio(T.region(Q_shared[16384], 1, 16384), T.region(K_shared[16384], 1, 16384), T.region(acc_s[16384], 3, 16384), T.bool(False), T.bool(True), T.bool(False), T.sync_token_id(3))
                for i in range(128):
                    scores_max_prev[i + 33024] = scores_max[i + 32896]
                    scores_max[i + 32896] = T.infinity("float16") * T.float16(-1.0)
                    for j in range(128):
                        T.wait_token(3)
                        scores_max[i + 32896] = T.max(scores_max[i + 32896], acc_s[i * 128 + j + 16384])
                    scores_max[i + 32896] = T.max(scores_max[i + 32896], scores_max_prev[i + 33024])
                T.dma_copy(T.region(K_1[bz * 16777216 + k * 524288 + by * 128 + 524288], 1, 520320), T.region(K_shared[16384], 2, 16384), T.sync_token_id(4))
                for i in range(128):
                    scores_sum[i + 33280] = T.float16(0.0)
                    for j in range(128):
                        acc_s[i * 128 + j + 16384] = T.Cast("float16", T.exp2(T.Cast("float32", acc_s[i * 128 + j + 16384]) * T.float32(0.1275174307460247) - T.Cast("float32", scores_max[i + 32896]) * T.float32(0.1275174307460247)))
                        scores_sum[i + 33280] = scores_sum[i + 33280] + acc_s[i * 128 + j + 16384]
                for i in T.unroll(4):
                    scores_scale[i * 32 + 33152:i * 32 + 33152 + 32] = T.Cast("float16x32", T.exp2(T.Cast("float32x32", scores_max_prev[i * 32 + 33024:i * 32 + 33024 + 32]) * T.Broadcast(T.float32(0.1275174307460247), 32) - T.Cast("float32x32", scores_max[i * 32 + 32896:i * 32 + 32896 + 32]) * T.Broadcast(T.float32(0.1275174307460247), 32)))
                T.wait_token(6)
                T.dma_copy(T.region(acc_s[16384], 1, 16384), T.region(acc_s_cast[0], 2, 16384), T.sync_token_id(5))
                for i in T.unroll(512):
                    acc_o[i * 32:i * 32 + 32] = acc_o[i * 32:i * 32 + 32] * T.Broadcast(scores_scale[i // 4 + 33152], 32)
                T.wait_token(2)
                T.wait_token(7)
                T.mma_sunmmio(T.region(acc_s_cast[0], 1, 16384), T.region(V_shared[0], 1, 16384), T.region(acc_o[0], 3, 16384), T.bool(False), T.bool(False), T.bool(False), T.sync_token_id(6))
                for i in T.unroll(2):
                    logsum[i * 64 + 32768:i * 64 + 32768 + 64] = logsum[i * 64 + 32768:i * 64 + 32768 + 64] * scores_scale[i * 64 + 33152:i * 64 + 33152 + 64] + scores_sum[i * 64 + 33280:i * 64 + 33280 + 64]
                for i in T.unroll(512):
                    acc_s[i * 32 + 16384:i * 32 + 16384 + 32] = T.Broadcast(T.float16(0.0), 32)
                T.dma_copy(T.region(V_1[bz * 16777216 + k * 524288 + by * 128 + 524288], 1, 520320), T.region(V_shared[0], 2, 16384), T.sync_token_id(7))
            T.wait_token(4)
            T.mma_sunmmio(T.region(Q_shared[16384], 1, 16384), T.region(K_shared[16384], 1, 16384), T.region(acc_s[16384], 3, 16384), T.bool(False), T.bool(True), T.bool(False), T.sync_token_id(8))
            for i in range(128):
                scores_max_prev[i + 33024] = scores_max[i + 32896]
                scores_max[i + 32896] = T.infinity("float16") * T.float16(-1.0)
                for j in range(128):
                    T.wait_token(8)
                    scores_max[i + 32896] = T.max(scores_max[i + 32896], acc_s[i * 128 + j + 16384])
                scores_max[i + 32896] = T.max(scores_max[i + 32896], scores_max_prev[i + 33024])
            for i in range(128):
                scores_sum[i + 33280] = T.float16(0.0)
                for j in range(128):
                    acc_s[i * 128 + j + 16384] = T.Cast("float16", T.exp2(T.Cast("float32", acc_s[i * 128 + j + 16384]) * T.float32(0.1275174307460247) - T.Cast("float32", scores_max[i + 32896]) * T.float32(0.1275174307460247)))
                    scores_sum[i + 33280] = scores_sum[i + 33280] + acc_s[i * 128 + j + 16384]
            for i in T.unroll(4):
                scores_scale[i * 32 + 33152:i * 32 + 33152 + 32] = T.Cast("float16x32", T.exp2(T.Cast("float32x32", scores_max_prev[i * 32 + 33024:i * 32 + 33024 + 32]) * T.Broadcast(T.float32(0.1275174307460247), 32) - T.Cast("float32x32", scores_max[i * 32 + 32896:i * 32 + 32896 + 32]) * T.Broadcast(T.float32(0.1275174307460247), 32)))
            T.dma_copy(T.region(acc_s[16384], 1, 16384), T.region(acc_s_cast[0], 2, 16384), T.sync_token_id(9))
            for i in T.unroll(512):
                acc_o[i * 32:i * 32 + 32] = acc_o[i * 32:i * 32 + 32] * T.Broadcast(scores_scale[i // 4 + 33152], 32)
            T.wait_token(9)
            T.wait_token(7)
            T.mma_sunmmio(T.region(acc_s_cast[0], 1, 16384), T.region(V_shared[0], 1, 16384), T.region(acc_o[0], 3, 16384), T.bool(False), T.bool(False), T.bool(False), T.sync_token_id(10))
            for i in T.unroll(2):
                logsum[i * 64 + 32768:i * 64 + 32768 + 64] = logsum[i * 64 + 32768:i * 64 + 32768 + 64] * scores_scale[i * 64 + 33152:i * 64 + 33152 + 64] + scores_sum[i * 64 + 33280:i * 64 + 33280 + 64]
            for i in T.unroll(512):
                T.wait_token(10)
                acc_o[i * 32:i * 32 + 32] = acc_o[i * 32:i * 32 + 32] / T.Broadcast(logsum[i // 4 + 32768], 32)
            O_shared = T.Buffer((16384,), "float16", data=buf_shmem, scope="shared.rsram")
            for v0 in T.serial(4, annotations={"tile.buffer_new_shape": [4, 4, 32, 32], "tile.dim_map": [-2, -1], "tile.execution": 1, "tile.loop_parallel": 1, "tile.loop_stage": 2, "tile.scope_entry": 1, "tile.tile_size": [32, 32], "tile.tiled_buffer": acc_o_1}):
                acc_o_1 = T.handle("float16", "shared.rsram")
                for v1 in T.serial(4, annotations={"tile.buffer_new_shape": [4, 4, 32, 32], "tile.dim_map": [-2, -1], "tile.execution": 1, "tile.loop_parallel": 1, "tile.loop_stage": 2, "tile.tile_size": [32, 32], "tile.tiled_buffer": acc_o_1}):
                    for ki in T.serial(32, annotations={"tile.interior": 1, "tile.interior_axis": 0, "tile.loop_stage": 2, "tile.tiled_buffer": acc_o_1}):
                        O_shared[v0 * 4096 + ki * 128 + v1 * 32 + 33792:v0 * 4096 + ki * 128 + v1 * 32 + 33792 + 32] = acc_o[v0 * 4096 + ki * 128 + v1 * 32:v0 * 4096 + ki * 128 + v1 * 32 + 32]
            Output_1 = T.Buffer((134217728,), "float16", data=Output)
            T.dma_copy(T.region(O_shared[33792], 1, 16384), T.region(Output_1[bz * 16777216 + bx * 524288 + by * 128], 2, 520320), T.sync_token_id(11))
            T.wait_token(11)
        return 0
    """

    def get_verify_merge_allocate():
        kernel_name = "main_kernel"
        # 65536 65536 100352
        block_m, block_n, dim = 128, 128, 128
        cnt_a = block_m * dim + block_m * block_n
        cnt_w = block_n * dim + block_n * dim
        # rsram mainly has three matrix blocks and 5 vector blocks, the matrix size is 128 (float16), the third matrix block is at the end, aligned to 2048,
        # the 5 vector blocks combined are less than 2048, so we just take the alignment size
        cnt_r = block_m * dim + block_m * block_n + block_m * dim + 1024
        cnt_a *= 2
        cnt_w *= 2
        cnt_r *= 2
        return build_verify_merge_allocate(kernel_name=kernel_name, cnt_a=cnt_a, cnt_w=cnt_w, cnt_r=cnt_r)

    test_config = {
        "LowerTileOp": {
            "script_expected": script_lower_tile_op,
        },
        "InjectSunmmioSync": {
            "script_expected": script_inject_sunmmio_sync,
        },
        "MergeSharedMemoryAllocationsSunmmio": {
            "formal_verify": get_verify_merge_allocate(),
        },
        "DeviceMode": {
            "script_expected": script_device_mode,
        },
    }

    test_config = get_or_add_default_verify(func, test_config)
    compile_test(func, target="Sunmmio", test_config=test_config)


if __name__ == "__main__":
    test_flashattn()
