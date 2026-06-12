import os

import pytest
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


def mla_decode_no_split(
    batch=1,
    heads=128,
    kv_head_num=1,
    seqlen_kv=128,
    dim=64,
    pe_dim=64,
    block_N=64,
    block_H=32,
    num_stages=0,
    softmax_scale=None,
):
    if softmax_scale is None:
        softmax_scale = (dim + pe_dim) ** -0.5
    scale = float(softmax_scale * 1.44269504)
    dtype = T.float16
    accum_dtype = T.float32
    kv_group_num = heads // kv_head_num
    valid_block_H = min(block_H, kv_group_num)

    assert kv_head_num == 1, "kv_head_num must be 1"

    q_shape = [batch, heads, dim]
    q_pe_shape = [batch, heads, pe_dim]
    kv_shape = [batch, seqlen_kv, kv_head_num, dim]
    k_pe_shape = [batch, seqlen_kv, kv_head_num, pe_dim]

    q_shard_policy = T.MeshShardingPolicy(y=0, x=1)
    kv_shard_policy = T.MeshShardingPolicy()
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    assert heads % ncols == 0
    assert (heads // ncols) % valid_block_H == 0

    Q_layout = make_zz_layout(q_shape, [1, 2], (32, 32))
    Q_pe_layout = make_zz_layout(q_pe_shape, [1, 2], (32, 32))
    KV_layout = make_zz_layout(kv_shape, [1, 3], (32, 32))
    K_pe_layout = make_zz_layout(k_pe_shape, [1, 3], (32, 32))
    O_layout = make_zz_layout(q_shape, [1, 2], (32, 32))

    @T.prim_func
    def main(
        Q: T.MeshTensor(q_shape, q_shard_policy, device_mesh_config, dtype, layout=Q_layout),  # type: ignore
        Q_pe: T.MeshTensor(q_pe_shape, q_shard_policy, device_mesh_config, dtype, layout=Q_pe_layout),  # type: ignore
        KV: T.MeshTensor(kv_shape, kv_shard_policy, device_mesh_config, dtype, layout=KV_layout),  # type: ignore
        K_pe: T.MeshTensor(k_pe_shape, kv_shard_policy, device_mesh_config, dtype, layout=K_pe_layout),  # type: ignore
        Output: T.MeshTensor(q_shape, q_shard_policy, device_mesh_config, dtype, layout=O_layout),  # type: ignore
    ):
        with T.Kernel(ncores) as _cid:
            sharded_batch = Q.shape[0]
            sharded_heads = Q.shape[1]

            Q_shared = T.alloc_shared([block_H, dim], dtype)
            S_shared_local = T.alloc_shared([block_H, block_N], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
            KV_shared_qk = T.alloc_shared([block_N, dim], dtype)
            KV_shared_pv = T.alloc_shared([block_N, dim], dtype)
            K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
            O_shared = T.alloc_shared([block_H, dim], dtype)
            acc_s = T.alloc_shared([block_H, block_N], accum_dtype, scope="shared.rsram")
            acc_o = T.alloc_shared([block_H, dim], accum_dtype)
            scores_max = T.alloc_shared([block_H], accum_dtype)
            scores_max_prev = T.alloc_shared([block_H], accum_dtype)
            scores_scale = T.alloc_shared([block_H], accum_dtype)
            scores_sum = T.alloc_shared([block_H], accum_dtype)
            logsum = T.alloc_shared([block_H], accum_dtype)

            for bid in T.serial(sharded_batch):
                for hid in T.serial(T.ceildiv(sharded_heads, valid_block_H)):
                    cur_kv_head = hid // (kv_group_num // valid_block_H)

                    T.copy(Q[bid, hid * valid_block_H : (hid + 1) * valid_block_H, :], Q_shared)
                    T.copy(Q_pe[bid, hid * valid_block_H : (hid + 1) * valid_block_H, :], Q_pe_shared)
                    T.fill(acc_o, 0)
                    T.fill(logsum, 0)
                    T.fill(scores_max, -T.infinity(accum_dtype))

                    loop_range = T.ceildiv(seqlen_kv, block_N)
                    for k in T.Pipelined(loop_range, num_stages=num_stages):
                        T.copy(KV[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], KV_shared_qk)
                        T.copy(KV[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], KV_shared_pv)
                        T.copy(K_pe[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], K_pe_shared)
                        T.gemm(Q_shared, KV_shared_qk, acc_s, transpose_B=True, clear_accum=True)
                        T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True)

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

                        for i, j in T.Tiles([block_H, block_N]):
                            S_shared_local[i, j] = acc_s[i, j]
                        T.copy(S_shared_local, S_shared)
                        for i in T.Tiles([block_H]):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        for i, j in T.Tiles([block_H, dim]):
                            acc_o[i, j] *= scores_scale[i]
                        T.gemm(S_shared, KV_shared_pv, acc_o)

                    for i, j in T.Tiles([block_H, dim]):
                        acc_o[i, j] /= logsum[i]
                    T.copy(acc_o, O_shared)
                    T.copy(O_shared, Output[bid, hid * valid_block_H : (hid + 1) * valid_block_H, :])

    return main


def mla_decode_dynamic_length_no_split(
    batch=1,
    heads=128,
    kv_head_num=1,
    dim=64,
    pe_dim=64,
    block_N=64,
    block_H=32,
    num_stages=0,
    softmax_scale=None,
):
    if softmax_scale is None:
        softmax_scale = (dim + pe_dim) ** -0.5
    scale = float(softmax_scale * 1.44269504)
    dtype = T.float16
    accum_dtype = T.float32
    kv_group_num = heads // kv_head_num
    valid_block_H = min(block_H, kv_group_num)
    seqlen_kv = T.dynamic("seqlen_kv")

    assert kv_head_num == 1, "kv_head_num must be 1"

    q_shape = [batch, heads, dim]
    q_pe_shape = [batch, heads, pe_dim]
    kv_shape = [batch, seqlen_kv, kv_head_num, dim]
    k_pe_shape = [batch, seqlen_kv, kv_head_num, pe_dim]

    q_shard_policy = T.MeshShardingPolicy(y=0, x=1)
    kv_shard_policy = T.MeshShardingPolicy()
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    assert heads % ncols == 0
    assert (heads // ncols) % valid_block_H == 0

    @T.prim_func
    def main(
        Q: T.MeshTensor(q_shape, q_shard_policy, device_mesh_config, dtype),  # type: ignore
        Q_pe: T.MeshTensor(q_pe_shape, q_shard_policy, device_mesh_config, dtype),  # type: ignore
        KV: T.MeshTensor(kv_shape, kv_shard_policy, device_mesh_config, dtype),  # type: ignore
        K_pe: T.MeshTensor(k_pe_shape, kv_shard_policy, device_mesh_config, dtype),  # type: ignore
        Output: T.MeshTensor(q_shape, q_shard_policy, device_mesh_config, dtype),  # type: ignore
    ):
        with T.Kernel(ncores) as _cid:
            sharded_batch = Q.shape[0]
            sharded_heads = Q.shape[1]

            Q_shared = T.alloc_shared([block_H, dim], dtype)
            S_shared_local = T.alloc_shared([block_H, block_N], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
            KV_shared_qk = T.alloc_shared([block_N, dim], dtype)
            KV_shared_pv = T.alloc_shared([block_N, dim], dtype)
            K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
            O_shared = T.alloc_shared([block_H, dim], dtype)
            acc_s = T.alloc_shared([block_H, block_N], accum_dtype, scope="shared.rsram")
            acc_o = T.alloc_shared([block_H, dim], accum_dtype)
            scores_max = T.alloc_shared([block_H], accum_dtype)
            scores_max_prev = T.alloc_shared([block_H], accum_dtype)
            scores_scale = T.alloc_shared([block_H], accum_dtype)
            scores_sum = T.alloc_shared([block_H], accum_dtype)
            logsum = T.alloc_shared([block_H], accum_dtype)

            for bid in T.serial(sharded_batch):
                for hid in T.serial(T.ceildiv(sharded_heads, valid_block_H)):
                    cur_kv_head = hid // (kv_group_num // valid_block_H)

                    T.copy(Q[bid, hid * valid_block_H : (hid + 1) * valid_block_H, :], Q_shared)
                    T.copy(Q_pe[bid, hid * valid_block_H : (hid + 1) * valid_block_H, :], Q_pe_shared)
                    T.fill(acc_o, 0)
                    T.fill(logsum, 0)
                    T.fill(scores_max, -T.infinity(accum_dtype))

                    loop_range = T.ceildiv(seqlen_kv, block_N)
                    for k in T.Pipelined(loop_range, num_stages=num_stages):
                        T.copy(KV[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], KV_shared_qk)
                        T.copy(KV[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], KV_shared_pv)
                        T.copy(K_pe[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], K_pe_shared)
                        T.gemm(Q_shared, KV_shared_qk, acc_s, transpose_B=True, clear_accum=True)
                        T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True)

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

                        for i, j in T.Tiles([block_H, block_N]):
                            S_shared_local[i, j] = acc_s[i, j]
                        T.copy(S_shared_local, S_shared)
                        for i in T.Tiles([block_H]):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        for i, j in T.Tiles([block_H, dim]):
                            acc_o[i, j] *= scores_scale[i]
                        T.gemm(S_shared, KV_shared_pv, acc_o)

                    for i, j in T.Tiles([block_H, dim]):
                        acc_o[i, j] /= logsum[i]
                    T.copy(acc_o, O_shared)
                    T.copy(O_shared, Output[bid, hid * valid_block_H : (hid + 1) * valid_block_H, :])

    return main


def mla_decode_split(
    batch=1,
    heads=128,
    kv_head_num=1,
    seqlen_kv=128,
    dim=64,
    pe_dim=64,
    block_N=64,
    block_H=32,
    num_split=2,
    num_stages=0,
    softmax_scale=None,
):
    if softmax_scale is None:
        softmax_scale = (dim + pe_dim) ** -0.5
    scale = float(softmax_scale * 1.44269504)
    dtype = T.float16
    accum_dtype = T.float32
    kv_group_num = heads // kv_head_num
    valid_block_H = min(block_H, kv_group_num)

    assert kv_head_num == 1, "kv_head_num must be 1"
    assert seqlen_kv % num_split == 0

    q_shape = [batch, heads, dim]
    q_pe_shape = [batch, heads, pe_dim]
    kv_shape = [batch, seqlen_kv, kv_head_num, dim]
    k_pe_shape = [batch, seqlen_kv, kv_head_num, pe_dim]
    glse_shape = [batch, heads, num_split]
    output_partial_shape = [batch, heads, num_split, dim]

    q_shard_policy = T.MeshShardingPolicy(y=0, x=1)
    kv_shard_policy = T.MeshShardingPolicy()
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    assert heads % ncols == 0
    assert (heads // ncols) % valid_block_H == 0

    Q_layout = make_zz_layout(q_shape, [1, 2], (32, 32))
    Q_pe_layout = make_zz_layout(q_pe_shape, [1, 2], (32, 32))
    KV_layout = make_zz_layout(kv_shape, [1, 3], (32, 32))
    K_pe_layout = make_zz_layout(k_pe_shape, [1, 3], (32, 32))
    glse_layout = make_zz_layout(glse_shape, [1, 2], (32, 2))
    output_partial_layout = make_zz_layout(output_partial_shape, [1, 3], (32, 32))

    @T.prim_func
    def main(
        Q: T.MeshTensor(q_shape, q_shard_policy, device_mesh_config, dtype, layout=Q_layout),  # type: ignore
        Q_pe: T.MeshTensor(q_pe_shape, q_shard_policy, device_mesh_config, dtype, layout=Q_pe_layout),  # type: ignore
        KV: T.MeshTensor(kv_shape, kv_shard_policy, device_mesh_config, dtype, layout=KV_layout),  # type: ignore
        K_pe: T.MeshTensor(k_pe_shape, kv_shard_policy, device_mesh_config, dtype, layout=K_pe_layout),  # type: ignore
        glse: T.MeshTensor(glse_shape, q_shard_policy, device_mesh_config, accum_dtype, layout=glse_layout),  # type: ignore
        Output_partial: T.MeshTensor(output_partial_shape, q_shard_policy, device_mesh_config, dtype, layout=output_partial_layout),  # type: ignore
    ):
        with T.Kernel(ncores) as _cid:
            sharded_batch = Q.shape[0]
            sharded_heads = Q.shape[1]

            Q_shared = T.alloc_shared([block_H, dim], dtype)
            S_shared_local = T.alloc_shared([block_H, block_N], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
            KV_shared_qk = T.alloc_shared([block_N, dim], dtype)
            KV_shared_pv = T.alloc_shared([block_N, dim], dtype)
            K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
            O_shared = T.alloc_shared([block_H, dim], dtype)
            acc_s = T.alloc_shared([block_H, block_N], accum_dtype, scope="shared.rsram")
            acc_o = T.alloc_shared([block_H, dim], accum_dtype)
            scores_max = T.alloc_shared([block_H], accum_dtype)
            scores_max_prev = T.alloc_shared([block_H], accum_dtype)
            scores_scale = T.alloc_shared([block_H], accum_dtype)
            scores_sum = T.alloc_shared([block_H], accum_dtype)
            logsum = T.alloc_shared([block_H], accum_dtype)

            for bid in T.serial(sharded_batch):
                for hid in T.serial(T.ceildiv(sharded_heads, valid_block_H)):
                    cur_kv_head = hid // (kv_group_num // valid_block_H)
                    for bz in T.serial(num_split):
                        T.copy(Q[bid, hid * valid_block_H : (hid + 1) * valid_block_H, :], Q_shared)
                        T.copy(Q_pe[bid, hid * valid_block_H : (hid + 1) * valid_block_H, :], Q_pe_shared)
                        T.fill(acc_o, 0)
                        T.fill(logsum, 0)
                        T.fill(scores_max, -T.infinity(accum_dtype))

                        loop_range = T.ceildiv(seqlen_kv // num_split, block_N)
                        for k in T.Pipelined(loop_range, num_stages=num_stages):
                            kv_start = (seqlen_kv // num_split) * bz + k * block_N
                            kv_end = (seqlen_kv // num_split) * bz + (k + 1) * block_N
                            T.copy(KV[bid, kv_start:kv_end, cur_kv_head, :], KV_shared_qk)
                            T.copy(KV[bid, kv_start:kv_end, cur_kv_head, :], KV_shared_pv)
                            T.copy(K_pe[bid, kv_start:kv_end, cur_kv_head, :], K_pe_shared)
                            T.gemm(Q_shared, KV_shared_qk, acc_s, transpose_B=True, clear_accum=True)
                            T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True)

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

                            for i, j in T.Tiles([block_H, block_N]):
                                S_shared_local[i, j] = acc_s[i, j]
                            T.copy(S_shared_local, S_shared)
                            for i in T.Tiles([block_H]):
                                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                            for i, j in T.Tiles([block_H, dim]):
                                acc_o[i, j] *= scores_scale[i]
                            T.gemm(S_shared, KV_shared_pv, acc_o)

                        for i, j in T.Tiles([block_H, dim]):
                            acc_o[i, j] /= logsum[i]
                        for i in T.Tiles([block_H]):
                            scores_scale[i] = scores_max[i] * scale
                        for i in T.Tiles([block_H]):
                            logsum[i] = T.log(logsum[i])
                        for i in T.Tiles([block_H]):
                            logsum[i] *= 1.44269504
                        for i in T.Tiles([block_H]):
                            logsum[i] += scores_scale[i]
                        T.copy(logsum, glse[bid, hid * valid_block_H : (hid + 1) * valid_block_H, bz])
                        T.copy(acc_o, O_shared)
                        T.copy(
                            O_shared,
                            Output_partial[bid, hid * valid_block_H : (hid + 1) * valid_block_H, bz, :],
                        )

    return main


def mla_decode_split_combine_probe(
    batch=1,
    heads=128,
    dim=64,
    num_split=2,
):
    dtype = T.float16
    accum_dtype = T.float32

    glse_shape = [batch, heads, num_split]
    output_partial_shape = [batch, heads, num_split, dim]
    output_shape = [batch, heads, dim]

    q_shard_policy = T.MeshShardingPolicy(y=0, x=1)
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    assert heads % ncols == 0

    glse_layout = make_zz_layout(glse_shape, [1, 2], (32, 2))
    output_partial_layout = make_zz_layout(output_partial_shape, [1, 3], (32, 32))
    Output_layout = make_zz_layout(output_shape, [1, 2], (32, 32))

    @T.prim_func
    def main(
        glse: T.MeshTensor(glse_shape, q_shard_policy, device_mesh_config, accum_dtype, layout=glse_layout),  # type: ignore
        Output_partial: T.MeshTensor(  # type: ignore
            output_partial_shape,
            q_shard_policy,
            device_mesh_config,
            dtype,
            layout=output_partial_layout,
        ),
        Output: T.MeshTensor(output_shape, q_shard_policy, device_mesh_config, dtype, layout=Output_layout),  # type: ignore
    ):
        with T.Kernel(ncores) as _cid:
            sharded_batch = Output.shape[0]
            sharded_heads = Output.shape[1]

            po_local = T.alloc_fragment([dim], dtype)
            o_accum_local = T.alloc_fragment([dim], accum_dtype)
            lse_local_split = T.alloc_var(accum_dtype)
            lse_logsum_local = T.alloc_var(accum_dtype)
            lse_max_local = T.alloc_var(accum_dtype)
            scale_local = T.alloc_var(accum_dtype)

            for bid in T.serial(sharded_batch):
                for hid in T.serial(sharded_heads):
                    T.clear(lse_logsum_local)
                    T.clear(o_accum_local)
                    lse_max_local = -T.infinity(accum_dtype)
                    for k in T.serial(num_split):
                        lse_max_local = T.max(lse_max_local, glse[bid, hid, k])
                    for k in T.Pipelined(num_split, num_stages=1):
                        lse_local_split = glse[bid, hid, k]
                        lse_logsum_local += T.exp2(lse_local_split - lse_max_local)
                    lse_logsum_local = T.log(lse_logsum_local) * 1.44269504 + lse_max_local
                    for k in T.serial(num_split):
                        for i in T.Tiles([dim]):
                            po_local[i] = Output_partial[bid, hid, k, i]
                        lse_local_split = glse[bid, hid, k]
                        scale_local = T.exp2(lse_local_split - lse_logsum_local)
                        for i in T.Tiles([dim]):
                            o_accum_local[i] += po_local[i] * scale_local
                    for i in T.Tiles([dim]):
                        Output[bid, hid, i] = o_accum_local[i]

    return main


def mla_decode_dynamic_kv_zzlayout_probe(
    batch=1,
    kv_head_num=1,
    dim=64,
):
    dtype = T.float16
    seqlen_kv = T.dynamic("seqlen_kv")
    kv_shape = [batch, seqlen_kv, kv_head_num, dim]

    kv_shard_policy = T.MeshShardingPolicy()
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    KV_layout = make_zz_layout(kv_shape, [1, 3], (32, 32))

    @T.prim_func
    def main(
        KV: T.MeshTensor(kv_shape, kv_shard_policy, device_mesh_config, dtype, layout=KV_layout),  # type: ignore
    ):
        with T.Kernel(1):
            pass

    return main


def test_mla_decode_no_split_codegen_passes_loose_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        mla_decode_no_split(),
        tmp_path,
        mlir_filename="mla_decode_no_split_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.tc.mma", "suvm.tile.reduce"),
        opt_args=("--verify-each",),
    )
    assert_source_contains(src, ("suvm.tc.mma", "suvm.tile.reduce"))


def test_mla_decode_dynamic_length_no_split_codegen_passes_loose_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        mla_decode_dynamic_length_no_split(),
        tmp_path,
        mlir_filename="mla_decode_dynamic_length_no_split_suvm.mlir",
        expected_tokens=("suvm.bind_layout", "suvm.copy_async", "suvm.tc.mma", "suvm.tile.reduce"),
        opt_args=("--verify-each",),
    )
    assert_source_contains(src, ("suvm.bind_layout", "suvm.tc.mma", "suvm.tile.reduce"))


# @pytest.mark.xfail(
#     strict=True,
#     reason="Check failed: (T.RegisterScratchBuffer && stage_layout.defined()) is false: sunmmio layout transform: cannot build the staging layout.",
# )
def test_mla_decode_split_codegen_passes_loose_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        mla_decode_split(),
        tmp_path,
        mlir_filename="mla_decode_split_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.tc.mma", "suvm.tile.reduce"),
        opt_args=("--verify-each",),
    )
    assert_source_contains(src, ("suvm.tc.mma", "suvm.tile.reduce"))


@pytest.mark.xfail(
    strict=True,
    reason="Sunmmio lowering does not support scalar local.var state used by MLA split combine yet.",
)
def test_mla_decode_split_combine_probe_codegen_passes_loose_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        mla_decode_split_combine_probe(),
        tmp_path,
        mlir_filename="mla_decode_split_combine_probe_suvm.mlir",
        expected_tokens=("suvm.tile_store",),
        opt_args=("--verify-each",),
    )
    assert_source_contains(src, ("suvm.tile_store",))


@pytest.mark.xfail(
    strict=True,
    reason="CuteLayout DeriveLayoutLike does not support 4D dynamic K/V zzlayout with more than one symbolic stride yet.",
)
def test_mla_decode_dynamic_kv_zzlayout_probe_codegen_passes_loose_npuir_opt(tmp_path):
    validate_sunmmio_codegen_with_npuir_opt(
        mla_decode_dynamic_kv_zzlayout_probe(),
        tmp_path,
        mlir_filename="mla_decode_dynamic_kv_zzlayout_probe_suvm.mlir",
        expected_tokens=("suvm.bind_layout",),
        opt_args=("--verify-each",),
    )


if __name__ == "__main__":
    tilelang.testing.main()
