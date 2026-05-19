"""Tests for the Sunmmio LegalizeSunmmioDataPath pass."""

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target
from tvm.tir import Block, BufferLoad, Call, Ramp
from tvm.tir.stmt_functor import post_order_visit

tilelang.env.disable_cache()


def gemm_matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    """Build a GEMM kernel that relies on InferSramScope to infer Sunmmio scopes."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return tvm.IRModule({"main": main})


def flashattn(batch, heads, seq_len, dim, is_causal, groups=1, block_M=64, block_N=64, num_stages=0, threads=128):
    """Build the FlashAttention kernel from the GQA example."""

    scale = (1.0 / dim) ** 0.5 * 1.44269504
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = T.float16
    accum_dtype = T.float32

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
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype))
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_len, -T.infinity(acc_s.dtype), 0)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[bz, k * block_N : (k + 1) * block_N, by // groups, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return tvm.IRModule({"main": main})


def run_legalize_sunmmio_datapath(mod, target):
    """Apply the real pre-split pipeline before running LegalizeSunmmioDataPath."""

    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.InferSramScope()(mod)
    mod = tilelang.transform.LegalizeSunmmioDataPath()(mod)
    return mod


def extract_copy_calls(func):
    """Collect all tl.tileop.copy calls from a PrimFunc."""

    copy_calls = []

    def visit(node):
        if isinstance(node, Call) and node.op == tvm.tir.op.Op.get("tl.tileop.copy"):
            copy_calls.append(node)

    post_order_visit(func.body, visit)
    return copy_calls


def extract_comm_calls(func, op_name="tl.tileop.comm_broadcast"):
    """Collect all calls for a specific comm op from a PrimFunc."""

    comm_calls = []

    def visit(node):
        if isinstance(node, Call) and node.op == tvm.tir.op.Op.get(op_name):
            comm_calls.append(node)

    post_order_visit(func.body, visit)
    return comm_calls


def normalize_comm_region(region):
    """Decode the src/dst region of a comm op call.

    After legalization, rewritten args use tl.tileop.region (Call), while
    untouched args remain as BufferLoad with Ramp indices (representing ranges).
    """

    if isinstance(region, Call) and region.op.name == "tl.tileop.region":
        assert isinstance(region.args[0], BufferLoad)
        return {
            "buffer": region.args[0].buffer,
            "mins": [expr_text(index) for index in region.args[0].indices],
            "access_mask": int(region.args[1]),
            "extents": [expr_text(extent) for extent in region.args[2:]],
        }
    elif isinstance(region, BufferLoad):
        # BufferLoad with Ramp indices: Ramp(base, stride=1, lanes) encodes range [base, base+lanes)
        mins = []
        extents = []
        for index in region.indices:
            if isinstance(index, Ramp):
                mins.append(expr_text(index.base))
                extents.append(expr_text(index.lanes))
            else:
                mins.append(expr_text(index))
                extents.append("1")
        return {
            "buffer": region.buffer,
            "mins": mins,
            "access_mask": None,
            "extents": extents,
        }
    else:
        raise ValueError(f"Unexpected region type: {type(region)}")


def extract_comm_edges(func, op_name="tl.tileop.comm_broadcast"):
    """Collect normalized src/dst metadata for every comm op call."""

    edges = []
    for call in extract_comm_calls(func, op_name):
        src = normalize_comm_region(call.args[0])
        dst = normalize_comm_region(call.args[1])
        edges.append(
            {
                "src_name": src["buffer"].name,
                "dst_name": dst["buffer"].name,
                "src_scope": src["buffer"].scope(),
                "dst_scope": dst["buffer"].scope(),
                "src_mins": src["mins"],
                "dst_mins": dst["mins"],
                "src_extents": src["extents"],
                "dst_extents": dst["extents"],
                "src_access_mask": src["access_mask"],
                "dst_access_mask": dst["access_mask"],
            }
        )
    return edges


def extract_blocks(func):
    """Collect all blocks from a PrimFunc."""

    blocks = []

    def visit(node):
        if isinstance(node, Block):
            blocks.append(node)

    post_order_visit(func.body, visit)
    return blocks


def expr_text(expr):
    """Convert a TIR expression into a stable string for structural assertions."""

    return str(expr)


def normalize_region(region):
    """Decode a tl.tileop.region call."""

    assert isinstance(region, Call)
    assert region.op.name == "tl.tileop.region"
    assert isinstance(region.args[0], BufferLoad)
    return {
        "buffer": region.args[0].buffer,
        "mins": [expr_text(index) for index in region.args[0].indices],
        "access_mask": int(region.args[1]),
        "extents": [expr_text(extent) for extent in region.args[2:]],
    }


def extract_copy_edges(func):
    """Collect normalized source/destination metadata for every tile copy."""

    edges = []
    for call in extract_copy_calls(func):
        src = normalize_region(call.args[0])
        dst = normalize_region(call.args[1])
        edges.append(
            {
                "src_name": src["buffer"].name,
                "dst_name": dst["buffer"].name,
                "src_scope": src["buffer"].scope(),
                "dst_scope": dst["buffer"].scope(),
                "src_mins": src["mins"],
                "dst_mins": dst["mins"],
                "src_extents": src["extents"],
                "dst_extents": dst["extents"],
                "src_access_mask": src["access_mask"],
                "dst_access_mask": dst["access_mask"],
            }
        )
    return edges


def test_legalize_sunmmio_datapath_on_gemm_kernel():
    target = determine_target("Sunmmio", return_object=True)

    with tvm.target.Target(target):
        mod = run_legalize_sunmmio_datapath(gemm_matmul(128, 128, 128, 32, 32, 32), target)

    func = mod["main"]
    root_block = next(block for block in extract_blocks(func) if block.name_hint == "tilelang_root")
    copy_edges = extract_copy_edges(func)
    stage_buffers = [buf for buf in root_block.alloc_buffers if buf.name.startswith("A_rsram_stage")]

    assert len(copy_edges) == 4
    assert len(stage_buffers) == 1
    assert stage_buffers[0].scope() == "shared.rsram"
    assert [int(dim) for dim in stage_buffers[0].shape] == [32, 32]

    a_global_to_stage = [edge for edge in copy_edges if edge["src_name"] == "A" and edge["dst_name"].startswith("A_rsram_stage")]
    a_stage_to_asram = [edge for edge in copy_edges if edge["src_name"].startswith("A_rsram_stage") and edge["dst_name"] == "A_shared"]
    b_direct_copy = [edge for edge in copy_edges if edge["src_name"] == "B" and edge["dst_name"] == "B_shared"]

    assert len(a_global_to_stage) == 1
    assert len(a_stage_to_asram) == 1
    assert len(b_direct_copy) == 1

    assert a_global_to_stage[0]["src_scope"] == "global"
    assert a_global_to_stage[0]["dst_scope"] == "shared.rsram"
    assert a_global_to_stage[0]["dst_mins"] == ["0", "0"]
    assert a_global_to_stage[0]["src_extents"] == ["32", "32"]
    assert a_global_to_stage[0]["dst_extents"] == ["32", "32"]
    assert a_global_to_stage[0]["src_access_mask"] == 1
    assert a_global_to_stage[0]["dst_access_mask"] == 2

    assert a_stage_to_asram[0]["src_scope"] == "shared.rsram"
    assert a_stage_to_asram[0]["dst_scope"] == "shared.asram"
    assert a_stage_to_asram[0]["src_mins"] == ["0", "0"]
    assert a_stage_to_asram[0]["dst_extents"] == ["32", "32"]
    assert a_stage_to_asram[0]["src_access_mask"] == 1
    assert a_stage_to_asram[0]["dst_access_mask"] == 2

    assert b_direct_copy[0]["src_scope"] == "global"
    assert b_direct_copy[0]["dst_scope"] == "shared.wsram"
    assert b_direct_copy[0]["dst_extents"] == ["32", "32"]
    assert b_direct_copy[0]["src_access_mask"] == 1
    assert b_direct_copy[0]["dst_access_mask"] == 2


def test_legalize_sunmmio_datapath_on_flashattn_kernel():
    target = determine_target("Sunmmio", return_object=True)

    with tvm.target.Target(target):
        mod = run_legalize_sunmmio_datapath(
            flashattn(1, 4, 128, 32, False, groups=1, block_M=32, block_N=32, num_stages=1, threads=128),
            target,
        )

    func = mod["main"]
    root_block = next(block for block in extract_blocks(func) if block.name_hint == "tilelang_root")
    copy_edges = extract_copy_edges(func)
    stage_buffers = [buf for buf in root_block.alloc_buffers if buf.name.startswith("Q_rsram_stage")]
    k_direct_copies = [edge for edge in copy_edges if edge["src_name"] == "K" and edge["dst_name"] == "K_shared"]
    v_direct_copies = [edge for edge in copy_edges if edge["src_name"] == "V" and edge["dst_name"] == "V_shared"]
    q_global_to_stage = [edge for edge in copy_edges if edge["src_name"] == "Q" and edge["dst_name"].startswith("Q_rsram_stage")]
    q_stage_to_asram = [edge for edge in copy_edges if edge["src_name"].startswith("Q_rsram_stage") and edge["dst_name"] == "Q_shared"]

    assert len(stage_buffers) == 1
    assert stage_buffers[0].scope() == "shared.rsram"
    assert [int(dim) for dim in stage_buffers[0].shape] == [1, 32, 1, 32]

    assert len(q_global_to_stage) == 1
    assert len(q_stage_to_asram) == 1
    assert q_global_to_stage[0]["src_scope"] == "global"
    assert q_global_to_stage[0]["dst_scope"] == "shared.rsram"
    assert q_global_to_stage[0]["dst_mins"] == ["0", "0", "0", "0"]
    assert q_global_to_stage[0]["src_extents"] == ["1", "32", "1", "32"]
    assert q_global_to_stage[0]["dst_extents"] == ["1", "32", "1", "32"]
    assert q_global_to_stage[0]["src_access_mask"] == 1
    assert q_global_to_stage[0]["dst_access_mask"] == 2
    assert q_stage_to_asram[0]["src_scope"] == "shared.rsram"
    assert q_stage_to_asram[0]["dst_scope"] == "shared.asram"
    assert q_stage_to_asram[0]["src_mins"] == ["0", "0", "0", "0"]
    assert q_stage_to_asram[0]["src_extents"] == ["1", "32", "1", "32"]
    assert q_stage_to_asram[0]["dst_extents"] == ["32", "32"]
    assert q_stage_to_asram[0]["src_access_mask"] == 1
    assert q_stage_to_asram[0]["dst_access_mask"] == 2

    assert k_direct_copies
    assert v_direct_copies
    assert all(edge["dst_scope"] == "shared.wsram" for edge in k_direct_copies)
    assert all(edge["dst_scope"] == "shared.wsram" for edge in v_direct_copies)
    assert not [buf for buf in root_block.alloc_buffers if buf.name.startswith("K_shared_rsram_stage")]
    assert not [buf for buf in root_block.alloc_buffers if buf.name.startswith("V_shared_rsram_stage")]


def gemm_broadcast_from_dram(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    """Build a GEMM kernel where A is loaded via broadcast from DRAM.

    A is broadcast from DRAM to A_shared, which is used as the left operand
    of T.gemm. InferSramScope should assign A_shared to shared.asram,
    triggering DRAM->ASRAM legalization for the broadcast.
    """

    @T.prim_func
    def main(
        A: T.Tensor((block_M, block_K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.clear(C_shared)
            T.comm.broadcast(A, A_shared, (0, 0), direction="h")
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return tvm.IRModule({"main": main})


def gemm_put_from_dram(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    """Build a GEMM kernel where A is loaded via put from DRAM.

    A is put from DRAM (core 0,0) to A_shared (core 1,1), and A_shared is used
    as the left operand of T.gemm. InferSramScope should assign A_shared to
    shared.asram, triggering DRAM->ASRAM legalization for the put.
    """

    @T.prim_func
    def main(
        A: T.Tensor((block_M, block_K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.clear(C_shared)
            T.comm.put(A, A_shared, (0, 0), (1, 1))
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return tvm.IRModule({"main": main})


def test_legalize_sunmmio_datapath_on_broadcast_from_dram():
    """Broadcast from DRAM to ASRAM should be split: copy global->rsram + broadcast rsram->asram."""
    target = determine_target("Sunmmio", return_object=True)

    with tvm.target.Target(target):
        mod = run_legalize_sunmmio_datapath(gemm_broadcast_from_dram(128, 128, 128, 32, 32, 32), target)

    func = mod["main"]
    root_block = next(block for block in extract_blocks(func) if block.name_hint == "tilelang_root")
    copy_edges = extract_copy_edges(func)
    comm_edges = extract_comm_edges(func, "tl.tileop.comm_broadcast")

    # Should have an RSRAM staging buffer for A
    stage_buffers = [buf for buf in root_block.alloc_buffers if buf.name.startswith("A_rsram_stage")]
    assert len(stage_buffers) == 1
    assert stage_buffers[0].scope() == "shared.rsram"
    assert [int(dim) for dim in stage_buffers[0].shape] == [32, 32]

    # Should have a copy: global A -> RSRAM staging
    a_global_to_stage = [edge for edge in copy_edges if edge["src_name"] == "A" and edge["dst_name"].startswith("A_rsram_stage")]
    assert len(a_global_to_stage) == 1
    assert a_global_to_stage[0]["src_scope"] == "global"
    assert a_global_to_stage[0]["dst_scope"] == "shared.rsram"
    assert a_global_to_stage[0]["src_access_mask"] == 1
    assert a_global_to_stage[0]["dst_access_mask"] == 2

    # Broadcast should now be from RSRAM staging -> ASRAM (not global -> ASRAM)
    a_broadcast = [edge for edge in comm_edges if edge["dst_name"] == "A_shared"]
    assert len(a_broadcast) == 1
    assert a_broadcast[0]["src_name"].startswith("A_rsram_stage")
    assert a_broadcast[0]["src_scope"] == "shared.rsram"
    assert a_broadcast[0]["dst_scope"] == "shared.asram"

    # No broadcast should remain with global source
    global_broadcasts = [edge for edge in comm_edges if edge["src_scope"] == "global"]
    assert len(global_broadcasts) == 0


def test_legalize_sunmmio_datapath_on_put_from_dram():
    """Put from DRAM to ASRAM should be split: copy global->rsram + put rsram->asram."""
    target = determine_target("Sunmmio", return_object=True)

    with tvm.target.Target(target):
        mod = run_legalize_sunmmio_datapath(gemm_put_from_dram(128, 128, 128, 32, 32, 32), target)

    func = mod["main"]
    root_block = next(block for block in extract_blocks(func) if block.name_hint == "tilelang_root")
    copy_edges = extract_copy_edges(func)
    # PutOp is lowered as comm_broadcast calls by LowerTileOp, but at this stage
    # it's still tl.tileop.comm_put
    comm_edges = extract_comm_edges(func, "tl.tileop.comm_put")

    # Should have an RSRAM staging buffer for A
    stage_buffers = [buf for buf in root_block.alloc_buffers if buf.name.startswith("A_rsram_stage")]
    assert len(stage_buffers) == 1
    assert stage_buffers[0].scope() == "shared.rsram"
    assert [int(dim) for dim in stage_buffers[0].shape] == [32, 32]

    # Should have a copy: global A -> RSRAM staging
    a_global_to_stage = [edge for edge in copy_edges if edge["src_name"] == "A" and edge["dst_name"].startswith("A_rsram_stage")]
    assert len(a_global_to_stage) == 1
    assert a_global_to_stage[0]["src_scope"] == "global"
    assert a_global_to_stage[0]["dst_scope"] == "shared.rsram"
    assert a_global_to_stage[0]["src_access_mask"] == 1
    assert a_global_to_stage[0]["dst_access_mask"] == 2

    # Put should now be from RSRAM staging -> ASRAM (not global -> ASRAM)
    a_put = [edge for edge in comm_edges if edge["dst_name"] == "A_shared"]
    assert len(a_put) == 1
    assert a_put[0]["src_name"].startswith("A_rsram_stage")
    assert a_put[0]["src_scope"] == "shared.rsram"
    assert a_put[0]["dst_scope"] == "shared.asram"


def allgather_dram_to_asram(block_M, block_K, dtype=T.float16):
    """Build a kernel where all_gather goes from DRAM to an explicitly ASRAM-scoped buffer.

    The recv buffer is allocated with explicit scope="shared.asram" to bypass
    InferSramScope (which wouldn't naturally assign a 3D buffer to ASRAM).
    This directly tests the legalize pass on the all_gather DRAM->ASRAM path.

    For horizontal all_gather with 4 cols: send [block_M, block_K] -> recv [4, block_M, block_K].
    """

    @T.prim_func
    def main(
        A: T.Tensor((block_M, block_K), dtype),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_gathered = T.alloc_shared([4, block_M, block_K], dtype, scope="shared.asram")
            T.comm.all_gather(A, A_gathered, direction="h")

    return tvm.IRModule({"main": main})


def annotated_copy_dram_to_asram(block_M, block_K, dtype=T.float16):
    """Build a DRAM->ASRAM copy carrying copy annotations."""

    @T.prim_func
    def main(
        A: T.Tensor((block_M, block_K), dtype),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared.asram")
            T.copy(A, A_shared, coalesced_width=4, disable_tma=True, eviction_policy="evict_last")

    return tvm.IRModule({"main": main})


def run_legalize_sunmmio_datapath_no_infer_scope(mod, target):
    """Apply passes before LegalizeSunmmioDataPath, skipping InferSramScope.

    Use this when the kernel already has explicit scope assignments.
    """

    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LegalizeSunmmioDataPath()(mod)
    return mod


def test_legalize_sunmmio_datapath_preserves_copy_annotations():
    """Staged copies should preserve annotations from the original T.copy."""
    target = determine_target("Sunmmio", return_object=True)

    with tvm.target.Target(target):
        mod = run_legalize_sunmmio_datapath_no_infer_scope(annotated_copy_dram_to_asram(32, 32), target)

    copy_calls = extract_copy_calls(mod["main"])
    assert len(copy_calls) == 2
    for call in copy_calls:
        annotations = call.annotations
        assert int(annotations["coalesced_width"]) == 4
        assert bool(annotations["disable_tma"])
        assert int(annotations["eviction_policy"]) == 2


def test_legalize_sunmmio_datapath_on_allgather_from_dram():
    """All_gather from DRAM to ASRAM should be split: copy global->rsram + allgather rsram->asram."""
    target = determine_target("Sunmmio", return_object=True)

    with tvm.target.Target(target):
        mod = run_legalize_sunmmio_datapath_no_infer_scope(allgather_dram_to_asram(32, 32), target)

    func = mod["main"]
    root_block = next(block for block in extract_blocks(func) if block.name_hint == "tilelang_root")
    copy_edges = extract_copy_edges(func)
    comm_edges = extract_comm_edges(func, "tl.tileop.comm_allgather")

    # Should have an RSRAM staging buffer for A (matching send shape [32, 32])
    stage_buffers = [buf for buf in root_block.alloc_buffers if buf.name.startswith("A_rsram_stage")]
    assert len(stage_buffers) == 1
    assert stage_buffers[0].scope() == "shared.rsram"
    assert [int(dim) for dim in stage_buffers[0].shape] == [32, 32]

    # Should have a copy: global A -> RSRAM staging
    a_global_to_stage = [edge for edge in copy_edges if edge["src_name"] == "A" and edge["dst_name"].startswith("A_rsram_stage")]
    assert len(a_global_to_stage) == 1
    assert a_global_to_stage[0]["src_scope"] == "global"
    assert a_global_to_stage[0]["dst_scope"] == "shared.rsram"
    assert a_global_to_stage[0]["src_access_mask"] == 1
    assert a_global_to_stage[0]["dst_access_mask"] == 2

    # Allgather should now be from RSRAM staging -> ASRAM (not global -> ASRAM)
    a_allgather = [edge for edge in comm_edges if edge["dst_name"] == "A_gathered"]
    assert len(a_allgather) == 1
    assert a_allgather[0]["src_name"].startswith("A_rsram_stage")
    assert a_allgather[0]["src_scope"] == "shared.rsram"
    assert a_allgather[0]["dst_scope"] == "shared.asram"
    # Staging buffer should match send shape, not recv shape
    assert a_allgather[0]["src_extents"] == ["32", "32"]
    assert a_allgather[0]["dst_extents"] == ["4", "32", "32"]

    # No allgather should remain with global source
    global_allgathers = [edge for edge in comm_edges if edge["src_scope"] == "global"]
    assert len(global_allgathers) == 0
