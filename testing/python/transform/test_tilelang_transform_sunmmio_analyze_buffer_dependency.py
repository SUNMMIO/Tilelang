import tilelang
import tilelang as tl
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from tvm import tir
from tvm import IRModule
from tvm.tir import stmt_functor


def apply_sunmmio_passes(mod, target):
    """Apply the full SUNMMIO pass pipeline used for DMA copy lowering."""
    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.InferSramScope()(mod)
    mod = tilelang.transform.LayoutReducer()(mod)
    mod = tilelang.transform.LayoutInference()(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = tl.transform.LegalizeTilesLoop()(mod)
    mod = tl.transform.TilesLoop()(mod)
    mod = tl.transform.AnalyzePipelinedBufferDependency()(mod)
    return mod


def _find_analyzed_pipeline_loop(stmt):
    loops = []

    def _visit(node):
        if isinstance(node, tir.For) and "tl_buffer_dependency_state_buffers" in node.annotations:
            loops.append(node)

    stmt_functor.post_order_visit(stmt, _visit)
    assert len(loops) == 1, f"Expected exactly one analyzed pipeline loop, got {len(loops)}"
    return loops[0]


def _annotation_strings(loop: tir.For, key: str) -> list[str]:
    values = loop.annotations[key]
    return [value.value if hasattr(value, "value") else str(value) for value in values]


def _structured_analysis(loop: tir.For):
    assert "tl_buffer_dependency_analysis" in loop.annotations
    return loop.annotations["tl_buffer_dependency_analysis"]


def flashattn(batch, heads, seq_len, dim, groups, block_M, block_N, num_stages):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
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
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch) as (bx, by, bz):
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
            for i in T.Tiles(acc_o):
                acc_o[i] = 0.0
            for i in T.Tiles(logsum):
                logsum[i] = 0.0
            for i in T.Tiles(scores_max):
                scores_max[i] = -T.infinity(accum_dtype)

            loop_range = (
                T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N))
            )
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by // groups, :], K_shared)
                for i, j in T.Tiles(acc_s):
                    acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype))
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)

                for i in T.Tiles(scores_max):
                    scores_max_prev[i] = scores_max[i]
                for i in T.Tiles(scores_max):
                    scores_max[i] = -T.infinity(accum_dtype)

                for i in T.Serial(block_M):
                    for j in T.Serial(block_N):
                        scores_max[i] = T.max(scores_max[i], acc_s[i, j])

                for i in T.Tiles(scores_max):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Tiles(scores_scale):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Tiles(acc_s):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)

                for i in T.Tiles(scores_sum):
                    scores_sum[i] = 0.0
                for i in T.Serial(block_M):
                    for j in T.Serial(block_N):
                        scores_sum[i] += acc_s[i, j]

                for i in T.Tiles(logsum):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Tiles(acc_s):
                    acc_s_cast[i, j] = T.cast(acc_s[i, j], dtype)

                for i, j in T.Tiles(acc_o):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[bz, k * block_N : (k + 1) * block_N, by // groups, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Tiles(acc_o):
                acc_o[i, j] /= logsum[i]
            for i, j in T.Tiles(acc_o):
                O_shared[i, j] = T.cast(acc_o[i, j], dtype)
            T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main


def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32, num_stages=2):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            acc_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, acc_shared)

            for i, j in T.Tiles(C_shared):
                C_shared[i, j] = T.cast(acc_shared[i, j], dtype)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def partial_region_mma(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32, num_stages=3):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm_v1(
                    A_shared[0:8, 16:32],
                    B_shared[0:16, 8:16],
                    C_shared[8:24, 16:32],
                    transpose_A=True,
                    transpose_B=True,
                )

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def clear_accum_matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32, num_stages=2):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared, clear_accum=True)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


@T.prim_func
def block_internal_sequence(A: T.Buffer((4,), "float32"), C: T.Buffer((1,), "float32")):
    with T.block("root"):
        tmp = T.alloc_buffer((1,), "float32", scope="shared")
        state = T.alloc_buffer((1,), "float32", scope="shared")
        state[0] = T.float32(0)
        for k in T.serial(4, annotations={"num_stages": 2}):
            with T.block("step"):
                tmp[0] = T.float32(0)
                tmp[0] = tmp[0] + A[k]
                state[0] = state[0] + tmp[0]
        C[0] = state[0]


@T.prim_func
def conditional_branch_sequence(A: T.Buffer((4,), "float32"), C: T.Buffer((1,), "float32")):
    with T.block("root"):
        tmp = T.alloc_buffer((1,), "float32", scope="shared")
        state = T.alloc_buffer((1,), "float32", scope="shared")
        state[0] = T.float32(0)
        for k in T.serial(4, annotations={"num_stages": 2}):
            if k % 2 == 0:
                tmp[0] = A[k]
            else:
                tmp[0] = A[k] + T.float32(1)
            state[0] = state[0] + tmp[0]
        C[0] = state[0]


def partial_overwrite_full_region(K=4):
    @T.prim_func
    def main(A: T.Tensor((K, 2), "float32"), C: T.Tensor((1,), "float32")):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            carry = T.alloc_shared((2,), "float32")
            total = T.alloc_shared((1,), "float32")

            total[0] = T.float32(0)
            carry[0] = T.float32(0)
            carry[1] = T.float32(0)
            for k in T.Pipelined(K, num_stages=2):
                carry[0] = T.float32(0)
                total[0] = total[0] + carry[1]
                T.copy(A[k, 0:2], carry)

            T.copy(total, C[0:1])

    return main


@T.prim_func
def mixed_region_roles(A: T.Buffer((4,), "float32"), C: T.Buffer((1,), "float32")):
    with T.block("root"):
        mix = T.alloc_buffer((2,), "float32", scope="shared")
        state = T.alloc_buffer((1,), "float32", scope="shared")
        mix[1] = T.float32(0)
        state[0] = T.float32(0)
        for k in T.serial(4, annotations={"num_stages": 2}):
            mix[0] = A[k]
            state[0] = state[0] + mix[1]
            mix[1] = state[0]
        C[0] = state[0]


@T.prim_func
def covered_nested_rewrite(A: T.Buffer((4,), "float32"), C: T.Buffer((1,), "float32")):
    with T.block("root"):
        scale = T.alloc_buffer((1,), "float32", scope="shared")
        acc = T.alloc_buffer((4,), "float32", scope="shared")
        state = T.alloc_buffer((1,), "float32", scope="shared")
        for i in T.serial(4):
            acc[i] = T.float32(1)
        state[0] = T.float32(0)
        for k in T.serial(4, annotations={"num_stages": 2}):
            scale[0] = A[k]
            for i in T.serial(4):
                acc[i] = acc[i] * scale[0]
            state[0] = state[0] + acc[0]
        C[0] = state[0]


@T.prim_func
def reduction_block_init(A: T.Buffer((4, 4), "float32"), C: T.Buffer((1,), "float32")):
    with T.block("root"):
        acc = T.alloc_buffer((1,), "float32", scope="shared")
        state = T.alloc_buffer((1,), "float32", scope="shared")
        state[0] = T.float32(0)
        for k in T.serial(4, annotations={"num_stages": 2}):
            for j in T.serial(4):
                with T.block("sum"):
                    vj = T.axis.reduce(4, j)
                    T.reads(A[k, vj], acc[0])
                    T.writes(acc[0])
                    with T.init():
                        acc[0] = T.float32(0)
                    acc[0] = acc[0] + A[k, vj]
            state[0] = state[0] + acc[0]
        C[0] = state[0]


@T.prim_func
def unknown_opaque_effect(A: T.Buffer((4,), "float32"), C: T.Buffer((4,), "float32")):
    with T.block("root"):
        tmp = T.alloc_buffer((1,), "float32", scope="shared")
        for k in T.serial(4, annotations={"num_stages": 2}):
            tmp[0] = A[k]
            T.evaluate(T.call_extern("handle", "mystery_runtime"))
            C[k] = tmp[0]


def test_flashattn_analysis():
    target = SUNMMIO_TARGET_DESC
    mod = IRModule.from_expr(flashattn(32, 64, 128, 256, 8, 64, 64, 2))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
        loop = _find_analyzed_pipeline_loop(mod["main"].body)
        state_buffers = set(_annotation_strings(loop, "tl_buffer_dependency_state_buffers"))
        channel_buffers = set(_annotation_strings(loop, "tl_buffer_dependency_channel_buffers"))
        intra_raw_edges = _annotation_strings(loop, "tl_buffer_dependency_intra_raw")
        inter_raw_edges = _annotation_strings(loop, "tl_buffer_dependency_inter_raw")
        covered_rewrites = _annotation_strings(loop, "tl_buffer_dependency_covered_rewrites")

        assert "tl_buffer_dependency_analysis" in loop.annotations
        assert state_buffers == {"acc_o", "logsum", "scores_max"}
        assert channel_buffers == {"K_shared", "V_shared", "acc_s", "acc_s_cast", "scores_max_prev", "scores_scale", "scores_sum"}
        assert state_buffers.isdisjoint(channel_buffers)
        assert any("acc_o" in detail for detail in covered_rewrites)
        assert any("K_shared" in edge for edge in intra_raw_edges)
        assert any("V_shared" in edge for edge in intra_raw_edges)
        assert any("scores_max" in edge for edge in inter_raw_edges)
        assert any("logsum" in edge for edge in inter_raw_edges)
        assert any("acc_o" in edge for edge in inter_raw_edges)
        assert not any("scores_sum" in edge for edge in inter_raw_edges)


def test_matmul_analysis():
    target = SUNMMIO_TARGET_DESC
    mod = IRModule.from_expr(matmul(128, 128, 128, 64, 64, 32, num_stages=2))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
        loop = _find_analyzed_pipeline_loop(mod["main"].body)
        state_buffers = set(_annotation_strings(loop, "tl_buffer_dependency_state_buffers"))
        channel_buffers = set(_annotation_strings(loop, "tl_buffer_dependency_channel_buffers"))
        intra_raw_edges = _annotation_strings(loop, "tl_buffer_dependency_intra_raw")
        inter_raw_edges = _annotation_strings(loop, "tl_buffer_dependency_inter_raw")

        assert "tl_buffer_dependency_analysis" in loop.annotations
        assert state_buffers == {"acc_shared"}
        assert channel_buffers == {"A_shared", "B_shared"}
        assert state_buffers.isdisjoint(channel_buffers)
        assert any("A_shared" in edge for edge in intra_raw_edges)
        assert any("B_shared" in edge for edge in intra_raw_edges)
        assert any("acc_shared" in edge for edge in inter_raw_edges)


def test_partial_region_mma_analysis():
    target = SUNMMIO_TARGET_DESC
    mod = IRModule.from_expr(partial_region_mma(128, 128, 128, 32, 32, 32))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
        loop = _find_analyzed_pipeline_loop(mod["main"].body)
        state_buffers = set(_annotation_strings(loop, "tl_buffer_dependency_state_buffers"))
        channel_buffers = set(_annotation_strings(loop, "tl_buffer_dependency_channel_buffers"))
        intra_raw_edges = _annotation_strings(loop, "tl_buffer_dependency_intra_raw")
        inter_raw_edges = _annotation_strings(loop, "tl_buffer_dependency_inter_raw")

        assert "tl_buffer_dependency_analysis" in loop.annotations
        assert state_buffers == {"C_shared"}
        assert channel_buffers == {"A_shared", "B_shared"}
        assert state_buffers.isdisjoint(channel_buffers)
        assert any("A_shared" in edge and "read=A_shared[0:8, 16:32]" in edge for edge in intra_raw_edges)
        assert any("B_shared" in edge and "read=B_shared[0:16, 8:16]" in edge for edge in intra_raw_edges)
        assert any("C_shared" in edge and "write=C_shared[8:24, 16:32]" in edge for edge in inter_raw_edges)


def test_clear_accum_true_analysis():
    target = SUNMMIO_TARGET_DESC
    mod = IRModule.from_expr(clear_accum_matmul(128, 128, 128, 64, 64, 32))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
        loop = _find_analyzed_pipeline_loop(mod["main"].body)
        state_buffers = set(_annotation_strings(loop, "tl_buffer_dependency_state_buffers"))
        channel_buffers = set(_annotation_strings(loop, "tl_buffer_dependency_channel_buffers"))
        inter_raw_edges = _annotation_strings(loop, "tl_buffer_dependency_inter_raw")

        assert "tl_buffer_dependency_analysis" in loop.annotations
        assert state_buffers == set()
        assert channel_buffers == {"A_shared", "B_shared", "C_shared"}
        assert not inter_raw_edges


def test_block_internal_sequence_analysis():
    mod = IRModule.from_expr(block_internal_sequence)
    mod = tl.transform.AnalyzePipelinedBufferDependency()(mod)
    loop = _find_analyzed_pipeline_loop(mod["block_internal_sequence"].body)
    state_buffers = set(_annotation_strings(loop, "tl_buffer_dependency_state_buffers"))
    channel_buffers = set(_annotation_strings(loop, "tl_buffer_dependency_channel_buffers"))
    intra_raw_edges = _annotation_strings(loop, "tl_buffer_dependency_intra_raw")
    inter_raw_edges = _annotation_strings(loop, "tl_buffer_dependency_inter_raw")

    assert "tl_buffer_dependency_analysis" in loop.annotations
    assert state_buffers == {"state"}
    assert channel_buffers == {"tmp"}
    assert state_buffers.isdisjoint(channel_buffers)
    assert any("tmp" in edge for edge in intra_raw_edges)
    assert any("state" in edge for edge in inter_raw_edges)
    assert not any("tmp" in edge for edge in inter_raw_edges)


def test_conditional_branch_sequence_analysis():
    mod = IRModule.from_expr(conditional_branch_sequence)
    mod = tl.transform.AnalyzePipelinedBufferDependency()(mod)
    loop = _find_analyzed_pipeline_loop(mod["conditional_branch_sequence"].body)
    state_buffers = set(_annotation_strings(loop, "tl_buffer_dependency_state_buffers"))
    channel_buffers = set(_annotation_strings(loop, "tl_buffer_dependency_channel_buffers"))
    intra_raw_edges = _annotation_strings(loop, "tl_buffer_dependency_intra_raw")
    inter_raw_edges = _annotation_strings(loop, "tl_buffer_dependency_inter_raw")

    assert "tl_buffer_dependency_analysis" in loop.annotations
    assert state_buffers == {"state"}
    assert channel_buffers == {"tmp"}
    assert state_buffers.isdisjoint(channel_buffers)
    assert any("tmp" in edge for edge in intra_raw_edges)
    assert any("state" in edge for edge in inter_raw_edges)
    assert not any("tmp" in edge for edge in inter_raw_edges)

def test_partial_overwrite_full_region_detection():
    target = SUNMMIO_TARGET_DESC
    mod = IRModule.from_expr(partial_overwrite_full_region())
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
        loop = _find_analyzed_pipeline_loop(mod["main"].body)
        state_buffers = set(_annotation_strings(loop, "tl_buffer_dependency_state_buffers"))
        channel_buffers = set(_annotation_strings(loop, "tl_buffer_dependency_channel_buffers"))
        partial_overwrite_hazards = _annotation_strings(loop, "tl_buffer_dependency_partial_overwrite_hazards")
        covered_rewrites = _annotation_strings(loop, "tl_buffer_dependency_covered_rewrites")

        assert "tl_buffer_dependency_analysis" in loop.annotations
        assert state_buffers == {"total"}
        assert "carry" in channel_buffers
        assert not any("carry" in detail for detail in covered_rewrites)
        assert any("carry" in detail for detail in partial_overwrite_hazards)

def test_mixed_region_roles_detection():
    mod = IRModule.from_expr(mixed_region_roles)
    mod = tl.transform.AnalyzePipelinedBufferDependency()(mod)
    loop = _find_analyzed_pipeline_loop(mod["mixed_region_roles"].body)
    state_buffers = set(_annotation_strings(loop, "tl_buffer_dependency_state_buffers"))
    channel_buffers = set(_annotation_strings(loop, "tl_buffer_dependency_channel_buffers"))
    mixed_role_buffers = set(_annotation_strings(loop, "tl_buffer_dependency_mixed_role_buffers"))
    mixed_role_details = _annotation_strings(loop, "tl_buffer_dependency_mixed_role_details")

    assert "tl_buffer_dependency_analysis" in loop.annotations
    assert "mix" in state_buffers
    assert "mix" not in channel_buffers
    assert mixed_role_buffers == {"mix"}
    assert any("mix" in detail for detail in mixed_role_details)


def test_covered_nested_rewrite_proof():
    mod = IRModule.from_expr(covered_nested_rewrite)
    mod = tl.transform.AnalyzePipelinedBufferDependency()(mod)
    loop = _find_analyzed_pipeline_loop(mod["covered_nested_rewrite"].body)
    state_buffers = set(_annotation_strings(loop, "tl_buffer_dependency_state_buffers"))
    channel_buffers = set(_annotation_strings(loop, "tl_buffer_dependency_channel_buffers"))
    covered_rewrites = _annotation_strings(loop, "tl_buffer_dependency_covered_rewrites")
    partial_overwrite_hazards = _annotation_strings(loop, "tl_buffer_dependency_partial_overwrite_hazards")

    assert "tl_buffer_dependency_analysis" in loop.annotations
    assert state_buffers == {"acc", "state"}
    assert channel_buffers == {"scale"}
    assert state_buffers.isdisjoint(channel_buffers)
    assert not partial_overwrite_hazards
    assert any("acc" in detail and "cover=acc[0:4]" in detail for detail in covered_rewrites)


def test_downstream_structured_analysis_example():
    mod = IRModule.from_expr(mixed_region_roles)
    mod = tl.transform.AnalyzePipelinedBufferDependency()(mod)
    loop = _find_analyzed_pipeline_loop(mod["mixed_region_roles"].body)
    analysis = _structured_analysis(loop)

    buffer_roles = {
        info.buffer.name: {
            "state": [str(region) for region in info.state_regions],
            "channel": [str(region) for region in info.channel_regions],
        }
        for info in analysis.buffers
    }
    loop_carried_buffers = {
        edge.buffer.name
        for edge in analysis.edges
        if edge.dep_kind == "RAW" and int(edge.distance) > 0
    }
    special_handling_buffers = {
        pattern.buffer.name
        for pattern in analysis.patterns
        if pattern.kind in {"mixed_role_regions", "partial_overwrite_remainder_read"}
    }
    plain_state_buffers = loop_carried_buffers - special_handling_buffers
    plain_channel_buffers = {
        name
        for name, roles in buffer_roles.items()
        if roles["channel"] and name not in special_handling_buffers
    }

    assert buffer_roles["mix"]["state"] == ["mix[1]"]
    assert buffer_roles["mix"]["channel"] == ["mix[0]"]
    assert buffer_roles["state"]["state"] == ["state[0]"]
    assert loop_carried_buffers == {"mix", "state"}
    assert special_handling_buffers == {"mix"}
    assert plain_state_buffers == {"state"}
    assert plain_channel_buffers == set()


def test_reduction_block_init_analysis():
    mod = IRModule.from_expr(reduction_block_init)
    mod = tl.transform.AnalyzePipelinedBufferDependency()(mod)
    loop = _find_analyzed_pipeline_loop(mod["reduction_block_init"].body)
    state_buffers = set(_annotation_strings(loop, "tl_buffer_dependency_state_buffers"))
    channel_buffers = set(_annotation_strings(loop, "tl_buffer_dependency_channel_buffers"))
    inter_raw_edges = _annotation_strings(loop, "tl_buffer_dependency_inter_raw")

    assert "tl_buffer_dependency_analysis" in loop.annotations
    assert state_buffers == {"state"}
    assert channel_buffers == {"acc"}
    assert state_buffers.isdisjoint(channel_buffers)
    assert any("state" in edge for edge in inter_raw_edges)
    assert not any("acc" in edge for edge in inter_raw_edges)


def test_unknown_opaque_effect_detection():
    mod = IRModule.from_expr(unknown_opaque_effect)
    mod = tl.transform.AnalyzePipelinedBufferDependency()(mod)
    loop = _find_analyzed_pipeline_loop(mod["unknown_opaque_effect"].body)
    analysis = _structured_analysis(loop)
    unknown_effects = _annotation_strings(loop, "tl_buffer_dependency_unknown_effects")

    assert any("tir.call_extern:mystery_runtime" in detail for detail in unknown_effects)
    assert any(pattern.kind == "unknown_effect" for pattern in analysis.patterns)
    assert any(
        pattern.kind == "unknown_effect" and "tir.call_extern:mystery_runtime" in pattern.detail
        for pattern in analysis.patterns
    )
