import tilelang
import pytest
from tilelang import tvm as tvm
from tilelang.layout.hierarchical_layout import make_blockwise_zz_layout
from tilelang.utils.target import determine_target
import tilelang as tl
import tilelang.language as T
from tvm.tir.stmt_functor import post_order_visit
from tvm.tir import BufferLoad, BufferStore, Buffer, Block
from typing import Set
from tilelang.tileview import make_tileview

tilelang.env.disable_cache()


def extract_buffers_from_kernel(func) -> Set[Buffer]:
    """Extract all buffers used in a TIR PrimFunc."""
    used_buffers = set()

    def visit_buffer_access(node):
        if isinstance(node, (BufferLoad, BufferStore)):
            used_buffers.add(node.buffer)

    # Visit function body to find buffer accesses
    post_order_visit(func.body, visit_buffer_access)

    # Also collect allocated buffers from function parameters
    for param in func.params:
        if param in func.buffer_map:
            used_buffers.add(func.buffer_map[param])

    return used_buffers


def extract_block_attrs_from_kernel(func):
    """Extract all block attrs used in a TIR PrimFunc."""
    attrs = []

    def visit_block_attr_access(node):
        if isinstance(node, Block):
            attrs.append(node.annotations)

    post_order_visit(func.body, visit_block_attr_access)

    return attrs


def gemm_matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
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


def gemm_matmul_specify_all_correct_scope(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared.asram")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared.wsram")
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype, scope="shared.rsram")

            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return tvm.IRModule({"main": main})


def gemm_matmul_specify_A_scope(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared.asram")
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return tvm.IRModule({"main": main})


def gemm_matmul_specify_B_scope(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared.wsram")
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return tvm.IRModule({"main": main})


def gemm_matmul_specify_C_scope(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype, scope="shared.rsram")

            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return tvm.IRModule({"main": main})


def example_kernel(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
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

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return tvm.IRModule({"main": main})


CORRECT_TEST_CASES = [
    (
        gemm_matmul(128, 128, 128, 32, 32, 32),
        {
            "A": "global",
            "B": "global",
            "C": "global",
            # T.gemm(A_shared, B_shared, C_shared)
            "A_shared": "shared.asram",
            "B_shared": "shared.wsram",
            "C_shared": "shared.rsram",
        },
    ),
    (
        gemm_matmul_specify_all_correct_scope(128, 128, 128, 32, 32, 32),
        {
            "A": "global",
            "B": "global",
            "C": "global",
            # T.gemm(A_shared, B_shared, C_shared)
            "A_shared": "shared.asram",
            "B_shared": "shared.wsram",
            "C_shared": "shared.rsram",
        },
    ),
    (
        gemm_matmul_specify_A_scope(128, 128, 128, 32, 32, 32),
        {
            "A": "global",
            "B": "global",
            "C": "global",
            # T.gemm(A_shared, B_shared, C_shared)
            "A_shared": "shared.asram",
            "B_shared": "shared.wsram",
            "C_shared": "shared.rsram",
        },
    ),
    (
        gemm_matmul_specify_B_scope(128, 128, 128, 32, 32, 32),
        {
            "A": "global",
            "B": "global",
            "C": "global",
            # T.gemm(A_shared, B_shared, C_shared)
            "A_shared": "shared.asram",
            "B_shared": "shared.wsram",
            "C_shared": "shared.rsram",
        },
    ),
    (
        gemm_matmul_specify_C_scope(128, 128, 128, 32, 32, 32),
        {
            "A": "global",
            "B": "global",
            "C": "global",
            # T.gemm(A_shared, B_shared, C_shared)
            "A_shared": "shared.asram",
            "B_shared": "shared.wsram",
            "C_shared": "shared.rsram",
        },
    ),
    (
        example_kernel(128, 128, 128, 32, 32, 32),
        {
            "A": "global",
            "B": "global",
            "C": "global",
            # T.gemm(A_shared, B_shared, C_shared)
            "A_shared": "shared.rsram",
            "B_shared": "shared.rsram",
            "C_shared": "shared.rsram",
        },
    ),
]


@pytest.mark.parametrize(
    "kernel, buffer_scope_dict",
    CORRECT_TEST_CASES,
)
def test_tilelang_correct_infer_sram_scope(kernel, buffer_scope_dict):
    target_name = "Sunmmio"
    # target_name = "cuda"
    target = determine_target(target_name, return_object=True)
    with tvm.target.Target(target):
        mod = kernel
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tl.transform.InferSramScope()(mod)

        func = list(mod.functions.values())[0]
        buffers = extract_buffers_from_kernel(func)
        for buf in buffers:
            assert buf.scope() == buffer_scope_dict[buf.name]


# from examples/flash_attention/example_gqa_fwd_bshd.py
def flashattn(batch, heads, seq_len, dim, is_causal, groups=1, block_M=64, block_N=64, num_stages=0, threads=128):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = "float16"
    accum_dtype = "float"

    @T.macro
    def MMA0(
        K: T.Tensor(kv_shape, dtype),
        Q_shared: T.SharedBuffer([block_M, dim], dtype),
        K_shared: T.SharedBuffer([block_N, dim], dtype),
        acc_s: T.SharedBuffer([block_M, block_N], accum_dtype),
        k: T.int32,
        bx: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        T.copy(K[bz, k * block_N : (k + 1) * block_N, by // groups, :], K_shared)
        if is_causal:
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype))
        else:
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_len, -T.infinity(acc_s.dtype), 0)
        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def MMA1(
        V: T.Tensor(kv_shape, dtype),
        V_shared: T.SharedBuffer([block_N, dim], dtype),
        acc_s_cast: T.SharedBuffer([block_M, block_N], dtype),
        acc_o: T.SharedBuffer([block_M, dim], accum_dtype),
        k: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        T.copy(V[bz, k * block_N : (k + 1) * block_N, by // groups, :], V_shared)
        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def Softmax(
        acc_s: T.SharedBuffer([block_M, block_N], accum_dtype),
        acc_s_cast: T.SharedBuffer([block_M, block_N], dtype),
        scores_max: T.SharedBuffer([block_M], accum_dtype),
        scores_max_prev: T.SharedBuffer([block_M], accum_dtype),
        scores_scale: T.SharedBuffer([block_M], accum_dtype),
        scores_sum: T.SharedBuffer([block_M], accum_dtype),
        logsum: T.SharedBuffer([block_M], accum_dtype),
    ):
        T.copy(scores_max, scores_max_prev)
        T.fill(scores_max, -T.infinity(accum_dtype))
        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
        for i in T.Parallel(block_M):
            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
        # To do causal softmax, we need to set the scores_max to 0 if it is -inf
        # This process is called Check_inf in FlashAttention3 code, and it only need to be done
        # in the first ceil_div(kBlockM, kBlockN) steps.
        # for i in T.Parallel(block_M):
        #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
        for i in T.Parallel(block_M):
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
        for i, j in T.Parallel(block_M, block_N):
            # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            # max * log_2(e)) This allows the compiler to use the ffma
            # instruction instead of fadd and fmul separately.
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
        T.reduce_sum(acc_s, scores_sum, dim=1)
        for i in T.Parallel(block_M):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
        T.copy(acc_s, acc_s_cast)

    @T.macro
    def Rescale(
        acc_o: T.SharedBuffer([block_M, dim], accum_dtype),
        scores_scale: T.SharedBuffer([block_M], accum_dtype),
    ):
        for i, j in T.Parallel(block_M, dim):
            acc_o[i, j] *= scores_scale[i]

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
                MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)
                Rescale(acc_o, scores_scale)
                MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return tvm.IRModule({"main": main})


# flashattn fail for the following reason:
#
# @macro
# def reduce_macro(buffer: tir.Buffer, out: tir.Buffer, reduce_type: str, dim: int, clear: bool):
#     if is_shared(buffer) and is_shared(out):
#         red_frag_in = alloc_fragment(buffer.shape, buffer.dtype)
#         red_frag_out = alloc_fragment(out.shape, out.dtype)

#         # rename buffers
#         IRBuilder.name(buffer.name + "_frag", red_frag_in)
#         IRBuilder.name(out.name + "_frag", red_frag_out)

#         copy(buffer, red_frag_in)
#
# so T.reduce will create a local.fragment buffer, causing error


def gemm_matmul_incorrect_A_scope(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared.wsram")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared.wsram")
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype, scope="shared.rsram")

            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return tvm.IRModule({"main": main})


def gemm_matmul_incorrect_B_scope(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared.asram")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared.asram")
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype, scope="shared.rsram")

            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return tvm.IRModule({"main": main})


def gemm_matmul_A_scope_infer_conflict(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
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
                T.gemm(B_shared, C_shared, A_shared)
                T.gemm(A_shared, B_shared, C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return tvm.IRModule({"main": main})


def gemm_matmul_B_scope_infer_conflict(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
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
                T.gemm(B_shared, C_shared, A_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return tvm.IRModule({"main": main})


def gemm_matmul_C_scope_infer_conflict(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
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
                T.gemm(C_shared, B_shared, A_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return tvm.IRModule({"main": main})


INCORRECT_TEST_CASES = [
    (gemm_matmul_incorrect_A_scope(128, 128, 128, 32, 32, 32), "Specify invalid scope shared.wsram of A_shared in GEMM Sunmmio."),
    (gemm_matmul_incorrect_B_scope(128, 128, 128, 32, 32, 32), "Specify invalid scope shared.asram of B_shared in GEMM Sunmmio."),
    (
        gemm_matmul_A_scope_infer_conflict(128, 128, 128, 32, 32, 32),
        "Infer scope shared.asram of A_shared in GEMM Sunmmio, but scope shared.rsram has been inferred for A_shared.",
    ),
    (
        gemm_matmul_B_scope_infer_conflict(128, 128, 128, 32, 32, 32),
        "Infer scope shared.asram of B_shared in GEMM Sunmmio, but scope shared.wsram has been inferred for B_shared.",
    ),
    (
        gemm_matmul_C_scope_infer_conflict(128, 128, 128, 32, 32, 32),
        "Infer scope shared.asram of C_shared in GEMM Sunmmio, but scope shared.rsram has been inferred for C_shared.",
    ),
    (
        flashattn(
            1,
            64,
            4096,
            128,
            False,
            groups=16,
        ),
        "Invalid scope local.fragment of acc_s_frag in Sunmmio.",
    ),
]


@pytest.mark.parametrize(
    "kernel, error_info",
    INCORRECT_TEST_CASES,
)
def test_tilelang_incorrect_infer_sram_scope(kernel, error_info):
    target_name = "Sunmmio"
    # target_name = "cuda"
    target = determine_target(target_name, return_object=True)
    with pytest.raises(tvm.error.InternalError, match=error_info), tvm.target.Target(target):
        mod = kernel
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tl.transform.InferSramScope()(mod)


def dot_mul_tiled_parallel(M, N, block_M, block_N, tile_size, index_map, dtype="float16", accum_dtype="float"):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            T.annotate_tileview({A_shared: make_tileview(A_shared, tile_size, index_map)})
            T.annotate_layout({A_shared: make_blockwise_zz_layout(A_shared)})
            B_shared = T.alloc_shared((block_M, block_N), dtype)
            T.annotate_tileview({B_shared: make_tileview(B_shared, tile_size, index_map)})
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            T.annotate_tileview({C_shared: make_tileview(C_shared, tile_size, index_map)})
            T.annotate_safe_value({C_shared: "test", A_shared: 1, B_shared: True})

            T.clear(C_shared)

            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[by * block_M, bx * block_N], B_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


BUG_CASES = [
    (
        dot_mul_tiled_parallel(128, 128, 128, 128, (32, 32), (-2, -1)),
        {
            "A": "global",
            "B": "global",
            "C": "global",
            "A_shared": "shared.rsram",
            "B_shared": "shared.rsram",
            "C_shared": "shared.rsram",
        },
    ),
]


@pytest.mark.parametrize(
    "kernel, buffer_scope_dict",
    BUG_CASES,
)
def test_tilelang_bug_case_infer_sram_scope(kernel, buffer_scope_dict):
    target_name = "Sunmmio"
    target = determine_target(target_name, return_object=True)
    with tvm.target.Target(target):
        mod = tvm.IRModule.from_expr(kernel.with_attr("global_symbol", "main"))
        mod = tvm.tir.transform.BindTarget(target)(mod)

        mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
        mod = tilelang.transform.LegalizeNegativeIndex()(mod)
        mod = tilelang.transform.InjectAssumes()(mod)
        mod = tilelang.transform.Simplify()(mod)
        before_attrs = extract_block_attrs_from_kernel(list(mod.functions.values())[0])
        mod = tl.transform.InferSramScope()(mod)
        func = list(mod.functions.values())[0]
        after_attrs = extract_block_attrs_from_kernel(func)
        buffers = extract_buffers_from_kernel(func)
        for buf in buffers:
            assert buf.scope() == buffer_scope_dict[buf.name]

        for i in range(len(before_attrs)):
            before_attr = before_attrs[i]
            after_attr = after_attrs[i]
            for key in before_attr.keys():
                before_keys = list(before_attr[key])
                after_keys = list(after_attr[key])
                before_values = list(before_attr[key].values())
                after_values = list(after_attr[key].values())
                for i in range(len(before_keys)):
                    assert before_keys[i].name == after_keys[i].name
                    assert before_keys[i].type_annotation.storage_scope == "shared.dyn"
                    assert after_keys[i].type_annotation.storage_scope == "shared.rsram"
                for i in range(len(before_values)):
                    if key == "safe_value_map":
                        assert tvm.tir.analysis.expr_deep_equal(before_values[i], after_values[i])
                    else:
                        assert tvm.ir.structural_equal(before_values[i], after_values[i])


def dot_mul_tiled_parallel_specified_scope(M, N, block_M, block_N, tile_size, index_map, dtype="float16", accum_dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        A_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            T.annotate_tileview({A_shared: make_tileview(A_shared, tile_size, index_map)})
            T.annotate_layout({A_shared: make_blockwise_zz_layout(A_shared)})

            B_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")
            T.annotate_tileview({B_shared: make_tileview(B_shared, tile_size, index_map)})
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype, scope="shared.rsram")
            T.annotate_tileview({C_shared: make_tileview(C_shared, tile_size, index_map)})
            T.annotate_safe_value({C_shared: "test", A_shared: 1, B_shared: True})

            T.clear(C_shared)

            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[by * block_M, bx * block_N], B_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


CONSISTENCY_CASES = [
    dot_mul_tiled_parallel_specified_scope(128, 128, 128, 128, (32, 32), (-2, -1)),
]


@pytest.mark.parametrize(
    "kernel",
    CONSISTENCY_CASES,
)
def test_tilelang_consistency_case_infer_sram_scope(kernel):
    target_name = "Sunmmio"
    target = determine_target(target_name, return_object=True)
    with tvm.target.Target(target):
        mod = tvm.IRModule.from_expr(kernel.with_attr("global_symbol", "main"))
        mod = tvm.tir.transform.BindTarget(target)(mod)

        mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
        mod = tilelang.transform.LegalizeNegativeIndex()(mod)
        mod = tilelang.transform.InjectAssumes()(mod)
        mod = tilelang.transform.Simplify()(mod)
        before_attrs = extract_block_attrs_from_kernel(list(mod.functions.values())[0])
        mod = tl.transform.InferSramScope()(mod)
        after_attrs = extract_block_attrs_from_kernel(list(mod.functions.values())[0])

        for i in range(len(before_attrs)):
            before_attr = before_attrs[i]
            after_attr = after_attrs[i]
            for key in before_attr.keys():
                before_keys = list(before_attr[key])
                after_keys = list(after_attr[key])
                before_values = list(before_attr[key].values())
                after_values = list(after_attr[key].values())
                for i in range(len(before_keys)):
                    assert before_keys[i].name == after_keys[i].name
                    assert before_keys[i].type_annotation.storage_scope == after_keys[i].type_annotation.storage_scope
                for i in range(len(before_values)):
                    if key == "safe_value_map":
                        assert tvm.tir.analysis.expr_deep_equal(before_values[i], after_values[i])
                    else:
                        assert tvm.ir.structural_equal(before_values[i], after_values[i])
