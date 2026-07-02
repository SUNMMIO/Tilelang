"""Tests for tilelang.transform.LegalizeSunmmioGemm.

The pass expands each bf16 ASRAM GEMM into the two-pass form required by
the Sunmmio bf16 tensor core (TC reads only ASRAM north). For each
matching GEMM, it:
  1. Duplicates the direct A-operand writer (Copy or AllgatherOp) with
     annotation ``src_offset_byte`` set to the HW ASRAM bank stripe (1024 B
     on A4E).
  2. Duplicates the GEMM with the ``accOffsetByte`` direct field set to
     ``(block_h / 2) * block_w * sizeof(C_dtype)`` (2048 for fp32 acc,
     1024 for bf16 acc).

These tests focus on bf16 GEMM scenarios per #133.

Note on K-loop trip counts: the kernels here all have single-iteration
K-loops (sharded_K <= block_K). That is sufficient for transform-level
coverage because the pass is trip-count-agnostic — it emits the same
loop body (for hoisted cases: [restore, gemm, restage, gemm']) whatever
the loop extent. The structural checkers verify that body, including the
mandatory stripe-0 restore that makes multi-iteration loops correct.
End-to-end multi-iteration correctness is exercised by the flash-attention
e2e test (issue #133, task #17), whose K-loop genuinely iterates.
"""

import pytest
import tilelang as tl
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.utils.target import SUNMMIO_TARGET_DESC


def apply_pipeline_up_to_legalize_gemm(mod):
    """Run the lowering pipeline up to and including LegalizeSunmmioGemm."""
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tl.transform.AddWrapperForSingleBufStore()(mod)
        mod = tl.transform.LegalizeNegativeIndex()(mod)
        mod = tl.transform.InjectAssumes()(mod)
        mod = tl.transform.Simplify()(mod)
        mod = tl.transform.InferSramScope()(mod)
        mod = tl.transform.LegalizeSunmmioDataPath()(mod)
        mod = tl.transform.LayoutReducer()(mod)
        mod = tl.transform.SunmmioLayoutInference()(mod)
        mod = tl.transform.LegalizeSunmmioGemm()(mod)
    return mod


def count_call_op(mod, op_name):
    """Count occurrences of a TIR op call in the function body."""
    count = [0]
    op = tvm.tir.op.Op.get(op_name)

    def visit(node):
        if isinstance(node, tvm.tir.Call) and node.op.same_as(op):
            count[0] += 1

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, visit)
    return count[0]


def find_call_args(mod, op_name):
    """Return a list of (args, annotations) for every TIR Call to op_name."""
    results = []
    op = tvm.tir.op.Op.get(op_name)

    def visit(node):
        if isinstance(node, tvm.tir.Call) and node.op.same_as(op):
            results.append((list(node.args), dict(node.annotations) if node.annotations else {}))

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, visit)
    return results


def bf16_gemm_with_allgather(M=32, N=32, K=32, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"):
    """A minimal bf16 GEMM kernel whose A-operand writer is comm.all_gather."""
    from tilelang.carver.arch import driver

    nrows, ncols = driver.get_sunmmio_device_mesh_config()
    ncores = nrows * ncols
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), accum_dtype, layout=C_layout),
    ):
        with T.Kernel(ncores) as (_cid):
            sharded_M, sharded_K = A.shape
            _, sharded_N = B.shape
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            A_shared_dist = T.alloc_shared((block_M, block_K * ncols), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            B_shared_dist = T.alloc_shared((block_K * nrows, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=2):
                T.copy(A[0, k * block_K], A_shared)
                T.comm.all_gather(A_shared, A_shared_dist, direction="h", axis=-1)
                T.copy(B[k * block_K, 0], B_shared)
                T.comm.all_gather(B_shared, B_shared_dist, direction="v", axis=0)
                T.gemm(A_shared_dist, B_shared_dist, C_shared)

            T.copy(C_shared, C[0, 0])

    return main


def test_legalize_sunmmio_gemm_on_allgather_writer():
    """bf16 GEMM whose A_dist writer is T.comm.all_gather (HLink broadcast).
    Expected: the all_gather and the gemm each duplicate; the cloned
    all_gather carries src_offset_byte=1024, the cloned gemm carries
    accOffsetByte=2048 for an fp32 accumulator."""
    func = bf16_gemm_with_allgather()
    mod = tvm.IRModule({"main": func})
    mod = apply_pipeline_up_to_legalize_gemm(mod)

    # The A-side all_gather should now appear twice (original + cloned with
    # src_offset_byte=1024). The B-side all_gather is unrelated to the
    # bf16-MMA constraint and should still appear once.
    allgather_calls = find_call_args(mod, "tl.tileop.comm_allgather")
    src_offset_annotated = [c for c in allgather_calls if "src_offset_byte" in c[1]]
    assert len(src_offset_annotated) == 1, (
        f"expected exactly one all_gather with src_offset_byte annotation, got {len(src_offset_annotated)}"
    )
    src_offset_val = int(src_offset_annotated[0][1]["src_offset_byte"])
    assert src_offset_val == 1024, f"expected src_offset_byte=1024 (HW ASRAM bank stripe), got {src_offset_val}"

    # The GEMM call (gemm_py for the v2 frontend) should now appear twice —
    # original with accOffsetByte=0 (args[19]=0) and cloned with
    # accOffsetByte=2048 for fp32 accumulator.
    gemm_calls = find_call_args(mod, "tl.tileop.gemm_py")
    assert len(gemm_calls) == 2, f"expected exactly two gemm_py calls after legalization, got {len(gemm_calls)}"
    # acc_offset_byte is at args[19] (after the 19 pre-existing positional args).
    acc_offsets = []
    for args, _ in gemm_calls:
        if len(args) > 19:
            acc_offsets.append(int(args[19]))
        else:
            acc_offsets.append(0)
    assert sorted(acc_offsets) == [0, 2048], f"expected acc_offset values [0, 2048] (fp32 acc, block 32x32), got {sorted(acc_offsets)}"


def bf16_gemm_with_copy_to_asram(M=32, N=32, K=32, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"):
    """A minimal bf16 GEMM whose A_dist writer is a plain T.copy (no
    all_gather / HLink). After LegalizeSunmmioDataPath stages the DRAM->ASRAM
    transfer through an RSRAM buffer, the direct writer of A_dist becomes a
    *RSRAM->ASRAM* T.copy. This is the case the user-facing C++ kernels
    (compiler-samples/05_matmul, 06_matmul) use when A is staged through
    RSRAM and then sent to ASRAM without HLink broadcast."""
    from tilelang.carver.arch import driver

    nrows, ncols = driver.get_sunmmio_device_mesh_config()
    ncores = nrows * ncols
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), accum_dtype, layout=C_layout),
    ):
        with T.Kernel(ncores) as (_cid):
            sharded_M, sharded_K = A.shape
            _, sharded_N = B.shape
            # A staged via direct copy (no all_gather): full DRAM->ASRAM transfer.
            A_dist = T.alloc_shared((block_M, block_K * ncols), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            B_shared_dist = T.alloc_shared((block_K * nrows, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=2):
                # Direct copy of an A slice into A_dist (ASRAM).
                T.copy(A[0, k * block_K], A_dist[0:block_M, 0:block_K])
                T.copy(B[k * block_K, 0], B_shared)
                T.comm.all_gather(B_shared, B_shared_dist, direction="v", axis=0)
                T.gemm(A_dist, B_shared_dist, C_shared)

            T.copy(C_shared, C[0, 0])

    return main


def test_legalize_sunmmio_gemm_on_rsram_to_asram_copy_writer():
    """bf16 GEMM whose A_dist writer is a plain T.copy with no HLink. After
    LegalizeSunmmioDataPath stages the DRAM->ASRAM transfer through an
    RSRAM buffer, the direct writer of A_dist is a *RSRAM->ASRAM* T.copy.
    Expected: only that final RSRAM->ASRAM copy gets duplicated with
    src_offset_byte=1024 (the upstream DRAM->RSRAM staging is reused, not
    duplicated). This is the path the C/C++ reference kernels
    (compiler-samples/05_matmul, 06_matmul) take when A is staged through
    RSRAM and copied to ASRAM without HLink broadcast."""
    func = bf16_gemm_with_copy_to_asram()
    mod = tvm.IRModule({"main": func})
    mod = apply_pipeline_up_to_legalize_gemm(mod)

    # Look for a Copy op call whose dst buffer is in ASRAM and whose
    # annotation has src_offset_byte=1024.
    copy_calls = find_call_args(mod, "tl.tileop.copy")
    src_annotated = [c for c in copy_calls if "src_offset_byte" in c[1]]
    assert len(src_annotated) == 1, (
        f"expected exactly one copy with src_offset_byte annotation, got {len(src_annotated)} (total copy calls: {len(copy_calls)})"
    )
    assert int(src_annotated[0][1]["src_offset_byte"]) == 1024

    # Gemm duplication, same as the all_gather case.
    gemm_calls = find_call_args(mod, "tl.tileop.gemm_py")
    assert len(gemm_calls) == 2
    acc_offsets = []
    for args, _ in gemm_calls:
        if len(args) > 19:
            acc_offsets.append(int(args[19]))
        else:
            acc_offsets.append(0)
    assert sorted(acc_offsets) == [0, 2048]


def bf16_gemm_with_hoisted_copy_writer(M=32, N=32, K=64, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"):
    """A minimal bf16 GEMM whose A-operand writer is HOISTED out of the
    K-loop — the flash-attention pattern. The single T.copy(A → A_shared)
    runs once at kernel scope; the gemm sits inside a T.Pipelined loop and
    reuses A_shared across iterations. After LegalizeSunmmioDataPath stages
    DRAM→ASRAM through an RSRAM buffer, the direct writer of A_shared is a
    RSRAM→ASRAM copy at kernel scope, while the gemm is in the inner
    loop's scope. HoistedShadow should kick in."""
    from tilelang.carver.arch import driver

    nrows, ncols = driver.get_sunmmio_device_mesh_config()
    ncores = nrows * ncols
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), accum_dtype, layout=C_layout),
    ):
        with T.Kernel(ncores) as (_cid):
            sharded_M, sharded_K = A.shape
            _, sharded_N = B.shape
            # A is loaded ONCE at kernel scope, before the K-loop. The
            # whole-K stripe of A is brought in; B is streamed inside the loop.
            A_shared = T.alloc_shared((block_M, sharded_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.copy(A[0, 0], A_shared)  # hoisted A writer
            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=2):
                T.copy(B[k * block_K, 0], B_shared)
                T.gemm(A_shared[0:block_M, k * block_K : (k + 1) * block_K], B_shared, C_shared)

            T.copy(C_shared, C[0, 0])

    return main


def test_legalize_sunmmio_gemm_on_hoisted_copy_writer():
    """bf16 GEMM whose A_shared writer is a HOISTED T.copy (flash-attention
    pattern). The intermediate RSRAM source (A_rsram_stage) is written
    once outside the K-loop and never mutated downstream, so the
    cleanliness analyzer picks HoistedDirect — no shadow buffer; the
    re-emit reads A_rsram_stage directly with src_offset_byte=1024."""
    func = bf16_gemm_with_hoisted_copy_writer()
    mod = tvm.IRModule({"main": func})
    mod = apply_pipeline_up_to_legalize_gemm(mod)
    assert_hoisted_direct_structure(mod, expected_acc_offset=2048, expected_pairs=1)


def bf16_gemm_with_hoisted_copy_writer_bf16_acc(M=32, N=32, K=64, block_M=32, block_N=32, block_K=32, dtype="bfloat16"):
    """Same hoisted-copy pattern as above but with a bf16 accumulator
    instead of fp32. The accOffsetByte formula
    (block_h/2)*block_w*sizeof(C_dtype) gives 1024 for bf16 acc with a
    32x32 block, vs 2048 for fp32 acc — this exercises the dtype-dependent
    arithmetic in ComputeAccOffsetBytes."""
    accum_dtype = dtype
    from tilelang.carver.arch import driver

    nrows, ncols = driver.get_sunmmio_device_mesh_config()
    ncores = nrows * ncols
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), accum_dtype, layout=C_layout),
    ):
        with T.Kernel(ncores) as (_cid):
            sharded_M, sharded_K = A.shape
            _, sharded_N = B.shape
            A_shared = T.alloc_shared((block_M, sharded_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.copy(A[0, 0], A_shared)
            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=2):
                T.copy(B[k * block_K, 0], B_shared)
                T.gemm(A_shared[0:block_M, k * block_K : (k + 1) * block_K], B_shared, C_shared)

            T.copy(C_shared, C[0, 0])

    return main


def test_legalize_sunmmio_gemm_hoisted_copy_bf16_acc():
    """HoistedDirect with a bf16 accumulator. Same kernel shape as the
    fp32 hoisted test, but C_shared is bf16 so accOffsetByte should be
    1024 (instead of 2048 for fp32). The dtype-dependent math
    (block_h/2 * block_w * sizeof(C)) is the unique invariant under test
    here; the structural shape is shared with the fp32 case."""
    func = bf16_gemm_with_hoisted_copy_writer_bf16_acc()
    mod = tvm.IRModule({"main": func})
    mod = apply_pipeline_up_to_legalize_gemm(mod)
    assert_hoisted_direct_structure(mod, expected_acc_offset=1024, expected_pairs=1)


def bf16_gemm_with_two_independent_hoisted_writers(
    M=32, N=32, K=64, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"
):
    """Two ASRAM A operands, each loaded by its own hoisted T.copy at
    kernel scope, each consumed by a distinct gemm in the same K-loop.
    Exercises the case where a single kernel needs multiple independent
    HoistedShadow rewrites — each A_dist must get its own shadow buffer
    co-located with it in the kernel Block's alloc_buffers."""
    from tilelang.carver.arch import driver

    nrows, ncols = driver.get_sunmmio_device_mesh_config()
    ncores = nrows * ncols
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=A_layout),
        A2: T.MeshTensor((M, K), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), accum_dtype, layout=C_layout),
        C2: T.MeshTensor((M, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), accum_dtype, layout=C_layout),
    ):
        with T.Kernel(ncores) as (_cid):
            sharded_M, sharded_K = A.shape
            _, sharded_N = B.shape
            A_shared = T.alloc_shared((block_M, sharded_K), dtype)
            A2_shared = T.alloc_shared((block_M, sharded_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            C2_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.copy(A[0, 0], A_shared)  # hoisted writer #1
            T.copy(A2[0, 0], A2_shared)  # hoisted writer #2
            T.clear(C_shared)
            T.clear(C2_shared)
            for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=2):
                T.copy(B[k * block_K, 0], B_shared)
                T.gemm(A_shared[0:block_M, k * block_K : (k + 1) * block_K], B_shared, C_shared)
                T.gemm(A2_shared[0:block_M, k * block_K : (k + 1) * block_K], B_shared, C2_shared)

            T.copy(C_shared, C[0, 0])
            T.copy(C2_shared, C2[0, 0])

    return main


def test_legalize_sunmmio_gemm_two_independent_hoisted_writers():
    """Two independent A_dists, each with its own hoisted writer and a
    clean source buffer. Each gemm gets its own HoistedDirect rewrite —
    no shadow buffers anywhere, each re-emit reads its own writer's
    source directly."""
    func = bf16_gemm_with_two_independent_hoisted_writers()
    mod = tvm.IRModule({"main": func})
    mod = apply_pipeline_up_to_legalize_gemm(mod)
    assert_hoisted_direct_structure(mod, expected_acc_offset=2048, expected_pairs=2)


def bf16_gemm_with_hoisted_writer_dirty_source(
    M=32, N=32, K=64, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"
):
    """Hoisted bf16 ASRAM A writer whose RSRAM source buffer is mutated
    inside the K-loop. Forces the cleanliness analyzer to flag the
    source as dirty so Phase 3 picks HoistedShadow instead of
    HoistedDirect.

    The kernel is synthetic — real flash-attention kernels don't write
    back into the hoisted writer's source inside the K-loop. But it
    exercises the exact code path that protects correctness when the
    source IS mutated (e.g., if a future kernel re-stages A on each
    iteration without re-issuing the ASRAM writer)."""
    from tilelang.carver.arch import driver

    nrows, ncols = driver.get_sunmmio_device_mesh_config()
    ncores = nrows * ncols
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), accum_dtype, layout=C_layout),
    ):
        with T.Kernel(ncores) as (_cid):
            sharded_M, sharded_K = A.shape
            _, sharded_N = B.shape
            # User-declared RSRAM scratch — visible by name to the test so
            # we can verify cleanliness detects writes to it.
            A_rsram = T.alloc_shared((block_M, sharded_K), dtype, scope="shared.rsram")
            A_shared = T.alloc_shared((block_M, sharded_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.copy(A[0, 0], A_rsram)  # initial source population
            T.copy(A_rsram, A_shared)  # hoisted writer (reads A_rsram)
            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=2):
                T.copy(A[0, 0], A_rsram)  # MUTATES the writer's source per iter
                T.copy(B[k * block_K, 0], B_shared)
                T.gemm(A_shared[0:block_M, k * block_K : (k + 1) * block_K], B_shared, C_shared)

            T.copy(C_shared, C[0, 0])

    return main


def test_legalize_sunmmio_gemm_hoisted_shadow_when_source_dirty():
    """Source-dirty hoisted case: cleanliness analyzer must detect the
    K-loop write to A_rsram and pick HoistedShadow (snapshot at writer
    site, re-emit reads the snapshot, shadow co-located with A_shared)."""
    func = bf16_gemm_with_hoisted_writer_dirty_source()
    mod = tvm.IRModule({"main": func})
    mod = apply_pipeline_up_to_legalize_gemm(mod)
    assert_hoisted_shadow_structure(mod, expected_acc_offset=2048, expected_pairs=1)


def bf16_gemm_with_hoisted_allgather(M=32, N=32, K=32, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"):
    """bf16 GEMM whose A-operand writer is a HOISTED comm.all_gather. A is
    copied and gathered once at kernel scope; the gemm in the inner K-loop
    reuses the gathered A_shared_dist. The all_gather's source (A_shared) is
    written once before the gather and never mutated — clean — so the pass
    should pick HoistedDirect and re-issue the all_gather inside the loop."""
    from tilelang.carver.arch import driver

    nrows, ncols = driver.get_sunmmio_device_mesh_config()
    ncores = nrows * ncols
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), accum_dtype, layout=C_layout),
    ):
        with T.Kernel(ncores) as (_cid):
            sharded_M, sharded_K = A.shape
            _, sharded_N = B.shape
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            A_shared_dist = T.alloc_shared((block_M, block_K * ncols), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            B_shared_dist = T.alloc_shared((block_K * nrows, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            # A loaded + gathered ONCE, hoisted out of the K-loop.
            T.copy(A[0, 0], A_shared)
            T.comm.all_gather(A_shared, A_shared_dist, direction="h", axis=-1)
            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=2):
                T.copy(B[k * block_K, 0], B_shared)
                T.comm.all_gather(B_shared, B_shared_dist, direction="v", axis=0)
                T.gemm(A_shared_dist, B_shared_dist, C_shared)

            T.copy(C_shared, C[0, 0])

    return main


def test_legalize_sunmmio_gemm_hoisted_allgather_direct():
    """Hoisted Allgather writer with a clean source → HoistedDirect. The
    all_gather is removed from outside the loop and re-issued inside it as
    the stripe-0 restore and the stripe-1 restage (both comm_allgather
    calls, the second carrying src_offset_byte=1024). No shadow buffer."""
    func = bf16_gemm_with_hoisted_allgather()
    mod = tvm.IRModule({"main": func})
    mod = apply_pipeline_up_to_legalize_gemm(mod)
    assert_hoisted_direct_structure(mod, expected_acc_offset=2048, expected_pairs=1, writer_op="tl.tileop.comm_allgather")


def bf16_gemm_with_hoisted_allgather_dirty_source(
    M=32, N=32, K=32, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"
):
    """Hoisted Allgather whose source (A_shared) is mutated inside the
    K-loop. The cleanliness analyzer must flag the source dirty so the
    pass picks HoistedShadow: snapshot A_shared into an RSRAM stage, then
    re-issue the all_gather from the stage inside the loop."""
    from tilelang.carver.arch import driver

    nrows, ncols = driver.get_sunmmio_device_mesh_config()
    ncores = nrows * ncols
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), accum_dtype, layout=C_layout),
    ):
        with T.Kernel(ncores) as (_cid):
            sharded_M, sharded_K = A.shape
            _, sharded_N = B.shape
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            A_shared_dist = T.alloc_shared((block_M, block_K * ncols), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            B_shared_dist = T.alloc_shared((block_K * nrows, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.copy(A[0, 0], A_shared)
            T.comm.all_gather(A_shared, A_shared_dist, direction="h", axis=-1)
            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=2):
                T.copy(A[0, 0], A_shared)  # mutates the all_gather's source
                T.copy(B[k * block_K, 0], B_shared)
                T.comm.all_gather(B_shared, B_shared_dist, direction="v", axis=0)
                T.gemm(A_shared_dist, B_shared_dist, C_shared)

            T.copy(C_shared, C[0, 0])

    return main


def test_legalize_sunmmio_gemm_hoisted_allgather_shadow():
    """Hoisted Allgather with a dirty source → HoistedShadow. The snapshot
    is a plain Copy (A_shared → shadow); the in-loop restore/restage are
    comm_allgather calls re-broadcasting from the shadow."""
    func = bf16_gemm_with_hoisted_allgather_dirty_source()
    mod = tvm.IRModule({"main": func})
    mod = apply_pipeline_up_to_legalize_gemm(mod)
    assert_hoisted_shadow_structure(mod, expected_acc_offset=2048, expected_pairs=1, writer_op="tl.tileop.comm_allgather")


def _collect_legalize_stage_shadows(mod):
    """Helper: return a list of (shadow_buffer, a_dist_buffer) tuples for
    every '_legalize_stage' shadow allocation found in the module. Asserts
    each shadow's A_dist counterpart is in the SAME Block's alloc_buffers
    (the co-location liveness invariant)."""
    results = []

    def visit(node):
        if not isinstance(node, tvm.tir.Block):
            return
        by_name = {b.name: b for b in node.alloc_buffers}
        for b in node.alloc_buffers:
            if not b.name.endswith("_legalize_stage"):
                continue
            a_dist_name = b.name[: -len("_legalize_stage")]
            assert a_dist_name in by_name, (
                f"shadow {b.name} expects A_dist '{a_dist_name}' in the same Block's alloc_buffers, found: {list(by_name.keys())}"
            )
            results.append((b, by_name[a_dist_name]))

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, visit)
    return results


def _iter_calls_in_stmt(stmt, op_name):
    """Yield (index_in_seq, call) for direct EvaluateNode(Call(op_name)) at
    the top level of `stmt` (treating SeqStmt as a flat sequence and any
    other Stmt as a singleton sequence). Skips nested control flow."""
    op = tvm.tir.op.Op.get(op_name)
    if isinstance(stmt, tvm.tir.SeqStmt):
        seq = list(stmt.seq)
    else:
        seq = [stmt]
    for i, s in enumerate(seq):
        if isinstance(s, tvm.tir.Evaluate) and isinstance(s.value, tvm.tir.Call) and s.value.op.same_as(op):
            yield i, s.value


def _find_k_loop_for(mod):
    """Locate the K-loop For node (the one annotated with num_stages=2 in
    these tests) and return it. There is exactly one in each kernel."""
    for_nodes = []

    def visit(node):
        if isinstance(node, tvm.tir.For) and node.annotations and "num_stages" in node.annotations:
            for_nodes.append(node)

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, visit)
    assert len(for_nodes) == 1, f"expected exactly one annotated K-loop For, got {len(for_nodes)}"
    return for_nodes[0]


def _kernel_body_seq(mod):
    """Return the SeqStmt that holds the kernel-scope statements (the body
    of the innermost tilelang_root block). Used to assert outer-scope
    structure for HoistedShadow."""
    found = [None]

    def visit(node):
        if isinstance(node, tvm.tir.Block) and node.name_hint == "tilelang_root":
            found[0] = node.body

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, visit)
    assert found[0] is not None, "tilelang_root block not found"
    return found[0]


def _is_call(s, op):
    """True iff `s` is Evaluate(Call(op))."""
    return isinstance(s, tvm.tir.Evaluate) and isinstance(s.value, tvm.tir.Call) and s.value.op.same_as(op)


def _copy_buf(call, arg_index):
    """Extract the Buffer behind an op's region arg. Different ops normalize
    region args differently: it may be a tl.tileop.region Call (whose first
    child is a BufferLoad), a raw BufferLoad, or a BufferRegion. All three
    expose the underlying buffer via `.buffer` once unwrapped."""
    expr = call.args[arg_index]
    if isinstance(expr, tvm.tir.Call):
        expr = expr.args[0]  # tl.tileop.region(BufferLoad, mask, extents...)
    return expr.buffer


def _src_offset(call):
    """Return the int src_offset_byte annotation on a copy call, or None."""
    anno = dict(call.annotations) if call.annotations else {}
    return int(anno["src_offset_byte"]) if "src_offset_byte" in anno else None


def _assert_hoisted_loop_body(mod, *, expected_acc_offset, expected_pairs, src_is_shadow, writer_op="tl.tileop.copy"):
    """Verify each hoisted gemm pair expands to the mandatory 4-statement
    K-loop body:

        [ restore : <writer_op>(src → A_dist)           no src_offset_byte,
          gemm pass-1 : accOffsetByte = 0,
          restage : <writer_op>(src → A_dist, src_offset_byte=1024),
          gemm pass-2 : accOffsetByte = expected_acc_offset ]

    The stripe-0 *restore* before pass-1 is the load-bearing part: the
    hoisted writer runs only once outside the loop, so every iteration
    must re-establish stripe-0 before pass-1 — otherwise iteration k+1
    reads the stripe-1 data left by iteration k.

    writer_op is the op of the restore/restage statements — 'tl.tileop.copy'
    for a Copy writer, 'tl.tileop.comm_allgather' for an Allgather writer
    (the rewrite re-issues the writer preserving its op type).

    src_is_shadow selects where restore/restage read from:
      True  -> a '<A_dist>_legalize_stage' shadow buffer  (HoistedShadow)
      False -> the writer's source buffer directly        (HoistedDirect)
    """
    gemm_op = tvm.tir.op.Op.get("tl.tileop.gemm_py")
    wop = tvm.tir.op.Op.get(writer_op)
    body = _find_k_loop_for(mod).body
    seq = list(body.seq) if isinstance(body, tvm.tir.SeqStmt) else [body]

    pairs = 0
    i = 0
    while i < len(seq):
        if not _is_call(seq[i], gemm_op):
            i += 1
            continue
        acc = int(seq[i].value.args[19]) if len(seq[i].value.args) > 19 else 0
        if acc != 0:
            i += 1
            continue

        # --- stripe-0 restore must immediately precede pass-1 ---
        assert i - 1 >= 0, "gemm pass-1 has no preceding stripe-0 restore"
        restore = seq[i - 1]
        assert _is_call(restore, wop), (
            f"expected a stripe-0 restore ({writer_op}) before gemm pass-1 at index {i}, got {type(restore).__name__}"
        )
        assert _src_offset(restore.value) is None, "stripe-0 restore must NOT carry src_offset_byte"

        # --- restage + cloned gemm must immediately follow pass-1 ---
        assert i + 2 < len(seq), f"gemm pass-1 at index {i} is missing its restage + pass-2"
        restage, cloned = seq[i + 1], seq[i + 2]
        assert _is_call(restage, wop), f"expected a stripe-1 restage ({writer_op}) at index {i + 1}"
        assert _src_offset(restage.value) == 1024, "stripe-1 restage copy must carry src_offset_byte=1024"
        assert _is_call(cloned, gemm_op), f"expected cloned gemm pass-2 at index {i + 2}"
        cloned_acc = int(cloned.value.args[19]) if len(cloned.value.args) > 19 else 0
        assert cloned_acc == expected_acc_offset, f"gemm pass-2 expected accOffsetByte={expected_acc_offset}, got {cloned_acc}"

        # --- restore & restage write the SAME A_dist, read the right src ---
        a_dist = _copy_buf(restore.value, 1)
        assert _copy_buf(restage.value, 1).same_as(a_dist), "restore and restage must write the same A_dist buffer"
        for label, cp in (("restore", restore.value), ("restage", restage.value)):
            src_name = _copy_buf(cp, 0).name
            if src_is_shadow:
                assert src_name.endswith("_legalize_stage"), f"HoistedShadow {label} must read the shadow buffer, got src={src_name}"
            else:
                assert not src_name.endswith("_legalize_stage"), (
                    f"HoistedDirect {label} must read the writer source directly, got src={src_name}"
                )

        pairs += 1
        i += 3
    assert pairs == expected_pairs, f"expected {expected_pairs} hoisted gemm pair(s) in the K-loop body, found {pairs}"


def _outer_copy_to_buffer_count(mod, buffer_name_suffix):
    """Count copies at kernel (outer) scope whose destination buffer name
    ends with `buffer_name_suffix`."""
    outer_seq = _kernel_body_seq(mod)
    count = 0
    for _, call in _iter_calls_in_stmt(outer_seq, "tl.tileop.copy"):
        if _copy_buf(call, 1).name.endswith(buffer_name_suffix):
            count += 1
    return count


def assert_hoisted_direct_structure(mod, *, expected_acc_offset, expected_pairs, writer_op="tl.tileop.copy"):
    """Verify HoistedDirect IR shape.

    HoistedDirect fires when the writer's source is provably clean. The
    shadow + snapshot are elided, and the original hoisted writer is
    removed entirely (it is re-issued inside the K-loop as the stripe-0
    restore). So we expect:
      * NO '_legalize_stage' shadow buffer anywhere;
      * the 4-statement loop body, restore/restage being `writer_op` calls
        that read the writer's source buffer directly.
    """
    shadows = _collect_legalize_stage_shadows(mod)
    assert len(shadows) == 0, f"HoistedDirect should elide shadows, found {len(shadows)}"
    _assert_hoisted_loop_body(
        mod, expected_acc_offset=expected_acc_offset, expected_pairs=expected_pairs, src_is_shadow=False, writer_op=writer_op
    )


def assert_hoisted_shadow_structure(mod, *, expected_acc_offset, expected_pairs, writer_op="tl.tileop.copy"):
    """Verify HoistedShadow IR shape.

    HoistedShadow fires when the writer's source is contested. We expect:
      * one shadow buffer per pair, shared.rsram, co-located in the same
        Block as its A_dist;
      * one snapshot copy per pair at outer scope (dst = shadow, no
        src_offset_byte) — it replaces the removed original writer. The
        snapshot is always a plain Copy (it just freezes the source),
        even when the writer itself is an Allgather;
      * the 4-statement loop body, restore/restage being `writer_op` calls
        that read the shadow.
    """
    # (a) shadow buffers — count, scope, co-location.
    shadows = _collect_legalize_stage_shadows(mod)
    assert len(shadows) == expected_pairs, f"expected {expected_pairs} shadow buffer(s), got {len(shadows)}"
    for shadow_buf, a_dist_buf in shadows:
        assert shadow_buf.scope() == "shared.rsram", f"shadow {shadow_buf.name} expected scope=shared.rsram, got {shadow_buf.scope()}"
        assert shadow_buf.name == a_dist_buf.name + "_legalize_stage"

    # (b) one snapshot copy per pair at outer scope (dst = shadow buffer).
    snapshots = _outer_copy_to_buffer_count(mod, "_legalize_stage")
    assert snapshots == expected_pairs, (
        f"expected {expected_pairs} snapshot copies at outer scope (dst=<*>_legalize_stage), got {snapshots}"
    )

    # (c) the 4-statement loop body, reading from the shadow.
    _assert_hoisted_loop_body(
        mod, expected_acc_offset=expected_acc_offset, expected_pairs=expected_pairs, src_is_shadow=True, writer_op=writer_op
    )


def _run_legalize_gemm_again(mod):
    """Run only the LegalizeSunmmioGemm pass on an already-pipelined module."""
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        return tl.transform.LegalizeSunmmioGemm()(mod)


def test_legalize_sunmmio_gemm_is_idempotent_inplace():
    """Running LegalizeSunmmioGemm on already-legalized IR must be a no-op.

    This guards two things that reduce to the same property:
      * a second pass invocation does not re-legalize its own output;
      * a developer who hand-writes the two-pass form (split A stripes +
        paired gemms with explicit accOffsetByte) is not double-legalized.
    Both are detected the same way: a bf16 ASRAM A operand that already has
    a gemm consumer with accOffsetByte != 0 is skipped entirely — even its
    pass-1 gemm, whose accOffsetByte is still 0."""
    func = bf16_gemm_with_copy_to_asram()
    mod_once = apply_pipeline_up_to_legalize_gemm(tvm.IRModule({"main": func}))
    gemms_once = count_call_op(mod_once, "tl.tileop.gemm_py")
    copies_once = count_call_op(mod_once, "tl.tileop.copy")
    assert gemms_once == 2, f"expected 2 gemms after one legalization, got {gemms_once}"

    mod_twice = _run_legalize_gemm_again(mod_once)
    assert count_call_op(mod_twice, "tl.tileop.gemm_py") == gemms_once, "second LegalizeSunmmioGemm run double-legalized the pass-1 gemm"
    assert count_call_op(mod_twice, "tl.tileop.copy") == copies_once, "second LegalizeSunmmioGemm run duplicated a writer copy"


def test_legalize_sunmmio_gemm_is_idempotent_hoisted():
    """Idempotency for the hoisted (HoistedDirect) path. The pass-1 gemm
    still carries accOffsetByte == 0 after legalization; without the
    already-legalized A_dist guard a second run would re-legalize it."""
    func = bf16_gemm_with_hoisted_copy_writer()
    mod_once = apply_pipeline_up_to_legalize_gemm(tvm.IRModule({"main": func}))
    gemms_once = count_call_op(mod_once, "tl.tileop.gemm_py")
    copies_once = count_call_op(mod_once, "tl.tileop.copy")
    assert gemms_once == 2

    mod_twice = _run_legalize_gemm_again(mod_once)
    assert count_call_op(mod_twice, "tl.tileop.gemm_py") == gemms_once, "second LegalizeSunmmioGemm run double-legalized the pass-1 gemm"
    assert count_call_op(mod_twice, "tl.tileop.copy") == copies_once, "second LegalizeSunmmioGemm run duplicated a writer copy"


def assert_inplace_group_structure(mod, *, expected_acc_offset, group_size):
    """Verify InPlaceGroup: a consecutive run of `group_size` consumer
    gemms sharing one A_dist is emitted as

        [ G1..Gn   (accOffsetByte = 0),
          W'       (one restage, src_offset_byte = 1024),
          G1'..Gn' (accOffsetByte = expected_acc_offset) ]

    The co-scoped writer is kept; a single restage W' serves the whole
    pass-2 run. No shadow buffer."""
    shadows = _collect_legalize_stage_shadows(mod)
    assert len(shadows) == 0, f"InPlaceGroup must not allocate shadows, found {len(shadows)}"

    gemm_op = tvm.tir.op.Op.get("tl.tileop.gemm_py")
    copy_op = tvm.tir.op.Op.get("tl.tileop.copy")
    body = _find_k_loop_for(mod).body
    seq = list(body.seq) if isinstance(body, tvm.tir.SeqStmt) else [body]

    def gemm_acc(s):
        return int(s.value.args[19]) if len(s.value.args) > 19 else 0

    # Start of the pass-1 run: first gemm with accOffsetByte == 0.
    start = next((i for i, s in enumerate(seq) if _is_call(s, gemm_op) and gemm_acc(s) == 0), None)
    assert start is not None, "no pass-1 gemm in K-loop body"

    # pass-1: group_size consecutive gemms, accOffsetByte == 0.
    for j in range(group_size):
        s = seq[start + j]
        assert _is_call(s, gemm_op) and gemm_acc(s) == 0, f"expected pass-1 gemm (acc=0) at index {start + j}"

    # exactly one restage between the runs, src_offset_byte == 1024.
    restage = seq[start + group_size]
    assert _is_call(restage, copy_op), f"expected one restage copy after the pass-1 run at index {start + group_size}"
    assert _src_offset(restage.value) == 1024, "group restage must carry src_offset_byte=1024"

    # pass-2: group_size consecutive cloned gemms, accOffsetByte == expected.
    for j in range(group_size):
        s = seq[start + group_size + 1 + j]
        assert _is_call(s, gemm_op) and gemm_acc(s) == expected_acc_offset, (
            f"expected pass-2 gemm (acc={expected_acc_offset}) at index {start + group_size + 1 + j}"
        )

    total = sum(1 for s in seq if _is_call(s, gemm_op))
    assert total == 2 * group_size, f"expected {2 * group_size} gemms in K-loop body, got {total}"


def bf16_gemm_multi_consumer_groupable(M=32, N=32, K=32, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"):
    """Two bf16 gemms that share one ASRAM A operand and sit back-to-back
    (textually adjacent) in the K-loop — a multi-B fan-out. The shared
    A_shared is written by a single co-scoped copy. Groupable: the pass
    should emit one [G1, G2, W', G1', G2'] block."""
    from tilelang.carver.arch import driver

    nrows, ncols = driver.get_sunmmio_device_mesh_config()
    ncores = nrows * ncols
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), accum_dtype, layout=C_layout),
        C2: T.MeshTensor((M, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), accum_dtype, layout=C_layout),
    ):
        with T.Kernel(ncores) as (_cid):
            sharded_M, sharded_K = A.shape
            _, sharded_N = B.shape
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            C2_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.clear(C_shared)
            T.clear(C2_shared)
            for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=2):
                T.copy(A[0, k * block_K], A_shared)
                T.copy(B[k * block_K, 0], B_shared)
                T.gemm(A_shared, B_shared, C_shared)  # consumer 1
                T.gemm(A_shared, B_shared, C2_shared)  # consumer 2 — adjacent

            T.copy(C_shared, C[0, 0])
            T.copy(C2_shared, C2[0, 0])

    return main


def test_legalize_sunmmio_gemm_multi_consumer_groupable():
    """Two adjacent gemms sharing A_shared → InPlaceGroup. Expect one
    [G1, G2, W'(+1024), G1', G2'] block: 4 gemm_py calls, exactly one
    copy carrying src_offset_byte=1024, no shadow buffer."""
    func = bf16_gemm_multi_consumer_groupable()
    mod = tvm.IRModule({"main": func})
    mod = apply_pipeline_up_to_legalize_gemm(mod)
    assert_inplace_group_structure(mod, expected_acc_offset=2048, group_size=2)
    # Exactly one restage for the whole group.
    copy_calls = find_call_args(mod, "tl.tileop.copy")
    src_annotated = [c for c in copy_calls if "src_offset_byte" in c[1]]
    assert len(src_annotated) == 1, f"groupable multi-consumer should emit ONE restage, got {len(src_annotated)}"


def bf16_gemm_multi_consumer_non_groupable(M=32, N=32, K=32, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"):
    """Two bf16 gemms sharing one ASRAM A operand, but with a copy between
    them — NOT textually adjacent. Non-groupable: the pass must fall back
    to per-gemm Reissue (each self-restores), and emit a warning."""
    from tilelang.carver.arch import driver

    nrows, ncols = driver.get_sunmmio_device_mesh_config()
    ncores = nrows * ncols
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=B_layout),
        B2: T.MeshTensor((K, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), accum_dtype, layout=C_layout),
        C2: T.MeshTensor((M, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), accum_dtype, layout=C_layout),
    ):
        with T.Kernel(ncores) as (_cid):
            sharded_M, sharded_K = A.shape
            _, sharded_N = B.shape
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            B2_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            C2_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.clear(C_shared)
            T.clear(C2_shared)
            for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=2):
                T.copy(A[0, k * block_K], A_shared)
                T.copy(B[k * block_K, 0], B_shared)
                T.gemm(A_shared, B_shared, C_shared)  # consumer 1
                T.copy(B2[k * block_K, 0], B2_shared)  # breaks adjacency
                T.gemm(A_shared, B2_shared, C2_shared)  # consumer 2

            T.copy(C_shared, C[0, 0])
            T.copy(C2_shared, C2[0, 0])

    return main


def test_legalize_sunmmio_gemm_multi_consumer_non_groupable():
    """Two gemms sharing A_shared with a copy between them → non-groupable.
    Each consumer becomes a self-restoring Reissue: the K-loop body has two
    independent [restore, gemm(0), restage(+1024), gemm(2048)] quadruples,
    so two src_offset_byte=1024 restage copies, no shadow buffer."""
    func = bf16_gemm_multi_consumer_non_groupable()
    mod = tvm.IRModule({"main": func})
    mod = apply_pipeline_up_to_legalize_gemm(mod)
    # Non-groupable multi-consumer reuses the ReissueDirect structure:
    # each of the 2 consumers expands to [restore, gemm, restage, gemm'].
    assert_hoisted_direct_structure(mod, expected_acc_offset=2048, expected_pairs=2)
    copy_calls = find_call_args(mod, "tl.tileop.copy")
    src_annotated = [c for c in copy_calls if "src_offset_byte" in c[1]]
    assert len(src_annotated) == 2, f"non-groupable multi-consumer should emit one restage per consumer (2 total), got {len(src_annotated)}"


def bf16_flash_attention_shaped(M=32, N=32, K=64, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"):
    """A flash-attention-shaped kernel: the QK→PV gemm skeleton that
    LegalizeSunmmioGemm must handle. Two bf16 gemms share one K-loop:

      * QK gemm — A operand Q_shared is copied ONCE outside the K-loop
        (hoisted, like flash attention's Q). Its source is clean, so the
        pass picks ReissueDirect. QK accumulates into a bf16 buffer, so the
        cloned gemm's accOffsetByte is 1024.
      * PV gemm — A operand S_cast is written by a copy INSIDE the K-loop
        (co-scoped, like flash attention's acc_s_cast). The pass picks
        InPlace. PV accumulates into an fp32 buffer → accOffsetByte 2048.

    The softmax between the two gemms in real flash attention is omitted —
    it does not affect the legalization (the pass only cares about which
    scope each A-operand writer lives in). The shipped example
    examples/flash_attention/sunmmio_example_gqa_fwd_bhsd.py is currently
    stale against the T.Persistent / T.Tiles APIs and does not lower; this
    self-contained kernel exercises the same two legalization patterns."""
    from tilelang.carver.arch import driver

    nrows, ncols = driver.get_sunmmio_device_mesh_config()
    ncores = nrows * ncols
    from tilelang.layout import make_zz_layout

    Q_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    Kt_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    V_layout = make_zz_layout((N, N), [0, 1], (32, 32))
    O_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        Q: T.MeshTensor((M, K), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=Q_layout),
        Kt: T.MeshTensor((K, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=Kt_layout),
        V: T.MeshTensor((N, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=V_layout),
        O: T.MeshTensor((M, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), accum_dtype, layout=O_layout),
    ):
        with T.Kernel(ncores) as (_cid):
            sharded_M, sharded_K = Q.shape
            Q_shared = T.alloc_shared((block_M, sharded_K), dtype)  # hoisted A (Q)
            K_shared = T.alloc_shared((block_K, block_N), dtype)
            S_shared = T.alloc_shared((block_M, block_N), dtype)  # bf16 QK output
            S_cast = T.alloc_shared((block_M, block_N), dtype)  # co-scoped A (PV)
            V_shared = T.alloc_shared((block_N, block_N), dtype)
            O_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.copy(Q[0, 0], Q_shared)  # hoisted Q writer
            T.clear(S_shared)
            T.clear(O_shared)
            for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=2):
                T.copy(Kt[k * block_K, 0], K_shared)
                T.gemm(Q_shared[0:block_M, k * block_K : (k + 1) * block_K], K_shared, S_shared)  # QK gemm: A = Q (hoisted)
                T.copy(S_shared, S_cast)  # RSRAM->ASRAM move, co-scoped
                T.copy(V[0, 0], V_shared)
                T.gemm(S_cast, V_shared, O_shared)  # PV gemm: A = S_cast (in-place)
            T.copy(O_shared, O[0, 0])

    return main


def test_legalize_sunmmio_gemm_flash_attention_shaped():
    """The QK→PV flash-attention skeleton: a hoisted-Q gemm and a
    co-scoped-S_cast gemm legalized together in one K-loop. Verifies both
    patterns coexist correctly — QK → ReissueDirect, PV → InPlace — with
    dtype-correct accOffsetByte on each cloned gemm."""
    func = bf16_flash_attention_shaped()
    mod = tvm.IRModule({"main": func})
    mod = apply_pipeline_up_to_legalize_gemm(mod)

    # Two gemms × two passes = four gemm_py calls. QK accumulates into bf16
    # (accOffsetByte 1024); PV into fp32 (2048).
    gemm_calls = find_call_args(mod, "tl.tileop.gemm_py")
    assert len(gemm_calls) == 4, f"expected 4 gemm_py calls (QK pair + PV pair), got {len(gemm_calls)}"
    acc = sorted(int(a[19]) if len(a) > 19 else 0 for a, _ in gemm_calls)
    assert acc == [0, 0, 1024, 2048], f"expected accOffsets [0, 0, 1024, 2048] — QK bf16 acc → 1024, PV fp32 acc → 2048 — got {acc}"

    # Both writer sources are clean: no shadow buffers.
    assert len(_collect_legalize_stage_shadows(mod)) == 0, "flash-attention skeleton sources are clean — no shadow expected"

    # One restage per gemm, each src_offset_byte=1024.
    copy_calls = find_call_args(mod, "tl.tileop.copy")
    restage = [c for c in copy_calls if "src_offset_byte" in c[1]]
    assert len(restage) == 2, f"expected two restage copies (one per gemm), got {len(restage)}"
    for _, anno in restage:
        assert int(anno["src_offset_byte"]) == 1024

    # QK gemm is hoisted (ReissueDirect): the Q writer is removed from
    # kernel scope — no outer-scope copy writes Q_shared.
    outer = _kernel_body_seq(mod)
    outer_dsts = [_copy_buf(c, 1).name for _, c in _iter_calls_in_stmt(outer, "tl.tileop.copy")]
    assert "Q_shared" not in outer_dsts, f"hoisted Q writer must be removed from kernel scope; outer copy destinations were {outer_dsts}"

    # Inside the K-loop: exactly one no-offset copy re-establishes Q_shared
    # (the ReissueDirect restore) and exactly one no-offset copy writes
    # S_cast (the kept InPlace writer).
    copy_op = tvm.tir.op.Op.get("tl.tileop.copy")
    body = _find_k_loop_for(mod).body
    seq = list(body.seq) if isinstance(body, tvm.tir.SeqStmt) else [body]
    no_offset_dsts = [_copy_buf(s.value, 1).name for s in seq if _is_call(s, copy_op) and _src_offset(s.value) is None]
    assert no_offset_dsts.count("Q_shared") == 1, (
        f"expected one in-loop stripe-0 restore of Q_shared (ReissueDirect), got {no_offset_dsts.count('Q_shared')}"
    )
    assert no_offset_dsts.count("S_cast") == 1, (
        f"expected the co-scoped S_cast writer kept in the loop (InPlace), got {no_offset_dsts.count('S_cast')}"
    )


def test_legalize_sunmmio_gemm_skips_small_block_m():
    """A bf16 ASRAM GEMM whose row count does not exceed the north-bank
    single-pass capacity (SunmmioTileProcessorConfig.bf16_gemm_single_pass_max_rows
    = 16 on A4E) fits entirely in the north bank in one pass —
    LegalizeSunmmioGemm must leave it untouched: no gemm duplication, no
    stripe-shifted writer."""
    func = bf16_gemm_with_copy_to_asram(block_M=16)
    mod = tvm.IRModule({"main": func})
    mod = apply_pipeline_up_to_legalize_gemm(mod)

    gemm_calls = find_call_args(mod, "tl.tileop.gemm_py")
    assert len(gemm_calls) == 1, (
        f"a block_M=16 bf16 gemm must NOT be legalized (M <= 16 fits one pass), but got {len(gemm_calls)} gemm_py calls"
    )
    # The single gemm must keep accOffsetByte == 0 (unlegalized).
    args = gemm_calls[0][0]
    acc = int(args[19]) if len(args) > 19 else 0
    assert acc == 0, f"unlegalized gemm must keep accOffsetByte=0, got {acc}"

    copy_calls = find_call_args(mod, "tl.tileop.copy")
    assert not any("src_offset_byte" in c[1] for c in copy_calls), (
        "a block_M=16 gemm must not get a stripe-shifted (src_offset_byte) writer clone"
    )


# ─── Reject-path diagnostics ──────────────────────────────────────────────
# LegalizeSunmmioGemm has three reachable LOG(FATAL) reject paths. These
# tests confirm each fires with its diagnostic instead of silently
# miscompiling. (A fourth check — "unsupported writer op" in Phase 2 — is
# dead code: Phase 1 finds writers only via GetDstBuffer, which yields only
# Copy / AllgatherOp, so the writer is always one of those.)


def bf16_gemm_no_reaching_writer(M=32, N=32, K=32, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"):
    """A bf16 ASRAM gemm whose A operand is never written — no reaching
    writer. Phase 1 records the gemm with reaching-def NONE; Phase 2 must
    reject it."""
    from tilelang.carver.arch import driver

    nrows, ncols = driver.get_sunmmio_device_mesh_config()
    ncores = nrows * ncols
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), accum_dtype, layout=C_layout),
    ):
        with T.Kernel(ncores) as (_cid):
            sharded_M, sharded_K = A.shape
            A_shared = T.alloc_shared((block_M, block_K), dtype)  # never written
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=2):
                T.copy(B[k * block_K, 0], B_shared)
                T.gemm(A_shared, B_shared, C_shared)
            T.copy(C_shared, C[0, 0])

    return main


def test_legalize_sunmmio_gemm_rejects_no_reaching_writer():
    """NONE reaching def: the gemm's A operand has no writer at all. The
    pass must fatal with a diagnostic, not silently legalize garbage."""
    func = bf16_gemm_no_reaching_writer()
    mod = tvm.IRModule({"main": func})
    with pytest.raises(Exception, match="has no reaching writer at the gemm site"):
        apply_pipeline_up_to_legalize_gemm(mod)


def bf16_gemm_multiple_writers(M=32, N=32, K=32, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"):
    """A bf16 ASRAM gemm whose A operand is written by divergent writers in
    the two branches of an if/else. Phase 1 joins them to reaching-def
    MULTIPLE; Phase 2 must reject it (per-branch legalization unsupported)."""
    from tilelang.carver.arch import driver

    nrows, ncols = driver.get_sunmmio_device_mesh_config()
    ncores = nrows * ncols
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=A_layout),
        A2: T.MeshTensor((M, K), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), accum_dtype, layout=C_layout),
    ):
        with T.Kernel(ncores) as (cid):
            sharded_M, sharded_K = A.shape
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=2):
                if cid > 0:
                    T.copy(A[0, k * block_K], A_shared)  # writer in then
                else:
                    T.copy(A2[0, k * block_K], A_shared)  # writer in else
                T.copy(B[k * block_K, 0], B_shared)
                T.gemm(A_shared, B_shared, C_shared)
            T.copy(C_shared, C[0, 0])

    return main


def test_legalize_sunmmio_gemm_rejects_multiple_writers():
    """MULTIPLE reaching def: divergent if/else writers join to MULTIPLE.
    The pass must fatal — per-branch legalization is not supported."""
    func = bf16_gemm_multiple_writers()
    mod = tvm.IRModule({"main": func})
    with pytest.raises(Exception, match="has multiple reaching writers at the gemm site"):
        apply_pipeline_up_to_legalize_gemm(mod)


def bf16_gemm_diverging_scopes(M=32, N=32, K=32, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"):
    """The A-operand writer is in one loop; the gemm is in a separate
    sibling loop. The writer reaches the gemm (UNIQUE), but their scope
    paths diverge — neither is a prefix of the other — so Phase 3 cannot
    classify it as InPlace or hoisted, and must reject."""
    from tilelang.carver.arch import driver

    nrows, ncols = driver.get_sunmmio_device_mesh_config()
    ncores = nrows * ncols
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.MeshShardingPolicy(y=0, x=1), (nrows, ncols), accum_dtype, layout=C_layout),
    ):
        with T.Kernel(ncores) as (_cid):
            sharded_M, sharded_K = A.shape
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            T.clear(C_shared)
            for _i in T.serial(1):
                T.copy(A[0, 0], A_shared)  # writer in loop _i
            for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=2):
                T.copy(B[k * block_K, 0], B_shared)
                T.gemm(A_shared, B_shared, C_shared)  # gemm in sibling loop k
            T.copy(C_shared, C[0, 0])

    return main


def test_legalize_sunmmio_gemm_rejects_diverging_scopes():
    """Writer and gemm in sibling loops → diverging scope paths. Phase 3
    has no co-scoped or hoisted (prefix) classification for this, so it
    must reject rather than guess."""
    func = bf16_gemm_diverging_scopes()
    mod = tvm.IRModule({"main": func})
    with pytest.raises(Exception, match="writer and gemm are in diverging scope paths"):
        apply_pipeline_up_to_legalize_gemm(mod)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
