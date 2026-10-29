"""Tests for tilelang.transform.LegalizeSunmmioGemm.

The pass expands each bf16 ASRAM GEMM into the two-pass form required by
the Sunmmio bf16 tensor core (TC reads only ASRAM north). For each
matching GEMM, it:
  1. Stages supported Broadcast/guarded Put through Allgather + local Copy.
     Duplicates the resulting A-operand writer (Copy or AllgatherOp) with
     annotation ``src_offset_byte`` set to the HW ASRAM bank stripe (1024 B
     on A4E).
  2. Duplicates the GEMM with the ``accOffsetByte`` direct field set to
     ``(block_h / 2) * block_w * sizeof(C_dtype)`` (2048 for fp32 acc,
     1024 for bf16 acc).

These tests cover bf16 GEMM scenarios from #133 and #268.

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
from tilelang.transform import PassConfigKey
from tilelang.utils.target import SUNMMIO_TARGET_DESC


def apply_pipeline_up_to_legalize_gemm(mod, *, legalize=True):
    """Run the lowering pipeline up to and including LegalizeSunmmioGemm."""
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tl.transform.ResolveSunmmioMeshSymbols()(mod)
        mod = tl.transform.AddWrapperForSingleBufStore()(mod)
        mod = tl.transform.LegalizeNegativeIndex()(mod)
        mod = tl.transform.InjectAssumes()(mod)
        mod = tl.transform.Simplify()(mod)
        mod = tl.transform.InferSramScope()(mod)
        mod = tl.transform.LegalizeSunmmioDataPath()(mod)
        mod = tl.transform.LayoutReducer()(mod)
        mod = tl.transform.SunmmioLayoutInference()(mod)
        if legalize:
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


def bf16_gemm_with_allgather(M=128, N=128, K=128, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"):
    """A minimal bf16 GEMM kernel whose A-operand writer is comm.all_gather."""
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.placement.full_shard(0, 1), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.placement.full_shard(0, 1), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.placement.full_shard(0, 1), accum_dtype, layout=C_layout),
    ):
        with T.Kernel() as (_cid):
            sharded_M, sharded_K = A.local_shape
            _, sharded_N = B.local_shape
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            A_shared_dist = T.alloc_shared((block_M, block_K * T.ncols()), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            B_shared_dist = T.alloc_shared((block_K * T.nrows(), block_N), dtype)
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


def bf16_gemm_with_broadcast(
    broadcast_size=-1,
    use_put=False,
    cross_row=False,
    hoisted=False,
    guarded_put=False,
    guard_core=1,
    put_target=(0, 1),
    outer_guard=False,
    source_col=0,
    broadcast_b=False,
    broadcast_b_direction="v",
    broadcast_b_size=-1,
    broadcast_b_source=None,
    put_source=(0, 0),
    block_shape=(32, 32, 32),
):
    """Broadcast one complete BF16 A tile along each core row before GEMM."""
    from tilelang.layout import make_zz_layout

    block_M, block_N, block_K = block_shape
    M, N, K = block_M * 4, block_N * 4, block_K * 4
    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.placement.full_shard(0, 1), "bfloat16", layout=A_layout),
        B: T.MeshTensor((K, N), T.placement.full_shard(0, 1), "bfloat16", layout=B_layout),
        C: T.MeshTensor((M, N), T.placement.full_shard(0, 1), "float32", layout=C_layout),
    ):
        with T.Kernel(T.mesh_ncores()) as cid:
            _, sharded_K = A.local_shape
            A_shared = T.alloc_shared((block_M, block_K), "bfloat16", scope="shared.rsram")
            A_dist = T.alloc_shared((block_M, block_K), "bfloat16")
            B_shared = T.alloc_shared((block_K, block_N), "bfloat16")
            C_shared = T.alloc_shared((block_M, block_N), "float32")

            if hoisted:
                T.copy(A[0, 0], A_shared)
                T.comm.broadcast(A_shared, A_dist, (cid // T.mesh_ncols(), source_col), direction="h")
            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=2):
                if outer_guard:
                    if cid != 0:
                        T.copy(A[0, k * block_K], A_shared)
                        T.copy(B[k * block_K, 0], B_shared)
                        T.comm.put(A_shared, A_dist, (0, 0), (0, 1))
                        if cid == 1:
                            T.gemm(A_dist, B_shared, C_shared)
                else:
                    if not hoisted:
                        T.copy(A[0, k * block_K], A_shared)
                    if guarded_put:
                        T.copy(B[k * block_K, 0], B_shared)
                    if use_put:
                        T.comm.put(A_shared, A_dist, put_source, put_target, size=broadcast_size)
                    elif not hoisted:
                        T.comm.broadcast(
                            A_shared,
                            A_dist,
                            (0, 0) if cross_row else (cid // T.mesh_ncols(), source_col),
                            direction="h",
                            size=broadcast_size,
                        )
                    if not guarded_put:
                        if broadcast_b:
                            T.comm.broadcast(
                                B[k * block_K : k * block_K + block_K, 0:block_N],
                                B_shared,
                                (0, cid % T.mesh_ncols()) if broadcast_b_source is None else broadcast_b_source,
                                direction=broadcast_b_direction,
                                size=broadcast_b_size,
                            )
                        else:
                            T.copy(B[k * block_K, 0], B_shared)
                    if guarded_put:
                        if cid == guard_core:
                            T.gemm(A_dist, B_shared, C_shared)
                    else:
                        T.gemm(A_dist, B_shared, C_shared)
            T.copy(C_shared, C[0, 0])

    return main


def test_legalize_sunmmio_gemm_on_broadcast_writer():
    mod = apply_pipeline_up_to_legalize_gemm(tvm.IRModule({"main": bf16_gemm_with_broadcast()}))
    broadcasts = find_call_args(mod, "tl.tileop.comm_broadcast")
    assert not broadcasts
    assert count_call_op(mod, "tl.tileop.comm_allgather") == 1
    copies = find_call_args(mod, "tl.tileop.copy")
    assert sum(int(attrs.get("src_offset_byte", 0)) == 1024 for _, attrs in copies) == 1
    gemms = find_call_args(mod, "tl.tileop.gemm_py")
    assert sorted(int(args[19]) if len(args) > 19 else 0 for args, _ in gemms) == [0, 2048]

    lowered = tl.transform.LowerTileOp()(mod)
    assert [int(args[4]) for args, _ in find_call_args(lowered, "tl.broadcast_")] == [0]
    assert sorted(int(args[6]) for args, _ in find_call_args(lowered, "tl.mma_sunmmio")) == [0, 2048]


def test_legalize_sunmmio_gemm_hoisted_broadcast_restores_each_iteration():
    mod = apply_pipeline_up_to_legalize_gemm(tvm.IRModule({"main": bf16_gemm_with_broadcast(hoisted=True)}))
    assert_hoisted_direct_structure(mod, expected_acc_offset=2048, expected_pairs=1, writer_op="tl.tileop.copy")


@pytest.mark.parametrize("source_col", [0, 2, 3])
def test_broadcast_gather_selects_source_slot_and_initializes_senders(source_col):
    mod = apply_pipeline_up_to_legalize_gemm(tvm.IRModule({"main": bf16_gemm_with_broadcast(source_col=source_col)}))
    gather_args, _ = find_call_args(mod, "tl.tileop.comm_allgather")[0]
    send = gather_args[0].args[0].buffer
    receive = gather_args[1].args[0].buffer
    assert send.scope() == receive.scope() == "shared.rsram"
    assert [int(x) for x in receive.shape] == [128, 32]
    selected = []
    for args, attrs in find_call_args(mod, "tl.tileop.copy"):
        region = args[0]
        if isinstance(region, tvm.tir.Call) and region.args[0].buffer.same_as(receive):
            selected.append((int(region.args[0].indices[0]), int(attrs.get("src_offset_byte", 0))))
    assert selected == [(source_col * 32, 0), (source_col * 32, 1024)]
    fills = find_call_args(mod, "tl.tileop.fill")
    assert any(args[0].args[0].buffer.same_as(send) and args[1].value == 0 for args, _ in fills)


def test_broadcast_gather_stages_full_mesh_b_operand():
    mod = apply_pipeline_up_to_legalize_gemm(
        tvm.IRModule(
            {
                "main": bf16_gemm_with_broadcast(
                    broadcast_b=True,
                    broadcast_b_direction="all",
                    broadcast_b_source=(2, 3),
                )
            }
        )
    )
    gathers = find_call_args(mod, "tl.tileop.comm_allgather")
    full_mesh = [(args, attrs) for args, attrs in gathers if int(args[2]) == 2]
    assert len(full_mesh) == 1
    receive = full_mesh[0][0][1].args[0].buffer
    assert [int(x) for x in receive.shape] == [16 * 32, 32]
    selected = []
    for args, _ in find_call_args(mod, "tl.tileop.copy"):
        region = args[0]
        if isinstance(region, tvm.tir.Call) and region.args[0].buffer.same_as(receive):
            selected.append(int(region.args[0].indices[0]))
    assert selected == [11 * 32]
    loop = _find_k_loop_for(mod)
    copy_op = tvm.tir.op.Op.get("tl.tileop.copy")
    assert any(_is_call(stmt, copy_op) and stmt.value.args[0].args[0].buffer.same_as(receive) for stmt in loop.body.seq)
    lowered = tl.transform.LowerTileOp()(mod)
    assert count_call_op(lowered, "tl.broadcast_") == 6


def test_legalize_sunmmio_gemm_preserves_unsupported_b_broadcast():
    mod = apply_pipeline_up_to_legalize_gemm(
        tvm.IRModule(
            {
                "main": bf16_gemm_with_broadcast(
                    broadcast_b=True,
                    broadcast_b_size=512,
                )
            }
        )
    )
    broadcasts = find_call_args(mod, "tl.tileop.comm_broadcast")
    assert len(broadcasts) == 1
    assert int(broadcasts[0][0][2]) == 512
    assert count_call_op(mod, "tl.tileop.comm_allgather") == 1
    assert count_call_op(mod, "tl.tileop.gemm_py") == 2
    tl.transform.LowerTileOp()(mod)


def test_legalize_sunmmio_gemm_rejects_partial_broadcast_writer():
    mod = tvm.IRModule({"main": bf16_gemm_with_broadcast(broadcast_size=512)})
    with pytest.raises(Exception, match="broadcast writer must transfer its full source region"):
        apply_pipeline_up_to_legalize_gemm(mod)


def test_legalize_sunmmio_gemm_rejects_cross_row_broadcast_writer():
    mod = tvm.IRModule({"main": bf16_gemm_with_broadcast(cross_row=True)})
    with pytest.raises(Exception, match="horizontal broadcast source must be in the GEMM core's row"):
        apply_pipeline_up_to_legalize_gemm(mod)


def test_legalize_sunmmio_gemm_rejects_single_target_put():
    mod = tvm.IRModule({"main": bf16_gemm_with_broadcast(use_put=True)})
    with pytest.raises(Exception, match="put writer for bf16 ASRAM A is not supported"):
        apply_pipeline_up_to_legalize_gemm(mod)


def test_legalize_sunmmio_gemm_keeps_gather_outside_target_guard():
    mod = tvm.IRModule({"main": bf16_gemm_with_broadcast(guarded_put=True)})
    mod = apply_pipeline_up_to_legalize_gemm(mod)
    loop = _find_k_loop_for(mod)
    assert count_call_op(mod, "tl.tileop.comm_allgather") == 1
    assert count_call_op(mod, "tl.tileop.gemm_py") == 2
    guard = list(loop.body.seq)[-1]
    assert isinstance(guard, tvm.tir.IfThenElse)
    inner = tvm.IRModule({"main": mod["main"].with_body(guard.then_case)})
    assert count_call_op(inner, "tl.tileop.comm_allgather") == 0


@pytest.mark.parametrize("broadcast_b", [False, True])
def test_broadcast_gemm_codegen_validates_with_npuir(tmp_path, broadcast_b):
    from testing.python.sunmmio.common.codegen_validation import validate_sunmmio_codegen_with_npuir_opt

    src = validate_sunmmio_codegen_with_npuir_opt(
        bf16_gemm_with_broadcast(broadcast_b=broadcast_b),
        tmp_path,
        mlir_filename="issue268_broadcast_gemm.mlir",
        expected_tokens=("suvm.mcast_async", "suvm.tc.mma"),
        opt_args=("--verify-each", "--suvm-to-llvm-pipeline"),
    )
    assert src.count("suvm.mcast_async") == 1 + int(broadcast_b)
    assert "source_mask" not in src
    assert "link.receive" not in src
    assert src.count("suvm.tc.mma") == 2


@pytest.mark.parametrize(
    "source,target,block_shape",
    [(0, 1, (32, 32, 32)), (3, 0, (32, 32, 64)), (7, 4, (64, 64, 64)), (12, 15, (32, 32, 32))],
)
def test_legalize_sunmmio_gemm_on_guarded_put_writer(source, target, block_shape):
    mod = apply_pipeline_up_to_legalize_gemm(
        tvm.IRModule(
            {
                "main": bf16_gemm_with_broadcast(
                    use_put=True,
                    guarded_put=True,
                    put_source=divmod(source, 4),
                    put_target=divmod(target, 4),
                    guard_core=target,
                    block_shape=block_shape,
                )
            }
        )
    )
    assert count_call_op(mod, "tl.tileop.comm_put") == 0
    broadcasts = find_call_args(mod, "tl.tileop.comm_broadcast")
    assert not broadcasts
    allgathers = find_call_args(mod, "tl.tileop.comm_allgather")
    assert len(allgathers) == 1
    send = allgathers[0][0][0].args[0].buffer
    gather = allgathers[0][0][1].args[0].buffer
    assert send.scope() == gather.scope() == "shared.rsram"
    assert [int(x) for x in gather.shape] == [block_shape[0] * 4, block_shape[2]]
    assert sorted(int(args[19]) if len(args) > 19 else 0 for args, _ in find_call_args(mod, "tl.tileop.gemm_py")) == [0, 2048]
    loop = _find_k_loop_for(mod)
    seq = list(loop.body.seq)
    assert isinstance(seq[-1], tvm.tir.IfThenElse)
    inner = list(seq[-1].then_case.seq)
    copy_op = tvm.tir.op.Op.get("tl.tileop.copy")
    assert [_is_call(stmt, copy_op) for stmt in inner] == [True, False, True, False]
    # Both passes read the source slot directly, without a private receive tile.
    for stmt in (inner[0], inner[2]):
        region = stmt.value.args[0].args[0]
        assert region.buffer.same_as(gather)
        assert int(region.indices[0]) == (source % 4) * block_shape[0]
    assert count_call_op(tvm.IRModule({"main": mod["main"].with_body(seq[-1])}), "tl.tileop.comm_allgather") == 0
    assert sorted(
        int(attrs.get("src_offset_byte", 0))
        for _, attrs in find_call_args(tvm.IRModule({"main": mod["main"].with_body(seq[-1])}), "tl.tileop.copy")
    ) == [0, 1024]
    lowered = tl.transform.LowerTileOp()(mod)
    # Allgather is lowered to the backend multicast primitive.  The important
    # property here is that the guarded Put contributes exactly one public
    # communication operation; there is no private put receive buffer or
    # second communication generated by the two GEMM passes.
    mcasts = find_call_args(lowered, "tl.broadcast_")
    assert len(mcasts) == 1
    assert int(mcasts[0][0][3]) == 15  # all four cores in the row
    assert int(mcasts[0][0][2]) == 0  # horizontal multicast
    assert int(mcasts[0][0][4]) == 0  # no communication source offset
    again = tl.transform.LegalizeSunmmioGemm()(mod)
    tvm.ir.assert_structural_equal(again, mod)
    assert count_call_op(again, "tl.tileop.gemm_py") == 2


def test_legalize_sunmmio_gemm_rejects_put_guard_on_wrong_core():
    mod = tvm.IRModule({"main": bf16_gemm_with_broadcast(use_put=True, guarded_put=True, guard_core=2)})
    with pytest.raises(Exception, match="put writer for bf16 ASRAM A is not supported"):
        apply_pipeline_up_to_legalize_gemm(mod)


def test_legalize_sunmmio_gemm_rejects_partial_guarded_put():
    mod = tvm.IRModule({"main": bf16_gemm_with_broadcast(use_put=True, guarded_put=True, broadcast_size=512)})
    with pytest.raises(Exception, match="put writer for bf16 ASRAM A is not supported"):
        apply_pipeline_up_to_legalize_gemm(mod)


def test_legalize_sunmmio_gemm_rejects_put_when_outer_guard_skips_source():
    mod = tvm.IRModule({"main": bf16_gemm_with_broadcast(outer_guard=True)})
    with pytest.raises(Exception, match="put writer for bf16 ASRAM A is not supported"):
        apply_pipeline_up_to_legalize_gemm(mod)


def test_legalize_sunmmio_gemm_diagonal_put_still_requires_rsram_receive():
    mod = tvm.IRModule({"main": bf16_gemm_with_broadcast(use_put=True, guarded_put=True, put_target=(1, 1), guard_core=5)})
    with pytest.raises(Exception, match="Unsupported communication path from shared.rsram to shared.asram"):
        apply_pipeline_up_to_legalize_gemm(mod)


def test_guarded_put_gemm_codegen_validates_with_npuir(tmp_path):
    from testing.python.sunmmio.common.codegen_validation import validate_sunmmio_codegen_with_npuir_opt

    src = validate_sunmmio_codegen_with_npuir_opt(
        bf16_gemm_with_broadcast(use_put=True, guarded_put=True),
        tmp_path,
        mlir_filename="issue268_guarded_put_gemm.mlir",
        expected_tokens=("suvm.mcast_async", "suvm.tc.mma"),
        opt_args=("--verify-each", "--suvm-to-llvm-pipeline"),
    )
    assert src.count("suvm.mcast_async") == 1
    assert "source_mask" not in src
    assert "link.receive" not in src
    assert src.count("suvm.tc.mma") == 2


@pytest.mark.parametrize("use_put", [False, True])
@pytest.mark.parametrize(
    "source,target,block_shape",
    [(0, 1, (32, 32, 32)), (3, 0, (32, 32, 64)), (7, 4, (64, 64, 64)), (12, 15, (32, 32, 32))],
)
def test_bf16_comm_gemm_two_communications(use_put, source, target, block_shape):
    func = bf16_gemm_with_broadcast(
        use_put=use_put,
        guarded_put=use_put,
        source_col=source % 4,
        put_source=divmod(source, 4),
        put_target=divmod(target, 4),
        guard_core=target,
        block_shape=block_shape,
    )
    with tvm.transform.PassContext(config={PassConfigKey.TL_SUNMMIO_GEMM_TWO_COMMUNICATIONS: True}):
        mod = apply_pipeline_up_to_legalize_gemm(tvm.IRModule({"main": func}))
        gathers = find_call_args(mod, "tl.tileop.comm_allgather")
        assert len(gathers) == 2
        assert [int(attrs.get("src_offset_byte", 0)) for _, attrs in gathers] == [0, 1024]
        receive = gathers[0][0][1].args[0].buffer
        assert receive.scope() == "shared.asram"
        assert [int(x) for x in receive.shape] == [4, 1, block_shape[0], block_shape[2]]
        gemms = find_call_args(mod, "tl.tileop.gemm_py")
        assert [int(args[19]) if len(args) > 19 else 0 for args, _ in gemms] == [0, 2048]
        for args, _ in gemms:
            a = args[0].args[0]
            assert a.buffer.same_as(receive)
            assert [int(x) for x in a.indices] == [source % 4, 0, 0, 0]
        for args, _ in find_call_args(mod, "tl.tileop.copy"):
            assert args[1].args[0].buffer.scope() != "shared.asram"

        def check_guard(node):
            if isinstance(node, tvm.tir.IfThenElse):
                guarded = tvm.IRModule({"main": mod["main"].with_body(node)})
                assert count_call_op(guarded, "tl.tileop.comm_allgather") == 0

        tvm.tir.stmt_functor.post_order_visit(mod["main"].body, check_guard)
        again = tl.transform.LegalizeSunmmioGemm()(mod)
        tvm.ir.assert_structural_equal(again, mod)
        lowered = tl.transform.LowerTileOp()(mod)
        mcasts = find_call_args(lowered, "tl.broadcast_")
        assert len(mcasts) == 2
        assert [int(args[4]) for args, _ in mcasts] == [0, 1024]


@pytest.mark.parametrize("use_put", [False, True])
def test_bf16_comm_gemm_default_is_one_communication(use_put):
    original = tvm.IRModule({"main": bf16_gemm_with_broadcast(use_put=use_put, guarded_put=use_put)})
    prepared = apply_pipeline_up_to_legalize_gemm(original, legalize=False)
    default = tl.transform.LegalizeSunmmioGemm()(prepared)
    with tvm.transform.PassContext(config={PassConfigKey.TL_SUNMMIO_GEMM_TWO_COMMUNICATIONS: False}):
        explicit = tl.transform.LegalizeSunmmioGemm()(prepared)
    # Fresh staging buffers are identity-keyed in layout_map. Check the data
    # path rather than equating separately allocated layout-map keys.
    for mod in (default, explicit):
        gathers = find_call_args(mod, "tl.tileop.comm_allgather")
        assert len(gathers) == 1
        assert gathers[0][0][1].args[0].buffer.scope() == "shared.rsram"
        copies = []
        for args, attrs in find_call_args(mod, "tl.tileop.copy"):
            dst = args[1].args[0] if isinstance(args[1], tvm.tir.Call) else args[1]
            if dst.buffer.scope() == "shared.asram":
                copies.append(attrs)
        assert [int(attrs.get("src_offset_byte", 0)) for attrs in copies] == [0, 1024]
        assert count_call_op(mod, "tl.tileop.gemm_py") == 2


@pytest.mark.parametrize("kwargs", [{"hoisted": True}, {"guarded_put": True}])
def test_two_communication_broadcast_rejects_unsafe_scope(kwargs):
    original = tvm.IRModule({"main": bf16_gemm_with_broadcast(**kwargs)})
    # The default staged strategy remains valid for these scopes.
    apply_pipeline_up_to_legalize_gemm(original)
    with (
        tvm.transform.PassContext(config={PassConfigKey.TL_SUNMMIO_GEMM_TWO_COMMUNICATIONS: True}),
        pytest.raises(Exception, match="writer's unconditional scope"),
    ):
        apply_pipeline_up_to_legalize_gemm(original)


@pytest.mark.parametrize("use_put", [False, True])
@pytest.mark.parametrize("block_shape", [(32, 32, 32), (64, 64, 64)])
def test_two_communication_gemm_codegen_validates_with_npuir(tmp_path, use_put, block_shape):
    from testing.python.sunmmio.common.codegen_validation import (
        codegen_sunmmio_suvm_mlir,
        validate_suvm_mlir_with_npuir_opt,
    )
    from testing.python.sunmmio.common.compile_pipeline import compile_test

    func = bf16_gemm_with_broadcast(use_put=use_put, guarded_put=use_put, broadcast_b=not use_put, block_shape=block_shape)
    _, device_mod = compile_test(
        func,
        target=tvm.target.Target(SUNMMIO_TARGET_DESC),
        remove_header=True,
        pass_configs={PassConfigKey.TL_SUNMMIO_GEMM_TWO_COMMUNICATIONS: True},
    )
    src = codegen_sunmmio_suvm_mlir(device_mod)
    validate_suvm_mlir_with_npuir_opt(src, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))
    assert src.count("suvm.mcast_async") == (2 if use_put else 3)
    assert src.count("suvm.tc.mma") == 2


@pytest.mark.parametrize("kind", ["copy", "allgather", "single_pass_broadcast"])
def test_two_communications_does_not_change_other_gemms(kind):
    if kind == "copy":
        func = bf16_gemm_with_copy_to_asram()
    elif kind == "allgather":
        func = bf16_gemm_with_allgather()
    else:
        func = bf16_gemm_with_broadcast(block_shape=(16, 32, 32))
    original = tvm.IRModule({"main": func})
    prepared = apply_pipeline_up_to_legalize_gemm(original, legalize=False)
    default = tl.transform.LegalizeSunmmioGemm()(prepared)
    with tvm.transform.PassContext(config={PassConfigKey.TL_SUNMMIO_GEMM_TWO_COMMUNICATIONS: True}):
        optional = tl.transform.LegalizeSunmmioGemm()(prepared)
    tvm.ir.assert_structural_equal(default, optional, map_free_vars=True)


def bf16_gemm_with_unrelated_broadcast(kind):
    from tilelang.layout import make_zz_layout

    layout = make_zz_layout((128, 128), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((128, 128), T.placement.full_shard(0, 1), "bfloat16", layout=layout),
        B: T.MeshTensor((128, 128), T.placement.full_shard(0, 1), "bfloat16", layout=layout),
        C: T.MeshTensor((128, 128), T.placement.full_shard(0, 1), "float32", layout=layout),
        D: T.MeshTensor((128, 128), T.placement.full_shard(0, 1), "bfloat16", layout=layout),
        flag: T.int32,
    ):
        with T.Kernel(T.mesh_ncores()) as cid:
            a = T.alloc_shared((32, 32), "bfloat16")
            b = T.alloc_shared((32, 32), "bfloat16")
            c = T.alloc_shared((32, 32), "float32")
            send = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            recv = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            T.copy(A[0:32, 0:32], a)
            T.copy(B[0:32, 0:32], b)
            T.gemm(a, b, c, clear_accum=True)
            T.copy(c, C[0:32, 0:32])
            T.copy(A[0:32, 0:32], send)
            T.clear(recv)
            if kind == "conditional":
                if flag > 0:
                    T.comm.broadcast(send, recv, (cid // T.mesh_ncols(), 0), direction="h")
            elif kind == "all":
                T.comm.broadcast(send, recv, (0, 0), direction="all")
            else:
                T.comm.broadcast(send, recv, (cid // T.mesh_ncols(), 0), direction="h")
            T.copy(recv, D[0:32, 0:32])

    return main


@pytest.mark.parametrize("twice", [False, True])
@pytest.mark.parametrize("kind", ["plain", "conditional", "all"])
def test_bf16_gemm_preserves_unrelated_broadcast(kind, twice):
    original = tvm.IRModule({"main": bf16_gemm_with_unrelated_broadcast(kind)})
    with tvm.transform.PassContext(config={PassConfigKey.TL_SUNMMIO_GEMM_TWO_COMMUNICATIONS: twice}):
        prepared = apply_pipeline_up_to_legalize_gemm(original, legalize=False)
        mod = tl.transform.LegalizeSunmmioGemm()(prepared)
    # The unrelated communication keeps its arguments, annotations and scope.
    before = find_call_args(prepared, "tl.tileop.comm_broadcast")
    after = find_call_args(mod, "tl.tileop.comm_broadcast")
    assert len(before) == len(after) == 1
    tvm.ir.assert_structural_equal(tvm.runtime.convert(before), tvm.runtime.convert(after))
    assert count_call_op(mod, "tl.tileop.comm_allgather") == 0
    assert count_call_op(mod, "tl.tileop.gemm_py") == 2
    tl.transform.LowerTileOp()(mod)


def bf16_gemm_with_nested_broadcast(sibling_consumer=False):
    from tilelang.layout import make_zz_layout

    layout = make_zz_layout((128, 128), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((128, 128), T.placement.full_shard(0, 1), "bfloat16", layout=layout),
        B: T.MeshTensor((128, 128), T.placement.full_shard(0, 1), "bfloat16", layout=layout),
        C: T.MeshTensor((128, 128), T.placement.full_shard(0, 1), "float32", layout=layout),
    ):
        with T.Kernel(T.mesh_ncores()) as cid:
            send = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            a = T.alloc_shared((32, 32), "bfloat16")
            b = T.alloc_shared((32, 32), "bfloat16")
            c = T.alloc_shared((32, 32), "float32")
            T.copy(A[0:32, 0:32], send)
            T.copy(B[0:32, 0:32], b)
            with T.block("writer"):
                T.comm.broadcast(send, a, (cid // T.mesh_ncols(), 0), direction="h")
            if sibling_consumer:
                with T.block("consumer"):
                    T.gemm(a, b, c, clear_accum=True)
            else:
                T.gemm(a, b, c, clear_accum=True)
            T.copy(c, C[0:32, 0:32])

    return main


@pytest.mark.parametrize("twice", [False, True])
@pytest.mark.parametrize("sibling_consumer", [False, True])
def test_broadcast_staging_outlives_writer_block(tmp_path, twice, sibling_consumer):
    from testing.python.sunmmio.common.codegen_validation import (
        codegen_sunmmio_suvm_mlir,
        validate_suvm_mlir_with_npuir_opt,
    )
    from testing.python.sunmmio.common.compile_pipeline import compile_test

    func = bf16_gemm_with_nested_broadcast(sibling_consumer)
    config = {PassConfigKey.TL_SUNMMIO_GEMM_TWO_COMMUNICATIONS: twice}
    with tvm.transform.PassContext(config=config):
        mod = apply_pipeline_up_to_legalize_gemm(tvm.IRModule({"main": func}))

    def check_allocations(node):
        if isinstance(node, tvm.tir.Block):
            staging = [buf for buf in node.alloc_buffers if "_broadcast_" in buf.name]
            if staging:
                assert str(node.name_hint) == "tilelang_root"
                assert len(staging) == 2
                assert all(buf in node.annotations["layout_map"] for buf in staging)

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, check_allocations)
    assert count_call_op(mod, "tl.tileop.comm_allgather") == (2 if twice else 1)
    _, device_mod = compile_test(func, target=tvm.target.Target(SUNMMIO_TARGET_DESC), remove_header=True, pass_configs=config)
    src = codegen_sunmmio_suvm_mlir(device_mod)
    validate_suvm_mlir_with_npuir_opt(src, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))
    assert src.count("suvm.tc.mma") == 2


def bf16_gemm_with_copy_to_asram(M=128, N=128, K=128, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"):
    """A minimal bf16 GEMM whose A_dist writer is a plain T.copy (no
    all_gather / HLink). After LegalizeSunmmioDataPath stages the DRAM->ASRAM
    transfer through an RSRAM buffer, the direct writer of A_dist becomes a
    *RSRAM->ASRAM* T.copy. This is the case the user-facing C++ kernels
    (compiler-samples/05_matmul, 06_matmul) use when A is staged through
    RSRAM and then sent to ASRAM without HLink broadcast."""
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.placement.full_shard(0, 1), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.placement.full_shard(0, 1), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.placement.full_shard(0, 1), accum_dtype, layout=C_layout),
    ):
        with T.Kernel() as (_cid):
            sharded_M, sharded_K = A.local_shape
            _, sharded_N = B.local_shape
            # A staged via direct copy (no all_gather): full DRAM->ASRAM transfer.
            A_dist = T.alloc_shared((block_M, block_K * T.ncols()), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            B_shared_dist = T.alloc_shared((block_K * T.nrows(), block_N), dtype)
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


def bf16_gemm_with_hoisted_copy_writer(M=128, N=128, K=128, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"):
    """A minimal bf16 GEMM whose A-operand writer is HOISTED out of the
    K-loop — the flash-attention pattern. The single T.copy(A → A_shared)
    runs once at kernel scope; the gemm sits inside a T.Pipelined loop and
    reuses A_shared across iterations. After LegalizeSunmmioDataPath stages
    DRAM→ASRAM through an RSRAM buffer, the direct writer of A_shared is a
    RSRAM→ASRAM copy at kernel scope, while the gemm is in the inner
    loop's scope. HoistedShadow should kick in."""
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.placement.full_shard(0, 1), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.placement.full_shard(0, 1), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.placement.full_shard(0, 1), accum_dtype, layout=C_layout),
    ):
        with T.Kernel() as (_cid):
            sharded_M, sharded_K = A.local_shape
            _, sharded_N = B.local_shape
            # A is loaded ONCE at kernel scope, before the K-loop. The
            # whole-K stripe of A is brought in; B is streamed inside the loop.
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.copy(A[0:block_M, 0:block_K], A_shared)  # hoisted A writer
            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=2):
                T.copy(B[k * block_K, 0], B_shared)
                T.gemm(A_shared, B_shared, C_shared)

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


def bf16_gemm_with_hoisted_copy_writer_bf16_acc(M=128, N=128, K=128, block_M=32, block_N=32, block_K=32, dtype="bfloat16"):
    """Same hoisted-copy pattern as above but with a bf16 accumulator
    instead of fp32. The accOffsetByte formula
    (block_h/2)*block_w*sizeof(C_dtype) gives 1024 for bf16 acc with a
    32x32 block, vs 2048 for fp32 acc — this exercises the dtype-dependent
    arithmetic in ComputeAccOffsetBytes."""
    accum_dtype = dtype
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.placement.full_shard(0, 1), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.placement.full_shard(0, 1), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.placement.full_shard(0, 1), accum_dtype, layout=C_layout),
    ):
        with T.Kernel() as (_cid):
            sharded_M, sharded_K = A.local_shape
            _, sharded_N = B.local_shape
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.copy(A[0:block_M, 0:block_K], A_shared)
            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=2):
                T.copy(B[k * block_K, 0], B_shared)
                T.gemm(A_shared, B_shared, C_shared)

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
    M=128, N=128, K=128, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"
):
    """Two ASRAM A operands, each loaded by its own hoisted T.copy at
    kernel scope, each consumed by a distinct gemm in the same K-loop.
    Exercises the case where a single kernel needs multiple independent
    HoistedShadow rewrites — each A_dist must get its own shadow buffer
    co-located with it in the kernel Block's alloc_buffers."""
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.placement.full_shard(0, 1), dtype, layout=A_layout),
        A2: T.MeshTensor((M, K), T.placement.full_shard(0, 1), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.placement.full_shard(0, 1), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.placement.full_shard(0, 1), accum_dtype, layout=C_layout),
        C2: T.MeshTensor((M, N), T.placement.full_shard(0, 1), accum_dtype, layout=C_layout),
    ):
        with T.Kernel() as (_cid):
            sharded_M, sharded_K = A.local_shape
            _, sharded_N = B.local_shape
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            A2_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            C2_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.copy(A[0:block_M, 0:block_K], A_shared)  # hoisted writer #1
            T.copy(A2[0:block_M, 0:block_K], A2_shared)  # hoisted writer #2
            T.clear(C_shared)
            T.clear(C2_shared)
            for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=2):
                T.copy(B[k * block_K, 0], B_shared)
                T.gemm(A_shared, B_shared, C_shared)
                T.gemm(A2_shared, B_shared, C2_shared)

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
    M=128, N=128, K=128, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"
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
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.placement.full_shard(0, 1), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.placement.full_shard(0, 1), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.placement.full_shard(0, 1), accum_dtype, layout=C_layout),
    ):
        with T.Kernel() as (_cid):
            sharded_M, sharded_K = A.local_shape
            _, sharded_N = B.local_shape
            # User-declared RSRAM scratch — visible by name to the test so
            # we can verify cleanliness detects writes to it.
            A_rsram = T.alloc_shared((block_M, block_K), dtype, scope="shared.rsram")
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.copy(A[0:block_M, 0:block_K], A_rsram)  # initial source population
            T.copy(A_rsram, A_shared)  # hoisted writer (reads A_rsram)
            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=2):
                T.copy(A[0:block_M, 0:block_K], A_rsram)  # MUTATES the writer's source per iter
                T.copy(B[k * block_K, 0], B_shared)
                T.gemm(A_shared, B_shared, C_shared)

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


def bf16_gemm_with_broadcast_dirtying_hoisted_source():
    """A broadcast after a hoisted writer must invalidate its RSRAM source."""
    from tilelang.layout import make_zz_layout

    layout = make_zz_layout((128, 128), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((128, 128), T.placement.full_shard(0, 1), "bfloat16", layout=layout),
        B: T.MeshTensor((128, 128), T.placement.full_shard(0, 1), "bfloat16", layout=layout),
        C: T.MeshTensor((128, 128), T.placement.full_shard(0, 1), "float32", layout=layout),
    ):
        with T.Kernel(T.mesh_ncores()) as cid:
            source = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            replacement = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            a_dist = T.alloc_shared((32, 32), "bfloat16")
            b_shared = T.alloc_shared((32, 32), "bfloat16")
            c_shared = T.alloc_shared((32, 32), "float32")
            T.copy(A[0, 0], source)
            T.copy(source, a_dist)
            T.clear(c_shared)
            for _k in T.Pipelined(1, num_stages=2):
                T.copy(A[0, 0], replacement)
                T.comm.broadcast(replacement, source, (cid // T.mesh_ncols(), 0), direction="h")
                T.copy(B[0, 0], b_shared)
                T.gemm(a_dist, b_shared, c_shared)
            T.copy(c_shared, C[0, 0])

    return main


def test_legalize_sunmmio_gemm_broadcast_marks_reissue_source_dirty():
    mod = apply_pipeline_up_to_legalize_gemm(tvm.IRModule({"main": bf16_gemm_with_broadcast_dirtying_hoisted_source()}))
    assert_hoisted_shadow_structure(mod, expected_acc_offset=2048, expected_pairs=1)


def bf16_gemm_with_hoisted_allgather(M=128, N=128, K=128, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"):
    """bf16 GEMM whose A-operand writer is a HOISTED comm.all_gather. A is
    copied and gathered once at kernel scope; the gemm in the inner K-loop
    reuses the gathered A_shared_dist. The all_gather's source (A_shared) is
    written once before the gather and never mutated — clean — so the pass
    should pick HoistedDirect and re-issue the all_gather inside the loop."""
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.placement.full_shard(0, 1), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.placement.full_shard(0, 1), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.placement.full_shard(0, 1), accum_dtype, layout=C_layout),
    ):
        with T.Kernel() as (_cid):
            sharded_M, sharded_K = A.local_shape
            _, sharded_N = B.local_shape
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            A_shared_dist = T.alloc_shared((block_M, block_K * T.ncols()), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            B_shared_dist = T.alloc_shared((block_K * T.nrows(), block_N), dtype)
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
    M=128, N=128, K=128, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"
):
    """Hoisted Allgather whose source (A_shared) is mutated inside the
    K-loop. The cleanliness analyzer must flag the source dirty so the
    pass picks HoistedShadow: snapshot A_shared into an RSRAM stage, then
    re-issue the all_gather from the stage inside the loop."""
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.placement.full_shard(0, 1), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.placement.full_shard(0, 1), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.placement.full_shard(0, 1), accum_dtype, layout=C_layout),
    ):
        with T.Kernel() as (_cid):
            sharded_M, sharded_K = A.local_shape
            _, sharded_N = B.local_shape
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            A_shared_dist = T.alloc_shared((block_M, block_K * T.ncols()), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            B_shared_dist = T.alloc_shared((block_K * T.nrows(), block_N), dtype)
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
        src_offset_byte), alongside the original writer for unexpanded readers.
        The snapshot is always a plain Copy (it just freezes the source),
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


def bf16_gemm_multi_consumer_groupable(M=128, N=128, K=128, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"):
    """Two bf16 gemms that share one ASRAM A operand and sit back-to-back
    (textually adjacent) in the K-loop — a multi-B fan-out. The shared
    A_shared is written by a single co-scoped copy. Groupable: the pass
    should emit one [G1, G2, W', G1', G2'] block."""
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.placement.full_shard(0, 1), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.placement.full_shard(0, 1), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.placement.full_shard(0, 1), accum_dtype, layout=C_layout),
        C2: T.MeshTensor((M, N), T.placement.full_shard(0, 1), accum_dtype, layout=C_layout),
    ):
        with T.Kernel() as (_cid):
            sharded_M, sharded_K = A.local_shape
            _, sharded_N = B.local_shape
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


def bf16_gemm_multi_consumer_non_groupable(
    M=128, N=128, K=128, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"
):
    """Two bf16 gemms sharing one ASRAM A operand, but with a copy between
    them — NOT textually adjacent. Non-groupable: the pass must fall back
    to per-gemm Reissue (each self-restores), and emit a warning."""
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.placement.full_shard(0, 1), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.placement.full_shard(0, 1), dtype, layout=B_layout),
        B2: T.MeshTensor((K, N), T.placement.full_shard(0, 1), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.placement.full_shard(0, 1), accum_dtype, layout=C_layout),
        C2: T.MeshTensor((M, N), T.placement.full_shard(0, 1), accum_dtype, layout=C_layout),
    ):
        with T.Kernel() as (_cid):
            sharded_M, sharded_K = A.local_shape
            _, sharded_N = B.local_shape
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


def test_legalize_sunmmio_gemm_singleton_branch_consumers():
    """Both singleton branches must restore A and execute both BF16 stripes."""
    from tilelang.layout import make_zz_layout

    layout = make_zz_layout((128, 128), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((128, 128), T.placement.full_shard(0, 1), "bfloat16", layout=layout),
        B: T.MeshTensor((128, 128), T.placement.full_shard(0, 1), "bfloat16", layout=layout),
        C: T.MeshTensor((128, 128), T.placement.full_shard(0, 1), "float32", layout=layout),
    ):
        with T.Kernel(T.mesh_ncores()) as cid:
            A_shared = T.alloc_shared((32, 32), "bfloat16")
            B_shared = T.alloc_shared((32, 32), "bfloat16")
            C_shared = T.alloc_shared((32, 32), "float32")
            T.copy(A[0:32, 0:32], A_shared)
            T.copy(B[0:32, 0:32], B_shared)
            if cid > 0:
                T.gemm(A_shared, B_shared, C_shared, clear_accum=True)
            else:
                T.gemm(A_shared, B_shared, C_shared)
            T.copy(C_shared, C[0:32, 0:32])

    mod = apply_pipeline_up_to_legalize_gemm(tvm.IRModule({"main": main}))
    assert count_call_op(mod, "tl.tileop.gemm_py") == 4
    branches = []
    tvm.tir.stmt_functor.post_order_visit(
        mod["main"].body, lambda node: branches.append(node) if isinstance(node, tvm.tir.IfThenElse) else None
    )
    assert len(branches) == 1
    for branch in (branches[0].then_case, branches[0].else_case):
        assert isinstance(branch, tvm.tir.SeqStmt)
        calls = [stmt.value for stmt in branch.seq]
        assert [call.op.name for call in calls] == ["tl.tileop.copy", "tl.tileop.gemm_py", "tl.tileop.copy", "tl.tileop.gemm_py"]
        assert _copy_buf(calls[0], 1).same_as(_copy_buf(calls[1], 0))
        assert _copy_buf(calls[2], 1).same_as(_copy_buf(calls[3], 0))
        assert int(calls[2].annotations["src_offset_byte"]) == 1024
        assert (int(calls[1].args[19]) if len(calls[1].args) > 19 else 0) == 0
        assert int(calls[3].args[19]) == 2048
    tvm.ir.assert_structural_equal(mod, _run_legalize_gemm_again(mod))


def bf16_flash_attention_shaped(M=128, N=128, K=128, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"):
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
    from tilelang.layout import make_zz_layout

    Q_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    Kt_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    V_layout = make_zz_layout((N, N), [0, 1], (32, 32))
    O_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        Q: T.MeshTensor((M, K), T.placement.full_shard(0, 1), dtype, layout=Q_layout),
        Kt: T.MeshTensor((K, N), T.placement.full_shard(0, 1), dtype, layout=Kt_layout),
        V: T.MeshTensor((N, N), T.placement.full_shard(0, 1), dtype, layout=V_layout),
        O: T.MeshTensor((M, N), T.placement.full_shard(0, 1), accum_dtype, layout=O_layout),
    ):
        with T.Kernel() as (_cid):
            sharded_M, sharded_K = Q.local_shape
            Q_shared = T.alloc_shared((block_M, block_K), dtype)  # hoisted A (Q)
            K_shared = T.alloc_shared((block_K, block_N), dtype)
            S_shared = T.alloc_shared((block_M, block_N), dtype)  # bf16 QK output
            S_cast = T.alloc_shared((block_M, block_N), dtype)  # co-scoped A (PV)
            V_shared = T.alloc_shared((block_N, block_N), dtype)
            O_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.copy(Q[0:block_M, 0:block_K], Q_shared)  # hoisted Q writer
            T.clear(S_shared)
            T.clear(O_shared)
            for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=2):
                T.copy(Kt[k * block_K, 0], K_shared)
                T.gemm(Q_shared, K_shared, S_shared)  # QK gemm: A = Q (hoisted)
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


def bf16_gemm_no_reaching_writer(M=128, N=128, K=128, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"):
    """A bf16 ASRAM gemm whose A operand is never written — no reaching
    writer. Phase 1 records the gemm with reaching-def NONE; Phase 2 must
    reject it."""
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.placement.full_shard(0, 1), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.placement.full_shard(0, 1), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.placement.full_shard(0, 1), accum_dtype, layout=C_layout),
    ):
        with T.Kernel() as (_cid):
            sharded_M, sharded_K = A.local_shape
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


def bf16_gemm_multiple_writers(M=128, N=128, K=128, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"):
    """A bf16 ASRAM gemm whose A operand is written by divergent writers in
    the two branches of an if/else. Phase 1 joins them to reaching-def
    MULTIPLE; Phase 2 must reject it (per-branch legalization unsupported)."""
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.placement.full_shard(0, 1), dtype, layout=A_layout),
        A2: T.MeshTensor((M, K), T.placement.full_shard(0, 1), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.placement.full_shard(0, 1), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.placement.full_shard(0, 1), accum_dtype, layout=C_layout),
    ):
        with T.Kernel(T.mesh_ncores()) as cid:
            sharded_M, sharded_K = A.local_shape
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


def bf16_gemm_diverging_scopes(M=128, N=128, K=128, block_M=32, block_N=32, block_K=32, dtype="bfloat16", accum_dtype="float32"):
    """The A-operand writer is in one loop; the gemm is in a separate
    sibling loop. The writer reaches the gemm (UNIQUE), but their scope
    paths diverge — neither is a prefix of the other — so Phase 3 cannot
    classify it as InPlace or hoisted, and must reject."""
    from tilelang.layout import make_zz_layout

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.placement.full_shard(0, 1), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.placement.full_shard(0, 1), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.placement.full_shard(0, 1), accum_dtype, layout=C_layout),
    ):
        with T.Kernel() as (_cid):
            sharded_M, sharded_K = A.local_shape
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


@pytest.mark.parametrize("branch_pv", [False, True])
def test_legalize_sunmmio_gemm_reused_activation_uses_reaching_writer(branch_pv):
    """Reusing Q's ASRAM for P must restage probabilities for both PV stripes."""
    from tilelang.layout import make_zz_layout

    layout = make_zz_layout((32, 32), [0, 1], (32, 32))

    @T.prim_func
    def main(
        Q: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
        K: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
        O: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
    ):
        with T.Kernel() as _cid:
            q_stage = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            activation = T.alloc_shared((32, 32), "bfloat16")
            weight = T.alloc_shared((32, 32), "bfloat16")
            scores = T.alloc_shared((32, 32), "bfloat16")
            out = T.alloc_shared((32, 32), "bfloat16")
            T.copy(Q, q_stage)
            T.copy(K, weight)
            for k in T.serial(2):
                T.copy(q_stage, activation)
                T.gemm(activation, weight, scores, clear_accum=True)
                T.copy(scores, activation)
                if branch_pv:
                    if k == 0:
                        T.gemm(activation, weight, out, clear_accum=True)
                    else:
                        T.gemm(activation, weight, out)
                else:
                    T.gemm(activation, weight, out, clear_accum=True)
            T.copy(out, O)

    mod = apply_pipeline_up_to_legalize_gemm(tvm.IRModule({"main": main}))
    current_source = {}
    observed = []

    def check(node):
        if not isinstance(node, tvm.tir.Call):
            return
        if node.op.name == "tl.tileop.copy" and _copy_buf(node, 1).name == "activation":
            current_source["activation"] = _copy_buf(node, 0).name
        elif node.op.name == "tl.tileop.gemm_py":
            expected = "q_stage" if _copy_buf(node, 2).name == "scores" else "scores"
            actual = current_source.get("activation")
            assert actual == expected, f"{_copy_buf(node, 2).name} consumes {actual}; expected {expected}"
            observed.append(actual)

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, check)
    assert observed.count("q_stage") == 2
    assert observed.count("scores") == (4 if branch_pv else 2)
    tvm.ir.assert_structural_equal(mod, _run_legalize_gemm_again(mod))


@pytest.mark.parametrize("consumer_kind", ["single", "group", "branch"])
def test_legalize_sunmmio_gemm_reused_activation_keeps_each_writer_shadow(consumer_kind):
    """Two dirty reaching sources sharing ASRAM need two separately owned snapshots."""
    from tilelang.layout import make_zz_layout

    layout = make_zz_layout((32, 32), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
        B: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
        C: T.MeshTensor((32, 32), T.placement.replicated(), "float32", layout=layout),
    ):
        with T.Kernel(T.mesh_ncores()) as cid:
            first = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            second = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            activation = T.alloc_shared((32, 32), "bfloat16")
            weight = T.alloc_shared((32, 32), "bfloat16")
            out = T.alloc_shared((32, 32), "float32")
            T.copy(A, first)
            T.copy(B, second)
            T.copy(B, weight)
            T.copy(first, activation)
            T.copy(B, first)
            if consumer_kind == "branch":
                if cid == 0:
                    T.gemm(activation, weight, out, clear_accum=True)
                else:
                    T.gemm(activation, weight, out)
            else:
                T.gemm(activation, weight, out, clear_accum=True)
                if consumer_kind == "group":
                    T.gemm(activation, weight, out)
            T.copy(second, activation)
            T.copy(A, second)
            if consumer_kind == "branch":
                if cid == 0:
                    T.gemm(activation, weight, out, clear_accum=True)
                else:
                    T.gemm(activation, weight, out)
            else:
                T.gemm(activation, weight, out, clear_accum=True)
                if consumer_kind == "group":
                    T.gemm(activation, weight, out)
            T.copy(out, C)

    mod = apply_pipeline_up_to_legalize_gemm(tvm.IRModule({"main": main}))
    shadows = _collect_legalize_stage_shadows(mod)
    assert len(shadows) == 2, "each reaching writer needs its own retained source snapshot"
    assert not shadows[0][0].same_as(shadows[1][0])
    assert shadows[0][1].same_as(shadows[1][1])
    copy_calls = find_call_args(mod, "tl.tileop.copy")
    snapshots = {}
    for args, _ in copy_calls:
        source, destination = args[0].args[0].buffer, args[1].args[0].buffer
        if any(destination.same_as(shadow) for shadow, _ in shadows):
            snapshots[source.name] = destination
    assert set(snapshots) == {"first", "second"}
    restored = []
    for args, annotations in copy_calls:
        if "src_offset_byte" in annotations:
            source = args[0].args[0].buffer
            restored.append(next(name for name, shadow in snapshots.items() if source.same_as(shadow)))
    repeats = 1 if consumer_kind == "single" else 2
    assert restored == ["first"] * repeats + ["second"] * repeats
    tvm.ir.assert_structural_equal(mod, _run_legalize_gemm_again(mod))


@pytest.mark.parametrize("mutation", ["fill", "gemm", "final_gemm", "opaque", "opaque_expr"])
def test_legalize_sunmmio_gemm_snapshots_all_source_mutations(mutation):
    """Both activation stripes retain the value copied before a source mutation."""
    from tilelang.layout import make_zz_layout

    layout = make_zz_layout((32, 32), [0, 1], (32, 32))

    @T.prim_func
    def main(
        Q: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
        K: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
        O: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
    ):
        with T.Kernel():
            first = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            second = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            activation = T.alloc_shared((32, 32), "bfloat16")
            other_activation = T.alloc_shared((32, 32), "bfloat16")
            weight = T.alloc_shared((32, 32), "bfloat16")
            out = T.alloc_shared((32, 32), "bfloat16")
            T.copy(Q, first)
            T.copy(K, second)
            T.copy(K, weight)
            T.copy(first, activation)
            if mutation == "fill":
                T.clear(first)
            elif mutation == "gemm":
                T.copy(second, other_activation)
                T.gemm(other_activation, weight, first, clear_accum=True)
            elif mutation == "opaque":
                T.call_extern("int32", "may_mutate_source")
            elif mutation == "opaque_expr":
                out[0, 0] = T.call_extern("bfloat16", "may_mutate_source")
            if mutation == "final_gemm":
                T.gemm(activation, weight, first, clear_accum=True)
                T.copy(first, O)
            else:
                T.gemm(activation, weight, out, clear_accum=True)
                T.copy(out, O)

    mod = apply_pipeline_up_to_legalize_gemm(tvm.IRModule({"main": main}))
    shadows = [(s, a) for s, a in _collect_legalize_stage_shadows(mod) if a.name == "activation"]
    assert len(shadows) == 1, f"{mutation} can overwrite the copied source before restaging"
    shadow, _ = shadows[0]
    current_source = None
    target_stripes = 0

    def check(node):
        nonlocal current_source, target_stripes
        if not isinstance(node, tvm.tir.Call):
            return
        if node.op.name == "tl.tileop.copy" and _copy_buf(node, 1).name == "activation":
            current_source = _copy_buf(node, 0)
        elif node.op.name == "tl.tileop.gemm_py" and _copy_buf(node, 0).name == "activation":
            assert current_source is not None and current_source.same_as(shadow)
            target_stripes += 1

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, check)
    assert target_stripes == 2
    tvm.ir.assert_structural_equal(mod, _run_legalize_gemm_again(mod))


@pytest.mark.parametrize("writer_block", [False, True])
def test_legalize_sunmmio_gemm_shadow_keeps_writer_for_small_reader(writer_block):
    """Snapshot insertion must not remove the definition used by an unexpanded GEMM."""
    from tilelang.layout import make_zz_layout

    layout = make_zz_layout((32, 32), [0, 1], (32, 32))

    @T.prim_func
    def main(
        Q: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
        K: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
        O: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
    ):
        with T.Kernel():
            first = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            activation = T.alloc_shared((32, 32), "bfloat16")
            weight = T.alloc_shared((32, 32), "bfloat16")
            small_out = T.alloc_shared((16, 32), "bfloat16")
            out = T.alloc_shared((32, 32), "bfloat16")
            T.copy(Q, first)
            T.copy(K, weight)
            if writer_block:
                with T.block("writer_block"):
                    T.copy(first, activation)
            else:
                T.copy(first, activation)
            T.gemm(activation[0:16, :], weight, small_out, clear_accum=True)
            T.copy(K, first)
            T.gemm(activation, weight, out, clear_accum=True)
            T.copy(small_out, O[0:16, :])
            T.copy(out[16:32, :], O[16:32, :])

    mod = apply_pipeline_up_to_legalize_gemm(tvm.IRModule({"main": main}))
    current_source = None
    initialized_shadows = set()
    small_readers = 0

    def check(node):
        nonlocal current_source, small_readers
        if not isinstance(node, tvm.tir.Call):
            return
        if node.op.name == "tl.tileop.copy":
            destination = _copy_buf(node, 1).name
            if destination.endswith("_legalize_stage"):
                initialized_shadows.add(destination)
            elif destination == "activation":
                current_source = _copy_buf(node, 0).name
        elif node.op.name == "tl.tileop.gemm_py" and int(node.args[5]) == 16:
            assert current_source == "first", "snapshot insertion removed the small GEMM's activation definition"
            small_readers += 1
        elif node.op.name == "tl.tileop.gemm_py":
            assert current_source in initialized_shadows, "large GEMM reads an uninitialized source snapshot"

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, check)
    assert small_readers == 1
    assert count_call_op(mod, "tl.tileop.gemm_py") == 3
    tvm.ir.assert_structural_equal(mod, _run_legalize_gemm_again(mod))


@pytest.mark.parametrize("small_inside_loop", [False, True])
def test_legalize_sunmmio_gemm_direct_reissue_keeps_writer_for_small_reader(small_inside_loop):
    """A clean hoisted writer must still initialize a small GEMM before the restore."""
    from tilelang.layout import make_zz_layout

    layout = make_zz_layout((32, 32), [0, 1], (32, 32))

    @T.prim_func
    def main(
        Q: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
        K: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
        O: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
    ):
        with T.Kernel():
            first = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            activation = T.alloc_shared((32, 32), "bfloat16")
            weight = T.alloc_shared((32, 32), "bfloat16")
            small_out = T.alloc_shared((16, 32), "bfloat16")
            out = T.alloc_shared((32, 32), "bfloat16")
            T.copy(Q, first)
            T.copy(K, weight)
            T.copy(first, activation)
            if not small_inside_loop:
                T.gemm(activation[0:16, :], weight, small_out, clear_accum=True)
            for _ in T.serial(2):
                if small_inside_loop:
                    T.gemm(activation[0:16, :], weight, small_out, clear_accum=True)
                T.gemm(activation, weight, out, clear_accum=True)
            T.copy(small_out, O[0:16, :])
            T.copy(out[16:32, :], O[16:32, :])

    mod = apply_pipeline_up_to_legalize_gemm(tvm.IRModule({"main": main}))
    outer = _kernel_body_seq(mod)
    outer_writes = [_copy_buf(call, 1).name for _, call in _iter_calls_in_stmt(outer, "tl.tileop.copy")]
    assert "activation" in outer_writes, "the clean hoisted writer was removed before the small GEMM"
    loops = []
    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, lambda node: loops.append(node) if isinstance(node, tvm.tir.For) else None)
    assert len(loops) == 1
    seq = list(loops[0].body.seq)
    current_stripe = 0
    small_readers = 0
    activation_copy_offsets = []
    for stmt in seq:
        if not isinstance(stmt, tvm.tir.Evaluate) or not isinstance(stmt.value, tvm.tir.Call):
            continue
        call = stmt.value
        if call.op.name == "tl.tileop.copy" and _copy_buf(call, 1).name == "activation":
            current_stripe = 1 if "src_offset_byte" in call.annotations else 0
            activation_copy_offsets.append(current_stripe)
        elif call.op.name == "tl.tileop.gemm_py" and int(call.args[5]) == 16:
            assert current_stripe == 0
            small_readers += 1

    assert small_readers == int(small_inside_loop)
    assert activation_copy_offsets == ([0, 1, 0] if small_inside_loop else [0, 1])
    tvm.ir.assert_structural_equal(mod, _run_legalize_gemm_again(mod))


def test_legalize_sunmmio_gemm_hoists_restore_out_of_nested_large_gemm_loop():
    """An outer-loop reader needs one restore after the inner loop, not per inner iteration."""
    from tilelang.layout import make_zz_layout

    layout = make_zz_layout((32, 32), [0, 1], (32, 32))

    @T.prim_func
    def main(
        Q: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
        K: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
        O: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
    ):
        with T.Kernel():
            first = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            activation = T.alloc_shared((32, 32), "bfloat16")
            weight = T.alloc_shared((32, 32), "bfloat16")
            small_out = T.alloc_shared((16, 32), "bfloat16")
            out = T.alloc_shared((32, 32), "bfloat16")
            T.copy(Q, first)
            T.copy(K, weight)
            T.copy(first, activation)
            for _ in T.serial(2):
                T.gemm(activation[0:16, :], weight, small_out, clear_accum=True)
                for _ in T.serial(3):
                    T.gemm(activation, weight, out, clear_accum=True)
            T.copy(small_out, O[0:16, :])
            T.copy(out[16:32, :], O[16:32, :])

    mod = apply_pipeline_up_to_legalize_gemm(tvm.IRModule({"main": main}))
    loops = []
    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, lambda node: loops.append(node) if isinstance(node, tvm.tir.For) else None)
    assert len(loops) == 2
    inner = next(loop for loop in loops if int(loop.extent) == 3)
    outer = next(loop for loop in loops if int(loop.extent) == 2)

    inner_offsets = [
        1 if "src_offset_byte" in call.annotations else 0
        for _, call in _iter_calls_in_stmt(inner.body, "tl.tileop.copy")
        if _copy_buf(call, 1).name == "activation"
    ]
    assert inner_offsets == [0, 1]
    outer_seq = list(outer.body.seq)
    inner_index = next(i for i, stmt in enumerate(outer_seq) if isinstance(stmt, tvm.tir.For))
    restore = outer_seq[inner_index + 1]
    assert isinstance(restore, tvm.tir.Evaluate) and isinstance(restore.value, tvm.tir.Call)
    assert restore.value.op.name == "tl.tileop.copy"
    assert _copy_buf(restore.value, 1).name == "activation"
    assert "src_offset_byte" not in restore.value.annotations
    tvm.ir.assert_structural_equal(mod, _run_legalize_gemm_again(mod))


def test_legalize_sunmmio_gemm_keeps_loop_variant_restore_inside_loop():
    """A writer indexed by the inner loop cannot be cloned after that loop."""
    from tilelang.layout import make_zz_layout

    layout = make_zz_layout((32, 32), [0, 1], (32, 32))

    @T.prim_func
    def main(
        K: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
        O: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
    ):
        with T.Kernel():
            source = T.alloc_shared((96, 32), "bfloat16", scope="shared.rsram")
            activation = T.alloc_shared((32, 32), "bfloat16")
            weight = T.alloc_shared((32, 32), "bfloat16")
            small_out = T.alloc_shared((16, 32), "bfloat16")
            out = T.alloc_shared((32, 32), "bfloat16")
            T.clear(source)
            T.copy(K, weight)
            for _ in T.serial(2):
                for j in T.serial(3):
                    T.copy(source[j * 32 : (j + 1) * 32, :], activation)
                    T.gemm(activation, weight, out, clear_accum=True)
                T.gemm(activation[0:16, :], weight, small_out, clear_accum=True)
            T.copy(small_out, O[0:16, :])
            T.copy(out[16:32, :], O[16:32, :])

    mod = apply_pipeline_up_to_legalize_gemm(tvm.IRModule({"main": main}))
    loops = []
    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, lambda node: loops.append(node) if isinstance(node, tvm.tir.For) else None)
    inner = next(loop for loop in loops if int(loop.extent) == 3)
    outer = next(loop for loop in loops if int(loop.extent) == 2)
    inner_offsets = [
        1 if "src_offset_byte" in call.annotations else 0
        for _, call in _iter_calls_in_stmt(inner.body, "tl.tileop.copy")
        if _copy_buf(call, 1).name == "activation"
    ]
    assert inner_offsets == [0, 1, 0]
    outer_seq = list(outer.body.seq)
    inner_index = next(i for i, stmt in enumerate(outer_seq) if isinstance(stmt, tvm.tir.For))
    assert not _is_call(outer_seq[inner_index + 1], tvm.tir.op.Op.get("tl.tileop.copy"))
    tvm.ir.assert_structural_equal(mod, _run_legalize_gemm_again(mod))


def test_legalize_sunmmio_gemm_keeps_branch_dependent_restore_inside_loop():
    """A restore cannot leave a branch nested inside the loop it exits."""
    from tilelang.layout import make_zz_layout

    layout = make_zz_layout((32, 32), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
        B: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
        C: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
    ):
        with T.Kernel(T.mesh_ncores()) as cid:
            first = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            second = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            activation = T.alloc_shared((32, 32), "bfloat16")
            weight = T.alloc_shared((32, 32), "bfloat16")
            large_out = T.alloc_shared((32, 32), "bfloat16")
            small_out = T.alloc_shared((16, 32), "bfloat16")
            T.copy(A, first)
            T.copy(B, second)
            T.copy(B, weight)
            for _ in T.serial(3):
                if cid == 0:
                    T.copy(first, activation)
                    T.gemm(activation, weight, large_out, clear_accum=True)
                else:
                    T.copy(second, activation)
                    T.gemm(activation, weight, large_out, clear_accum=True)
            T.gemm(activation[0:16, :], weight, small_out, clear_accum=True)
            T.copy(small_out, C[0:16, :])
            T.copy(large_out[16:32, :], C[16:32, :])

    mod = apply_pipeline_up_to_legalize_gemm(tvm.IRModule({"main": main}))
    outer = list(_kernel_body_seq(mod).seq)
    loop_index = next(i for i, stmt in enumerate(outer) if isinstance(stmt, tvm.tir.For))
    unconditional_restores = [
        _copy_buf(stmt.value, 0).name
        for stmt in outer[loop_index + 1 :]
        if isinstance(stmt, tvm.tir.Evaluate)
        and isinstance(stmt.value, tvm.tir.Call)
        and stmt.value.op.name == "tl.tileop.copy"
        and _copy_buf(stmt.value, 1).name == "activation"
    ]
    assert not unconditional_restores

    branches = []
    tvm.tir.stmt_functor.post_order_visit(
        outer[loop_index], lambda node: branches.append(node) if isinstance(node, tvm.tir.IfThenElse) else None
    )
    assert len(branches) == 1
    for source, branch in zip(("first", "second"), (branches[0].then_case, branches[0].else_case)):
        activation_copies = [call for _, call in _iter_calls_in_stmt(branch, "tl.tileop.copy") if _copy_buf(call, 1).name == "activation"]
        assert _copy_buf(activation_copies[-1], 0).name == source
        assert "src_offset_byte" not in activation_copies[-1].annotations
    tvm.ir.assert_structural_equal(mod, _run_legalize_gemm_again(mod))


@pytest.mark.parametrize("group_size", [1, 2])
def test_legalize_sunmmio_gemm_restores_stripe_zero_for_later_small_reader(group_size):
    """A small GEMM after a legalized consumer must not observe stripe-1 data."""
    from tilelang.layout import make_zz_layout

    layout = make_zz_layout((32, 32), [0, 1], (32, 32))

    @T.prim_func
    def main(
        Q: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
        K: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
        O: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
    ):
        with T.Kernel():
            first = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            activation = T.alloc_shared((32, 32), "bfloat16")
            weight = T.alloc_shared((32, 32), "bfloat16")
            small_out = T.alloc_shared((16, 32), "bfloat16")
            out = T.alloc_shared((32, 32), "bfloat16")
            other_out = T.alloc_shared((32, 32), "bfloat16")
            T.copy(Q, first)
            T.copy(K, weight)
            T.copy(first, activation)
            T.gemm(activation, weight, out, clear_accum=True)
            if group_size == 2:
                T.gemm(activation, weight, other_out, clear_accum=True)
            T.gemm(activation[0:16, :], weight, small_out, clear_accum=True)
            T.copy(small_out, O[0:16, :])
            T.copy(out[16:32, :], O[16:32, :])

    mod = apply_pipeline_up_to_legalize_gemm(tvm.IRModule({"main": main}))
    seq = list(_kernel_body_seq(mod).seq)
    current_stripe = None
    small_readers = 0
    for stmt in seq:
        if not isinstance(stmt, tvm.tir.Evaluate) or not isinstance(stmt.value, tvm.tir.Call):
            continue
        call = stmt.value
        if call.op.name == "tl.tileop.copy" and _copy_buf(call, 1).name == "activation":
            current_stripe = 1 if "src_offset_byte" in call.annotations else 0
        elif call.op.name == "tl.tileop.gemm_py" and int(call.args[5]) == 16:
            assert current_stripe == 0, "the small GEMM observed the legalized consumer's stripe-1 staging"
            small_readers += 1

    assert small_readers == 1
    tvm.ir.assert_structural_equal(mod, _run_legalize_gemm_again(mod))


def test_legalize_sunmmio_gemm_restores_after_divergent_writers_for_joined_small_reader():
    """A post-join small GEMM must get stripe-0 from either branch writer."""
    from tilelang.layout import make_zz_layout

    layout = make_zz_layout((32, 32), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
        B: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
        C: T.MeshTensor((32, 32), T.placement.replicated(), "bfloat16", layout=layout),
    ):
        with T.Kernel(T.mesh_ncores()) as cid:
            first = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            second = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            activation = T.alloc_shared((32, 32), "bfloat16")
            weight = T.alloc_shared((32, 32), "bfloat16")
            large_out = T.alloc_shared((32, 32), "bfloat16")
            small_out = T.alloc_shared((16, 32), "bfloat16")
            T.copy(A, first)
            T.copy(B, second)
            T.copy(B, weight)
            if cid == 0:
                T.copy(first, activation)
                T.gemm(activation, weight, large_out, clear_accum=True)
            else:
                T.copy(second, activation)
                T.gemm(activation, weight, large_out, clear_accum=True)
            T.gemm(activation[0:16, :], weight, small_out, clear_accum=True)
            T.copy(small_out, C[0:16, :])
            T.copy(large_out[16:32, :], C[16:32, :])

    mod = apply_pipeline_up_to_legalize_gemm(tvm.IRModule({"main": main}))
    branches = []
    tvm.tir.stmt_functor.post_order_visit(
        mod["main"].body, lambda node: branches.append(node) if isinstance(node, tvm.tir.IfThenElse) else None
    )
    assert len(branches) == 1
    for branch in (branches[0].then_case, branches[0].else_case):
        seq = list(branch.seq)
        last = seq[-1]
        assert isinstance(last, tvm.tir.Evaluate) and isinstance(last.value, tvm.tir.Call)
        assert last.value.op.name == "tl.tileop.copy"
        assert _copy_buf(last.value, 1).name == "activation"
        assert "src_offset_byte" not in last.value.annotations
    tvm.ir.assert_structural_equal(mod, _run_legalize_gemm_again(mod))


def test_legalize_sunmmio_gemm_rejects_hoisted_collective_in_one_sided_branch():
    """The pass must not move a mesh collective under a core-dependent branch."""
    from tilelang.layout import make_zz_layout

    layout = make_zz_layout((128, 128), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((128, 128), T.placement.full_shard(0, 1), "bfloat16", layout=layout),
        B: T.MeshTensor((128, 128), T.placement.full_shard(0, 1), "bfloat16", layout=layout),
        C: T.MeshTensor((128, 128), T.placement.full_shard(0, 1), "float32", layout=layout),
    ):
        with T.Kernel(T.mesh_ncores()) as cid:
            send = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            activation = T.alloc_shared((32, 32 * T.mesh_ncols()), "bfloat16")
            weight_send = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            weight = T.alloc_shared((32 * T.mesh_nrows(), 32), "bfloat16")
            out = T.alloc_shared((32, 32), "float32")
            T.copy(A[0:32, 0:32], send)
            T.comm.all_gather(send, activation, direction="h", axis=-1)
            T.copy(B[0:32, 0:32], weight_send)
            T.comm.all_gather(weight_send, weight, direction="v", axis=0)
            if cid == 0:
                T.gemm(activation, weight, out, clear_accum=True)
            T.copy(out, C[0:32, 0:32])

    with pytest.raises(Exception, match="cannot replay collective writer inside a conditional GEMM scope"):
        apply_pipeline_up_to_legalize_gemm(tvm.IRModule({"main": main}))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
