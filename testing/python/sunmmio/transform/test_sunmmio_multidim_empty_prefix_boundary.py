"""First-K initialization proof for an empty WorkItem prefix.

These are analysis tests, not performance or backend acceptance tests.  The
fixture uses the lowered BF16 sequence (A/B broadcasts, MMA stripe 0, A restage,
MMA stripe 1) and real canonical layouts.  No kernel-name dispatch is involved.
"""

import pytest
import tilelang as tl
from tilelang import tvm
from tilelang.layout import make_nz_layout, make_row_major, make_zz_layout
from tvm import tir


ELIGIBLE = "tl.sunmmio.boundary_overlap.eligible"
REASON = "tl.sunmmio.boundary_overlap.reject_reason"
RESIDENCY_REASON = "tl.sunmmio.residency.reject_reason"


def _buffer(name, shape, scope, dtype="bfloat16", **kwargs):
    data = tir.Var(name, tvm.ir.PointerType(tvm.ir.PrimType(dtype), scope))
    return tir.decl_buffer(shape, dtype, name=name, data=data, scope=scope, **kwargs)


def _region(buffer, access, indices=None, extents=None, predicate=None):
    indices = [0] * len(buffer.shape) if indices is None else indices
    extents = list(buffer.shape) if extents is None else extents
    return tir.Call("handle", tvm.ir.Op.get("tl.tileop.region"),
                    [tir.BufferLoad(buffer, indices, predicate=predicate),
                     tir.IntImm("int32", access), *extents])


def _op(name, *args):
    return tir.Evaluate(tir.Call("handle", tvm.ir.Op.get(name), list(args)))


def _fixture(fault=None, *, rows=32, cols=64, reduction=64, kmin=0,
             accumulators=1, partitioned=False, tag=True):
    n, k, bx = (tir.Var(name, "int32") for name in ("output_tile", "reduction_tile", "bx"))
    ksteps = 2
    shard = reduction // 4 if partitioned else reduction
    source = _buffer("input", (rows, ksteps * shard), "global")
    weights = _buffer("weights", (ksteps * shard, 3 * cols), "global")
    output = _buffer("output", (3, rows, cols), "global")
    cache = _buffer("read_only_cache", source.shape, "shared.rsram")
    a = _buffer("a", (rows, reduction), "shared.asram")
    b = _buffer("b", (reduction, cols), "shared.wsram")
    accumulators = 2 if fault == "other_state_uninitialized" else accumulators
    accs = [_buffer("acc" + str(i), (rows, cols), "shared.rsram", "float32")
            for i in range(accumulators)]
    out = _buffer("out", (rows, cols), "shared.rsram")
    spare = _buffer("spare", (rows, cols), "shared.rsram", "float32")
    if fault == "loaded_metadata":
        accs[0] = _buffer("acc_loaded_offset", (rows, cols), "shared.rsram", "float32",
                          elem_offset=tir.Cast("int32", spare[0, 0]))
    if fault == "nonzero_elem_offset":
        accs[0] = _buffer("acc_offset", (rows, cols), "shared.rsram", "float32", elem_offset=16)
    if fault == "strided_accumulator":
        accs[0] = _buffer("acc_stride", (rows, cols), "shared.rsram", "float32", strides=[cols, 1])

    def broadcast_a(offset):
        return _op("tl.broadcast_",
                   _region(cache, 1, [0, (k - kmin) * shard], [rows, shard]),
                   _region(a, 2, [0, bx % 4 * shard if partitioned else 0], [rows, shard]),
                   0, tir.IntImm("int64", 7 if fault == "partial_collective" else 15), offset)

    bcast_b = _op("tl.broadcast_",
                 _region(weights, 1, [(k - kmin) * shard, n * cols], [shard, cols]),
                 _region(b, 2, [bx // 4 * shard if partitioned else 0, 0], [shard, cols]),
                 1, tir.IntImm("int64", 15), 0)

    def mma(acc, offset):
        clear = k == kmin
        if fault == "clear_zero" or (fault == "second_stripe_not_clear" and offset):
            clear = tir.const(False, "bool")
        if fault == "clear_wrong_min":
            clear = k == kmin + 1
        if fault == "clear_always":
            clear = tir.const(True, "bool")
        if fault == "other_state_uninitialized" and acc.same_as(accs[1]):
            clear = tir.const(False, "bool")
        if fault == "same_storage_view" and offset:
            acc = tir.decl_buffer(acc.shape, acc.dtype, name="alias", data=acc.data, scope=acc.scope())
        c = _region(acc, 3, extents=[rows // 2, cols] if fault == "partial_accumulator" else None,
                    predicate=spare[0, 0] > 0 if fault == "region_predicate" else None)
        args = [_region(a, 1), _region(b, 1), c, tir.const(False, "bool"),
                tir.const(False, "bool"), clear,
                0 if fault == "duplicate_stripe" else (offset + 64 if fault == "wrong_offset" else offset)]
        if fault == "separate_output":
            args.extend([_region(spare, 2), k == kmin + ksteps - 1])
        result = _op("tl.mma_sunmmio", *args)
        if fault == "conditional_clear":
            result = tir.IfThenElse(k == kmin, result, None)
        return result

    core_stmts = [broadcast_a(0), bcast_b]
    for acc in accs:
        core_stmts.append(mma(acc, 0))
    core_stmts.append(broadcast_a(1024))
    if fault != "missing_stripe":
        for acc in accs:
            core_stmts.append(mma(acc, 2048))
    read_acc = _op("tl.dma_copy", _region(accs[0], 1), _region(spare, 2), 0)
    if fault == "read_before_first_stripe":
        core_stmts.insert(0, read_acc)
    if fault == "read_between_stripes":
        core_stmts.insert(3, read_acc)
    if fault == "read_after_full_init":
        core_stmts.append(read_acc)
    if fault == "duplicate_full_pair":
        core_stmts.extend([mma(accs[0], 0), mma(accs[0], 2048)])
    core = tir.For(k, kmin, ksteps, tir.ForKind.SERIAL, tir.SeqStmt(core_stmts),
                   annotations={"num_stages": 2})
    row, col = tir.Var("row", "int32"), tir.Var("col", "int32")
    value = tir.Cast("bfloat16", accs[0][row, col])
    for acc in accs[1:]:
        value = value + tir.Cast("bfloat16", acc[row, col])
    if fault == "uninitialized_suffix_state":
        value = value + tir.Cast("bfloat16", spare[row, col])
    if fault == "cross_workitem_read":
        value = value + output[(n + 1) % 3, row, col]
    load_predicate = spare[0, 0] > 0 if fault == "suffix_load_predicate" else None
    if load_predicate is not None:
        value = tir.Cast("bfloat16", tir.BufferLoad(accs[0], [row, col], predicate=load_predicate))
    cast_store = tir.BufferStore(out, value, [row, col],
                                predicate=spare[0, 0] > 0 if fault == "suffix_store_predicate" else None)
    cast = tir.For(row, 0, rows, tir.ForKind.SERIAL,
                   tir.For(col, 0, cols, tir.ForKind.VECTORIZED, cast_store))
    store = _op("tl.dma_copy", _region(out, 1),
                _region(output, 2, [n % 2 if fault == "output_overlap" else n, 0, 0], [1, rows, cols]), 0)
    annotations = {RESIDENCY_REASON: "no_invariant_transfer_prefix"} if tag else {}
    outer = tir.For(n, 0, 3, tir.ForKind.SERIAL, tir.SeqStmt([core, cast, store]),
                    annotations=annotations)
    preload = _op("tl.dma_copy", _region(source, 1), _region(cache, 2), 0)
    body = tir.SeqStmt([preload, outer])
    if fault == "raw_escape":
        body = tir.SeqStmt([body, tir.Evaluate(tir.call_extern("int32", "consume", accs[0].data))])
    if fault == "handle_alias":
        alias = tir.Var("alias", "handle")
        body = tir.LetStmt(alias, accs[0].data, body)
    if fault == "match_buffer_alias":
        view = _buffer("matched", (rows, cols), "shared.rsram", "float32")
        match = tir.MatchBufferRegion(view, tir.BufferRegion(accs[0], [
            tvm.ir.Range.from_min_extent(0, rows), tvm.ir.Range.from_min_extent(0, cols)]))
        body = tir.BlockRealize([], tir.const(True, "bool"),
            tir.Block([], [], [], "alias_block", body, match_buffers=[match]))
    if partitioned:
        thread = tir.IterVar(None, bx, tir.IterVar.ThreadIndex, "blockIdx.x")
        body = tir.AttrStmt(thread, "thread_extent", 16, body)
    layouts = {a: make_zz_layout(a.shape), b: make_nz_layout(b.shape, [0, 1], [32, 32]),
               **{acc: make_zz_layout(acc.shape) for acc in accs}}
    if fault == "noncanonical_layout":
        layouts[accs[0]] = make_row_major(accs[0].shape)
    if fault == "wrong_block_layout":
        layouts[accs[0]] = make_zz_layout(accs[0].shape, block_shape=[16, 32])
    block_layouts = dict(layouts)
    if fault == "conflicting_block_layout":
        block_layouts[accs[0]] = make_row_major(accs[0].shape)
    root = tir.Block([], [], [], "root", body, alloc_buffers=[cache, a, b, *accs, out, spare],
                     annotations={"layout_map": block_layouts})
    func = tir.PrimFunc([source.data, weights.data, output.data],
                       tir.BlockRealize([], tir.const(True, "bool"), root),
                       buffer_map={source.data: source, weights.data: weights, output.data: output})
    func = func.with_attr("global_symbol", "main").with_attr("tir.noalias", fault != "missing_noalias")
    if fault != "missing_layout":
        func = func.with_attr("layout_map", layouts)
    if fault != "missing_target":
        target = "llvm" if fault == "non_sunmmio_target" else (
            "llvm -mcpu=sunmmio-unknown" if fault == "unknown_target" else
            "llvm -mcpu=sunmmio-a4e -mattr=device_mesh_nrow_4,device_mesh_ncol_4")
        func = func.with_attr("target", tvm.target.Target(target))
    return tvm.IRModule.from_expr(func)


def _outer(mod):
    loops = []
    tir.stmt_functor.post_order_visit(mod["main"].body,
        lambda node: loops.append(node) if isinstance(node, tir.For) and
        str(node.loop_var.name) == "output_tile" else None)
    assert len(loops) == 1
    return loops[0]


def _analyze(mod, policy="auto"):
    return tvm.get_global_func("tl.transform.AnalyzeSunmmioBoundaryOverlap")(policy)(mod)


@pytest.fixture(autouse=True)
def _environment(monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_MULTIDIM_BOUNDARY_OVERLAP", "1")
    monkeypatch.delenv("TL_SUNMMIO_RESIDENCY_REPORT", raising=False)


@pytest.mark.parametrize("rows, cols, reduction, kmin, accumulators, partitioned", [
    (32, 32, 32, 0, 1, False),
    (32, 512, 1024, 0, 1, True),  # real O lowering geometry / collective shards
    (64, 96, 64, 3, 1, False),   # multiple physical blocks, nonzero K minimum
    (32, 64, 64, 7, 2, False),   # every independent accumulator must initialize
])
def test_full_first_k_stripes_prove_empty_prefix(rows, cols, reduction, kmin, accumulators, partitioned):
    before = _fixture(rows=rows, cols=cols, reduction=reduction, kmin=kmin,
                      accumulators=accumulators, partitioned=partitioned)
    result = _analyze(before)
    loop = _outer(result)
    assert bool(loop.annotations[ELIGIBLE])
    assert str(loop.annotations[RESIDENCY_REASON]) == "no_invariant_transfer_prefix"
    assert tvm.ir.structural_equal(_outer(before).body, loop.body)
    assert tvm.ir.structural_equal(before["main"].attrs, result["main"].attrs)
    assert all(a.same_as(b) for a, b in zip(before["main"].body.block.alloc_buffers,
                                          result["main"].body.block.alloc_buffers))
    assert len(loop.annotations["tl.sunmmio.boundary_overlap.state_buffers"]) == accumulators
    assert tvm.tir.analysis.verify_ssa(result["main"])


@pytest.mark.parametrize("fault, reason", [
    ("read_before_first_stripe", "empty_prefix_state_read_before_init"),
    ("read_between_stripes", "empty_prefix_state_read_before_init"),
    ("missing_stripe", "empty_prefix_incomplete_stripes"),
    ("duplicate_stripe", "empty_prefix_stripe_order_or_offset"),
    ("wrong_offset", "empty_prefix_stripe_order_or_offset"),
    ("duplicate_full_pair", "empty_prefix_stripe_order_or_offset"),
    ("clear_zero", "empty_prefix_first_k_clear_required"),
    ("second_stripe_not_clear", "empty_prefix_first_k_clear_required"),
    ("other_state_uninitialized", "empty_prefix_first_k_clear_required"),
    ("clear_wrong_min", "reduction_clear_order_unproven"),
    ("clear_always", "reduction_clear_order_unproven"),
    ("conditional_clear", "core_control_or_side_effect"),
    ("partial_accumulator", "empty_prefix_full_mma_required"),
    ("nonzero_elem_offset", "empty_prefix_full_mma_required"),
    ("strided_accumulator", "empty_prefix_full_mma_required"),
    ("separate_output", "empty_prefix_mma_form_unproven"),
    ("missing_layout", "empty_prefix_layout_unproven"),
    ("noncanonical_layout", "empty_prefix_layout_unproven"),
    ("wrong_block_layout", "empty_prefix_layout_unproven"),
    ("conflicting_block_layout", "empty_prefix_layout_unproven"),
    ("missing_target", "empty_prefix_target_required"),
    ("non_sunmmio_target", "empty_prefix_target_unproven"),
    ("unknown_target", "empty_prefix_target_unproven"),
    ("loaded_metadata", "empty_prefix_hidden_access_or_alias"),
    ("region_predicate", "empty_prefix_hidden_access_or_alias"),
    ("suffix_load_predicate", "empty_prefix_hidden_access_or_alias"),
    ("suffix_store_predicate", "boundary_control_or_side_effect"),
    ("handle_alias", "empty_prefix_hidden_access_or_alias"),
    ("match_buffer_alias", "empty_prefix_hidden_access_or_alias"),
    ("same_storage_view", "ambiguous_buffer_alias"),
    ("raw_escape", "written_state_escapes_region"),
    ("uninitialized_suffix_state", "suffix_reads_carried_or_ephemeral_state"),
    ("partial_collective", "collective_coverage_unproven"),
    ("missing_noalias", "global_alias_unproven"),
    ("cross_workitem_read", "suffix_global_read_write_overlap_across_workitems"),
    ("output_overlap", "global_output_overlap_across_workitems"),
])
def test_empty_prefix_rejects_unsafe_or_unmodeled_structure(fault, reason):
    loop = _outer(_analyze(_fixture(fault, partitioned=fault == "partial_collective")))
    assert ELIGIBLE not in loop.annotations
    assert str(loop.annotations[REASON]) == reason
    assert str(loop.annotations[RESIDENCY_REASON]) == "no_invariant_transfer_prefix"


@pytest.mark.parametrize("rows, cols", [(16, 32), (48, 64), (32, 48)])
def test_partial_physical_blocks_remain_explicitly_unsupported(rows, cols):
    loop = _outer(_analyze(_fixture(rows=rows, cols=cols)))
    assert ELIGIBLE not in loop.annotations
    assert str(loop.annotations[REASON]) == "empty_prefix_partial_physical_tile"


def test_state_read_after_complete_initialization_is_not_a_pre_read():
    assert bool(_outer(_analyze(_fixture("read_after_full_init"))).annotations[ELIGIBLE])


@pytest.mark.parametrize("policy", ["off", "shadow"])
def test_nonmutating_policies_preserve_empty_prefix(policy):
    before = _fixture()
    assert tvm.ir.structural_equal(before, _analyze(before, policy))


def test_forced_mode_keeps_initialization_gate():
    with pytest.raises(tvm.TVMError, match="empty_prefix_incomplete_stripes"):
        _analyze(_fixture("missing_stripe"), "forced")


def test_real_residency_routing_does_not_forge_residency_success():
    rejected = tl.transform.AnalyzeSunmmioTileResidency("auto_strict")(_fixture(tag=False))
    assert str(_outer(rejected).annotations[RESIDENCY_REASON]) == "no_invariant_transfer_prefix"
    proved = _analyze(rejected)
    assert bool(_outer(proved).annotations[ELIGIBLE])
    assert "tl.sunmmio.residency.generic_eligible" not in _outer(proved).annotations
