"""Independent original-order boundary-overlap proof, not performance tests.

The positive fixture models a read-only cache and BF16's repeated A writes.
It intentionally fails tile residency.  Passing this proof only authorizes
the existing region planner to try its own dependence/epoch/coverage checks.
"""

import pytest
import tilelang as tl
from tilelang import tvm
from tvm import tir


ELIGIBLE = "tl.sunmmio.boundary_overlap.eligible"
REASON = "tl.sunmmio.boundary_overlap.reject_reason"
RESIDENCY_REASON = "tl.sunmmio.residency.reject_reason"


def _buffer(name, shape, scope, dtype="bfloat16"):
    data = tir.Var(name, tvm.ir.PointerType(tvm.ir.PrimType(dtype), scope))
    return tir.decl_buffer(shape, dtype, name=name, data=data, scope=scope)


def _region(buffer, access, indices, extents):
    return tir.Call("handle", tvm.ir.Op.get("tl.tileop.region"), [tir.BufferLoad(buffer, indices), tir.IntImm("int32", access), *extents])


def _op(name, *args):
    return tir.Evaluate(tir.Call("handle", tvm.ir.Op.get(name), list(args)))


def _tiled_store(buffer, value, *, rows=32, step=1):
    row = tir.Var(buffer.name + "_row", "int32")
    tile = tir.Var(buffer.name + "_tile", "int32")
    lane = tir.Var(buffer.name + "_lane", "int32")
    expr = value(row, tile * 16 + lane)
    return tir.For(
        row,
        0,
        rows,
        tir.ForKind.SERIAL,
        tir.For(
            tile,
            0,
            2,
            tir.ForKind.SERIAL,
            tir.For(lane, 0, 16, tir.ForKind.VECTORIZED, tir.BufferStore(buffer, expr, [row, tile * 16 + lane])),
        ),
        step=step,
    )


def make_boundary_module(fault=None, *, tag=True):
    n = tir.Var("output_tile", "int32")
    k = tir.Var("reduction_tile", "int32")
    source = _buffer("input", (32, 32), "global")
    weights = _buffer("weights", (3, 2, 32, 32), "global")
    output = _buffer("output", (3, 32, 32), "global")
    cache = _buffer("read_only_cache", (32, 32), "shared.rsram")
    a = _buffer("a", (32, 32), "shared.asram")
    b = _buffer("b", (32, 32), "shared.wsram")
    acc = _buffer("acc", (32, 32), "shared.rsram", "float32")
    out = _buffer("out", (32, 32), "shared.rsram")
    a_second = a
    if fault == "same_storage_alias":
        a_second = tir.decl_buffer(a.shape, a.dtype, name="alias", data=a.data, scope=a.scope())

    def broadcast_a(mode, dest):
        return _op(
            "tl.broadcast_", _region(cache, 1, [0, 0], [32, 32]), _region(dest, 2, [0, 0], [32, 32]), 0, tir.IntImm("int64", 15), mode
        )

    def mma(mode):
        return _op(
            "tl.mma_sunmmio",
            _region(a, 1, [0, 0], [32, 32]),
            _region(b, 1, [0, 0], [32, 32]),
            _region(acc, 3, [0, 0], [32, 32]),
            tir.IntImm("bool", 0),
            tir.IntImm("bool", 0),
            tir.IntImm("bool", 0),
            mode,
        )

    core_stmts = [
        broadcast_a(0, a),
        _op(
            "tl.broadcast_",
            _region(weights, 1, [n, k, 0, 0], [1, 1, 32, 32]),
            _region(b, 2, [0, 0], [32, 32]),
            1,
            tir.IntImm("int64", 15),
            0,
        ),
        mma(0),
        broadcast_a(1024, a_second),
        mma(2048),
    ]
    if fault == "unknown_core_effect":
        core_stmts.insert(0, tir.Evaluate(tir.call_extern("int32", "unknown_effect")))
    if fault == "core_global_write":
        core_stmts.append(_op("tl.dma_copy", _region(a, 1, [0, 0], [32, 32]), _region(source, 2, [0, 0], [32, 32]), 0))
    if fault == "scratch_read_before_write":
        core_stmts.insert(0, mma(0))
    if fault == "core_if_else":
        core_stmts[0] = tir.IfThenElse(n == 0, core_stmts[0], core_stmts[0])
    core = tir.For(
        k, 0, 2, tir.ForKind.SERIAL, tir.SeqStmt(core_stmts), annotations={"num_stages": 2}, step=2 if fault == "core_step" else 1
    )
    clear = _tiled_store(
        acc,
        lambda row, col: tir.FloatImm("float32", 0),
        rows=16 if fault == "partial_clear" else 32,
        step=2 if fault == "nonunit_clear" else 1,
    )
    prefix = [clear]
    if fault == "prefix_global_write":
        prefix.append(tir.BufferStore(source, tir.const(0, "bfloat16"), [0, 0]))

    def cast_value(row, col):
        value = tir.Cast("bfloat16", acc[row, col])
        if fault == "suffix_reads_other_output":
            value = value + output[(n + 1) % 3, row, col]
        if fault == "suffix_nonadjacent_read_conflict":
            value = value + output[tir.Select(n == 2, 0, n), row, col]
        if fault == "suffix_reads_own_output":
            value = value + output[n, row, col]
        if fault == "suffix_reads_independent_input":
            value = value + source[row, col]
        return value

    cast = _tiled_store(out, cast_value, rows=16 if fault == "partial_output" else 32)
    destination_n = n % 2 if fault == "third_output_revisits_first" else n
    suffix = [cast, _op("tl.dma_copy", _region(out, 1, [0, 0], [32, 32]), _region(output, 2, [destination_n, 0, 0], [1, 32, 32]), 0)]
    if fault == "unknown_suffix_effect":
        suffix.append(tir.Evaluate(tir.call_extern("int32", "unknown_effect")))
    outer = tir.For(
        n,
        0,
        3,
        tir.ForKind.SERIAL,
        tir.SeqStmt([*prefix, core, *suffix]),
        annotations={RESIDENCY_REASON: "no_invariant_transfer_prefix"} if tag else {},
        step=2 if fault == "outer_step" else 1,
    )
    preload = _op("tl.dma_copy", _region(source, 1, [0, 0], [32, 32]), _region(cache, 2, [0, 0], [32, 32]), 0)
    body = [preload, outer]
    if fault == "state_escape":
        body.append(tir.Evaluate(acc[0, 0]))
    if fault == "raw_pointer_escape":
        body.append(tir.Evaluate(tir.call_extern("int32", "consume_pointer", acc.data)))
    if fault == "pointer_escape_before_region":
        body.insert(1, tir.Evaluate(tir.call_extern("int32", "save_pointer", acc.data)))
    if fault == "let_pointer_alias_escape":
        alias_pointer = tir.Var("saved_acc_pointer", "handle")
        body.append(tir.LetStmt(alias_pointer, acc.data, tir.Evaluate(tir.call_extern("int32", "consume_pointer", alias_pointer))))
    if fault == "same_data_view_escape":
        view = tir.decl_buffer(acc.shape, acc.dtype, name="acc_view", data=acc.data, scope=acc.scope())
        body.append(tir.Evaluate(view[0, 0]))
    if fault == "match_buffer_view_escape":
        view = _buffer("matched_acc", (32, 32), "shared.rsram", "float32")
        source_region = tir.BufferRegion(acc, [tvm.ir.Range.from_min_extent(0, 32), tvm.ir.Range.from_min_extent(0, 32)])
        match = tir.MatchBufferRegion(view, source_region)
        alias_block = tir.Block([], [], [], "read_matched_acc", tir.Evaluate(view[0, 0]), match_buffers=[match])
        body.append(tir.BlockRealize([], tir.const(True, "bool"), alias_block))
    root = tir.Block([], [], [], "root", tir.SeqStmt(body), alloc_buffers=[cache, a, b, acc, out])
    func = tir.PrimFunc(
        [source.data, weights.data, output.data],
        tir.BlockRealize([], tir.const(True, "bool"), root),
        buffer_map={source.data: source, weights.data: weights, output.data: output},
    )
    func = func.with_attr("global_symbol", "main")
    if fault != "missing_noalias":
        func = func.with_attr("tir.noalias", True)
    return tvm.IRModule.from_expr(func)


def _outer(mod):
    loops = []
    tir.stmt_functor.post_order_visit(
        mod["main"].body,
        lambda node: loops.append(node) if isinstance(node, tir.For) and str(node.loop_var.name) == "output_tile" else None,
    )
    assert len(loops) == 1
    return loops[0]


def _analyze(mod, policy="auto"):
    return tvm.get_global_func("tl.transform.AnalyzeSunmmioBoundaryOverlap")(policy)(mod)


@pytest.fixture(autouse=True)
def _environment(monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_MULTIDIM_BOUNDARY_OVERLAP", "1")
    monkeypatch.delenv("TL_SUNMMIO_RESIDENCY_REPORT", raising=False)


def test_original_residency_rejects_repeated_a_destination():
    # Runnable against the frozen pre-change library: establishes the baseline.
    result = tl.transform.AnalyzeSunmmioTileResidency("auto_strict")(make_boundary_module(tag=False))
    assert str(_outer(result).annotations[RESIDENCY_REASON]) == "no_invariant_transfer_prefix"


def test_independent_boundary_proof_keeps_residency_rejection_and_cache():
    result = _analyze(make_boundary_module())
    loop = _outer(result)
    assert bool(loop.annotations[ELIGIBLE])
    assert int(loop.annotations["tl.sunmmio.boundary_overlap.proof_version"]) == 1
    assert str(loop.annotations[RESIDENCY_REASON]) == "no_invariant_transfer_prefix"
    assert [str(b.name) for b in loop.annotations["tl.sunmmio.boundary_overlap.read_only_buffers"]] == ["read_only_cache"]
    assert [str(b.name) for b in loop.annotations["tl.sunmmio.boundary_overlap.state_buffers"]] == ["acc"]


@pytest.mark.parametrize(
    "fault, reason",
    [
        ("missing_noalias", "global_alias_unproven"),
        ("same_storage_alias", "ambiguous_buffer_alias"),
        ("partial_clear", "accumulator_not_fully_initialized"),
        ("nonunit_clear", "boundary_control_or_side_effect"),
        ("partial_output", "suffix_state_not_fully_defined"),
        ("unknown_core_effect", "core_control_or_side_effect"),
        ("core_if_else", "core_control_or_side_effect"),
        ("core_global_write", "core_global_write"),
        ("prefix_global_write", "prefix_writes_non_state"),
        ("unknown_suffix_effect", "boundary_control_or_side_effect"),
        ("scratch_read_before_write", "scratch_read_before_full_write"),
        ("third_output_revisits_first", "global_output_overlap_across_workitems"),
        ("state_escape", "written_state_escapes_region"),
        ("raw_pointer_escape", "written_state_escapes_region"),
        ("pointer_escape_before_region", "written_state_escapes_region"),
        ("let_pointer_alias_escape", "written_state_escapes_region"),
        ("same_data_view_escape", "written_state_escapes_region"),
        ("match_buffer_view_escape", "match_buffer_alias_unmodeled"),
        ("suffix_reads_other_output", "suffix_global_read_write_overlap_across_workitems"),
        ("suffix_nonadjacent_read_conflict", "suffix_global_read_write_overlap_across_workitems"),
        ("outer_step", "unit_serial_outer_required"),
        ("core_step", "unit_serial_core_required"),
    ],
)
def test_independent_proof_rejects_unsafe_candidate(fault, reason):
    loop = _outer(_analyze(make_boundary_module(fault)))
    assert ELIGIBLE not in loop.annotations
    assert str(loop.annotations[REASON]) == reason
    assert str(loop.annotations[RESIDENCY_REASON]) == "no_invariant_transfer_prefix"


@pytest.mark.parametrize("fault", ["suffix_reads_own_output", "suffix_reads_independent_input"])
def test_proved_noninterfering_global_epilogue_reads_remain_eligible(fault):
    loop = _outer(_analyze(make_boundary_module(fault)))
    assert bool(loop.annotations[ELIGIBLE])


@pytest.mark.parametrize("policy", ["off", "shadow"])
def test_nonmutating_policies_preserve_ir(policy):
    before = make_boundary_module()
    assert tvm.ir.structural_equal(before, _analyze(before, policy))


def test_disabled_experiment_preserves_ir(monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_MULTIDIM_BOUNDARY_OVERLAP", "0")
    before = make_boundary_module()
    assert tvm.ir.structural_equal(before, _analyze(before))


def test_forced_mode_does_not_bypass_independent_proof():
    with pytest.raises(tvm.TVMError, match="global_output_overlap_across_workitems"):
        _analyze(make_boundary_module("third_output_revisits_first"), "forced")


def test_normalizer_does_not_turn_boundary_proof_into_interchange(monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_MULTIDIM_INTERCHANGE", "1")
    proved = _analyze(make_boundary_module())
    result = tl.transform.SunmmioMultidimRegion("auto")(proved)
    loop = _outer(result)
    assert ELIGIBLE in loop.annotations
    assert "tl.sunmmio.multidim.strategy" not in loop.annotations
    assert str(loop.annotations[RESIDENCY_REASON]) == "no_invariant_transfer_prefix"
    assert sum(isinstance(stmt, tir.For) and "num_stages" in stmt.annotations for stmt in loop.body.seq) == 1
