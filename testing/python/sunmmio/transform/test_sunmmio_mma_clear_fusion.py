"""TIR-only coverage for the opt-in clear fusion in LegalizeSunmmioGemm.

These tests run a transform, never codegen, a kernel, or a simulator.  The
BF16 fixtures include the real RSRAM -> ASRAM reaching copy required by the
existing two-stripe legalizer.  Safety regressions retain BF16 inputs, so the
dtype eligibility guard cannot mask the safety property being tested.
Passing these tests is not numerical or backend validation.
"""

import json

import pytest
import tilelang as tl
from tilelang import tvm
from tilelang.utils.target import SUNMMIO_TARGET_DESC


tir = tvm.tir
OPTION = "tl.sunmmio_mma_clear_fusion"
FILL = "tl.tileop.fill"
COPY = "tl.tileop.copy"
REGION = "tl.tileop.region"
GEMMS = ("tl.tileop.gemm", "tl.tileop.gemm_py")


def _call(op, args, annotations=None):
    return tir.Call("handle", tvm.ir.Op.get(op), args, annotations=annotations)


def _region(buffer, *, extents=None, starts=None, access=3, predicate=None):
    starts = [0] * len(buffer.shape) if starts is None else starts
    extents = list(buffer.shape) if extents is None else extents
    load = tir.BufferLoad(buffer, starts, predicate=predicate)
    return _call(REGION, [load, access, *extents])


def _copy(source, destination):
    return tir.Evaluate(_call(COPY, [
        _region(source, access=1), _region(destination, access=2),
    ]))


def _nodes(statement, node_type):
    result = []

    def visit(node):
        if isinstance(node, node_type):
            result.append(node)
        return True

    tir.stmt_functor.pre_order_visit(statement, visit)
    return result


def _calls(statement, names):
    if isinstance(names, str):
        names = (names,)
    return [node for node in _nodes(statement, tir.Call)
            if isinstance(node.op, tvm.ir.Op) and node.op.name in names]


def _fixture(case="valid", *, dtype="bfloat16", weight_dtype=None, op=GEMMS[0], minimum=0,
             extent=3, explicit_offset=True, ramp_fill=False):
    """Build post-layout IR, with one locally owned full FP32 accumulator."""
    n = tir.Var("n", "int32")
    a_shape = (2, 32, 32) if case == "rank3_input" else (32, 32)
    c_shape = (2, 32, 32) if case == "rank3_accumulator" else (32, 32)
    if case == "dynamic_shape":
        c_shape = (n, 32)
    source = tir.decl_buffer(a_shape, dtype, name="A_source", scope="shared.rsram")
    weight_dtype = dtype if weight_dtype is None else weight_dtype
    weight = tir.decl_buffer((32, 32), weight_dtype, name="B_source", scope="shared.rsram")
    a = tir.decl_buffer(a_shape, dtype, name="A", scope="shared.asram")
    b = tir.decl_buffer((32, 32), weight_dtype, name="B", scope="shared.wsram",
                       strides=[n, 1] if case == "dynamic_operand_stride" else None,
                       elem_offset=n if case == "dynamic_operand_offset" else 0)
    c = tir.decl_buffer(c_shape, "float32", name="C", scope="shared.rsram",
                       strides=[32, 1] if case == "explicit_accumulator_strides" else None,
                       elem_offset=1 if case == "accumulator_offset" else 0)
    output = tir.decl_buffer(c_shape, "float32", name="output")
    local_buffers = [a, b, c]
    params = [source.data, weight.data, output.data]
    buffer_map = {source.data: source, weight.data: weight, output.data: output}
    if case in ("dynamic_extent", "dynamic_minimum", "dynamic_shape", "dynamic_ramp_stride",
                "dynamic_operand_stride", "dynamic_operand_offset"):
        params.append(n)

    fill_region = _region(c, access=2)
    if case == "partial_extent":
        fill_region = _region(c, extents=[16, 32], access=2)
    elif case == "partial_start":
        fill_region = _region(c, starts=[1, 0], extents=[31, 32], access=2)
    elif case in ("ramp_stride_zero", "ramp_stride_two", "dynamic_ramp_stride"):
        stride = {"ramp_stride_zero": 0, "ramp_stride_two": 2,
                  "dynamic_ramp_stride": n}[case]
        fill_region = tir.BufferLoad(c, [tir.Ramp(0, stride, 32), tir.Ramp(0, 1, 32)])
    elif case == "scalable_ramp_lanes":
        # A last-dimension scalable Ramp is legal TIR but is not a static
        # region.  Do not feed it to the legacy GEMM region normalizer.
        fill_region = tir.BufferLoad(c, [0, tir.Ramp(0, 1, tir.vscale() * 4)])
    elif case == "load_predicate":
        fill_region = _region(c, access=2, predicate=tir.IntImm("bool", True))
    elif ramp_fill:
        fill_region = tir.BufferLoad(c, [tir.Ramp(0, 1, 32), tir.Ramp(0, 1, 32)])
    zero = tir.FloatImm("float32", -0.0 if case == "negative_zero" else 0.0)
    fill = tir.Evaluate(_call(FILL, [fill_region, zero]))

    args = [_region(a, access=1), _region(b, access=1), _region(c),
            False, True, 32, 32, 32, 0, False, 32, 32, 0, 0, 1, 0, 0, 0, 0]
    if explicit_offset or case in ("offset", "optional_d"):
        args.append(2048 if case == "offset" else 0)
    if case == "optional_d":
        args.append(_region(output, access=2))
    elif case == "already_clear":
        args[9] = tir.IntImm("bool", True)
    elif case == "nonzero_c_coordinates":
        args[17] = 1
    elif case == "alias_view":
        view = tir.decl_buffer(c_shape, "float32", name="C_view", data=c.data,
                               scope="shared.rsram")
        args[2] = _region(view)
    elif case == "hidden_c":
        # mbar is a PrimExpr/BufferLoad slot.  Using policy/stride instead
        # would fail the old GEMM constructor's IntImm cast before proof.
        args[16] = tir.BufferLoad(c, [0, 0])
    annotations = None
    if case == "call_annotations":
        annotations = {"unreviewed_effect": tir.IntImm("int32", 1)}
    elif case == "annotation_hidden_c":
        annotations = {"unreviewed_effect": tir.BufferLoad(c, [0, 0])}
    gemm = tir.Evaluate(_call(op, args, annotations))
    loop_body = [_copy(source, a), _copy(weight, b), gemm]
    if case == "two_gemms_one_loop":
        loop_body.append(tir.Evaluate(_call(op, list(args))))
    elif case == "c_read_between_updates":
        loop_body.insert(0, _copy(c, output))
    elif case == "unknown_external":
        loop_body.insert(0, tir.Evaluate(tir.call_extern("int32", "unknown_effect")))
    elif case == "unknown_pure":
        loop_body.insert(0, tir.Evaluate(tir.call_pure_extern("int32", "unknown_pure")))
    elif case == "store_predicate":
        loop_body.insert(0, tir.BufferStore(c, tir.FloatImm("float32", 1), [0, 0],
                                          predicate=tir.IntImm("bool", True)))

    if case == "zero_extent":
        extent = 0
    elif case == "dynamic_extent":
        extent = n
    elif case == "dynamic_minimum":
        minimum = n
    k = tir.Var("k", "int32")
    loop_span = tvm.ir.Span(tvm.ir.SourceName("mma_clear_fixture"), 10, 14, 1, 1)
    loop = tir.For(k, minimum, extent, tir.ForKind.SERIAL, tir.SeqStmt(loop_body),
                   annotations={"num_stages": 2, "clear_test.k_metadata": 7},
                   step=2 if case == "step2" else 1, span=loop_span)
    tile_sequence = [fill, loop, _copy(c, output)]
    if case == "same_c_gemm_outside_candidate":
        # Its own writer avoids unrelated diverging-scope BF16 diagnostics.
        tile_sequence.extend([_copy(source, a), tir.Evaluate(_call(op, list(args)))])
    elif case == "two_same_c_candidates":
        k2 = tir.Var("k2", "int32")
        second = tir.For(k2, 4, 2, tir.ForKind.SERIAL,
                         tir.SeqStmt([_copy(source, a), _copy(weight, b),
                                      tir.Evaluate(_call(op, list(args)))]),
                         annotations={"num_stages": 2})
        tile_sequence.extend([tir.Evaluate(_call(FILL, [_region(c, access=2), zero])),
                              second, _copy(c, output)])

    outer = tir.Var("outer", "int32")
    body = tir.For(outer, 2, 4, tir.ForKind.SERIAL, tir.SeqStmt(tile_sequence),
                   annotations={"clear_test.outer_metadata": 11}, step=1,
                   span=tvm.ir.Span(tvm.ir.SourceName("mma_clear_fixture"), 8, 18, 1, 1))
    if case == "outer_handle_alias":
        body = tir.LetStmt(tir.Var("alias", "handle"), c.data, body)
    elif case == "pointer_escape":
        body = tir.SeqStmt([tir.Evaluate(c.data), body])
    elif case == "outer_unknown_pure":
        body = tir.SeqStmt([tir.Evaluate(tir.call_pure_extern("int32", "outer_unknown")), body])
    elif case == "outer_load_predicate":
        load = tir.BufferLoad(weight, [0, 0], predicate=tir.IntImm("bool", True))
        body = tir.SeqStmt([tir.Evaluate(load), body])
    elif case == "outer_store_predicate":
        store = tir.BufferStore(c, tir.FloatImm("float32", 1), [0, 0],
                                predicate=tir.IntImm("bool", True))
        body = tir.SeqStmt([store, body])

    matches = []
    if case == "match_buffer":
        view = tir.decl_buffer(c_shape, "float32", name="matched_C", scope="shared.rsram")
        matches = [tir.MatchBufferRegion(view, tir.BufferRegion(c, [
            tvm.ir.Range.from_min_extent(0, size) for size in c_shape
        ]))]
    root = tir.Block([], [], [], "root", body, alloc_buffers=local_buffers,
                     match_buffers=matches, annotations={"clear_test.root_metadata": 13})
    func = tir.PrimFunc(params, tir.BlockRealize([], True, root), buffer_map=buffer_map)
    func = func.with_attr("global_symbol", "main")
    if case != "missing_noalias":
        func = func.with_attr("tir.noalias", case != "false_noalias")
    if case == "nonrestrict":
        func = func.with_attr("tl.non_restrict_params", [source.data])
    return func


def _legalize(func, enabled=None, *, target_description=SUNMMIO_TARGET_DESC):
    target = tvm.target.Target(target_description)
    config = {} if enabled is None else {OPTION: enabled}
    with target, tvm.transform.PassContext(config=config):
        mod = tvm.IRModule({"main": func.with_attr("target", target)})
        return tl.transform.LegalizeSunmmioGemm()(mod)["main"]


def _assert_loop_headers_preserved(before, after):
    original = _nodes(before.body, tir.For)
    changed = _nodes(after.body, tir.For)
    assert len(original) == len(changed)
    for old, new in zip(original, changed):
        assert old.loop_var.same_as(new.loop_var)
        for field in ("min", "extent", "annotations"):
            tvm.ir.assert_structural_equal(getattr(old, field), getattr(new, field))
        assert old.kind == new.kind
        assert old.thread_binding is None and new.thread_binding is None
        assert (old.step is None) == (new.step is None)
        if old.step is not None:
            tvm.ir.assert_structural_equal(old.step, new.step)
        # This branch serializes inherited Stmt.span but does not expose it
        # as a Python For property. Serialize together to preserve object
        # identity: equal graph references are stronger than equal values.
        graph = json.loads(tvm.ir.save_json(tvm.runtime.convert([old, new])))
        nodes = graph["nodes"]
        pair = nodes[graph["root_index"]]
        assert pair["type"] == "ffi.Array" and len(pair["data"]) == 2
        old_node, new_node = [nodes[index] for index in pair["data"]]
        assert old_node["type"] == new_node["type"] == "tir.For"
        old_span, new_span = old_node["data"]["span"], new_node["data"]["span"]
        assert nodes[old_span]["type"] == nodes[new_span]["type"] == "ir.Span"
        assert old_span == new_span


@pytest.mark.parametrize("op", GEMMS)
@pytest.mark.parametrize("minimum,extent", [(0, 1), (0, 3), (5, 4)])
@pytest.mark.parametrize("explicit_offset", [False, True])
def test_bf16_first_k_clear_preserves_two_stripes_and_loop_metadata(op, minimum, extent, explicit_offset):
    func = _fixture(op=op, minimum=minimum, extent=extent, explicit_offset=explicit_offset)
    off = _legalize(func, False)
    on = _legalize(func, True)
    assert len(_calls(off.body, FILL)) == 1
    assert not _calls(on.body, FILL)
    before_gemms, after_gemms = _calls(off.body, GEMMS), _calls(on.body, GEMMS)
    assert len(before_gemms) == len(after_gemms) == 2
    offsets = [int(call.args[19]) if len(call.args) > 19 else 0 for call in after_gemms]
    assert offsets == [0, 2048]
    k_loop = next(loop for loop in _nodes(on.body, tir.For) if loop.loop_var.name == "k")
    for old, new in zip(before_gemms, after_gemms):
        assert int(old.args[9]) == 0
        tvm.ir.assert_structural_equal(new.args[9], k_loop.loop_var == k_loop.min)
        assert old.op.same_as(new.op) and len(old.args) == len(new.args)
        for index, (old_arg, new_arg) in enumerate(zip(old.args, new.args)):
            if index != 9:
                tvm.ir.assert_structural_equal(old_arg, new_arg)
        tvm.ir.assert_structural_equal(old.annotations, new.annotations)
    before_copies, after_copies = _calls(off.body, COPY), _calls(on.body, COPY)
    assert len(before_copies) == len(after_copies) == 4
    for old, new in zip(before_copies, after_copies):
        tvm.ir.assert_structural_equal(old, new)
    restages = [call for call in after_copies if "src_offset_byte" in call.annotations]
    assert len(restages) == 1 and int(restages[0].annotations["src_offset_byte"]) == 1024
    _assert_loop_headers_preserved(func, on)
    tvm.ir.assert_structural_equal(off.attrs, on.attrs)
    tvm.ir.assert_structural_equal(off.body.block.annotations, on.body.block.annotations)
    assert tir.analysis.verify_ssa(on)


@pytest.mark.parametrize("op", GEMMS)
def test_default_matches_explicit_enabled_fusion(op):
    func = _fixture(op=op)
    default, enabled = _legalize(func), _legalize(func, True)
    tvm.ir.assert_structural_equal(default, enabled)
    assert not _calls(default.body, FILL)
    assert len(_calls(default.body, GEMMS)) == 2
    loop = next(loop for loop in _nodes(default.body, tir.For) if loop.loop_var.name == "k")
    for gemm in _calls(default.body, GEMMS):
        tvm.ir.assert_structural_equal(gemm.args[9], loop.loop_var == loop.min)


def test_unit_stride_full_ramp_fill_is_eligible():
    result = _legalize(_fixture(ramp_fill=True), True)
    assert not _calls(result.body, FILL)
    loop = next(loop for loop in _nodes(result.body, tir.For) if loop.loop_var.name == "k")
    gemms = _calls(result.body, GEMMS)
    assert len(gemms) == 2
    for gemm in gemms:
        tvm.ir.assert_structural_equal(gemm.args[9], loop.loop_var == loop.min)


REJECTED_CASES = [
    "negative_zero", "partial_extent", "partial_start", "offset", "optional_d",
    "already_clear", "nonzero_c_coordinates", "rank3_input", "rank3_accumulator",
    "alias_view", "hidden_c", "c_read_between_updates", "two_gemms_one_loop",
    "same_c_gemm_outside_candidate", "two_same_c_candidates", "zero_extent",
    "dynamic_extent", "dynamic_minimum", "dynamic_shape", "step2",
    "explicit_accumulator_strides", "accumulator_offset",
    "dynamic_operand_stride", "dynamic_operand_offset",
    "ramp_stride_zero", "ramp_stride_two", "dynamic_ramp_stride", "scalable_ramp_lanes",
    "load_predicate", "store_predicate", "outer_load_predicate", "outer_store_predicate",
    "outer_handle_alias", "match_buffer", "missing_noalias", "false_noalias", "nonrestrict",
    "pointer_escape", "unknown_external", "unknown_pure", "outer_unknown_pure",
    "call_annotations", "annotation_hidden_c",
]


@pytest.mark.parametrize("case", REJECTED_CASES)
def test_unsafe_candidate_keeps_disabled_legalizer_result(case):
    # Every safety case passes the BF16 dtype gate.  Malformed fill regions
    # are not operands of the old GEMM normalizer: clear fusion must reject
    # them without throwing, while ordinary BF16 legalization still runs.
    func = _fixture(case)
    disabled = _legalize(func, False)
    enabled = _legalize(func, True)
    tvm.ir.assert_structural_equal(enabled, disabled)
    assert len(_calls(enabled.body, FILL)) == (2 if case == "two_same_c_candidates" else 1)


@pytest.mark.parametrize("dtype,weight_dtype", [
    ("float32", "float32"), ("float16", "float16"),
    ("bfloat16", "float32"), ("float32", "bfloat16"),
])
def test_non_bf16_and_mixed_inputs_keep_existing_path(dtype, weight_dtype):
    func = _fixture(dtype=dtype, weight_dtype=weight_dtype)
    disabled, enabled = _legalize(func, False), _legalize(func, True)
    tvm.ir.assert_structural_equal(enabled, disabled)
    assert len(_calls(enabled.body, FILL)) == 1
    assert all(int(call.args[9]) == 0 for call in _calls(enabled.body, GEMMS))


@pytest.mark.parametrize("case", ["two_gemms_one_loop", "same_c_gemm_outside_candidate",
                                 "two_same_c_candidates"])
def test_same_accumulator_ambiguity_also_preserves_bf16_legalization(case):
    func = _fixture(case)
    disabled, enabled = _legalize(func, False), _legalize(func, True)
    tvm.ir.assert_structural_equal(enabled, disabled)
    assert len(_calls(enabled.body, GEMMS)) == 4
    assert all(int(call.args[9]) == 0 for call in _calls(enabled.body, GEMMS))


def test_enabled_pass_is_idempotent_after_bf16_stripe_expansion():
    once = _legalize(_fixture(), True)
    twice = _legalize(once, True)
    tvm.ir.assert_structural_equal(once, twice)


def test_option_does_not_rewrite_non_sunmmio_target():
    func = _fixture(dtype="float32")
    disabled = _legalize(func, False, target_description="llvm")
    enabled = _legalize(func, True, target_description="llvm")
    tvm.ir.assert_structural_equal(enabled, disabled)
    assert len(_calls(enabled.body, FILL)) == 1
