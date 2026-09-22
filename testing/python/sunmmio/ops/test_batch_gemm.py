"""Operator-level TIR checks for Sunmmio Batch GEMM.

Representative input/lowered IR is saved under build/batch-gemm-ops-ir/.
"""

from pathlib import Path

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.utils.target import determine_target
from tvm import tir


tilelang.env.disable_cache()


def batch_gemm(
    batch=2,
    a_batched=True,
    w_batched=True,
    output_batched=True,
    partial=False,
    w_batch=None,
):
    a_shape = (batch, 32, 32) if a_batched else (32, 32)
    w_shape = (w_batch if w_batch is not None else batch, 32, 32) if w_batched else (32, 32)
    c_shape = (batch, 32, 32) if output_batched else (32, 32)

    @T.prim_func
    def main(A: T.Tensor(a_shape, "bfloat16"), W: T.Tensor(w_shape, "bfloat16")):
        with T.Kernel():
            a = T.alloc_shared(a_shape, "bfloat16")
            w = T.alloc_shared(w_shape, "bfloat16")
            c = T.alloc_shared(c_shape, "float32")
            if a_batched:
                T.copy(A[0, 0, 0], a)
            else:
                T.copy(A[0, 0], a)
            if w_batched:
                T.copy(W[0, 0, 0], w)
            else:
                T.copy(W[0, 0], w)
            if partial:
                T.gemm(a[1:3, :, :], w[1:3, :, :], c[1:3, :, :], clear_accum=True)
            else:
                T.gemm(a, w, c, clear_accum=True)

    return tvm.IRModule({"main": main})


def lower_batch_gemm(mod, ir_case=None):
    if ir_case is not None:
        ir_dir = Path(__file__).resolve().parents[4] / "build" / "batch-gemm-ops-ir"
        ir_dir.mkdir(parents=True, exist_ok=True)
        (ir_dir / f"{ir_case}.input.tir").write_text(mod.script(show_meta=True), encoding="utf-8")
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.LegalizeSunmmioBatchGemmViews()(mod)
        mod = tilelang.transform.InferSramScope()(mod)
        mod = tilelang.transform.LegalizeSunmmioDataPath()(mod)
        mod = tilelang.transform.LayoutReducer()(mod)
        mod = tilelang.transform.SunmmioLayoutInference()(mod)
        mod = tilelang.transform.LegalizeSunmmioGemm()(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
    if ir_case is not None:
        (ir_dir / f"{ir_case}.lowered.tir").write_text(mod.script(show_meta=True), encoding="utf-8")
    return mod


def collect_calls(mod, op_name):
    calls = []

    def visit(node):
        if isinstance(node, tir.Call) and hasattr(node.op, "name") and node.op.name == op_name:
            calls.append(node)

    tir.stmt_functor.post_order_visit(mod["main"].body, visit)
    return calls


def region_shape(region):
    assert region.op.name == "tl.tileop.region"
    return tuple(int(extent) for extent in region.args[2:])


@pytest.mark.parametrize(
    ("a_batched", "w_batched", "output_batched"),
    [
        (True, True, True),
        (False, True, True),
        (True, False, True),
        (False, False, True),
        (True, True, False),
        (False, True, False),
    ],
)
def test_batch_gemm_operand_regions(a_batched, w_batched, output_batched):
    ir_case = {
        (True, True, True): "batched-b2",
        (True, False, True): "shared-w-b2",
        (True, True, False): "reduction-b2",
    }.get((a_batched, w_batched, output_batched))
    mod = lower_batch_gemm(
        batch_gemm(
            a_batched=a_batched,
            w_batched=w_batched,
            output_batched=output_batched,
        ),
        ir_case=ir_case,
    )
    mmas = collect_calls(mod, "tl.mma_sunmmio")
    assert len(mmas) == 2  # BF16 A uses the two ASRAM stripes for M=32.
    for mma in mmas:
        expected_a = (2, 32, 32) if a_batched else (32, 32)
        expected_w = (2, 32, 32) if w_batched else (32, 32)
        expected_c = (2, 32, 32) if output_batched else (32, 32)
        assert [region_shape(region) for region in mma.args[:3]] == [expected_a, expected_w, expected_c]
        if not output_batched:
            assert not bool(mma.args[5])  # Clear once, then accumulate across batches.


def test_batch_one_keeps_rank_three():
    mod = lower_batch_gemm(batch_gemm(batch=1), ir_case="unit-b1")
    mmas = collect_calls(mod, "tl.mma_sunmmio")
    assert len(mmas) == 2
    for mma in mmas:
        assert [region_shape(region) for region in mma.args[:3]] == [(1, 32, 32), (1, 32, 32), (1, 32, 32)]


def test_partial_batch_uses_compact_regions():
    mod = lower_batch_gemm(batch_gemm(batch=4, partial=True), ir_case="partial-b1-e2-of4")
    mmas = collect_calls(mod, "tl.mma_sunmmio")
    assert len(mmas) == 2
    for mma in mmas:
        for region in mma.args[:3]:
            assert region_shape(region) == (2, 32, 32)
            assert "compact" in region.args[0].buffer.name


def test_partial_batch_views_are_legalized_before_sram_inference():
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        full = tir.transform.BindTarget(target)(batch_gemm(batch=4))
        assert tvm.ir.structural_equal(tilelang.transform.LegalizeSunmmioBatchGemmViews()(full), full)

        mod = tir.transform.BindTarget(target)(batch_gemm(batch=4, partial=True))
        views = tilelang.transform.LegalizeSunmmioBatchGemmViews()(mod)
        view_ir = views.script()
        assert "_batch_a_compact_" in view_ir
        assert "_batch_b_compact_" in view_ir
        assert "_batch_c_compact_" in view_ir
        assert 'scope="shared.dyn"' in view_ir

        scoped = tilelang.transform.InferSramScope()(views)
        scoped_ir = scoped.script()
        assert 'scope="shared.asram"' in scoped_ir
        assert 'scope="shared.wsram"' in scoped_ir
        assert 'scope="shared.rsram"' in scoped_ir


@pytest.mark.parametrize("output_batched", [True, False])
def test_batch_gemm_rejects_mismatched_input_batches(output_batched):
    with pytest.raises(ValueError, match="batch extent"):
        batch_gemm(output_batched=output_batched, w_batch=3)


def test_weight_copy_expands_but_mma_stays_batched():
    mod = lower_batch_gemm(batch_gemm())
    weight_copies = [call for call in collect_calls(mod, "tl.dma_copy") if call.args[0].args[0].buffer.name == "W"]
    assert len(weight_copies) == 2
    assert all(region_shape(call.args[0]) == (1, 32, 32) for call in weight_copies)
    assert all(region_shape(mma.args[1]) == (2, 32, 32) for mma in collect_calls(mod, "tl.mma_sunmmio"))
