"""DMA alignment proofs use stable facts from the enclosing TIR scope."""

import pytest

import tilelang.language as T
from tilelang import tvm
from tilelang.engine.lower import lower
from tilelang.layout import make_mxzz_layout, make_zz_layout
from testing.python.sunmmio.common.codegen_validation import validate_suvm_mlir_with_npuir_opt
from testing.python.sunmmio.common.compile_pipeline import target


@target("sunmmio")
def contextual_dma_kernel(dtype=T.bfloat16, store=False, context="then", explicit=False):
    shape = (64, 128)
    layout = make_zz_layout(shape) if dtype == T.bfloat16 else make_mxzz_layout(shape, dtype=dtype)

    def transfer(a, b, tile, index):
        start = index // 32 * 32 if explicit else index
        return T.copy(tile, b[start, 0]) if store else T.copy(a[start, 0], tile)

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, T.placement.replicated(), dtype, layout=layout),
        B: T.MeshTensor(shape, T.placement.replicated(), dtype, layout=layout),
        slot: T.int32,
    ):
        with T.Kernel():
            tile = T.alloc_shared((32, 128), dtype, scope="shared.rsram")
            T.copy(A[0, 0], tile)
            if context == "then":
                if slot >= 0 and slot <= 32 and slot % 32 == 0:
                    transfer(A, B, tile, slot)
            elif context == "else":
                if slot < 0 or slot > 32 or slot % 32 != 0:
                    T.evaluate(0)
                else:
                    transfer(A, B, tile, slot)
            elif context == "let":
                row = slot + 32
                if row >= 0 and row <= 32 and row % 32 == 0:
                    transfer(A, B, tile, row)
            elif context == "loop":
                for row in T.serial(64):
                    if row % 32 == 0:
                        transfer(A, B, tile, row)
            elif context == "assume":
                with T.attr(slot >= 0 and slot <= 32 and slot % 32 == 0, "tl.assume", "aligned"):
                    transfer(A, B, tile, slot)
            elif context == "misaligned":
                if slot >= 0 and slot < 32 and slot % 32 == 16:
                    transfer(A, B, tile, slot)
            elif context == "leak":
                if slot >= 0 and slot <= 32 and slot % 32 == 0:
                    transfer(A, B, tile, slot)
                transfer(A, B, tile, slot)
            elif context == "mutable":
                row = T.alloc_var("int32", init=slot)
                if row % 32 == 0:
                    row = row + 1
                    transfer(A, B, tile, row)
            T.copy(tile, B[0, 0])

    return main


def check_contextual_copy(dtype, store, context, tmp_path):
    actual = lower(contextual_dma_kernel(dtype, store, context), target="sunmmio")
    control = lower(contextual_dma_kernel(dtype, store, context, explicit=True), target="sunmmio")
    assert actual.kernel_source == control.kernel_source
    validate_suvm_mlir_with_npuir_opt(actual.kernel_source, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))


@pytest.mark.parametrize("dtype", [T.bfloat16, T.mxfp8, T.mxfp4])
@pytest.mark.parametrize("store", [False, True])
def test_guarded_dma_matches_explicit_alignment(dtype, store, tmp_path):
    check_contextual_copy(dtype, store, "then", tmp_path)


@pytest.mark.parametrize("context", ["else", "let", "loop", "assume"])
def test_dma_alignment_uses_enclosing_context(context, tmp_path):
    check_contextual_copy(T.bfloat16, False, context, tmp_path)


@pytest.mark.parametrize("context", ["misaligned", "leak", "mutable"])
def test_dma_alignment_does_not_accept_invalid_or_unstable_proofs(context):
    with pytest.raises(tvm.error.InternalError, match="must be provably aligned to region extent"):
        lower(contextual_dma_kernel(context=context), target="sunmmio")
