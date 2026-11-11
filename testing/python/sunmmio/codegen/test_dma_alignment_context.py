"""DMA alignment proofs use stable facts from the enclosing TIR scope."""

import pytest

import tilelang.language as T
from tilelang import tvm
from tilelang.engine.lower import lower
from tilelang.layout import make_mxzz_layout, make_row_major, make_zz_layout
from testing.python.sunmmio.common.codegen_validation import codegen_sunmmio_suvm_mlir, validate_suvm_mlir_with_npuir_opt
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


def scope_copy_source(make_body, kind):
    """Exercise codegen directly, preserving Var identity across lexical scopes."""
    tir = tvm.tir
    # Layout transforms require full matrix extents. Select one of two complete
    # matrices, while retaining a row offset that must be proven zero.
    shape = (64, 128) if kind == "dma" else (2, 32, 128)
    src_scope = "global" if kind == "dma" else "shared.rsram"
    src = tir.decl_buffer(shape, "bfloat16", name="Src", scope=src_scope)
    dst = tir.decl_buffer(shape, "bfloat16", name="Dst", scope="shared.rsram")
    slot = tir.Var("slot", "int32")
    unit = tir.Call("handle", tvm.ir.Op.get("tl.odma_unit"), [tir.StringImm("odma0" if kind == "dma" else "odma1")])

    def region(buffer, access, index):
        indices = [index, 0] if kind == "dma" else [index // 32, index % 32, 0]
        extents = [32, 128] if kind == "dma" else [1, 32, 128]
        return tir.Call("handle", tvm.ir.Op.get("tl.tileop.region"), [tir.BufferLoad(buffer, indices), access, *extents])

    def copy(index, destination=False):
        args = [region(src, 1, 0 if destination else index), region(dst, 2, index if destination else 0)]
        if kind == "dma":
            args.append(tir.IntImm("int32", 0))
        args.append(unit)
        call = tir.Call("handle", tvm.ir.Op.get("tl.dma_copy" if kind == "dma" else "tl.sunmmio_layout_transform"), args)
        sync = tir.Call("handle", tvm.ir.Op.get("tl.sunmmio_sync"), [1 if kind == "dma" else 2])
        return tir.SeqStmt([tir.Evaluate(call), tir.Evaluate(sync)])

    body = tir.DeclBuffer(src, tir.DeclBuffer(dst, make_body(copy, slot)))
    func = tir.PrimFunc([src.data, dst.data, slot], body)
    func = func.with_attr("global_symbol", "main").with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
    zz = make_zz_layout(shape, axes=[len(shape) - 2, len(shape) - 1])
    func = func.with_attr("layout_map", {src: zz, dst: zz if kind == "dma" else make_row_major(shape)})
    return codegen_sunmmio_suvm_mlir(tvm.IRModule({"main": func}))


def scoped_copy_body(scope, copy, slot):
    tir = tvm.tir
    v = tir.Var("v", "int32")
    zero = tir.Evaluate(0)
    if scope == "sibling_let":
        return tir.SeqStmt([tir.LetStmt(v, 0, copy(v)), tir.LetStmt(v, 32, copy(v))])
    if scope == "expr_then_stmt":
        return tir.SeqStmt([tir.Evaluate(tir.Let(v, 0, v + 1)), tir.LetStmt(v, 32, copy(v))])
    if scope == "let_value":
        row = tir.Var("row", "int32")
        return tir.LetStmt(row, tir.Let(v, 0, v), tir.LetStmt(v, 32, copy(v)))
    if scope == "loop_bound_let":
        i = tir.Var("i", "int32")
        return tir.For(i, tir.Let(v, 0, v), 1, tir.ForKind.SERIAL, tir.LetStmt(v, 32, copy(v)))
    if scope == "condition_let":
        return tir.IfThenElse(tir.Let(v, 0, v) == 0, tir.LetStmt(v, 32, copy(v)), zero)
    if scope == "region_let":
        return tir.LetStmt(v, 0, copy(tir.Let(v, 32, v)))
    if scope == "branch_let":
        return tir.IfThenElse(slot == 0, tir.LetStmt(v, 0, copy(v)), tir.LetStmt(v, 32, copy(v)))
    if scope == "loop_then_let":
        return tir.SeqStmt([tir.For(v, 0, 1, tir.ForKind.SERIAL, copy(v)), tir.LetStmt(v, 32, copy(v))])
    if scope == "block_then_let":
        # Codegen requires block variables to already have a runtime binding.
        iv = tir.IterVar(tvm.ir.Range.from_min_extent(0, 1), slot, tir.IterVar.DataPar)
        block = tir.Block([iv], [], [], "singleton", copy(slot))
        return tir.SeqStmt([block, tir.LetStmt(slot, 32, copy(slot))])
    if scope == "thread_then_let":
        iv = tir.IterVar(tvm.ir.Range.from_min_extent(0, 1), v, tir.IterVar.ThreadIndex, "blockIdx.x")
        return tir.SeqStmt([tir.AttrStmt(iv, "thread_extent", 1, copy(v)), tir.LetStmt(v, 32, copy(v))])
    if scope == "assert":
        return tir.AssertStmt(slot % 32 == 0, tir.StringImm("aligned"), copy(slot))
    if scope == "ordered_facts":
        # The range and subsequent binding rely on the enclosing constraints.
        row = tir.Var("row", "int32")
        return tir.AttrStmt(
            slot == 32,
            "tl.assume",
            0,
            tir.For(v, slot, 1, tir.ForKind.SERIAL, tir.LetStmt(row, v, copy(row))),
        )
    if scope == "unreachable":
        return tir.IfThenElse(slot == 0, tir.IfThenElse(slot != 0, copy(1), copy(slot)), zero)
    if scope == "empty_loop":
        return tir.For(v, 0, 0, tir.ForKind.SERIAL, copy(1))
    raise AssertionError(scope)


@pytest.mark.parametrize("kind", ["dma", "layout_transform"])
@pytest.mark.parametrize(
    "scope",
    [
        "sibling_let",
        "expr_then_stmt",
        "let_value",
        "loop_bound_let",
        "condition_let",
        "region_let",
        "branch_let",
        "loop_then_let",
        "block_then_let",
        "thread_then_let",
        "assert",
        "ordered_facts",
        "unreachable",
        "empty_loop",
    ],
)
def test_copy_alignment_uses_only_active_bindings(scope, kind, tmp_path):
    def make_body(copy, slot):
        return scoped_copy_body(scope, copy, slot)

    src = scope_copy_source(make_body, kind)
    validate_suvm_mlir_with_npuir_opt(src, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))


@pytest.mark.parametrize("kind", ["dma", "layout_transform"])
def test_copy_alignment_rejects_misaligned_expression_local_var(kind):
    def make_body(copy, slot):
        v = tvm.tir.Var("v", "int32")
        return tvm.tir.LetStmt(v, 0, copy(tvm.tir.Let(v, 16, v)))

    with pytest.raises(tvm.error.InternalError, match="must be provably aligned to region extent"):
        scope_copy_source(make_body, kind)


@pytest.mark.parametrize("kind", ["dma", "layout_transform"])
@pytest.mark.parametrize("scope", ["branch", "assume", "assert", "loop", "block", "thread"])
@pytest.mark.parametrize("destination", [False, True])
def test_copy_alignment_rejects_expired_constraints(scope, kind, destination, tmp_path):
    def make_body(copy, slot):
        tir = tvm.tir
        v = tir.Var("v", "int32")
        first = copy(slot, destination)
        second = copy(slot, destination)
        if scope == "branch":
            # A valid then branch must not authorize the else branch.
            return tir.IfThenElse(slot % 32 == 0, first, copy(slot, destination))
        if scope == "assume":
            first = tir.AttrStmt(slot % 32 == 0, "tl.assume", 0, first)
        elif scope == "assert":
            first = tir.AssertStmt(slot % 32 == 0, tir.StringImm("aligned"), first)
        elif scope == "loop":
            first = tir.For(v, 0, 1, tir.ForKind.SERIAL, copy(v, destination))
            second = tir.LetStmt(v, slot, copy(v, destination))
        elif scope == "block":
            iv = tir.IterVar(tvm.ir.Range.from_min_extent(0, 1), slot, tir.IterVar.DataPar)
            first = tir.Block([iv], [], [], "singleton", first)
        else:
            iv = tir.IterVar(tvm.ir.Range.from_min_extent(0, 1), v, tir.IterVar.ThreadIndex, "blockIdx.x")
            first = tir.AttrStmt(iv, "thread_extent", 1, copy(v, destination))
            second = tir.LetStmt(v, slot, copy(v, destination))
        # The later copy must not inherit the earlier scope's alignment facts.
        return tir.SeqStmt([first, second])

    operand = "destination" if destination else "source"
    with pytest.raises(tvm.error.InternalError, match=f"copy {operand} buffer .*must be provably aligned to region extent"):
        scope_copy_source(make_body, kind)

    # Check that the same scopes are supported when each copy has a live proof.
    def guarded_body(copy, slot):
        def guarded_copy(index, destination=False):
            return tvm.tir.IfThenElse(index % 32 == 0, copy(index, destination), None)

        return make_body(guarded_copy, slot)

    src = scope_copy_source(guarded_body, kind)
    validate_suvm_mlir_with_npuir_opt(src, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))
