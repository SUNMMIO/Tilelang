import pytest

from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing


def _check(original, expected):
    """Helper function to verify structural equality after transformations"""
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tl.transform.LegalizeNegativeIndex()(mod)
    expected = tvm.IRModule.from_expr(expected.with_attr("global_symbol", "main"))
    tvm.ir.assert_structural_equal(mod["main"], expected["main"], True)


def test_buffer_load_negative_index_legalized():
    """
    Test that negative indices are legalized by adding buffer extent.
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        value = A[-1]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value

    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        value = A[1023]  # A[-1] becomes A[1023]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value

    _check(before, after)


def test_buffer_load_mixed_negative_positive_indices():
    """
    Test mixed negative and positive indices - only negative ones are legalized.
    """

    @T.prim_func
    def before(A: T.Tensor((1024, 512), T.float32)):
        value = A[-1, 10]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value

    @T.prim_func
    def after(A: T.Tensor((1024, 512), T.float32)):
        value = A[1023, 10]  # A[-1, 10] becomes A[1023, 10]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value

    _check(before, after)


def test_buffer_load_multiple_negative_indices():
    """
    Test multiple negative indices in different dimensions.
    """

    @T.prim_func
    def before(A: T.Tensor((1024, 512, 256), T.float32)):
        value = A[-1, -2, -3]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value

    @T.prim_func
    def after(A: T.Tensor((1024, 512, 256), T.float32)):
        value = A[1023, 510, 253]  # -1+1024=1023, -2+512=510, -3+256=253
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value

    _check(before, after)


def test_buffer_load_negative_index_in_expression():
    """
    Test negative index as part of a larger expression.
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        B = T.alloc_buffer((1024,), T.float32)
        for i in T.serial(1, 1024):
            value = A[-i]
            B[-i] = value

    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        B = T.alloc_buffer((1024,), T.float32)
        for i in T.serial(1, 1024):
            value = A[1024 - i]
            B[1024 - i] = value

    _check(before, after)


def test_buffer_load_non_negative_index_unchanged():
    """
    Test that non-negative indices remain unchanged.
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        value = A[0]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value

    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        # No changes expected for non-negative indices
        value = A[0]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value

    _check(before, after)


def test_buffer_load_unknown_sign_index_warning():
    """
    Test that indices with unknown sign trigger warnings but are processed.
    This test mainly checks that the pass doesn't crash on unknown signs.
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        i = T.Var("i", T.int32)
        value = A[i]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value

    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        i = T.Var("i", T.int32)
        # Unknown sign indices should remain unchanged
        value = A[i]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value

    _check(before, after)


def test_buffer_load_vector_index_negative_broadcast():
    """
    Test negative indices in vectorized operations (broadcast case).
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        vec = T.Broadcast(-1, 4)
        value = A[vec]
        B = T.alloc_buffer((4,), T.float32)
        B[T.Ramp(0, 1, 4)] = value

    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        # vec is unused and can be delimed by Simplify.
        vec = T.Broadcast(-1, 4)  # noqa: F841
        value = A[T.Broadcast(1023, 4)]
        B = T.alloc_buffer((4,), T.float32)
        B[T.Ramp(0, 1, 4)] = value

    _check(before, after)


def test_buffer_load_vector_index_negative_ramp():
    """
    Test negative indices in vectorized operations (ramp case).
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        vec = T.Ramp(-4, 1, 4)  # indices: [-4, -3, -2, -1]
        value = A[vec]
        B = T.alloc_buffer((4,), T.float32)
        B[T.Ramp(0, 1, 4)] = value

    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        # vec is unused and can be delimed by Simplify.
        vec = T.Ramp(-4, 1, 4)  # noqa: F841
        value = A[T.Ramp(1020, 1, 4)]
        B = T.alloc_buffer((4,), T.float32)
        B[T.Ramp(0, 1, 4)] = value

    _check(before, after)


def test_buffer_load_nested_buffer_loads():
    """
    Test legalization with nested buffer load expressions.
    """

    @T.prim_func
    def before(A: T.Tensor((1024, 512), T.float32)):
        inner_val = A[-1, 10]
        outer_val = A[inner_val.astype(T.int32), -2]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = outer_val

    @T.prim_func
    def after(A: T.Tensor((1024, 512), T.float32)):
        inner_val = A[1023, 10]
        outer_val = A[inner_val.astype(T.int32), 510]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = outer_val

    _check(before, after)


def test_buffer_store_negative_index():
    """
    Test negative indices in buffer store operations are legalized.
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        A[-1] = 42.0

    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        A[1023] = 42.0

    _check(before, after)


def test_buffer_store_mixed_negative_positive_indices():
    """
    Test mixed negative and positive indices in buffer store.
    """

    @T.prim_func
    def before(A: T.Tensor((1024, 512), T.float32)):
        A[-1, 10] = 42.0

    @T.prim_func
    def after(A: T.Tensor((1024, 512), T.float32)):
        A[1023, 10] = 42.0

    _check(before, after)


def test_buffer_store_multiple_negative_indices():
    """
    Test multiple negative indices in different dimensions for buffer store.
    """

    @T.prim_func
    def before(A: T.Tensor((1024, 512, 256), T.float32)):
        A[-1, -2, -3] = 42.0

    @T.prim_func
    def after(A: T.Tensor((1024, 512, 256), T.float32)):
        A[1023, 510, 253] = 42.0  # -1+1024=1023, -2+512=510, -3+256=253

    _check(before, after)


def test_buffer_store_negative_index_in_expression():
    """
    Test negative index as part of a larger expression in buffer store.
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        for i in T.serial(1, 1024):
            A[-i] = i * 2.0

    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        for i in T.serial(1, 1024):
            A[1024 - i] = i * 2.0

    _check(before, after)


def test_buffer_store_vector_index_negative_broadcast():
    """
    Test negative indices in vectorized store operations (broadcast case).
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        vec = T.Broadcast(-1, 4)
        values = T.Broadcast(42.0, 4)
        A[vec] = values

    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        # vec is unused and can be delimed by Simplify.
        vec = T.Broadcast(-1, 4)  # noqa: F841
        values = T.Broadcast(42.0, 4)
        A[T.Broadcast(1023, 4)] = values

    _check(before, after)


def test_buffer_store_vector_index_negative_ramp():
    """
    Test negative indices in vectorized store operations (ramp case).
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        vec = T.Ramp(-4, 1, 4)  # indices: [-4, -3, -2, -1]
        values = T.Ramp(0.0, 1.0, 4)  # values: [0.0, 1.0, 2.0, 3.0]
        A[vec] = values

    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        # vec is unused and can be delimed by Simplify.
        vec = T.Ramp(-4, 1, 4)  # noqa: F841
        values = T.Ramp(0.0, 1.0, 4)
        A[T.Ramp(1020, 1, 4)] = values

    _check(before, after)


def test_buffer_store_nested_in_condition():
    """
    Test negative index buffer store within conditional statements.
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32), flag: T.int32):
        if flag > 0:
            A[-1] = 42.0
        else:
            A[-2] = 24.0

    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32), flag: T.int32):
        if flag > 0:
            A[1023] = 42.0
        else:
            A[1022] = 24.0

    _check(before, after)


def _modulo_branch_module(kind, force_rewrite=False, normalized=False):
    """Keep let bindings explicit so modulo range analysis runs in the branch."""
    tir = tvm.tir
    ids = tir.decl_buffer((32,), "int32", name="ids")
    out = tir.decl_buffer((32,), "int32", name="out")
    token = tir.Var("token", "int32")
    expert = tir.Var("expert", "int32")
    local = tir.Var("local_expert", "int32")
    tokens = tir.Var("tokens", "int32")
    is_live_case = kind in {"reachable", "unknown", "select"}
    true_index = 31 if normalized and is_live_case else -1
    value = tir.Let(
        expert,
        tir.BufferLoad(ids, [token]),
        tir.Let(local, tir.floormod(expert, 2), tir.BufferLoad(ids, [true_index]) + local),
    )
    unsafe = tir.LetStmt(
        expert,
        tir.BufferLoad(ids, [token]),
        tir.LetStmt(local, tir.floormod(expert, 2), tir.BufferStore(out, tir.BufferLoad(ids, [true_index]), [local])),
    )
    false_value = tir.BufferLoad(ids, [30 if normalized else -2]) if is_live_case else tir.const(0, "int32")
    safe = tir.BufferStore(out, false_value, [0])
    if kind == "then":
        body = tir.IfThenElse(token < 0, unsafe, safe)
    elif kind == "else":
        body = tir.IfThenElse(token >= 0, safe, unsafe)
    elif kind == "nested":
        body = tir.IfThenElse(token < 8, tir.IfThenElse(token >= 8, unsafe, safe), None)
    elif kind == "likely":
        body = tir.IfThenElse(tir.likely(token < 0), unsafe, safe)
    elif kind == "expression_then":
        body = tir.BufferStore(out, tir.if_then_else(token < 0, value, 0), [0])
    elif kind == "expression_else":
        body = tir.BufferStore(out, tir.if_then_else(token >= 0, 0, value), [0])
    elif kind == "expression_likely":
        body = tir.BufferStore(out, tir.if_then_else(tir.likely(token < 0), value, 0), [0])
    elif kind == "reachable":
        body = tir.IfThenElse(token < 16, unsafe, safe)
    elif kind == "unknown":
        body = tir.IfThenElse(token < tokens, unsafe, safe)
    elif kind == "select":
        # Select evaluates both values; the false predicate must not suppress
        # negative-index normalization (or introduce contradictory constraints).
        body = tir.BufferStore(out, tir.Select(token < 0, value, false_value), [0])
    else:
        raise ValueError(kind)
    body = tir.For(token, 0, 32, tir.ForKind.SERIAL, body)
    if force_rewrite:
        body = tir.SeqStmt([tir.BufferStore(out, 7, [31 if normalized else -1]), body])
    func = tir.PrimFunc([ids.data, out.data, tokens], body, buffer_map={ids.data: ids, out.data: out})
    return tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))


@pytest.mark.parametrize("kind", ["then", "else", "nested", "likely", "expression_then", "expression_else", "expression_likely"])
@pytest.mark.parametrize("force_rewrite", [False, True])
def test_unreachable_modulo_branch_is_not_analyzed_or_rewritten(kind, force_rewrite):
    original = _modulo_branch_module(kind, force_rewrite)
    result = tl.transform.LegalizeNegativeIndex()(original)
    expected = _modulo_branch_module(kind, force_rewrite, normalized=True)
    # Unreachable code is retained unchanged. A live negative index outside the
    # branch forces the second traversal and must still be legalized.
    tvm.ir.assert_structural_equal(result, expected, map_free_vars=True)


@pytest.mark.parametrize("kind", ["reachable", "unknown", "select"])
def test_conditional_negative_indices_keep_live_branch_semantics(kind):
    original = _modulo_branch_module(kind)
    result = tl.transform.LegalizeNegativeIndex()(original)
    expected = _modulo_branch_module(kind, normalized=True)
    tvm.ir.assert_structural_equal(result, expected, map_free_vars=True)


def test_negative_indices_in_condition_are_still_rewritten():
    @T.prim_func
    def before(A: T.Tensor((32,), T.int32), B: T.Tensor((32,), T.int32)):
        if A[-1] > 0:
            B[-1] = 1
        else:
            B[-2] = 2

    @T.prim_func
    def after(A: T.Tensor((32,), T.int32), B: T.Tensor((32,), T.int32)):
        if A[31] > 0:
            B[31] = 1
        else:
            B[30] = 2

    _check(before, after)


def test_empty_input_lowers_without_kernel_workaround():
    from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget
    from tilelang.utils.target import determine_target, target_is_sunmmio

    target = tvm.target.Target(determine_target("sunmmio", return_object=True), tvm.target.Target("llvm"))
    with target:

        @T.prim_func
        def main(ids: T.Tensor((32,), T.int32), output: T.Tensor((2,), T.int32)):
            with T.Kernel():
                output[0] = 0
                output[1] = 0
                for token in T.serial(32):
                    if token < 0:
                        expert = ids[token]
                        local_expert = expert % 2
                        output[local_expert] = 1

        with tvm.transform.PassContext(opt_level=3):
            mod = LowerAndLegalize(tvm.IRModule.from_expr(main), target)
            mod = OptimizeForTarget(mod, target)

    # The production simplifier removes the dead branch after legalization.
    device_funcs = [
        func for func in mod.functions.values() if func.attrs and "target" in func.attrs and target_is_sunmmio(func.attrs["target"])
    ]
    assert device_funcs
    remaining = []
    for func in device_funcs:
        tvm.tir.stmt_functor.post_order_visit(
            func.body,
            lambda node: remaining.append(node) if isinstance(node, (tvm.tir.FloorMod, tvm.tir.BufferLoad)) else None,
        )
    assert not remaining


if __name__ == "__main__":
    tilelang.testing.main()
