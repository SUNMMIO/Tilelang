import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.layout import make_zz_layout
from tilelang.tileview import make_tileview
from tilelang.utils.target import SUNMMIO_TARGET_DESC
import pytest

tilelang.env.disable_cache()


def apply_sunmmio_passes(mod, target):
    """Apply the SUNMMIO pass pipeline used for Reduce lowering."""
    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.InferSramScope()(mod)
    mod = tilelang.transform.LegalizeSunmmioDataPath()(mod)
    mod = tilelang.transform.LayoutReducer()(mod)
    mod = tilelang.transform.SunmmioLayoutInference()(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    return mod


def assert_reduce_lowering_is_ssa(mod):
    """Reduce lowering should be SSA-clean before LowerOpaqueBlock/ConvertSSA."""
    assert tvm.tir.analysis.verify_ssa(mod["main"]), mod.script()


@tvm.tir.functor.visitor
class ReduceIRChecker(tvm.tir.PyStmtExprVisitor):
    def __init__(self, target_buffer_name="Out_shared"):
        super().__init__()
        self.target_buffer_name = target_buffer_name
        self.has_in_tile_reduce = False
        self.scope_root = None
        self.scope_entry_count = 0
        self.execution_axes = []
        self.interior_axes = []
        self.saw_legacy_stage = False
        self.saw_legacy_execution = False

    def visit_for_(self, op):
        ann = op.annotations
        if ann:
            if "tile.domain" in ann:
                self.scope_root = op
            if ann.get("tile.scope_entry", 0) == 1:
                self.scope_entry_count += 1
            if "tile.execution_axis" in ann:
                self.execution_axes.append(int(ann["tile.execution_axis"]))
            if ann.get("tile.interior", 0) == 1:
                self.interior_axes.append(int(ann["tile.interior_axis"]))
            if "tile.loop_stage" in ann:
                self.saw_legacy_stage = True
            if "tile.execution" in ann:
                self.saw_legacy_execution = True

        super().visit_for_(op)

    def visit_call_(self, op):
        if op.op.name == "tl.vector_core_in_tile_reduce":
            self.has_in_tile_reduce = True
        super().visit_call_(op)


def reduce_kernel_builder(shape, reduce_axis, dtype="float16"):
    out_shape = list(shape[:reduce_axis]) + list(shape[reduce_axis + 1 :])
    if not out_shape:  # Handle scalar reduction case
        out_shape = [1]

    @T.prim_func
    def main(A: T.Tensor(shape, dtype), Out: T.Tensor(out_shape, dtype)):
        with T.Kernel(1, threads=128) as (bx,):
            # For Sunmmio, src and dst must be in shared.rsram for vector core operations
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")

            T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=reduce_axis)
            T.copy(Out_shared, Out)

    return tvm.IRModule({"main": main})


def reduce_kernel_with_blockwise_layout_builder(shape, reduce_axis, dtype="float32"):
    out_shape = list(shape[:reduce_axis]) + list(shape[reduce_axis + 1 :])
    if not out_shape:
        out_shape = [1]

    @T.prim_func
    def main(A: T.Tensor(shape, dtype), Out: T.Tensor(out_shape, dtype)):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")

            T.annotate_layout(
                {
                    A_shared: make_zz_layout(A_shared),
                }
            )

            T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=reduce_axis)
            T.copy(Out_shared, Out)

    return tvm.IRModule({"main": main})


def apply_reduce_op(reduce_op, buffer, out, reduce_axis, clear=True):
    if reduce_op == "sum":
        T.reduce_sum(buffer, out, dim=reduce_axis, clear=clear)
    elif reduce_op == "max":
        T.reduce_max(buffer, out, dim=reduce_axis, clear=clear)
    elif reduce_op == "min":
        T.reduce_min(buffer, out, dim=reduce_axis, clear=clear)
    else:
        raise ValueError(f"Unsupported reduce_op={reduce_op}")


def reduce_kernel_with_tileview_builder(shape, reduce_axis, tile_size=(8, 32), dtype="float16", clear=True, reduce_op="sum"):
    out_shape = list(shape[:reduce_axis]) + list(shape[reduce_axis + 1 :])
    if not out_shape:
        out_shape = [1]

    @T.prim_func
    def main(A: T.Tensor(shape, dtype), Out: T.Tensor(out_shape, dtype)):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")

            T.annotate_tileview({A_shared: make_tileview(A_shared, tile_size, (-2, -1))})
            T.copy(A, A_shared)
            if not clear:
                T.copy(Out, Out_shared)
            apply_reduce_op(reduce_op, A_shared, Out_shared, reduce_axis, clear=clear)
            T.copy(Out_shared, Out)

    return tvm.IRModule({"main": main})


def unaligned_reduce_kernel_builder(shape, reduce_axis, dtype="float16", clear=True, reduce_op="sum"):
    out_shape = list(shape[:reduce_axis]) + list(shape[reduce_axis + 1 :])
    if not out_shape:
        out_shape = [1]

    @T.prim_func
    def main(A: T.Tensor(shape, dtype), Out: T.Tensor(out_shape, dtype)):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")

            T.copy(A, A_shared)
            if not clear:
                T.copy(Out, Out_shared)
            apply_reduce_op(reduce_op, A_shared, Out_shared, reduce_axis, clear=clear)
            T.copy(Out_shared, Out)

    return tvm.IRModule({"main": main})


def multi_reduce_kernel_builder(shape=(32, 128, 128), reduce_axis=1, dtype="float16"):
    out_shape = list(shape[:reduce_axis]) + list(shape[reduce_axis + 1 :])
    if not out_shape:
        out_shape = [1]

    @T.prim_func
    def main(A: T.Tensor(shape, dtype), Out0: T.Tensor(out_shape, dtype), Out1: T.Tensor(out_shape, dtype)):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out0_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")
            Out1_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")

            T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out0_shared, dim=reduce_axis)
            T.reduce_sum(A_shared, Out1_shared, dim=reduce_axis)
            T.copy(Out0_shared, Out0)
            T.copy(Out1_shared, Out1)

    return tvm.IRModule({"main": main})


def _collect_buffer_loads(func, buffer_name):
    loads = []

    def visitor(node):
        if isinstance(node, tvm.tir.BufferLoad) and node.buffer.name == buffer_name:
            loads.append(node)

    tvm.tir.stmt_functor.post_order_visit(func.body, visitor)
    return loads


def _collect_buffer_stores(func, buffer_name):
    stores = []

    def visitor(node):
        if isinstance(node, tvm.tir.BufferStore) and node.buffer.name == buffer_name:
            stores.append(node)

    tvm.tir.stmt_functor.post_order_visit(func.body, visitor)
    return stores


def _collect_tile_loop_extents(func):
    extents = []

    @tvm.tir.functor.visitor
    class TileLoopVisitor(tvm.tir.PyStmtExprVisitor):
        def visit_for_(self, op):
            if op.annotations and "tile.domain" in op.annotations or op.annotations and "tile.execution_axis" in op.annotations:
                extents.append(int(op.extent))
            super().visit_for_(op)

    TileLoopVisitor().visit_stmt(func.body)
    return extents


def _collect_tile_domain_roots(func):
    roots = []

    @tvm.tir.functor.visitor
    class TileDomainVisitor(tvm.tir.PyStmtExprVisitor):
        def visit_for_(self, op):
            if op.annotations and "tile.domain" in op.annotations:
                roots.append(op)
            super().visit_for_(op)

    TileDomainVisitor().visit_stmt(func.body)
    return roots


def _ceildiv(lhs, rhs):
    return (lhs + rhs - 1) // rhs


def _expected_tiled_reduce(shape, reduce_axis):
    return len(shape) == 1 or reduce_axis >= len(shape) - 2


def _tail_identity_text(reduce_op):
    if reduce_op == "sum":
        return "T.float16(0.0)"
    if reduce_op == "max":
        return 'T.float16("-inf")'
    if reduce_op == "min":
        return 'T.float16("inf")'
    raise ValueError(f"Unsupported reduce_op={reduce_op}")


# (Shape, ReduceAxis, ExpectedInTileReduce)
# For Sunmmio, all dimensions should be multiples of 32 for simplicity in these tests.
REDUCE_TEST_CASES = [
    ((1024,), 0, True),
    ((32, 1024), 1, True),
    # 2D
    ((128, 128), 1, True),
    ((128, 128), 0, True),
    # 3D
    ((32, 128, 128), 2, True),
    ((32, 128, 128), 1, True),
    ((32, 128, 128), 0, False),
    # 4D
    ((32, 32, 128, 128), 3, True),
    ((32, 32, 128, 128), 1, False),
    # 5D
    ((32, 32, 32, 128, 128), 4, True),
    ((32, 32, 32, 128, 128), 0, False),
]


@pytest.mark.parametrize("shape, reduce_axis, expected_in_tile", REDUCE_TEST_CASES)
def test_tilelang_reduce_sunmmio(shape, reduce_axis, expected_in_tile):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = reduce_kernel_builder(shape, reduce_axis)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_reduce_lowering_is_ssa(mod)

    checker = ReduceIRChecker()
    checker.visit_stmt(mod["main"].body)

    assert checker.scope_root is not None, "Missing tile.domain root on lowered reduction"
    root_ann = checker.scope_root.annotations
    tile_size = [int(x) for x in root_ann["tile.tile_size"]]
    execution_domain_axes = [int(x) for x in root_ann["tile.execution_domain_axes"]]

    if expected_in_tile:
        assert checker.has_in_tile_reduce, "Expected vector_core_in_tile_reduce intrinsic but not found"
    else:
        assert not checker.has_in_tile_reduce, "Did not expect vector_core_in_tile_reduce intrinsic but found it"

    assert checker.scope_entry_count == 1, "Expected exactly one tile.scope_entry annotation"
    assert not checker.saw_legacy_stage, "Reduction should not emit legacy tile.loop_stage annotations"
    assert not checker.saw_legacy_execution, "Reduction should not emit legacy tile.execution annotations"
    assert sorted(checker.execution_axes) == list(range(len(tile_size))), (
        "tile.execution_axis annotations should cover every execution axis"
    )
    assert len(execution_domain_axes) == len(tile_size), "tile.execution_domain_axes rank must match tile.tile_size"
    assert set(checker.interior_axes).issuperset(set(range(len(tile_size)))), "Missing tile.interior annotations for one or more tile axes"


def test_tilelang_reduce_sunmmio_preserves_blockwise_kept_axis_tile():
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = reduce_kernel_with_blockwise_layout_builder((32, 32), 1, dtype="float32")

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_reduce_lowering_is_ssa(mod)

    checker = ReduceIRChecker()
    checker.visit_stmt(mod["main"].body)

    assert checker.scope_root is not None, "Missing tile.domain root on lowered reduction"
    root_ann = checker.scope_root.annotations
    tile_size = [int(x) for x in root_ann["tile.tile_size"]]
    execution_domain_axes = [int(x) for x in root_ann["tile.execution_domain_axes"]]

    assert checker.has_in_tile_reduce, "Expected vector_core_in_tile_reduce intrinsic but not found"
    assert tile_size == [4, 32], "Reduction should preserve the blockwise kept-axis tile instead of collapsing to [1, 32]"
    assert execution_domain_axes == [0, 1]


def test_tilelang_reduce_sunmmio_multiple_reduces_are_ssa_clean():
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = multi_reduce_kernel_builder()

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    assert_reduce_lowering_is_ssa(mod)


def test_tilelang_reduce_sunmmio_tiled_axis_tail_load_is_predicated():
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = reduce_kernel_with_tileview_builder((8, 63, 250), reduce_axis=2)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_reduce_lowering_is_ssa(mod)

    func = mod["main"]
    tile_loop_extents = _collect_tile_loop_extents(func)
    assert tile_loop_extents.count(8) >= 3, "Expected ceildiv tile extents for dimensions 8, 63, and 250"
    assert 7 not in tile_loop_extents, "truncdiv(63, 8) would drop the spatial tail tile"

    a_load_predicates = [load.predicate for load in _collect_buffer_loads(func, "A_shared") if load.predicate is not None]
    assert a_load_predicates, "Expected predicated source loads for reduce-axis tail tile"
    script = mod.script()
    assert "T.if_then_else" in script
    assert "T.float16(0.0)" in script
    assert any("< 250" in str(predicate) or "<250" in str(predicate) for predicate in a_load_predicates)
    assert all("< 63" not in str(predicate) and "<63" not in str(predicate) for predicate in a_load_predicates)

    out_store_predicates = [store.predicate for store in _collect_buffer_stores(func, "Out_shared") if store.predicate is not None]
    assert not out_store_predicates, "Reduce final write-back should remain unpredicated"


@pytest.mark.parametrize(
    "reduce_op",
    [
        "max",
        "min",
    ],
)
def test_tilelang_reduce_sunmmio_tiled_axis_tail_uses_predicated_update(reduce_op):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = reduce_kernel_with_tileview_builder((8, 63, 250), reduce_axis=2, reduce_op=reduce_op)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_reduce_lowering_is_ssa(mod)

    script = mod.script()
    assert "T.if_then_else" in script
    expected_identity = 'T.float16("-inf")' if reduce_op == "max" else 'T.float16("inf")'
    assert expected_identity in script
    a_load_predicates = [load.predicate for load in _collect_buffer_loads(mod["main"], "A_shared") if load.predicate is not None]
    acc_store_predicates = [
        store.predicate for store in _collect_buffer_stores(mod["main"], "Out_shared_acc") if store.predicate is not None
    ]
    assert a_load_predicates
    assert not acc_store_predicates
    assert any("< 250" in str(predicate) or "<250" in str(predicate) for predicate in a_load_predicates)


UNALIGNED_REDUCE_CASES = [
    ((1000,), 0, True),
    ((1000,), 0, False),
    ((33, 50), 0, True),
    ((33, 50), 0, False),
    ((33, 50), 1, True),
    ((33, 50), 1, False),
    ((5, 43, 249), 0, True),
    ((5, 43, 249), 0, False),
    ((5, 43, 249), 1, True),
    ((5, 43, 249), 1, False),
    ((5, 43, 249), 2, True),
    ((5, 43, 249), 2, False),
]


@pytest.mark.parametrize("shape, reduce_axis, clear", UNALIGNED_REDUCE_CASES)
def test_tilelang_reduce_sunmmio_unaligned_cases_from_tir_dump(shape, reduce_axis, clear):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = unaligned_reduce_kernel_builder(shape, reduce_axis=reduce_axis, clear=clear)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_reduce_lowering_is_ssa(mod)

    func = mod["main"]
    checker = ReduceIRChecker()
    checker.visit_stmt(func.body)
    assert checker.scope_root is not None, "Missing tile.domain root on lowered unaligned reduction"
    assert checker.has_in_tile_reduce == _expected_tiled_reduce(shape, reduce_axis)

    roots = _collect_tile_domain_roots(func)
    assert len(roots) == 1
    root = roots[0]
    tile_size = [int(x) for x in root.annotations["tile.tile_size"]]
    execution_domain_axes = [int(x) for x in root.annotations["tile.execution_domain_axes"]]
    domain = [int(x) for x in root.annotations["tile.domain"]]
    assert domain == list(shape)
    assert len(tile_size) == len(execution_domain_axes)

    execution_extents = []

    @tvm.tir.functor.visitor
    class ExecutionLoopVisitor(tvm.tir.PyStmtExprVisitor):
        def visit_for_(self, op):
            if op.annotations and "tile.execution_axis" in op.annotations:
                axis = int(op.annotations["tile.execution_axis"])
                execution_extents.append((axis, int(op.extent)))
            super().visit_for_(op)

    ExecutionLoopVisitor().visit_stmt(func.body)
    assert len(execution_extents) == len(tile_size)
    for axis, extent in execution_extents:
        domain_axis = execution_domain_axes[axis]
        expected_extent = _ceildiv(shape[domain_axis], tile_size[axis])
        assert extent == expected_extent
        if shape[domain_axis] % tile_size[axis] != 0:
            assert extent > shape[domain_axis] // tile_size[axis]

    script = mod.script()
    a_load_predicates = [load.predicate for load in _collect_buffer_loads(func, "A_shared") if load.predicate is not None]
    acc_store_predicates = [store.predicate for store in _collect_buffer_stores(func, "Out_shared_acc") if store.predicate is not None]
    assert not acc_store_predicates

    is_reduce_axis_tiled = reduce_axis in execution_domain_axes
    reduce_tile_size = tile_size[execution_domain_axes.index(reduce_axis)] if is_reduce_axis_tiled else 1
    has_reduce_axis_tail = is_reduce_axis_tiled and shape[reduce_axis] % reduce_tile_size != 0
    if has_reduce_axis_tail:
        assert a_load_predicates
        assert "T.if_then_else" in script
        assert _tail_identity_text("sum") in script
        assert any(
            f"< {shape[reduce_axis]}" in str(predicate) or f"<{shape[reduce_axis]}" in str(predicate) for predicate in a_load_predicates
        )
    else:
        assert not a_load_predicates

    if is_reduce_axis_tiled:
        assert ("Out_shared_res" in script) == (not clear)


@pytest.mark.parametrize("reduce_op", ["sum", "max", "min"])
def test_tilelang_reduce_sunmmio_unaligned_tiled_axis_tail_identity(reduce_op):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = unaligned_reduce_kernel_builder((5, 43, 249), reduce_axis=2, clear=True, reduce_op=reduce_op)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_reduce_lowering_is_ssa(mod)

    script = mod.script()
    assert "T.if_then_else" in script
    assert _tail_identity_text(reduce_op) in script
    assert "predicate=i2 * 32 + kj < 249" in script


def test_tilelang_reduce_sunmmio_non_tiled_axis_tail_has_no_reduce_predicate():
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = reduce_kernel_with_tileview_builder((8, 63, 250), reduce_axis=0)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_reduce_lowering_is_ssa(mod)

    checker = ReduceIRChecker()
    checker.visit_stmt(mod["main"].body)
    assert not checker.has_in_tile_reduce

    a_load_predicates = [load.predicate for load in _collect_buffer_loads(mod["main"], "A_shared") if load.predicate is not None]
    out_store_predicates = [store.predicate for store in _collect_buffer_stores(mod["main"], "Out_shared") if store.predicate is not None]
    assert not a_load_predicates
    assert not out_store_predicates


if __name__ == "__main__":
    pytest.main([__file__])
