import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.layout import make_zz_layout, make_aligned_row_major, make_row_major
from tilelang.layout.cute_layout import is_same_layout
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from tvm.tir import Block
from tvm.tir.stmt_functor import post_order_visit
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


def _infer_layout_map(mod, target):
    """Run the Sunmmio pipeline up to SunmmioLayoutInference and return the
    inferred {buffer_name: layout} from the block's layout_map annotation."""
    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.InferSramScope()(mod)
    mod = tilelang.transform.LegalizeSunmmioDataPath()(mod)
    mod = tilelang.transform.LayoutReducer()(mod)
    mod = tilelang.transform.SunmmioLayoutInference()(mod)

    result = {}

    def visit(node):
        if isinstance(node, Block) and "layout_map" in node.annotations:
            for buf, layout in node.annotations["layout_map"].items():
                result[buf.name] = layout

    post_order_visit(mod["main"].body, visit)
    return result


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

    checker = ReduceIRChecker()
    checker.visit_stmt(mod["main"].body)

    assert checker.scope_root is not None, "Missing tile.domain root on lowered reduction"
    root_ann = checker.scope_root.annotations
    tile_size = [int(x) for x in root_ann["tile.tile_size"]]
    execution_domain_axes = [int(x) for x in root_ann["tile.execution_domain_axes"]]

    assert checker.has_in_tile_reduce, "Expected vector_core_in_tile_reduce intrinsic but not found"
    assert tile_size == [4, 32], "Reduction should preserve the blockwise kept-axis tile instead of collapsing to [1, 32]"
    assert execution_domain_axes == [0, 1]


def test_tilelang_reduce_sunmmio_blocked_axis_yields_aligned_rowmajor():
    """Reducing a blocked (ZZ) axis to a non-32-multiple output gives the dst an
    alignment-padded row-major layout (e.g. (40,) -> covered 64), not plain."""
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)

    @T.prim_func
    def main(A: T.Tensor((64, 40), "float16"), Out: T.Tensor((40,), "float16")):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((64, 40), "float16", scope="shared.rsram")
            Out_shared = T.alloc_shared((40,), "float16", scope="shared.rsram")
            T.annotate_layout({A_shared: make_zz_layout(A_shared)})
            T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=0)
            T.copy(Out_shared, Out)

    with tvm.target.Target(target):
        layouts = _infer_layout_map(tvm.IRModule({"main": main}), target)

    out = layouts["Out_shared"]
    assert is_same_layout(out, make_aligned_row_major((40,), "float16", 64))
    assert not is_same_layout(out, make_row_major((40,)))


def test_tilelang_reduce_sunmmio_3d_blocked_axis_aligned():
    """3D ZZ source, reduce the inner blocked axis -> 2D aligned row-major dst
    with a non-32-multiple inner extent (the (2,40,256) -> (2,40) shape)."""
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)

    @T.prim_func
    def main(A: T.Tensor((2, 40, 256), "float16"), Out: T.Tensor((2, 40), "float16")):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((2, 40, 256), "float16", scope="shared.rsram")
            Out_shared = T.alloc_shared((2, 40), "float16", scope="shared.rsram")
            T.annotate_layout({A_shared: make_zz_layout(A_shared)})
            T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=2)
            T.copy(Out_shared, Out)

    with tvm.target.Target(target):
        layouts = _infer_layout_map(tvm.IRModule({"main": main}), target)
    assert is_same_layout(layouts["Out_shared"], make_aligned_row_major((2, 40), "float16", 64))


def test_tilelang_reduce_sunmmio_chained_reduce_stays_aligned():
    """Reducing twice: the first reduce makes an (8,40) aligned row-major; the
    second reduce off that *unblocked* buffer must stay aligned (the chained
    path goes through DeriveLayoutLike -> MakeAlignedRowMajor, not plain)."""
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)

    @T.prim_func
    def main(A: T.Tensor((8, 64, 40), "float16"), Out: T.Tensor((40,), "float16")):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((8, 64, 40), "float16", scope="shared.rsram")
            M_shared = T.alloc_shared((8, 40), "float16", scope="shared.rsram")
            Out_shared = T.alloc_shared((40,), "float16", scope="shared.rsram")
            T.annotate_layout({A_shared: make_zz_layout(A_shared)})
            T.copy(A, A_shared)
            T.reduce_sum(A_shared, M_shared, dim=1)  # blocked axis -> aligned (8,40)
            T.reduce_sum(M_shared, Out_shared, dim=0)  # off unblocked -> aligned
            T.copy(Out_shared, Out)

    with tvm.target.Target(target):
        layouts = _infer_layout_map(tvm.IRModule({"main": main}), target)
    assert is_same_layout(layouts["M_shared"], make_aligned_row_major((8, 40), "float16", 64))
    assert is_same_layout(layouts["Out_shared"], make_aligned_row_major((40,), "float16", 64))
    assert not is_same_layout(layouts["Out_shared"], make_row_major((40,)))


def test_tilelang_reduce_sunmmio_nonblocked_reduce_preserves_zz():
    """Reducing a NON-blocked leading axis keeps the surviving ZZ block
    structure (DeriveLayoutLike path), not a flat row-major."""
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)

    @T.prim_func
    def main(A: T.Tensor((40, 64, 64), "float16"), Out: T.Tensor((64, 64), "float16")):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((40, 64, 64), "float16", scope="shared.rsram")
            Out_shared = T.alloc_shared((64, 64), "float16", scope="shared.rsram")
            T.annotate_layout({A_shared: make_zz_layout(A_shared)})
            T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=0)
            T.copy(Out_shared, Out)

    with tvm.target.Target(target):
        layouts = _infer_layout_map(tvm.IRModule({"main": main}), target)
    out = layouts["Out_shared"]
    assert is_same_layout(out, make_zz_layout((64, 64), [0, 1], (32, 32)))
    assert not is_same_layout(out, make_row_major((64, 64)))


def test_tilelang_reduce_sunmmio_aligned_dst_is_noop_when_32_multiple():
    """A 32-multiple reduce output gets no padding: aligned row-major collapses
    to plain row-major (no spurious covered-extent inflation)."""
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)

    @T.prim_func
    def main(A: T.Tensor((64, 64), "float16"), Out: T.Tensor((64,), "float16")):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((64, 64), "float16", scope="shared.rsram")
            Out_shared = T.alloc_shared((64,), "float16", scope="shared.rsram")
            T.annotate_layout({A_shared: make_zz_layout(A_shared)})
            T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=0)
            T.copy(Out_shared, Out)

    with tvm.target.Target(target):
        layouts = _infer_layout_map(tvm.IRModule({"main": main}), target)
    assert is_same_layout(layouts["Out_shared"], make_row_major((64,)))


def test_tilelang_reduce_sunmmio_aligned_output_lowers_end_to_end():
    """End-to-end (through LowerTileOp): reduce dim1 of a ZZ (40,64) source — a
    blocked axis with a 32-multiple inner extent — yields an aligned, covered-
    padded (40,) output that lowers and stores to unpadded DRAM via an unpad
    transform.  Two sunmmio_layout_transforms: ZZ-reblock on the load, unpad on
    the store.  This proves the aligned (covered != logical) reduce dst is fully
    lowerable, not just inferred."""
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)

    @T.prim_func
    def main(A: T.Tensor((40, 64), "float16"), Out: T.Tensor((40,), "float16")):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((40, 64), "float16", scope="shared.rsram")
            Out_shared = T.alloc_shared((40,), "float16", scope="shared.rsram")
            T.annotate_layout({A_shared: make_zz_layout(A_shared)})
            T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=1)
            T.copy(Out_shared, Out)

    mod_in = tvm.IRModule({"main": main})
    with tvm.target.Target(target):
        layouts = _infer_layout_map(tvm.IRModule({"main": main}), target)
        assert is_same_layout(layouts["Out_shared"], make_aligned_row_major((40,), "float16", 64))
        mod = apply_sunmmio_passes(mod_in, target)

    names = []
    post_order_visit(
        mod["main"].body,
        lambda n: names.append(n.op.name) if isinstance(n, tvm.tir.Call) and hasattr(n.op, "name") else None,
    )
    assert names.count("tl.sunmmio_layout_transform") == 2, names


@pytest.mark.xfail(
    strict=True,
    reason="Pre-existing reduce TileView planner limitation (unrelated to "
    "alignment): PlanReduceTileViews requires the source's innermost extent to "
    "be a 32-multiple, so a (64,40) source (inner=40) has no compatible tile "
    "candidate. Remove this xfail if the planner gains non-32-multiple support.",
)
def test_tilelang_reduce_sunmmio_non32_multiple_inner_source_unsupported():
    """Documents that reducing a source whose innermost extent is not a
    32-multiple is not lowerable by the current reduce TileView planner."""
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)

    @T.prim_func
    def main(A: T.Tensor((64, 40), "float16"), Out: T.Tensor((40,), "float16")):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((64, 40), "float16", scope="shared.rsram")
            Out_shared = T.alloc_shared((40,), "float16", scope="shared.rsram")
            T.annotate_layout({A_shared: make_zz_layout(A_shared)})
            T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=0)
            T.copy(Out_shared, Out)

    with tvm.target.Target(target):
        apply_sunmmio_passes(tvm.IRModule({"main": main}), target)


if __name__ == "__main__":
    pytest.main([__file__])
