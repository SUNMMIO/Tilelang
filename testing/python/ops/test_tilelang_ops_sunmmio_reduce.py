import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
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
    mod = tilelang.transform.LayoutReducer()(mod)
    mod = tilelang.transform.LayoutInference()(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    return mod


@tvm.tir.functor.visitor
class ReduceIRChecker(tvm.tir.PyStmtExprVisitor):
    def __init__(self, target_buffer_name="Out_shared"):
        super().__init__()
        self.target_buffer_name = target_buffer_name
        self.max_loop_depth = 0
        self.current_loop_depth = 0
        self.found_ktiled_stage = True
        self.has_in_tile_reduce = False
        self.scope_entry_count = 0
        self.execution_loop_count = 0

    def visit_for_(self, op):
        self.current_loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.current_loop_depth)

        ann = op.annotations
        if ann:
            if "tile.loop_stage" in ann and ann["tile.loop_stage"] != 2:
                self.found_ktiled_stage = False

            if ann.get("tile.scope_entry", 0) == 1:
                self.scope_entry_count += 1
            if ann.get("tile.execution", 0) == 1:
                self.execution_loop_count += 1

        self.visit_stmt(op.body)
        self.current_loop_depth -= 1

    def visit_call_(self, op):
        if op.op.name == "tl.vector_core_in_tile_reduce":
            self.has_in_tile_reduce = True
        super().visit_call_(op)


def reduce_kernel_builder(rank, shape, tile_size, index_map, reduce_axis, dtype="float16"):
    out_shape = list(shape[:reduce_axis]) + list(shape[reduce_axis + 1 :])
    if not out_shape:  # Handle scalar reduction case
        out_shape = [1]

    # Pre-calculate output tileview info outside T.prim_func
    out_tile_size = []
    out_index_map = []
    for i, axis in enumerate(index_map):
        if axis == reduce_axis:
            continue
        out_tile_size.append(tile_size[i])
        if axis > reduce_axis:
            out_index_map.append(axis - 1)
        else:
            out_index_map.append(axis)

    @T.prim_func
    def main(A: T.Tensor(shape, dtype), Out: T.Tensor(out_shape, dtype)):
        with T.Kernel(1, threads=128) as (bx,):
            # For Sunmmio, src and dst must be in shared.rsram for vector core operations
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")

            T.annotate_tileview({A_shared: make_tileview(A_shared, tile_size, index_map)})
            T.annotate_tileview({Out_shared: make_tileview(Out_shared, out_tile_size, out_index_map)})

            T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=reduce_axis)
            T.copy(Out_shared, Out)

    return tvm.IRModule({"main": main})


# (Rank, Shape, TileSize, IndexMap, ReduceAxis, ExpectedDepth, ExpectedInTileReduce)
# For Sunmmio, all dimensions should be multiples of 32 for simplicity in these tests.
REDUCE_TEST_CASES = [
    # 1D (Using asram)
    (1, (1024,), (32,), (0,), 0, 2, True),
    # 2D (Used as 1D equivalent: reducing last dim of a 2D tensor)
    (2, (32, 1024), (32,), (1,), 1, 3, True),
    # 2D
    (2, (128, 128), (32, 32), (0, 1), 1, 4, True),
    (2, (128, 128), (32, 32), (0, 1), 0, 4, True),
    # 3D
    (3, (32, 128, 128), (32, 32), (1, 2), 2, 5, True),
    (3, (32, 128, 128), (32, 32), (1, 2), 1, 5, True),
    (3, (32, 128, 128), (32, 32), (1, 2), 0, 5, False),
    # 4D
    (4, (32, 32, 128, 128), (32, 32), (2, 3), 3, 6, True),
    (4, (32, 32, 128, 128), (32, 32), (2, 3), 1, 6, False),
    # 5D
    (5, (32, 32, 32, 128, 128), (32, 32), (3, 4), 4, 7, True),
    (5, (32, 32, 32, 128, 128), (32, 32), (3, 4), 0, 7, False),
]


@pytest.mark.parametrize("rank, shape, tile_size, index_map, reduce_axis, expected_depth, expected_in_tile", REDUCE_TEST_CASES)
def test_tilelang_reduce_sunmmio(rank, shape, tile_size, index_map, reduce_axis, expected_depth, expected_in_tile):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = reduce_kernel_builder(rank, shape, tile_size, index_map, reduce_axis)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    checker = ReduceIRChecker()
    checker.visit_stmt(mod["main"].body)

    assert checker.found_ktiled_stage, "All loops in the reduction block should have tile.loop_stage == 2"
    assert checker.max_loop_depth == expected_depth, f"Expected depth {expected_depth}, but found {checker.max_loop_depth} for rank {rank}"

    if expected_in_tile:
        assert checker.has_in_tile_reduce, "Expected vector_core_in_tile_reduce intrinsic but not found"
    else:
        assert not checker.has_in_tile_reduce, "Did not expect vector_core_in_tile_reduce intrinsic but found it"

    assert checker.scope_entry_count >= 1, "Missing tile.scope_entry annotation"
    assert checker.execution_loop_count >= len(tile_size), "Missing tile.execution annotation"


if __name__ == "__main__":
    pytest.main([__file__])
