import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
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


if __name__ == "__main__":
    pytest.main([__file__])
