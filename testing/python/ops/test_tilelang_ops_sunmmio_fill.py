import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.tileview import make_tileview
from tilelang.utils.target import SUNMMIO_TARGET_DESC
import pytest

tilelang.env.disable_cache()


def apply_sunmmio_passes(mod, target):
    """Apply the full SUNMMIO pass pipeline used for Fill lowering."""
    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.InferSramScope()(mod)
    mod = tilelang.transform.LayoutReducer()(mod)
    mod = tilelang.transform.LayoutInference()(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = tilelang.transform.LegalizeTilesLoop()(mod)
    mod = tilelang.transform.TilesLoop()(mod)
    return mod


def fill_kernel(B, M, N, block_B, block_M, block_N, tile_size, index_map, dtype="float16"):
    @T.prim_func
    def main(A: T.Tensor((B, M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), T.ceildiv(B, block_B), threads=128) as (bx, by, bz):
            A_shared = T.alloc_shared((block_B, block_M, block_N), dtype)
            # Annotate tileview for the shared buffer
            T.annotate_tileview({A_shared: make_tileview(A_shared, tile_size, index_map)})

            # 1. Fill the entire buffer -> value 1.0
            T.fill(A_shared, T.float16(1.0))

            # 2. Fill a region of the buffer -> value 2.0
            # Region: First half of M dimension
            T.fill(A_shared[0:block_B, 0 : block_M // 2, 0:block_N], T.float16(2.0))

            # 3. Fill a region with offset -> value 3.0
            # Region: Second half of M dimension
            T.fill(A_shared[0:block_B, block_M // 2 : block_M, 0:block_N], T.float16(3.0))

            # 4. Clear the buffer -> value 0.0
            T.clear(A_shared)

            # Dummy output to prevent dead code elimination if we were to go further,
            # but for LowerTileOp check it is not strictly necessary.
            # We add a dma_copy just to match the user's provided example structure which had a copy.
            T.copy(A_shared, A[bz * block_B : (bz + 1) * block_B, by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])

    return tvm.IRModule({"main": main})


@tvm.tir.functor.visitor
class LoopNestChecker(tvm.tir.PyStmtExprVisitor):
    def __init__(self, target_buffer_name="A_shared"):
        super().__init__()
        self.target_buffer_name = target_buffer_name
        self.max_loop_depth = 0
        self.current_loop_depth = 0
        self.found_5_layer_loop = False

    def visit_for_(self, op):
        self.current_loop_depth += 1
        self.visit_stmt(op.body)
        self.current_loop_depth -= 1

    def visit_buffer_store_(self, op):
        if op.buffer.name.startswith(self.target_buffer_name):
            # Check if we are inside a 5-layer loop nest
            # The 5 layers are: 3 tiled loops + 1 inner loop + 1 vectorized loop
            if self.current_loop_depth == 5:
                self.found_5_layer_loop = True
            # Update max depth just in case
            self.max_loop_depth = max(self.max_loop_depth, self.current_loop_depth)


def fill_kernel_config(B, M, N, block_B, block_M, block_N, tile_size, index_map, fill_type="full", dtype="float16"):
    @T.prim_func
    def main(A: T.Tensor((B, M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), T.ceildiv(B, block_B), threads=128) as (bx, by, bz):
            A_shared = T.alloc_shared((block_B, block_M, block_N), dtype)
            # Annotate tileview for the shared buffer
            T.annotate_tileview({A_shared: make_tileview(A_shared, tile_size, index_map)})

            if fill_type == "full":
                T.fill(A_shared, T.float16(1.0))
            elif fill_type == "region_first_half":
                # Fill first half of M dimension
                T.fill(A_shared[0:block_B, 0 : block_M // 2, 0:block_N], T.float16(2.0))
            elif fill_type == "region_second_half":
                # Fill second half of M dimension
                T.fill(A_shared[0:block_B, block_M // 2 : block_M, 0:block_N], T.float16(3.0))
            elif fill_type == "clear":
                T.clear(A_shared)

            # Dummy output
            T.copy(A_shared, A[bz * block_B : (bz + 1) * block_B, by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])

    return tvm.IRModule({"main": main})


FILL_TEST_CASES = [
    ("full", 1.0),
    ("region_first_half", 2.0),
    ("region_second_half", 3.0),
    ("clear", 0.0),
]


@pytest.mark.parametrize("fill_type, expected_val", FILL_TEST_CASES)
@pytest.mark.parametrize("dtype", ["float16"])
def test_tilelang_fill_sunmmio(fill_type, expected_val, dtype):
    # Parameters
    B, M, N = 64, 512, 1024
    block_B, block_M, block_N = 16, 256, 128
    tile_size = (32, 32)
    index_map = (-2, -1)

    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = fill_kernel_config(B, M, N, block_B, block_M, block_N, tile_size, index_map, fill_type=fill_type, dtype=dtype)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    script = mod.script()
    assert "tl.tileop.fill" not in script, "tl.tileop.fill should be lowered"

    # Check for loop nest depth
    checker = LoopNestChecker()
    checker.visit_stmt(mod["main"].body)

    # All fill operations (full, region, clear) should be tiled
    assert checker.found_5_layer_loop, (
        f"Expected 5-layer nested loop for fill type {fill_type}, but found max depth {checker.max_loop_depth}"
    )

    # Verify the specific value is stored
    # Simplified check as per user suggestion
    if expected_val == 0.0:
        # T.clear() typically lowers to T.float16(0.0) or T.float16(0) or T.Cast("float16", 0)
        expected_patterns = ["T.float16(0.0)", "T.float16(0)", 'T.Cast("float16", 0)']
    else:
        # For non-zero values
        val_int = int(expected_val)
        if expected_val == val_int:
            # e.g. 1.0 -> T.float16(1) or T.float16(1.0)
            expected_patterns = [f"T.float16({val_int})", f"T.float16({expected_val})"]
        else:
            expected_patterns = [f"T.float16({expected_val})"]

    found = any(pattern in script for pattern in expected_patterns)
    assert found, f"Should find store of {expected_val} for fill type {fill_type}. Looked for: {expected_patterns}"


if __name__ == "__main__":
    test_tilelang_fill_sunmmio("float16")
