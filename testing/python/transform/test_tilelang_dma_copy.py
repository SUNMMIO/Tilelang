"""Test that SUNMMIO copy lowering emits tl.dma_copy with tl.tileop.region args,
and that each region can be normalized back to a BufferRegion with full metadata."""

import tilelang
import tilelang as tl
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from tvm import tir
from tvm.tir import PyStmtExprVisitor
import tilelang.env as env
import pytest

tilelang.env.disable_cache()


def simple_copy_kernel(M, N, block_M, block_N, dtype="float16"):
    """A minimal kernel with T.copy from global to shared memory."""

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype),):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            T.copy(A[by * block_M, bx * block_N], A_shared)

    return tvm.IRModule({"main": main})


def gemm_copy_kernel(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    """A kernel with T.copy + T.gemm to trigger layout inference."""

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return tvm.IRModule({"main": main})


class RegionRange:
    """Simple container for (min, extent) of a region axis."""

    def __init__(self, min_val, extent):
        self.min = min_val
        self.extent = extent


def normalize_region(region_call):
    """Decode a tl.tileop.region Call back into (buffer, extents, access_mask).

    This mirrors what NormalizeToBufferRegion does in C++:
      args[0] = BufferLoad (indices are per-axis minima)
      args[1] = access_mask (int)
      args[2+i] = extent for axis i

    On SUNMMIO, buffer remap is disabled so buffers retain their original
    N-D shape and indices always match the number of extents.

    Returns (buffer, extents, access_mask) where extents is a list of
    RegionRange objects with .min and .extent attributes.
    """
    assert isinstance(region_call, tir.Call)
    assert region_call.op.name == "tl.tileop.region"

    load = region_call.args[0]
    assert isinstance(load, tir.BufferLoad)

    access_mask = int(region_call.args[1])
    num_extents = len(region_call.args) - 2

    assert len(
        load.indices) == num_extents, (f"Expected {num_extents} indices, got {len(load.indices)}")

    ranges = []
    for i in range(num_extents):
        ranges.append(RegionRange(load.indices[i], region_call.args[2 + i]))

    return load.buffer, ranges, access_mask


@tir.functor.visitor
class _DmaCopyVisitor(PyStmtExprVisitor):
    """Walk TIR and collect tl.dma_copy calls and their region arguments."""

    def __init__(self):
        super().__init__()
        self.dma_copy_calls = []
        self.layout_map = {}

    def visit_block_(self, op: tir.Block) -> None:
        if "layout_map" in op.annotations:
            for key, layout in op.annotations["layout_map"].items():
                self.layout_map[key.name] = layout
        self.visit_stmt(op.body)

    def visit_call_(self, op: tir.Call) -> None:
        if hasattr(op, "op") and hasattr(op.op, "name") and op.op.name == "tl.dma_copy":
            self.dma_copy_calls.append(op)
        # Visit children
        for arg in op.args:
            self.visit_expr(arg)

    def visit_evaluate_(self, op: tir.Evaluate) -> None:
        self.visit_expr(op.value)


TEST_CASES = [
    # (M, N, block_M, block_N)
    (128, 128, 32, 32),
    (256, 256, 64, 64),
    (128, 256, 32, 64),
]


@pytest.mark.parametrize("M, N, block_M, block_N", TEST_CASES)
def test_tilelang_dma_copy(M, N, block_M, block_N):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = simple_copy_kernel(M, N, block_M, block_N)

    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)
        print(mod)

    # Walk the lowered IR and find dma_copy calls
    visitor = _DmaCopyVisitor()
    func = mod["main"]
    visitor.visit_stmt(func.body)

    # Verify that at least one tl.dma_copy call was emitted
    assert len(visitor.dma_copy_calls) > 0, ("Expected at least one tl.dma_copy call in lowered IR")

    for call in visitor.dma_copy_calls:
        # dma_copy should have exactly 2 arguments (src_region, dst_region)
        assert len(call.args) == 2, (f"Expected 2 args for dma_copy, got {len(call.args)}")

        # Each argument should be a tl.tileop.region Call
        for i, arg in enumerate(call.args):
            assert isinstance(arg,
                              tir.Call), (f"dma_copy arg[{i}] should be a Call, got {type(arg)}")
            assert hasattr(arg.op, "name") and arg.op.name == "tl.tileop.region", (
                f"dma_copy arg[{i}] should be tl.tileop.region, got {arg.op.name}")

        # --- Normalize regions back to buffer metadata ---
        src_buf, src_ranges, src_mask = normalize_region(call.args[0])
        dst_buf, dst_ranges, dst_mask = normalize_region(call.args[1])

        # access_mask: 1=read for src, 2=write for dst
        assert src_mask == 1, f"Source access_mask should be 1 (read), got {src_mask}"
        assert dst_mask == 2, f"Destination access_mask should be 2 (write), got {dst_mask}"

        # Buffer dtype
        assert src_buf.dtype == "float16", f"Source dtype should be float16, got {src_buf.dtype}"
        assert dst_buf.dtype == "float16", f"Dest dtype should be float16, got {dst_buf.dtype}"

        # Buffer shapes: both stay 2D (no flattening on SUNMMIO)
        assert len(src_buf.shape) == 2, f"Source buffer should be 2D, got {len(src_buf.shape)}D"
        assert int(src_buf.shape[0]) == M, f"Source shape[0] should be {M}, got {src_buf.shape[0]}"
        assert int(src_buf.shape[1]) == N, f"Source shape[1] should be {N}, got {src_buf.shape[1]}"

        assert len(dst_buf.shape) == 2, f"Dest buffer should be 2D, got {len(dst_buf.shape)}D"
        assert int(dst_buf.shape[0]) == block_M, (
            f"Dest shape[0] should be {block_M}, got {dst_buf.shape[0]}")
        assert int(dst_buf.shape[1]) == block_N, (
            f"Dest shape[1] should be {block_N}, got {dst_buf.shape[1]}")

        # Buffer scope
        src_scope = src_buf.scope()
        assert src_scope == "" or src_scope == "global", (
            f"Source buffer should be in global scope, got '{src_scope}'")
        dst_scope = dst_buf.scope()
        assert "shared" in dst_scope, (
            f"Destination buffer should be in shared scope, got '{dst_scope}'")

        # Region extents match block dimensions
        assert len(src_ranges) == 2
        src_extent_0 = int(src_ranges[0].extent)
        src_extent_1 = int(src_ranges[1].extent)
        assert src_extent_0 == block_M, (
            f"Source extent[0] should be {block_M}, got {src_extent_0}")
        assert src_extent_1 == block_N, (
            f"Source extent[1] should be {block_N}, got {src_extent_1}")

        assert len(dst_ranges) == 2
        dst_extent_0 = int(dst_ranges[0].extent)
        dst_extent_1 = int(dst_ranges[1].extent)
        assert dst_extent_0 == block_M, (f"Dest extent[0] should be {block_M}, got {dst_extent_0}")
        assert dst_extent_1 == block_N, (f"Dest extent[1] should be {block_N}, got {dst_extent_1}")


GEMM_TEST_CASES = [
    # (M, N, K, block_M, block_N, block_K)
    (128, 128, 128, 32, 32, 32),
    (128, 128, 128, 64, 64, 64),
]


@pytest.mark.parametrize("M, N, K, block_M, block_N, block_K", GEMM_TEST_CASES)
def test_tilelang_dma_copy_layout_query(M, N, K, block_M, block_N, block_K):
    """Verify that after LayoutInference, the layout_map annotation is populated
    for shared buffers, and that a downstream pass can look up layouts by buffer.

    NOTE: This test only checks layout annotations after LayoutInference.
    LowerTileOp is not called here because gemm does not yet have a SUNMMIO
    lowering path.  The dma_copy lowering for copy ops is verified separately
    in test_tilelang_dma_copy above.
    """
    env.TILELANG_USE_GEMM_V1 = 0
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = gemm_copy_kernel(M, N, K, block_M, block_N, block_K)

    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tl.transform.LayoutInference()(mod)

        # After LayoutInference but before LowerTileOp, the layout_map
        # annotation is present on the Block node.
        visitor = _DmaCopyVisitor()
        visitor.visit_stmt(mod["main"].body)
        assert len(visitor.layout_map) > 0, ("Expected layout_map annotation after LayoutInference")

        # The shared buffers should have layout entries
        assert "A_shared" in visitor.layout_map, (
            f"Expected A_shared in layout_map, got keys: {list(visitor.layout_map.keys())}")
        layout_a = visitor.layout_map["A_shared"]
        input_shape_a = layout_a.input_size
        assert len(input_shape_a) == 2
        assert int(input_shape_a[0]) == block_M
        assert int(input_shape_a[1]) == block_K

        assert "B_shared" in visitor.layout_map
        layout_b = visitor.layout_map["B_shared"]
        input_shape_b = layout_b.input_size
        assert len(input_shape_b) == 2
        assert int(input_shape_b[0]) == block_K
        assert int(input_shape_b[1]) == block_N


if __name__ == "__main__":
    tilelang.testing.main()
