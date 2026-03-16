"""Regression tests for Sunmmio shared.rsram -> shared.rsram copy lowering."""

import pytest
import tilelang
import tilelang as tl
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target
from tvm import tir

tilelang.env.disable_cache()


def apply_sunmmio_passes(mod, target):
    """Apply the Sunmmio lowering pipeline through tile-loop legalization."""

    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.InferSramScope()(mod)
    mod = tilelang.transform.LayoutReducer()(mod)
    mod = tilelang.transform.LayoutInference()(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    print(mod)
    mod = tl.transform.LegalizeTilesLoop()(mod)
    mod = tl.transform.TilesLoop()(mod)
    print(mod)
    return mod


def rsram_copy(block_M=64, block_N=64, dtype="float16", dst_dtype=None):
    """Build a kernel with a single shared.rsram -> shared.rsram T.copy."""

    if dst_dtype is None:
        dst_dtype = dtype

    @T.prim_func
    def main():
        with T.Kernel(1, 1, threads=128) as (_bx, _by):
            A_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")
            B_shared = T.alloc_shared((block_M, block_N), dst_dtype, scope="shared.rsram")
            T.copy(A_shared, B_shared)

    return tvm.IRModule({"main": main})


def rsram_copy_region(block_M=64, block_N=64, dtype="float16"):
    """Build a kernel with a sliced shared.rsram -> shared.rsram T.copy."""

    @T.prim_func
    def main():
        with T.Kernel(1, 1, threads=128) as (_bx, _by):
            A_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")
            B_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")
            T.copy(A_shared[8:24, 16:48], B_shared[0:16, 0:32])

    return tvm.IRModule({"main": main})


def extract_dma_copy_lines(mod):
    return [line.lstrip() for line in mod.script().split("\n") if "T.dma_copy" in line]


def collect_loops_with_attr(func, attr_name):
    loops = []

    def visit(stmt):
        if isinstance(stmt, tir.For):
            ann = stmt.annotations
            if ann is not None and ann.get(attr_name, 0) == 1:
                loops.append(stmt)

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return loops


def collect_buffer_stores(func, buffer_name):
    stores = []

    def visit(stmt):
        if isinstance(stmt, tir.BufferStore) and stmt.buffer.name == buffer_name:
            stores.append(stmt)

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return stores


def expr_has_cast(expr, dst_dtype, src_dtype):
    found = False

    def visit(node):
        nonlocal found
        if isinstance(node, tir.Cast) and str(node.dtype) == dst_dtype and str(node.value.dtype) == src_dtype:
            found = True

    tvm.tir.stmt_functor.post_order_visit(expr, visit)
    return found


def expr_has_if_then_else(expr):
    found = False

    def visit(node):
        nonlocal found
        if isinstance(node, tir.Call) and hasattr(node.op, "name") and node.op.name == "tir.if_then_else":
            found = True

    tvm.tir.stmt_functor.post_order_visit(expr, visit)
    return found


def test_tilelang_rsram_copy_lowers_to_tile_loop():
    target = determine_target("Sunmmio", return_object=True)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(rsram_copy(), target)

    func = mod["main"]
    stores = collect_buffer_stores(func, "B_shared")

    assert extract_dma_copy_lines(mod) == []
    assert collect_loops_with_attr(func, "tile.scope_entry")
    assert collect_loops_with_attr(func, "tile.interior")
    assert stores
    assert not any(expr_has_if_then_else(store.value) for store in stores)


def test_tilelang_rsram_copy_inserts_cast_for_dtype_mismatch():
    target = determine_target("Sunmmio", return_object=True)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(
            rsram_copy(dtype="float16", dst_dtype="float32"),
            target,
        )

    func = mod["main"]
    stores = collect_buffer_stores(func, "B_shared")

    assert extract_dma_copy_lines(mod) == []
    assert collect_loops_with_attr(func, "tile.scope_entry")
    assert collect_loops_with_attr(func, "tile.interior")
    assert stores
    assert not any(expr_has_if_then_else(store.value) for store in stores)
    assert any(
        isinstance(store.value, tir.Cast) and str(store.value.dtype) == "float32" and str(store.value.value.dtype) == "float16"
        for store in stores
    )


def test_tilelang_rsram_region_copy_is_rejected_for_now():
    target = determine_target("Sunmmio", return_object=True)

    with (
        pytest.raises(
            tvm.error.InternalError,
            match="Unsupported copy from shared.rsram to shared.rsram of Sunmmio target.",
        ),
        tvm.target.Target(target),
    ):
        mod = rsram_copy_region()
        apply_sunmmio_passes(mod, target)


def extract_block_attr_lines(mod):
    """Extract block attributes from TIR script"""
    return [line.lstrip() for line in mod.script().split("\n") if "T.block_attr" in line]


def test_tilelang_rsram_copy_layout_inference():
    target = determine_target("Sunmmio", return_object=True)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(rsram_copy(64, 64), target)

    # Check layout map
    texts = extract_block_attr_lines(mod)
    for text in texts:
        assert '"layout_map"' in text and 'A_shared: metadata["tl.Layout"]' in text and 'B_shared: metadata["tl.Layout"]' in text
