"""Tests for the Sunmmio DRAM<->RSRAM layout-transform lowering.

A copy between a row-major DRAM buffer and a ZZ RSRAM buffer is split, in
CopyNode::LowerSunmmioDmaCopy, into a plain tl.dma_copy through a staging
RSRAM buffer plus a tl.sunmmio_layout_transform that re-blocks the data.  A
copy whose src/dst layouts already match stays a plain tl.dma_copy.

The kernels declare their DRAM tensors the way examples/gemm/
sunmmio_example_gemm.py does — T.MeshTensor on the Sunmmio device mesh,
carrying an explicit layout.
"""

import tilelang
import tilelang as tl
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target
from tilelang.layout import make_row_major, make_zz_layout
from tilelang.language.mesh_tensor import MeshReplicationType
from tilelang.carver.arch import driver
from tvm import tir
from tvm.tir import Block
from tvm.tir.stmt_functor import post_order_visit

tilelang.env.disable_cache()

M, N = 128, 128
DTYPE = T.float16


def apply_passes_through_lower_tile_op(mod, target):
    """Apply the Sunmmio pass pipeline up to and including LowerTileOp."""
    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tl.transform.AddWrapperForSingleBufStore()(mod)
    mod = tl.transform.LegalizeNegativeIndex()(mod)
    mod = tl.transform.InjectAssumes()(mod)
    mod = tl.transform.Simplify()(mod)
    mod = tl.transform.InferSramScope()(mod)
    mod = tl.transform.LegalizeSunmmioDataPath()(mod)
    mod = tl.transform.LayoutReducer()(mod)
    mod = tl.transform.SunmmioLayoutInference()(mod)
    mod = tl.transform.LowerTileOp()(mod)
    return mod


def collect_call_names(func):
    """Return the list of intrinsic call op-names in the function body."""
    names = []

    def visit(node):
        if isinstance(node, tir.Call) and hasattr(node.op, "name"):
            names.append(node.op.name)

    post_order_visit(func.body, visit)
    return names


def collect_alloc_buffers(func):
    """Return all buffers declared in Block alloc_buffers."""
    bufs = []

    def visit(node):
        if isinstance(node, Block):
            bufs.extend(node.alloc_buffers)

    post_order_visit(func.body, visit)
    return bufs


def staging_buffers(func):
    return [b for b in collect_alloc_buffers(func) if b.name.endswith("_layout_stage")]


def extract_layout_map(func):
    """Return the layout_map from the function's block annotations."""
    layout_map = {}

    def visit(node):
        nonlocal layout_map
        if isinstance(node, Block) and "layout_map" in node.annotations:
            layout_map = node.annotations["layout_map"]

    post_order_visit(func.body, visit)
    return layout_map


# ---------------------------------------------------------------------------
# Kernels — declared like examples/gemm/sunmmio_example_gemm.py
# ---------------------------------------------------------------------------


def _dram(shape, layout):
    """A DRAM tensor on the Sunmmio device mesh, carrying an explicit layout —
    the T.MeshTensor declaration style of sunmmio_example_gemm.py.

    A fully-replicated policy keeps each core's buffer at the declared shape:
    these are single-core layout tests (unlike the sharded GEMM example), so
    sharding must not perturb the shapes the assertions rely on.
    """
    mesh = driver.get_sunmmio_device_mesh_config()
    policy = T.MeshShardingPolicy(replicate=MeshReplicationType.ALL)
    return T.MeshTensor(shape, policy, mesh, DTYPE, layout=layout)


def mismatched_copy_kernel():
    """Row-major DRAM <-> ZZ RSRAM (default), both directions — combos 1 & 2."""

    @T.prim_func
    def main(
        A: _dram((M, N), make_row_major((M, N))),
        B: _dram((M, N), make_row_major((M, N))),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), DTYPE, scope="shared.rsram")
            T.copy(A, A_shared)  # DRAM(row-major) -> RSRAM(ZZ default)
            T.copy(A_shared, B)  # RSRAM(ZZ) -> DRAM(row-major)

    return tvm.IRModule({"main": main})


def zz_dram_kernel():
    """ZZ DRAM <-> row-major RSRAM, both directions — combos 3 & 4."""

    @T.prim_func
    def main(
        A: _dram((M, N), make_zz_layout((M, N), [0, 1], (32, 32))),
        B: _dram((M, N), make_zz_layout((M, N), [0, 1], (32, 32))),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), DTYPE, scope="shared.rsram")
            T.annotate_layout({A_shared: make_row_major((M, N))})
            T.copy(A, A_shared)  # DRAM(ZZ) -> RSRAM(row-major)
            T.copy(A_shared, B)  # RSRAM(row-major) -> DRAM(ZZ)

    return tvm.IRModule({"main": main})


def matched_copy_kernel():
    """Row-major DRAM and row-major RSRAM (annotated) — layouts match, so the
    copy stays a plain dma_copy."""

    @T.prim_func
    def main(A: _dram((M, N), make_row_major((M, N)))):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), DTYPE, scope="shared.rsram")
            T.annotate_layout({A_shared: make_row_major((M, N))})
            T.copy(A, A_shared)  # DRAM(row-major) -> RSRAM(row-major), matched

    return tvm.IRModule({"main": main})


def zz_dram_reduce_kernel():
    """End-to-end: ZZ DRAM (3x256x256) -> RSRAM, axis-2 reduction -> row-major
    RSRAM (3x256), copied back to ZZ DRAM (3x256).

    Out_sh needs no annotation: the reduction is over axis 2 — a blocked dim
    of the ZZ layout — so ReduceOp::InferLayout assigns Out_sh a row-major
    layout, making the final store a ZZ <- row-major transform.  The matched
    ZZ->ZZ load stays a plain dma_copy.
    """
    b, h, w = 3, 256, 256

    @T.prim_func
    def main(
        A: _dram((b, h, w), make_zz_layout((b, h, w), [1, 2], (32, 32))),
        Out: _dram((b, h), make_zz_layout((b, h), [0, 1], (32, 32))),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_sh = T.alloc_shared((b, h, w), DTYPE, scope="shared.rsram")
            Out_sh = T.alloc_shared((b, h), DTYPE, scope="shared.rsram")
            T.copy(A, A_sh)  # ZZ DRAM -> ZZ RSRAM (matched)
            T.reduce_sum(A_sh, Out_sh, dim=2)  # axis-2 reduce -> row-major Out_sh
            T.copy(Out_sh, Out)  # row-major RSRAM -> ZZ DRAM (transform)

    return tvm.IRModule({"main": main})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mismatched_copy_splits_into_dma_and_transform():
    """A row-major DRAM <-> ZZ RSRAM copy is split via an RSRAM staging buffer."""
    target = determine_target("Sunmmio", return_object=True)
    mod = mismatched_copy_kernel()
    with tvm.target.Target(target):
        mod = apply_passes_through_lower_tile_op(mod, target)
        func = list(mod.functions.values())[0]

        names = collect_call_names(func)
        # One dma_copy + one transform per copy, both directions.
        assert names.count("tl.dma_copy") == 2, names
        assert names.count("tl.sunmmio_layout_transform") == 2, names

        # One RSRAM staging buffer per split copy.
        stages = staging_buffers(func)
        assert len(stages) == 2, [b.name for b in stages]
        for b in stages:
            assert b.scope() == "shared.rsram", b.scope()


def test_zz_dram_splits_with_zz_staging():
    """ZZ DRAM <-> row-major RSRAM (combos 3 & 4): splits, and the staging
    buffer is registered with the ZZ layout so its dma_copy leg is matched."""
    target = determine_target("Sunmmio", return_object=True)
    mod = zz_dram_kernel()
    with tvm.target.Target(target):
        mod = apply_passes_through_lower_tile_op(mod, target)
        func = list(mod.functions.values())[0]

        names = collect_call_names(func)
        assert names.count("tl.dma_copy") == 2, names
        assert names.count("tl.sunmmio_layout_transform") == 2, names

        stages = staging_buffers(func)
        assert len(stages) == 2, [b.name for b in stages]

        # The staging buffers must be registered in layout_map (the
        # RegisterLayout callback) — that is what makes the dma_copy leg a
        # layout-matched ZZ->ZZ transfer.
        layout_map = extract_layout_map(func)
        registered = {b.name for b in layout_map if b.name.endswith("_layout_stage")}
        assert len(registered) == 2, registered


def test_matched_layout_stays_plain_dma_copy():
    """A copy whose src/dst layouts match keeps a bare dma_copy — no transform."""
    target = determine_target("Sunmmio", return_object=True)
    mod = matched_copy_kernel()
    with tvm.target.Target(target):
        mod = apply_passes_through_lower_tile_op(mod, target)
        func = list(mod.functions.values())[0]

        names = collect_call_names(func)
        assert names.count("tl.dma_copy") == 1, names
        assert "tl.sunmmio_layout_transform" not in names, names
        assert staging_buffers(func) == [], [b.name for b in staging_buffers(func)]


def test_sync_injected_between_dma_and_transform():
    """InjectSunmmioSync inserts a wait_token for the staging-buffer dependency."""
    target = determine_target("Sunmmio", return_object=True)
    mod = mismatched_copy_kernel()
    with tvm.target.Target(target):
        mod = apply_passes_through_lower_tile_op(mod, target)
        # InjectSunmmioSync only processes global device functions; that attr is
        # set by the host/device split later in the real pipeline, so set it
        # here to exercise the pass directly on the layout-transform op.
        func = mod["main"].with_attr("tir.is_global_func", True)
        mod = tvm.IRModule({"main": func})
        mod = tl.transform.InjectSunmmioSync()(mod)
        func = list(mod.functions.values())[0]

        names = collect_call_names(func)
        # The transform reads the staging buffer the dma_copy wrote -> RAW.
        assert "tl.wait_token" in names, names
        # The async op itself is tagged with a sync token.
        assert "tl.sync_token_id" in names, names


def test_end_to_end_zz_dram_reduce_roundtrip():
    """ZZ DRAM -> RSRAM -> axis-2 reduce -> row-major RSRAM -> ZZ DRAM.

    The matched ZZ->ZZ load stays a plain dma_copy; only the row-major -> ZZ
    store is split into a sunmmio_layout_transform + dma_copy.
    """
    target = determine_target("Sunmmio", return_object=True)
    mod = zz_dram_reduce_kernel()
    with tvm.target.Target(target):
        mod = apply_passes_through_lower_tile_op(mod, target)
        func = list(mod.functions.values())[0]
        print(func)

        names = collect_call_names(func)
        # ZZ->ZZ load (plain) + the transform's dma leg = 2 dma_copy.
        assert names.count("tl.dma_copy") == 2, names
        # Only the row-major -> ZZ store needs a transform.
        assert names.count("tl.sunmmio_layout_transform") == 1, names
        # The axis-2 reduction lowered to the vector-core in-tile reduce.
        assert names.count("tl.vector_core_in_tile_reduce") == 1, names

        # One ZZ staging buffer (mirrors the ZZ DRAM output), registered in
        # layout_map by the RegisterLayout callback.
        stages = staging_buffers(func)
        assert len(stages) == 1, [b.name for b in stages]
        assert stages[0].scope() == "shared.rsram", stages[0].scope()
        layout_map = extract_layout_map(func)
        registered = {b.name for b in layout_map if b.name.endswith("_layout_stage")}
        assert registered == {stages[0].name}, registered


if __name__ == "__main__":
    tilelang.testing.main()
