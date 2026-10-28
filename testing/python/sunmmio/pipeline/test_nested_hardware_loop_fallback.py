"""Fallback coverage for unsupported nested hardware loops in pipeline planning."""

import pytest
import tilelang as tl
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.phase import should_force_let_inline
from tilelang.layout import make_zz_layout
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from tvm import tir


def _nested_copy_kernel():
    placement = T.placement.full_shard(0, 1)

    @T.prim_func
    def main(
        A: T.MeshTensor((128, 1024), placement, T.float32, layout=make_zz_layout((128, 1024))),
        C: T.MeshTensor((128, 512), placement, T.float32, layout=make_zz_layout((128, 512))),
    ):
        with T.Kernel(T.mesh_ncores()):
            tile = T.alloc_shared((32, 32), T.float32, scope="shared.rsram")
            for ni in T.Pipelined(4, num_stages=2):
                for ki in T.serial(2):
                    T.copy(A[0:32, (ni * 2 + ki) * 32 : (ni * 2 + ki + 1) * 32], tile)
                T.copy(tile, C[0:32, ni * 32 : (ni + 1) * 32])

    return main


def _vector_loop_kernel():
    placement = T.placement.full_shard(0, 1)

    @T.prim_func
    def main(
        A: T.MeshTensor((128, 128), placement, T.float32, layout=make_zz_layout((128, 128))),
        C: T.MeshTensor((128, 512), placement, T.float32, layout=make_zz_layout((128, 512))),
    ):
        with T.Kernel(T.mesh_ncores()):
            tile = T.alloc_shared((32, 32), T.float32, scope="shared.rsram")
            T.copy(A[0:32, 0:32], tile)
            for ni in T.Pipelined(4, num_stages=2):
                for i, j in T.Tiles((32, 32), parallel=True):
                    tile[i, j] = tile[i, j] + T.float32(1)
                T.copy(tile, C[0:32, ni * 32 : (ni + 1) * 32])

    return main


def _lower_before_planning(func):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with target:
        mod = tir.transform.BindTarget(target)(tvm.IRModule.from_expr(func))
        mod = tl.transform.ResolveSunmmioMeshSymbols()(mod)
        if should_force_let_inline():
            mod = tl.transform.LetInline()(mod)
        for transform in (
            tl.transform.LegalizeNegativeIndex,
            tl.transform.InjectAssumes,
            tl.transform.Simplify,
            tl.transform.InferSramScope,
            tl.transform.LegalizeSunmmioDataPath,
            tl.transform.SunmmioLayoutInference,
            tl.transform.LegalizeSunmmioGemm,
            tl.transform.LowerTileOp,
            tl.transform.LegalizeTilesLoop,
            tl.transform.TilesLoop,
            tl.transform.LegalizeVectorizedLoop,
            tl.transform.LegalizeSafeMemoryAccess,
            tl.transform.LowerAccessPtr,
            tl.transform.Simplify,
            tl.transform.HoistNonRestrictParams,
            tl.transform.HoistBlockAnnotationsToFuncAttrs,
            tl.transform.ResolveSunmmioUnit,
            tl.transform.IfStmtBinding,
        ):
            mod = transform()(mod)
    return mod


def _planner(mode):
    if mode == "greedy":
        return tl.transform.SunmmioPipelinePlanning()
    return tl.transform.SunmmioPipelinePlanningILP(debug=False)


@pytest.mark.parametrize("mode", ["greedy", "ilp"])
def test_nested_dma_loop_falls_back_without_vector_costing(mode):
    planned = _planner(mode)(_lower_before_planning(_nested_copy_kernel()))
    script = planned.script()

    assert '"tl.sunmmio.pipeline.applied": T.bool(False)' in script
    assert '"tl.sunmmio.pipeline.fallback_reason": "unsupported_nested_hardware_loop"' in script
    assert "T.dma_copy" in script
    assert "for ki in range(2)" in script


@pytest.mark.parametrize("mode", ["greedy", "ilp"])
def test_pure_vector_tile_loop_remains_supported(mode):
    planned = _planner(mode)(_lower_before_planning(_vector_loop_kernel()))
    script = planned.script()
    assert '"tl.sunmmio.pipeline.applied": T.bool(True)' in script
    assert "unsupported_nested_hardware_loop" not in script
