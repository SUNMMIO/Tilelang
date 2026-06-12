import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.layout import make_row_major, make_zz_layout
from tilelang.utils.target import SUNMMIO_TARGET_DESC


DTYPE = "float16"


@pytest.fixture(autouse=True)
def disable_tilelang_cache():
    cache_was_enabled = tilelang.env.is_cache_enabled()
    tilelang.env.disable_cache()
    try:
        yield
    finally:
        if cache_was_enabled:
            tilelang.env.enable_cache()
        else:
            tilelang.env.disable_cache()


def _make_copy_kernel(copy_case):
    if copy_case.startswith("zz_"):
        shape = (96, 96)
        layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))
    else:
        shape = (128, 128)
        layout = make_row_major(shape)

    @T.prim_func
    def kernel(A: T.Tensor(shape, DTYPE)):
        with T.Kernel(1):
            A_shared = T.alloc_shared(shape, DTYPE)
            T.annotate_layout({A: layout, A_shared: layout})

            if copy_case == "row_major_aligned_grid_tile":
                T.copy(A[0:64, 0:64], A_shared[64:128, 64:128])
            elif copy_case == "row_major_min_not_region_extent_aligned":
                T.copy(A[32:96, 0:64], A_shared[0:64, 0:64])
            elif copy_case == "row_major_extent_not_buffer_shape_factor":
                T.copy(A[0:96, 0:96], A_shared[0:96, 0:96])
            elif copy_case == "row_major_1d_tile_view":
                T.copy(A[0, 64:128], A_shared[1, 64:128])
            elif copy_case == "zz_block_equal":
                T.copy(A[0:32, 0:32], A_shared[32:64, 32:64])
            elif copy_case == "zz_block_inner_split":
                T.copy(A[0:16, 0:16], A_shared[16:32, 16:32])
            elif copy_case == "zz_whole_dim":
                T.copy(A[0:96, 0:96], A_shared[0:96, 0:96])
            elif copy_case == "zz_extent_not_coalesced_compatible":
                T.copy(A[0:48, 0:48], A_shared[0:48, 0:48])
            else:
                raise ValueError(f"unknown copy case: {copy_case}")

    return tvm.IRModule({"main": kernel})


def _run_validate_copy_tile_view(copy_case):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = _make_copy_kernel(copy_case)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        return tilelang.transform.ValidateCopyTileView()(mod)


def _make_inferred_layout_copy_kernel(copy_case):
    shape = (96, 96) if copy_case.startswith("zz_") else (128, 128)

    @T.prim_func
    def kernel(A: T.Tensor(shape, DTYPE)):
        with T.Kernel(1):
            A_shared = T.alloc_shared(shape, DTYPE)

            if copy_case == "row_major_aligned_grid_tile":
                T.copy(A[0:64, 0:64], A_shared[64:128, 64:128])
            elif copy_case == "row_major_min_not_region_extent_aligned":
                T.copy(A[32:96, 0:64], A_shared[0:64, 0:64])
            elif copy_case == "zz_extent_not_coalesced_compatible":
                T.copy(A[0:48, 0:48], A_shared[0:48, 0:48])
            else:
                raise ValueError(f"unknown copy case: {copy_case}")

    return tvm.IRModule({"main": kernel})


def _run_kernel_pipeline_to_validate_copy_tile_view(copy_case):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = _make_inferred_layout_copy_kernel(copy_case)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.InferSramScope()(mod)
        mod = tilelang.transform.LegalizeSunmmioDataPath()(mod)
        mod = tilelang.transform.SunmmioLayoutInference()(mod)
        return tilelang.transform.ValidateCopyTileView()(mod)


@pytest.mark.parametrize(
    "copy_case",
    [
        "row_major_aligned_grid_tile",
        "row_major_1d_tile_view",
        "zz_block_equal",
        "zz_block_inner_split",
        "zz_whole_dim",
    ],
)
def test_sunmmio_validate_copy_tile_view_accepts_legal_regions(copy_case):
    _run_validate_copy_tile_view(copy_case)


@pytest.mark.parametrize(
    "copy_case, error_msg",
    [
        (
            "row_major_min_not_region_extent_aligned",
            "must align to region extent",
        ),
        (
            "row_major_extent_not_buffer_shape_factor",
            "must divide buffer shape",
        ),
        (
            "zz_extent_not_coalesced_compatible",
            "must be compatible with coalesced extent",
        ),
    ],
)
def test_sunmmio_validate_copy_tile_view_rejects_illegal_regions(copy_case, error_msg):
    with pytest.raises(tvm.error.InternalError, match=error_msg):
        _run_validate_copy_tile_view(copy_case)


def test_sunmmio_validate_copy_tile_view_accepts_kernel_after_layout_inference():
    _run_kernel_pipeline_to_validate_copy_tile_view("row_major_aligned_grid_tile")


@pytest.mark.parametrize(
    "copy_case, error_msg",
    [
        (
            "row_major_min_not_region_extent_aligned",
            "must align to region extent",
        ),
        (
            "zz_extent_not_coalesced_compatible",
            "must be compatible with coalesced extent",
        ),
    ],
)
def test_sunmmio_validate_copy_tile_view_rejects_kernel_after_layout_inference(copy_case, error_msg):
    with pytest.raises(tvm.error.InternalError, match=error_msg):
        _run_kernel_pipeline_to_validate_copy_tile_view(copy_case)


if __name__ == "__main__":
    tilelang.testing.main()
