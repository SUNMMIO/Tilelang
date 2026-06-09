import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.tileview import make_tileview
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
    @T.prim_func
    def kernel(A: T.Tensor((128, 128), DTYPE)):
        with T.Kernel(1):
            A_shared = T.alloc_shared((128, 128), DTYPE)
            T.annotate_tileview({A_shared: make_tileview(A_shared, (32, 32), (-2, -1))})

            if copy_case == "aligned":
                T.copy(A[32:96, 32:96], A_shared[32:96, 32:96])
            elif copy_case == "min_not_tile_aligned":
                T.copy(A[1:65, 32:96], A_shared[32:96, 32:96])
            elif copy_case == "min_not_shape_factor":
                T.copy(A[32:64, 32:64], A_shared[96:128, 96:128])
            elif copy_case == "extent_not_shape_factor":
                T.copy(A[32:128, 32:128], A_shared[32:128, 32:128])
            else:
                raise ValueError(f"unknown copy case: {copy_case}")

    return tvm.IRModule({"main": kernel})


def _run_validate_copy_tile_view(copy_case):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = _make_copy_kernel(copy_case)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        return tilelang.transform.ValidateCopyTileView()(mod)


def test_sunmmio_validate_copy_tile_view_accepts_aligned_region():
    _run_validate_copy_tile_view("aligned")


@pytest.mark.parametrize(
    "copy_case, error_msg",
    [
        ("min_not_tile_aligned", "must align to TileView tile size"),
        ("min_not_shape_factor", "must be zero or a factor of buffer shape"),
        ("extent_not_shape_factor", "must be a factor of buffer shape"),
    ],
)
def test_sunmmio_validate_copy_tile_view_rejects_misaligned_regions(copy_case, error_msg):
    with pytest.raises(tvm.error.InternalError, match=error_msg):
        _run_validate_copy_tile_view(copy_case)


if __name__ == "__main__":
    tilelang.testing.main()
