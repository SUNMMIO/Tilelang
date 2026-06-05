import re
import warnings

import pytest
import torch

_torch_cuda_is_available = torch.cuda.is_available
torch.cuda.is_available = lambda: False
try:
    import tilelang.language as T
    import tilelang.testing
    from tilelang import tvm
    from tilelang.utils.target import determine_target
finally:
    torch.cuda.is_available = _torch_cuda_is_available


DTYPE = "float16"


def _emit_copy_case(
    copy_case,
    A_128x128x128_global,
    A_128x128x1_global,
    A_128x1x128_global,
    A_1x128x128_global,
    A_128x128_global,
    A_1x128x1x64_global,
    Q_1x128x1x64_global,
):
    if copy_case == "buffer_to_buffer_same_shape":
        B_128x128x128_shared = T.alloc_shared((128, 128, 128), DTYPE)
        return T.copy(A_128x128x128_global, B_128x128x128_shared)
    if copy_case == "buffer_to_buffer_same_rank_last_mismatch":
        B_128x128x128_shared = T.alloc_shared((128, 128, 128), DTYPE)
        return T.copy(A_128x128x1_global, B_128x128x128_shared)
    if copy_case == "buffer_to_buffer_same_rank_middle_mismatch":
        B_128x128x128_shared = T.alloc_shared((128, 128, 128), DTYPE)
        return T.copy(A_128x1x128_global, B_128x128x128_shared)
    if copy_case == "buffer_to_buffer_same_rank_leading_mismatch":
        B_128x128x128_shared = T.alloc_shared((128, 128, 128), DTYPE)
        return T.copy(A_1x128x128_global, B_128x128x128_shared)
    if copy_case == "buffer_to_buffer_same_rank_no_dim_reorder":
        B_128x1x128_shared = T.alloc_shared((128, 1, 128), DTYPE)
        return T.copy(A_1x128x128_global, B_128x1x128_shared)
    if copy_case == "buffer_to_buffer_src_leading_one_rank2":
        B_128x128_shared = T.alloc_shared((128, 128), DTYPE)
        return T.copy(A_1x128x128_global, B_128x128_shared)
    if copy_case == "buffer_to_buffer_dst_leading_one_rank3":
        B_1x128x128_shared = T.alloc_shared((1, 128, 128), DTYPE)
        return T.copy(A_128x128_global, B_1x128x128_shared)
    if copy_case == "buffer_to_buffer_src_middle_one_rank2":
        B_128x128_shared = T.alloc_shared((128, 128), DTYPE)
        return T.copy(A_128x1x128_global, B_128x128_shared)
    if copy_case == "buffer_to_buffer_rank4_middle_singleton":
        B_128x64_shared = T.alloc_shared((128, 64), DTYPE)
        return T.copy(A_1x128x1x64_global, B_128x64_shared)

    if copy_case == "buffer_to_region_equal":
        C_256x256x256_shared = T.alloc_shared((256, 256, 256), DTYPE)
        return T.copy(A_128x128x128_global, C_256x256x256_shared[0:128, 0:128, 0:128])
    if copy_case == "buffer_to_region_larger_dst":
        C_256x256x256_shared = T.alloc_shared((256, 256, 256), DTYPE)
        return T.copy(A_128x128x128_global, C_256x256x256_shared[0:256, 0:256, 0:256])
    if copy_case == "buffer_to_region_smaller_dst":
        C_128x128x32_shared = T.alloc_shared((128, 128, 32), DTYPE)
        return T.copy(A_128x128x128_global, C_128x128x32_shared[0:128, 0:128, 0:32])
    if copy_case == "buffer_to_region_explicit_dst_oob":
        C_128x128x32_shared = T.alloc_shared((128, 128, 32), DTYPE)
        return T.copy(A_128x128x128_global, C_128x128x32_shared[0:128, 0:128, 0:128])

    if copy_case == "buffer_to_load_point_dst":
        C_256x256x256_shared = T.alloc_shared((256, 256, 256), DTYPE)
        return T.copy(A_128x128x128_global, C_256x256x256_shared[0, 0, 0])
    if copy_case == "buffer_to_load_point_dst_oob":
        C_128x128x32_shared = T.alloc_shared((128, 128, 32), DTYPE)
        return T.copy(A_128x128x128_global, C_128x128x32_shared[0, 0, 0])
    if copy_case == "buffer_to_load_point_dst_unaligned":
        C_256x256x256_shared = T.alloc_shared((256, 256, 256), DTYPE)
        return T.copy(A_128x128x128_global, C_256x256x256_shared[1, 0, 0])

    if copy_case == "region_to_buffer_explicit_src":
        C_128x128x32_shared = T.alloc_shared((128, 128, 32), DTYPE)
        return T.copy(A_128x128x128_global[0:32, 0:32, 0:32], C_128x128x32_shared)
    if copy_case == "region_to_buffer_small_tile":
        C_128x128x32_shared = T.alloc_shared((128, 128, 32), DTYPE)
        return T.copy(A_128x128x128_global[0:16, 0:16, 0:16], C_128x128x32_shared)
    if copy_case == "region_to_buffer_dst_too_small":
        C_128x128x32_shared = T.alloc_shared((128, 128, 32), DTYPE)
        return T.copy(A_128x128x128_global[0:64, 0:64, 0:64], C_128x128x32_shared)
    if copy_case == "region_to_buffer_src_unaligned":
        C_128x128x32_shared = T.alloc_shared((128, 128, 32), DTYPE)
        return T.copy(A_128x128x128_global[1:33, 0:32, 0:32], C_128x128x32_shared)

    if copy_case == "region_to_region_equal":
        C_128x128x32_shared = T.alloc_shared((128, 128, 32), DTYPE)
        return T.copy(A_128x128x128_global[0:32, 0:32, 0:32], C_128x128x32_shared[0:32, 0:32, 0:32])
    if copy_case == "region_to_region_small_tile":
        C_128x128x32_shared = T.alloc_shared((128, 128, 32), DTYPE)
        return T.copy(A_128x128x128_global[0:16, 0:16, 0:16], C_128x128x32_shared[0:16, 0:16, 0:16])
    if copy_case == "region_to_region_dst_oob":
        C_128x128x32_shared = T.alloc_shared((128, 128, 32), DTYPE)
        return T.copy(A_128x128x128_global[0:64, 0:64, 0:64], C_128x128x32_shared[0:64, 0:64, 0:64])
    if copy_case == "region_to_region_src_gt_dst":
        C_128x128x32_shared = T.alloc_shared((128, 128, 32), DTYPE)
        return T.copy(A_128x128x128_global[0:32, 0:32, 0:32], C_128x128x32_shared[0:16, 0:16, 0:16])
    if copy_case == "region_to_region_src_lt_dst":
        C_128x128x32_shared = T.alloc_shared((128, 128, 32), DTYPE)
        return T.copy(A_128x128x128_global[0:16, 0:16, 0:16], C_128x128x32_shared[0:32, 0:32, 0:32])
    if copy_case == "region_to_region_extent_mismatch_dst_oob":
        C_128x128x32_shared = T.alloc_shared((128, 128, 32), DTYPE)
        return T.copy(A_128x128x128_global[0:32, 0:32, 0:32], C_128x128x32_shared[0:64, 0:64, 0:64])
    if copy_case == "region_to_region_1d_tile_view":
        C_128x128x32_shared = T.alloc_shared((128, 128, 32), DTYPE)
        return T.copy(A_128x128x128_global[0, 0, 0:32], C_128x128x32_shared[0:1, 0:1, 0:32])
    if copy_case == "region_to_region_no_dim_reorder":
        C_128x128x32_shared = T.alloc_shared((128, 128, 32), DTYPE)
        return T.copy(A_128x128x128_global[0:32, 0, 0:32], C_128x128x32_shared[0:1, 0:32, 0:32])
    if copy_case == "region_to_region_rank_suffix_compatible":
        E_128x128_shared = T.alloc_shared((128, 128), DTYPE)
        return T.copy(E_128x128_shared[:32, :32], A_128x128x128_global[1, :32, :32])
    if copy_case == "region_to_region_rank_squeeze_middle_singleton":
        Q_64x64_shared = T.alloc_shared((64, 64), DTYPE)
        return T.copy(Q_1x128x1x64_global[0, 0:64, 0, 0:64], Q_64x64_shared)
    if copy_case == "region_to_region_rank_mismatch_non1_leading":
        E_128x128_shared = T.alloc_shared((128, 128), DTYPE)
        return T.copy(A_128x128x128_global[0:32, 0:32, 0:32], E_128x128_shared[:32, :32])

    if copy_case == "region_to_load_point_dst":
        C_128x128x32_shared = T.alloc_shared((128, 128, 32), DTYPE)
        return T.copy(A_128x128x128_global[0:32, 0:32, 0:32], C_128x128x32_shared[0, 0, 0])
    if copy_case == "region_to_load_point_dst_oob":
        C_128x128x32_shared = T.alloc_shared((128, 128, 32), DTYPE)
        return T.copy(A_128x128x128_global[0:32, 0:32, 0:32], C_128x128x32_shared[32, 32, 32])
    if copy_case == "region_to_load_point_dst_unaligned":
        C_256x256x256_shared = T.alloc_shared((256, 256, 256), DTYPE)
        return T.copy(A_128x128x128_global[0:32, 0:32, 0:32], C_256x256x256_shared[1, 0, 0])

    if copy_case == "load_to_buffer_full_dst":
        C_128x128x32_shared = T.alloc_shared((128, 128, 32), DTYPE)
        return T.copy(A_128x128x128_global[0, 0, 0], C_128x128x32_shared)
    if copy_case == "load_to_buffer_rank_lower_full_dst":
        E_128x128_shared = T.alloc_shared((128, 128), DTYPE)
        return T.copy(A_128x128x128_global[0, 0, 0], E_128x128_shared)
    if copy_case == "load_to_buffer_clipped_unaligned":
        C_128x128x32_shared = T.alloc_shared((128, 128, 32), DTYPE)
        return T.copy(A_128x128x128_global[1, 2, 3], C_128x128x32_shared)
    if copy_case == "load_to_buffer_clipped_legal":
        D_128x128x64_shared = T.alloc_shared((128, 128, 64), DTYPE)
        return T.copy(A_128x128x128_global[0, 0, 96], D_128x128x64_shared)

    if copy_case == "load_to_region_explicit_dst":
        C_128x128x32_shared = T.alloc_shared((128, 128, 32), DTYPE)
        return T.copy(A_128x128x128_global[0, 0, 0], C_128x128x32_shared[0:128, 0:128, 0:32])
    if copy_case == "load_to_region_clipped_unaligned":
        C_256x256x256_shared = T.alloc_shared((256, 256, 256), DTYPE)
        return T.copy(A_128x128x128_global[1, 2, 3], C_256x256x256_shared[0:128, 0:128, 0:32])
    if copy_case == "load_to_region_clipped_legal":
        C_256x256x256_shared = T.alloc_shared((256, 256, 256), DTYPE)
        return T.copy(A_128x128x128_global[0, 0, 96], C_256x256x256_shared[0:128, 0:128, 0:64])
    if copy_case == "load_to_region_dst_oob":
        C_128x128x32_shared = T.alloc_shared((128, 128, 32), DTYPE)
        return T.copy(A_128x128x128_global[0, 0, 0], C_128x128x32_shared[0:128, 0:128, 0:64])
    if copy_case == "load_to_load_scalar":
        C_128x128x32_shared = T.alloc_shared((128, 128, 32), DTYPE)
        return T.copy(A_128x128x128_global[1, 2, 3], C_128x128x32_shared[0, 0, 0])

    raise AssertionError(f"unknown copy case: {copy_case}")


def _make_copy_kernel(copy_case):
    @T.prim_func
    def kernel(
        A_128x128x128_global: T.Tensor((128, 128, 128), DTYPE),
        A_128x128x1_global: T.Tensor((128, 128, 1), DTYPE),
        A_128x1x128_global: T.Tensor((128, 1, 128), DTYPE),
        A_1x128x128_global: T.Tensor((1, 128, 128), DTYPE),
        A_128x128_global: T.Tensor((128, 128), DTYPE),
        A_1x128x1x64_global: T.Tensor((1, 128, 1, 64), DTYPE),
        Q_1x128x1x64_global: T.Tensor((1, 128, 1, 64), DTYPE),
    ):
        with T.Kernel(1):
            _emit_copy_case(
                copy_case,
                A_128x128x128_global,
                A_128x128x1_global,
                A_128x1x128_global,
                A_1x128x128_global,
                A_128x128_global,
                A_1x128x1x64_global,
                Q_1x128x1x64_global,
            )

    return kernel


def _build_script(copy_case):
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        return tvm.IRModule({"main": _make_copy_kernel(copy_case)}).script()


def _assert_region_extents(script, buffer_name, access_mask, extents):
    extent_pattern = r",\s*".join(str(extent) for extent in extents)
    name_pattern = r"[A-Za-z_]\w*" if buffer_name is None else re.escape(buffer_name)
    pattern = rf"T\.region\({name_pattern}\[[^\]]+\],\s*{access_mask},\s*{extent_pattern}\)"
    label = buffer_name or f"any buffer with access mask {access_mask}"
    assert re.search(pattern, script), f"missing region for {label} with extents {extents}:\n{script}"


FRONTEND_VALID_CASES = [
    "buffer_to_buffer_same_shape",
    "buffer_to_buffer_src_leading_one_rank2",
    "buffer_to_buffer_dst_leading_one_rank3",
    "buffer_to_region_equal",
    "buffer_to_region_larger_dst",
    "buffer_to_load_point_dst",
    "buffer_to_load_point_dst_unaligned",
    "region_to_buffer_explicit_src",
    "region_to_buffer_small_tile",
    "region_to_buffer_src_unaligned",
    "region_to_region_equal",
    "region_to_region_small_tile",
    "region_to_region_src_lt_dst",
    "region_to_region_extent_mismatch_dst_oob",
    "region_to_region_1d_tile_view",
    "region_to_region_no_dim_reorder",
    "region_to_region_rank_suffix_compatible",
    "region_to_region_rank_squeeze_middle_singleton",
    "region_to_load_point_dst",
    "region_to_load_point_dst_unaligned",
    "load_to_buffer_full_dst",
    "load_to_buffer_rank_lower_full_dst",
    "load_to_buffer_clipped_unaligned",
    "load_to_buffer_clipped_legal",
    "load_to_region_explicit_dst",
    "load_to_region_clipped_unaligned",
    "load_to_region_clipped_legal",
    "load_to_region_dst_oob",
    "load_to_load_scalar",
]


FRONTEND_INVALID_CASES = [
    "buffer_to_buffer_same_rank_last_mismatch",
    "buffer_to_buffer_same_rank_middle_mismatch",
    "buffer_to_buffer_same_rank_leading_mismatch",
    "buffer_to_buffer_same_rank_no_dim_reorder",
    "buffer_to_buffer_src_middle_one_rank2",
    "buffer_to_buffer_rank4_middle_singleton",
    "buffer_to_region_smaller_dst",
    "buffer_to_region_explicit_dst_oob",
    "buffer_to_load_point_dst_oob",
    "region_to_buffer_dst_too_small",
    "region_to_region_dst_oob",
    "region_to_region_src_gt_dst",
    "region_to_region_rank_mismatch_non1_leading",
    "region_to_load_point_dst_oob",
]


@pytest.mark.parametrize("copy_case", FRONTEND_VALID_CASES)
def test_sunmmio_copy_frontend_accepts_valid_cases(copy_case):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _build_script(copy_case)


@pytest.mark.parametrize("copy_case", FRONTEND_INVALID_CASES)
def test_sunmmio_copy_frontend_rejects_invalid_cases(copy_case):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            _build_script(copy_case)


def test_sunmmio_copy_frontend_shrinks_larger_explicit_destination():
    script = _build_script("buffer_to_region_larger_dst")

    _assert_region_extents(script, "A_128x128x128_global", 1, [128, 128, 128])
    _assert_region_extents(script, None, 2, [128, 128, 128])


def test_sunmmio_copy_frontend_shrinks_dst_when_src_is_smaller():
    script = _build_script("region_to_region_src_lt_dst")

    _assert_region_extents(script, "A_128x128x128_global", 1, [16, 16, 16])
    _assert_region_extents(script, None, 2, [16, 16, 16])


def test_sunmmio_copy_frontend_clips_explicit_dst_before_inferring_load_src():
    with pytest.warns(UserWarning, match="will be clipped"):
        script = _build_script("load_to_region_dst_oob")

    _assert_region_extents(script, "A_128x128x128_global", 1, [128, 128, 32])
    _assert_region_extents(script, None, 2, [128, 128, 32])


def test_sunmmio_copy_frontend_supports_rank_lower_bufferload_to_buffer():
    script = _build_script("load_to_buffer_rank_lower_full_dst")

    _assert_region_extents(script, "A_128x128x128_global", 1, [1, 128, 128])
    _assert_region_extents(script, None, 2, [128, 128])


def test_sunmmio_copy_frontend_keeps_dynamic_bufferload_extents_static():
    @T.prim_func
    def kernel(A: T.Tensor((128, 128), DTYPE)):
        with T.Kernel(2, 2) as (bx, by):
            A_shared = T.alloc_shared((64, 32), DTYPE)
            T.copy(A[by * 64, bx * 32], A_shared)

    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        script = tvm.IRModule({"main": kernel}).script()

    _assert_region_extents(script, "A", 1, [64, 32])
    _assert_region_extents(script, None, 2, [64, 32])
    assert "T.min" not in script


def test_sunmmio_copy_frontend_supports_squeezed_middle_singleton_dims():
    script = _build_script("region_to_region_rank_squeeze_middle_singleton")

    _assert_region_extents(script, "Q_1x128x1x64_global", 1, [1, 64, 1, 64])
    _assert_region_extents(script, None, 2, [64, 64])


def test_sunmmio_copy_frontend_keeps_bufferload_to_bufferload_as_store():
    script = _build_script("load_to_load_scalar")

    assert "T.copy(" not in script
    assert re.search(r"\w+\[0, 0, 0\] = A_128x128x128_global\[1, 2, 3\]", script)


if __name__ == "__main__":
    tilelang.testing.main()
