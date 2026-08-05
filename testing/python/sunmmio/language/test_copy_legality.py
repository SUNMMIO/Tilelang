import re
import warnings

import pytest

import tilelang.utils.target as _target_utils
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.layout import make_zz_layout
from testing.python.sunmmio.common.compile_pipeline import target


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
    if copy_case == "region_to_region_src_oob_clips_dst":
        C_128x128x32_shared = T.alloc_shared((128, 128, 32), DTYPE)
        return T.copy(A_128x128x128_global[127:129, 0:32, 0:32], C_128x128x32_shared[0:2, 0:32, 0:32])
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


@target("Sunmmio")
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
        with T.Kernel():
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
    return tvm.IRModule({"main": _make_copy_kernel(copy_case)}).script()


@target("Sunmmio")
def _make_let_bound_mesh_copy_kernel():
    global_shape = (256, 256)
    shard_policy = T.MeshShardingPolicy(y=0, x=1)
    tensor_layout = make_zz_layout(global_shape, axes=[0, 1], block_shape=(32, 32))

    @T.prim_func
    def kernel(
        A: T.MeshTensor(global_shape, shard_policy, T.bfloat16, layout=tensor_layout),  # type: ignore
    ):
        with T.Kernel():
            local_m, local_n = A.local_shape
            A_shared = T.alloc_shared((local_m, local_n), T.bfloat16)
            T.copy(A, A_shared)

    return kernel


@target("Sunmmio")
def _make_mismatched_let_bound_mesh_copy_kernel():
    global_shape = (256, 256)
    shard_policy = T.MeshShardingPolicy(y=0, x=1)
    tensor_layout = make_zz_layout(global_shape, axes=[0, 1], block_shape=(32, 32))

    @T.prim_func
    def kernel(
        A: T.MeshTensor(global_shape, shard_policy, T.bfloat16, layout=tensor_layout),  # type: ignore
    ):
        with T.Kernel():
            local_m, local_n = A.local_shape
            A_shared = T.alloc_shared((local_m, local_n + 1), T.bfloat16)
            T.copy(A, A_shared)

    return kernel


@target("Sunmmio")
def _make_let_bound_singleton_region_copy_kernel():
    @T.prim_func
    def kernel(A_1x8_global: T.Tensor((1, 8), DTYPE)):
        with T.Kernel():
            A_8_shared = T.alloc_shared((8,), DTYPE)
            with T.LetStmt(1) as singleton:
                T.copy(A_1x8_global[0:singleton, 0:8], A_8_shared)

    return kernel


@target("Sunmmio")
def _make_let_bound_oversized_region_copy_kernel():
    @T.prim_func
    def kernel(A_8_global: T.Tensor((8,), DTYPE)):
        with T.Kernel():
            A_4_shared = T.alloc_shared((4,), DTYPE)
            with T.LetStmt(8) as extent:
                T.copy(A_8_global[0:extent], A_4_shared[0:4])

    return kernel


@target("Sunmmio")
def _make_symbolic_unknown_region_copy_kernel():
    n = T.dynamic("n")

    @T.prim_func
    def kernel(src: T.Tensor((n,), DTYPE), dst: T.Tensor((4,), DTYPE)):
        with T.Kernel():
            T.copy(src[0:n], dst[0:4])

    return kernel


@target("Sunmmio")
def _make_memory_backed_let_extent_copy_kernel():
    @T.prim_func
    def kernel(
        shape: T.Tensor((1,), T.int32),
        src: T.Tensor((8,), DTYPE),
        dst: T.Tensor((8,), DTYPE),
    ):
        with T.Kernel(), T.LetStmt(shape[0]) as extent:
            shape[0] = 4
            T.copy(src[0:extent], dst[0:extent])

    return kernel


@target("Sunmmio")
def _make_memory_backed_let_allocation_kernel():
    @T.prim_func
    def kernel(shape: T.Tensor((1,), T.int32)):
        with T.Kernel(), T.LetStmt(shape[0]) as extent:
            T.alloc_shared((extent,), DTYPE)

    return kernel


def _assert_region_extents(script, buffer_name, access_mask, extents):
    extent_pattern = r",\s*".join(str(extent) for extent in extents)
    name_pattern = r"[A-Za-z_]\w*" if buffer_name is None else re.escape(buffer_name)
    pattern = rf"T\.region\({name_pattern}\[[^\]]+\],\s*{access_mask},\s*{extent_pattern}\)"
    label = buffer_name or f"any buffer with access mask {access_mask}"
    assert re.search(pattern, script), f"missing region for {label} with extents {extents}:\n{script}"


@pytest.fixture
def _strict_region_validation():
    previous = _target_utils.ENABLE_SUNMMIO_REGION_VALIDATION
    _target_utils.set_sunmmio_region_validation(True)
    try:
        yield
    finally:
        _target_utils.set_sunmmio_region_validation(previous)


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
    "region_to_region_src_oob_clips_dst",
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
def test_sunmmio_copy_frontend_accepts_valid_cases(copy_case, _strict_region_validation):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _build_script(copy_case)


@pytest.mark.parametrize("copy_case", FRONTEND_INVALID_CASES)
def test_sunmmio_copy_frontend_rejects_invalid_cases(copy_case, _strict_region_validation):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            _build_script(copy_case)


def test_sunmmio_copy_frontend_shrinks_larger_explicit_destination(_strict_region_validation):
    script = _build_script("buffer_to_region_larger_dst")

    _assert_region_extents(script, "A_128x128x128_global", 1, [128, 128, 128])
    _assert_region_extents(script, None, 2, [128, 128, 128])


def test_sunmmio_copy_frontend_shrinks_dst_when_src_is_smaller(_strict_region_validation):
    script = _build_script("region_to_region_src_lt_dst")

    _assert_region_extents(script, "A_128x128x128_global", 1, [16, 16, 16])
    _assert_region_extents(script, None, 2, [16, 16, 16])


def test_sunmmio_copy_compact_path_skips_strict_region_validation():
    previous = _target_utils.ENABLE_SUNMMIO_REGION_VALIDATION
    _target_utils.set_sunmmio_region_validation(False)
    try:
        script = _build_script("buffer_to_region_smaller_dst")

        _assert_region_extents(script, "A_128x128x128_global", 1, [128, 128, 128])
        _assert_region_extents(script, None, 2, [128, 128, 32])
    finally:
        _target_utils.set_sunmmio_region_validation(previous)


@pytest.mark.parametrize("strict", [False, True], ids=["compact", "strict"])
def test_sunmmio_copy_frontend_accepts_let_bound_mesh_shape(strict):
    previous = _target_utils.ENABLE_SUNMMIO_REGION_VALIDATION
    _target_utils.set_sunmmio_region_validation(strict)
    try:
        script = tvm.IRModule({"main": _make_let_bound_mesh_copy_kernel()}).script()
    finally:
        _target_utils.set_sunmmio_region_validation(previous)

    assert script.strip()
    assert "T.alloc_buffer((local_m, local_n)" not in script


@pytest.mark.parametrize("strict", [False, True], ids=["compact", "strict"])
def test_sunmmio_copy_frontend_rejects_mismatched_let_bound_mesh_shape(strict):
    previous = _target_utils.ENABLE_SUNMMIO_REGION_VALIDATION
    _target_utils.set_sunmmio_region_validation(strict)
    try:
        with pytest.raises(ValueError):
            _make_mismatched_let_bound_mesh_copy_kernel()
    finally:
        _target_utils.set_sunmmio_region_validation(previous)


def test_sunmmio_copy_strict_squeezes_let_bound_singleton_extent(_strict_region_validation):
    script = tvm.IRModule({"main": _make_let_bound_singleton_region_copy_kernel()}).script()

    assert "T.min" not in script
    _assert_region_extents(script, "A_1x8_global", 1, [1, 8])
    _assert_region_extents(script, None, 2, [8])


def test_sunmmio_copy_strict_rejects_let_bound_oversized_extent(_strict_region_validation):
    with pytest.raises(ValueError, match="src extent is larger than dst extent"):
        _make_let_bound_oversized_region_copy_kernel()


def test_sunmmio_copy_strict_warns_and_preserves_symbolic_unknown_regions(_strict_region_validation):
    with pytest.warns(UserWarning, match="cannot prove.*less than or equal"):
        script = tvm.IRModule({"main": _make_symbolic_unknown_region_copy_kernel()}).script()

    assert "T.region(src[0], 1, n)" in script
    assert "T.region(dst[0], 2, 4)" in script


def test_sunmmio_copy_strict_preserves_memory_backed_let_snapshot(_strict_region_validation):
    script = tvm.IRModule({"main": _make_memory_backed_let_extent_copy_kernel()}).script()

    match = re.search(r"(?P<extent>\w+): T\.int32 = shape\[0\]", script)
    assert match, f"missing captured let value:\n{script}"
    extent = match.group("extent")
    assert f"T.region(src[0], 1, {extent})" in script
    assert f"T.region(dst[0], 2, {extent})" in script
    assert "T.region(src[0], 1, shape[0])" not in script
    assert "T.region(dst[0], 2, shape[0])" not in script


def test_sunmmio_allocation_rejects_memory_backed_let_shape():
    with pytest.raises(ValueError, match="non-invariant let binding"):
        _make_memory_backed_let_allocation_kernel()


def test_sunmmio_copy_frontend_clips_src_before_shrinking_dst(_strict_region_validation):
    with pytest.warns(UserWarning, match="will be clipped"):
        script = _build_script("region_to_region_src_oob_clips_dst")

    _assert_region_extents(script, "A_128x128x128_global", 1, [1, 32, 32])
    _assert_region_extents(script, None, 2, [1, 32, 32])


def test_sunmmio_copy_frontend_clips_explicit_dst_before_inferring_load_src(_strict_region_validation):
    with pytest.warns(UserWarning, match="will be clipped"):
        script = _build_script("load_to_region_dst_oob")

    _assert_region_extents(script, "A_128x128x128_global", 1, [128, 128, 32])
    _assert_region_extents(script, None, 2, [128, 128, 32])


@target("Sunmmio")
def _make_dynamic_explicit_region_kernel():
    @T.prim_func
    def kernel(A_256x128_global: T.Tensor((256, 128), DTYPE)):
        with T.Kernel():
            A_shared = T.alloc_shared((64, 128), DTYPE)
            for bx in T.serial(4):
                T.copy(A_256x128_global[bx * 32 : bx * 32 + 64, 0:128], A_shared)

    return kernel


def test_sunmmio_copy_frontend_keeps_dynamic_explicit_region_extent():
    script = tvm.IRModule({"main": _make_dynamic_explicit_region_kernel()}).script()

    assert "T.min" not in script
    _assert_region_extents(script, "A_256x128_global", 1, [64, 128])
    _assert_region_extents(script, None, 2, [64, 128])


def test_sunmmio_copy_frontend_supports_rank_lower_bufferload_to_buffer():
    script = _build_script("load_to_buffer_rank_lower_full_dst")

    _assert_region_extents(script, "A_128x128x128_global", 1, [1, 128, 128])
    _assert_region_extents(script, None, 2, [128, 128])


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
