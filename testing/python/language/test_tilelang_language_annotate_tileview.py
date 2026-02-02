"""Tests for TileView and annotate_tileview functionality."""
import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm as tvm
from tilelang.tileview import TileView, make_tileview
from tvm import tir

# =============================================================================
# Test cases for TileView creation
# Format: (buffer_shape, tile_shape, index_map, expected_tiled_shape, expected_vector_lanes)
# =============================================================================
TILEVIEW_TEST_CASES = [
    # Basic 2D cases
    pytest.param([64, 128], [16, 32], [-2, -1], [4, 4, 16, 32],
                 512,
                 id="2d_basic_64x128_tile_16x32"),
    pytest.param([128, 256], [32, 64], [-2, -1], [4, 4, 32, 64], 2048, id="2d_128x256_tile_32x64"),
    pytest.param([32, 32], [8, 8], [-2, -1], [4, 4, 8, 8], 64, id="2d_square_32x32_tile_8x8"),
    pytest.param([64, 128], [16, 32], [0, 1], [4, 4, 16, 32], 512, id="2d_positive_index_map"),
    # 3D cases - tiling last 2 dims
    pytest.param([8, 64, 128], [16, 32], [-2, -1], [8, 4, 4, 16, 32],
                 512,
                 id="3d_8x64x128_tile_last2"),
    pytest.param([4, 32, 64], [8, 16], [-2, -1], [4, 4, 4, 8, 16], 128, id="3d_4x32x64_tile_last2"),
    # 4D cases - tiling last 2 dims
    pytest.param([2, 4, 64, 128], [16, 32], [-2, -1], [2, 4, 4, 4, 16, 32],
                 512,
                 id="4d_2x4x64x128_tile_last2"),
    # Single tile (buffer_shape == tile_shape for tiled dims)
    pytest.param([16, 32], [16, 32], [-2, -1], [1, 1, 16, 32], 512, id="2d_single_tile"),
    # Asymmetric tiling
    pytest.param([64, 64], [8, 32], [-2, -1], [8, 2, 8, 32], 256, id="2d_asymmetric_tile"),
]


@pytest.mark.parametrize(
    "buffer_shape, tile_shape, index_map, expected_tiled_shape, expected_vector_lanes",
    TILEVIEW_TEST_CASES)
def test_tileview_creation(buffer_shape, tile_shape, index_map, expected_tiled_shape,
                           expected_vector_lanes):
    """Test TileView creation with various shapes and configurations."""
    tv = TileView(buffer_shape, tile_shape, index_map)

    # Verify buffer_shape
    assert len(tv.buffer_shape) == len(buffer_shape)
    for i, expected in enumerate(buffer_shape):
        assert int(tv.buffer_shape[i]) == expected

    # Verify tile_shape
    assert len(tv.tile_shape) == len(tile_shape)
    for i, expected in enumerate(tile_shape):
        assert int(tv.tile_shape[i]) == expected

    # Verify index_map
    assert len(tv.index_map) == len(index_map)
    for i, expected in enumerate(index_map):
        assert int(tv.index_map[i]) == expected

    # Verify tiled_buffer_shape
    assert len(tv.tiled_buffer_shape) == len(expected_tiled_shape)
    for i, expected in enumerate(expected_tiled_shape):
        assert int(tv.tiled_buffer_shape[i]) == expected

    # Verify vector_lanes
    assert int(tv.vector_lanes) == expected_vector_lanes


# =============================================================================
# Test cases for TileView equality
# Format: (tv1_args, tv2_args, should_be_equal)
# =============================================================================
TILEVIEW_EQUALITY_CASES = [
    pytest.param(([64, 128], [16, 32], [-2, -1]), ([64, 128], [16, 32], [-2, -1]),
                 True,
                 id="equal_same_params"),
    pytest.param(([64, 128], [16, 32], [-2, -1]), ([64, 128], [8, 32], [-2, -1]),
                 False,
                 id="different_tile_shape"),
    pytest.param(([64, 128], [16, 32], [-2, -1]), ([128, 128], [16, 32], [-2, -1]),
                 False,
                 id="different_buffer_shape"),
    pytest.param(([64, 128], [16, 32], [0, 1]), ([64, 128], [16, 32], [-2, -1]),
                 False,
                 id="different_index_map"),
]


@pytest.mark.parametrize("tv1_args, tv2_args, should_be_equal", TILEVIEW_EQUALITY_CASES)
def test_tileview_equality(tv1_args, tv2_args, should_be_equal):
    """Test TileView equality comparison."""
    tv1 = TileView(*tv1_args)
    tv2 = TileView(*tv2_args)

    assert tv1.is_equal(tv2) == should_be_equal
    assert tv2.is_equal(tv1) == should_be_equal  # Symmetry


# =============================================================================
# Test cases for non-divisible dimensions (should raise error)
# Format: (buffer_shape, tile_shape, index_map)
# =============================================================================
TILEVIEW_ERROR_CASES = [
    pytest.param([64, 128], [15, 32], [-2, -1], id="dim0_not_divisible"),
    pytest.param([64, 128], [16, 30], [-2, -1], id="dim1_not_divisible"),
    pytest.param([64, 128], [17, 33], [-2, -1], id="both_dims_not_divisible"),
    pytest.param([100, 100], [32, 32], [-2, -1], id="100_not_divisible_by_32"),
]


@pytest.mark.parametrize("buffer_shape, tile_shape, index_map", TILEVIEW_ERROR_CASES)
def test_tileview_non_divisible_error(buffer_shape, tile_shape, index_map):
    """Test that TileView raises error for non-divisible dimensions."""
    with pytest.raises(ValueError):
        TileView(buffer_shape, tile_shape, index_map)


def test_tileview_repr():
    """Test TileView string representation."""
    tv = TileView([64, 128], [16, 32], [-2, -1])
    repr_str = repr(tv)

    assert "TileView" in repr_str
    assert "buffer_shape" in repr_str
    assert "tile_shape" in repr_str
    assert "index_map" in repr_str
    assert "tiled_shape" in repr_str


# =============================================================================
# Helper function to extract tileviews from IR
# =============================================================================
def _extract_tileviews(func: tir.PrimFunc) -> dict:
    """Extract tileview_map from a PrimFunc's block annotations.

    Recursively traverses the IR to find blocks with tileview_map annotations.
    """
    collected_tileviews = {}

    def visit(stmt):
        """Recursively visit statements to find tileview_map annotations."""
        if stmt is None:
            return

        if isinstance(stmt, tir.Block):
            if "tileview_map" in stmt.annotations:
                tileview_map = stmt.annotations["tileview_map"]
                for key, tileview in tileview_map.items():
                    collected_tileviews[key.name] = tileview
            visit(stmt.body)
        elif isinstance(stmt, tir.BlockRealize):
            visit(stmt.block)
        elif isinstance(stmt, tir.SeqStmt):
            for s in stmt.seq:
                visit(s)
        elif isinstance(stmt, (tir.For, tir.AttrStmt, tir.LetStmt)):
            visit(stmt.body)
        elif isinstance(stmt, tir.IfThenElse):
            visit(stmt.then_case)
            if stmt.else_case:
                visit(stmt.else_case)

    visit(func.body)
    return collected_tileviews


# =============================================================================
# Test cases for annotate_tileview in kernel context
# Format: (buffer_shape, tile_shape, index_map, expected_tiled_shape)
# =============================================================================
ANNOTATE_TILEVIEW_CASES = [
    pytest.param((64, 128), [16, 32], [-2, -1], [4, 4, 16, 32], id="2d_basic"),
    pytest.param((128, 256), [32, 64], [-2, -1], [4, 4, 32, 64], id="2d_larger"),
    pytest.param((32, 32), [8, 8], [-2, -1], [4, 4, 8, 8], id="2d_square"),
    pytest.param((64, 64), [8, 32], [-2, -1], [8, 2, 8, 32], id="2d_asymmetric"),
]


@pytest.mark.parametrize("buffer_shape, tile_shape, index_map, expected_tiled_shape",
                         ANNOTATE_TILEVIEW_CASES)
def test_annotate_tileview_with_tuple(buffer_shape, tile_shape, index_map, expected_tiled_shape):
    """Test annotate_tileview with tuple shorthand in kernel context."""
    M, N = buffer_shape

    @T.prim_func
    def kernel(A: T.Tensor((M, N), "float16"),):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared((M, N), "float16")
            T.annotate_tileview({A_shared: (tile_shape, index_map)})
            for i, j in T.Parallel(M, N):
                A_shared[i, j] = A[i, j]

    # Extract and verify tileview
    tileviews = _extract_tileviews(kernel)

    assert "A_shared" in tileviews, "TileView should be attached to A_shared"
    tv = tileviews["A_shared"]

    # Verify buffer_shape
    assert int(tv.buffer_shape[0]) == M
    assert int(tv.buffer_shape[1]) == N

    # Verify tile_shape
    for i, expected in enumerate(tile_shape):
        assert int(tv.tile_shape[i]) == expected

    # Verify index_map
    for i, expected in enumerate(index_map):
        assert int(tv.index_map[i]) == expected

    # Verify tiled_buffer_shape
    assert len(tv.tiled_buffer_shape) == len(expected_tiled_shape)
    for i, expected in enumerate(expected_tiled_shape):
        assert int(tv.tiled_buffer_shape[i]) == expected


@pytest.mark.parametrize("buffer_shape, tile_shape, index_map, expected_tiled_shape",
                         ANNOTATE_TILEVIEW_CASES)
def test_annotate_tileview_with_make_tileview(buffer_shape, tile_shape, index_map,
                                              expected_tiled_shape):
    """Test annotate_tileview with make_tileview in kernel context."""
    M, N = buffer_shape

    @T.prim_func
    def kernel(A: T.Tensor((M, N), "float16"),):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared((M, N), "float16")
            T.annotate_tileview({A_shared: make_tileview(A_shared, tile_shape, index_map)})
            for i, j in T.Parallel(M, N):
                A_shared[i, j] = A[i, j]

    # Extract and verify tileview
    tileviews = _extract_tileviews(kernel)

    assert "A_shared" in tileviews
    tv = tileviews["A_shared"]
    assert isinstance(tv, TileView)

    # Verify tiled_buffer_shape
    for i, expected in enumerate(expected_tiled_shape):
        assert int(tv.tiled_buffer_shape[i]) == expected


# =============================================================================
# Test cases for multiple buffer annotations
# Format: (buffers_config) where each entry is (name, buffer_shape, tile_shape, index_map, expected_tiled_shape)
# =============================================================================
MULTI_BUFFER_CASES = [
    pytest.param([
        ("A_shared", (64, 128), [16, 32], [-2, -1], [4, 4, 16, 32]),
        ("B_shared", (128, 64), [32, 16], [-2, -1], [4, 4, 32, 16]),
    ],
                 id="two_buffers_different_shapes"),
    pytest.param([
        ("A_shared", (64, 64), [8, 8], [-2, -1], [8, 8, 8, 8]),
        ("B_shared", (64, 64), [16, 16], [-2, -1], [4, 4, 16, 16]),
    ],
                 id="two_buffers_same_shape_different_tiles"),
    pytest.param([
        ("A_shared", (32, 32), [8, 8], [-2, -1], [4, 4, 8, 8]),
        ("B_shared", (32, 64), [8, 16], [-2, -1], [4, 4, 8, 16]),
        ("C_shared", (64, 32), [16, 8], [-2, -1], [4, 4, 16, 8]),
    ],
                 id="three_buffers"),
]


@pytest.mark.parametrize("buffers_config", MULTI_BUFFER_CASES)
def test_annotate_tileview_multiple_buffers(buffers_config):
    """Test annotate_tileview with multiple buffers."""
    # Use the first two buffers for the kernel (simplification)
    buf_a = buffers_config[0]
    buf_b = buffers_config[1] if len(buffers_config) > 1 else buf_a

    M_a, N_a = buf_a[1]
    M_b, N_b = buf_b[1]

    @T.prim_func
    def kernel(
            A: T.Tensor((M_a, N_a), "float16"),
            B: T.Tensor((M_b, N_b), "float16"),
    ):
        with T.Kernel(1, 1) as (bx, by):
            A_shared = T.alloc_shared((M_a, N_a), "float16")
            B_shared = T.alloc_shared((M_b, N_b), "float16")

            T.annotate_tileview({
                A_shared: (buf_a[2], buf_a[3]),
                B_shared: (buf_b[2], buf_b[3]),
            })

            for i, j in T.Parallel(M_a, N_a):
                A_shared[i, j] = A[i, j]
            for i, j in T.Parallel(M_b, N_b):
                B_shared[i, j] = B[i, j]

    # Extract and verify tileviews
    tileviews = _extract_tileviews(kernel)

    # Verify A_shared
    assert "A_shared" in tileviews
    tv_a = tileviews["A_shared"]
    for i, expected in enumerate(buf_a[4]):
        assert int(tv_a.tiled_buffer_shape[i]) == expected

    # Verify B_shared
    assert "B_shared" in tileviews
    tv_b = tileviews["B_shared"]
    for i, expected in enumerate(buf_b[4]):
        assert int(tv_b.tiled_buffer_shape[i]) == expected


def test_annotate_tileview_invalid_input():
    """Test annotate_tileview raises error for invalid input."""
    with pytest.raises(ValueError):

        @T.prim_func
        def kernel(A: T.Tensor((64, 128), "float16"),):
            with T.Kernel(1, 1) as (bx, by):
                A_shared = T.alloc_shared((64, 128), "float16")
                # Invalid: passing a single list instead of tuple
                T.annotate_tileview({A_shared: [16, 32]})


if __name__ == "__main__":
    tilelang.testing.main()
