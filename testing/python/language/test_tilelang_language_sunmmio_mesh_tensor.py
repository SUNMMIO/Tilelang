import math
import pytest

import tilelang.language as T
from tilelang.language.mesh_tensor import MeshTensorProxy, MeshShardingPolicy, MeshReplicationType
from tilelang.layout import make_zz_layout, make_zn_layout
from tvm import tir


# ─── _get_sharded_shape tests (unchanged — pure shape math) ─────────────


@pytest.mark.parametrize(
    "shape, nrows, ncols",
    [
        ((100, 200, 300), 2, 2),
        ((64, 128), 4, 1),
        ((10, 20, 30, 40), 8, 8),
    ],
)
def test_get_sharded_shape_replicate_all(shape, nrows, ncols):
    policy = MeshShardingPolicy(replicate=MeshReplicationType.ALL)
    expected_shape = shape
    assert MeshTensorProxy._get_sharded_shape(shape, policy, nrows, ncols) == expected_shape


@pytest.mark.parametrize(
    "shape, cross_mesh_dim, nrows, ncols",
    [
        ((100, 200, 300), 1, 2, 2),
        ((100, 203, 300), 1, 2, 2),
        ((128, 256, 512), 0, 4, 4),
        ((128, 256, 512), 2, 2, 8),
    ],
)
def test_get_sharded_shape_cross_mesh_dim(shape, cross_mesh_dim, nrows, ncols):
    policy = MeshShardingPolicy(cross_mesh_dim=cross_mesh_dim)
    total_cores = nrows * ncols

    expected_shape_list = list(shape)
    expected_shape_list[cross_mesh_dim] = math.ceil(shape[cross_mesh_dim] / total_cores)
    expected_shape = tuple(expected_shape_list)

    assert MeshTensorProxy._get_sharded_shape(shape, policy, nrows, ncols) == expected_shape


@pytest.mark.parametrize(
    "shape, cross_mesh_dim, nrows, ncols",
    [
        ((100, 200, 300), 3, 2, 2),
        ((100, 200), 2, 2, 2),
    ],
)
def test_get_sharded_shape_cross_mesh_dim_invalid(shape, cross_mesh_dim, nrows, ncols):
    with pytest.raises(ValueError, match="Invalid cross_mesh_dim"):
        MeshTensorProxy._get_sharded_shape(shape, MeshShardingPolicy(cross_mesh_dim=cross_mesh_dim), nrows, ncols)


@pytest.mark.parametrize(
    "shape, y_dim, nrows, ncols",
    [
        ((100, 200, 300), 0, 4, 4),
        ((103, 200, 300), 0, 4, 4),
        ((128, 256, 512), 2, 2, 8),
    ],
)
def test_get_sharded_shape_replicate_row(shape, y_dim, nrows, ncols):
    policy = MeshShardingPolicy(y=y_dim, replicate=MeshReplicationType.ROW)

    expected_shape_list = list(shape)
    expected_shape_list[y_dim] = math.ceil(shape[y_dim] / nrows)
    expected_shape = tuple(expected_shape_list)

    assert MeshTensorProxy._get_sharded_shape(shape, policy, nrows, ncols) == expected_shape


@pytest.mark.parametrize(
    "shape, policy, nrows, ncols, error_msg",
    [
        (
            (100, 200, 300),
            MeshShardingPolicy(x=1, y=0, replicate=MeshReplicationType.ROW),
            4,
            4,
            "Cannot shard on x-axis when replicating on rows",
        ),
        ((100, 200, 300), MeshShardingPolicy(y=3, replicate=MeshReplicationType.ROW), 4, 4, "Invalid y-split dimension"),
    ],
)
def test_get_sharded_shape_replicate_row_invalid(shape, policy, nrows, ncols, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        MeshTensorProxy._get_sharded_shape(shape, policy, nrows, ncols)


@pytest.mark.parametrize(
    "shape, x_dim, nrows, ncols",
    [
        ((100, 200, 300), 1, 4, 4),
        ((100, 203, 300), 1, 4, 4),
        ((128, 256, 512), 0, 2, 8),
    ],
)
def test_get_sharded_shape_replicate_column(shape, x_dim, nrows, ncols):
    policy = MeshShardingPolicy(x=x_dim, replicate=MeshReplicationType.COLUMN)

    expected_shape_list = list(shape)
    expected_shape_list[x_dim] = math.ceil(shape[x_dim] / ncols)
    expected_shape = tuple(expected_shape_list)

    assert MeshTensorProxy._get_sharded_shape(shape, policy, nrows, ncols) == expected_shape


@pytest.mark.parametrize(
    "shape, policy, nrows, ncols, error_msg",
    [
        (
            (100, 200, 300),
            MeshShardingPolicy(x=1, y=0, replicate=MeshReplicationType.COLUMN),
            4,
            4,
            "Cannot shard on y-axis when replicating on columns",
        ),
        ((100, 200, 300), MeshShardingPolicy(x=3, replicate=MeshReplicationType.COLUMN), 4, 4, "Invalid x-split dimension"),
    ],
)
def test_get_sharded_shape_replicate_column_invalid(shape, policy, nrows, ncols, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        MeshTensorProxy._get_sharded_shape(shape, policy, nrows, ncols)


@pytest.mark.parametrize(
    "shape, x_dim, y_dim, nrows, ncols",
    [
        ((100, 200, 300), 1, 0, 2, 2),
        ((101, 203, 300), 1, 0, 2, 2),
        ((100, 200, 300), 1, None, 2, 2),
        ((100, 200, 300), None, 0, 2, 2),
    ],
)
def test_get_sharded_shape_none_replication(shape, x_dim, y_dim, nrows, ncols):
    policy = MeshShardingPolicy(x=x_dim, y=y_dim, replicate=MeshReplicationType.NONE)

    expected_shape_list = list(shape)
    if y_dim is not None:
        expected_shape_list[y_dim] = math.ceil(shape[y_dim] / nrows)
    if x_dim is not None:
        expected_shape_list[x_dim] = math.ceil(shape[x_dim] / ncols)
    expected_shape = tuple(expected_shape_list)

    assert MeshTensorProxy._get_sharded_shape(shape, policy, nrows, ncols) == expected_shape


@pytest.mark.parametrize(
    "shape, policy, nrows, ncols, error_msg",
    [
        ((100, 200, 300), MeshShardingPolicy(x=3, replicate=MeshReplicationType.NONE), 2, 2, "Invalid x-split dimension"),
        ((100, 200, 300), MeshShardingPolicy(y=3, replicate=MeshReplicationType.NONE), 2, 2, "Invalid y-split dimension"),
    ],
)
def test_get_sharded_shape_none_replication_invalid(shape, policy, nrows, ncols, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        MeshTensorProxy._get_sharded_shape(shape, policy, nrows, ncols)


# ─── MeshTensor __call__ basic tests ─────────────────────────────────────


@pytest.mark.parametrize(
    "shape, device_mesh_config, policy",
    [
        ((100, 200, 300), (2, 2), MeshShardingPolicy(replicate=MeshReplicationType.ALL)),
        ((100, 200, 300), (2, 2), MeshShardingPolicy(cross_mesh_dim=1)),
        ((100, 200, 300), (2, 2), MeshShardingPolicy(y=0, replicate=MeshReplicationType.ROW)),
        ((100, 200, 300), (2, 2), MeshShardingPolicy(x=1, replicate=MeshReplicationType.COLUMN)),
        ((100, 200, 300), (2, 2), MeshShardingPolicy(y=0, x=1, replicate=MeshReplicationType.NONE)),
        ((128, 256), (4, 2), MeshShardingPolicy(y=0, x=1, replicate=MeshReplicationType.NONE)),
        ((128, 256, 512), (2, 4), MeshShardingPolicy(cross_mesh_dim=2)),
    ],
)
def test_call_method(shape, device_mesh_config, policy):
    proxy = MeshTensorProxy()
    nrows, ncols = device_mesh_config

    result = proxy(shape, policy, device_mesh_config)

    expected_shape = MeshTensorProxy._get_sharded_shape(shape, policy, nrows, ncols)
    assert tuple(result.buffer.shape) == expected_shape


# ─── Default row-major layout tests ──────────────────────────────────────


@pytest.mark.parametrize(
    "shape, device_mesh_config, policy",
    [
        ((128, 256), (2, 4), MeshShardingPolicy(y=0, x=1, replicate=MeshReplicationType.NONE)),
        ((100, 200, 300), (2, 2), MeshShardingPolicy(cross_mesh_dim=1)),
        ((128, 256, 512), (2, 4), MeshShardingPolicy(replicate=MeshReplicationType.ALL)),
    ],
)
def test_default_row_major_layout(shape, device_mesh_config, policy):
    proxy = MeshTensorProxy()
    nrows, ncols = device_mesh_config

    tensor_with_meta = proxy(shape, policy, device_mesh_config)
    sharded_shape = MeshTensorProxy._get_sharded_shape(shape, policy, nrows, ncols)

    # Verify the sharded buffer shape
    assert tuple(tensor_with_meta.buffer.shape) == sharded_shape

    # Verify CuteLayout metadata is present
    meta = tensor_with_meta.meta_data
    assert meta["global_shape"] == shape
    assert "global_layout" in meta
    assert "sharded_layout" in meta

    # Global layout should have dim_levels = (1, 1, ...) for row-major
    import tvm_ffi

    _dim_levels = tvm_ffi.get_global_func("tl.CuteLayout_dim_levels")
    global_dl = _dim_levels(meta["global_layout"])
    assert len(global_dl) == len(shape)
    for dl in global_dl:
        assert dl == 1

    sharded_dl = _dim_levels(meta["sharded_layout"])
    assert len(sharded_dl) == len(shape)
    for dl in sharded_dl:
        assert dl == 1


# ─── Layout parameter tests ─────────────────────────────────────────────


class TestLayoutParam:
    """Tests for MeshTensor with explicit layout= parameter."""

    def test_zz_layout_basic(self):
        """ZZ layout is stored and sharded correctly."""
        proxy = MeshTensorProxy()
        shape = (128, 128)
        policy = MeshShardingPolicy(y=0, x=1, replicate=MeshReplicationType.NONE)
        device_mesh = (2, 2)

        layout = make_zz_layout(shape, [0, 1], (32, 32))
        tensor = proxy(shape, policy, device_mesh, layout=layout)

        meta = tensor.meta_data
        assert "global_layout" in meta
        assert "sharded_layout" in meta

        # Global layout should have 2 modes per dim (block + grid)
        import tvm_ffi

        _dim_levels = tvm_ffi.get_global_func("tl.CuteLayout_dim_levels")
        _mode_shape = tvm_ffi.get_global_func("tl.CuteLayout_mode_shape")

        global_dl = _dim_levels(meta["global_layout"])
        assert tuple(int(v) for v in global_dl) == (2, 2)

        # Sharded: each dim sharded by 2 → grid goes from 4 to 2
        sharded_ms = _mode_shape(meta["sharded_layout"])
        # block preserved at 32, grid = 64/32 = 2
        assert int(sharded_ms[0]) == 32  # block dim0
        assert int(sharded_ms[1]) == 2  # grid dim0 (128/2/32 = 2)
        assert int(sharded_ms[2]) == 32  # block dim1
        assert int(sharded_ms[3]) == 2  # grid dim1

    def test_zn_layout(self):
        """ZN layout is stored and sharded correctly."""
        proxy = MeshTensorProxy()
        shape = (128, 128)
        policy = MeshShardingPolicy(y=0, x=1, replicate=MeshReplicationType.NONE)
        device_mesh = (2, 2)

        layout = make_zn_layout(shape, [0, 1], (32, 32))
        tensor = proxy(shape, policy, device_mesh, layout=layout)

        import tvm_ffi

        _dim_levels = tvm_ffi.get_global_func("tl.CuteLayout_dim_levels")
        _mode_stride = tvm_ffi.get_global_func("tl.CuteLayout_mode_stride")

        meta = tensor.meta_data
        global_dl = _dim_levels(meta["global_layout"])
        assert tuple(int(v) for v in global_dl) == (2, 2)

        # ZN has column-major inner block — stride pattern differs from ZZ
        global_strides = _mode_stride(meta["global_layout"])
        sharded_strides = _mode_stride(meta["sharded_layout"])

        # Inner block strides should match (same block ordering)
        # The physical ordering is preserved by DeriveLayoutLike
        assert int(global_strides[0]) == int(sharded_strides[0])  # innermost stride preserved
        assert int(global_strides[2]) == int(sharded_strides[2])  # inner block dim1 stride

    def test_replicate_all_preserves_layout(self):
        """ALL replication leaves layout unchanged."""
        proxy = MeshTensorProxy()
        shape = (128, 128)
        policy = MeshShardingPolicy(replicate=MeshReplicationType.ALL)
        device_mesh = (2, 2)

        layout = make_zz_layout(shape, [0, 1], (32, 32))
        tensor = proxy(shape, policy, device_mesh, layout=layout)

        import tvm_ffi

        _mode_shape = tvm_ffi.get_global_func("tl.CuteLayout_mode_shape")
        _same_layout = tvm_ffi.get_global_func("tl.IsSameLayout")

        meta = tensor.meta_data
        # Global and sharded should be identical for ALL replication
        assert _same_layout(meta["global_layout"], meta["sharded_layout"])

    def test_4d_zz_layout(self):
        """4D tensor with ZZ tiling on selected axes."""
        proxy = MeshTensorProxy()
        shape = (2, 128, 4, 128)
        policy = MeshShardingPolicy(y=1, x=3, replicate=MeshReplicationType.NONE)
        device_mesh = (2, 2)

        layout = make_zz_layout(shape, [1, 3], (32, 32))
        tensor = proxy(shape, policy, device_mesh, layout=layout)

        import tvm_ffi

        _dim_levels = tvm_ffi.get_global_func("tl.CuteLayout_dim_levels")
        _mode_shape = tvm_ffi.get_global_func("tl.CuteLayout_mode_shape")

        meta = tensor.meta_data
        sharded_dl = _dim_levels(meta["sharded_layout"])
        # dims 0, 2 have 1 level; dims 1, 3 have 2 levels
        assert tuple(int(v) for v in sharded_dl) == (1, 2, 1, 2)

        sharded_ms = _mode_shape(meta["sharded_layout"])
        # dim 0: untiled, shape 2 (not sharded)
        assert int(sharded_ms[0]) == 2
        # dim 1: sharded by nrows=2 → 64, block=32, grid=2
        assert int(sharded_ms[1]) == 32
        assert int(sharded_ms[2]) == 2
        # dim 2: untiled, shape 4 (not sharded)
        assert int(sharded_ms[3]) == 4
        # dim 3: sharded by ncols=2 → 64, block=32, grid=2
        assert int(sharded_ms[4]) == 32
        assert int(sharded_ms[5]) == 2


# ─── Kernel integration test ────────────────────────────────────────────


def test_mesh_tensor_in_kernel():
    """MeshTensor with layout flows through kernel definition correctly."""
    policy = MeshShardingPolicy(y=0, x=1, replicate=MeshReplicationType.NONE)
    device_mesh = (2, 2)
    M, N, K = 128, 128, 128

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    A_tensor = T.MeshTensor((M, K), policy, device_mesh, "float16", layout=A_layout)
    B_tensor = T.MeshTensor((K, N), policy, device_mesh, "float16", layout=B_layout)
    C_tensor = T.MeshTensor((M, N), policy, device_mesh, "float32", layout=C_layout)

    @T.prim_func
    def kernel(A: A_tensor, B: B_tensor, C: C_tensor):
        sharded_M, sharded_K = A.shape
        _, sharded_N = B.shape

    assert "tensor_meta" in kernel.attrs
    tensor_meta = kernel.attrs["tensor_meta"]
    assert "A" in tensor_meta
    assert "sharded_layout" in tensor_meta["A"]
    assert "global_layout" in tensor_meta["A"]


# ─── Dynamic shape tests ────────────────────────────────────────────────


class TestDynamicShapeSharding:
    """Tests for _get_sharded_shape with symbolic PrimExpr dimensions."""

    @pytest.mark.parametrize(
        "nrows, ncols, policy",
        [
            (2, 2, MeshShardingPolicy(y=0, x=1, replicate=MeshReplicationType.NONE)),
            (4, 2, MeshShardingPolicy(y=0, x=1, replicate=MeshReplicationType.NONE)),
            (2, 4, MeshShardingPolicy(x=1, replicate=MeshReplicationType.COLUMN)),
            (4, 1, MeshShardingPolicy(y=0, replicate=MeshReplicationType.ROW)),
            (2, 2, MeshShardingPolicy(cross_mesh_dim=0)),
        ],
    )
    def test_get_sharded_shape_symbolic(self, nrows, ncols, policy):
        """_get_sharded_shape returns PrimExpr ceildiv for symbolic dims."""
        M = tir.Var("M", "int32")
        N = tir.Var("N", "int32")
        shape = (M, N)
        result = MeshTensorProxy._get_sharded_shape(shape, policy, nrows, ncols)

        # Verify result is a tuple of length 2
        assert len(result) == 2

        # Substitute concrete values and verify against the concrete-shape path
        concrete_M, concrete_N = 128, 256
        concrete_shape = (concrete_M, concrete_N)
        concrete_result = MeshTensorProxy._get_sharded_shape(concrete_shape, policy, nrows, ncols)

        for i in range(2):
            if isinstance(result[i], tir.PrimExpr):
                substituted = tir.stmt_functor.substitute(
                    result[i], {M: tir.IntImm("int32", concrete_M), N: tir.IntImm("int32", concrete_N)}
                )
                from tvm import arith

                analyzer = arith.Analyzer()
                simplified = analyzer.simplify(substituted)
                assert simplified.value == concrete_result[i], (
                    f"dim {i}: symbolic result evaluates to {simplified.value}, expected {concrete_result[i]}"
                )
            else:
                assert result[i] == concrete_result[i]

    def test_get_sharded_shape_replicate_all_symbolic(self):
        """ALL replication returns shape unchanged for symbolic dims."""
        M = tir.Var("M", "int32")
        N = tir.Var("N", "int32")
        shape = (M, N)
        policy = MeshShardingPolicy(replicate=MeshReplicationType.ALL)
        result = MeshTensorProxy._get_sharded_shape(shape, policy, 4, 4)
        assert result == shape

    def test_get_sharded_shape_mixed_symbolic_concrete(self):
        """Mix of symbolic and concrete dims: only the sharded dim becomes PrimExpr."""
        M = tir.Var("M", "int32")
        shape = (M, 256)
        policy = MeshShardingPolicy(y=0, x=1, replicate=MeshReplicationType.NONE)
        result = MeshTensorProxy._get_sharded_shape(shape, policy, 2, 4)

        # dim 0 (y-sharded by nrows=2): symbolic ceildiv
        assert isinstance(result[0], tir.PrimExpr)
        # dim 1 (x-sharded by ncols=4): concrete 256/4 = 64
        assert result[1] == 64


class TestDynamicShapeCallMethod:
    """Tests for MeshTensorProxy.__call__ with symbolic shapes."""

    @pytest.mark.parametrize(
        "policy",
        [
            MeshShardingPolicy(y=0, x=1, replicate=MeshReplicationType.NONE),
            MeshShardingPolicy(replicate=MeshReplicationType.ALL),
            MeshShardingPolicy(cross_mesh_dim=1),
            MeshShardingPolicy(y=0, replicate=MeshReplicationType.ROW),
            MeshShardingPolicy(x=1, replicate=MeshReplicationType.COLUMN),
        ],
    )
    def test_call_default_row_major_symbolic(self, policy):
        """MeshTensor construction with symbolic shapes produces valid metadata."""
        M = tir.Var("M", "int32")
        K = tir.Var("K", "int32")
        shape = (M, K)
        device_mesh = (2, 2)
        proxy = MeshTensorProxy()

        tensor_with_meta = proxy(shape, policy, device_mesh)
        meta = tensor_with_meta.meta_data

        # Global shape should be the original symbolic shape
        assert meta["global_shape"] == shape

        # CuteLayout objects should be present
        assert "global_layout" in meta
        assert "sharded_layout" in meta

        # Buffer should be created successfully
        assert len(tensor_with_meta.buffer.shape) == 2
