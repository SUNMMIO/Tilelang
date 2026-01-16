import math
import pytest

import tilelang.language as T
from tilelang.language.v2.annot import MeshTensorAnnot, MeshShardingPolicy, \
    MeshReplicationType, TensorAnnot


@pytest.mark.parametrize("shape, nrows, ncols", [
    ((100, 200, 300), 2, 2),
    ((64, 128), 4, 1),
    ((10, 20, 30, 40), 8, 8),
])
def test_get_sharded_shape_replicate_all(shape, nrows, ncols):
    policy = MeshShardingPolicy(replicate=MeshReplicationType.ALL)
    expected_shape = shape
    assert MeshTensorAnnot._get_sharded_shape(shape, policy, nrows, ncols) == expected_shape


@pytest.mark.parametrize("shape, cross_mesh_dim, nrows, ncols", [
    ((100, 200, 300), 1, 2, 2),
    ((100, 203, 300), 1, 2, 2),
    ((128, 256, 512), 0, 4, 4),
    ((128, 256, 512), 2, 2, 8),
])
def test_get_sharded_shape_cross_mesh_dim(shape, cross_mesh_dim, nrows, ncols):
    policy = MeshShardingPolicy(cross_mesh_dim=cross_mesh_dim)
    total_cores = nrows * ncols

    expected_shape_list = list(shape)
    expected_shape_list[cross_mesh_dim] = math.ceil(shape[cross_mesh_dim] / total_cores)
    expected_shape = tuple(expected_shape_list)

    assert MeshTensorAnnot._get_sharded_shape(shape, policy, nrows, ncols) == expected_shape


@pytest.mark.parametrize("shape, cross_mesh_dim, nrows, ncols", [
    ((100, 200, 300), 3, 2, 2),
    ((100, 200), 2, 2, 2),
])
def test_get_sharded_shape_cross_mesh_dim_invalid(shape, cross_mesh_dim, nrows, ncols):
    with pytest.raises(ValueError, match="Invalid cross_mesh_dim"):
        MeshTensorAnnot._get_sharded_shape(shape, MeshShardingPolicy(cross_mesh_dim=cross_mesh_dim),
                                           nrows, ncols)


@pytest.mark.parametrize("shape, y_dim, nrows, ncols", [
    ((100, 200, 300), 0, 4, 4),
    ((103, 200, 300), 0, 4, 4),
    ((128, 256, 512), 2, 2, 8),
])
def test_get_sharded_shape_replicate_row(shape, y_dim, nrows, ncols):
    policy = MeshShardingPolicy(y=y_dim, replicate=MeshReplicationType.ROW)

    expected_shape_list = list(shape)
    expected_shape_list[y_dim] = math.ceil(shape[y_dim] / nrows)
    expected_shape = tuple(expected_shape_list)

    assert MeshTensorAnnot._get_sharded_shape(shape, policy, nrows, ncols) == expected_shape


@pytest.mark.parametrize("shape, policy, nrows, ncols, error_msg", [
    ((100, 200, 300), MeshShardingPolicy(x=1, y=0, replicate=MeshReplicationType.ROW), 4, 4,
     "Cannot shard on x-axis when replicating on rows"),
    ((100, 200, 300), MeshShardingPolicy(
        y=3, replicate=MeshReplicationType.ROW), 4, 4, "Invalid y-split dimension"),
])
def test_get_sharded_shape_replicate_row_invalid(shape, policy, nrows, ncols, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        MeshTensorAnnot._get_sharded_shape(shape, policy, nrows, ncols)


@pytest.mark.parametrize("shape, x_dim, nrows, ncols", [
    ((100, 200, 300), 1, 4, 4),
    ((100, 203, 300), 1, 4, 4),
    ((128, 256, 512), 0, 2, 8),
])
def test_get_sharded_shape_replicate_column(shape, x_dim, nrows, ncols):
    policy = MeshShardingPolicy(x=x_dim, replicate=MeshReplicationType.COLUMN)

    expected_shape_list = list(shape)
    expected_shape_list[x_dim] = math.ceil(shape[x_dim] / ncols)
    expected_shape = tuple(expected_shape_list)

    assert MeshTensorAnnot._get_sharded_shape(shape, policy, nrows, ncols) == expected_shape


@pytest.mark.parametrize("shape, policy, nrows, ncols, error_msg", [
    ((100, 200, 300), MeshShardingPolicy(x=1, y=0, replicate=MeshReplicationType.COLUMN), 4, 4,
     "Cannot shard on y-axis when replicating on columns"),
    ((100, 200, 300), MeshShardingPolicy(
        x=3, replicate=MeshReplicationType.COLUMN), 4, 4, "Invalid x-split dimension"),
])
def test_get_sharded_shape_replicate_column_invalid(shape, policy, nrows, ncols, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        MeshTensorAnnot._get_sharded_shape(shape, policy, nrows, ncols)


@pytest.mark.parametrize("shape, x_dim, y_dim, nrows, ncols", [
    ((100, 200, 300), 1, 0, 2, 2),
    ((101, 203, 300), 1, 0, 2, 2),
    ((100, 200, 300), 1, None, 2, 2),
    ((100, 200, 300), None, 0, 2, 2),
])
def test_get_sharded_shape_none_replication(shape, x_dim, y_dim, nrows, ncols):
    policy = MeshShardingPolicy(x=x_dim, y=y_dim, replicate=MeshReplicationType.NONE)

    expected_shape_list = list(shape)
    if y_dim is not None:
        expected_shape_list[y_dim] = math.ceil(shape[y_dim] / nrows)
    if x_dim is not None:
        expected_shape_list[x_dim] = math.ceil(shape[x_dim] / ncols)
    expected_shape = tuple(expected_shape_list)

    assert MeshTensorAnnot._get_sharded_shape(shape, policy, nrows, ncols) == expected_shape


@pytest.mark.parametrize("shape, policy, nrows, ncols, error_msg", [
    ((100, 200, 300), MeshShardingPolicy(
        x=3, replicate=MeshReplicationType.NONE), 2, 2, "Invalid x-split dimension"),
    ((100, 200, 300), MeshShardingPolicy(
        y=3, replicate=MeshReplicationType.NONE), 2, 2, "Invalid y-split dimension"),
])
def test_get_sharded_shape_none_replication_invalid(shape, policy, nrows, ncols, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        MeshTensorAnnot._get_sharded_shape(shape, policy, nrows, ncols)


@pytest.mark.parametrize("shape, device_mesh_config, policy", [
    ((100, 200, 300), (2, 2), MeshShardingPolicy(replicate=MeshReplicationType.ALL)),
    ((100, 200, 300), (2, 2), MeshShardingPolicy(cross_mesh_dim=1)),
    ((100, 200, 300), (2, 2), MeshShardingPolicy(y=0, replicate=MeshReplicationType.ROW)),
    ((100, 200, 300), (2, 2), MeshShardingPolicy(x=1, replicate=MeshReplicationType.COLUMN)),
    ((100, 200, 300), (2, 2), MeshShardingPolicy(y=0, x=1, replicate=MeshReplicationType.NONE)),
    ((128, 256), (4, 2), MeshShardingPolicy(y=0, x=1, replicate=MeshReplicationType.NONE)),
    ((128, 256, 512), (2, 4), MeshShardingPolicy(cross_mesh_dim=2)),
])
def test_call_method(shape, device_mesh_config, policy):
    proxy = MeshTensorAnnot()
    nrows, ncols = device_mesh_config

    buffer = proxy(shape, policy, device_mesh_config)

    expected_shape = MeshTensorAnnot._get_sharded_shape(shape, policy, nrows, ncols)

    assert tuple(buffer.buffer.shape) == expected_shape


@pytest.mark.parametrize("M_val, N_val, K_val, device_mesh_config, policyA, policyB, policyC", [
    (100, 200, 300, (4, 4), MeshShardingPolicy(x=1, y=0), MeshShardingPolicy(
        x=1, y=0), MeshShardingPolicy(x=1, y=0)),
    (128, 256, 512, (2, 8), MeshShardingPolicy(x=1, y=0), MeshShardingPolicy(
        x=1, y=0), MeshShardingPolicy(x=1, y=0)),
    (100, 200, 300, (2, 2), MeshShardingPolicy(cross_mesh_dim=1),
     MeshShardingPolicy(cross_mesh_dim=1), MeshShardingPolicy(cross_mesh_dim=1)),
    (100, 200, 300, (2, 2), MeshShardingPolicy(replicate=MeshReplicationType.ALL),
     MeshShardingPolicy(cross_mesh_dim=1), MeshShardingPolicy(y=0, x=1)),
])
def test_mesh_tensor_annot(M_val, N_val, K_val, device_mesh_config, policyA, policyB, policyC):

    def get_expected_str(buffer_name, handle_name, shape, policy):
        annot = MeshTensorAnnot()
        sharded_buffer = annot(shape, policy, device_mesh_config)
        sharded_shape = tuple(sharded_buffer.buffer.shape)
        # Assumes 2D row-major
        strides = (sharded_shape[1], 1) if len(sharded_shape) == 2 else ()
        stride_str = f", strides={strides}" if strides else ""
        return f"{buffer_name} = T.match_buffer({handle_name}, {sharded_shape}{stride_str})"

    def example_tensor_annot(M: T.PrimExpr, N: T.PrimExpr, K: T.PrimExpr):

        A_tensor = T.MeshTensor((
            M,
            K,
        ), policyA, device_mesh_config, dtype="float32")
        B_tensor = T.MeshTensor((
            K,
            N,
        ), policyB, device_mesh_config, dtype="float32")
        C_tensor = T.MeshTensor((
            M,
            N,
        ), policyC, device_mesh_config, dtype="float32")

        @T.prim_func
        def kernel(A: A_tensor, B: B_tensor, C: C_tensor):
            sharded_M, sharded_K = A.shape
            _, sharded_N = B.shape

        return kernel

    ker = example_tensor_annot(M_val, N_val, K_val)
    script = str(ker)
    # print(script) # Removed print for clean output

    expected_A_str = get_expected_str("A", "A_handle", (M_val, K_val), policyA)
    expected_B_str = get_expected_str("B", "B_handle", (K_val, N_val), policyB)
    expected_C_str = get_expected_str("C", "C_handle", (M_val, N_val), policyC)

    assert expected_A_str in script
    assert expected_B_str in script
    assert expected_C_str in script

    # Assertions for tensor_meta
    assert "tensor_meta" in ker.attrs
    tensor_meta = ker.attrs["tensor_meta"]

    expected_A_global_shape = (M_val, K_val)
    expected_A_global_strides = TensorAnnot._construct_strides(expected_A_global_shape)
    assert tuple(tensor_meta["A"]["global_shape"]) == expected_A_global_shape
    assert tuple(tensor_meta["A"]["global_strides"]) == expected_A_global_strides

    expected_B_global_shape = (K_val, N_val)
    expected_B_global_strides = TensorAnnot._construct_strides(expected_B_global_shape)
    assert tuple(tensor_meta["B"]["global_shape"]) == expected_B_global_shape
    assert tuple(tensor_meta["B"]["global_strides"]) == expected_B_global_strides

    expected_C_global_shape = (M_val, N_val)
    expected_C_global_strides = TensorAnnot._construct_strides(expected_C_global_shape)
    assert tuple(tensor_meta["C"]["global_shape"]) == expected_C_global_shape
    assert tuple(tensor_meta["C"]["global_strides"]) == expected_C_global_strides
