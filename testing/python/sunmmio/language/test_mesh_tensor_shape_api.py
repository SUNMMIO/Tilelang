import pytest

import tilelang.language as T
from tilelang import tvm
from tilelang.utils.target import determine_target


SUNMMIO_TARGET = determine_target("Sunmmio", return_object=True)


def test_mesh_sharding_policy_normalizes_integer_replicate():
    policy = T.MeshShardingPolicy(replicate=0)
    assert policy.replicate is T.MeshReplicationType.NONE

    tensor = T.MeshTensor((8, 16), policy, (2, 4), "float16")
    assert tensor.local_shape == (8, 16)


@pytest.mark.parametrize(
    "placement, expected_kind, expected_local_shape, expected_extents",
    [
        (T.placement.replicated(), 0, (5,), [5, 5, 5, 5]),
        (T.placement.row_shard(0), 1, (3,), [3, 3, 2, 2]),
        (T.placement.col_shard(0), 2, (3,), [3, 2, 3, 2]),
        (T.placement.full_shard(0, 0), 3, (2,), [2, 1, 1, 1]),
        (T.placement.mesh_as_line(0), 4, (2,), [2, 1, 1, 1]),
    ],
)
def test_placement_constructors_and_metadata(placement, expected_kind, expected_local_shape, expected_extents):
    tensor = T.MeshTensor((5,), placement, (2, 2), "float16")

    assert placement.kind == expected_kind
    assert tensor.meta_data["placement_kind"] == expected_kind
    assert tensor.local_shape == expected_local_shape
    assert [tensor.get_local_extent(cid)[0] for cid in range(4)] == expected_extents


def test_full_shard_same_dim_differs_from_mesh_as_line():
    full_shard = T.MeshTensor((6,), T.placement.full_shard(0, 0), (2, 2), "float16")
    mesh_as_line = T.MeshTensor((6,), T.placement.mesh_as_line(0), (2, 2), "float16")

    assert full_shard.local_shape == mesh_as_line.local_shape == (2,)
    assert [full_shard.get_local_extent(cid)[0] for cid in range(4)] == [2, 1, 2, 1]
    assert [mesh_as_line.get_local_extent(cid)[0] for cid in range(4)] == [2, 2, 1, 1]
    assert full_shard.meta_data["placement_kind"] != mesh_as_line.meta_data["placement_kind"]


@pytest.mark.parametrize(
    "make_placement",
    [
        lambda: T.placement.row_shard(-1),
        lambda: T.placement.col_shard(-1),
        lambda: T.placement.full_shard(-1, 0),
        lambda: T.placement.full_shard(0, -1),
        lambda: T.placement.mesh_as_line(-1),
    ],
)
def test_placement_constructors_reject_negative_dimensions(make_placement):
    with pytest.raises(ValueError, match="must be a non-negative int"):
        make_placement()


@pytest.mark.parametrize(
    "placement",
    [
        T.placement.row_shard(1),
        T.placement.col_shard(1),
        T.placement.full_shard(0, 1),
        T.placement.full_shard(1, 0),
        T.placement.mesh_as_line(1),
    ],
)
def test_placement_rejects_dimensions_outside_tensor_rank(placement):
    with pytest.raises(ValueError, match=r"Invalid (row|col) shard dimension"):
        T.MeshTensor((6,), placement, (2, 2), "float16")


def test_mesh_tensor_shape_api_in_kernel():
    tensor = T.MeshTensor(
        (513, 4097),
        T.MeshShardingPolicy(y=0, x=1),
        (4, 4),
        "float16",
    )

    assert tensor.global_shape == (513, 4097)
    assert not hasattr(tensor, "shape")
    assert tensor.local_shape == (129, 1025)
    assert tensor.get_local_extent(0) == (129, 1025)
    assert tensor.get_local_extent(1) == (129, 1024)
    assert tensor.get_local_extent(4) == (128, 1025)
    assert tensor.get_local_extent(15) == (128, 1024)

    with tvm.target.Target(SUNMMIO_TARGET):

        @T.prim_func
        def kernel(A: tensor):
            with T.Kernel(T.mesh_ncores()) as cid:
                global_m, global_n = A.global_shape
                local_m, local_n = A.local_shape
                valid_m, valid_n = A.get_local_extent(cid)
                core0_m, core0_n = A.get_local_extent(0)
                core15_m, core15_n = A.get_local_extent(15)

                assert not hasattr(A, "shape")
                assert global_m == 513
                assert global_n == 4097
                assert local_m == 129
                assert local_n == 1025
                assert valid_m <= local_m
                assert valid_n <= local_n
                assert core0_m == 129
                assert core0_n == 1025
                assert core15_m == 128
                assert core15_n == 1024

    assert "tensor_meta" in kernel.attrs
    assert "threadIdx" not in tvm.IRModule({"main": kernel}).script()


def test_mesh_tensor_same_dim_row_then_col_extent():
    tensor = T.MeshTensor(
        (65, 9),
        T.MeshShardingPolicy(y=0, x=0),
        (4, 4),
        "float16",
    )

    assert tensor.global_shape == (65, 9)
    assert not hasattr(tensor, "shape")
    assert tensor.local_shape == (5, 9)
    assert tensor.get_local_extent(0) == (5, 9)
    assert tensor.get_local_extent(1) == (4, 9)
    assert tensor.get_local_extent(15) == (4, 9)

    with tvm.target.Target(SUNMMIO_TARGET):

        @T.prim_func
        def kernel(A: tensor):
            with T.Kernel(T.mesh_ncores()) as cid:
                global_m, global_n = A.global_shape
                local_m, local_n = A.local_shape
                valid_m, valid_n = A.get_local_extent(cid)
                core0_m, core0_n = A.get_local_extent(0)
                core1_m, core1_n = A.get_local_extent(1)
                core15_m, core15_n = A.get_local_extent(15)

                assert not hasattr(A, "shape")
                assert global_m == 65
                assert global_n == 9
                assert local_m == 5
                assert local_n == 9
                assert valid_m <= local_m
                assert valid_n <= local_n
                assert core0_m == 5
                assert core0_n == 9
                assert core1_m == 4
                assert core1_n == 9
                assert core15_m == 4
                assert core15_n == 9

    assert "tensor_meta" in kernel.attrs
    assert "threadIdx" not in tvm.IRModule({"main": kernel}).script()
