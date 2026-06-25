import tilelang.language as T
from tilelang import tvm
from tilelang.utils.target import determine_target


SUNMMIO_TARGET = determine_target("Sunmmio", return_object=True)


def test_mesh_tensor_shape_api_in_kernel():
    tensor = T.MeshTensor(
        (513, 4097),
        T.MeshShardingPolicy(y=0, x=1),
        (4, 4),
        "float16",
    )

    assert tensor.shape == (513, 4097)
    assert tensor.local_shape == (129, 1025)
    assert tensor.get_local_extent(0) == (129, 1025)
    assert tensor.get_local_extent(1) == (129, 1024)
    assert tensor.get_local_extent(4) == (128, 1025)
    assert tensor.get_local_extent(15) == (128, 1024)

    with tvm.target.Target(SUNMMIO_TARGET):

        @T.prim_func
        def kernel(A: tensor):
            with T.Kernel(16) as cid:
                global_m, global_n = A.shape
                local_m, local_n = A.local_shape
                valid_m, valid_n = A.get_local_extent(cid)
                core0_m, core0_n = A.get_local_extent(0)
                core15_m, core15_n = A.get_local_extent(15)

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

    assert tensor.shape == (65, 9)
    assert tensor.local_shape == (5, 9)
    assert tensor.get_local_extent(0) == (5, 9)
    assert tensor.get_local_extent(1) == (4, 9)
    assert tensor.get_local_extent(15) == (4, 9)

    with tvm.target.Target(SUNMMIO_TARGET):

        @T.prim_func
        def kernel(A: tensor):
            with T.Kernel(16) as cid:
                global_m, global_n = A.shape
                local_m, local_n = A.local_shape
                valid_m, valid_n = A.get_local_extent(cid)
                core0_m, core0_n = A.get_local_extent(0)
                core1_m, core1_n = A.get_local_extent(1)
                core15_m, core15_n = A.get_local_extent(15)

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
