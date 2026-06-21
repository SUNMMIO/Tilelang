import pytest

import tilelang
import tilelang.language as T

from tilelang import tvm as tvm
from tilelang.utils.target import determine_target


@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype, accum_dtype",
    [
        (1024, 1024, 128, 128, "float16", "float"),
    ],
)
def test_comm_python_api(M, N, block_M, block_N, dtype, accum_dtype):
    func_str = """
        T.comm_broadcast(A_shared[0:128, 0:128], B_shared[0:128, 0:128], -1, 6, 2)
        T.comm_put(A_shared[0:128, 0:128], B_shared[0:128, 0:128], -1, 6, 11)
        T.comm_allgather(A_shared[0:128, 0:128], C_shared[0:16, 0:128, 0:128], 2, -1, -1)
        T.comm_allgather(A_shared[0:128, 0:128], R0_shared[0:2048, 0:128], 2, -1, 0)
        T.comm_allgather(A_shared[0:128, 0:128], R1_shared[0:128, 0:2048], 2, -1, 1)""".strip()

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            B_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            C_shared = T.alloc_shared([16, block_M, block_N], dtype, scope="shared.rsram")
            R0_shared = T.alloc_shared([16 * block_M, block_N], dtype, scope="shared.rsram")
            R1_shared = T.alloc_shared([block_M, 16 * block_N], dtype, scope="shared.rsram")
            T.copy(A[by * block_M, bx * block_N], A_shared)

            T.comm.broadcast(A_shared, B_shared, (1, 2), direction="all")
            T.comm.put(A_shared, B_shared, (1, 2), (2, 3))
            T.comm.all_gather(A_shared, C_shared, direction="all")
            T.comm.all_gather(A_shared, R0_shared, direction="all", axis=0)
            T.comm.all_gather(A_shared, R1_shared, direction="all", axis=-1)

    assert main.script()[-len(func_str) :] == func_str, "The generated script does not match the expected output."


@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype, accum_dtype",
    [
        (1024, 1024, 128, 128, "float16", "float"),
    ],
)
def test_comm_broadcast_lower(M, N, block_M, block_N, dtype, accum_dtype):
    func_str = """
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 16384, 6, 1, 0)
            T.broadcast_(T.region(B_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 16384, 2, 0, 0)
            T.broadcast_(T.region(B_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 16384, 6, 0, 0)
            T.broadcast_(T.region(B_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 16384, 10, 0, 0)
            T.broadcast_(T.region(B_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 16384, 14, 0, 0)""".strip()

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            B_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            T.copy(A[by * block_M, bx * block_N], A_shared)

            T.comm.broadcast(A_shared, B_shared, (1, 2), direction="all")

    mod = tvm.IRModule({"main": main})
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
        assert mod.script()[-len(func_str) :] == func_str, "The generated script does not match the expected output."


@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype",
    [
        (1024, 1024, 128, 128, "float16"),
    ],
)
def test_comm_broadcast_lower_custom_mesh(M, N, block_M, block_N, dtype):
    func_str = """
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 16384, 5, 1, 0)
            T.broadcast_(T.region(B_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 16384, 2, 0, 0)
            T.broadcast_(T.region(B_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 16384, 5, 0, 0)""".strip()

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            B_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            T.copy(A[by * block_M, bx * block_N], A_shared)

            T.evaluate(
                tvm.tir.call_intrin(
                    "handle",
                    tvm.tir.op.Op.get("tl.tileop.comm_broadcast"),
                    A_shared[0:128, 0:128],
                    B_shared[0:128, 0:128],
                    -1,
                    5,
                    2,
                )
            )

    mod = tvm.IRModule({"main": main})
    target = determine_target("llvm -mcpu=sunmmio-a4e -mattr=device_mesh_nrow_2,device_mesh_ncol_3", return_object=True)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
        assert mod.script()[-len(func_str) :] == func_str, "The generated script does not match the expected output."


@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype, accum_dtype",
    [
        (1024, 1024, 128, 128, "float16", "float"),
    ],
)
def test_comm_put_lower(M, N, block_M, block_N, dtype, accum_dtype):
    func_str = """
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 16384, 6, 1, 0, 0, 1, 3)
            T.broadcast_(T.region(B_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 16384, 10, 0, 0, 0, 1, 2)""".strip()

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            B_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            T.copy(A[by * block_M, bx * block_N], A_shared)

            T.comm.put(A_shared, B_shared, (1, 2), (2, 3))

    mod = tvm.IRModule({"main": main})
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
        assert mod.script()[-len(func_str) :] == func_str, "The generated script does not match the expected output."


@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype, accum_dtype",
    [
        (1024, 1024, 128, 128, "float16", "float"),
    ],
)
def test_comm_all_gather_lower(M, N, block_M, block_N, dtype, accum_dtype):
    func_str = """
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(C_shared[0, 0, 0], 2, 1, 128, 128), 16384, 0, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(C_shared[1, 0, 0], 2, 1, 128, 128), 16384, 1, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(C_shared[2, 0, 0], 2, 1, 128, 128), 16384, 2, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(C_shared[3, 0, 0], 2, 1, 128, 128), 16384, 3, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(C_shared[4, 0, 0], 2, 1, 128, 128), 16384, 4, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(C_shared[5, 0, 0], 2, 1, 128, 128), 16384, 5, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(C_shared[6, 0, 0], 2, 1, 128, 128), 16384, 6, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(C_shared[7, 0, 0], 2, 1, 128, 128), 16384, 7, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(C_shared[8, 0, 0], 2, 1, 128, 128), 16384, 8, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(C_shared[9, 0, 0], 2, 1, 128, 128), 16384, 9, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(C_shared[10, 0, 0], 2, 1, 128, 128), 16384, 10, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(C_shared[11, 0, 0], 2, 1, 128, 128), 16384, 11, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(C_shared[12, 0, 0], 2, 1, 128, 128), 16384, 12, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(C_shared[13, 0, 0], 2, 1, 128, 128), 16384, 13, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(C_shared[14, 0, 0], 2, 1, 128, 128), 16384, 14, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(C_shared[15, 0, 0], 2, 1, 128, 128), 16384, 15, 0, 0)
            T.broadcast_(T.region(C_shared[0, 0, 0], 1, 4, 128, 128), T.region(C_shared[0, 0, 0], 2, 4, 128, 128), 65536, 0, 1, 0)
            T.broadcast_(T.region(C_shared[4, 0, 0], 1, 4, 128, 128), T.region(C_shared[4, 0, 0], 2, 4, 128, 128), 65536, 4, 1, 0)
            T.broadcast_(T.region(C_shared[8, 0, 0], 1, 4, 128, 128), T.region(C_shared[8, 0, 0], 2, 4, 128, 128), 65536, 8, 1, 0)
            T.broadcast_(T.region(C_shared[12, 0, 0], 1, 4, 128, 128), T.region(C_shared[12, 0, 0], 2, 4, 128, 128), 65536, 12, 1, 0)
            T.broadcast_(T.region(C_shared[0, 0, 0], 1, 4, 128, 128), T.region(C_shared[0, 0, 0], 2, 4, 128, 128), 65536, 1, 1, 0)
            T.broadcast_(T.region(C_shared[4, 0, 0], 1, 4, 128, 128), T.region(C_shared[4, 0, 0], 2, 4, 128, 128), 65536, 5, 1, 0)
            T.broadcast_(T.region(C_shared[8, 0, 0], 1, 4, 128, 128), T.region(C_shared[8, 0, 0], 2, 4, 128, 128), 65536, 9, 1, 0)
            T.broadcast_(T.region(C_shared[12, 0, 0], 1, 4, 128, 128), T.region(C_shared[12, 0, 0], 2, 4, 128, 128), 65536, 13, 1, 0)
            T.broadcast_(T.region(C_shared[0, 0, 0], 1, 4, 128, 128), T.region(C_shared[0, 0, 0], 2, 4, 128, 128), 65536, 2, 1, 0)
            T.broadcast_(T.region(C_shared[4, 0, 0], 1, 4, 128, 128), T.region(C_shared[4, 0, 0], 2, 4, 128, 128), 65536, 6, 1, 0)
            T.broadcast_(T.region(C_shared[8, 0, 0], 1, 4, 128, 128), T.region(C_shared[8, 0, 0], 2, 4, 128, 128), 65536, 10, 1, 0)
            T.broadcast_(T.region(C_shared[12, 0, 0], 1, 4, 128, 128), T.region(C_shared[12, 0, 0], 2, 4, 128, 128), 65536, 14, 1, 0)
            T.broadcast_(T.region(C_shared[0, 0, 0], 1, 4, 128, 128), T.region(C_shared[0, 0, 0], 2, 4, 128, 128), 65536, 3, 1, 0)
            T.broadcast_(T.region(C_shared[4, 0, 0], 1, 4, 128, 128), T.region(C_shared[4, 0, 0], 2, 4, 128, 128), 65536, 7, 1, 0)
            T.broadcast_(T.region(C_shared[8, 0, 0], 1, 4, 128, 128), T.region(C_shared[8, 0, 0], 2, 4, 128, 128), 65536, 11, 1, 0)
            T.broadcast_(T.region(C_shared[12, 0, 0], 1, 4, 128, 128), T.region(C_shared[12, 0, 0], 2, 4, 128, 128), 65536, 15, 1, 0)""".strip()

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            C_shared = T.alloc_shared([16, block_M, block_N], dtype, scope="shared.rsram")
            T.copy(A[by * block_M, bx * block_N], A_shared)

            T.comm.all_gather(A_shared, C_shared, direction="all")

    mod = tvm.IRModule({"main": main})
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
        assert mod.script()[-len(func_str) :] == func_str, "The generated script does not match the expected output."


@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype",
    [
        (1024, 1024, 128, 128, "float16"),
    ],
)
def test_comm_all_gather_axis0_all_lower(M, N, block_M, block_N, dtype):
    func_str = """
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 0], 2, 128, 128), 16384, 0, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[128, 0], 2, 128, 128), 16384, 1, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[256, 0], 2, 128, 128), 16384, 2, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[384, 0], 2, 128, 128), 16384, 3, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[512, 0], 2, 128, 128), 16384, 4, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[640, 0], 2, 128, 128), 16384, 5, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[768, 0], 2, 128, 128), 16384, 6, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[896, 0], 2, 128, 128), 16384, 7, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[1024, 0], 2, 128, 128), 16384, 8, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[1152, 0], 2, 128, 128), 16384, 9, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[1280, 0], 2, 128, 128), 16384, 10, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[1408, 0], 2, 128, 128), 16384, 11, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[1536, 0], 2, 128, 128), 16384, 12, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[1664, 0], 2, 128, 128), 16384, 13, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[1792, 0], 2, 128, 128), 16384, 14, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[1920, 0], 2, 128, 128), 16384, 15, 0, 0)
            T.broadcast_(T.region(R_shared[0, 0], 1, 512, 128), T.region(R_shared[0, 0], 2, 512, 128), 65536, 0, 1, 0)
            T.broadcast_(T.region(R_shared[512, 0], 1, 512, 128), T.region(R_shared[512, 0], 2, 512, 128), 65536, 4, 1, 0)
            T.broadcast_(T.region(R_shared[1024, 0], 1, 512, 128), T.region(R_shared[1024, 0], 2, 512, 128), 65536, 8, 1, 0)
            T.broadcast_(T.region(R_shared[1536, 0], 1, 512, 128), T.region(R_shared[1536, 0], 2, 512, 128), 65536, 12, 1, 0)
            T.broadcast_(T.region(R_shared[0, 0], 1, 512, 128), T.region(R_shared[0, 0], 2, 512, 128), 65536, 1, 1, 0)
            T.broadcast_(T.region(R_shared[512, 0], 1, 512, 128), T.region(R_shared[512, 0], 2, 512, 128), 65536, 5, 1, 0)
            T.broadcast_(T.region(R_shared[1024, 0], 1, 512, 128), T.region(R_shared[1024, 0], 2, 512, 128), 65536, 9, 1, 0)
            T.broadcast_(T.region(R_shared[1536, 0], 1, 512, 128), T.region(R_shared[1536, 0], 2, 512, 128), 65536, 13, 1, 0)
            T.broadcast_(T.region(R_shared[0, 0], 1, 512, 128), T.region(R_shared[0, 0], 2, 512, 128), 65536, 2, 1, 0)
            T.broadcast_(T.region(R_shared[512, 0], 1, 512, 128), T.region(R_shared[512, 0], 2, 512, 128), 65536, 6, 1, 0)
            T.broadcast_(T.region(R_shared[1024, 0], 1, 512, 128), T.region(R_shared[1024, 0], 2, 512, 128), 65536, 10, 1, 0)
            T.broadcast_(T.region(R_shared[1536, 0], 1, 512, 128), T.region(R_shared[1536, 0], 2, 512, 128), 65536, 14, 1, 0)
            T.broadcast_(T.region(R_shared[0, 0], 1, 512, 128), T.region(R_shared[0, 0], 2, 512, 128), 65536, 3, 1, 0)
            T.broadcast_(T.region(R_shared[512, 0], 1, 512, 128), T.region(R_shared[512, 0], 2, 512, 128), 65536, 7, 1, 0)
            T.broadcast_(T.region(R_shared[1024, 0], 1, 512, 128), T.region(R_shared[1024, 0], 2, 512, 128), 65536, 11, 1, 0)
            T.broadcast_(T.region(R_shared[1536, 0], 1, 512, 128), T.region(R_shared[1536, 0], 2, 512, 128), 65536, 15, 1, 0)""".strip()

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            # 4x4 mesh -> 16 cores gathered along axis=0: [16 * block_M, block_N]
            R_shared = T.alloc_shared([16 * block_M, block_N], dtype, scope="shared.rsram")
            T.copy(A[by * block_M, bx * block_N], A_shared)

            T.comm.all_gather(A_shared, R_shared, direction="all", axis=0)

    mod = tvm.IRModule({"main": main})
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
        assert mod.script()[-len(func_str) :] == func_str, "The generated script does not match the expected output."


@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype",
    [
        (1024, 1024, 128, 128, "float16"),
    ],
)
def test_comm_all_gather_axis_last_horizontal_lower(M, N, block_M, block_N, dtype):
    func_str = """
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 0], 2, 128, 128), 16384, 0, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 128], 2, 128, 128), 16384, 1, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 256], 2, 128, 128), 16384, 2, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 384], 2, 128, 128), 16384, 3, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 0], 2, 128, 128), 16384, 4, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 128], 2, 128, 128), 16384, 5, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 256], 2, 128, 128), 16384, 6, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 384], 2, 128, 128), 16384, 7, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 0], 2, 128, 128), 16384, 8, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 128], 2, 128, 128), 16384, 9, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 256], 2, 128, 128), 16384, 10, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 384], 2, 128, 128), 16384, 11, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 0], 2, 128, 128), 16384, 12, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 128], 2, 128, 128), 16384, 13, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 256], 2, 128, 128), 16384, 14, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 384], 2, 128, 128), 16384, 15, 0, 0)""".strip()

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            # Horizontal allgather (K = mesh_ncol = 4) along last axis: [block_M, 4 * block_N]
            R_shared = T.alloc_shared([block_M, 4 * block_N], dtype, scope="shared.rsram")
            T.copy(A[by * block_M, bx * block_N], A_shared)

            T.comm.all_gather(A_shared, R_shared, direction="horizontal", axis=-1)

    mod = tvm.IRModule({"main": main})
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
        assert mod.script()[-len(func_str) :] == func_str, "The generated script does not match the expected output."


@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype",
    [
        (1024, 1024, 128, 128, "float16"),
    ],
)
def test_comm_all_gather_axis_last_all_lower(M, N, block_M, block_N, dtype):
    func_str = """
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 0], 2, 128, 128), 16384, 0, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 128], 2, 128, 128), 16384, 1, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 256], 2, 128, 128), 16384, 2, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 384], 2, 128, 128), 16384, 3, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 512], 2, 128, 128), 16384, 4, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 640], 2, 128, 128), 16384, 5, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 768], 2, 128, 128), 16384, 6, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 896], 2, 128, 128), 16384, 7, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 1024], 2, 128, 128), 16384, 8, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 1152], 2, 128, 128), 16384, 9, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 1280], 2, 128, 128), 16384, 10, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 1408], 2, 128, 128), 16384, 11, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 1536], 2, 128, 128), 16384, 12, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 1664], 2, 128, 128), 16384, 13, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 1792], 2, 128, 128), 16384, 14, 0, 0)
            T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(R_shared[0, 1920], 2, 128, 128), 16384, 15, 0, 0)
            T.broadcast_(T.region(R_shared[0, 0], 1, 128, 512), T.region(R_shared[0, 0], 2, 128, 512), 65536, 0, 1, 0)
            T.broadcast_(T.region(R_shared[0, 512], 1, 128, 512), T.region(R_shared[0, 512], 2, 128, 512), 65536, 4, 1, 0)
            T.broadcast_(T.region(R_shared[0, 1024], 1, 128, 512), T.region(R_shared[0, 1024], 2, 128, 512), 65536, 8, 1, 0)
            T.broadcast_(T.region(R_shared[0, 1536], 1, 128, 512), T.region(R_shared[0, 1536], 2, 128, 512), 65536, 12, 1, 0)
            T.broadcast_(T.region(R_shared[0, 0], 1, 128, 512), T.region(R_shared[0, 0], 2, 128, 512), 65536, 1, 1, 0)
            T.broadcast_(T.region(R_shared[0, 512], 1, 128, 512), T.region(R_shared[0, 512], 2, 128, 512), 65536, 5, 1, 0)
            T.broadcast_(T.region(R_shared[0, 1024], 1, 128, 512), T.region(R_shared[0, 1024], 2, 128, 512), 65536, 9, 1, 0)
            T.broadcast_(T.region(R_shared[0, 1536], 1, 128, 512), T.region(R_shared[0, 1536], 2, 128, 512), 65536, 13, 1, 0)
            T.broadcast_(T.region(R_shared[0, 0], 1, 128, 512), T.region(R_shared[0, 0], 2, 128, 512), 65536, 2, 1, 0)
            T.broadcast_(T.region(R_shared[0, 512], 1, 128, 512), T.region(R_shared[0, 512], 2, 128, 512), 65536, 6, 1, 0)
            T.broadcast_(T.region(R_shared[0, 1024], 1, 128, 512), T.region(R_shared[0, 1024], 2, 128, 512), 65536, 10, 1, 0)
            T.broadcast_(T.region(R_shared[0, 1536], 1, 128, 512), T.region(R_shared[0, 1536], 2, 128, 512), 65536, 14, 1, 0)
            T.broadcast_(T.region(R_shared[0, 0], 1, 128, 512), T.region(R_shared[0, 0], 2, 128, 512), 65536, 3, 1, 0)
            T.broadcast_(T.region(R_shared[0, 512], 1, 128, 512), T.region(R_shared[0, 512], 2, 128, 512), 65536, 7, 1, 0)
            T.broadcast_(T.region(R_shared[0, 1024], 1, 128, 512), T.region(R_shared[0, 1024], 2, 128, 512), 65536, 11, 1, 0)
            T.broadcast_(T.region(R_shared[0, 1536], 1, 128, 512), T.region(R_shared[0, 1536], 2, 128, 512), 65536, 15, 1, 0)""".strip()

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            # All direction (K = 16) gathered along last axis: [block_M, 16 * block_N]
            R_shared = T.alloc_shared([block_M, 16 * block_N], dtype, scope="shared.rsram")
            T.copy(A[by * block_M, bx * block_N], A_shared)

            T.comm.all_gather(A_shared, R_shared, direction="all", axis=-1)

    mod = tvm.IRModule({"main": main})
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
        assert mod.script()[-len(func_str) :] == func_str, "The generated script does not match the expected output."


'''
@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype, accum_dtype",
    [
        (1024 * 128, 1024 * 128, 1024, 1024, "float16", "float"),
    ],
)
def test_comm_all_reduce_lower(M, N, block_M, block_N, dtype, accum_dtype):
    func_str = """
                for i in T.parallel(1024):
                    for j in T.parallel(256):
                        for vec in T.vectorized(4):
                            A_shared[i * 8 + (j * 4 + vec) // 512 * 4 + (j * 4 + vec) % 4] = T.Cast("float32", A[by * 1024 + i, bx * 1024 + (j * 4 + vec)])
                for i in T.unroll(1024, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    buffer_2[i] = T.float32(0.0)
                    for rv in T.unroll(8, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        buffer_2[i] = buffer_2[i] + A_shared[i * 8 + rv % 2 * 4 + rv // 2]
                    buffer_2[i] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 128, 1, 0>::run", buffer_2[i], T.tvm_access_ptr(T.type_annotation("float32"), workspace.data, 0, 128, 2))
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 0, 1024, 2), 1024, 0, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 1024, 1024, 2), 1024, 1, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 2048, 1024, 2), 1024, 2, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 3072, 1024, 2), 1024, 3, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 0, 1024, 2), 1024, 4, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 1024, 1024, 2), 1024, 5, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 2048, 1024, 2), 1024, 6, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 3072, 1024, 2), 1024, 7, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 0, 1024, 2), 1024, 8, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 1024, 1024, 2), 1024, 9, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 2048, 1024, 2), 1024, 10, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 3072, 1024, 2), 1024, 11, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 0, 1024, 2), 1024, 12, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 1024, 1024, 2), 1024, 13, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 2048, 1024, 2), 1024, 14, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 3072, 1024, 2), 1024, 15, 0)
                for i in T.unroll(1024, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    buffer_2[i] = T.float32(0.0)
                    for rv in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        buffer_2[i] = buffer_2[i] + buffer[rv * 8 + i // 512 * 4 + i % 4]
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 0, 1024, 2), 1024, 0, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 1024, 1024, 2), 1024, 4, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 2048, 1024, 2), 1024, 8, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 3072, 1024, 2), 1024, 12, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 0, 1024, 2), 1024, 1, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 1024, 1024, 2), 1024, 5, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 2048, 1024, 2), 1024, 9, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 3072, 1024, 2), 1024, 13, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 0, 1024, 2), 1024, 2, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 1024, 1024, 2), 1024, 6, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 2048, 1024, 2), 1024, 10, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 3072, 1024, 2), 1024, 14, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 0, 1024, 2), 1024, 3, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 1024, 1024, 2), 1024, 7, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 2048, 1024, 2), 1024, 11, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 3072, 1024, 2), 1024, 15, 1)
                E_shared_clear = T.allocate([1024], "float32", "local")
                for i in T.unroll(1024, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    E_shared_clear_1 = T.Buffer((1024,), data=E_shared_clear, scope="local")
                    E_shared_clear_1[i] = T.float32(0.0)
                    for rv in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        E_shared_clear_1[i] = E_shared_clear_1[i] + buffer_1[rv * 8 + i // 512 * 4 + i % 4]
                    E_shared[i] = E_shared[i] + E_shared_clear_1[i]

# Metadata omitted. Use show_meta=True in script() method to show it.""".strip()

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            E_shared = T.alloc_shared([block_M], dtype, scope="shared.rsram")
            T.copy(A[by * block_M, bx * block_N], A_shared)

            T.comm.all_reduce(A_shared, E_shared, "sum", "all", dim=-1, clear=False)

    mod = tvm.IRModule({"main": main})
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.LayoutInference()(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
        assert mod.script()[-len(func_str):] == func_str, "The generated script does not match the expected output."
'''

if __name__ == "__main__":
    test_comm_python_api(1024, 1024, 128, 128, "float16", "float")
    test_comm_broadcast_lower(1024, 1024, 128, 128, "float16", "float")
    test_comm_put_lower(1024, 1024, 128, 128, "float16", "float")
    test_comm_all_gather_lower(1024, 1024, 128, 128, "float16", "float")
    test_comm_all_gather_axis0_all_lower(1024, 1024, 128, 128, "float16")
    test_comm_all_gather_axis_last_horizontal_lower(1024, 1024, 128, 128, "float16")
    test_comm_all_gather_axis_last_all_lower(1024, 1024, 128, 128, "float16")
    # test_comm_all_reduce_lower(1024 * 128, 1024 * 128, 1024, 1024, "float16", "float")
