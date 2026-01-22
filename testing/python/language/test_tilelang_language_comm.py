import pytest

import tilelang
import tilelang.language as T

from tilelang import tvm as tvm
from tilelang.utils.target import determine_target


@pytest.mark.parametrize("M, N, block_M, block_N, dtype, accum_dtype", [
    (1024, 1024, 128, 128, "float16", "float"),
])
def test_comm_python_api(M, N, block_M, block_N, dtype="float16", accum_dtype="float"):
    func_str = """# from tvm.script import tir as T

@T.prim_func
def main(A_handle: T.handle):
    A = T.match_buffer(A_handle, (1024, 1024), "float16", strides=(1024, 1))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", 8)
    by = T.launch_thread("blockIdx.y", 8)
    tx = T.launch_thread("threadIdx.x", 128)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(A[by * 128, bx * 128])
        T.writes()
        A_local = T.alloc_buffer((128, 128), scope="local.fragment")
        B_local = T.alloc_buffer((128, 128), scope="local.fragment")
        C_local = T.alloc_buffer((16, 128, 128), scope="local.fragment")
        T.copy(T.region(A[by * 128, bx * 128], 1, 128, 128), T.region(A_local[0, 0], 2, 128, 128), -1, T.bool(False), 0)
        T.comm_broadcast(A_local[0:128, 0:128], B_local[0:128, 0:128], -1, 0, 6, 2)
        T.comm_put(A_local[0:128, 0:128], B_local[0:128, 0:128], -1, 6, 11)
        T.comm_allgather(A_local[0:128, 0:128], C_local[0:16, 0:128, 0:128], 2, -1)"""

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype),):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_local = T.alloc_fragment([block_M, block_N], accum_dtype)
            B_local = T.alloc_fragment([block_M, block_N], accum_dtype)
            C_local = T.alloc_fragment([16, block_M, block_N], accum_dtype)
            T.copy(A[by * block_M, bx * block_N], A_local)

            T.comm.broadcast(A_local, B_local, (1, 2), direction="all")
            T.comm.put(A_local, B_local, (1, 2), (2, 3))
            T.comm.all_gather(A_local, C_local, direction="all")

    assert main.script() == func_str, "The generated script does not match the expected output."


@pytest.mark.parametrize("M, N, block_M, block_N, dtype, accum_dtype", [
    (1024, 1024, 128, 128, "float16", "float"),
])
def test_comm_broadcast_lower(M, N, block_M, block_N, dtype="float16", accum_dtype="float"):
    func_str = """# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A_handle: T.handle):
        T.func_attr({"target": T.target({"keys": ["cpu"], "kind": "llvm", "mattr": ["device_mesh_nrow_4", "device_mesh_ncol_4"], "mcpu": "sunmmio-a4e", "tag": ""})})
        A = T.match_buffer(A_handle, (1024, 1024), "float16", strides=(1024, 1))
        # with T.block("root"):
        bx = T.launch_thread("blockIdx.x", 8)
        by = T.launch_thread("blockIdx.y", 8)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        with T.block("tilelang_root"):
            T.reads(A[by * 128, bx * 128])
            T.writes()
            A_local = T.alloc_buffer((128, 128), scope="local.fragment")
            B_local = T.alloc_buffer((128, 128), scope="local.fragment")
            for i in T.parallel(128):
                for j in T.parallel(32):
                    for vec in T.vectorized(4):
                        A_local[i, j * 4 + vec] = T.Cast("float32", A[by * 128 + i, bx * 128 + (j * 4 + vec)])
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), A_local.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), B_local.data, 0, 16384, 2), 16384, 6, 1)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), B_local.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), B_local.data, 0, 16384, 2), 16384, 2, 0, 2)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), B_local.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), B_local.data, 0, 16384, 2), 16384, 6, 0, 2)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), B_local.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), B_local.data, 0, 16384, 2), 16384, 10, 0, 2)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), B_local.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), B_local.data, 0, 16384, 2), 16384, 14, 0, 2)"""

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype),):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_local = T.alloc_fragment([block_M, block_N], accum_dtype)
            B_local = T.alloc_fragment([block_M, block_N], accum_dtype)
            T.copy(A[by * block_M, bx * block_N], A_local)

            T.comm.broadcast(A_local, B_local, (1, 2), direction="all")

    mod = tvm.IRModule({'main': main})
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
        assert mod.script() == func_str, "The generated script does not match the expected output."


@pytest.mark.parametrize("M, N, block_M, block_N, dtype, accum_dtype", [
    (1024, 1024, 128, 128, "float16", "float"),
])
def test_comm_put_lower(M, N, block_M, block_N, dtype="float16", accum_dtype="float"):
    func_str = """# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A_handle: T.handle):
        T.func_attr({"target": T.target({"keys": ["cpu"], "kind": "llvm", "mattr": ["device_mesh_nrow_4", "device_mesh_ncol_4"], "mcpu": "sunmmio-a4e", "tag": ""})})
        A = T.match_buffer(A_handle, (1024, 1024), "float16", strides=(1024, 1))
        # with T.block("root"):
        bx = T.launch_thread("blockIdx.x", 8)
        by = T.launch_thread("blockIdx.y", 8)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        with T.block("tilelang_root"):
            T.reads(A[by * 128, bx * 128])
            T.writes()
            A_local = T.alloc_buffer((128, 128), scope="local.fragment")
            B_local = T.alloc_buffer((128, 128), scope="local.fragment")
            for i in T.parallel(128):
                for j in T.parallel(32):
                    for vec in T.vectorized(4):
                        A_local[i, j * 4 + vec] = T.Cast("float32", A[by * 128 + i, bx * 128 + (j * 4 + vec)])
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), A_local.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), B_local.data, 0, 16384, 2), 16384, 6, 1, 0, 1, 3)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), B_local.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), B_local.data, 0, 16384, 2), 16384, 7, 0, 0, 1, 2)"""

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype),):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_local = T.alloc_fragment([block_M, block_N], accum_dtype)
            B_local = T.alloc_fragment([block_M, block_N], accum_dtype)
            T.copy(A[by * block_M, bx * block_N], A_local)

            T.comm.put(A_local, B_local, (1, 2), (2, 3))

    mod = tvm.IRModule({'main': main})
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
        assert mod.script() == func_str, "The generated script does not match the expected output."


@pytest.mark.parametrize("M, N, block_M, block_N, dtype, accum_dtype", [
    (1024, 1024, 128, 128, "float16", "float"),
])
def test_comm_all_gather_lower(M, N, block_M, block_N, dtype="float16", accum_dtype="float"):
    func_str = """# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A_handle: T.handle):
        T.func_attr({"target": T.target({"keys": ["cpu"], "kind": "llvm", "mattr": ["device_mesh_nrow_4", "device_mesh_ncol_4"], "mcpu": "sunmmio-a4e", "tag": ""})})
        A = T.match_buffer(A_handle, (1024, 1024), "float16", strides=(1024, 1))
        # with T.block("root"):
        bx = T.launch_thread("blockIdx.x", 8)
        by = T.launch_thread("blockIdx.y", 8)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        with T.block("tilelang_root"):
            T.reads(A[by * 128, bx * 128])
            T.writes()
            A_local = T.alloc_buffer((128, 128), scope="local.fragment")
            C_local = T.alloc_buffer((16, 128, 128), scope="local.fragment")
            for i in T.parallel(128):
                for j in T.parallel(32):
                    for vec in T.vectorized(4):
                        A_local[i, j * 4 + vec] = T.Cast("float32", A[by * 128 + i, bx * 128 + (j * 4 + vec)])
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), A_local.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 0, 16384, 2), 16384, 0, 0)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), A_local.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 16384, 16384, 2), 16384, 1, 0)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), A_local.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 32768, 16384, 2), 16384, 2, 0)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), A_local.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 49152, 16384, 2), 16384, 3, 0)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), A_local.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 65536, 16384, 2), 16384, 4, 0)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), A_local.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 81920, 16384, 2), 16384, 5, 0)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), A_local.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 98304, 16384, 2), 16384, 6, 0)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), A_local.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 114688, 16384, 2), 16384, 7, 0)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), A_local.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 131072, 16384, 2), 16384, 8, 0)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), A_local.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 147456, 16384, 2), 16384, 9, 0)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), A_local.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 163840, 16384, 2), 16384, 10, 0)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), A_local.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 180224, 16384, 2), 16384, 11, 0)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), A_local.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 196608, 16384, 2), 16384, 12, 0)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), A_local.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 212992, 16384, 2), 16384, 13, 0)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), A_local.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 229376, 16384, 2), 16384, 14, 0)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), A_local.data, 0, 16384, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 245760, 16384, 2), 16384, 15, 0)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 0, 65536, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 0, 65536, 2), 65536, 0, 1, 0)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 65536, 65536, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 65536, 65536, 2), 65536, 4, 1, 1)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 131072, 65536, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 131072, 65536, 2), 65536, 8, 1, 2)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 196608, 65536, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 196608, 65536, 2), 65536, 12, 1, 3)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 0, 65536, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 0, 65536, 2), 65536, 1, 1, 0)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 65536, 65536, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 65536, 65536, 2), 65536, 5, 1, 1)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 131072, 65536, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 131072, 65536, 2), 65536, 9, 1, 2)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 196608, 65536, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 196608, 65536, 2), 65536, 13, 1, 3)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 0, 65536, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 0, 65536, 2), 65536, 2, 1, 0)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 65536, 65536, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 65536, 65536, 2), 65536, 6, 1, 1)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 131072, 65536, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 131072, 65536, 2), 65536, 10, 1, 2)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 196608, 65536, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 196608, 65536, 2), 65536, 14, 1, 3)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 0, 65536, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 0, 65536, 2), 65536, 3, 1, 0)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 65536, 65536, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 65536, 65536, 2), 65536, 7, 1, 1)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 131072, 65536, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 131072, 65536, 2), 65536, 11, 1, 2)
            T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 196608, 65536, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 196608, 65536, 2), 65536, 15, 1, 3)"""

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype),):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_local = T.alloc_fragment([block_M, block_N], accum_dtype)
            C_local = T.alloc_fragment([16, block_M, block_N], accum_dtype)
            T.copy(A[by * block_M, bx * block_N], A_local)

            T.comm.all_gather(A_local, C_local, direction="all")

    mod = tvm.IRModule({'main': main})
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
        assert mod.script() == func_str, "The generated script does not match the expected output."


if __name__ == "__main__":
    test_comm_python_api(1024, 1024, 128, 128, "float16", "float")
    test_comm_broadcast_lower(1024, 1024, 128, 128, "float16", "float")
    test_comm_put_lower(1024, 1024, 128, 128, "float16", "float")
    test_comm_all_gather_lower(1024, 1024, 128, 128, "float16", "float")
