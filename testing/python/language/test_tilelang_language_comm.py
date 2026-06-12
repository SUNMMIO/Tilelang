import pytest

import tilelang
import tilelang.language as T

from tilelang import tvm as tvm
from tilelang.utils.target import determine_target


def _broadcast_lines(script):
    return [line.strip() for line in script.splitlines() if "T.broadcast_(" in line]


def _collect_calls(func, op_name):
    calls = []
    op = tvm.tir.op.Op.get(op_name)

    def visit(node):
        if isinstance(node, tvm.tir.Call) and node.op.same_as(op):
            calls.append(node)

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return calls


def _broadcast_line(src, dst, direction, mask, core, src_offset=0):
    return f"T.broadcast_({src}, {dst}, {direction}, T.int64({mask}), {src_offset}, {core})"


def _broadcast_line_no_core(src, dst, direction, mask, src_offset=0):
    return f"T.broadcast_({src}, {dst}, {direction}, {mask}, {src_offset})"


_ROW_MASK_BX = "T.int64(15)"

_COL_MASK_BX = "T.int64(15)"


def _expected_axis0_all_lines(buffer):
    src = "T.region(A_shared[0, 0], 1, 128, 128)"
    return [
        _broadcast_line_no_core(src, f"T.region({buffer}[bx * 128, 0], 2, 128, 128)", 0, _ROW_MASK_BX),
        _broadcast_line_no_core(
            f"T.region({buffer}[bx // 4 * 512, 0], 1, 512, 128)",
            f"T.region({buffer}[bx // 4 * 512, 0], 2, 512, 128)",
            1,
            _COL_MASK_BX,
        ),
    ]


def _expected_axis_last_horizontal_lines(buffer):
    src = "T.region(A_shared[0, 0], 1, 128, 128)"
    return [
        _broadcast_line_no_core(
            src,
            f"T.region({buffer}[0, bx % 4 * 128], 2, 128, 128)",
            0,
            _ROW_MASK_BX,
        )
    ]


def _expected_axis_last_all_lines(buffer):
    src = "T.region(A_shared[0, 0], 1, 128, 128)"
    return [
        _broadcast_line_no_core(src, f"T.region({buffer}[0, bx * 128], 2, 128, 128)", 0, _ROW_MASK_BX),
        _broadcast_line_no_core(
            f"T.region({buffer}[0, bx // 4 * 512], 1, 128, 512)",
            f"T.region({buffer}[0, bx // 4 * 512], 2, 128, 512)",
            1,
            _COL_MASK_BX,
        ),
    ]


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
        T.comm_allgather(A_shared[0:128, 0:128], C_shared[0:16, 0:128, 0:128], 2, -1, -1, bx)
        T.comm_allgather(A_shared[0:128, 0:128], R0_shared[0:2048, 0:128], 2, -1, 0, bx)
        T.comm_allgather(A_shared[0:128, 0:128], R1_shared[0:128, 0:2048], 2, -1, 1, bx)""".strip()

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


def test_comm_buffer_like_region_python_api():
    @T.prim_func
    def main(A: T.Tensor((128, 128), "float32")):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared([128, 128], "float32", scope="shared.rsram")
            B_shared = T.alloc_shared([128, 128], "float32", scope="shared.rsram")
            C_shared = T.alloc_shared([4, 64, 64], "float32", scope="shared.rsram")
            Out_shared = T.alloc_shared([128], "float32", scope="shared.rsram")
            T.copy(A, A_shared)

            T.comm.broadcast(A_shared[8:72, 16:80], B_shared[24:88, 32:96], (0, 0), direction="h")
            T.comm.put(A_shared[0:64, 0:64], B_shared[32:96, 32:96], (0, 0), (0, 1))
            T.comm.all_gather(A_shared[8:72, 16:80], C_shared[0:4, 0:64, 0:64], direction="h")
            T.comm.all_reduce(A_shared[8:72, 16:80], Out_shared[32:96], "sum", "h", dim=1)

    script = main.script()
    assert "T.comm_broadcast(A_shared[8:72, 16:80], B_shared[24:88, 32:96], -1, 0, 0)" in script
    assert "T.comm_put(A_shared[0:64, 0:64], B_shared[32:96, 32:96], -1, 0, 1)" in script
    assert "T.comm_allgather(A_shared[8:72, 16:80], C_shared[0:4, 0:64, 0:64], 0, -1, -1, bx)" in script
    assert (
        'T.comm_allreduce(A_shared[8:72, 16:80], Out_shared[32:96], buffer[0:4, 0:64], buffer_1[0:4, 0:64], "sum", 0, 1, T.bool(True), bx)'
    ) in script


def test_comm_dynamic_core_python_api():
    @T.prim_func
    def main(A: T.Tensor((128, 128), "float32")):
        with T.Kernel(16, threads=128) as bx:
            A_shared = T.alloc_shared([128, 128], "float32", scope="shared.rsram")
            B_shared = T.alloc_shared([128, 128], "float32", scope="shared.rsram")
            T.copy(A, A_shared)

            T.comm.broadcast(A_shared, B_shared, bx, direction="h")
            T.comm.put(A_shared, B_shared, bx, (bx + 1) % 16)

    broadcast_calls = _collect_calls(main, "tl.tileop.comm_broadcast")
    put_calls = _collect_calls(main, "tl.tileop.comm_put")
    assert len(broadcast_calls) == 1
    assert len(put_calls) == 1
    assert isinstance(broadcast_calls[0].args[3], tvm.tir.Var)
    assert tvm.ir.structural_equal(broadcast_calls[0].args[3], put_calls[0].args[3])
    assert tvm.ir.structural_equal(put_calls[0].args[4], (put_calls[0].args[3] + 1) % 16)


def test_comm_dynamic_core_lower():
    @T.prim_func
    def main(A: T.Tensor((128, 128), "float32")):
        with T.Kernel(16, threads=128) as bx:
            A_shared = T.alloc_shared([128, 128], "float32", scope="shared.rsram")
            B_shared = T.alloc_shared([128, 128], "float32", scope="shared.rsram")
            T.copy(A, A_shared)

            T.comm.broadcast(A_shared, B_shared, bx, direction="all")
            T.comm.put(A_shared, B_shared, bx, (bx + 1) % 16)

    mod = tvm.IRModule({"main": main})
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.LowerTileOp()(mod)

    broadcast_calls = _collect_calls(mod["main"], "tl.broadcast_")
    if_nodes = []

    def visit(node):
        if isinstance(node, tvm.tir.IfThenElse):
            if_nodes.append(node)

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, visit)
    assert len(broadcast_calls) == 9
    assert all(len(call.args) == 6 for call in broadcast_calls)
    assert if_nodes, "Dynamic put lowering should emit runtime routing branches."


@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype, accum_dtype",
    [
        (1024, 1024, 128, 128, "float16", "float"),
    ],
)
def test_comm_broadcast_lower(M, N, block_M, block_N, dtype, accum_dtype):
    expected = [
        "T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 1, T.int64(15), 0, 6)",
        "T.broadcast_(T.region(B_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 0, T.int64(15), 0, 2)",
        "T.broadcast_(T.region(B_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 0, T.int64(15), 0, 6)",
        "T.broadcast_(T.region(B_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 0, T.int64(15), 0, 10)",
        "T.broadcast_(T.region(B_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 0, T.int64(15), 0, 14)",
    ]

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
        assert _broadcast_lines(mod.script()) == expected, "The generated script does not match the expected output."


@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype",
    [
        (1024, 1024, 128, 128, "float16"),
    ],
)
def test_comm_broadcast_lower_custom_mesh(M, N, block_M, block_N, dtype):
    expected = [
        "T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 1, T.int64(3), 0, 5)",
        "T.broadcast_(T.region(B_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 0, T.int64(7), 0, 2)",
        "T.broadcast_(T.region(B_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 0, T.int64(7), 0, 5)",
    ]

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
        assert _broadcast_lines(mod.script()) == expected, "The generated script does not match the expected output."


@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype, accum_dtype",
    [
        (1024, 1024, 128, 128, "float16", "float"),
    ],
)
def test_comm_put_lower(M, N, block_M, block_N, dtype, accum_dtype):
    expected = [
        "T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 1, T.int64(4), 0, 6)",
        "T.broadcast_(T.region(B_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 0, T.int64(8), 0, 10)",
    ]

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
        assert _broadcast_lines(mod.script()) == expected, "The generated script does not match the expected output."


@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype, accum_dtype",
    [
        (1024, 1024, 128, 128, "float16", "float"),
    ],
)
def test_comm_all_gather_lower(M, N, block_M, block_N, dtype, accum_dtype):
    expected = [
        _broadcast_line_no_core(
            "T.region(A_shared[0, 0], 1, 128, 128)",
            "T.region(C_shared[bx, 0, 0], 2, 1, 128, 128)",
            0,
            _ROW_MASK_BX,
        ),
        _broadcast_line_no_core(
            "T.region(C_shared[bx // 4 * 4, 0, 0], 1, 4, 128, 128)",
            "T.region(C_shared[bx // 4 * 4, 0, 0], 2, 4, 128, 128)",
            1,
            _COL_MASK_BX,
        ),
    ]

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
        assert _broadcast_lines(mod.script()) == expected, "The generated script does not match the expected output."


@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype",
    [
        (1024, 1024, 128, 128, "float16"),
    ],
)
def test_comm_all_gather_axis0_all_lower(M, N, block_M, block_N, dtype):
    expected = _expected_axis0_all_lines("R_shared")

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
        assert _broadcast_lines(mod.script()) == expected, "The generated script does not match the expected output."


@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype",
    [
        (1024, 1024, 128, 128, "float16"),
    ],
)
def test_comm_all_gather_axis_last_horizontal_lower(M, N, block_M, block_N, dtype):
    expected = _expected_axis_last_horizontal_lines("R_shared")

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
        assert _broadcast_lines(mod.script()) == expected, "The generated script does not match the expected output."


@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype",
    [
        (1024, 1024, 128, 128, "float16"),
    ],
)
def test_comm_all_gather_axis_last_all_lower(M, N, block_M, block_N, dtype):
    expected = _expected_axis_last_all_lines("R_shared")

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
        assert _broadcast_lines(mod.script()) == expected, "The generated script does not match the expected output."


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
