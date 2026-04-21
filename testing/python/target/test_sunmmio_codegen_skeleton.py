import pytest
import re
import numpy as np
import os
import json
import tilelang.language as T
import tilelang.testing
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target

# PRINT = os.getenv("SUNMMIO_TEST_PRINT", "false").lower() in ("1", "true", "yes", "on")
PRINT = True


def _maybe_print_kernel_and_mlir(func, src: str):
    if not PRINT:
        return
    print("=== TVM Kernel ===")
    if hasattr(func, "script"):
        print(func.script())
    else:
        print(func)
    print("=== SunMMIO Text MLIR ===")
    print(src)


# SunMMIO codegen traversal contract:
# Supported now:
# - core scalar arithmetic/control flow
# - For, IfThenElse
# - Allocate, AllocateConst
# - DeclBuffer, BufferRealize, BufferLoad, BufferStore
# - Block, BlockRealize
# - Ramp, Broadcast
# - TileLang/SunMMIO intrinsic Call
#
# Intentionally unsupported:
# - While
# - Shuffle
# - legacy Load
# - Any


def simple_add_kernel(n: int = 16):
    @T.prim_func
    def main(
        A: T.Tensor((n,), dtype=T.float32),
        B: T.Tensor((n,), dtype=T.float32),
    ):
        with T.Kernel(1, 1) as (bx, by):
            for i in T.serial(n):
                B[i] = A[i] + T.float32(1.0)

    return main


# Additional kernel builders copied from custom compile smoke tests
# (`test_with_custom_compile_2.py`) for richer SunMMIO codegen inputs.
def flashattn_kernel(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    block_M=64,
    block_N=64,
    num_stages=1,
    threads=1,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    shape = [batch, seq_len, heads, dim]
    dtype = T.float16
    accum_dtype = T.float16

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (
            bx,
            by,
            bz,
        ):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_shared([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_shared([block_M, block_N], accum_dtype, scope="shared.asram")
            acc_o = T.alloc_shared([block_M, dim], accum_dtype, scope="shared.rsram")
            scores_max = T.alloc_shared([block_M], accum_dtype)
            scores_max_prev = T.alloc_shared([block_M], accum_dtype)
            scores_scale = T.alloc_shared([block_M], accum_dtype)
            scores_sum = T.alloc_shared([block_M], accum_dtype)
            logsum = T.alloc_shared([block_M], accum_dtype)

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N)) if is_causal else T.ceildiv(seq_len, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            bx * block_M + i >= k * block_N + j,
                            0,
                            -T.infinity(acc_s.dtype),
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_len, -T.infinity(acc_s.dtype), 0)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)

                for i in T.serial(0, block_M):
                    scores_max_prev[i] = scores_max[i]
                    scores_max[i] = -T.infinity(accum_dtype)
                    for j in T.serial(0, block_N):
                        scores_max[i] = T.max(scores_max[i], acc_s[i, j])
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

                for i in T.serial(0, block_M):
                    scores_sum[i] = T.cast(0, accum_dtype)
                    for j in T.serial(0, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        scores_sum[i] = scores_sum[i] + acc_s[i, j]

                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main


def func_comm_kernel(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    mesh_device_config = (4, 4)

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.MeshShardingPolicy(x=1, y=0), mesh_device_config, dtype),
        B: T.MeshTensor((K, N), T.MeshShardingPolicy(x=1, y=0), mesh_device_config, dtype),
        C: T.MeshTensor((M, N), T.MeshShardingPolicy(x=1, y=0), mesh_device_config, accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), dtype=dtype)
            A_remote_1 = T.alloc_shared((block_M, block_K), dtype=dtype)
            A_remote_2 = T.alloc_shared((block_M, block_K), dtype=dtype)
            A_remote_3 = T.alloc_shared((block_M, block_K), dtype=dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype=dtype)
            B_remote_1 = T.alloc_shared((block_K, block_N), dtype=dtype)
            B_remote_2 = T.alloc_shared((block_K, block_N), dtype=dtype)
            B_remote_3 = T.alloc_shared((block_K, block_N), dtype=dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype=accum_dtype)
            C_allgather_1 = T.alloc_shared((16, block_M, block_N), dtype=accum_dtype)
            C_allgather_2 = T.alloc_shared((4, block_M, block_N), dtype=accum_dtype)
            C_allgather_3 = T.alloc_shared((4, block_M, block_N), dtype=accum_dtype)

            T.clear(A_shared)
            T.clear(B_shared)
            T.clear(C_shared)
            T.comm.broadcast(A_shared, A_remote_1, (0, 0), direction="all")
            T.comm.broadcast(A_shared, A_remote_2, (0, 0), direction="h")
            T.comm.broadcast(A_shared, A_remote_3, (0, 0), direction="v")
            T.comm.put(B_shared, B_remote_1, (1, 2), (2, 3))
            T.comm.put(B_shared, B_remote_2, (1, 2), (1, 3))
            T.comm.put(B_shared, B_remote_3, (1, 2), (3, 2))
            T.comm.all_gather(C_shared, C_allgather_1, direction="all")
            T.comm.all_gather(C_shared, C_allgather_2, direction="h")
            T.comm.all_gather(C_shared, C_allgather_3, direction="v")

    return main


def func_sync_kernel(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((M, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=1) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.asram")
            B_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.wsram")
            C_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.rsram")
            D_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.rsram")
            E_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.rsram")

            T.gemm(A_shared, B_shared, C_shared)
            if by <= 2:
                T.copy(C_shared, D_shared)

            if bx <= 2:
                T.clear(D_shared)

            for i in range(5):
                C_shared[i, 0] = C_shared[i, 0] + 1.0

            for _i in range(10):
                T.comm.broadcast(D_shared, E_shared, (0, 0), direction="h")
                E_shared[0, 0] = E_shared[0, 0] + 1.0
                T.comm.broadcast(E_shared, D_shared, (0, 0), direction="h")

    return kernel


def build_sunmmio_module_without_compile(func):
    target = determine_target("Sunmmio", return_object=True)
    func = func.with_attr("global_symbol", "main")
    func = func.with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
    mod = tvm.IRModule({"main": func})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    return builder(mod, target, "text")


def build_sunmmio_source_without_compile(func):
    src = build_sunmmio_module_without_compile(func).inspect_source()
    _maybe_print_kernel_and_mlir(func, src)
    return src


def build_sunmmio_source_from_stmt(stmt):
    target = determine_target("Sunmmio", return_object=True)
    func = tvm.tir.PrimFunc([], stmt)
    func = func.with_attr("global_symbol", "main")
    func = func.with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
    mod = tvm.IRModule({"main": func})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    src = builder(mod, target, "text").inspect_source()
    _maybe_print_kernel_and_mlir(func, src)
    return src


def test_sunmmio_codegen_without_compile_emits_mlir_source():
    src = build_sunmmio_source_without_compile(simple_add_kernel())
    assert "module {" in src
    assert "func.func @main" in src
    assert "scf.for" in src
    assert "memref.load" in src
    assert "arith.addf" in src
    assert "memref.store" in src
    assert "return" in src


def test_sunmmio_codegen_no_placeholder_summary_text():
    src = build_sunmmio_source_without_compile(simple_add_kernel())
    assert "sunmmio.traversal_summary" not in src
    assert "status: traversal_only_no_emission" not in src


def test_sunmmio_codegen_emits_multidim_store_indices():
    @T.prim_func
    def main(A: T.Tensor((4, 4), dtype=T.float32), B: T.Tensor((4, 4), dtype=T.float32)):
        with T.attr(0, "sunmmio.test_attr", 7):
            for i, j in T.grid(2, 3):
                with T.block("B0"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    T.reads(A[vi, vj])
                    T.writes(B[vi, vj])
                    B[vi, vj] = A[vi, vj] + T.float32(1.0)

    src = build_sunmmio_source_without_compile(main)
    assert "scf.for" in src
    assert "memref.store" in src
    assert re.search(r"memref\.store .*?\[[^,\]]+,\s*[^,\]]+\]", src), src


def test_sunmmio_codegen_classifies_sunmmio_intrinsic_calls():
    dma_call = tvm.tir.Call("handle", tvm.ir.Op.get("tl.dma_copy"), [])
    mma_call = tvm.tir.Call("handle", tvm.ir.Op.get("tl.mma_sunmmio"), [])
    body = tvm.tir.SeqStmt([tvm.tir.Evaluate(dma_call), tvm.tir.Evaluate(mma_call)])
    src = build_sunmmio_source_from_stmt(body)
    assert 'sunmmio.call @"tl.dma_copy"(' in src
    assert 'sunmmio.call @"tl.mma_sunmmio"(' in src
    assert 'category = "sunmmio_intrinsic"' in src


def test_sunmmio_codegen_block_predicate_emits_control_flow():
    @T.prim_func
    def main(A: T.Tensor((8,), dtype=T.float32), B: T.Tensor((8,), dtype=T.float32)):
        for i in T.serial(8):
            with T.block("blk"):
                vi = T.axis.spatial(8, i)
                T.where(vi < 4)
                T.reads(A[vi])
                T.writes(B[vi])
                B[vi] = A[vi] + T.float32(1.0)

    src = build_sunmmio_source_without_compile(main)
    assert "scf.if" in src
    assert "memref.store" in src


def test_sunmmio_codegen_block_annotations_are_traversed():
    @T.prim_func
    def main(A: T.Tensor((8,), dtype=T.float32), B: T.Tensor((8,), dtype=T.float32)):
        for i in T.serial(8):
            with T.block("blk"):
                vi = T.axis.spatial(8, i)
                T.block_attr({"sunmmio.anno_expr": vi + 1, "sunmmio.anno_const": 7})
                T.reads(A[vi])
                T.writes(B[vi])
                B[vi] = A[vi] + T.float32(1.0)

    src = build_sunmmio_source_without_compile(main)
    assert "func.func @main" in src
    assert "arith.addi" in src
    assert "memref.store" in src


def test_sunmmio_codegen_allocate_const_is_handled():
    dev = tvm.cpu(0)
    dtype = "float32"
    shape = (4,)
    cbuf = tvm.tir.decl_buffer(shape, dtype, name="C")
    data = tvm.runtime.tensor(np.array([1, 2, 3, 4], dtype=dtype), device=dev)
    body = tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0))
    stmt = tvm.tir.AllocateConst(cbuf.data, dtype, shape, data, body)
    src = build_sunmmio_source_from_stmt(stmt)
    assert "memref.alloc" in src


def test_sunmmio_codegen_block_alloc_buffers_are_handled():
    @T.prim_func
    def main(A: T.Tensor((8,), dtype=T.float32), B: T.Tensor((8,), dtype=T.float32)):
        for i in T.serial(8):
            with T.block("blk"):
                vi = T.axis.spatial(8, i)
                tmp = T.alloc_buffer((8,), dtype=T.float32, scope="local")
                T.reads(A[vi])
                T.writes(B[vi], tmp[vi])
                tmp[vi] = A[vi]
                B[vi] = tmp[vi] + T.float32(1.0)

    src = build_sunmmio_source_without_compile(main)
    assert "memref.alloc" in src
    assert "memref.store" in src


def test_sunmmio_codegen_unsupported_stmt_fails_loudly():
    cond = tvm.tir.LT(tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 1))
    body = tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0))
    stmt = tvm.tir.While(cond, body)
    target = determine_target("Sunmmio", return_object=True)
    func = tvm.tir.PrimFunc([], stmt)
    func = func.with_attr("global_symbol", "main")
    func = func.with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
    mod = tvm.IRModule({"main": func})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(Exception, match="CodeGenTileLangSunMMIO unsupported stmt: tir.While"):
        builder(mod, target, "text")


def test_sunmmio_codegen_shuffle_fails_loudly():
    shuffle = tvm.tir.Shuffle(
        [tvm.tir.Broadcast(tvm.tir.IntImm("int32", 7), 4)],
        [tvm.tir.IntImm("int32", 0)],
    )
    stmt = tvm.tir.Evaluate(shuffle)
    target = determine_target("Sunmmio", return_object=True)
    func = tvm.tir.PrimFunc([], stmt)
    func = func.with_attr("global_symbol", "main")
    func = func.with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
    mod = tvm.IRModule({"main": func})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(Exception, match="CodeGenTileLangSunMMIO unsupported expr: tir.Shuffle"):
        builder(mod, target, "text")


def test_sunmmio_codegen_ramp_is_supported():
    ramp = tvm.tir.Ramp(tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 1), 4)
    stmt = tvm.tir.Evaluate(ramp)
    src = build_sunmmio_source_from_stmt(stmt)
    assert "sunmmio.ramp" in src
    assert "vector<4xi32>" in src


def test_sunmmio_codegen_broadcast_is_supported():
    bcast = tvm.tir.Broadcast(tvm.tir.FloatImm("float32", 1.5), 4)
    stmt = tvm.tir.Evaluate(bcast)
    src = build_sunmmio_source_from_stmt(stmt)
    assert "vector.broadcast" in src
    assert "vector<4xf32>" in src


def test_sunmmio_codegen_compile_path_not_implemented():
    target = determine_target("Sunmmio", return_object=True)
    func = simple_add_kernel().with_attr("global_symbol", "main")
    func = func.with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
    mod = tvm.IRModule({"main": func})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio")
    with pytest.raises(Exception, match="not implemented yet"):
        builder(mod, target)


@pytest.mark.parametrize(
    "kernel_name,kernel_factory",
    [
        ("simple_add", lambda: simple_add_kernel()),
        (
            "flashattn",
            lambda: flashattn_kernel(1, 1, 128, 32, False, block_M=32, block_N=32, num_stages=1, threads=1),
        ),
        ("func_sync", lambda: func_sync_kernel(256, 256, 256, 64, 64, 64)),
    ],
)
def test_sunmmio_codegen_coverage_report_has_no_missing_entries(tmp_path, kernel_name, kernel_factory):
    report_path = tmp_path / f"codegen_coverage_{kernel_name}.json"
    old_path = os.environ.get("TL_SUNMMIO_CODEGEN_COVERAGE_PATH")
    old_strict = os.environ.get("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT")
    os.environ["TL_SUNMMIO_CODEGEN_COVERAGE_PATH"] = str(report_path)
    os.environ["TL_SUNMMIO_CODEGEN_COVERAGE_STRICT"] = "1"
    try:
        _ = build_sunmmio_source_without_compile(kernel_factory())
    finally:
        if old_path is None:
            os.environ.pop("TL_SUNMMIO_CODEGEN_COVERAGE_PATH", None)
        else:
            os.environ["TL_SUNMMIO_CODEGEN_COVERAGE_PATH"] = old_path
        if old_strict is None:
            os.environ.pop("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT", None)
        else:
            os.environ["TL_SUNMMIO_CODEGEN_COVERAGE_STRICT"] = old_strict

    assert report_path.exists(), f"coverage report not generated: {report_path}"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required_keys = [
        "expected_node_types",
        "visited_node_types",
        "missing_node_types",
        "expected_call_ops",
        "visited_call_ops",
        "missing_call_ops",
    ]
    for key in required_keys:
        assert key in report, f"missing coverage key: {key}"
    assert report["missing_node_types"] == []
    assert report["missing_call_ops"] == []
    expected_nodes = set(report["expected_node_types"])
    visited_nodes = set(report["visited_node_types"])
    missing_nodes = set(report["missing_node_types"])
    expected_calls = set(report["expected_call_ops"])
    visited_calls = set(report["visited_call_ops"])
    missing_calls = set(report["missing_call_ops"])
    assert expected_nodes - visited_nodes == missing_nodes
    assert expected_calls - visited_calls == missing_calls
    assert "tir.For" in expected_nodes
    assert "tir.For" in visited_nodes


if __name__ == "__main__":
    tilelang.testing.main()
import pytest
import re
import numpy as np
import os
import json
import tilelang.language as T
import tilelang.testing
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target

# PRINT = os.getenv("SUNMMIO_TEST_PRINT", "false").lower() in ("1", "true", "yes", "on")
PRINT = True


def _maybe_print_kernel_and_mlir(func, src: str):
    if not PRINT:
        return
    print("=== TVM Kernel ===")
    if hasattr(func, "script"):
        print(func.script())
    else:
        print(func)
    print("=== SunMMIO Text MLIR ===")
    print(src)


# SunMMIO codegen traversal contract:
# Supported now:
# - core scalar arithmetic/control flow
# - For, IfThenElse
# - Allocate, AllocateConst
# - DeclBuffer, BufferRealize, BufferLoad, BufferStore
# - Block, BlockRealize
# - Ramp, Broadcast
# - TileLang/SunMMIO intrinsic Call
#
# Intentionally unsupported:
# - While
# - Shuffle
# - legacy Load
# - Any


def simple_add_kernel(n: int = 16):
    @T.prim_func
    def main(
        A: T.Tensor((n,), dtype=T.float32),
        B: T.Tensor((n,), dtype=T.float32),
    ):
        with T.Kernel(1, 1) as (bx, by):
            for i in T.serial(n):
                B[i] = A[i] + T.float32(1.0)

    return main


# Additional kernel builders copied from custom compile smoke tests
# (`test_with_custom_compile_2.py`) for richer SunMMIO codegen inputs.
def flashattn_kernel(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    block_M=64,
    block_N=64,
    num_stages=1,
    threads=1,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    shape = [batch, seq_len, heads, dim]
    dtype = T.float16
    accum_dtype = T.float16

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (
            bx,
            by,
            bz,
        ):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_shared([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_shared([block_M, block_N], accum_dtype, scope="shared.asram")
            acc_o = T.alloc_shared([block_M, dim], accum_dtype, scope="shared.rsram")
            scores_max = T.alloc_shared([block_M], accum_dtype)
            scores_max_prev = T.alloc_shared([block_M], accum_dtype)
            scores_scale = T.alloc_shared([block_M], accum_dtype)
            scores_sum = T.alloc_shared([block_M], accum_dtype)
            logsum = T.alloc_shared([block_M], accum_dtype)

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N)) if is_causal else T.ceildiv(seq_len, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            bx * block_M + i >= k * block_N + j,
                            0,
                            -T.infinity(acc_s.dtype),
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_len, -T.infinity(acc_s.dtype), 0)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)

                for i in T.serial(0, block_M):
                    scores_max_prev[i] = scores_max[i]
                    scores_max[i] = -T.infinity(accum_dtype)
                    for j in T.serial(0, block_N):
                        scores_max[i] = T.max(scores_max[i], acc_s[i, j])
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

                for i in T.serial(0, block_M):
                    scores_sum[i] = T.cast(0, accum_dtype)
                    for j in T.serial(0, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        scores_sum[i] = scores_sum[i] + acc_s[i, j]

                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main


def func_comm_kernel(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    mesh_device_config = (4, 4)

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.MeshShardingPolicy(x=1, y=0), mesh_device_config, dtype),
        B: T.MeshTensor((K, N), T.MeshShardingPolicy(x=1, y=0), mesh_device_config, dtype),
        C: T.MeshTensor((M, N), T.MeshShardingPolicy(x=1, y=0), mesh_device_config, accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), dtype=dtype)
            A_remote_1 = T.alloc_shared((block_M, block_K), dtype=dtype)
            A_remote_2 = T.alloc_shared((block_M, block_K), dtype=dtype)
            A_remote_3 = T.alloc_shared((block_M, block_K), dtype=dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype=dtype)
            B_remote_1 = T.alloc_shared((block_K, block_N), dtype=dtype)
            B_remote_2 = T.alloc_shared((block_K, block_N), dtype=dtype)
            B_remote_3 = T.alloc_shared((block_K, block_N), dtype=dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype=accum_dtype)
            C_allgather_1 = T.alloc_shared((16, block_M, block_N), dtype=accum_dtype)
            C_allgather_2 = T.alloc_shared((4, block_M, block_N), dtype=accum_dtype)
            C_allgather_3 = T.alloc_shared((4, block_M, block_N), dtype=accum_dtype)

            T.clear(A_shared)
            T.clear(B_shared)
            T.clear(C_shared)
            T.comm.broadcast(A_shared, A_remote_1, (0, 0), direction="all")
            T.comm.broadcast(A_shared, A_remote_2, (0, 0), direction="h")
            T.comm.broadcast(A_shared, A_remote_3, (0, 0), direction="v")
            T.comm.put(B_shared, B_remote_1, (1, 2), (2, 3))
            T.comm.put(B_shared, B_remote_2, (1, 2), (1, 3))
            T.comm.put(B_shared, B_remote_3, (1, 2), (3, 2))
            T.comm.all_gather(C_shared, C_allgather_1, direction="all")
            T.comm.all_gather(C_shared, C_allgather_2, direction="h")
            T.comm.all_gather(C_shared, C_allgather_3, direction="v")

    return main


def func_sync_kernel(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((M, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=1) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.asram")
            B_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.wsram")
            C_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.rsram")
            D_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.rsram")
            E_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.rsram")

            T.gemm(A_shared, B_shared, C_shared)
            if by <= 2:
                T.copy(C_shared, D_shared)

            if bx <= 2:
                T.clear(D_shared)

            for i in range(5):
                C_shared[i, 0] = C_shared[i, 0] + 1.0

            for _i in range(10):
                T.comm.broadcast(D_shared, E_shared, (0, 0), direction="h")
                E_shared[0, 0] = E_shared[0, 0] + 1.0
                T.comm.broadcast(E_shared, D_shared, (0, 0), direction="h")

    return kernel


def build_sunmmio_module_without_compile(func):
    target = determine_target("Sunmmio", return_object=True)
    func = func.with_attr("global_symbol", "main")
    func = func.with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
    mod = tvm.IRModule({"main": func})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    return builder(mod, target)


def build_sunmmio_source_without_compile(func):
    src = build_sunmmio_module_without_compile(func).inspect_source()
    _maybe_print_kernel_and_mlir(func, src)
    return src


def build_sunmmio_source_from_stmt(stmt):
    target = determine_target("Sunmmio", return_object=True)
    func = tvm.tir.PrimFunc([], stmt)
    func = func.with_attr("global_symbol", "main")
    func = func.with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
    mod = tvm.IRModule({"main": func})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    src = builder(mod, target).inspect_source()
    _maybe_print_kernel_and_mlir(func, src)
    return src


def test_sunmmio_codegen_without_compile_emits_mlir_source():
    src = build_sunmmio_source_without_compile(simple_add_kernel())
    assert "module {" in src
    assert "func.func @main" in src
    assert "scf.for" in src
    assert "memref.load" in src
    assert "arith.addf" in src
    assert "memref.store" in src
    assert "return" in src


def test_sunmmio_codegen_no_placeholder_summary_text():
    src = build_sunmmio_source_without_compile(simple_add_kernel())
    assert "sunmmio.traversal_summary" not in src
    assert "status: traversal_only_no_emission" not in src


def test_sunmmio_codegen_emits_multidim_store_indices():
    @T.prim_func
    def main(A: T.Tensor((4, 4), dtype=T.float32), B: T.Tensor((4, 4), dtype=T.float32)):
        with T.attr(0, "sunmmio.test_attr", 7):
            for i, j in T.grid(2, 3):
                with T.block("B0"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    T.reads(A[vi, vj])
                    T.writes(B[vi, vj])
                    B[vi, vj] = A[vi, vj] + T.float32(1.0)

    src = build_sunmmio_source_without_compile(main)
    assert "scf.for" in src
    assert "memref.store" in src
    assert re.search(r"memref\.store .*?\[[^,\]]+,\s*[^,\]]+\]", src), src


def test_sunmmio_codegen_classifies_sunmmio_intrinsic_calls():
    dma_call = tvm.tir.Call("handle", tvm.ir.Op.get("tl.dma_copy"), [])
    mma_call = tvm.tir.Call("handle", tvm.ir.Op.get("tl.mma_sunmmio"), [])
    body = tvm.tir.SeqStmt([tvm.tir.Evaluate(dma_call), tvm.tir.Evaluate(mma_call)])
    src = build_sunmmio_source_from_stmt(body)
    assert 'sunmmio.call @"tl.dma_copy"(' in src
    assert 'sunmmio.call @"tl.mma_sunmmio"(' in src
    assert 'category = "sunmmio_intrinsic"' in src


def test_sunmmio_codegen_block_predicate_emits_control_flow():
    @T.prim_func
    def main(A: T.Tensor((8,), dtype=T.float32), B: T.Tensor((8,), dtype=T.float32)):
        for i in T.serial(8):
            with T.block("blk"):
                vi = T.axis.spatial(8, i)
                T.where(vi < 4)
                T.reads(A[vi])
                T.writes(B[vi])
                B[vi] = A[vi] + T.float32(1.0)

    src = build_sunmmio_source_without_compile(main)
    assert "scf.if" in src
    assert "memref.store" in src


def test_sunmmio_codegen_block_annotations_are_traversed():
    @T.prim_func
    def main(A: T.Tensor((8,), dtype=T.float32), B: T.Tensor((8,), dtype=T.float32)):
        for i in T.serial(8):
            with T.block("blk"):
                vi = T.axis.spatial(8, i)
                T.block_attr({"sunmmio.anno_expr": vi + 1, "sunmmio.anno_const": 7})
                T.reads(A[vi])
                T.writes(B[vi])
                B[vi] = A[vi] + T.float32(1.0)

    src = build_sunmmio_source_without_compile(main)
    assert "func.func @main" in src
    assert "arith.addi" in src
    assert "memref.store" in src


def test_sunmmio_codegen_allocate_const_is_handled():
    dev = tvm.cpu(0)
    dtype = "float32"
    shape = (4,)
    cbuf = tvm.tir.decl_buffer(shape, dtype, name="C")
    data = tvm.runtime.tensor(np.array([1, 2, 3, 4], dtype=dtype), device=dev)
    body = tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0))
    stmt = tvm.tir.AllocateConst(cbuf.data, dtype, shape, data, body)
    src = build_sunmmio_source_from_stmt(stmt)
    assert "memref.alloc" in src


def test_sunmmio_codegen_block_alloc_buffers_are_handled():
    @T.prim_func
    def main(A: T.Tensor((8,), dtype=T.float32), B: T.Tensor((8,), dtype=T.float32)):
        for i in T.serial(8):
            with T.block("blk"):
                vi = T.axis.spatial(8, i)
                tmp = T.alloc_buffer((8,), dtype=T.float32, scope="local")
                T.reads(A[vi])
                T.writes(B[vi], tmp[vi])
                tmp[vi] = A[vi]
                B[vi] = tmp[vi] + T.float32(1.0)

    src = build_sunmmio_source_without_compile(main)
    assert "memref.alloc" in src
    assert "memref.store" in src


def test_sunmmio_codegen_unsupported_stmt_fails_loudly():
    cond = tvm.tir.LT(tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 1))
    body = tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0))
    stmt = tvm.tir.While(cond, body)
    target = determine_target("Sunmmio", return_object=True)
    func = tvm.tir.PrimFunc([], stmt)
    func = func.with_attr("global_symbol", "main")
    func = func.with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
    mod = tvm.IRModule({"main": func})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(Exception, match="CodeGenTileLangSunMMIO unsupported stmt: tir.While"):
        builder(mod, target)


def test_sunmmio_codegen_shuffle_fails_loudly():
    shuffle = tvm.tir.Shuffle(
        [tvm.tir.Broadcast(tvm.tir.IntImm("int32", 7), 4)],
        [tvm.tir.IntImm("int32", 0)],
    )
    stmt = tvm.tir.Evaluate(shuffle)
    target = determine_target("Sunmmio", return_object=True)
    func = tvm.tir.PrimFunc([], stmt)
    func = func.with_attr("global_symbol", "main")
    func = func.with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
    mod = tvm.IRModule({"main": func})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(Exception, match="CodeGenTileLangSunMMIO unsupported expr: tir.Shuffle"):
        builder(mod, target)


def test_sunmmio_codegen_ramp_is_supported():
    ramp = tvm.tir.Ramp(tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 1), 4)
    stmt = tvm.tir.Evaluate(ramp)
    src = build_sunmmio_source_from_stmt(stmt)
    assert "sunmmio.ramp" in src
    assert "vector<4xi32>" in src


def test_sunmmio_codegen_broadcast_is_supported():
    bcast = tvm.tir.Broadcast(tvm.tir.FloatImm("float32", 1.5), 4)
    stmt = tvm.tir.Evaluate(bcast)
    src = build_sunmmio_source_from_stmt(stmt)
    assert "vector.broadcast" in src
    assert "vector<4xf32>" in src


def test_sunmmio_codegen_compile_path_not_implemented():
    target = determine_target("Sunmmio", return_object=True)
    func = simple_add_kernel().with_attr("global_symbol", "main")
    func = func.with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
    mod = tvm.IRModule({"main": func})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio")
    with pytest.raises(Exception, match="not implemented yet"):
        builder(mod, target)


@pytest.mark.parametrize(
    "kernel_name,kernel_factory",
    [
        ("simple_add", lambda: simple_add_kernel()),
        (
            "flashattn",
            lambda: flashattn_kernel(1, 1, 128, 32, False, block_M=32, block_N=32, num_stages=1, threads=1),
        ),
        ("func_sync", lambda: func_sync_kernel(256, 256, 256, 64, 64, 64)),
    ],
)
def test_sunmmio_codegen_coverage_report_has_no_missing_entries(tmp_path, kernel_name, kernel_factory):
    report_path = tmp_path / f"codegen_coverage_{kernel_name}.json"
    old_path = os.environ.get("TL_SUNMMIO_CODEGEN_COVERAGE_PATH")
    old_strict = os.environ.get("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT")
    os.environ["TL_SUNMMIO_CODEGEN_COVERAGE_PATH"] = str(report_path)
    os.environ["TL_SUNMMIO_CODEGEN_COVERAGE_STRICT"] = "1"
    try:
        _ = build_sunmmio_source_without_compile(kernel_factory())
    finally:
        if old_path is None:
            os.environ.pop("TL_SUNMMIO_CODEGEN_COVERAGE_PATH", None)
        else:
            os.environ["TL_SUNMMIO_CODEGEN_COVERAGE_PATH"] = old_path
        if old_strict is None:
            os.environ.pop("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT", None)
        else:
            os.environ["TL_SUNMMIO_CODEGEN_COVERAGE_STRICT"] = old_strict

    assert report_path.exists(), f"coverage report not generated: {report_path}"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required_keys = [
        "expected_node_types",
        "visited_node_types",
        "missing_node_types",
        "expected_call_ops",
        "visited_call_ops",
        "missing_call_ops",
    ]
    for key in required_keys:
        assert key in report, f"missing coverage key: {key}"
    assert report["missing_node_types"] == []
    assert report["missing_call_ops"] == []
    expected_nodes = set(report["expected_node_types"])
    visited_nodes = set(report["visited_node_types"])
    missing_nodes = set(report["missing_node_types"])
    expected_calls = set(report["expected_call_ops"])
    visited_calls = set(report["visited_call_ops"])
    missing_calls = set(report["missing_call_ops"])
    assert expected_nodes - visited_nodes == missing_nodes
    assert expected_calls - visited_calls == missing_calls
    assert "tir.For" in expected_nodes
    assert "tir.For" in visited_nodes


if __name__ == "__main__":
    tilelang.testing.main()
