import tilelang
import tilelang.language as T
import tilelang.transform as tl_transform
import tvm_ffi

from tilelang import tvm as tvm
from tilelang.engine.phase import LowerAndLegalize, OptimizeForSunmmio, PreLowerSemanticCheck
from tilelang.layout import make_zz_layout
from tilelang.utils.target import determine_target, target_is_sunmmio
from tvm import tir
from tvm.tir.stmt_functor import post_order_visit

tilelang.env.disable_cache()

_MESH_OP_NAMES = ("tl.mesh_nrows", "tl.mesh_ncols", "tl.mesh_ncores")
_layout_logical_shape = tvm_ffi.get_global_func("tl.CuteLayout_logical_shape")


def _sunmmio_target(nrows=4, ncols=4):
    return determine_target(f"llvm -mcpu=sunmmio-a4e -mattr=device_mesh_nrow_{nrows},device_mesh_ncol_{ncols}", return_object=True)


def _sunmmio_target_with_host(nrows=4, ncols=4):
    return tvm.target.Target(_sunmmio_target(nrows, ncols), tvm.target.Target.canon_target("llvm"))


def _mesh_calls_in_stmt(func):
    ops = {name: tir.op.Op.get(name) for name in _MESH_OP_NAMES}
    calls = []

    def visit(node):
        if isinstance(node, tir.Call):
            for name, op in ops.items():
                if hasattr(node.op, "same_as") and node.op.same_as(op):
                    calls.append(name)

    post_order_visit(func.body, visit)
    return calls


def _mesh_calls_in_mod(mod):
    calls = []
    for func in mod.functions.values():
        if isinstance(func, tir.PrimFunc):
            calls.extend(_mesh_calls_in_stmt(func))
            calls.extend(_mesh_calls_in_exprs(_exprs_in_buffer_metadata(func)))
    return calls


def _mesh_calls_in_exprs(exprs):
    ops = {name: tir.op.Op.get(name) for name in _MESH_OP_NAMES}
    calls = []

    def visit(node):
        if isinstance(node, tir.Call):
            for name, op in ops.items():
                if hasattr(node.op, "same_as") and node.op.same_as(op):
                    calls.append(name)

    for expr in exprs:
        tir.stmt_functor.post_order_visit(expr, visit)
    return calls


def _exprs_in_buffer_metadata(func):
    exprs = []
    for _, buffer in func.buffer_map.items():
        exprs.extend(buffer.shape)
        exprs.extend(buffer.strides)
        exprs.append(buffer.elem_offset)

    def visit(node):
        if isinstance(node, tir.Block):
            for buffer in node.alloc_buffers:
                exprs.extend(buffer.shape)
                exprs.extend(buffer.strides)
                exprs.append(buffer.elem_offset)
            for region in list(node.reads) + list(node.writes):
                for rng in region.region:
                    exprs.append(rng.min)
                    exprs.append(rng.extent)

    post_order_visit(func.body, visit)
    return exprs


def _thread_extents(func):
    extents = {}

    def visit(node):
        if isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent":
            extents[node.node.thread_tag] = int(node.value)

    post_order_visit(func.body, visit)
    return extents


def _alloc_shapes(func):
    shapes = []

    def visit(node):
        if isinstance(node, tir.Block):
            for buffer in node.alloc_buffers:
                shapes.append(tuple(int(extent) for extent in buffer.shape))

    post_order_visit(func.body, visit)
    return shapes


def _sunmmio_device_func(mod):
    funcs = [
        func
        for func in mod.functions.values()
        if isinstance(func, tir.PrimFunc)
        and func.attrs is not None
        and func.attrs.get("tir.is_global_func", False)
        and "target" in func.attrs
        and target_is_sunmmio(func.attrs["target"])
    ]
    assert len(funcs) == 1, f"Expected one Sunmmio device function, got {len(funcs)}.\n{mod.script()}"
    return funcs[0]


def _bind_and_resolve(mod, target):
    mod = tvm.tir.transform.BindTarget(target)(mod)
    return tl_transform.ResolveSunmmioMeshSymbols()(mod)


def _lower_and_optimize(mod, target):
    with tvm.transform.PassContext(opt_level=3), tvm.target.Target(target):
        PreLowerSemanticCheck(mod)
        lowered = LowerAndLegalize(mod, target)

    assert _mesh_calls_in_mod(lowered) == []
    assert _thread_extents(lowered["main"])["blockIdx.x"] == 16

    with tvm.transform.PassContext(opt_level=3), tvm.target.Target(target):
        optimized = OptimizeForSunmmio(lowered, target)

    device_func = _sunmmio_device_func(optimized)
    assert _mesh_calls_in_mod(optimized) == []
    assert _thread_extents(device_func)["blockIdx.x"] == 16
    return optimized


def _assert_frontend_uses_symbolic_mesh(func):
    assert "tl.mesh_ncores" in _mesh_calls_in_stmt(func)
    metadata_calls = _mesh_calls_in_exprs(_exprs_in_buffer_metadata(func))
    assert "tl.mesh_nrows" in metadata_calls
    assert "tl.mesh_ncols" in metadata_calls


def _symbolic_mesh_gemm_mod():
    M, N, K = 128, 128, 128
    block_M, block_N, block_K = 32, 32, 32
    dtype = "bfloat16"
    accum_dtype = "float32"
    policy = T.MeshShardingPolicy(y=0, x=1)
    a_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    b_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    c_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    with tvm.target.Target(_sunmmio_target_with_host()):

        @T.prim_func
        def main(
            A: T.MeshTensor((M, K), policy, dtype, layout=a_layout),
            B: T.MeshTensor((K, N), policy, dtype, layout=b_layout),
            C: T.MeshTensor((M, N), policy, accum_dtype, layout=c_layout),
        ):
            with T.Kernel() as _cid:
                sharded_M, sharded_K = A.local_shape
                _, sharded_N = B.local_shape

                A_shared = T.alloc_shared((block_M, block_K), dtype)
                A_shared_dist = T.alloc_shared((block_M, block_K * T.mesh_ncols()), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                B_shared_dist = T.alloc_shared((block_K * T.mesh_nrows(), block_N), dtype)
                C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

                for bx in T.serial(T.ceildiv(sharded_M, block_M)):
                    for by in T.serial(T.ceildiv(sharded_N, block_N)):
                        T.clear(C_shared)
                        for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=1):
                            T.copy(A[bx * block_M, k * block_K], A_shared)
                            T.comm.all_gather(A_shared, A_shared_dist, direction="horizontal", axis=-1)
                            T.copy(B[k * block_K, by * block_N], B_shared)
                            T.comm.all_gather(B_shared, B_shared_dist, direction="vertical", axis=0)
                            T.gemm(A_shared_dist, B_shared_dist, C_shared)
                        T.copy(C_shared, C[bx * block_M, by * block_N])

    return tvm.IRModule({"main": main})


def _symbolic_mesh_all_gather_mod():
    block_M, block_N = 32, 32
    dtype = "bfloat16"

    with tvm.target.Target(_sunmmio_target_with_host()):

        @T.prim_func
        def main(A: T.Tensor((block_M, block_N), dtype)):
            with T.Kernel() as _cid:
                A_shared = T.alloc_shared((block_M, block_N), dtype)
                A_shared_dist = T.alloc_shared((block_M, block_N * T.mesh_ncols()), dtype)

                T.copy(A, A_shared)
                T.comm.all_gather(A_shared, A_shared_dist, direction="horizontal", axis=-1)

    return tvm.IRModule({"main": main})


def _symbolic_mesh_gqa_mod():
    batch, heads, seq_len, dim, groups = 2, 4, 32, 32, 2
    block_M, block_N = 32, 32
    head_kv = heads // groups
    dtype = "bfloat16"
    accum_dtype = "bfloat16"
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    policy = T.MeshShardingPolicy(y=0, x=2)
    q_layout = make_zz_layout(q_shape, [1, 3], (32, 32))
    kv_layout = make_zz_layout(kv_shape, [1, 3], (32, 32))

    with tvm.target.Target(_sunmmio_target_with_host()):

        @T.prim_func
        def main(
            Q: T.MeshTensor(q_shape, policy, dtype, layout=q_layout),
            K: T.MeshTensor(kv_shape, policy, dtype, layout=kv_layout),
            V: T.MeshTensor(kv_shape, policy, dtype, layout=kv_layout),
            Output: T.MeshTensor(q_shape, policy, dtype, layout=q_layout),
        ):
            with T.Kernel() as _cid:
                sharded_batch = Q.local_shape[0]
                sharded_heads = Q.local_shape[2]

                Q_shared = T.alloc_shared((block_M, dim), dtype)
                K_shared = T.alloc_shared((block_N, dim), dtype)
                V_shared = T.alloc_shared((block_N, dim), dtype)
                O_shared = T.alloc_shared((block_M, dim), dtype)
                acc_s = T.alloc_shared((block_M, block_N), accum_dtype)
                acc_s_cast = T.alloc_shared((block_M, block_N), dtype)
                acc_o = T.alloc_shared((block_M, dim), accum_dtype)

                for bz in T.serial(sharded_batch):
                    for by in T.serial(sharded_heads):
                        for bx in T.serial(T.ceildiv(seq_len, block_M)):
                            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
                            T.clear(acc_s)
                            T.clear(acc_o)
                            for k in T.Pipelined(T.ceildiv(seq_len, block_N), num_stages=1):
                                T.copy(K[bz, k * block_N : (k + 1) * block_N, by // groups, :], K_shared)
                                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                                T.copy(acc_s, acc_s_cast)
                                T.copy(V[bz, k * block_N : (k + 1) * block_N, by // groups, :], V_shared)
                                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                            T.copy(acc_o, O_shared)
                            T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return tvm.IRModule({"main": main})


def test_resolve_replaces_mesh_intrinsics_in_body_and_kernel_extent():
    target = _sunmmio_target()

    with tvm.target.Target(target):

        @T.prim_func
        def main():
            with T.Kernel() as cid:
                scratch = T.alloc_shared((T.mesh_nrows(), T.mesh_ncols(), T.mesh_ncores()), "float32")
                scratch[0, 0, 0] = T.if_then_else(
                    cid < T.mesh_ncores(),
                    T.Cast("float32", T.mesh_nrows() + T.mesh_ncols() + T.mesh_ncores()),
                    T.float32(0),
                )

        mod = tvm.IRModule({"main": main})

    assert set(_mesh_calls_in_stmt(mod["main"])) == set(_MESH_OP_NAMES)

    mod = _bind_and_resolve(mod, target)
    func = mod["main"]

    assert _mesh_calls_in_stmt(func) == []
    assert _mesh_calls_in_exprs(_exprs_in_buffer_metadata(func)) == []
    assert _thread_extents(func)["blockIdx.x"] == 16
    assert (4, 4, 16) in _alloc_shapes(func)


def test_resolve_updates_default_mesh_tensor_buffer_map_and_layout_metadata():
    target = _sunmmio_target()
    policy = T.MeshShardingPolicy(y=0, x=1)
    layout = make_zz_layout((128, 96), [0, 1], (32, 32))

    with tvm.target.Target(target):

        @T.prim_func
        def main(A: T.MeshTensor((128, 96), policy, "float16", layout=layout)):
            with T.Kernel() as _cid:
                valid_M, valid_N = A.get_local_extent(_cid)
                for i in T.serial(valid_M):
                    for j in T.serial(valid_N):
                        A[i, j] = A[i, j]

        mod = tvm.IRModule({"main": main})

    unresolved_buffer = mod["main"].buffer_map[mod["main"].params[0]]
    assert _mesh_calls_in_exprs(unresolved_buffer.shape) == ["tl.mesh_nrows", "tl.mesh_ncols"]

    mod = _bind_and_resolve(mod, target)
    func = mod["main"]
    buffer = func.buffer_map[func.params[0]]
    tensor_meta = func.attrs["tensor_meta"]["A"]

    assert _mesh_calls_in_exprs(_exprs_in_buffer_metadata(func)) == []
    assert tuple(int(extent) for extent in buffer.shape) == (32, 24)
    assert tuple(int(extent) for extent in _layout_logical_shape(tensor_meta["sharded_layout"])) == (32, 24)


def test_lower_and_optimize_resolve_kernel_default_mesh_ncores():
    target = _sunmmio_target_with_host()

    with tvm.target.Target(target):

        @T.prim_func
        def main():
            with T.Kernel() as cid:
                scratch = T.alloc_shared((1,), "float32")
                scratch[0] = T.if_then_else(
                    cid < T.mesh_ncores(),
                    T.Cast("float32", T.mesh_nrows() + T.mesh_ncols() + T.mesh_ncores()),
                    T.float32(0),
                )

        mod = tvm.IRModule({"main": main})

    assert set(_mesh_calls_in_stmt(mod["main"])) == set(_MESH_OP_NAMES)

    with tvm.transform.PassContext(opt_level=3), tvm.target.Target(target):
        PreLowerSemanticCheck(mod)
        lowered = LowerAndLegalize(mod, target)

    assert _mesh_calls_in_mod(lowered) == []
    assert _thread_extents(lowered["main"])["blockIdx.x"] == 16

    with tvm.transform.PassContext(opt_level=3), tvm.target.Target(target):
        optimized = OptimizeForSunmmio(lowered, target)

    device_func = _sunmmio_device_func(optimized)
    assert _mesh_calls_in_mod(optimized) == []
    assert _thread_extents(device_func)["blockIdx.x"] == 16


def test_lower_and_optimize_symbolic_mesh_gemm_kernel():
    target = _sunmmio_target_with_host()
    mod = _symbolic_mesh_gemm_mod()

    _assert_frontend_uses_symbolic_mesh(mod["main"])
    optimized = _lower_and_optimize(mod, target)

    device_func = _sunmmio_device_func(optimized)
    script = device_func.script()
    assert "A_shared" in script
    assert "A_shared_dist" in script
    assert "B_shared" in script
    assert "B_shared_dist" in script
    assert "T.broadcast_" in script
    assert "T.mma_sunmmio" in script
    assert "tl.mesh_" not in script


def test_lower_and_optimize_symbolic_mesh_all_gather_shape_kernel():
    target = _sunmmio_target_with_host()
    mod = _symbolic_mesh_all_gather_mod()

    assert "tl.mesh_ncores" in _mesh_calls_in_stmt(mod["main"])
    assert "tl.mesh_ncols" in _mesh_calls_in_exprs(_exprs_in_buffer_metadata(mod["main"]))
    optimized = _lower_and_optimize(mod, target)

    device_func = _sunmmio_device_func(optimized)
    script = device_func.script()
    assert "T.broadcast_" in script
    assert "tl.mesh_" not in script


def test_lower_and_optimize_symbolic_mesh_gqa_kernel():
    target = _sunmmio_target_with_host()
    mod = _symbolic_mesh_gqa_mod()

    _assert_frontend_uses_symbolic_mesh(mod["main"])
    optimized = _lower_and_optimize(mod, target)

    device_func = _sunmmio_device_func(optimized)
    script = device_func.script()
    assert "Q_shared" in script
    assert "K_shared" in script
    assert "V_shared" in script
    assert "T.mma_sunmmio" in script
    assert "tl.mesh_" not in script
