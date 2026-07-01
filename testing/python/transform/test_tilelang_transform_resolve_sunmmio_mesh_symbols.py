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


def _sunmmio_target(nrows=2, ncols=3):
    return determine_target(f"llvm -mcpu=sunmmio-a4e -mattr=device_mesh_nrow_{nrows},device_mesh_ncol_{ncols}", return_object=True)


def _sunmmio_target_with_host(nrows=2, ncols=3):
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


def test_resolve_replaces_mesh_intrinsics_in_body_and_kernel_extent():
    target = _sunmmio_target(2, 3)

    with tvm.target.Target(target):

        @T.prim_func
        def main(A: T.Tensor((16,), "int32"), B: T.Tensor((16,), "int32")):
            with T.Kernel() as cid:
                scratch = T.alloc_shared((T.mesh_nrows(), T.mesh_ncols(), T.mesh_ncores()), "float32")
                scratch[0, 0, 0] = T.if_then_else(cid < T.mesh_ncores(), T.float32(1), T.float32(0))
                B[cid] = A[cid] + T.mesh_nrows() + T.mesh_ncols() + T.mesh_ncores()

        mod = tvm.IRModule({"main": main})

    assert set(_mesh_calls_in_stmt(mod["main"])) == set(_MESH_OP_NAMES)

    mod = _bind_and_resolve(mod, target)
    func = mod["main"]

    assert _mesh_calls_in_stmt(func) == []
    assert _mesh_calls_in_exprs(_exprs_in_buffer_metadata(func)) == []
    assert _thread_extents(func)["blockIdx.x"] == 6
    assert (2, 3, 6) in _alloc_shapes(func)


def test_resolve_updates_default_mesh_tensor_buffer_map_and_layout_metadata():
    target = _sunmmio_target(2, 3)
    policy = T.MeshShardingPolicy(y=0, x=1)
    layout = make_zz_layout((128, 96), [0, 1], (32, 32))

    with tvm.target.Target(target):

        @T.prim_func
        def main(A: T.MeshTensor((128, 96), policy, "float16", layout=layout)):
            with T.Kernel() as cid:
                A[cid, 0] = A[cid, 0]

        mod = tvm.IRModule({"main": main})

    unresolved_buffer = mod["main"].buffer_map[mod["main"].params[0]]
    assert _mesh_calls_in_exprs(unresolved_buffer.shape) == ["tl.mesh_nrows", "tl.mesh_ncols"]

    mod = _bind_and_resolve(mod, target)
    func = mod["main"]
    buffer = func.buffer_map[func.params[0]]
    tensor_meta = func.attrs["tensor_meta"]["A"]

    assert _mesh_calls_in_exprs(_exprs_in_buffer_metadata(func)) == []
    assert tuple(int(extent) for extent in buffer.shape) == (64, 32)
    assert tuple(int(extent) for extent in _layout_logical_shape(tensor_meta["sharded_layout"])) == (64, 32)


def test_lower_and_optimize_resolve_kernel_default_mesh_ncores():
    target = _sunmmio_target_with_host(2, 3)

    with tvm.target.Target(target):

        @T.prim_func
        def main(A: T.Tensor((16,), "int32"), B: T.Tensor((16,), "int32")):
            with T.Kernel() as cid:
                B[cid] = A[cid] + T.mesh_nrows() + T.mesh_ncols() + T.mesh_ncores()

        mod = tvm.IRModule({"main": main})

    assert _mesh_calls_in_stmt(mod["main"]) == [
        "tl.mesh_ncores",
        "tl.mesh_nrows",
        "tl.mesh_ncols",
        "tl.mesh_ncores",
    ]

    with tvm.transform.PassContext(opt_level=3), tvm.target.Target(target):
        PreLowerSemanticCheck(mod)
        lowered = LowerAndLegalize(mod, target)

    assert _mesh_calls_in_mod(lowered) == []
    assert _thread_extents(lowered["main"])["blockIdx.x"] == 6

    with tvm.transform.PassContext(opt_level=3), tvm.target.Target(target):
        optimized = OptimizeForSunmmio(lowered, target)

    device_func = _sunmmio_device_func(optimized)
    assert _mesh_calls_in_mod(optimized) == []
    assert _thread_extents(device_func)["blockIdx.x"] == 6
