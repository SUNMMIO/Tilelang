import importlib.util
from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize, PreLowerSemanticCheck
from tilelang.utils.target import determine_target
from tvm import tir


tilelang.env.disable_cache()


def _load_elementwise_add_example():
    example_path = Path(__file__).resolve().parents[4] / "examples" / "sunmmio" / "elementwise" / "elementwise_add.py"
    spec = importlib.util.spec_from_file_location("tilelang_sunmmio_elementwise_add_example_for_make_packed_api", example_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sunmmio_target():
    return tvm.target.Target(determine_target("sunmmio", return_object=True), tvm.target.Target.canon_target("llvm"))


def _elementwise_add_prim_func(M, N):
    example = _load_elementwise_add_example()
    return example._elementwise_add_prim_func(
        M,
        N,
        32,
        32,
        T.bfloat16,
        T.float32,
    )


def _lower_until_make_packed_api_input(kernel) -> tvm.IRModule:
    target = _sunmmio_target()
    mod = tvm.IRModule({kernel.attrs["global_symbol"]: kernel})

    with tvm.transform.PassContext(opt_level=3), target:
        PreLowerSemanticCheck(mod)
        mod = LowerAndLegalize(mod, target)
        for pass_obj in (
            tilelang.transform.IfStmtBinding(),
            tilelang.transform.SunmmioPipelinePlanning(debug=False),
            tilelang.transform.InjectSunmmioPipeline(),
            tilelang.transform.LowerOpaqueBlock(),
            tilelang.transform.Simplify(),
            tir.transform.NarrowDataType(32),
            tilelang.transform.ConfigIndexBitwidth(),
            tir.transform.Simplify(),
            tilelang.transform.LoopUnswitching(),
            tir.transform.UnrollLoop(),
            tir.transform.RenormalizeSplitPattern(),
            tir.transform.Simplify(),
            tir.transform.RemoveNoOp(),
            tir.transform.HoistIfThenElse(),
            tir.transform.VerifyMemory(),
            tir.transform.AnnotateEntryFunc(),
            tilelang.transform.AnnotateDeviceRegions(),
            tilelang.transform.SplitHostDevice(),
            tilelang.transform.AnnotateReadOnlyParams(),
            tilelang.transform.MergeIfStmt(),
            tilelang.transform.InjectSunmmioSync(),
        ):
            mod = pass_obj(mod)

    return mod


def _apply_make_packed_api(mod: tvm.IRModule) -> tvm.IRModule:
    with tvm.transform.PassContext(opt_level=3), _sunmmio_target():
        mod = tilelang.transform.MakePackedAPI()(mod)
        return tilelang.transform.Simplify()(mod)


def _var_name(var):
    return getattr(var, "name_hint", getattr(var, "name", None))


def _vars_by_name(exprs):
    vars_by_name = {}

    def visit(node):
        if isinstance(node, tir.Var):
            name = _var_name(node)
            if name in vars_by_name:
                assert vars_by_name[name].same_as(node), f"Var {name} has inconsistent object identity"
            else:
                vars_by_name[name] = node

    for expr in exprs:
        tir.stmt_functor.post_order_visit(expr, visit)
    return vars_by_name


def _buffer_by_name(func: tir.PrimFunc, name: str):
    for param in func.params:
        if param in func.buffer_map:
            buffer = func.buffer_map[param]
            if buffer.name == name:
                return buffer
    raise AssertionError(f"Buffer {name!r} not found in {func.buffer_map}")


def test_split_host_device_keeps_tensor_meta_dynamic_vars_in_host_scope():
    kernel = _elementwise_add_prim_func(T.dynamic("m"), T.dynamic("n"))
    mod = _lower_until_make_packed_api_input(kernel)
    func = mod["elem_add"]

    buffer = _buffer_by_name(func, "A")
    tensor_meta = func.attrs["tensor_meta"]["A"]

    physical_vars = _vars_by_name(list(buffer.shape) + list(buffer.strides))
    logical_vars = _vars_by_name(list(tensor_meta["global_shape"]) + list(tensor_meta["global_strides"]))

    for name in ("m", "n"):
        assert name in logical_vars
        assert name in physical_vars
        assert logical_vars[name].same_as(physical_vars[name])


def test_make_packed_api_binds_dynamic_mesh_tensor_from_logical_shape():
    kernel = _elementwise_add_prim_func(T.dynamic("m"), T.dynamic("n"))
    mod = _apply_make_packed_api(_lower_until_make_packed_api_input(kernel))

    script = mod["elem_add"].script()
    assert 'with T.LetStmt(T.Cast("int32", elem_add_A_shape_1[0]), var=m)' in script
    assert 'with T.LetStmt(T.Cast("int32", elem_add_A_shape_1[1]), var=n)' in script
    assert "elem_add_kernel(A.data, B.data, C.data, m, n)" in script
    assert "TVM is unable to solve" not in script


def test_make_packed_api_validates_static_mesh_tensor_logical_shape():
    kernel = _elementwise_add_prim_func(1024, 1024)
    mod = _apply_make_packed_api(_lower_until_make_packed_api_input(kernel))

    script = mod["elem_add"].script()
    assert '"shape[0]", T.int64(1024)' in script
    assert '"shape[1]", T.int64(1024)' in script
    assert '"shape[0]", T.int64(256)' not in script
    assert '"shape[1]", T.int64(256)' not in script
