"""Sunmmio tests for SplitHostDevice device-function attribute propagation."""

import tilelang
import tilelang.language as T
import tilelang.transform
from tilelang import tvm as tvm
import tilelang.testing
from tilelang.engine.lower import canon_target_host
from tilelang.layout.cute_layout import CuteLayout
from tilelang.utils.target import determine_target, target_is_sunmmio
from tvm import tir

tilelang.env.disable_cache()


def make_sunmmio_target_with_host():
    target = determine_target("Sunmmio", return_object=True)
    target_host = tvm.target.Target.canon_target(canon_target_host(target, None))
    return tvm.target.Target(target, target_host)


def get_device_func(mod):
    device_funcs = [
        func
        for func in mod.functions.values()
        if isinstance(func, tir.PrimFunc)
        and func.attrs is not None
        and func.attrs.get("tir.is_global_func", False)
        and "target" in func.attrs
        and target_is_sunmmio(func.attrs["target"])
    ]
    assert len(device_funcs) == 1, (
        f"Expected exactly one Sunmmio device function, got {len(device_funcs)}.\n"
        f"Functions: { {gv.name_hint: dict(func.attrs) for gv, func in mod.functions.items()} }"
    )
    return device_funcs[0]


def get_host_func(mod):
    host_funcs = [
        func for func in mod.functions.values() if isinstance(func, tir.PrimFunc) and not func.attrs.get("tir.is_global_func", False)
    ]
    assert len(host_funcs) == 1, (
        f"Expected exactly one host function, got {len(host_funcs)}.\n"
        f"Functions: { {gv.name_hint: dict(func.attrs) for gv, func in mod.functions.items()} }"
    )
    return host_funcs[0]


def get_param_by_name(func, name):
    for param in func.params:
        if getattr(param, "name_hint", getattr(param, "name", None)) == name:
            return param
    return None


def run_sunmmio_split(mod, *, hoist=False):
    target = make_sunmmio_target_with_host()
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        if hoist:
            mod = tilelang.transform.HoistBlockAnnotationsToFuncAttrs()(mod)
        mod = tilelang.transform.LowerOpaqueBlock()(mod)
        mod = tilelang.transform.AnnotateDeviceRegions()(mod)
        mod = tilelang.transform.SplitHostDevice()(mod)
    return mod


def test_sunmmio_split_host_device_propagates_hoisted_layout_attrs():
    n = T.dynamic("n")

    @T.prim_func
    def before(A: T.Buffer((n,), "float32")):
        B = T.alloc_buffer((n,), "float32", scope="shared")
        with T.block("producer"):
            T.block_attr(
                {
                    "layout_map": {B.data: T.Layout((n,), lambda i: i)},
                    "global_layout_map": {A.data: T.Layout((n,), lambda i: i)},
                }
            )
            with T.Kernel(1):
                B[0] = A[0]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = run_sunmmio_split(mod, hoist=True)

    host_func = get_host_func(mod)
    device_func = get_device_func(mod)

    assert "tl.device_func_attr_keys" not in host_func.attrs
    assert "tl.device_func_attr_keys" not in device_func.attrs
    assert "layout_map" in device_func.attrs
    assert "global_layout_map" in device_func.attrs

    device_n = get_param_by_name(device_func, "n")
    device_a = get_param_by_name(device_func, "A")
    device_b = get_param_by_name(device_func, "B")
    assert device_n is not None
    assert device_a is not None
    assert device_b is not None

    shared_key = next(iter(device_func.attrs["layout_map"].keys()))
    global_key = next(iter(device_func.attrs["global_layout_map"].keys()))
    assert shared_key.same_as(device_b)
    assert global_key.same_as(device_a)

    shared_layout = next(iter(device_func.attrs["layout_map"].values()))
    global_layout = next(iter(device_func.attrs["global_layout_map"].values()))
    assert shared_layout.get_input_shape()[0].same_as(device_n)
    assert global_layout.get_input_shape()[0].same_as(device_n)


def test_sunmmio_split_host_device_preserves_cute_layout_attr_type():
    @T.prim_func
    def before(A: T.Buffer((16, 16), "float32")):
        with T.Kernel(1):
            A[0, 0] = A[0, 0]

    func = before.with_attr("global_symbol", "main")
    buffer = func.buffer_map[func.params[0]]
    layout = CuteLayout([16, 16], [4, 4, 16], [64, 16, 1], [2, 1])._inner
    func = func.with_attr("layout_map", {buffer: layout}).with_attr("tl.device_func_attr_keys", ["layout_map"])

    mod = run_sunmmio_split(tvm.IRModule.from_expr(func))
    device_func = get_device_func(mod)

    remapped_layout = next(iter(device_func.attrs["layout_map"].values()))
    assert remapped_layout.__class__.__name__ == "CuteLayout"
    assert list(remapped_layout.mode_shape) == list(layout.mode_shape)
    assert list(remapped_layout.mode_stride) == list(layout.mode_stride)
    assert list(remapped_layout.dim_levels) == list(layout.dim_levels)


def test_sunmmio_split_host_device_remaps_marked_simple_attrs():
    n = T.dynamic("n")

    @T.prim_func
    def before(A: T.Buffer((n,), "float32")):
        B = T.alloc_buffer((n,), "float32", scope="shared")
        with T.Kernel(1):
            B[0] = A[0]

    func = before.with_attr("global_symbol", "main")
    buffer_a = func.buffer_map[func.params[0]]
    data_a = buffer_a.data
    n_var = buffer_a.shape[0]

    func = (
        func.with_attr("var_attr", data_a)
        .with_attr("buffer_attr", buffer_a)
        .with_attr("expr_attr", n_var + 1)
        .with_attr("array_attr", [data_a, "keep_me", n_var + 2])
        .with_attr("map_attr", {data_a: n_var + 3, "plain_key": data_a})
        .with_attr("unmarked_attr", data_a)
        .with_attr(
            "tl.device_func_attr_keys",
            [
                "var_attr",
                "buffer_attr",
                "expr_attr",
                "array_attr",
                "map_attr",
            ],
        )
    )

    mod = run_sunmmio_split(tvm.IRModule.from_expr(func))
    device_func = get_device_func(mod)
    device_a = get_param_by_name(device_func, "A")
    device_n = get_param_by_name(device_func, "n")
    assert device_a is not None
    assert device_n is not None

    assert device_func.attrs["var_attr"].same_as(device_a)
    assert device_func.attrs["buffer_attr"].data.same_as(device_a)
    assert device_func.attrs["buffer_attr"].shape[0].same_as(device_n)
    assert device_func.attrs["expr_attr"].a.same_as(device_n)

    array_attr = device_func.attrs["array_attr"]
    assert array_attr[0].same_as(device_a)
    assert array_attr[1] == "keep_me"
    assert array_attr[2].a.same_as(device_n)

    map_attr = device_func.attrs["map_attr"]
    remapped_key = next(key for key in map_attr.keys() if hasattr(key, "same_as"))
    assert remapped_key.same_as(device_a)
    assert map_attr[remapped_key].a.same_as(device_n)
    assert map_attr["plain_key"].same_as(device_a)

    assert "unmarked_attr" not in device_func.attrs
    assert "tl.device_func_attr_keys" not in device_func.attrs


if __name__ == "__main__":
    tilelang.testing.main()
