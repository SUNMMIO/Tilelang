import tilelang
import tilelang.language as T
import tilelang.transform
from tilelang import tvm as tvm
from tvm import tir
from tvm.tir.stmt_functor import post_order_visit
from tilelang.utils.target import determine_target, target_is_sunmmio
from tilelang.engine.lower import canon_target_host

tilelang.env.disable_cache()


def make_sunmmio_target_with_host():
    """Build the Sunmmio target with a host, matching what lower() does."""
    target = determine_target("Sunmmio", return_object=True)
    target_host = tvm.target.Target.canon_target(canon_target_host(target, None))
    return tvm.target.Target(target, target_host)


def run_pre_split_passes(mod, target):
    """Run the passes required before AnnotateDeviceRegions, matching the real pipeline."""
    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    return mod


def collect_attr_stmts(stmt, attr_key):
    """Walk TIR and return all AttrStmt nodes with the given attr_key."""
    found = []

    def fvisit(node):
        if isinstance(node, tir.AttrStmt) and node.attr_key == attr_key:
            found.append(node)

    post_order_visit(stmt, fvisit)
    return found


def get_device_func(mod):
    """
    Return the device function after SplitHostDevice.
    For CPU-based targets (kDLCPU) like Sunmmio, SplitHostDevice marks the
    device function with kIsGlobalFunc=True (not DEVICE_KERNEL_LAUNCH which
    is GPU-only). We identify it by kIsGlobalFunc + Sunmmio target.
    """
    candidates = [
        f
        for f in mod.functions.values()
        if f.attrs.get("tir.is_global_func", False) and "target" in f.attrs and target_is_sunmmio(f.attrs["target"])
    ]
    return candidates


def simple_kernel(M, N, dtype=T.float32):
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(M, N) as (bx, by):
            B[bx, by] = A[bx, by]

    return tvm.IRModule({"main": main})


def test_annotate_device_regions_wraps_thread_extent_with_target():
    """AnnotateDeviceRegions must wrap thread_extent regions with a 'target' AttrStmt
    pointing to the Sunmmio device target (no host)."""
    target = make_sunmmio_target_with_host()

    with tvm.target.Target(target):
        mod = simple_kernel(4, 4)

    mod = run_pre_split_passes(mod, target)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)

    func = mod["main"]
    target_attrs = collect_attr_stmts(func.body, "target")

    assert len(target_attrs) >= 1, (
        f"Expected at least one 'target' AttrStmt after AnnotateDeviceRegions, found none.\nFunction body:\n{func.script()}"
    )

    device_target_node = target_attrs[0].node
    assert target_is_sunmmio(device_target_node), (
        f"Expected 'target' AttrStmt to annotate with the Sunmmio device target, got: {device_target_node}"
    )

    inner_stmt = target_attrs[0].body
    assert isinstance(inner_stmt, tir.AttrStmt) and inner_stmt.attr_key == "thread_extent", (
        f"Expected the body of the 'target' AttrStmt to be a 'thread_extent' AttrStmt (the thread launch region), got: {type(inner_stmt)}"
    )


def test_split_host_device_produces_two_functions():
    """SplitHostDevice should split the module into a host and a device function."""
    target = make_sunmmio_target_with_host()

    with tvm.target.Target(target):
        mod = simple_kernel(4, 4)

    mod = run_pre_split_passes(mod, target)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)

    funcs = dict(mod.functions)
    assert len(funcs) == 2, f"Expected 2 functions after SplitHostDevice, got {len(funcs)}: {list(funcs.keys())}"


def test_split_host_device_device_func_has_sunmmio_target():
    """The device function produced by SplitHostDevice should carry the Sunmmio device target.

    SplitHostDevice does not set CallingConv.DEVICE_KERNEL_LAUNCH; the device function
    is identified by its target attribute having no host.
    """
    target = make_sunmmio_target_with_host()

    with tvm.target.Target(target):
        mod = simple_kernel(4, 4)

    mod = run_pre_split_passes(mod, target)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)

    device_funcs = get_device_func(mod)
    assert len(device_funcs) == 1, (
        f"Expected exactly 1 Sunmmio device function, got {len(device_funcs)}.\n"
        f"Functions: { {gv.name_hint: dict(f.attrs) for gv, f in mod.functions.items()} }"
    )
    assert target_is_sunmmio(device_funcs[0].attrs["target"]), (
        f"Expected device function target to be Sunmmio, got: {device_funcs[0].attrs['target']}"
    )


def test_split_host_device_device_func_has_no_thread_bindings():
    """The device function body must contain no threadIdx bindings (Sunmmio is fully threadless)."""
    target = make_sunmmio_target_with_host()

    with tvm.target.Target(target):
        mod = simple_kernel(4, 4)

    mod = run_pre_split_passes(mod, target)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)

    device_funcs = get_device_func(mod)
    assert len(device_funcs) == 1, (
        "Expected a device function (Sunmmio target + kIsGlobalFunc) after SplitHostDevice, "
        f"found none.\nFunctions in module: {[str(gv) for gv in mod.functions]}"
    )
    device_func = device_funcs[0]

    thread_extents = {}

    def fvisit(node):
        if isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent":
            thread_extents[node.node.thread_tag] = int(node.value)

    post_order_visit(device_func.body, fvisit)

    assert "threadIdx.x" not in thread_extents, (
        f"Sunmmio device function must have no threadIdx bindings (threadless). "
        f"Found thread extents: {thread_extents}\n"
        f"Device function:\n{device_func.script()}"
    )


if __name__ == "__main__":
    test_annotate_device_regions_wraps_thread_extent_with_target()
    test_split_host_device_produces_two_functions()
    test_split_host_device_device_func_has_sunmmio_target()
    test_split_host_device_device_func_has_no_thread_bindings()
    print("All tests passed.")
