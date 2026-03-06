import tilelang
import tilelang.language as T
import tilelang.transform
from tilelang import tvm as tvm
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


def get_device_func(mod):
    """
    Return the device function after SplitHostDevice.
    For CPU-based targets (kDLCPU) like Sunmmio, SplitHostDevice marks the
    device function with kIsGlobalFunc=True (not DEVICE_KERNEL_LAUNCH which
    is GPU-only). We identify it by kIsGlobalFunc + Sunmmio target.
    """
    candidates = [
        f for f in mod.functions.values()
        if f.attrs.get("tir.is_global_func", False) and
           "target" in f.attrs and target_is_sunmmio(f.attrs["target"])
    ]
    return candidates


def simple_kernel(M, N, dtype=T.float32):

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(M, N) as (bx, by):
            B[bx, by] = A[bx, by]

    return tvm.IRModule({"main": main})


def test_annotate_device_regions_runs_on_sunmmio():
    """AnnotateDeviceRegions should annotate thread_extent regions with the Sunmmio target."""
    target = make_sunmmio_target_with_host()

    with tvm.target.Target(target):
        mod = simple_kernel(4, 4)

    mod = run_pre_split_passes(mod, target)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)

    script = mod.script()
    assert "sunmmio" in script.lower(), (
        f"Expected Sunmmio target annotation in script, got:\n{script}"
    )
    print("test_annotate_device_regions_runs_on_sunmmio passed.")


def test_split_host_device_produces_two_functions():
    """SplitHostDevice should split the module into a host and a device function."""
    target = make_sunmmio_target_with_host()

    with tvm.target.Target(target):
        mod = simple_kernel(4, 4)

    mod = run_pre_split_passes(mod, target)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)

    funcs = dict(mod.functions)
    assert len(funcs) == 2, (
        f"Expected 2 functions after SplitHostDevice, got {len(funcs)}: {list(funcs.keys())}"
    )
    print("test_split_host_device_produces_two_functions passed.")


def test_split_host_device_device_func_has_sunmmio_target():
    """The device function produced by SplitHostDevice should carry the Sunmmio target."""
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
    print("test_split_host_device_device_func_has_sunmmio_target passed.")


def test_split_host_device_device_func_is_threadless():
    """The device function should have threadIdx.x with extent 1 (threadless)."""
    target = make_sunmmio_target_with_host()

    with tvm.target.Target(target):
        mod = simple_kernel(4, 4)

    mod = run_pre_split_passes(mod, target)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)

    device_funcs = get_device_func(mod)
    assert len(device_funcs) == 1
    script = device_funcs[0].script()
    assert "threadIdx.x\", 1)" in script, (
        f"Expected threadIdx.x extent 1 in device function, got:\n{script}"
    )
    print("test_split_host_device_device_func_is_threadless passed.")


if __name__ == "__main__":
    test_annotate_device_regions_runs_on_sunmmio()
    test_split_host_device_produces_two_functions()
    test_split_host_device_device_func_has_sunmmio_target()
    test_split_host_device_device_func_is_threadless()
    print("All tests passed.")
