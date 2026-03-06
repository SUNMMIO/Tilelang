import tilelang
import tilelang.language as T
import tilelang.transform
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target, target_is_sunmmio
from tilelang.engine.lower import canon_target_host
from tvm.ir import CallingConv

tilelang.env.disable_cache()


def make_sunmmio_target_with_host():
    """Build the Sunmmio target with an llvm host, matching what lower() does."""
    target = determine_target("Sunmmio", return_object=True)
    target_host = tvm.target.Target.canon_target(canon_target_host(target, None))
    return tvm.target.Target(target, target_host)


def simple_kernel(M, N, dtype=T.float32):

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(M, N) as (bx, by):
            B[bx, by] = A[bx, by]

    return tvm.IRModule({"main": main})


def test_annotate_device_regions_runs_on_sunmmio():
    """AnnotateDeviceRegions should not crash and should annotate thread_extent regions."""
    target = make_sunmmio_target_with_host()

    with tvm.target.Target(target):
        mod = simple_kernel(4, 4)

    mod = tvm.tir.transform.BindTarget(target)(mod)
    # Should not raise
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)

    # The device target on the annotated region must be the Sunmmio target
    script = mod.script()
    assert "sunmmio" in script.lower(), (
        f"Expected Sunmmio target annotation in script, got:\n{script}"
    )


def test_split_host_device_produces_two_functions():
    """SplitHostDevice should split the module into a host and a device function."""
    target = make_sunmmio_target_with_host()

    with tvm.target.Target(target):
        mod = simple_kernel(4, 4)

    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)

    funcs = dict(mod.functions)
    assert len(funcs) == 2, (
        f"Expected 2 functions after SplitHostDevice, got {len(funcs)}: {list(funcs.keys())}"
    )


def test_split_host_device_device_func_has_sunmmio_target():
    """The device function produced by SplitHostDevice should carry the Sunmmio target."""
    target = make_sunmmio_target_with_host()

    with tvm.target.Target(target):
        mod = simple_kernel(4, 4)

    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)

    device_funcs = [
        f for f in mod.functions.values()
        if f.attrs.get("calling_conv", None) == CallingConv.DEVICE_KERNEL_LAUNCH
    ]
    assert len(device_funcs) == 1, (
        f"Expected exactly 1 device function, got {len(device_funcs)}"
    )
    device_func = device_funcs[0]
    func_target = device_func.attrs["target"]
    assert target_is_sunmmio(func_target), (
        f"Expected device function target to be Sunmmio, got: {func_target}"
    )


def test_split_host_device_device_func_is_threadless():
    """The device function should have threadIdx.x with extent 1 (threadless)."""
    target = make_sunmmio_target_with_host()

    with tvm.target.Target(target):
        mod = simple_kernel(4, 4)

    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)

    device_funcs = [
        f for f in mod.functions.values()
        if f.attrs.get("calling_conv", None) == CallingConv.DEVICE_KERNEL_LAUNCH
    ]
    assert len(device_funcs) == 1
    script = device_funcs[0].script()
    assert "threadIdx.x\", 1)" in script, (
        f"Expected threadIdx.x extent 1 in device function, got:\n{script}"
    )


if __name__ == "__main__":
    test_annotate_device_regions_runs_on_sunmmio()
    test_split_host_device_produces_two_functions()
    test_split_host_device_device_func_has_sunmmio_target()
    test_split_host_device_device_func_is_threadless()
    print("All tests passed.")
