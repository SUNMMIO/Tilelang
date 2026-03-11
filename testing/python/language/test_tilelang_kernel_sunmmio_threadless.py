import pytest
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target
import tilelang.language as T


def make_kernel(threads):

    @T.prim_func
    def kernel(A: T.Tensor((128,), T.float32)):
        with T.Kernel(128, threads=threads):
            pass

    return tvm.IRModule({"main": kernel})


@pytest.mark.parametrize("requested_threads", [1, 32, 128, 256])
def test_sunmmio_kernel_threads_always_one(requested_threads):
    """T.Kernel threads= is overridden to 1 for Sunmmio target, regardless of the requested value."""
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = make_kernel(requested_threads)

    script = mod.script()
    assert "threadIdx_x, 1)" in script, f"Expected threadIdx_x extent to be 1, but got:\n{script}"


def test_sunmmio_kernel_default_threads_is_one():
    """When threads= is not specified, Sunmmio target still defaults to 1 (not 128)."""
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):

        @T.prim_func
        def kernel(A: T.Tensor((128,), T.float32)):
            with T.Kernel(128):
                pass

        mod = tvm.IRModule({"main": kernel})

    script = mod.script()
    assert "threadIdx_x, 1)" in script, f"Expected threadIdx_x extent to be 1, but got:\n{script}"


def test_non_sunmmio_kernel_respects_threads():
    """T.Kernel threads= is not overridden for non-Sunmmio targets."""
    with tvm.target.Target("llvm"):
        mod = make_kernel(128)

    script = mod.script()
    assert "threadIdx_x, 128)" in script, f"Expected threadIdx_x extent to be 128, but got:\n{script}"
