# type: ignore
import torch
import tilelang
import tilelang.testing
import tilelang.language as T


# TODO(dyq) It intentionally triggers a device-side assert so we can't include this in CI
# Please run manually when you want to verify that device_assert actually traps on GPU.
def _manual_device_assert_triggered():

    @T.prim_func
    def program():
        with T.Kernel(threads=128):
            tid = T.get_thread_binding()
            T.device_assert(tid > 0, "Assertion Trigger !")

    jit_kernel = tilelang.compile(program, target="cuda")
    profiler = jit_kernel.get_profiler()
    profiler.run_once()


@tilelang.testing.requires_cuda
def test_device_assert_no_trigger():
    # Ensure a valid CUDA runtime context is current on this thread before
    # using driver API calls. Without this, cuModuleLoadData can fail with
    # CUDA_ERROR_INVALID_CONTEXT for kernels that don't touch any device
    # memory (i.e., no tensor parameters to implicitly create a context).
    # See upstream: testing/python/issue/test_tilelang_issue_830.py
    torch.cuda.set_device(0)

    @T.prim_func
    def program():
        with T.Kernel(threads=128):
            tid = T.get_thread_binding()
            T.device_assert(tid == tid)

    jit_kernel = tilelang.compile(program, target="cuda")
    profiler = jit_kernel.get_profiler()
    profiler.run_once()


if __name__ == "__main__":
    _manual_device_assert_triggered()
