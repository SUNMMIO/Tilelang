import pytest
import tilelang.language as T
import tilelang.testing
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target


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


def build_sunmmio_module_without_compile(func):
    target = determine_target("Sunmmio", return_object=True)
    func = func.with_attr("global_symbol", "main")
    func = func.with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
    mod = tvm.IRModule({"main": func})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    return builder(mod, target)


def test_sunmmio_codegen_without_compile_emits_skeleton_source():
    src = build_sunmmio_module_without_compile(simple_add_kernel()).inspect_source()
    print(src)
    assert "sunmmio.module {" in src
    assert "sunmmio.func" in src
    assert "sunmmio.for" in src
    assert "sunmmio.load" in src
    assert "sunmmio.add" in src
    assert "sunmmio.store" in src
    assert "sunmmio.return" in src


def test_sunmmio_codegen_compile_path_not_implemented():
    target = determine_target("Sunmmio", return_object=True)
    func = simple_add_kernel().with_attr("global_symbol", "main")
    func = func.with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
    mod = tvm.IRModule({"main": func})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio")
    with pytest.raises(Exception, match="not implemented yet"):
        builder(mod, target)


if __name__ == "__main__":
    tilelang.testing.main()
