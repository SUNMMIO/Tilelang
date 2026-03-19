import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm as tvm


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


def lower_to_sunmmio_source(func) -> str:
    with tvm.transform.PassContext(), tvm.target.Target("Sunmmio"):
        artifact = tilelang.lower(func, target="Sunmmio", enable_device_compile=False)
    assert artifact.kernel_source is not None
    return artifact.kernel_source


def test_sunmmio_codegen_without_compile_emits_skeleton_source():
    src = lower_to_sunmmio_source(simple_add_kernel())
    assert "sunmmio.module {" in src
    assert "sunmmio.func" in src
    assert "sunmmio.for" in src
    assert "sunmmio.load" in src
    assert "sunmmio.addf" in src
    assert "sunmmio.store" in src
    assert "sunmmio.return" in src


def test_sunmmio_codegen_compile_path_not_implemented():
    with tvm.transform.PassContext(), tvm.target.Target("Sunmmio"):
        with pytest.raises(Exception, match="not implemented yet"):
            tilelang.lower(simple_add_kernel(), target="Sunmmio", enable_device_compile=True)


if __name__ == "__main__":
    tilelang.testing.main()
