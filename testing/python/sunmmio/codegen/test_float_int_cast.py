import pytest

import tilelang
import tilelang.language as T
from tilelang.layout import make_row_major
from testing.python.sunmmio.common.codegen_validation import (
    codegen_sunmmio_suvm_mlir,
    lower_sunmmio_kernel_to_device_tir,
    validate_suvm_mlir_with_npuir_opt,
)


def cast_kernel(dst_dtype, count=1024, vector=True, src_dtype="float32"):
    shape = (1024,)
    placement = T.placement.replicated()
    layout = make_row_major(shape)

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, placement, src_dtype, layout=layout),
        B: T.MeshTensor(shape, placement, dst_dtype, layout=layout),
    ):
        with T.Kernel():
            a = T.alloc_shared(shape, src_dtype, scope="shared.rsram")
            b = T.alloc_shared(shape, dst_dtype, scope="shared.rsram")
            T.copy(A, a)
            if count < 1024:
                T.fill(b, T.cast(73, dst_dtype))
            if vector:
                for j in T.Tiles((count,), parallel=True):
                    b[j] = T.cast(a[j], dst_dtype)
            else:
                for j in T.serial(count):
                    b[j] = T.cast(a[j], dst_dtype)
            T.copy(b, B)

    return main


@pytest.mark.parametrize("dst_dtype", ["int32", "uint32"])
@pytest.mark.parametrize("count", [1024, 257])
def test_float_int_cast_codegen(dst_dtype, count, tmp_path):
    src = codegen_sunmmio_suvm_mlir(lower_sunmmio_kernel_to_device_tir(cast_kernel(dst_dtype, count)))
    lowered = validate_suvm_mlir_with_npuir_opt(src, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))
    assert "llvm.riscv.sunmmio.vftrunc" in lowered.stdout
    assert f"llvm.riscv.sunmmio.vcvt.{dst_dtype}.fp32" in lowered.stdout
    if dst_dtype == "uint32":
        assert "llvm.riscv.sunmmio.vcvt.int32.fp32" not in lowered.stdout


@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [("int32", "uint16"), ("uint32", "uint16"), ("float32", "bfloat16"), ("bfloat16", "float32")],
)
def test_other_casts_keep_conversion_semantics(src_dtype, dst_dtype, tmp_path):
    src = codegen_sunmmio_suvm_mlir(lower_sunmmio_kernel_to_device_tir(cast_kernel(dst_dtype, src_dtype=src_dtype)))
    lowered = validate_suvm_mlir_with_npuir_opt(src, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))
    assert "llvm.riscv.sunmmio.vftrunc" not in lowered.stdout
    if dst_dtype == "uint16":
        assert "llvm.riscv.sunmmio.vcvt.uint16.uint32" in lowered.stdout
        assert "llvm.riscv.sunmmio.vcvt.uint16.int32" not in lowered.stdout


@pytest.mark.sunmmio_closed_runtime
@pytest.mark.sunmmio_toolchain
@pytest.mark.sunmmio_sunsim
@pytest.mark.parametrize("dst_dtype", ["int32", "uint32"])
@pytest.mark.parametrize("count", [1024, 257])
@pytest.mark.parametrize("vector", [True, False], ids=["vector", "scalar"])
def test_float_int_cast_device(dst_dtype, count, vector, tmp_path):
    np = pytest.importorskip("numpy")
    sunsim = pytest.importorskip("sunsim")
    values = [0.0, -0.0, 0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2.5, 3.5, -0.25, -0.5, -0.75, 2**24, 2**30]
    if dst_dtype == "int32":
        values += [-1.25, -1.5, -1.75, -2.5, -3.5, -(2**31), 2**31 - 128]
    else:
        values += [2**31, 3 * 2**30, 2**32 - 256]
    data = np.resize(np.array(values, dtype=np.float32), 1024)
    rng = np.random.default_rng(20261124)
    data[256:] = rng.uniform(-1000 if dst_dtype == "int32" else 0, 1000, 768).astype(np.float32)
    # All truncated inputs fit the destination. The suffix checks tail stores.
    expected = np.full(data.shape, 73, dtype=dst_dtype)
    expected[:count] = data[:count].astype(dst_dtype)
    kernel = tilelang.compile(cast_kernel(dst_dtype, count, vector), target="sunmmio", execution_backend="sunmmio_sunsim")
    args = dict(placement=[sunsim.R(), sunsim.R()], layout=sunsim.layout.row_major())
    result = kernel(
        sunsim.Input(data, **args),
        sunsim.Output(data.shape, np.dtype(dst_dtype), **args),
        working_dir=tmp_path,
        timeout=120,
    )
    assert result.exit_code == 0
    for core in range(16):
        staged = np.fromfile(tmp_path / "inputs" / "arg0" / f"core{core}.dat", np.float32)
        actual = np.fromfile(tmp_path / "outputs" / "arg1" / f"core{core}.dat", dst_dtype)
        assert staged.tobytes() == data.tobytes()
        np.testing.assert_array_equal(actual, expected, err_msg=f"core {core}")


@pytest.mark.sunmmio_closed_runtime
@pytest.mark.sunmmio_toolchain
@pytest.mark.sunmmio_sunsim
@pytest.mark.parametrize("src_dtype", ["int32", "uint32"])
def test_integer_narrowing_device(src_dtype, tmp_path):
    np = pytest.importorskip("numpy")
    sunsim = pytest.importorskip("sunsim")
    bits = np.resize(np.array([0, 1, 65535, 65536, 0x80000000, 0xFFFFFFFF, 0xFFFF8001], dtype=np.uint32), 1024)
    data = bits.view(src_dtype)
    kernel = tilelang.compile(cast_kernel("uint16", src_dtype=src_dtype), target="sunmmio", execution_backend="sunmmio_sunsim")
    args = dict(placement=[sunsim.R(), sunsim.R()], layout=sunsim.layout.row_major())
    result = kernel(
        sunsim.Input(data, **args),
        sunsim.Output(data.shape, np.uint16, **args),
        working_dir=tmp_path,
        timeout=120,
    )
    assert result.exit_code == 0
    for core in range(16):
        actual = np.fromfile(tmp_path / "outputs" / "arg1" / f"core{core}.dat", np.uint16)
        np.testing.assert_array_equal(actual, data.astype(np.uint16), err_msg=f"core {core}")
