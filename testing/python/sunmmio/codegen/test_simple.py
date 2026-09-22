import os
import subprocess

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import make_mxznz_layout, make_mxzz_layout, make_zz_layout

from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import (
    validate_sunmmio_codegen_with_npuir_opt,
)
from tilelang.jit.adapter.sunmmio.libgen import find_npuir_tool


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
# os.environ["SUNMMIO_TEST_LOG_IR"] = "1"

LOOSE_OPT_ARGS = ("--verify-each",)


def _is_mx_dtype(dtype):
    return str(T.dtype(dtype)) in {"custom[mxfp8]8", "custom[mxfp4]4"}


@target("Sunmmio")
def simple_global_copy_gemm_kernel(
    M=32,
    N=32,
    K=32,
    dtype="bfloat16",
    accum_dtype="float32",
    clear_accum=False,
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel():
            A_shared = T.alloc_shared((M, K), dtype)
            B_shared = T.alloc_shared((K, N), dtype)
            C_shared = T.alloc_shared((M, N), accum_dtype)

            T.copy(A[0, 0], A_shared)
            T.copy(B[0, 0], B_shared)
            T.gemm(A_shared, B_shared, C_shared, clear_accum=clear_accum)
            T.copy(C_shared, C[0, 0])

    return main


@target("Sunmmio")
def simple_dynamic_accumulate_gemm_kernel(
    M=32,
    N=32,
    K=32,
    dtype="bfloat16",
    accum_dtype="float32",
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel():
            A_shared = T.alloc_shared((M, K), dtype)
            B_shared = T.alloc_shared((K, N), dtype)
            C_shared = T.alloc_shared((M, N), accum_dtype)

            T.copy(A[0, 0], A_shared)
            T.copy(B[0, 0], B_shared)
            for ko in T.serial(2):
                T.gemm(A_shared, B_shared, C_shared, clear_accum=ko == 0)
            T.copy(C_shared, C[0, 0])

    return main


@target("Sunmmio")
def batch_gemm_kernel(
    batch=2,
    M=32,
    N=32,
    K=32,
    a_batched=True,
    b_batched=True,
    output_batched=True,
    transpose_b=False,
    clear_accum=True,
    load_initial_c=False,
    dynamic_clear=False,
    version=2,
    dtype="bfloat16",
    a_dtype=None,
    b_dtype=None,
    accum_dtype="float32",
):
    a_dtype = dtype if a_dtype is None else a_dtype
    b_dtype = dtype if b_dtype is None else b_dtype
    A_shape = (batch, M, K) if a_batched else (M, K)
    B_matrix_shape = (N, K) if transpose_b else (K, N)
    B_shape = (batch, *B_matrix_shape) if b_batched else B_matrix_shape
    C_shape = (batch, M, N) if output_batched else (M, N)
    placement = T.placement.replicated()
    A_axes = [len(A_shape) - 2, len(A_shape) - 1]
    B_axes = [len(B_shape) - 2, len(B_shape) - 1]
    A_layout = make_mxzz_layout(A_shape, A_axes, dtype=a_dtype) if _is_mx_dtype(a_dtype) else make_zz_layout(A_shape, A_axes, (32, 32))
    B_layout = (
        make_mxzz_layout(B_shape, B_axes, dtype=b_dtype)
        if _is_mx_dtype(b_dtype) and transpose_b
        else make_mxznz_layout(B_shape, B_axes, dtype=b_dtype)
        if _is_mx_dtype(b_dtype)
        else make_zz_layout(B_shape, B_axes, (32, 32))
    )
    C_layout = make_zz_layout(C_shape, [len(C_shape) - 2, len(C_shape) - 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor(A_shape, placement, a_dtype, layout=A_layout),  # type: ignore
        B: T.MeshTensor(B_shape, placement, b_dtype, layout=B_layout),  # type: ignore
        C: T.MeshTensor(C_shape, placement, accum_dtype, layout=C_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(A_shape, a_dtype)
            B_shared = T.alloc_shared(B_shape, b_dtype)
            C_shared = T.alloc_shared(C_shape, accum_dtype)

            if a_batched:
                T.copy(A[0, 0, 0], A_shared)
            else:
                T.copy(A[0, 0], A_shared)
            if b_batched:
                T.copy(B[0, 0, 0], B_shared)
            else:
                T.copy(B[0, 0], B_shared)
            if load_initial_c:
                if output_batched:
                    T.copy(C[0, 0, 0], C_shared)
                else:
                    T.copy(C[0, 0], C_shared)
            gemm = T.gemm_v1 if version == 1 else T.gemm_v2
            if dynamic_clear:
                for ko in T.serial(2):
                    gemm(
                        A_shared,
                        B_shared,
                        C_shared,
                        transpose_B=transpose_b,
                        clear_accum=ko == 0,
                    )
            else:
                gemm(
                    A_shared,
                    B_shared,
                    C_shared,
                    transpose_B=transpose_b,
                    clear_accum=clear_accum,
                )
            if output_batched:
                T.copy(C_shared, C[0, 0, 0])
            else:
                T.copy(C_shared, C[0, 0])

    return main


@target("Sunmmio")
def legacy_singleton_gemm_kernel(version=2):
    A_shape = (1, 1, 32, 32)
    B_shape = (1, 1, 32, 32)
    C_shape = (32, 32)
    placement = T.placement.replicated()
    A_layout = make_zz_layout(A_shape, [2, 3], (32, 32))
    B_layout = make_zz_layout(B_shape, [2, 3], (32, 32))
    C_layout = make_zz_layout(C_shape, [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor(A_shape, placement, "bfloat16", layout=A_layout),  # type: ignore
        B: T.MeshTensor(B_shape, placement, "bfloat16", layout=B_layout),  # type: ignore
        C: T.MeshTensor(C_shape, placement, "float32", layout=C_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(A_shape, "bfloat16")
            B_shared = T.alloc_shared(B_shape, "bfloat16")
            C_shared = T.alloc_shared(C_shape, "float32")
            T.copy(A[0, 0, 0, 0], A_shared)
            T.copy(B[0, 0, 0, 0], B_shared)
            gemm = T.gemm_v1 if version == 1 else T.gemm_v2
            gemm(A_shared, B_shared, C_shared, clear_accum=True)
            T.copy(C_shared, C[0, 0])

    return main


@target("Sunmmio")
def partial_batch_gemm_kernel(
    version=2,
    batch_parent=4,
    batch_begin=1,
    batch_extent=2,
    output_batched=True,
    clear_accum=True,
):
    M = N = K = 32
    A_shape = (batch_parent, M, K)
    B_shape = (batch_parent, K, N)
    C_shape = (batch_parent, M, N) if output_batched else (M, N)
    placement = T.placement.replicated()
    A_layout = make_zz_layout(A_shape, [1, 2], (32, 32))
    B_layout = make_zz_layout(B_shape, [1, 2], (32, 32))
    C_layout = make_zz_layout(C_shape, [len(C_shape) - 2, len(C_shape) - 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor(A_shape, placement, "bfloat16", layout=A_layout),  # type: ignore
        B: T.MeshTensor(B_shape, placement, "bfloat16", layout=B_layout),  # type: ignore
        C: T.MeshTensor(C_shape, placement, "float32", layout=C_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(A_shape, "bfloat16")
            B_shared = T.alloc_shared(B_shape, "bfloat16")
            C_shared = T.alloc_shared(C_shape, "float32")
            T.copy(A[0, 0, 0], A_shared)
            T.copy(B[0, 0, 0], B_shared)
            if not clear_accum:
                if output_batched:
                    T.copy(C[0, 0, 0], C_shared)
                else:
                    T.copy(C[0, 0], C_shared)

            batch_end = batch_begin + batch_extent
            gemm = T.gemm_v1 if version == 1 else T.gemm_v2
            if output_batched:
                gemm(
                    A_shared[batch_begin:batch_end, :, :],
                    B_shared[batch_begin:batch_end, :, :],
                    C_shared[batch_begin:batch_end, :, :],
                    clear_accum=clear_accum,
                )
                T.copy(C_shared, C[0, 0, 0])
            else:
                gemm(
                    A_shared[batch_begin:batch_end, :, :],
                    B_shared[batch_begin:batch_end, :, :],
                    C_shared,
                    clear_accum=clear_accum,
                )
                T.copy(C_shared, C[0, 0])

    return main


def test_simple_global_copy_gemm_codegen_validates_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        simple_global_copy_gemm_kernel(),
        tmp_path,
        mlir_filename="simple_global_copy_gemm_suvm.mlir",
        expected_tokens=(
            "suvm.copy_async",
            "suvm.tc.mma",
        ),
        opt_args=LOOSE_OPT_ARGS,
    )

    assert "sunmmio.fake" not in src
    assert src.count("suvm.copy_async") >= 3
    assert src.count("suvm.tc.mma") == 2
    assert "arith.constant true" in src
    assert src.count("acc = %") == src.count("suvm.tc.mma")


def test_simple_global_copy_gemm_clear_accum_codegen_validates_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        simple_global_copy_gemm_kernel(clear_accum=True),
        tmp_path,
        mlir_filename="simple_global_copy_gemm_clear_accum_suvm.mlir",
        expected_tokens=(
            "suvm.copy_async",
            "suvm.tc.mma",
        ),
        opt_args=LOOSE_OPT_ARGS,
    )

    assert "arith.constant false" in src
    assert src.count("acc = %") == src.count("suvm.tc.mma")


def test_simple_dynamic_accumulate_gemm_codegen_validates_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        simple_dynamic_accumulate_gemm_kernel(),
        tmp_path,
        mlir_filename="simple_dynamic_accumulate_gemm_suvm.mlir",
        expected_tokens=(
            "arith.cmpi ne",
            "suvm.tc.mma",
        ),
        opt_args=LOOSE_OPT_ARGS,
    )

    assert "arith.cmpi ne" in src
    assert src.count("acc = %") == src.count("suvm.tc.mma")


@pytest.mark.parametrize("version", [1, 2])
def test_batch_gemm_rank3_output_codegen_validates_with_npuir_opt(tmp_path, version):
    mlir_filename = f"batch_gemm_v{version}_rank3_output_suvm.mlir"
    src = validate_sunmmio_codegen_with_npuir_opt(
        batch_gemm_kernel(version=version),
        tmp_path,
        mlir_filename=mlir_filename,
        expected_tokens=("suvm.copy_async", "suvm.tc.mma"),
        opt_args=LOOSE_OPT_ARGS,
    )

    assert src.count("suvm.tc.mma") == 2
    assert "!suvm.tile_view<2x32x32xbf16>" in src
    assert "!suvm.tile_view<2x32x32xf32>" in src

    llvm_path = tmp_path / f"batch_gemm_v{version}.ll"
    command = [
        str(find_npuir_tool("npuir-compile")),
        "--target=sunmmio-a4e",
        "--emit=llvm-ir",
        "-o",
        str(llvm_path),
        str(tmp_path / mlir_filename),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    llvm_ir = llvm_path.read_text(encoding="utf-8")
    assert "call void @su_tc_bmm_init(i64 0, i64 2048, i64 2048, i64 4096, i64 4096)" in llvm_ir
    assert llvm_ir.count("i32 2056, i64 2") == 2


def test_batch_gemm_reduction_codegen_validates_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        batch_gemm_kernel(output_batched=False),
        tmp_path,
        mlir_filename="batch_gemm_reduction_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.tc.mma"),
        opt_args=LOOSE_OPT_ARGS,
    )

    assert src.count("suvm.tc.mma") == 2
    assert "!suvm.tile_view<2x32x32xbf16>" in src
    assert "!suvm.tile_view<32x32xf32>" in src
    assert src.count("arith.constant true") == 2
    assert src.count("suvm.tile.fill") == 1


@pytest.mark.parametrize(
    ("a_batched", "b_batched", "output_batched"),
    [
        pytest.param(False, False, True, id="shared-a-shared-w-batched-c"),
        pytest.param(False, True, True, id="shared-a-batched-w-batched-c"),
        pytest.param(True, False, True, id="batched-a-shared-w-batched-c"),
        pytest.param(False, True, False, id="shared-a-batched-w-reduction"),
        pytest.param(True, False, False, id="batched-a-shared-w-reduction"),
    ],
)
def test_batch_gemm_mixed_rank_codegen_validates_with_npuir_opt(tmp_path, a_batched, b_batched, output_batched):
    src = validate_sunmmio_codegen_with_npuir_opt(
        batch_gemm_kernel(
            a_batched=a_batched,
            b_batched=b_batched,
            output_batched=output_batched,
        ),
        tmp_path,
        mlir_filename=(f"batch_gemm_a{3 if a_batched else 2}_w{3 if b_batched else 2}_c{3 if output_batched else 2}_suvm.mlir"),
        expected_tokens=("suvm.copy_async", "suvm.tc.mma"),
        opt_args=LOOSE_OPT_ARGS,
    )

    assert src.count("suvm.tc.mma") == 2
    if not output_batched:
        assert "arith.constant true" in src


def test_batch_gemm_preserves_rank3_unit_batch(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        batch_gemm_kernel(batch=1),
        tmp_path,
        mlir_filename="batch_gemm_unit_batch_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.tc.mma"),
        opt_args=LOOSE_OPT_ARGS,
    )

    assert src.count("suvm.tc.mma") == 2
    assert "!suvm.tile_view<1x32x32xbf16>" in src
    assert "!suvm.tile_view<1x32x32xf32>" in src


def test_batch_gemm_reduction_without_clear_still_accumulates(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        batch_gemm_kernel(output_batched=False, clear_accum=False),
        tmp_path,
        mlir_filename="batch_gemm_reduction_no_clear_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.tc.mma"),
        opt_args=LOOSE_OPT_ARGS,
    )

    assert src.count("suvm.tc.mma") == 2
    assert "arith.constant true" in src


def test_batch_gemm_reduction_dynamic_clear_is_conditional(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        batch_gemm_kernel(output_batched=False, dynamic_clear=True),
        tmp_path,
        mlir_filename="batch_gemm_reduction_dynamic_clear_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.tc.mma"),
        opt_args=LOOSE_OPT_ARGS,
    )

    assert src.count("suvm.tc.mma") == 2
    assert src.count("arith.constant true") == 2
    assert src.count("suvm.tile.fill") == 1
    assert "scf.if" in src


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("output_batched", [True, False])
def test_batch_gemm_m16_uses_full_physical_mma_carrier(tmp_path, version, output_batched):
    output_rank = 3 if output_batched else 2
    mlir_filename = f"batch_gemm_v{version}_m16_c{output_rank}_suvm.mlir"
    src = validate_sunmmio_codegen_with_npuir_opt(
        batch_gemm_kernel(M=16, version=version, output_batched=output_batched),
        tmp_path,
        mlir_filename=mlir_filename,
        expected_tokens=("suvm.copy_async", "suvm.tc.mma"),
        opt_args=LOOSE_OPT_ARGS,
    )

    assert "!suvm.tile_view<2x32x32xbf16>" in src
    mma_output_type = "!suvm.tile_view<2x32x32xf32>" if output_batched else "!suvm.tile_view<32x32xf32>"
    assert mma_output_type in src
    assert "suvm.transform_layout_async" not in src
    logical_output_type = "!suvm.tile_view<2x16x32xf32>" if output_batched else "!suvm.tile_view<16x32xf32>"
    assert logical_output_type in src

    llvm_path = tmp_path / f"batch_gemm_v{version}_m16_c{output_rank}.ll"
    result = subprocess.run(
        [
            str(find_npuir_tool("npuir-compile")),
            "--target=sunmmio-a4e",
            "--emit=llvm-ir",
            "-o",
            str(llvm_path),
            str(tmp_path / mlir_filename),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("output_batched", [True, False])
def test_batch_gemm_transposed_weight_validates(tmp_path, output_batched):
    src = validate_sunmmio_codegen_with_npuir_opt(
        batch_gemm_kernel(transpose_b=True, output_batched=output_batched),
        tmp_path,
        mlir_filename=f"batch_gemm_trans_b_c{3 if output_batched else 2}.mlir",
        expected_tokens=("suvm.tc.mma",),
        opt_args=LOOSE_OPT_ARGS,
    )

    assert src.count("suvm.tc.mma") == 2
    assert "trans" in src


@pytest.mark.parametrize("output_batched", [True, False])
def test_batch_gemm_bf16_output_validates(tmp_path, output_batched):
    src = validate_sunmmio_codegen_with_npuir_opt(
        batch_gemm_kernel(accum_dtype="bfloat16", output_batched=output_batched),
        tmp_path,
        mlir_filename=f"batch_gemm_bf16_c{3 if output_batched else 2}.mlir",
        expected_tokens=("suvm.tc.mma",),
        opt_args=LOOSE_OPT_ARGS,
    )

    assert "xbf16>" in src


@pytest.mark.parametrize(
    ("mx_dtype", "a_is_mx", "output_batched", "transpose_b"),
    [
        pytest.param(mx_dtype, a_is_mx, output_batched, transpose_b, id=case_id)
        for mx_dtype, mx_name in ((T.mxfp8, "mxfp8"), (T.mxfp4, "mxfp4"))
        for a_is_mx, output_batched, transpose_b, case_id in (
            (False, True, True, f"bf16-{mx_name}-output-transposed"),
            (False, False, True, f"bf16-{mx_name}-reduction-transposed"),
            (True, True, True, f"{mx_name}-{mx_name}-output-transposed"),
            (False, True, False, f"bf16-{mx_name}-output-nontransposed"),
        )
    ],
)
def test_batch_gemm_mx_codegen_and_physical_stride(tmp_path, mx_dtype, a_is_mx, output_batched, transpose_b):
    a_dtype = mx_dtype if a_is_mx else T.bfloat16
    mx_name = "mxfp8" if mx_dtype == T.mxfp8 else "mxfp4"
    case = f"batch_gemm_{mx_name if a_is_mx else 'bf16'}_{mx_name}_c{3 if output_batched else 2}_{'mt' if transpose_b else 'mn'}"
    mlir_path = tmp_path / f"{case}.mlir"
    src = validate_sunmmio_codegen_with_npuir_opt(
        batch_gemm_kernel(
            K=64,
            a_dtype=a_dtype,
            b_dtype=mx_dtype,
            output_batched=output_batched,
            transpose_b=transpose_b,
        ),
        tmp_path,
        mlir_filename=mlir_path.name,
        expected_tokens=(f"!suvm.{mx_name}", "suvm.copy_async", "suvm.tc.mma"),
        opt_args=LOOSE_OPT_ARGS,
    )

    mma_count = 1 if _is_mx_dtype(a_dtype) else 2
    assert src.count("suvm.tc.mma") == mma_count
    b_matrix_shape = "32x64" if transpose_b else "64x32"
    assert f"!suvm.tile_view<2x{b_matrix_shape}x!suvm.{mx_name}>" in src
    if _is_mx_dtype(a_dtype):
        assert src.count(f"!suvm.tile_view<2x32x64x!suvm.{mx_name}>") >= 4

    llvm_path = tmp_path / f"{case}.ll"
    result = subprocess.run(
        [
            str(find_npuir_tool("npuir-compile")),
            "--target=sunmmio-a4e",
            "--emit=llvm-ir",
            "-o",
            str(llvm_path),
            str(mlir_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    llvm_ir = llvm_path.read_text(encoding="utf-8")
    # Both MXFP8 and MXFP4 occupy 1024 bytes per 32x32 block in the
    # TensorCore-facing ASRAM/WSRAM storage view.
    mx_stride = 2048
    a_stride = mx_stride if _is_mx_dtype(a_dtype) else 4096
    c_stride = 4096 if output_batched else 0
    assert f"call void @su_tc_bmm_init(i64 0, i64 {a_stride}, i64 {mx_stride}, i64 {c_stride}, i64 {c_stride})" in llvm_ir
    assert llvm_ir.count("i32 2056, i64 2") == mma_count


def test_batch_gemm_batch4_validates(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        batch_gemm_kernel(batch=4),
        tmp_path,
        mlir_filename="batch_gemm_b4.mlir",
        expected_tokens=("suvm.tc.mma",),
        opt_args=LOOSE_OPT_ARGS,
    )

    assert "!suvm.tile_view<4x32x32xbf16>" in src


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("output_batched", [True, False])
def test_partial_batch_gemm_uses_compact_physical_allocation(tmp_path, version, output_batched):
    output_rank = 3 if output_batched else 2
    mlir_path = tmp_path / f"partial_batch_v{version}_c{output_rank}.mlir"
    src = validate_sunmmio_codegen_with_npuir_opt(
        partial_batch_gemm_kernel(version=version, output_batched=output_batched),
        tmp_path,
        mlir_filename=mlir_path.name,
        expected_tokens=("suvm.copy_async", "suvm.tc.mma"),
        opt_args=LOOSE_OPT_ARGS,
    )

    assert src.count("suvm.tc.mma") == 2
    assert src.count("suvm.memtensor<2x32x32xbf16") >= 2
    assert "!suvm.tile_view<2x32x32xbf16>" in src
    if output_batched:
        assert "suvm.memtensor<2x32x32xf32" in src
        assert "!suvm.tile_view<2x32x32xf32>" in src

    llvm_path = tmp_path / f"partial_batch_v{version}_c{output_rank}.ll"
    result = subprocess.run(
        [
            str(find_npuir_tool("npuir-compile")),
            "--target=sunmmio-a4e",
            "--emit=llvm-ir",
            "-o",
            str(llvm_path),
            str(mlir_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    llvm_ir = llvm_path.read_text(encoding="utf-8")
    c_stride = 4096 if output_batched else 0
    assert f"call void @su_tc_bmm_init(i64 0, i64 2048, i64 2048, i64 {c_stride}, i64 {c_stride})" in llvm_ir


@pytest.mark.parametrize("version", [1, 2])
def test_batch_gemm_dynamic_batch_has_target_diagnostic(tmp_path, version):
    with pytest.raises(Exception, match="positive static IntImm"):
        validate_sunmmio_codegen_with_npuir_opt(
            batch_gemm_kernel(batch=T.dynamic(f"batch_v{version}"), version=version),
            tmp_path,
            mlir_filename=f"batch_gemm_dynamic_v{version}.mlir",
            opt_args=LOOSE_OPT_ARGS,
        )


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_rank4_singleton_gemm_stays_supported(tmp_path, version):
    src = validate_sunmmio_codegen_with_npuir_opt(
        legacy_singleton_gemm_kernel(version=version),
        tmp_path,
        mlir_filename=f"legacy_singleton_gemm_v{version}.mlir",
        expected_tokens=("suvm.tc.mma",),
        opt_args=LOOSE_OPT_ARGS,
    )

    assert "!suvm.tile_view<32x32xbf16>" in src
    assert "!suvm.tile_view<1x1x32x32xbf16>" not in src


if __name__ == "__main__":
    tilelang.testing.main()
