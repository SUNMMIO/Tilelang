"""Compare contextual negative copies with equivalent positive-coordinate IR."""

import pytest
import tilelang
from tilelang import tvm
from tilelang.engine.lower import lower
from testing.python.sunmmio.common.mx_view_kernels import context_kernel, mesh_target
from testing.python.sunmmio.common.codegen_validation import validate_suvm_mlir_with_npuir_opt
from testing.python.sunmmio.common.mx_view_validation import assert_mx_view_resolved


@pytest.mark.parametrize("case", ["then", "else", "nested", "thread", "loop", "late_shape", "slice"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("force_inline", [False, True])
def test_contextual_copy_matches_positive_ir(case, reverse, force_inline, tmp_path):
    with tilelang.transform.PassContext(config={"tl.force_let_inline": force_inline}):
        negative = lower(context_kernel(case, reverse=reverse), target=mesh_target((4, 4)))
        positive = lower(context_kernel(case, reverse=reverse, positive=True), target=mesh_target((4, 4)))
    assert_mx_view_resolved(negative.device_mod)
    assert tvm.tir.analysis.verify_well_formed(negative.device_mod)
    for function in negative.device_mod.functions.values():
        assert tvm.tir.analysis.verify_ssa(function)
    # Compare the actual device program, including reads and writes. Matching
    # a positive-coordinate control rules out replacing a valid read with fill.
    assert negative.kernel_source == positive.kernel_source
    validate_suvm_mlir_with_npuir_opt(negative.kernel_source, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))


@pytest.mark.parametrize("bits", [8, 4])
def test_aligned_scale_load_preserves_column_broadcast(bits, tmp_path):
    artifact = lower(context_kernel("then", bits=bits), target=mesh_target((4, 4)))
    source = artifact.kernel_source
    # peer[0, j] varies with the column loop; a padded 1D load must become a
    # 1x32 tile before it is broadcast to the 32x32 output.
    assert "to !suvm.tile<1x32xf8E8M0FNU>" in source or "-> !suvm.tile<1x32xf8E8M0FNU>" in source
    validate_suvm_mlir_with_npuir_opt(source, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("layout_kind", ["mxzz", "mxznz", "row_major"])
@pytest.mark.parametrize("reverse", [False, True])
def test_contextual_copy_layouts_on_small_mesh(bits, layout_kind, reverse, tmp_path):
    # Keep layout/dtype and small-mesh coverage independent of simulator tests.
    with tilelang.transform.PassContext(config={"tl.force_let_inline": True}):
        negative = lower(context_kernel("thread", bits, reverse, layout_kind), target=mesh_target((2, 2)))
        positive = lower(context_kernel("thread", bits, reverse, layout_kind, positive=True), target=mesh_target((2, 2)))
    assert_mx_view_resolved(negative.device_mod)
    assert tvm.tir.analysis.verify_well_formed(negative.device_mod)
    for function in negative.device_mod.functions.values():
        assert tvm.tir.analysis.verify_ssa(function)
    assert negative.kernel_source == positive.kernel_source
    validate_suvm_mlir_with_npuir_opt(negative.kernel_source, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))
