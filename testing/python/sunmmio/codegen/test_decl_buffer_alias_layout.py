"""Explicit DRAM buffer aliases used by C4."""

import pytest
from tilelang import tvm
import tilelang.language as T
from tilelang.layout import make_aligned_row_major, make_row_major, make_zz_layout

from testing.python.sunmmio.common.codegen_validation import (
    codegen_sunmmio_suvm_mlir,
    lower_sunmmio_kernel_to_device_tir,
    validate_sunmmio_codegen_with_npuir_opt,
)
from testing.python.sunmmio.common.compile_pipeline import target


@target("Sunmmio")
def c4_sharded_copy(layout_kind="dense"):
    shape = (32 * T.mesh_nrows(), 512 * T.mesh_ncols())
    placement = T.placement.full_shard(0, 1)
    layout = make_row_major(shape)
    if layout_kind == "blocked":
        layout = make_zz_layout(shape)
    elif layout_kind == "padded":
        shape = (32, 128)
        placement = T.placement.replicated()
        layout = make_aligned_row_major(shape, T.bfloat16, 1024)

    @T.prim_func
    def main(
        source: T.MeshTensor(shape, placement, T.bfloat16, layout=layout),
        output: T.MeshTensor(shape, placement, T.bfloat16, layout=layout),
    ):
        with (
            T.Kernel(T.mesh_ncores()),
            T.decl_buffer((source.local_shape[0] * source.local_shape[1],), T.bfloat16, data=source.data) as src,
            T.decl_buffer((output.local_shape[0] * output.local_shape[1],), T.bfloat16, data=output.data) as dst,
        ):
            local = T.alloc_shared((source.local_shape[0] * source.local_shape[1],), T.bfloat16, scope="shared.rsram")
            T.copy(src, local)
            T.copy(local, dst)

    return main


def test_sharded_decl_buffer_views(tmp_path):
    mlir = validate_sunmmio_codegen_with_npuir_opt(c4_sharded_copy(), tmp_path)
    assert "16384xbf16, #suvm.layout<(16384), (1)>, #suvm.memory_space<global>" in mlir
    assert "!suvm.tile_view<16384xbf16>" in mlir
    assert mlir.count("suvm.copy_async") == 2
    assert "suvm.transform_layout_async" not in mlir


@pytest.mark.parametrize("layout_kind", ["blocked", "padded"])
def test_non_dense_decl_buffer_is_not_flattened(layout_kind, tmp_path):
    with pytest.raises(tvm.error.InternalError, match="layout_dim_levels rank mismatch"):
        validate_sunmmio_codegen_with_npuir_opt(c4_sharded_copy(layout_kind), tmp_path)


def declared_flat_copy(shape, strides, *, write_to_matrix=False):
    length = shape[0] * shape[1]

    @T.prim_func
    def main(matrix: T.StridedTensor(shape, strides, T.bfloat16), vector: T.Tensor((length,), T.bfloat16)):
        with T.Kernel(T.mesh_ncores()), T.decl_buffer((length,), T.bfloat16, data=matrix.data) as flat:
            local = T.alloc_shared((length,), T.bfloat16, scope="shared.rsram")
            if write_to_matrix:
                T.copy(vector, local)
                T.copy(local, flat)
            else:
                T.copy(flat, local)
                T.copy(local, vector)

    return main


@pytest.mark.parametrize("strides", [(1024, 1), (1, 8)])
def test_decl_buffer_rejects_noncompact_backing_strides(strides, tmp_path):
    with pytest.raises(tvm.error.InternalError, match="layout_dim_levels rank mismatch"):
        validate_sunmmio_codegen_with_npuir_opt(declared_flat_copy((8, 512), strides), tmp_path)


def test_flat_sibling_layout_cannot_hide_backing_strides():
    mod = lower_sunmmio_kernel_to_device_tir(declared_flat_copy((8, 512), (1024, 1), write_to_matrix=True))
    ((symbol, func),) = mod.functions.items()
    # The flat alias and the backing parameter share a data variable. Put
    # the inferred flat entry before the parameter, as can happen after lowering.
    layouts = dict(sorted(func.attrs["global_layout_map"].items(), key=lambda item: len(item[0].shape)))
    mod.update_func(symbol, func.with_attr("global_layout_map", layouts))
    with pytest.raises(tvm.error.InternalError, match="layout_dim_levels rank mismatch"):
        codegen_sunmmio_suvm_mlir(mod)


@pytest.mark.parametrize("shape,strides", [((8, 512), (512, 1)), ((1, 512), (1024, 1))])
def test_decl_buffer_accepts_contiguous_backing_strides(shape, strides, tmp_path):
    ir = validate_sunmmio_codegen_with_npuir_opt(declared_flat_copy(shape, strides), tmp_path)
    assert ir.count("suvm.copy_async") == 2
    assert "suvm.transform_layout_async" not in ir


@target("Sunmmio")
def read_declared_row():
    placement = T.placement.replicated()

    @T.prim_func
    def main(
        source: T.MeshTensor((1024,), placement, T.bfloat16, layout=make_row_major((1024,))),
        output: T.MeshTensor((512,), placement, T.bfloat16, layout=make_row_major((512,))),
    ):
        with T.Kernel(T.mesh_ncores()), T.decl_buffer((2, 512), T.bfloat16, data=source.data) as matrix:
            row = T.alloc_shared((512,), T.bfloat16, scope="shared.rsram")
            T.copy(matrix[1, 0:512], row)
            T.copy(row, output)

    return main


def test_decl_buffer_can_address_second_row(tmp_path):
    mlir = validate_sunmmio_codegen_with_npuir_opt(read_declared_row(), tmp_path)
    assert "2x512xbf16, #suvm.layout<(2, 512), (512, 1)>, #suvm.memory_space<global>" in mlir
    assert mlir.count("suvm.copy_async") == 2
    assert "suvm.transform_layout_async" not in mlir
