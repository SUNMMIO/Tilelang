"""Explicit DRAM buffer aliases used by C4."""

import pytest
from tilelang import tvm
import tilelang.language as T
from tilelang.layout import make_aligned_row_major, make_row_major, make_zz_layout

from testing.python.sunmmio.common.codegen_validation import validate_sunmmio_codegen_with_npuir_opt
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
