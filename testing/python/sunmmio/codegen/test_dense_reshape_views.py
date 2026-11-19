"""Rank-changing views of per-core contiguous, row-major storage."""

import pytest
from tilelang import tvm
import tilelang.language as T
from tilelang.layout import make_aligned_row_major, make_row_major, make_zz_layout

from testing.python.sunmmio.common.codegen_validation import validate_sunmmio_codegen_with_npuir_opt
from testing.python.sunmmio.common.compile_pipeline import target


@target("Sunmmio")
def reshape_copy(source_shape, view_shape, *, layout_kind="dense"):
    placement = T.placement.replicated()
    source_layout = make_row_major(source_shape)
    if layout_kind == "blocked":
        source_layout = make_zz_layout(source_shape)
    elif layout_kind == "padded":
        source_layout = make_aligned_row_major(source_shape, T.bfloat16, 1024)
    view_layout = make_row_major(view_shape)

    @T.prim_func
    def main(
        source: T.MeshTensor(source_shape, placement, T.bfloat16, layout=source_layout),
        output: T.MeshTensor(view_shape, placement, T.bfloat16, layout=view_layout),
    ):
        with T.Kernel(T.mesh_ncores()):
            source_view = T.reshape(source.buffer, view_shape)
            local = T.alloc_shared(view_shape, T.bfloat16, scope="shared.rsram")
            # Isolate the DRAM reshape from any conversion of the RSRAM staging buffer.
            T.annotate_layout({local: view_layout})
            T.copy(source_view, local)
            T.copy(local, output)

    return main


@pytest.mark.parametrize(
    "source_shape,view_shape,expected_layout",
    [
        ((8, 128), (1024,), "(1024), (1)"),
        ((1024,), (8, 128), "(8, 128), (128, 1)"),
        ((2, 4, 128), (1024,), "(1024), (1)"),
        ((1024,), (2, 4, 128), "(2, 4, 128), (512, 128, 1)"),
    ],
)
def test_dense_reshape(source_shape, view_shape, expected_layout, tmp_path):
    mlir = validate_sunmmio_codegen_with_npuir_opt(reshape_copy(source_shape, view_shape), tmp_path)
    shape = "x".join(map(str, view_shape))
    arg_type = f"!suvm.memtensor<{shape}xbf16, #suvm.layout<{expected_layout}>, #suvm.memory_space<global>>"
    signature = next(line for line in mlir.splitlines() if "func.func @main_kernel(" in line)
    assert signature.count(arg_type) == 2
    assert mlir.count("suvm.copy_async") == 2
    assert "suvm.transform_layout_async" not in mlir


@pytest.mark.parametrize("layout_kind", ["padded", "blocked"])
def test_non_dense_reshape_requires_relayout(layout_kind, tmp_path):
    with pytest.raises(tvm.error.InternalError, match="requires compact row-major storage"):
        validate_sunmmio_codegen_with_npuir_opt(reshape_copy((32, 128), (4096,), layout_kind=layout_kind), tmp_path)


@target("Sunmmio")
def c4_sharded_copy():
    shape = (32 * T.mesh_nrows(), 512 * T.mesh_ncols())
    placement = T.placement.full_shard(0, 1)
    layout = make_row_major(shape)

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
    assert "!suvm.tile_view<16384xbf16>" in mlir
    assert mlir.count("suvm.copy_async") == 2
    assert "suvm.transform_layout_async" not in mlir


@target("Sunmmio")
def partial_alias_copy(length, stride, offset):
    shape = (8, 128)
    placement = T.placement.replicated()

    @T.prim_func
    def main(
        source: T.MeshTensor(shape, placement, T.bfloat16, layout=make_row_major(shape)),
        output: T.MeshTensor((length,), placement, T.bfloat16, layout=make_row_major((length,))),
    ):
        with (
            T.Kernel(T.mesh_ncores()),
            T.decl_buffer((length,), T.bfloat16, data=source.data, strides=(stride,), elem_offset=offset) as view,
        ):
            local = T.alloc_shared((length,), T.bfloat16, scope="shared.rsram")
            T.copy(view, local)
            T.copy(local, output)

    return main


@pytest.mark.parametrize("length,stride,offset", [(512, 1, 0), (1024, 2, 0), (1024, 1, 1)])
def test_partial_or_strided_alias_is_not_a_full_reshape(length, stride, offset, tmp_path):
    with pytest.raises(tvm.error.InternalError, match="requires compact row-major storage"):
        validate_sunmmio_codegen_with_npuir_opt(partial_alias_copy(length, stride, offset), tmp_path)
