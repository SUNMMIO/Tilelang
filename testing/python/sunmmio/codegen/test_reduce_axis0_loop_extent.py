"""Non-tiled reduction loops retain their source dimension after loop reordering."""

import re

import pytest

import tilelang.language as T
from testing.python.sunmmio.common.codegen_validation import (
    codegen_sunmmio_suvm_mlir,
    lower_sunmmio_kernel_to_device_tir,
    simplify_suvm_mlir,
)
from testing.python.sunmmio.common.compile_pipeline import target
from tilelang import tvm
from tilelang.layout import make_row_major


@target("Sunmmio")
def axis0_reduce_kernel(shape, operation):
    out_shape = (shape[1],)

    @T.prim_func
    def main(
        Output: T.MeshTensor(out_shape, T.placement.replicated(), T.float32, layout=make_row_major(out_shape)),
    ):
        with T.Kernel():
            local_values = T.alloc_shared(out_shape, T.float32)
            values = T.alloc_shared(shape, T.float32)
            result = T.alloc_shared(out_shape, T.float32)
            T.fill(local_values, 1)
            T.comm.all_gather(local_values, values, direction="h")
            if operation == "max":
                T.reduce_max(values, result, dim=0)
            else:
                T.reduce_sum(values, result, dim=0)
            T.copy(result, Output)

    return main


@pytest.mark.parametrize("shape", [(4, 256), (4, 512)])
@pytest.mark.parametrize("operation", ["max", "sum"])
def test_axis0_reduction_preserves_loop_extent_and_view_bounds(shape, operation, tmp_path):
    device = lower_sunmmio_kernel_to_device_tir(axis0_reduce_kernel(shape, operation))
    (tmp_path / "device.tir").write_text(device.script())
    source_extents = []

    def inspect_loop(node):
        if not isinstance(node, tvm.tir.For):
            return

        def inspect_load(load):
            if not isinstance(load, tvm.tir.BufferLoad):
                return
            if tuple(int(x) for x in load.buffer.shape) == shape and load.indices[0].same_as(node.loop_var):
                source_extents.append(int(node.extent))

        tvm.tir.stmt_functor.post_order_visit(node.body, inspect_load)

    for function in device.functions.values():
        tvm.tir.stmt_functor.post_order_visit(function.body, inspect_loop)
    assert source_extents and set(source_extents) == {shape[0]}

    source = codegen_sunmmio_suvm_mlir(device)
    (tmp_path / "kernel.mlir").write_text(source)
    simplified = simplify_suvm_mlir(source)
    (tmp_path / "canonical.mlir").write_text(simplified)
    constants = dict((name, int(value)) for name, value in re.findall(r"(%\w+) = arith.constant (-?\d+) : index", simplified))
    loops = dict(
        (variable, (lower, upper, step))
        for variable, lower, upper, step in re.findall(r"scf.for (%\w+) = (%\w+) to (%\w+) step (%\w+)", simplified)
    )
    views = re.findall(
        rf"(%\w+) = suvm.get_partitioned_tile_view .*?indices = \[(%\w+), (%\w+)\] tiled_dims = \[1\]"
        rf" : !suvm.memtensor<{shape[0]}x{shape[1]}xf32,.*? -> !suvm.tile_view<(\d+)xf32>",
        simplified,
    )
    views = [view for view in views if f"suvm.tile.load {view[0]} :" in simplified]
    assert views, simplified
    for _, row_index, column_tile_index, tile_width in views:
        assert tuple(constants[v] for v in loops[row_index]) == (0, shape[0], 1), simplified
        if column_tile_index in loops:
            lower, upper, step = (constants[v] for v in loops[column_tile_index])
            assert lower == 0 and step == 1
            assert upper * int(tile_width) <= shape[1], simplified
        else:
            assert constants[column_tile_index] == 0
            assert int(tile_width) <= shape[1], simplified
