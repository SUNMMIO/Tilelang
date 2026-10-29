import os

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.engine.lower import lower
from tilelang.layout import make_mx_row_major_layout, make_mxzz_layout, make_row_major, make_zz_layout

from testing.python.sunmmio.common.codegen_validation import (
    lower_sunmmio_kernel_to_device_tir,
    validate_sunmmio_codegen_with_npuir_opt,
    validate_suvm_mlir_with_npuir_opt,
)
from testing.python.sunmmio.common.compile_pipeline import target


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")

DTYPE = "bfloat16"
LOOSE_OPT_ARGS = ("--verify-each",)
_SUMMA_COPY_SHAPES = (
    (256, 128, 128),
    (128, 128, 128),
    (256, 256, 128),
    (256, 256, 256),
)


@target("Sunmmio")
def zz_major_sub_block_copy_kernel():
    shape = (128, 128)
    placement = T.placement.replicated()
    global_layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, placement, DTYPE, layout=global_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, DTYPE, scope="shared.rsram")
            T.annotate_layout({A_shared: global_layout})
            T.copy(A[0:8, 0:32], A_shared[0:8, 0:32])

    return main


@target("Sunmmio")
def zz_non_major_sub_block_copy_kernel():
    shape = (128, 128)
    placement = T.placement.replicated()
    global_layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, placement, DTYPE, layout=global_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, DTYPE, scope="shared.rsram")
            T.annotate_layout({A_shared: global_layout})
            T.copy(A[0:32, 0:8], A_shared[0:32, 0:8])

    return main


@target("Sunmmio")
def zz_both_dims_sub_block_copy_kernel():
    shape = (128, 128)
    placement = T.placement.replicated()
    global_layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, placement, DTYPE, layout=global_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, DTYPE, scope="shared.rsram")
            T.annotate_layout({A_shared: global_layout})
            T.copy(A[0:8, 0:8], A_shared[0:8, 0:8])

    return main


@target("Sunmmio")
def zz_major_sub_block_multi_block_copy_kernel():
    shape = (128, 128)
    placement = T.placement.replicated()
    global_layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, placement, DTYPE, layout=global_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, DTYPE, scope="shared.rsram")
            T.annotate_layout({A_shared: global_layout})
            T.copy(A[0:8, 0:64], A_shared[0:8, 0:64])

    return main


@target("Sunmmio")
def zz_fully_coalesced_full_block_copy_kernel():
    shape = (64, 32)
    region_shape = (32, 32)
    placement = T.placement.replicated()
    global_layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))
    shared_layout = make_row_major(region_shape)

    @T.prim_func
    def main(
        B: T.MeshTensor(shape, placement, DTYPE, layout=global_layout),  # type: ignore
    ):
        with T.Kernel():
            B_shared = T.alloc_shared(region_shape, DTYPE, scope="shared.rsram")
            T.annotate_layout({B_shared: shared_layout})
            for bk in T.serial(2):
                T.copy(B[bk * 32, 0], B_shared)

    return main


@target("Sunmmio")
def zz_fully_coalesced_non_major_sub_block_copy_kernel():
    shape = (64, 32)
    placement = T.placement.replicated()
    global_layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))

    @T.prim_func
    def main(
        B: T.MeshTensor(shape, placement, DTYPE, layout=global_layout),  # type: ignore
    ):
        with T.Kernel():
            B_shared = T.alloc_shared(shape, DTYPE, scope="shared.rsram")
            T.annotate_layout({B_shared: global_layout})
            T.copy(B[0:32, 0:8], B_shared[0:32, 0:8])

    return main


@target("Sunmmio")
def let_bound_mesh_local_shape_copy_kernel():
    global_shape = (256, 256)
    shard_policy = T.placement.full_shard(0, 1)
    tensor_layout = make_zz_layout(global_shape, axes=[0, 1], block_shape=(32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor(global_shape, shard_policy, DTYPE, layout=tensor_layout),  # type: ignore
    ):
        with T.Kernel():
            local_m, local_n = A.local_shape
            A_shared = T.alloc_shared((local_m, local_n), DTYPE)
            T.copy(A, A_shared)

    return main


@target("Sunmmio")
def singleton_dimension_broadcast_kernel():
    src_shape = (1, 32, 32)
    dst_shape = (1, 32, 32)
    src_layout = make_zz_layout(src_shape, axes=[1, 2], block_shape=(32, 32))
    dst_layout = make_zz_layout(dst_shape, axes=[1, 2], block_shape=(32, 32))

    @T.prim_func
    def main():
        with T.Kernel():
            src_shared = T.alloc_shared(src_shape, DTYPE, scope="shared.rsram")
            dst_shared = T.alloc_shared(dst_shape, DTYPE, scope="shared.rsram")
            T.annotate_layout(
                {
                    src_shared: src_layout,
                    dst_shared: dst_layout,
                }
            )
            T.comm.broadcast(src_shared, dst_shared, (0, 0), direction="h")

    return main


@target("Sunmmio")
def summa_output_copy_kernel(
    M=128,
    N=128,
    K=128,
    block_M=32,
    block_N=32,
    dtype="float16",
    accum_dtype="float32",
):
    shard_policy = T.placement.full_shard(0, 1)
    A_shape = (M, K)
    B_shape = (K, N)
    C_shape = (M, N)
    A_layout = make_zz_layout(A_shape, axes=[0, 1], block_shape=(32, 32))
    B_layout = make_zz_layout(B_shape, axes=[0, 1], block_shape=(32, 32))
    C_layout = make_zz_layout(C_shape, axes=[0, 1], block_shape=(32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor(A_shape, shard_policy, dtype, layout=A_layout),  # type: ignore
        B: T.MeshTensor(B_shape, shard_policy, dtype, layout=B_layout),  # type: ignore
        C: T.MeshTensor(C_shape, shard_policy, accum_dtype, layout=C_layout),  # type: ignore
    ):
        with T.Kernel():
            sharded_M, _ = A.local_shape
            _, sharded_N = B.local_shape
            C_local = T.alloc_shared((block_M, block_N), accum_dtype)

            for bx in T.serial(T.ceildiv(sharded_M, block_M)):
                for by in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.copy(C_local, C[bx * block_M, by * block_N])

    return main


def _validate_copy_codegen(kernel, tmp_path, mlir_filename, tile_shape, element_type="bf16"):
    src = validate_sunmmio_codegen_with_npuir_opt(
        kernel,
        tmp_path,
        mlir_filename=mlir_filename,
        expected_tokens=("suvm.copy_async", "suvm.get_partitioned_tile_view"),
        opt_args=LOOSE_OPT_ARGS,
    )
    assert f"!suvm.tile_view<{tile_shape}x{element_type}>" in src
    return src


def test_zz_major_sub_block_copy_passes(tmp_path):
    _validate_copy_codegen(
        zz_major_sub_block_copy_kernel(),
        tmp_path,
        "zz_major_sub_block_copy.mlir",
        "8x32",
    )


def test_zz_non_major_sub_block_copy_lowers_in_compact_mode():
    lowered = lower_sunmmio_kernel_to_device_tir(zz_non_major_sub_block_copy_kernel())

    assert lowered.get_global_vars()


def test_zz_both_dims_sub_block_copy_lowers_in_compact_mode():
    lowered = lower_sunmmio_kernel_to_device_tir(zz_both_dims_sub_block_copy_kernel())

    assert lowered.get_global_vars()


def test_zz_major_sub_block_multi_block_copy_passes(tmp_path):
    _validate_copy_codegen(
        zz_major_sub_block_multi_block_copy_kernel(),
        tmp_path,
        "zz_major_sub_block_multi_block_copy.mlir",
        "8x64",
    )


def test_zz_fully_coalesced_full_block_copy_passes(tmp_path):
    _validate_copy_codegen(
        zz_fully_coalesced_full_block_copy_kernel(),
        tmp_path,
        "zz_fully_coalesced_full_block_copy.mlir",
        "32x32",
    )


def test_zz_fully_coalesced_non_major_sub_block_copy_lowers_in_compact_mode():
    lowered = lower_sunmmio_kernel_to_device_tir(zz_fully_coalesced_non_major_sub_block_copy_kernel())

    assert lowered.get_global_vars()


def test_let_bound_mesh_local_shape_copy_codegen_passes(tmp_path):
    src = _validate_copy_codegen(
        let_bound_mesh_local_shape_copy_kernel(),
        tmp_path,
        "let_bound_mesh_local_shape_copy.mlir",
        "64x64",
    )

    assert "sunmmio.fake" not in src
    assert src.count("suvm.copy_async") == 1
    assert src.count("suvm.get_partitioned_tile_view") == 2


def test_singleton_dimension_broadcast_codegen_passes(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        singleton_dimension_broadcast_kernel(),
        tmp_path,
        mlir_filename="singleton_dimension_broadcast.mlir",
        expected_tokens=(
            "suvm.get_partitioned_tile_view",
            "suvm.mcast_async",
            "suvm.sync",
        ),
        opt_args=LOOSE_OPT_ARGS,
    )
    assert "!suvm.tile_view<32x32xbf16>" in src
    assert "tiled_dims = [1, 2]" in src
    assert src.count("suvm.mcast_async") >= 1
    assert "sunmmio.fake" not in src


@pytest.mark.parametrize("M,N,K", _SUMMA_COPY_SHAPES)
def test_summa_output_copy_codegen_passes(tmp_path, M, N, K):
    src = _validate_copy_codegen(
        summa_output_copy_kernel(M=M, N=N, K=K),
        tmp_path,
        f"summa_output_copy_m{M}_n{N}_k{K}.mlir",
        "32x32",
        element_type="f32",
    )
    assert "sunmmio.fake" not in src
    assert src.count("suvm.copy_async") >= 1


@target("sunmmio")
def mx_copy_slot_kernel(
    dtype=T.mxfp8,
    route="rr",
    src_rank=2,
    dst_rank=3,
    extent=(32, 128),
    src_matrix=(32, 128),
    dst_matrix=(32, 128),
    src_min=(0, 0),
    dst_min=(0, 0),
    dynamic_slot=False,
    dst_extent=None,
    row_major=False,
    dynamic_row_stride=None,
):
    src_shape = (2,) * (src_rank - 2) + src_matrix
    dst_shape = (2,) * (dst_rank - 2) + dst_matrix
    dst_extent = extent if dst_extent is None else dst_extent

    def layout(shape):
        if dtype == T.bfloat16:
            return make_zz_layout(shape, axes=[len(shape) - 2, len(shape) - 1], block_shape=(32, 32))
        if row_major:
            return make_mx_row_major_layout(shape, dtype=dtype)
        return make_mxzz_layout(shape, dtype=dtype)

    def region(buffer, shape, start, size, slot):
        buffer = getattr(buffer, "buffer", buffer)
        mins = [slot] * (len(shape) - 2) + list(start)
        extents = [1] * (len(shape) - 2) + list(size)
        return tvm.tir.BufferRegion(buffer, [tvm.ir.Range.from_min_extent(m, e) for m, e in zip(mins, extents)])

    def copy_region(a, b, slot):
        start = src_min if dynamic_row_stride is None else (slot * dynamic_row_stride, 0)
        return T.copy(region(a, src_shape, start, extent, slot), region(b, dst_shape, dst_min, dst_extent, slot))

    @T.prim_func
    def main(
        G: T.MeshTensor(src_shape, T.placement.replicated(), dtype, layout=layout(src_shape)),
        H: T.MeshTensor(dst_shape, T.placement.replicated(), dtype, layout=layout(dst_shape)),
        slot: T.int32,
    ):
        with T.Kernel():
            A = T.alloc_shared(src_shape, dtype, scope="shared.rsram")
            B = T.alloc_shared(dst_shape, dtype, scope="shared.rsram")
            T.annotate_layout({A: layout(src_shape), B: layout(dst_shape)})
            if route == "gr":
                copy_region(G, B, slot if dynamic_slot else 1)
            elif route == "rg":
                copy_region(A, H, slot if dynamic_slot else 1)
            else:
                copy_region(A, B, slot if dynamic_slot else 1)

    return main


def _production_copy_artifact(kernel, pass_configs=None):
    with tvm.transform.PassContext(opt_level=3, config=pass_configs or {}):
        return lower(kernel, target="sunmmio")


def _calls_named(mod, name):
    calls = []

    def visit(node):
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == name:
            calls.append(node)

    for func in mod.functions.values():
        tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return calls


@pytest.mark.parametrize("dtype", [T.bfloat16, T.mxfp8, T.mxfp4])
@pytest.mark.parametrize("route", ["rr", "gr", "rg"])
@pytest.mark.parametrize("src_rank,dst_rank", [(2, 2), (2, 3), (3, 2), (3, 3), (4, 2), (2, 4)])
def test_copy_matrix_slots_production_pipeline(tmp_path, dtype, route, src_rank, dst_rank):
    # Use a nondegenerate block grid for the ordinary ZZ control: generic
    # MeshTensor derivation currently changes a size-one mode's stride in the
    # 3D single-block-height case. MX tests retain the original [32, 128] case.
    matrix = (64, 128) if dtype == T.bfloat16 else (32, 128)
    artifact = _production_copy_artifact(
        mx_copy_slot_kernel(
            dtype,
            route,
            src_rank,
            dst_rank,
            src_matrix=matrix,
            dst_matrix=matrix,
        )
    )
    source = artifact.kernel_source
    if dtype != T.bfloat16 or route != "rr" or src_rank != dst_rank:
        copies = _calls_named(artifact.device_mod, "tl.dma_copy")
        assert len(copies) == 1
        transforms = _calls_named(artifact.device_mod, "tl.sunmmio_layout_transform")
        original_endpoints = []
        for transfer in copies + transforms:
            for region_arg in transfer.args[:2]:
                load = region_arg.args[0]
                rank = len(load.indices)
                assert [int(x) for x in region_arg.args[2:]] == [1] * (rank - 2) + [32, 128]
                is_stage = load.buffer.name.endswith("_layout_stage")
                assert [int(x) for x in load.indices] == [0 if is_stage else 1] * (rank - 2) + [0, 0]
                if not is_stage:
                    original_endpoints.append(rank)
        assert sorted(original_endpoints) == sorted([src_rank, dst_rank])
        if dtype != T.bfloat16:
            assert not transforms
            assert "_layout_stage" not in artifact.device_mod.script()
        assert "suvm.tile.load" not in source
        expected_unit = "odma1" if route == "rr" else "odma0"
        assert f"unit = {expected_unit}" in source
        assert f"suvm.sync {expected_unit}" in source
    validate_suvm_mlir_with_npuir_opt(source, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))


@pytest.mark.parametrize("dtype", [T.bfloat16, T.mxfp8, T.mxfp4])
@pytest.mark.parametrize("reverse", [False, True])
def test_copy_matrix_slots_with_different_parent_strides(tmp_path, dtype, reverse):
    # Two block rows: the larger parent has a different block-row stride.
    ranks = (3, 2) if reverse else (2, 3)
    matrices = ((128, 256), (64, 128)) if reverse else ((64, 128), (128, 256))
    mins = ((64, 128), (0, 0)) if reverse else ((0, 0), (64, 128))
    artifact = _production_copy_artifact(
        mx_copy_slot_kernel(
            dtype,
            "rr",
            *ranks,
            extent=(64, 128),
            src_matrix=matrices[0],
            dst_matrix=matrices[1],
            src_min=mins[0],
            dst_min=mins[1],
            dynamic_slot=True,
        )
    )
    copies = _calls_named(artifact.device_mod, "tl.dma_copy")
    assert len(copies) == 1
    for arg, start in zip(copies[0].args[:2], mins):
        assert [int(x) for x in arg.args[0].indices[-2:]] == list(start)
    assert "_layout_stage" not in artifact.device_mod.script()
    validate_suvm_mlir_with_npuir_opt(artifact.kernel_source, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))


@pytest.mark.parametrize("dtype", [T.mxfp8, T.mxfp4])
def test_mx_row_major_slot_copy_preserves_full_rows(tmp_path, dtype):
    artifact = _production_copy_artifact(mx_copy_slot_kernel(dtype, "rr", 3, 2, row_major=True))
    assert len(_calls_named(artifact.device_mod, "tl.dma_copy")) == 1
    validate_suvm_mlir_with_npuir_opt(artifact.kernel_source, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))


@pytest.mark.parametrize("dtype", [T.mxfp8, T.mxfp4])
def test_mx_copy_rejects_partial_blocks(dtype):
    with pytest.raises(tvm.error.InternalError, match="requires whole blocks"):
        _production_copy_artifact(mx_copy_slot_kernel(dtype, extent=(16, 128)))


@pytest.mark.parametrize("dtype", [T.bfloat16, T.mxfp8, T.mxfp4])
def test_dma_copy_rejects_same_element_count_with_different_matrix_shape(dtype):
    with pytest.raises(tvm.error.InternalError, match="matching effective region shapes"):
        _production_copy_artifact(mx_copy_slot_kernel(dtype, dst_matrix=(64, 128), dst_extent=(64, 64)))


@pytest.mark.parametrize("dtype", [T.bfloat16, T.mxfp8, T.mxfp4])
@pytest.mark.parametrize("extent,offset", [((32, 128), 16), ((64, 128), 32)])
def test_dma_copy_rejects_truncated_partition_start_in_compact_mode(dtype, extent, offset):
    import tilelang.utils.target as target_utils

    previous = target_utils.ENABLE_SUNMMIO_REGION_VALIDATION
    target_utils.set_sunmmio_region_validation(False)
    try:
        with pytest.raises(tvm.error.InternalError, match="must be provably aligned to region extent"):
            _production_copy_artifact(
                mx_copy_slot_kernel(
                    dtype,
                    extent=extent,
                    src_matrix=extent,
                    dst_matrix=(128, 128),
                    dst_min=(offset, 0),
                )
            )
    finally:
        target_utils.set_sunmmio_region_validation(previous)


@pytest.mark.parametrize("stride", [1, 32])
def test_mx_copy_dynamic_start_requires_exact_partition(tmp_path, stride):
    kernel = mx_copy_slot_kernel(
        T.mxfp8,
        "gr",
        2,
        2,
        src_matrix=(128, 128),
        dynamic_slot=True,
        dynamic_row_stride=stride,
    )
    if stride == 1:
        with pytest.raises(tvm.error.InternalError, match="must be provably aligned"):
            _production_copy_artifact(kernel)
    else:
        artifact = _production_copy_artifact(kernel)
        validate_suvm_mlir_with_npuir_opt(artifact.kernel_source, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))


@target("sunmmio")
def mx_copy_nontrailing_axes_kernel():
    @T.prim_func
    def main():
        with T.Kernel():
            A = T.alloc_shared((32, 2, 128), T.mxfp8, scope="shared.rsram")
            B = T.alloc_shared((32, 128), T.mxfp8, scope="shared.rsram")
            T.annotate_layout(
                {
                    A: make_mxzz_layout((32, 2, 128), axes=[0, 2], dtype=T.mxfp8),
                    B: make_mxzz_layout((32, 128), dtype=T.mxfp8),
                }
            )
            T.copy(A[:, 1, :], B)

    return main


def test_mx_copy_nontrailing_matrix_axes_are_rejected():
    with pytest.raises(tvm.error.InternalError, match="requires trailing matrix axes"):
        _production_copy_artifact(mx_copy_nontrailing_axes_kernel())


@target("sunmmio")
def mx_copy_pipeline_kernel(dtype):
    shape = (4, 32, 128)

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, T.placement.replicated(), dtype, layout=make_mxzz_layout(shape, dtype=dtype)),
        B: T.MeshTensor(shape, T.placement.replicated(), dtype, layout=make_mxzz_layout(shape, dtype=dtype)),
    ):
        with T.Kernel():
            X = T.alloc_shared((32, 128), dtype, scope="shared.rsram")
            Y = T.alloc_shared((32, 128), dtype, scope="shared.rsram")
            T.annotate_layout({X: make_mxzz_layout((32, 128), dtype=dtype), Y: make_mxzz_layout((32, 128), dtype=dtype)})
            for k in T.Pipelined(4, num_stages=2):
                T.copy(A[k, :, :], X)
                T.copy(X, Y)
                T.copy(Y, B[k, :, :])

    return main


@target("sunmmio")
def mx_copy_chain_kernel(dtype):
    shape = (32, 128)

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, T.placement.replicated(), dtype, layout=make_mxzz_layout(shape, dtype=dtype)),
        B: T.MeshTensor(shape, T.placement.replicated(), dtype, layout=make_mxzz_layout(shape, dtype=dtype)),
    ):
        with T.Kernel():
            X = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Y = T.alloc_shared(shape, dtype, scope="shared.rsram")
            T.annotate_layout({X: make_mxzz_layout(shape, dtype=dtype), Y: make_mxzz_layout(shape, dtype=dtype)})
            T.copy(A, X)
            T.copy(X, Y)
            T.copy(Y, B)

    return main


@pytest.mark.parametrize("dtype", [T.mxfp8, T.mxfp4])
def test_mx_copy_waits_before_other_unit_consumes_result(tmp_path, dtype):
    artifact = _production_copy_artifact(mx_copy_chain_kernel(dtype))
    lines = artifact.kernel_source.splitlines()
    transfers = [i for i, line in enumerate(lines) if "suvm.copy_async" in line]
    assert len(transfers) == 3
    assert "unit = odma0" in lines[transfers[0]]
    assert "unit = odma1" in lines[transfers[1]]
    assert "unit = odma0" in lines[transfers[2]]
    for begin, end, unit in [(transfers[0], transfers[1], "odma0"), (transfers[1], transfers[2], "odma1")]:
        assert any("suvm.sync" in line and unit in line for line in lines[begin + 1 : end])
    validate_suvm_mlir_with_npuir_opt(artifact.kernel_source, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))


@pytest.mark.parametrize("dtype", [T.mxfp8, T.mxfp4])
@pytest.mark.parametrize("mode", ["greedy", "ilp"])
def test_mx_copy_pipeline_keeps_dma_dependencies(tmp_path, dtype, mode):
    artifact = _production_copy_artifact(mx_copy_pipeline_kernel(dtype), {"tl.sunmmio_pipeline_mode": mode})
    source = artifact.kernel_source
    assert "suvm.tile.load" not in source
    assert "suvm.copy_async" in source
    assert "suvm.sync" in source
    validate_suvm_mlir_with_npuir_opt(source, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))


if __name__ == "__main__":
    tilelang.testing.main()
