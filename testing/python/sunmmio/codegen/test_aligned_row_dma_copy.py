import re

import pytest
import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.layout import make_aligned_row_major

from testing.python.sunmmio.common.codegen_validation import (
    lower_sunmmio_kernel_to_device_tir,
    codegen_sunmmio_suvm_mlir,
    validate_suvm_mlir_with_npuir_opt,
    validate_sunmmio_codegen_with_npuir_opt,
)
from testing.python.sunmmio.common.compile_pipeline import target


tilelang.env.disable_cache()


@target("Sunmmio")
def aligned_row_vector_copy_kernel(direction="load", rsram_rank=1, cols=64, dtype=T.bfloat16):
    dram_shape = (3, cols)
    rsram_shape = (cols,) if rsram_rank == 1 else (1, cols)
    dram_layout = make_aligned_row_major(dram_shape, dtype, align_bytes=1024)
    rsram_layout = make_aligned_row_major(rsram_shape, dtype, align_bytes=1024)

    if direction == "load":

        @T.prim_func
        def main(
            src: T.MeshTensor(dram_shape, T.placement.replicated(), dtype, layout=dram_layout),  # type: ignore
        ):
            with T.Kernel():
                dst = T.alloc_shared(rsram_shape, dtype, scope="shared.rsram")
                T.annotate_layout({dst: rsram_layout})
                T.copy(src[1:2, :], dst)

        return main

    @T.prim_func
    def main(
        dst: T.MeshTensor(dram_shape, T.placement.replicated(), dtype, layout=dram_layout),  # type: ignore
    ):
        with T.Kernel():
            src = T.alloc_shared(rsram_shape, dtype, scope="shared.rsram")
            T.annotate_layout({src: rsram_layout})
            T.copy(src, dst[1:2, :])

    return main


@target("Sunmmio")
def aligned_row_matrix_copy_kernel(
    direction="load",
    rows=500,
    cols=500,
    row_start=0,
    copy_rows=None,
    dtype=T.bfloat16,
):
    copy_rows = rows if copy_rows is None else copy_rows
    dram_shape = (5, rows, cols)
    rsram_shape = (copy_rows, cols)
    dram_layout = make_aligned_row_major(dram_shape, dtype, align_bytes=1024)
    rsram_layout = make_aligned_row_major(rsram_shape, dtype, align_bytes=1024)

    if direction == "load":

        @T.prim_func
        def main(
            src: T.MeshTensor(dram_shape, T.placement.replicated(), dtype, layout=dram_layout),  # type: ignore
        ):
            with T.Kernel():
                dst = T.alloc_shared(rsram_shape, dtype, scope="shared.rsram")
                T.annotate_layout({dst: rsram_layout})
                T.copy(src[2, row_start : row_start + copy_rows, :], dst)

        return main

    @T.prim_func
    def main(
        dst: T.MeshTensor(dram_shape, T.placement.replicated(), dtype, layout=dram_layout),  # type: ignore
    ):
        with T.Kernel():
            src = T.alloc_shared(rsram_shape, dtype, scope="shared.rsram")
            T.annotate_layout({src: rsram_layout})
            T.copy(src, dst[2, row_start : row_start + copy_rows, :])

    return main


@target("Sunmmio")
def aligned_row_non_singleton_reshape_kernel(direction="load", dtype=T.bfloat16):
    dram_shape, rsram_shape = (2, 32), (64,)
    dram_layout = make_aligned_row_major(dram_shape, dtype, align_bytes=1024)
    rsram_layout = make_aligned_row_major(rsram_shape, dtype, align_bytes=1024)

    if direction == "load":

        @T.prim_func
        def main(
            src: T.MeshTensor(dram_shape, T.placement.replicated(), dtype, layout=dram_layout),  # type: ignore
        ):
            with T.Kernel():
                dst = T.alloc_shared(rsram_shape, dtype, scope="shared.rsram")
                T.annotate_layout({dst: rsram_layout})
                T.copy(src[0:2, 0:32], dst[0:64])

        return main

    @T.prim_func
    def main(
        dst: T.MeshTensor(dram_shape, T.placement.replicated(), dtype, layout=dram_layout),  # type: ignore
    ):
        with T.Kernel():
            src = T.alloc_shared(rsram_shape, dtype, scope="shared.rsram")
            T.annotate_layout({src: rsram_layout})
            T.copy(src[0:64], dst[0:2, 0:32])

    return main


@target("Sunmmio")
def aligned_row_byte_aligned_reshape_kernel(direction="load", dtype=T.bfloat16):
    dram_shape, rsram_shape = (2, 512), (1024,)
    dram_layout = make_aligned_row_major(dram_shape, dtype, align_bytes=1024)
    rsram_layout = make_aligned_row_major(rsram_shape, dtype, align_bytes=1024)

    if direction == "load":

        @T.prim_func
        def main(
            src: T.MeshTensor(dram_shape, T.placement.replicated(), dtype, layout=dram_layout),  # type: ignore
        ):
            with T.Kernel():
                dst = T.alloc_shared(rsram_shape, dtype, scope="shared.rsram")
                T.annotate_layout({dst: rsram_layout})
                T.copy(src[0:2, 0:512], dst[0:1024])

        return main

    @T.prim_func
    def main(
        dst: T.MeshTensor(dram_shape, T.placement.replicated(), dtype, layout=dram_layout),  # type: ignore
    ):
        with T.Kernel():
            src = T.alloc_shared(rsram_shape, dtype, scope="shared.rsram")
            T.annotate_layout({src: rsram_layout})
            T.copy(src[0:1024], dst[0:2, 0:512])

    return main


@target("Sunmmio")
def aligned_row_incompatible_singleton_kernel(direction="load", dtype=T.bfloat16):
    dram_shape, rsram_shape = (64,), (64, 1)
    dram_layout = make_aligned_row_major(dram_shape, dtype, align_bytes=1024)
    rsram_layout = make_aligned_row_major(rsram_shape, dtype, align_bytes=1024)

    if direction == "load":

        @T.prim_func
        def main(
            src: T.MeshTensor(dram_shape, T.placement.replicated(), dtype, layout=dram_layout),  # type: ignore
        ):
            with T.Kernel():
                dst = T.alloc_shared(rsram_shape, dtype, scope="shared.rsram")
                T.annotate_layout({dst: rsram_layout})
                T.copy(src[0:64], dst[0:64, 0:1])

        return main

    @T.prim_func
    def main(
        dst: T.MeshTensor(dram_shape, T.placement.replicated(), dtype, layout=dram_layout),  # type: ignore
    ):
        with T.Kernel():
            src = T.alloc_shared(rsram_shape, dtype, scope="shared.rsram")
            T.annotate_layout({src: rsram_layout})
            T.copy(src[0:64, 0:1], dst[0:64])

    return main


@target("Sunmmio")
def aligned_row_middle_singleton_kernel(direction="load", dtype=T.bfloat16):
    dram_shape, rsram_shape = (500, 1, 500), (500, 500)
    dram_layout = make_aligned_row_major(dram_shape, dtype, align_bytes=1024)
    rsram_layout = make_aligned_row_major(rsram_shape, dtype, align_bytes=1024)

    if direction == "load":

        @T.prim_func
        def main(
            src: T.MeshTensor(dram_shape, T.placement.replicated(), dtype, layout=dram_layout),  # type: ignore
        ):
            with T.Kernel():
                dst = T.alloc_shared(rsram_shape, dtype, scope="shared.rsram")
                T.annotate_layout({dst: rsram_layout})
                T.copy(src[0:500, 0:1, 0:500], dst[0:500, 0:500])

        return main

    @T.prim_func
    def main(
        dst: T.MeshTensor(dram_shape, T.placement.replicated(), dtype, layout=dram_layout),  # type: ignore
    ):
        with T.Kernel():
            src = T.alloc_shared(rsram_shape, dtype, scope="shared.rsram")
            T.annotate_layout({src: rsram_layout})
            T.copy(src[0:500, 0:500], dst[0:500, 0:1, 0:500])

    return main


@target("Sunmmio")
def aligned_row_partial_row_kernel(direction="load", dtype=T.bfloat16):
    dram_shape, rsram_shape = (3, 64), (32,)
    dram_layout = make_aligned_row_major(dram_shape, dtype, align_bytes=1024)
    rsram_layout = make_aligned_row_major(rsram_shape, dtype, align_bytes=1024)

    if direction == "load":

        @T.prim_func
        def main(
            src: T.MeshTensor(dram_shape, T.placement.replicated(), dtype, layout=dram_layout),  # type: ignore
        ):
            with T.Kernel():
                dst = T.alloc_shared(rsram_shape, dtype, scope="shared.rsram")
                T.annotate_layout({dst: rsram_layout})
                T.copy(src[1, 0:32], dst)

        return main

    @T.prim_func
    def main(
        dst: T.MeshTensor(dram_shape, T.placement.replicated(), dtype, layout=dram_layout),  # type: ignore
    ):
        with T.Kernel():
            src = T.alloc_shared(rsram_shape, dtype, scope="shared.rsram")
            T.annotate_layout({src: rsram_layout})
            T.copy(src, dst[1, 0:32])

    return main


@target("Sunmmio")
def aligned_row_byte_aligned_partial_row_kernel(
    direction="load", dtype=T.bfloat16, cols=1024, start=512, length=512, rsram_start=0, rsram_cols=None
):
    dram_shape, rsram_shape = (3, cols), (rsram_cols or length,)
    dram_layout = make_aligned_row_major(dram_shape, dtype, align_bytes=1024)
    rsram_layout = make_aligned_row_major(rsram_shape, dtype, align_bytes=1024)

    if direction == "load":

        @T.prim_func
        def main(
            src: T.MeshTensor(dram_shape, T.placement.replicated(), dtype, layout=dram_layout),  # type: ignore
        ):
            with T.Kernel():
                dst = T.alloc_shared(rsram_shape, dtype, scope="shared.rsram")
                T.annotate_layout({dst: rsram_layout})
                T.copy(src[1, start : start + length], dst[rsram_start : rsram_start + length])

        return main

    @T.prim_func
    def main(
        dst: T.MeshTensor(dram_shape, T.placement.replicated(), dtype, layout=dram_layout),  # type: ignore
    ):
        with T.Kernel():
            src = T.alloc_shared(rsram_shape, dtype, scope="shared.rsram")
            T.annotate_layout({src: rsram_layout})
            T.copy(src[rsram_start : rsram_start + length], dst[1, start : start + length])

    return main


@target("Sunmmio")
def aligned_row_effective_rank3_kernel(direction="load", dtype=T.bfloat16):
    shape = (2, 3, 64)
    dram_layout = make_aligned_row_major(shape, dtype, align_bytes=1024)
    rsram_layout = make_aligned_row_major(shape, dtype, align_bytes=1024)

    if direction == "load":

        @T.prim_func
        def main(
            src: T.MeshTensor(shape, T.placement.replicated(), dtype, layout=dram_layout),  # type: ignore
        ):
            with T.Kernel():
                dst = T.alloc_shared(shape, dtype, scope="shared.rsram")
                T.annotate_layout({dst: rsram_layout})
                T.copy(src, dst)

        return main

    @T.prim_func
    def main(
        dst: T.MeshTensor(shape, T.placement.replicated(), dtype, layout=dram_layout),  # type: ignore
    ):
        with T.Kernel():
            src = T.alloc_shared(shape, dtype, scope="shared.rsram")
            T.annotate_layout({src: rsram_layout})
            T.copy(src, dst)

    return main


@target("Sunmmio")
def aligned_row_alignment_mismatch_kernel(direction="load", dtype=T.bfloat16):
    dram_shape, rsram_shape = (3, 64), (2, 64)
    dram_layout = make_aligned_row_major(dram_shape, dtype, align_bytes=1024)
    rsram_layout = make_aligned_row_major(rsram_shape, dtype, align_bytes=64)

    if direction == "load":

        @T.prim_func
        def main(
            src: T.MeshTensor(dram_shape, T.placement.replicated(), dtype, layout=dram_layout),  # type: ignore
        ):
            with T.Kernel():
                dst = T.alloc_shared(rsram_shape, dtype, scope="shared.rsram")
                T.annotate_layout({dst: rsram_layout})
                T.copy(src[0:2, :], dst)

        return main

    @T.prim_func
    def main(
        dst: T.MeshTensor(dram_shape, T.placement.replicated(), dtype, layout=dram_layout),  # type: ignore
    ):
        with T.Kernel():
            src = T.alloc_shared(rsram_shape, dtype, scope="shared.rsram")
            T.annotate_layout({src: rsram_layout})
            T.copy(src, dst[0:2, :])

    return main


@pytest.mark.parametrize("direction", ["load", "store"])
@pytest.mark.parametrize("rsram_rank", [1, 2])
def test_aligned_row_vector_copy_uses_1024_byte_carrier(direction, rsram_rank, tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        aligned_row_vector_copy_kernel(direction=direction, rsram_rank=rsram_rank),
        tmp_path,
        mlir_filename=f"aligned_row_vector_{direction}_rank{rsram_rank}.mlir",
        expected_tokens=("suvm.copy_async", "!suvm.tile_view<512xbf16>"),
        opt_args=("--verify-each", "--suvm-to-llvm-pipeline"),
    )
    assert src.count("suvm.copy_async") == 1
    assert "suvm.transform_layout_async" not in src


@pytest.mark.parametrize(
    "direction,rows,cols,row_start,copy_rows,dtype,expected_view",
    [
        pytest.param(
            direction,
            500,
            500,
            0,
            None,
            T.bfloat16,
            "!suvm.tile_view<500x512xbf16>",
            id=f"full-bf16-{direction}",
        )
        for direction in ("load", "store")
    ]
    + [
        pytest.param(
            direction,
            20,
            500,
            7,
            5,
            T.bfloat16,
            "!suvm.tile_view<512xbf16>",
            id=f"row-subset-bf16-{direction}",
        )
        for direction in ("load", "store")
    ]
    + [
        pytest.param(
            direction,
            3,
            250,
            0,
            None,
            T.float32,
            "!suvm.tile_view<3x256xf32>",
            id=f"full-fp32-{direction}",
        )
        for direction in ("load", "store")
    ]
    + [
        pytest.param(
            direction,
            3,
            512,
            0,
            None,
            T.bfloat16,
            "!suvm.tile_view<3x512xbf16>",
            id=f"already-aligned-{direction}",
        )
        for direction in ("load", "store")
    ],
)
def test_aligned_row_matrix_copy_uses_rank2_dma(
    direction,
    rows,
    cols,
    row_start,
    copy_rows,
    dtype,
    expected_view,
    tmp_path,
):
    src = validate_sunmmio_codegen_with_npuir_opt(
        aligned_row_matrix_copy_kernel(
            direction=direction,
            rows=rows,
            cols=cols,
            row_start=row_start,
            copy_rows=copy_rows,
            dtype=dtype,
        ),
        tmp_path,
        mlir_filename=f"aligned_row_matrix_{direction}_{rows}_{cols}_{row_start}_{copy_rows}.mlir",
        expected_tokens=("suvm.copy_async", expected_view),
        opt_args=("--verify-each", "--suvm-to-llvm-pipeline"),
    )
    assert src.count("suvm.copy_async") == 1
    assert "suvm.transform_layout_async" not in src


@pytest.mark.parametrize("direction", ["load", "store"])
def test_alignment_mismatch_falls_back_to_existing_layout_transform_path(direction):
    mod = lower_sunmmio_kernel_to_device_tir(aligned_row_alignment_mismatch_kernel(direction=direction))
    src = mod.script()
    assert src.count("T.dma_copy") == 1
    assert src.count("T.sunmmio_layout_transform") == 1


@pytest.mark.parametrize("direction", ["load", "store"])
def test_aligned_row_effective_rank3_uses_direct_dma(direction, tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        aligned_row_effective_rank3_kernel(direction=direction),
        tmp_path,
        mlir_filename=f"aligned_row_rank3_{direction}.mlir",
        expected_tokens=("suvm.copy_async", "!suvm.tile_view<2x3x512xbf16>"),
        opt_args=("--verify-each", "--suvm-to-llvm-pipeline"),
    )
    assert src.count("suvm.copy_async") == 1
    assert "suvm.transform_layout_async" not in src


@pytest.mark.parametrize(
    "factory,reason",
    [
        pytest.param(
            aligned_row_non_singleton_reshape_kernel,
            "canonical logical shapes do not match",
            id="non-singleton-reshape",
        ),
        pytest.param(
            aligned_row_byte_aligned_reshape_kernel,
            "canonical logical shapes do not match",
            id="byte-aligned-reshape",
        ),
        pytest.param(
            aligned_row_incompatible_singleton_kernel,
            "canonical logical shapes do not match",
            id="incompatible-singleton",
        ),
        pytest.param(
            aligned_row_middle_singleton_kernel,
            "canonical logical shapes do not match",
            id="middle-singleton",
        ),
        pytest.param(
            aligned_row_partial_row_kernel,
            "partial-row byte size is not a multiple of 1024",
            id="partial-row",
        ),
    ],
)
@pytest.mark.parametrize("direction", ["load", "store"])
def test_aligned_row_carrier_rejection_is_actionable(factory, reason, direction, tmp_path):
    with pytest.raises(tvm.error.InternalError, match=reason):
        validate_sunmmio_codegen_with_npuir_opt(
            factory(direction=direction),
            tmp_path,
            mlir_filename=f"aligned_row_reject_{factory.__name__}_{direction}.mlir",
            opt_args=("--verify-each", "--suvm-to-llvm-pipeline"),
        )


@pytest.mark.parametrize("direction", ["load", "store"])
def test_aligned_row_byte_aligned_partial_row_uses_exact_segment(direction, tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        aligned_row_byte_aligned_partial_row_kernel(direction=direction),
        tmp_path,
        mlir_filename=f"aligned_row_partial_segment_{direction}.mlir",
        expected_tokens=("suvm.copy_async", "!suvm.tile_view<512xbf16>"),
        opt_args=("--verify-each", "--suvm-to-llvm-pipeline"),
    )
    assert src.count("suvm.copy_async") == 1
    assert "suvm.transform_layout_async" not in src


def _lower_dma_to_llvm_with_constant_addresses(kernel, tmp_path):
    mod = lower_sunmmio_kernel_to_device_tir(kernel)
    # Unroll small constant loops only in this test to expose every byte GEP.
    with tvm.transform.PassContext(config={"tir.UnrollLoop": {"auto_max_step": 128}}):
        mod = tvm.tir.transform.UnrollLoop()(mod)
        mod = tvm.tir.transform.Simplify()(mod)
    src = codegen_sunmmio_suvm_mlir(mod)
    result = validate_suvm_mlir_with_npuir_opt(src, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline", "--canonicalize"))
    return result.stdout


@pytest.mark.parametrize("direction", ["load", "store"])
@pytest.mark.parametrize(
    "dtype,cols,start,length,expected_offsets",
    [
        (T.bfloat16, 2048, 512, 1024, [5120, 6144]),
        (T.bfloat16, 1536, 0, 1024, [3072, 4096]),
        (T.float32, 1024, 256, 512, [5120, 6144]),
        (T.float32, 768, 0, 512, [3072, 4096]),
        (T.bfloat16, 1024, 512, 512, [3072]),
        (T.bfloat16, 1600, 512, 1024, [5120, 6144]),
    ],
)
def test_aligned_row_segment_preserves_byte_addresses(direction, dtype, cols, start, length, expected_offsets, tmp_path):
    llvm = _lower_dma_to_llvm_with_constant_addresses(
        aligned_row_byte_aligned_partial_row_kernel(direction, dtype, cols, start, length), tmp_path
    )
    offsets = [int(x) for x in re.findall(r"llvm.getelementptr %arg0\[(\d+)\].*, i8", llvm)]
    assert offsets == expected_offsets, llvm


@pytest.mark.parametrize("direction", ["load", "store"])
def test_aligned_row_subset_preserves_outer_byte_addresses(direction, tmp_path):
    llvm = _lower_dma_to_llvm_with_constant_addresses(
        aligned_row_matrix_copy_kernel(direction, rows=20, cols=500, row_start=7, copy_rows=5), tmp_path
    )
    offsets = [int(x) for x in re.findall(r"llvm.getelementptr %arg0\[(\d+)\].*, i8", llvm)]
    # Batch 2, rows 7..11, each row physically padded to 1024 bytes.
    assert offsets == [48128, 49152, 50176, 51200, 52224], llvm


@pytest.mark.parametrize("direction", ["load", "store"])
def test_aligned_row_segment_preserves_rsram_byte_addresses(direction, tmp_path):
    llvm = _lower_dma_to_llvm_with_constant_addresses(
        aligned_row_byte_aligned_partial_row_kernel(direction, T.bfloat16, 2048, 0, 1024, rsram_start=512, rsram_cols=2048), tmp_path
    )
    rsram = re.search(r"(%\w+) = llvm.mlir.addressof @sunmmio_rsram_buf", llvm).group(1)
    offsets = [int(x) for x in re.findall(rf"llvm.getelementptr {rsram}\[(\d+)\].*, i8", llvm)]
    assert offsets == [1024, 2048], llvm


if __name__ == "__main__":
    tilelang.testing.main()
