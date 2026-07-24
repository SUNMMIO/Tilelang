import os

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import make_aligned_row_major, make_zz_layout

from testing.python.sunmmio.common.codegen_validation import validate_sunmmio_codegen_with_npuir_opt
from testing.python.sunmmio.common.compile_pipeline import target


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")

STRICT_OPT_ARGS = ("--verify-each", "--suvm-to-llvm-pipeline")


def _matrix_output_spec(h, w, dtype):
    shape = (16, h, w)
    return shape, T.MeshShardingPolicy(cross_mesh_dim=0), make_zz_layout(shape, [1, 2], (32, 32))


@target("Sunmmio")
def serialized_rank1_zz_slices_kernel(h=4, dtype=T.float32):
    output_rows = 8
    padded_width = 32
    out_shape, token_policy, out_layout = _matrix_output_spec(output_rows, padded_width, dtype)
    cm_layout = make_zz_layout((h, h), [0, 1], (32, 32))
    comb_layout = make_zz_layout((output_rows, padded_width), [0, 1], (32, 32))

    @T.prim_func
    def main(
        out: T.MeshTensor(out_shape, token_policy, dtype, layout=out_layout),  # type: ignore
    ):
        with T.Kernel():
            cm = T.alloc_shared((h, h), dtype)
            comb = T.alloc_shared((output_rows, padded_width), dtype)
            T.annotate_layout({cm: cm_layout, comb: comb_layout})

            T.fill(comb, 0)
            for i in T.serial(h):
                for j in T.serial(h):
                    for k in T.Tiles([h], parallel=True):
                        comb[i, j * h + k] = cm[j, k]

            T.copy(comb, out[0, 0:output_rows, 0:padded_width])

    return main


@target("Sunmmio")
def temp_stage_subaligned_then_direct_kernel(h=4, w=32, dtype=T.float32):
    output_rows = 8
    out_shape, token_policy, out_layout = _matrix_output_spec(output_rows, w, dtype)
    matrix_layout = make_zz_layout((output_rows, w), [0, 1], (32, 32))
    vector_layout = make_aligned_row_major((w,), dtype, align_bytes=64)

    @T.prim_func
    def main(
        out: T.MeshTensor(out_shape, token_policy, dtype, layout=out_layout),  # type: ignore
    ):
        with T.Kernel():
            cm = T.alloc_shared((output_rows, w), dtype)
            m_shared = T.alloc_shared((w,), dtype)
            temp = T.alloc_shared((w,), dtype)
            T.annotate_layout({cm: matrix_layout, m_shared: vector_layout, temp: vector_layout})

            T.fill(cm, 0)
            T.fill(m_shared, 1)
            for i in T.serial(h):
                T.fill(temp, 0)
                for j in T.Tiles([h], parallel=True):
                    temp[j] = m_shared[i * h + j]
                for j in T.Tiles([w], parallel=True):
                    cm[i, j] = temp[j]

            T.copy(cm, out[0, 0:output_rows, 0:w])

    return main


@target("Sunmmio")
def packed_1d_to_2d_rank1_fallback_kernel(h=4, base=8, vector_size=128, matrix_size=32, dtype=T.float32):
    assert base + h * h <= vector_size
    out_shape, token_policy, out_layout = _matrix_output_spec(matrix_size, matrix_size, dtype)
    vector_layout = make_aligned_row_major((vector_size,), dtype, align_bytes=64)
    matrix_layout = make_zz_layout((matrix_size, matrix_size), [0, 1], (32, 32))

    @T.prim_func
    def main(
        out: T.MeshTensor(out_shape, token_policy, dtype, layout=out_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared((vector_size,), dtype)
            B_shared = T.alloc_shared((matrix_size, matrix_size), dtype)
            T.annotate_layout({A_shared: vector_layout, B_shared: matrix_layout})

            T.fill(A_shared, 1)
            T.fill(B_shared, 0)
            for i, j in T.Tiles([h, h], parallel=True):
                B_shared[i, j] = A_shared[base + i * h + j]

            T.copy(B_shared, out[0, 0:matrix_size, 0:matrix_size])

    return main


@target("Sunmmio")
def packed_2d_to_1d_rank1_fallback_kernel(h=4, base=8, vector_size=128, matrix_size=32, dtype=T.float32):
    assert base + h * h <= vector_size
    out_shape, token_policy, out_layout = _matrix_output_spec(matrix_size, matrix_size, dtype)
    vector_layout = make_aligned_row_major((vector_size,), dtype, align_bytes=64)
    matrix_layout = make_zz_layout((matrix_size, matrix_size), [0, 1], (32, 32))

    @T.prim_func
    def main(
        out: T.MeshTensor(out_shape, token_policy, dtype, layout=out_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared((vector_size,), dtype)
            B_shared = T.alloc_shared((matrix_size, matrix_size), dtype)
            T.annotate_layout({A_shared: vector_layout, B_shared: matrix_layout})

            T.fill(A_shared, 0)
            T.fill(B_shared, 1)
            for i, j in T.Tiles([h, h], parallel=True):
                A_shared[base + i * h + j] = B_shared[i, j]

            for j in T.Tiles([matrix_size], parallel=True):
                B_shared[matrix_size - 1, j] = A_shared[j]
            T.copy(B_shared, out[0, 0:matrix_size, 0:matrix_size])

    return main


def _validate_aligned_1d_bridge(kernel, tmp_path, filename):
    src = validate_sunmmio_codegen_with_npuir_opt(
        kernel,
        tmp_path,
        mlir_filename=filename,
        expected_tokens=("suvm.tile.extract_slice", "suvm.tile.insert_slice", "!suvm.tile_view<16xf32>"),
        opt_args=STRICT_OPT_ARGS,
    )
    assert "suvm.tile.pick" not in src
    assert "suvm.tile.set" not in src


def test_serialized_rank1_zz_slices_lower_through_aligned_carriers(tmp_path):
    _validate_aligned_1d_bridge(
        serialized_rank1_zz_slices_kernel(),
        tmp_path,
        "serialized_rank1_zz_slices_suvm.mlir",
    )


def test_temp_stage_subaligned_then_direct_lowers_to_llvm(tmp_path):
    _validate_aligned_1d_bridge(
        temp_stage_subaligned_then_direct_kernel(),
        tmp_path,
        "temp_stage_subaligned_then_direct_suvm.mlir",
    )


def test_packed_1d_to_2d_falls_back_to_rank1_carriers(tmp_path):
    _validate_aligned_1d_bridge(
        packed_1d_to_2d_rank1_fallback_kernel(),
        tmp_path,
        "packed_1d_to_2d_rank1_fallback_suvm.mlir",
    )


def test_packed_2d_to_1d_falls_back_to_rank1_carriers(tmp_path):
    _validate_aligned_1d_bridge(
        packed_2d_to_1d_rank1_fallback_kernel(),
        tmp_path,
        "packed_2d_to_1d_rank1_fallback_suvm.mlir",
    )


if __name__ == "__main__":
    tilelang.testing.main()
