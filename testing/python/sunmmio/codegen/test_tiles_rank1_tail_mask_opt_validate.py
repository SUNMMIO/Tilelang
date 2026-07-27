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


@target("Sunmmio")
def dynamic_rank1_tail_mask_kernel(h=4, base=8, vector_size=128, matrix_size=32, dtype=T.float32):
    out_shape = (16, matrix_size, matrix_size)
    token_policy = T.MeshShardingPolicy(cross_mesh_dim=0)
    out_layout = make_zz_layout(out_shape, [1, 2], (32, 32))
    lengths_shape = (128,)
    lengths_layout = make_aligned_row_major(lengths_shape, T.int32, align_bytes=1024)
    vector_layout = make_aligned_row_major((vector_size,), dtype, align_bytes=64)
    matrix_layout = make_zz_layout((matrix_size, matrix_size), [0, 1], (32, 32))

    @T.prim_func
    def main(
        out: T.MeshTensor(out_shape, token_policy, dtype, layout=out_layout),  # type: ignore
        lengths: T.MeshTensor(lengths_shape, T.MeshShardingPolicy(), T.int32, layout=lengths_layout),  # type: ignore
    ):
        with T.Kernel():
            packed = T.alloc_shared((vector_size,), dtype)
            matrix = T.alloc_shared((matrix_size, matrix_size), dtype)
            lengths_shared = T.alloc_shared(lengths_shape, T.int32)
            valid = T.alloc_var(T.int32, init=0)
            T.annotate_layout(
                {
                    packed: vector_layout,
                    matrix: matrix_layout,
                    lengths_shared: lengths_layout,
                }
            )

            T.copy(lengths, lengths_shared)
            valid = T.max(T.min(lengths_shared[0], h), 0)
            T.fill(packed, 1)
            T.fill(matrix, 0)
            for i, j in T.Tiles([h, valid], parallel=True):
                matrix[i, j] = packed[base + i * h + j]

            T.copy(matrix, out[0, 0:matrix_size, 0:matrix_size])

    return main


def test_dynamic_rank1_tail_uses_axis_mask(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        dynamic_rank1_tail_mask_kernel(),
        tmp_path,
        mlir_filename="dynamic_rank1_tail_mask_suvm.mlir",
        expected_tokens=("suvm.tile.range : !suvm.tile<4xi32>", "-> !suvm.tile<4xi1>"),
        opt_args=STRICT_OPT_ARGS,
    )
    # TileAxisMask expands to tile.range + cmpi in the SUVM builder.
    assert "suvm.tile.rect_mask" not in src


if __name__ == "__main__":
    tilelang.testing.main()
