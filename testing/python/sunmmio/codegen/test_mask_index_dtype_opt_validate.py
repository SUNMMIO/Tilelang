import os
import re

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import make_aligned_row_major

from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import validate_sunmmio_codegen_with_npuir_opt


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
os.environ["SUNMMIO_TEST_LOG_IR"] = "1"

LOOSE_OPT_ARGS = ("--verify-each",)


@target("Sunmmio")
def predicated_store_mask_dtype_kernel(dtype, extent=512, valid_n=33):
    shard_policy = T.MeshShardingPolicy()
    tensor_shape = (extent,)
    tensor_layout = make_aligned_row_major(tensor_shape, dtype, align_bytes=1024)

    @T.prim_func
    def main(out: T.MeshTensor(tensor_shape, shard_policy, dtype, layout=tensor_layout)):  # type: ignore
        with T.Kernel():
            out_shared = T.alloc_shared(tensor_shape, dtype)
            T.annotate_layout({out_shared: make_aligned_row_major(tensor_shape, dtype, align_bytes=1024)})

            T.clear(out_shared)
            for i in T.Tiles([valid_n], parallel=True):
                out_shared[i] = T.Cast(dtype, 7)
            T.copy(out_shared, out[0:extent])

    return main


@pytest.mark.parametrize(
    "dtype,expected_index_dtype",
    [
        (T.bfloat16, "i16"),
        (T.int16, "i16"),
        (T.float32, "i32"),
        (T.int32, "i32"),
    ],
)
def test_predicated_store_mask_range_uses_value_dtype_width(dtype, expected_index_dtype, tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        predicated_store_mask_dtype_kernel(dtype),
        tmp_path,
        mlir_filename=f"predicated_store_mask_{expected_index_dtype}_suvm.mlir",
        expected_tokens=("suvm.tile.range", "suvm.tile.cmpi", "suvm.tile.select"),
        opt_args=LOOSE_OPT_ARGS,
    )
    assert re.search(rf"suvm\.tile\.range : !suvm\.tile<[^>]*x{expected_index_dtype}>", src)


def test_float16_sunmmio_codegen_reports_unsupported_dtype(tmp_path):
    with pytest.raises(
        Exception,
        match="Unsupported SunMMIO SUVM dtype.*float16.*does not support float16.*Use bfloat16",
    ):
        validate_sunmmio_codegen_with_npuir_opt(
            predicated_store_mask_dtype_kernel(T.float16),
            tmp_path,
            mlir_filename="predicated_store_mask_float16_rejected_suvm.mlir",
        )


if __name__ == "__main__":
    tilelang.testing.main()
