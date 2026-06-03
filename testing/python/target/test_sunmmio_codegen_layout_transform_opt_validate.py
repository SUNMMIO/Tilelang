import os

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.carver.arch import driver
from tilelang.language.mesh_tensor import MeshReplicationType
from tilelang.layout import make_row_major, make_zz_layout

from sunmmio_codegen_validation_utils import (
    assert_source_contains,
    validate_sunmmio_codegen_with_npuir_opt,
)


tilelang.env.disable_cache()
os.environ["SUNMMIO_TEST_PRINT"] = "0"
# os.environ["SUNMMIO_TEST_LOG_IR"] = "1"


def layout_transform_roundtrip_kernel(
    m=128,
    n=128,
    dtype=T.float16,
):
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    shard_policy = T.MeshShardingPolicy(replicate=MeshReplicationType.ALL)
    dram_layout = make_zz_layout((m, n), axes=[0, 1], block_shape=(32, 32))
    rsram_layout = make_row_major((m, n))

    @T.prim_func
    def main(
        A: T.MeshTensor((m, n), shard_policy, device_mesh_config, dtype, layout=dram_layout),  # type: ignore
        B: T.MeshTensor((m, n), shard_policy, device_mesh_config, dtype, layout=dram_layout),  # type: ignore
    ):
        with T.Kernel(ncores, threads=128) as _cid:
            A_rsram = T.alloc_shared((m, n), dtype, scope="shared.rsram")
            T.annotate_layout({A_rsram: rsram_layout})

            T.copy(A, A_rsram)
            T.copy(A_rsram, B)

    return main


def test_layout_transform_codegen_validates_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        layout_transform_roundtrip_kernel(),
        tmp_path,
        mlir_filename="layout_transform_roundtrip_suvm.mlir",
        expected_tokens=(
            "suvm.copy_async",
            "suvm.transform_layout_async",
            "suvm.wait_token",
        ),
    )
    assert_source_contains(src, ("!suvm.token", "suvm.get_partitioned_tile_view"))
    assert src.count("suvm.transform_layout_async") >= 2
    assert "sunmmio.fake" not in src


if __name__ == "__main__":
    tilelang.testing.main()
