import os

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.carver.arch import driver
from tilelang.layout import make_zz_layout

from compile_pipeline import target
from sunmmio_codegen_validation_utils import (
    assert_source_contains,
    validate_sunmmio_codegen_with_npuir_opt,
)


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
# os.environ["SUNMMIO_TEST_LOG_IR"] = "1"

LOOSE_OPT_ARGS = ("--verify-each",)


@target("Sunmmio")
def reduce_max_mul(
    m=10240,
    n=10240,
    block_m=128,
    block_n=128,
    dtype=T.float16,
):
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    shard_policy = T.MeshShardingPolicy(y=0, x=1)
    tensor_shape = (m, n)
    tensor_layout = make_zz_layout(tensor_shape, [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor(tensor_shape, shard_policy, device_mesh_config, dtype, layout=tensor_layout),  # type: ignore
        B: T.MeshTensor(tensor_shape, shard_policy, device_mesh_config, dtype, layout=tensor_layout),  # type: ignore
        C: T.MeshTensor(tensor_shape, shard_policy, device_mesh_config, dtype, layout=tensor_layout),  # type: ignore
    ):
        with T.Kernel(ncores):
            sharded_m, sharded_n = A.shape

            A_tile = T.alloc_shared((block_m, block_n), dtype, scope="shared.rsram")
            B_tile = T.alloc_shared((block_m, block_n), dtype, scope="shared.rsram")
            C_tile = T.alloc_shared((block_m, block_n), dtype, scope="shared.rsram")
            row_max = T.alloc_shared((block_m), dtype, scope="shared.rsram")

            for by in T.serial(T.ceildiv(sharded_m, block_m)):
                T.fill(row_max, -T.infinity(dtype))

                for bx in T.serial(T.ceildiv(sharded_n, block_n)):
                    T.copy(
                        B[by * block_m : (by + 1) * block_m, bx * block_n : (bx + 1) * block_n],
                        B_tile,
                    )
                    T.comm.all_reduce(B_tile, row_max, "max", direction="h", dim=1, clear=False)

                for bx in T.serial(T.ceildiv(sharded_n, block_n)):
                    T.copy(
                        A[by * block_m : (by + 1) * block_m, bx * block_n : (bx + 1) * block_n],
                        A_tile,
                    )
                    for i, j in T.Tiles([block_m, block_n]):
                        C_tile[i, j] = A_tile[i, j] * row_max[i]
                    T.copy(
                        C_tile,
                        C[by * block_m : (by + 1) * block_m, bx * block_n : (bx + 1) * block_n],
                    )

    return main


def test_reduce_max_mul_codegen_validates_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        reduce_max_mul(),
        tmp_path,
        mlir_filename="reduce_max_mul_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.mcast_tok", "suvm.tile.reduce"),
        opt_args=LOOSE_OPT_ARGS,
    )
    assert_source_contains(src, ("suvm.mcast_tok", "direction =  row", "suvm.tile.reduce", "max"))
    assert "sunmmio.fake" not in src
    assert "fake_missing" not in src

    rank1_broadcast_dst = [
        line
        for line in src.splitlines()
        if "suvm.get_partitioned_tile_view" in line
        and "!suvm.memtensor<4x128xbf16" in line
        and "tiled_dims = [1]" in line
        and "-> !suvm.tile_view<128xbf16>" in line
    ]
    assert rank1_broadcast_dst


if __name__ == "__main__":
    tilelang.testing.main()
