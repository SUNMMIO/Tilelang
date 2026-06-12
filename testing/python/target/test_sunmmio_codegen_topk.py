import os

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.carver.arch import driver
from tilelang.layout import make_zz_layout
from tilelang.tileview import make_tileview

from sunmmio_codegen_validation_utils import (
    assert_source_contains,
    validate_sunmmio_codegen_with_npuir_opt,
)


tilelang.env.disable_cache()
# os.environ["SUNMMIO_TEST_PRINT"] = "0"
os.environ["SUNMMIO_TEST_LOG_IR"] = "1"


def topk_full_kernel(
    M=128,
    N=128,
    topk=2,
    block_M=32,
):
    dtype = T.float32
    grid_m = T.ceildiv(M, block_M)

    assert M % block_M == 0
    assert N == 128
    assert block_M == 32
    assert topk == 2

    shard_policy = T.MeshShardingPolicy()
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    logits_shape = (M, N)
    topk_shape = (M, topk)
    logits_layout = make_zz_layout(logits_shape, [0, 1], (32, 32))
    topk_gates_layout = make_zz_layout(topk_shape, [0, 1], (32, 2))
    topk_indices_layout = make_zz_layout(topk_shape, [0, 1], (32, 2))

    @T.prim_func
    def main(
        logits: T.MeshTensor(logits_shape, shard_policy, device_mesh_config, dtype, layout=logits_layout),  # type: ignore
        topk_gates: T.MeshTensor(topk_shape, shard_policy, device_mesh_config, dtype, layout=topk_gates_layout),  # type: ignore
        topk_indices: T.MeshTensor(topk_shape, shard_policy, device_mesh_config, T.int32, layout=topk_indices_layout),  # type: ignore
    ):
        with T.Kernel(ncores) as _cid:
            logits_shared = T.alloc_shared((block_M, N), dtype, scope="shared.rsram")
            expand_max_idx = T.alloc_shared((block_M, N), T.int32, scope="shared.rsram")
            max_val = T.alloc_shared((block_M,), dtype, scope="shared.rsram")
            max_idx = T.alloc_shared((block_M,), T.int32, scope="shared.rsram")

            T.annotate_tileview(
                {
                    logits_shared: make_tileview(logits_shared, (4, 32), (-2, -1)),
                    expand_max_idx: make_tileview(expand_max_idx, (4, 32), (-2, -1)),
                }
            )

            for bx in T.serial(grid_m):
                T.copy(logits[bx * block_M, 0], logits_shared)

                for k in T.serial(topk):
                    T.fill(expand_max_idx, T.int32(-1))
                    T.reduce_max(logits_shared, max_val, dim=1, clear=True)

                    for i, j in T.Tiles([block_M, N], parallel=True):
                        expand_max_idx[i, j] = T.if_then_else(
                            max_val[i] == logits_shared[i, j],
                            T.cast(j, T.int32),
                            expand_max_idx[i, j],
                        )

                    T.reduce_max(expand_max_idx, max_idx, dim=1, clear=True)

                    for i, j in T.Tiles([block_M, N], parallel=True):
                        logits_shared[i, j] = T.if_then_else(
                            max_val[i] == logits_shared[i, j],
                            T.float32(-10000.0),
                            logits_shared[i, j],
                        )

                    T.copy(max_val, topk_gates[bx * block_M : (bx + 1) * block_M, k : k + 1])
                    T.copy(max_idx, topk_indices[bx * block_M : (bx + 1) * block_M, k : k + 1])

    return main


def topk_transition_kernel(
    M=128,
    N=128,
    topk=2,
    block_M=32,
):
    dtype = T.float32
    grid_m = T.ceildiv(M, block_M)

    assert M % block_M == 0
    assert N == 128
    assert block_M == 32
    assert topk == 2

    shard_policy = T.MeshShardingPolicy()
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    logits_shape = (M, N)
    topk_shape = (M, topk)
    logits_layout = make_zz_layout(logits_shape, [0, 1], (32, 32))
    topk_gates_layout = make_zz_layout(topk_shape, [0, 1], (32, 2))
    topk_indices_layout = make_zz_layout(topk_shape, [0, 1], (32, 2))

    @T.prim_func
    def main(
        logits: T.MeshTensor(logits_shape, shard_policy, device_mesh_config, dtype, layout=logits_layout),  # type: ignore
        topk_gates: T.MeshTensor(topk_shape, shard_policy, device_mesh_config, dtype, layout=topk_gates_layout),  # type: ignore
        topk_indices: T.MeshTensor(topk_shape, shard_policy, device_mesh_config, T.int32, layout=topk_indices_layout),  # type: ignore
    ):
        with T.Kernel(ncores) as _cid:
            logits_shared = T.alloc_shared((block_M, N), dtype, scope="shared.rsram")
            expand_max_idx = T.alloc_shared((block_M, N), T.int32, scope="shared.rsram")
            max_val = T.alloc_shared((block_M,), dtype, scope="shared.rsram")
            max_idx = T.alloc_shared((block_M,), T.int32, scope="shared.rsram")

            for bx in T.serial(grid_m):
                T.copy(logits[bx * block_M, 0], logits_shared)

                for k in T.serial(topk):
                    T.fill(expand_max_idx, T.int32(-1))
                    T.reduce_max(logits_shared, max_val, dim=1, clear=True)

                    for i, j in T.Tiles([block_M, N], parallel=True):
                        expand_max_idx[i, j] = T.if_then_else(
                            max_val[i] == logits_shared[i, j],
                            T.cast(j, T.int32),
                            expand_max_idx[i, j],
                        )

                    T.reduce_max(expand_max_idx, max_idx, dim=1, clear=True)

                    for i, j in T.Tiles([block_M, N], parallel=True):
                        logits_shared[i, j] = T.if_then_else(
                            max_val[i] == logits_shared[i, j],
                            T.float32(-10000.0),
                            logits_shared[i, j],
                        )

                    T.copy(max_val, topk_gates[bx * block_M : (bx + 1) * block_M, k : k + 1])
                    T.copy(max_idx, topk_indices[bx * block_M : (bx + 1) * block_M, k : k + 1])

    return main


def test_topk_codegen_generates_expected_ops(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        topk_transition_kernel(),
        tmp_path,
        mlir_filename="topk_transition_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.tile.store"),
        opt_args=("--verify-each",),
    )
    assert_source_contains(
        src,
        (
            "suvm.copy_async",
            "suvm.tile.store",
        ),
    )


if __name__ == "__main__":
    tilelang.testing.main()
