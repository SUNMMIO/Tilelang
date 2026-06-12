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
os.environ["SUNMMIO_TEST_LOG_IR"] = "1"

LOOSE_OPT_ARGS = ("--verify-each",)


@target("Sunmmio")
def summa_matmul(
    M=128,
    N=128,
    K=128,
    block_M=32,
    block_N=32,
    block_K=32,
    dtype="float16",
    accum_dtype="float32",
):
    shard_policy = T.MeshShardingPolicy(y=0, x=1)
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    A_shape = (M, K)
    B_shape = (K, N)
    C_shape = (M, N)
    A_layout = make_zz_layout(A_shape, [0, 1], (32, 32))
    B_layout = make_zz_layout(B_shape, [0, 1], (32, 32))
    C_layout = make_zz_layout(C_shape, [0, 1], (32, 32))

    @T.prim_func
    def kernel(
        A: T.MeshTensor(A_shape, shard_policy, device_mesh_config, dtype, layout=A_layout),  # type: ignore
        B: T.MeshTensor(B_shape, shard_policy, device_mesh_config, dtype, layout=B_layout),  # type: ignore
        C: T.MeshTensor(C_shape, shard_policy, device_mesh_config, accum_dtype, layout=C_layout),  # type: ignore
    ):
        with T.Kernel(ncores) as _cid:
            sharded_M, _ = A.shape
            _, sharded_N = B.shape
            core_row = _cid // ncols
            core_col = _cid % ncols

            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_shared((block_M, block_N), accum_dtype)

            for bx in T.serial(T.ceildiv(sharded_M, block_M)):
                for by in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.clear(C_local)
                    K_steps = T.ceildiv(K, block_K)

                    for k_tile in range(K_steps):
                        a_src_col = k_tile % ncols
                        b_src_row = k_tile % nrows
                        a_local_k = (k_tile // ncols) * block_K
                        b_local_k = (k_tile // nrows) * block_K

                        T.comm.broadcast(
                            A[
                                bx * block_M : bx * block_M + block_M,
                                a_local_k : a_local_k + block_K,
                            ],
                            A_shared,
                            (core_row, a_src_col),
                            direction="h",
                        )
                        T.comm.broadcast(
                            B[
                                b_local_k : b_local_k + block_K,
                                by * block_N : by * block_N + block_N,
                            ],
                            B_shared,
                            (b_src_row, core_col),
                            direction="v",
                        )
                        # Sunmmio MMA consumes B in the transposed operand mode;
                        # algorithmically this is still the SUMMA B(k, j) panel.
                        T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                    T.copy(C_local, C[bx * block_M, by * block_N])

    return kernel


def test_summa_matmul_codegen_validates_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        summa_matmul(),
        tmp_path,
        mlir_filename="summa_matmul_suvm.mlir",
        expected_tokens=(
            "suvm.copy_async",
            "suvm.mcast_tok",
            "suvm.tc.mma",
            "suvm.wait_token",
        ),
        opt_args=LOOSE_OPT_ARGS,
    )

    assert "sunmmio.fake" not in src
    assert src.count("suvm.mcast_tok") >= 2
    assert src.count("suvm.tc.mma") >= 1
    assert src.count("suvm.wait_token") >= 4


def test_summa_matmul_codegen_contains_broadcast_sync_sequence(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        summa_matmul(),
        tmp_path,
        mlir_filename="summa_matmul_sync_suvm.mlir",
        expected_tokens=(
            "suvm.barrier.init",
            "suvm.barrier.arrive_and_wait",
            "suvm.mcast_tok",
            "suvm.tc.mma",
            "suvm.copy_async",
        ),
        opt_args=LOOSE_OPT_ARGS,
    )

    assert_source_contains(src, ("suvm.wait_token", "suvm.tile.fill"))


if __name__ == "__main__":
    tilelang.testing.main()
