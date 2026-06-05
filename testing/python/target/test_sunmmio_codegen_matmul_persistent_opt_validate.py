import os

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.carver.arch import driver
from tilelang.layout import make_zz_layout

from compile_pipeline import target
from sunmmio_codegen_validation_utils import (
    assert_source_contains,
    lower_sunmmio_kernel_to_device_tir,
    validate_sunmmio_codegen_with_npuir_opt,
    write_sunmmio_codegen_logs,
)


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
# os.environ["SUNMMIO_TEST_LOG_IR"] = "1"

LOOSE_OPT_ARGS = ("--verify-each",)


@target("Sunmmio")
def matmul_persistent_kernel(
    M=128,
    N=128,
    K=128,
    block_M=32,
    block_N=32,
    block_K=32,
    num_stages=3,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    shard_policy = T.MeshShardingPolicy(y=0, x=1)
    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), shard_policy, device_mesh_config, dtype, layout=A_layout),  # type: ignore
        B: T.MeshTensor((K, N), shard_policy, device_mesh_config, dtype, layout=B_layout),  # type: ignore
        C: T.MeshTensor((M, N), shard_policy, device_mesh_config, accum_dtype, layout=C_layout),  # type: ignore
    ):
        with T.Kernel(ncores) as _cid:
            sharded_M, sharded_K = A.shape
            _, sharded_N = B.shape

            A_shared_dist = T.alloc_shared((block_M, block_K * ncols), dtype)
            B_shared_dist = T.alloc_shared((block_K * nrows, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            for bx in T.serial(T.ceildiv(sharded_M, block_M)):
                for by in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.clear(C_shared)
                    for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=num_stages):
                        T.comm.all_gather(
                            A[
                                bx * block_M : (bx + 1) * block_M,
                                k * block_K : (k + 1) * block_K,
                            ],
                            A_shared_dist,
                            direction="horizontal",
                            axis=-1,
                        )
                        T.comm.all_gather(
                            B[
                                k * block_K : (k + 1) * block_K,
                                by * block_N : (by + 1) * block_N,
                            ],
                            B_shared_dist,
                            direction="vertical",
                            axis=0,
                        )
                        T.gemm(A_shared_dist, B_shared_dist, C_shared)

                    T.copy(C_shared, C[bx * block_M, by * block_N])

    return main


def test_matmul_persistent_lowers_to_expected_sunmmio_device_tir():
    block_M = 32
    block_N = 32
    block_K = 32
    num_stages = 3
    nrows, ncols = driver.get_sunmmio_device_mesh_config()

    device_mod = lower_sunmmio_kernel_to_device_tir(
        matmul_persistent_kernel(
            block_M=block_M,
            block_N=block_N,
            block_K=block_K,
            num_stages=num_stages,
        )
    )
    src = device_mod.script(show_meta=True)
    write_sunmmio_codegen_logs(
        case_name="matmul_persistent_device_tir",
        tir_src=src,
    )

    assert_source_contains(
        src,
        (
            "T.dma_copy",
            "T.broadcast_",
            "T.wait_token",
            "T.barrier_arrive_and_wait",
            "T.mma_sunmmio",
        ),
    )
    assert_source_contains(
        src,
        (
            f"T.decl_buffer(({num_stages}, {block_M}, {block_K * ncols})",
            f"T.decl_buffer(({num_stages}, {block_K * nrows}, {block_N})",
            f"T.decl_buffer(({num_stages}, {block_M}, {block_K})",
        ),
    )


@pytest.mark.xfail(
    raises=tvm.error.InternalError,
    reason=(
        "matmul_persistent all_gather pipeline currently lowers staged shared buffers "
        "to rank-3 decl_buffer views, while SunMMIO MLIR type mapping still expects "
        "layout rank to match the original 2-D buffer shape."
    ),
    strict=True,
)
def test_matmul_persistent_codegen_generates_expected_suvm_ops(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        matmul_persistent_kernel(),
        tmp_path,
        mlir_filename="matmul_persistent_suvm.mlir",
        expected_tokens=(
            "suvm.mcast_tok",
            "suvm.tc.mma",
            "suvm.copy_async",
            "suvm.wait_token",
        ),
        opt_args=LOOSE_OPT_ARGS,
    )

    assert "sunmmio.fake" not in src
    assert src.count("suvm.mcast_tok") >= 2
    assert src.count("suvm.tc.mma") >= 1


if __name__ == "__main__":
    tilelang.testing.main()
