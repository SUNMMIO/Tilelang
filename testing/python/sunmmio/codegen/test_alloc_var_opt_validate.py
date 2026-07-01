import os
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.carver.arch import driver
from tilelang.layout import make_zz_layout

from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import (
    assert_source_contains,
    lower_sunmmio_kernel_to_device_tir,
    validate_sunmmio_codegen_with_npuir_opt,
)

os.environ["SUNMMIO_TEST_LOG_IR"] = "1"

tilelang.env.disable_cache()


@target("Sunmmio")
def alloc_var_scalar_state_kernel(
    M=32,
    N=32,
    dtype=T.float32,
):
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    shard_policy = T.MeshShardingPolicy(y=0, x=1)
    A_layout = make_zz_layout((M, N))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, N), shard_policy, device_mesh_config, dtype, layout=A_layout),  # type: ignore
    ):
        with T.Kernel(ncores) as _cid:
            running = T.alloc_var(dtype, init=1.0)

            for _i in T.serial(4):
                running = running + T.float32(1.0)

            if running > T.float32(3.0):
                running = running * T.float32(2.0)
            else:
                running = running - T.float32(1.0)

            T.evaluate(running)

    return main


@target("Sunmmio")
def alloc_var_all_control_flow_kernel(
    M=32,
    N=32,
    dtype=T.float32,
):
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    shard_policy = T.MeshShardingPolicy(y=0, x=1)
    A_layout = make_zz_layout((M, N))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, N), shard_policy, device_mesh_config, dtype, layout=A_layout),  # type: ignore
    ):
        with T.Kernel(ncores) as _cid:
            idx = T.alloc_var(T.int32, init=0)
            running = T.alloc_var(dtype, init=1.0)

            for _i in T.serial(4):
                running = running + T.float32(1.0)

            if running > T.float32(3.0):
                running = running * T.float32(2.0)
                idx = idx + 1
            else:
                running = running - T.float32(1.0)
                idx = idx + 2

            while idx < 4:
                running = running + T.float32(1.0)
                idx = idx + 1

            T.evaluate(running)

    return main


@target("Sunmmio")
def alloc_var_copy_mma_control_flow_kernel(
    M=128,
    N=128,
    K=128,
    block_M=32,
    block_N=32,
    block_K=32,
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
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            idx = T.alloc_var(T.int32, init=0)
            running = T.alloc_var(accum_dtype, init=1.0)

            T.clear(C_shared)
            for _i in T.serial(1):
                T.copy(A[0, 0], A_shared)
                T.copy(B[0, 0], B_shared)
                T.gemm(A_shared, B_shared, C_shared)
                running = running + T.float32(1.0)

            if running > T.float32(0.0):
                T.copy(C_shared, C[0, 0])
                running = running * T.float32(2.0)
                idx = idx + 1
            else:
                T.copy(C_shared, C[0, 0])
                running = running - T.float32(1.0)
                idx = idx + 2

            while idx < 2:
                T.copy(A[0, 0], A_shared)
                T.copy(B[0, 0], B_shared)
                T.gemm(A_shared, B_shared, C_shared)
                running = running + T.float32(1.0)
                idx = idx + 1

            T.copy(C_shared, C[0, 0])
            T.evaluate(running)

    return main


@target("Sunmmio")
def alloc_var_tiles_kernel(
    M=32,
    N=32,
    dtype=T.float32,
):
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    shard_policy = T.MeshShardingPolicy(y=0, x=1)
    A_layout = make_zz_layout((M, N))
    B_layout = make_zz_layout((M, N))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, N), shard_policy, device_mesh_config, dtype, layout=A_layout),  # type: ignore
        B: T.MeshTensor((M, N), shard_policy, device_mesh_config, dtype, layout=B_layout),  # type: ignore
    ):
        with T.Kernel(ncores) as _cid:
            A_shared = T.alloc_shared((M, N), dtype)
            bias = T.alloc_var(dtype, init=1.0)

            T.copy(A[0, 0], A_shared)
            for i, j in T.Tiles(A_shared, parallel=True):
                A_shared[i, j] = A_shared[i, j] + bias
            T.copy(A_shared, B[0, 0])

    return main


def test_alloc_var_survives_sunmmio_device_lowering():
    device_mod = lower_sunmmio_kernel_to_device_tir(alloc_var_scalar_state_kernel())
    tir_src = device_mod.script()
    assert_source_contains(
        tir_src,
        (
            "local.var",
            "T.allocate",
            "tl.local_var_init",
            "T.decl_buffer",
            "running",
            "[0]",
        ),
    )


def test_alloc_var_scalar_state_codegen_validates_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        alloc_var_scalar_state_kernel(),
        tmp_path,
        mlir_filename="alloc_var_scalar_state_suvm.mlir",
        expected_tokens=(
            "scf.for",
            "scf.if",
            "iter_args",
            "arith.addf",
            "arith.mulf",
        ),
    )
    assert "suvm.alloc" not in src


def test_alloc_var_all_control_flow_kernel_codegen_validates_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        alloc_var_all_control_flow_kernel(),
        tmp_path,
        mlir_filename="alloc_var_all_control_flow_suvm.mlir",
        expected_tokens=(
            "scf.for",
            "scf.if",
            "scf.while",
            "scf.condition",
            "iter_args",
            "arith.addi",
            "arith.addf",
            "arith.mulf",
        ),
    )
    assert "suvm.alloc" not in src


def test_alloc_var_copy_mma_control_flow_kernel_codegen_validates_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        alloc_var_copy_mma_control_flow_kernel(),
        tmp_path,
        mlir_filename="alloc_var_copy_mma_control_flow_suvm.mlir",
        expected_tokens=(
            "scf.for",
            "scf.if",
            "scf.while",
            "scf.condition",
            "-> (!suvm.token, !suvm.token, f32, i32)",
            "-> (!suvm.token, f32, i32)",
            "!suvm.token",
            "suvm.copy_async",
            "suvm.tc.mma",
            "suvm.wait_token",
            "arith.addi",
            "arith.addf",
            "arith.mulf",
        ),
    )
    assert "sunmmio.fake" not in src


def test_alloc_var_inside_tiles_codegen_validates_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        alloc_var_tiles_kernel(),
        tmp_path,
        mlir_filename="alloc_var_tiles_suvm.mlir",
        expected_tokens=(
            "suvm.tile.addf",
            "suvm.tile.store",
        ),
    )
    assert "sunmmio.fake" not in src


if __name__ == "__main__":
    tilelang.testing.main()
