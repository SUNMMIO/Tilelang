import os

import tilelang
import tilelang.language as T
import tvm
from compile_pipeline import compile_test
from sunmmio_test_utils import save_final_ast, save_final_mlir, verify_final_mlir
from tilelang.carver.arch import driver
from tilelang.utils.target import determine_target

tilelang.env.disable_cache()


def tile_elementwise_ops_test(
    batch,
    m,
    n,
    block_b,
    block_m,
    block_n,
    dtype="float16",
):
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols
    grid_b = T.ceildiv(batch, block_b)
    grid_m = T.ceildiv(m, block_m)
    grid_n = T.ceildiv(n, block_n)

    @T.prim_func
    def main(
        A: T.Tensor((batch, m, n), dtype),
        B: T.Tensor((batch, m, n), dtype),
        C: T.Tensor((batch, m, n), dtype),
    ):
        with T.Kernel(ncores) as _cid:
            A_shared = T.alloc_shared((block_b, block_m, block_n), dtype)
            B_shared = T.alloc_shared((block_b, block_m, block_n), dtype)
            C_shared = T.alloc_shared((block_b, block_m, block_n), dtype)

            for bz in T.serial(grid_b):
                for by in T.serial(grid_m):
                    for bx in T.serial(grid_n):
                        if (bz * grid_m * grid_n + by * grid_n + bx) % ncores == _cid:
                            T.copy(
                                A[
                                    bz * block_b : (bz + 1) * block_b,
                                    by * block_m : (by + 1) * block_m,
                                    bx * block_n : (bx + 1) * block_n,
                                ],
                                A_shared,
                            )
                            T.copy(
                                B[
                                    bz * block_b : (bz + 1) * block_b,
                                    by * block_m : (by + 1) * block_m,
                                    bx * block_n : (bx + 1) * block_n,
                                ],
                                B_shared,
                            )

                            for b, i, j in T.Tiles(A_shared, parallel=True):
                                x = T.max(A_shared[b, i, j], B_shared[b, i, j])
                                y = T.min(x, T.exp(B_shared[b, i, j]))
                                z = T.log(T.abs(y) + T.float32(1.0))
                                C_shared[b, i, j] = T.if_then_else(
                                    z > T.float32(0.5),
                                    T.rsqrt(z + T.float32(1.0)),
                                    T.ieee_frcp(z + T.float32(2.0)),
                                )

                            for b, i, j in T.Tiles(A_shared, parallel=True):
                                neg_a = T.float32(0.0) - T.Cast("float32", A_shared[b, i, j])
                                rem_b = T.fmod(T.Cast("float32", B_shared[b, i, j]), T.float32(3.0))
                                C_shared[b, i, j] = (
                                    T.ceil(C_shared[b, i, j])
                                    + T.floor(A_shared[b, i, j])
                                    + T.round(B_shared[b, i, j])
                                    + T.trunc(A_shared[b, i, j])
                                    + neg_a
                                    + rem_b
                                )

                            T.copy(
                                C_shared,
                                C[
                                    bz * block_b : (bz + 1) * block_b,
                                    by * block_m : (by + 1) * block_m,
                                    bx * block_n : (bx + 1) * block_n,
                                ],
                            )

    return main


PYTEST_TILE_OPS_CONFIG = dict(
    batch=2,
    m=256,
    n=128,
    block_b=2,
    block_m=256,
    block_n=128,
    dtype="float16",
)

REQUIRED_TILE_OPS = [
    "suvm.tile.abs",
    "suvm.tile.ceil",
    "suvm.tile.cmpf",
    "suvm.tile.floor",
    "suvm.tile.ln",
    "suvm.tile.maxf",
    "suvm.tile.minf",
    "suvm.tile.neg",
    "suvm.tile.recip",
    "suvm.tile.remf",
    "suvm.tile.round",
    "suvm.tile.rsqrt",
    "suvm.tile.select",
    "suvm.tile.trunc",
]


def build_pytest_tile_ops_case():
    return tile_elementwise_ops_test(**PYTEST_TILE_OPS_CONFIG)


def compile_tile_ops_case(func, log_dir):
    log_dir = os.fspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    _, device_mod = compile_test(
        func,
        out_idx=[2],
        target="Sunmmio",
        log_pass_output=True,
        log_dir=log_dir,
    )
    save_final_ast(log_dir, device_mod, echo=False)

    target = determine_target("Sunmmio", return_object=True)
    mlir_mod = tvm.get_global_func("target.build.tilelang_sunmmio_without_compile")(device_mod, target, "suvm")
    mlir_source = mlir_mod.inspect_source()
    mlir_path = save_final_mlir(log_dir, mlir_source, echo=False)
    verify_final_mlir(log_dir, mlir_path)

    return {
        "device_mod": device_mod,
        "log_dir": log_dir,
        "mlir_source": mlir_source,
        "mlir_path": mlir_path,
    }


def test_tile_ops_lower_core_tile_math(tmp_path):
    result = compile_tile_ops_case(build_pytest_tile_ops_case(), tmp_path / "tile_ops_core_math")
    missing_ops = [op for op in REQUIRED_TILE_OPS if op not in result["mlir_source"]]
    assert not missing_ops, f"Missing expected tile ops: {missing_ops}"


def test_tile_ops_should_not_use_fake_tile_addf(tmp_path):
    result = compile_tile_ops_case(build_pytest_tile_ops_case(), tmp_path / "tile_ops_no_fake_add")
    assert "fake_tile_addf" not in result["mlir_source"]


def test_tile_ops_should_copy_between_global_and_rsram(tmp_path):
    result = compile_tile_ops_case(build_pytest_tile_ops_case(), tmp_path / "tile_ops_copy_paths")
    mlir_source = result["mlir_source"]
    assert "get_partitioned_tile_view %arg0" in mlir_source
    assert "get_partitioned_tile_view %arg1" in mlir_source
    assert "get_partitioned_tile_view %arg2" in mlir_source
    assert "suvm.copy_async" in mlir_source


def main():
    func = tile_elementwise_ops_test(2, 256, 128, 2, 256, 128)
    log_dir = os.environ.get(
        "TL_TILE_OPS_LOG_DIR",
        os.path.join(os.path.dirname(__file__), "logs_tile_ops_sunmmio"),
    )
    os.makedirs(log_dir, exist_ok=True)

    _, device_mod = compile_test(
        func,
        out_idx=[2],
        target="Sunmmio",
        log_pass_output=True,
        log_dir=log_dir,
    )
    save_final_ast(log_dir, device_mod)

    target = determine_target("Sunmmio", return_object=True)
    mlir_mod = tvm.get_global_func("target.build.tilelang_sunmmio_without_compile")(device_mod, target, "suvm")
    mlir_source = mlir_mod.inspect_source()
    missing_ops = [op for op in REQUIRED_TILE_OPS if op not in mlir_source]
    assert not missing_ops, f"Missing expected tile ops: {missing_ops}"
    mlir_path = save_final_mlir(log_dir, mlir_source, echo=True)
    verify_final_mlir(log_dir, mlir_path)


if __name__ == "__main__":
    main()
