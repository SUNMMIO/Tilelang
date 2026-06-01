import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.utils.target import determine_target
from compile_pipeline import compile_test
from sunmmio_test_utils import save_final_ast, save_final_mlir, verify_final_mlir

tilelang.env.disable_cache()


def fill_tiled_test(B, M, N, block_B, block_M, block_N, tile_size, index_map, dtype="float16"):
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols
    grid_b = T.ceildiv(B, block_B)
    grid_m = T.ceildiv(M, block_M)
    grid_n = T.ceildiv(N, block_N)

    @T.prim_func
    def main(A: T.Tensor((B, M, N), dtype)):
        with T.Kernel(ncores) as _cid:
            A_shared = T.alloc_shared((block_B, block_M, block_N), dtype)
            # T.annotate_tileview({A_shared: make_tileview(A_shared, tile_size, index_map)})

            for bz in T.serial(grid_b):
                for by in T.serial(grid_m):
                    for bx in T.serial(grid_n):
                        if (bz * grid_m * grid_n + by * grid_n + bx) % ncores == _cid:
                            # 1. Fill the entire buffer
                            T.fill(A_shared, T.float16(1.0))

                            # 2. Fill a region of the buffer (e.g., first half of M dimension)
                            T.fill(A_shared[0:block_B, 0 : block_M // 2, 0:block_N], T.float16(2.0))

                            # 3. Fill a region with offset (e.g., second half of M dimension)
                            T.fill(A_shared[0:block_B, block_M // 2 : block_M, 0:block_N], T.float16(3.0))

                            T.clear(A_shared)

                            T.copy(
                                A_shared,
                                A[
                                    bz * block_B : (bz + 1) * block_B,
                                    by * block_M : (by + 1) * block_M,
                                    bx * block_N : (bx + 1) * block_N,
                                ],
                            )

    return main


PYTEST_FILL_CONFIG = dict(
    B=16,
    M=256,
    N=128,
    block_B=16,
    block_M=256,
    block_N=128,
    tile_size=(32, 32),
    index_map=(-2, -1),
)


def build_pytest_fill_case(dtype="float16"):
    return fill_tiled_test(dtype=dtype, **PYTEST_FILL_CONFIG)


def compile_fill_case(func, log_dir):
    log_dir = os.fspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    host_mod, device_mod = compile_test(
        func,
        out_idx=[0],
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
        "host_mod": host_mod,
        "device_mod": device_mod,
        "log_dir": log_dir,
        "mlir_source": mlir_source,
        "mlir_path": mlir_path,
    }


def test_fill_lowers_to_tile_fill_and_tile_store(tmp_path):
    result = compile_fill_case(build_pytest_fill_case(), tmp_path / "fill_lowering")
    mlir_source = result["mlir_source"]
    assert "suvm.tile.fill" in mlir_source
    assert "suvm.tile.store" in mlir_source


def test_fill_should_write_back_to_global_memory(tmp_path):
    result = compile_fill_case(build_pytest_fill_case(), tmp_path / "fill_writeback")
    mlir_source = result["mlir_source"]
    assert "get_partitioned_tile_view %arg0" in mlir_source
    assert "suvm.copy_async" in mlir_source
    assert '{sunmmio.fake = "call"}' not in mlir_source


B = 64
M = 512
N = 1024
block_B = 16
block_M = 256
block_N = 128
tile_size = (32, 32)
index_map = (-2, -1)

func = fill_tiled_test(B, M, N, block_B, block_M, block_N, tile_size, index_map)
func.show()

# try:
#     jit_kernel = tilelang.compile(func, target="Sunmmio", verbose=True)
# except Exception as e:
#     print(e)
import os

log_dir = os.path.join(os.path.dirname(__file__), "logs_fill_sunmmio")

os.makedirs(log_dir, exist_ok=True)
host_mod, device_mod = compile_test(func, out_idx=[2], target="Sunmmio", log_pass_output=True, log_dir=log_dir)
device_mod.show()
save_final_ast(log_dir, device_mod, echo=True)

print("--- Running CodeGenTileLangSunMMIO ---")
import tvm

target = determine_target("Sunmmio", return_object=True)
mlir_mod = tvm.get_global_func("target.build.tilelang_sunmmio_without_compile")(device_mod, target, "suvm")
mlir_source = mlir_mod.inspect_source()
mlir_path = save_final_mlir(log_dir, mlir_source, echo=True)
verify_final_mlir(log_dir, mlir_path)
