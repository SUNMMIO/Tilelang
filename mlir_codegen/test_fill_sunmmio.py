import tilelang
import tilelang.language as T
from tilelang.utils.target import determine_target
from compile_pipeline import compile_test
from sunmmio_test_utils import save_final_ast, save_final_mlir, verify_final_mlir

tilelang.env.disable_cache()


def fill_tiled_test(B, M, N, block_B, block_M, block_N, tile_size, index_map, dtype="float16"):
    @T.prim_func
    def main(A: T.Tensor((B, M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), T.ceildiv(B, block_B), threads=1) as (bx, by, bz):
            A_shared = T.alloc_shared((block_B, block_M, block_N), dtype)
            # T.annotate_tileview({A_shared: make_tileview(A_shared, tile_size, index_map)})

            # 1. Fill the entire buffer
            T.fill(A_shared, T.float16(1.0))

            # 2. Fill a region of the buffer (e.g., first half of M dimension)
            # Region: A_shared[0:block_B, 0:block_M//2, 0:block_N]
            # Since block_M=256, block_M//2 = 128. tile_size[0]=32, so 128 is divisible by 32.
            # This should generate a loop over 4 tiles (128/32).
            T.fill(A_shared[0:block_B, 0 : block_M // 2, 0:block_N], T.float16(2.0))

            # 3. Fill a region with offset (e.g., second half of M dimension)
            # Region: A_shared[0:block_B, block_M//2:block_M, 0:block_N]
            # Offset is 128 (4 tiles), extent is 128 (4 tiles).
            T.fill(A_shared[0:block_B, block_M // 2 : block_M, 0:block_N], T.float16(3.0))

            T.clear(A_shared)

            T.copy(A_shared, A[bz * block_B : (bz + 1) * block_B, by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])

    return main


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
log_dir = "/home/cedu/projects/Tilelang-Mesh/tilelang_mesh/mlir_codegen/logs_fill_sunmmio"
import os

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
