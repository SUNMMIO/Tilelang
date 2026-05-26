import tilelang
import tilelang.language as T
from tilelang.utils.target import determine_target
from compile_pipeline import compile_test
from sunmmio_test_utils import save_final_ast, save_final_mlir, verify_final_mlir

tilelang.env.disable_cache()


def dot_mul_tiled_parallel_3D(Batch, M, N, block_B, block_M, block_N, tile_size, index_map, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
        A: T.Tensor((Batch, M, N), dtype),
        B: T.Tensor((Batch, M, N), dtype),
        C: T.Tensor((Batch, M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), T.ceildiv(Batch, block_B), threads=1) as (bx, by, bz):
            A_shared = T.alloc_shared((block_B, block_M, block_N), dtype)

            B_shared = T.alloc_shared((block_B, block_M, block_N), dtype)

            C_shared = T.alloc_shared((block_B, block_M, block_N), accum_dtype)

            # T.clear(C_shared)

            T.copy(A[bz * block_B : (bz + 1) * block_B, by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N], A_shared)
            T.copy(B[bz * block_B : (bz + 1) * block_B, by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N], B_shared)
            # T.copy(D[by * block_M:(by + 1) * block_M], D_shared)
            for b, i, j in T.Tiles(A_shared, parallel=True):
                # temp = A_shared[b, i, j] * B_shared[b, i, j]
                # A_shared[b, i, j] = temp * T.float32(5.0)
                A_shared[b, i, j] = A_shared[b, i, j] * T.float32(2.0)
                B_shared[b, i, j] = A_shared[b, i, j] * B_shared[b, i, j]
                C_shared[b, i, j] = T.exp(A_shared[b, i, j]) + T.exp(B_shared[b, i, j])

            T.copy(C_shared, C[bz * block_B : (bz + 1) * block_B, by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])

    return main


# broadcast test
def test_tiles_broadcast(Batch, M, N, block_B, block_M, block_N, tile_size, index_map, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
        A: T.Tensor((Batch, M, N), dtype), B: T.Tensor((Batch, M, N), dtype), C: T.Tensor((Batch, M, N), dtype), D: T.Tensor((M,), dtype)
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), T.ceildiv(Batch, block_B), threads=1) as (bx, by, bz):
            A_shared = T.alloc_shared((block_B, block_M, block_N), dtype)
            # T.annotate_tileview({A_shared: make_tileview(A_shared, tile_size, index_map)})
            B_shared = T.alloc_shared((block_B, block_M, block_N), dtype)
            # T.annotate_tileview({B_shared: make_tileview(B_shared, tile_size, index_map)})
            C_shared = T.alloc_shared((block_B, block_M, block_N), accum_dtype)
            # T.annotate_tileview({C_shared: make_tileview(C_shared, tile_size, index_map)})
            D_shared = T.alloc_shared((block_M,), dtype)

            # T.clear(C_shared)

            T.copy(A[bz * block_B : (bz + 1) * block_B, by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N], A_shared)
            T.copy(B[bz * block_B : (bz + 1) * block_B, by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N], B_shared)
            T.copy(D[by * block_M : (by + 1) * block_M], D_shared)
            for b, i, j in T.Tiles(A_shared, parallel=True):
                A_shared[b, i, j] = A_shared[b, i, j] + D_shared[i]
                A_shared[b, i, j] = A_shared[b, i, j] * T.float32(2.0)
                C_shared[b, i, j] = A_shared[b, i, j] * B_shared[b, i, j]
                # C_shared[b, i, j] = T.exp(A_shared[b, i, j]) + T.exp(B_shared[b, i, j])

            for b, i, j in T.Tiles(A_shared, parallel=True):
                A_shared[b, i, j] = A_shared[b, i, j] + D_shared[j]
                A_shared[b, i, j] = A_shared[b, i, j] * T.float32(2.0)
                C_shared[b, i, j] = A_shared[b, i, j] * B_shared[b, i, j]
            T.copy(C_shared, C[bz * block_B : (bz + 1) * block_B, by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])

    return main


# 1D buffer_store
def test_1d_tiles(M, block_M, tile_size, index_map, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
        A: T.Tensor((M,), dtype),
        B: T.Tensor((M,), dtype),
        C: T.Tensor((M,), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=1) as (by,):
            A_shared = T.alloc_shared((block_M,), dtype)
            B_shared = T.alloc_shared((block_M,), dtype)
            C_shared = T.alloc_shared((block_M,), accum_dtype)

            T.clear(C_shared)

            T.copy(A[by * block_M : (by + 1) * block_M], A_shared)
            T.copy(B[by * block_M : (by + 1) * block_M], B_shared)
            for i in T.Tiles(A_shared, parallel=True):
                C_shared[i] = A_shared[i] * B_shared[i]

            T.copy(C_shared, C[by * block_M : (by + 1) * block_M])

    return main


# tail tiles kernel

# 64Byte align tile.slice

# reduce

B = 64
M = 512
N = 1024

# T.handle()

block_B = 2
block_M = 256
block_N = 128

tile_size = (4, 32)
index_map = (-2, -1)


# func = dot_mul_tiled_parallel_3D(B, M, N, block_B, block_M, block_N, tile_size, index_map, "bfloat16", "bfloat16")
# func = test_tiles_broadcast(B, M, N, block_B, block_M, block_N, tile_size, index_map, "bfloat16", "bfloat16")
func = test_1d_tiles(M, block_M, tile_size, index_map, "bfloat16", "bfloat16")
# func = dot_mul_tiled_parallel_3D(64, 510, 254, 32, 255, 127, tile_size, index_map, dtype="float16", accum_dtype="float16")
func.show()


import os

log_dir = os.path.join(os.path.dirname(__file__), "logs_tiles_sunmmio")

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
print("Done.")
print("---------------------------------------")


# def dot_mul_tiled_parallel_3D_expected(Batch, M, N, block_B, block_M, block_N, dtype="float16", accum_dtype="float"):

#     @T.prim_func
#     def main(
#             A: T.Tensor((Batch, M, N), dtype),
#             B: T.Tensor((Batch, M, N), dtype),
#             C: T.Tensor((Batch, M, N), dtype),
#     ):
#         # Initialize Kernel Context
#         with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), T.ceildiv(Batch, block_B), threads=128) as (bx, by, bz):
#             A_shared = T.alloc_shared((block_B, block_M, block_N), dtype)
#             B_shared = T.alloc_shared((block_B, block_M, block_N), dtype)
#             C_shared = T.alloc_fragment((block_B, block_M, block_N), accum_dtype)
#             # T.block_attr({"tile_view":{
#             #     A_shared.data:{"tile_size": (T.int32(32), T.int32(32)), "dim_map": (T.int32(0), T.int32(1)), "new_shape": (T.int32(6), T.int32(6))},
#             #     B_shared.data:{"tile_size": (T.int32(32), T.int32(32)), "dim_map": (T.int32(0), T.int32(1)), "new_shape": (T.int32(6), T.int32(6))},
#             #     C_shared.data:{"tile_size": (T.int32(32), T.int32(32)), "dim_map": (T.int32(0), T.int32(1)), "new_shape": (T.int32(6), T.int32(6))}
#             # }
#             # })

#             T.clear(C_shared)

#             T.copy(A[bz * block_B:(bz + 1) * block_B, by * block_M:(by + 1) * block_M, bx * block_N:(bx + 1) * block_N], A_shared)
#             T.copy(B[bz * block_B:(bz + 1) * block_B, by * block_M:(by + 1) * block_M, bx * block_N:(bx + 1) * block_N], B_shared)

#             # for b, i, j in T.Tiles(A_shared, parallel=True):
#             #     C_shared[b, i, j] = A_shared[b, i, j] * B_shared[b, i, j]
#             for b in T.serial(A_shared.shape[0]):
#                 for i in T.serial(A_shared.shape[1] // tile_size[0]):
#                     for j in T.serial(A_shared.shape[2] // tile_size[1]):
#                         for ki in T.serial(tile_size[0]):
#                             for kj in T.vectorized(tile_size[1]):
#                                 C_shared[b, i * tile_size[0] + ki, j * tile_size[1] + kj] = \
#                                     A_shared[b, i * tile_size[0] + ki, j * tile_size[1] + kj] * \
#                                     B_shared[b, i * tile_size[0] + ki, j * tile_size[1] + kj]


#             # for i, j in T.Tiles((T.ceildiv(block_M, 32), T.ceildiv(block_N, 32)), parallel=True):
#             #     C_shared[i, j] = A_shared[i, j] * B_shared[i, j]

#             # for i, j in T.Tiles((T.ceildiv(block_M, 32), T.ceildiv(block_N, 32)), tile_size=(32, 32), dim_map=(0, 1), parallel=True):
#             #     C_shared[i, j] = A_shared[i, j] * B_shared[i, j]
#             T.copy(C_shared, C[bz * block_B:(bz + 1) * block_B, by * block_M:(by + 1) * block_M, bx * block_N:(bx + 1) * block_N])

#     return main
# ref_primfunc = dot_mul_tiled_parallel_3D_expected(B, M, N, block_B, block_M, block_N,"float16","float16")
# ref_mod = IRModule.from_expr(ref_primfunc.with_attr("global_symbol", "main"))
# ir.assert_structural_equal(mod["main"], ref_mod["main"])

#     C_shared[i, j] = A_shared[i, j] * B_shared[i, j]

# for i, j in T.Tiles(A_shared, parallel=True):
#     C_shared[Ramp(i...), Ramp(j...)] = A_shared[Ramp(i...), Ramp(j...)] * B_shared[Ramp(i...), Ramp(j...)]

# for i in serial(0, new_shape[0]):
#     for j in serial(0, new_shape[1]):
#         for k in serial(i*tile_size[0], (i+1)*tile_size[0]):
#             for l in vectorized(j*tile_size[1], (j+1)*tile_size[1]):
#                 C_shared[k, l] = A_shared[k, l] * B_shared[k, l]


# for i in serial(0, new_shape[0]):
#     for j in serial(0, new_shape[1]):
#         for k in range(tile_size[0]): # serial
#             for l in vectorized(tile_size[1]): # vectorized
#                 C_shared[i*tile_size[0]+k, j*tile_size[1]+l] = A_shared[i*tile_size[0]+k, j*tile_size[1]+l] * B_shared[i*tile_size[0]+k, j*tile_size[1]+l]


# for b, i, j in T.Tiles(A_shared, parallel=True):
#     C_shared[b, i, j] = A_shared[b, i, j] * B_shared[b, i, j]

# for b in serial(new_shape[0]):
#     for i in serial(new_shape[1]):
#         for j in serial(new_shape[2]):
#             for k in serial(tile_size[0]):
#                 for l in vectorized(tile_size[1]):
#                     C_shared[b, i*tile_size[0]+k, j*tile_size[1]+l] = A_shared[b, i*tile_size[0]+k, j*tile_size[1]+l] * B_shared[b, i*tile_size[0]+k, j*tile_size[1]+l]
