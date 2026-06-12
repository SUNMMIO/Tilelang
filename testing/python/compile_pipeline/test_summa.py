import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.layout import make_zz_layout

from compile_pipeline import compile_test, target
from formal_verify_funcs import *


@target("Sunmmio")
def summa_matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    """
    SUMMA (Scalable Universal Matrix Multiplication Algorithm)
    for a 4x4 mesh.

    Grid size: (N/block_N, M/block_M) = (4, 4)
    """
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
        A: T.MeshTensor(A_shape, shard_policy, device_mesh_config, dtype, layout=A_layout),
        B: T.MeshTensor(B_shape, shard_policy, device_mesh_config, dtype, layout=B_layout),
        C: T.MeshTensor(C_shape, shard_policy, device_mesh_config, accum_dtype, layout=C_layout),
    ):
        # Assume the current is a 4x4 processor grid (Mesh)
        # Each core is responsible for outputting a 32x32 block of matrix C
        with T.Kernel(ncores) as _cid:
            sharded_M, sharded_K = A.shape
            _, sharded_N = B.shape

            # Allocate local SRAM cache
            # A_shared is placed in ASRAM (usually used for A matrix cache)
            # B_shared is placed in WSRAM (usually used for B matrix cache)
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)

            # Local accumulator, placed in RSRAM
            C_local = T.alloc_shared((block_M, block_N), accum_dtype)
            for bx in T.serial(T.ceildiv(sharded_M, block_M)):
                for by in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.clear(C_local)

                    # Number of iterations in K dimension.
                    K_steps = T.ceildiv(sharded_K, block_K)

                    # Core loop of SUMMA algorithm
                    for k_tile in range(K_steps):
                        # --- Step 1: Broadcast row block of matrix A ---
                        # Broadcast directly from DRAM to asram of each core
                        T.comm.broadcast(
                            A[
                                bx * block_M : bx * block_M + block_M,
                                k_tile * block_K : k_tile * block_K + block_K,
                            ],
                            A_shared,
                            (0, 0),
                            direction="h",
                        )

                        # --- Step 2: Broadcast column block of matrix B ---
                        # Broadcast directly from DRAM to wsram of each core
                        T.comm.broadcast(
                            B[
                                k_tile * block_K : k_tile * block_K + block_K,
                                by * block_N : by * block_N + block_N,
                            ],
                            B_shared,
                            (0, 0),
                            direction="v",
                        )

                        # --- Step 3: Local computation ---
                        # Each core performs local GEMM using broadcasted A_shared and B_shared
                        T.gemm(A_shared, B_shared, C_local)

                    # After the loop ends, write local computation result back to DRAM
                    T.copy(C_local, C[bx * block_M, by * block_N])

    return kernel


def test_summa():
    func = summa_matmul(128, 128, 128, 32, 32, 32)

    script_device_mode = [
        '"thread_extent": {"blockIdx.x": 16}',
        'with T.launch_thread("blockIdx.x", 16) as bx:',
        "T.dma_copy(T.region(A_1[0, 0], 1, 32, 32), T.region(A_rsram_stage[0, 0], 2, 32, 32), 0, T.sync_token_id(0))",
        "T.broadcast_(T.region(B_1[0, 0], 1, 32, 32), T.region(B_shared[0, 0], 2, 32, 32), 1, 15, 0, 0, T.sync_token_id(2))",
        "T.dma_copy(T.region(C_local[0, 0], 1, 32, 32), T.region(C_1[0, 0], 2, 32, 32), 0, T.sync_token_id(4))",
    ]

    script_lower_tile_op = [
        'A = T.match_buffer(A_handle, (32, 32), "float16", strides=(32, 1))',
        'B = T.match_buffer(B_handle, (32, 32), "float16", strides=(32, 1))',
        "C = T.match_buffer(C_handle, (32, 32), strides=(32, 1))",
        'bx = T.launch_thread("blockIdx.x", 16)',
        "for bx_1, by in T.grid(1, 1):",
        "T.dma_copy(T.region(A[0, 0], 1, 32, 32), T.region(A_rsram_stage[0, 0], 2, 32, 32), 0)",
        "T.broadcast_(T.region(A_rsram_stage[0, 0], 1, 32, 32), T.region(A_shared[0, 0], 2, 32, 32), 0, T.int64(15), 0, 0)",
        "T.broadcast_(T.region(B[0, 0], 1, 32, 32), T.region(B_shared[0, 0], 2, 32, 32), 1, T.int64(15), 0, 0)",
        "T.dma_copy(T.region(C_local[0, 0], 1, 32, 32), T.region(C[0, 0], 2, 32, 32), 0)",
    ]

    script_InjectSunmmioSync = [
        'with T.launch_thread("blockIdx.x", 16) as bx:',
        "T.dma_copy(T.region(A_1[0, 0], 1, 32, 32), T.region(A_rsram_stage[0, 0], 2, 32, 32), 0, T.sync_token_id(0))",
        "T.wait_token(0)",
        "T.barrier_init(T.int64(15))",
        "T.barrier_init(T.int64(4369))",
        "T.barrier_arrive_and_wait(T.int64(15))",
        "T.broadcast_(T.region(A_rsram_stage[0, 0], 1, 32, 32), T.region(A_shared[0, 0], 2, 32, 32), 0, 15, 0, 0, T.sync_token_id(1))",
        "T.barrier_arrive_and_wait(T.int64(4369))",
        "T.broadcast_(T.region(B_1[0, 0], 1, 32, 32), T.region(B_shared[0, 0], 2, 32, 32), 1, 15, 0, 0, T.sync_token_id(2))",
        "T.wait_token(1)",
        "T.barrier_arrive_and_wait(T.int64(15))",
        "T.wait_token(2)",
        "T.barrier_arrive_and_wait(T.int64(4369))",
        "T.mma_sunmmio(T.region(A_shared[0, 0], 1, 32, 32), T.region(B_shared[0, 0], 1, 32, 32), T.region(C_local[0, 0], 3, 32, 32), T.bool(False), T.bool(False), T.bool(False), 0, T.sync_token_id(3))",
        "T.wait_token(3)",
        "T.dma_copy(T.region(C_local[0, 0], 1, 32, 32), T.region(C_1[0, 0], 2, 32, 32), 0, T.sync_token_id(4))",
        "T.wait_token(4)",
    ]

    test_config = {
        "LowerTileOp": {
            "script_expected": script_lower_tile_op,
        },
        "InjectSunmmioSync": {
            "script_expected": script_InjectSunmmioSync,
        },
        "DeviceMod": {
            "script_expected": script_device_mode,
        },
    }
    test_config = get_or_add_default_verify(func, test_config)
    compile_test(func, target="Sunmmio", test_config=test_config)


if __name__ == "__main__":
    test_summa()
