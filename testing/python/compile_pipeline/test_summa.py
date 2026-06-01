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
            for bx, by in T.Persistent(
                [T.ceildiv(sharded_N, block_N), T.ceildiv(sharded_M, block_M)],
                ncores,
                _cid,
            ):
                T.clear(C_local)

                # Number of iterations in K dimension.
                K_steps = T.ceildiv(sharded_K, block_K)

                # Core loop of SUMMA algorithm
                for k_tile in range(K_steps):
                    # --- Step 1: Broadcast row block of matrix A ---
                    # Broadcast directly from DRAM to asram of each core
                    # Source core coordinate is (by, k_tile), which is responsible for reading from DRAM and broadcasting to all cores in the same row
                    T.comm.broadcast(
                        A[
                            by * block_M : by * block_M + block_M,
                            k_tile * block_K : k_tile * block_K + block_K,
                        ],
                        A_shared,
                        (0, 0),
                        direction="h",
                    )

                    # --- Step 2: Broadcast column block of matrix B ---
                    # Broadcast directly from DRAM to wsram of each core
                    # Source core coordinate is (k_tile, bx), which is responsible for reading from DRAM and broadcasting to all cores in the same column
                    T.comm.broadcast(
                        B[
                            k_tile * block_K : k_tile * block_K + block_K,
                            bx * block_N : bx * block_N + block_N,
                        ],
                        B_shared,
                        (0, 0),
                        direction="v",
                    )

                    # --- Step 3: Local computation ---
                    # Each core performs local GEMM using broadcasted A_shared and B_shared
                    T.gemm(A_shared, B_shared, C_local)

                # After the loop ends, write local computation result back to DRAM
                T.copy(C_local, C[by * block_M, bx * block_N])

    return kernel


def test_summa():
    func = summa_matmul(128, 128, 128, 32, 32, 32)

    script_device_mode = """
    @T.prim_func
    def kernel_kernel(A: T.handle("float16", "global"), B: T.handle("float16", "global"), C: T.handle("float32", "global")) -> T.int32:
        T.func_attr({"target": T.target({"keys": ["cpu"], "kind": "llvm", "mattr": ["device_mesh_nrow_4", "device_mesh_ncol_4"], "mcpu": "sunmmio-a4e", "tag": ""}), "thread_extent": {"blockIdx.x": 4, "blockIdx.y": 4, "threadIdx.x": 128, "threadIdx.y": 1, "threadIdx.z": 1}, "tir.is_global_func": T.bool(True), "tir.noalias": True, "tl.non_restrict_params": [], "tl.readonly_param_indices": [0, 1, 2]})
        with T.launch_thread("blockIdx.x", 4) as bx:
            C_local = T.allocate([1024], "float32", "shared.rsram")
            A_shared = T.allocate([1024], "float16", "shared.asram")
            B_shared = T.allocate([1024], "float16", "shared.wsram")
            T.barrier_init(T.int64(15))
            T.barrier_init(T.int64(4369))
            by = T.launch_thread("blockIdx.y", 4)
            tx = T.launch_thread("threadIdx.x", 128)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            C_local_1 = T.Buffer((1024,), data=C_local, scope="shared.rsram")
            C_local_1[tx * 8:tx * 8 + 8] = T.Broadcast(T.float32(0.0), 8)
            T.sync_null_token(2)
            for k_tile in range(4):
                T.wait_token(2)
                T.barrier_arrive_and_wait(T.int64(15))
                A_1 = T.Buffer((16384,), "float16", data=A)
                A_shared_1 = T.Buffer((1024,), "float16", data=A_shared, scope="shared.asram")
                T.broadcast_(T.region(A_1[by * 4096 + k_tile * 32], 1, 4000), T.region(A_shared_1[0], 2, 1024), 1024, by * 4 + k_tile, 0, T.sync_token_id(0))
                B_1 = T.Buffer((16384,), "float16", data=B)
                B_shared_1 = T.Buffer((1024,), "float16", data=B_shared, scope="shared.wsram")
                T.barrier_arrive_and_wait(T.int64(4369))
                T.broadcast_(T.region(B_1[k_tile * 4096 + bx * 32], 1, 4000), T.region(B_shared_1[0], 2, 1024), 1024, k_tile * 4 + bx, 1, T.sync_token_id(1))
                T.wait_token(0)
                T.barrier_arrive_and_wait(T.int64(15))
                T.wait_token(1)
                T.barrier_arrive_and_wait(T.int64(4369))
                T.mma_sunmmio(T.region(A_shared_1[0], 1, 1024), T.region(B_shared_1[0], 1, 1024), T.region(C_local_1[0], 3, 1024), T.bool(False), T.bool(False), T.bool(False), T.sync_token_id(2))
            T.wait_token(2)
            C_1 = T.Buffer((16384,), data=C)
            T.dma_copy(T.region(C_local_1[0], 1, 1024), T.region(C_1[by * 4096 + bx * 32], 2, 4000), T.sync_token_id(3))
            T.wait_token(3)
        return 0
    """

    script_lower_tile_op = [
        'A = T.match_buffer(A_handle, (32, 32), "float16", strides=(32, 1))',
        'B = T.match_buffer(B_handle, (32, 32), "float16", strides=(32, 1))',
        "C = T.match_buffer(C_handle, (32, 32), strides=(32, 1))",
        'bx = T.launch_thread("blockIdx.x", 16)',
        "for w in range(1):",
        "T.broadcast_(T.region(A_rsram_stage[0, 0], 1, 32, 32), T.region(A_shared[0, 0], 2, 32, 32), 0, T.int64(15), 0, 0)",
        "T.broadcast_(T.region(B[0, 0], 1, 32, 32), T.region(B_shared[0, 0], 2, 32, 32), 1, T.int64(4369), 0, 0)",
    ]

    script_InjectSunmmioSync = [
        'with T.launch_thread("blockIdx.x", 16) as bx:',
        "T.dma_copy(T.region(A_1[bx * 32, 0], 1, 32, 32), T.region(A_rsram_stage[0, 0], 2, 32, 32), 0, T.sync_token_id(0))",
        "T.barrier_init(T.int64(15))",
        "T.barrier_init(T.int64(4369))",
        "T.barrier_arrive_and_wait(T.int64(15))",
        "T.broadcast_(T.region(A_rsram_stage[0, 0], 1, 32, 32), T.region(A_shared[0, 0], 2, 32, 32), 0, 15, 0, 0, T.sync_token_id(1))",
        "T.barrier_arrive_and_wait(T.int64(4369))",
        "T.broadcast_(T.region(B_1[0, 0], 1, 32, 32), T.region(B_shared[0, 0], 2, 32, 32), 1, 4369, 0, 0, T.sync_token_id(2))",
        "T.barrier_arrive_and_wait(T.int64(15))",
        "T.barrier_arrive_and_wait(T.int64(4369))",
        "T.mma_sunmmio(T.region(A_shared[0, 0], 1, 32, 32), T.region(B_shared[0, 0], 1, 32, 32), T.region(C_local[0, 0], 3, 32, 32), T.bool(False), T.bool(False), T.bool(False), 0, T.sync_token_id(3))",
        "T.dma_copy(T.region(C_local[0, 0], 1, 32, 32), T.region(C_1[bx * 32, 0], 2, 32, 32), 0, T.sync_token_id(4))",
        "T.wait_token(4)",
    ]

    test_config = {
        "LowerTileOp": {
            "script_expected": script_lower_tile_op,
        },
        "InjectSunmmioSync": {
            "script_expected": script_InjectSunmmioSync,
        },
        "DeviceMode": {
            "script_expected": script_device_mode,
        },
    }
    test_config = get_or_add_default_verify(func, test_config)
    compile_test(func, target="Sunmmio", test_config=test_config)


if __name__ == "__main__":
    test_summa()
