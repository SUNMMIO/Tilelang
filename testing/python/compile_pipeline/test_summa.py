import tilelang.language as T
from compile_pipeline import compile_test
from formal_verify_funcs import *


def summa_matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    """
    SUMMA (Scalable Universal Matrix Multiplication Algorithm)
    for a 4x4 mesh.

    Grid size: (N/block_N, M/block_M) = (4, 4)
    """

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        # Assume the current is a 4x4 processor grid (Mesh)
        # Each core is responsible for outputting a 32x32 block of matrix C
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            # Allocate local SRAM cache
            # A_shared is placed in ASRAM (usually used for A matrix cache)
            # B_shared is placed in WSRAM (usually used for B matrix cache)
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared.asram")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared.wsram")

            # Local accumulator, placed in RSRAM
            C_local = T.alloc_shared((block_M, block_N), accum_dtype, scope="shared.rsram")
            T.clear(C_local)

            # Number of iterations in K dimension (for 128/32 = 4 steps)
            K_steps = T.ceildiv(K, block_K)

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
                    (by, k_tile),
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
                    (k_tile, bx),
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

    script_lower_tile_op = """
            with T.block("tilelang_root"):
                T.reads(A[by * 32:by * 32 + 32, 0:128], B[0:128, bx * 32:bx * 32 + 32], C[by * 32, bx * 32])
                T.writes()
                T.block_attr({"layout_map": {C_local: metadata["tl.Layout"][0], A_shared: metadata["tl.Layout"][1], B_shared: metadata["tl.Layout"][2]}})
                A_shared = T.alloc_buffer((32, 32), "float16", data=A_shared.data, scope="shared.asram")
                B_shared = T.alloc_buffer((32, 32), "float16", data=B_shared.data, scope="shared.wsram")
                C_local = T.alloc_buffer((32, 32), data=C_local.data, scope="shared.rsram")
                for i0 in T.serial(32, annotations={"tile.domain": [32, 32], "tile.loop_parallel": 1, "tile.loop_stage": 0}):
                    for i1 in T.serial(32, annotations={"tile.loop_parallel": 1, "tile.loop_stage": 0}):
                        C_local[i0, i1] = T.Cast("float32", 0)
                for k_tile in range(4):
                    T.broadcast_(T.region(A[by * 32, k_tile * 32], 1, 32, 32), T.region(A_shared[0, 0], 2, 32, 32), 1024, by * 4 + k_tile, 0)
                    T.broadcast_(T.region(B[k_tile * 32, bx * 32], 1, 32, 32), T.region(B_shared[0, 0], 2, 32, 32), 1024, k_tile * 4 + bx, 1)
                    with T.block("_gemm_sss"):
                        T.reads()
                        T.writes()
                        T.mma_sunmmio(T.region(A_shared[0, 0], 1, 32, 32), T.region(B_shared[0, 0], 1, 32, 32), T.region(C_local[0, 0], 3, 32, 32), T.bool(False), T.bool(False), T.bool(False))
                T.dma_copy(T.region(C_local[0, 0], 1, 32, 32), T.region(C[by * 32, bx * 32], 2, 32, 32))
    """

    script_InjectSunmmioSync = """
            with T.decl_buffer((32, 32), scope="shared.rsram") as C_local:
                for i0 in T.serial(8, annotations={"tile.domain": [32, 32], "tile.execution_axis": 0, "tile.execution_domain_axes": [0, 1], "tile.scope_entry": 1, "tile.tile_size": [4, 32]}):
                    for i1 in T.serial(1, annotations={"tile.execution_axis": 1}):
                        for ki in T.serial(4, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                            for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                C_local[i0 * 4 + ki, kj] = T.float32(0.0)
                T.sync_null_token(2)
                for k_tile in range(4):
                    A_shared = T.decl_buffer((32, 32), "float16", scope="shared.asram")
                    B_shared = T.decl_buffer((32, 32), "float16", scope="shared.wsram")
                    T.wait_token(2)
                    A_2 = T.Buffer((128, 128), "float16", data=A, strides=(128, 1))
                    T.broadcast_(T.region(A_2[by * 32, k_tile * 32], 1, 32, 32), T.region(A_shared[0, 0], 2, 32, 32), 1024, by * 4 + k_tile, 0, T.sync_token_id(0))
                    T.barrier_init(0, by * 4 + k_tile, k_tile // 4 * 4 + by * 4, k_tile // 4 * 4 + by * 4 + 1, k_tile // 4 * 4 + by * 4 + 2, k_tile // 4 * 4 + by * 4 + 3)
                    B_2 = T.Buffer((128, 128), "float16", data=B, strides=(128, 1))
                    T.broadcast_(T.region(B_2[k_tile * 32, bx * 32], 1, 32, 32), T.region(B_shared[0, 0], 2, 32, 32), 1024, k_tile * 4 + bx, 1, T.sync_token_id(1))
                    T.barrier_init(1, k_tile * 4 + bx, bx % 4, bx % 4 + 4, bx % 4 + 8, bx % 4 + 12)
                    T.wait_token(0)
                    T.barrier_arrive_and_wait(0)
                    T.wait_token(1)
                    T.barrier_arrive_and_wait(1)
                    T.mma_sunmmio(T.region(A_shared[0, 0], 1, 32, 32), T.region(B_shared[0, 0], 1, 32, 32), T.region(C_local[0, 0], 3, 32, 32), T.bool(False), T.bool(False), T.bool(False), T.sync_token_id(2))
                T.wait_token(2)
                C_2 = T.Buffer((128, 128), data=C, strides=(128, 1))
                T.dma_copy(T.region(C_local[0, 0], 1, 32, 32), T.region(C_2[by * 32, bx * 32], 2, 32, 32), T.sync_token_id(3))
            T.wait_token(3)
    """

    script_device_mode = """
    @T.prim_func
    def kernel_kernel(A: T.handle("float16", "global"), B: T.handle("float16", "global"), C: T.handle("float32", "global")) -> T.int32:
        T.func_attr({"target": T.target({"keys": ["cpu"], "kind": "llvm", "mattr": ["device_mesh_nrow_4", "device_mesh_ncol_4"], "mcpu": "sunmmio-a4e", "tag": ""}), "thread_extent": {"blockIdx.x": 4, "blockIdx.y": 4, "threadIdx.x": 128, "threadIdx.y": 1, "threadIdx.z": 1}, "tir.is_global_func": T.bool(True), "tir.noalias": True, "tl.non_restrict_params": [], "tl.readonly_param_indices": [0, 1, 2]})
        with T.launch_thread("blockIdx.x", 4) as bx:
            C_local = T.allocate([1024], "float32", "shared.rsram")
            A_shared = T.allocate([1024], "float16", "shared.asram")
            B_shared = T.allocate([1024], "float16", "shared.wsram")
            by = T.launch_thread("blockIdx.y", 4)
            tx = T.launch_thread("threadIdx.x", 128)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            C_local_1 = T.Buffer((1024,), data=C_local, scope="shared.rsram")
            C_local_1[tx * 8:tx * 8 + 8] = T.Broadcast(T.float32(0.0), 8)
            T.sync_null_token(2)
            for k_tile in range(4):
                T.wait_token(2)
                A_1 = T.Buffer((16384,), "float16", data=A)
                A_shared_1 = T.Buffer((1024,), "float16", data=A_shared, scope="shared.asram")
                T.broadcast_(T.region(A_1[by * 4096 + k_tile * 32], 1, 4000), T.region(A_shared_1[0], 2, 1024), 1024, by * 4 + k_tile, 0, T.sync_token_id(0))
                T.barrier_init(0, by * 4 + k_tile, by * 4, by * 4 + 1, by * 4 + 2, by * 4 + 3)
                B_1 = T.Buffer((16384,), "float16", data=B)
                B_shared_1 = T.Buffer((1024,), "float16", data=B_shared, scope="shared.wsram")
                T.broadcast_(T.region(B_1[k_tile * 4096 + bx * 32], 1, 4000), T.region(B_shared_1[0], 2, 1024), 1024, k_tile * 4 + bx, 1, T.sync_token_id(1))
                T.barrier_init(1, k_tile * 4 + bx, bx, bx + 4, bx + 8, bx + 12)
                T.wait_token(0)
                T.barrier_arrive_and_wait(0)
                T.wait_token(1)
                T.barrier_arrive_and_wait(1)
                T.mma_sunmmio(T.region(A_shared_1[0], 1, 1024), T.region(B_shared_1[0], 1, 1024), T.region(C_local_1[0], 3, 1024), T.bool(False), T.bool(False), T.bool(False), T.sync_token_id(2))
            T.wait_token(2)
            C_1 = T.Buffer((16384,), data=C)
            T.dma_copy(T.region(C_local_1[0], 1, 1024), T.region(C_1[by * 4096 + bx * 32], 2, 4000), T.sync_token_id(3))
            T.wait_token(3)
        return 0
    """

    def get_verify_merge_allocate():
        kernel_name = "kernel_kernel"
        # Only one buffer per scope, no need to verify merge size
        return build_verify_merge_allocate(kernel_name=kernel_name)

    test_config = {
        "LowerTileOp": {
            "script_expected": script_lower_tile_op,
        },
        "InjectSunmmioSync": {
            "script_expected": script_InjectSunmmioSync,
        },
        "MergeSharedMemoryAllocationsSunmmio": {
            "formal_verify": get_verify_merge_allocate(),
        },
        "DeviceMode": {
            "script_expected": script_device_mode,
        },
    }
    test_config = get_or_add_default_verify(func, test_config)
    compile_test(func, target="Sunmmio", test_config=test_config)


if __name__ == "__main__":
    test_summa()
