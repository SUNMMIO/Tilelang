import tilelang.language as T
from compile_pipeline import compile_test
from formal_verify_funcs import *


def kernel_overall(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    mesh_device_config = (4, 4)

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.MeshShardingPolicy(x=1, y=0), mesh_device_config, dtype),
        B: T.MeshTensor((K, N), T.MeshShardingPolicy(x=1, y=0), mesh_device_config, dtype),
        Bias: T.MeshTensor((M, N), T.MeshShardingPolicy(x=1, y=0), mesh_device_config, accum_dtype),
        C: T.MeshTensor((M, N), T.MeshShardingPolicy(x=1, y=0), mesh_device_config, accum_dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            # [wanghz18] Automatic SRAM Scope Inference
            # We declare generic 'shared' scope, expecting InferSramScope pass to
            # refine them to 'shared.asram', 'shared.wsram', 'shared.rsram'
            A_shared = T.alloc_shared((block_M, block_K), dtype=dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype=dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            Bias_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.clear(C_shared)  # Avoid Fill op unsupported scope error

            # [wanghz18] GEMM Lowering to mma_sunmmio intrinsic
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)

            # Load Bias
            T.copy(Bias[by * block_M, bx * block_N], Bias_shared)

            # [weizzh] Tiles Loop for Element-wise operation
            # This loop should be legalized and vectorized by LegalizeTilesLoop/TilesLoop passes
            for i, j in T.Tiles(C_shared, parallel=True):
                C_shared[i, j] = C_shared[i, j] + Bias_shared[i, j]

            # [xiaoyao-NKU] Inter-core Communication (Broadcast)
            C_remote = T.alloc_shared((block_M, block_N), accum_dtype)
            T.comm.broadcast(C_shared, C_remote, (0, 0), direction="h")

            # Store result
            T.copy(C_remote, C[by * block_M, bx * block_N])

    return main


def test_overall():
    func = kernel_overall(128, 128, 128, 64, 64, 32)
    script_lower_tile_op = """
            with T.block("tilelang_root"):
                T.reads(A[by * 64, 0:97], B[0:97, bx * 64], Bias[by * 64, bx * 64], C[by * 64, bx * 64])
                T.writes()
                T.block_attr({"global_layout_map": {A: metadata["tl.Layout"][0], B: metadata["tl.Layout"][1], Bias: metadata["tl.Layout"][2], C: metadata["tl.Layout"][3]}, "layout_map": {A_shared: metadata["tl.Layout"][4], B_shared: metadata["tl.Layout"][5], Bias_shared: metadata["tl.Layout"][6], C_shared: metadata["tl.Layout"][7], C_remote: metadata["tl.Layout"][8]}})
                A_shared = T.alloc_buffer((64, 32), "float16", data=A_shared.data, scope="shared.asram")
                B_shared = T.alloc_buffer((32, 64), "float16", data=B_shared.data, scope="shared.wsram")
                C_shared = T.alloc_buffer((64, 64), data=C_shared.data, scope="shared.rsram")
                Bias_shared = T.alloc_buffer((64, 64), data=Bias_shared.data, scope="shared.rsram")
                C_remote = T.alloc_buffer((64, 64), data=C_remote.data, scope="shared.rsram")
                for i0 in T.serial(64, annotations={"tile.domain": [64, 64], "tile.loop_parallel": 1, "tile.loop_stage": 0}):
                    for i1 in T.serial(64, annotations={"tile.loop_parallel": 1, "tile.loop_stage": 0}):
                        C_shared[i0, i1] = T.Cast("float32", 0)
                for k in T.serial(4, annotations={"num_stages": 2}):
                    T.dma_copy(T.region(A[by * 64, k * 32], 1, 64, 32), T.region(A_shared[0, 0], 2, 64, 32))
                    T.dma_copy(T.region(B[k * 32, bx * 64], 1, 32, 64), T.region(B_shared[0, 0], 2, 32, 64))
                    with T.block("_gemm_sss"):
                        T.reads()
                        T.writes()
                        T.mma_sunmmio(T.region(A_shared[0, 0], 1, 64, 32), T.region(B_shared[0, 0], 1, 32, 64), T.region(C_shared[0, 0], 3, 64, 64), T.bool(False), T.bool(False), T.bool(False))
                T.dma_copy(T.region(Bias[by * 64, bx * 64], 1, 64, 64), T.region(Bias_shared[0, 0], 2, 64, 64))
                for i in T.serial(64, annotations={"tile.domain": [64, 64], "tile.loop_parallel": 1, "tile.loop_stage": 0}):
                    for j in T.serial(64, annotations={"tile.loop_parallel": 1, "tile.loop_stage": 0}):
                        C_shared[i, j] = C_shared[i, j] + Bias_shared[i, j]
                T.broadcast_(T.region(C_shared[0, 0], 1, 64, 64), T.region(C_remote[0, 0], 2, 64, 64), 4096, 0, 0)
                T.dma_copy(T.region(C_remote[0, 0], 1, 64, 64), T.region(C[by * 64, bx * 64], 2, 64, 64))
    """
    script_InjectSunmmioSync = """
        with T.launch_thread("blockIdx.x", 2) as bx:
            by = T.launch_thread("blockIdx.y", 2)
            tx = T.launch_thread("threadIdx.x", 128)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            with T.decl_buffer((2, 64, 32), "float16", scope="shared.asram") as A_shared:
                B_shared = T.decl_buffer((2, 32, 64), "float16", scope="shared.wsram")
                C_shared = T.decl_buffer((64, 64), scope="shared.rsram")
                Bias_shared = T.decl_buffer((64, 64), scope="shared.rsram")
                C_remote = T.decl_buffer((64, 64), scope="shared.rsram")
                for i0 in T.serial(16, annotations={"tile.domain": [64, 64], "tile.execution_axis": 0, "tile.execution_domain_axes": [0, 1], "tile.scope_entry": 1, "tile.tile_size": [4, 32]}):
                    for i1 in T.serial(2, annotations={"tile.execution_axis": 1}):
                        for ki in T.serial(4, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                            for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                C_shared[i0 * 4 + ki, i1 * 32 + kj] = T.float32(0.0)
                A_2 = T.Buffer((32, 32), "float16", data=A, strides=(32, 1))
                T.dma_copy(T.region(A_2[by * 64, 0], 1, 64, 32), T.region(A_shared[0, 0, 0], 2, 1, 64, 32), T.sync_token_id(0))
                B_2 = T.Buffer((32, 32), "float16", data=B, strides=(32, 1))
                T.dma_copy(T.region(B_2[0, bx * 64], 1, 32, 64), T.region(B_shared[0, 0, 0], 2, 1, 32, 64), T.sync_token_id(1))
                T.wait_token(0)
                T.wait_token(1)
                T.mma_sunmmio(T.region(A_shared[0, 0, 0], 1, 1, 64, 32), T.region(B_shared[0, 0, 0], 1, 1, 32, 64), T.region(C_shared[0, 0], 3, 64, 64), T.bool(False), T.bool(False), T.bool(False), T.sync_token_id(2))
                T.dma_copy(T.region(A_2[by * 64, 32], 1, 64, 32), T.region(A_shared[1, 0, 0], 2, 1, 64, 32), T.sync_token_id(3))
                T.dma_copy(T.region(B_2[32, bx * 64], 1, 32, 64), T.region(B_shared[1, 0, 0], 2, 1, 32, 64), T.sync_token_id(4))
                T.wait_token(3)
                T.wait_token(4)
                T.wait_token(2)
                T.mma_sunmmio(T.region(A_shared[1, 0, 0], 1, 1, 64, 32), T.region(B_shared[1, 0, 0], 1, 1, 32, 64), T.region(C_shared[0, 0], 3, 64, 64), T.bool(False), T.bool(False), T.bool(False), T.sync_token_id(5))
                T.dma_copy(T.region(A_2[by * 64, 64], 1, 64, 32), T.region(A_shared[0, 0, 0], 2, 1, 64, 32), T.sync_token_id(6))
                T.dma_copy(T.region(B_2[64, bx * 64], 1, 32, 64), T.region(B_shared[0, 0, 0], 2, 1, 32, 64), T.sync_token_id(7))
                T.wait_token(6)
                T.wait_token(7)
                T.wait_token(5)
                T.mma_sunmmio(T.region(A_shared[0, 0, 0], 1, 1, 64, 32), T.region(B_shared[0, 0, 0], 1, 1, 32, 64), T.region(C_shared[0, 0], 3, 64, 64), T.bool(False), T.bool(False), T.bool(False), T.sync_token_id(8))
                T.dma_copy(T.region(A_2[by * 64, 96], 1, 64, 32), T.region(A_shared[1, 0, 0], 2, 1, 64, 32), T.sync_token_id(9))
                T.dma_copy(T.region(B_2[96, bx * 64], 1, 32, 64), T.region(B_shared[1, 0, 0], 2, 1, 32, 64), T.sync_token_id(10))
                T.wait_token(9)
                T.wait_token(10)
                T.wait_token(8)
                T.mma_sunmmio(T.region(A_shared[1, 0, 0], 1, 1, 64, 32), T.region(B_shared[1, 0, 0], 1, 1, 32, 64), T.region(C_shared[0, 0], 3, 64, 64), T.bool(False), T.bool(False), T.bool(False), T.sync_token_id(11))
                Bias_2 = T.Buffer((32, 32), data=Bias, strides=(32, 1))
                T.dma_copy(T.region(Bias_2[by * 64, bx * 64], 1, 64, 64), T.region(Bias_shared[0, 0], 2, 64, 64), T.sync_token_id(12))
                for i in T.serial(16, annotations={"tile.domain": [64, 64], "tile.execution_axis": 0, "tile.execution_domain_axes": [0, 1], "tile.scope_entry": 1, "tile.tile_size": [4, 32]}):
                    for j in T.serial(2, annotations={"tile.execution_axis": 1}):
                        for ki in T.serial(4, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                            for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                T.wait_token(11)
                                T.wait_token(12)
                                C_shared[i * 4 + ki, j * 32 + kj] = C_shared[i * 4 + ki, j * 32 + kj] + Bias_shared[i * 4 + ki, j * 32 + kj]
                T.broadcast_(T.region(C_shared[0, 0], 1, 64, 64), T.region(C_remote[0, 0], 2, 64, 64), 4096, 0, 0, T.sync_token_id(13))
                T.barrier_init(0, 0, 1, 2, 3)
                T.wait_token(13)
                T.barrier_arrive_and_wait(0)
                C_2 = T.Buffer((32, 32), data=C, strides=(32, 1))
                T.dma_copy(T.region(C_remote[0, 0], 1, 64, 64), T.region(C_2[by * 64, bx * 64], 2, 64, 64), T.sync_token_id(14))
            T.wait_token(14)
    """
    script_device_mode = """
        with T.launch_thread("blockIdx.x", 2) as bx:
            buf_shmem = T.allocate([32768], "uint8", "shared.rsram")
            A_shared = T.allocate([4096], "float16", "shared.asram")
            B_shared = T.allocate([4096], "float16", "shared.wsram")
            by = T.launch_thread("blockIdx.y", 2)
            tx = T.launch_thread("threadIdx.x", 128)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            C_shared = T.Buffer((4096,), data=buf_shmem, scope="shared.rsram")
            for i0 in T.serial(16, annotations={"tile.domain": [64, 64], "tile.execution_axis": 0, "tile.execution_domain_axes": [0, 1], "tile.scope_entry": 1, "tile.tile_size": [4, 32]}):
                for i1 in T.serial(2, annotations={"tile.execution_axis": 1}):
                    for ki in T.serial(4, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                        C_shared[i0 * 256 + ki * 64 + i1 * 32:i0 * 256 + ki * 64 + i1 * 32 + 32] = T.Broadcast(T.float32(0.0), 32)
            A_1 = T.Buffer((1024,), "float16", data=A)
            A_shared_1 = T.Buffer((4096,), "float16", data=A_shared, scope="shared.asram")
            T.dma_copy(T.region(A_1[by * 2048], 1, 2048), T.region(A_shared_1[0], 2, 2048), T.sync_token_id(0))
            B_1 = T.Buffer((1024,), "float16", data=B)
            B_shared_1 = T.Buffer((4096,), "float16", data=B_shared, scope="shared.wsram")
            T.dma_copy(T.region(B_1[bx * 64], 1, 1056), T.region(B_shared_1[0], 2, 2048), T.sync_token_id(1))
            T.wait_token(0)
            T.wait_token(1)
            T.mma_sunmmio(T.region(A_shared_1[0], 1, 2048), T.region(B_shared_1[0], 1, 2048), T.region(C_shared[0], 3, 4096), T.bool(False), T.bool(False), T.bool(False), T.sync_token_id(2))
            T.dma_copy(T.region(A_1[by * 2048 + 32], 1, 2048), T.region(A_shared_1[2048], 2, 2048), T.sync_token_id(3))
            T.dma_copy(T.region(B_1[bx * 64 + 1024], 1, 1056), T.region(B_shared_1[2048], 2, 2048), T.sync_token_id(4))
            T.wait_token(3)
            T.wait_token(4)
            T.wait_token(2)
            T.mma_sunmmio(T.region(A_shared_1[2048], 1, 2048), T.region(B_shared_1[2048], 1, 2048), T.region(C_shared[0], 3, 4096), T.bool(False), T.bool(False), T.bool(False), T.sync_token_id(5))
            T.dma_copy(T.region(A_1[by * 2048 + 64], 1, 2048), T.region(A_shared_1[0], 2, 2048), T.sync_token_id(6))
            T.dma_copy(T.region(B_1[bx * 64 + 2048], 1, 1056), T.region(B_shared_1[0], 2, 2048), T.sync_token_id(7))
            T.wait_token(6)
            T.wait_token(7)
            T.wait_token(5)
            T.mma_sunmmio(T.region(A_shared_1[0], 1, 2048), T.region(B_shared_1[0], 1, 2048), T.region(C_shared[0], 3, 4096), T.bool(False), T.bool(False), T.bool(False), T.sync_token_id(8))
            T.dma_copy(T.region(A_1[by * 2048 + 96], 1, 2048), T.region(A_shared_1[2048], 2, 2048), T.sync_token_id(9))
            T.dma_copy(T.region(B_1[bx * 64 + 3072], 1, 1056), T.region(B_shared_1[2048], 2, 2048), T.sync_token_id(10))
            T.wait_token(9)
            T.wait_token(10)
            T.wait_token(8)
            T.mma_sunmmio(T.region(A_shared_1[2048], 1, 2048), T.region(B_shared_1[2048], 1, 2048), T.region(C_shared[0], 3, 4096), T.bool(False), T.bool(False), T.bool(False), T.sync_token_id(11))
            Bias_1 = T.Buffer((1024,), data=Bias)
            Bias_shared = T.Buffer((4096,), data=buf_shmem, scope="shared.rsram")
            T.dma_copy(T.region(Bias_1[by * 2048 + bx * 64], 1, 2080), T.region(Bias_shared[4096], 2, 4096), T.sync_token_id(12))
            for i in T.serial(16, annotations={"tile.domain": [64, 64], "tile.execution_axis": 0, "tile.execution_domain_axes": [0, 1], "tile.scope_entry": 1, "tile.tile_size": [4, 32]}):
                for j in T.serial(2, annotations={"tile.execution_axis": 1}):
                    for ki in T.serial(4, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                        T.wait_token(11)
                        T.wait_token(12)
                        C_shared[i * 256 + ki * 64 + j * 32:i * 256 + ki * 64 + j * 32 + 32] = C_shared[i * 256 + ki * 64 + j * 32:i * 256 + ki * 64 + j * 32 + 32] + Bias_shared[i * 256 + ki * 64 + j * 32 + 4096:i * 256 + ki * 64 + j * 32 + 4096 + 32]
            C_remote = T.Buffer((4096,), data=buf_shmem, scope="shared.rsram")
            T.broadcast_(T.region(C_shared[0], 1, 4096), T.region(C_remote[4096], 2, 4096), 4096, 0, 0, T.sync_token_id(13))
            T.barrier_init(0, 0, 1, 2, 3)
            T.wait_token(13)
            T.barrier_arrive_and_wait(0)
            C_1 = T.Buffer((1024,), data=C)
            T.dma_copy(T.region(C_remote[4096], 1, 4096), T.region(C_1[by * 2048 + bx * 64], 2, 2080), T.sync_token_id(14))
            T.wait_token(14)
    """

    def get_verify_merge_allocate():
        kernel_name = "main_kernel"
        #  65536  65536  100352
        block_m, block_n, block_k = 64, 64, 32
        cnt_a = block_m * block_k * 2
        cnt_w = block_k * block_n * 2
        # c_shared, bias_shared and c_remote are all on rsram, dtype = *4, bias and c_remote reuse, so only 2 buffer spaces are needed * 2
        cnt_r = block_m * block_n * 2 * 4
        return build_verify_merge_allocate(kernel_name=kernel_name, cnt_a=cnt_a, cnt_w=cnt_w, cnt_r=cnt_r)

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
    compile_test(func, out_idx=[2], target="Sunmmio", test_config=test_config)


if __name__ == "__main__":
    test_overall()
