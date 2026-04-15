import tilelang.language as T
from compile_pipeline import compile_test
from formal_verify_funcs import *


def kernel_sync(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((M, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=1) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.asram")
            B_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.wsram")
            C_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.rsram")
            D_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.rsram")
            E_shared = T.alloc_shared((1024, 1024), dtype, scope="shared.rsram")

            T.gemm(A_shared, B_shared, C_shared)
            if bx <= 2:
                T.clear(D_shared)

            for i in range(5):
                C_shared[i, 0] = C_shared[i, 0] + 1.0

            for _i in range(10):
                T.comm.broadcast(D_shared, E_shared, (0, 0), direction="h")
                E_shared[0, 0] = E_shared[0, 0] + 1.0
                T.comm.broadcast(E_shared, D_shared, (0, 0), direction="h")

    return kernel


def test_sync():
    func = kernel_sync(1024 * 16, 1024 * 16, 1024 * 16, 1024, 1024, 1024)

    script_lower_tile_op = """
                with T.block("_gemm_sss"):
                    T.reads()
                    T.writes()
                    T.mma_sunmmio(T.region(A_shared[0, 0], 1, 1024, 1024), T.region(B_shared[0, 0], 1, 1024, 1024), T.region(C_shared[0, 0], 3, 1024, 1024), T.bool(False), T.bool(False), T.bool(False))
                if bx <= 2:
                    for i0 in T.serial(1024, annotations={"tile.domain": [1024, 1024], "tile.loop_parallel": 1, "tile.loop_stage": 0}):
                        for i1 in T.serial(1024, annotations={"tile.loop_parallel": 1, "tile.loop_stage": 0}):
                            D_shared[i0, i1] = T.Cast("float16", 0)
                for i in range(5):
                    C_shared[i, 0] = T.Cast("float16", T.Cast("float32", C_shared[i, 0]) + T.float32(1.0))
                for _i in range(10):
                    T.broadcast_(T.region(D_shared[0, 0], 1, 1024, 1024), T.region(E_shared[0, 0], 2, 1024, 1024), 1048576, 0, 0)
                    E_shared[0, 0] = T.Cast("float16", T.Cast("float32", E_shared[0, 0]) + T.float32(1.0))
                    T.broadcast_(T.region(E_shared[0, 0], 1, 1024, 1024), T.region(D_shared[0, 0], 2, 1024, 1024), 1048576, 0, 0)
    """

    script_InjectSunmmioSync = """
                with T.decl_buffer((1024, 1024), "float16", scope="shared.asram") as A_shared:
                    B_shared = T.decl_buffer((1024, 1024), "float16", scope="shared.wsram")
                    T.mma_sunmmio(T.region(A_shared[0, 0], 1, 1024, 1024), T.region(B_shared[0, 0], 1, 1024, 1024), T.region(C_shared[0, 0], 3, 1024, 1024), T.bool(False), T.bool(False), T.bool(False), T.sync_token_id(0))
                if bx <= 2:
                    for i0 in T.serial(1024, annotations={"tile.domain": [1024, 1024], "tile.execution_axis": 0, "tile.execution_domain_axes": [0, 1], "tile.scope_entry": 1, "tile.tile_size": [1, 256]}):
                        for i1 in T.serial(4, annotations={"tile.execution_axis": 1}):
                            for ki in T.serial(1, annotations={"tile.interior": 1, "tile.interior_axis": 0}):
                                for kj in T.serial(4, annotations={"tile.interior": 1, "tile.interior_axis": 1}):
                                    for vec in T.vectorized(64):
                                        D_shared[i0, i1 * 256 + kj * 64 + vec] = T.float16(0.0)
                for i in range(5):
                    T.wait_token(0)
                    C_shared[i, 0] = T.Cast("float16", T.Cast("float32", C_shared[i, 0]) + T.float32(1.0))
                T.sync_null_token(2)
                T.barrier_init(1, 0, 1, 2, 3)
                for _i in range(10):
                    E_shared = T.decl_buffer((1024, 1024), "float16", scope="shared.rsram")
                    T.wait_token(2)
                    T.barrier_arrive_and_wait(1)
                    T.broadcast_(T.region(D_shared[0, 0], 1, 1024, 1024), T.region(E_shared[0, 0], 2, 1024, 1024), 1048576, 0, 0, T.sync_token_id(1))
                    T.barrier_init(0, 0, 1, 2, 3)
                    T.wait_token(1)
                    T.barrier_arrive_and_wait(0)
                    E_shared[0, 0] = T.Cast("float16", T.Cast("float32", E_shared[0, 0]) + T.float32(1.0))
                    T.broadcast_(T.region(E_shared[0, 0], 1, 1024, 1024), T.region(D_shared[0, 0], 2, 1024, 1024), 1048576, 0, 0, T.sync_token_id(2))
                    T.barrier_init(1, 0, 1, 2, 3)
            T.wait_token(2)
            T.barrier_arrive_and_wait(1)
    """

    script_device_mode = """
    @T.prim_func(private=True)
    def kernel_kernel() -> T.int32:
        T.func_attr({"target": T.target({"keys": ["cpu"], "kind": "llvm", "mattr": ["device_mesh_nrow_4", "device_mesh_ncol_4"], "mcpu": "sunmmio-a4e", "tag": ""}), "tir.is_global_func": True, "tir.noalias": True, "tl.non_restrict_params": []})
        with T.launch_thread("blockIdx.x", 16) as bx:
            by = T.launch_thread("blockIdx.y", 16)
            tx = T.launch_thread("threadIdx.x", 1)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            with T.allocate([1048576], "float16", "shared.rsram") as C_shared:
                D_shared = T.allocate([1048576], "float16", "shared.rsram")
                C_shared_1 = T.Buffer((1048576,), "float16", data=C_shared, scope="shared.rsram")
                with T.allocate([1048576], "float16", "shared.asram") as A_shared:
                    B_shared = T.allocate([1048576], "float16", "shared.wsram")
                    A_shared_1 = T.Buffer((1048576,), "float16", data=A_shared, scope="shared.asram")
                    B_shared_1 = T.Buffer((1048576,), "float16", data=B_shared, scope="shared.wsram")
                    T.mma_sunmmio(T.region(A_shared_1[0], 1, 1048576), T.region(B_shared_1[0], 1, 1048576), T.region(C_shared_1[0], 3, 1048576), T.bool(False), T.bool(False), T.bool(False), T.sync_token_id(0))
                D_shared_1 = T.Buffer((1048576,), "float16", data=D_shared, scope="shared.rsram")
                if bx <= 2:
                    for i in T.unroll(16384):
                        D_shared_1[i * 64:i * 64 + 64] = T.Broadcast(T.float16(0.0), 64)
                for i in range(5):
                    T.wait_token(0)
                    C_shared_1[i * 1024] = T.Cast("float16", T.Cast("float32", C_shared_1[i * 1024]) + T.float32(1.0))
                T.sync_null_token(2)
                T.barrier_init(1, 0, 1, 2, 3)
                for _i in range(10):
                    E_shared = T.allocate([1048576], "float16", "shared.rsram")
                    T.wait_token(2)
                    T.barrier_arrive_and_wait(1)
                    E_shared_1 = T.Buffer((1048576,), "float16", data=E_shared, scope="shared.rsram")
                    T.broadcast_(T.region(D_shared_1[0], 1, 1048576), T.region(E_shared_1[0], 2, 1048576), 1048576, 0, 0, T.sync_token_id(1))
                    T.barrier_init(0, 0, 1, 2, 3)
                    T.wait_token(1)
                    T.barrier_arrive_and_wait(0)
                    E_shared_1[0] = T.Cast("float16", T.Cast("float32", E_shared_1[0]) + T.float32(1.0))
                    T.broadcast_(T.region(E_shared_1[0], 1, 1048576), T.region(D_shared_1[0], 2, 1048576), 1048576, 0, 0, T.sync_token_id(2))
                    T.barrier_init(1, 0, 1, 2, 3)
            T.wait_token(2)
            T.barrier_arrive_and_wait(1)
        return 0
    """

    script_mere_allocate = [
        """
            buf_shmem = T.allocate([4194304], "uint8", "shared.rsram")
            A_shared = T.allocate([1048576], "float16", "shared.asram")
            B_shared = T.allocate([1048576], "float16", "shared.wsram")
        """
    ]

    def get_verify_merge_allocate():
        kernel_name = "kernel_kernel"
        # a, w only have one, no change in size,
        # r has five, 2 are not used, remaining cde, ce reuse, so only need the size of two buffers (*2), float(*2)
        cnt_a = 1024 * 1024
        cnt_w = 1024 * 1024
        cnt_r = 1024 * 1024 * 2 * 2
        return build_verify_merge_allocate(kernel_name=kernel_name, cnt_a=cnt_a, cnt_w=cnt_w, cnt_r=cnt_r)

    test_config = {
        "LowerTileOp": {
            "script_expected": script_lower_tile_op,
        },
        "InjectSunmmioSync": {
            "script_expected": script_InjectSunmmioSync,
        },
        "MergeSharedMemoryAllocationsSunmmio": {
            "script_expected": script_mere_allocate,
            "formal_verify": get_verify_merge_allocate(),
        },
        "DeviceMode": {
            "script_expected": script_device_mode,
        },
    }
    test_config = get_or_add_default_verify(func, test_config)
    compile_test(func, out_idx=[2], target="Sunmmio", test_config=test_config)


if __name__ == "__main__":
    test_sync()
