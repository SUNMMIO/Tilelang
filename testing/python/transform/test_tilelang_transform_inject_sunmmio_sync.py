import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.utils.target import SUNMMIO_TARGET_DESC


def simple_copy_kernel(M, N, block_M, block_N, dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(A_shared, B[by * block_M, bx * block_N])

    return tvm.IRModule({"main": main})


def mma_kernel(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            # Load A and B
            T.copy(A[by * block_M, 0], A_shared)
            T.copy(B[0, bx * block_N], B_shared)

            # GEMM
            T.gemm(A_shared, B_shared, C_shared)

            # Store C
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return tvm.IRModule({"main": main})


def broadcast_kernel(M, N, block_M, block_N, dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            B_shared = T.alloc_shared((block_M, block_N), dtype)

            # Load A
            T.copy(A[by * block_M, bx * block_N], A_shared)

            # Broadcast A to B
            T.comm.broadcast(A_shared, B_shared, (0, 0), direction="h")

            # Store B
            T.copy(B_shared, B[by * block_M, bx * block_N])

    return tvm.IRModule({"main": main})


def apply_sunmmio_lowering(mod, target):
    # This sequence lowers T.copy to tl.dma_copy and T.gemm to tl.mma_sunmmio
    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.InferSramScope()(mod)
    mod = tilelang.transform.LayoutReducer()(mod)
    mod = tilelang.transform.LayoutInference()(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    return mod


def test_inject_sunmmio_sync_dma():
    M, N = 128, 128
    block_M, block_N = 32, 32
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = simple_copy_kernel(M, N, block_M, block_N)

    with tvm.target.Target(target):
        # Lower to tl.dma_copy
        mod = apply_sunmmio_lowering(mod, target)

        # Now apply the sync injection pass
        mod = tilelang.transform.InjectSunmmioSync()(mod)

    script = mod.script()
    print(script)

    # Check for inserted sync calls
    # We expect wait_token calls to be inserted for synchronization
    # The script output uses T.wait_token
    assert "wait_token" in script
    # dma_copy should still be present
    assert "dma_copy" in script

    # Also check that we have a sync_token_id call or similar inside the dma_copy args or around it
    assert "sync_token_id" in script

    # Ensure order: dma(0) -> wait(0) -> dma(1)
    lines = [l.strip() for l in script.split("\n")]
    dma_lines = [l for l in lines if "dma_copy" in l]
    wait_lines = [l for l in lines if "wait_token" in l]

    assert len(dma_lines) == 2
    assert len(wait_lines) == 1

    assert "sync_token_id(0)" in dma_lines[0]
    assert "wait_token(0)" in wait_lines[0]
    assert "sync_token_id(1)" in dma_lines[1]

    # Check that wait(0) is between dma(0) and dma(1) in the full script
    idx_dma0 = script.find("sync_token_id(0)")
    idx_wait0 = script.find("wait_token(0)")
    idx_dma1 = script.find("sync_token_id(1)")

    assert idx_dma0 < idx_wait0 < idx_dma1


def test_inject_sunmmio_sync_mma():
    M, N, K = 128, 128, 128
    block_M, block_N, block_K = 32, 32, 32
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = mma_kernel(M, N, K, block_M, block_N, block_K)

    with tvm.target.Target(target):
        mod = apply_sunmmio_lowering(mod, target)
        mod = tilelang.transform.InjectSunmmioSync()(mod)

    script = mod.script()
    print(script)

    assert "mma_sunmmio" in script
    assert "wait_token" in script
    assert "sync_token_id" in script

    # Check that mma depends on previous copies
    # Copies (Token 0, 1) -> Wait(0), Wait(1) -> MMA (Token 2) -> Wait(2) -> Copy (Token 3)
    # The exact token IDs depend on the order of operations

    # Expected sequence roughly:
    # dma_copy(token=0) (load A)
    # dma_copy(token=1) (load B)
    # wait_token(0)
    # wait_token(1)
    # mma_sunmmio(token=2)
    # wait_token(2)
    # dma_copy(token=3) (store C)

    lines = [l.strip() for l in script.split("\n")]
    dma_lines = [l for l in lines if "dma_copy" in l]
    mma_lines = [l for l in lines if "mma_sunmmio" in l]
    wait_lines = [l for l in lines if "wait_token" in l]

    assert len(dma_lines) == 3  # 2 loads + 1 store
    assert len(mma_lines) == 1
    # We expect at least 3 wait tokens (0, 1, 2)
    assert len(wait_lines) >= 3

    # Check instruction order:
    # 1. dma_copy (load A) -> token 0
    # 2. dma_copy (load B) -> token 1
    # 3. wait_token(0)
    # 4. wait_token(1)
    # 5. mma_sunmmio -> token 2
    # 6. wait_token(2)
    # 7. dma_copy (store C) -> token 3

    idx_dma0 = script.find("sync_token_id(0)")
    idx_dma1 = script.find("sync_token_id(1)")
    idx_wait0 = script.find("wait_token(0)")
    idx_wait1 = script.find("wait_token(1)")
    idx_mma = script.find("mma_sunmmio")
    idx_token2 = script.find("sync_token_id(2)")  # token 2 is inside mma
    idx_wait2 = script.find("wait_token(2)")
    idx_dma2 = script.find("sync_token_id(3)")

    assert idx_dma0 < idx_wait0
    assert idx_dma1 < idx_wait1
    # Both wait(0) and wait(1) must be before MMA execution (which generates token 2)
    # Note: MMA instruction emits token 2, so it logically happens after waits for 0/1
    assert idx_wait0 < idx_mma
    assert idx_wait1 < idx_mma

    # Token 2 is generated by MMA
    assert idx_mma < idx_token2

    # Wait(2) must be after Token 2 generation and before DMA store (Token 3)
    assert idx_token2 < idx_wait2
    assert idx_wait2 < idx_dma2


def test_inject_sunmmio_sync_broadcast():
    M, N = 128, 128
    block_M, block_N = 32, 32
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = broadcast_kernel(M, N, block_M, block_N)

    with tvm.target.Target(target):
        mod = apply_sunmmio_lowering(mod, target)
        mod = tilelang.transform.InjectSunmmioSync()(mod)

    script = mod.script()
    print(script)

    assert "broadcast_" in script
    assert "barrier_init" in script
    assert "barrier_arrive_and_wait" in script

    # Broadcast usually involves barrier synchronization
    # dma_copy(token=0) -> wait_token(0) -> broadcast(token=1) -> barrier_wait?

    lines = [l.strip() for l in script.split("\n")]
    dma_lines = [l for l in lines if "dma_copy" in l]
    bcast_lines = [l for l in lines if "broadcast_" in l]
    barrier_lines = [l for l in lines if "barrier_arrive_and_wait" in l]
    wait_lines = [l for l in lines if "wait_token" in l]

    assert len(dma_lines) == 2
    assert len(bcast_lines) == 1
    assert len(barrier_lines) >= 1
    assert len(wait_lines) >= 2

    # Check instruction order:
    # 1. dma_copy (load A) -> token 0
    # 2. wait_token(0)
    # 3. broadcast_ -> token 1
    # 4. barrier_init
    # 5. wait_token(1)
    # 6. barrier_arrive_and_wait
    # 7. dma_copy (store B) -> token 2

    idx_dma0 = script.find("sync_token_id(0)")
    idx_wait0 = script.find("wait_token(0)")
    idx_bcast = script.find("broadcast_")
    idx_token1 = script.find("sync_token_id(1)", idx_bcast)  # token 1 should be in broadcast call
    idx_barrier_init = script.find("barrier_init")
    idx_wait1 = script.find("wait_token(1)")
    idx_barrier_wait = script.find("barrier_arrive_and_wait")
    idx_dma1 = script.find("sync_token_id(2)")

    # Verify order
    assert idx_dma0 < idx_wait0
    assert idx_wait0 < idx_bcast
    assert idx_bcast < idx_token1  # token 1 is inside broadcast
    assert idx_bcast < idx_barrier_init  # barrier init is usually after broadcast call or around it
    assert idx_barrier_init < idx_wait1
    assert idx_wait1 < idx_barrier_wait
    assert idx_barrier_wait < idx_dma1


def test_inject_sunmmio_sync_if():
    def kernel(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), accum_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
                D_shared = T.alloc_shared((block_M, block_N), accum_dtype)

                # Load A and B
                T.copy(A[by * block_M, 0], A_shared)
                T.copy(B[0, bx * block_N], B_shared)

                # GEMM
                T.gemm(A_shared, B_shared, C_shared)

                if by == 0:
                    T.comm.broadcast(C_shared, D_shared, (0, 0), direction="h")

                # Store C
                C_shared[0, 0] = C_shared[0, 0] + 1

        return tvm.IRModule({"main": main})

    func_str = """
                T.dma_copy(T.region(A[by * 32, 0], 1, 32, 32), T.region(A_shared[0, 0], 2, 32, 32), T.sync_token_id(0))
                T.dma_copy(T.region(B[0, bx * 32], 1, 32, 32), T.region(B_shared[0, 0], 2, 32, 32), T.sync_token_id(1))
                with T.block("_gemm_sss"):
                    T.reads()
                    T.writes()
                    T.wait_token(0)
                    T.wait_token(1)
                    T.mma_sunmmio(T.region(A_shared[0, 0], 1, 32, 32), T.region(B_shared[0, 0], 1, 32, 32), T.region(C_shared[0, 0], 3, 32, 32), T.bool(False), T.bool(False), T.bool(False), T.sync_token_id(2))
                if by == 0:
                    T.wait_token(2)
                    T.broadcast_(T.region(C_shared[0, 0], 1, 32, 32), T.region(D_shared[0, 0], 2, 32, 32), 1024, 0, 0, T.sync_token_id(3))
                    T.barrier_init(0, 0, 1, 2, 3)
                T.wait_token(3)
                T.barrier_arrive_and_wait(0)
                T.wait_token(2)
                C_shared[0, 0] = C_shared[0, 0] + T.float32(1.0)
    """.strip()

    M, N, K = 128, 128, 128
    block_M, block_N, block_K = 32, 32, 32
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = kernel(M, N, K, block_M, block_N, block_K)

    with tvm.target.Target(target):
        mod = apply_sunmmio_lowering(mod, target)
        mod = tilelang.transform.InjectSunmmioSync()(mod)

    script = mod.script(show_meta=True)
    assert script[-len(func_str) :] == func_str, "The generated script does not match the expected output."
    print(script)


def test_inject_sunmmio_sync_loop():
    def kernel(M, N, block_M, block_N, accum_dtype="float32"):
        @T.prim_func
        def main(
            C: T.Tensor((M, N), accum_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
                D_shared = T.alloc_shared((block_M, block_N), accum_dtype)
                T.copy(C[by * block_M, bx * block_N], D_shared)

                for _i in range(10):
                    T.copy(D_shared, C_shared)
                    T.copy(C_shared, D_shared)

                for _i in range(10):
                    T.comm.broadcast(C_shared, D_shared, (0, 0), direction="h")
                    T.comm.broadcast(D_shared, C_shared, (0, 0), direction="h")

        return tvm.IRModule({"main": main})

    func_str = """
                T.dma_copy(T.region(C[by * 32, bx * 32], 1, 32, 32), T.region(D_shared[0, 0], 2, 32, 32), T.sync_token_id(0))
                for _i in range(10):
                    T.wait_token(0)
                    T.wait_token(2)
                    T.dma_copy(T.region(D_shared[0, 0], 1, 32, 32), T.region(C_shared[0, 0], 2, 32, 32), T.sync_token_id(1))
                    T.wait_token(1)
                    T.dma_copy(T.region(C_shared[0, 0], 1, 32, 32), T.region(D_shared[0, 0], 2, 32, 32), T.sync_token_id(2))
                for _i in range(10):
                    T.wait_token(1)
                    T.wait_token(4)
                    T.barrier_arrive_and_wait(1)
                    T.wait_token(0)
                    T.wait_token(2)
                    T.broadcast_(T.region(C_shared[0, 0], 1, 32, 32), T.region(D_shared[0, 0], 2, 32, 32), 1024, 0, 0, T.sync_token_id(3))
                    T.barrier_init(0, 0, 1, 2, 3)
                    T.wait_token(3)
                    T.barrier_arrive_and_wait(0)
                    T.broadcast_(T.region(D_shared[0, 0], 1, 32, 32), T.region(C_shared[0, 0], 2, 32, 32), 1024, 0, 0, T.sync_token_id(4))
                    T.barrier_init(1, 0, 1, 2, 3)
    """.strip()

    M, N = 128, 128
    block_M, block_N = 32, 32
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = kernel(M, N, block_M, block_N)

    with tvm.target.Target(target):
        mod = apply_sunmmio_lowering(mod, target)
        mod = tilelang.transform.InjectSunmmioSync()(mod)

    script = mod.script(show_meta=True)
    print(script)
    assert script[-len(func_str) :] == func_str, "The generated script does not match the expected output."


if __name__ == "__main__":
    test_inject_sunmmio_sync_dma()
    test_inject_sunmmio_sync_mma()
    test_inject_sunmmio_sync_broadcast()
    test_inject_sunmmio_sync_if()
    test_inject_sunmmio_sync_loop()
