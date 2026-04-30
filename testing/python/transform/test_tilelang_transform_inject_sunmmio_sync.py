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
    mod = tilelang.transform.LegalizeSunmmioCopyPath()(mod)
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
    assert len(wait_lines) == 2

    assert "sync_token_id(0)" in dma_lines[0]
    assert "wait_token(0)" in wait_lines[0]
    assert "sync_token_id(1)" in dma_lines[1]
    assert "wait_token(1)" in wait_lines[1]

    # Check that wait(0) is between dma(0) and dma(1) in the full script
    idx_dma0 = script.find("sync_token_id(0)")
    idx_wait0 = script.find("wait_token(0)")
    idx_dma1 = script.find("sync_token_id(1)")
    idx_wait1 = script.find("wait_token(1)")

    assert idx_dma0 < idx_wait0 < idx_dma1 < idx_wait1


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
    # dma_copy(token=0) (load A')
    # wait_token(0)
    # dma_copy(token=1) (load A)
    # dma_copy(token=2) (load B)
    # wait_token(1)
    # wait_token(2)
    # mma_sunmmio(token=3)
    # wait_token(3)
    # dma_copy(token=4) (store C)
    # wait_token(4)

    lines = [l.strip() for l in script.split("\n")]

    def extract_token_id(line, marker):
        prefix = f"{marker}("
        start = line.find(prefix)
        assert start != -1, f"Cannot find {marker} in line: {line}"
        start += len(prefix)
        end = line.find(")", start)
        assert end != -1, f"Cannot parse {marker} in line: {line}"
        return int(line[start:end])

    dma_entries = [
        (idx, line, extract_token_id(line, "sync_token_id"))
        for idx, line in enumerate(lines)
        if "dma_copy" in line and "sync_token_id(" in line
    ]
    mma_entries = [
        (idx, line, extract_token_id(line, "sync_token_id"))
        for idx, line in enumerate(lines)
        if "mma_sunmmio" in line and "sync_token_id(" in line
    ]
    wait_entries = [(idx, line, extract_token_id(line, "wait_token")) for idx, line in enumerate(lines) if "wait_token(" in line]

    assert len(dma_entries) >= 3
    assert len(mma_entries) == 1
    assert len(wait_entries) >= 4

    mma_idx, _, mma_token = mma_entries[0]
    pre_mma_dma_tokens = [token for idx, _, token in dma_entries if idx < mma_idx]
    post_mma_dma_entries = [(idx, token) for idx, _, token in dma_entries if idx > mma_idx]
    pre_mma_wait_tokens = {token for idx, _, token in wait_entries if idx < mma_idx}

    # A/B loads may include an extra staging DMA, but every DMA before MMA must
    # be waited on before the MMA executes.
    assert len(pre_mma_dma_tokens) >= 2
    assert set(pre_mma_dma_tokens).issubset(pre_mma_wait_tokens)

    # The MMA-generated token must be waited on before any DMA that consumes its
    # result, such as the final store.
    assert post_mma_dma_entries
    first_post_mma_dma_idx = min(idx for idx, _ in post_mma_dma_entries)
    mma_wait_indices = [idx for idx, _, token in wait_entries if token == mma_token and idx > mma_idx]
    assert mma_wait_indices
    assert min(mma_wait_indices) < first_post_mma_dma_idx

    # Every DMA after MMA should eventually be waited on as well.
    for dma_idx, dma_token in post_mma_dma_entries:
        wait_indices = [idx for idx, _, token in wait_entries if token == dma_token and idx > dma_idx]
        assert wait_indices, f"Missing wait_token({dma_token}) after DMA line {dma_idx}"


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
    assert len(wait_lines) >= 3

    # Check instruction order:
    # 1. dma_copy (load A) -> token 0
    # 2. wait_token(0)
    # 3. broadcast_ -> token 1
    # 4. barrier_init
    # 5. wait_token(1)
    # 6. barrier_arrive_and_wait
    # 7. dma_copy (store B) -> token 2
    # 8. wait_token(2)

    idx_dma0 = script.find("sync_token_id(0)")
    idx_wait0 = script.find("wait_token(0)")
    idx_bcast = script.find("broadcast_")
    idx_token1 = script.find("sync_token_id(1)", idx_bcast)  # token 1 should be in broadcast call
    idx_barrier_init = script.find("barrier_init")
    idx_wait1 = script.find("wait_token(1)")
    idx_barrier_wait = script.find("barrier_arrive_and_wait")
    idx_dma1 = script.find("sync_token_id(2)")
    idx_wait2 = script.find("wait_token(2)")

    # Verify order
    assert idx_dma0 < idx_wait0
    assert idx_wait0 < idx_bcast
    assert idx_bcast < idx_token1  # token 1 is inside broadcast
    assert idx_bcast < idx_barrier_init  # barrier init is usually after broadcast call or around it
    assert idx_barrier_init < idx_wait1
    assert idx_wait1 < idx_barrier_wait
    assert idx_barrier_wait < idx_dma1
    assert idx_dma1 < idx_wait2


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

    M, N, K = 128, 128, 128
    block_M, block_N, block_K = 32, 32, 32
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = kernel(M, N, K, block_M, block_N, block_K)

    with tvm.target.Target(target):
        mod = apply_sunmmio_lowering(mod, target)
        mod = tilelang.transform.InjectSunmmioSync()(mod)

    script = mod.script(show_meta=True)
    print(script)
    assert "mma_sunmmio" in script
    assert "broadcast_" in script
    assert "barrier_init" in script
    assert "barrier_arrive_and_wait" in script
    assert "if by == 0:" in script

    lines = [l.strip() for l in script.split("\n")]

    def extract_token_id(line, marker):
        prefix = f"{marker}("
        start = line.find(prefix)
        assert start != -1, f"Cannot find {marker} in line: {line}"
        start += len(prefix)
        end = line.find(")", start)
        assert end != -1, f"Cannot parse {marker} in line: {line}"
        return int(line[start:end])

    dma_entries = [
        (idx, line, extract_token_id(line, "sync_token_id"))
        for idx, line in enumerate(lines)
        if "dma_copy" in line and "sync_token_id(" in line
    ]
    mma_entries = [
        (idx, line, extract_token_id(line, "sync_token_id"))
        for idx, line in enumerate(lines)
        if "mma_sunmmio" in line and "sync_token_id(" in line
    ]
    broadcast_entries = [
        (idx, line, extract_token_id(line, "sync_token_id"))
        for idx, line in enumerate(lines)
        if "broadcast_" in line and "sync_token_id(" in line
    ]
    wait_entries = [(idx, line, extract_token_id(line, "wait_token")) for idx, line in enumerate(lines) if "wait_token(" in line]

    assert len(dma_entries) >= 2
    assert len(mma_entries) == 1
    assert len(broadcast_entries) == 1

    mma_idx, _, mma_token = mma_entries[0]
    broadcast_idx, _, broadcast_token = broadcast_entries[0]
    if_idx = next(idx for idx, line in enumerate(lines) if line == "if by == 0:")
    barrier_init_idx = next(idx for idx, line in enumerate(lines) if "barrier_init" in line)
    barrier_wait_idx = next(idx for idx, line in enumerate(lines) if "barrier_arrive_and_wait" in line)
    final_store_idx = next(idx for idx, line in enumerate(lines) if "C_shared[0, 0] = C_shared[0, 0] +" in line)

    pre_mma_dma_tokens = [token for idx, _, token in dma_entries if idx < mma_idx]
    pre_mma_wait_tokens = {token for idx, _, token in wait_entries if idx < mma_idx}

    # All DMA loads that happen before the MMA must be waited on first.
    assert set(pre_mma_dma_tokens).issubset(pre_mma_wait_tokens)

    # The conditional branch should wait for the MMA result before broadcasting it.
    branch_wait_indices = [idx for idx, _, token in wait_entries if token == mma_token and if_idx < idx < broadcast_idx]
    assert branch_wait_indices
    assert mma_idx < min(branch_wait_indices) < broadcast_idx

    # The branch-local broadcast should be followed by barrier setup.
    assert if_idx < broadcast_idx < barrier_init_idx

    # The broadcast token must be waited on before the outer barrier wait.
    broadcast_wait_indices = [idx for idx, _, token in wait_entries if token == broadcast_token and idx > barrier_init_idx]
    assert broadcast_wait_indices
    assert barrier_init_idx < min(broadcast_wait_indices) < barrier_wait_idx

    # The MMA token should also be waited on after the branch before C_shared is consumed.
    post_branch_mma_wait_indices = [idx for idx, _, token in wait_entries if token == mma_token and idx > barrier_wait_idx]
    assert post_branch_mma_wait_indices
    assert barrier_wait_idx < min(post_branch_mma_wait_indices) < final_store_idx


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
                    T.comm.broadcast(C_shared, D_shared, (0, 0), direction="h")
                    T.comm.broadcast(D_shared, C_shared, (0, 0), direction="h")

        return tvm.IRModule({"main": main})

    func_str = """
                T.dma_copy(T.region(C[by * 32, bx * 32], 1, 32, 32), T.region(D_shared[0, 0], 2, 32, 32), T.sync_token_id(0))
                T.sync_null_token(2)
                T.barrier_init(1, 0, 1, 2, 3)
                for _i in range(10):
                    T.wait_token(2)
                    T.barrier_arrive_and_wait(1)
                    T.wait_token(0)
                    T.broadcast_(T.region(C_shared[0, 0], 1, 32, 32), T.region(D_shared[0, 0], 2, 32, 32), 1024, 0, 0, T.sync_token_id(1))
                    T.barrier_init(0, 0, 1, 2, 3)
                    T.wait_token(1)
                    T.barrier_arrive_and_wait(0)
                    T.broadcast_(T.region(D_shared[0, 0], 1, 32, 32), T.region(C_shared[0, 0], 2, 32, 32), 1024, 0, 0, T.sync_token_id(2))
                    T.barrier_init(1, 0, 1, 2, 3)
                T.wait_token(0)
                T.wait_token(2)
                T.barrier_arrive_and_wait(1)
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
