import re

import tilelang
import tilelang.language as T
from tilelang import tvm
from tvm import tir, IRModule
from tilelang.utils.target import determine_target
from tvm.target import Target


def get_target(target_str: str):
    target = determine_target(target_str, return_object=True)
    target_host = "llvm" if tvm.runtime.enabled("llvm") else "c"
    target_host = tvm.target.Target.canon_target(target_host)
    target = tvm.target.Target(target, target_host)
    return target


def LowerAndLegalize_sunmmio(
    mod: IRModule,
    target: Target,
) -> IRModule:
    mod = tir.transform.BindTarget(target)(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.InferSramScope()(mod)
    mod = tilelang.transform.LegalizeSunmmioDataPath()(mod)
    mod = tilelang.transform.SunmmioLayoutInference()(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = tilelang.transform.LegalizeTilesLoop()(mod)
    mod = tilelang.transform.TilesLoop()(mod)
    mod = tilelang.transform.DecoupleTypeCast()(mod)
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tilelang.transform.LowerAccessPtr()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.HoistNonRestrictParams()(mod)
    return mod


def OptimizeForSunmmio_patial(
    mod: IRModule,
    target: Target,
) -> IRModule:
    mod = tilelang.transform.IfStmtBinding()(mod)
    mod = tilelang.transform.SunmmioPipelinePlanning(debug=False)(mod)
    mod = tilelang.transform.InjectSunmmioPipeline()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tilelang.transform.LoopUnswitching()(mod)
    mod = tir.transform.UnrollLoop()(mod)
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.RemoveNoOp()(mod)
    mod = tir.transform.HoistIfThenElse()(mod)
    mod = tir.transform.VerifyMemory()(mod)
    mod = tir.transform.AnnotateEntryFunc()(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    return mod


def inject_sunmmio_sync_script(func, show_meta=True):
    target = get_target("Sunmmio")
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    mod = LowerAndLegalize_sunmmio(mod, target)
    mod = OptimizeForSunmmio_patial(mod, target)
    mod = tilelang.transform.InjectSunmmioSync()(mod)
    return mod.script(show_meta=show_meta)


def extract_call_id(line, marker):
    match = re.search(rf"{re.escape(marker)}\((\d+)\)", line)
    assert match, f"Cannot parse {marker} in line: {line}"
    return int(match.group(1))


def call_entries(lines, op_name, marker="sync_token_id"):
    return [
        (idx, line.strip(), extract_call_id(line, marker)) for idx, line in enumerate(lines) if op_name in line and f"{marker}(" in line
    ]


def barrier_init_entries(lines):
    entries = []
    for idx, line in enumerate(lines):
        match = re.search(r"barrier_init\(([^)]*)\)", line)
        if not match:
            continue
        args = match.group(1).strip()
        first_arg = args.split(",", 1)[0].strip()
        if first_arg.isdigit():
            entries.append((idx, line.strip(), int(first_arg), "," not in args))
    return entries


def line_indent(line):
    return len(line) - len(line.lstrip())


def scope_end(lines, start_idx):
    base_indent = line_indent(lines[start_idx])
    for idx in range(start_idx + 1, len(lines)):
        if lines[idx].strip() and line_indent(lines[idx]) <= base_indent:
            return idx
    return len(lines)


def kernel_simple_copy(M, N, block_M, block_N, dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(A_shared, B[by * block_M, bx * block_N])

    return main


def test_inject_sunmmio_sync_dma():
    M, N = 128, 128
    block_M, block_N = 32, 32
    target = get_target("Sunmmio")
    func = kernel_simple_copy(M, N, block_M, block_N)

    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    mod = LowerAndLegalize_sunmmio(mod, target)
    mod = OptimizeForSunmmio_patial(mod, target)

    mod = tilelang.transform.InjectSunmmioSync()(mod)
    script = mod.script()

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


def kernel_mma(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
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

    return main


def test_inject_sunmmio_sync_mma():
    M, N, K = 128, 128, 128
    block_M, block_N, block_K = 32, 32, 32
    target = get_target("Sunmmio")
    func = kernel_mma(M, N, K, block_M, block_N, block_K)

    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    mod = LowerAndLegalize_sunmmio(mod, target)
    mod = OptimizeForSunmmio_patial(mod, target)

    mod = tilelang.transform.InjectSunmmioSync()(mod)

    script = mod.script()

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


def kernel_broadcast(M, N, block_M, block_N, dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            B_shared = T.alloc_shared((block_M, block_N), dtype)

            # Load A
            T.copy(A[by * block_M, bx * block_N], A_shared)

            # Broadcast A to B
            T.comm.broadcast(A_shared, B_shared, (0, 0), direction="h")

            # Store B
            T.copy(B_shared, B[by * block_M, bx * block_N])

    return main


def test_inject_sunmmio_sync_broadcast():
    M, N = 128, 128
    block_M, block_N = 32, 32
    target = get_target("Sunmmio")
    func = kernel_broadcast(M, N, block_M, block_N)

    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    mod = LowerAndLegalize_sunmmio(mod, target)
    mod = OptimizeForSunmmio_patial(mod, target)

    mod = tilelang.transform.InjectSunmmioSync()(mod)
    script = mod.script()

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


def kernel_sync_if(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
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

    return main


def test_inject_sunmmio_sync_if():
    M, N, K = 128, 128, 128
    block_M, block_N, block_K = 32, 32, 32
    target = get_target("Sunmmio")
    func = kernel_sync_if(M, N, K, block_M, block_N, block_K)

    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    mod = LowerAndLegalize_sunmmio(mod, target)
    mod = OptimizeForSunmmio_patial(mod, target)

    mod = tilelang.transform.InjectSunmmioSync()(mod)
    script = mod.script(show_meta=True)
    # print(script)
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


def kernel_sync_loop(M, N, block_M, block_N, accum_dtype="float32"):
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

    return main


def test_inject_sunmmio_sync_loop():
    func_str = """
        with T.launch_thread("blockIdx.x", 4) as bx:
            by = T.launch_thread("blockIdx.y", 4)
            tx = T.launch_thread("threadIdx.x", 128)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            with T.decl_buffer((32, 32), scope="shared.rsram") as C_shared:
                D_shared = T.decl_buffer((32, 32), scope="shared.rsram")
                C_2 = T.Buffer((128, 128), data=C, strides=(128, 1))
                T.dma_copy(T.region(C_2[by * 32, bx * 32], 1, 32, 32), T.region(D_shared[0, 0], 2, 32, 32), T.sync_token_id(0))
                T.sync_null_token(2)
                T.barrier_init(1)
                T.wait_token(0)
                for _i in range(10):
                    T.wait_token(2)
                    T.barrier_arrive_and_wait(1)
                    T.broadcast_(T.region(C_shared[0, 0], 1, 32, 32), T.region(D_shared[0, 0], 2, 32, 32), 1024, 0, 0, T.sync_token_id(1))
                    T.barrier_init(0, 0, 1, 2, 3)
                    T.wait_token(1)
                    T.barrier_arrive_and_wait(0)
                    T.broadcast_(T.region(D_shared[0, 0], 1, 32, 32), T.region(C_shared[0, 0], 2, 32, 32), 1024, 0, 0, T.sync_token_id(2))
                    T.barrier_init(1, 0, 1, 2, 3)
            T.wait_token(2)
            T.barrier_arrive_and_wait(1)
    """.strip()

    M, N = 128, 128
    block_M, block_N = 32, 32
    target = get_target("Sunmmio")
    func = kernel_sync_loop(M, N, block_M, block_N)

    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    mod = LowerAndLegalize_sunmmio(mod, target)
    mod = OptimizeForSunmmio_patial(mod, target)

    mod = tilelang.transform.InjectSunmmioSync()(mod)
    script = mod.script(show_meta=True)
    assert func_str in script, "The generated script does not match the expected output."


def kernel_summa_loop_carried(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_shared((block_M, block_N), accum_dtype)

            for k_tile in range(T.ceildiv(K, block_K)):
                T.comm.broadcast(
                    A[
                        by * block_M : by * block_M + block_M,
                        k_tile * block_K : k_tile * block_K + block_K,
                    ],
                    A_shared,
                    (by, k_tile),
                    direction="h",
                )
                T.comm.broadcast(
                    B[
                        k_tile * block_K : k_tile * block_K + block_K,
                        bx * block_N : bx * block_N + block_N,
                    ],
                    B_shared,
                    (k_tile, bx),
                    direction="v",
                )
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def test_inject_sunmmio_sync_summa_loop_carried():
    func = kernel_summa_loop_carried(128, 128, 128, 32, 32, 32)
    script = inject_sunmmio_sync_script(func)
    lines = script.split("\n")

    broadcast_entries = call_entries(lines, "broadcast_")
    mma_entries = call_entries(lines, "mma_sunmmio")
    assert len(broadcast_entries) >= 2
    assert len(mma_entries) == 1

    first_bcast_idx, _, first_bcast_token = broadcast_entries[0]
    second_bcast_idx, _, second_bcast_token = broadcast_entries[1]
    mma_idx, _, mma_token = mma_entries[0]
    loop_idx = next(idx for idx, line in enumerate(lines) if "for k_tile in range(" in line)
    first_dma_idx, _, _ = next(entry for entry in call_entries(lines, "dma_copy") if loop_idx < entry[0] < first_bcast_idx)

    null_idx = next(idx for idx, line in enumerate(lines) if f"sync_null_token({mma_token})" in line)
    assert null_idx < loop_idx

    first_wait = next(
        idx for idx, line in enumerate(lines) if idx > loop_idx and idx < first_bcast_idx and f"wait_token({mma_token})" in line
    )
    assert first_dma_idx < first_wait < first_bcast_idx
    assert not any(f"wait_token({mma_token})" in line for line in lines[first_bcast_idx + 1 : second_bcast_idx])

    assert first_bcast_idx < second_bcast_idx < mma_idx
    assert first_bcast_token != mma_token
    assert second_bcast_token != mma_token
    assert "k_tile = T.int32()" not in script
    assert not any("barrier_init" in line and line_indent(line) <= line_indent(lines[loop_idx]) for line in lines[null_idx + 1 : loop_idx])


def kernel_loop_wait_sunk_to_conflicting_broadcast(M, N, block_M, block_N, dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")
            B_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")
            C_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")
            D_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")

            T.copy(A[by * block_M, bx * block_N], A_shared)
            for _i in range(10):
                T.comm.broadcast(A_shared, C_shared, (0, 0), direction="h")
                T.comm.broadcast(D_shared, B_shared, (0, 0), direction="h")
            T.copy(B_shared, B[by * block_M, bx * block_N])

    return main


def test_inject_sunmmio_sync_loop_wait_sunk_to_conflicting_broadcast():
    func = kernel_loop_wait_sunk_to_conflicting_broadcast(128, 128, 32, 32)
    script = inject_sunmmio_sync_script(func)
    lines = script.split("\n")

    loop_idx = next(idx for idx, line in enumerate(lines) if "for _i in range(10):" in line)
    loop_end = scope_end(lines, loop_idx)
    loop_broadcasts = [(idx, line, token) for idx, line, token in call_entries(lines, "broadcast_") if loop_idx < idx < loop_end]
    assert len(loop_broadcasts) == 2

    first_bcast_idx, _, _ = loop_broadcasts[0]
    second_bcast_idx, second_bcast_line, second_bcast_token = loop_broadcasts[1]
    assert "B_shared" in second_bcast_line

    second_barrier_idx, _, second_barrier, _ = next(entry for entry in barrier_init_entries(lines) if entry[0] > second_bcast_idx)
    assert second_barrier_idx < loop_end

    null_idx = next(idx for idx, line in enumerate(lines) if f"sync_null_token({second_bcast_token})" in line)
    entry_barrier_idx = next(
        idx for idx, _, barrier_id, is_loop_entry in barrier_init_entries(lines) if is_loop_entry and barrier_id == second_barrier
    )
    assert null_idx < entry_barrier_idx < loop_idx

    wait_idx = next(
        idx for idx, line in enumerate(lines) if first_bcast_idx < idx < second_bcast_idx and f"wait_token({second_bcast_token})" in line
    )
    barrier_wait_idx = next(
        idx for idx, line in enumerate(lines) if wait_idx < idx < second_bcast_idx and f"barrier_arrive_and_wait({second_barrier})" in line
    )
    assert first_bcast_idx < wait_idx < barrier_wait_idx < second_bcast_idx


def kernel_barrier_cf(M, N, block_N, dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=128) as bx:
            A_shared = T.alloc_shared((M, block_N), dtype, scope="shared.rsram")
            B_shared = T.alloc_shared((M, block_N), dtype, scope="shared.rsram")

            T.copy(A[0, bx * block_N], A_shared)
            if bx == 0:
                T.comm.broadcast(A_shared, B_shared, (0, 0), direction="h")
            T.copy(B_shared, B[0, bx * block_N])

    return main


def test_inject_sunmmio_sync_barrier_control_flow():
    func = kernel_barrier_cf(128, 128, 32)
    script = inject_sunmmio_sync_script(func)
    lines = script.split("\n")

    if_idx = next(idx for idx, line in enumerate(lines) if "if bx == 0:" in line)
    if_end = scope_end(lines, if_idx)
    broadcast_idx, _, broadcast_token = next(entry for entry in call_entries(lines, "broadcast_") if if_idx < entry[0] < if_end)
    barrier_init_idx, _, barrier_id, _ = next(entry for entry in barrier_init_entries(lines) if if_idx < entry[0] < if_end)

    assert if_idx < broadcast_idx < barrier_init_idx < if_end

    wait_idx = next(idx for idx, line in enumerate(lines) if idx >= if_end and f"wait_token({broadcast_token})" in line)
    barrier_wait_idx = next(idx for idx, line in enumerate(lines) if idx > wait_idx and f"barrier_arrive_and_wait({barrier_id})" in line)
    store_idx = next(idx for idx, line in enumerate(lines) if idx > barrier_wait_idx and "dma_copy" in line and "B_shared" in line)

    assert wait_idx < barrier_wait_idx < store_idx


def kernel_while_loop_carried_dma(dtype="float16"):
    def make_region(buffer, access_mask):
        return tir.Call(
            "handle",
            tvm.ir.Op.get("tl.tileop.region"),
            [
                tir.BufferLoad(buffer, [tir.IntImm("int32", 0)]),
                tir.IntImm("int32", access_mask),
                tir.IntImm("int32", 16),
            ],
        )

    A_shared = tir.decl_buffer((16,), dtype, name="A_shared", scope="shared.rsram")
    B_shared = tir.decl_buffer((16,), dtype, name="B_shared", scope="shared.rsram")
    first_dma = tir.Evaluate(
        tir.Call(
            "handle",
            tvm.ir.Op.get("tl.dma_copy"),
            [make_region(B_shared, 1), make_region(A_shared, 2)],
        )
    )
    second_dma = tir.Evaluate(
        tir.Call(
            "handle",
            tvm.ir.Op.get("tl.dma_copy"),
            [make_region(A_shared, 1), make_region(B_shared, 2)],
        )
    )
    loop = tir.While(tir.IntImm("bool", True), tir.SeqStmt([first_dma, second_dma]))
    body = tir.Allocate(
        A_shared.data,
        dtype,
        [16],
        tir.IntImm("bool", True),
        tir.Allocate(B_shared.data, dtype, [16], tir.IntImm("bool", True), loop),
    )
    return tir.PrimFunc([], body).with_attr("tir.is_global_func", True)


def test_inject_sunmmio_sync_while_loop_carried():
    target = get_target("Sunmmio")
    func = kernel_while_loop_carried_dma()

    mod = tvm.IRModule({"main": func})
    mod = tir.transform.BindTarget(target)(mod)
    mod = tilelang.transform.InjectSunmmioSync()(mod)
    script = mod.script(show_meta=True)
    lines = script.split("\n")

    dma_entries = call_entries(lines, "dma_copy")
    assert len(dma_entries) == 2

    first_dma_idx, _, first_dma_token = dma_entries[0]
    second_dma_idx, _, second_dma_token = dma_entries[1]
    assert first_dma_token != second_dma_token
    assert first_dma_idx < second_dma_idx

    while_idx = next(idx for idx, line in enumerate(lines) if "while " in line)
    null_idx = next(idx for idx, line in enumerate(lines) if f"sync_null_token({second_dma_token})" in line)
    wait_idx = next(idx for idx, line in enumerate(lines) if while_idx < idx < first_dma_idx and f"wait_token({second_dma_token})" in line)

    assert null_idx < while_idx < wait_idx < first_dma_idx


if __name__ == "__main__":
    test_inject_sunmmio_sync_dma()
    test_inject_sunmmio_sync_mma()
    test_inject_sunmmio_sync_broadcast()
    test_inject_sunmmio_sync_if()
    test_inject_sunmmio_sync_loop()
    test_inject_sunmmio_sync_summa_loop_carried()
    test_inject_sunmmio_sync_loop_wait_sunk_to_conflicting_broadcast()
    test_inject_sunmmio_sync_barrier_control_flow()
    test_inject_sunmmio_sync_while_loop_carried()
