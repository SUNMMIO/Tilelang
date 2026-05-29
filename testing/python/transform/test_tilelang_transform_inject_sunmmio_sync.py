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
    # mod = tilelang.transform.SunmmioPipelinePlanning(debug=False)(mod)
    # mod = tilelang.transform.InjectSunmmioPipeline()(mod)
    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)
    mod = tir.transform.HoistIfThenElse()(mod)
    mod = tilelang.transform.LoopUnswitching()(mod)
    mod = tir.transform.UnrollLoop()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.VerifyMemory()(mod)
    mod = tir.transform.AnnotateEntryFunc()(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    return mod


def simple_copy_kernel(M, N, block_M, block_N, dtype="float16"):
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


def mma_kernel(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
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


def broadcast_kernel(M, N, block_M, block_N, dtype="float16"):
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


def _pointer_var(name, dtype="float16", scope="shared.rsram"):
    return tir.Var(name, tvm.ir.PointerType(tvm.ir.PrimType(dtype), scope))


def _region(buf, access):
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.region"),
        tir.BufferLoad(buf, [tir.IntImm("int32", 0), tir.IntImm("int32", 0)]),
        tir.IntImm("int32", access),
        tir.IntImm("int32", 32),
        tir.IntImm("int32", 32),
    )


def _make_leaf_broadcast_without_src_core_mod(target):
    src_data = _pointer_var("src")
    dst_data = _pointer_var("dst")
    src_buf = tir.decl_buffer(
        (32, 32),
        "float16",
        name="src_buf",
        data=src_data,
        scope="shared.rsram",
    )
    dst_buf = tir.decl_buffer(
        (32, 32),
        "float16",
        name="dst_buf",
        data=dst_data,
        scope="shared.rsram",
    )
    broadcast = tir.Evaluate(
        tir.call_intrin(
            "handle",
            tir.op.Op.get("tl.broadcast_"),
            _region(src_buf, 1),
            _region(dst_buf, 2),
            tir.IntImm("int32", 0),
            tir.IntImm("int64", 15),
            tir.IntImm("int32", 0),
        )
    )
    consume_dst = tir.Evaluate(tir.BufferLoad(dst_buf, [tir.IntImm("int32", 0), tir.IntImm("int32", 0)]))
    body = tir.DeclBuffer(
        src_buf,
        tir.DeclBuffer(dst_buf, tir.SeqStmt([broadcast, consume_dst])),
    )
    func = tir.PrimFunc([src_data, dst_data], body)
    func = func.with_attr("global_symbol", "main")
    func = func.with_attr("tir.is_global_func", True)
    mod = tvm.IRModule({"main": func})
    return tir.transform.BindTarget(target)(mod)


def _make_dynamic_row_broadcast_mod(target):
    src_data = _pointer_var("src")
    dst_data = _pointer_var("dst")
    src_buf = tir.decl_buffer(
        (32, 32),
        "float16",
        name="src_buf",
        data=src_data,
        scope="shared.rsram",
    )
    dst_buf = tir.decl_buffer(
        (32, 32),
        "float16",
        name="dst_buf",
        data=dst_data,
        scope="shared.rsram",
    )
    bx = tir.Var("bx", "int32")
    mask = tir.IntImm("int64", 15)
    broadcast = tir.Evaluate(
        tir.call_intrin(
            "handle",
            tir.op.Op.get("tl.broadcast_"),
            _region(src_buf, 1),
            _region(dst_buf, 2),
            tir.IntImm("int32", 0),
            mask,
            tir.IntImm("int32", 0),
            bx * tir.IntImm("int32", 4),
        )
    )
    consume_dst = tir.Evaluate(tir.BufferLoad(dst_buf, [tir.IntImm("int32", 0), tir.IntImm("int32", 0)]))
    loop = tir.For(
        bx,
        tir.IntImm("int32", 0),
        tir.IntImm("int32", 4),
        tir.ForKind.SERIAL,
        tir.SeqStmt([broadcast, consume_dst]),
    )
    body = tir.DeclBuffer(
        src_buf,
        tir.DeclBuffer(dst_buf, loop),
    )
    func = tir.PrimFunc([src_data, dst_data], body)
    func = func.with_attr("global_symbol", "main")
    func = func.with_attr("tir.is_global_func", True)
    mod = tvm.IRModule({"main": func})
    return tir.transform.BindTarget(target)(mod)


def _make_dynamic_row_pair_broadcast_mod(target):
    src_data = _pointer_var("src")
    dst_data = _pointer_var("dst")
    src_buf = tir.decl_buffer(
        (32, 32),
        "float16",
        name="src_buf",
        data=src_data,
        scope="shared.rsram",
    )
    dst_buf = tir.decl_buffer(
        (32, 32),
        "float16",
        name="dst_buf",
        data=dst_data,
        scope="shared.rsram",
    )
    bx = tir.Var("bx", "int32")
    dst_col = bx + tir.IntImm("int32", 1)
    pair_mask = tir.shift_left(tir.IntImm("int64", 1), tir.Cast("int64", dst_col))
    broadcast = tir.Evaluate(
        tir.call_intrin(
            "handle",
            tir.op.Op.get("tl.broadcast_"),
            _region(src_buf, 1),
            _region(dst_buf, 2),
            tir.IntImm("int32", 0),
            pair_mask,
            tir.IntImm("int32", 0),
            bx,
        )
    )
    consume_dst = tir.Evaluate(tir.BufferLoad(dst_buf, [tir.IntImm("int32", 0), tir.IntImm("int32", 0)]))
    loop = tir.For(
        bx,
        tir.IntImm("int32", 0),
        tir.IntImm("int32", 3),
        tir.ForKind.SERIAL,
        tir.SeqStmt([broadcast, consume_dst]),
    )
    body = tir.DeclBuffer(
        src_buf,
        tir.DeclBuffer(dst_buf, loop),
    )
    func = tir.PrimFunc([src_data, dst_data], body)
    func = func.with_attr("global_symbol", "main")
    func = func.with_attr("tir.is_global_func", True)
    mod = tvm.IRModule({"main": func})
    return tir.transform.BindTarget(target)(mod)


def _parse_numeric_barrier_mask(line, marker="barrier_init"):
    match = re.search(rf"{marker}\((?:T\.int64\()?(-?\d+)\)?\)", line)
    assert match, f"expected {marker}(participant_mask), got: {line}"
    return int(match.group(1))


def _parse_barrier_args(line, marker="barrier_init"):
    match = re.search(rf"{marker}\((.*)\)", line)
    assert match, f"expected {marker}(...), got: {line}"
    args = match.group(1)
    values = []
    for explicit_i64, bare_int in re.findall(
        r"T\.int64\((-?\d+)\)|(?<![A-Za-z_])(-?\d+)(?![A-Za-z_])",
        args,
    ):
        values.append(int(explicit_i64 or bare_int))
    return values


def apply_sunmmio_lowering(mod, target):
    # This sequence lowers T.copy to tl.dma_copy and T.gemm to tl.mma_sunmmio
    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.InferSramScope()(mod)
    mod = tilelang.transform.LegalizeSunmmioDataPath()(mod)
    mod = tilelang.transform.LayoutReducer()(mod)
    mod = tilelang.transform.SunmmioLayoutInference()(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    return mod


def test_inject_sunmmio_sync_dma():
    M, N = 128, 128
    block_M, block_N = 32, 32
    target = get_target("Sunmmio")
    func = simple_copy_kernel(M, N, block_M, block_N)

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


def test_inject_sunmmio_sync_mma():
    M, N, K = 128, 128, 128
    block_M, block_N, block_K = 32, 32, 32
    target = get_target("Sunmmio")
    func = mma_kernel(M, N, K, block_M, block_N, block_K)

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


def test_inject_sunmmio_sync_broadcast():
    M, N = 128, 128
    block_M, block_N = 32, 32
    target = get_target("Sunmmio")
    func = broadcast_kernel(M, N, block_M, block_N)

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
    assert len(barrier_lines) >= 2
    assert len(wait_lines) >= 3

    # Check instruction order:
    # 1. barrier_init(participant_mask)
    # 2. dma_copy (load A) -> token 0
    # 3. wait_token(0)
    # 4. barrier_arrive_and_wait(participant_mask)
    # 5. broadcast_ -> token 1
    # 6. wait_token(1)
    # 7. barrier_arrive_and_wait(participant_mask)
    # 8. dma_copy (store B) -> token 2
    # 9. wait_token(2)

    idx_dma0 = script.find("sync_token_id(0)")
    idx_wait0 = script.find("wait_token(0)")
    idx_bcast = script.find("broadcast_")
    idx_token1 = script.find("sync_token_id(1)", idx_bcast)  # token 1 should be in broadcast call
    idx_barrier_init = script.find("barrier_init")
    idx_wait1 = script.find("wait_token(1)")
    idx_pre_barrier_wait = script.find("barrier_arrive_and_wait", idx_wait0)
    idx_dma1 = script.find("sync_token_id(2)")
    idx_post_barrier_wait = script.find("barrier_arrive_and_wait", idx_wait1)
    idx_wait2 = script.find("wait_token(2)")

    # Verify order
    assert idx_barrier_init < idx_dma0
    assert idx_dma0 < idx_wait0
    assert idx_wait0 < idx_pre_barrier_wait < idx_bcast
    assert idx_bcast < idx_token1  # token 1 is inside broadcast
    assert idx_barrier_init < idx_wait1
    assert idx_wait1 < idx_post_barrier_wait
    assert idx_post_barrier_wait < idx_dma1
    assert idx_dma1 < idx_wait2

    # Regression (PR #164): broadcast_ carries a receiving mask at arg slot 3,
    # src_offset_byte at slot 4, and optional src_core before the sync token.
    # The barrier parser must decode the bitmask instead of deriving write
    # cores from the offset/source-core slots.
    # A horizontal broadcast from core (0,0) writes the whole mesh row 0 =
    # cores {0,1,2,3}. The reusable barrier is keyed by all participating
    # cores, so read/write masks are merged into a single participant mask.
    barrier_init_lines = [l for l in lines if "barrier_init" in l]
    assert barrier_init_lines, "expected a barrier_init for the broadcast"
    assert _parse_numeric_barrier_mask(barrier_init_lines[0]) == 15
    assert all(_parse_numeric_barrier_mask(line, "barrier_arrive_and_wait") == 15 for line in barrier_lines)


def test_inject_sunmmio_sync_broadcast_without_src_core_full_mesh_barrier():
    target = get_target("Sunmmio")
    mod = _make_leaf_broadcast_without_src_core_mod(target)

    mod = tilelang.transform.InjectSunmmioSync()(mod)
    script = mod.script()

    assert "broadcast_" in script
    assert "barrier_init" in script
    assert "barrier_arrive_and_wait" in script

    lines = [l.strip() for l in script.split("\n")]
    broadcast_lines = [l for l in lines if "broadcast_" in l]
    assert len(broadcast_lines) == 1
    assert "T.sync_token_id(0)" in broadcast_lines[0]
    assert ", 0, T.sync_token_id(0)" in broadcast_lines[0]

    barrier_init_lines = [l for l in lines if "barrier_init" in l]
    assert len(barrier_init_lines) == 1
    assert _parse_numeric_barrier_mask(barrier_init_lines[0]) == (1 << 16) - 1

    barrier_wait_lines = [l for l in lines if "barrier_arrive_and_wait" in l]
    assert len(barrier_wait_lines) == 2
    assert all(_parse_numeric_barrier_mask(line, "barrier_arrive_and_wait") == (1 << 16) - 1 for line in barrier_wait_lines)


def test_inject_sunmmio_sync_dynamic_broadcast_mask_candidates():
    target = get_target("Sunmmio")
    mod = _make_dynamic_row_broadcast_mod(target)

    mod = tilelang.transform.InjectSunmmioSync()(mod)
    script = mod.script()

    lines = [l.strip() for l in script.split("\n")]
    barrier_init_lines = [l for l in lines if "barrier_init(" in l]
    barrier_wait_lines = [l for l in lines if "barrier_arrive_and_wait(" in l]

    assert len(barrier_init_lines) == 1
    assert _parse_barrier_args(barrier_init_lines[0]) == [-1, 15, 240, 3840, 61440]
    assert barrier_wait_lines
    for line in barrier_wait_lines:
        args = _parse_barrier_args(line, "barrier_arrive_and_wait")
        assert args[-4:] == [15, 240, 3840, 61440]


def test_inject_sunmmio_sync_dynamic_pair_mask_candidates():
    target = get_target("Sunmmio")
    mod = _make_dynamic_row_pair_broadcast_mod(target)

    mod = tilelang.transform.InjectSunmmioSync()(mod)
    script = mod.script()

    pair_candidates = [3, 6, 12]
    lines = [l.strip() for l in script.split("\n")]
    barrier_init_lines = [l for l in lines if "barrier_init(" in l]
    barrier_wait_lines = [l for l in lines if "barrier_arrive_and_wait(" in l]

    assert len(barrier_init_lines) == 1
    assert _parse_barrier_args(barrier_init_lines[0]) == [-1] + pair_candidates
    assert barrier_wait_lines
    for line in barrier_wait_lines:
        assert _parse_barrier_args(line, "barrier_arrive_and_wait")[-len(pair_candidates) :] == pair_candidates


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

        return main

    M, N, K = 128, 128, 128
    block_M, block_N, block_K = 32, 32, 32
    target = get_target("Sunmmio")
    func = kernel(M, N, K, block_M, block_N, block_K)

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
    barrier_wait_entries = [(idx, line) for idx, line in enumerate(lines) if "barrier_arrive_and_wait" in line]
    final_store_idx = next(idx for idx, line in enumerate(lines) if "C_shared[0, 0] = C_shared[0, 0] +" in line)

    pre_mma_dma_tokens = [token for idx, _, token in dma_entries if idx < mma_idx]
    pre_mma_wait_tokens = {token for idx, _, token in wait_entries if idx < mma_idx}

    # All DMA loads that happen before the MMA must be waited on first.
    assert set(pre_mma_dma_tokens).issubset(pre_mma_wait_tokens)

    # The conditional branch should wait for the MMA result before broadcasting it.
    branch_wait_indices = [idx for idx, _, token in wait_entries if token == mma_token and if_idx < idx < broadcast_idx]
    assert branch_wait_indices
    assert mma_idx < min(branch_wait_indices) < broadcast_idx

    # The reusable barrier is initialized once at device function entry.
    assert barrier_init_idx < if_idx < broadcast_idx

    # The branch waits on the participant cores before launching the broadcast.
    pre_broadcast_barrier_indices = [idx for idx, _ in barrier_wait_entries if if_idx < idx < broadcast_idx]
    assert pre_broadcast_barrier_indices

    # The broadcast token must be waited on before the outer barrier wait.
    broadcast_wait_indices = [idx for idx, _, token in wait_entries if token == broadcast_token and idx > barrier_init_idx]
    assert broadcast_wait_indices
    post_broadcast_barrier_indices = [idx for idx, _ in barrier_wait_entries if idx > min(broadcast_wait_indices)]
    assert post_broadcast_barrier_indices
    barrier_wait_idx = min(post_broadcast_barrier_indices)
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

        return main

    M, N = 128, 128
    block_M, block_N = 32, 32
    target = get_target("Sunmmio")
    func = kernel(M, N, block_M, block_N)

    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    mod = LowerAndLegalize_sunmmio(mod, target)
    mod = OptimizeForSunmmio_patial(mod, target)

    mod = tilelang.transform.InjectSunmmioSync()(mod)
    script = mod.script(show_meta=True)
    lines = script.splitlines()

    def extract_call_id(line, marker):
        match = re.search(rf"{re.escape(marker)}\((\d+)\)", line)
        assert match, f"Cannot parse {marker} in line: {line}"
        return int(match.group(1))

    broadcast_entries = [
        (idx, line.strip(), extract_call_id(line, "sync_token_id"))
        for idx, line in enumerate(lines)
        if "broadcast_" in line and "sync_token_id(" in line
    ]
    assert len(broadcast_entries) == 2

    first_bcast_idx, first_bcast_line, first_token = broadcast_entries[0]
    second_bcast_idx, second_bcast_line, second_token = broadcast_entries[1]
    assert first_token == 1
    assert second_token == 2
    assert ", 0, 15, 0, 0, T.sync_token_id(1)" in first_bcast_line
    assert ", 0, 15, 0, 0, T.sync_token_id(2)" in second_bcast_line

    wait_entries = [(idx, line.strip(), extract_call_id(line, "wait_token")) for idx, line in enumerate(lines) if "wait_token(" in line]
    barrier_init_entries = [
        (idx, line.strip(), _parse_numeric_barrier_mask(line)) for idx, line in enumerate(lines) if "barrier_init(" in line
    ]
    barrier_wait_entries = [
        (idx, line.strip(), _parse_numeric_barrier_mask(line, "barrier_arrive_and_wait"))
        for idx, line in enumerate(lines)
        if "barrier_arrive_and_wait(" in line
    ]
    assert len(barrier_init_entries) == 1
    assert barrier_init_entries[0][2] == 15
    assert all(mask == 15 for _, _, mask in barrier_wait_entries)

    wait_token_2_before_first = [idx for idx, _, token in wait_entries if token == 2 and idx < first_bcast_idx]
    wait_token_0_before_first = [idx for idx, _, token in wait_entries if token == 0 and idx < first_bcast_idx]
    wait_token_1_between = [idx for idx, _, token in wait_entries if token == 1 and first_bcast_idx < idx < second_bcast_idx]
    assert wait_token_2_before_first
    assert wait_token_0_before_first
    assert wait_token_1_between

    barrier_wait_before_first = [idx for idx, _, _ in barrier_wait_entries if min(wait_token_2_before_first) < idx < first_bcast_idx]
    barrier_wait_between = [idx for idx, _, _ in barrier_wait_entries if min(wait_token_1_between) < idx < second_bcast_idx]
    barrier_wait_after_second = [idx for idx, _, _ in barrier_wait_entries if idx > second_bcast_idx]
    assert barrier_wait_before_first
    assert barrier_wait_between
    assert barrier_wait_after_second


if __name__ == "__main__":
    test_inject_sunmmio_sync_dma()
    test_inject_sunmmio_sync_mma()
    test_inject_sunmmio_sync_broadcast()
    test_inject_sunmmio_sync_broadcast_without_src_core_full_mesh_barrier()
    test_inject_sunmmio_sync_dynamic_broadcast_mask_candidates()
    test_inject_sunmmio_sync_dynamic_pair_mask_candidates()
    test_inject_sunmmio_sync_if()
    test_inject_sunmmio_sync_loop()
