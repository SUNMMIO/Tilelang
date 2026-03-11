"""Cascaded tests for OptimizeForSunmmio passes on threadless Sunmmio kernels.

Each test function first runs LowerAndLegalize (Phase 1) to produce a correctly
lowered module, then cascades through every OptimizeForSunmmio pass (Phase 2) in
order, asserting threadless invariants after each pass.

Passes run in order (mirroring OptimizeForSunmmio in phase.py):
    01  IfStmtBinding
    --  InjectGreedyPipeline     SKIPPED (under implementation)
    --  InjectAsyncToken         SKIPPED (under implementation)
    02  MergeIfStmt
    03  LowerOpaqueBlock
    04  NarrowDataType
    05  FlattenBuffer
    06  ConfigIndexBitwidth
    07  Simplify[1]
    08  VectorizeLoop
    09  UnrollLoop
    10  Simplify[2]
    11  RemoveNoOp
    12  HoistIfThenElse
    13  AnnotateDeviceRegions       ← splits the IR into host/device regions
    14  SplitHostDevice             ← mod["main"] becomes host; device func extracted
    --  MergeSRAMAllocations     SKIPPED (under implementation)
    15  MakePackedAPI
    16  LowerDeviceKernelLaunch

Assertion strategy:
    - Passes 01-13: assert on mod["main"] — no threadIdx, no v_thread, blockIdx present
    - Pass 14 (SplitHostDevice) onward: assert on the device function (identified
      by its target attribute having no host)

Kernel variants covered:
    1. Simple element-wise copy    — no shared memory, no T.Parallel
    2. T.Parallel + shared memory  — original ThreadBindingCollector bug trigger
    3. 3-D grid (bx, by, bz)       — multi-dimensional grid with arithmetic

Note on target context: VectorizePlanner::Plan inside LayoutInference calls
Target::Current() at runtime. All test functions run entirely inside a
`with tvm.target.Target(target):` block to keep the context active.
"""

import tilelang
import tilelang.language as T
import tilelang.transform as tl_transform
from tilelang import tvm as tvm
from tilelang.engine.lower import canon_target_host
from tilelang.engine.phase import LowerAndLegalize
from tilelang.utils.target import determine_target
from tvm import tir
from tvm.tir.stmt_functor import post_order_visit

tilelang.env.disable_cache()

# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------


def make_elementwise_copy_kernel(M, N, dtype=T.float32):

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(M, N) as (bx, by):
            B[bx, by] = A[bx, by]

    return tvm.IRModule({"main": main})


def make_parallel_shared_kernel(M, N, dtype=T.float32):

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, 32), T.ceildiv(N, 32)) as (bx, by):
            A_shared = T.alloc_shared((32, 32), dtype)
            for i, j in T.Parallel(32, 32):
                A_shared[i, j] = A[bx * 32 + i, by * 32 + j]
            for i, j in T.Parallel(32, 32):
                B[bx * 32 + i, by * 32 + j] = A_shared[i, j]

    return tvm.IRModule({"main": main})


def make_3d_grid_kernel(M, N, K, dtype=T.float32):

    @T.prim_func
    def main(A: T.Tensor((M, N, K), dtype), B: T.Tensor((M, N, K), dtype)):
        with T.Kernel(M, N, K) as (bx, by, bz):
            B[bx, by, bz] = A[bx, by, bz] + T.float32(1.0)

    return tvm.IRModule({"main": main})


def make_sunmmio_target_with_host():
    target = determine_target("Sunmmio", return_object=True)
    target_host = tvm.target.Target.canon_target(canon_target_host(target, None))
    return tvm.target.Target(target, target_host)


# ---------------------------------------------------------------------------
# IR inspection helpers
# ---------------------------------------------------------------------------


def collect_thread_extents(func):
    extents = {}

    def fvisit(node):
        if isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent":
            extents[node.node.thread_tag] = int(node.value)

    post_order_visit(func.body, fvisit)
    return extents


def collect_var_names(func):
    names = set()

    def fvisit(node):
        if isinstance(node, tir.Var):
            names.add(node.name)

    post_order_visit(func.body, fvisit)
    return names


def get_device_func(mod):
    """Return the device PrimFunc: the one whose target attribute has no host."""
    for func in mod.functions.values():
        if isinstance(func, tir.PrimFunc) and func.attrs is not None and "target" in func.attrs and not func.attrs["target"].host:
            return func
    return None


def assert_threadless_invariants(func, after_pass):
    """Assert all three threadless invariants on a PrimFunc.

    - No threadIdx.* bindings introduced
    - No free v_thread variable
    - blockIdx.* bindings still present
    """
    extents = collect_thread_extents(func)
    var_names = collect_var_names(func)

    threadidx_tags = [t for t in extents if t.startswith("threadIdx")]
    blockidx_tags = [t for t in extents if t.startswith("blockIdx")]

    assert not threadidx_tags, (
        f"After {after_pass}: threadIdx bindings must not exist in a threadless "
        f"Sunmmio kernel. Found: {threadidx_tags}\nAll extents: {extents}\n"
        f"IR:\n{func.script()}"
    )

    assert "v_thread" not in var_names, (
        f"After {after_pass}: free v_thread variable found. A pass incorrectly "
        f"treated blockIdx as a thread binding and called PartitionLoop.\n"
        f"All vars: {sorted(var_names)}\nIR:\n{func.script()}"
    )

    assert blockidx_tags, (
        f"After {after_pass}: blockIdx bindings must be preserved in a threadless "
        f"Sunmmio kernel. Found extents: {extents}\nIR:\n{func.script()}"
    )


# ---------------------------------------------------------------------------
# Cascaded pipeline helper
# ---------------------------------------------------------------------------


def run_optimize_for_sunmmio_cascade(mod, target):
    """Run every implemented pass in OptimizeForSunmmio in order, asserting
    threadless invariants after each one.

    Pre-split assertions target mod["main"].
    Post-split assertions target the device function (target attr with no host).

    Skipped passes (under implementation):
        InjectGreedyPipeline, InjectAsyncToken, MergeSRAMAllocations
    """
    # --- Pre-split phase: assert on mod["main"] ---

    mod = tl_transform.IfStmtBinding()(mod)
    assert_threadless_invariants(mod["main"], "IfStmtBinding")

    # InjectGreedyPipeline  -- SKIPPED (under implementation)
    # InjectAsyncToken      -- SKIPPED (under implementation)

    mod = tl_transform.MergeIfStmt()(mod)
    assert_threadless_invariants(mod["main"], "MergeIfStmt")

    mod = tl_transform.LowerOpaqueBlock()(mod)
    assert_threadless_invariants(mod["main"], "LowerOpaqueBlock")

    mod = tir.transform.NarrowDataType(32)(mod)
    assert_threadless_invariants(mod["main"], "NarrowDataType")

    mod = tl_transform.FlattenBuffer()(mod)
    assert_threadless_invariants(mod["main"], "FlattenBuffer")

    mod = tl_transform.ConfigIndexBitwidth()(mod)
    assert_threadless_invariants(mod["main"], "ConfigIndexBitwidth")

    mod = tir.transform.Simplify()(mod)
    assert_threadless_invariants(mod["main"], "Simplify[1]")

    mod = tl_transform.VectorizeLoop(enable_vectorize=True)(mod)
    assert_threadless_invariants(mod["main"], "VectorizeLoop")

    mod = tir.transform.UnrollLoop()(mod)
    assert_threadless_invariants(mod["main"], "UnrollLoop")

    mod = tir.transform.Simplify()(mod)
    assert_threadless_invariants(mod["main"], "Simplify[2]")

    mod = tir.transform.RemoveNoOp()(mod)
    assert_threadless_invariants(mod["main"], "RemoveNoOp")

    mod = tir.transform.HoistIfThenElse()(mod)
    assert_threadless_invariants(mod["main"], "HoistIfThenElse")

    mod = tl_transform.AnnotateDeviceRegions()(mod)
    assert_threadless_invariants(mod["main"], "AnnotateDeviceRegions")

    # --- Host/device split: mod["main"] becomes the host launcher ---

    mod = tl_transform.SplitHostDevice()(mod)
    device_func = get_device_func(mod)
    assert device_func is not None, (
        "After SplitHostDevice: expected a device function (target attr with no host), "
        f"found none. Functions: {[str(gv) for gv in mod.functions]}"
    )
    assert_threadless_invariants(device_func, "SplitHostDevice")

    # MergeSRAMAllocations  -- SKIPPED (under implementation)

    # --- Post-split phase: assert on device function ---

    mod = tl_transform.MakePackedAPI()(mod)
    device_func = get_device_func(mod)
    assert device_func is not None, f"After MakePackedAPI: device function lost. Functions: {[str(gv) for gv in mod.functions]}"
    assert_threadless_invariants(device_func, "MakePackedAPI")

    mod = tl_transform.LowerDeviceKernelLaunch()(mod)
    # LowerDeviceKernelLaunch rewrites the host launcher; device body is unchanged.
    device_func = get_device_func(mod)
    assert device_func is not None, f"After LowerDeviceKernelLaunch: device function lost. Functions: {[str(gv) for gv in mod.functions]}"
    assert_threadless_invariants(device_func, "LowerDeviceKernelLaunch")

    return mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_optimize_for_sunmmio_cascade_elementwise_copy():
    """All OptimizeForSunmmio passes must preserve threadless invariants for a copy kernel.

    Runs LowerAndLegalize first (Phase 1), then cascades every implemented
    OptimizeForSunmmio pass (Phase 2) asserting after each one:
      - no threadIdx bindings injected
      - no free v_thread variable introduced
      - blockIdx bindings preserved (pre-split: main; post-split: device func)
    """
    target = make_sunmmio_target_with_host()
    device_target = determine_target("Sunmmio", return_object=True)

    with tvm.target.Target(target):
        mod = make_elementwise_copy_kernel(16, 16)
        mod = LowerAndLegalize(mod, device_target)
        run_optimize_for_sunmmio_cascade(mod, target)


def test_optimize_for_sunmmio_cascade_parallel_shared():
    """All OptimizeForSunmmio passes must preserve threadless invariants for a
    T.Parallel + shared memory kernel — the original ThreadBindingCollector bug trigger.

    Runs LowerAndLegalize first (Phase 1), then cascades every implemented
    OptimizeForSunmmio pass (Phase 2) asserting after each one:
      - no threadIdx bindings injected
      - no free v_thread variable introduced
      - blockIdx bindings preserved (pre-split: main; post-split: device func)
    """
    target = make_sunmmio_target_with_host()
    device_target = determine_target("Sunmmio", return_object=True)

    with tvm.target.Target(target):
        mod = make_parallel_shared_kernel(64, 64)
        mod = LowerAndLegalize(mod, device_target)
        run_optimize_for_sunmmio_cascade(mod, target)


def test_optimize_for_sunmmio_cascade_3d_grid():
    """All OptimizeForSunmmio passes must preserve threadless invariants for a
    3-D grid kernel.

    Runs LowerAndLegalize first (Phase 1), then cascades every implemented
    OptimizeForSunmmio pass (Phase 2) asserting after each one:
      - no threadIdx bindings injected
      - no free v_thread variable introduced
      - blockIdx bindings preserved (pre-split: main; post-split: device func)
    """
    target = make_sunmmio_target_with_host()
    device_target = determine_target("Sunmmio", return_object=True)

    with tvm.target.Target(target):
        mod = make_3d_grid_kernel(4, 4, 4)
        mod = LowerAndLegalize(mod, device_target)
        run_optimize_for_sunmmio_cascade(mod, target)


if __name__ == "__main__":
    test_optimize_for_sunmmio_cascade_elementwise_copy()
    test_optimize_for_sunmmio_cascade_parallel_shared()
    test_optimize_for_sunmmio_cascade_3d_grid()
    print("All tests passed.")
