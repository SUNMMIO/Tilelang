"""End-to-end cascaded tests for LowerAndLegalize on threadless Sunmmio kernels.

Each test function runs the passes in LowerAndLegalize **in sequence**, asserting
the threadless invariants (no threadIdx, no v_thread, blockIdx preserved) after
every individual pass. When a pass breaks a threadless kernel, the assertion
failure names exactly which pass is responsible.

Passes run in order (mirroring phase.py LowerAndLegalize):
    01  BindTarget
    02  AddWrapperForSingleBufStore
    03  LegalizeNegativeIndex
    04  InjectAssumes
    05  Simplify
    06  InferSramScope
    07  LayoutReducer
    08  LayoutInference
    09  LowerTileOp
    10  LowerL2Persistent
    11  LegalizeVectorizedLoop
    12  LegalizeSafeMemoryAccess
    13  Simplify (2nd)

Kernel variants covered:
    1. Simple element-wise copy    — no shared memory, no T.Parallel
    2. T.Parallel + shared memory  — the original ThreadBindingCollector bug trigger
    3. Multi-block element-wise    — 3-D grid (bx, by, bz) with arithmetic

Note on target context: VectorizePlanner::Plan (called from ParallelOpNode::InferLayout
inside LayoutInference) calls Target::Current() at runtime. For kernels that use
T.Parallel, the target context must remain active for the duration of the pass
cascade. All test functions therefore run entirely inside a with tvm.target.Target block.
"""

import tilelang
import tilelang.language as T
import tilelang.transform as tl_transform
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target
from tvm import tir
from tvm.tir.stmt_functor import post_order_visit

tilelang.env.disable_cache()

# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------


def make_elementwise_copy_kernel(M, N, dtype=T.float32):
    """Simplest threadless kernel: copy A → B element-by-element."""

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(M, N) as (bx, by):
            B[bx, by] = A[bx, by]

    return tvm.IRModule({"main": main})


def make_parallel_shared_kernel(M, N, dtype=T.float32):
    """Threadless kernel with T.Parallel loops and shared memory.

    This is the exact scenario that previously triggered the
    ThreadBindingCollector bug in LayoutInference.
    """

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
    """Threadless kernel with a 3-D grid (bx, by, bz) and elementwise arithmetic."""

    @T.prim_func
    def main(A: T.Tensor((M, N, K), dtype), B: T.Tensor((M, N, K), dtype)):
        with T.Kernel(M, N, K) as (bx, by, bz):
            B[bx, by, bz] = A[bx, by, bz] + T.float32(1.0)

    return tvm.IRModule({"main": main})


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


def assert_threadless_invariants(mod, after_pass):
    """Assert all three threadless invariants on mod["main"].

    - No threadIdx.* bindings introduced
    - No free v_thread variable
    - blockIdx.* bindings still present
    """
    func = mod["main"]
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
        f"treated blockIdx as a thread binding and called PartitionLoop with the "
        f"unbound dummy thread variable.\nAll vars: {sorted(var_names)}\n"
        f"IR:\n{func.script()}"
    )

    assert blockidx_tags, (
        f"After {after_pass}: blockIdx bindings must be preserved in a threadless "
        f"Sunmmio kernel. Found extents: {extents}\nIR:\n{func.script()}"
    )


# ---------------------------------------------------------------------------
# Cascaded pipeline helper
# ---------------------------------------------------------------------------


def run_lower_and_legalize_cascade(mod, target):
    """Run every pass in LowerAndLegalize in order, asserting threadless invariants
    after each one. Returns the final transformed module.

    Mirrors the pass sequence in tilelang/engine/phase.py LowerAndLegalize.
    """
    mod = tir.transform.BindTarget(target)(mod)
    assert_threadless_invariants(mod, "BindTarget")

    mod = tl_transform.AddWrapperForSingleBufStore()(mod)
    assert_threadless_invariants(mod, "AddWrapperForSingleBufStore")

    mod = tl_transform.LegalizeNegativeIndex()(mod)
    assert_threadless_invariants(mod, "LegalizeNegativeIndex")

    mod = tl_transform.InjectAssumes()(mod)
    assert_threadless_invariants(mod, "InjectAssumes")

    mod = tl_transform.Simplify()(mod)
    assert_threadless_invariants(mod, "Simplify[1]")

    mod = tl_transform.InferSramScope()(mod)
    assert_threadless_invariants(mod, "InferSramScope")

    mod = tl_transform.LayoutReducer()(mod)
    assert_threadless_invariants(mod, "LayoutReducer")

    mod = tl_transform.LayoutInference()(mod)
    assert_threadless_invariants(mod, "LayoutInference")

    mod = tl_transform.LowerTileOp()(mod)
    assert_threadless_invariants(mod, "LowerTileOp")

    mod = tl_transform.LowerL2Persistent()(mod)
    assert_threadless_invariants(mod, "LowerL2Persistent")

    mod = tl_transform.LegalizeVectorizedLoop()(mod)
    assert_threadless_invariants(mod, "LegalizeVectorizedLoop")

    mod = tl_transform.LegalizeSafeMemoryAccess()(mod)
    assert_threadless_invariants(mod, "LegalizeSafeMemoryAccess")

    mod = tl_transform.Simplify()(mod)
    assert_threadless_invariants(mod, "Simplify[2]")

    return mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_lower_and_legalize_cascade_elementwise_copy():
    """All passes in LowerAndLegalize must preserve threadless invariants for a copy kernel.

    Checks after every pass:
      - no threadIdx bindings injected
      - no free v_thread variable introduced
      - blockIdx bindings preserved
    """
    target = determine_target("Sunmmio", return_object=True)

    with tvm.target.Target(target):
        mod = make_elementwise_copy_kernel(16, 16)

    run_lower_and_legalize_cascade(mod, target)


def test_lower_and_legalize_cascade_parallel_shared():
    """All passes in LowerAndLegalize must preserve threadless invariants for a
    T.Parallel + shared memory kernel — the original ThreadBindingCollector bug trigger.

    Checks after every pass:
      - no threadIdx bindings injected
      - no free v_thread variable introduced
      - blockIdx bindings preserved

    Note: target context must stay active throughout because VectorizePlanner::Plan
    (inside LayoutInference) calls Target::Current() at runtime.
    """
    target = determine_target("Sunmmio", return_object=True)

    with tvm.target.Target(target):
        mod = make_parallel_shared_kernel(64, 64)
        run_lower_and_legalize_cascade(mod, target)


def test_lower_and_legalize_cascade_3d_grid():
    """All passes in LowerAndLegalize must preserve threadless invariants for a
    3-D grid kernel.

    Checks after every pass:
      - no threadIdx bindings injected
      - no free v_thread variable introduced
      - blockIdx bindings preserved
    """
    target = determine_target("Sunmmio", return_object=True)

    with tvm.target.Target(target):
        mod = make_3d_grid_kernel(4, 4, 4)

    run_lower_and_legalize_cascade(mod, target)


if __name__ == "__main__":
    test_lower_and_legalize_cascade_elementwise_copy()
    test_lower_and_legalize_cascade_parallel_shared()
    test_lower_and_legalize_cascade_3d_grid()
    print("All tests passed.")
