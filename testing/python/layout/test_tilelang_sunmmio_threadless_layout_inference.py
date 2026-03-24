"""Tests that LayoutInference handles threadless Sunmmio kernels correctly.

Background — the ThreadBindingCollector fix (layout_inference.cc):

  Before the fix, ThreadBindingCollector collected *all* thread_extent bindings,
  including blockIdx.*. On Sunmmio, T.Kernel always forces threads=None (threadless),
  so the IR only ever has blockIdx.* bindings — never threadIdx.*. The collector
  saw blockIdx and set has_thread_binding=True, so skip_thread_partition was False.

  When such a kernel also had T.Parallel loops accessing shared memory,
  BufferUseDefCollector encountered those parallel ForNodes. Because no threadIdx
  AttrStmt was present, thread_var_ was never updated from its default dummy IterVar
  (named "v_thread", range [0,1)). PartitionLoop was then called with that unbound
  dummy variable, producing incorrect IR (free variable v_thread in the output).

  After the fix, only threadIdx.* bindings are collected. A pure-blockIdx kernel
  correctly yields has_thread_binding=False → skip_thread_partition=True →
  PartitionLoop is never called → correct threadless IR.

  Note: on Sunmmio, T.Kernel unconditionally sets threads=None regardless of any
  explicit threads= argument. There is no "threaded Sunmmio" path; all Sunmmio
  kernels are threadless.

Regression coverage:
  1. Threadless kernel without T.Parallel — basic smoke tests.
  2. Threadless kernel WITH T.Parallel + shared memory — the exact bug trigger,
     verifying that PartitionLoop is not called with the dummy v_thread variable.
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


def make_threadless_copy_kernel(M, N, dtype=T.float32):
    """Threadless element-wise copy — no T.Parallel, no shared memory."""

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(M, N) as (bx, by):
            B[bx, by] = A[bx, by]

    return tvm.IRModule({"main": main})


def make_sunmmio_parallel_shared_kernel(M, N, dtype=T.float32):
    """Threadless Sunmmio kernel with T.Parallel + shared memory.

    This is the exact trigger for the ThreadBindingCollector bug:
    - blockIdx.* bindings are present (from T.Kernel)
    - threadIdx.* bindings are absent (threads=None on Sunmmio)
    - T.Parallel produces parallel ForNodes accessing shared memory
    Before the fix, blockIdx was collected as a "thread binding", causing
    PartitionLoop to be called with the unbound dummy v_thread variable.
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


# ---------------------------------------------------------------------------
# IR inspection helpers
# ---------------------------------------------------------------------------


def collect_var_names(func):
    """Return names of all Var nodes reachable from the function body."""
    names = set()

    def fvisit(node):
        if isinstance(node, tir.Var):
            names.add(node.name)

    post_order_visit(func.body, fvisit)
    return names


def collect_thread_extents(func):
    """Return {thread_tag: extent} for every thread_extent AttrStmt in the body."""
    extents = {}

    def fvisit(node):
        if isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent":
            extents[node.node.thread_tag] = int(node.value)

    post_order_visit(func.body, fvisit)
    return extents


def has_parallel_for(func):
    """Return True if any T.parallel ForNode survives in the body."""
    found = []

    def fvisit(node):
        if isinstance(node, tir.For) and node.kind == tir.ForKind.PARALLEL:
            found.append(node)

    post_order_visit(func.body, fvisit)
    return bool(found)


# ---------------------------------------------------------------------------
# Tests — threadless path (regression for the ThreadBindingCollector fix)
# ---------------------------------------------------------------------------


def test_layout_inference_threadless_kernel_completes_without_error():
    """LayoutInference must not crash on a threadless Sunmmio kernel.

    This is the base smoke test: even without T.Parallel or shared memory,
    a threadless kernel (blockIdx only, no threadIdx) must pass through
    LayoutInference cleanly.

    Before the fix, blockIdx was collected as a "thread binding", setting
    skip_thread_partition=False. For kernels with parallel for loops + shared
    memory this caused PartitionLoop to be called with an unbound dummy thread
    variable, producing incorrect IR or an ICHECK failure.
    """
    target = determine_target("Sunmmio", return_object=True)

    with tvm.target.Target(target):
        mod = make_threadless_copy_kernel(16, 16)

    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tl_transform.InferSramScope()(mod)
    mod = tl_transform.LayoutInference()(mod)  # must not crash or raise


def test_layout_inference_threadless_kernel_has_no_threadidx_bindings():
    """After LayoutInference a threadless kernel must carry no threadIdx bindings.

    The threadless execution model (skip_thread_partition=True) must be preserved
    end-to-end. If blockIdx were still collected as a thread binding the pass
    might inject spurious threadIdx annotations.
    """
    target = determine_target("Sunmmio", return_object=True)

    with tvm.target.Target(target):
        mod = make_threadless_copy_kernel(16, 16)

    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tl_transform.InferSramScope()(mod)
    mod = tl_transform.LayoutInference()(mod)

    thread_extents = collect_thread_extents(mod["main"])
    threadidx_tags = [t for t in thread_extents if t.startswith("threadIdx")]

    assert not threadidx_tags, (
        f"Threadless Sunmmio kernel must have no threadIdx bindings after LayoutInference. "
        f"Found: {thread_extents}\n"
        f"IR:\n{mod['main'].script()}"
    )


def test_layout_inference_threadless_kernel_preserves_blockidx_bindings():
    """After LayoutInference, blockIdx bindings must still be present.

    Guards against the fix accidentally stripping blockIdx from the IR.
    """
    target = determine_target("Sunmmio", return_object=True)

    with tvm.target.Target(target):
        mod = make_threadless_copy_kernel(16, 16)

    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tl_transform.InferSramScope()(mod)
    mod = tl_transform.LayoutInference()(mod)

    thread_extents = collect_thread_extents(mod["main"])
    blockidx_tags = [t for t in thread_extents if t.startswith("blockIdx")]

    assert blockidx_tags, (
        f"Threadless Sunmmio kernel must still have blockIdx bindings after LayoutInference. "
        f"Found extents: {thread_extents}\n"
        f"IR:\n{mod['main'].script()}"
    )


# ---------------------------------------------------------------------------
# Tests — T.Parallel + shared memory (the actual ThreadBindingCollector bug trigger)
# ---------------------------------------------------------------------------


def test_layout_inference_parallel_shared_kernel_completes_without_error():
    """LayoutInference must not crash on a Sunmmio kernel with T.Parallel + shared memory.

    This is the exact scenario that triggered the ThreadBindingCollector bug:
    - Only blockIdx.* bindings (no threadIdx on Sunmmio)
    - T.Parallel for loops accessing shared memory → BufferUseDefCollector builds
      ParallelOpNode entries, and in Run() retrieves thread_var for each

    Before the fix: blockIdx collected → skip_thread_partition=False →
    PartitionLoop called with unbound dummy v_thread → incorrect IR or crash.
    After the fix: blockIdx ignored → skip_thread_partition=True →
    PartitionLoop skipped → clean threadless IR.
    """
    target = determine_target("Sunmmio", return_object=True)

    with tvm.target.Target(target):
        mod = make_sunmmio_parallel_shared_kernel(64, 64)
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tl_transform.InferSramScope()(mod)
        mod = tl_transform.LayoutInference()(mod)  # must not crash or raise


def test_layout_inference_parallel_shared_kernel_has_no_threadidx_bindings():
    """After LayoutInference a Sunmmio T.Parallel kernel must have no threadIdx bindings.

    Sunmmio is unconditionally threadless; LayoutInference must not inject threadIdx
    when processing T.Parallel loops. If blockIdx were still collected as a thread
    binding the pass might introduce spurious threadIdx annotations.
    """
    target = determine_target("Sunmmio", return_object=True)

    with tvm.target.Target(target):
        mod = make_sunmmio_parallel_shared_kernel(64, 64)
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tl_transform.InferSramScope()(mod)
        mod = tl_transform.LayoutInference()(mod)

    thread_extents = collect_thread_extents(mod["main"])
    threadidx_tags = [t for t in thread_extents if t.startswith("threadIdx")]

    assert not threadidx_tags, (
        f"Sunmmio kernel with T.Parallel must have no threadIdx bindings after "
        f"LayoutInference (Sunmmio is fully threadless). Found: {thread_extents}\n"
        f"IR:\n{mod['main'].script()}"
    )


def test_layout_inference_parallel_shared_kernel_has_no_v_thread_variable():
    """After LayoutInference the IR must not contain a free v_thread variable.

    v_thread is the dummy IterVar used by PartitionLoop when no real thread binding
    exists. If it appears in the output IR it means PartitionLoop was called with
    an unbound variable — the exact bug this fix addresses.

    Trigger condition: T.Parallel loop + shared memory + no threadIdx binding.
    Before fix: blockIdx collected → PartitionLoop(v_thread) → v_thread in IR.
    After fix: skip_thread_partition=True → PartitionLoop not called → no v_thread.
    """
    target = determine_target("Sunmmio", return_object=True)

    with tvm.target.Target(target):
        mod = make_sunmmio_parallel_shared_kernel(64, 64)
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tl_transform.InferSramScope()(mod)
        mod = tl_transform.LayoutInference()(mod)

    var_names = collect_var_names(mod["main"])

    assert "v_thread" not in var_names, (
        "LayoutInference introduced a free v_thread variable for a Sunmmio kernel "
        "with T.Parallel. This means ThreadBindingCollector incorrectly collected "
        "blockIdx.* as a thread binding, causing PartitionLoop to run with an "
        "unbound dummy thread variable.\n"
        f"All vars found: {sorted(var_names)}\n"
        f"IR:\n{mod['main'].script()}"
    )


if __name__ == "__main__":
    test_layout_inference_threadless_kernel_completes_without_error()
    test_layout_inference_threadless_kernel_has_no_threadidx_bindings()
    test_layout_inference_threadless_kernel_preserves_blockidx_bindings()
    test_layout_inference_parallel_shared_kernel_completes_without_error()
    test_layout_inference_parallel_shared_kernel_has_no_threadidx_bindings()
    test_layout_inference_parallel_shared_kernel_has_no_v_thread_variable()
    print("All tests passed.")
