import pytest
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target
import tilelang.language as T


def make_kernel(threads):
    @T.prim_func
    def kernel(A: T.Tensor((128,), T.float32)):
        with T.Kernel(128, threads=threads):
            pass

    return tvm.IRModule({"main": kernel})


def collect_calls(func, op_name):
    calls = []
    op = tvm.tir.op.Op.get(op_name)

    def visit(node):
        if isinstance(node, tvm.tir.Call) and hasattr(node.op, "same_as") and node.op.same_as(op):
            calls.append(node)

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return calls


@pytest.mark.parametrize("requested_threads", [1, 32, 128, 256])
def test_sunmmio_kernel_has_no_threadidx(requested_threads):
    """T.Kernel on Sunmmio emits no threadIdx bindings regardless of the threads= argument.

    Sunmmio is unconditionally threadless: threads=None is forced at the compiler
    level. No threadIdx AttrStmt should appear in the IR.
    """
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = make_kernel(requested_threads)

    script = mod.script()
    assert "threadIdx" not in script, f"Sunmmio kernel must have no threadIdx bindings (threadless). Got:\n{script}"


def test_sunmmio_kernel_default_has_no_threadidx():
    """When threads= is not specified, Sunmmio kernel still emits no threadIdx bindings."""
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):

        @T.prim_func
        def kernel(A: T.Tensor((128,), T.float32)):
            with T.Kernel(128):
                pass

        mod = tvm.IRModule({"main": kernel})

    script = mod.script()
    assert "threadIdx" not in script, f"Sunmmio kernel must have no threadIdx bindings (threadless). Got:\n{script}"


def test_sunmmio_kernel_without_blocks_defaults_to_mesh_ncores():
    """T.Kernel() on Sunmmio uses the symbolic mesh core count as block extent."""
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):

        @T.prim_func
        def kernel(A: T.Tensor((128,), T.float32)):
            with T.Kernel() as cid:
                A[cid] = A[cid]

        mod = tvm.IRModule({"main": kernel})

    script = mod.script()
    assert "threadIdx" not in script, f"Sunmmio kernel must have no threadIdx bindings (threadless). Got:\n{script}"
    assert collect_calls(mod["main"], "tl.mesh_ncores"), f"T.Kernel() should default to symbolic Sunmmio mesh cores. Got:\n{script}"


def test_non_sunmmio_kernel_respects_threads():
    """T.Kernel threads= is not overridden for non-Sunmmio targets."""
    with tvm.target.Target("llvm"):
        mod = make_kernel(128)

    script = mod.script()
    assert 'threadIdx.x", 128)' in script, f"Expected threadIdx.x extent to be 128, but got:\n{script}"
