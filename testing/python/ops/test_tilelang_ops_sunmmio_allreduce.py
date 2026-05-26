import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from tvm import tir

tilelang.env.disable_cache()


def apply_sunmmio_passes(mod, target):
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


def allreduce_kernel(direction="all", clear=True, dtype="float32"):
    shape = (32, 32)
    out_shape = (32,)

    @T.prim_func
    def main(A: T.Tensor(shape, dtype), Out: T.Tensor(out_shape, dtype)):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")

            T.copy(A, A_shared)
            if not clear:
                T.copy(Out, Out_shared)
            T.comm.all_reduce(A_shared, Out_shared, "sum", direction, dim=1, clear=clear)
            T.copy(Out_shared, Out)

    return tvm.IRModule({"main": main})


@tir.functor.visitor
class AllreduceIRChecker(tir.PyStmtExprVisitor):
    def __init__(self):
        super().__init__()
        self.broadcast_count = 0
        self.in_tile_reduce_count = 0
        self.rsram_alloc_names = []

    def visit_block_(self, op):
        for buf in op.alloc_buffers:
            if buf.scope() == "shared.rsram":
                self.rsram_alloc_names.append(buf.name)
        super().visit_block_(op)

    def visit_call_(self, op):
        if hasattr(op, "op") and hasattr(op.op, "name"):
            if op.op.name == "tl.broadcast_":
                self.broadcast_count += 1
            elif op.op.name == "tl.vector_core_in_tile_reduce":
                self.in_tile_reduce_count += 1
        super().visit_call_(op)


def test_tilelang_allreduce_frontend_allocates_rsram_temporaries():
    script = allreduce_kernel(direction="all", clear=False).script()

    assert 'buffer = T.alloc_buffer((4, 32), scope="shared.rsram")' in script
    assert 'buffer_1 = T.alloc_buffer((4, 32), scope="shared.rsram")' in script
    assert 'buffer_2 = T.alloc_buffer((32,), scope="shared.rsram")' in script
    assert (
        "T.comm_allreduce(A_shared[0:32, 0:32], Out_shared[0:32], "
        'buffer[0:4, 0:32], buffer_1[0:4, 0:32], "sum", 2, 1, '
        "T.bool(False), buffer_2[0:32], bx)"
    ) in script


@pytest.mark.parametrize(
    "direction, clear, expected_broadcasts",
    [
        ("h", True, 1),
        ("v", True, 1),
        ("all", True, 2),
        ("h", False, 1),
        ("v", False, 1),
        ("all", False, 2),
    ],
)
def test_tilelang_allreduce_sunmmio_lowers_to_broadcast_and_tile_reduce(direction, clear, expected_broadcasts):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = allreduce_kernel(direction=direction, clear=clear)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    checker = AllreduceIRChecker()
    checker.visit_stmt(mod["main"].body)

    assert checker.broadcast_count == expected_broadcasts
    assert checker.in_tile_reduce_count >= 2
