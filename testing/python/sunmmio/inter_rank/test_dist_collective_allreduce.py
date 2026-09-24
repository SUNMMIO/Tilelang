"""Rank all-reduce frontend and gather-reduce lowering tests."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import (
    collect_op_names,
    lower_collectives,
)


@tilelang.jit(target="sunmmio")
def allreduce_kernel_factory(reduce_type="sum", group=None, signal_kind=None, expect_mode=None, compute_loops=False, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.bfloat16)
            dst = T.alloc_shared((4, 8), T.bfloat16)
            signal = T.dist.signal(kind=signal_kind, expect=expect_mode)
            if compute_loops:
                for i in T.serial(4):
                    src[i, 0] = 0
            T.dist.all_reduce(src, dst, reduce_type, signal=signal, group=group)
            if compute_loops:
                for i in T.serial(4):
                    dst[i, 0] += 1

    return main


@tilelang.jit(target="sunmmio")
def repeated_allreduce_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.float32)
            dst = T.alloc_shared((4, 8), T.float32)
            sum_signal = T.dist.signal()
            max_signal = T.dist.signal()
            T.dist.all_reduce(src, dst, "sum", signal=sum_signal)
            T.dist.all_reduce(src, dst, "max", signal=max_signal)

    return main


@tilelang.jit(target="sunmmio")
def loop_allreduce_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.bfloat16)
            dst = T.alloc_shared((4, 8), T.bfloat16)
            signal = T.dist.signal()
            for _ in T.serial(3):
                T.dist.all_reduce(src, dst, signal=signal)
                src[0, 0] = dst[0, 0]

    return main


@tilelang.jit(target="sunmmio")
def allreduce_loop_boundary_kernel_factory(
    loop_kind="serial",
    iterations=2,
    distinct_members=False,
    world_size: int = 1,
):
    @T.prim_func
    def main(rank_id: T.dist.RankId, rounds: T.int32):
        with T.Kernel() as core_id:
            src = T.alloc_shared((4, 8), T.bfloat16)
            dst = T.alloc_shared((4, 8), T.bfloat16)
            signals = T.dist.signals(2, kind=T.dist.SignalKind.SRAM_FLAGREG_INC)
            step = T.alloc_var(T.int32, init=0)
            if loop_kind == "serial":
                for i in T.serial(iterations):
                    T.dist.all_reduce(src, dst, signal=signals[i] if distinct_members else signals[0])
            elif loop_kind == "unroll":
                for i in T.unroll(iterations):
                    T.dist.all_reduce(src, dst, signal=signals[i] if distinct_members else signals[0])
            elif loop_kind == "dynamic":
                for _ in T.serial(rounds):
                    T.dist.all_reduce(src, dst, signal=signals[0])
            elif loop_kind == "nested":
                for _ in T.serial(iterations):
                    if core_id % T.mesh_ncols() == 0:
                        while step < iterations:
                            T.dist.all_reduce(src, dst, signal=signals[0])
                            step += 1
            else:
                while step < iterations:
                    T.dist.all_reduce(src, dst, signal=signals[0])
                    step += 1

    return main


@tilelang.jit(target="sunmmio")
def mismatched_allreduce_shape_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.bfloat16)
            dst = T.alloc_shared((4, 7), T.bfloat16)
            signal = T.dist.signal()
            T.dist.all_reduce(src, dst, signal=signal)

    return main


@tilelang.jit(target="sunmmio")
def mismatched_allreduce_dtype_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.bfloat16)
            dst = T.alloc_shared((4, 8), T.float32)
            signal = T.dist.signal()
            T.dist.all_reduce(src, dst, signal=signal)

    return main


@tilelang.jit(target="sunmmio")
def overlapping_allreduce_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            data = T.alloc_shared((4, 8), T.bfloat16)
            signal = T.dist.signal()
            T.dist.all_reduce(data, data, signal=signal)

    return main


@tilelang.jit(target="sunmmio")
def global_allreduce_source_kernel_factory(world_size: int = 1):
    placement = T.placement.replicated()

    @T.prim_func
    def main(
        src: T.MeshTensor((4, 8), placement, T.bfloat16),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            dst = T.alloc_shared((4, 8), T.bfloat16)
            signal = T.dist.signal()
            T.dist.all_reduce(src, dst, signal=signal)

    return main


@pytest.mark.parametrize("reduce_type", ["sum", "max", "min", "SUM"])
def test_allreduce_frontend_normalizes_reduce_type_and_requires_signal(reduce_type):
    func = allreduce_kernel_factory.get_tir(reduce_type=reduce_type, world_size=4)
    names = collect_op_names(func)

    assert names.count("tl.tileop.dist_allreduce") == 1
    assert names.count("tl.dist_signal_decl") == 1
    assert "tl.tileop.dist_put" not in names
    expected = "sum" if reduce_type == "SUM" else reduce_type
    assert f'"{expected}"' in func.script()


@pytest.mark.parametrize("reduce_type", ["sum", "max", "min"])
def test_lower_dist_collectives_builds_gather_then_local_reduce(reduce_type):
    func = allreduce_kernel_factory.get_tir(reduce_type=reduce_type, world_size=4)
    lowered = lower_collectives(func)
    names = collect_op_names(lowered)
    script = lowered.script()

    assert "tl.tileop.dist_allreduce" not in names
    assert names.count("tl.tileop.copy") == 1
    assert names.count("tl.tileop.dist_put") == 3
    assert names.count("tl.dist_wait_signal") == 1
    assert names.count("tl.tileop.reduce") == 1
    assert names.count("tl.dist_submit") == 1
    assert names.count("tl.dist_wait_send") == 1
    assert "dist_allreduce_gather_0 = T.alloc_buffer((4, 4, 8)" in script
    assert f'), "{reduce_type}", 0, T.bool(True))' in script
    assert script.index("T.dist_wait_signal") < script.index("T.reduce(")
    assert script.index("T.reduce(") < script.index("T.dist_wait_send()")

    lowered_again = tilelang.transform.LowerDistCollectives()(lowered)
    assert tvm.ir.structural_equal(lowered_again, lowered)


@pytest.mark.parametrize("reduce_type", ["sum", "max", "min"])
def test_allreduce_reaches_device_tir(reduce_type):
    func = allreduce_kernel_factory.get_tir(reduce_type=reduce_type, world_size=4)
    result = lower_to_device_tir(
        func,
        capture_passes=("tl.LowerDistCollectives", "tl.PlanDistSignals"),
    )

    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    device_names = collect_op_names(result.device_mod)
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 1
    assert device_names.count("tl.dist_put_") == 3
    assert device_names.count("tl.dist_wait_signal_") == 1
    assert device_names.count("tl.dist_wait_send") == 2
    assert device_names.count("tl.dist_submit_") == 1
    assert "tl.tileop.dist_allreduce" not in device_names
    assert "tl.tileop.reduce" not in device_names


def test_repeated_allreduce_uses_independent_explicit_signals_and_staging():
    func = repeated_allreduce_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes=("tl.LowerDistCollectives", "tl.PlanDistSignals"))

    lowered_script = result.pass_snapshot("tl.LowerDistCollectives").mod.script()
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert "dist_allreduce_gather_0" in lowered_script
    assert "dist_allreduce_gather_1" in lowered_script
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 2
    # Receive storage for the two static call sites must remain distinct in final device TIR.
    gather_allocations = []
    for func in result.device_mod.functions.values():
        tvm.tir.stmt_functor.post_order_visit(
            func.body,
            lambda node: gather_allocations.append(node.buffer_var)
            if isinstance(node, tvm.tir.Allocate) and node.buffer_var.name.startswith("dist_allreduce_gather_")
            else None,
        )
    assert len(gather_allocations) == 2
    assert not gather_allocations[0].same_as(gather_allocations[1])


def test_allreduce_loop_rejects_receive_staging_reuse():
    func = loop_allreduce_kernel_factory.get_tir(world_size=4)
    with pytest.raises(tvm.error.InternalError, match="all_reduce inside For/While"):
        lower_to_device_tir(func)


@pytest.mark.parametrize("loop_kind", ["serial", "unroll", "while", "nested", "dynamic"])
@pytest.mark.parametrize("iterations", [1, 2])
def test_allreduce_rejects_any_tir_loop_before_expansion(loop_kind, iterations):
    func = allreduce_loop_boundary_kernel_factory.get_tir(
        loop_kind=loop_kind,
        iterations=iterations,
        world_size=4,
    )
    with pytest.raises(tvm.error.InternalError, match="all_reduce inside For/While"):
        lower_collectives(func)


@pytest.mark.parametrize("loop_kind", ["serial", "unroll"])
def test_allreduce_distinct_iteration_signals_do_not_protect_staging(loop_kind):
    func = allreduce_loop_boundary_kernel_factory.get_tir(
        loop_kind=loop_kind,
        distinct_members=True,
        world_size=4,
    )
    with pytest.raises(tvm.error.InternalError, match="receive staging.*consumed"):
        lower_collectives(func)


def test_allreduce_outside_local_compute_loops_remains_supported():
    result = lower_to_device_tir(
        allreduce_kernel_factory.get_tir(
            compute_loops=True,
            world_size=4,
        )
    )
    assert collect_op_names(result.device_mod).count("tl.dist_put_") == 3


def test_allreduce_rejects_missing_signal():
    with pytest.raises(TypeError, match="signal"):

        @tilelang.jit(target="sunmmio")
        def missing_signal_kernel_factory(world_size: int = 1):
            @T.prim_func
            def main(rank_id: T.dist.RankId):
                with T.Kernel():
                    src = T.alloc_shared((4, 8), T.bfloat16)
                    dst = T.alloc_shared((4, 8), T.bfloat16)
                    T.dist.all_reduce(src, dst)

            return main

        missing_signal_kernel_factory.get_tir(world_size=4)


def test_allreduce_rejects_dram_signal():
    with pytest.raises(ValueError, match="requires an SRAM INC signal"):
        allreduce_kernel_factory.get_tir(signal_kind=T.dist.SignalKind.DRAM_FLAGREG_INC, world_size=4)


@pytest.mark.parametrize(
    "kind, expect_mode, error",
    [
        (T.dist.SignalKind.SRAM_FLAGREG_VALUE, None, "automatic or INC flagreg"),
        (None, T.dist.ExpectMode.MANUAL, "cannot use expect='manual'"),
    ],
)
def test_allreduce_rejects_incompatible_signal(kind, expect_mode, error):
    with pytest.raises(ValueError, match=error):
        allreduce_kernel_factory.get_tir(signal_kind=kind, expect_mode=expect_mode, world_size=4)


def test_allreduce_world_size_one_is_rejected_by_collective_pass():
    func = allreduce_kernel_factory.get_tir(world_size=1)
    with pytest.raises(tvm.error.InternalError, match="world_size > 1"):
        lower_collectives(func)


def test_allreduce_rejects_nonempty_group():
    with pytest.raises(NotImplementedError, match="group is reserved"):
        allreduce_kernel_factory.get_tir(group=(0, 1), world_size=4)


@pytest.mark.parametrize("reduce_type", ["abssum", "absmax", "bitand", "bitor", "bitxor", "prod"])
def test_allreduce_rejects_unsupported_reduce_type(reduce_type):
    with pytest.raises(ValueError, match="reduce_type must be one of"):
        allreduce_kernel_factory.get_tir(reduce_type=reduce_type, world_size=4)


def test_allreduce_rejects_non_string_reduce_type():
    with pytest.raises(TypeError, match="reduce_type must be a string"):
        allreduce_kernel_factory.get_tir(reduce_type=0, world_size=4)


def test_allreduce_rejects_mismatched_shape():
    with pytest.raises(ValueError, match="identical shapes"):
        mismatched_allreduce_shape_kernel_factory.get_tir(world_size=4)


def test_allreduce_rejects_mismatched_dtype():
    with pytest.raises(TypeError, match="dtypes must match"):
        mismatched_allreduce_dtype_kernel_factory.get_tir(world_size=4)


def test_allreduce_rejects_overlapping_storage():
    with pytest.raises(ValueError, match="overlapping source and destination"):
        overlapping_allreduce_kernel_factory.get_tir(world_size=4)


def test_allreduce_rejects_global_source():
    with pytest.raises(ValueError, match="source must use RSRAM"):
        global_allreduce_source_kernel_factory.get_tir(world_size=4)
