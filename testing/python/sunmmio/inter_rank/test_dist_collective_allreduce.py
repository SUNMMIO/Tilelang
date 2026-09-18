"""Rank all-reduce frontend and gather-reduce lowering tests."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.utils.target import determine_target
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir


@tilelang.jit(target="sunmmio")
def allreduce_kernel_factory(reduce_type="sum", group=None, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.bfloat16)
            dst = T.alloc_shared((4, 8), T.bfloat16)
            T.dist.all_reduce(src, dst, reduce_type, group=group)

    return main


@tilelang.jit(target="sunmmio")
def repeated_allreduce_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.float32)
            dst = T.alloc_shared((4, 8), T.float32)
            T.dist.all_reduce(src, dst, "sum")
            T.dist.all_reduce(src, dst, "max")

    return main


@tilelang.jit(target="sunmmio")
def mismatched_allreduce_shape_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.bfloat16)
            dst = T.alloc_shared((4, 7), T.bfloat16)
            T.dist.all_reduce(src, dst)

    return main


@tilelang.jit(target="sunmmio")
def mismatched_allreduce_dtype_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.bfloat16)
            dst = T.alloc_shared((4, 8), T.float32)
            T.dist.all_reduce(src, dst)

    return main


@tilelang.jit(target="sunmmio")
def overlapping_allreduce_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            data = T.alloc_shared((4, 8), T.bfloat16)
            T.dist.all_reduce(data, data)

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
            T.dist.all_reduce(src, dst)

    return main


def _collect_op_names(func_or_mod):
    names = []
    funcs = func_or_mod.functions.values() if isinstance(func_or_mod, tvm.IRModule) else (func_or_mod,)

    def visit(node):
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op):
            names.append(node.op.name)

    for func in funcs:
        if isinstance(func, tvm.tir.PrimFunc):
            tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return names


def _base_mod(func):
    target = tvm.target.Target(determine_target("sunmmio", return_object=True))
    mod = tvm.IRModule({"main": func.with_attr("target", target)})
    return tilelang.transform.ResolveSunmmioMeshSymbols()(mod)


def _lower_collectives(func):
    return tilelang.transform.LowerDistCollectives()(_base_mod(func))


@pytest.mark.parametrize("reduce_type", ["sum", "max", "min", "SUM"])
def test_allreduce_frontend_normalizes_reduce_type_and_hides_signal(reduce_type):
    func = allreduce_kernel_factory.get_tir(reduce_type=reduce_type, world_size=4)
    names = _collect_op_names(func)

    assert names.count("tl.tileop.dist_allreduce") == 1
    assert names.count("tl.dist_signal_decl") == 1
    assert "tl.tileop.dist_put" not in names
    expected = "sum" if reduce_type == "SUM" else reduce_type
    assert f'"{expected}"' in func.script()


@pytest.mark.parametrize("reduce_type", ["sum", "max", "min"])
def test_lower_dist_collectives_builds_gather_then_local_reduce(reduce_type):
    func = allreduce_kernel_factory.get_tir(reduce_type=reduce_type, world_size=4)
    lowered = _lower_collectives(func)
    names = _collect_op_names(lowered)
    script = lowered.script()

    assert "tl.tileop.dist_allreduce" not in names
    assert names.count("tl.tileop.copy") == 1
    assert names.count("tl.tileop.dist_put") == 3
    assert names.count("tl.tileop.dist_wait_signal") == 1
    assert names.count("tl.tileop.reduce") == 1
    assert names.count("tl.dist_wait_send") == 1
    assert "dist_allreduce_gather_0 = T.alloc_buffer((4, 4, 8)" in script
    assert f'), "{reduce_type}", 0, T.bool(True))' in script
    assert script.index("T.dist_wait_signal") < script.index("T.reduce(")

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
    device_names = _collect_op_names(result.device_mod)
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 1
    assert device_names.count("tl.dist_put_") == 3
    assert device_names.count("tl.dist_wait_signal_") == 1
    assert device_names.count("tl.dist_wait_send") == 1
    assert "tl.tileop.dist_allreduce" not in device_names
    assert "tl.tileop.reduce" not in device_names


def test_repeated_allreduce_uses_independent_internal_signals_and_staging():
    func = repeated_allreduce_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func, capture_passes=("tl.LowerDistCollectives", "tl.PlanDistSignals"))

    lowered_script = result.pass_snapshot("tl.LowerDistCollectives").mod.script()
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]
    assert "dist_allreduce_gather_0" in lowered_script
    assert "dist_allreduce_gather_1" in lowered_script
    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 2


def test_allreduce_world_size_one_is_rejected_by_collective_pass():
    func = allreduce_kernel_factory.get_tir(world_size=1)
    with pytest.raises(tvm.error.InternalError, match="world_size > 1"):
        _lower_collectives(func)


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
