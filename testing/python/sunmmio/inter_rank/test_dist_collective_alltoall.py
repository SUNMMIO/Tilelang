"""Static Rank all-to-all frontend and lowering tests."""

from collections import Counter

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.utils.target import determine_target
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir


def _is_row_domain(domain):
    return domain == "row" or domain == T.dist.CollectiveDomain.ROW


def _alltoall_shape(world_size, domain):
    if _is_row_domain(domain):
        return world_size, T.mesh_nrows(), 4, 8
    return world_size, 4, 8


@tilelang.jit(target="sunmmio")
def alltoall_kernel_factory(
    domain=T.dist.CollectiveDomain.RANK,
    signal_kind=None,
    group=None,
    world_size: int = 1,
):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            shape = _alltoall_shape(world_size, domain)
            src = T.alloc_shared(shape, T.bfloat16)
            dst = T.alloc_shared(shape, T.bfloat16)
            signal = T.dist.signal(kind=signal_kind)
            T.dist.all_to_all(src, dst, signal=signal, domain=domain, group=group)
            T.dist.wait_signal(signal, dst=dst)

    return main


@tilelang.jit(target="sunmmio")
def invalid_alltoall_rank_extent_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((world_size + 1, 4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size + 1, 4, 8), T.bfloat16)
            signal = T.dist.signal()
            T.dist.all_to_all(src, dst, signal=signal)

    return main


@tilelang.jit(target="sunmmio")
def invalid_alltoall_row_extent_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((world_size, 3, 4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 3, 4, 8), T.bfloat16)
            signal = T.dist.signal()
            T.dist.all_to_all(src, dst, signal=signal, domain="row")

    return main


@tilelang.jit(target="sunmmio")
def mismatched_alltoall_shape_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 7), T.bfloat16)
            signal = T.dist.signal()
            T.dist.all_to_all(src, dst, signal=signal)

    return main


@tilelang.jit(target="sunmmio")
def mismatched_alltoall_dtype_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 8), T.float32)
            signal = T.dist.signal()
            T.dist.all_to_all(src, dst, signal=signal)

    return main


@tilelang.jit(target="sunmmio")
def overlapping_alltoall_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            storage = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            signal = T.dist.signal()
            T.dist.all_to_all(storage, storage, signal=signal)

    return main


@tilelang.jit(target="sunmmio")
def alltoall_scope_combinations_kernel_factory(world_size: int = 1):
    placement = T.placement.replicated()

    @T.prim_func
    def main(
        A: T.MeshTensor((world_size, 32, 32), placement, T.bfloat16),  # type: ignore
        B: T.MeshTensor((world_size, 32, 32), placement, T.bfloat16),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            src = T.alloc_shared((world_size, 32, 32), T.bfloat16)
            dst = T.alloc_shared((world_size, 32, 32), T.bfloat16)
            signal0 = T.dist.signal()
            signal1 = T.dist.signal()
            signal2 = T.dist.signal()
            signal3 = T.dist.signal()
            T.dist.all_to_all(src, dst, signal=signal0)
            T.dist.wait_signal(signal0, dst=dst)
            T.dist.all_to_all(src, B, signal=signal1)
            T.dist.wait_signal(signal1, dst=B)
            T.dist.all_to_all(A, dst, signal=signal2)
            T.dist.wait_signal(signal2, dst=dst)
            T.dist.all_to_all(A, B, signal=signal3)
            T.dist.wait_signal(signal3, dst=B)

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


def _collect_dist_put_kinds(mod):
    kinds = []

    def visit(node):
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.dist_put_":
            kinds.append(str(node.args[3].value))

    for func in mod.functions.values():
        if isinstance(func, tvm.tir.PrimFunc):
            tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return kinds


def _base_mod(func):
    target = tvm.target.Target(determine_target("sunmmio", return_object=True))
    mod = tvm.IRModule({"main": func.with_attr("target", target)})
    return tilelang.transform.ResolveSunmmioMeshSymbols()(mod)


def _lower_collectives(func):
    return tilelang.transform.LowerDistCollectives()(_base_mod(func))


def _lower_routing(func):
    mod = _lower_collectives(func)
    mod = tilelang.transform.InferSramScope()(mod)
    mod = tilelang.transform.PlanDistSignals()(mod)
    return tilelang.transform.LowerDistRouting()(mod)


@pytest.mark.parametrize(
    "domain, expected",
    [
        (T.dist.CollectiveDomain.RANK, "rank"),
        ("rank", "rank"),
        (T.dist.CollectiveDomain.ROW, "row"),
        ("row", "row"),
    ],
)
def test_alltoall_frontend_normalizes_domain(domain, expected):
    func = alltoall_kernel_factory.get_tir(domain=domain, world_size=4)
    names = _collect_op_names(func)

    assert names.count("tl.tileop.dist_alltoall") == 1
    assert f'"{expected}"' in func.script()
    assert "tl.tileop.dist_put" not in names


@pytest.mark.parametrize(
    "domain, logical_puts",
    [
        (T.dist.CollectiveDomain.RANK, 4),
        (T.dist.CollectiveDomain.ROW, 16),
    ],
)
def test_lower_dist_collectives_expands_alltoall_endpoint_domain(domain, logical_puts):
    func = alltoall_kernel_factory.get_tir(domain=domain, world_size=4)
    lowered = _lower_collectives(func)
    names = _collect_op_names(lowered)

    assert "tl.tileop.dist_alltoall" not in names
    assert names.count("tl.tileop.dist_put") == logical_puts
    assert names.count("tl.tileop.dist_wait_signal") == 1
    assert names.count("tl.dist_wait_send") == 1


def test_alltoall_collective_does_not_insert_receiver_wait():
    func = alltoall_kernel_factory.get_tir(world_size=4)
    before = _collect_op_names(func).count("tl.tileop.dist_wait_signal")
    lowered = _lower_collectives(func)
    assert _collect_op_names(lowered).count("tl.tileop.dist_wait_signal") == before


def test_alltoall_rank_domain_lowers_local_and_peer_routes():
    func = alltoall_kernel_factory.get_tir(world_size=4)
    routed = _lower_routing(func)
    names = _collect_op_names(routed)

    assert "tl.tileop.dist_put" not in names
    assert names.count("tl.tileop.copy") == 1
    assert names.count("tl.tileop.dist_peer_put") == 3
    assert "tl.tileop.comm_put" not in names
    assert "tl.tileop.dist_routed_peer_put" not in names


def test_alltoall_row_domain_uses_local_and_remote_row_routes():
    func = alltoall_kernel_factory.get_tir(domain="row", world_size=2)
    collective = _lower_collectives(func)
    collective_script = collective.script()
    assert "src[(rank_id + 1) % 2, 3, 0, 0]" in collective_script
    assert "dst[rank_id, bx // 4, 0, 0]" in collective_script

    routed = _lower_routing(func)
    names = _collect_op_names(routed)
    assert "tl.tileop.dist_put" not in names
    assert names.count("tl.tileop.copy") == 4
    assert names.count("tl.tileop.comm_put") == 24
    assert names.count("tl.tileop.dist_routed_peer_put") == 4
    assert names.count("tl.dist_peer_route") == 16


def test_alltoall_rank_world_size_two_reaches_device_tir():
    func = alltoall_kernel_factory.get_tir(world_size=2)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistCollectives")

    collective_names = _collect_op_names(result.pass_snapshot("tl.LowerDistCollectives").mod)
    device_names = _collect_op_names(result.device_mod)
    assert "tl.tileop.dist_alltoall" not in collective_names
    assert device_names.count("tl.dist_put_") == 1
    assert device_names.count("tl.dist_wait_signal_") == 1
    assert device_names.count("tl.dist_wait_send") == 1


@pytest.mark.parametrize("world_size", [2, 4])
def test_alltoall_row_reaches_device_tir(world_size):
    func = alltoall_kernel_factory.get_tir(domain="row", world_size=world_size)
    result = lower_to_device_tir(
        func,
        capture_passes=("tl.SunmmioLayoutInference", "tl.LowerDistCommunication"),
    )

    layout_script = result.pass_snapshot("tl.SunmmioLayoutInference").mod.script()
    device_names = _collect_op_names(result.device_mod)
    assert "layout_map" in layout_script
    assert device_names.count("tl.dist_put_") == 16 * (world_size - 1)
    assert device_names.count("tl.dist_wait_signal_") == 1
    assert device_names.count("tl.dist_wait_send") == 1


def test_alltoall_supports_all_p2p_scope_combinations():
    func = alltoall_scope_combinations_kernel_factory.get_tir(world_size=2)
    result = lower_to_device_tir(func, capture_passes="tl.PlanDistSignals")
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]

    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 2
    assert int(planned.attrs["tl.dist.signal_counts"]["dram_flagreg_inc"]) == 2
    assert Counter(_collect_dist_put_kinds(result.device_mod)) == {
        "sram_flagreg_inc": 2,
        "dram_flagreg_inc": 2,
    }


def test_alltoall_world_size_one_is_rejected_by_collective_pass():
    func = alltoall_kernel_factory.get_tir(world_size=1)
    with pytest.raises(tvm.error.InternalError, match="world_size > 1"):
        _lower_collectives(func)


def test_alltoall_rejects_invalid_domain():
    with pytest.raises(ValueError, match="domain must be 'rank' or 'row'"):
        alltoall_kernel_factory.get_tir(domain="ROW", world_size=4)


def test_alltoall_rejects_nonempty_group():
    with pytest.raises(NotImplementedError, match="group is reserved"):
        alltoall_kernel_factory.get_tir(group=(0, 1), world_size=4)


def test_alltoall_rejects_value_signal():
    with pytest.raises(ValueError, match="automatic or INC flagreg signal"):
        alltoall_kernel_factory.get_tir(
            signal_kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE,
            world_size=4,
        )


def test_alltoall_rejects_invalid_rank_extent():
    with pytest.raises(ValueError, match="leading extent 0"):
        invalid_alltoall_rank_extent_kernel_factory.get_tir(world_size=4)


def test_alltoall_rejects_invalid_row_extent():
    with pytest.raises(ValueError, match="leading extent 1"):
        invalid_alltoall_row_extent_kernel_factory.get_tir(world_size=4)


def test_alltoall_rejects_mismatched_shape():
    with pytest.raises(ValueError, match="identical shapes"):
        mismatched_alltoall_shape_kernel_factory.get_tir(world_size=4)


def test_alltoall_rejects_mismatched_dtype():
    with pytest.raises(TypeError, match="dtypes must match"):
        mismatched_alltoall_dtype_kernel_factory.get_tir(world_size=4)


def test_alltoall_rejects_overlapping_storage():
    with pytest.raises(ValueError, match="overlapping source and destination"):
        overlapping_alltoall_kernel_factory.get_tir(world_size=4)
