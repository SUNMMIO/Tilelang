"""Static Rank all-to-all frontend and lowering tests."""

from collections import Counter

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import (
    collect_calls,
    collect_dist_put_kinds,
    collect_nodes,
    collect_op_names,
    lower_collectives,
)


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
            T.dist.all_to_all(
                src,
                dst,
                signal=signal,
                domain=domain,
                group=group,
                submit=True,
            )
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def invalid_alltoall_rank_extent_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((world_size + 1, 4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size + 1, 4, 8), T.bfloat16)
            signal = T.dist.signal()
            T.dist.all_to_all(src, dst, signal=signal, submit=True)

    return main


@tilelang.jit(target="sunmmio")
def invalid_alltoall_row_extent_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((world_size, 3, 4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 3, 4, 8), T.bfloat16)
            signal = T.dist.signal()
            T.dist.all_to_all(src, dst, signal=signal, domain="row", submit=True)

    return main


@tilelang.jit(target="sunmmio")
def mismatched_alltoall_shape_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 7), T.bfloat16)
            signal = T.dist.signal()
            T.dist.all_to_all(src, dst, signal=signal, submit=True)

    return main


@tilelang.jit(target="sunmmio")
def mismatched_alltoall_dtype_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 8), T.float32)
            signal = T.dist.signal()
            T.dist.all_to_all(src, dst, signal=signal, submit=True)

    return main


@tilelang.jit(target="sunmmio")
def overlapping_alltoall_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            storage = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            signal = T.dist.signal()
            T.dist.all_to_all(storage, storage, signal=signal, submit=True)

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
            T.dist.all_to_all(src, dst, signal=signal0, submit=True)
            T.dist.wait_signal(signal0)
            T.dist.all_to_all(src, B, signal=signal1, submit=True)
            T.dist.wait_signal(signal1)
            T.dist.all_to_all(A, dst, signal=signal2, submit=True)
            T.dist.wait_signal(signal2)
            T.dist.all_to_all(A, B, signal=signal3, submit=True)
            T.dist.wait_signal(signal3)

    return main


def _lower_routing(func):
    mod = lower_collectives(func)
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
    names = collect_op_names(func)

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
    lowered = lower_collectives(func)
    names = collect_op_names(lowered)

    assert "tl.tileop.dist_alltoall" not in names
    assert names.count("tl.tileop.dist_put") == logical_puts
    assert names.count("tl.dist_wait_signal") == 1
    assert names.count("tl.dist_submit") == 1
    assert "tl.dist_wait_send" not in names


def test_alltoall_collective_does_not_insert_receiver_wait():
    func = alltoall_kernel_factory.get_tir(world_size=4)
    before = collect_op_names(func).count("tl.dist_wait_signal")
    lowered = lower_collectives(func)
    assert collect_op_names(lowered).count("tl.dist_wait_signal") == before


def test_alltoall_rank_domain_lowers_local_and_peer_routes():
    func = alltoall_kernel_factory.get_tir(world_size=4)
    routed = _lower_routing(func)
    names = collect_op_names(routed)

    assert "tl.tileop.dist_put" not in names
    assert names.count("tl.tileop.copy") == 1
    assert names.count("tl.tileop.dist_peer_put") == 3
    assert "tl.tileop.comm_put" not in names
    assert "tl.tileop.dist_routed_peer_put" not in names


def test_alltoall_row_domain_uses_local_and_remote_row_routes():
    func = alltoall_kernel_factory.get_tir(domain="row", world_size=2)
    collective = lower_collectives(func)
    collective_script = collective.script()
    assert "src[(rank_id + 1) % 2, 3, 0, 0]" in collective_script
    assert "dst[rank_id, bx // 4, 0, 0]" in collective_script

    routed = _lower_routing(func)
    names = collect_op_names(routed)
    assert "tl.tileop.dist_put" not in names
    assert names.count("tl.tileop.copy") == 4
    assert names.count("tl.tileop.comm_put") == 24
    assert names.count("tl.tileop.dist_routed_peer_put") == 4
    assert names.count("tl.dist_peer_route") == 16


def test_alltoall_rank_world_size_two_reaches_device_tir():
    func = alltoall_kernel_factory.get_tir(world_size=2)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistCollectives")

    collective_names = collect_op_names(result.pass_snapshot("tl.LowerDistCollectives").mod)
    device_names = collect_op_names(result.device_mod)
    assert "tl.tileop.dist_alltoall" not in collective_names
    assert device_names.count("tl.dist_put_") == 1
    assert device_names.count("tl.dist_wait_signal_") == 1
    assert device_names.count("tl.dist_wait_send") == 2
    assert device_names.count("tl.dist_submit_") == 1


@pytest.mark.parametrize("world_size", [2, 4])
def test_alltoall_row_reaches_device_tir(world_size):
    func = alltoall_kernel_factory.get_tir(domain="row", world_size=world_size)
    result = lower_to_device_tir(
        func,
        capture_passes=("tl.SunmmioLayoutInference", "tl.LowerDistCommunication"),
    )

    layout_script = result.pass_snapshot("tl.SunmmioLayoutInference").mod.script()
    device_names = collect_op_names(result.device_mod)
    assert "layout_map" in layout_script
    assert device_names.count("tl.dist_put_") == 16 * (world_size - 1)
    assert device_names.count("tl.dist_wait_signal_") == 1
    assert device_names.count("tl.dist_wait_send") == 2
    assert device_names.count("tl.dist_submit_") == 1


def test_alltoall_supports_all_p2p_scope_combinations():
    func = alltoall_scope_combinations_kernel_factory.get_tir(world_size=2)
    result = lower_to_device_tir(func, capture_passes="tl.PlanDistSignals")
    planned = result.pass_snapshot("tl.PlanDistSignals").mod["main"]

    assert int(planned.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 2
    assert int(planned.attrs["tl.dist.signal_counts"]["dram_flagreg_inc"]) == 2
    assert Counter(collect_dist_put_kinds(result.device_mod)) == {
        "sram_flagreg_inc": 2,
        "dram_flagreg_inc": 2,
    }


def test_alltoall_world_size_one_is_rejected_by_collective_pass():
    func = alltoall_kernel_factory.get_tir(world_size=1)
    with pytest.raises(tvm.error.InternalError, match="world_size > 1"):
        lower_collectives(func)


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


@tilelang.jit(target="sunmmio")
def alltoall_signals_kernel_factory(
    domain="rank",
    wait_mode="all",
    signal_kind=None,
    expect=None,
    count=None,
    form="group",
    src_dram=False,
    dst_dram=False,
    submit=True,
    world_size: int = 1,
):
    shape = _alltoall_shape(world_size, domain)
    endpoint_count = world_size * (4 if _is_row_domain(domain) else 1)

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, T.placement.replicated(), T.bfloat16),
        B: T.MeshTensor(shape, T.placement.replicated(), T.bfloat16),
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            src = T.alloc_shared(shape, T.bfloat16)
            dst = T.alloc_shared(shape, T.bfloat16)
            ready = T.dist.signals(endpoint_count if count is None else count, kind=signal_kind, expect=expect)
            single = T.dist.signal()
            if form == "both":
                T.dist.all_to_all(src, dst, signal=single, signals=ready, domain=domain)
            elif form == "neither":
                T.dist.all_to_all(src, dst, domain=domain)
            elif form == "wrong_type":
                T.dist.all_to_all(src, dst, signals=single, domain=domain)
            else:
                done = T.dist.all_to_all(A if src_dram else src, B if dst_dram else dst, signals=ready, domain=domain, submit=submit)
                if not submit:
                    T.dist.submit()
                if wait_mode == "any":
                    while T.dist.has_pending(done):
                        source = T.dist.wait_any(done)
                        T.evaluate(source)
                elif wait_mode == "group":
                    T.dist.wait_all(ready)
                else:
                    T.dist.wait_all(done)

    return main


@pytest.mark.parametrize("domain,world_size", [("rank", 2), ("rank", 4), ("row", 2)])
def test_alltoall_signals_route_members_and_completion_deltas(domain, world_size):
    func = alltoall_signals_kernel_factory.get_tir(domain=domain, world_size=world_size)
    result = lower_to_device_tir(func, capture_passes=("tl.LowerDistCollectives", "tl.LowerDistCommunication"))
    collective = result.pass_snapshot("tl.LowerDistCollectives").mod
    calls = collect_calls(collective, "tl.tileop.dist_put")
    rows = 4 if domain == "row" else 1
    endpoints = world_size * rows
    assert len(calls) == endpoints
    assert collect_op_names(collective).count("tl.dist_completion") == 1
    assert str(collect_calls(collective, "tl.dist_completion")[0].args[-1].value) == f"all_to_all_{domain}"
    rank_var = func.params[int(func.attrs["tl.dist.rank_id_param_index"])]
    analyzer = tvm.arith.Analyzer()
    for call in calls:
        route = call.args[4]
        assert route.op.name == "tl.dist_signal_group_route"
        for rank in range(world_size):
            for row in range(4):
                values = {rank_var: tvm.tir.IntImm("int32", rank), call.args[5]: tvm.tir.IntImm("int32", row * 4 + 1)}

                def evaluate(expr, substitution=values):
                    return int(analyzer.simplify(tvm.tir.stmt_functor.substitute(expr, substitution)))

                source = rank * rows + (row if domain == "row" else 0)
                destination = evaluate(call.args[2]) * rows + (evaluate(call.args[3]) if domain == "row" else 0)
                assert evaluate(route.args[1]) == source
                assert evaluate(route.args[2]) == source * endpoints + destination
                assert evaluate(route.args[3]) == 1

    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod
    init = collect_calls(lowered, "tl.dist_completion_init_")
    assert len(init) == 1
    assert len(init[0].args[5:]) == endpoints
    for rank in range(world_size):
        deltas = [
            int(analyzer.simplify(tvm.tir.stmt_functor.substitute(delta, {rank_var: tvm.tir.IntImm("int32", rank)})))
            for delta in init[0].args[5:]
        ]
        assert deltas == [int(index // rows != rank) for index in range(endpoints)]
    names = collect_op_names(result.device_mod)
    assert "tl.tileop.dist_alltoall" not in names
    assert "tl.dist_completion_init_" not in names
    assert names.count("tl.dist_wait_any_") == 1
    assert names.count("tl.dist_submit_") == 1


@pytest.mark.parametrize("domain", ["rank", "row"])
@pytest.mark.parametrize("kind_name", ["FLAGREG_INC", "FLAGREG_VALUE", "MEMORY"])
@pytest.mark.parametrize("dst_dram", [False, True])
def test_alltoall_signals_support_kinds_and_generation_layout(domain, kind_name, dst_dram):
    kind = getattr(T.dist.SignalKind, ("DRAM_" if dst_dram else "SRAM_") + kind_name)
    func = alltoall_signals_kernel_factory.get_tir(
        domain=domain,
        signal_kind=kind,
        dst_dram=dst_dram,
        wait_mode="any",
        world_size=2,
    )
    result = lower_to_device_tir(func, capture_passes=("tl.PlanDistSignals", "tl.LowerDistCommunication"))
    planned = next(iter(result.pass_snapshot("tl.PlanDistSignals").mod.functions.values()))
    endpoints = 8 if domain == "row" else 2
    assert int(planned.attrs["tl.dist.signal_counts"][kind.value]) == endpoints
    lower_comm = result.pass_snapshot("tl.LowerDistCommunication").mod
    allocations = collect_nodes(lower_comm, tvm.tir.Allocate, lambda node: "generation" in node.buffer_var.name)
    if kind_name == "FLAGREG_INC":
        assert not allocations
    else:
        assert [tuple(int(extent) for extent in node.extents) for node in allocations] == [(endpoints * endpoints,)]
    assert collect_op_names(result.device_mod).count("tl.dist_wait_any_") == 1


@pytest.mark.parametrize("src_dram,dst_dram", [(False, False), (False, True), (True, False), (True, True)])
def test_alltoall_signals_scope_combinations_and_explicit_submit(src_dram, dst_dram):
    result = lower_to_device_tir(
        alltoall_signals_kernel_factory.get_tir(
            src_dram=src_dram,
            dst_dram=dst_dram,
            submit=False,
            expect="auto",
            world_size=2,
        )
    )
    names = collect_op_names(result.device_mod)
    assert names.count("tl.dist_submit_") == 1
    assert names.count("tl.dist_wait_any_") == 1


@pytest.mark.parametrize("domain,count", [("rank", 3), ("row", 2)])
def test_alltoall_signals_require_one_member_per_source_endpoint(domain, count):
    with pytest.raises(ValueError, match="requires .* signals"):
        alltoall_signals_kernel_factory.get_tir(domain=domain, count=count, world_size=2)


@pytest.mark.parametrize("form", ["both", "neither", "wrong_type"])
def test_alltoall_signals_reject_invalid_signal_forms(form):
    with pytest.raises(TypeError, match="exactly one of signal or signals|signals must be created"):
        alltoall_signals_kernel_factory.get_tir(form=form, world_size=2)


def test_alltoall_signals_reject_manual_expect():
    with pytest.raises(ValueError, match="cannot use expect='manual'"):
        alltoall_signals_kernel_factory.get_tir(expect="manual", world_size=2)


@tilelang.jit(target="sunmmio")
def repeated_alltoall_signals_kernel_factory(wait_mode="all", world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            dst0 = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            dst1 = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            ready = T.dist.signals(world_size, kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE)
            first = T.dist.all_to_all(src, dst0, signals=ready, submit=True)
            if wait_mode == "all":
                T.dist.wait_all(first)
            elif wait_mode == "any":
                while T.dist.has_pending(first):
                    T.dist.wait_any(first)
            elif wait_mode == "partial":
                T.dist.wait_any(first)
            second = T.dist.all_to_all(src, dst1, signals=ready, submit=True)
            if wait_mode == "group":
                T.dist.wait_all(ready)
            else:
                T.dist.wait_all(second)

    return main


@pytest.mark.parametrize("wait_mode", ["all", "any", "group"])
def test_alltoall_signals_reuse_with_independent_payload_slots(wait_mode):
    result = lower_to_device_tir(
        repeated_alltoall_signals_kernel_factory.get_tir(wait_mode=wait_mode, world_size=2), capture_passes="tl.LowerDistCommunication"
    )
    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod
    allocations = collect_nodes(lowered, tvm.tir.Allocate, lambda node: "generation" in node.buffer_var.name)
    assert [tuple(int(extent) for extent in node.extents) for node in allocations] == [(4,)]
    names = collect_op_names(result.device_mod)
    assert names.count("tl.dist_put_") == 2
    assert names.count("tl.dist_wait_any_") == (0 if wait_mode == "group" else 2)


def test_alltoall_signals_reject_reuse_after_partial_progress():
    with pytest.raises(tvm.error.InternalError, match="previous completion must be drained"):
        lower_to_device_tir(repeated_alltoall_signals_kernel_factory.get_tir(wait_mode="partial", world_size=2))


@tilelang.jit(target="sunmmio")
def alltoall_signals_offset_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((2 * world_size, 4, 8), T.bfloat16)
            dst = T.alloc_shared((2 * world_size, 4, 8), T.bfloat16)
            ready = T.dist.signals(world_size)
            for e in T.serial(2):
                done = T.dist.all_to_all(
                    src[e * world_size : (e + 1) * world_size, :, :],
                    dst[e * world_size : (e + 1) * world_size, :, :],
                    signals=ready,
                    submit=True,
                )
                T.dist.wait_all(done)

    return main


def test_alltoall_signals_support_independent_loop_slots():
    func = alltoall_signals_offset_kernel_factory.get_tir(world_size=2)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistCollectives")
    expanded = result.pass_snapshot("tl.LowerDistCollectives").mod
    puts = collect_calls(expanded, "tl.tileop.dist_put")
    assert len(puts) == 2
    loop = collect_nodes(expanded, tvm.tir.For)[0]
    rank = func.params[int(func.attrs["tl.dist.rank_id_param_index"])]
    analyzer = tvm.arith.Analyzer()
    for iteration in range(2):
        for source_rank in range(2):
            values = {loop.loop_var: tvm.tir.IntImm("int32", iteration), rank: tvm.tir.IntImm("int32", source_rank)}
            for put in puts:
                target_rank = int(analyzer.simplify(tvm.tir.stmt_functor.substitute(put.args[2], values)))
                for operand, peer in [(0, target_rank), (1, source_rank)]:
                    region = put.args[operand]
                    index = region.args[0].indices[0]
                    assert int(analyzer.simplify(tvm.tir.stmt_functor.substitute(index, values))) == iteration * 2 + peer
                    assert [int(x) for x in region.args[2:]] == [1, 4, 8]
    assert collect_op_names(result.device_mod).count("tl.dist_put_") == 1
    assert collect_op_names(result.device_mod).count("tl.dist_wait_any_") == 1


@tilelang.jit(target="sunmmio")
def nested_offset_alltoall_kernel_factory(error=None, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4 * world_size, 32, 128), T.bfloat16)
            dst = T.alloc_shared((4 * world_size, 32, 128), T.bfloat16)
            ready = T.dist.signals(world_size)
            for expert in T.serial(2):
                for block in T.serial(2):
                    begin = (expert * 2 + block) * world_size
                    output_begin = block * world_size if error == "reuse" else begin
                    if error != "rank_condition" or rank_id == 0:
                        done = T.dist.all_to_all(
                            src[begin : begin + world_size, :, :],
                            dst[output_begin : output_begin + world_size, :, :],
                            signals=ready,
                            submit=True,
                        )
                        T.dist.wait_all(done)
                    if error == "early_exit":
                        break

    return main


def test_alltoall_loop_offsets_support_flattened_nested_slots():
    result = lower_to_device_tir(nested_offset_alltoall_kernel_factory.get_tir(world_size=2), capture_passes="tl.LowerDistCollectives")
    expanded = result.pass_snapshot("tl.LowerDistCollectives").mod
    assert len(collect_nodes(expanded, tvm.tir.For)) == 2
    assert len(collect_calls(expanded, "tl.tileop.dist_put")) == 2
    assert collect_op_names(result.device_mod).count("tl.dist_wait_any_") == 1


@pytest.mark.parametrize(
    "error,message",
    [
        ("reuse", "independent destination slots"),
        ("rank_condition", "Rank/core-uniform control flow"),
        ("early_exit", "without early exits"),
    ],
)
def test_alltoall_loop_offsets_reject_unsafe_loop_context(error, message):
    func = nested_offset_alltoall_kernel_factory.get_tir(error=error, world_size=2)
    with pytest.raises(tvm.error.InternalError, match=message):
        lower_collectives(func)


def test_alltoall_loop_offsets_reject_rank_dependent_block_predicate():
    func = nested_offset_alltoall_kernel_factory.get_tir(world_size=2)
    rank = func.params[int(func.attrs["tl.dist.rank_id_param_index"])]
    guarded = tvm.tir.BlockRealize([], rank == 0, tvm.tir.Block([], [], [], "guard", func.body))
    with pytest.raises(tvm.error.InternalError, match="Rank/core-uniform control flow"):
        lower_collectives(func.with_body(guarded))


@tilelang.jit(target="sunmmio")
def row_offset_alltoall_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            rows = T.mesh_nrows()
            src = T.alloc_shared((2 * world_size, rows, 32, 128), T.bfloat16)
            dst = T.alloc_shared((2 * world_size, rows, 32, 128), T.bfloat16)
            ready = T.dist.signals(world_size * rows)
            for e in T.serial(2):
                done = T.dist.all_to_all(
                    src[e * world_size : (e + 1) * world_size, :, :, :],
                    dst[e * world_size : (e + 1) * world_size, :, :, :],
                    signals=ready,
                    domain="row",
                    submit=True,
                )
                T.dist.wait_all(done)

    return main


def test_alltoall_loop_offsets_do_not_enable_cross_row_routing():
    with pytest.raises(tvm.error.InternalError, match="only support the RANK domain"):
        lower_collectives(row_offset_alltoall_kernel_factory.get_tir(world_size=2))
