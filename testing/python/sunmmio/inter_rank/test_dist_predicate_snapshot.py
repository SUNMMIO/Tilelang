"""Eliminating a manual dist if preserves when and how often it is evaluated."""

import numpy as np
import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tvm import tir

from testing.python.sunmmio.common.codegen_validation import codegen_sunmmio_suvm_mlir, validate_suvm_mlir_with_npuir_opt
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import base_mod, collect_nodes, single_prim_func


@tilelang.jit(target="sunmmio")
def predicate_kernel_factory(case="shared", cross_row=False, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, enabled: T.int32):
        with T.Kernel() as core_id:
            zeros = T.alloc_shared((32,), T.int32)
            state = T.alloc_shared((32,), T.int32)
            payload = T.alloc_shared((32,), T.int32)
            destination = T.alloc_shared((32,), T.int32)
            T.fill(zeros, 0)
            T.fill(state, enabled + core_id // T.mesh_ncols())
            T.fill(payload, 3)
            signals = T.dist.signals(2, kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            peer = (rank_id + 1) % world_size
            row = core_id // T.mesh_ncols()
            dst_row = (row + 1) % T.mesh_nrows() if cross_row else row
            if case == "pure":
                if enabled > 0:
                    T.dist.put(payload, destination, peer, dst_row=dst_row, signal=signals[1])
            elif case == "loop":
                for _ in T.serial(4):
                    if state[0] > 0:
                        T.dist.put(zeros, state, rank_id, signal=signals[0])
                        T.dist.put(payload, destination, peer, dst_row=dst_row, signal=signals[1])
            else:
                if state[0] > 0:
                    if case in ("shared", "nested_late", "independent"):
                        T.dist.put(zeros, state, rank_id, signal=signals[0])
                    if case == "nested_shared":
                        if state[0] > 0:
                            T.dist.put(zeros, state, rank_id, signal=signals[0])
                            T.dist.put(payload, destination, peer, dst_row=dst_row, signal=signals[1])
                    elif case == "nested_late":
                        if state[0] > 0:
                            T.dist.put(payload, destination, peer, dst_row=dst_row, signal=signals[1])
                    elif case != "independent":
                        T.dist.put(payload, destination, peer, dst_row=dst_row, signal=signals[1])
                if case == "independent":  # noqa: SIM102 -- static DSL selection
                    if state[0] > 0:
                        T.dist.put(payload, destination, peer, dst_row=dst_row, signal=signals[1])
            T.dist.submit()

    return main


def _planned(case="shared", cross_row=False):
    func = predicate_kernel_factory.get_tir(case=case, cross_row=cross_row, world_size=2)
    mod = tilelang.transform.LowerDistCollectives()(base_mod(func))
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.InferSramScope()(mod)
    return tilelang.transform.PlanDistSignals()(mod)


def _region(expr):
    if isinstance(expr, tir.BufferRegion):
        return expr
    if isinstance(expr, tir.BufferLoad):
        ranges = [
            tvm.ir.Range(index.base, index.lanes) if isinstance(index, tir.Ramp) else tvm.ir.Range(index, 1) for index in expr.indices
        ]
        return tir.BufferRegion(expr.buffer, ranges)
    assert isinstance(expr, tir.Call) and expr.op.name == "tl.tileop.region"
    load = expr.args[0]
    return tir.BufferRegion(load.buffer, [tvm.ir.Range(index, extent) for index, extent in zip(load.indices, expr.args[2:])])


def _scalar_transfer(dst, value):
    assert len(dst.region) == 1
    lane = tir.Var("lane", "int32")
    return tir.For(lane, 0, dst.region[0].extent, tir.ForKind.SERIAL, tir.BufferStore(dst.buffer, value(lane), [dst.region[0].min + lane]))


def _project_sender_trace(mod, core_id=0):
    """Execute local copies and record sends/source active values, without a NIC.

    Cross-row transport is not simulated: slots 0-3 record the active value
    produced on each source row; slot 4 counts same-row remote enqueues.
    """
    func = single_prim_func(mod)
    trace = tir.decl_buffer((5,), "int32", name="trace")
    bindings = {}

    def rewrite(node):
        if isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent":
            bindings[node.node.var] = tir.IntImm("int32", core_id)
            return node.body
        if isinstance(node, tir.LetStmt) and isinstance(node.value, tir.Call) and node.value.op.name == "tl.dist_signal_group":
            return node.body
        if isinstance(node, tir.BufferStore) and node.buffer.name.startswith("dist_route_active_"):
            source_row = int(node.buffer.name.rsplit("_", 1)[1])
            return tir.BufferStore(trace, tir.Cast("int32", node.value), [source_row])
        if not isinstance(node, tir.Evaluate) or not isinstance(node.value, tir.Call):
            return None
        call = node.value
        name = call.op.name
        if name == "tl.tileop.fill":
            return _scalar_transfer(_region(call.args[0]), lambda lane: call.args[1])
        if name == "tl.tileop.copy":
            src, dst = map(_region, call.args[:2])
            return _scalar_transfer(dst, lambda lane: tir.BufferLoad(src.buffer, [src.region[0].min + lane]))
        if name == "tl.tileop.dist_peer_put":
            active = call.args[3].args[-1]
            return tir.IfThenElse(active, tir.BufferStore(trace, trace[4] + 1, [4]), None)
        if name.startswith("tl.dist_") or name in ("tl.tileop.comm_put", "tl.tileop.dist_routed_peer_put"):
            return tir.Evaluate(0)
        return None

    body = tir.stmt_functor.ir_transform(func.body, None, rewrite, ["tir.AttrStmt", "tir.LetStmt", "tir.BufferStore", "tir.Evaluate"])
    body = tir.stmt_functor.substitute(body, bindings)
    func = tir.PrimFunc([*func.params, trace.data], body, buffer_map={**dict(func.buffer_map), trace.data: trace})
    return func.with_attr("global_symbol", "main").with_attr("tir.noalias", True)


def _execute_trace(mod, enabled, core_id=0, probe=None, addon=None):
    func = _project_sender_trace(mod, core_id)
    executable = tvm.compile(func, target="c").jit(addons=[str(addon)] if addon else None, options=["-std=c++17"])
    trace = tvm.runtime.tensor(np.zeros(5, dtype="int32"))
    args = [0, enabled]
    if probe is not None:
        args.append(probe)
    executable(*args, trace)
    return trace.numpy()


@pytest.mark.parametrize("enabled", [0, 1])
@pytest.mark.parametrize("case", ["shared", "nested_shared", "nested_late", "independent", "loop"])
def test_manual_peer_if_preserves_source_semantics(case, enabled):
    routed = tilelang.transform.LowerDistRouting()(_planned(case))
    expected = int(enabled > 0 and case in ("shared", "nested_shared", "loop"))
    assert _execute_trace(routed, enabled)[4] == expected


@pytest.mark.parametrize("source_row", range(4))
@pytest.mark.parametrize("case", ["shared", "nested_shared", "nested_late"])
def test_cross_row_active_uses_the_source_snapshot(case, source_row):
    routed = tilelang.transform.LowerDistRouting()(_planned(case, cross_row=True))
    trace = _execute_trace(routed, enabled=-2, core_id=source_row * 4)
    expected = np.zeros(4, dtype="int32")
    expected[source_row] = int(source_row > 2 and case != "nested_late")
    np.testing.assert_array_equal(trace[:4], expected)


def test_pure_condition_keeps_existing_routing_shape():
    routed = tilelang.transform.LowerDistRouting()(_planned("pure"))
    snapshots = collect_nodes(routed, tir.LetStmt, lambda node: node.var.name.startswith("dist_condition_snapshot_"))
    assert snapshots == []
    assert _execute_trace(routed, 1)[4] == 1
    assert _execute_trace(routed, 0)[4] == 0


@pytest.mark.parametrize("cross_row", [False, True])
@pytest.mark.parametrize("case", ["shared", "nested_shared", "nested_late", "independent"])
def test_snapshot_survives_production_lowering(case, cross_row):
    func = predicate_kernel_factory.get_tir(case=case, cross_row=cross_row, world_size=2)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistRouting")
    routed = result.pass_snapshot("tl.LowerDistRouting").mod
    routed_snapshots = collect_nodes(routed, tir.LetStmt, lambda node: node.var.name.startswith("dist_condition_snapshot_"))
    final_snapshots = collect_nodes(result.device_mod, tir.LetStmt, lambda node: node.var.name.startswith("dist_condition_snapshot_"))
    expected = 1 if case == "shared" else 2
    assert len(routed_snapshots) == expected
    assert len(final_snapshots) == expected


@pytest.mark.parametrize("nested, enabled", [(False, 1), (True, 0), (True, 1)])
def test_side_effecting_condition_is_evaluated_once_when_reached(tmp_path, nested, enabled):
    planned = _planned("nested_shared" if nested else "shared")
    conditions = collect_nodes(planned, tir.IfThenElse)
    target = conditions[0] if nested else conditions[-1]
    probe_buffer = tir.decl_buffer((1,), "int32", name="probe")
    condition = tir.call_extern("int32", "dist_condition_probe", tir.address_of(probe_buffer[0])) > 0

    def replace(node):
        if node.same_as(target):
            return tir.IfThenElse(condition, node.then_case, node.else_case)
        return None

    func = single_prim_func(planned)
    body = tir.stmt_functor.ir_transform(func.body, None, replace, ["tir.IfThenElse"])
    func = tir.PrimFunc([*func.params, probe_buffer.data], body, buffer_map={probe_buffer.data: probe_buffer}, attrs=func.attrs)
    routed = tilelang.transform.LowerDistRouting()(tvm.IRModule({"main": func}))
    addon = tmp_path / "probe.cc"
    addon.write_text(
        '#include <stdint.h>\nextern "C" int32_t dist_condition_probe(int32_t* calls) { ++*calls; return 1; }\n',
        encoding="utf-8",
    )
    probe = tvm.runtime.tensor(np.zeros(1, dtype="int32"))
    trace = _execute_trace(routed, enabled, probe=probe, addon=addon)
    expected = int(not nested or enabled > 0)
    assert probe.numpy()[0] == expected
    assert trace[4] == expected


@pytest.mark.parametrize("case", ["nested_shared", "nested_late"])
def test_nested_snapshot_uses_lazy_scalar_control_flow_in_codegen(tmp_path, case):
    routed = tilelang.transform.LowerDistRouting()(_planned(case))
    func = single_prim_func(routed)

    def local_transport(node):
        if isinstance(node, tir.LetStmt) and isinstance(node.value, tir.Call) and node.value.op.name == "tl.dist_signal_group":
            return node.body
        if not isinstance(node, tir.Evaluate) or not isinstance(node.value, tir.Call):
            return None
        call = node.value
        if call.op.name == "tl.tileop.dist_peer_put":
            copy = tir.Evaluate(tir.Call("handle", tvm.ir.Op.get("tl.tileop.copy"), list(call.args[:2])))
            return tir.IfThenElse(call.args[3].args[-1], copy, None)
        if call.op.name.startswith("tl.dist_"):
            return tir.Evaluate(0)
        return None

    # Rank send codegen is still pending. Replace only the send with a local
    # copy to validate the real snapshot/control-flow path through SunMMIO.
    body = tir.stmt_functor.ir_transform(func.body, None, local_transport, ["tir.LetStmt", "tir.Evaluate"])
    result = lower_to_device_tir(func.with_body(body))
    src = codegen_sunmmio_suvm_mlir(result.device_mod)
    assert "scf.if" in src
    assert "arith.select" not in src
    validate_suvm_mlir_with_npuir_opt(src, tmp_path)
