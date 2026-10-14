"""Signal member bounds: static proofs and runtime guards before state access."""

import numpy as np
import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import base_mod, collect_calls, collect_nodes, single_prim_func

tir = tvm.tir


@tilelang.jit(target="sunmmio")
def wait_index_kernel(index_kind="rank", count=2, delta=True, guarded=False, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, runtime_index: T.int32, enabled: T.int32):
        with T.Kernel() as cid:
            signals = T.dist.signals(count, kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE, expect="manual" if delta else None)
            indices = T.alloc_shared((32,), T.int32)
            indices[0] = runtime_index
            if index_kind == "rank":
                index = rank_id
            elif index_kind == "negative":
                index = rank_id - 1
            elif index_kind == "core":
                index = cid
            elif index_kind == "mod":
                index = runtime_index % count
            elif index_kind == "buffer":
                index = indices[0]
            else:
                index = runtime_index
            if guarded:
                if T.And(index >= 0, index < count):
                    T.dist.wait_signal(signals[indices[0] if index_kind == "buffer" else index], expected_delta=1 if delta else None)
            else:
                if enabled != 0:
                    T.dist.wait_signal(signals[indices[0] if index_kind == "buffer" else index], expected_delta=1 if delta else None)
                T.dist.wait_signal(signals[indices[0] if index_kind == "buffer" else index], expected_delta=0 if delta else None)

    return main


@tilelang.jit(target="sunmmio")
def rank_wait_kernel(count=2, guarded=False, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            signals = T.dist.signals(count, expect="manual")
            if guarded:
                if rank_id < count:
                    T.dist.wait_signal(signals[rank_id], expected_delta=1)
            else:
                T.dist.wait_signal(signals[rank_id], expected_delta=1)

    return main


@tilelang.jit(target="sunmmio")
def wave_wait_kernel(guarded=True, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            signals = T.dist.signals(20, kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE, expect="manual")
            for wave in T.serial(3):
                for lane in T.serial(8):
                    index = wave * 8 + lane
                    if guarded:
                        if index < 20:
                            T.dist.wait_signal(signals[index], expected_delta=1)
                    else:
                        T.dist.wait_signal(signals[index], expected_delta=1)

    return main


@tilelang.jit(target="sunmmio")
def invalid_send_kernel(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signals = T.dist.signals(2, expect="manual")
            T.dist.put(src, dst, (rank_id + 1) % world_size, signal=signals[rank_id], submit=True)

    return main


@tilelang.jit(target="sunmmio")
def changing_index_kernel(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, value: T.int32):
        with T.Kernel():
            signals = T.dist.signals(2, expect="manual")
            indices = T.alloc_shared((32,), T.int32)
            indices[0] = value
            T.dist.wait_signal(signals[indices[0]], expected_delta=1)
            indices[0] = value + 1
            T.dist.wait_signal(signals[indices[0]], expected_delta=1)

    return main


@tilelang.jit(target="sunmmio")
def mutated_guard_kernel(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, value: T.int32):
        with T.Kernel():
            signals = T.dist.signals(2, expect="manual")
            indices = T.alloc_shared((32,), T.int32)
            indices[0] = value
            if indices[0] == 0:
                indices[0] = value + 3
                T.dist.wait_signal(signals[indices[0]], expected_delta=1)

    return main


@tilelang.jit(target="sunmmio")
def constant_index_kernel(index=-1, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            signals = T.dist.signals(2, expect="manual")
            T.dist.wait_signal(signals[index], expected_delta=1)

    return main


def _plan(func):
    mod = tilelang.transform.LowerDistCollectives()(base_mod(func))
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.InferSramScope()(mod)
    return tilelang.transform.PlanDistSignals()(mod)


def _bounds_asserts(mod):
    return collect_nodes(mod, tir.AssertStmt, lambda node: "SignalList index" in str(node.message))


def test_rank_wait_rejects_known_out_of_bounds_member():
    with pytest.raises(tvm.error.InternalError, match="SignalList index.*out of range"):
        _plan(rank_wait_kernel.get_tir(world_size=4))


@pytest.mark.parametrize("delta", [False, True])
def test_wait_rejects_known_negative_member(delta):
    with pytest.raises(tvm.error.InternalError, match="SignalList index.*out of range"):
        _plan(wait_index_kernel.get_tir(index_kind="negative", delta=delta, world_size=4))


@pytest.mark.parametrize("count,guarded", [(4, False), (2, True)])
def test_rank_wait_proves_group_range_or_branch_guard(count, guarded):
    result = lower_to_device_tir(rank_wait_kernel.get_tir(count=count, guarded=guarded, world_size=4))
    assert not _bounds_asserts(result.device_mod)
    assert collect_calls(result.device_mod, "tl.dist_wait_signal_")


@pytest.mark.parametrize("index_kind,count,guarded", [("core", 16, False), ("mod", 2, False), ("runtime", 2, True)])
def test_wait_uses_core_arithmetic_and_user_guard_facts(index_kind, count, guarded):
    result = lower_to_device_tir(wait_index_kernel.get_tir(index_kind=index_kind, count=count, guarded=guarded, world_size=4))
    assert not _bounds_asserts(result.device_mod)


def test_wave_tail_guard_is_respected():
    result = lower_to_device_tir(wave_wait_kernel.get_tir(world_size=4))
    assert not _bounds_asserts(result.device_mod)
    with pytest.raises(tvm.error.InternalError, match="SignalList index.*out of range"):
        _plan(wave_wait_kernel.get_tir(guarded=False, world_size=4))


def test_send_topology_still_rejects_out_of_bounds_member():
    with pytest.raises(tvm.error.InternalError, match="member.*group_counts"):
        _plan(invalid_send_kernel.get_tir(world_size=4))


def test_python_negative_index_keeps_normalization():
    result = lower_to_device_tir(constant_index_kernel.get_tir(world_size=4))
    assert not _bounds_asserts(result.device_mod)
    wait = collect_calls(result.device_mod, "tl.dist_wait_signal_")[0]
    assert int(wait.args[1]) == 1
    with pytest.raises(IndexError, match="out of range"):
        constant_index_kernel.get_tir(index=2, world_size=4)


def test_mutable_index_is_sampled_once_per_wait_and_rechecked():
    result = lower_to_device_tir(changing_index_kernel.get_tir(world_size=4))
    guards = _bounds_asserts(result.device_mod)
    assert len(guards) == 2
    snapshots = collect_nodes(
        result.device_mod,
        tir.LetStmt,
        lambda node: isinstance(node.value, tir.BufferLoad) and node.value.buffer.name == "indices",
    )
    assert len(snapshots) == 2
    for snapshot in snapshots:
        body = tir.PrimFunc([], snapshot.body)
        assert not collect_nodes(body, tir.BufferLoad, lambda node: node.buffer.name == "indices")
        wait = collect_calls(body, "tl.dist_wait_signal_")[0]
        assert wait.args[1].same_as(snapshot.var)
        assert wait.args[2].indices[0].same_as(snapshot.var)
        updates = collect_nodes(body, tir.BufferStore, lambda node, data=wait.args[2].buffer.data: node.buffer.data.same_as(data))
        assert len(updates) == 1
        assert updates[0].indices[0].same_as(snapshot.var)


def test_planning_bounds_is_idempotent():
    first = _plan(wait_index_kernel.get_tir(index_kind="runtime", world_size=4))
    second = tilelang.transform.PlanDistSignals()(first)
    assert tvm.ir.structural_equal(first, second)


def test_buffer_write_invalidates_old_branch_range():
    result = lower_to_device_tir(mutated_guard_kernel.get_tir(world_size=4))
    assert len(_bounds_asserts(result.device_mod)) == 1


@pytest.mark.parametrize("delta", [False, True])
@pytest.mark.parametrize("index_kind", ["runtime", "buffer"])
def test_unknown_wait_index_is_guarded_before_expected_and_wait(delta, index_kind):
    result = lower_to_device_tir(
        wait_index_kernel.get_tir(index_kind=index_kind, delta=delta, world_size=4),
        capture_passes=("tl.LowerDistCommunication", "tl.LowerDistSync"),
    )
    lowered = result.pass_snapshot("tl.LowerDistCommunication").mod
    guards = _bounds_asserts(lowered)
    assert len(guards) == 2
    for guard in guards:
        body = tir.PrimFunc([], guard.body)
        assert len(collect_calls(body, "tl.dist_wait_signal_")) == 1
        assert len(collect_calls(body, "tl.dist_expect_")) == int(delta)
    assert len(_bounds_asserts(result.device_mod)) >= 1


def _cpu_wait_trace(device_mod):
    """Execute index/expected logic on CPU; replace physical waits with a trace."""
    func = single_prim_func(device_mod)
    trace = tir.decl_buffer((3,), "int32", name="trace")
    bindings = {}

    def rewrite(node):
        if isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent":
            bindings[node.node.var] = tir.IntImm("int32", 0)
            return node.body
        if isinstance(node, tir.Evaluate) and isinstance(node.value, tir.Call):
            call = node.value
            if call.op.name == "tl.dist_wait_signal_":
                return tir.SeqStmt(
                    [
                        tir.BufferStore(trace, tir.Cast("int32", call.args[1]), [0]),
                        tir.BufferStore(trace, tir.Cast("int32", call.args[2]), [1]),
                        tir.BufferStore(trace, trace[2] + 1, [2]),
                    ]
                )
            if call.op.name == "tl.sunmmio_sync":
                return tir.Evaluate(0)
        return None

    body = tir.stmt_functor.ir_transform(func.body, None, rewrite, ["tir.AttrStmt", "tir.Evaluate"])
    body = tir.stmt_functor.substitute(body, bindings)
    by_name = {param.name: param for param in func.params}
    ordered = [by_name["rank_id"], by_name.get("runtime_index", by_name.get("index")), by_name["enabled"]]
    func = tir.PrimFunc([*ordered, trace.data], body, buffer_map={trace.data: trace})
    func = func.with_attr("global_symbol", "main").with_attr("tir.noalias", True)
    return tvm.compile(func, target="c").jit(options=["-std=c++17"])


@pytest.mark.parametrize("index_kind", ["runtime", "buffer"])
def test_runtime_assert_prevents_bad_index_and_allows_valid_wait(index_kind):
    result = lower_to_device_tir(wait_index_kernel.get_tir(index_kind=index_kind, world_size=4))
    executable = _cpu_wait_trace(result.device_mod)
    for index in (-1, 2, 17):
        trace = tvm.runtime.tensor(np.full(3, 0, dtype="int32"))
        with pytest.raises(Exception, match="SignalList index out of range"):
            executable(0, index, 1, trace)
        np.testing.assert_array_equal(trace.numpy(), [0, 0, 0])
    for index in (0, 1):
        trace = tvm.runtime.tensor(np.full(3, 0, dtype="int32"))
        executable(0, index, 1, trace)
        np.testing.assert_array_equal(trace.numpy(), [index, 1, 2])
