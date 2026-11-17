"""Direct batch scopes, loop-carried handles, and staged diagnostics."""

import os
import re

import pytest
import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm

from testing.python.sunmmio.common.codegen_validation import (
    codegen_sunmmio_suvm_mlir,
    validate_suvm_mlir_with_npuir_opt,
    write_sunmmio_codegen_logs,
)
from testing.python.sunmmio.common.suvm_trace import trace_suvm_control_flow as _trace_reconstructed_source
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import collect_calls, collect_nodes, single_prim_func

tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
os.environ.setdefault("SUNMMIO_TEST_LOG_IR", "0")


@tilelang.jit(target="sunmmio")
def batch_kernel(mode="conditional_append", rounds=3, dynamic=False, literal=None, world_size: int = 1):
    shape = (128,)

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, T.placement.replicated(), T.float32),
        B: T.MeshTensor(shape, T.placement.replicated(), T.float32),
        rank_id: T.dist.RankId,
        active: T.int32,
        limit: T.int32,
    ):
        with T.Kernel():
            ready = T.dist.signal(kind=T.dist.SignalKind.DRAM_FLAGREG_VALUE, expect="manual")
            peer = (rank_id + 1) % world_size
            T.assume(limit >= 0)
            T.assume(limit <= 3)
            n = limit if dynamic else rounds
            condition = active > 0 if literal is None else literal
            if mode == "conditional_append":
                if condition:
                    T.dist.put(A[0:32], B[0:32], peer, signal=ready)
                T.dist.submit()
            if mode == "nested_append":
                if condition:
                    if active < 5:
                        T.dist.put(A[0:32], B[0:32], peer, signal=ready)
                else:
                    T.dist.put(A[32:64], B[32:64], peer, signal=ready)
                T.dist.submit()
            if mode == "prefix_conditional":
                T.dist.put(A[0:32], B[0:32], peer, signal=ready)
                if condition:
                    T.dist.put(A[32:64], B[32:64], peer, signal=ready)
                T.dist.submit()
            if mode == "branch_local":
                if condition:
                    T.dist.put(A[0:32], B[0:32], peer, signal=ready, submit=True)
                else:
                    T.dist.put(A[32:64], B[32:64], peer, signal=ready, submit=True)
            if mode == "for_append":
                for _i in T.serial(n):
                    T.dist.put(A[0:32], B[0:32], peer, signal=ready)
                T.dist.submit()
            if mode == "for_local":
                for _i in T.serial(n):
                    T.dist.put(A[0:32], B[0:32], peer, signal=ready, submit=True)
            if mode == "while_append" or mode == "while_local":
                i = T.alloc_var(T.int32, init=0)
                while i < n:
                    T.dist.put(A[0:32], B[0:32], peer, signal=ready)
                    if mode == "while_local":
                        T.dist.submit()
                    i += 1
                if mode == "while_append":
                    T.dist.submit()
            if mode == "for_carried" or mode == "for_empty_output" or mode == "for_empty_input":
                if mode != "for_empty_input":
                    T.dist.put_signal(ready, peer)
                for _i in T.serial(n):
                    T.dist.submit()
                    if mode != "for_empty_output":
                        T.dist.put_signal(ready, peer)
                T.dist.submit()
            if mode == "for_carried_payload":
                T.dist.put(A[0:32], B[0:32], peer, signal=ready)
                for _i in T.serial(n):
                    T.dist.put(A[32:64], B[32:64], peer, signal=ready)
                    T.dist.submit()
                    T.dist.put(A[64:96], B[64:96], peer, signal=ready)
                T.dist.submit()
            if mode == "nested_carried":
                T.dist.put_signal(ready, peer)
                for _outer in T.serial(n):
                    for _inner in T.serial(2):
                        T.dist.submit()
                        T.dist.put_signal(ready, peer)
                T.dist.submit()
            if mode == "conditional_submit":
                T.dist.put(A[0:32], B[0:32], peer, signal=ready)
                if condition:
                    T.dist.submit()
                T.dist.put(A[32:64], B[32:64], peer, signal=ready)
                T.dist.submit()
            if mode == "while_carried":
                i = T.alloc_var(T.int32, init=0)
                T.dist.put_signal(ready, peer)
                while i < n:
                    T.dist.submit()
                    T.dist.put_signal(ready, peer)
                    i += 1
                T.dist.submit()
            if mode == "conditional_for_local":
                for _i in T.serial(n):
                    if condition:
                        T.dist.put(A[0:32], B[0:32], peer, signal=ready, submit=True)
            if mode == "partial_branch":
                if condition:
                    T.dist.put(A[0:32], B[0:32], peer, signal=ready, submit=True)
                else:
                    T.dist.put(A[32:64], B[32:64], peer, signal=ready)
                T.dist.submit()
            T.dist.wait_signal(ready, expected_delta=0)

    return main


@tilelang.jit(target="sunmmio")
def mixed_compute_kernel(world_size: int = 1):
    shape = (32, 32)
    out_shape = (64, 32)

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, T.placement.replicated(), T.bfloat16),
        B: T.MeshTensor(shape, T.placement.replicated(), T.bfloat16),
        D: T.MeshTensor(out_shape, T.placement.replicated(), T.bfloat16),
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            ready = T.dist.signal()
            a = T.alloc_shared(shape, T.bfloat16)
            b = T.alloc_shared(shape, T.bfloat16)
            c = T.alloc_shared(shape, T.float32)
            T.dist.put(A, D[0:32, :], (rank_id + 1) % world_size, signal=ready)
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, c, clear_accum=True)
            T.dist.put(B, D[32:64, :], (rank_id + 1) % world_size, signal=ready)
            T.dist.submit()
            T.dist.wait_signal(ready)

    return main


@tilelang.jit(target="sunmmio")
def reconstruction_kernel(mode="two", kind=T.dist.SignalKind.DRAM_FLAGREG_VALUE, dst_shared=False, world_size: int = 1):
    shape = (128,)

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, T.placement.replicated(), T.float32),
        B: T.MeshTensor(shape, T.placement.replicated(), T.float32),
        rank_id: T.dist.RankId,
        early: T.int32,
        second: T.int32,
        active: T.int32,
        probe_index: T.int32,
    ):
        with T.Kernel():
            ready = T.dist.signal(kind=kind, expect="manual")
            local_dst = T.alloc_shared(shape, T.float32)
            target = local_dst if dst_shared else B
            T.assume(second >= 0)
            T.assume(second <= 3)
            if mode == "hybrid":
                for _prefix in T.serial(second):
                    T.dist.put(A[96:128], target[96:128], (rank_id + 1) % world_size, signal=ready)
                T.dist.submit()
            slot = T.alloc_var(T.int32, init=0)
            gate = T.alloc_var(T.int32, init=early)
            table = T.alloc_shared((16,), T.int32)
            table[0] = 0
            if mode == "conditional" or mode == "array" or mode == "snapshot":
                if active > 0:
                    start = table[probe_index] * 32 if mode != "conditional" else slot * 32
                    if mode == "snapshot":
                        table[probe_index] = 1
                    T.dist.put(A[start : start + 32], target[start : start + 32], (rank_id + 1) % world_size, signal=ready)
            else:
                T.dist.put(A[slot * 32 : slot * 32 + 32], target[slot * 32 : slot * 32 + 32], (rank_id + 1) % world_size, signal=ready)
            slot += 1
            table[0] = 1
            if mode == "both_branches":
                if gate > 0:
                    T.dist.submit()
                else:
                    T.dist.submit()
            else:
                if gate > 0:
                    T.dist.submit()
            gate = 1 - gate
            if mode != "no_b" and mode != "both_branches":
                destination = (slot + gate * 2) * 32
                T.dist.put(A[slot * 32 : slot * 32 + 32], target[destination : destination + 32], (rank_id + 1) % world_size, signal=ready)
            if mode == "multiple":
                if second > 0:
                    T.dist.submit()
                slot += 1
                T.dist.put(A[slot * 32 : slot * 32 + 32], target[slot * 32 : slot * 32 + 32], (rank_id + 1) % world_size, signal=ready)
            if mode != "both_branches":
                T.dist.submit()
            T.dist.wait_signal(ready, expected_delta=0)

    return main


@tilelang.jit(target="sunmmio")
def reconstruction_loop_kernel(loop="for", history=False, world_size: int = 1):
    shape = (128,)

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, T.placement.replicated(), T.float32),
        B: T.MeshTensor(shape, T.placement.replicated(), T.float32),
        rank_id: T.dist.RankId,
        early: T.int32,
        limit: T.int32,
    ):
        with T.Kernel():
            ready = T.dist.signal(kind=T.dist.SignalKind.DRAM_FLAGREG_VALUE, expect="manual")
            T.assume(limit >= 0)
            T.assume(limit <= 3)
            if history:
                T.dist.put_signal(ready, (rank_id + 1) % world_size)
            if loop == "for":
                for _i in T.serial(limit):
                    if not history:
                        T.dist.put(A[0:32], B[0:32], (rank_id + 1) % world_size, signal=ready)
                    if early > 0:
                        T.dist.submit()
                    T.dist.put(A[32:64], B[32:64], (rank_id + 1) % world_size, signal=ready)
                    if not history:
                        T.dist.submit()
            else:
                i = T.alloc_var(T.int32, init=0)
                while i < limit:
                    if not history:
                        T.dist.put(A[0:32], B[0:32], (rank_id + 1) % world_size, signal=ready)
                    if early > 0:
                        T.dist.submit()
                    T.dist.put(A[32:64], B[32:64], (rank_id + 1) % world_size, signal=ready)
                    if not history:
                        T.dist.submit()
                    i += 1
            T.dist.submit()
            T.dist.wait_signal(ready, expected_delta=0)

    return main


@tilelang.jit(target="sunmmio")
def routing_capture_kernel(world_size: int = 1):
    shape = (64,)

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, T.placement.replicated(), T.float32),
        B: T.MeshTensor(shape, T.placement.replicated(), T.float32),
        rank_id: T.dist.RankId,
        early: T.int32,
    ):
        with T.Kernel():
            ready = T.dist.signals(world_size, kind=T.dist.SignalKind.DRAM_FLAGREG_INC, expect="manual")
            peer = (rank_id + 1) % world_size
            member = rank_id
            T.dist.put(A[0:32], B[0:32], peer, signal=ready[member])
            if early > 0:
                T.dist.submit()
            next_peer = (rank_id + 2) % world_size
            next_member = (rank_id + 1) % world_size
            T.dist.put(A[32:64], B[32:64], next_peer, signal=ready[next_member])
            T.dist.submit()

    return main


@tilelang.jit(target="sunmmio")
def unavailable_capture_root_kernel(world_size: int = 1):
    @T.prim_func
    def main(A: T.MeshTensor((32,), T.placement.replicated(), T.float32), rank_id: T.dist.RankId, active: T.int32, early: T.int32):
        with T.Kernel():
            ready = T.dist.signal(expect="manual")
            if active > 0:
                dst = T.alloc_shared((32,), T.float32)
                T.dist.put(A, dst, (rank_id + 1) % world_size, signal=ready)
            if early > 0:
                T.dist.submit()
            T.dist.submit()

    return main


def generate(factory, tmp_path, **kwargs):
    case_name = tmp_path.name
    kernel = factory.get_tir(world_size=4, **kwargs)
    write_sunmmio_codegen_logs(case_name=case_name, kernel=kernel)
    result = lower_to_device_tir(kernel)
    write_sunmmio_codegen_logs(case_name=case_name, tir_mod=result.device_mod)
    source = codegen_sunmmio_suvm_mlir(result.device_mod, log_case_name=case_name)
    validate_suvm_mlir_with_npuir_opt(source, tmp_path, opt_args=("--verify-each", "--suvm-device-validate"))
    return result, source


@pytest.mark.parametrize("mode", ["conditional_append", "nested_append"])
def test_conditional_appends_share_outer_batch(tmp_path, mode):
    _, source = generate(batch_kernel, tmp_path, mode=mode)
    begin = source.index("suvm.dist.batch_begin")
    condition = source.index("scf.if")
    submit = source.index("suvm.dist.batch_submit")
    assert begin < condition < source.index("suvm.dist.batch_put") < submit
    assert source.count("suvm.dist.batch_begin") == source.count("suvm.dist.batch_submit") == 1
    handles = re.findall(r"suvm.dist.batch_(?:put|signal|submit) (%\w+)", source)
    assert len(set(handles)) == 1
    # False branches can append nothing; the unconditional submit remains.
    assert "} else {" in source[condition:submit]


@pytest.mark.parametrize("mode", ["branch_local", "partial_branch", "conditional_for_local"])
def test_branches_can_finish_independent_batches(tmp_path, mode):
    _, source = generate(batch_kernel, tmp_path, mode=mode)
    assert "scf.if" in source
    assert source.count("suvm.dist.batch_begin") == source.count("suvm.dist.batch_submit")
    assert source.count("suvm.dist.batch_begin") >= 1


@pytest.mark.parametrize("mode", ["for_append", "for_local", "while_append", "while_local"])
@pytest.mark.parametrize("rounds", [0, 1, 3])
def test_loop_batches_validate_for_zero_one_many_iterations(tmp_path, mode, rounds):
    result, source = generate(batch_kernel, tmp_path, mode=mode, rounds=rounds)
    if collect_calls(result.device_mod, "tl.dist_put_"):
        assert source.count("suvm.dist.batch_put") == source.count("suvm.dist.batch_signal") == 1
        loop_op = "scf.while" if mode.startswith("while") else "scf.for"
        if loop_op not in source:
            assert rounds == 1
            return
        loop = source.index(loop_op)
        begin = source.index("suvm.dist.batch_begin")
        submit = source.index("suvm.dist.batch_submit")
        if mode.endswith("append"):
            assert begin < loop < submit
        else:
            assert loop < begin < submit


@pytest.mark.parametrize("mode", ["for_append", "for_local", "while_append", "while_local"])
def test_dynamic_loops_use_direct_batches(tmp_path, mode):
    _, source = generate(batch_kernel, tmp_path, mode=mode, dynamic=True)
    assert "suvm.dist.batch_put" in source
    if mode.startswith("while"):
        signature = next(line for line in source.splitlines() if "scf.while" in line)
        assert "!suvm.dist.batch" not in signature


@pytest.mark.parametrize("mode", ["for_carried", "for_empty_input", "for_empty_output"])
@pytest.mark.parametrize("rounds", [0, 1, 3])
def test_for_batch_handoff_keeps_zero_iteration_path(tmp_path, mode, rounds):
    _, source = generate(batch_kernel, tmp_path, mode=mode, rounds=rounds)
    assert "suvm.dist.batch_submit" in source or (mode == "for_empty_input" and rounds == 0)
    if rounds > 1:
        loop = next(line for line in source.splitlines() if "scf.for" in line)
        assert "iter_args" in loop and "!suvm.dist.batch" in loop
        assert re.search(r"scf.yield.*!suvm.dist.batch", source)
        last_submit = source.rindex("suvm.dist.batch_submit")
        assert "scf.yield" in source[:last_submit]


@pytest.mark.parametrize("mode", ["for_carried", "for_empty_input", "for_empty_output"])
def test_dynamic_for_carries_handle_with_scalar_state(tmp_path, mode):
    _, source = generate(batch_kernel, tmp_path, mode=mode, dynamic=True)
    loop = next(line for line in source.splitlines() if "scf.for" in line)
    assert "!suvm.dist.batch" in loop
    if mode != "for_empty_output":
        assert "i8" in loop
    assert source.count("suvm.dist.batch_submit") == 2
    pairs = re.findall(r"(%\w+) = (%\w+(?:#\d+)?)", loop.split("iter_args(", 1)[1].split(")", 1)[0])
    batch_arg, initial = pairs[-1]
    loop_start = source.index(loop)
    assert f"{initial} = suvm.dist.batch_begin" in source[:loop_start]
    assert f"suvm.dist.batch_submit {batch_arg}" in source[loop_start:]
    yielded = next(line for line in source.splitlines() if "scf.yield" in line and "!suvm.dist.batch" in line)
    batch_yield = yielded.split("scf.yield ", 1)[1].split(" : ", 1)[0].split(", ")[-1]
    assert f"{batch_yield} = suvm.dist.batch_begin" in source[loop_start:]
    results = re.search(r"(%\w+)(?::(\d+))? = scf.for", loop)
    assert results
    result = results[1] + (f"#{len(pairs) - 1}" if results[2] else "")
    assert f"suvm.dist.batch_submit {result}" in source


@pytest.mark.parametrize("literal", [False, True])
def test_constant_branch_extremes_keep_only_executed_sends(tmp_path, literal):
    _, source = generate(batch_kernel, tmp_path, literal=literal)
    assert source.count("suvm.dist.batch_put") == int(literal)
    assert source.count("suvm.dist.batch_signal") == int(literal)


@pytest.mark.parametrize("mode", ["conditional_submit", "while_carried"])
def test_finite_reconstruction_extends_direct_control_flow(tmp_path, mode):
    _, source = generate(batch_kernel, tmp_path, mode=mode)
    assert "suvm.dist.batch_submit" in source
    assert all("!suvm.dist.batch" not in line for line in source.splitlines() if "scf.while" in line)


def test_copy_and_gemm_keep_original_position_inside_batch(tmp_path):
    _, source = generate(mixed_compute_kernel, tmp_path)
    first = source.index("suvm.dist.batch_put")
    last = source.rindex("suvm.dist.batch_put")
    assert "suvm.copy_async" in source[first:last]
    assert "suvm.tc.mma" in source[first:last]
    assert source.count("suvm.dist.batch_submit") == 1


def test_nested_for_handoffs_use_legal_iter_args(tmp_path):
    _, source = generate(batch_kernel, tmp_path, mode="nested_carried", dynamic=True)
    loops = [line for line in source.splitlines() if "scf.for" in line]
    assert len(loops) == 2
    assert all("!suvm.dist.batch" in line and "iter_args" in line for line in loops)
    assert source.count("suvm.dist.batch_signal") == 2
    assert source.count("suvm.dist.batch_submit") == 2


def test_conditional_append_keeps_preceding_entries_in_same_batch(tmp_path):
    _, source = generate(batch_kernel, tmp_path, mode="prefix_conditional")
    first = source.index("suvm.dist.batch_put")
    last = source.rindex("suvm.dist.batch_put")
    assert first < source.index("scf.if") < last
    handles = re.findall(r"suvm.dist.batch_(?:put|signal|submit) (%\w+)", source)
    assert len(set(handles)) == 1
    assert source.count("suvm.dist.batch_submit") == 1


def test_for_argument_accepts_payload_appends_before_consumption(tmp_path):
    _, source = generate(batch_kernel, tmp_path, mode="for_carried_payload", dynamic=True)
    loop = next(line for line in source.splitlines() if "scf.for" in line)
    pairs = re.findall(r"(%\w+) = (%\w+)", loop.split("iter_args(", 1)[1].split(")", 1)[0])
    argument = pairs[-1][0]
    put = source.index(f"suvm.dist.batch_put {argument},")
    submit = source.index(f"suvm.dist.batch_submit {argument}")
    assert put < submit
    assert source.count("suvm.dist.batch_put") == source.count("suvm.dist.batch_signal") == 3


@pytest.mark.parametrize(
    "mode,early,second,active",
    [(mode, early, 0, 1) for mode in ("two", "no_b") for early in (0, 1)]
    + [(mode, early, 0, active) for mode in ("conditional", "array") for early in (0, 1) for active in (0, 1)]
    + [("multiple", early, second, 1) for early in (0, 1) for second in (0, 1)],
)
def test_reconstruction_preserves_original_values_and_exact_send_order(tmp_path, mode, early, second, active):
    result, source = generate(reconstruction_kernel, tmp_path, mode=mode)
    batches, events = _trace_reconstructed_source(result, source, rank_id=0, early=early, second=second, active=active, probe_index=0)
    expected, queue, generation = [], [], 0

    def append(offset, destination=None):
        nonlocal generation
        generation += 1
        queue.extend([("data", 1, "A", offset, "B", offset if destination is None else destination, 32), ("signal", 1, 0, generation)])

    def submit():
        if queue:
            expected.append(tuple(queue))
        queue.clear()

    if mode not in ("conditional", "array") or active:
        append(0)
    if early:
        submit()
    if mode != "no_b":
        append(32, 32 if early else 96)
    if mode == "multiple":
        if second:
            submit()
        append(64)
    submit()
    assert batches == expected
    for i, event in enumerate(events):
        if event[0] == "submit":
            assert events[i - 1][0] == "flush"
    assert "memref." not in source


def test_inactive_capture_does_not_evaluate_branch_only_address(tmp_path):
    result, source = generate(reconstruction_kernel, tmp_path, mode="array")
    batches, _ = _trace_reconstructed_source(result, source, rank_id=0, early=1, second=0, active=0, probe_index=999)
    assert [entry[3] for batch in batches for entry in batch if entry[0] == "data"] == [32]


@pytest.mark.parametrize("loop", ["for", "while"])
@pytest.mark.parametrize("early,limit", [(0, 0), (0, 3), (1, 3)])
def test_reconstruction_is_finite_with_iteration_local_drain(tmp_path, loop, early, limit):
    result, source = generate(reconstruction_loop_kernel, tmp_path, loop=loop)
    batches, _ = _trace_reconstructed_source(result, source, rank_id=0, early=early, limit=limit)
    signals = [[item[3] for item in batch if item[0] == "signal"] for batch in batches]
    expected = [[value] for value in range(1, limit * 2 + 1)] if early else [[value, value + 1] for value in range(1, limit * 2 + 1, 2)]
    assert signals == expected


@pytest.mark.parametrize("loop", ["for", "while"])
def test_accumulating_instances_reject_dynamic_send_history(tmp_path, loop):
    with pytest.raises(tvm.error.InternalError, match="dynamic send history"):
        generate(reconstruction_loop_kernel, tmp_path, loop=loop, history=True)


@pytest.mark.parametrize("early", [0, 1])
def test_capture_keeps_original_peer_and_member(tmp_path, early):
    result, source = generate(routing_capture_kernel, tmp_path)
    batches, _ = _trace_reconstructed_source(result, source, rank_id=0, early=early)
    signals = [item[1:] for batch in batches for item in batch if item[0] == "signal"]
    assert signals == [(1, 0, None), (2, 1, None)]


def test_capture_uses_actual_tir_root_allocation_scope(tmp_path):
    _, source = generate(unavailable_capture_root_kernel, tmp_path)
    assert source.index("suvm.alloc") < source.index("scf.if")
    assert all("!suvm.tile_view" not in line for line in source.splitlines() if "scf.if" in line)


def test_capture_does_not_extend_root_allocation_lifetime(tmp_path):
    kernel = unavailable_capture_root_kernel.get_tir(world_size=4)
    write_sunmmio_codegen_logs(case_name=tmp_path.name, kernel=kernel)
    result = lower_to_device_tir(kernel)
    func = single_prim_func(result.device_mod)
    destination = next(node.buffer for node in collect_nodes(func, tvm.tir.DeclBuffer) if node.buffer.name == "dst")

    def restrict_lifetime(node):
        if isinstance(node, tvm.tir.DeclBuffer) and node.buffer.data.same_as(destination.data):
            return node.body
        if isinstance(node, tvm.tir.Allocate) and node.buffer_var.same_as(destination.data):
            return node.body
        if isinstance(node, tvm.tir.IfThenElse) and collect_calls(tvm.tir.PrimFunc([], node.then_case), "tl.dist_put_"):
            declared = tvm.tir.DeclBuffer(destination, node.then_case)
            scoped = tvm.tir.Allocate(destination.data, destination.dtype, destination.shape, tvm.tir.const(True), declared)
            return tvm.tir.IfThenElse(node.condition, scoped, node.else_case)

    # Exercise the backend lifetime boundary on final device IR; the ordinary
    # frontend deliberately hoists this allocation to a longer-lived scope.
    body = tvm.tir.stmt_functor.ir_transform(func.body, None, restrict_lifetime, ["tir.Allocate", "tir.DeclBuffer", "tir.IfThenElse"])
    mod = tvm.IRModule({"restricted_root": func.with_body(body)})
    write_sunmmio_codegen_logs(case_name=tmp_path.name, tir_mod=mod)
    with pytest.raises(tvm.error.InternalError, match="root storage is unavailable at submit"):
        codegen_sunmmio_suvm_mlir(mod, log_case_name=tmp_path.name)


@pytest.mark.parametrize("kind,dst_shared", [(T.dist.SignalKind.DRAM_FLAGREG_INC, False), (T.dist.SignalKind.SRAM_MEMORY, True)])
@pytest.mark.parametrize("early", [0, 1])
def test_capture_respects_increment_and_memory_interfaces(tmp_path, kind, dst_shared, early):
    result, source = generate(reconstruction_kernel, tmp_path, kind=kind, dst_shared=dst_shared)
    batches, _ = _trace_reconstructed_source(result, source, rank_id=0, early=early, second=0, active=1, probe_index=0)
    notifications = [item[3] for batch in batches for item in batch if item[0] == "signal"]
    assert notifications == ([None, None] if kind == T.dist.SignalKind.DRAM_FLAGREG_INC else [1, 2])
    if dst_shared:
        assert "i32 to i64" in source


@pytest.mark.parametrize("early,prefix", [(0, 0), (1, 0), (0, 3), (1, 3)])
def test_reconstruction_keeps_independent_dynamic_batches_direct(tmp_path, early, prefix):
    result, source = generate(reconstruction_kernel, tmp_path, mode="hybrid")
    batches, _ = _trace_reconstructed_source(result, source, rank_id=0, early=early, second=prefix, active=1, probe_index=0)
    offsets = [item[3] for batch in batches for item in batch if item[0] == "data"]
    generations = [item[3] for batch in batches for item in batch if item[0] == "signal"]
    assert offsets == [96] * prefix + [0, 32]
    assert generations == list(range(1, prefix + 3))
    loop = next(line for line in source.splitlines() if "scf.for" in line)
    assert "!suvm.dist.batch" not in loop and "index" not in loop.split("->", 1)[1]


@pytest.mark.parametrize("limit", [0, 3])
def test_finite_while_queue_carries_saved_values_as_scalars(tmp_path, limit):
    result, source = generate(batch_kernel, tmp_path, mode="while_carried", dynamic=True)
    batches, _ = _trace_reconstructed_source(result, source, rank_id=0, active=1, limit=limit, n=limit)
    generations = [item[3] for batch in batches for item in batch if item[0] == "signal"]
    assert generations == list(range(1, limit + 2))
    header = next(line for line in source.splitlines() if "scf.while" in line)
    assert "!suvm.dist.batch" not in header


@pytest.mark.parametrize("early", [0, 1])
def test_snapshot_let_is_not_reevaluated_after_memory_write(tmp_path, early):
    result, source = generate(reconstruction_kernel, tmp_path, mode="snapshot")
    batches, events = _trace_reconstructed_source(result, source, rank_id=0, early=early, second=0, active=1, probe_index=0)
    offsets = [item[3] for batch in batches for item in batch if item[0] == "data"]
    assert offsets == [0, 32]
    assert [event[1] for event in events if event[0] == "pick"] == [0]


@pytest.mark.parametrize("early", [0, 1])
def test_both_submit_branches_consume_saved_entry_once(tmp_path, early):
    result, source = generate(reconstruction_kernel, tmp_path, mode="both_branches")
    batches, events = _trace_reconstructed_source(result, source, rank_id=0, early=early, second=0, active=1, probe_index=0)
    assert len(batches) == 1
    assert [item[3] for item in batches[0] if item[0] == "data"] == [0]
    assert [item[3] for item in batches[0] if item[0] == "signal"] == [1]
    assert sum(event[0] == "submit" for event in events) == 1
    assert source.count("suvm.dist.batch_submit") == 2


if __name__ == "__main__":
    tilelang.testing.main()
