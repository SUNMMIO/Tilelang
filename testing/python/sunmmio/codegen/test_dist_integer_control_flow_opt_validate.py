"""Integer control flow preserves TIR values, lazy reads, and signal guards."""

import pytest
import re
from dataclasses import replace

import tilelang
import tilelang.language as T
from tilelang import tvm

from testing.python.sunmmio.common.codegen_validation import (
    codegen_sunmmio_suvm_mlir,
    simplify_suvm_mlir,
    validate_suvm_mlir_with_npuir_opt,
    write_sunmmio_codegen_logs,
)
from testing.python.sunmmio.common.suvm_trace import trace_suvm_control_flow
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import collect_nodes

tilelang.env.disable_cache()


@tilelang.jit(target="sunmmio")
def integer_loop_kernel(width=8, mode="count", signed=False, world_size: int = 1):
    dtype = T.int32 if signed else {8: T.uint8, 16: T.uint16, 32: T.uint32}[width]

    @T.prim_func
    def main(rank_id: T.dist.RankId, n: dtype):
        with T.Kernel():
            ready = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            if mode == "count":
                for _ in T.serial(n):
                    T.dist.put_signal(ready, (rank_id + 1) % world_size, submit=True)
            else:
                for i in T.serial(n, n + T.Cast(dtype, 2)):
                    T.dist.wait_signal(ready, expected_delta=T.Cast("uint32", i - n if signed else i))

    return main


@tilelang.jit(target="sunmmio")
def conditional_read_kernel(rebuild=False, mode="lazy", world_size: int = 1):
    @T.prim_func
    def main(
        A: T.MeshTensor((768,), T.placement.replicated(), T.float32),
        B: T.MeshTensor((768,), T.placement.replicated(), T.float32),
        rank_id: T.dist.RankId,
        pos: T.int32,
    ):
        with T.Kernel():
            table = T.alloc_shared((16,), T.int32)
            ready = T.dist.signal(expect="manual")
            T.fill(table, 1)
            if mode == "nested":
                off = T.if_then_else(pos >= 0, T.if_then_else(pos < 16, table[pos], 0), 0) * 32
            elif mode == "select":
                off = T.Select(pos > 0, table[3], table[4]) * 32
            elif mode == "division":
                table[0] = T.if_then_else(pos != 0, T.truncdiv(1, pos), 0)
                off = table[0] * 32
            else:
                off = T.if_then_else((pos >= 0) & (pos < 16), table[pos], 0) * 32
            T.dist.put(A[off : off + 32], B[0:32], (rank_id + 1) % world_size, signal=ready)
            if rebuild:  # noqa: SIM102 - compile-time choice enclosing a TIR branch
                if pos > 0:
                    T.dist.submit()
            T.dist.submit()
            T.dist.wait_signal(ready, expected_delta=1)

    return main


@tilelang.jit(target="sunmmio")
def guarded_computation_kernel(communication="none", world_size: int = 1):
    @T.prim_func
    def main(B: T.MeshTensor((256,), T.placement.replicated(), T.int32), rank_id: T.dist.RankId, n: T.int32):
        with T.Kernel():
            tmp = T.alloc_shared((256,), T.int32)
            if communication == "group":
                ready = T.dist.signals(2, kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            elif communication == "single":
                ready = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            T.fill(tmp, 0)
            if n != 0:  # noqa: SIM102 - the outer branch protects the division
                if T.truncdiv(96, n) > 1:
                    tmp[0] = 42
            T.copy(tmp, B)
            if communication != "none":
                T.dist.put_signal(ready[0] if communication == "group" else ready, (rank_id + 1) % world_size, submit=True)
                T.dist.wait_signal(ready[0] if communication == "group" else ready, expected_delta=1)

    return main


@tilelang.jit(target="sunmmio")
def guarded_member_kernel(read=False, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, n: T.int32):
        with T.Kernel():
            ready = T.dist.signals(2, kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            if read:
                table = T.alloc_shared((16,), T.int32)
                T.fill(table, 96)
            if n != 0:
                index = T.truncdiv(table[0] if read else 96, n)
                T.dist.wait_signal(ready[index], expected_delta=0)

    return main


@tilelang.jit(target="sunmmio")
def masked_delta_kernel(mask=1, tail=1, group=False, reverse=False, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, n: T.int32):
        with T.Kernel():
            if group:
                ready = T.dist.signals(2, kind=T.dist.SignalKind.SRAM_FLAGREG_INC, expect="manual")
            else:
                ready = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_INC, expect="manual")
            delta = T.bitwise_and(mask, n) if reverse else T.bitwise_and(n, mask)
            T.dist.wait_signal(ready[0] if group else ready, expected_delta=delta)
            T.dist.wait_signal(ready[0] if group else ready, expected_delta=tail)

    return main


@tilelang.jit(target="sunmmio")
def wide_window_kernel(signed=False, span=2, stride=1, mode="notify", mutate=False, world_size: int = 1):
    dtype = T.int64 if signed else T.uint64
    stride_value = tvm.tir.const(stride, "int64" if signed else "uint64")

    @T.prim_func
    def main(rank_id: T.dist.RankId, n: dtype):
        with T.Kernel():
            ready = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            base = T.alloc_var(dtype, init=n)
            for i in T.serial(base if mutate else n, (base if mutate else n) + T.Cast(dtype, span), stride_value):
                if mode == "notify":
                    T.dist.put_signal(ready, (rank_id + 1) % world_size, submit=True)
                else:
                    delta = T.truncdiv(i, T.Cast(dtype, 1 << 32)) if mode == "high" else i - n
                    T.dist.wait_signal(ready, expected_delta=T.Cast("uint32", delta))
                if mutate:
                    base = 0

    return main


@tilelang.jit(target="sunmmio")
def guarded_flag_kernel(group=False, memory=False, positive=False, amount=1, raw_delta=False, world_size: int = 1):
    kind = T.dist.SignalKind.SRAM_MEMORY if memory else T.dist.SignalKind.SRAM_FLAGREG_INC

    @T.prim_func
    def main(rank_id: T.dist.RankId, n: T.int32):
        with T.Kernel():
            if group:
                ready = T.dist.signals(2, kind=kind, expect="manual")
            else:
                ready = T.dist.signal(kind=kind, expect="manual")
            T.dist.put_signal(ready[0] if group else ready, (rank_id + 1) % world_size, submit=True)
            if n > 0 if positive else n != 0:  # noqa: SIM102 - protect the division
                if T.truncdiv(96, n) > 1:
                    T.dist.wait_signal(ready[0] if group else ready, expected_delta=T.truncdiv(96, n) if raw_delta else amount)

    return main


def generate(factory, tmp_path, **kwargs):
    kernel = factory.get_tir(world_size=2, **kwargs)
    write_sunmmio_codegen_logs(case_name=tmp_path.name, kernel=kernel)
    result = lower_to_device_tir(kernel)
    write_sunmmio_codegen_logs(case_name=tmp_path.name, tir_mod=result.device_mod)
    source = codegen_sunmmio_suvm_mlir(result.device_mod, log_case_name=tmp_path.name)
    validate_suvm_mlir_with_npuir_opt(source, tmp_path, opt_args=("--verify-each", "--suvm-device-validate"))
    return result, source


def test_uint8_loop_keeps_unsigned_trip_count(tmp_path):
    result, source = generate(integer_loop_kernel, tmp_path)
    for n in (0, 1, 127, 128, 255):
        batches, _ = trace_suvm_control_flow(result, source, rank_id=0, n=n)
        assert len(batches) == n
        assert all(len(batch) == 1 and batch[0][0] == "signal" for batch in batches)


@pytest.mark.parametrize("width", [8, 16, 32])
def test_unsigned_loop_min_and_iv_keep_tir_dtype(tmp_path, width):
    result, source = generate(integer_loop_kernel, tmp_path, width=width, mode="window")
    for n in (0, (1 << (width - 1)) - 1, 1 << (width - 1), (1 << width) - 3):
        _, events = trace_suvm_control_flow(result, source, rank_id=0, n=n)
        assert [e[1] for e in events if e[0] == "wait"] == [n, (2 * n + 1) & 0xFFFFFFFF]


def test_signed_loop_keeps_negative_start(tmp_path):
    result, source = generate(integer_loop_kernel, tmp_path, signed=True, mode="window")
    _, events = trace_suvm_control_flow(result, source, rank_id=0, n=-2)
    assert [e[1] for e in events if e[0] == "wait"] == [0, 1]


@pytest.mark.parametrize("rebuild", [False, True])
@pytest.mark.parametrize("mode", ["lazy", "nested"])
def test_if_then_else_reads_only_selected_branch(tmp_path, rebuild, mode):
    result, source = generate(conditional_read_kernel, tmp_path, rebuild=rebuild, mode=mode)
    for pos in (-1, 0, 3, 15, 16, 64):
        batches, events = trace_suvm_control_flow(result, source, rank_id=0, pos=pos)
        selected = 0 <= pos < 16
        data = [p for batch in batches for p in batch if p[0] == "data"]
        assert len(data) == 1 and data[0][3] == (32 if selected else 0)
        assert [e for e in events if e[0] == "pick"] == ([("pick", 1)] if selected else [])


@pytest.mark.parametrize("rebuild", [False, True])
def test_select_retains_eager_operand_evaluation(tmp_path, rebuild):
    result, source = generate(conditional_read_kernel, tmp_path, rebuild=rebuild, mode="select")
    for pos in (-1, 1):
        batches, events = trace_suvm_control_flow(result, source, rank_id=0, pos=pos)
        assert [p[3] for batch in batches for p in batch if p[0] == "data"] == [32]
        assert [e for e in events if e[0] == "pick"] == [("pick", 1), ("pick", 1)]


@pytest.mark.parametrize("rebuild", [False, True])
def test_if_then_else_does_not_speculate_guarded_division(tmp_path, rebuild):
    result, source = generate(conditional_read_kernel, tmp_path, rebuild=rebuild, mode="division")
    for pos, offset in ((0, 0), (1, 32), (2, 0)):
        batches, _ = trace_suvm_control_flow(result, source, rank_id=0, pos=pos)
        assert [p[3] for batch in batches for p in batch if p[0] == "data"] == [offset]


@pytest.mark.parametrize("communication", ["none", "single", "group"])
def test_unrelated_guarded_division_is_accepted(tmp_path, communication):
    _, source = generate(guarded_computation_kernel, tmp_path, communication=communication)
    assert "arith.divsi" in source


@pytest.mark.parametrize("read", [False, True])
def test_unproven_member_keeps_runtime_range_guard(tmp_path, read):
    result, source = generate(guarded_member_kernel, tmp_path, read=read)
    guards = collect_nodes(result.device_mod, tvm.tir.AssertStmt, lambda node: "SignalList index" in str(node.message))
    assert guards
    for n in (0, 64, 96):
        _, events = trace_suvm_control_flow(result, source, rank_id=0, n=n)
        assert [e[1] for e in events if e[0] == "wait"] == ([] if n == 0 else [0])
    for n in (1, -96):
        with pytest.raises(AssertionError):
            trace_suvm_control_flow(result, source, rank_id=0, n=n)


@pytest.mark.parametrize("group", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("mask,tail", [(0, 1), (1, 1), (255, 0)])
def test_constant_mask_preserves_nonnegative_expected_range(tmp_path, group, reverse, mask, tail):
    result, source = generate(masked_delta_kernel, tmp_path, mask=mask, tail=tail, group=group, reverse=reverse)
    for n in (-(1 << 31), -1, 0, 1, 255, (1 << 31) - 1):
        _, events = trace_suvm_control_flow(result, source, rank_id=0, n=n)
        assert [e[1] for e in events if e[0] == "wait"] == [n & mask, (n & mask) + tail]


@pytest.mark.parametrize("group", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_constant_mask_does_not_hide_flag_peak_overflow(group, reverse):
    kernel = masked_delta_kernel.get_tir(mask=255, tail=1, group=group, reverse=reverse, world_size=2)
    with pytest.raises(tvm.error.InternalError, match="cumulative expected"):
        lower_to_device_tir(kernel)


@pytest.mark.parametrize("span,stride", [(2, 1), (5, 2), (3, 3)])
def test_uint64_window_counts_from_zero_and_preserves_submissions(tmp_path, span, stride):
    result, source = generate(wide_window_kernel, tmp_path, span=span, stride=stride)
    count = (span + stride - 1) // stride
    for n in (0, (1 << 63) - 1, 1 << 63, (1 << 64) - span):
        batches, _ = trace_suvm_control_flow(result, source, rank_id=0, n=n)
        assert len(batches) == count
        assert all(len(batch) == 1 and batch[0][0] == "signal" for batch in batches)


def test_uint64_boundary_window_keeps_a_two_iteration_cf_loop(tmp_path):
    result, source = generate(wide_window_kernel, tmp_path)
    func = next(iter(result.device_mod.functions.values()))
    n_arg = next(i for i, p in enumerate(func.params) if p.name == "n")
    lines = source.splitlines()
    function = next(i for i, line in enumerate(lines) if line.strip().startswith("func.func"))
    lines[function + 1 :] = [re.sub(rf"%arg{n_arg}\b", "%fixed_start", line) for line in lines[function + 1 :]]
    lines.insert(function + 1, f"    %fixed_start = arith.constant {(1 << 63) - 1} : i64")
    cf = simplify_suvm_mlir("\n".join(lines), opt_args=("--convert-scf-to-cf", "--canonicalize", "--sccp", "--canonicalize"))
    (tmp_path / "boundary_cf.mlir").write_text(cf)
    constants = {name: int(value) for name, value in re.findall(r"(%[\w-]+) = arith.constant (-?\d+) : index", cf)}
    comparisons = re.findall(r"arith.cmpi slt, (%[\w-]+), (%[\w-]+) : index", cf)
    assert comparisons and constants[comparisons[0][1]] == 2
    entry = re.search(r"cf.br \^bb\d+\((%[\w-]+)[^\n]*: index", cf)
    assert entry and constants[entry[1]] == 0
    increments = re.findall(rf"arith.addi {re.escape(comparisons[0][0])}, (%[\w-]+) : index", cf)
    assert increments and constants[increments[0]] == 1
    assert "suvm.dist.batch_submit" in cf


def test_uint64_window_restores_full_width_iv(tmp_path):
    result, source = generate(wide_window_kernel, tmp_path, span=5, stride=2, mode="high")
    for n in (0, (1 << 63) - 1, 1 << 63, (1 << 64) - 5):
        _, events = trace_suvm_control_flow(result, source, rank_id=0, n=n)
        total, expected = 0, []
        for offset in (0, 2, 4):
            total = (total + ((n + offset) >> 32)) & 0xFFFFFFFF
            expected.append(total)
        assert [e[1] for e in events if e[0] == "wait"] == expected


@pytest.mark.parametrize("stride", [2, 1 << 63])
def test_native_uint64_for_step_is_applied_to_logical_iv(tmp_path, stride):
    # Public stepped serial is normalized by the frontend. Exercise For.step
    # directly on device TIR so this tests the backend's own trip-count formula.
    result = lower_to_device_tir(wide_window_kernel.get_tir(span=5, mode="offset", world_size=2))
    func = next(iter(result.device_mod.functions.values()))
    step = tvm.tir.Cast("uint64", tvm.tir.IntImm("int64", stride - (1 << 64))) if stride >= 1 << 63 else tvm.tir.const(stride, "uint64")

    def rewrite(node):
        if isinstance(node, tvm.tir.For):
            return tvm.tir.For(node.loop_var, node.min, node.extent, node.kind, node.body, node.thread_binding, node.annotations, step=step)

    body = tvm.tir.stmt_functor.ir_transform(func.body, None, rewrite, ["tir.For"])
    result = replace(result, device_mod=tvm.IRModule({"main": func.with_body(body)}))
    source = codegen_sunmmio_suvm_mlir(result.device_mod, log_case_name=tmp_path.name)
    validate_suvm_mlir_with_npuir_opt(source, tmp_path, opt_args=("--verify-each", "--suvm-device-validate"))
    for n in (0, (1 << 63) - 1, (1 << 64) - 5):
        _, events = trace_suvm_control_flow(result, source, rank_id=0, n=n)
        expected, total = [], 0
        for offset in range(0, 5, stride):
            total += offset
            expected.append(total)
        assert [e[1] for e in events if e[0] == "wait"] == expected


@pytest.mark.parametrize("signed", [False, True])
def test_wide_loop_uses_entry_min_after_a_local_write(tmp_path, signed):
    result, source = generate(wide_window_kernel, tmp_path, signed=signed, mode="offset", mutate=True)
    for n in (-2, 5, (1 << 63) - 2) if signed else (5, (1 << 63) - 1):
        _, events = trace_suvm_control_flow(result, source, rank_id=0, n=n)
        assert [e[1] for e in events if e[0] == "wait"] == [0, 1]


def test_int64_window_keeps_negative_start_and_last_representable_iv(tmp_path):
    result, source = generate(wide_window_kernel, tmp_path, signed=True, mode="offset")
    for n in (-2, (1 << 63) - 2):
        _, events = trace_suvm_control_flow(result, source, rank_id=0, n=n)
        assert [e[1] for e in events if e[0] == "wait"] == [0, 1]


@pytest.mark.parametrize("span", [0, -1])
def test_int64_nonpositive_extent_has_no_iterations(tmp_path, span):
    result, source = generate(wide_window_kernel, tmp_path, signed=True, span=span)
    batches, _ = trace_suvm_control_flow(result, source, rank_id=0, n=-(1 << 63))
    assert not batches


@pytest.mark.parametrize("mode", ["count", "step", "zero_step"])
def test_uint64_unproven_control_bounds_are_rejected(mode):
    func = wide_window_kernel.get_tir(world_size=2)
    result = lower_to_device_tir(func)
    device_func = next(iter(result.device_mod.functions.values()))
    n = next(p for p in device_func.params if p.name == "n")

    def rewrite(node):
        if isinstance(node, tvm.tir.For):
            return tvm.tir.For(
                node.loop_var,
                node.min,
                n if mode == "count" else node.extent,
                node.kind,
                node.body,
                node.thread_binding,
                node.annotations,
                step=n if mode == "step" else tvm.tir.const(0, "uint64") if mode == "zero_step" else node.step,
            )

    body = tvm.tir.stmt_functor.ir_transform(device_func.body, None, rewrite, ["tir.For"])
    mod = tvm.IRModule({"main": device_func.with_body(body)})
    with pytest.raises(tvm.error.InternalError, match="trip count must fit|provably positive step"):
        codegen_sunmmio_suvm_mlir(mod)


@pytest.mark.parametrize("count,valid", [((1 << 63) - 1, True), (1 << 63, False)])
def test_uint64_trip_count_checks_the_signed_index_limit(tmp_path, count, valid):
    result = lower_to_device_tir(wide_window_kernel.get_tir(world_size=2))
    func = next(iter(result.device_mod.functions.values()))

    def rewrite(node):
        if isinstance(node, tvm.tir.For):
            return tvm.tir.For(
                node.loop_var,
                node.min,
                tvm.tir.const(count, "uint64"),
                node.kind,
                node.body,
                node.thread_binding,
                node.annotations,
                step=node.step,
            )

    body = tvm.tir.stmt_functor.ir_transform(func.body, None, rewrite, ["tir.For"])
    mod = tvm.IRModule({"main": func.with_body(body)})
    if valid:
        source = codegen_sunmmio_suvm_mlir(mod)
        validate_suvm_mlir_with_npuir_opt(source, tmp_path, opt_args=("--verify-each", "--suvm-device-validate"))
    else:
        with pytest.raises(tvm.error.InternalError, match="trip count must fit"):
            codegen_sunmmio_suvm_mlir(mod)


@pytest.mark.parametrize("group", [False, True])
@pytest.mark.parametrize("memory,positive", [(False, False), (False, True), (True, False)])
def test_guarded_flag_wait_merges_unknown_condition_conservatively(tmp_path, group, memory, positive):
    result, source = generate(guarded_flag_kernel, tmp_path, group=group, memory=memory, positive=positive)
    for n in (0, -1, 1, 48, 49):
        _, events = trace_suvm_control_flow(result, source, rank_id=0, n=n)
        assert [e[1] for e in events if e[0] == "wait"] == ([1] if 0 < n <= 48 else [])


@pytest.mark.parametrize("group", [False, True])
@pytest.mark.parametrize("amount", [255, 256])
def test_guarded_flag_wait_still_checks_the_peak(tmp_path, group, amount):
    if amount == 256:
        with pytest.raises(tvm.error.InternalError, match="cumulative expected"):
            lower_to_device_tir(guarded_flag_kernel.get_tir(group=group, amount=amount, world_size=2))
    else:
        generate(guarded_flag_kernel, tmp_path, group=group, amount=amount)


def test_condition_fallback_does_not_relax_delta_proofs():
    with pytest.raises(tvm.error.InternalError, match="finite-width integer|non-negative"):
        lower_to_device_tir(guarded_flag_kernel.get_tir(raw_delta=True, world_size=2))
