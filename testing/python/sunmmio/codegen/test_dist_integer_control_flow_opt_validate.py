"""Integer control flow preserves TIR values, lazy reads, and signal guards."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm

from testing.python.sunmmio.common.codegen_validation import (
    codegen_sunmmio_suvm_mlir,
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
