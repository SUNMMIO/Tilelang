"""Inter-rank resources, sequential batches, regions, and source completion."""

import os
import re

import pytest
import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm
from tilelang.layout import make_mx_row_major_layout
from tilelang.utils import target as target_utils

from testing.python.sunmmio.common.codegen_validation import (
    codegen_sunmmio_suvm_mlir,
    validate_suvm_mlir_with_npuir_opt,
    write_sunmmio_codegen_logs,
)
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import collect_calls
from testing.python.sunmmio.common.suvm_trace import trace_suvm_control_flow

tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
os.environ.setdefault("SUNMMIO_TEST_LOG_IR", "0")


@tilelang.jit(target="sunmmio")
def payload_kernel(src_global=False, dst_global=False, offset=32, repeated=False, dynamic=False, kind=None, world_size: int = 1):
    shape = (128,)

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, T.placement.replicated(), T.float32),  # type: ignore
        B: T.MeshTensor(shape, T.placement.replicated(), T.float32),  # type: ignore
        rank_id: T.dist.RankId,  # type: ignore
        position: T.int32,
    ):
        with T.Kernel():
            src = T.alloc_shared(shape, T.float32)
            dst = T.alloc_shared(shape, T.float32)
            ready = T.dist.signal(kind=kind)
            if not src_global:
                T.fill(src, T.float32(1))
            begin = position * 32 if dynamic else offset
            T.dist.put(
                A[begin : begin + 32] if src_global else src[begin : begin + 32],
                B[32:64] if dst_global else dst[32:64],
                (rank_id + 1) % world_size,
                signal=ready,
            )
            if repeated:
                T.dist.put(
                    A[64:96] if src_global else src[64:96], B[64:96] if dst_global else dst[64:96], (rank_id + 2) % world_size, signal=ready
                )
            T.dist.submit()
            T.dist.wait_signal(ready)

    return main


@tilelang.jit(target="sunmmio")
def notification_kernel(kind=T.dist.SignalKind.SRAM_FLAGREG_INC, two_batches=False, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            ready = T.dist.signal(kind=kind)
            T.dist.put_signal(ready, (rank_id + 1) % world_size)
            T.dist.submit()
            T.dist.wait_signal(ready)
            if two_batches:
                T.dist.put_signal(ready, (rank_id + 1) % world_size, submit=True)
                T.dist.wait_signal(ready)

    return main


@tilelang.jit(target="sunmmio")
def source_scalar_write_kernel(submitted=True, index=16, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((256,), T.float32)
            dst = T.alloc_shared((16,), T.float32)
            ready = T.dist.signal()
            T.fill(src, T.float32(1))
            T.dist.put(src[0:16], dst, (rank_id + 1) % world_size, signal=ready)
            if submitted:
                T.dist.submit()
            src[index] = T.float32(2)
            if not submitted:
                T.dist.submit()
            T.dist.wait_signal(ready)

    return main


@tilelang.jit(target="sunmmio")
def group_state_kernel(memory=False, world_size: int = 1):
    kind = T.dist.SignalKind.SRAM_MEMORY if memory else T.dist.SignalKind.SRAM_FLAGREG_VALUE

    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            ready = T.dist.signals(2, kind=kind)
            T.dist.put_signal(ready[0], (rank_id + 1) % world_size)
            T.dist.put_signal(ready[1], (rank_id + 1) % world_size)
            T.dist.submit()
            T.dist.wait_signal(ready[0])
            T.dist.wait_signal(ready[1])

    return main


@tilelang.jit(target="sunmmio")
def mx_storage_kernel(row_source=False, world_size: int = 1):
    shape = (32, 128)
    # The rejection case compares MX row-major storage against default MXZZ.
    src_layout = make_mx_row_major_layout(shape, dtype=T.mxfp8) if row_source else None

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, T.placement.replicated(), T.mxfp8, layout=src_layout),
        B: T.MeshTensor(shape, T.placement.replicated(), T.mxfp8),
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            ready = T.dist.signal()
            T.dist.put(A, B, (rank_id + 1) % world_size, signal=ready, submit=True)
            T.dist.wait_signal(ready)

    return main


@tilelang.jit(target="sunmmio")
def dynamic_layout_kernel(world_size: int = 1):
    n = T.dynamic("n")

    @T.prim_func
    def main(
        A: T.MeshTensor((n,), T.placement.replicated(), T.float32),
        B: T.MeshTensor((n,), T.placement.replicated(), T.float32),
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            ready = T.dist.signal()
            T.dist.put(A[0:32], B[0:32], (rank_id + 1) % world_size, signal=ready, submit=True)
            T.dist.wait_signal(ready)

    return main


@tilelang.jit(target="sunmmio")
def alignment_snapshot_kernel(mode="difference", destination=False, world_size: int = 1):
    @T.prim_func
    def main(
        A: T.MeshTensor((128,), T.placement.replicated(), T.float32),
        B: T.MeshTensor((128,), T.placement.replicated(), T.float32),
        rank_id: T.dist.RankId,
        probe_index: T.int32,
    ):
        with T.Kernel():
            table = T.alloc_shared((16,), T.int32)
            ready = T.dist.signal(kind=T.dist.SignalKind.DRAM_FLAGREG_VALUE, expect="manual")
            index = probe_index % 16
            table[index] = 0
            before = table[index]
            table[index] = before + 1
            after = table[index]
            if mode == "difference":
                begin = after - before
            elif mode == "scaled_difference":
                begin = (after - before) * 32
            else:
                begin = table[index] * 32
                table[index] = 2
            T.dist.put(
                A[0:32] if destination else A[begin : begin + 32],
                B[begin : begin + 32] if destination else B[0:32],
                (rank_id + 1) % world_size,
                signal=ready,
            )
            T.dist.submit()
            T.dist.wait_signal(ready, expected_delta=0)

    return main


@tilelang.jit(target="sunmmio")
def integer_address_kernel(mode="narrow", extent=96, width=32, destination=False, keep_let=False, world_size: int = 1):
    dtype = T.int32 if mode in ("floor", "floor_mod") else {8: T.uint8, 16: T.uint16, 32: T.uint32, 64: T.uint64}[width]

    @T.prim_func
    def main(
        A: T.MeshTensor((768,), T.placement.replicated(), T.float32),
        B: T.MeshTensor((768,), T.placement.replicated(), T.float32),
        rank_id: T.dist.RankId,
        pos: dtype,
    ):
        with T.Kernel():
            ready = T.dist.signal(expect="manual")
            if mode == "narrow":
                off = T.Cast("int32", T.Cast("uint8", pos * T.Cast(dtype, extent)))
            elif mode == "mul":
                off = T.Cast("int64", pos * T.Cast(dtype, extent))
            elif mode == "add":
                off = T.Cast("int32", T.Cast("uint8", pos) * T.uint8(extent) + T.uint8(extent))
            elif mode == "sub":
                off = T.Cast("int32", T.Cast("uint8", pos) * T.uint8(extent) - T.uint8(extent))
            elif mode == "bounded":
                off = T.Cast("int32", T.Cast("uint8", (pos % T.Cast(dtype, 3)) * T.Cast(dtype, extent)))
            elif mode == "widen":
                off = T.Cast("int64", pos) * T.int64(extent)
            elif mode == "signed":
                off = T.Cast("int32", T.Cast("int8", pos * T.Cast(dtype, extent))) + 256
            elif mode == "bool":
                off = T.Cast("int32", T.Cast("bool", pos)) * extent
            elif mode == "floor":
                off = ((pos - 3) // 2 + 2) * extent
            else:
                off = ((pos - 3) % 2) * extent
            if keep_let:
                scratch = T.alloc_shared((16,), T.int32)
                scratch[0] = T.Cast("int32", off)
            T.dist.put(
                A[0:extent] if destination else A[off : off + extent],
                B[off : off + extent] if destination else B[0:extent],
                (rank_id + 1) % world_size,
                signal=ready,
            )
            T.dist.submit()
            T.dist.wait_signal(ready, expected_delta=1)

    return main


@tilelang.jit(target="sunmmio")
def integer_wait_index_kernel(mode="valid", world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, pos: T.uint32):
        with T.Kernel():
            ready = T.dist.signals(2, kind=T.dist.SignalKind.SRAM_FLAGREG_INC, expect="manual")
            if mode == "valid":
                index = T.Cast("int32", T.Cast("uint8", rank_id % 2))
            elif mode == "invalid":
                index = T.Cast("int32", T.Cast("uint8", rank_id - 1))
            else:
                index = T.Cast("int32", T.Cast("uint8", pos))
            T.dist.wait_signal(ready[index], expected_delta=0)

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


@pytest.mark.parametrize("src_global,dst_global", [(False, False), (False, True), (True, False), (True, True)])
def test_payload_scopes_and_destination_identity(tmp_path, src_global, dst_global):
    result, source = generate(payload_kernel, tmp_path, src_global=src_global, dst_global=dst_global)
    assert len(collect_calls(result.device_mod, "tl.dist_put_")) == 1
    assert source.count("suvm.dist.batch_put") == source.count("suvm.dist.batch_signal") == 1
    assert source.count("peer_writable") >= 2
    declaration = next(line for line in source.splitlines() if "= suvm.dist.signal" in line)
    assert f"space = {'global' if dst_global else 'rsram'}" in declaration
    if dst_global:
        signature = next(line for line in source.splitlines() if "func.func" in line)
        assert "peer_writable" in signature
    for before, after in [
        ("batch_begin", "batch_put"),
        ("batch_put", "batch_signal"),
        ("batch_signal", "flush"),
        ("flush", "batch_submit"),
        ("batch_submit", "signal_wait"),
    ]:
        assert source.index("suvm.dist." + before) < source.index("suvm.dist." + after)
    assert source.rindex("suvm.dist.flush") > source.index("suvm.dist.signal_wait")
    assert "suvm.dist.put " not in source
    assert "suvm.dist.my_pe" not in source and "suvm.dist.n_pes" not in source
    assert "signal_set" not in source


@pytest.mark.parametrize("kind", list(T.dist.SignalKind))
def test_signal_resources_and_widths(tmp_path, kind):
    if kind == T.dist.SignalKind.DRAM_MEMORY:
        with pytest.raises(tvm.error.InternalError, match="DRAM MEMORY.*ABI"):
            generate(notification_kernel, tmp_path, kind=kind)
        return
    _, source = generate(notification_kernel, tmp_path, kind=kind)
    assert source.count("= suvm.dist.signal :") == 1
    update = next(line for line in source.splitlines() if "suvm.dist.batch_signal" in line)
    wait = next(line for line in source.splitlines() if "suvm.dist.signal_wait" in line)
    assert "cmp = ge" in wait
    if kind.value.endswith("inc"):
        assert "] =" not in update
    else:
        assert "] =" in update
        assert (", i64" if kind.value.endswith("memory") else ", i8") in update
    assert ", i64" in wait if kind.value.endswith("memory") else ", i8" in wait


def test_multiple_peers_share_one_batch(tmp_path):
    _, source = generate(payload_kernel, tmp_path, repeated=True)
    assert source.count("suvm.dist.batch_begin") == source.count("suvm.dist.batch_submit") == 1
    assert source.count("suvm.dist.batch_put") == source.count("suvm.dist.batch_signal") == 2
    batches = re.findall(r"suvm.dist.batch_(?:put|signal) (%\w+)", source)
    assert len(set(batches)) == 1


@pytest.mark.parametrize(
    "kind,dst_global",
    [
        (T.dist.SignalKind.SRAM_FLAGREG_VALUE, False),
        (T.dist.SignalKind.DRAM_FLAGREG_VALUE, True),
        (T.dist.SignalKind.SRAM_MEMORY, False),
    ],
)
def test_payload_uses_existing_generation_once(tmp_path, kind, dst_global):
    result, source = generate(payload_kernel, tmp_path, kind=kind, dst_global=dst_global)
    put = collect_calls(result.device_mod, "tl.dist_put_")[0]
    assert put.args[5].dtype == ("uint64" if kind.value.endswith("memory") else "uint8")
    assert source.count("suvm.dist.batch_signal") == 1
    assert "] =" in next(line for line in source.splitlines() if "suvm.dist.batch_signal" in line)


def test_successive_submissions_have_distinct_handles(tmp_path):
    _, source = generate(notification_kernel, tmp_path, two_batches=True)
    handles = re.findall(r"suvm.dist.batch_submit (%\w+)", source)
    assert len(handles) == len(set(handles)) == 2


def test_aligned_dynamic_region_uses_exact_partition(tmp_path):
    _, source = generate(payload_kernel, tmp_path, dynamic=True)
    assert "suvm.dist.batch_put" in source
    assert "arith.index_cast" in source


def test_nonzero_payload_offsets_are_preserved(tmp_path):
    _, source = generate(payload_kernel, tmp_path)
    operands = re.search(r"suvm.dist.batch_put %\w+, %\w+, (%\w+), (%\w+)", source)
    assert operands
    constants = dict(re.findall(r"(%\w+) = arith.constant (\d+) : (?:i\d+|index)", source))
    casts = dict(re.findall(r"(%\w+) = arith.index_cast (%\w+)", source))
    views = dict(re.findall(r"(%\w+) = suvm.get_partitioned_tile_view %\w+ indices = \[(%\w+)\]", source))
    for operand in operands.groups():
        index = views[operand]
        while index in casts:
            index = casts[index]
        # Both original regions are [32:64], partition 1 of an extent-32 view.
        assert constants[index] == "1"


def test_unaligned_region_is_rejected_even_with_optional_checks_off(tmp_path, monkeypatch):
    monkeypatch.setattr(target_utils, "ENABLE_SUNMMIO_REGION_VALIDATION", False)
    with pytest.raises(tvm.error.InternalError, match="dist region start.*aligned"):
        generate(payload_kernel, tmp_path, offset=1)


@pytest.mark.parametrize("validation_enabled", [False, True])
@pytest.mark.parametrize("destination", [False, True])
def test_alignment_does_not_equate_loads_across_a_write(tmp_path, monkeypatch, validation_enabled, destination):
    monkeypatch.setattr(target_utils, "ENABLE_SUNMMIO_REGION_VALIDATION", validation_enabled)
    with pytest.raises(tvm.error.InternalError, match="dist region start.*aligned"):
        generate(alignment_snapshot_kernel, tmp_path, destination=destination)


@pytest.mark.parametrize("mode", ["scaled_difference", "scaled_load"])
@pytest.mark.parametrize("destination", [False, True])
def test_alignment_preserves_arithmetic_around_read_snapshots(tmp_path, mode, destination):
    result, source = generate(alignment_snapshot_kernel, tmp_path, mode=mode, destination=destination)
    batches, _ = trace_suvm_control_flow(result, source, rank_id=0, probe_index=3)
    data = [item for batch in batches for item in batch if item[0] == "data"]
    assert len(data) == 1
    assert (data[0][3], data[0][5]) == ((0, 32) if destination else (32, 0))


def test_queued_source_rejects_scalar_write_to_same_physical_tile(tmp_path):
    kernel = source_scalar_write_kernel.get_tir(submitted=False, world_size=4)
    write_sunmmio_codegen_logs(case_name=tmp_path.name, kernel=kernel)
    with pytest.raises(tvm.error.InternalError, match="before.*submit|unsubmitted"):
        lower_to_device_tir(kernel)


def test_inflight_source_flushes_before_scalar_tile_write(tmp_path):
    _, source = generate(source_scalar_write_kernel, tmp_path)
    submit = source.index("suvm.dist.batch_submit")
    store = source.index("suvm.tile.store", submit)
    assert "suvm.dist.flush" in source[submit:store]


@pytest.mark.parametrize("submitted", [False, True])
def test_scalar_write_to_disjoint_tile_preserves_overlap(tmp_path, submitted):
    _, source = generate(source_scalar_write_kernel, tmp_path, index=128, submitted=submitted)
    if submitted:
        submit = source.index("suvm.dist.batch_submit")
        store = source.index("suvm.tile.store", submit)
        assert "suvm.dist.flush" not in source[submit:store]


@pytest.mark.parametrize("memory", [False, True])
def test_group_arrays_use_existing_storage_and_keep_previous_tile_updates(tmp_path, memory):
    _, source = generate(group_state_kernel, tmp_path, memory=memory)
    declarations = [line for line in source.splitlines() if "= suvm.dist.signal :" in line]
    assert len(declarations) == 1 and "count = 2" in declarations[0]
    assert "suvm.tile.pick" in source and "suvm.tile.set" in source
    updates = re.findall(r"^    (%\w+) = suvm.tile.set %\w+, (%\w+)\[", source, re.MULTILINE)
    assert len(updates) >= 4
    # Tile memory operations are ordered by SUVM. Each update must reload its
    # base tile rather than reuse a tile from before a previous write.
    old_tiles = [old for _, old in updates]
    assert len(old_tiles) == len(set(old_tiles))
    for updated, old in updates:
        assert f"{old} = suvm.tile.load" in source
        assert f"suvm.tile.store {updated}," in source
    assert "i32 to i64" in source if memory else "i32 to i8" in source


def test_mx_whole_storage_payload_validates(tmp_path):
    _, source = generate(mx_storage_kernel, tmp_path)
    assert "!suvm.mxfp8" in source and "suvm.dist.batch_put" in source


def test_a4e_verifier_rejects_different_physical_storage_sizes(tmp_path):
    with pytest.raises(tvm.error.InternalError, match="A4E constraints"):
        generate(mx_storage_kernel, tmp_path, row_source=True)


def test_dynamic_destination_layout_keeps_peer_writable(tmp_path):
    _, source = generate(dynamic_layout_kernel, tmp_path)
    bindings = [line for line in source.splitlines() if "suvm.bind_layout" in line]
    assert bindings and any("peer_writable" in line for line in bindings)


@pytest.mark.parametrize("mode", ["narrow", "mul", "add", "sub"])
@pytest.mark.parametrize("destination", [False, True])
@pytest.mark.parametrize("validation_enabled", [False, True])
@pytest.mark.parametrize("keep_let", [False, True])
def test_direct_rejects_lost_alignment_after_integer_wrap(tmp_path, monkeypatch, mode, destination, validation_enabled, keep_let):
    monkeypatch.setattr(target_utils, "ENABLE_SUNMMIO_REGION_VALIDATION", validation_enabled)
    with pytest.raises(tvm.error.InternalError, match="region start.*aligned"):
        generate(integer_address_kernel, tmp_path, mode=mode, destination=destination, keep_let=keep_let)


@pytest.mark.parametrize(
    "mode,extent,positions",
    [
        ("narrow", 32, [2, 3, 134217729, 2**32 - 1]),
        ("bounded", 96, [0, 1, 2, 3, 2**32 - 1]),
        ("widen", 96, [0, 2, 3, 7]),
        ("signed", 32, [0, 4, 7, 255]),
        ("bool", 32, [0, 1, 2, 255]),
        ("floor", 32, [-1, 0, 2, 3]),
        ("floor_mod", 32, [-4, -3, 0, 2]),
    ],
)
@pytest.mark.parametrize("destination", [False, True])
@pytest.mark.parametrize("keep_let", [False, True])
def test_direct_preserves_finite_width_descriptor_values(tmp_path, mode, extent, positions, destination, keep_let):
    result, source = generate(integer_address_kernel, tmp_path, mode=mode, extent=extent, destination=destination, keep_let=keep_let)
    for pos in positions:
        reference = {
            "narrow": (pos * extent) & 255,
            "bounded": (pos % 3) * extent,
            "widen": pos * extent,
            "signed": (((pos * extent) & 255) ^ 128) - 128 + 256,
            "bool": int(pos != 0) * extent,
            "floor": ((pos - 3) // 2 + 2) * extent,
            "floor_mod": ((pos - 3) % 2) * extent,
        }[mode]
        batches, _ = trace_suvm_control_flow(result, source, rank_id=0, pos=pos)
        data = [item for batch in batches for item in batch if item[0] == "data"]
        assert len(data) == 1
        assert (data[0][3], data[0][5]) == ((0, reference) if destination else (reference, 0))


@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_direct_byte_narrowing_keeps_32_alignment_at_width_boundaries(tmp_path, width):
    result, source = generate(integer_address_kernel, tmp_path, extent=32, width=width)
    positions = range(256) if width == 8 else [0, 1, 7, 8, (1 << (width - 1)) - 1, 1 << (width - 1), (1 << width) - 1]
    for pos in positions:
        batches, _ = trace_suvm_control_flow(result, source, rank_id=0, pos=pos)
        data = [item for batch in batches for item in batch if item[0] == "data"]
        assert len(data) == 1 and data[0][3] == (pos * 32) & 255


def test_integer_wait_index_keeps_known_rank_bounds(tmp_path):
    result, source = generate(integer_wait_index_kernel, tmp_path)
    for rank in range(4):
        _, events = trace_suvm_control_flow(result, source, rank_id=rank, pos=0)
        assert [e[1] for e in events if e[0] == "wait"] == [0]


def test_integer_wait_index_rejects_concrete_wrapped_rank():
    with pytest.raises(tvm.error.InternalError, match="out of range"):
        lower_to_device_tir(integer_wait_index_kernel.get_tir(mode="invalid", world_size=4))


def test_integer_wait_index_keeps_runtime_guard_when_unknown(tmp_path):
    result, source = generate(integer_wait_index_kernel, tmp_path, mode="runtime")
    _, events = trace_suvm_control_flow(result, source, rank_id=0, pos=256)
    assert [e[1] for e in events if e[0] == "wait"] == [0]
    with pytest.raises(AssertionError):
        trace_suvm_control_flow(result, source, rank_id=0, pos=255)


if __name__ == "__main__":
    tilelang.testing.main()
