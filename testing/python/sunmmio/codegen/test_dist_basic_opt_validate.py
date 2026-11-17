"""Inter-rank resources, sequential batches, regions, and source completion."""

import re

import pytest
import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.layout import make_mx_row_major_layout
from tilelang.utils import target as target_utils

from testing.python.sunmmio.common.codegen_validation import (
    codegen_sunmmio_suvm_mlir,
    validate_suvm_mlir_with_npuir_opt,
)
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import collect_calls


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


def generate(factory, tmp_path, **kwargs):
    result = lower_to_device_tir(factory.get_tir(world_size=4, **kwargs))
    source = codegen_sunmmio_suvm_mlir(result.device_mod, log_ir=False)
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
        result = lower_to_device_tir(notification_kernel.get_tir(kind=kind, world_size=4))
        with pytest.raises(tvm.error.InternalError, match="DRAM MEMORY.*ABI"):
            codegen_sunmmio_suvm_mlir(result.device_mod, log_ir=False)
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
    result = lower_to_device_tir(payload_kernel.get_tir(offset=1, world_size=4))
    with pytest.raises(tvm.error.InternalError, match="dist region start.*aligned"):
        codegen_sunmmio_suvm_mlir(result.device_mod, log_ir=False)


def test_queued_source_rejects_scalar_write_to_same_physical_tile():
    with pytest.raises(tvm.error.InternalError, match="before.*submit|unsubmitted"):
        lower_to_device_tir(source_scalar_write_kernel.get_tir(submitted=False, world_size=4))


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
    result = lower_to_device_tir(mx_storage_kernel.get_tir(row_source=True, world_size=4))
    with pytest.raises(tvm.error.InternalError, match="A4E constraints"):
        codegen_sunmmio_suvm_mlir(result.device_mod, log_ir=False)


def test_dynamic_destination_layout_keeps_peer_writable(tmp_path):
    _, source = generate(dynamic_layout_kernel, tmp_path)
    bindings = [line for line in source.splitlines() if "suvm.bind_layout" in line]
    assert bindings and any("peer_writable" in line for line in bindings)
