"""Completion scanning and scalar signal observation through real SUVM codegen."""

import json

import pytest
import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.layout import make_aligned_row_major
from tilelang.utils.target import determine_target

from testing.python.sunmmio.common.codegen_validation import codegen_sunmmio_suvm_mlir, validate_suvm_mlir_with_npuir_opt
from testing.python.sunmmio.common.suvm_trace import trace_suvm_control_flow
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import collect_calls, single_prim_func
from tilelang.language.utils import region


@tilelang.jit(target="sunmmio")
def completion_kernel(count=3, base=0, delta=1, mode="progress", kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE, world_size: int = 1):
    offset = 3
    padded = ((count + offset + 15) // 16) * 16

    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            if base > 0:
                reserved = T.dist.signals(base, kind=kind, expect="manual")
                T.dist.wait_signal(reserved[0], expected_delta=0)
            signals = T.dist.signals(count, kind=kind, expect="manual")
            deltas = T.alloc_shared((padded,), T.uint32)
            T.fill(deltas, T.uint32(delta))
            if mode == "nested":
                for _round in T.serial(2):
                    done = T.dist.completion(signals, expected_deltas=deltas[offset : offset + count])
                    while T.dist.has_pending(done):
                        member = T.dist.wait_any(done)
                        T.evaluate(member)
            else:
                done = T.dist.completion(signals, expected_deltas=deltas[offset : offset + count])
                if mode == "wait_all":
                    T.dist.wait_all(done)
                elif mode == "evaluate":
                    T.dist.wait_any(done)
                else:
                    while T.dist.has_pending(done):
                        member = T.dist.wait_any(done)
                        T.evaluate(member)

    return main


def leaf_module(kind, count, base=0, snapshot_origin=0, pending_origin=0, selected=0, size=None):
    """Stable leaf boundary for offsets not produced by public Completion allocation."""
    # Final device TIR bypasses scope/layout inference, so describe storage here.
    dtype = "uint32" if kind.endswith("memory") else "int32"
    size = size or ((max(snapshot_origin, pending_origin) + count + 15) // 16) * 16
    snapshot = tvm.tir.decl_buffer((size,), dtype, name="snapshot", scope="shared.rsram")
    pending = tvm.tir.decl_buffer((size,), "int32", name="pending", scope="shared.rsram")
    call = tvm.tir.Call(
        "int32",
        tvm.ir.Op.get("tl.dist_wait_any_"),
        [
            tvm.tir.StringImm(kind),
            base,
            count,
            region(snapshot[snapshot_origin], "r", count),
            region(pending[pending_origin], "r", count),
            count,
        ],
    )
    member = tvm.tir.Var("selected_member", "int32")
    check = tvm.tir.AssertStmt(member == selected, tvm.tir.StringImm("logical member result"), tvm.tir.Evaluate(0))
    body = tvm.tir.LetStmt(member, call, check)
    body = tvm.tir.DeclBuffer(snapshot, tvm.tir.DeclBuffer(pending, body))
    func = tvm.tir.PrimFunc([snapshot.data, pending.data], body)
    layouts = {
        snapshot: make_aligned_row_major((size,), dtype, align_bytes=64),
        pending: make_aligned_row_major((size,), "int32", align_bytes=64),
    }
    func = func.with_attr("target", determine_target("sunmmio", return_object=True))
    func = func.with_attr("global_symbol", "leaf_probe").with_attr("tir.is_global_func", True)
    func = func.with_attr("layout_map", layouts).with_attr("tl.dist.signal_counts", {kind: base + count})
    return tvm.IRModule({"leaf_probe": func}), size


def producer_before_wait_any(*, placement="before", read="snapshot", evaluate=False, disjoint=False):
    size = 256 if disjoint else (128 if placement == "backedge" else 64)
    mod, _ = leaf_module("sram_flagreg_value", 1, snapshot_origin=16, pending_origin=16, size=size)
    func = single_prim_func(mod)
    snapshot_decl = func.body
    pending_decl = snapshot_decl.body
    snapshot, pending = snapshot_decl.buffer, pending_decl.buffer
    producer = tvm.tir.decl_buffer((16,), "int32", name="producer", scope="global")
    destination = snapshot if read == "snapshot" else pending
    i = tvm.tir.Var("phase", "int32")
    write_origin = 192 if disjoint else (48 + i * 16 if placement == "backedge" else 48)
    copy = tvm.tir.Evaluate(
        tvm.tir.Call(
            "handle",
            tvm.ir.Op.get("tl.dma_copy"),
            [
                region(producer[0], "r", 16),
                region(destination[write_origin], "w", 16),
                0,
                tvm.tir.Call("handle", tvm.ir.Op.get("tl.odma_unit"), [tvm.tir.StringImm("odma0")]),
            ],
        )
    )
    original = pending_decl.body
    if evaluate:
        original = tvm.tir.Evaluate(original.value)
    if placement == "backedge":
        original = tvm.tir.For(i, 0, 2, tvm.tir.ForKind.SERIAL, tvm.tir.SeqStmt([original, copy]))
    else:
        original = tvm.tir.SeqStmt([copy, original])
    body = tvm.tir.DeclBuffer(producer, tvm.tir.DeclBuffer(snapshot, tvm.tir.DeclBuffer(pending, original)))
    func = tvm.tir.PrimFunc([*func.params, producer.data], body, attrs=func.attrs)
    func = func.with_attr("global_layout_map", {producer: make_aligned_row_major((16,), "int32", align_bytes=1024)})
    return tvm.IRModule({"leaf_probe": func})


def generate(factory, tmp_path, **kwargs):
    result = lower_to_device_tir(factory.get_tir(world_size=4, **kwargs))
    source = codegen_sunmmio_suvm_mlir(result.device_mod, log_ir=False)
    validate_suvm_mlir_with_npuir_opt(source, tmp_path, opt_args=("--verify-each", "--suvm-device-validate"))
    return result, source


def trace_leaf(
    tmp_path,
    *,
    kind="sram_flagreg_value",
    expected,
    pending,
    observations,
    selected,
    base=0,
    snapshot_origin=0,
    pending_origin=0,
    size=None,
):
    mod, physical = leaf_module(kind, len(expected), base, snapshot_origin, pending_origin, selected, size)
    source = codegen_sunmmio_suvm_mlir(mod, log_ir=False)
    validate_suvm_mlir_with_npuir_opt(source, tmp_path, opt_args=("--verify-each", "--suvm-device-validate"))
    snapshots = [0] * physical
    snapshots[snapshot_origin : snapshot_origin + len(expected)] = expected
    flags = [1] * physical
    flags[pending_origin : pending_origin + len(pending)] = pending
    from types import SimpleNamespace

    result = SimpleNamespace(device_mod=mod)
    _, events = trace_suvm_control_flow(
        result,
        source,
        signal_values=observations,
        snapshot={"name": "snapshot", "data": snapshots},
        pending={"name": "pending", "data": flags},
    )
    assert "suvm.tile.store" not in source
    return source, [event for event in events if event[0] == "fetch"]


@pytest.mark.parametrize("kind", list(T.dist.SignalKind))
def test_completion_uses_each_supported_signal_interface(tmp_path, kind):
    if kind == T.dist.SignalKind.DRAM_MEMORY:
        result = lower_to_device_tir(completion_kernel.get_tir(kind=kind, world_size=4))
        with pytest.raises(tvm.error.InternalError, match="DRAM MEMORY.*ABI"):
            codegen_sunmmio_suvm_mlir(result.device_mod, log_ir=False)
        return
    result, source = generate(completion_kernel, tmp_path, kind=kind)
    _, events = trace_suvm_control_flow(result, source, signal_values={0: 1, 1: 1, 2: 1}, rank_id=0)
    fetches = [event for event in events if event[0] == "fetch"]
    assert [event[3] for event in fetches] == [0, 1, 2]
    assert "suvm.dist.signal_fetch" in source and "arith.cmpi uge" in source
    assert "suvm.dist.signal_set" not in source
    assert "memref." not in source


@pytest.mark.parametrize(
    "count,kind",
    [(1, T.dist.SignalKind.SRAM_FLAGREG_VALUE), (17, T.dist.SignalKind.SRAM_FLAGREG_VALUE), (40, T.dist.SignalKind.SRAM_MEMORY)],
)
def test_wait_all_completion_drains_logical_members_only(tmp_path, count, kind):
    result, source = generate(completion_kernel, tmp_path, count=count, kind=kind, mode="wait_all")
    _, events = trace_suvm_control_flow(result, source, signal_values=dict.fromkeys(range(count), 1), rank_id=0)
    assert [event[3] for event in events if event[0] == "fetch"] == list(range(count))


@pytest.mark.parametrize("mode", ["progress", "nested", "evaluate"])
def test_wait_any_results_flow_through_user_control_flow(tmp_path, mode):
    count = 1 if mode == "evaluate" else 3
    result, source = generate(completion_kernel, tmp_path, count=count, base=2, mode=mode)
    leaf = collect_calls(result.device_mod, "tl.dist_wait_any_")[0]
    assert int(leaf.args[1]) == 2
    _, events = trace_suvm_control_flow(result, source, signal_values=dict.fromkeys(range(2, 2 + count), 2), rank_id=0)
    fetches = [event[3] for event in events if event[0] == "fetch"]
    assert fetches == list(range(2, 2 + count)) * (2 if mode == "nested" else 1)


def test_partial_pending_and_nonzero_regions_return_logical_index(tmp_path):
    _, fetches = trace_leaf(
        tmp_path,
        expected=[1, 2, 3],
        pending=[0, 1, 1],
        observations={3: 1, 4: 3},
        selected=2,
        base=2,
        snapshot_origin=13,
        pending_origin=17,
        size=48,
    )
    assert [(event[3], event[4]) for event in fetches] == [(3, 1), (4, 3)]


def test_polling_restarts_scan_and_reads_fresh_signal_values(tmp_path):
    _, fetches = trace_leaf(tmp_path, expected=[2, 4], pending=[1, 1], observations={0: [0, 2], 1: [0]}, selected=0)
    assert [event[3] for event in fetches] == [0, 1, 0]


def test_first_ready_member_stops_remaining_fetches(tmp_path):
    _, fetches = trace_leaf(tmp_path, expected=[1, 1], pending=[1, 1], observations={0: 1}, selected=0)
    assert len(fetches) == 1


def test_flag_observation_is_zero_extended_before_unsigned_compare(tmp_path):
    source, fetches = trace_leaf(tmp_path, expected=[200], pending=[1], observations={0: [128, 200]}, selected=0)
    assert [event[4] for event in fetches] == [128, 200]
    assert "i8 to i32" in source and "arith.extui" in source


@pytest.mark.parametrize("expected,observed", [(2**31 + 7, 2**32 + 7), (2**32 - 1, 2**32 + 2), (2**32 - 1, 2**63 + 2)])
def test_memory_snapshot_zero_extension_and_full_observed_width(tmp_path, expected, observed):
    source, fetches = trace_leaf(
        tmp_path, kind="sram_memory", expected=[expected], pending=[1], observations={0: [2**31 - 1, observed]}, selected=0
    )
    assert [event[4] for event in fetches] == [2**31 - 1, observed]
    assert "i32 to i64" in source and "arith.extui" in source


@pytest.mark.parametrize("count,kind", [(1, "sram_flagreg_value"), (17, "sram_flagreg_value"), (40, "sram_memory")])
def test_polling_never_visits_physical_padding(tmp_path, count, kind):
    _, fetches = trace_leaf(
        tmp_path, kind=kind, expected=[1] * count, pending=[0] * (count - 1) + [1], observations={count - 1: [0, 1]}, selected=count - 1
    )
    assert [event[3] for event in fetches] == [count - 1, count - 1]


@pytest.mark.parametrize("read", ["snapshot", "pending"])
@pytest.mark.parametrize("evaluate", [False, True])
def test_wait_any_reads_wait_for_overlapping_physical_tile_producer(tmp_path, read, evaluate):
    # Logical read [16:17] and write [48:64] differ; the scalar loader reads
    # their shared 64-element tile, so an ODMA dependency is still required.
    mod = tilelang.transform.InjectSunmmioSync()(producer_before_wait_any(read=read, evaluate=evaluate))
    syncs = collect_calls(mod, "tl.sunmmio_sync")
    assert any(int(call.args[0]) & 1 for call in syncs)
    source = codegen_sunmmio_suvm_mlir(mod, log_ir=False)
    validate_suvm_mlir_with_npuir_opt(source, tmp_path, opt_args=("--verify-each", "--suvm-device-validate"))
    assert source.index("suvm.sync odma0") < source.index("suvm.dist.signal_fetch")


@pytest.mark.parametrize("read", ["snapshot", "pending"])
def test_loop_backedge_completes_producer_before_next_scan(tmp_path, read):
    mod = tilelang.transform.InjectSunmmioSync()(producer_before_wait_any(placement="backedge", read=read))
    func = single_prim_func(mod)
    calls = collect_calls(func)
    write = next(i for i, call in enumerate(calls) if call.op.name == "tl.dma_copy")
    assert any(call.op.name == "tl.sunmmio_sync" and int(call.args[0]) & 1 for call in calls[write + 1 :])
    source = codegen_sunmmio_suvm_mlir(mod, log_ir=False)
    validate_suvm_mlir_with_npuir_opt(source, tmp_path, opt_args=("--verify-each", "--suvm-device-validate"))


@pytest.mark.parametrize("delta", [2**31 + 7, 2**32 - 1])
def test_real_completion_preserves_memory_software_high_bits(tmp_path, delta):
    result, source = generate(completion_kernel, tmp_path, count=1, delta=delta, kind=T.dist.SignalKind.SRAM_MEMORY)
    observed = 2**32 + delta
    _, events = trace_suvm_control_flow(result, source, signal_values={0: observed}, rank_id=0)
    assert [event[4] for event in events if event[0] == "fetch"] == [observed]


def test_real_completion_keeps_flag_boundary_unsigned(tmp_path):
    result, source = generate(completion_kernel, tmp_path, count=1, delta=255)
    _, events = trace_suvm_control_flow(result, source, signal_values={0: [128, 255]}, rank_id=0)
    assert [event[4] for event in events if event[0] == "fetch"] == [128, 255]


@pytest.mark.parametrize("read", ["snapshot", "pending"])
def test_disjoint_producer_is_not_drained_before_scan(tmp_path, read):
    mod = tilelang.transform.InjectSunmmioSync()(producer_before_wait_any(read=read, disjoint=True))
    assert not any(int(call.args[0]) & 1 for call in collect_calls(mod, "tl.sunmmio_sync"))
    source = codegen_sunmmio_suvm_mlir(mod, log_ir=False)
    validate_suvm_mlir_with_npuir_opt(source, tmp_path, opt_args=("--verify-each", "--suvm-device-validate"))
    assert source.index("suvm.sync odma0") > source.index("suvm.dist.signal_fetch")


def test_wait_any_tracks_all_original_input_nodes(tmp_path, monkeypatch):
    report_path = tmp_path / "wait_any_coverage.json"
    monkeypatch.setenv("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT", "1")
    monkeypatch.setenv("TL_SUNMMIO_CODEGEN_COVERAGE_PATH", str(report_path))
    generate(completion_kernel, tmp_path)
    report = json.loads(report_path.read_text())
    assert report["main"]["missing_node_types"] == []
    assert report["main"]["missing_call_ops"] == []
