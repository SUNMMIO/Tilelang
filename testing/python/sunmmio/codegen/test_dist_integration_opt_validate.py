"""Production collective/routing and computation integration for dist codegen."""

import os

import pytest
import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm
from tilelang.layout import make_aligned_row_major
from tilelang.utils import target as target_utils

from testing.python.sunmmio.common.codegen_validation import (
    codegen_sunmmio_suvm_mlir,
    validate_suvm_mlir_with_npuir_opt,
    write_sunmmio_codegen_logs,
)
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import collect_calls, single_prim_func

tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
os.environ.setdefault("SUNMMIO_TEST_LOG_IR", "0")


@tilelang.jit(target="sunmmio")
def collective_kernel(operation="gather", row=False, grouped=True, src_dram=False, dst_dram=False, capacity=1, world_size: int = 1):
    endpoints = world_size * 4 if row else world_size
    chunk = (32, 32)
    send_shape = (
        chunk if operation in ("gather", "reduce") else ((world_size, 4, capacity, 32, 32) if row else (world_size, capacity, 32, 32))
    )
    if operation == "alltoall":
        send_shape = (world_size, 4, 32, 32) if row else (world_size, 32, 32)
    recv_shape = (world_size, 32, 32) if operation == "gather" else send_shape
    kind = T.dist.SignalKind.DRAM_FLAGREG_VALUE if dst_dram else T.dist.SignalKind.SRAM_FLAGREG_VALUE

    @T.prim_func
    def main(
        A: T.MeshTensor(send_shape, T.placement.replicated(), T.bfloat16),
        B: T.MeshTensor(recv_shape, T.placement.replicated(), T.bfloat16),
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            src = T.alloc_shared(send_shape, T.bfloat16)
            dst = T.alloc_shared(recv_shape, T.bfloat16)
            if not src_dram:
                T.fill(src, T.bfloat16(1))
            if operation == "gather":
                if grouped:
                    ready = T.dist.signals(world_size, kind=kind)
                    done = T.dist.all_gather(A if src_dram else src, B if dst_dram else dst, signals=ready, submit=True)
                    T.dist.wait_all(done)
                else:
                    ready = T.dist.signal()
                    T.dist.all_gather(A if src_dram else src, B if dst_dram else dst, signal=ready, submit=True)
                    T.dist.wait_signal(ready)
            if operation == "alltoall":
                ready = T.dist.signals(endpoints, kind=kind)
                done = T.dist.all_to_all(
                    A if src_dram else src,
                    B if dst_dram else dst,
                    signals=ready,
                    domain=T.dist.CollectiveDomain.ROW if row else T.dist.CollectiveDomain.RANK,
                    submit=True,
                )
                T.dist.wait_all(done)
            if operation == "alltoallv":
                count_shape = (world_size, 4) if row else (world_size,)
                send_counts = T.alloc_shared(count_shape, T.int32)
                recv_counts = T.alloc_shared(count_shape, T.int32)
                T.fill(send_counts, 1)
                T.fill(recv_counts, 1)
                ready = T.dist.signals(endpoints, kind=kind)
                done = T.dist.all_to_allv(
                    A if src_dram else src,
                    B if dst_dram else dst,
                    send_counts=send_counts,
                    recv_counts=recv_counts,
                    signals=ready,
                    domain=T.dist.CollectiveDomain.ROW if row else T.dist.CollectiveDomain.RANK,
                    submit=True,
                )
                T.dist.wait_all(done)
            if operation == "reduce":
                ready = T.dist.signal()
                T.dist.all_reduce(src, dst, "sum", signal=ready)
            if operation == "barrier":
                ready = T.dist.signal()
                T.dist.barrier(signal=ready)

    return main


@tilelang.jit(target="sunmmio")
def cross_row_kernel(src_dram=False, dst_dram=False, local=False, world_size: int = 1):
    shape = (32, 32)

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, T.placement.replicated(), T.bfloat16),
        B: T.MeshTensor(shape, T.placement.replicated(), T.bfloat16),
        rank_id: T.dist.RankId,
    ):
        with T.Kernel() as cid:
            src = T.alloc_shared(shape, T.bfloat16)
            dst = T.alloc_shared(shape, T.bfloat16)
            T.fill(src, T.bfloat16(1))
            ready = T.dist.signal()
            row = cid // T.mesh_ncols()
            T.dist.put(
                A if src_dram else src,
                B if dst_dram else dst,
                rank_id if local else (rank_id + 1) % world_size,
                dst_row=(row + 1) % T.mesh_nrows(),
                signal=ready,
                submit=True,
            )
            T.dist.wait_signal(ready)

    return main


@tilelang.jit(target="sunmmio")
def lifecycle_kernel(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((128,), T.float32)
            dst = T.alloc_shared((64,), T.float32)
            ready = T.dist.signal()
            T.fill(src, T.float32(1))
            T.dist.put(src[0:32], dst[0:32], (rank_id + 1) % world_size, signal=ready, submit=True)
            src[0] = T.float32(2)
            for i in T.Tiles(src):
                src[i] = src[i] + T.float32(1)
            T.dist.put(src[32:64], dst[32:64], (rank_id + 1) % world_size, signal=ready, submit=True)
            T.dist.wait_signal(ready)

    return main


@tilelang.jit(target="sunmmio")
def isolation_kernel(mode="plain", world_size: int = 1):
    shape = (64,)

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, T.placement.replicated(), T.float32),
        B: T.MeshTensor(shape, T.placement.replicated(), T.float32),
        rank_id: T.dist.RankId,
        early: T.int32,
    ):
        with T.Kernel():
            if mode == "plain":
                stage = T.alloc_shared(shape, T.float32)
                # Match the external 1D buffer's 1KiB padding so these copies
                # preserve layout; generic rank-1 layout conversion is limited.
                T.annotate_layout({stage: make_aligned_row_major(shape, T.float32, align_bytes=1024)})
                T.copy(A, stage)
                T.copy(stage, B)
            if mode == "send":
                ready = T.dist.signal()
                T.dist.put(A[0:32], B[0:32], (rank_id + 1) % world_size, signal=ready, submit=True)
                T.dist.wait_signal(ready)
            if mode == "capture":
                ready = T.dist.signal(kind=T.dist.SignalKind.DRAM_FLAGREG_VALUE, expect="manual")
                T.dist.put(A[0:32], B[0:32], (rank_id + 1) % world_size, signal=ready)
                if early > 0:
                    T.dist.submit()
                T.dist.put(A[32:64], B[32:64], (rank_id + 1) % world_size, signal=ready)
                T.dist.submit()
                T.dist.wait_signal(ready, expected_delta=0)
            if mode == "receive":
                ready = T.dist.signals(2, kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
                deltas = T.alloc_shared((16,), T.uint32)
                T.fill(deltas, T.uint32(1))
                done = T.dist.completion(ready, expected_deltas=deltas[0:2])
                T.dist.wait_all(done)

    return main


@tilelang.jit(target="sunmmio")
def mx_collective_kernel(dtype=T.mxfp8, capacity=1, loop=False, direct=False, world_size: int = 1):
    shape = (world_size * (3 if loop else 1), capacity, 32, 128)

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, T.placement.replicated(), dtype),
        B: T.MeshTensor(shape, T.placement.replicated(), dtype),
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            send = T.alloc_shared((16,), T.int32)
            recv = T.alloc_shared((16,), T.int32)
            T.fill(send, 1)
            T.fill(recv, 1)
            ready = T.dist.signals(world_size, kind=T.dist.SignalKind.DRAM_FLAGREG_VALUE)
            for epoch in T.serial(1 if loop else 0, 3 if loop else 1):
                begin = epoch * world_size
                if direct:
                    T.dist.put(A[begin, :, :, :], B[begin, :, :, :], (rank_id + 1) % world_size, signal=ready[0], submit=True)
                    T.dist.wait_signal(ready[0])
                else:
                    done = T.dist.all_to_allv(
                        A[begin : begin + world_size, :, :, :],
                        B[begin : begin + world_size, :, :, :],
                        send_counts=send[begin : begin + world_size],
                        recv_counts=recv[begin : begin + world_size],
                        signals=ready,
                        submit=True,
                    )
                    T.dist.wait_all(done)

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


def possible_remote_peers(func, rank, core=0):
    """Project statically known topology; data predicates conservatively fork."""
    rank_var = func.params[int(func.attrs["tl.dist.rank_id_param_index"])]
    analyzer = tvm.arith.Analyzer()
    peers = []
    contains_send = {}

    def has_send(stmt):
        if stmt not in contains_send:
            if isinstance(stmt, tvm.tir.SeqStmt):
                result = any(has_send(child) for child in stmt.seq)
            elif isinstance(stmt, tvm.tir.IfThenElse):
                result = has_send(stmt.then_case) or (stmt.else_case is not None and has_send(stmt.else_case))
            elif hasattr(stmt, "body"):
                result = has_send(stmt.body)
            else:
                result = any(call.op.name in ("tl.dist_put_", "tl.dist_signal_put_") for call in collect_calls(tvm.tir.PrimFunc([], stmt)))
            contains_send[stmt] = result
        return contains_send[stmt]

    def value(expr, env):
        simplified = analyzer.simplify(tvm.tir.stmt_functor.substitute(expr, env))
        return int(simplified) if isinstance(simplified, tvm.tir.IntImm) else None

    def visit(stmt, env):
        if not has_send(stmt):
            return
        if isinstance(stmt, tvm.tir.SeqStmt):
            for child in stmt.seq:
                visit(child, env)
        elif isinstance(stmt, tvm.tir.AttrStmt):
            updated = env.copy()
            if stmt.attr_key == "thread_extent" and stmt.node.thread_tag == "blockIdx.x":
                updated[stmt.node.var] = tvm.tir.const(core, str(stmt.node.var.dtype))
            visit(stmt.body, updated)
        elif isinstance(stmt, tvm.tir.LetStmt):
            updated = env.copy()
            constant = value(stmt.value, env)
            if constant is not None:
                updated[stmt.var] = tvm.tir.const(constant, str(stmt.var.dtype))
            visit(stmt.body, updated)
        elif isinstance(stmt, tvm.tir.IfThenElse):
            condition = value(stmt.condition, env)
            if condition is None or condition:
                visit(stmt.then_case, env)
            if stmt.else_case is not None and (condition is None or not condition):
                visit(stmt.else_case, env)
        elif isinstance(stmt, tvm.tir.For):
            start, extent = value(stmt.min, env), value(stmt.extent, env)
            assert start is not None and extent is not None
            step = value(stmt.step, env) if stmt.step is not None else 1
            for i in range(start, start + extent, step):
                updated = env.copy()
                updated[stmt.loop_var] = tvm.tir.const(i, str(stmt.loop_var.dtype))
                visit(stmt.body, updated)
        elif isinstance(stmt, tvm.tir.Evaluate) and isinstance(stmt.value, tvm.tir.Call):
            call = stmt.value
            if call.op.name in ("tl.dist_put_", "tl.dist_signal_put_"):
                peer = value(call.args[2 if call.op.name == "tl.dist_put_" else 0], env)
                assert peer is not None
                peers.append(peer)
        elif hasattr(stmt, "body"):
            visit(stmt.body, env)
        else:
            raise AssertionError(f"unsupported topology statement {type(stmt)}")

    visit(func.body, {rank_var: tvm.tir.const(rank, str(rank_var.dtype))})
    return peers


@pytest.mark.parametrize(
    "operation,row",
    [
        ("gather", False),
        ("alltoall", False),
        ("alltoall", True),
        ("alltoallv", False),
        ("alltoallv", True),
        ("reduce", False),
        ("barrier", False),
    ],
)
def test_real_collective_expansions_validate(tmp_path, operation, row):
    result, source = generate(collective_kernel, tmp_path, operation=operation, row=row)
    leaves = collect_calls(result.device_mod, "tl.dist_put_")
    updates = collect_calls(result.device_mod, "tl.dist_signal_put_")
    assert source.count("suvm.dist.batch_put") == len(leaves)
    assert source.count("suvm.dist.batch_signal") == len(leaves) + len(updates)
    assert "suvm.dist.batch_submit" in source and "suvm.dist.flush" in source
    assert "suvm.dist.signal_fetch" in source if operation in ("gather", "alltoall", "alltoallv") else "suvm.dist.signal_wait" in source
    if operation == "barrier":
        assert not leaves
        assert "suvm.dist.barrier" not in source
    if row:
        assert "suvm.mcast_async" in source and "suvm.barrier.arrive_and_wait" in source
        # A4E link writes require 1024B even though scalar tile accesses need
        # only 64B. Transport the dedicated predicate buffer, including padding.
        assert "!suvm.tile_view<256xi32>" in source


@pytest.mark.parametrize("src_dram,dst_dram", [(False, False), (False, True), (True, False), (True, True)])
def test_gather_all_payload_spaces_and_aggregate_wait(tmp_path, src_dram, dst_dram):
    _, source = generate(collective_kernel, tmp_path, operation="gather", grouped=False, src_dram=src_dram, dst_dram=dst_dram)
    assert "suvm.dist.signal_wait" in source and "suvm.dist.signal_fetch" not in source
    assert "suvm.copy_async" in source


@pytest.mark.parametrize("row", [False, True])
def test_alltoallv_transfers_complete_capacity(tmp_path, row):
    _, source = generate(collective_kernel, tmp_path, operation="alltoallv", row=row, capacity=2)
    assert "!suvm.tile_view<2x32x32xbf16>" in source


@pytest.mark.parametrize("src_dram,dst_dram", [(False, False), (True, False), (False, True), (True, True)])
def test_cross_row_staging_connects_local_and_remote_engines(tmp_path, src_dram, dst_dram):
    _, source = generate(cross_row_kernel, tmp_path, src_dram=src_dram, dst_dram=dst_dram)
    assert "suvm.mcast_async" in source and "suvm.dist.batch_put" in source
    assert "suvm.barrier.arrive_and_wait" in source


@pytest.mark.parametrize("dst_dram", [False, True])
def test_local_rank_cross_row_path_has_no_remote_notifications(tmp_path, dst_dram):
    result, source = generate(cross_row_kernel, tmp_path, local=True, dst_dram=dst_dram)
    assert not collect_calls(result.device_mod, "tl.dist_put_")
    assert "suvm.mcast_async" in source and "suvm.dist.batch_signal" not in source


@pytest.mark.parametrize(
    "operation,row",
    [("gather", False), ("alltoall", False), ("alltoall", True), ("alltoallv", False), ("alltoallv", True), ("barrier", False)],
)
def test_collectives_never_notify_local_rank(operation, row):
    # The matching cases above validate actual codegen. Project topology here
    # directly from production TIR to check all Rank/core paths independently.
    result = lower_to_device_tir(collective_kernel.get_tir(world_size=4, operation=operation, row=row))
    func = single_prim_func(result.device_mod)
    for rank in range(4):
        for core in [0, 5, 10, 15] if row else [0]:
            peers = possible_remote_peers(func, rank, core)
            assert set(peers) == set(range(4)) - {rank}
            assert all(peer != rank for peer in peers)


def test_source_reuse_tiles_and_natural_exit_keep_completion_order(tmp_path):
    _, source = generate(lifecycle_kernel, tmp_path)
    first_submit = source.index("suvm.dist.batch_submit")
    first_store = source.index("suvm.tile.store", first_submit)
    second_put = source.rindex("suvm.dist.batch_put")
    assert "suvm.dist.flush" in source[first_submit:first_store]
    assert "suvm.tile.addf" in source[first_store:second_put]
    assert source.rindex("suvm.dist.flush") < source.index("return")
    assert source.count("suvm.dist.batch_submit") == 2


def test_multiple_functions_do_not_share_resources_or_capture_state(tmp_path):
    modes = ["capture", "receive", "send", "plain"]
    functions = {}
    for mode in modes:
        kernel = isolation_kernel.get_tir(mode=mode, world_size=4)
        write_sunmmio_codegen_logs(case_name=f"{tmp_path.name}_{mode}", kernel=kernel)
        func = single_prim_func(lower_to_device_tir(kernel).device_mod)
        functions["isolation_" + mode] = func.with_attr("global_symbol", "isolation_" + mode)
    mod = tvm.IRModule(functions)
    write_sunmmio_codegen_logs(case_name=tmp_path.name, tir_mod=mod)
    source = codegen_sunmmio_suvm_mlir(mod, log_case_name=tmp_path.name)
    validate_suvm_mlir_with_npuir_opt(source, tmp_path, opt_args=("--verify-each", "--suvm-device-validate"))
    sections = source.split("func.func @")[1:]
    for section in sections:
        mode = section.split("(", 1)[0].removeprefix("isolation_")
        if mode == "plain":
            assert "suvm.dist." not in section and "peer_writable" not in section
        if mode == "receive":
            assert "suvm.dist.signal_fetch" in section and "count = 2" in section
            assert "peer_writable" not in section and "suvm.dist.batch_begin" not in section
        if mode == "send":
            assert "peer_writable" in section and "kind = increment, space = global, count = 1" in section
            assert section.count("suvm.dist.batch_begin") == 1
        if mode == "capture":
            assert "kind = assign, space = global, count = 1" in section and "scf.if" in section


@pytest.mark.parametrize("dtype", [T.mxfp8, T.mxfp4])
@pytest.mark.parametrize("checks", [False, True])
def test_mx_collective_and_validation_switch_preserve_physical_views(tmp_path, monkeypatch, dtype, checks):
    monkeypatch.setattr(target_utils, "ENABLE_SUNMMIO_REGION_VALIDATION", checks)
    _, source = generate(mx_collective_kernel, tmp_path, dtype=dtype)
    assert "!suvm.tile_view<32x128x!suvm.mxfp" in source
    assert "suvm.dist.batch_signal" in source


@pytest.mark.parametrize("dtype", [T.mxfp8, T.mxfp4])
def test_mx_capacity_preserves_existing_local_copy_slot_limit(dtype):
    with pytest.raises(tvm.error.InternalError, match="MX copy requires a single matrix slot"):
        lower_to_device_tir(mx_collective_kernel.get_tir(dtype=dtype, capacity=2, world_size=4))


@pytest.mark.parametrize("dtype", [T.mxfp8, T.mxfp4])
@pytest.mark.parametrize("checks", [False, True])
def test_mx_batched_remote_payload_preserves_storage_units(tmp_path, monkeypatch, dtype, checks):
    monkeypatch.setattr(target_utils, "ENABLE_SUNMMIO_REGION_VALIDATION", checks)
    _, source = generate(mx_collective_kernel, tmp_path, dtype=dtype, capacity=2, direct=True)
    assert "!suvm.tile_view<2x32x128x!suvm.mxfp" in source
    assert "suvm.dist.batch_put" in source


@pytest.mark.parametrize("dtype", [T.bfloat16, T.mxfp8, T.mxfp4])
@pytest.mark.parametrize("checks", [False, True])
def test_collective_loop_offsets_keep_payload_and_count_bases(tmp_path, monkeypatch, dtype, checks):
    monkeypatch.setattr(target_utils, "ENABLE_SUNMMIO_REGION_VALIDATION", checks)
    _, source = generate(mx_collective_kernel, tmp_path, dtype=dtype, loop=True)
    assert "scf.for" in source and "suvm.dist.signal_fetch" in source
    assert "suvm.dist.batch_put" in source and "suvm.get_partitioned_tile_view" in source


if __name__ == "__main__":
    tilelang.testing.main()
