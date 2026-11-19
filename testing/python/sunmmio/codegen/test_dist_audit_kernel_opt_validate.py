"""Audit regressions from public kernels through the production pipeline."""

import pytest
from testing.python.sunmmio.codegen.test_dist_integer_control_flow_opt_validate import wide_window_kernel
from tilelang.layout import make_aligned_row_major
from testing.python.sunmmio.common.suvm_trace import trace_suvm_control_flow
import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.common.codegen_validation import codegen_sunmmio_suvm_mlir, validate_suvm_mlir_with_npuir_opt

tilelang.env.disable_cache()


@tilelang.jit(target="sunmmio")
def guarded_partition_kernel(use="then", world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, n: T.uint64):
        with T.Kernel():
            src = T.alloc_shared((4, 32), T.int32)
            dst = T.alloc_shared((1, 32), T.int32)
            s = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            T.fill(src, 1)
            if T.truncdiv(n, T.uint64(4)) == 0 or use == "else":
                T.dist.put(src[n : n + 1, :], dst, (rank_id + 1) % world_size, signal=s, submit=True)
            if use == "after":
                T.dist.put(src[n : n + 1, :], dst, (rank_id + 1) % world_size, signal=s, submit=True)
            T.dist.wait_signal(s, expected_delta=0)

    return main


@tilelang.jit(target="sunmmio")
def snapshot_guard_kernel(stale=False, world_size: int = 1):
    kind = T.dist.SignalKind.DRAM_FLAGREG_INC if stale else T.dist.SignalKind.SRAM_MEMORY

    @T.prim_func
    def main(
        A: T.MeshTensor((128,), T.placement.replicated(), T.int32),
        B: T.MeshTensor((128,), T.placement.replicated(), T.int32),
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            table = T.alloc_shared((16,), T.int32)
            src = T.alloc_shared((4, 32), T.int32)
            dst = T.alloc_shared((1, 32), T.int32)
            s = T.dist.signals(1, kind=kind, expect="manual")
            T.fill(table, 0)
            T.fill(src, 1)
            before = T.Cast("uint64", table[0])
            if T.truncdiv(before, T.uint64(4)) == 0:
                table[0] = 1 if stale else 32
                if stale:
                    T.dist.put(A[table[0] : table[0] + 32], B[0:32], (rank_id + 1) % world_size, signal=s[0], submit=True)
                else:
                    T.dist.put(src[before : before + 1, :], dst, (rank_id + 1) % world_size, signal=s[0], submit=True)
            T.dist.wait_signal(s[0], expected_delta=1)

    return main


@tilelang.jit(target="sunmmio")
def rank_partition_kernel(nested=False, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, n: T.int32):
        with T.Kernel():
            src = T.alloc_shared((4, 32), T.int32)
            dst = T.alloc_shared((1, 32), T.int32)
            s = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            T.fill(src, 1)
            if not nested:
                if rank_id == 0:
                    T.dist.put(src[0:1, :], dst, 1, signal=s, submit=True)
            elif n > 0:  # noqa: SIM102 - exercise two nested guards on the same value
                if n < 4:
                    T.dist.put(src[n : n + 1, :], dst, (rank_id + 1) % world_size, signal=s, submit=True)

    return main


@tilelang.jit(target="sunmmio")
def scalar_write_kernel(position="before", constant=False, dynamic_source=False, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, n: T.int32):
        with T.Kernel():
            src = T.alloc_shared((8192,), T.int32)
            dst = T.alloc_shared((32,), T.int32)
            s = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            T.fill(src, 1)
            address = T.Cast("int32", T.Cast("uint8", n % 2 - 1)) * 32
            pos = 8160 if constant else address
            if position == "before":
                src[pos] = 2
            T.dist.put(src[address : address + 32] if dynamic_source else src[8160:8192], dst, (rank_id + 1) % world_size, signal=s)
            if position == "queued":
                src[pos] = 2
            T.dist.submit()
            if position == "submitted":
                src[pos] = 2
            T.dist.wait_signal(s, expected_delta=1)

    return main


@tilelang.jit(target="sunmmio")
def disjoint_fill_kernel(submitted=False, parallel=False, two_dim=False, world_size: int = 1):
    shape = (4, 32) if two_dim else (256,)
    dst_shape = (1, 32) if two_dim else (32,)

    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared(shape, T.int32)
            dst = T.alloc_shared(dst_shape, T.int32)
            s = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            T.fill(src, 1)
            T.dist.put(src[0:1, :] if two_dim else src[0:32], dst, (rank_id + 1) % world_size, signal=s)
            if submitted:
                T.dist.submit()
            if parallel:
                for i in T.Parallel(32):
                    if two_dim:
                        src[3, i] = 2
                    else:
                        src[i + 128] = 2
            else:
                T.fill(src[3:4, :] if two_dim else src[128:160], 2)
            if not submitted:
                T.dist.submit()
            T.dist.wait_signal(s, expected_delta=1)

    return main


@tilelang.jit(target="sunmmio")
def short_fill_kernel(submitted=True, cross=False, world_size: int = 1):
    start, width = (28, 4) if cross else (0, 8)

    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((256,), T.float32)
            dst = T.alloc_shared((width,), T.float32)
            T.fill(src, 1)
            s = T.dist.signal(expect="manual")
            T.dist.put(src[start : start + width], dst, (rank_id + 1) % world_size, signal=s)
            if submitted:
                T.dist.submit()
            if cross:
                for lane in T.Tiles([14], parallel=True):
                    src[14 + lane] = T.float32(2)
            else:
                T.fill(src[8:16], 2)
            if not submitted:
                T.dist.submit()
            T.dist.wait_signal(s, expected_delta=1)

    return main


@tilelang.jit(target="sunmmio")
def small_tile_kernel(submitted=True, world_size: int = 1):
    shape = (64, 32)
    layout = make_aligned_row_major(shape, T.float32, align_bytes=64)

    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared(shape, T.float32)
            other = T.alloc_shared(shape, T.float32)
            dst = T.alloc_shared((1, 16), T.float32)
            T.annotate_layout({src: layout, other: layout})
            s = T.dist.signal(expect="manual")
            T.fill(src, 1)
            T.fill(other, 2)
            T.dist.put(src[0:1, 16:32], dst, (rank_id + 1) % world_size, signal=s)
            if submitted:
                T.dist.submit()
            for i, j in T.Tiles([4, 4], parallel=True):
                src[i, j] = other[i, j] * 2.0
            if not submitted:
                T.dist.submit()
            T.dist.wait_signal(s, expected_delta=1)

    return main


@tilelang.jit(target="sunmmio")
def next_tile_kernel(submitted=False, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((640,), T.int32)
            dst = T.alloc_shared((32,), T.int32)
            s = T.dist.signal(expect="manual")
            T.fill(src, 1)
            for i in T.serial(4):
                T.dist.put(src[i * 128 : i * 128 + 32], dst, (rank_id + 1) % world_size, signal=s)
                if submitted:
                    T.dist.submit()
                src[(i + 1) * 128] = 2
                if not submitted:
                    T.dist.submit()
                T.dist.wait_signal(s, expected_delta=1)

    return main


@tilelang.jit(target="sunmmio")
def integer_range_kernel(step=None, width=8, signed=False, world_size: int = 1):
    dtype = getattr(T, f"{'int' if signed else 'uint'}{width}")

    @T.prim_func
    def main(rank_id: T.dist.RankId, start: dtype, stop: dtype):
        with T.Kernel():
            s = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            for _ in T.serial(start, stop, step):
                T.dist.put_signal(s, (rank_id + 1) % world_size, submit=True)

    return main


@tilelang.jit(target="sunmmio")
def signed_step_kernel(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, start: T.int32, stop: T.int32):
        with T.Kernel():
            s = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            for _ in T.serial(start, stop, T.int32(2147483647)):
                T.dist.put_signal(s, (rank_id + 1) % world_size, submit=True)

    return main


@tilelang.jit(target="sunmmio")
def unroll_window_kernel(mutable=False, bounded=False, world_size: int = 1):
    dtype = T.int32 if mutable else T.uint8

    @T.prim_func
    def main(rank_id: T.dist.RankId, n: dtype):
        with T.Kernel():
            s = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            if mutable:
                base = T.alloc_var(T.int32, init=n)
                for i in T.unroll(base, base + 5, 1, explicit=True):
                    T.dist.wait_signal(s, expected_delta=T.Cast("uint32", i - n))
                    base = 0
            else:
                bounded_start = n % T.uint8(16)
                for i in T.unroll(bounded_start if bounded else n, (bounded_start if bounded else n) + T.uint8(5), 2, explicit=True):
                    T.dist.wait_signal(s, expected_delta=T.Cast("uint32", i - (bounded_start if bounded else n)))

    return main


@tilelang.jit(target="sunmmio")
def large_constant_kernel(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, n: T.uint64):
        with T.Kernel():
            s = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            T.dist.wait_signal(s, expected_delta=T.Cast("uint32", T.truncdiv(n, T.uint64(1 << 63))))

    return main


@tilelang.jit(target="sunmmio")
def scalar_snapshot_kernel(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, n: T.uint32):
        with T.Kernel():
            value = T.alloc_var(T.uint32, init=n)
            s = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            before = T.Cast("uint32", value)
            value += T.uint32(1)
            T.dist.wait_signal(s, expected_delta=before - n)

    return main


@tilelang.jit(target="sunmmio")
def nested_loop_kernel(world_size: int = 1):
    @T.prim_func
    def main(
        A: T.MeshTensor((256,), T.placement.replicated(), T.float32),
        B: T.MeshTensor((256,), T.placement.replicated(), T.float32),
        rank_id: T.dist.RankId,
        rows: T.int32,
        cols: T.int32,
    ):
        with T.Kernel():
            s = T.dist.signal(kind=T.dist.SignalKind.DRAM_FLAGREG_INC, expect="manual")
            row_count = T.min(T.max(rows, 0), 2)
            col_count = T.min(T.max(cols, 0), 3)
            for i in T.serial(row_count):
                for j in T.serial(col_count):
                    begin = (i * col_count + j) * 32
                    T.dist.put(A[begin : begin + 32], B[begin : begin + 32], (rank_id + 1) % world_size, signal=s, submit=True)
                    T.dist.wait_signal(s, expected_delta=1)

    return main


@tilelang.jit(target="sunmmio")
def runtime_stride_kernel(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, start: T.int32, stop: T.int32, stride: T.int32):
        with T.Kernel():
            s = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            for i in T.serial(start, stop, stride):
                T.dist.put_signal(s, (rank_id + 1) % world_size, submit=True)
                T.dist.wait_signal(s, expected_delta=T.Cast("uint32", i - start))

    return main


def generate(factory, tmp_path, **kwargs):
    kernel = factory.get_tir(world_size=2, **kwargs)
    result = lower_to_device_tir(kernel, capture_passes=("tl.LowerDistSync",))
    source = codegen_sunmmio_suvm_mlir(result.device_mod)
    validate_suvm_mlir_with_npuir_opt(source, tmp_path, opt_args=("--verify-each", "--suvm-device-validate"))
    return result, source


def source_events(result):
    events = []

    def visit(node):
        if (
            isinstance(node, tvm.tir.BufferStore)
            and node.buffer.name == "src"
            and not (isinstance(node.value, (tvm.tir.IntImm, tvm.tir.FloatImm)) and node.value.value == 1)
        ):
            events.append("write")
        if isinstance(node, tvm.tir.Evaluate) and isinstance(node.value, tvm.tir.Call):
            name = node.value.op.name
            if name == "tl.dist_submit_":
                events.append("submit")
            elif name == "tl.dist_wait_send":
                events.append("flush")

    for func in result.pass_snapshot("tl.LowerDistSync").mod.functions.values():
        tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return events


def test_uint64_partition_uses_immutable_guard(tmp_path):
    generate(guarded_partition_kernel, tmp_path)


def test_guard_follows_immutable_snapshot_after_memory_changes(tmp_path):
    generate(snapshot_guard_kernel, tmp_path)


def test_guard_does_not_apply_to_a_later_read_from_same_address(tmp_path):
    with pytest.raises(tvm.error.InternalError, match="aligned"):
        generate(snapshot_guard_kernel, tmp_path, stale=True)


@pytest.mark.parametrize("nested", [False, True])
def test_partition_intersects_existing_bound(tmp_path, nested):
    generate(rank_partition_kernel, tmp_path, nested=nested)


@pytest.mark.parametrize("use", ["else", "after"])
def test_branch_bound_does_not_escape(tmp_path, use):
    with pytest.raises(tvm.error.InternalError, match="cannot be represented without loss"):
        generate(guarded_partition_kernel, tmp_path, use=use)


def test_scalar_narrowed_index_uses_real_partition_and_lane(tmp_path):
    result, source = generate(scalar_write_kernel, tmp_path)
    for n in (0, 1):
        batches, events = trace_suvm_control_flow(result, source, rank_id=0, n=n, trace_memory=True)
        assert len(batches) == 1
        assert ("set", 96 if n == 0 else 0, 2) in events
        assert any(e[0] == "store" and e[2:] == ((8064 if n == 0 else 0), 128) for e in events)


def test_queued_source_rejects_narrowed_scalar_overlap(tmp_path):
    with pytest.raises(tvm.error.InternalError, match="source is modified"):
        generate(scalar_write_kernel, tmp_path, position="queued")


def test_submitted_source_waits_before_narrowed_scalar_overlap(tmp_path):
    result, _ = generate(scalar_write_kernel, tmp_path, position="submitted")
    events = source_events(result)
    assert "flush" in events[events.index("submit") + 1 : events.index("write")]


@pytest.mark.parametrize("submitted", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("two_dim", [False, True])
def test_disjoint_tile_fill_keeps_overlap(tmp_path, submitted, parallel, two_dim):
    # The 2-D Parallel currently lowers to scalar 4x32 RMW, unlike the
    # one-row fill's 32-element tile store. Its physical write includes row 0.
    physical_overlap = two_dim and parallel
    if physical_overlap and not submitted:
        with pytest.raises(tvm.error.InternalError, match="source is modified"):
            generate(disjoint_fill_kernel, tmp_path, submitted=submitted, parallel=parallel, two_dim=two_dim)
        return
    result, source = generate(disjoint_fill_kernel, tmp_path, submitted=submitted, parallel=parallel, two_dim=two_dim)
    if submitted:
        events = source_events(result)
        waits = events[events.index("submit") + 1 : events.index("write")]
        assert ("flush" in waits) == physical_overlap
    if physical_overlap:
        assert "suvm.tile.set" in source and "!suvm.tile<4x32xi32>" in source


def test_queued_source_also_models_narrowed_send_address(tmp_path):
    with pytest.raises(tvm.error.InternalError, match="source is modified"):
        generate(scalar_write_kernel, tmp_path, position="queued", constant=True, dynamic_source=True)


def test_submitted_source_also_models_narrowed_send_address(tmp_path):
    result, _ = generate(scalar_write_kernel, tmp_path, position="submitted", constant=True, dynamic_source=True)
    events = source_events(result)
    assert "flush" in events[events.index("submit") + 1 : events.index("write")]


@pytest.mark.parametrize(
    "factory,kwargs,carrier",
    [(short_fill_kernel, {}, "16xf32"), (short_fill_kernel, {"cross": True}, "16xf32"), (small_tile_kernel, {}, "4x32xf32")],
)
@pytest.mark.parametrize("submitted", [True, False])
def test_store_protects_entire_physical_carrier(tmp_path, factory, kwargs, carrier, submitted):
    if not submitted:
        with pytest.raises(tvm.error.InternalError, match="source is modified"):
            generate(factory, tmp_path, submitted=False, **kwargs)
        return
    result, source = generate(factory, tmp_path, submitted=True, **kwargs)
    events = source_events(result)
    assert "flush" in events[events.index("submit") + 1 : events.index("write")]
    after = source.split("suvm.dist.batch_submit", 1)[1]
    assert after.index("suvm.dist.flush") < after.index("suvm.tile.store")
    if kwargs.get("cross"):
        assert after.count("suvm.tile.store") >= 2
    assert f"!suvm.tile<{carrier}>" in after.split("suvm.tile.store", 1)[1].splitlines()[0]


@pytest.mark.parametrize("submitted", [True, False])
def test_store_in_next_tile_is_disjoint_this_iteration(tmp_path, submitted):
    result, source = generate(next_tile_kernel, tmp_path, submitted=submitted)
    events = source_events(result)
    if submitted:
        assert "flush" not in events[events.index("submit") + 1 : events.index("write")]
        assert "suvm.dist.flush" not in source.split("suvm.dist.batch_submit", 1)[1].split("suvm.tile.store", 1)[0]
    assert "!suvm.tile<128xi32>" in source.split("suvm.dist.batch_put", 1)[1]


@pytest.mark.parametrize("step", [None, 1, 2])
@pytest.mark.parametrize("width", [8, 16, 32])
def test_unsigned_range_orders_original_bounds(tmp_path, step, width):
    result, source = generate(integer_range_kernel, tmp_path, step=step, width=width)
    maximum = (1 << width) - 1
    for start, stop in ((5, 3), (3, 3), (3, 5), (maximum, maximum - 2), (maximum - 2, maximum), (0, 3)):
        batches, _ = trace_suvm_control_flow(result, source, rank_id=0, start=start, stop=stop)
        assert len(batches) == len(range(start, stop, step or 1))


@pytest.mark.parametrize("step", [None, 1])
@pytest.mark.parametrize("width", [8, 16, 32])
def test_signed_unit_range_does_not_wrap(tmp_path, step, width):
    result, source = generate(integer_range_kernel, tmp_path, step=step, width=width, signed=True)
    low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
    cases = [(high, low), (low, low), (high - 2, high), (low, low + 2), (-2, 2)]
    if width == 8:
        cases.extend([(100, -100), (-100, 100), (low, high)])
    for start, stop in cases:
        batches, _ = trace_suvm_control_flow(result, source, rank_id=0, start=start, stop=stop)
        assert len(batches) == len(range(start, stop))


def test_stepped_loop_captures_mutable_start(tmp_path):
    result, source = generate(wide_window_kernel, tmp_path, span=5, stride=2, mutate=True, mode="offset")
    _, events = trace_suvm_control_flow(result, source, rank_id=0, n=5)
    assert [e[1] for e in events if e[0] == "wait"] == [0, 2, 6]


def test_public_large_uint64_step_is_constant(tmp_path):
    result, source = generate(wide_window_kernel, tmp_path, span=3, stride=1 << 63)
    batches, _ = trace_suvm_control_flow(result, source, rank_id=0, n=0)
    assert len(batches) == 1


def test_public_large_int32_step_has_one_iteration(tmp_path):
    result, source = generate(signed_step_kernel, tmp_path)
    for start, stop in ((0, 3), (-(1 << 31), (1 << 31) - 1), ((1 << 31) - 4, (1 << 31) - 1), (3, 0)):
        batches, _ = trace_suvm_control_flow(result, source, rank_id=0, start=start, stop=stop)
        assert len(batches) == len(range(start, stop, (1 << 31) - 1))


def test_large_uint64_constant_preserves_unsigned_bits(tmp_path):
    result, source = generate(large_constant_kernel, tmp_path)
    for n in (0, (1 << 63) - 1, 1 << 63, (1 << 64) - 1):
        _, events = trace_suvm_control_flow(result, source, rank_id=0, n=n)
        assert [e[1] for e in events if e[0] == "wait"] == [n >> 63]


def test_scalar_read_is_an_immutable_value(tmp_path):
    result, source = generate(scalar_snapshot_kernel, tmp_path)
    for n in (0, 5, (1 << 32) - 1):
        _, events = trace_suvm_control_flow(result, source, rank_id=0, n=n)
        assert [e[1] for e in events if e[0] == "wait"] == [0]


def test_nested_loops_preserve_both_index_identities(tmp_path):
    result, source = generate(nested_loop_kernel, tmp_path)
    batches, _ = trace_suvm_control_flow(result, source, rank_id=0, rows=2, cols=3)
    assert [p[3] for batch in batches for p in batch if p[0] == "data"] == [0, 32, 64, 96, 128, 160]


def test_runtime_stride_uses_dominating_positive_assertion(tmp_path):
    result, source = generate(runtime_stride_kernel, tmp_path)
    assert source.index("cf.assert") < source.index("arith.divsi")
    for start, stop, stride in ((0, 5, 2), (-4, 5, 3), (3, 0, 2)):
        batches, events = trace_suvm_control_flow(result, source, rank_id=0, start=start, stop=stop, stride=stride)
        indices = range(start, stop, stride)
        assert len(batches) == len(indices)
        target = 0
        expected = []
        for i in indices:
            target += i - start
            expected.append(target)
        assert [event[1] for event in events if event[0] == "wait"] == expected
    for stride in (0, -1):
        with pytest.raises(AssertionError):
            trace_suvm_control_flow(result, source, rank_id=0, start=0, stop=5, stride=stride)


def test_explicit_unroll_uses_the_entry_start_for_every_copy(tmp_path):
    result, source = generate(unroll_window_kernel, tmp_path, mutable=True)
    for n, expected in ((5, [0, 1, 3, 6, 10]), (-3, [0, 1, 3, 6, 10]), ((1 << 31) - 1, [])):
        _, events = trace_suvm_control_flow(result, source, rank_id=0, n=n)
        assert [e[1] for e in events if e[0] == "wait"] == expected


@pytest.mark.parametrize("bounded", [True, False])
def test_explicit_unroll_keeps_known_trip_count(tmp_path, bounded):
    result, source = generate(unroll_window_kernel, tmp_path, bounded=bounded)
    assert "scf.for" not in source
    for n in (0, 15, 250, 251, 255):
        _, events = trace_suvm_control_flow(result, source, rank_id=0, n=n)
        assert [e[1] for e in events if e[0] == "wait"] == ([0, 2, 6] if bounded or n <= 250 else [])
