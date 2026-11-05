"""Distributed send-queue submission tests."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import (
    collect_nodes,
    collect_op_names,
)


@tilelang.jit(target="sunmmio")
def explicit_batch_kernel_factory(world_size: int = 1):
    placement = T.placement.replicated()

    @T.prim_func
    def main(
        A: T.MeshTensor((64,), placement, T.float32),  # type: ignore
        B: T.MeshTensor((64,), placement, T.float32),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            signals = T.dist.signals(2)
            peer = (rank_id + 1) % world_size
            T.dist.put(A[0:32], B[0:32], peer, signal=signals[0])
            T.dist.put(A[32:64], B[32:64], peer, signal=signals[1])
            T.dist.submit()
            T.dist.wait_all(signals)

    return main


@tilelang.jit(target="sunmmio")
def submit_sugar_kernel_factory(world_size: int = 1):
    placement = T.placement.replicated()

    @T.prim_func
    def main(
        A: T.MeshTensor((32,), placement, T.float32),  # type: ignore
        B: T.MeshTensor((32,), placement, T.float32),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            signal = T.dist.signal()
            T.dist.put(
                A,
                B,
                (rank_id + 1) % world_size,
                signal=signal,
                submit=True,
            )
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def collective_submit_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((4, 8), T.bfloat16)
            dst = T.alloc_shared((world_size, 4, 8), T.bfloat16)
            signal = T.dist.signal()
            T.dist.all_gather(src, dst, signal=signal, submit=True)
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def barrier_arrive_submit_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            signal = T.dist.signal()
            T.dist.barrier_arrive(signal, submit=True)
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def missing_submit_kernel_factory(world_size: int = 1):
    placement = T.placement.replicated()

    @T.prim_func
    def main(
        A: T.MeshTensor((32,), placement, T.float32),  # type: ignore
        B: T.MeshTensor((32,), placement, T.float32),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            signal = T.dist.signal()
            T.dist.put(A, B, (rank_id + 1) % world_size, signal=signal)

    return main


@tilelang.jit(target="sunmmio")
def nonempty_sugar_kernel_factory(world_size: int = 1):
    placement = T.placement.replicated()

    @T.prim_func
    def main(
        A: T.MeshTensor((32,), placement, T.float32),  # type: ignore
        B: T.MeshTensor((32,), placement, T.float32),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            signals = T.dist.signals(2)
            peer = (rank_id + 1) % world_size
            T.dist.put(A, B, peer, signal=signals[0])
            T.dist.put(A, B, peer, signal=signals[1], submit=True)

    return main


@tilelang.jit(target="sunmmio")
def local_empty_submit_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            T.dist.put(src, dst, rank_id, signal=signal)
            T.dist.submit()

    return main


@tilelang.jit(target="sunmmio")
def cross_iteration_submit_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            for _ in T.serial(2):
                T.dist.submit()
                T.dist.wait_signal(signal)
                T.dist.put(src, dst, (rank_id + 1) % world_size, signal=signal)
            T.dist.submit()
            T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def cross_iteration_while_submit_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signals = T.dist.signals(1, expect="manual")
            iteration = T.alloc_var(T.int32, init=0)
            while iteration < 2:
                T.dist.submit()
                T.dist.wait_signal(signals[0], expected_delta=T.if_then_else(iteration == 0, 0, 1))
                T.dist.put(src, dst, (rank_id + 1) % world_size, signal=signals[0])
                iteration += 1
            T.dist.submit()
            T.dist.wait_signal(signals[0], expected_delta=1)

    return main


@tilelang.jit(target="sunmmio")
def early_return_kernel_factory(control_flow="if", world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, stop: T.int32):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            step = T.alloc_var(T.int32, init=0)
            T.dist.put(src, dst, (rank_id + 1) % world_size, signal=signal, submit=True)
            if control_flow == "for":
                for _ in T.serial(2):
                    if stop != 0:
                        T.evaluate(tvm.tir.ret(1))
            elif control_flow == "while":
                while step < 2:
                    if stop != 0:
                        T.evaluate(tvm.tir.ret(1))
                    step += 1
            else:
                if stop != 0:
                    T.evaluate(tvm.tir.ret(1))

    return main


@tilelang.jit(target="sunmmio")
def source_lifetime_kernel_factory(mode="submitted_overlap", redundant_submit=False, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((64,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal()
            T.dist.put(src[0:32], dst, (rank_id + 1) % world_size, signal=signal)
            if mode == "queued_overlap":
                src[0] = 0
            T.dist.submit()
            if redundant_submit:
                T.dist.submit()
            if mode == "submitted_overlap":
                src[0] = 0
            elif mode == "submitted_disjoint":
                src[32] = 0
            elif mode == "submitted_dma_overlap":
                T.copy(dst, src[0:32])

    return main


@tilelang.jit(target="sunmmio")
def mutable_source_index_kernel_factory(submitted=True, mutate=True, world_size: int = 1):
    @T.prim_func
    def main(initial: T.int32, rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((128,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            position = T.alloc_var(T.int32, init=initial)
            signal = T.dist.signal(expect="manual")
            T.dist.put(
                src[position * 32 : (position + 1) * 32],
                dst,
                (rank_id + 1) % world_size,
                signal=signal,
            )
            if submitted:
                T.dist.submit()
            if mutate:
                position += 1
                src[(position - 1) * 32] = 0
            else:
                src[(position + 1) * 32] = 0
            if not submitted:
                T.dist.submit()

    return main


@pytest.mark.parametrize("submitted", [False, True])
def test_mutable_source_index_update_preserves_source_protection(submitted):
    func = mutable_source_index_kernel_factory.get_tir(submitted=submitted, world_size=2)
    if not submitted:
        with pytest.raises(tvm.error.InternalError, match="modified before its queued batch is submitted"):
            lower_to_device_tir(func)
        return
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistSync")
    for mod in (result.pass_snapshot("tl.LowerDistSync").mod, result.device_mod):
        events = _source_write_events(mod)
        assert "tl.dist_wait_send" in events[events.index("tl.dist_submit_") + 1 : events.index("source_write")]


def _source_write_events(mod):
    events = []
    for node in collect_nodes(mod, (tvm.tir.Evaluate, tvm.tir.BufferStore)):
        if isinstance(node, tvm.tir.BufferStore):
            if node.buffer.name == "src":
                events.append("source_write")
        elif isinstance(node.value, tvm.tir.Call) and isinstance(node.value.op, tvm.ir.Op):
            events.append(node.value.op.name)
    return events


@pytest.mark.parametrize("submitted", [False, True])
def test_unchanged_source_index_keeps_disjoint_access_legal(submitted):
    result = lower_to_device_tir(mutable_source_index_kernel_factory.get_tir(submitted=submitted, mutate=False, world_size=2))
    events = _source_write_events(result.device_mod)
    # At most the existing pre-submit wait may precede the disjoint store.
    assert events[: events.index("source_write")].count("tl.dist_wait_send") == int(submitted)


@tilelang.jit(target="sunmmio")
def loop_source_lifetime_kernel_factory(loop_kind="serial", submit_each=True, world_size: int = 1):
    step = 2 if loop_kind == "step" else 1
    inner_count = 2 if loop_kind == "nested" else 1

    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((512,), T.bfloat16)
            dst = T.alloc_shared((512,), T.bfloat16)
            signal = T.dist.signal(expect="manual")
            if loop_kind == "while":
                position = T.alloc_var(T.int32, init=1)
                while position < 4:
                    src[(position - 1) * 32] = 0
                    T.dist.put(
                        src[position * 32 : (position + 1) * 32],
                        dst[position * 32 : (position + 1) * 32],
                        (rank_id + 1) % world_size,
                        signal=signal,
                    )
                    if submit_each:
                        T.dist.submit()
                    position += 1
            else:
                for outer in T.serial(step, 4 * step, step=step):
                    for inner in T.serial(inner_count):
                        offset = (outer * inner_count + inner) * 32
                        src[offset - step * 32] = 0
                        T.dist.put(
                            src[offset : offset + 32],
                            dst[offset : offset + 32],
                            (rank_id + 1) % world_size,
                            signal=signal,
                        )
                        if submit_each:
                            T.dist.submit()
                        if loop_kind == "continue":
                            continue
            if not submit_each:
                T.dist.submit()

    return main


@tilelang.jit(target="sunmmio")
def loop_jump_kernel_factory(jump="break", submit_after_loop=False, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal(expect="manual")
            for _ in T.serial(2):
                T.dist.put(src, dst, (rank_id + 1) % world_size, signal=signal)
                if rank_id == 0:
                    if jump == "break":
                        break
                    else:
                        continue
                T.dist.submit()
            if submit_after_loop:
                T.dist.submit()

    return main


def test_explicit_submit_batches_two_puts():
    result = lower_to_device_tir(
        explicit_batch_kernel_factory.get_tir(world_size=4),
        capture_passes=("tl.LowerDistCommunication", "tl.LowerDistSync"),
    )

    lowered_names = collect_op_names(result.pass_snapshot("tl.LowerDistCommunication").mod)
    assert lowered_names.count("tl.dist_put_") == 2
    assert lowered_names.count("tl.dist_submit_") == 1

    injected_names = collect_op_names(result.pass_snapshot("tl.LowerDistSync").mod)
    injected_script = result.pass_snapshot("tl.LowerDistSync").mod.script()
    assert injected_names.count("tl.dist_submit_") == 1
    assert "tl.dist_batch_begin_" not in injected_names
    assert injected_names.count("tl.dist_wait_send") == 2
    first_put = injected_script.index("T.dist_put_(")
    pre_submit_wait = injected_script.index("T.dist_wait_send()")
    submit = injected_script.index("T.dist_submit_()")
    exit_wait = injected_script.rindex("T.dist_wait_send()")
    assert first_put < pre_submit_wait < submit < exit_wait
    device_script = result.device_mod.script()
    assert device_script.rindex("T.dist_wait_send()") < device_script.index("return 0")
    assert "            T.dist_wait_send()\n        return 0" in device_script


def test_submit_true_marks_one_strict_batch():
    func = submit_sugar_kernel_factory.get_tir(world_size=4)
    frontend_names = collect_op_names(func)
    assert frontend_names.count("tl.dist_batch_begin") == 1
    assert frontend_names.count("tl.dist_submit") == 1

    result = lower_to_device_tir(func, capture_passes="tl.LowerDistSync")
    names = collect_op_names(result.pass_snapshot("tl.LowerDistSync").mod)
    assert names.count("tl.dist_put_") == 1
    assert names.count("tl.dist_submit_") == 1
    assert "tl.dist_batch_begin_" not in names


@pytest.mark.parametrize(
    "factory, send_op, send_count",
    [
        (collective_submit_kernel_factory, "tl.dist_put_", 3),
        (barrier_arrive_submit_kernel_factory, "tl.dist_signal_put_", 3),
    ],
)
def test_collective_submit_true_submits_one_chain(factory, send_op, send_count):
    result = lower_to_device_tir(factory.get_tir(world_size=4), capture_passes="tl.LowerDistSync")
    names = collect_op_names(result.pass_snapshot("tl.LowerDistSync").mod)
    assert names.count(send_op) == send_count
    assert names.count("tl.dist_submit_") == 1


def test_missing_submit_is_rejected():
    with pytest.raises(tvm.error.InternalError, match="send queue is not submitted"):
        lower_to_device_tir(missing_submit_kernel_factory.get_tir(world_size=4))


def test_submit_true_rejects_a_nonempty_queue():
    with pytest.raises(tvm.error.InternalError, match="requires an empty send queue"):
        lower_to_device_tir(nonempty_sugar_kernel_factory.get_tir(world_size=4))


def test_statically_empty_submit_is_removed():
    result = lower_to_device_tir(
        local_empty_submit_kernel_factory.get_tir(world_size=4),
        capture_passes="tl.LowerDistSync",
    )
    names = collect_op_names(result.pass_snapshot("tl.LowerDistSync").mod)
    assert "tl.dist_put_" not in names
    assert "tl.dist_submit_" not in names
    assert "tl.dist_wait_send" not in names


@pytest.mark.parametrize(
    "factory",
    [cross_iteration_submit_kernel_factory, cross_iteration_while_submit_kernel_factory],
)
def test_loop_submit_keeps_a_submission_needed_on_later_iterations(factory):
    result = lower_to_device_tir(
        factory.get_tir(world_size=4),
        capture_passes="tl.LowerDistSync",
    )
    injected = result.pass_snapshot("tl.LowerDistSync").mod
    script = injected.script()
    assert collect_op_names(injected).count("tl.dist_submit_") == 2
    assert "T.dist_submit_()" in script
    assert script.count("T.dist_submit_()") == 2

    device = next(func for func in result.device_mod.functions.values() if isinstance(func, tvm.tir.PrimFunc))
    source_allocations = []
    tvm.tir.stmt_functor.post_order_visit(
        device.body,
        lambda node: source_allocations.append(node) if isinstance(node, tvm.tir.Allocate) and node.buffer_var.name == "src" else None,
    )
    assert len(source_allocations) == 1
    source_ops = []
    tvm.tir.stmt_functor.post_order_visit(
        source_allocations[0].body,
        lambda node: source_ops.append(node.op.name) if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op) else None,
    )
    assert source_ops[-1] == "tl.dist_wait_send"


def _has_wait_before_return(func_or_mod, return_value):
    found = []
    funcs = func_or_mod.functions.values() if isinstance(func_or_mod, tvm.IRModule) else (func_or_mod,)

    def visit(node):
        if not isinstance(node, tvm.tir.SeqStmt):
            return
        for previous, current in zip(node.seq, node.seq[1:]):
            if (
                isinstance(current, tvm.tir.Evaluate)
                and isinstance(current.value, tvm.tir.Call)
                and current.value.op.same_as(tvm.ir.Op.get("tir.ret"))
                and int(current.value.args[0]) == return_value
            ):
                found.append(
                    isinstance(previous, tvm.tir.Evaluate)
                    and isinstance(previous.value, tvm.tir.Call)
                    and previous.value.op.same_as(tvm.ir.Op.get("tl.dist_wait_send"))
                )

    for func in funcs:
        tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return bool(found) and all(found)


@pytest.mark.parametrize("control_flow", ["if", "for", "while"])
def test_early_return_waits_on_every_in_flight_exit(control_flow):
    result = lower_to_device_tir(
        early_return_kernel_factory.get_tir(control_flow, world_size=4),
        capture_passes="tl.LowerDistSync",
    )
    assert _has_wait_before_return(result.pass_snapshot("tl.LowerDistSync").mod, 1)
    assert _has_wait_before_return(result.device_mod, 1)


def test_manual_sender_wait_is_rejected():
    with pytest.raises(RuntimeError, match="inserted automatically"):
        T.dist.wait()


def test_source_write_before_submit_is_rejected():
    with pytest.raises(tvm.error.InternalError, match="modified before its queued batch is submitted"):
        lower_to_device_tir(source_lifetime_kernel_factory.get_tir(mode="queued_overlap", world_size=4))


@pytest.mark.parametrize("redundant_submit", [False, True])
def test_source_write_after_submit_gets_automatic_wait(redundant_submit):
    result = lower_to_device_tir(
        source_lifetime_kernel_factory.get_tir(mode="submitted_overlap", redundant_submit=redundant_submit, world_size=4),
        capture_passes="tl.LowerDistSync",
    )
    for mod in (result.pass_snapshot("tl.LowerDistSync").mod, result.device_mod):
        script = mod.script()
        assert collect_op_names(mod).count("tl.dist_submit_") == 1
        submit = script.index("T.dist_submit_()")
        wait = script.index("T.dist_wait_send()", submit)
        store = script.index("src[0] =", submit)
        assert submit < wait < store


@pytest.mark.parametrize("redundant_submit", [False, True])
def test_disjoint_source_write_keeps_wait_at_scope_exit(redundant_submit):
    result = lower_to_device_tir(
        source_lifetime_kernel_factory.get_tir(mode="submitted_disjoint", redundant_submit=redundant_submit, world_size=4),
        capture_passes="tl.LowerDistSync",
    )
    script = result.pass_snapshot("tl.LowerDistSync").mod.script()
    submit = script.index("T.dist_submit_()")
    store = script.index("src[32] =", submit)
    wait = script.index("T.dist_wait_send()", submit)
    assert submit < store < wait


def test_async_source_write_after_submit_gets_automatic_wait():
    result = lower_to_device_tir(
        source_lifetime_kernel_factory.get_tir(mode="submitted_dma_overlap", world_size=4),
        capture_passes="tl.LowerDistSync",
    )
    script = result.pass_snapshot("tl.LowerDistSync").mod.script()
    submit = script.index("T.dist_submit_()")
    wait = script.index("T.dist_wait_send()", submit)
    copy_loop = script.index("for i in", submit)
    assert submit < wait < copy_loop
    assert "T.dist_wait_send()" not in script[copy_loop:]


@pytest.mark.parametrize("loop_kind", ["serial", "step", "nested", "continue", "while"])
def test_loop_carried_source_overwrite_waits_for_previous_iteration(loop_kind):
    result = lower_to_device_tir(
        loop_source_lifetime_kernel_factory.get_tir(loop_kind=loop_kind, world_size=4),
        capture_passes="tl.LowerDistSync",
    )
    for mod in (result.pass_snapshot("tl.LowerDistSync").mod, result.device_mod):
        protected_stores = []
        for seq in collect_nodes(mod, tvm.tir.SeqStmt):
            for index, stmt in enumerate(seq.seq):
                if not isinstance(stmt, tvm.tir.BufferStore) or stmt.buffer.name != "src":
                    continue
                previous = seq.seq[index - 1] if index else None
                protected_stores.append(
                    isinstance(previous, tvm.tir.Evaluate)
                    and isinstance(previous.value, tvm.tir.Call)
                    and previous.value.op.same_as(tvm.ir.Op.get("tl.dist_wait_send"))
                )
        assert protected_stores and all(protected_stores)


@pytest.mark.parametrize("loop_kind", ["serial", "step", "nested", "continue", "while"])
def test_loop_carried_source_overwrite_before_submit_is_rejected(loop_kind):
    with pytest.raises(tvm.error.InternalError, match="modified before its queued batch is submitted"):
        lower_to_device_tir(loop_source_lifetime_kernel_factory.get_tir(loop_kind=loop_kind, submit_each=False, world_size=4))


def _dist_leaf(name, *args):
    return tvm.tir.Evaluate(tvm.tir.call_intrin("handle", tvm.ir.Op.get("tl." + name), *args))


def _queued_signal_put():
    return _dist_leaf("dist_signal_put_", 1, "sram_flagreg_inc", 0, tvm.tir.const(0, "uint8"))


def _encoded_region(buffer, access_mask):
    return tvm.tir.Call(
        "handle",
        tvm.ir.Op.get("tl.tileop.region"),
        [buffer[0], access_mask, buffer.shape[0]],
    )


def _local_source_loop(jump, submitted):
    src = tvm.tir.decl_buffer((32,), "bfloat16", name="src", scope="shared.rsram")
    dst = tvm.tir.decl_buffer((32,), "bfloat16", name="dst", scope="shared.rsram")
    statements = [
        _dist_leaf(
            "dist_put_",
            _encoded_region(src, 1),
            _encoded_region(dst, 2),
            1,
            "sram_flagreg_inc",
            0,
            tvm.tir.const(0, "uint8"),
        )
    ]
    if submitted:
        statements.append(_dist_leaf("dist_submit_"))
    statements.append(_loop_jump(jump))
    body = tvm.tir.Allocate(src.data, src.dtype, src.shape, True, tvm.tir.SeqStmt(statements))
    loop = tvm.tir.For(tvm.tir.Var("i", "int32"), 0, 2, tvm.tir.ForKind.SERIAL, body)
    return tvm.tir.Allocate(dst.data, dst.dtype, dst.shape, True, loop)


def _inject_queue_body(body, params=()):
    core = tvm.te.thread_axis((0, 4), "blockIdx.x")
    body = tvm.tir.SeqStmt(
        [
            tvm.tir.AttrStmt(core, "thread_extent", 4, body),
            tvm.tir.Evaluate(tvm.tir.ret(0)),
        ]
    )
    func = tvm.tir.PrimFunc(params, body).with_attr("tl.dist.world_size", 4)
    mod = tilelang.transform.LowerDistSync()(tvm.IRModule({"main": func}))
    return mod["main"]


@pytest.mark.parametrize("submitted", [False, True])
@pytest.mark.parametrize("index_write", ["alias", "async"])
@pytest.mark.parametrize("payload_write", ["store", "async"])
def test_index_storage_writes_preserve_queued_and_inflight_sources(submitted, index_write, payload_write):
    src = tvm.tir.decl_buffer((128,), "bfloat16", name="src", scope="shared.rsram")
    dst = tvm.tir.decl_buffer((32,), "bfloat16", name="dst", scope="shared.rsram")
    index_scope = "shared.rsram" if index_write == "async" else "local.var"
    index = tvm.tir.decl_buffer((32,), "int32", name="index", scope=index_scope)
    alias = tvm.tir.decl_buffer(index.shape, index.dtype, name="alias", scope=index_scope, data=index.data)
    seed = tvm.tir.decl_buffer((32,), "int32", name="seed", scope="global")

    def region(buffer, access, start=0):
        return tvm.tir.Call("handle", tvm.ir.Op.get("tl.tileop.region"), [buffer[start], access, 32])

    def copy(source, destination):
        unit = tvm.tir.Call("handle", tvm.ir.Op.get("tl.odma_unit"), [tvm.tir.StringImm("odma0")])
        return _dist_leaf("dma_copy", source, destination, 0, unit)

    statements = [
        _dist_leaf(
            "dist_put_",
            region(src, 1, index[0] * 32),
            region(dst, 2),
            1,
            "sram_flagreg_inc",
            0,
            tvm.tir.const(0, "uint8"),
        )
    ]
    if submitted:
        statements.append(_dist_leaf("dist_submit_"))
    statements.append(copy(region(seed, 1), region(index, 2)) if index_write == "async" else tvm.tir.BufferStore(alias, alias[0] + 1, [0]))
    position = (index[0] - 1) * 32
    statements.append(
        copy(region(dst, 1), region(src, 2, position))
        if payload_write == "async"
        else tvm.tir.BufferStore(src, tvm.tir.const(0, "bfloat16"), [position])
    )
    if not submitted:
        statements.append(_dist_leaf("dist_submit_"))
    body = tvm.tir.SeqStmt(statements)
    params = [src.data, dst.data, index.data, seed.data]
    if not submitted:
        with pytest.raises(tvm.error.InternalError, match="modified before its queued batch is submitted"):
            _inject_queue_body(body, params)
        return
    func = _inject_queue_body(body, params)
    events = _source_write_events(func)
    overwrite = events.index("source_write") if payload_write == "store" else len(events) - 1 - events[::-1].index("tl.dma_copy")
    assert "tl.dist_wait_send" in events[events.index("tl.dist_submit_") + 1 : overwrite]


def test_loop_carried_source_preserves_invariant_disjoint_dimensions():
    src = tvm.tir.decl_buffer((2, 128), "bfloat16", name="src", scope="shared.rsram")
    dst = tvm.tir.decl_buffer((2, 128), "bfloat16", name="dst", scope="shared.rsram")
    i = tvm.tir.Var("i", "int32")

    def region(buffer, access):
        return tvm.tir.Call("handle", tvm.ir.Op.get("tl.tileop.region"), [buffer[0, i * 32], access, 1, 32])

    loop = tvm.tir.For(
        i,
        1,
        3,
        tvm.tir.ForKind.SERIAL,
        tvm.tir.SeqStmt(
            [
                tvm.tir.BufferStore(src, tvm.tir.const(0, "bfloat16"), [1, (i - 1) * 32]),
                _dist_leaf("dist_put_", region(src, 1), region(dst, 2), 1, "sram_flagreg_inc", 0, tvm.tir.const(0, "uint8")),
                _dist_leaf("dist_submit_"),
            ]
        ),
    )
    func = _inject_queue_body(loop, [src.data, dst.data])
    # Only the existing pre-submit wait and final drain are needed: row 1 is
    # disjoint from every previous send, even as the column changes each round.
    assert collect_op_names(func).count("tl.dist_wait_send") == 2


def test_loop_local_let_address_is_not_reused_as_previous_iteration_value():
    src = tvm.tir.decl_buffer((128,), "bfloat16", name="src", scope="shared.rsram")
    dst = tvm.tir.decl_buffer((128,), "bfloat16", name="dst", scope="shared.rsram")
    i = tvm.tir.Var("i", "int32")
    offset = tvm.tir.Var("offset", "int32")

    def region(buffer, access):
        return tvm.tir.Call("handle", tvm.ir.Op.get("tl.tileop.region"), [buffer[offset], access, 32])

    store = tvm.tir.BufferStore(src, tvm.tir.const(0, "bfloat16"), [offset - 32])
    body = tvm.tir.LetStmt(
        offset,
        i * 32,
        tvm.tir.SeqStmt(
            [
                store,
                _dist_leaf("dist_put_", region(src, 1), region(dst, 2), 1, "sram_flagreg_inc", 0, tvm.tir.const(0, "uint8")),
                _dist_leaf("dist_submit_"),
            ]
        ),
    )
    func = _inject_queue_body(tvm.tir.For(i, 1, 3, tvm.tir.ForKind.SERIAL, body), [src.data, dst.data])
    loops = collect_nodes(func, tvm.tir.For)
    assert len(loops) == 1
    assert isinstance(loops[0].body, tvm.tir.LetStmt)
    statements = loops[0].body.body.seq
    assert statements[0].value.op.same_as(tvm.ir.Op.get("tl.dist_wait_send"))
    assert tvm.ir.structural_equal(statements[1], store)


@pytest.mark.parametrize("strict_batch", [False, True])
def test_early_return_rejects_unsubmitted_queue(strict_batch):
    stop = tvm.tir.Var("stop", "int32")
    statements = [_queued_signal_put()]
    if strict_batch:
        statements.insert(0, _dist_leaf("dist_batch_begin_"))
    statements.extend(
        [
            tvm.tir.IfThenElse(stop != 0, tvm.tir.Evaluate(tvm.tir.ret(1)), None),
            _dist_leaf("dist_submit_"),
            _dist_leaf("dist_wait_send"),
        ]
    )
    with pytest.raises(tvm.error.InternalError, match="before return"):
        _inject_queue_body(tvm.tir.SeqStmt(statements), [stop])


def test_unreachable_put_does_not_affect_exit_validation():
    func = _inject_queue_body(
        tvm.tir.SeqStmt(
            [
                _queued_signal_put(),
                _dist_leaf("dist_submit_"),
                tvm.tir.Evaluate(tvm.tir.ret(1)),
                _queued_signal_put(),
            ]
        )
    )
    assert _has_wait_before_return(func, 1)


def test_returned_branch_does_not_add_wait_to_completed_branch():
    stop = tvm.tir.Var("stop", "int32")
    func = _inject_queue_body(
        tvm.tir.SeqStmt(
            [
                _queued_signal_put(),
                _dist_leaf("dist_submit_"),
                tvm.tir.IfThenElse(
                    stop != 0,
                    tvm.tir.Evaluate(tvm.tir.ret(1)),
                    _dist_leaf("dist_wait_send"),
                ),
                tvm.tir.Evaluate(tvm.tir.ret(2)),
            ]
        ),
        [stop],
    )
    assert _has_wait_before_return(func, 1)
    # One pre-submit wait, one early-exit wait, one explicit else-branch wait.
    assert collect_op_names(func).count("tl.dist_wait_send") == 3
    assert not _has_wait_before_return(func, 2)


@pytest.mark.parametrize("loop_kind", ["for", "while"])
def test_early_return_accounts_for_in_flight_loop_backedge(loop_kind):
    stop = tvm.tir.Var("stop", "int32")
    body = tvm.tir.SeqStmt(
        [
            tvm.tir.IfThenElse(stop != 0, tvm.tir.Evaluate(tvm.tir.ret(1)), None),
            _queued_signal_put(),
            _dist_leaf("dist_submit_"),
        ]
    )
    if loop_kind == "for":
        body = tvm.tir.For(tvm.tir.Var("i", "int32"), 0, 2, tvm.tir.ForKind.SERIAL, body)
    else:
        body = tvm.tir.While(stop < 2, body)
    func = _inject_queue_body(
        tvm.tir.SeqStmt(
            [
                body,
                _dist_leaf("dist_wait_send"),
            ]
        ),
        [stop],
    )
    assert _has_wait_before_return(func, 1)


def _loop_jump(jump):
    return tvm.tir.Evaluate(tvm.tir.call_intrin("int32", "tir." + jump + "_loop"))


def _queue_loop(loop_kind, body):
    if loop_kind == "for":
        return tvm.tir.For(tvm.tir.Var("i", "int32"), 0, 2, tvm.tir.ForKind.SERIAL, body)
    counter = tvm.tir.decl_buffer((1,), "int32", name="step", scope="local")
    loop = tvm.tir.While(
        counter[0] < 2,
        tvm.tir.SeqStmt([tvm.tir.BufferStore(counter, counter[0] + 1, [0]), body]),
    )
    return tvm.tir.Allocate(
        counter.data,
        counter.dtype,
        [1],
        True,
        tvm.tir.SeqStmt([tvm.tir.BufferStore(counter, 0, [0]), loop]),
    )


@pytest.mark.parametrize("jump", ["break", "continue"])
@pytest.mark.parametrize("submit_after_loop", [False, True])
def test_loop_jump_submission_through_device_tir(jump, submit_after_loop):
    func = loop_jump_kernel_factory.get_tir(jump, submit_after_loop, world_size=4)
    if not submit_after_loop:
        with pytest.raises(tvm.error.InternalError, match="send queue is not submitted"):
            lower_to_device_tir(func)
        return
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistSync")
    for mod in (result.pass_snapshot("tl.LowerDistSync").mod, result.device_mod):
        names = collect_op_names(mod)
        assert "tir." + jump + "_loop" in names
        assert names.count("tl.dist_submit_") == 2
        # Two pre-submit waits and one final drain; none at the jump itself.
        assert names.count("tl.dist_wait_send") == 3


@pytest.mark.parametrize("jump", ["break", "continue"])
def test_loop_jump_waits_before_releasing_local_source(jump):
    script = _inject_queue_body(_local_source_loop(jump, submitted=True)).script()
    jump_index = script.index("T." + jump + "_loop()")
    wait_index = script.rfind("T.dist_wait_send()", 0, jump_index)
    submit_index = script.rfind("T.dist_submit_()", 0, jump_index)
    assert submit_index < wait_index < jump_index


@pytest.mark.parametrize("jump", ["break", "continue"])
def test_loop_jump_rejects_releasing_unsubmitted_local_source(jump):
    with pytest.raises(tvm.error.InternalError, match="released by .* before its queued batch is submitted"):
        _inject_queue_body(_local_source_loop(jump, submitted=False))


@pytest.mark.parametrize("loop_kind", ["for", "while"])
@pytest.mark.parametrize("jump", ["break", "continue"])
@pytest.mark.parametrize("submit_after_loop", [False, True])
def test_loop_jump_preserves_pending_queue(loop_kind, jump, submit_after_loop):
    stop = tvm.tir.Var("stop", "int32")
    body = _queue_loop(
        loop_kind,
        tvm.tir.SeqStmt(
            [
                _queued_signal_put(),
                tvm.tir.IfThenElse(stop != 0, _loop_jump(jump), None),
                _dist_leaf("dist_submit_"),
            ]
        ),
    )
    if not submit_after_loop:
        with pytest.raises(tvm.error.InternalError, match="send queue is not submitted"):
            _inject_queue_body(body, [stop])
        return
    func = _inject_queue_body(tvm.tir.SeqStmt([body, _dist_leaf("dist_submit_")]), [stop])
    names = collect_op_names(func)
    assert names.count("tl.dist_submit_") == 2
    assert names.count("tl.dist_wait_send") == 3


@pytest.mark.parametrize("loop_kind", ["for", "while"])
@pytest.mark.parametrize("jump", ["break", "continue"])
def test_loop_jump_skipping_wait_keeps_exit_drain(loop_kind, jump):
    stop = tvm.tir.Var("stop", "int32")
    body = _queue_loop(
        loop_kind,
        tvm.tir.SeqStmt(
            [
                _queued_signal_put(),
                _dist_leaf("dist_submit_"),
                tvm.tir.IfThenElse(stop != 0, _loop_jump(jump), None),
                _dist_leaf("dist_wait_send"),
            ]
        ),
    )
    func = _inject_queue_body(body, [stop])
    exit_waits = []

    def visit(node):
        if isinstance(node, tvm.tir.SeqStmt):
            for previous, current in zip(node.seq, node.seq[1:]):
                if isinstance(previous, (tvm.tir.For, tvm.tir.While)):
                    exit_waits.append(
                        isinstance(current, tvm.tir.Evaluate)
                        and isinstance(current.value, tvm.tir.Call)
                        and current.value.op.same_as(tvm.ir.Op.get("tl.dist_wait_send"))
                    )

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    assert exit_waits == [True]
    # A reachable explicit wait after the loop already supplies the drain.
    explicit = _inject_queue_body(tvm.tir.SeqStmt([body, _dist_leaf("dist_wait_send")]), [stop])
    assert collect_op_names(explicit).count("tl.dist_wait_send") == 3


@pytest.mark.parametrize("loop_kind", ["for", "while"])
@pytest.mark.parametrize("jump", ["break", "continue"])
def test_loop_jump_cannot_leave_strict_batch_open(loop_kind, jump):
    body = _queue_loop(
        loop_kind,
        tvm.tir.SeqStmt(
            [
                _dist_leaf("dist_batch_begin_"),
                _queued_signal_put(),
                _loop_jump(jump),
                _dist_leaf("dist_submit_"),
            ]
        ),
    )
    with pytest.raises(tvm.error.InternalError, match="submit boundary before " + jump):
        _inject_queue_body(tvm.tir.SeqStmt([body, _dist_leaf("dist_submit_")]))


@pytest.mark.parametrize("loop_kind", ["for", "while"])
@pytest.mark.parametrize("jump", ["break", "continue"])
def test_only_continue_propagates_pending_queue_to_loop_header(loop_kind, jump):
    body = _queue_loop(
        loop_kind,
        tvm.tir.SeqStmt(
            [
                _dist_leaf("dist_batch_begin_"),
                _queued_signal_put(),
                _dist_leaf("dist_submit_"),
                _queued_signal_put(),
                _loop_jump(jump),
            ]
        ),
    )
    body = tvm.tir.SeqStmt([body, _dist_leaf("dist_submit_")])
    if jump == "continue":
        with pytest.raises(tvm.error.InternalError, match="requires an empty send queue"):
            _inject_queue_body(body)
    else:
        assert collect_op_names(_inject_queue_body(body)).count("tl.dist_submit_") == 2


@pytest.mark.parametrize("jump", ["break", "continue"])
def test_nested_loop_jump_targets_innermost_loop(jump):
    inner = _queue_loop("while", tvm.tir.SeqStmt([_queued_signal_put(), _loop_jump(jump)]))
    outer = _queue_loop("for", tvm.tir.SeqStmt([inner, _dist_leaf("dist_submit_")]))
    func = _inject_queue_body(outer)
    assert collect_op_names(func).count("tl.dist_submit_") == 1
    assert collect_op_names(func).count("tl.dist_wait_send") == 2


@pytest.mark.parametrize("jump", ["break", "continue"])
def test_loop_jump_unreachable_suffix_does_not_affect_queue(jump):
    body = _queue_loop(
        "for",
        tvm.tir.SeqStmt(
            [
                _loop_jump(jump),
                _queued_signal_put(),
                tvm.tir.Evaluate(tvm.tir.ret(1)),
            ]
        ),
    )
    func = _inject_queue_body(body)
    assert "tl.dist_wait_send" not in collect_op_names(func)


def test_continue_retains_submit_needed_by_later_iterations():
    body = _queue_loop(
        "for",
        tvm.tir.SeqStmt(
            [
                _dist_leaf("dist_submit_"),
                _queued_signal_put(),
                _loop_jump("continue"),
            ]
        ),
    )
    func = _inject_queue_body(tvm.tir.SeqStmt([body, _dist_leaf("dist_submit_")]))
    assert collect_op_names(func).count("tl.dist_submit_") == 2


@pytest.mark.parametrize("jump", ["break", "continue"])
@pytest.mark.parametrize("submitted", [False, True])
def test_loop_jump_preserves_state_at_return(jump, submitted):
    statements = [_queued_signal_put()]
    if submitted:
        statements.append(_dist_leaf("dist_submit_"))
    statements.extend([_loop_jump(jump), _dist_leaf("dist_wait_send")])
    body = tvm.tir.SeqStmt(
        [
            _queue_loop("for", tvm.tir.SeqStmt(statements)),
            tvm.tir.Evaluate(tvm.tir.ret(1)),
        ]
    )
    if not submitted:
        with pytest.raises(tvm.error.InternalError, match="send queue is not submitted before return"):
            _inject_queue_body(body)
    else:
        assert _has_wait_before_return(_inject_queue_body(body), 1)
