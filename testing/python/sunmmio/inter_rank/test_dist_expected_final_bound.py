"""Final mathematical flag peaks are checked in LowerDistSync before narrowing."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.utils.target import determine_target

from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import base_mod, collect_calls, collect_nodes


@tilelang.jit(target="sunmmio")
def final_bound_kernel(mode="sequence", amount=255, group=False, loop_delta=1, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, choice: T.int32):
        with T.Kernel():
            if group:
                ready = T.dist.signals(2, kind=T.dist.SignalKind.SRAM_FLAGREG_INC, expect="auto" if mode == "auto_completion" else "manual")
            else:
                ready = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_INC, expect="manual")
            if mode == "sequence":
                T.dist.wait_signal(ready[0] if group else ready, expected_delta=250)
                T.dist.submit()
                T.dist.wait_signal(ready[0] if group else ready, expected_delta=0)
                T.dist.wait_signal(ready[0] if group else ready, expected_delta=amount - 250)
            elif mode == "loop":
                for _ in T.serial(amount):
                    T.dist.wait_signal(ready[0] if group else ready, expected_delta=loop_delta)
            elif mode in ("while", "while_unknown", "while_no_progress"):
                iteration = T.alloc_var(T.int32, init=0)
                while iteration < (choice if mode == "while_unknown" else amount):
                    T.dist.wait_signal(ready[0] if group else ready, expected_delta=loop_delta)
                    if mode != "while_no_progress":
                        iteration += 1
            elif mode == "nonmonotone":
                T.dist.wait_signal(ready[0] if group else ready, expected_delta=300)
                T.dist.wait_signal(ready[0] if group else ready, expected_delta=T.int32(-100))
            elif mode == "members":
                T.dist.wait_signal(ready[0], expected_delta=200)
                T.dist.wait_signal(ready[1], expected_delta=200)
                T.dist.wait_signal(ready[0], expected_delta=amount - 200)
            elif mode == "branch":
                if choice > 0:
                    T.dist.wait_signal(ready[0] if group else ready, expected_delta=200)
                else:
                    T.dist.wait_signal(ready[0] if group else ready, expected_delta=100)
                T.dist.wait_signal(ready[0] if group else ready, expected_delta=amount - 200)
            elif mode == "completion":
                deltas = T.alloc_shared((16,), T.int32)
                T.fill(deltas, 200)
                first = T.dist.completion(ready, expected_deltas=deltas[0:2])
                T.dist.wait_all(first)
                T.fill(deltas, amount - 200)
                second = T.dist.completion(ready, expected_deltas=deltas[0:2])
                T.dist.wait_all(second)
            elif mode == "auto_completion":
                source = T.alloc_shared((32, 32), T.bfloat16)
                destination = T.alloc_shared((32, 32), T.bfloat16)
                T.fill(source, T.bfloat16(1))
                for _ in T.serial(amount):
                    T.dist.put(source, destination, (rank_id + 1) % world_size, signal=ready[0], submit=True)
                    done = T.dist.completion(ready)
                    T.dist.wait_all(done)

    return main


@tilelang.jit(target="sunmmio")
def for_bound_snapshot_kernel(entry=2, first=200, second=200, group=False, maybe_zero=False, mode="shrink", world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, choice: T.int32):
        with T.Kernel():
            if group:
                ready = T.dist.signals(2, kind=T.dist.SignalKind.SRAM_FLAGREG_INC, expect="manual")
            else:
                ready = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_INC, expect="manual")
            limit = T.alloc_var(T.int32, init=entry)
            if maybe_zero and choice > 0:
                limit = 0
            if mode == "unknown":
                limit = choice
            for index in T.serial(limit):
                if mode == "correlated":
                    delta = T.if_then_else(limit == 0, 256, first)
                else:
                    delta = T.if_then_else(index == 0, first, second)
                T.dist.wait_signal(ready[0] if group else ready, expected_delta=delta)
                limit = 0

    return main


@tilelang.jit(target="sunmmio")
def integer_delta_kernel(mode="cast", group=False, through_array=False, tail=0, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            if group:
                ready = T.dist.signals(2, kind=T.dist.SignalKind.SRAM_FLAGREG_INC, expect="manual")
            else:
                ready = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_INC, expect="manual")
            if mode == "loop":
                limit = T.Cast("int32", T.Cast("uint8", rank_id - 1))
                for _ in T.serial(limit):
                    T.dist.wait_signal(ready[0] if group else ready, expected_delta=1)
            else:
                if mode == "cast":
                    delta = T.Cast("uint8", rank_id - 1)
                elif mode == "add":
                    delta = T.Cast("uint8", rank_id) + T.uint8(255)
                elif mode == "sub":
                    delta = T.uint8(0) - T.Cast("uint8", rank_id)
                else:
                    delta = T.if_then_else(T.Cast("uint8", rank_id - 1) == 0, 0, 255)
                if through_array:
                    deltas = T.alloc_shared((16,), T.int32)
                    deltas[0] = T.Cast("int32", delta)
                    T.dist.wait_signal(ready[0] if group else ready, expected_delta=deltas[0])
                else:
                    T.dist.wait_signal(ready[0] if group else ready, expected_delta=delta)
            T.dist.wait_signal(ready[0] if group else ready, expected_delta=tail)

    return main


def planned_expected_module(mode="reset"):
    """Internal fixture with logical kind and the ordinary signal-plan attrs."""
    rank = tvm.tir.Var("rank", "int32")
    expected = tvm.tir.decl_buffer((1,), "uint8", name="expected", scope="local.var")
    annotations = {"tl.dist.expected_signal_kind": tvm.tir.StringImm("sram_flagreg_inc")}

    def advance(value):
        return tvm.tir.Evaluate(tvm.tir.Call("handle", tvm.ir.Op.get("tl.dist_expect_"), [expected[0], value], annotations=annotations))

    if mode == "reset":
        body = tvm.tir.SeqStmt([advance(256), tvm.tir.BufferStore(expected, tvm.tir.const(0, "uint8"), [0]), advance(1)])
    elif mode == "read":
        body = tvm.tir.SeqStmt([advance(200), advance(tvm.tir.Cast("int32", expected[0]))])
    elif mode == "branch_read":
        body = tvm.tir.SeqStmt([tvm.tir.IfThenElse(rank > 0, advance(200), advance(100)), advance(tvm.tir.Cast("int32", expected[0]))])
    elif mode == "nested_loop":
        inner = tvm.tir.For(tvm.tir.Var("inner", "int32"), 0, tvm.tir.Cast("int32", expected[0]), tvm.tir.ForKind.SERIAL, advance(1))
        outer = tvm.tir.For(tvm.tir.Var("outer", "int32"), 0, 5000, tvm.tir.ForKind.SERIAL, inner)
        body = tvm.tir.SeqStmt([advance(1), outer])
    body = tvm.tir.DeclBuffer(expected, body)
    body = tvm.tir.Allocate(
        expected.data, "uint8", [1], tvm.tir.const(True), body, annotations={"tl.local_var_init": tvm.tir.const(0, "uint8")}
    )
    func = tvm.tir.PrimFunc([rank], body).with_attrs(
        {
            "target": determine_target("sunmmio", return_object=True),
            "tl.dist.world_size": 2,
            "tl.dist.rank_id_param_index": 0,
            "tl.dist.signal_counts": {"sram_flagreg_inc": 1},
        }
    )
    return tvm.IRModule({"main": func})


@pytest.mark.parametrize(
    "mode,group",
    [
        ("sequence", False),
        ("sequence", True),
        ("loop", False),
        ("while", False),
        ("members", True),
        ("branch", True),
        ("completion", True),
        ("auto_completion", True),
    ],
)
@pytest.mark.parametrize("amount", [255, 256])
def test_final_peak_is_checked_before_storage_cast(mode, group, amount):
    func = final_bound_kernel.get_tir(mode=mode, group=group, amount=amount, world_size=2)
    if amount == 256:
        with pytest.raises(tvm.error.InternalError, match="cumulative expected"):
            lower_to_device_tir(func)
    else:
        result = lower_to_device_tir(func, capture_passes="tl.LowerDistCommunication")
        markers = collect_calls(result.pass_snapshot("tl.LowerDistCommunication").mod)
        updates = [call for call in markers if call.op.name in ("tl.dist_expect_", "tl.dist_completion_init_")]
        assert updates and all(call.annotations["tl.dist.expected_signal_kind"].value == "sram_flagreg_inc" for call in updates)
        assert not collect_calls(result.device_mod, "tl.dist_expect_")


def test_communication_lowering_defers_the_limit_to_sync():
    mod = base_mod(final_bound_kernel.get_tir(amount=256, world_size=2))
    for transform in (
        tilelang.transform.InferSramScope(),
        tilelang.transform.PlanDistSignals(),
        tilelang.transform.LowerDistRouting(),
        tilelang.transform.LowerDistCommunication(),
    ):
        mod = transform(mod)
    assert collect_calls(mod, "tl.dist_expect_")
    with pytest.raises(tvm.error.InternalError, match="cumulative expected"):
        tilelang.transform.LowerDistSync()(mod)


def test_later_reset_does_not_hide_a_previous_peak():
    with pytest.raises(tvm.error.InternalError, match="cumulative expected"):
        tilelang.transform.LowerDistSync()(planned_expected_module())


@pytest.mark.parametrize("mode", ["read", "branch_read", "nested_loop"])
def test_later_expression_reads_the_latest_counter_bound(mode):
    with pytest.raises(tvm.error.InternalError, match="cumulative expected"):
        tilelang.transform.LowerDistSync()(planned_expected_module(mode))


@pytest.mark.parametrize("loop_delta", [0, 1])
def test_large_invariant_loop_uses_a_growth_summary(loop_delta):
    func = final_bound_kernel.get_tir(mode="loop", amount=5000, loop_delta=loop_delta, world_size=2)
    if loop_delta:
        with pytest.raises(tvm.error.InternalError, match="cumulative expected"):
            lower_to_device_tir(func)
    else:
        lower_to_device_tir(func)


def test_negative_delta_cannot_make_a_final_value_look_safe():
    with pytest.raises(tvm.error.InternalError, match="non-negative"):
        lower_to_device_tir(final_bound_kernel.get_tir(mode="nonmonotone", world_size=2))


@pytest.mark.parametrize("mode", ["while_unknown", "while_no_progress"])
def test_unproven_while_growth_is_rejected_with_a_bounded_analysis(mode):
    with pytest.raises(tvm.error.InternalError, match="cumulative expected.*while"):
        lower_to_device_tir(final_bound_kernel.get_tir(mode=mode, world_size=2))


def test_planned_updates_must_keep_their_signal_kind():
    def remove_kind(node):
        if isinstance(node, tvm.tir.Call) and node.op.name == "tl.dist_expect_":
            return tvm.tir.Call(node.dtype, node.op, node.args)

    mod = planned_expected_module()
    func = mod["main"]
    body = tvm.tir.stmt_functor.ir_transform(func.body, None, remove_kind, ["tir.Call"])
    with pytest.raises(tvm.error.InternalError, match="Missing expected signal kind"):
        tilelang.transform.LowerDistSync()(tvm.IRModule({"main": func.with_body(body)}))


@pytest.mark.parametrize("group", [False, True])
@pytest.mark.parametrize("maybe_zero", [False, True])
@pytest.mark.parametrize(
    "entry,first,second,valid", [(0, 200, 200, True), (1, 200, 200, True), (2, 127, 128, True), (2, 128, 128, False), (2, 200, 200, False)]
)
def test_for_expected_peak_uses_entry_bound_snapshot(group, maybe_zero, entry, first, second, valid):
    func = for_bound_snapshot_kernel.get_tir(entry=entry, first=first, second=second, group=group, maybe_zero=maybe_zero, world_size=2)
    if not valid:
        with pytest.raises(tvm.error.InternalError, match="cumulative expected"):
            lower_to_device_tir(func)
    else:
        result = lower_to_device_tir(func, capture_passes="tl.LowerDistCommunication")
        if entry:
            loops = collect_nodes(result.pass_snapshot("tl.LowerDistCommunication").mod, tvm.tir.For)
            assert any(isinstance(loop.extent, tvm.tir.BufferLoad) for loop in loops)


@pytest.mark.parametrize("group", [False, True])
def test_for_entry_snapshot_keeps_zero_iteration_constraints(group):
    lower_to_device_tir(
        for_bound_snapshot_kernel.get_tir(entry=1, first=200, group=group, maybe_zero=True, mode="correlated", world_size=2)
    )


@pytest.mark.parametrize("group", [False, True])
def test_for_unproven_entry_bound_is_rejected(group):
    with pytest.raises(tvm.error.InternalError, match="cumulative expected.*loop"):
        lower_to_device_tir(for_bound_snapshot_kernel.get_tir(group=group, mode="unknown", world_size=2))


@pytest.mark.parametrize("mode", ["cast", "add", "sub", "select"])
@pytest.mark.parametrize("group", [False, True])
@pytest.mark.parametrize("through_array", [False, True])
@pytest.mark.parametrize("tail", [0, 1])
def test_flag_bounds_cover_integer_casts_and_wrapping_arithmetic(mode, group, through_array, tail):
    func = integer_delta_kernel.get_tir(mode=mode, group=group, through_array=through_array, tail=tail, world_size=2)
    if tail:
        with pytest.raises(tvm.error.InternalError, match="cumulative expected"):
            lower_to_device_tir(func)
    else:
        lower_to_device_tir(func)


@pytest.mark.parametrize("group", [False, True])
@pytest.mark.parametrize("tail", [0, 1])
def test_for_entry_bound_cast_cannot_hide_flag_overflow(group, tail):
    func = integer_delta_kernel.get_tir(mode="loop", group=group, tail=tail, world_size=2)
    if tail:
        with pytest.raises(tvm.error.InternalError, match="cumulative expected"):
            lower_to_device_tir(func)
    else:
        lower_to_device_tir(func)
