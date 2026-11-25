"""Cross-pass semantic regressions, compiled from the public SunMMIO DSL."""

import functools
import re

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.common.codegen_validation import (
    codegen_sunmmio_suvm_mlir,
    simplify_suvm_mlir,
    validate_suvm_mlir_with_npuir_opt,
)
from testing.python.sunmmio.common.suvm_trace import trace_suvm_control_flow
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir

tilelang.env.disable_cache()


selectors = (T.if_then_else,)
partial_choice = functools.partial(T.if_then_else)
partial_script_choice = functools.partial(tvm.script.ir_builder.tir.if_then_else)


@T.macro
def wait_member_alias(c):
    member = T.dist.wait_any(c)
    return member


def plain_next_or_none(c):
    return T.if_then_else(T.dist.has_pending(c), wait_member_alias(c), -1)


@T.macro
def nested_member(c):
    return wait_member_alias(c)


def plain_nested_wait(c):
    return T.if_then_else(T.dist.has_pending(c), nested_member(c), -1)


@T.macro
def checked_next_or_none(c):
    member = T.alloc_var(T.int32, init=-1)
    if T.dist.has_pending(c):
        member = nested_member(c)
    return T.Cast("int32", member)


def plain_saved_member(used, member):
    return T.if_then_else(used, member, -1)


@tilelang.jit(target="sunmmio")
def helper_wait_kernel(mode, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, n: T.uint32):
        with T.Kernel():
            group = T.dist.signals(1, kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            delta = T.alloc_shared((16,), T.uint32)
            T.fill(delta, n % T.uint32(2))
            if n % T.uint32(2) != 0:
                T.dist.put_signal(group[0], (rank_id + 1) % world_size, submit=True)
            done = T.dist.completion(group, expected_deltas=delta[0:1])
            result = T.alloc_var(T.int32, init=-1)
            if mode == "plain":
                result = plain_next_or_none(done)
            elif mode == "nested":
                result = plain_nested_wait(done)
            elif mode == "checked":
                result = checked_next_or_none(done)
            else:
                active = T.dist.has_pending(done)
                if active:
                    saved = nested_member(done)
                    result = plain_saved_member(active, saved)
            assert T.Cast("uint32", result + 1) == n % T.uint32(2), "wrong member"
            T.dist.wait_all(done)

    return main


@tilelang.jit(target="sunmmio")
def consuming_wait_kernel(mode, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, n: T.uint32):
        with T.Kernel():
            group = T.dist.signals(1, kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            result = T.alloc_var(T.int32, init=-1)
            delta = T.alloc_shared((16,), T.uint32)
            T.fill(delta, n % T.uint32(2))
            for _ in T.serial(T.Cast("int32", n % T.uint32(2))):
                T.dist.put_signal(group[0], (rank_id + 1) % world_size, submit=True)
            done = T.dist.completion(group, expected_deltas=delta[0:1])
            if mode == "intrinsic":
                j = T.if_then_else(T.dist.has_pending(done), T.dist.wait_any(done), -1)
                result = j
            elif mode == "alias_macro":
                choose = T.if_then_else
                result = choose(T.dist.has_pending(done), wait_member_alias(done), -1)
            elif mode == "computed":
                result = selectors[0](T.dist.has_pending(done), wait_member_alias(done), -1)
            elif mode == "star":
                result = T.if_then_else(*(T.dist.has_pending(done), wait_member_alias(done), -1))
            elif mode == "mapping":
                result = T.if_then_else(**dict(cond=T.dist.has_pending(done), t=wait_member_alias(done), f=-1))
            elif mode == "ternary":
                result = T.dist.wait_any(done) if T.dist.has_pending(done) else -1
            elif mode == "short":
                result = T.Cast("int32", T.dist.has_pending(done) and T.dist.wait_any(done) == 0)
            else:
                if T.dist.has_pending(done):
                    if mode == "assignment":
                        result = T.dist.wait_any(done)
                    else:
                        j = T.dist.wait_any(done)
                        result = j
            assert T.Cast("uint32", result + 1) == n % T.uint32(2), "wrong member"
            T.dist.wait_all(done)

    return main


@tilelang.jit(target="sunmmio")
def partial_wait_kernel(binding, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            group = T.dist.signals(1, kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            T.dist.put_signal(group[0], (rank_id + 1) % world_size, submit=True)
            delta = T.alloc_shared((16,), T.uint32)
            T.fill(delta, T.uint32(1))
            done = T.dist.completion(group, expected_deltas=delta[0:1])
            if binding == "unbound":
                value = partial_choice(T.dist.has_pending(done), wait_member_alias(done), -1)
            elif binding == "call":
                value = partial_choice.__call__(T.dist.has_pending(done), wait_member_alias(done), -1)
            elif binding == "script":
                value = partial_script_choice(T.dist.has_pending(done), wait_member_alias(done), -1)
            elif binding == "cond":
                choose = functools.partial(partial_choice, T.dist.has_pending(done))
                value = choose(wait_member_alias(done), -1)
            elif binding == "cond_true":
                choose = functools.partial(T.if_then_else, T.dist.has_pending(done), -1)
                value = choose(wait_member_alias(done))
            elif binding == "hole":
                choose = functools.partial(T.if_then_else, T.dist.has_pending(done), functools.Placeholder, -1)
                value = choose(wait_member_alias(done))
            elif binding == "keyword":
                choose = functools.partial(T.if_then_else, cond=T.dist.has_pending(done), f=-1)
                value = choose(t=wait_member_alias(done))
            else:
                choose = functools.partial(T.if_then_else, t=1, f=0)
                value = choose(wait_member_alias(done) == 0)
            assert value == 1, "expected ready member"
            T.dist.wait_all(done)

    return main


@T.macro
def read_member(buf, i):
    value = buf[i]
    return value


@tilelang.jit(target="sunmmio")
def lazy_read_kernel(bound=False, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, i: T.int32):
        with T.Kernel():
            s = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            buf = T.alloc_shared((16,), T.uint32)
            T.fill(buf, T.uint32(7))
            if bound:
                choose = functools.partial(partial_choice, i >= 0, f=T.uint32(0))
                value = choose(read_member(buf, i))
            else:
                value = selectors[0](i >= 0, read_member(buf, i), T.uint32(0))
            T.dist.wait_signal(s, expected_delta=value)

    return main


@tilelang.jit(target="sunmmio")
def ordinary_route_kernel(mode, manual, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel() as core:
            src = T.alloc_shared((256,), T.int32)
            dst = T.alloc_shared((256,), T.int32)
            signal = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_INC, expect="manual" if manual else "auto")
            T.fill(src, 1)
            if mode == "ordinary_local":
                if core // T.mesh_ncols() == 0:
                    T.dist.put(src, dst, rank_id, dst_row=0, signal=signal)
            elif mode == "ordinary_cross":
                T.dist.put(src, dst, rank_id, dst_row=1 - (core // T.mesh_ncols()) % 2, signal=signal)
            elif mode == "ordinary_iv":
                for i in T.serial(1, world_size):
                    if rank_id == i:
                        T.dist.put(src, dst, rank_id - i, signal=signal)
            T.dist.submit()
            if manual:
                T.dist.wait_signal(signal, expected_delta=0)
            else:
                T.dist.wait_signal(signal)

    return main


@tilelang.jit(target="sunmmio")
def nested_init_kernel(mode="literal", world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, m: T.int32, z: T.int32, rounds: T.int32):
        with T.Kernel():
            s = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            for _ in T.serial(rounds):
                j = T.alloc_var(T.int32, init=z if mode == "runtime" else T.int32(0) if mode == "typed" else 0)
                if mode == "reset":
                    j = 0
                while j < m:
                    T.dist.put_signal(s, (rank_id + 1) % world_size, submit=True)
                    j += 1

    return main


@tilelang.jit(target="sunmmio")
def topology_kernel(mode, manual=False, kind=T.dist.SignalKind.SRAM_MEMORY, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel() as core:
            src = T.alloc_shared((256,), T.int32)
            dst = T.alloc_shared((256,), T.int32)
            signal = T.dist.signal(kind=kind, expect="manual" if manual else "auto")
            T.fill(src, 1)
            # Independent compile-time mode checks keep the generated DSL blocks shallow.
            if mode == "self":
                T.dist.put_signal(signal, rank_id)
            if mode == "partial":
                T.dist.put_signal(signal, 0)
            if mode == "column" and rank_id < 2 and core % T.mesh_ncols() == rank_id:
                T.dist.put_signal(signal, 2)
            if mode == "iv_source":
                for i in T.serial(1, 3):
                    if rank_id == i:
                        T.dist.put_signal(signal, 0)
            if mode == "two_senders" and rank_id != 2:
                T.dist.put_signal(signal, 2)
            if mode == "ring":
                for i in T.serial(1, world_size):
                    T.dist.put_signal(signal, (rank_id + i) % world_size)
            if mode == "local_guard" and core // T.mesh_ncols() == 0:
                T.dist.routed_put(src, dst, routes=[[0, rank_id, 1]], signal=signal)
            if mode == "local":
                T.dist.routed_put(src, dst, routes=[[0, rank_id, 1]], signal=signal)
            if mode == "rank_guard" and rank_id == 1:
                T.dist.routed_put(src, dst, routes=[[0, rank_id - 1, 0]], signal=signal)
            if mode == "inactive":
                T.dist.routed_put(src, dst, src_rank=1, routes=[[0, rank_id - 1, 0]], signal=signal)
            if mode == "mixed":
                T.dist.routed_put(src, dst, routes=[[0, 1 - rank_id, rank_id]], signal=signal)
            T.dist.submit()
            if manual:
                T.dist.wait_signal(signal, expected_delta=0)
            else:
                T.dist.wait_signal(signal)

    return main


@T.macro
def count_in_scope(m):
    j = T.alloc_var(T.int32, init=0)
    while j < m:
        j += 1
    return T.Cast("int32", j)


@tilelang.jit(target="sunmmio")
def scoped_macro_kernel(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, rounds: T.int32, m: T.int32, skip: T.int32):
        with T.Kernel():
            signal = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            total = T.alloc_var(T.int32, init=0)
            for i in T.serial(rounds):
                if i != skip:
                    total += count_in_scope(m)
                T.dist.wait_signal(signal, expected_delta=T.Cast("uint32", total))

    return main


def generate(kernel, tmp_path, world_size=2, **kwargs):
    result = lower_to_device_tir(kernel.get_tir(world_size=world_size, **kwargs))
    source = codegen_sunmmio_suvm_mlir(result.device_mod)
    validate_suvm_mlir_with_npuir_opt(source, tmp_path, opt_args=("--verify-each", "--suvm-device-validate"))
    return result, source


@pytest.mark.parametrize("mode", ["intrinsic", "alias_macro", "computed", "star", "mapping", "ternary", "short"])
def test_wait_any_rejects_runtime_lazy_operand(mode):
    with pytest.raises((ValueError, tvm.error.InternalError), match="statement branch"):
        lower_to_device_tir(consuming_wait_kernel.get_tir(mode, world_size=2))


@pytest.mark.parametrize(
    "binding",
    [
        "unbound",
        "call",
        "script",
        "cond",
        "cond_true",
        "keyword",
        pytest.param("hole", marks=pytest.mark.skipif(not hasattr(functools, "Placeholder"), reason="requires Python 3.14")),
    ],
)
def test_partial_wait_any_in_lazy_operand_is_rejected_at_construction(binding):
    with pytest.raises(ValueError, match="wait_any must be consumed in a statement branch"):
        partial_wait_kernel.get_tir(binding, world_size=2)


def test_partial_wait_any_in_eager_condition_is_consumed_once(tmp_path):
    result, source = generate(partial_wait_kernel, tmp_path, binding="eager_condition")
    _, events = trace_suvm_control_flow(result, source, rank_id=0, signal_values={0: 1}, max_loop_iterations=4)
    assert sum(event[0] == "fetch" for event in events) == 1


@pytest.mark.parametrize("mode", ["plain", "nested"])
def test_plain_helper_cannot_hoist_macro_wait(mode):
    with pytest.raises(ValueError, match=r"wait_any.*ordinary Python helper.*@T.macro"):
        helper_wait_kernel.get_tir(mode, world_size=2)


@pytest.mark.parametrize("mode", ["saved", "checked"])
def test_explicit_helper_wait_has_statement_semantics(tmp_path, mode):
    result, source = generate(helper_wait_kernel, tmp_path, mode=mode)
    for rank in (0, 1):
        for n in (0, 1):
            _, events = trace_suvm_control_flow(result, source, rank_id=rank, n=n, signal_values={0: 1}, max_loop_iterations=4)
            assert sum(event[0] == "fetch" for event in events) == n


@pytest.mark.parametrize("mode", ["statement", "assignment"])
def test_wait_any_consumes_once_in_taken_branch(tmp_path, mode):
    r, src = generate(consuming_wait_kernel, tmp_path, mode=mode)
    for n in (0, 1):
        _, events = trace_suvm_control_flow(r, src, rank_id=0, n=n, signal_values={0: 1}, max_loop_iterations=4)
        assert sum(e[0] == "fetch" for e in events) == n


@pytest.mark.parametrize("mode", ["literal", "typed", "runtime", "reset"])
def test_local_initializer_executes_each_declaration(tmp_path, mode):
    r, src = generate(nested_init_kernel, tmp_path, mode=mode)
    for rounds in (0, 1, 2):
        for m in (0, 1, 2):
            batches, _ = trace_suvm_control_flow(r, src, rank_id=0, rounds=rounds, m=m, z=0)
            assert [x[3] for batch in batches for x in batch if x[0] == "signal"] == list(range(1, rounds * m + 1))


@pytest.mark.parametrize("mode", ["self", "partial"])
@pytest.mark.parametrize("manual", [True, False])
def test_self_notify_is_rejected_before_expected_planning(mode, manual):
    with pytest.raises(tvm.error.InternalError, match="remote peer Rank"):
        lower_to_device_tir(topology_kernel.get_tir(mode, manual=manual, world_size=2))


def test_local_routed_barrier_requires_both_rows(tmp_path):
    with pytest.raises(tvm.error.InternalError, match="uniform across rows"):
        lower_to_device_tir(topology_kernel.get_tir("local_guard", manual=True, world_size=2))
    _, source = generate(topology_kernel, tmp_path, mode="local", manual=True)
    for col in range(4):
        mask = (1 << col) | (1 << (4 + col))
        for row in (0, 1):
            core = 4 * row + col
            fixed = simplify_suvm_mlir(
                source.replace("suvm.get_core_id : i64", f"arith.constant {core} : i64"),
                opt_args=("--canonicalize", "--cse", "--sccp", "--canonicalize"),
            )
            barrier = re.search(rf"(%[\w-]+) = suvm.barrier.init mask = {mask} :", fixed)[1]
            arrivals = f"suvm.barrier.arrive_and_wait {barrier} :"
            assert fixed.count(arrivals) == 2
            if row == 0:
                assert fixed.index(arrivals) < fixed.index("suvm.mcast_async") < fixed.rindex(arrivals)


@pytest.mark.parametrize("mode", ["inactive", "rank_guard", "mixed"])
def test_routed_endpoint_is_classified_per_participating_rank(tmp_path, mode):
    generate(topology_kernel, tmp_path, mode=mode)


@pytest.mark.parametrize("manual", [False, True])
def test_static_signal_peer_loop(tmp_path, manual):
    r, source = generate(topology_kernel, tmp_path, mode="ring", world_size=3, manual=manual, kind=T.dist.SignalKind.SRAM_FLAGREG_INC)
    for rank in range(3):
        batches, events = trace_suvm_control_flow(r, source, rank_id=rank)
        assert [x[1] for batch in batches for x in batch if x[0] == "signal"] == [(rank + 1) % 3, (rank + 2) % 3]
        assert [x[1] for x in events if x[0] == "wait"] == [0 if manual else 2]


@pytest.mark.parametrize("kind", [T.dist.SignalKind.SRAM_MEMORY, T.dist.SignalKind.SRAM_FLAGREG_VALUE, T.dist.SignalKind.SRAM_FLAGREG_INC])
def test_two_remote_senders_are_checked_per_member(tmp_path, kind):
    opts = dict(mode="two_senders", manual=True, kind=kind, world_size=3)
    if kind == T.dist.SignalKind.SRAM_FLAGREG_INC:
        generate(topology_kernel, tmp_path, **opts)
    else:
        with pytest.raises(tvm.error.InternalError, match="sender"):
            lower_to_device_tir(topology_kernel.get_tir(**opts))


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("mode", ["column", "iv_source"])
def test_static_source_filters_drive_expected_and_uniqueness(tmp_path, manual, mode):
    kind = T.dist.SignalKind.SRAM_MEMORY if mode == "column" else T.dist.SignalKind.SRAM_FLAGREG_INC
    result, source = generate(topology_kernel, tmp_path, world_size=3, mode=mode, manual=manual, kind=kind)
    for col in range(4):
        specialized = source.replace("suvm.get_core_id : i64", f"arith.constant {col} : i64")
        for rank in range(3):
            batches, events = trace_suvm_control_flow(result, specialized, rank_id=rank)
            peers = [x[1] for batch in batches for x in batch if x[0] == "signal"]
            if mode == "column":
                assert peers == ([2] if rank < 2 and col == rank else [])
                target = int(rank == 2 and col < 2)
            else:
                assert peers == ([0] if rank in (1, 2) else [])
                target = 2 if rank == 0 else 0
            assert [x[1] for x in events if x[0] == "wait"] == [0 if manual else target]


def test_macro_local_reset_and_outer_lifetime(tmp_path):
    result, src = generate(scoped_macro_kernel, tmp_path)
    for rounds in (0, 1, 3):
        for skip in (0, 1, 3):
            _, events = trace_suvm_control_flow(result, src, rank_id=0, rounds=rounds, m=2, skip=skip)
            total = target = 0
            expected = []
            for i in range(rounds):
                if i != skip:
                    total += 2
                target += total
                expected.append(target)
            assert [e[1] for e in events if e[0] == "wait"] == expected


@pytest.mark.parametrize("bound", [False, True])
def test_read_macro_remains_in_lazy_operand(tmp_path, bound):
    result, source = generate(lazy_read_kernel, tmp_path, bound=bound)
    for index, expected in ((-1, 0), (0, 7)):
        _, events = trace_suvm_control_flow(result, source, rank_id=0, i=index)
        assert [event[1] for event in events if event[0] == "wait"] == [expected]


@pytest.mark.parametrize("manual", [True, False])
def test_ordinary_local_put_uses_only_participating_sources(tmp_path, manual):
    _, source = generate(ordinary_route_kernel, tmp_path, mode="ordinary_local", manual=manual)
    assert "suvm.barrier.arrive_and_wait" not in source
    assert "suvm.dist.batch_put" not in source
    for row in range(4):
        fixed = simplify_suvm_mlir(
            source.replace("suvm.get_core_id : i64", f"arith.constant {4 * row} : i64"),
            opt_args=("--canonicalize", "--cse", "--sccp", "--canonicalize"),
        )
        assert ("suvm.tile.load" in fixed) == (row == 0)


@pytest.mark.parametrize("manual", [True, False])
def test_ordinary_put_validates_only_active_rank_loop_pairs(tmp_path, manual):
    result, source = generate(ordinary_route_kernel, tmp_path, world_size=3, mode="ordinary_iv", manual=manual)
    for rank in range(3):
        batches, events = trace_suvm_control_flow(result, source, rank_id=rank)
        assert [x[1] for batch in batches for x in batch if x[0] == "data"] == ([0] if rank else [])
        assert [e[1] for e in events if e[0] == "wait"] == [0 if manual or rank else 2]


def test_ordinary_cross_row_put_keeps_matching_participants(tmp_path):
    _, source = generate(ordinary_route_kernel, tmp_path, mode="ordinary_cross", manual=True)
    for col in range(4):
        expected_arrivals = {}
        for row in range(4):
            dst_row = 1 - row % 2
            mask = (1 << (4 * row + col)) | (1 << (4 * dst_row + col))
            expected_arrivals[mask] = expected_arrivals.get(mask, 0) + 2
        for row in range(4):
            core = 4 * row + col
            fixed = simplify_suvm_mlir(
                source.replace("suvm.get_core_id : i64", f"arith.constant {core} : i64"),
                opt_args=("--canonicalize", "--cse", "--sccp", "--canonicalize"),
            )
            for mask, count in expected_arrivals.items():
                if mask & (1 << core):
                    barriers = re.findall(rf"(%[\w-]+) = suvm.barrier.init mask = {mask} :", fixed)
                    assert sum(fixed.count(f"suvm.barrier.arrive_and_wait {b} :") for b in barriers) == count
