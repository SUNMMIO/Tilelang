"""Cross-pass semantic regressions, compiled from the public SunMMIO DSL."""

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


@T.macro
def wait_member_alias(c):
    member = T.dist.wait_any(c)
    return member


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
            if mode == "self":
                T.dist.put_signal(signal, rank_id)
            elif mode == "partial":
                T.dist.put_signal(signal, 0)
            elif mode == "column":
                if rank_id < 2 and core % T.mesh_ncols() == rank_id:
                    T.dist.put_signal(signal, 2)
            elif mode == "iv_source":
                for i in T.serial(1, 3):
                    if rank_id == i:
                        T.dist.put_signal(signal, 0)
            elif mode == "two_senders":
                if rank_id != 2:
                    T.dist.put_signal(signal, 2)
            elif mode == "ring":
                for i in T.serial(1, world_size):
                    T.dist.put_signal(signal, (rank_id + i) % world_size)
            elif mode == "local_guard":
                if core // T.mesh_ncols() == 0:
                    T.dist.routed_put(src, dst, routes=[[0, rank_id, 1]], signal=signal)
            elif mode == "local":
                T.dist.routed_put(src, dst, routes=[[0, rank_id, 1]], signal=signal)
            elif mode == "rank_guard":
                if rank_id == 1:
                    T.dist.routed_put(src, dst, routes=[[0, rank_id - 1, 0]], signal=signal)
            elif mode == "inactive":
                T.dist.routed_put(src, dst, src_rank=1, routes=[[0, rank_id - 1, 0]], signal=signal)
            else:
                T.dist.routed_put(src, dst, routes=[[0, 1 - rank_id, rank_id]], signal=signal)
            T.dist.submit()
            if manual:
                T.dist.wait_signal(signal, expected_delta=0)
            else:
                T.dist.wait_signal(signal)

    return main


def generate(kernel, tmp_path, world_size=2, **kwargs):
    result = lower_to_device_tir(kernel.get_tir(world_size=world_size, **kwargs))
    source = codegen_sunmmio_suvm_mlir(result.device_mod)
    validate_suvm_mlir_with_npuir_opt(source, tmp_path, opt_args=("--verify-each", "--suvm-device-validate"))
    return result, source


@pytest.mark.parametrize("mode", ["intrinsic", "alias_macro", "ternary", "short"])
def test_wait_any_rejects_runtime_lazy_operand(mode):
    with pytest.raises((ValueError, tvm.error.InternalError), match="statement branch"):
        lower_to_device_tir(consuming_wait_kernel.get_tir(mode, world_size=2))


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
