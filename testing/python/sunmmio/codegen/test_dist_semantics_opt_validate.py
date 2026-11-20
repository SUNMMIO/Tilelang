"""Cross-pass semantic regressions, compiled from the public SunMMIO DSL."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.common.codegen_validation import codegen_sunmmio_suvm_mlir, validate_suvm_mlir_with_npuir_opt
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


def generate(kernel, tmp_path, **kwargs):
    result = lower_to_device_tir(kernel.get_tir(world_size=2, **kwargs))
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
