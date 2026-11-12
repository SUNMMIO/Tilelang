"""Communication array parameters require local RSRAM; copies are explicit."""

import pytest
import tilelang
import tilelang.language as T
from tilelang import tvm

from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import base_mod, collect_calls


@tilelang.jit(target="sunmmio")
def auxiliary_kernel(api="wait_signals", scope="shared", copied=False, offset=0, world_size: int = 1):
    @T.prim_func
    def main(remote: T.MeshTensor((32,), T.placement.replicated(), T.uint32), rank_id: T.dist.RankId):
        with T.Kernel():
            local = T.alloc_shared((32,), T.uint32, scope="shared" if scope == "global" else scope)
            other_counts = T.alloc_shared((32,), T.uint32)
            T.fill(local, 1)
            T.fill(other_counts, 1)
            if copied:
                T.copy(remote, local)
            if api == "wait_signals":
                signals = T.dist.signals(world_size, kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
                T.dist.wait_signals(
                    signals,
                    expected_deltas=remote[offset : offset + world_size]
                    if scope == "global" and not copied
                    else local[offset : offset + world_size],
                )
            elif api == "completion":
                signals = T.dist.signals(world_size, kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
                done = T.dist.completion(
                    signals,
                    expected_deltas=remote[offset : offset + world_size]
                    if scope == "global" and not copied
                    else local[offset : offset + world_size],
                )
                T.dist.wait_all(done)
            else:
                source = T.alloc_shared((world_size, 4, 32), T.bfloat16)
                destination = T.alloc_shared((world_size, 4, 32), T.bfloat16)
                signals = T.dist.signals(world_size)
                done = T.dist.all_to_allv(
                    source,
                    destination,
                    send_counts=(
                        remote[offset : offset + world_size] if scope == "global" and not copied else local[offset : offset + world_size]
                    )
                    if api == "send_counts"
                    else other_counts[offset : offset + world_size],
                    recv_counts=(
                        remote[offset : offset + world_size] if scope == "global" and not copied else local[offset : offset + world_size]
                    )
                    if api == "recv_counts"
                    else other_counts[offset : offset + world_size],
                    signals=signals,
                    submit=True,
                )
                T.dist.wait_all(done)

    return main


@pytest.mark.parametrize("api", ["wait_signals", "completion", "send_counts", "recv_counts"])
def test_auxiliary_dram_array_is_rejected_with_explicit_copy_hint(api):
    with pytest.raises(tvm.error.InternalError, match="(expected_deltas|send_counts|recv_counts).*shared.rsram.*T.copy"):
        lower_to_device_tir(auxiliary_kernel.get_tir(api, "global", world_size=2))


@pytest.mark.parametrize("api", ["wait_signals", "completion", "send_counts", "recv_counts"])
@pytest.mark.parametrize("scope", ["shared", "shared.dyn", "shared.rsram"])
def test_auxiliary_shared_array_resolves_to_rsram(api, scope):
    result = lower_to_device_tir(auxiliary_kernel.get_tir(api, scope, offset=1, world_size=2), capture_passes="tl.PlanDistSignals")
    planned = result.pass_snapshot("tl.PlanDistSignals").mod
    if api == "wait_signals":
        args = collect_calls(planned, "tl.dist_wait_signals")[0].args
    else:
        args = collect_calls(planned, "tl.dist_completion")[0].args
    payload = args[1]
    load = payload if isinstance(payload, tvm.tir.BufferLoad) else payload.args[0]
    assert load.buffer.scope() == "shared.rsram"
    index = load.indices[0]
    assert int(index.base if isinstance(index, tvm.tir.Ramp) else index) == 1


@pytest.mark.parametrize("api", ["wait_signals", "completion", "send_counts", "recv_counts"])
def test_auxiliary_explicit_copy_accepts_local_region(api):
    result = lower_to_device_tir(auxiliary_kernel.get_tir(api, "global", copied=True, offset=1, world_size=2))
    assert collect_calls(result.device_mod, "tl.dma_copy")


@pytest.mark.parametrize("api", ["wait_signals", "completion", "send_counts", "recv_counts"])
@pytest.mark.parametrize("scope", ["shared.asram", "shared.wsram", "local"])
def test_auxiliary_other_scopes_are_rejected_by_owning_op_check(api, scope):
    mod = base_mod(auxiliary_kernel.get_tir(api, scope, world_size=2))
    # Exercise the parameter check directly for explicit spaces; inference
    # can reject the ordinary reads from these spaces even earlier.
    transform = tilelang.transform.PlanDistSignals() if api in ("wait_signals", "completion") else tilelang.transform.LowerDistCollectives()
    with pytest.raises(tvm.error.InternalError, match="(expected_deltas|send_counts|recv_counts).*shared.rsram.*T.copy"):
        transform(mod)


@tilelang.jit(target="sunmmio")
def scalar_delta_kernel(world_size: int = 1):
    @T.prim_func
    def main(delta: T.uint32, rank_id: T.dist.RankId):
        with T.Kernel():
            signal = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            T.dist.wait_signal(signal, expected_delta=delta)

    return main


def test_scalar_expected_delta_does_not_require_array_scope():
    result = lower_to_device_tir(scalar_delta_kernel.get_tir(world_size=2))
    assert collect_calls(result.device_mod, "tl.dist_wait_signal_")
