"""Expected bounds survive payload branches correlated with earlier deltas."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm

from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import collect_calls


@tilelang.jit(target="sunmmio")
def correlated_payload_kernel(tail_delta=254, world_size: int = 1):
    @T.prim_func
    def main(
        input_counts: T.MeshTensor((2, 1, 2), T.placement.replicated(), T.int32),
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            counts = T.alloc_shared((2, 1, 2), T.int32)
            payload = T.alloc_shared((32, 32), T.float32)
            ready = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_INC, expect="manual")
            delta = T.alloc_var(T.uint32, init=0)
            T.copy(input_counts, counts)
            if counts[0, 0, 0] > 0:
                delta += 1
            T.dist.wait_signal(ready, expected_delta=delta)

            # These branches share the count used above, but only write payload.
            # The expected bound must remain [0, 1] after every branch join.
            for item in T.serial(32):
                if item < counts[0, 0, 0]:
                    for hidden in T.serial(32):
                        payload[item, hidden] = T.float32(1)
            T.dist.wait_signal(ready, expected_delta=tail_delta)

    return main


@pytest.mark.parametrize("tail_delta,valid", [(254, True), (255, False)])
def test_payload_joins_preserve_cumulative_expected_bound(tail_delta, valid):
    func = correlated_payload_kernel.get_tir(tail_delta=tail_delta, world_size=2)
    if valid:
        result = lower_to_device_tir(func)
        assert len(collect_calls(result.device_mod, "tl.dist_wait_signal_")) == 2
    else:
        with pytest.raises(tvm.error.InternalError, match="cumulative expected"):
            lower_to_device_tir(func)
