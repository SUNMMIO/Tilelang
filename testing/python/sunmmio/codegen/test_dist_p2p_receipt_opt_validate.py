"""Two-rank execution handoff: device assertions check data and both guard regions."""

import pytest
import tilelang
import tilelang.language as T
from testing.python.sunmmio.codegen.test_dist_semantics_opt_validate import generate


@tilelang.jit(target="sunmmio")
def p2p_receipt_kernel(rounds=2, world_size: int = 2):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((512,), T.int32)
            window = T.alloc_shared((768,), T.int32)
            data = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            ack = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            if rank_id == 0:
                for lane in T.Parallel(512):
                    src[lane] = 1024 + lane
                T.dist.wait_signal(ack, expected_delta=1)
            else:
                T.fill(window, -1)
                assert window[256] == -1, "receive slot initialization"
                T.dist.put_signal(ack, 0, submit=True)
            for it in T.serial(rounds):
                if rank_id == 0:
                    T.dist.put(src[it * 256 : it * 256 + 256], window[256:512], 1, signal=data, submit=True)
                    T.dist.wait_signal(ack, expected_delta=1)
                else:
                    T.dist.wait_signal(data, expected_delta=1)
                    for lane in T.serial(256):
                        assert window[256 + lane] == 1024 + it * 256 + lane, "incorrect received payload"
                        assert window[lane] == -1 and window[512 + lane] == -1, "receive slot guard overwritten"
                    T.dist.put_signal(ack, 0, submit=True)

    return main


@pytest.mark.parametrize("rounds", [1, 2])
def test_two_rank_payload_and_guard_fixture_compiles(tmp_path, rounds):
    # The assertions below check the generated receiver path. Running the
    # kernel on two ranks will check every payload element and both guards.
    _, source = generate(p2p_receipt_kernel, tmp_path, rounds=rounds)
    receiver_wait = source.rindex("suvm.dist.signal_wait")
    payload_check = source.index("incorrect received payload")
    guard_check = source.index("receive slot guard overwritten")
    ack = source.rindex("suvm.dist.batch_signal")
    assert source.index("suvm.dist.batch_put") < receiver_wait < payload_check < guard_check < ack
