"""Device-index representability from original typed loop endpoints and guards."""

import pytest
import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.codegen.test_dist_semantics_opt_validate import generate
from testing.python.sunmmio.common.suvm_trace import trace_suvm_control_flow


@tilelang.jit(target="sunmmio")
def range_kernel(mode, width=8, unsigned_step=False, world_size: int = 1):
    dtype = T.int64 if mode == "bounded" else T.int32 if mode == "stride" else getattr(T, f"uint{width}")
    stride_type = T.uint64 if unsigned_step else T.int64
    high = tvm.tir.const((1 << width) - 6, dtype) if mode in ("left", "right") else 0

    @T.prim_func
    def main(rank_id: T.dist.RankId, a: dtype, b: dtype, stride: stride_type):
        with T.Kernel():
            s = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            for _ in T.serial(
                high if mode == "left" else a % dtype(8) if mode == "bounded" else a,
                high if mode == "right" else b % dtype(8) if mode == "bounded" else b,
                stride if mode == "stride" else None,
            ):
                T.dist.put_signal(s, (rank_id + 1) % world_size, submit=True)

    return main


@tilelang.jit(target="sunmmio")
def quotient_guard_kernel(mode, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, n: T.uint64):
        with T.Kernel():
            src = T.alloc_shared((4, 32), T.int32)
            dst = T.alloc_shared((1, 32), T.int32)
            s = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            T.fill(src, 1)
            if mode == "floor":
                if n // T.uint64(4) == T.uint64(0):
                    T.dist.put(src[n : n + 1, :], dst, (rank_id + 1) % world_size, signal=s, submit=True)
            elif mode == "trunc":
                if T.truncdiv(n, T.uint64(4)) != T.uint64(0):
                    T.dist.put_signal(s, (rank_id + 1) % world_size, submit=True)
                else:
                    T.dist.put(src[n : n + 1, :], dst, (rank_id + 1) % world_size, signal=s, submit=True)
            else:
                if T.truncdiv(n, T.uint64(4)) == T.uint64(0):
                    T.dist.put_signal(s, (rank_id + 1) % world_size, submit=True)
                else:
                    T.dist.put(src[n : n + 1, :], dst, (rank_id + 1) % world_size, signal=s, submit=True)

    return main


@pytest.mark.parametrize(
    "mode,width", [("bounded", 8), ("stride", 8), *((side, bits) for side in ("left", "right") for bits in (8, 16, 32))]
)
def test_range_accepts_representable_counts(tmp_path, mode, width):
    result, src = generate(range_kernel, tmp_path, mode=mode, width=width)
    high = (1 << width) - 6
    a, b = (9, 15) if mode == "bounded" else (-3, 6) if mode == "stride" else (high - 2, high + 3)
    batches, _ = trace_suvm_control_flow(result, src, rank_id=0, a=a, b=b, stride=2)
    expected = len(range(a % 8, b % 8)) if mode == "bounded" else len(range(a, b, 2)) if mode == "stride" else 3 if mode == "left" else 2
    assert len(batches) == expected


def test_unsigned_wide_stride(tmp_path):
    r, s = generate(range_kernel, tmp_path, mode="stride", unsigned_step=True)
    for step in (2, 1 << 63):
        batches, _ = trace_suvm_control_flow(r, s, rank_id=0, a=0, b=5, stride=step)
        assert len(batches) == len(range(0, 5, step))


@pytest.mark.parametrize("mode", ["floor", "trunc", "unsafe"])
def test_partition_guard_in_real_else(tmp_path, mode):
    if mode == "unsafe":
        with pytest.raises(tvm.error.InternalError, match="without loss"):
            generate(quotient_guard_kernel, tmp_path, mode=mode)
    else:
        generate(quotient_guard_kernel, tmp_path, mode=mode)


def test_uint64_high_constant_only_bounds_one_direction(tmp_path):
    r, s = generate(range_kernel, tmp_path, mode="left", width=64)
    high = (1 << 64) - 6
    for stop, expected in ((high - 1, 0), (high, 0), (high + 5, 5)):
        batches, _ = trace_suvm_control_flow(r, s, rank_id=0, a=0, b=stop, stride=1)
        assert len(batches) == expected
    with pytest.raises(tvm.error.InternalError, match="trip count must fit"):
        generate(range_kernel, tmp_path, mode="right", width=64)
