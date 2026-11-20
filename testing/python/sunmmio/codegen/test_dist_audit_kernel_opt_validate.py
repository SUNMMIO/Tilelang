"""Audit regressions from public kernels through the production pipeline."""

import pytest
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


def generate(factory, tmp_path, **kwargs):
    kernel = factory.get_tir(world_size=2, **kwargs)
    result = lower_to_device_tir(kernel, capture_passes=("tl.LowerDistSync",))
    source = codegen_sunmmio_suvm_mlir(result.device_mod)
    validate_suvm_mlir_with_npuir_opt(source, tmp_path, opt_args=("--verify-each", "--suvm-device-validate"))
    return result, source


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
