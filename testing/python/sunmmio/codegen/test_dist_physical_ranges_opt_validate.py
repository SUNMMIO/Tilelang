"""Physical source ranges for local producer readiness and sender reuse."""

import re

import pytest
import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.codegen.test_dist_semantics_opt_validate import generate
from testing.python.sunmmio.common.codegen_validation import simplify_suvm_mlir


@tilelang.jit(target="sunmmio")
def producer_kernel(mode, world_size: int = 1):
    @T.prim_func
    def main(A: T.MeshTensor((256,), T.placement.replicated(), T.int32), rank_id: T.dist.RankId, n: T.int32):
        with T.Kernel():
            src = T.alloc_shared((4096,), T.int32)
            dst = T.alloc_shared((16,), T.int32)
            sig = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            T.fill(src, 1)
            b = T.Cast("int32", T.Cast("uint8", n % 2 - 1))
            # Every DMA view is aligned, including the dynamic one.
            producer = (n % 2) * 3840 if mode == "dynamic_dma" else 3840
            start = b * 16 if mode == "dynamic_source" else 0 if mode == "separate" else 4080
            T.copy(A, src[producer : producer + 256])
            T.dist.put(src[start : start + 16], dst, (rank_id + 1) % world_size, signal=sig, submit=True)
            T.dist.wait_signal(sig, expected_delta=T.Cast("uint32", n % 2))

    return main


@tilelang.jit(target="sunmmio")
def disjoint_kernel(matrix, submitted, overlap, world_size: int = 1):
    shape = (32, 32) if matrix else (640,)

    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared(shape, T.int32)
            dst = T.alloc_shared((1, 32) if matrix else (32,), T.int32)
            s = T.dist.signal(expect="manual")
            T.fill(src, 1)
            for i in T.serial(4):
                T.dist.put(src[i * 4 : i * 4 + 1, :] if matrix else src[i * 128 : i * 128 + 32], dst, (rank_id + 1) % world_size, signal=s)
                if submitted:
                    T.dist.submit()
                for _ in T.serial(1 if matrix else 3):
                    if not matrix:
                        T.dist.wait_signal(s, expected_delta=0)
                    if matrix:
                        src[(i + (0 if overlap else 1)) * 4, 0] = 2
                    else:
                        src[(i + (0 if overlap else 1)) * 128] = 2
                if not submitted:
                    T.dist.submit()
                T.dist.wait_signal(s, expected_delta=1)

    return main


@pytest.mark.parametrize("mode", ["dynamic_source", "dynamic_dma", "constant", "separate"])
def test_local_producer_ready_before_transfer(tmp_path, mode):
    result, source = generate(producer_kernel, tmp_path, mode=mode)
    interval = source.split("suvm.copy_async", 1)[1].split("suvm.dist.batch_put", 1)[0]
    assert ("suvm.sync odma0" in interval) == (mode != "separate")

    # Specialize the already-generated dynamic MLIR, leaving the compiler's
    # descriptor construction and sync decisions intact.
    func = next(iter(result.device_mod.functions.values()))
    arg = next(i for i, v in enumerate(func.params) if v.name == "n")
    for n in (0, 1):
        head, sep, body = source.partition(") {")
        assert sep
        body = re.sub(rf"%arg{arg}\b", "%fixed_n", body)
        fixed = simplify_suvm_mlir(
            f"{head}{sep}\n    %fixed_n = arith.constant {n} : i32\n{body}",
            opt_args=("--canonicalize", "--cse", "--sccp", "--canonicalize"),
        )
        consts = dict(re.findall(r"(%[\w-]+) = arith.constant (\d+) : index", fixed))
        views = {
            v: (buf, int(consts[ix]) * int(width), int(width))
            for v, buf, ix, width in re.findall(
                r"(%[\w-]+) = suvm.get_partitioned_tile_view (%[\w-]+) indices = \[(%[\w-]+)\][^\n]* -> !suvm.tile_view<(\d+)xi32>", fixed
            )
        }
        dest = re.search(r"suvm.copy_async %[^,]+, (%[\w-]+)", fixed)[1]
        src = re.search(r"suvm.dist.batch_put %[^,]+, %[^,]+, (%[\w-]+)", fixed)[1]
        assert views[src][0] == views[dest][0]
        assert views[dest][1:] == ((n % 2) * 3840 if mode == "dynamic_dma" else 3840, 256)
        assert views[src][1:] == ((((n % 2 - 1) & 255) * 16 if mode == "dynamic_source" else 0 if mode == "separate" else 4080), 16)


@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("submitted", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_source_reuse_uses_invariant_physical_dimensions(tmp_path, matrix, submitted, overlap):
    opts = dict(matrix=matrix, submitted=submitted, overlap=overlap)
    if overlap and not submitted:
        with pytest.raises(tvm.error.InternalError, match="source is modified"):
            generate(disjoint_kernel, tmp_path, **opts)
        return
    _, source = generate(disjoint_kernel, tmp_path, **opts)
    if submitted:
        window = source.split("suvm.dist.batch_submit", 1)[1].split("suvm.tile.store", 1)[0]
        assert ("suvm.dist.flush" in window) == overlap
