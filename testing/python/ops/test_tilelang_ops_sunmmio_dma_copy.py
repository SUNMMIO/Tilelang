"""Test that SUNMMIO copy lowering emits tl.dma_copy with tl.tileop.region args,
and that each region can be normalized back to a BufferRegion with full metadata."""

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.utils.target import SUNMMIO_TARGET_DESC, determine_target
from tilelang.language.v2.annot import MeshShardingPolicy
from tvm import tir
from tvm.tir import PyStmtExprVisitor
import pytest

tilelang.env.disable_cache()


def simple_copy_kernel(M, N, block_M, block_N, dtype="float16"):
    """A minimal kernel with T.copy from global to shared memory."""

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype),):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            T.copy(A[by * block_M, bx * block_N], A_shared)

    return tvm.IRModule({"main": main})


def apply_sunmmio_passes(mod, target):
    """Apply the full SUNMMIO pass pipeline used for DMA copy lowering."""
    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.InferSramScope()(mod)
    mod = tilelang.transform.LayoutReducer()(mod)
    mod = tilelang.transform.LayoutInference()(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    return mod


class RegionRange:
    """Simple container for (min, extent) of a region axis."""

    def __init__(self, min_val, extent):
        self.min = min_val
        self.extent = extent


def normalize_region(region_call):
    """Decode a tl.tileop.region Call back into (buffer, extents, access_mask).

    This mirrors what NormalizeToBufferRegion does in C++:
      args[0] = BufferLoad (indices are per-axis minima)
      args[1] = access_mask (int)
      args[2+i] = extent for axis i

    On SUNMMIO, buffer remap is disabled so buffers retain their original
    N-D shape and indices always match the number of extents.

    Returns (buffer, extents, access_mask) where extents is a list of
    RegionRange objects with .min and .extent attributes.
    """
    assert isinstance(region_call, tir.Call)
    assert region_call.op.name == "tl.tileop.region"

    load = region_call.args[0]
    assert isinstance(load, tir.BufferLoad)

    access_mask = int(region_call.args[1])
    num_extents = len(region_call.args) - 2

    assert len(
        load.indices) == num_extents, (f"Expected {num_extents} indices, got {len(load.indices)}")

    ranges = []
    for i in range(num_extents):
        ranges.append(RegionRange(load.indices[i], region_call.args[2 + i]))

    return load.buffer, ranges, access_mask


@tir.functor.visitor
class _DmaCopyVisitor(PyStmtExprVisitor):
    """Walk TIR and collect tl.dma_copy calls and their region arguments."""

    def __init__(self):
        super().__init__()
        self.dma_copy_calls = []
        self.layout_map = {}

    def visit_block_(self, op: tir.Block) -> None:
        if "layout_map" in op.annotations:
            for key, layout in op.annotations["layout_map"].items():
                self.layout_map[key.name] = layout
        self.visit_stmt(op.body)

    def visit_call_(self, op: tir.Call) -> None:
        if hasattr(op, "op") and hasattr(op.op, "name") and op.op.name == "tl.dma_copy":
            self.dma_copy_calls.append(op)
        # Visit children
        for arg in op.args:
            self.visit_expr(arg)

    def visit_evaluate_(self, op: tir.Evaluate) -> None:
        self.visit_expr(op.value)


def extract_dma_copy_lines(mod):
    """Extract T.dma_copy lines from TIR script, robust to formatting changes."""
    return [line.lstrip() for line in mod.script().split('\n') if 'T.dma_copy' in line]


SIMPLE_COPY_CASES = [
    # (M, N, block_M, block_N)
    (128, 128, 32, 32),
    (256, 256, 64, 64),
    (128, 256, 32, 64),
]


@pytest.mark.parametrize("M, N, block_M, block_N", SIMPLE_COPY_CASES)
def test_tilelang_dma_copy(M, N, block_M, block_N):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = simple_copy_kernel(M, N, block_M, block_N)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    # Walk the lowered IR and find dma_copy calls
    visitor = _DmaCopyVisitor()
    func = mod["main"]
    visitor.visit_stmt(func.body)

    # Verify that exactly one tl.dma_copy call was emitted
    assert len(visitor.dma_copy_calls) == 1, (
        f"Expected exactly 1 tl.dma_copy call, got {len(visitor.dma_copy_calls)}")

    call = visitor.dma_copy_calls[0]

    # dma_copy should have exactly 2 arguments (src_region, dst_region)
    assert len(call.args) == 2, (f"Expected 2 args for dma_copy, got {len(call.args)}")

    # Each argument should be a tl.tileop.region Call
    for i, arg in enumerate(call.args):
        assert isinstance(arg, tir.Call), (f"dma_copy arg[{i}] should be a Call, got {type(arg)}")
        assert hasattr(arg.op, "name") and arg.op.name == "tl.tileop.region", (
            f"dma_copy arg[{i}] should be tl.tileop.region, got {arg.op.name}")

    # --- Normalize regions back to buffer metadata ---
    src_buf, src_ranges, src_mask = normalize_region(call.args[0])
    dst_buf, dst_ranges, dst_mask = normalize_region(call.args[1])

    # access_mask: 1=read for src, 2=write for dst
    assert src_mask == 1, f"Source access_mask should be 1 (read), got {src_mask}"
    assert dst_mask == 2, f"Destination access_mask should be 2 (write), got {dst_mask}"

    # Buffer dtype
    assert src_buf.dtype == "float16", f"Source dtype should be float16, got {src_buf.dtype}"
    assert dst_buf.dtype == "float16", f"Dest dtype should be float16, got {dst_buf.dtype}"

    # Buffer shapes: both stay 2D (no flattening on SUNMMIO)
    assert len(src_buf.shape) == 2, f"Source buffer should be 2D, got {len(src_buf.shape)}D"
    assert int(src_buf.shape[0]) == M, f"Source shape[0] should be {M}, got {src_buf.shape[0]}"
    assert int(src_buf.shape[1]) == N, f"Source shape[1] should be {N}, got {src_buf.shape[1]}"

    assert len(dst_buf.shape) == 2, f"Dest buffer should be 2D, got {len(dst_buf.shape)}D"
    assert int(
        dst_buf.shape[0]) == block_M, (f"Dest shape[0] should be {block_M}, got {dst_buf.shape[0]}")
    assert int(
        dst_buf.shape[1]) == block_N, (f"Dest shape[1] should be {block_N}, got {dst_buf.shape[1]}")

    # Buffer scope: InferSramScope assigns shared.rsram for a plain alloc_shared
    src_scope = src_buf.scope()
    assert src_scope == "" or src_scope == "global", (
        f"Source buffer should be in global scope, got '{src_scope}'")
    dst_scope = dst_buf.scope()
    assert dst_scope == "shared.rsram", (
        f"Destination buffer should be shared.rsram after InferSramScope, got '{dst_scope}'")

    # Region extents match block dimensions
    assert len(src_ranges) == 2
    src_extent_0 = int(src_ranges[0].extent)
    src_extent_1 = int(src_ranges[1].extent)
    assert src_extent_0 == block_M, (f"Source extent[0] should be {block_M}, got {src_extent_0}")
    assert src_extent_1 == block_N, (f"Source extent[1] should be {block_N}, got {src_extent_1}")

    assert len(dst_ranges) == 2
    dst_extent_0 = int(dst_ranges[0].extent)
    dst_extent_1 = int(dst_ranges[1].extent)
    assert dst_extent_0 == block_M, (f"Dest extent[0] should be {block_M}, got {dst_extent_0}")
    assert dst_extent_1 == block_N, (f"Dest extent[1] should be {block_N}, got {dst_extent_1}")


def wrong_copy(M,
               N,
               K,
               block_M,
               block_N,
               block_K,
               error_type,
               dtype="float16",
               accum_dtype="float16"):

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), accum_dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared.asram")
            A_shared_2 = T.alloc_shared((block_M, block_K), dtype, scope="shared.asram")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared.wsram")
            B_shared_2 = T.alloc_shared((block_K, block_N), dtype, scope="shared.wsram")
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype, scope="shared.rsram")

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                if error_type == 'A->D':
                    T.copy(A_shared, C[by * block_M, ko * block_K])
                elif error_type == 'W->D':
                    T.copy(B_shared, C[by * block_M, ko * block_K])
                elif error_type == 'A->R':
                    T.copy(A_shared, C_shared)
                elif error_type == 'W->R':
                    T.copy(B_shared, C_shared)
                elif error_type == 'D<->D':
                    T.copy(C[by * block_M, ko * block_K], B[by * block_M, ko * block_K])
                elif error_type == 'A<->A':
                    T.copy(A_shared, A_shared_2)
                elif error_type == 'W<->W':
                    T.copy(B_shared, B_shared_2)
                elif error_type == 'A->W':
                    T.copy(A_shared, B_shared)
                elif error_type == 'W->A':
                    T.copy(B_shared, A_shared)

    return tvm.IRModule({'main': main})


WRONG_TEST_CASES = [
    (128, 128, 128, 32, 32, 32, "A->D",
     "Unsupported copy from shared.asram to global of Sunmmio target."),
    (128, 128, 128, 32, 32, 32, "W->D",
     "Unsupported copy from shared.wsram to global of Sunmmio target."),
    (128, 128, 128, 32, 32, 32, "A->R",
     "Unsupported copy from shared.asram to shared.rsram of Sunmmio target."),
    (128, 128, 128, 32, 32, 32, "W->R",
     "Unsupported copy from shared.wsram to shared.rsram of Sunmmio target."),
    # (128, 128, 128, 32, 32, 32, "D<->D",
    #  "Unsupported copy from global to global of Sunmmio target."),
    # D<->D not work now
    (128, 128, 128, 32, 32, 32, "A<->A",
     "Unsupported copy from shared.asram to shared.asram of Sunmmio target."),
    (128, 128, 128, 32, 32, 32, "W<->W",
     "Unsupported copy from shared.wsram to shared.wsram of Sunmmio target."),
    (128, 128, 128, 32, 32, 32, "A->W",
     "Unsupported copy from shared.asram to shared.wsram of Sunmmio target."),
    (128, 128, 128, 32, 32, 32, "W->A",
     "Unsupported copy from shared.wsram to shared.asram of Sunmmio target."),
]


@pytest.mark.parametrize(
    "M, N, K, block_M, block_N, block_K, error_type, error_msg",
    WRONG_TEST_CASES,
)
def test_tilelang_mesh_wrong_copy_to_dma(M, N, K, block_M, block_N, block_K, error_type, error_msg):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with pytest.raises(tvm.error.InternalError, match=error_msg), tvm.target.Target(target):
        mod = wrong_copy(M, N, K, block_M, block_N, block_K, error_type)
        mod = apply_sunmmio_passes(mod, target)


def copy(K, block_M, block_N, block_K, dtype="float32", accum_dtype="float32"):
    MyTensor = T.MeshTensor((128, 128),
                            sharding_policy=MeshShardingPolicy(cross_mesh_dim=0),
                            device_mesh_config=(2, 2),
                            hierarchical_dims=(4, 32, 128),
                            hierarchical_groups=((0, 2), (2, 3)),
                            hierarchical_strides=(32, 1, 4096))

    @T.prim_func
    def main(C: MyTensor):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(128, block_N), T.ceildiv(128, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.asram")
            B_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.wsram")
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype, scope="shared.rsram")
            D_shared = T.alloc_shared((block_M, block_N), accum_dtype, scope="shared.rsram")

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # DRAM -> RSRAM
                T.copy(C[by * block_M, ko * block_K], C_shared)
                # DRAM -> WSRAM
                T.copy(C[by * block_M, ko * block_K], B_shared)
                # DRAM <- RSRAM
                T.copy(C_shared, C[by * block_M, ko * block_K])
                # DRAM -> ASRAM
                T.copy(C[by * block_M, ko * block_K], A_shared)
                # RSRAM -> ASRAM
                T.copy(C_shared[8:24, 16:48], A_shared[24:40, 8:40])
                # RSRAM -> WSRAM
                T.copy(C_shared[8:32, 48:56], B_shared[40:64, 0:8])
                # RSRAM <-> RSRAM
                T.copy(C_shared, D_shared)

    return tvm.IRModule({'main': main})


# fmt: off
MESH_COPY_CASES = [
    (
        128, 64, 64, 32,
        [
            # DRAM -> RSRAM
            'T.dma_copy(T.region(C[by * 64, ko * 32], 1, 64, 64), T.region(C_shared[0, 0], 2, 64, 64))',
            # DRAM -> WSRAM
            'T.dma_copy(T.region(C[by * 64, ko * 32], 1, 64, 64), T.region(B_shared[0, 0], 2, 64, 64))',
            # DRAM <- RSRAM
            'T.dma_copy(T.region(C_shared[0, 0], 1, 64, 64), T.region(C[by * 64, ko * 32], 2, 64, 64))',
            # DRAM -> ASRAM
            'T.dma_copy(T.region(C[by * 64, ko * 32], 1, 64, 64), T.region(A_shared[0, 0], 2, 64, 64))',
            # RSRAM -> ASRAM
            'T.dma_copy(T.region(C_shared[8, 16], 1, 16, 32), T.region(A_shared[24, 8], 2, 16, 32))',
            # RSRAM -> WSRAM
            'T.dma_copy(T.region(C_shared[8, 48], 1, 24, 8), T.region(B_shared[40, 0], 2, 24, 8))',
            # RSRAM <-> RSRAM
            'T.dma_copy(T.region(C_shared[0, 0], 1, 64, 64), T.region(D_shared[0, 0], 2, 64, 64))',
        ],
    ),
    (
        256, 64, 64, 64,
        [
            # DRAM -> RSRAM
            'T.dma_copy(T.region(C[by * 64, ko * 64], 1, 64, 64), T.region(C_shared[0, 0], 2, 64, 64))',
            # DRAM -> WSRAM
            'T.dma_copy(T.region(C[by * 64, ko * 64], 1, 64, 64), T.region(B_shared[0, 0], 2, 64, 64))',
            # DRAM <- RSRAM
            'T.dma_copy(T.region(C_shared[0, 0], 1, 64, 64), T.region(C[by * 64, ko * 64], 2, 64, 64))',
            # DRAM -> ASRAM
            'T.dma_copy(T.region(C[by * 64, ko * 64], 1, 64, 64), T.region(A_shared[0, 0], 2, 64, 64))',
            # RSRAM -> ASRAM
            'T.dma_copy(T.region(C_shared[8, 16], 1, 16, 32), T.region(A_shared[24, 8], 2, 16, 32))',
            # RSRAM -> WSRAM
            'T.dma_copy(T.region(C_shared[8, 48], 1, 24, 8), T.region(B_shared[40, 0], 2, 24, 8))',
            # RSRAM <-> RSRAM
            'T.dma_copy(T.region(C_shared[0, 0], 1, 64, 64), T.region(D_shared[0, 0], 2, 64, 64))',
        ],
    ),
]
# fmt: on


@pytest.mark.parametrize(
    "K, block_M, block_N, block_K, lower_stmt",
    MESH_COPY_CASES,
)
def test_tilelang_mesh_copy_to_dma(K, block_M, block_N, block_K, lower_stmt):
    target_name = "Sunmmio"
    target = determine_target(target_name, return_object=True)
    with tvm.target.Target(target):
        mod = copy(K, block_M, block_N, block_K)
        mod = apply_sunmmio_passes(mod, target)
        texts = extract_dma_copy_lines(mod)
        assert len(texts) == len(lower_stmt), (
            f"Expected {len(lower_stmt)} dma_copy lines, got {len(texts)}")
        for i in range(len(texts)):
            assert texts[i] == lower_stmt[i], (
                f"Line {i} mismatch:\n  actual:   {texts[i]}\n  expected: {lower_stmt[i]}")
