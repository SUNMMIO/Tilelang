"""
Test global buffer layout extraction from tensor_meta for Sunmmio target.

This tests the C++ implementation in:
- src/transform/global_layout_utils.cc
- Integration in src/transform/layout_inference.cc
"""

from tilelang import tvm as tvm
from tilelang.utils.target import determine_target, SUNMMIO_TARGET_DESC, target_is_sunmmio
import tilelang as tl
import tilelang.language as T
from tilelang.language.v2.annot import MeshShardingPolicy, MeshReplicationType
from tvm import tir
from tvm.tir import PyStmtExprVisitor
from tvm.tir.transform import prim_func_pass
from tvm.target import Target

# Global dict to collect layout_map from block annotations
collected_layout_map = {}


@tir.functor.visitor
class _LayoutMapCollector(PyStmtExprVisitor):
    """Visitor to extract layout_map from block annotations after LayoutInference."""

    def __init__(self):
        super().__init__()

    def visit_block_(self, op: tir.Block) -> None:
        if "layout_map" in op.annotations:
            layout_map = op.annotations["layout_map"]
            collected_layout_map.clear()
            for key, layout in layout_map.items():
                # key is a Buffer, use its name as dict key
                collected_layout_map[key.name] = layout


def CollectLayoutMap():
    """TIR pass to collect layout_map from block annotations."""

    def pass_fn(func: tir.PrimFunc, mod, ctx):
        _LayoutMapCollector().visit_stmt(func.body)
        return func

    return prim_func_pass(pass_fn, opt_level=0)


def test_sunmmio_target_detection():
    """Verify Sunmmio target detection works correctly."""
    target = Target(SUNMMIO_TARGET_DESC)
    assert target_is_sunmmio(target), "Should detect Sunmmio target"

    cuda_target = Target("cuda")
    assert not target_is_sunmmio(cuda_target), "CUDA should not be detected as Sunmmio"


def test_global_buffer_layout_populated_for_sunmmio():
    """
    Test that global buffer layouts from tensor_meta are populated into layout_map
    during LayoutInference pass for Sunmmio target.

    Uses a GEMM-style kernel that triggers proper layout inference for shared buffers.
    """
    policy = MeshShardingPolicy(y=0, x=1, replicate=MeshReplicationType.NONE)
    device_mesh = (2, 2)

    # Use shapes that match the test requirements
    M, N, K = 64, 64, 64
    block_M, block_N, block_K = 32, 32, 32

    # Simple row-major hierarchical layout
    A_hdims = (64, 64)
    A_hgroups = ((0, 1), (1, 2))
    A_hstrides = (64, 1)

    A_tensor = T.MeshTensor(
        (M, K),
        policy,
        device_mesh,
        dtype="float16",
        hierarchical_dims=A_hdims,
        hierarchical_groups=A_hgroups,
        hierarchical_strides=A_hstrides,
    )

    B_tensor = T.MeshTensor(
        (K, N),
        policy,
        device_mesh,
        dtype="float16",
        hierarchical_dims=(64, 64),
        hierarchical_groups=((0, 1), (1, 2)),
        hierarchical_strides=(64, 1),
    )

    C_tensor = T.MeshTensor(
        (M, N),
        policy,
        device_mesh,
        dtype="float32",
        hierarchical_dims=(64, 64),
        hierarchical_groups=((0, 1), (1, 2)),
        hierarchical_strides=(64, 1),
    )

    @T.prim_func
    def kernel(
        A: A_tensor,
        B: B_tensor,
        C: C_tensor,
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_shared = T.alloc_shared((block_M, block_N), "float32")

            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    # Verify tensor_meta exists before compilation
    assert "tensor_meta" in kernel.attrs, "Kernel should have tensor_meta attribute"

    # Get Sunmmio target
    target = determine_target("Sunmmio", return_object=True)

    # Create IR module and run passes
    mod = tvm.IRModule({'main': kernel})

    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tl.transform.LayoutInference()(mod)
        CollectLayoutMap()(mod)

    # Verify that global buffer 'A' has a layout in the layout_map
    assert "A" in collected_layout_map, \
        f"Global buffer 'A' should be in layout_map after LayoutInference. Got: {list(collected_layout_map.keys())}"

    # Verify the layout is a Layout object (hierarchical layout)
    a_layout = collected_layout_map["A"]
    assert a_layout is not None, "Layout for 'A' should not be None"

    # The layout should have 2 input dimensions matching the buffer shape
    assert len(a_layout.input_size) == 2, f"Expected 2 input dims, got {len(a_layout.input_size)}"


def test_global_buffer_layout_not_populated_for_cuda():
    """
    Test that global buffer layouts are NOT populated for non-Sunmmio targets (CUDA).
    The PopulateGlobalBufferLayouts function should return early for CUDA.

    Uses a simple copy kernel to avoid GEMM instruction selection issues.
    """
    M, N = 64, 64
    block_M, block_N = 32, 32

    @T.prim_func
    def kernel(
            A: T.Tensor((M, N), "float16"),
            B: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), "float16")

            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(A_shared, B[by * block_M, bx * block_N])

    # Use CUDA target
    target = Target("cuda")

    mod = tvm.IRModule({'main': kernel})

    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tl.transform.LayoutInference()(mod)
        CollectLayoutMap()(mod)

    # For CUDA without MeshTensor, global buffer 'A' should NOT be in layout_map
    # (only fragment/shared buffers get layouts inferred)
    # This verifies that our code path for Sunmmio is not triggered for CUDA
    assert "A" not in collected_layout_map, \
        "Global buffer 'A' should NOT be in layout_map for CUDA target"


def test_hierarchical_layout_values():
    """
    Test that the hierarchical layout created from tensor_meta produces
    correct forward index mapping.
    """
    policy = MeshShardingPolicy(y=0, x=1, replicate=MeshReplicationType.NONE)
    device_mesh = (2, 2)

    M, N, K = 64, 64, 64
    block_M, block_N, block_K = 32, 32, 32

    # Simple row-major style hierarchical layout
    # After sharding by 2x2: sharded_shape = (32, 32)
    # sharded_hdims = (32, 32), sharded_hstrides = (32, 1)
    hdims = (64, 64)
    hgroups = ((0, 1), (1, 2))
    hstrides = (64, 1)  # row-major

    A_tensor = T.MeshTensor(
        (M, K),
        policy,
        device_mesh,
        dtype="float16",
        hierarchical_dims=hdims,
        hierarchical_groups=hgroups,
        hierarchical_strides=hstrides,
    )

    B_tensor = T.MeshTensor(
        (K, N),
        policy,
        device_mesh,
        dtype="float16",
        hierarchical_dims=(64, 64),
        hierarchical_groups=((0, 1), (1, 2)),
        hierarchical_strides=(64, 1),
    )

    C_tensor = T.MeshTensor(
        (M, N),
        policy,
        device_mesh,
        dtype="float32",
        hierarchical_dims=(64, 64),
        hierarchical_groups=((0, 1), (1, 2)),
        hierarchical_strides=(64, 1),
    )

    @T.prim_func
    def kernel(
        A: A_tensor,
        B: B_tensor,
        C: C_tensor,
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_shared = T.alloc_shared((block_M, block_N), "float32")

            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({'main': kernel})

    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tl.transform.LayoutInference()(mod)
        CollectLayoutMap()(mod)

    assert "A" in collected_layout_map, "Global buffer 'A' should be in layout_map"

    a_layout = collected_layout_map["A"]

    # Verify the layout computes correct physical offsets
    # For row-major with sharded_strides (32, 1):
    # offset(i,j) = i * 32 + j * 1

    # Test a few index mappings
    offset_0_0 = a_layout.map_forward_index([0, 0])
    offset_0_1 = a_layout.map_forward_index([0, 1])
    offset_1_0 = a_layout.map_forward_index([1, 0])

    # offset(0,0) = 0
    # offset(0,1) = 1
    # offset(1,0) = 32
    assert offset_0_0[0] == 0, f"Expected offset(0,0)=0, got {offset_0_0[0]}"
    assert offset_0_1[0] == 1, f"Expected offset(0,1)=1, got {offset_0_1[0]}"
    assert offset_1_0[0] == 32, f"Expected offset(1,0)=32, got {offset_1_0[0]}"


if __name__ == "__main__":
    test_sunmmio_target_detection()
    print("PASSED: test_sunmmio_target_detection")

    test_global_buffer_layout_populated_for_sunmmio()
    print("PASSED: test_global_buffer_layout_populated_for_sunmmio")

    test_global_buffer_layout_not_populated_for_cuda()
    print("PASSED: test_global_buffer_layout_not_populated_for_cuda")

    test_hierarchical_layout_values()
    print("PASSED: test_hierarchical_layout_values")

    print("\nAll tests passed!")
