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
from tilelang.language.mesh_tensor import MeshShardingPolicy, MeshReplicationType
from tilelang.layout import make_row_major
from tvm import tir
from tvm.tir import PyStmtExprVisitor
from tvm.tir.transform import prim_func_pass
from tvm.target import Target

# Global dict to collect layout_map from block annotations
collected_layout_map = {}


@tir.functor.visitor
class _LayoutMapCollector(PyStmtExprVisitor):
    """Visitor to extract layout_map from block annotations after SunmmioLayoutInference."""

    def __init__(self):
        super().__init__()

    def visit_block_(self, op: tir.Block) -> None:
        if "layout_map" in op.annotations:
            layout_map = op.annotations["layout_map"]
            collected_layout_map.clear()
            for key, layout in layout_map.items():
                collected_layout_map[key.name] = layout
        if "global_layout_map" in op.annotations:
            global_layout_map = op.annotations["global_layout_map"]
            for key, layout in global_layout_map.items():
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
    during SunmmioLayoutInference pass for Sunmmio target.
    """
    policy = MeshShardingPolicy(y=0, x=1, replicate=MeshReplicationType.NONE)
    device_mesh = (2, 2)

    M, N, K = 64, 64, 64
    block_M, block_N, block_K = 32, 32, 32

    A_layout = make_row_major((M, K))
    B_layout = make_row_major((K, N))
    C_layout = make_row_major((M, N))

    A_tensor = T.MeshTensor((M, K), policy, device_mesh, dtype="float16", layout=A_layout)
    B_tensor = T.MeshTensor((K, N), policy, device_mesh, dtype="float16", layout=B_layout)
    C_tensor = T.MeshTensor((M, N), policy, device_mesh, dtype="float32", layout=C_layout)

    @T.prim_func
    def kernel(A: A_tensor, B: B_tensor, C: C_tensor):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_shared = T.alloc_shared((block_M, block_N), "float32")

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
    mod = tvm.IRModule({"main": kernel})

    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tl.transform.InferSramScope()(mod)
        mod = tl.transform.LegalizeSunmmioCopyPath()(mod)
        mod = tl.transform.LayoutReducer()(mod)
        mod = tl.transform.SunmmioLayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)
        CollectLayoutMap()(mod)

    # Verify that global buffer 'A' has a layout in the layout_map
    assert "A" in collected_layout_map, (
        f"Global buffer 'A' should be in layout_map after SunmmioLayoutInference. Got: {list(collected_layout_map.keys())}"
    )

    a_layout = collected_layout_map["A"]
    assert a_layout is not None, "Layout for 'A' should not be None"
    assert len(a_layout.input_size) == 2, f"Expected 2 input dims, got {len(a_layout.input_size)}"


def test_global_buffer_layout_not_populated_for_cuda():
    """
    Test that global buffer layouts are NOT populated for non-Sunmmio targets (CUDA).
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

    target = Target("cuda")
    mod = tvm.IRModule({"main": kernel})

    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tl.transform.InferSramScope()(mod)
        mod = tl.transform.LegalizeSunmmioCopyPath()(mod)
        mod = tl.transform.LayoutInference()(mod)
        CollectLayoutMap()(mod)

    assert "A" not in collected_layout_map, "Global buffer 'A' should NOT be in layout_map for CUDA target"


def test_row_major_global_layout_values():
    """
    Test that the layout created from tensor_meta produces correct forward index mapping.
    """
    policy = MeshShardingPolicy(y=0, x=1, replicate=MeshReplicationType.NONE)
    device_mesh = (2, 2)

    M, N, K = 64, 64, 64
    block_M, block_N, block_K = 32, 32, 32

    A_layout = make_row_major((M, K))
    B_layout = make_row_major((K, N))
    C_layout = make_row_major((M, N))

    A_tensor = T.MeshTensor((M, K), policy, device_mesh, dtype="float16", layout=A_layout)
    B_tensor = T.MeshTensor((K, N), policy, device_mesh, dtype="float16", layout=B_layout)
    C_tensor = T.MeshTensor((M, N), policy, device_mesh, dtype="float32", layout=C_layout)

    @T.prim_func
    def kernel(A: A_tensor, B: B_tensor, C: C_tensor):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_shared = T.alloc_shared((block_M, block_N), "float32")

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": kernel})

    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tl.transform.InferSramScope()(mod)
        mod = tl.transform.LegalizeSunmmioCopyPath()(mod)
        mod = tl.transform.LayoutReducer()(mod)
        mod = tl.transform.SunmmioLayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)
        CollectLayoutMap()(mod)

    assert "A" in collected_layout_map, "Global buffer 'A' should be in layout_map"

    a_layout = collected_layout_map["A"]

    # For row-major with sharded shape (32, 32), strides (32, 1):
    offset_0_0 = a_layout.map_forward_index([0, 0])
    offset_0_1 = a_layout.map_forward_index([0, 1])
    offset_1_0 = a_layout.map_forward_index([1, 0])

    assert offset_0_0[0] == 0, f"Expected offset(0,0)=0, got {offset_0_0[0]}"
    assert offset_0_1[0] == 1, f"Expected offset(0,1)=1, got {offset_0_1[0]}"
    assert offset_1_0[0] == 32, f"Expected offset(1,0)=32, got {offset_1_0[0]}"


def test_dynamic_shape_global_buffer_layout():
    """
    Test that ParseGlobalBufferLayout correctly handles symbolic (dynamic) shapes.
    """
    M_var = tir.Var("M", "int32")
    K_var = tir.Var("K", "int32")

    block_M, block_N, block_K = 32, 32, 32

    policy = MeshShardingPolicy(replicate=MeshReplicationType.ALL)
    device_mesh = (2, 2)

    A_tensor = T.MeshTensor((M_var, K_var), policy, device_mesh, dtype="float16")
    B_tensor = T.MeshTensor((K_var, 64), policy, device_mesh, dtype="float16")
    C_tensor = T.MeshTensor((M_var, 64), policy, device_mesh, dtype="float32")

    @T.prim_func
    def kernel(A: A_tensor, B: B_tensor, C: C_tensor):
        with T.Kernel(T.ceildiv(64, block_N), T.ceildiv(M_var, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_shared = T.alloc_shared((block_M, block_N), "float32")

            for k in T.Pipelined(T.ceildiv(K_var, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    assert "tensor_meta" in kernel.attrs, "Kernel should have tensor_meta attribute"

    # Run through the Sunmmio layout inference pipeline
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": kernel})

    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tl.transform.InferSramScope()(mod)
        mod = tl.transform.LegalizeSunmmioCopyPath()(mod)
        mod = tl.transform.LayoutReducer()(mod)
        mod = tl.transform.SunmmioLayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)
        CollectLayoutMap()(mod)

    assert "A" in collected_layout_map, (
        f"Global buffer 'A' with symbolic shape should be in layout_map. Got: {list(collected_layout_map.keys())}"
    )

    a_layout = collected_layout_map["A"]
    assert a_layout is not None, "Layout for 'A' should not be None"
    assert len(a_layout.input_size) == 2, f"Expected 2 input dims, got {len(a_layout.input_size)}"
