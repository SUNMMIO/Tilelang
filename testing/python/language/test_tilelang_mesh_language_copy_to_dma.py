import tilelang
import pytest
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target
import tilelang.language as T
from tilelang.language.v2.annot import MeshShardingPolicy


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
                # DRAM <- RSRAM
                T.copy(C_shared, C[by * block_M, ko * block_K])
                # RSRAM -> ASRAM
                T.copy(C_shared[8:24, 16:48], A_shared[24:40, 8:40])
                # RSRAM -> WSRAM
                T.copy(C_shared[8:32, 48:56], B_shared[40:64, 0:8])
                # RSRAM <-> RSRAM
                T.copy(C_shared, D_shared)

    return tvm.IRModule({'main': main})


TEST_CASES = [
    (128, 64, 64, 32),
]


@pytest.mark.parametrize(
    "K, block_M, block_N, block_K",
    TEST_CASES,
)
def test_tilelang_mesh_copy_to_dma(K, block_M, block_N, block_K):
    target_name = "Sunmmio"
    target = determine_target(target_name, return_object=True)
    with tvm.target.Target(target):
        mod = copy(K, block_M, block_N, block_K)
        mod = tvm.tir.transform.BindTarget(target)(mod)
        # Add wrapper for single buf store
        mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
        # Normalize negative indices to canonical non-negative form
        mod = tilelang.transform.LegalizeNegativeIndex()(mod)
        # Inject assumes to speedup tvm prover
        mod = tilelang.transform.InjectAssumes()(mod)
        # Simplify the IR expressions
        mod = tilelang.transform.Simplify()(mod)
        # Infer shared memory SRAM scope
        mod = tilelang.transform.InferSramScope()(mod)
        # Set layouts for reducers
        mod = tilelang.transform.LayoutReducer()(mod)
        # Infer memory layouts for fragments and shared memory
        mod = tilelang.transform.LayoutInference()(mod)
        # Lower high-level tile operations to low-level operations
        mod = tilelang.transform.LowerTileOp()(mod)
        print(mod)


def wrong_copy_1(M,
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
            C: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared.asram")
            A_shared_2 = T.alloc_shared((block_M, block_K), dtype, scope="shared.asram")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared.wsram")
            B_shared_2 = T.alloc_shared((block_K, block_N), dtype, scope="shared.wsram")
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype, scope="shared.rsram")

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                if error_type == 'D->A':
                    T.copy(C[by * block_M, ko * block_K], A_shared)
                elif error_type == 'A->D':
                    T.copy(A_shared, C[by * block_M, ko * block_K])
                elif error_type == 'D->W':
                    T.copy(C[by * block_M, ko * block_K], B_shared)
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

    return tvm.IRModule({'main': main})


WRONG_TEST_CASES = [
    (128, 128, 128, 32, 32, 32, "D->A",
     "Unsupported copy from global to shared.asram of Sunmmio target."),
    (128, 128, 128, 32, 32, 32, "A->D",
     "Unsupported copy from shared.asram to global of Sunmmio target."),
    (128, 128, 128, 32, 32, 32, "D->W",
     "Unsupported copy from global to shared.wsram of Sunmmio target."),
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
]


@pytest.mark.parametrize(
    "M, N, K, block_M, block_N, block_K, error_type, error_msg",
    WRONG_TEST_CASES,
)
def test_tilelang_mesh_wrong_copy_to_dma_1(M, N, K, block_M, block_N, block_K, error_type,
                                           error_msg):
    target_name = "Sunmmio"
    target = determine_target(target_name, return_object=True)
    with pytest.raises(tvm.error.InternalError, match=error_msg), tvm.target.Target(target):
        mod = wrong_copy_1(M, N, K, block_M, block_N, block_K, error_type)
        mod = tvm.tir.transform.BindTarget(target)(mod)
        # Add wrapper for single buf store
        mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
        # Normalize negative indices to canonical non-negative form
        mod = tilelang.transform.LegalizeNegativeIndex()(mod)
        # Inject assumes to speedup tvm prover
        mod = tilelang.transform.InjectAssumes()(mod)
        # Simplify the IR expressions
        mod = tilelang.transform.Simplify()(mod)
        # Infer shared memory SRAM scope
        mod = tilelang.transform.InferSramScope()(mod)
        # Set layouts for reducers
        mod = tilelang.transform.LayoutReducer()(mod)
        # Infer memory layouts for fragments and shared memory
        mod = tilelang.transform.LayoutInference()(mod)
        # Lower high-level tile operations to low-level operations
        mod = tilelang.transform.LowerTileOp()(mod)


def wrong_copy_2(M,
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
            C: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared.asram")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared.wsram")
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype, scope="shared.rsram")
            D_shared = T.alloc_shared((block_M, block_N), accum_dtype, scope="shared.rsram")

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # DRAM -> RSRAM
                T.copy(C[by * block_M, ko * block_K], C_shared)
                # DRAM <- RSRAM
                T.copy(C_shared, C[by * block_M, ko * block_K])
                # RSRAM -> ASRAM
                T.copy(C_shared, A_shared)
                # RSRAM -> WSRAM
                T.copy(C_shared, B_shared)
                # RSRAM <-> RSRAM
                T.copy(C_shared, D_shared)
                if error_type == 'A->W':
                    T.copy(A_shared, B_shared)
                elif error_type == 'W->A':
                    T.copy(B_shared, A_shared)

    return tvm.IRModule({'main': main})


WRONG_TEST_CASES = [
    (128, 128, 128, 32, 32, 32, "A->W",
     "Unsupported copy from shared.asram to shared.wsram of Sunmmio target."),
    (128, 128, 128, 32, 32, 32, "W->A",
     "Unsupported copy from shared.wsram to shared.asram of Sunmmio target."),
]


@pytest.mark.parametrize(
    "M, N, K, block_M, block_N, block_K, error_type, error_msg",
    WRONG_TEST_CASES,
)
def test_tilelang_mesh_wrong_copy_to_dma_2(M, N, K, block_M, block_N, block_K, error_type,
                                           error_msg):
    target_name = "Sunmmio"
    target = determine_target(target_name, return_object=True)
    with pytest.raises(tvm.error.InternalError, match=error_msg), tvm.target.Target(target):
        mod = wrong_copy_2(M, N, K, block_M, block_N, block_K, error_type)
        mod = tvm.tir.transform.BindTarget(target)(mod)
        # Add wrapper for single buf store
        mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
        # Normalize negative indices to canonical non-negative form
        mod = tilelang.transform.LegalizeNegativeIndex()(mod)
        # Inject assumes to speedup tvm prover
        mod = tilelang.transform.InjectAssumes()(mod)
        # Simplify the IR expressions
        mod = tilelang.transform.Simplify()(mod)
        # Infer shared memory SRAM scope
        mod = tilelang.transform.InferSramScope()(mod)
        # Set layouts for reducers
        mod = tilelang.transform.LayoutReducer()(mod)
        # Infer memory layouts for fragments and shared memory
        mod = tilelang.transform.LayoutInference()(mod)
        # Lower high-level tile operations to low-level operations
        mod = tilelang.transform.LowerTileOp()(mod)
