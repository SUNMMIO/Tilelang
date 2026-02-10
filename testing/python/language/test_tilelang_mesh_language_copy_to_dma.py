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


TEST_CASES = [
    (
        128,
        64,
        64,
        32,
        [
            # DRAM -> RSRAM
            # T.copy(C[by * block_M, ko * block_K], C_shared)
            "T.dma_copy(\"global\", \"shared.rsram\", 7, 2, 64, 64, 32, 128, _j * 32 + _i, 2, 64, 64, 64, 64, _i // 32 * 2048 + _j // 32 * 1024 + _i % 32 * 32 + _j % 32, C.data, by * 64, ko * 32, T.tvm_access_ptr(T.type_annotation(\"float32\"), C_shared.data, 0, 4096, 2), 0, 0)",
            # DRAM -> WSRAM
            # T.copy(C[by * block_M, ko * block_K], B_shared)
            "T.dma_copy(\"global\", \"shared.wsram\", 7, 2, 64, 64, 32, 128, _j * 32 + _i, 2, 64, 64, 64, 64, _i // 32 * 2048 + _j // 32 * 1024 + _i % 32 * 32 + _j % 32, C.data, by * 64, ko * 32, T.tvm_access_ptr(T.type_annotation(\"float32\"), B_shared.data, 0, 4096, 2), 0, 0)",
            # DRAM <- RSRAM
            # T.copy(C_shared, C[by * block_M, ko * block_K])
            "T.dma_copy(\"shared.rsram\", \"global\", 7, 2, 64, 64, 64, 64, _i // 32 * 2048 + _j // 32 * 1024 + _i % 32 * 32 + _j % 32, 2, 64, 64, 32, 128, _j * 32 + _i, T.tvm_access_ptr(T.type_annotation(\"float32\"), C_shared.data, 0, 4096, 1), 0, 0, C.data, by * 64, ko * 32)",
            # DRAM -> ASRAM
            # T.copy(C[by * block_M, ko * block_K], A_shared)
            "T.dma_copy(\"global\", \"shared.asram\", 7, 2, 64, 64, 32, 128, _j * 32 + _i, 2, 64, 64, 64, 64, _i // 32 * 2048 + _j // 32 * 1024 + _i % 32 * 32 + _j % 32, C.data, by * 64, ko * 32, T.tvm_access_ptr(T.type_annotation(\"float32\"), A_shared.data, 0, 4096, 2), 0, 0)",
            # RSRAM -> ASRAM
            # T.copy(C_shared[8:24, 16:48], A_shared[24:40, 8:40])
            "T.dma_copy(\"shared.rsram\", \"shared.asram\", 7, 2, 16, 32, 64, 64, _i // 32 * 2048 + _j // 32 * 1024 + _i % 32 * 32 + _j % 32, 2, 16, 32, 64, 64, _i // 32 * 2048 + _j // 32 * 1024 + _i % 32 * 32 + _j % 32, T.tvm_access_ptr(T.type_annotation(\"float32\"), C_shared.data, 0, 4096, 1), 8, 16, T.tvm_access_ptr(T.type_annotation(\"float32\"), A_shared.data, 0, 4096, 2), 24, 8)",
            # RSRAM -> WSRAM
            # T.copy(C_shared[8:32, 48:56], B_shared[40:64, 0:8])
            "T.dma_copy(\"shared.rsram\", \"shared.wsram\", 7, 2, 24, 8, 64, 64, _i // 32 * 2048 + _j // 32 * 1024 + _i % 32 * 32 + _j % 32, 2, 24, 8, 64, 64, _i // 32 * 2048 + _j // 32 * 1024 + _i % 32 * 32 + _j % 32, T.tvm_access_ptr(T.type_annotation(\"float32\"), C_shared.data, 0, 4096, 1), 8, 48, T.tvm_access_ptr(T.type_annotation(\"float32\"), B_shared.data, 0, 4096, 2), 40, 0)",
            # RSRAM <-> RSRAM
            # T.copy(C_shared, D_shared)
            "T.dma_copy(\"shared.rsram\", \"shared.rsram\", 7, 2, 64, 64, 64, 64, _i // 32 * 2048 + _j // 32 * 1024 + _i % 32 * 32 + _j % 32, 2, 64, 64, 64, 64, _i // 32 * 2048 + _j // 32 * 1024 + _i % 32 * 32 + _j % 32, T.tvm_access_ptr(T.type_annotation(\"float32\"), C_shared.data, 0, 4096, 1), 0, 0, T.tvm_access_ptr(T.type_annotation(\"float32\"), D_shared.data, 0, 4096, 2), 0, 0)",
        ]),
]


@pytest.mark.parametrize(
    "K, block_M, block_N, block_K, lower_stmt",
    TEST_CASES,
)
def test_tilelang_mesh_copy_to_dma(K, block_M, block_N, block_K, lower_stmt):
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
        texts = mod.script().split('\n')
        texts = texts[29:-2]
        texts = [it.lstrip() for it in texts]
        for i in range(len(texts)):
            assert texts[i] == lower_stmt[i]


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
    target_name = "Sunmmio"
    target = determine_target(target_name, return_object=True)
    with pytest.raises(tvm.error.InternalError, match=error_msg), tvm.target.Target(target):
        mod = wrong_copy(M, N, K, block_M, block_N, block_K, error_type)
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
