import sys
import contextlib
from pathlib import Path

# 防止调用到官方tileLang
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
COMPILE_PIPELINE_DIR = REPO_ROOT / "testing" / "python" / "compile_pipeline"
if str(COMPILE_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(COMPILE_PIPELINE_DIR))

import tilelang
from tilelang import tvm
from tilelang.utils.target import determine_target
import tilelang.language as T
from compile_pipeline import compile_test

COPY_CASES = {
    "buffer_to_buffer_same_shape": "T.copy(A_128x128x128_global, B_128x128x128_shared)",
    "buffer_to_buffer_same_rank_last_mismatch": "T.copy(A_128x128x1_global, B_128x128x128_shared)",
    "buffer_to_buffer_same_rank_middle_mismatch": "T.copy(A_128x1x128_global, B_128x128x128_shared)",
    "buffer_to_buffer_same_rank_leading_mismatch": "T.copy(A_1x128x128_global, B_128x128x128_shared)",
    "buffer_to_buffer_same_rank_no_dim_reorder": "T.copy(A_1x128x128_global, B_128x1x128_shared)",
    "buffer_to_buffer_src_leading_one_rank2": "T.copy(A_1x128x128_global, B_128x128_shared)",
    "buffer_to_buffer_dst_leading_one_rank3": "T.copy(A_128x128_global, B_1x128x128_shared)",
    "buffer_to_buffer_src_middle_one_rank2": "T.copy(A_128x1x128_global, B_128x128_shared)",
    "buffer_to_buffer_rank4_middle_singleton": "T.copy(A_1x128x1x64_global, B_128x64_shared)",
    "buffer_to_region_equal": "T.copy(A_128x128x128_global, C_256x256x256_shared[0:128, 0:128, 0:128])",
    "buffer_to_region_larger_dst": "T.copy(A_128x128x128_global, C_256x256x256_shared[0:256, 0:256, 0:256])",
    "buffer_to_region_smaller_dst": "T.copy(A_128x128x128_global, C_128x128x32_shared[0:128, 0:128, 0:32])",
    "buffer_to_region_explicit_dst_oob": "T.copy(A_128x128x128_global, C_128x128x32_shared[0:128, 0:128, 0:128])",
    "buffer_to_load_point_dst": "T.copy(A_128x128x128_global, C_256x256x256_shared[0, 0, 0])",
    "buffer_to_load_point_dst_oob": "T.copy(A_128x128x128_global, C_128x128x32_shared[0, 0, 0])",
    "buffer_to_load_point_dst_unaligned": "T.copy(A_128x128x128_global, C_256x256x256_shared[1, 0, 0])",
    "region_to_buffer_explicit_src": "T.copy(A_128x128x128_global[0:32, 0:32, 0:32], C_128x128x32_shared)",
    "region_to_buffer_small_tile": "T.copy(A_128x128x128_global[0:16, 0:16, 0:16], C_128x128x32_shared)",
    "region_to_buffer_dst_too_small": "T.copy(A_128x128x128_global[0:64, 0:64, 0:64], C_128x128x32_shared)",
    "region_to_buffer_src_unaligned": "T.copy(A_128x128x128_global[1:33, 0:32, 0:32], C_128x128x32_shared)",
    "region_to_region_equal": "T.copy(A_128x128x128_global[0:32, 0:32, 0:32], C_128x128x32_shared[0:32, 0:32, 0:32])",
    "region_to_region_small_tile": "T.copy(A_128x128x128_global[0:16, 0:16, 0:16], C_128x128x32_shared[0:16, 0:16, 0:16])",
    "region_to_region_dst_oob": "T.copy(A_128x128x128_global[0:64, 0:64, 0:64], C_128x128x32_shared[0:64, 0:64, 0:64])",
    "region_to_region_src_gt_dst": "T.copy(A_128x128x128_global[0:32, 0:32, 0:32], C_128x128x32_shared[0:16, 0:16, 0:16])",
    "region_to_region_src_lt_dst": "T.copy(A_128x128x128_global[0:16, 0:16, 0:16], C_128x128x32_shared[0:32, 0:32, 0:32])",
    "region_to_region_extent_mismatch_dst_oob": "T.copy(A_128x128x128_global[0:32, 0:32, 0:32], C_128x128x32_shared[0:64, 0:64, 0:64])",
    "region_to_region_1d_tile_view": "T.copy(A_128x128x128_global[0, 0, 0:32], C_128x128x32_shared[0:1, 0:1, 0:32])",
    "region_to_region_no_dim_reorder": "T.copy(A_128x128x128_global[0:32, 0, 0:32], C_128x128x32_shared[0:1, 0:32, 0:32])",
    "region_to_region_rank_suffix_compatible": "T.copy(E_128x128_shared[:32, :32], A_128x128x128_global[1, :32, :32])",
    "region_to_region_rank_suffix_mismatch_middle_singleton": "T.copy(Q_1x128x1x64_global[0, 0:64, 0, 0:64], Q_64x64_shared)",
    "region_to_region_rank_mismatch_non1_leading": "T.copy(A_128x128x128_global[0:32, 0:32, 0:32], E_128x128_shared[:32, :32])",
    "region_to_load_point_dst": "T.copy(A_128x128x128_global[0:32, 0:32, 0:32], C_128x128x32_shared[0, 0, 0])",
    "region_to_load_point_dst_oob": "T.copy(A_128x128x128_global[0:32, 0:32, 0:32], C_128x128x32_shared[32, 32, 32])",
    "region_to_load_point_dst_unaligned": "T.copy(A_128x128x128_global[0:32, 0:32, 0:32], C_256x256x256_shared[1, 0, 0])",
    "load_to_buffer_full_dst": "T.copy(A_128x128x128_global[0, 0, 0], C_128x128x32_shared)",
    "load_to_buffer_rank_lower_full_dst": "T.copy(A_128x128x128_global[0, 0, 0], E_128x128_shared)",
    "load_to_buffer_clipped_unaligned": "T.copy(A_128x128x128_global[1, 2, 3], C_128x128x32_shared)",
    "load_to_buffer_clipped_legal": "T.copy(A_128x128x128_global[0, 0, 96], D_128x128x64_shared)",
    "load_to_region_explicit_dst": "T.copy(A_128x128x128_global[0, 0, 0], C_128x128x32_shared[0:128, 0:128, 0:32])",
    "load_to_region_clipped_unaligned": "T.copy(A_128x128x128_global[1, 2, 3], C_256x256x256_shared[0:128, 0:128, 0:32])",
    "load_to_region_clipped_legal": "T.copy(A_128x128x128_global[0, 0, 96], C_256x256x256_shared[0:128, 0:128, 0:64])",
    "load_to_region_dst_oob": "T.copy(A_128x128x128_global[0, 0, 0], C_128x128x32_shared[0:128, 0:128, 0:64])",
    "load_to_load_scalar": "T.copy(A_128x128x128_global[1, 2, 3], C_128x128x32_shared[0, 0, 0])",
}

COPY_CASE_ALLOCATIONS = {
    "buffer_to_buffer_same_shape": ["B_128x128x128_shared = T.alloc_shared((M, N, K), dtype)"],
    "buffer_to_buffer_same_rank_last_mismatch": ["B_128x128x128_shared = T.alloc_shared((M, N, K), dtype)"],
    "buffer_to_buffer_same_rank_middle_mismatch": ["B_128x128x128_shared = T.alloc_shared((M, N, K), dtype)"],
    "buffer_to_buffer_same_rank_leading_mismatch": ["B_128x128x128_shared = T.alloc_shared((M, N, K), dtype)"],
    "buffer_to_buffer_same_rank_no_dim_reorder": ["B_128x1x128_shared = T.alloc_shared((M, 1, K), dtype)"],
    "buffer_to_buffer_src_leading_one_rank2": ["B_128x128_shared = T.alloc_shared((M, N), dtype)"],
    "buffer_to_buffer_dst_leading_one_rank3": ["B_1x128x128_shared = T.alloc_shared((1, N, K), dtype)"],
    "buffer_to_buffer_src_middle_one_rank2": ["B_128x128_shared = T.alloc_shared((M, N), dtype)"],
    "buffer_to_buffer_rank4_middle_singleton": ["B_128x64_shared = T.alloc_shared((N, 64), dtype)"],
    "buffer_to_region_equal": ["C_256x256x256_shared = T.alloc_shared((256, 256, 256), dtype)"],
    "buffer_to_region_larger_dst": ["C_256x256x256_shared = T.alloc_shared((256, 256, 256), dtype)"],
    "buffer_to_region_smaller_dst": ["C_128x128x32_shared = T.alloc_shared((block_M, block_N, block_K), dtype)"],
    "buffer_to_region_explicit_dst_oob": ["C_128x128x32_shared = T.alloc_shared((block_M, block_N, block_K), dtype)"],
    "buffer_to_load_point_dst": ["C_256x256x256_shared = T.alloc_shared((256, 256, 256), dtype)"],
    "buffer_to_load_point_dst_oob": ["C_128x128x32_shared = T.alloc_shared((block_M, block_N, block_K), dtype)"],
    "buffer_to_load_point_dst_unaligned": ["C_256x256x256_shared = T.alloc_shared((256, 256, 256), dtype)"],
    "region_to_buffer_explicit_src": ["C_128x128x32_shared = T.alloc_shared((block_M, block_N, block_K), dtype)"],
    "region_to_buffer_small_tile": ["C_128x128x32_shared = T.alloc_shared((block_M, block_N, block_K), dtype)"],
    "region_to_buffer_dst_too_small": ["C_128x128x32_shared = T.alloc_shared((block_M, block_N, block_K), dtype)"],
    "region_to_buffer_src_unaligned": ["C_128x128x32_shared = T.alloc_shared((block_M, block_N, block_K), dtype)"],
    "region_to_region_equal": ["C_128x128x32_shared = T.alloc_shared((block_M, block_N, block_K), dtype)"],
    "region_to_region_small_tile": ["C_128x128x32_shared = T.alloc_shared((block_M, block_N, block_K), dtype)"],
    "region_to_region_dst_oob": ["C_128x128x32_shared = T.alloc_shared((block_M, block_N, block_K), dtype)"],
    "region_to_region_src_gt_dst": ["C_128x128x32_shared = T.alloc_shared((block_M, block_N, block_K), dtype)"],
    "region_to_region_src_lt_dst": ["C_128x128x32_shared = T.alloc_shared((block_M, block_N, block_K), dtype)"],
    "region_to_region_extent_mismatch_dst_oob": ["C_128x128x32_shared = T.alloc_shared((block_M, block_N, block_K), dtype)"],
    "region_to_region_1d_tile_view": ["C_128x128x32_shared = T.alloc_shared((block_M, block_N, block_K), dtype)"],
    "region_to_region_no_dim_reorder": ["C_128x128x32_shared = T.alloc_shared((block_M, block_N, block_K), dtype)"],
    "region_to_region_rank_suffix_compatible": ["E_128x128_shared = T.alloc_shared((block_M, block_N), dtype)"],
    "region_to_region_rank_suffix_mismatch_middle_singleton": ["Q_64x64_shared = T.alloc_shared((64, 64), dtype)"],
    "region_to_region_rank_mismatch_non1_leading": ["E_128x128_shared = T.alloc_shared((block_M, block_N), dtype)"],
    "region_to_load_point_dst": ["C_128x128x32_shared = T.alloc_shared((block_M, block_N, block_K), dtype)"],
    "region_to_load_point_dst_oob": ["C_128x128x32_shared = T.alloc_shared((block_M, block_N, block_K), dtype)"],
    "region_to_load_point_dst_unaligned": ["C_256x256x256_shared = T.alloc_shared((256, 256, 256), dtype)"],
    "load_to_buffer_full_dst": ["C_128x128x32_shared = T.alloc_shared((block_M, block_N, block_K), dtype)"],
    "load_to_buffer_rank_lower_full_dst": ["E_128x128_shared = T.alloc_shared((block_M, block_N), dtype)"],
    "load_to_buffer_clipped_unaligned": ["C_128x128x32_shared = T.alloc_shared((block_M, block_N, block_K), dtype)"],
    "load_to_buffer_clipped_legal": ["D_128x128x64_shared = T.alloc_shared((block_M, block_N, 64), dtype)"],
    "load_to_region_explicit_dst": ["C_128x128x32_shared = T.alloc_shared((block_M, block_N, block_K), dtype)"],
    "load_to_region_clipped_unaligned": ["C_256x256x256_shared = T.alloc_shared((256, 256, 256), dtype)"],
    "load_to_region_clipped_legal": ["C_256x256x256_shared = T.alloc_shared((256, 256, 256), dtype)"],
    "load_to_region_dst_oob": ["C_128x128x32_shared = T.alloc_shared((block_M, block_N, block_K), dtype)"],
    "load_to_load_scalar": ["C_128x128x32_shared = T.alloc_shared((block_M, block_N, block_K), dtype)"],
}

DEFAULT_COPY_CASE = "load_to_buffer_clipped_unaligned"


def select_copy_case() -> str:
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        for name, instr in COPY_CASES.items():
            print(f"{name}: {instr}")
        raise SystemExit(0)
    copy_case = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_COPY_CASE
    if copy_case not in COPY_CASES:
        raise ValueError(f"Unknown copy case: {copy_case}. Use --list to show available cases.")
    return copy_case


COPY_CASE = select_copy_case()


def emit_copy_case(
    copy_case,
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype,
    A_128x128x128_global,
    A_128x128x1_global,
    A_128x1x128_global,
    A_1x128x128_global,
    A_128x128_global,
    A_1x128x1x64_global,
    Q_1x128x1x64_global,
):
    env = {
        "T": T,
        "M": M,
        "N": N,
        "K": K,
        "block_M": block_M,
        "block_N": block_N,
        "block_K": block_K,
        "dtype": dtype,
        "A_128x128x128_global": A_128x128x128_global,
        "A_128x128x1_global": A_128x128x1_global,
        "A_128x1x128_global": A_128x1x128_global,
        "A_1x128x128_global": A_1x128x128_global,
        "A_128x128_global": A_128x128_global,
        "A_1x128x1x64_global": A_1x128x1x64_global,
        "Q_1x128x1x64_global": Q_1x128x1x64_global,
    }
    for stmt in COPY_CASE_ALLOCATIONS[copy_case]:
        exec(stmt, globals(), env)
    return eval(COPY_CASES[copy_case], globals(), env)


def mma_3times_single_thread(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32, dtype="float16", copy_case=DEFAULT_COPY_CASE):

    @T.prim_func
    def mma_3times_kernel(
        A_128x128x128_global: T.Tensor((M, N, K), dtype),
        A_128x128x1_global: T.Tensor((M, N, 1), dtype),
        A_128x1x128_global: T.Tensor((M, 1, K), dtype),
        A_1x128x128_global: T.Tensor((1, N, K), dtype),
        A_128x128_global: T.Tensor((M, N), dtype),
        A_1x128x1x64_global: T.Tensor((1, N, 1, 64), dtype),
        Q_1x128x1x64_global: T.Tensor((1, N, 1, 64), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            emit_copy_case(
                copy_case,
                M,
                N,
                K,
                block_M,
                block_N,
                block_K,
                dtype,
                A_128x128x128_global,
                A_128x128x1_global,
                A_128x1x128_global,
                A_1x128x128_global,
                A_128x128_global,
                A_1x128x1x64_global,
                Q_1x128x1x64_global,
            )

    return mma_3times_kernel


mma_3times_kernel = mma_3times_single_thread(M=128, N=128, K=128, copy_case=COPY_CASE)


def codegen_sunmmio_suvm_from_kernel(kernel):
    output_dir = REPO_ROOT / "tmp"
    output_dir.mkdir(exist_ok=True)
    ast_path = output_dir / "t_example_copy_ast.txt"
    with ast_path.open("w", encoding="utf-8") as f:
        with contextlib.redirect_stdout(f):
            compile_test(
                kernel,
                target="Sunmmio",
                pass_configs={tilelang.PassConfigKey.TL_AST_PRINT_ENABLE: True},
            )

    pass_dump_dir = output_dir / "t_example_copy_passes"
    host_mod, device_mod = compile_test(
        kernel,
        target="Sunmmio",
        log_pass_output=True,
        show_meta=False,
        log_dir=str(pass_dump_dir),
        log_passes=[
            "SunmmioLayoutInference",
            "LowerTileOp"
        ],
    )

    target = determine_target("Sunmmio", return_object=True)
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    try:
        src = builder(device_mod, target, "suvm").inspect_source()
    except TypeError:
        src = builder(device_mod, target).inspect_source()

    device_tir_path = output_dir / "t_example_copy_device_tir.py"
    suvm_mlir_path = output_dir / "t_example_copy_suvm.mlir"

    device_tir_path.write_text(device_mod.script(), encoding="utf-8")
    suvm_mlir_path.write_text(src, encoding="utf-8")

    print(f"Saved AST to {ast_path}")
    print(f"Saved pass dumps to {pass_dump_dir}")
    print(f"Saved device TIR to {device_tir_path}")
    print(f"Saved SUVM MLIR to {suvm_mlir_path}")


def t1():
    print(f"Active copy case: {COPY_CASE}")
    print(f"Instruction: {COPY_CASES[COPY_CASE]}")
    codegen_sunmmio_suvm_from_kernel(mma_3times_kernel)
    # print("codegen_str:")
    # print(codegen_str)


if __name__ == "__main__":
    t1()
