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

def mma_3times_single_thread(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32, dtype="float16"):

    @T.prim_func
    def mma_3times_kernel(
        A_128x128x128_global: T.Tensor((M, N, K), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            B_128x128x128_shared = T.alloc_shared((M, N, K), dtype)
            C_128x128x32_shared = T.alloc_shared((block_M, block_N, block_K), dtype)
            C_256x256x256_shared = T.alloc_shared((256, 256, 256), dtype)
            D_128x128x32_shared = T.alloc_shared((block_M, block_N, block_K), dtype)
            E_128x128_shared = T.alloc_shared((block_N, block_K), dtype)

            # T.copy(A_128x128x128_global, B_128x128x128_shared)
            # T.copy(A_128x128x128_global[1, 2, 3], C_128x128x32_shared[0, 0, 0])
            T.copy(A_128x128x128_global[1, 2, 3], C_128x128x32_shared)
            # T.copy(A_128x128x128_global[1, 2, 3], C_256x256x256_shared[0:128, 0:128, 0:32])

            # BufferRegion -> buffer
            # T.copy(A_128x128x128_global[0:16, 0:16, 0:16], C_128x128x32_shared)

            # 会把dst extent补成src 的 extent, 语义：A[0:16, 0:16, 0:16] -> A_shared1[0:16, 0:16, 0:16]
            # 由于A 和 A_shared1的shape不同，导致span不一致
            # BufferRegion -> Bufferload
            # T.copy(A_128x128x128_global[0:16, 0:16, 0:16], C_128x128x32_shared[0, 0, 0])

            # 依旧会导致 bounding span不一致
            # BufferRegion -> BufferRegion.
            # T.copy(A_128x128x128_global[0:16, 0:16, 0:16], C_128x128x32_shared[0:16, 0:16, 0:16])
            # T.copy(A_128x128x128_global[0:32, 0:32, 0:32], C_128x128x32_shared[0:16, 0:16, 0:16]) # 这句在官方那会直接报错
            # T.copy(A_128x128x128_global[0:16, 0:16, 0:16], C_128x128x32_shared[0:32, 0:32, 0:32])

            # 特殊的BufferRegion -> BufferRegion.，实际上还是extend = 1,1,64
            # 
            # 1D region -> 3D region with matching tail extent.
            # T.copy(A_128x128x128_global[0, 0, 0:64], C_128x128x32_shared[0:1, 0:1, 0:64])

            # 【rule】 extent(!=1)的数量不同
            # bufferreion 扩写后都要min原始extent，所以结果就是基本不变
            # T.copy(A_128x128x128_global[0:32, 0:16, 0:64], C_128x128x32_shared[0:32, 0:1, 0:64])
            # T.copy(A[0:32, 0:1, 0:64], A_shared1[0:32, 0:16, 0:64])

            # T.copy(A[0:32, 0:16, 0:64], A_shared1[0:16, 0:1, 0:16])

            # T.copy(A[0:128, 0:128, 0:128], A_shared1[0:16, 0:1, 0:16])

            # T.copy(E_128x128_shared[:16, :16], A_128x128x128_global[1, :16, :16])

            # T.copy(A_128x128x128_global[0:16, 0:16, 0:16], E_128x128_shared)

            # 【rule】rsam,转为tile操作
            # T.copy(C_128x128x32_shared[0:16, 0:32, 0:32], D_128x128x32_shared[0:16, 0:32, 0:32])
            # T.copy(A_shared1, A_shared2)

    return mma_3times_kernel


mma_3times_kernel = mma_3times_single_thread(M=128, N=128, K=128)


def codegen_sunmmio_suvm_from_kernel(kernel):
    # sunmmio的处理pass，可以替换成piepeline的 compile_test
    # _, device_mod = compile_with_log(
    #     kernel,
    #     target="Sunmmio",
    #     log_file="t_codegen_tmp",
    #     split_flag=False,
    #     origin_flag=False,
    # )
    # print("device_mod:")
    # print(device_mod.show())
    # target = determine_target("Sunmmio", return_object=True)
    # builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    # return builder(device_mod, target, "suvm").inspect_source()
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

    host_mod, device_mod = compile_test(kernel, target="Sunmmio")

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
    print(f"Saved device TIR to {device_tir_path}")
    print(f"Saved SUVM MLIR to {suvm_mlir_path}")


def t1():
    codegen_sunmmio_suvm_from_kernel(mma_3times_kernel)
    # print("codegen_str:")
    # print(codegen_str)


if __name__ == "__main__":
    t1()
