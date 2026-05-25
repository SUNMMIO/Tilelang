import sys
import contextlib
from pathlib import Path

# 防止调用到官方tileLang
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tilelang
from tilelang import tvm
from tilelang.engine.lower import canon_target_host
from tilelang.engine.phase import PreLowerSemanticCheck, LowerAndLegalize, OptimizeForTarget
from tilelang.utils.target import determine_target, target_is_sunmmio
import tilelang.language as T

# 替换成pipeline文件的compile_test
# from testing.my_test.python.comile_sunmmio.compile_with_log import compile_with_log

def compile_test_local(func_or_mod, target="Sunmmio"):
    target = determine_target(target, return_object=True)
    target_host = tvm.target.Target.canon_target(canon_target_host(target, None))
    target_with_host = tvm.target.Target(target, target_host)

    if isinstance(func_or_mod, tvm.tir.PrimFunc):
        func = (
            func_or_mod.with_attr("global_symbol", "main")
            .with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
            .with_attr("tir.is_global_func", True)
        )
        mod = tvm.IRModule({"main": func})
    else:
        mod = func_or_mod

    with tvm.target.Target(target_with_host):
        # Before lowering, do semantic check
        PreLowerSemanticCheck(mod)
        # Phase 1: Lower and legalize the IR
        mod = LowerAndLegalize(mod, target_with_host)
        # Phase 2: Optimize the IR for the target
        mod = OptimizeForTarget(mod, target_with_host)

    host_funcs = {}
    device_funcs = {}

    for gv, func in mod.functions.items():
        attrs = getattr(func, "attrs", None)
        target_attr = attrs.get("target", None) if attrs else None
        is_sunmmio_device = (
            target_attr is not None
            and target_is_sunmmio(target_attr)
            and not target_attr.host
        )

        if is_sunmmio_device:
            device_funcs[gv] = (
                func.with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
                .with_attr("tir.is_global_func", True)
            )
        else:
            host_funcs[gv] = func

    return tvm.IRModule(host_funcs), tvm.IRModule(device_funcs)

def mma_3times_single_thread(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32, dtype="float16"):

    @T.prim_func
    def mma_3times_kernel(
        A: T.Tensor((M, N, K), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_like_share = T.alloc_shared((M, N, K), dtype)
            A_shared1 = T.alloc_shared((block_M, block_N, block_K), dtype)
            A_shared2 = T.alloc_shared((block_M, block_N, block_K), dtype)
            A_shared3 = T.alloc_shared((block_N, block_K), dtype)

            # 完整buffer时，要求shape完全一致
            # 【base_rule】 type: buffer , buffer_load, buffer_region
            # 【base_rule】extent: shape, None(补1)， region_extend
            # 【rule】buffer copy: 校验shape,不一致，报错,一致，extend = shape
            # 生成AST:
            # T.copy(T.region(A[0, 0, 0], 1, 128, 128, 128),
            #        T.region(A_like_share[0, 0, 0], 2, 128, 128, 128))
            # 1表示read，2表示write
            # T.copy(A, A_shared1)
            T.copy(A, A_like_share)

            # 标量copy变为普通store操作，和官方实现基本一致：
            # seq1(Stmt): BufferStore
            #   buffer: A_shared1
            #   value: A[1, 2, 3]
            #   indices: [0, 0, 0]
            # 【rule】Scalar BufferLoad -> scalar BufferLoad, lowered as direct store.
            # T.copy(A[1, 2, 3], A_shared1[0, 0, 0])

            # 第一项被当作起始地址，extent怎么算？
            # src_extent = [1, 1, 1], dst_extent = [128, 128, 32]
            # 随后src会被扩成与dst一样，但是这样就会越界
            # 【rule】 extent(!=1)的数量相同的，不校验
            # BufferLoad -> Buffer.
            # T.copy(A[1, 2, 3], A_shared1)

            # src和dst的region extent不一样，且看不出真实硬件行为
            # 由于16*16*16的区间，在A中的索引是从0到247695，所以TIR中src长度为247695，dst长度为524288
            # BufferRegion -> buffer
            # T.copy(A[0:16, 0:16, 0:16], A_shared1)

            # 会把dst extent补成src 的 extent, 语义：A[0:16, 0:16, 0:16] -> A_shared1[0:16, 0:16, 0:16]
            # 由于A 和 A_shared1的shape不同，导致span不一致
            # BufferRegion -> Bufferload
            # T.copy(A[0:16, 0:16, 0:16], A_shared1[0, 0, 0])

            # 依旧会导致 bounding span不一致
            # BufferRegion -> BufferRegion.
            # T.copy(A[0:16, 0:16, 0:16], A_shared1[0:16, 0:16, 0:16])
            # T.copy(A[0:32, 0:32, 0:32], A_shared1[0:16, 0:16, 0:16]) # 这句在官方那会直接报错

            # 特殊的BufferRegion -> BufferRegion.，实际上还是extend = 1,1,64
            # 
            # 1D region -> 3D region with matching tail extent.
            # T.copy(A[0, 0, 0:64], A_shared1[0:1, 0:1, 0:64])

            # 【rule】 extent(!=1)的数量不同
            # bufferreion 扩写后都要min原始extent，所以结果就是基本不变
            # T.copy(A[0:32, 0:16, 0:64], A_shared1[0:32, 0:1, 0:64])
            # T.copy(A[0:32, 0:1, 0:64], A_shared1[0:32, 0:16, 0:64])

            # T.copy(A[0:32, 0:16, 0:64], A_shared1[0:16, 0:1, 0:16])

            # T.copy(A[0:128, 0:128, 0:128], A_shared1[0:16, 0:1, 0:16])

            # T.copy(A_shared3[:16, :16], A[1, :16, :16])

            # T.copy(A[0:16, 0:16, 0:16], A_shared3)

            # 【rule】rsam,转为tile操作
            # T.copy(A_shared1[0:16, 0:32, 0:32], A_shared2[0:16, 0:32, 0:32])
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
            with tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_AST_PRINT_ENABLE: True}):
                compile_test_local(kernel, target="Sunmmio")

    host_mod, device_mod = compile_test_local(kernel, target="Sunmmio")

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
