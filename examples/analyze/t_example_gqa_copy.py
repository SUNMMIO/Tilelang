import argparse
import contextlib
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
COMPILE_PIPELINE_DIR = REPO_ROOT / "testing" / "python" / "compile_pipeline"
if str(COMPILE_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(COMPILE_PIPELINE_DIR))

import tilelang
from tilelang import tvm
from tilelang.carver.arch import driver
from tilelang.layout import make_zz_layout
from tilelang.utils.target import determine_target
import tilelang.language as T
from compile_pipeline import compile_test


DEFAULT_LOG_PASSES = [
    "BindTarget",
    "InferSramScope",
    "LegalizeSunmmioDataPath",
    "SunmmioLayoutInference",
    "LowerTileOp",
    "DeviceMod",
]


def flashattn(batch, heads, seq_len, dim, groups=1, block_M=64, block_N=64, num_stages=0):
    if heads % groups != 0:
        raise ValueError(f"heads must be divisible by groups, got heads={heads}, groups={groups}")

    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    shard_policy = T.MeshShardingPolicy(y=0, x=2)
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    Q_layout = make_zz_layout(q_shape, [1, 3], (32, 32))
    K_layout = make_zz_layout(kv_shape, [1, 3], (32, 32))
    V_layout = make_zz_layout(kv_shape, [1, 3], (32, 32))
    O_layout = make_zz_layout(q_shape, [1, 3], (32, 32))

    @T.prim_func
    def main(
        Q: T.MeshTensor(q_shape, shard_policy, device_mesh_config, dtype, layout=Q_layout),
        K: T.MeshTensor(kv_shape, shard_policy, device_mesh_config, dtype, layout=K_layout),
        V: T.MeshTensor(kv_shape, shard_policy, device_mesh_config, dtype, layout=V_layout),
        Output: T.MeshTensor(q_shape, shard_policy, device_mesh_config, dtype, layout=O_layout),
    ):
        with T.Kernel(ncores) as _cid:
            # Get sharded logical shapes
            sharded_batch = Q.shape[0]
            sharded_heads = Q.shape[2]

            # Declare shared memory buffers
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_shared([block_M, block_N], accum_dtype)
            # RSRAM-resident dtype-cast staging for the PV gemm input. A
            # Sunmmio DMA copy cannot change dtype, so the fp32->dtype cast
            # is done here by the Tile unit (in RSRAM) and only then DMA'd
            # into acc_s_cast (ASRAM, the PV gemm's A operand).
            acc_s_cast_local = T.alloc_shared([block_M, block_N], dtype)
            acc_s_cast = T.alloc_shared([block_M, block_N], dtype)
            acc_o = T.alloc_shared([block_M, dim], accum_dtype)
            scores_max = T.alloc_shared([block_M], accum_dtype)
            scores_max_prev = T.alloc_shared([block_M], accum_dtype)
            scores_scale = T.alloc_shared([block_M], accum_dtype)
            scores_sum = T.alloc_shared([block_M], accum_dtype)
            logsum = T.alloc_shared([block_M], accum_dtype)

            # Each core iterates its own sharded work domain with plain
            # nested loops. The MeshTensor sharding already distributes data
            # (batch / heads) across the core mesh, so no persistent
            # core-distribution loop is needed here.
            for bz in T.serial(sharded_batch):
                for by in T.serial(sharded_heads):
                    for bx in T.serial(T.ceildiv(seq_len, block_M)):
                        T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
                        T.fill(acc_o, 0)
                        T.fill(logsum, 0)
                        T.fill(scores_max, -T.infinity(accum_dtype))

                        loop_range = T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N))

                        # loop over K/V blocks
                        for k in T.Pipelined(loop_range, num_stages=num_stages):
                            T.copy(K[bz, k * block_N : (k + 1) * block_N, by // groups, :], K_shared)
                            for i, j in T.Tiles([block_M, block_N]):
                                acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype))
                            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)

                            T.copy(scores_max, scores_max_prev)
                            T.fill(scores_max, -T.infinity(accum_dtype))
                            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                            for i in T.Tiles([block_M]):
                                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                            for i in T.Tiles([block_M]):
                                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                            for i, j in T.Tiles([block_M, block_N]):
                                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                            T.reduce_sum(acc_s, scores_sum, dim=1)
                            for i in T.Tiles([block_M]):
                                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                            # Cast fp32 probabilities -> dtype with the Tile
                            # unit in RSRAM, then DMA the same-dtype result
                            # into ASRAM.
                            for i, j in T.Tiles([block_M, block_N]):
                                acc_s_cast_local[i, j] = acc_s[i, j]
                            T.copy(acc_s_cast_local, acc_s_cast)

                            for i, j in T.Tiles([block_M, dim]):
                                acc_o[i, j] *= scores_scale[i]

                            T.copy(V[bz, k * block_N : (k + 1) * block_N, by // groups, :], V_shared)
                            T.gemm(acc_s_cast, V_shared, acc_o)

                        for i, j in T.Tiles([block_M, dim]):
                            acc_o[i, j] /= logsum[i]
                        T.copy(acc_o, O_shared)
                        T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main


def codegen_sunmmio_suvm_from_kernel(kernel, *, log_passes=None, show_meta=False):
    output_dir = REPO_ROOT / "tmp"
    output_dir.mkdir(exist_ok=True)

    kernel_tir_path = output_dir / "t_example_gqa_copy_kernel_tir.py"
    ast_path = output_dir / "t_example_gqa_copy_ast.txt"
    pass_dump_dir = output_dir / "t_example_gqa_copy_passes"
    device_tir_path = output_dir / "t_example_gqa_copy_device_tir.py"
    suvm_mlir_path = output_dir / "t_example_gqa_copy_suvm.mlir"
    suvm_error_path = output_dir / "t_example_gqa_copy_suvm_error.txt"

    kernel_tir_path.write_text(kernel.script(show_meta=show_meta), encoding="utf-8")

    with ast_path.open("w", encoding="utf-8") as f:
        with contextlib.redirect_stdout(f):
            compile_test(
                kernel,
                target="Sunmmio",
                pass_configs={tilelang.PassConfigKey.TL_AST_PRINT_ENABLE: True},
            )

    host_mod, device_mod = compile_test(
        kernel,
        target="Sunmmio",
        log_pass_output=True,
        show_meta=show_meta,
        log_dir=str(pass_dump_dir),
        log_passes=log_passes,
    )
    del host_mod

    device_tir_path.write_text(device_mod.script(show_meta=show_meta), encoding="utf-8")

    target = determine_target("Sunmmio", return_object=True)
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    try:
        try:
            src = builder(device_mod, target, "suvm").inspect_source()
        except TypeError:
            src = builder(device_mod, target).inspect_source()
    except Exception:
        suvm_error_path.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"Saved kernel TIR to {kernel_tir_path}")
        print(f"Saved AST to {ast_path}")
        print(f"Saved pass dumps to {pass_dump_dir}")
        print(f"Saved device TIR to {device_tir_path}")
        print(f"Saved SUVM codegen error to {suvm_error_path}")
        raise

    suvm_mlir_path.write_text(src, encoding="utf-8")

    print(f"Saved kernel TIR to {kernel_tir_path}")
    print(f"Saved AST to {ast_path}")
    print(f"Saved pass dumps to {pass_dump_dir}")
    print(f"Saved device TIR to {device_tir_path}")
    print(f"Saved SUVM MLIR to {suvm_mlir_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile the SunMMIO GQA copy example and dump copy-related IR passes."
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--num-stages", type=int, default=0)
    parser.add_argument(
        "--log-pass",
        action="append",
        dest="log_passes",
        default=None,
        help="Pass name to dump. Can be repeated. Defaults to copy-related passes.",
    )
    parser.add_argument(
        "--all-passes",
        action="store_true",
        help="Dump every pass, including the Initial Mod sections.",
    )
    parser.add_argument("--show-meta", action="store_true", help="Keep TVM metadata in dumped TIR.")
    return parser.parse_args()


def main():
    args = parse_args()
    log_passes = None if args.all_passes else (args.log_passes or DEFAULT_LOG_PASSES)
    config_summary = (
        "Active GQA config: "
        f"batch={args.batch}, heads={args.heads}, seq_len={args.seq_len}, dim={args.dim}, "
        f"groups={args.groups}, block_M={args.block_m}, block_N={args.block_n}, "
        f"num_stages={args.num_stages}"
    )
    print(config_summary)
    if log_passes is None:
        print("Logging passes: all")
    else:
        print(f"Logging passes: {', '.join(log_passes)}")

    try:
        kernel = flashattn(
            args.batch,
            args.heads,
            args.seq_len,
            args.dim,
            groups=args.groups,
            block_M=args.block_m,
            block_N=args.block_n,
            num_stages=args.num_stages,
        )
        codegen_sunmmio_suvm_from_kernel(kernel, log_passes=log_passes, show_meta=args.show_meta)
    except Exception:
        output_dir = REPO_ROOT / "tmp"
        output_dir.mkdir(exist_ok=True)
        error_path = output_dir / "t_example_gqa_copy_error.txt"
        error_path.write_text(config_summary + "\n\n" + traceback.format_exc(), encoding="utf-8")
        print(f"Saved compile error to {error_path}")
        raise


if __name__ == "__main__":
    main()
