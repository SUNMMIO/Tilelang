import json
import subprocess
from pathlib import Path

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.language.mesh_tensor import MeshReplicationType
from tilelang.layout import make_zz_layout
from tilelang.jit.adapter.sunmmio.libgen import find_npuir_tool
from tilelang.transform import PassConfigKey

from testing.python.sunmmio.common.codegen_validation import (
    validate_sunmmio_codegen_with_npuir_opt,
)
from testing.python.sunmmio.common.compile_pipeline import target


tilelang.env.disable_cache()

_DEBUG_ROOT = Path(__file__).with_name("_debug") / Path(__file__).stem


@target("Sunmmio")
def pipelined_gemm_kernel(
    M=512,
    N=512,
    K=256,
    block_M=128,
    block_N=128,
    block_K=32,
    num_stages=4,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    a_policy = T.MeshShardingPolicy(y=0, replicate=MeshReplicationType.ROW)
    b_policy = T.MeshShardingPolicy(x=1, replicate=MeshReplicationType.COLUMN)
    c_policy = T.MeshShardingPolicy(y=0, x=1)
    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), a_policy, (4, 4), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), b_policy, (4, 4), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), c_policy, (4, 4), accum_dtype, layout=C_layout),
    ):
        with T.Kernel() as _cid:
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[0:block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(B[k * block_K : (k + 1) * block_K, 0:block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)
            T.copy(C_shared, C[0:block_M, 0:block_N])

    return main


def _run_logged(command, output_dir, phase):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    (output_dir / f"{phase}.command.log").write_text(" ".join(str(part) for part in command) + "\n", encoding="utf-8")
    (output_dir / f"{phase}.stdout.log").write_text(result.stdout, encoding="utf-8")
    (output_dir / f"{phase}.stderr.log").write_text(result.stderr, encoding="utf-8")
    return result


def _isolated_global_to_wsram_mlir(version_extent):
    if version_extent is None:
        dst_shape = "32x128"
        dst_layout = "((32, 1), (32, 4)), ((1, 4096), (32, 1024))"
        indices = "%c0, %c0"
        tiled_dims = "0, 1"
    else:
        dst_shape = f"{version_extent}x32x128"
        dst_layout = f"({version_extent}, (32, 1), (32, 4)), (4096, (1, 4096), (32, 1024))"
        indices = "%c0, %c0, %c0"
        tiled_dims = "1, 2"

    src_type = "!suvm.memtensor<256x128xbf16, #suvm.layout<((32, 8), (32, 4)), ((32, 4096), (1, 1024))>, #suvm.memory_space<global>>"
    dst_type = f"!suvm.memtensor<{dst_shape}xbf16, #suvm.layout<{dst_layout}>, #suvm.memory_space<wsram>>"
    return f"""module attributes {{suvm.device_arch = #suvm.device_arch<a4e>}} {{
  func.func @isolated_global_to_wsram(%src: {src_type}) {{
    %dst = suvm.alloc() {{suvm.ping_pong = #suvm.ping_pong<ping>}} : {dst_type}
    %c0 = arith.constant 0 : index
    %src_view = suvm.get_partitioned_tile_view %src
      indices = [%c0, %c0] tiled_dims = [0, 1]
      : {src_type} -> !suvm.tile_view<32x128xbf16>
    %dst_view = suvm.get_partitioned_tile_view %dst
      indices = [{indices}] tiled_dims = [{tiled_dims}]
      : {dst_type} -> !suvm.tile_view<32x128xbf16>
    %token = suvm.copy_async %src_view, %dst_view
      : !suvm.tile_view<32x128xbf16>, !suvm.tile_view<32x128xbf16> -> !suvm.token
    suvm.wait_token %token : !suvm.token
    return
  }}
}}
"""


def test_isolate_wsram_version_axis_address_materialization():
    output_dir = _DEBUG_ROOT / "isolation"
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = {
        "rank2_no_version_axis": None,
        "rank3_version_extent1": 1,
        "rank3_version_extent2": 2,
    }
    summary = {}

    for case_name, version_extent in cases.items():
        mlir_path = output_dir / f"{case_name}.mlir"
        mlir_path.write_text(_isolated_global_to_wsram_mlir(version_extent), encoding="utf-8")
        opt_result = _run_logged(
            [
                str(find_npuir_tool("npuir-opt")),
                str(mlir_path),
                "-suvm-device-validate",
            ],
            output_dir,
            f"{case_name}.npuir_opt_validate",
        )
        llvm_path = output_dir / f"{case_name}.ll"
        compile_result = _run_logged(
            [
                str(find_npuir_tool("npuir-compile")),
                "--target=sunmmio-a4e",
                "--emit=llvm-ir",
                str(mlir_path),
                "-o",
                str(llvm_path),
            ],
            output_dir,
            f"{case_name}.npuir_compile_llvm",
        )
        summary[case_name] = {
            "version_extent": version_extent,
            "npuir_opt_validate_returncode": opt_result.returncode,
            "npuir_compile_llvm_returncode": compile_result.returncode,
            "llvm_ir": str(llvm_path) if llvm_path.exists() else None,
        }

    (output_dir / "00_isolation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    assert all(case["npuir_opt_validate_returncode"] == 0 for case in summary.values())
    assert summary["rank2_no_version_axis"]["npuir_compile_llvm_returncode"] == 0


@pytest.mark.parametrize(
    "pipeline_mode",
    (
        "ilp",
        pytest.param(
            "greedy",
            marks=pytest.mark.xfail(
                strict=True,
                reason="greedy version-axis addresses fail SUVM-to-LLVM materialization",
            ),
        ),
    ),
)
def test_pipeline_ping_pong_codegen_phase_diagnostics(tmp_path, pipeline_mode):
    output_dir = _DEBUG_ROOT / pipeline_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    mlir_filename = f"pipeline_ping_pong_{pipeline_mode}_suvm.mlir"
    src = validate_sunmmio_codegen_with_npuir_opt(
        pipelined_gemm_kernel(),
        tmp_path,
        pass_configs={PassConfigKey.TL_SUNMMIO_PIPELINE_MODE: pipeline_mode},
        mlir_filename=mlir_filename,
        expected_tokens=(
            "suvm.alloc",
            "suvm.copy_async",
            "suvm.tc.mma",
            "suvm.wait_token",
            "suvm.ping_pong",
        ),
        log_ir=True,
        log_dir=_DEBUG_ROOT,
        log_subdir=pipeline_mode,
    )
    assert "#suvm.ping_pong<ping>" in src
    assert "#suvm.ping_pong<pong>" in src

    mlir_path = output_dir / f"{Path(mlir_filename).stem}.mlir.log"
    opt_result = _run_logged(
        [str(find_npuir_tool("npuir-opt")), str(mlir_path), "-suvm-device-validate"],
        output_dir,
        "03_npuir_opt_validate",
    )
    llvm_path = output_dir / f"pipeline_ping_pong_{pipeline_mode}.ll"
    compile_result = _run_logged(
        [
            str(find_npuir_tool("npuir-compile")),
            "--target=sunmmio-a4e",
            "--emit=llvm-ir",
            str(mlir_path),
            "-o",
            str(llvm_path),
        ],
        output_dir,
        "04_npuir_compile_llvm",
    )
    summary = {
        "pipeline_mode": pipeline_mode,
        "npuir_opt_validate_returncode": opt_result.returncode,
        "npuir_compile_llvm_returncode": compile_result.returncode,
        "llvm_ir": str(llvm_path) if llvm_path.exists() else None,
        "failed_phase": (
            "npuir-opt-device-validate"
            if opt_result.returncode != 0
            else "npuir-compile-suvm-to-llvm"
            if compile_result.returncode != 0
            else None
        ),
    }
    (output_dir / "00_phase_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    assert opt_result.returncode == 0
    assert compile_result.returncode == 0


if __name__ == "__main__":
    tilelang.testing.main()
