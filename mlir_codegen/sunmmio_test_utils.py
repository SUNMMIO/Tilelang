import contextlib
import io
import os
import shutil
import subprocess
from pathlib import Path

import tilelang


def _write_text(log_dir, filename, text):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        if text and not text.endswith("\n"):
            f.write("\n")
    return path


def render_ast(mod):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        tilelang.analysis.ASTPrinter()(mod)
    return stream.getvalue()


def save_final_ast(log_dir, mod, filename="final_ast.txt", echo=False):
    ast_text = render_ast(mod)
    path = _write_text(log_dir, filename, ast_text)
    if echo:
        print(ast_text, end="" if ast_text.endswith("\n") else "\n")
    print(f"Saved final AST to {path}")
    return path


def save_final_mlir(log_dir, mlir_source, filename="final_mlir.mlir", echo=False):
    path = _write_text(log_dir, filename, mlir_source)
    if echo:
        print(mlir_source, end="" if mlir_source.endswith("\n") else "\n")
    print(f"Saved final MLIR to {path}")
    return path


def _find_npuir_opt():
    override = os.environ.get("NPUIR_OPT")
    if override:
        return override

    repo_root = Path(__file__).resolve().parents[1]
    build_tool = repo_root / "build" / "bin" / "npuir-opt"
    if build_tool.exists():
        return str(build_tool)

    return shutil.which("npuir-opt")


def _run_npuir_opt(cmd, log_dir, stdout_name, stderr_name):
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    _write_text(log_dir, stdout_name, result.stdout)
    _write_text(log_dir, stderr_name, result.stderr)
    return result


def verify_final_mlir(log_dir, mlir_path):
    if os.environ.get("TL_SUNMMIO_SKIP_NPUIR_VERIFY", "0") == "1":
        print("Skipped npuir-opt verification because TL_SUNMMIO_SKIP_NPUIR_VERIFY=1")
        return

    npuir_opt = _find_npuir_opt()
    if not npuir_opt:
        raise FileNotFoundError("npuir-opt not found. Build it with: ninja -C build npuir-opt")

    verified_path = os.path.join(log_dir, "final_mlir.verified.mlir")
    verify_cmd = [npuir_opt, "--verify-each", mlir_path, "-o", verified_path]
    result = _run_npuir_opt(
        verify_cmd,
        log_dir,
        "npuir_opt_verify.stdout.log",
        "npuir_opt_verify.stderr.log",
    )
    if result.returncode != 0:
        raise RuntimeError(f"npuir-opt --verify-each failed for {mlir_path}. See {log_dir}")
    print(f"npuir-opt --verify-each passed: {verified_path}")

    if os.environ.get("TL_SUNMMIO_DEVICE_VALIDATE", "0") != "1":
        return

    device_verified_path = os.path.join(log_dir, "final_mlir.device_verified.mlir")
    device_cmd = [
        npuir_opt,
        "--suvm-device-validate",
        "--verify-each",
        mlir_path,
        "-o",
        device_verified_path,
    ]
    device_result = _run_npuir_opt(
        device_cmd,
        log_dir,
        "npuir_opt_device_validate.stdout.log",
        "npuir_opt_device_validate.stderr.log",
    )
    if device_result.returncode == 0:
        print(f"npuir-opt --suvm-device-validate passed: {device_verified_path}")
        return

    message = f"npuir-opt --suvm-device-validate failed for {mlir_path}. See {log_dir}"
    if os.environ.get("TL_SUNMMIO_DEVICE_VALIDATE_STRICT", "0") == "1":
        raise RuntimeError(message)
    print(message)
