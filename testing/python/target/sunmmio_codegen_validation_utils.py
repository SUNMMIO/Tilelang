import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest
from compile_pipeline import compile_test
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target


CODEGEN_BACKEND = "suvm"
NPUIR_OPT_ENV = "NPUIR_OPT"
PRINT_ENV = "SUNMMIO_TEST_PRINT"
BASE_EXPECTED_TOKENS = (
    "module",
    "suvm.device_arch",
    "func.func @",
)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _print_enabled(value: bool | None) -> bool:
    if value is not None:
        return value
    return _env_flag(PRINT_ENV)


def _repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "CMakeLists.txt").exists() and (parent / "3rdparty" / "NPU-IR").exists():
            return parent
    raise RuntimeError(f"failed to locate repo root from {path}")


def find_npuir_opt() -> Path:
    candidates = []
    if os.getenv(NPUIR_OPT_ENV):
        candidates.append(Path(os.environ[NPUIR_OPT_ENV]))
    candidates.append(_repo_root() / "build" / "bin" / "npuir-opt")

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    candidate_text = "\n".join(str(candidate) for candidate in candidates)
    pytest.fail(f"npuir-opt executable not found. Checked:\n{candidate_text}")


def assert_source_contains(src: str, tokens: Sequence[str]) -> None:
    missing = [token for token in tokens if token not in src]
    assert not missing, f"missing expected SUVM MLIR tokens: {missing}\n{src}"


def lower_sunmmio_kernel_to_device_tir(
    kernel,
    *,
    print_ir: bool | None = None,
):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    _, device_mod = compile_test(kernel, target=target, remove_header=True)

    assert isinstance(device_mod, tvm.IRModule)
    assert device_mod.get_global_vars()
    script = device_mod.script()
    assert script.strip()

    if _print_enabled(print_ir):
        print("===================== Lowered SunMMIO Device TIR =====================")
        print(script)

    return device_mod


def codegen_sunmmio_suvm_mlir(
    device_mod,
    *,
    expected_tokens: Sequence[str] = (),
    print_ir: bool | None = None,
) -> str:
    target = determine_target("Sunmmio", return_object=True)
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    runtime_mod = builder(device_mod, target, CODEGEN_BACKEND)
    src = runtime_mod.inspect_source()

    assert src.strip()
    assert_source_contains(src, (*BASE_EXPECTED_TOKENS, *expected_tokens))

    if _print_enabled(print_ir):
        print("===================== SunMMIO SUVM MLIR =====================")
        print(src)

    return src


def validate_suvm_mlir_with_npuir_opt(
    src: str,
    tmp_path: Path,
    *,
    mlir_filename: str = "generated_suvm.mlir",
    opt_args: Sequence[str] = ("-suvm-device-validate",),
    print_output: bool | None = None,
):
    npuir_opt = find_npuir_opt()
    mlir_path = Path(tmp_path) / mlir_filename
    mlir_path.write_text(src, encoding="utf-8")

    command = [str(npuir_opt), str(mlir_path), *opt_args]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if _print_enabled(print_output):
        print("===================== npuir-opt command =====================")
        print(" ".join(command))
        print("===================== npuir-opt stderr =====================")
        print(result.stderr)

    assert result.returncode == 0, (
        "npuir-opt SUVM validation failed\n"
        f"command: {' '.join(command)}\n"
        f"mlir: {mlir_path}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return result


def validate_sunmmio_codegen_with_npuir_opt(
    kernel,
    tmp_path: Path,
    *,
    mlir_filename: str = "generated_suvm.mlir",
    expected_tokens: Sequence[str] = (),
    opt_args: Sequence[str] = ("-suvm-device-validate",),
    print_ir: bool | None = None,
    print_opt: bool | None = None,
) -> str:
    device_mod = lower_sunmmio_kernel_to_device_tir(
        kernel,
        print_ir=print_ir,
    )
    src = codegen_sunmmio_suvm_mlir(
        device_mod,
        expected_tokens=expected_tokens,
        print_ir=print_ir,
    )
    validate_suvm_mlir_with_npuir_opt(
        src,
        tmp_path,
        mlir_filename=mlir_filename,
        opt_args=opt_args,
        print_output=print_opt,
    )
    return src
