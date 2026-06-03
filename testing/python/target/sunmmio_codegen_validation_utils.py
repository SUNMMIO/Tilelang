import inspect
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
# SUNMMIO_TEST_LOG_IR=1 writes kernel/TIR/MLIR snapshots for codegen tests.
# SUNMMIO_TEST_LOG_DIR overrides the default log root:
# testing/python/target/_debug/sunmmio_codegen_logs/<test_file>/
LOG_IR_ENV = "SUNMMIO_TEST_LOG_IR"
LOG_DIR_ENV = "SUNMMIO_TEST_LOG_DIR"
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


def _log_ir_enabled(value: bool | None) -> bool:
    if value is not None:
        return value
    return _env_flag(LOG_IR_ENV)


def _default_codegen_log_root() -> Path:
    return Path(__file__).resolve().parent / "_debug" / "sunmmio_codegen_logs"


def _current_test_file_stem() -> str:
    current_test = os.getenv("PYTEST_CURRENT_TEST")
    if current_test:
        test_path = current_test.split("::", 1)[0]
        stem = Path(test_path).stem
        if stem:
            return stem

    this_file = Path(__file__).resolve()
    for frame_info in inspect.stack():
        frame_path = Path(frame_info.filename).resolve()
        if frame_path == this_file:
            continue
        if frame_path.name.startswith("test_") and frame_path.suffix == ".py":
            return frame_path.stem

    return "sunmmio_codegen"


def _resolve_codegen_log_dir(log_dir: Path | str | None, log_subdir: str | None) -> Path:
    if log_dir is not None:
        root = Path(log_dir)
    elif os.getenv(LOG_DIR_ENV):
        root = Path(os.environ[LOG_DIR_ENV])
    else:
        root = _default_codegen_log_root()
    return root / (log_subdir or _current_test_file_stem())


def write_sunmmio_codegen_logs(
    *,
    case_name: str,
    kernel=None,
    tir_mod=None,
    kernel_src: str | None = None,
    tir_src: str | None = None,
    mlir_src: str | None = None,
    log_ir: bool | None = None,
    log_dir: Path | str | None = None,
    log_subdir: str | None = None,
) -> tuple[Path, ...]:
    if not _log_ir_enabled(log_ir):
        return ()

    if kernel_src is None and kernel is not None:
        kernel_src = kernel.script(show_meta=True) if hasattr(kernel, "script") else str(kernel)
    if tir_src is None and tir_mod is not None:
        tir_src = tir_mod.script(show_meta=True)
    if kernel_src is None and tir_src is None and mlir_src is None:
        return ()

    output_dir = _resolve_codegen_log_dir(log_dir, log_subdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(case_name).stem or "generated_suvm"
    written_paths = []
    if kernel_src is not None:
        kernel_path = output_dir / f"{stem}.kernel.log"
        kernel_path.write_text(kernel_src, encoding="utf-8")
        written_paths.append(kernel_path)
    if tir_src is not None:
        tir_path = output_dir / f"{stem}.tir.log"
        tir_path.write_text(tir_src, encoding="utf-8")
        written_paths.append(tir_path)
    if mlir_src is not None:
        mlir_path = output_dir / f"{stem}.mlir.log"
        mlir_path.write_text(mlir_src, encoding="utf-8")
        written_paths.append(mlir_path)

    return tuple(written_paths)


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
    log_ir: bool | None = None,
    log_dir: Path | str | None = None,
    log_subdir: str | None = None,
) -> str:
    log_enabled = _log_ir_enabled(log_ir)
    if log_enabled:
        write_sunmmio_codegen_logs(
            case_name=mlir_filename,
            kernel=kernel,
            log_ir=True,
            log_dir=log_dir,
            log_subdir=log_subdir,
        )
    device_mod = lower_sunmmio_kernel_to_device_tir(
        kernel,
        print_ir=print_ir,
    )
    if log_enabled:
        write_sunmmio_codegen_logs(
            case_name=mlir_filename,
            tir_mod=device_mod,
            log_ir=True,
            log_dir=log_dir,
            log_subdir=log_subdir,
        )
    src = codegen_sunmmio_suvm_mlir(
        device_mod,
        expected_tokens=expected_tokens,
        print_ir=print_ir,
    )
    if log_enabled:
        write_sunmmio_codegen_logs(
            case_name=mlir_filename,
            mlir_src=src,
            log_ir=True,
            log_dir=log_dir,
            log_subdir=log_subdir,
        )
    validate_suvm_mlir_with_npuir_opt(
        src,
        tmp_path,
        mlir_filename=mlir_filename,
        opt_args=opt_args,
        print_output=print_opt,
    )
    return src
