"""Persistent edit session for TileLang-generated Sunmmio SUVM MLIR."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import difflib
import json
from pathlib import Path
import subprocess
from typing import Any

import tilelang
from tilelang import tvm
from tilelang.engine.param import KernelParam
from tilelang.utils.target import determine_target

from .adapter import SunmmioKernelSuDeckAdapter, SunmmioSunsimKernelAdapter
from .abi import SunmmioKernelABI
from .libgen import (
    SUNMMIO_KERNEL_ELF_FILE,
    SUNMMIO_KERNEL_LLVM_IR_FILE,
    SUNMMIO_KERNEL_MLIR_FILE,
    SUNMMIO_KERNEL_OBJ_FILE,
    SUNMMIO_SUDECK_LAUNCHER_CPP_FILE,
    SUNMMIO_SUDECK_LAUNCHER_LIB_FILE,
    SUNMMIO_SUDECK_LAUNCH_MODULE_FILE,
    SunmmioKernelArtifact,
    SunmmioSuDeckLibraryGenerator,
    SunmmioSunsimLibraryGenerator,
    find_npuir_tool,
)


ORIGINAL_MLIR = "kernel.original.mlir"
EDITED_MLIR = "kernel.edited.mlir"
DEVICE_TIR = "kernel.tir"
DEVICE_VALIDATED_MLIR = "kernel.device-validated.mlir"
LOWERED_MLIR = "kernel.lowered.mlir"
ABI_FILE = "abi.json"
DIFF_FILE = "kernel.diff"
MANIFEST_FILE = "manifest.json"
MANIFEST_SCHEMA_VERSION = 2

_COMPILED_ARTIFACTS = (
    DEVICE_VALIDATED_MLIR,
    LOWERED_MLIR,
    SUNMMIO_KERNEL_MLIR_FILE,
    SUNMMIO_KERNEL_LLVM_IR_FILE,
    SUNMMIO_KERNEL_ELF_FILE,
    SUNMMIO_KERNEL_OBJ_FILE,
    DIFF_FILE,
    "main_thunk.cpp",
    "main_thunk.o",
    "CMakeLists.txt",
    "device_sudeck.ld",
    SUNMMIO_SUDECK_LAUNCHER_CPP_FILE,
    SUNMMIO_SUDECK_LAUNCHER_LIB_FILE,
    SUNMMIO_SUDECK_LAUNCH_MODULE_FILE,
)


@dataclass(frozen=True)
class SunmmioSuvmArtifacts:
    """Stable artifact paths for a manual SUVM edit session."""

    work_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "work_dir", Path(self.work_dir).resolve())

    def path(self, name: str) -> Path:
        return self.work_dir / name

    @property
    def edited_mlir(self) -> Path:
        return self.path(EDITED_MLIR)

    @property
    def diff(self) -> Path:
        return self.path(DIFF_FILE)

    @property
    def llvm_ir(self) -> Path:
        return self.path(SUNMMIO_KERNEL_LLVM_IR_FILE)

    @property
    def elf(self) -> Path:
        return self.path(SUNMMIO_KERNEL_ELF_FILE)

    def managed_paths(self) -> list[Path]:
        sources = (ORIGINAL_MLIR, EDITED_MLIR, DEVICE_TIR, ABI_FILE, MANIFEST_FILE)
        return [self.path(name) for name in (*sources, *_COMPILED_ARTIFACTS)]


@dataclass
class SunmmioSuvmEditSession:
    """Emit editable SUVM MLIR and compile it into a callable Sunmmio kernel.

    A minimal two-stage workflow looks like this::

        from pathlib import Path

        from tilelang.jit.adapter.sunmmio.suvm_edit_session import SunmmioSuvmEditSession

        session = SunmmioSuvmEditSession(Path("manual_suvm"))

        # Stage 1: lower a @tilelang.jit kernel only as far as SUVM MLIR.
        artifacts = session.emit(jit_kernel.get_tir(...))
        print(f"Edit this file: {artifacts.edited_mlir}")

        # Stop here and manually edit manual_suvm/kernel.edited.mlir.

        # Stage 2: in a later run, reuse the same directory. This validates and
        # compiles the edited MLIR, then accepts normal torch_sunmmio tensors.
        kernel = session.compile()
        kernel(a_dev, b_dev, c_dev)

    Do not call :meth:`emit` again after editing: a new emit starts a fresh
    session input and archives the previous ``kernel.edited.mlir``.
    """

    work_dir: Path
    target: str | tvm.target.Target = "sunmmio"
    opt_level: int = 3
    timeout: float | None = 240.0

    def __post_init__(self) -> None:
        self.work_dir = Path(self.work_dir).resolve()

    @property
    def artifacts(self) -> SunmmioSuvmArtifacts:
        return SunmmioSuvmArtifacts(self.work_dir)

    def emit(self, kernel: tvm.tir.PrimFunc | tvm.IRModule) -> SunmmioSuvmArtifacts:
        """Lower a kernel, archive the previous edit, and write a fresh editable MLIR."""
        if not isinstance(kernel, (tvm.tir.PrimFunc, tvm.IRModule)):
            raise TypeError("SunmmioSuvmEditSession.emit expects a PrimFunc or IRModule.")

        resolved_target = determine_target(self.target, return_object=True)
        with tvm.transform.PassContext(opt_level=self.opt_level), resolved_target:
            lowered = tilelang.lower(
                kernel,
                target=resolved_target,
                enable_host_codegen=False,
                enable_device_compile=False,
            )
        if not lowered.kernel_source or not lowered.kernel_source.strip():
            raise RuntimeError("Sunmmio lowering produced no SUVM MLIR.")
        if lowered.device_mod is None:
            raise RuntimeError("Sunmmio lowering produced no device TIR.")

        params = lowered.params or _kernel_params(kernel)
        abi = SunmmioKernelABI.from_modules(
            func_or_mod=kernel,
            host_mod=lowered.host_mod,
            device_mod=lowered.device_mod,
            params=params,
        )
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "target": str(resolved_target),
            "opt_level": self.opt_level,
            "parameters": [
                {
                    "name": name,
                    "kind": "scalar" if param.is_scalar() else "tensor",
                    "dtype": str(param.dtype),
                    "shape": [str(dim) for dim in param.shape],
                }
                for name, param in zip(abi.public_param_names, params)
            ],
        }

        artifacts = self.artifacts
        _prepare_emit(artifacts)
        artifacts.path(ORIGINAL_MLIR).write_text(lowered.kernel_source, encoding="utf-8")
        artifacts.edited_mlir.write_text(lowered.kernel_source, encoding="utf-8")
        artifacts.path(DEVICE_TIR).write_text(lowered.device_mod.script(), encoding="utf-8")
        _write_json(artifacts.path(ABI_FILE), abi.to_json_dict())
        _write_json(artifacts.path(MANIFEST_FILE), manifest)
        return artifacts

    def compile(self) -> SunmmioKernelSuDeckAdapter:
        """Validate and compile the edited MLIR for the regular SuDeck runtime."""
        artifacts, manifest, abi, edited_source = self._prepare_compile()
        target = determine_target(manifest["target"], return_object=True)
        generator = SunmmioSuDeckLibraryGenerator(target, abi.kernel_name)
        generator.update_launcher_specs(
            [(name, "tensor" if dtype == "handle" else dtype) for name, dtype in zip(abi.device_param_names, abi.device_param_dtypes)]
        )
        generator.update_mlir_source(edited_source)
        tir_path = artifacts.path(DEVICE_TIR)
        if tir_path.is_file():
            generator.update_device_tir_source(tir_path.read_text(encoding="utf-8"))
        generator.compile_lib(timeout=self.timeout, output_dir=artifacts.work_dir)
        if generator.artifact is None:
            raise RuntimeError("Sunmmio SuDeck generator produced no artifact.")
        return SunmmioKernelSuDeckAdapter.from_compiled_artifact(
            target=target,
            abi=abi,
            kernel_lib_path=generator.artifact.elf_path,
        )

    def compile_sunsim(self) -> SunmmioSunsimKernelAdapter:
        """Validate the edited MLIR and compile it into a callable sunsim ELF."""
        artifacts, manifest, abi, edited_source = self._prepare_compile()
        target = determine_target(manifest["target"], return_object=True)
        generator = SunmmioSunsimLibraryGenerator(target, abi.kernel_name)
        generator.update_mlir_source(edited_source)
        tir_path = artifacts.path(DEVICE_TIR)
        if tir_path.is_file():
            generator.update_device_tir_source(tir_path.read_text(encoding="utf-8"))
        generator.compile_lib(timeout=self.timeout, output_dir=artifacts.work_dir)
        if generator.artifact is None:
            raise RuntimeError("Sunmmio sunsim generator produced no artifact.")
        _validate_sunsim_elf_abi(generator.artifact, abi)
        return SunmmioSunsimKernelAdapter.from_compiled_artifact(
            target=target,
            abi=abi,
            parameter_kinds=_parameter_kinds(manifest, abi),
            kernel_lib_path=generator.artifact.elf_path,
        )

    def _prepare_compile(
        self,
    ) -> tuple[SunmmioSuvmArtifacts, dict[str, Any], SunmmioKernelABI, str]:
        artifacts = self.artifacts
        manifest = _load_manifest(artifacts)
        abi = SunmmioKernelABI.from_json_dict(_read_json(artifacts.path(ABI_FILE), "Sunmmio ABI metadata"))
        edited_source = _read_nonempty(artifacts.edited_mlir, "edited SUVM MLIR")
        _write_diff(artifacts.path(ORIGINAL_MLIR), artifacts.edited_mlir, artifacts.diff)
        self._validate_mlir(artifacts)
        return artifacts, manifest, abi, edited_source

    def _validate_mlir(self, artifacts: SunmmioSuvmArtifacts) -> None:
        npuir_opt = find_npuir_tool("npuir-opt")
        _run_checked(
            [
                npuir_opt,
                artifacts.edited_mlir,
                "--verify-each",
                "--suvm-device-validate",
                "-o",
                artifacts.path(DEVICE_VALIDATED_MLIR),
            ],
            "NPU-IR device validation",
            self.timeout,
        )
        _run_checked(
            [
                npuir_opt,
                artifacts.edited_mlir,
                "--verify-each",
                "--suvm-to-llvm-pipeline",
                "-o",
                artifacts.path(LOWERED_MLIR),
            ],
            "NPU-IR full lowering validation",
            self.timeout,
        )


def _kernel_params(kernel: tvm.tir.PrimFunc | tvm.IRModule) -> list[KernelParam]:
    if isinstance(kernel, tvm.tir.PrimFunc):
        function = kernel
    else:
        functions = [func for func in kernel.functions.values() if isinstance(func, tvm.tir.PrimFunc)]
        if len(functions) != 1:
            raise ValueError(f"Manual SUVM edit session requires one PrimFunc, got {len(functions)}.")
        function = functions[0]
    return [
        KernelParam.from_buffer(function.buffer_map[param]) if param in function.buffer_map else KernelParam.from_var(param)
        for param in function.params
    ]


def _prepare_emit(artifacts: SunmmioSuvmArtifacts) -> None:
    artifacts.work_dir.mkdir(parents=True, exist_ok=True)
    edited = artifacts.edited_mlir
    if edited.is_file() or edited.is_symlink():
        archived = _timestamped_archive_path(edited)
        edited.replace(archived)
        print(f"Archived previous edit: {archived}")
    elif edited.exists():
        raise IsADirectoryError(f"Edited SUVM MLIR path is not a file: {edited}")

    for path in artifacts.managed_paths():
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.exists():
            raise IsADirectoryError(f"Managed artifact path is not a file: {path}")


def _timestamped_archive_path(path: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    archived = path.with_name(f"{path.stem}.{timestamp}{path.suffix}")
    sequence = 1
    while archived.exists():
        archived = path.with_name(f"{path.stem}.{timestamp}.{sequence}{path.suffix}")
        sequence += 1
    return archived


def _parameter_kinds(manifest: dict[str, Any], abi: SunmmioKernelABI) -> tuple[str, ...]:
    parameters = manifest["parameters"]
    if len(parameters) != abi.public_arg_count:
        raise ValueError("Manual SUVM manifest parameter count does not match ABI metadata.")
    kinds = []
    for index, (parameter, name) in enumerate(zip(parameters, abi.public_param_names, strict=True)):
        if not isinstance(parameter, dict) or parameter.get("name") != name:
            raise ValueError(f"Manual SUVM manifest parameter {index} does not match {name!r}.")
        kind = parameter.get("kind")
        if kind not in {"tensor", "scalar"}:
            raise ValueError(f"Invalid parameter kind at index {index}: {kind!r}.")
        kinds.append(kind)
    return tuple(kinds)


def _validate_sunsim_elf_abi(
    artifact: SunmmioKernelArtifact,
    abi: SunmmioKernelABI,
) -> None:
    """Check that an edited kernel kept the ABI captured during TileLang lowering."""
    from sunsim.notes import ArgumentKind, find_kernel

    try:
        metadata = find_kernel(artifact.elf_path, abi.kernel_name)
    except KeyError as exc:
        raise ValueError(f"Edited kernel changed or removed ABI symbol {abi.kernel_name!r}: {exc}") from exc
    if len(metadata.args) != abi.full_arg_count:
        raise ValueError(f"Edited kernel ABI has {len(metadata.args)} arguments, but TileLang lowering recorded {abi.full_arg_count}.")

    mismatches = []
    for index, (argument, dtype) in enumerate(zip(metadata.args, abi.device_param_dtypes, strict=True)):
        expected = ArgumentKind.GLOBAL_BUFFER if dtype == "handle" else ArgumentKind.BY_VALUE
        if argument.kind != expected:
            mismatches.append(f"arg {index} ({abi.device_param_names[index]}): expected {expected.name}, got {argument.kind.name}")
    if mismatches:
        raise ValueError("Edited kernel ABI is incompatible: " + "; ".join(mismatches))


def _write_diff(original: Path, edited: Path, output: Path) -> None:
    before = _read_nonempty(original, "original SUVM MLIR").splitlines(keepends=True)
    after = _read_nonempty(edited, "edited SUVM MLIR").splitlines(keepends=True)
    output.write_text(
        "".join(difflib.unified_diff(before, after, fromfile=original.name, tofile=edited.name)),
        encoding="utf-8",
    )


def _load_manifest(artifacts: SunmmioSuvmArtifacts) -> dict[str, Any]:
    path = artifacts.path(MANIFEST_FILE)
    manifest = _read_json(path, "manual SUVM manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported manual SUVM manifest schema in {path}; rerun emit.")
    if not isinstance(manifest.get("target"), str) or not isinstance(manifest.get("parameters"), list):
        raise ValueError(f"Incomplete manual SUVM manifest: {path}")
    return manifest


def _read_nonempty(path: Path, description: str) -> str:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"{description} does not exist: {path}") from None
    if not content.strip():
        raise ValueError(f"{description} is empty: {path}")
    return content


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{description} does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return value


def _run_checked(
    command: Sequence[str | Path],
    description: str,
    timeout: float | None,
) -> None:
    command_text = [str(part) for part in command]
    try:
        result = subprocess.run(
            command_text,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{description} timed out after {timeout} seconds\ncommand: {' '.join(command_text)}") from exc
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed\ncommand: {' '.join(command_text)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
