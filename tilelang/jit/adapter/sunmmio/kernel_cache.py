from __future__ import annotations

import json
from pathlib import Path

from tilelang.cache.kernel_cache import KernelCache
from tilelang.jit import JITKernel
from .abi import SUNMMIO_ABI_METADATA_FILE, SunmmioKernelABI
from .libgen import (
    SUNMMIO_KERNEL_ELF_FILE,
    SUNMMIO_KERNEL_LLVM_IR_FILE,
    SUNMMIO_KERNEL_MLIR_FILE,
    SUNMMIO_KERNEL_TIR_FILE,
    SUNMMIO_SUDECK_LAUNCH_MODULE_FILE,
    SUNMMIO_SUDECK_LAUNCHER_LIB_FILE,
)


class SunmmioKernelCache(KernelCache):
    _instance = None
    # Base KernelCache protocol fields. The actual filenames are owned by libgen.
    device_kernel_path = SUNMMIO_KERNEL_MLIR_FILE
    kernel_lib_path = SUNMMIO_KERNEL_ELF_FILE
    device_tir_path = SUNMMIO_KERNEL_TIR_FILE
    abi_metadata_path = SUNMMIO_ABI_METADATA_FILE

    def _save_kernel_source_code_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        return

    def _save_wrapper_kernel_code_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        return

    @staticmethod
    def _copy_file(src: str | Path, dst: Path) -> None:
        source = Path(src)
        KernelCache._safe_write_file(str(dst), "wb", lambda file: file.write(source.read_bytes()))

    def _save_so_cubin_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        if verbose:
            self.logger.debug(f"Saving Sunmmio ELF to cache directory: {cache_path}")

        cache_dir = Path(cache_path)
        artifact = kernel.adapter.lib_generator.artifact
        if artifact is None:
            raise RuntimeError("Sunmmio libgen did not materialize an ELF before cache persistence.")

        self._copy_file(artifact.elf_path, cache_dir / artifact.elf_path.name)

        if artifact.mlir_path is None:
            raise RuntimeError("Sunmmio cache persistence requires an MLIR artifact path.")
        self._copy_file(artifact.mlir_path, cache_dir / artifact.mlir_path.name)

        if artifact.tir_path is not None and artifact.tir_path.exists():
            self._copy_file(artifact.tir_path, cache_dir / artifact.tir_path.name)

        self._copy_file(artifact.llvm_ir_path, cache_dir / artifact.llvm_ir_path.name)

        metadata = kernel.adapter.abi.to_json_dict()
        metadata_path = cache_dir / self.abi_metadata_path
        KernelCache._safe_write_file(str(metadata_path), "w", lambda file: json.dump(metadata, file, sort_keys=True))

        # SuDeck emits a tvm-ffi launcher .so + Python launch module beside the ELF; copy
        # them into the cache dir so a cache hit can reload the launch module.
        build_dir = artifact.build_dir
        for name in (SUNMMIO_SUDECK_LAUNCHER_LIB_FILE, SUNMMIO_SUDECK_LAUNCH_MODULE_FILE):
            src = build_dir / name
            dst = cache_dir / name
            if src.exists() and src.resolve() != dst.resolve():
                self._copy_file(src, dst)

    def _get_required_files(self, cache_path: str) -> list[str]:
        cache_dir = Path(cache_path)
        return [
            str(cache_dir / name)
            for name in (
                self.device_kernel_path,
                self.kernel_lib_path,
                SUNMMIO_KERNEL_LLVM_IR_FILE,
                self.abi_metadata_path,
                self.params_path,
            )
        ]

    def _load_kernel_source(self, device_kernel_path: str, host_kernel_path: str, verbose: bool = False) -> tuple[str | None, str | None]:
        try:
            return Path(device_kernel_path).read_text(encoding="utf-8"), None
        except OSError:
            self.logger.exception("Error loading Sunmmio kernel source code from disk")
            return None, None

    def _build_kernel(
        self,
        func,
        host_kernel_source: str | None,
        device_kernel_source: str | None,
        kernel_lib_path: str | None,
        kernel_params,
        target,
        target_host,
        out_idx,
        execution_backend,
        **_cache_options,
    ) -> JITKernel | None:
        if not device_kernel_source or not kernel_params:
            return None

        abi = self._load_abi_metadata(kernel_lib_path)
        if abi is None:
            self.logger.warning("Ignoring cached Sunmmio kernel: missing or invalid ABI metadata")
            return None
        if execution_backend != "sunmmio_sunsim" and not self._has_sudeck_launch_artifacts(kernel_lib_path):
            self.logger.warning("Cannot build Sunmmio SuDeck kernel from cache: missing launch module artifacts")
            return None
        kernel = JITKernel(
            func=func,
            out_idx=out_idx,
            execution_backend=execution_backend,
            target=target,
            target_host=target_host,
            from_database=True,
        )
        if execution_backend == "sunmmio_sunsim":
            from tilelang.jit.adapter.sunmmio import SunmmioSunsimKernelAdapter as adapter_cls
        else:
            from tilelang.jit.adapter.sunmmio import SunmmioKernelSuDeckAdapter as adapter_cls

        kernel.adapter = adapter_cls.from_database(
            params=kernel_params,
            result_idx=out_idx,
            target=target,
            func_or_mod=func,
            host_kernel_source=host_kernel_source,
            device_kernel_source=device_kernel_source,
            kernel_lib_path=kernel_lib_path,
            abi=abi,
        )
        kernel.torch_function = kernel.adapter.func
        return kernel

    def _has_sudeck_launch_artifacts(self, kernel_lib_path: str | None) -> bool:
        if kernel_lib_path is None:
            return False
        cache_dir = Path(kernel_lib_path).parent
        return all((cache_dir / name).exists() for name in (SUNMMIO_SUDECK_LAUNCHER_LIB_FILE, SUNMMIO_SUDECK_LAUNCH_MODULE_FILE))

    def _load_abi_metadata(self, kernel_lib_path: str | None) -> SunmmioKernelABI | None:
        if kernel_lib_path is None:
            return None
        metadata_path = Path(kernel_lib_path).parent / self.abi_metadata_path
        try:
            with metadata_path.open(encoding="utf-8") as file:
                metadata = json.load(file)
        except (OSError, json.JSONDecodeError):
            self.logger.warning("Ignoring unreadable cached Sunmmio ABI metadata; recompiling")
            return None
        if not isinstance(metadata, dict):
            self.logger.warning("Ignoring malformed cached Sunmmio ABI metadata; recompiling")
            return None
        try:
            return SunmmioKernelABI.from_json_dict(metadata)
        except (AttributeError, KeyError, TypeError, ValueError):
            # Stale/corrupt cached ABI: treat as a cache miss and recompile.
            self.logger.warning("Ignoring inconsistent cached Sunmmio ABI metadata; recompiling")
        return None
