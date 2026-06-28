from __future__ import annotations

import os

from tilelang.cache.kernel_cache import KernelCache
from tilelang.jit import JITKernel


class SunmmioKernelCache(KernelCache):
    _instance = None
    device_kernel_path = "device_kernel.mlir"
    host_kernel_path = "host_kernel.py"
    kernel_lib_path = "kernel.elf"
    llvm_ir_path = "kernel.ll"

    def _save_wrapper_kernel_code_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        return

    def _save_so_cubin_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        if verbose:
            self.logger.debug(f"Saving Sunmmio ELF to cache directory: {cache_path}")

        artifact = kernel.adapter.lib_generator.artifact
        if artifact is None:
            raise RuntimeError("Sunmmio libgen did not materialize an ELF before cache persistence.")

        kernel_elf_path = os.path.join(cache_path, self.kernel_lib_path)
        KernelCache._safe_write_file(kernel_elf_path, "wb", lambda file: file.write(KernelCache._load_binary(artifact.elf_path)))

        kernel_ll_path = os.path.join(cache_path, self.llvm_ir_path)
        KernelCache._safe_write_file(kernel_ll_path, "w", lambda file: file.write(artifact.llvm_ir_source))
        kernel.adapter.lib_generator.load_lib(kernel_elf_path)

    def _get_required_files(self, cache_path: str) -> list[str]:
        return [
            os.path.join(cache_path, self.device_kernel_path),
            os.path.join(cache_path, self.kernel_lib_path),
            os.path.join(cache_path, self.llvm_ir_path),
            os.path.join(cache_path, self.params_path),
        ]

    def _load_kernel_source(self, device_kernel_path: str, host_kernel_path: str, verbose: bool = False) -> tuple[str | None, str | None]:
        try:
            with open(device_kernel_path) as f:
                return f.read(), None
        except Exception:
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
        pass_configs,
        compile_flags,
    ) -> JITKernel | None:
        if not device_kernel_source or not kernel_params:
            return None

        return JITKernel.from_database(
            func=func,
            host_kernel_source=host_kernel_source,
            device_kernel_source=device_kernel_source,
            kernel_lib_path=kernel_lib_path,
            params=kernel_params,
            target=target,
            target_host=target_host,
            out_idx=out_idx,
            execution_backend=execution_backend,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
        )


class SunmmioSunsimKernelCache(SunmmioKernelCache):
    _instance = None
