from __future__ import annotations

import json
import os

from tilelang.cache.kernel_cache import KernelCache
from tilelang.jit import JITKernel


class SunmmioKernelCache(KernelCache):
    _instance = None
    device_kernel_path = "device_kernel.mlir"
    host_kernel_path = "host_kernel.py"
    kernel_lib_path = "kernel.elf"
    llvm_ir_path = "kernel.ll"
    abi_metadata_path = "abi.json"

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

        abi = getattr(kernel.adapter, "abi", None)
        metadata = abi.to_json_dict() if abi is not None else {"kernel_name": artifact.runtime_kernel_name}
        metadata_path = os.path.join(cache_path, self.abi_metadata_path)
        KernelCache._safe_write_file(metadata_path, "w", lambda file: json.dump(metadata, file, sort_keys=True))

        kernel.adapter.lib_generator.load_lib(kernel_elf_path)

    def _get_required_files(self, cache_path: str) -> list[str]:
        return [
            os.path.join(cache_path, self.device_kernel_path),
            os.path.join(cache_path, self.kernel_lib_path),
            os.path.join(cache_path, self.llvm_ir_path),
            os.path.join(cache_path, self.abi_metadata_path),
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

        abi, kernel_name = self._load_abi_metadata(kernel_lib_path)
        kernel = JITKernel(
            func=func,
            out_idx=out_idx,
            execution_backend=execution_backend,
            target=target,
            target_host=target_host,
            from_database=True,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
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
            pass_configs=pass_configs,
            compile_flags=compile_flags,
            kernel_name=kernel_name,
            abi=abi,
        )
        kernel.torch_function = kernel.adapter.func
        return kernel

    def _load_abi_metadata(self, kernel_lib_path: str | None):
        if kernel_lib_path is None:
            return None, None
        metadata_path = os.path.join(os.path.dirname(kernel_lib_path), self.abi_metadata_path)
        with open(metadata_path, encoding="utf-8") as file:
            metadata = json.load(file)
        kernel_name = metadata.get("kernel_name")
        if {"public_arg_count", "public_param_names", "device_param_names", "runtime_scalars"}.issubset(metadata):
            from tilelang.jit.adapter.sunmmio import SunmmioKernelABI

            return SunmmioKernelABI.from_json_dict(metadata), kernel_name
        return None, kernel_name
