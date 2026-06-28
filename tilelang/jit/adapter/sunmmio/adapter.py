from __future__ import annotations

import os
from typing import Any, Callable
from collections.abc import Sequence
from dataclasses import replace

from tvm import tir
from tvm.target import Target

from tilelang import tvm
from tilelang.engine.param import KernelParam
from tilelang.jit.adapter.base import BaseKernelAdapter
from tilelang.utils.target import determine_target, target_is_sunmmio

from .abi import RuntimeScalarSourceKind, SunmmioKernelABI
from .libgen import (
    SunmmioLibraryGenerator,
    SunmmioSuDeckLibraryGenerator,
    SunmmioSunsimLibraryGenerator,
)


class SunmmioKernelAdapter(BaseKernelAdapter):
    """Shared adapter base for Sunmmio runtime backends."""

    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int] | int,
        target: str | Target,
        func_or_mod: tir.PrimFunc | tvm.IRModule,
        host_mod: tvm.IRModule | None = None,
        device_mod: tvm.IRModule | None = None,
        device_kernel_source: str | None = None,
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
        kernel_lib_path: str | os.PathLike[str] | None = None,
        kernel_name: str | None = None,
    ):
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self.target = Target.canon_target(determine_target(target))
        if not target_is_sunmmio(self.target):
            raise ValueError(f"SunmmioKernelAdapter requires a Sunmmio target, got {self.target}")

        if isinstance(func_or_mod, tir.PrimFunc):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod

        self.abi = SunmmioKernelABI.from_modules(
            func_or_mod=self.ir_module,
            host_mod=host_mod,
            device_mod=device_mod,
            params=params,
        )
        if kernel_name is not None:
            self.abi = replace(self.abi, kernel_name=kernel_name)
        self.kernel_name = self.abi.kernel_name
        self.runtime_kernel_name = self.kernel_name
        self.host_mod = host_mod
        self.device_mod = device_mod
        self.device_kernel_source = device_kernel_source or ""
        self.kernel_global_source = self.device_kernel_source
        self.verbose = verbose
        self.pass_configs = pass_configs
        self.compile_flags = compile_flags

        self.lib_generator = self._make_lib_generator(verbose)
        self.lib_generator.assign_pass_configs(pass_configs)
        self.lib_generator.assign_compile_flags(compile_flags)
        self.lib_generator.update_mlir_source(self.device_kernel_source)
        if kernel_lib_path is None:
            self.lib_generator.compile_lib()
            self.lib_generator.load_lib()
        else:
            self.lib_generator.load_lib(kernel_lib_path)
        self.runtime_kernel_name = getattr(self.lib_generator, "runtime_kernel_name", self.runtime_kernel_name)
        self._post_init()

    @classmethod
    def from_database(
        cls,
        params: list[KernelParam],
        result_idx: list[int] | int,
        target: str | Target,
        func_or_mod: tir.PrimFunc | tvm.IRModule,
        host_kernel_source: str | None,
        device_kernel_source: str | None,
        kernel_lib_path: str | None = None,
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
        kernel_name: str | None = None,
    ):
        if kernel_lib_path is None or not os.path.exists(kernel_lib_path):
            raise FileNotFoundError(f"Cached Sunmmio kernel artifact does not exist: {kernel_lib_path}")

        return cls(
            params=params,
            result_idx=result_idx,
            target=target,
            func_or_mod=func_or_mod,
            host_mod=None,
            device_mod=None,
            device_kernel_source=device_kernel_source,
            verbose=verbose,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
            kernel_lib_path=kernel_lib_path,
            kernel_name=kernel_name,
        )

    def _convert_torch_func(self) -> Callable[..., Any]:
        def func(*args: Any, **kwargs: Any):
            raise RuntimeError(
                "Sunmmio JIT runtime execution is not implemented yet. Use get_kernel_source() to inspect the generated SUVM MLIR."
            )

        return func

    def _make_lib_generator(self, verbose: bool) -> SunmmioLibraryGenerator:
        return SunmmioLibraryGenerator(self.target, verbose)

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        return self.device_kernel_source

    def get_host_source(self) -> str:
        if self.host_mod is None:
            return ""
        return self.host_mod.script()


class SunmmioKernelSuDeckAdapter(SunmmioKernelAdapter):
    """Adapter for the real Sunmmio runtime path backed by SuDeck."""

    def _make_lib_generator(self, verbose: bool) -> SunmmioSuDeckLibraryGenerator:
        return SunmmioSuDeckLibraryGenerator(self.target, self.kernel_name, verbose)

    def _convert_torch_func(self) -> Callable[..., Any]:
        def func(*args: Any, **kwargs: Any):
            raise RuntimeError(
                "Sunmmio SuDeck JIT runtime execution is not implemented yet. Use get_kernel_source() to inspect the generated SUVM MLIR."
            )

        return func


class SunmmioSunsimKernelAdapter(SunmmioKernelAdapter):
    """Runtime adapter for executing Sunmmio kernels through sunsim.

    This adapter intentionally accepts native sunsim markers instead of wrapping
    NumPy arrays implicitly. Outputs must be passed explicitly; out_idx-based
    auto-allocation is reserved for a later runtime TensorSpec path.
    """

    def _make_lib_generator(self, verbose: bool) -> SunmmioSunsimLibraryGenerator:
        return SunmmioSunsimLibraryGenerator(self.target, self.kernel_name, verbose)

    def _convert_torch_func(self) -> Callable[..., Any]:
        if self.result_idx:
            raise NotImplementedError(
                "sunmmio_sunsim requires explicit output arguments. "
                "Do not pass out_idx; use sunsim.Output or sunsim.Inout at the call site."
            )

        def func(*args: Any, **kwargs: Any):
            sunsim = self._import_sunsim()
            runtime_args = self._prepare_sunsim_args(args, sunsim)
            artifact = self.lib_generator.artifact
            if artifact is None:
                raise RuntimeError(
                    "Sunmmio sunsim adapter was created without a materialized ELF artifact. "
                    "Materialize the artifact through SunmmioSunsimLibraryGenerator before constructing the adapter."
                )
            self.runtime_kernel_name = artifact.runtime_kernel_name
            kwargs.setdefault("kernel_name", artifact.runtime_kernel_name)
            return sunsim.run(elf=artifact.elf_path, args=runtime_args, **kwargs)

        return func

    @staticmethod
    def _import_sunsim():
        try:
            import sunsim  # type: ignore[import-not-found]
        except ImportError as err:
            raise ImportError(
                "sunmmio_sunsim execution requires the sunsim Python package. "
                "Install TileLang with the sunmmio-sunsim extra, or set PYTHONPATH "
                "to compiler-samples/sunsim/src."
            ) from err
        return sunsim

    def _prepare_sunsim_args(self, args: Sequence[Any], sunsim) -> list[Any]:
        expected_public_args = self.abi.public_arg_count
        expected_full_args = self.abi.full_arg_count
        if len(args) not in (expected_public_args, expected_full_args):
            if self.abi.runtime_scalar_count:
                raise ValueError(
                    f"Sunmmio sunsim kernel expected {expected_public_args} public arguments "
                    f"or {expected_full_args} explicit ABI arguments, got {len(args)}"
                )
            raise ValueError(f"Sunmmio sunsim kernel expected {expected_public_args} arguments, got {len(args)}")

        marker_types = (sunsim.Input, sunsim.Output, sunsim.Inout)
        descriptor_type = sunsim.Descriptor
        for index, (arg, param) in enumerate(zip(args[:expected_public_args], self.params)):
            if param.is_scalar():
                if isinstance(arg, marker_types):
                    raise TypeError(
                        f"Sunmmio sunsim argument {index} is a scalar slot, but got {type(arg).__name__}. "
                        "Use scalar values or an explicit sunsim.Descriptor when the kernel ABI expects a descriptor pointer."
                    )
                continue

            if isinstance(arg, descriptor_type):
                raise TypeError(
                    f"Sunmmio sunsim argument {index} is a tensor slot, but got sunsim.Descriptor. "
                    "Pass Descriptor only for descriptor pointer parameters."
                )
            if not isinstance(arg, marker_types):
                raise TypeError(
                    f"Sunmmio sunsim argument {index} is a tensor slot; expected sunsim.Input, "
                    f"sunsim.Output, or sunsim.Inout, got {type(arg).__name__}."
                )

        if len(args) == expected_full_args:
            return list(args)

        return self.abi.materialize_runtime_args(args, self._resolve_sunsim_marker_dim)

    @staticmethod
    def _sunsim_marker_shape(marker: Any) -> tuple[int, ...]:
        shape = getattr(marker, "shape", None)
        if shape is not None:
            return tuple(int(dim) for dim in shape)

        array = getattr(marker, "array", None)
        if array is not None:
            return tuple(int(dim) for dim in array.shape)

        data = getattr(marker, "data", None)
        if data is not None:
            return tuple(int(dim) for dim in data.shape)

        raise ValueError(f"Cannot infer shape from sunsim marker {type(marker).__name__}.")

    @classmethod
    def _resolve_sunsim_marker_dim(cls, marker: Any, source_kind: RuntimeScalarSourceKind, dim_index: int) -> int:
        if source_kind == "shape":
            shape = cls._sunsim_marker_shape(marker)
            if dim_index >= len(shape):
                raise ValueError(f"Cannot infer dynamic shape dim {dim_index} from sunsim marker with shape {shape}.")
            return int(shape[dim_index])

        raise NotImplementedError(
            "Sunmmio sunsim dynamic stride ABI inference is not implemented yet. "
            "Pass explicit scalar ABI arguments for stride-dependent kernels."
        )
