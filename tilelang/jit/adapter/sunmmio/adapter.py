from __future__ import annotations

import os
from typing import Any, Callable
from collections.abc import Sequence

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
        kernel_lib_path: str | os.PathLike[str] | None = None,
        abi: SunmmioKernelABI | None = None,
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

        self.abi = abi or SunmmioKernelABI.from_modules(
            func_or_mod=self.ir_module,
            host_mod=host_mod,
            device_mod=device_mod,
            params=params,
        )
        self.host_mod = host_mod
        self.device_mod = device_mod
        self.device_kernel_source = device_kernel_source or ""
        self.verbose = verbose

        self.lib_generator = self._make_lib_generator(verbose)
        self.lib_generator.update_device_tir_source(self._format_device_mod_tir(device_mod))
        self.lib_generator.update_mlir_source(self.device_kernel_source)
        if kernel_lib_path is None:
            self.lib_generator.compile_lib()
            self.lib_generator.load_lib()
        else:
            self.lib_generator.load_lib(kernel_lib_path)
        self._post_init()

    @classmethod
    def from_compiled_artifact(
        cls,
        *,
        target: str | Target,
        abi: SunmmioKernelABI,
        kernel_lib_path: str | os.PathLike[str],
        verbose: bool = False,
    ):
        """Load a compiled Sunmmio artifact without rebuilding its frontend kernel."""
        if not os.path.exists(kernel_lib_path):
            raise FileNotFoundError(f"Compiled Sunmmio kernel artifact does not exist: {kernel_lib_path}")

        instance = cls.__new__(cls)
        instance.params = []
        instance.result_idx = []
        instance.target = Target.canon_target(determine_target(target))
        if not target_is_sunmmio(instance.target):
            raise ValueError(f"SunmmioKernelAdapter requires a Sunmmio target, got {instance.target}")
        instance.ir_module = None
        instance.abi = abi
        instance.host_mod = None
        instance.device_mod = None
        instance.device_kernel_source = ""
        instance.verbose = verbose
        instance.lib_generator = instance._make_lib_generator(verbose)
        instance.lib_generator.load_lib(kernel_lib_path)
        instance._post_init()
        return instance

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
        abi: SunmmioKernelABI | None = None,
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
            kernel_lib_path=kernel_lib_path,
            abi=abi,
        )

    def _convert_torch_func(self) -> Callable[..., Any]:
        def func(*args: Any, **kwargs: Any):
            raise RuntimeError(
                "Sunmmio JIT runtime execution is not implemented yet. Use get_kernel_source() to inspect the generated SUVM MLIR."
            )

        return func

    def _make_lib_generator(self, verbose: bool) -> SunmmioLibraryGenerator:
        return SunmmioLibraryGenerator(self.target, verbose)

    @staticmethod
    def _format_device_mod_tir(device_mod: tvm.IRModule | None) -> str:
        if device_mod is None:
            return ""
        try:
            return device_mod.script()
        except Exception:
            return str(device_mod)

    @property
    def kernel_name(self) -> str:
        return self.abi.kernel_name

    def _validate_abi_arg_count(self, args: Sequence[Any]) -> None:
        expected_public_args = self.abi.public_arg_count
        expected_full_args = self.abi.full_arg_count
        if len(args) in (expected_public_args, expected_full_args):
            return

        if self.abi.runtime_scalar_count:
            raise ValueError(
                f"Sunmmio kernel expected {expected_public_args} public arguments "
                f"or {expected_full_args} public arguments plus explicit scalar ABI arguments, got {len(args)}"
            )
        raise ValueError(f"Sunmmio kernel expected {expected_public_args} arguments, got {len(args)}")

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        return self.device_kernel_source

    def get_host_source(self) -> str:
        if self.host_mod is None:
            return ""
        return self.host_mod.script()


class SunmmioKernelSuDeckAdapter(SunmmioKernelAdapter):
    """Adapter for the real Sunmmio runtime path backed by SuDeck."""

    def _make_lib_generator(self, verbose: bool) -> SunmmioSuDeckLibraryGenerator:
        lib_generator = SunmmioSuDeckLibraryGenerator(self.target, self.kernel_name, verbose)
        lib_generator.update_launcher_specs(self._build_launcher_specs())
        return lib_generator

    def _build_launcher_specs(self) -> list[tuple[str, str]]:
        """Launcher specs in device-ABI order: "tensor" for handles, else the scalar dtype."""
        return [
            (name, "tensor" if dtype == "handle" else dtype)
            for name, dtype in zip(self.abi.device_param_names, self.abi.device_param_dtypes)
        ]

    def _resolve_binding(self, source: Any, source_kind: RuntimeScalarSourceKind, dim_index: int) -> int:
        if source_kind == "shape":
            return int(source.shape[dim_index])
        return int(source.stride()[dim_index])

    def _convert_torch_func(self) -> Callable[..., Any]:
        if self.result_idx:
            raise NotImplementedError(
                "sunmmio SuDeck execution requires explicit output tensors; do not pass out_idx. "
                "Allocate outputs as torch_sunmmio device tensors and pass them at the call site."
            )

        def func(*args: Any, **kwargs: Any):
            import torch
            import torch_sunmmio  # noqa: F401 - registers torch.sunmmio
            from torch_sunmmio.sunmmio import unsafe_get_sutensor_handle

            pymodule = self.lib_generator.pymodule
            if pymodule is None:
                raise RuntimeError("Sunmmio SuDeck launch module was not built; compile the kernel before calling it.")
            self._validate_abi_arg_count(args)
            runtime_args = self.abi.materialize_runtime_args(args, self._resolve_binding)
            call_args = []
            for dtype, value in zip(self.abi.device_param_dtypes, runtime_args):
                if dtype == "handle":
                    call_args.append(unsafe_get_sutensor_handle(value))
                elif dtype.startswith(("float", "bfloat")):
                    call_args.append(float(value))
                else:
                    call_args.append(int(value))
            stream = torch.sunmmio.current_stream()
            stream_handle = torch.sunmmio.unsafe_get_sudeck_stream_handle(stream)
            pymodule.call(*call_args, stream_handle)

        return func


class SunmmioSunsimKernelAdapter(SunmmioKernelAdapter):
    """Runtime adapter for executing Sunmmio kernels through sunsim.

    This adapter intentionally accepts native sunsim markers instead of wrapping
    NumPy arrays implicitly. Outputs must be passed explicitly; out_idx-based
    auto-allocation is reserved for a later runtime TensorSpec path.
    """

    def _make_lib_generator(self, verbose: bool) -> SunmmioSunsimLibraryGenerator:
        return SunmmioSunsimLibraryGenerator(self.target, self.kernel_name, verbose)

    @classmethod
    def from_compiled_artifact(
        cls,
        *,
        target: str | Target,
        abi: SunmmioKernelABI,
        kernel_lib_path: str | os.PathLike[str],
        parameter_kinds: Sequence[str],
        verbose: bool = False,
    ):
        invalid_kinds = set(parameter_kinds) - {"tensor", "scalar"}
        if invalid_kinds:
            raise ValueError(f"Invalid Sunmmio parameter kinds: {sorted(invalid_kinds)}")
        if len(parameter_kinds) != abi.public_arg_count:
            raise ValueError(f"Sunmmio artifact has {len(parameter_kinds)} parameter kinds but ABI expects {abi.public_arg_count}.")

        instance = super().from_compiled_artifact(
            target=target,
            abi=abi,
            kernel_lib_path=kernel_lib_path,
            verbose=verbose,
        )
        instance._artifact_parameter_kinds = tuple(parameter_kinds)
        return instance

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
        self._validate_abi_arg_count(args)

        marker_types = (sunsim.Input, sunsim.Output, sunsim.Inout)
        descriptor_type = sunsim.Descriptor
        parameter_kinds = getattr(self, "_artifact_parameter_kinds", None)
        if parameter_kinds is None:
            parameter_kinds = tuple("scalar" if param.is_scalar() else "tensor" for param in self.params)
        for index, (arg, kind) in enumerate(zip(args[: self.abi.public_arg_count], parameter_kinds)):
            if kind == "scalar":
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
