from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from tvm import tir
from tvm.tir.stmt_functor import post_order_visit

from tilelang import tvm
from tilelang.engine.param import KernelParam


RuntimeScalarSourceKind = Literal["shape", "stride"]


@dataclass(frozen=True)
class SunmmioRuntimeScalar:
    """One hidden scalar argument in the lowered Sunmmio kernel ABI."""

    name: str
    source_param_index: int | None = None
    source_kind: RuntimeScalarSourceKind | None = None
    source_dim: int | None = None

    @property
    def is_bound(self) -> bool:
        return self.source_param_index is not None and self.source_kind is not None and self.source_dim is not None


@dataclass(frozen=True)
class SunmmioKernelABI:
    """Runtime-agnostic Sunmmio kernel ABI metadata.

    The ABI object does not compile, load, or execute kernels. It only describes
    the public JIT parameters and any hidden scalar arguments required by the
    lowered device kernel.
    """

    kernel_name: str
    public_arg_count: int
    device_param_names: tuple[str, ...]
    runtime_scalars: tuple[SunmmioRuntimeScalar, ...]

    @classmethod
    def from_modules(
        cls,
        func_or_mod: tir.PrimFunc | tvm.IRModule,
        host_mod: tvm.IRModule | None,
        device_mod: tvm.IRModule | None,
        params: Sequence[KernelParam],
    ) -> SunmmioKernelABI:
        prim_func = _get_primary_prim_func(func_or_mod)
        public_arg_count = len(params)

        device_func, device_name = _get_device_prim_func(device_mod)
        host_func = _get_first_prim_func(host_mod)
        host_call = _find_host_kernel_call(host_func, device_name)

        kernel_name = host_call.name if host_call is not None else device_name
        if kernel_name is None:
            kernel_name = _get_prim_func_symbol(prim_func)

        device_param_names = _collect_device_param_names(device_func)
        if not device_param_names:
            device_param_names = _collect_fallback_param_names(prim_func, public_arg_count)

        runtime_scalar_names = device_param_names[public_arg_count:]
        if not runtime_scalar_names:
            runtime_scalar_names = _collect_tensor_meta_runtime_scalar_names(prim_func, device_param_names[:public_arg_count])
            if not runtime_scalar_names:
                runtime_scalar_names = _collect_fallback_runtime_scalar_names(prim_func, public_arg_count)
            device_param_names = device_param_names + runtime_scalar_names

        metadata_func = host_func if host_func is not None else prim_func
        symbol_sources = _collect_tensor_meta_symbol_sources(metadata_func, device_param_names[:public_arg_count])
        if not symbol_sources:
            symbol_sources = _collect_prim_func_symbol_sources(prim_func, public_arg_count)

        scalar_exprs = _collect_runtime_scalar_exprs(host_call, public_arg_count, len(runtime_scalar_names))
        if not scalar_exprs:
            scalar_exprs = list(runtime_scalar_names)

        runtime_scalars = _build_runtime_scalars(runtime_scalar_names, scalar_exprs, symbol_sources)

        return cls(
            kernel_name=kernel_name,
            public_arg_count=public_arg_count,
            device_param_names=tuple(device_param_names),
            runtime_scalars=tuple(runtime_scalars),
        )

    @property
    def runtime_scalar_names(self) -> tuple[str, ...]:
        return tuple(scalar.name for scalar in self.runtime_scalars)

    @property
    def runtime_scalar_count(self) -> int:
        return len(self.runtime_scalars)

    @property
    def full_arg_count(self) -> int:
        return self.public_arg_count + self.runtime_scalar_count

    def materialize_runtime_args(
        self,
        args: Sequence[Any],
        resolve_binding: Callable[[Any, RuntimeScalarSourceKind, int], int],
    ) -> list[Any]:
        """Return public args plus hidden scalar ABI args.

        `resolve_binding` is supplied by the runtime adapter because sunsim
        markers, torch tensors, and torch-sunmmio tensors expose shape/stride
        metadata differently.
        """

        if len(args) == self.full_arg_count:
            return list(args)

        if len(args) != self.public_arg_count:
            if self.runtime_scalar_count:
                raise ValueError(
                    f"Sunmmio kernel expected {self.public_arg_count} public arguments "
                    f"or {self.full_arg_count} explicit ABI arguments, got {len(args)}"
                )
            raise ValueError(f"Sunmmio kernel expected {self.public_arg_count} arguments, got {len(args)}")

        missing = [scalar.name for scalar in self.runtime_scalars if not scalar.is_bound]
        if missing:
            missing_names = ", ".join(missing)
            raise ValueError(
                "Sunmmio kernel has hidden scalar ABI arguments that cannot be inferred "
                f"from tensor shape/stride metadata: {missing_names}. Pass explicit ABI arguments."
            )

        runtime_args = list(args)
        for scalar in self.runtime_scalars:
            assert scalar.source_param_index is not None
            assert scalar.source_kind is not None
            assert scalar.source_dim is not None
            source_arg = args[scalar.source_param_index]
            runtime_args.append(resolve_binding(source_arg, scalar.source_kind, scalar.source_dim))
        return runtime_args


@dataclass(frozen=True)
class _HostKernelCall:
    name: str
    op_name: str
    args: tuple[Any, ...]


def _get_primary_prim_func(func_or_mod: tir.PrimFunc | tvm.IRModule) -> tir.PrimFunc:
    if isinstance(func_or_mod, tir.PrimFunc):
        return func_or_mod

    if "main" in func_or_mod:
        func = func_or_mod["main"]
        if isinstance(func, tir.PrimFunc):
            return func

    for _, func in func_or_mod.functions.items():
        if isinstance(func, tir.PrimFunc):
            return func

    raise ValueError("Sunmmio ABI requires an IRModule containing a tir.PrimFunc")


def _get_first_prim_func(mod: tvm.IRModule | None) -> tir.PrimFunc | None:
    if mod is None:
        return None
    for _, func in mod.functions.items():
        if isinstance(func, tir.PrimFunc):
            return func
    return None


def _get_device_prim_func(mod: tvm.IRModule | None) -> tuple[tir.PrimFunc | None, str | None]:
    if mod is None:
        return None, None

    for global_var, func in mod.functions.items():
        if not isinstance(func, tir.PrimFunc):
            continue
        attrs = func.attrs
        if attrs and attrs.get("tir.is_global_func", False):
            return func, global_var.name_hint

    for global_var, func in mod.functions.items():
        if isinstance(func, tir.PrimFunc):
            return func, global_var.name_hint

    return None, None


def _get_prim_func_symbol(func: tir.PrimFunc) -> str:
    if func.attrs and "global_symbol" in func.attrs:
        return str(func.attrs["global_symbol"])
    return "main"


def _collect_device_param_names(func: tir.PrimFunc | None) -> tuple[str, ...]:
    if func is None:
        return ()

    names = []
    for index, param in enumerate(func.params):
        if isinstance(param, tir.Var):
            names.append(param.name)
        else:
            names.append(f"arg{index}")
    return tuple(names)


def _collect_fallback_param_names(func: tir.PrimFunc, public_arg_count: int) -> tuple[str, ...]:
    names = []
    for index, param in enumerate(func.params[:public_arg_count]):
        if param in func.buffer_map:
            names.append(func.buffer_map[param].data.name)
        elif isinstance(param, tir.Var):
            names.append(param.name)
        else:
            names.append(f"arg{index}")
    return tuple(names)


def _collect_fallback_runtime_scalar_names(func: tir.PrimFunc, public_arg_count: int) -> tuple[str, ...]:
    names: dict[str, None] = {}
    for param in func.params[:public_arg_count]:
        if param not in func.buffer_map:
            continue
        buffer = func.buffer_map[param]
        for dim in buffer.shape:
            if isinstance(dim, tir.Var):
                names.setdefault(dim.name, None)
        if buffer.strides is None:
            continue
        for stride in buffer.strides:
            if isinstance(stride, tir.Var):
                names.setdefault(stride.name, None)
    return tuple(names.keys())


def _collect_tensor_meta_runtime_scalar_names(func: tir.PrimFunc, tensor_names: Sequence[str]) -> tuple[str, ...]:
    if func.attrs is None:
        return ()
    tensor_meta = func.attrs.get("tensor_meta")
    if tensor_meta is None:
        return ()

    names: dict[str, None] = {}

    def record_name(expr: Any) -> None:
        if isinstance(expr, tir.Var):
            names.setdefault(expr.name, None)

    for tensor_name in tensor_names:
        if tensor_name not in tensor_meta:
            continue
        meta = tensor_meta[tensor_name]
        for expr in meta.get("global_shape", []):
            record_name(expr)
        for expr in meta.get("global_strides", []):
            record_name(expr)

    return tuple(names.keys())


def _find_host_kernel_call(host_func: tir.PrimFunc | None, expected_name: str | None) -> _HostKernelCall | None:
    if host_func is None:
        return None

    calls: list[_HostKernelCall] = []

    def visitor(node):
        if not isinstance(node, tir.Call):
            return
        op_name = _call_op_name(node)
        if op_name not in {"tir.call_extern", "tir.call_packed", "tir.tvm_call_packed"}:
            return
        if not node.args:
            return
        name = _string_value(node.args[0])
        if name is None:
            return
        calls.append(_HostKernelCall(name=name, op_name=op_name, args=tuple(node.args[1:])))

    post_order_visit(host_func.body, visitor)

    if expected_name is not None:
        for call in calls:
            if call.name == expected_name:
                return call

    for call in calls:
        if call.op_name == "tir.call_extern" and not call.name.startswith("__tvm_"):
            return call
    for call in calls:
        if not call.name.startswith("__tvm_"):
            return call
    return None


def _call_op_name(call: tir.Call) -> str:
    op = call.op
    name = getattr(op, "name", None)
    if name is not None:
        return str(name)
    return str(op)


def _string_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, tir.StringImm):
        return str(value.value)
    raw_value = getattr(value, "value", None)
    if isinstance(raw_value, str):
        return raw_value
    return None


def _collect_runtime_scalar_exprs(host_call: _HostKernelCall | None, public_arg_count: int, runtime_scalar_count: int) -> list[Any]:
    if host_call is None or runtime_scalar_count == 0:
        return []
    return list(host_call.args[public_arg_count : public_arg_count + runtime_scalar_count])


def _collect_tensor_meta_symbol_sources(
    host_func: tir.PrimFunc | None,
    tensor_names: Sequence[str],
) -> dict[str, tuple[int, RuntimeScalarSourceKind, int]]:
    if host_func is None or host_func.attrs is None:
        return {}
    tensor_meta = host_func.attrs.get("tensor_meta")
    if tensor_meta is None:
        return {}

    symbol_sources: dict[str, tuple[int, RuntimeScalarSourceKind, int]] = {}
    for tensor_index, tensor_name in enumerate(tensor_names):
        if tensor_name not in tensor_meta:
            continue
        meta = tensor_meta[tensor_name]
        for dim_index, expr in enumerate(meta.get("global_shape", [])):
            _record_symbol_source(symbol_sources, expr, (tensor_index, "shape", dim_index))
        for dim_index, expr in enumerate(meta.get("global_strides", [])):
            _record_symbol_source(symbol_sources, expr, (tensor_index, "stride", dim_index))
    return symbol_sources


def _collect_prim_func_symbol_sources(
    func: tir.PrimFunc,
    public_arg_count: int,
) -> dict[str, tuple[int, RuntimeScalarSourceKind, int]]:
    symbol_sources: dict[str, tuple[int, RuntimeScalarSourceKind, int]] = {}
    for tensor_index, param in enumerate(func.params[:public_arg_count]):
        if param not in func.buffer_map:
            continue
        buffer = func.buffer_map[param]
        for dim_index, expr in enumerate(buffer.shape):
            _record_symbol_source(symbol_sources, expr, (tensor_index, "shape", dim_index))
        if buffer.strides is None:
            continue
        for dim_index, expr in enumerate(buffer.strides):
            _record_symbol_source(symbol_sources, expr, (tensor_index, "stride", dim_index))
    return symbol_sources


def _record_symbol_source(
    symbol_sources: dict[str, tuple[int, RuntimeScalarSourceKind, int]],
    expr: Any,
    source: tuple[int, RuntimeScalarSourceKind, int],
) -> None:
    if isinstance(expr, tir.Var):
        symbol_sources.setdefault(expr.name, source)


def _build_runtime_scalars(
    runtime_scalar_names: Sequence[str],
    scalar_exprs: Sequence[Any],
    symbol_sources: dict[str, tuple[int, RuntimeScalarSourceKind, int]],
) -> list[SunmmioRuntimeScalar]:
    scalars = []
    for scalar_index, scalar_name in enumerate(runtime_scalar_names):
        expr = scalar_exprs[scalar_index] if scalar_index < len(scalar_exprs) else scalar_name
        source = _lookup_symbol_source(expr, symbol_sources)
        if source is None:
            scalars.append(SunmmioRuntimeScalar(name=scalar_name))
            continue
        tensor_index, source_kind, dim_index = source
        scalars.append(
            SunmmioRuntimeScalar(
                name=scalar_name,
                source_param_index=tensor_index,
                source_kind=source_kind,
                source_dim=dim_index,
            )
        )
    return scalars


def _lookup_symbol_source(
    expr: Any,
    symbol_sources: dict[str, tuple[int, RuntimeScalarSourceKind, int]],
) -> tuple[int, RuntimeScalarSourceKind, int] | None:
    if isinstance(expr, str):
        return symbol_sources.get(expr)
    if isinstance(expr, tir.Var):
        return symbol_sources.get(expr.name)

    var_names: list[str] = []

    def visitor(node):
        if isinstance(node, tir.Var):
            var_names.append(node.name)

    try:
        post_order_visit(expr, visitor)
    except TypeError:
        return None

    for name in var_names:
        if name in symbol_sources:
            return symbol_sources[name]
    return None
