from __future__ import annotations

from typing import Any, Literal
from dataclasses import dataclass
from collections.abc import Callable, Sequence

from tvm import tir
from tvm.tir.stmt_functor import post_order_visit

from tilelang import tvm
from tilelang.engine.param import KernelParam


RuntimeScalarSourceKind = Literal["shape", "stride"]
SUNMMIO_ABI_METADATA_FILE = "abi.json"


@dataclass(frozen=True)
class SunmmioRuntimeScalar:
    """One hidden scalar argument in the lowered Sunmmio kernel ABI."""

    name: str
    source_param_index: int | None = None
    source_kind: RuntimeScalarSourceKind | None = None
    source_dim: int | None = None

    def __post_init__(self) -> None:
        source = (self.source_param_index, self.source_kind, self.source_dim)
        if any(field is not None for field in source) and not all(field is not None for field in source):
            raise ValueError(
                f"Sunmmio runtime scalar {self.name!r} has a partial source binding {source}; set all three source fields or none."
            )

    @property
    def is_bound(self) -> bool:
        return self.source_param_index is not None and self.source_kind is not None and self.source_dim is not None


@dataclass(frozen=True)
class SunmmioKernelABI:
    """Runtime-agnostic Sunmmio kernel ABI metadata.

    ABI describes the public JIT parameters, the device kernel argument order, and hidden scalar
    arguments required by the lowered device kernel.
    """

    kernel_name: str
    public_arg_count: int
    public_param_names: tuple[str, ...]
    device_param_names: tuple[str, ...]
    device_param_dtypes: tuple[str, ...]
    runtime_scalars: tuple[SunmmioRuntimeScalar, ...]

    def __post_init__(self) -> None:
        if len(self.public_param_names) != self.public_arg_count:
            raise ValueError(
                f"Sunmmio ABI has {len(self.public_param_names)} public param names but public_arg_count={self.public_arg_count}."
            )
        scalar_names = self.runtime_scalar_names
        for group, names in (("public", self.public_param_names), ("scalar", scalar_names)):
            if len(set(names)) != len(names):
                raise ValueError(f"Sunmmio ABI has duplicate {group} param names: {names}.")
        overlap = set(self.public_param_names) & set(scalar_names)
        if overlap:
            raise ValueError(f"Sunmmio ABI public and scalar names collide: {sorted(overlap)}.")
        known = set(self.public_param_names) | set(scalar_names)
        if len(self.device_param_names) != self.full_arg_count or set(self.device_param_names) != known:
            raise ValueError(
                f"Sunmmio device param names {self.device_param_names} are not a permutation of "
                f"public + scalar names {tuple(sorted(known))}."
            )
        if len(self.device_param_dtypes) != len(self.device_param_names):
            raise ValueError(
                f"Sunmmio ABI has {len(self.device_param_dtypes)} device param dtypes but {len(self.device_param_names)} device params."
            )

    @classmethod
    def from_modules(
        cls,
        func_or_mod: tir.PrimFunc | tvm.IRModule,
        host_mod: tvm.IRModule | None,
        device_mod: tvm.IRModule | None,
        params: Sequence[KernelParam],
    ) -> SunmmioKernelABI:
        """Build ABI metadata from lowered modules using strict source ownership.

        Source ownership:
        - public signature comes from the original PrimFunc;
        - device symbol, argument order, and dtypes come from device_mod;
        - hidden scalar expressions come from the host call into the device kernel;
        - shape/stride symbol bindings come from tensor metadata or public buffers.
        """

        prim_func = _require_primary_prim_func(func_or_mod)
        public_signature = _read_public_signature(prim_func, params)
        device_signature = _read_device_signature(device_mod)

        public_name_set = set(public_signature.param_names)
        runtime_scalar_names = tuple(name for name in device_signature.param_names if name not in public_name_set)
        host_launch = None
        if runtime_scalar_names:
            host_launch = _read_host_launch(host_mod, device_signature.kernel_name, len(device_signature.param_names))

        symbol_sources = _read_symbol_sources(prim_func, public_signature.param_names, public_signature.arg_count)
        runtime_scalars = _build_runtime_scalars(runtime_scalar_names, device_signature.param_names, host_launch, symbol_sources)

        return cls(
            kernel_name=device_signature.kernel_name,
            public_arg_count=public_signature.arg_count,
            public_param_names=public_signature.param_names,
            device_param_names=device_signature.param_names,
            runtime_scalars=tuple(runtime_scalars),
            device_param_dtypes=device_signature.param_dtypes,
        )

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> SunmmioKernelABI:
        required = {
            "kernel_name",
            "public_arg_count",
            "public_param_names",
            "device_param_names",
            "device_param_dtypes",
            "runtime_scalars",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"Sunmmio ABI metadata is missing required field(s): {', '.join(missing)}.")

        runtime_scalars = tuple(
            SunmmioRuntimeScalar(
                name=str(scalar["name"]),
                source_param_index=scalar.get("source_param_index"),
                source_kind=scalar.get("source_kind"),
                source_dim=scalar.get("source_dim"),
            )
            for scalar in data["runtime_scalars"]
        )
        return cls(
            kernel_name=str(data["kernel_name"]),
            public_arg_count=int(data["public_arg_count"]),
            public_param_names=tuple(str(name) for name in data["public_param_names"]),
            device_param_names=tuple(str(name) for name in data["device_param_names"]),
            runtime_scalars=runtime_scalars,
            device_param_dtypes=tuple(str(dtype) for dtype in data["device_param_dtypes"]),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "kernel_name": self.kernel_name,
            "public_arg_count": self.public_arg_count,
            "public_param_names": list(self.public_param_names),
            "device_param_names": list(self.device_param_names),
            "device_param_dtypes": list(self.device_param_dtypes),
            "runtime_scalars": [
                {
                    "name": scalar.name,
                    "source_param_index": scalar.source_param_index,
                    "source_kind": scalar.source_kind,
                    "source_dim": scalar.source_dim,
                }
                for scalar in self.runtime_scalars
            ],
        }

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
        """Return runtime arguments in device ABI order."""

        if self.runtime_scalar_count and len(args) == self.full_arg_count:
            public_values = args[: self.public_arg_count]
            explicit_scalar_values = args[self.public_arg_count :]
        elif len(args) == self.public_arg_count:
            public_values = args
            explicit_scalar_values = None
        else:
            if self.runtime_scalar_count:
                raise ValueError(
                    f"Sunmmio kernel expected {self.public_arg_count} public arguments "
                    f"or {self.full_arg_count} public arguments plus explicit scalar ABI arguments, got {len(args)}"
                )
            raise ValueError(f"Sunmmio kernel expected {self.public_arg_count} arguments, got {len(args)}")

        scalar_values: dict[str, Any] = {}
        if explicit_scalar_values is not None:
            scalar_values = dict(zip(self.runtime_scalar_names, explicit_scalar_values))
        else:
            missing = [scalar.name for scalar in self.runtime_scalars if not scalar.is_bound]
            if missing:
                missing_names = ", ".join(missing)
                raise ValueError(
                    "Sunmmio kernel has hidden scalar ABI arguments that cannot be inferred "
                    f"from tensor shape/stride metadata: {missing_names}. Pass explicit scalar ABI arguments after the public arguments."
                )
            for scalar in self.runtime_scalars:
                assert scalar.source_param_index is not None
                assert scalar.source_kind is not None
                assert scalar.source_dim is not None
                source_arg = public_values[scalar.source_param_index]
                scalar_values[scalar.name] = resolve_binding(source_arg, scalar.source_kind, scalar.source_dim)

        public_args = _build_public_arg_map(self.public_param_names, public_values)
        runtime_args = []
        for name in self.device_param_names:
            if name in public_args:
                runtime_args.append(public_args[name])
            elif name in scalar_values:
                runtime_args.append(scalar_values[name])
            else:
                raise ValueError(f"Cannot materialize Sunmmio device ABI argument {name!r}.")
        return runtime_args


@dataclass(frozen=True)
class _PublicSignature:
    arg_count: int
    param_names: tuple[str, ...]


@dataclass(frozen=True)
class _DeviceSignature:
    kernel_name: str
    param_names: tuple[str, ...]
    param_dtypes: tuple[str, ...]


@dataclass(frozen=True)
class _HostLaunch:
    kernel_name: str
    arg_exprs: tuple[Any, ...]


def _require_primary_prim_func(func_or_mod: tir.PrimFunc | tvm.IRModule) -> tir.PrimFunc:
    if isinstance(func_or_mod, tir.PrimFunc):
        return func_or_mod

    funcs = [func for _, func in func_or_mod.functions.items() if isinstance(func, tir.PrimFunc)]
    if len(funcs) != 1:
        raise ValueError(f"Sunmmio ABI requires exactly one public PrimFunc, got {len(funcs)}.")
    return funcs[0]


def _read_public_signature(func: tir.PrimFunc, params: Sequence[KernelParam]) -> _PublicSignature:
    arg_count = len(params)
    if len(func.params) < arg_count:
        raise ValueError(f"Sunmmio public PrimFunc has {len(func.params)} params but expected at least {arg_count}.")

    names = []
    for index, param in enumerate(func.params[:arg_count]):
        if param in func.buffer_map:
            names.append(func.buffer_map[param].data.name)
        elif isinstance(param, tir.Var):
            names.append(param.name)
        else:
            raise ValueError(f"Unsupported Sunmmio public ABI param at index {index}: {param!r}.")
    return _PublicSignature(arg_count=arg_count, param_names=tuple(names))


def _read_device_signature(mod: tvm.IRModule | None) -> _DeviceSignature:
    if mod is None:
        raise ValueError("Sunmmio ABI extraction requires device_mod.")

    candidates: list[tuple[str, tir.PrimFunc]] = []
    for global_var, func in mod.functions.items():
        if not isinstance(func, tir.PrimFunc):
            continue
        attrs = func.attrs
        if attrs and attrs.get("tir.is_global_func", False):
            candidates.append((global_var.name_hint, func))

    if len(candidates) != 1:
        raise ValueError(f"Sunmmio ABI extraction requires exactly one device kernel, got {len(candidates)}.")

    kernel_name, device_func = candidates[0]
    param_names = []
    param_dtypes = []
    for index, param in enumerate(device_func.params):
        if not isinstance(param, tir.Var):
            raise ValueError(f"Unsupported Sunmmio device ABI param at index {index}: {param!r}.")
        param_names.append(param.name)
        param_dtypes.append(str(param.dtype))
    return _DeviceSignature(kernel_name=kernel_name, param_names=tuple(param_names), param_dtypes=tuple(param_dtypes))


def _read_host_launch(host_mod: tvm.IRModule | None, kernel_name: str, expected_arg_count: int) -> _HostLaunch:
    if host_mod is None:
        raise ValueError(f"Sunmmio ABI extraction requires host_mod to bind hidden scalar args for {kernel_name!r}.")

    calls: list[_HostLaunch] = []
    for _, func in host_mod.functions.items():
        if not isinstance(func, tir.PrimFunc):
            continue

        def visitor(node):
            if not isinstance(node, tir.Call):
                return
            if _call_op_name(node) not in {"tir.call_extern", "tir.call_packed", "tir.tvm_call_packed"}:
                return
            if not node.args:
                return
            name = _string_value(node.args[0])
            if name == kernel_name:
                calls.append(_HostLaunch(kernel_name=name, arg_exprs=tuple(node.args[1:])))

        post_order_visit(func.body, visitor)

    if len(calls) != 1:
        raise ValueError(f"Sunmmio ABI extraction expected one host launch for {kernel_name!r}, got {len(calls)}.")
    host_launch = calls[0]
    if len(host_launch.arg_exprs) < expected_arg_count:
        raise ValueError(
            f"Sunmmio host launch for {kernel_name!r} has {len(host_launch.arg_exprs)} args, expected at least {expected_arg_count}."
        )
    return host_launch


def _read_symbol_sources(
    func: tir.PrimFunc,
    public_param_names: Sequence[str],
    public_arg_count: int,
) -> dict[str, tuple[int, RuntimeScalarSourceKind, int]]:
    symbol_sources: dict[str, tuple[int, RuntimeScalarSourceKind, int]] = {}
    _record_tensor_meta_symbol_sources(symbol_sources, func, public_param_names)
    _record_buffer_symbol_sources(symbol_sources, func, public_arg_count)
    return symbol_sources


def _record_tensor_meta_symbol_sources(
    symbol_sources: dict[str, tuple[int, RuntimeScalarSourceKind, int]],
    func: tir.PrimFunc,
    tensor_names: Sequence[str],
) -> None:
    if func.attrs is None:
        return
    tensor_meta = func.attrs.get("tensor_meta")
    if tensor_meta is None:
        return

    for tensor_index, tensor_name in enumerate(tensor_names):
        if tensor_name not in tensor_meta:
            continue
        meta = tensor_meta[tensor_name]
        for dim_index, expr in enumerate(meta.get("global_shape", [])):
            _record_symbol_source(symbol_sources, expr, (tensor_index, "shape", dim_index))
        for dim_index, expr in enumerate(meta.get("global_strides", [])):
            _record_symbol_source(symbol_sources, expr, (tensor_index, "stride", dim_index))


def _record_buffer_symbol_sources(
    symbol_sources: dict[str, tuple[int, RuntimeScalarSourceKind, int]],
    func: tir.PrimFunc,
    public_arg_count: int,
) -> None:
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


def _record_symbol_source(
    symbol_sources: dict[str, tuple[int, RuntimeScalarSourceKind, int]],
    expr: Any,
    source: tuple[int, RuntimeScalarSourceKind, int],
) -> None:
    if isinstance(expr, tir.Var):
        symbol_sources.setdefault(expr.name, source)


def _build_runtime_scalars(
    runtime_scalar_names: Sequence[str],
    device_param_names: Sequence[str],
    host_launch: _HostLaunch | None,
    symbol_sources: dict[str, tuple[int, RuntimeScalarSourceKind, int]],
) -> list[SunmmioRuntimeScalar]:
    scalars = []
    for scalar_name in runtime_scalar_names:
        if host_launch is None:
            raise ValueError(f"Sunmmio hidden scalar {scalar_name!r} requires a host launch expression.")
        device_index = device_param_names.index(scalar_name)
        expr = host_launch.arg_exprs[device_index]
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
    var = _identity_var(expr)
    if var is None:
        return None
    return symbol_sources.get(var.name)


def _identity_var(expr: Any) -> tir.Var | None:
    if isinstance(expr, tir.Var):
        return expr
    if isinstance(expr, tir.Cast):
        return _identity_var(expr.value)
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


def _build_public_arg_map(public_param_names: Sequence[str], args: Sequence[Any]) -> dict[str, Any]:
    public_args: dict[str, Any] = {}
    for index, name in enumerate(public_param_names):
        if name in public_args:
            raise ValueError(f"Duplicate Sunmmio public ABI argument name {name!r}.")
        public_args[name] = args[index]
    return public_args
