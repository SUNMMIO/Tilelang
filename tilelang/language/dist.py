"""Rank-level distributed language constructs."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, IntEnum
import threading
from collections.abc import Iterator

from tvm import tir
from tvm.script.ir_builder import tir as tir_builder


_WORLD_SIZE_ATTR = "tl.dist.world_size"
_RANK_ID_PARAM_INDEX_ATTR = "tl.dist.rank_id_param_index"
_thread_local = threading.local()


def _validate_world_size(world_size: int) -> int:
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError(f"world_size must be a positive int, got {world_size!r}")
    return world_size


@contextmanager
def _world_context(world_size: int):
    """Temporarily make one compile-time world size available to the DSL."""

    world_size = _validate_world_size(world_size)
    previous = getattr(_thread_local, "world_size", None)
    if previous is not None and previous != world_size:
        raise ValueError(f"Conflicting nested world_size values: {previous} and {world_size}")
    _thread_local.world_size = world_size
    try:
        yield
    finally:
        if previous is None:
            del _thread_local.world_size
        else:
            _thread_local.world_size = previous


def _current_world_size(*, allow_none: bool = False) -> int | None:
    world_size = getattr(_thread_local, "world_size", None)
    if world_size is None:
        return None if allow_none else 1
    return world_size


class _RankPlacementKind(IntEnum):
    REPLICATED = 0
    SHARD = 1


@dataclass(frozen=True)
class RankPlacementSpec:
    """Immutable description of how a tensor is placed across Ranks."""

    _kind: _RankPlacementKind
    dim: int = -1

    def __post_init__(self) -> None:
        if not isinstance(self._kind, _RankPlacementKind):
            raise TypeError("RankPlacementSpec values must be constructed with T.dist.placement")
        if self._kind == _RankPlacementKind.REPLICATED:
            if self.dim != -1:
                raise ValueError("Replicated Rank placement cannot have a shard dimension")
        elif isinstance(self.dim, bool) or not isinstance(self.dim, int) or self.dim < 0:
            raise ValueError(f"Rank shard dim must be a non-negative int, got {self.dim!r}")

    @property
    def kind(self) -> int:
        return int(self._kind)

    def __repr__(self) -> str:
        if self._kind == _RankPlacementKind.REPLICATED:
            return "RankReplicated()"
        return f"RankShard({self.dim})"


class _PlacementNamespace:
    @staticmethod
    def replicated() -> RankPlacementSpec:
        """Give every Rank the complete declared tensor shape."""

        return RankPlacementSpec(_RankPlacementKind.REPLICATED)

    @staticmethod
    def shard(dim: int) -> RankPlacementSpec:
        """Shard one tensor dimension across all Ranks."""

        return RankPlacementSpec(_RankPlacementKind.SHARD, dim)


placement = _PlacementNamespace()


class SignalKind(str, Enum):
    """Physical Rank receiver-signal kinds."""

    SRAM_FLAGREG_INC = "sram_flagreg_inc"
    DRAM_FLAGREG_INC = "dram_flagreg_inc"
    SRAM_FLAGREG_VALUE = "sram_flagreg_value"
    DRAM_FLAGREG_VALUE = "dram_flagreg_value"
    SRAM_MEMORY = "sram_memory"
    DRAM_MEMORY = "dram_memory"


def _validate_rank_placement(value: RankPlacementSpec, tensor_rank: int) -> RankPlacementSpec:
    if not isinstance(value, RankPlacementSpec):
        raise TypeError(f"rank_placement must be a RankPlacementSpec constructed with T.dist.placement, got {type(value).__name__}")
    if value.dim >= tensor_rank:
        raise ValueError(f"Invalid Rank shard dimension: {value.dim}, tensor rank is {tensor_rank}")
    return value


def _rank_placement_metadata(value: RankPlacementSpec) -> tuple[int, int]:
    return value.kind, value.dim


@dataclass(frozen=True)
class _RankIdAnnotation:
    dtype: str = "int32"


RankId = _RankIdAnnotation()


@dataclass(frozen=True)
class Signal:
    """Frontend handle for one endpoint-local logical receiver signal."""

    _requested_kind: SignalKind | None
    _logical_id: int
    handle: tir.Var
    _builder: object


@dataclass(frozen=True)
class SignalList:
    """A compile-time-sized group of independent receiver signals."""

    _signals: tuple[Signal, ...]

    def __len__(self) -> int:
        return len(self._signals)

    def __iter__(self) -> Iterator[Signal]:
        return iter(self._signals)

    def __getitem__(self, index: int) -> Signal:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("T.dist.SignalList only supports compile-time integer indexing")
        return self._signals[index]


def _is_rank_id_annotation(value) -> bool:
    return isinstance(value, _RankIdAnnotation)


def world_size() -> tir.IntImm:
    """Return the compile-time number of Ranks in the current kernel factory."""

    value = _current_world_size()
    from tilelang.language.eager.builder import Builder

    builder = Builder.current()
    if builder is not None:
        builder.mark_dist_world_size(value)
    return tir.IntImm("int32", value)


def _current_dist_builder():
    from tilelang.language.eager.builder import Builder

    builder = Builder.current()
    if builder is None:
        raise RuntimeError("T.dist operations can only be used while constructing a PrimFunc")
    if builder._dist_world_size is None:
        builder.mark_dist_world_size(1)
    if builder._dist_rank_id_var is None:
        raise RuntimeError("T.dist operations require one PrimFunc parameter annotated with T.dist.RankId")
    return builder


def _check_signal(signal: Signal) -> None:
    if not isinstance(signal, Signal):
        raise TypeError(f"signal must be created by T.dist.signal, got {type(signal).__name__}")
    builder = _current_dist_builder()
    if signal._builder is not builder:
        raise ValueError("A T.dist Signal cannot be used across different PrimFuncs")


def signal(*, kind: SignalKind | None = None) -> Signal:
    """Declare one receiver signal for compiler resource planning."""

    if kind is not None and not isinstance(kind, SignalKind):
        raise TypeError(f"kind must be a T.dist.SignalKind, got {kind!r}")

    from tilelang.language.kernel import KernelLaunchFrame

    if KernelLaunchFrame.Current() is None:
        raise RuntimeError("T.dist.signal must be called inside T.Kernel()")
    builder = _current_dist_builder()
    logical_id = builder.allocate_dist_signal_decl()
    requested_kind = "auto" if kind is None else kind.value
    signal_call = tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.dist_signal_decl"),
        tir.StringImm(requested_kind),
        tir.IntImm("int32", logical_id),
    )
    signal_frame = tir_builder.LetStmt(signal_call)
    signal_handle = signal_frame.var
    builder.enter_frame(signal_frame)
    return Signal(kind, logical_id, signal_handle, builder)


def signals(count: int, *, kind: SignalKind | None = None) -> SignalList:
    """Declare a compile-time-sized group of independent receiver signals."""

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(f"count must be a positive compile-time int, got {count!r}")
    return SignalList(tuple(signal(kind=kind) for _ in range(count)))


def _current_core_id() -> tir.PrimExpr:
    from tilelang.language.kernel import KernelLaunchFrame

    frame = KernelLaunchFrame.Current()
    if frame is None:
        raise RuntimeError("T.dist operations must be called inside T.Kernel()")
    return frame.get_block_binding(0)


def put(src, dst, dst_rank, *, dst_row=None, signal: Signal):
    """Asynchronously write one RSRAM region to another Rank."""

    _check_signal(signal)
    if isinstance(dst_rank, bool) or not isinstance(dst_rank, (int, tir.PrimExpr)):
        raise TypeError(f"dst_rank must be an integer or TIR PrimExpr, got {type(dst_rank).__name__}")
    dst_rank_int = int(dst_rank) if isinstance(dst_rank, (int, tir.IntImm)) else None
    current_world_size = _current_world_size()
    if dst_rank_int is not None and not 0 <= dst_rank_int < current_world_size:
        raise ValueError(f"dst_rank {dst_rank_int} is outside [0, {current_world_size})")

    current_core = _current_core_id()
    if dst_row is None:
        from tilelang.language.mesh_symbols import mesh_ncols

        normalized_dst_row = current_core // mesh_ncols()
    else:
        if isinstance(dst_row, bool) or not isinstance(dst_row, (int, tir.PrimExpr)):
            raise TypeError(f"dst_row must be an integer or TIR PrimExpr, got {type(dst_row).__name__}")
        normalized_dst_row = dst_row

    from tilelang.language.comm import _prepare_one_to_one_operands

    src_region, dst_region = _prepare_one_to_one_operands(src, dst, "T.dist.put")
    if src_region.buffer.dtype != dst_region.buffer.dtype:
        raise TypeError(f"T.dist.put source and destination dtypes must match, got {src_region.buffer.dtype} and {dst_region.buffer.dtype}")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.dist_put"),
        src_region.region,
        dst_region.region,
        dst_rank,
        normalized_dst_row,
        signal.handle,
        current_core,
    )


def routed_put(src, dst, routes, *, signal: Signal, src_rank=None):
    """Execute a compile-time same-column row routing table."""

    _check_signal(signal)
    if not isinstance(routes, (list, tuple)) or not routes:
        raise TypeError("routes must be a non-empty list or tuple of route entries")

    if src_rank is None:
        normalized_src_rank = _current_dist_builder()._dist_rank_id_var
    else:
        if isinstance(src_rank, bool) or not isinstance(src_rank, int):
            raise TypeError(f"src_rank must be a compile-time integer, got {src_rank!r}")
        current_world_size = _current_world_size()
        if not 0 <= src_rank < current_world_size:
            raise ValueError(f"src_rank {src_rank} is outside [0, {current_world_size})")
        normalized_src_rank = tir.IntImm("int32", src_rank)

    from tilelang.language.comm import _prepare_one_to_one_operands

    src_region, dst_region = _prepare_one_to_one_operands(src, dst, "T.dist.routed_put")
    if src_region.buffer.dtype != dst_region.buffer.dtype:
        raise TypeError(
            f"T.dist.routed_put source and destination dtypes must match, got {src_region.buffer.dtype} and {dst_region.buffer.dtype}"
        )

    route_entries = []
    for route in routes:
        if not isinstance(route, (list, tuple)) or len(route) != 3:
            raise TypeError("each route must be [src_row, dst_rank, dst_row]")
        src_row, dst_rank, dst_row = route
        for name, value in (
            ("src_row", src_row),
            ("dst_rank", dst_rank),
            ("dst_row", dst_row),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, tir.PrimExpr)):
                raise TypeError(f"route {name} must be an integer or TIR PrimExpr")
        route_entries.append(
            tir.call_intrin(
                "handle",
                tir.op.Op.get("tl.dist_route"),
                src_row,
                dst_rank,
                dst_row,
            )
        )

    route_table = tir.call_intrin("handle", tir.op.Op.get("tl.dist_route_table"), *route_entries)
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.dist_rank_routed_put"),
        src_region.region,
        dst_region.region,
        route_table,
        normalized_src_rank,
        signal.handle,
        _current_core_id(),
    )


def wait_signal(signal: Signal, *, dst):
    """Wait for the next receiver-signal generation and make ``dst`` visible."""

    _check_signal(signal)
    from tilelang.language.comm import _prepare_comm_region_compact

    dst_region = _prepare_comm_region_compact(dst, "w")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.dist_wait_signal"),
        signal.handle,
        dst_region.region,
    )


def wait_all(signal_list: SignalList, *, dst):
    """Wait until every signal in ``signal_list`` reaches its expectation."""

    if not isinstance(signal_list, SignalList):
        raise TypeError(f"signal_list must be created by T.dist.signals, got {type(signal_list).__name__}")
    for item in signal_list:
        _check_signal(item)

    from tilelang.language.comm import _prepare_comm_region_compact

    dst_region = _prepare_comm_region_compact(dst, "w")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.dist_wait_all"),
        dst_region.region,
        *(item.handle for item in signal_list),
    )


def wait():
    """Wait for all previously submitted local Rank sends to complete."""

    _current_dist_builder()
    _current_core_id()
    return tir.call_intrin("handle", tir.op.Op.get("tl.dist_wait_send"))


__all__ = [
    "RankId",
    "RankPlacementSpec",
    "Signal",
    "SignalList",
    "SignalKind",
    "placement",
    "put",
    "routed_put",
    "signal",
    "signals",
    "wait",
    "wait_all",
    "wait_signal",
    "world_size",
]
