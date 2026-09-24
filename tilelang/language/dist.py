"""Rank-level distributed language constructs."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, IntEnum
import threading
from collections.abc import Iterator

from tvm import arith, tir
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


class ExpectMode(str, Enum):
    """Owner of receiver-side expected-value updates."""

    AUTO = "auto"
    MANUAL = "manual"


def _normalize_expect_mode(expect: ExpectMode | str | None) -> ExpectMode | None:
    if expect is None or isinstance(expect, ExpectMode):
        return expect
    if isinstance(expect, str):
        try:
            return ExpectMode(expect)
        except ValueError:
            pass
    raise ValueError(f"expect must be None, 'auto', or 'manual', got {expect!r}")


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
    handle: tir.PrimExpr
    _builder: object
    _group_member: bool = False
    _expect_mode: ExpectMode | None = None
    _group_handle: tir.Var | None = None
    _group_index: int | tir.PrimExpr | None = None
    _group_count: int | None = None


@dataclass(frozen=True)
class SignalList:
    """A compile-time-sized group of homogeneous receiver signals."""

    _requested_kind: SignalKind | None
    _logical_id: int
    _count: int
    handle: tir.Var
    _builder: object
    _expect_mode: ExpectMode | None = None

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[Signal]:
        return (self[index] for index in range(self._count))

    def __getitem__(self, index: int | tir.PrimExpr) -> Signal:
        if isinstance(index, bool):
            raise TypeError("T.dist.SignalList index must be an integer")
        if isinstance(index, int):
            if index < 0:
                index += self._count
            if not 0 <= index < self._count:
                raise IndexError("T.dist.SignalList index out of range")
            normalized_index: int | tir.PrimExpr = index
        elif isinstance(index, tir.PrimExpr):
            _current_dist_builder()
            if self._expect_mode == ExpectMode.AUTO:
                raise TypeError("Dynamic SignalList indexing cannot use expect='auto'; use expect=None or expect='manual'")
            if not str(index.dtype).startswith("int"):
                raise TypeError("Dynamic SignalList index must have integer dtype")
            normalized_index = index
        else:
            raise TypeError("T.dist.SignalList index must be an integer or TIR PrimExpr")
        member = tir.call_intrin(
            "handle",
            tir.op.Op.get("tl.dist_signal_ref"),
            self.handle,
            tir.IntImm("int32", normalized_index) if isinstance(normalized_index, int) else normalized_index,
        )
        return Signal(
            self._requested_kind,
            self._logical_id,
            member,
            self._builder,
            _group_member=True,
            _expect_mode=self._expect_mode,
            _group_handle=self.handle,
            _group_index=normalized_index,
            _group_count=self._count,
        )

    def _as_group_handle(self) -> tir.Var:
        builder = _current_dist_builder()
        if self._builder is not builder:
            raise ValueError("A T.dist SignalList cannot be used across different PrimFuncs")
        return self.handle


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


def signal(*, kind: SignalKind | None = None, expect: ExpectMode | str | None = None) -> Signal:
    """Declare one receiver signal for compiler resource planning."""

    if kind is not None and not isinstance(kind, SignalKind):
        raise TypeError(f"kind must be a T.dist.SignalKind, got {kind!r}")
    expect_mode = _normalize_expect_mode(expect)

    from tilelang.language.kernel import KernelLaunchFrame

    if KernelLaunchFrame.Current() is None:
        raise RuntimeError("T.dist.signal must be called inside T.Kernel()")
    builder = _current_dist_builder()
    logical_id = builder.allocate_dist_signal_decl()
    requested_kind = "auto" if kind is None else kind.value
    requested_expect = "infer" if expect_mode is None else expect_mode.value
    signal_call = tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.dist_signal_decl"),
        tir.StringImm(requested_kind),
        tir.IntImm("int32", logical_id),
        tir.StringImm(requested_expect),
    )
    signal_frame = tir_builder.LetStmt(signal_call)
    signal_handle = signal_frame.var
    builder.enter_frame(signal_frame)
    return Signal(kind, logical_id, signal_handle, builder, _expect_mode=expect_mode)


def _resolve_signal_count(count) -> int:
    if isinstance(count, int) and not isinstance(count, bool):
        return count
    if isinstance(count, tir.PrimExpr):
        from tilelang.language.comm import get_target_mesh_shape
        from tilelang.language.mesh_symbols import _mesh_ncols_symbol, _mesh_nrows_symbol

        mesh = get_target_mesh_shape()
        resolved = tir.stmt_functor.substitute(
            count,
            {
                _mesh_nrows_symbol(): tir.IntImm("int32", mesh["nrow"]),
                _mesh_ncols_symbol(): tir.IntImm("int32", mesh["ncol"]),
            },
        )
        resolved = arith.Analyzer().simplify(resolved)
        if isinstance(resolved, tir.IntImm):
            return int(resolved.value)
    raise ValueError(f"count must be a positive compile-time int, got {count!r}")


def signals(
    count: int | tir.PrimExpr,
    *,
    kind: SignalKind | None = None,
    expect: ExpectMode | str | None = None,
) -> SignalList:
    """Declare a compile-time-sized homogeneous receiver-signal group."""

    if kind is not None and not isinstance(kind, SignalKind):
        raise TypeError(f"kind must be a T.dist.SignalKind, got {kind!r}")
    expect_mode = _normalize_expect_mode(expect)
    count = _resolve_signal_count(count)
    if count <= 0:
        raise ValueError(f"count must be a positive compile-time int, got {count!r}")
    from tilelang.language.kernel import KernelLaunchFrame

    if KernelLaunchFrame.Current() is None:
        raise RuntimeError("T.dist.signals must be called inside T.Kernel()")
    builder = _current_dist_builder()
    logical_id = builder.allocate_dist_signal_decl()
    requested_kind = "auto" if kind is None else kind.value
    requested_expect = "infer" if expect_mode is None else expect_mode.value
    group_call = tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.dist_signal_group_decl"),
        tir.StringImm(requested_kind),
        tir.IntImm("int32", logical_id),
        tir.IntImm("int32", count),
        tir.StringImm(requested_expect),
    )
    group_frame = tir_builder.LetStmt(group_call)
    builder.enter_frame(group_frame)
    return SignalList(kind, logical_id, count, group_frame.var, builder, expect_mode)


def _current_core_id() -> tir.PrimExpr:
    from tilelang.language.kernel import KernelLaunchFrame

    frame = KernelLaunchFrame.Current()
    if frame is None:
        raise RuntimeError("T.dist operations must be called inside T.Kernel()")
    return frame.get_block_binding(0)


def _begin_submit_sugar(submit: bool) -> None:
    if not isinstance(submit, bool):
        raise TypeError(f"submit must be a bool, got {submit!r}")
    if submit:
        _current_dist_builder().eval(tir.call_intrin("handle", tir.op.Op.get("tl.dist_batch_begin")))


def _finish_enqueue(call: tir.PrimExpr, submit: bool):
    if not submit:
        return call
    _current_dist_builder().eval(call)
    return tir.call_intrin("handle", tir.op.Op.get("tl.dist_submit"))


def put(src, dst, dst_rank, *, dst_row=None, signal: Signal, submit: bool = False):
    """Enqueue one region write to another Rank."""

    _check_signal(signal)
    _begin_submit_sugar(submit)
    manual_group_member = signal._expect_mode == ExpectMode.MANUAL and signal._group_member and signal._group_index is not None
    if signal._expect_mode == ExpectMode.MANUAL and not manual_group_member:
        raise ValueError("T.dist.put with expect='manual' requires a SignalList member")
    if signal._expect_mode is None and signal._group_member and isinstance(signal._group_index, tir.PrimExpr):
        raise ValueError(
            "A dynamic SignalList member passed to T.dist.put requires expect='manual'; expect=None is reserved for protocol inference"
        )
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
    signal_handle = signal.handle
    if manual_group_member:
        member_index = signal._group_index
        assert signal._group_handle is not None
        generation_index = member_index * current_world_size + dst_rank
        signal_handle = tir.call_intrin(
            "handle",
            tir.op.Op.get("tl.dist_signal_group_route"),
            signal._group_handle,
            member_index,
            generation_index,
            tir.IntImm("bool", 1),
        )
    call = tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.dist_put"),
        src_region.region,
        dst_region.region,
        dst_rank,
        normalized_dst_row,
        signal_handle,
        current_core,
    )
    return _finish_enqueue(call, submit)


def routed_put(
    src,
    dst,
    routes,
    *,
    signal: Signal,
    src_rank=None,
    submit: bool = False,
):
    """Enqueue a compile-time same-column row routing table."""

    _check_signal(signal)
    _begin_submit_sugar(submit)
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
    call = tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.dist_rank_routed_put"),
        src_region.region,
        dst_region.region,
        route_table,
        normalized_src_rank,
        signal.handle,
        _current_core_id(),
    )
    return _finish_enqueue(call, submit)


def put_signal(signal: Signal, dst_rank, *, submit: bool = False):
    """Enqueue a receiver-signal update for a peer Rank."""

    _check_signal(signal)
    _begin_submit_sugar(submit)
    if signal._expect_mode == ExpectMode.MANUAL:
        raise ValueError("T.dist.put_signal with expect='manual' is not supported yet; use a sender-indexed SignalList with T.dist.put")
    if isinstance(dst_rank, bool) or not isinstance(dst_rank, (int, tir.PrimExpr)):
        raise TypeError(f"dst_rank must be an integer or TIR PrimExpr, got {type(dst_rank).__name__}")
    dst_rank_int = int(dst_rank) if isinstance(dst_rank, (int, tir.IntImm)) else None
    current_world_size = _current_world_size()
    if dst_rank_int is not None and not 0 <= dst_rank_int < current_world_size:
        raise ValueError(f"dst_rank {dst_rank_int} is outside [0, {current_world_size})")
    call = tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.dist_signal_put"),
        signal.handle,
        dst_rank,
        _current_core_id(),
    )
    return _finish_enqueue(call, submit)


def submit():
    """Submit all descriptors in this core's current distributed-send queue."""

    _current_dist_builder()
    _current_core_id()
    return tir.call_intrin("handle", tir.op.Op.get("tl.dist_submit"))


def wait_signal(signal: Signal, *, expected_delta=None):
    """Wait until a receiver signal reaches its current expected value."""

    _check_signal(signal)
    if expected_delta is not None:
        if signal._expect_mode != ExpectMode.MANUAL:
            raise ValueError("T.dist.wait_signal expected_delta requires expect='manual'")
        if isinstance(expected_delta, bool) or not isinstance(expected_delta, (int, tir.PrimExpr)):
            raise TypeError("expected_delta must be an integer or TIR PrimExpr")
        if isinstance(expected_delta, int):
            if expected_delta < 0:
                raise ValueError("expected_delta must be non-negative")
            expected_delta = tir.IntImm("int32", expected_delta)
        elif not (str(expected_delta.dtype).startswith("int") or str(expected_delta.dtype).startswith("uint")):
            raise TypeError("expected_delta must have integer dtype")
        return tir.call_intrin(
            "handle",
            tir.op.Op.get("tl.dist_wait_signal_delta"),
            signal.handle,
            expected_delta,
        )
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.dist_wait_signal"),
        signal.handle,
    )


def wait_all(signal_list):
    """Wait for a static signal list or drain an all-to-allv completion."""

    if isinstance(signal_list, DistCompletion):
        return _wait_completion_all(signal_list)

    if not isinstance(signal_list, SignalList):
        raise TypeError(f"T.dist.wait_all expects a SignalList or DistCompletion, got {type(signal_list).__name__}")
    group_handle = signal_list._as_group_handle()
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.dist_wait_all"),
        group_handle,
    )


def wait_signals(signal_list: SignalList, *, expected_deltas):
    """Advance per-member expected values and wait for every signal."""

    if not isinstance(signal_list, SignalList):
        raise TypeError(f"T.dist.wait_signals expects a SignalList, got {type(signal_list).__name__}")
    if signal_list._expect_mode != ExpectMode.MANUAL:
        raise ValueError("T.dist.wait_signals requires a SignalList declared with expect='manual'")
    group_handle = signal_list._as_group_handle()

    from .comm import _compact_shape_equal, _prepare_comm_region_compact

    deltas = _prepare_comm_region_compact(expected_deltas, "r")
    if str(deltas.buffer.dtype) not in ("int32", "uint32"):
        raise TypeError("T.dist.wait_signals expected_deltas must use int32 or uint32")
    expected_shape = [tir.IntImm("int32", len(signal_list))]
    if not _compact_shape_equal(deltas.extents, expected_shape):
        raise ValueError(f"T.dist.wait_signals expected_deltas must have shape ({len(signal_list)},), got {tuple(deltas.extents)}")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.dist_wait_signals"),
        group_handle,
        deltas.region,
    )


def wait():
    """Wait for all previously submitted local Rank sends to complete."""

    _current_dist_builder()
    _current_core_id()
    return tir.call_intrin("handle", tir.op.Op.get("tl.dist_wait_send"))


from .dist_collective import (  # noqa: E402
    CollectiveDomain,
    DistCompletion,
    _wait_completion_all,
    all_gather,
    all_reduce,
    all_to_all,
    all_to_allv,
    barrier,
    barrier_arrive,
    completion,
    has_pending,
    wait_any,
)


__all__ = [
    "RankId",
    "RankPlacementSpec",
    "Signal",
    "SignalList",
    "SignalKind",
    "ExpectMode",
    "CollectiveDomain",
    "DistCompletion",
    "all_gather",
    "all_reduce",
    "all_to_all",
    "all_to_allv",
    "barrier",
    "barrier_arrive",
    "completion",
    "placement",
    "put",
    "put_signal",
    "routed_put",
    "signal",
    "signals",
    "submit",
    "has_pending",
    "wait",
    "wait_all",
    "wait_any",
    "wait_signal",
    "wait_signals",
    "world_size",
]
