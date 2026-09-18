"""Rank-level distributed collective language constructs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .dist import (
    Signal,
    SignalKind,
    SignalList,
    _check_signal,
    _current_core_id,
    _current_dist_builder,
    _current_world_size,
    signal,
)
from tvm import tir
from tvm.script.ir_builder import tir as tir_builder


class CollectiveDomain(str, Enum):
    """Endpoint domains supported by Rank collectives."""

    RANK = "rank"
    ROW = "row"


@dataclass(frozen=True)
class DistCompletion:
    """Frontend handle for one in-flight variable-size collective."""

    handle: tir.Var
    signals: SignalList
    _builder: object


_COLLECTIVE_SIGNAL_KINDS = {
    SignalKind.SRAM_FLAGREG_INC,
    SignalKind.DRAM_FLAGREG_INC,
}

_DIST_ALLREDUCE_TYPES = ("sum", "max", "min")


def _normalize_collective_domain(domain: CollectiveDomain | str) -> CollectiveDomain:
    if isinstance(domain, CollectiveDomain):
        return domain
    if isinstance(domain, str):
        try:
            return CollectiveDomain(domain)
        except ValueError:
            pass
    raise ValueError("T.dist collective domain must be 'rank' or 'row'")


def _check_collective_signal(signal: Signal, op_name: str) -> None:
    _check_signal(signal)
    if signal._group_member and signal._requested_kind is None:
        raise ValueError(f"{op_name} requires a single automatic signal or an explicitly INC SignalList member")
    if signal._requested_kind is not None and signal._requested_kind not in _COLLECTIVE_SIGNAL_KINDS:
        raise ValueError(f"{op_name} requires an automatic or INC flagreg signal")


def _check_completion(completion: DistCompletion) -> None:
    if not isinstance(completion, DistCompletion):
        raise TypeError(f"completion must be returned by T.dist.all_to_allv, got {type(completion).__name__}")
    if completion._builder is not _current_dist_builder():
        raise ValueError("A T.dist DistCompletion cannot be used across different PrimFuncs")


def _check_count_region(region, expected_shape, name: str) -> None:
    from .comm import _compact_shape_equal

    if str(region.buffer.dtype) not in ("int32", "uint32"):
        raise TypeError(f"T.dist.all_to_allv {name} must use int32 or uint32 values")
    if not _compact_shape_equal(region.extents, expected_shape):
        raise ValueError(f"T.dist.all_to_allv {name} shape must be {tuple(expected_shape)}, got {tuple(region.extents)}")


def all_gather(src, dst, *, signal: Signal, axis: int | None = None, group=None):
    """Submit a gather of corresponding core data from every Rank.

    The caller must wait on ``signal`` before reading ``dst``.
    """

    if group is not None:
        raise NotImplementedError("T.dist.all_gather group is reserved for future subgroup support")
    if axis is not None:
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError(f"T.dist.all_gather axis must be None or an integer, got {axis!r}")
        if axis not in (0, -1):
            raise ValueError(f"T.dist.all_gather only supports axis=None, axis=0, or axis=-1, got {axis}")

    _check_collective_signal(signal, "T.dist.all_gather")

    from .comm import _prepare_allgather_operands

    world_size = _current_world_size()
    src_region, dst_region, normalized_axis = _prepare_allgather_operands(
        src,
        dst,
        world_size,
        axis,
        "T.dist.all_gather",
    )
    if src_region.buffer.dtype != dst_region.buffer.dtype:
        raise TypeError(
            f"T.dist.all_gather source and destination dtypes must match, got {src_region.buffer.dtype} and {dst_region.buffer.dtype}"
        )
    if src_region.buffer.data.same_as(dst_region.buffer.data):
        raise ValueError("T.dist.all_gather does not support overlapping source and destination storage")

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.dist_allgather"),
        src_region.region,
        dst_region.region,
        tir.IntImm("int32", normalized_axis),
        signal.handle,
        _current_core_id(),
    )


def all_to_all(
    src,
    dst,
    *,
    signal: Signal,
    domain: CollectiveDomain | str = CollectiveDomain.RANK,
    group=None,
):
    """Synchronously exchange equal chunks among Rank or Rank-row endpoints."""

    if group is not None:
        raise NotImplementedError("T.dist.all_to_all group is reserved for future subgroup support")
    normalized_domain = _normalize_collective_domain(domain)
    _check_collective_signal(signal, "T.dist.all_to_all")

    from .comm import _compact_shape_equal, _const_int, _prepare_comm_region_compact, get_target_mesh_shape

    src_region = _prepare_comm_region_compact(src, "r")
    dst_region = _prepare_comm_region_compact(dst, "w")
    if src_region.buffer.dtype != dst_region.buffer.dtype:
        raise TypeError(
            f"T.dist.all_to_all source and destination dtypes must match, got {src_region.buffer.dtype} and {dst_region.buffer.dtype}"
        )
    if src_region.buffer.data.same_as(dst_region.buffer.data):
        raise ValueError("T.dist.all_to_all does not support overlapping source and destination storage")
    if not _compact_shape_equal(src_region.extents, dst_region.extents):
        raise ValueError(
            "T.dist.all_to_all source and destination regions must have identical shapes, "
            f"got {src_region.extents} and {dst_region.extents}"
        )

    world_size = _current_world_size()
    prefix_extents = [world_size]
    if normalized_domain == CollectiveDomain.ROW:
        prefix_extents.append(get_target_mesh_shape()["nrow"])
    if len(src_region.extents) < len(prefix_extents):
        raise ValueError(f"T.dist.all_to_all domain={normalized_domain.value!r} requires {len(prefix_extents)} leading endpoint dimensions")
    for dim, expected in enumerate(prefix_extents):
        actual = _const_int(src_region.extents[dim])
        if actual is not None and actual != expected:
            raise ValueError(
                f"T.dist.all_to_all domain={normalized_domain.value!r} expects leading extent {dim} to be {expected}, got {actual}"
            )

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.dist_alltoall"),
        src_region.region,
        dst_region.region,
        tir.StringImm(normalized_domain.value),
        signal.handle,
        _current_core_id(),
    )


def all_reduce(src, dst, reduce_type: str = "sum", *, group=None):
    """Synchronously reduce corresponding Rank data into every Rank."""

    if group is not None:
        raise NotImplementedError("T.dist.all_reduce group is reserved for future subgroup support")
    if not isinstance(reduce_type, str):
        raise TypeError(f"T.dist.all_reduce reduce_type must be a string, got {reduce_type!r}")
    normalized_reduce_type = reduce_type.lower()
    if normalized_reduce_type not in _DIST_ALLREDUCE_TYPES:
        raise ValueError(f"T.dist.all_reduce reduce_type must be one of {_DIST_ALLREDUCE_TYPES}, got {reduce_type!r}")

    from .comm import _compact_shape_equal, _prepare_comm_region_compact

    src_region = _prepare_comm_region_compact(src, "r")
    dst_region = _prepare_comm_region_compact(dst, "w")
    if src_region.buffer.dtype != dst_region.buffer.dtype:
        raise TypeError(
            f"T.dist.all_reduce source and destination dtypes must match, got {src_region.buffer.dtype} and {dst_region.buffer.dtype}"
        )
    if src_region.buffer.data.same_as(dst_region.buffer.data):
        raise ValueError("T.dist.all_reduce does not support overlapping source and destination storage")
    if not _compact_shape_equal(src_region.extents, dst_region.extents):
        raise ValueError(
            "T.dist.all_reduce source and destination regions must have identical shapes, "
            f"got {src_region.extents} and {dst_region.extents}"
        )
    allowed_scopes = {"shared", "shared.dyn", "shared.rsram"}
    for name, buffer in (("source", src_region.buffer), ("destination", dst_region.buffer)):
        if buffer.scope() not in allowed_scopes:
            raise ValueError(f"T.dist.all_reduce {name} must use RSRAM or generic shared scope, got {buffer.scope()!r}")

    internal_signal = signal()
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.dist_allreduce"),
        src_region.region,
        dst_region.region,
        tir.StringImm(normalized_reduce_type),
        internal_signal.handle,
        _current_core_id(),
    )


def all_to_allv(
    src,
    dst,
    *,
    send_counts,
    recv_counts,
    signal: Signal | None = None,
    signals: SignalList | None = None,
    domain: CollectiveDomain | str = CollectiveDomain.RANK,
    group=None,
):
    """Submit a fixed-capacity exchange; SignalList mode returns a completion."""

    if group is not None:
        raise NotImplementedError("T.dist.all_to_allv group is reserved for future subgroup support")
    normalized_domain = _normalize_collective_domain(domain)
    if (signal is None) == (signals is None):
        raise TypeError("T.dist.all_to_allv requires exactly one of signal or signals")
    if signal is not None:
        if not isinstance(signal, Signal):
            raise TypeError(f"signal must be created by T.dist.signal, got {type(signal).__name__}")
        _check_collective_signal(signal, "T.dist.all_to_allv")
        signal_resource = signal.handle
    else:
        if not isinstance(signals, SignalList):
            raise TypeError(f"signals must be created by T.dist.signals, got {type(signals).__name__}")
        for item in signals:
            _check_signal(item)
        signal_resource = signals._as_group_handle()

    from .comm import _compact_shape_equal, _const_int, _prepare_comm_region_compact, get_target_mesh_shape

    src_region = _prepare_comm_region_compact(src, "r")
    dst_region = _prepare_comm_region_compact(dst, "w")
    send_counts_region = _prepare_comm_region_compact(send_counts, "r")
    recv_counts_region = _prepare_comm_region_compact(recv_counts, "r")
    if src_region.buffer.dtype != dst_region.buffer.dtype:
        raise TypeError(
            f"T.dist.all_to_allv source and destination dtypes must match, got {src_region.buffer.dtype} and {dst_region.buffer.dtype}"
        )
    if src_region.buffer.data.same_as(dst_region.buffer.data):
        raise ValueError("T.dist.all_to_allv does not support overlapping source and destination storage")
    if not _compact_shape_equal(src_region.extents, dst_region.extents):
        raise ValueError(
            "T.dist.all_to_allv source and destination regions must have identical shapes, "
            f"got {src_region.extents} and {dst_region.extents}"
        )

    world_size = _current_world_size()
    endpoint_shape = [world_size]
    if normalized_domain == CollectiveDomain.ROW:
        endpoint_shape.append(get_target_mesh_shape()["nrow"])
    if len(src_region.extents) < len(endpoint_shape) + 1:
        raise ValueError(
            f"T.dist.all_to_allv domain={normalized_domain.value!r} requires endpoint dimensions followed by a capacity dimension"
        )
    for dim, expected in enumerate(endpoint_shape):
        actual = _const_int(src_region.extents[dim])
        if actual is not None and actual != expected:
            raise ValueError(
                f"T.dist.all_to_allv domain={normalized_domain.value!r} expects leading extent {dim} to be {expected}, got {actual}"
            )

    _check_count_region(send_counts_region, endpoint_shape, "send_counts")
    _check_count_region(recv_counts_region, endpoint_shape, "recv_counts")
    endpoint_count = 1
    for extent in endpoint_shape:
        endpoint_count *= extent
    if signals is not None and len(signals) != endpoint_count:
        raise ValueError(f"T.dist.all_to_allv domain={normalized_domain.value!r} requires {endpoint_count} signals, got {len(signals)}")

    builder = _current_dist_builder()
    call = tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.dist_alltoallv"),
        src_region.region,
        dst_region.region,
        send_counts_region.region,
        recv_counts_region.region,
        tir.StringImm(normalized_domain.value),
        signal_resource,
        _current_core_id(),
    )
    if signal is not None:
        return call
    completion_frame = tir_builder.LetStmt(call)
    builder.enter_frame(completion_frame)
    return DistCompletion(completion_frame.var, signals, builder)


def barrier(*, group=None, signal: Signal | None = None):
    """Synchronize corresponding cores across all Ranks."""

    if group is not None:
        raise NotImplementedError("T.dist.barrier group is reserved for future subgroup support")
    if signal is None:
        from .dist import signal as create_signal

        signal = create_signal(kind=SignalKind.SRAM_FLAGREG_INC)
    elif not isinstance(signal, Signal):
        raise TypeError(f"signal must be created by T.dist.signal, got {type(signal).__name__}")
    _check_collective_signal(signal, "T.dist.barrier")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.dist_barrier"),
        signal.handle,
        _current_core_id(),
    )


def has_pending(completion: DistCompletion):
    """Return whether an all-to-allv completion still has remote sources."""

    _check_completion(completion)
    return tir.call_intrin(
        "bool",
        tir.op.Op.get("tl.dist_completion_has_pending"),
        completion.handle,
    )


def wait_any(completion: DistCompletion):
    """Wait for and consume one pending source, returning its logical index."""

    _check_completion(completion)
    builder = _current_dist_builder()
    wait_call = tir.call_intrin(
        "int32",
        tir.op.Op.get("tl.dist_wait_any"),
        completion.handle,
    )
    wait_frame = tir_builder.LetStmt(wait_call)
    builder.enter_frame(wait_frame)
    return wait_frame.var


def _wait_completion_all(completion: DistCompletion):
    _check_completion(completion)
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.dist_wait_completion_all"),
        completion.handle,
    )


__all__ = [
    "CollectiveDomain",
    "DistCompletion",
    "barrier",
    "all_gather",
    "all_reduce",
    "all_to_all",
    "all_to_allv",
    "has_pending",
    "wait_any",
]
