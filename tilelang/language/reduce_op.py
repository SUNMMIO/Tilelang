from tvm import tir
from tvm.target import Target
from tilelang.language import copy, macro, alloc_fragment
from tilelang.utils.language import to_buffer_region, is_shared, is_fragment
from tilelang.utils.target import target_is_sunmmio
from tvm.script.ir_builder import IRBuilder


class ReduceKind:
    Sum = "sum"
    Max = "max"
    Min = "min"
    AbsSum = "abssum"
    AbsMax = "absmax"


_REDUCE_OP_KEY = "tl.tileop.reduce"


def reduce(buffer: tir.Buffer, out: tir.Buffer, reduce_type: ReduceKind, dim: int, clear: bool) -> None:
    """Perform a reduction operation on a buffer along a specified dimension.

    Args:
        buffer (tir.Buffer): Input buffer to reduce
        out (tir.Buffer): Output buffer to store results
        reduce_type (str): Type of reduction ('max', 'min', 'sum', 'abssum')
        dim (int): Dimension along which to perform reduction
        clear (bool): Whether to initialize the output buffer before reduction
    """

    # Relaxed shape check: ignore dimensions of size 1
    def _get_filtered_shape(shape):
        return [int(x) for x in shape if int(x) != 1]

    buffer_filtered = _get_filtered_shape(buffer.shape)
    out_filtered = _get_filtered_shape(out.shape)

    # After filtering 1s, the rank should decrease by 1 if the reduced dimension was not 1.
    # If the reduced dimension was 1, the rank remains the same after filtering.
    # This is a very loose check.
    if len(buffer_filtered) - len(out_filtered) not in [0, 1]:
        raise ValueError(f"Invalid reduce output shape, buffer shape is {buffer.shape}, dim is {dim}, output shape is {out.shape}")

    @macro
    def reduce_macro(buffer: tir.Buffer, out: tir.Buffer, reduce_type: str, dim: int, clear: bool) -> None:
        target = Target.current()
        # Sunmmio uses direct builtins for ReduceOp in LowerTileOp
        # Check for Sunmmio target or specific Sunmmio shared memory scopes
        is_sunmmio_scope = any(scope in (buffer.scope(), out.scope()) for scope in ("shared.rsram", "shared.asram", "shared.wsram"))
        if (target and target_is_sunmmio(target)) or is_sunmmio_scope:
            tir.call_intrin(
                "handle",
                tir.op.Op.get(_REDUCE_OP_KEY),
                to_buffer_region(buffer, access_type="r"),
                to_buffer_region(out, access_type="w"),
                reduce_type,
                dim,
                clear,
            )
            return

        if is_shared(buffer) and is_shared(out):
            red_frag_in = alloc_fragment(buffer.shape, buffer.dtype)
            red_frag_out = alloc_fragment(out.shape, out.dtype)

            # rename buffers
            IRBuilder.name(buffer.name + "_frag", red_frag_in)
            IRBuilder.name(out.name + "_frag", red_frag_out)

            if not clear:
                copy(out, red_frag_out)

            copy(buffer, red_frag_in)
            tir.call_intrin(
                "handle",
                tir.op.Op.get(_REDUCE_OP_KEY),
                to_buffer_region(red_frag_in, access_type="r"),
                to_buffer_region(red_frag_out, access_type="w"),
                reduce_type,
                dim,
                clear,
            )
            copy(red_frag_out, out)
        elif is_shared(buffer) and is_fragment(out):
            red_frag_in = alloc_fragment(buffer.shape, buffer.dtype)
            IRBuilder.name(buffer.name + "_frag", red_frag_in)

            copy(buffer, red_frag_in)
            tir.call_intrin(
                "handle",
                tir.op.Op.get(_REDUCE_OP_KEY),
                to_buffer_region(red_frag_in, access_type="r"),
                to_buffer_region(out, access_type="w"),
                reduce_type,
                dim,
                clear,
            )
        elif is_fragment(buffer) and is_shared(out):
            red_frag_out = alloc_fragment(out.shape, out.dtype)
            IRBuilder.name(out.name + "_frag", red_frag_out)

            if not clear:
                copy(out, red_frag_out)

            tir.call_intrin(
                "handle",
                tir.op.Op.get(_REDUCE_OP_KEY),
                to_buffer_region(buffer, access_type="r"),
                to_buffer_region(red_frag_out, access_type="w"),
                reduce_type,
                dim,
                clear,
            )
            copy(red_frag_out, out)
        elif is_fragment(buffer) and is_fragment(out):
            tir.call_intrin(
                "handle",
                tir.op.Op.get(_REDUCE_OP_KEY),
                to_buffer_region(buffer, access_type="r"),
                to_buffer_region(out, access_type="w"),
                reduce_type,
                dim,
                clear,
            )
        else:
            raise ValueError(f"Invalid buffer scopes: {buffer.scope()} and {out.scope()}")

    return reduce_macro(buffer, out, reduce_type, dim, clear)


def reduce_sum(buffer: tir.Buffer, out: tir.Buffer, dim: int, clear: bool = True) -> None:
    reduce(buffer, out, "sum", dim, clear)


def reduce_max(buffer: tir.Buffer, out: tir.Buffer, dim: int, clear: bool = True) -> None:
    reduce(buffer, out, "max", dim, clear)


def reduce_min(buffer: tir.Buffer, out: tir.Buffer, dim: int, clear: bool = True) -> None:
    reduce(buffer, out, "min", dim, clear)


def reduce_abssum(buffer: tir.Buffer, out: tir.Buffer, dim: int, clear: bool = True) -> None:
    reduce(buffer, out, "abssum", dim, clear)


def reduce_absmax(buffer: tir.Buffer, out: tir.Buffer, dim: int, clear: bool = True) -> None:
    reduce(buffer, out, "absmax", dim, clear)


def reduce_bitand(buffer: tir.Buffer, out: tir.Buffer, dim: int, clear: bool = True) -> None:
    reduce(buffer, out, "bitand", dim, clear)


def reduce_bitor(buffer: tir.Buffer, out: tir.Buffer, dim: int, clear: bool = True) -> None:
    reduce(buffer, out, "bitor", dim, clear)


def reduce_bitxor(buffer: tir.Buffer, out: tir.Buffer, dim: int, clear: bool = True) -> None:
    reduce(buffer, out, "bitxor", dim, clear)


@macro
def cumsum(buffer: tir.Buffer, out: tir.Buffer, dim: int, reverse: bool = False) -> None:
    tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.cumsum"),
        to_buffer_region(buffer, access_type="r"),
        to_buffer_region(out, access_type="w"),
        dim,
        reverse,
    )


@macro
def finalize_reducer(buffer: tir.Buffer, out: tir.Buffer) -> None:
    tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.finalize_reducer"),
        to_buffer_region(buffer, access_type="r"),
        to_buffer_region(out, access_type="w"),
    )


def warp_reduce_sum(buffer: tir.Buffer) -> None:
    tir.call_intrin("handle", tir.op.Op.get("tl.warp_reduce_sum"), buffer)


def warp_reduce_max(buffer: tir.Buffer) -> None:
    tir.call_intrin("handle", tir.op.Op.get("tl.warp_reduce_max"), buffer)


def warp_reduce_min(buffer: tir.Buffer) -> None:
    tir.call_intrin("handle", tir.op.Op.get("tl.warp_reduce_min"), buffer)


def warp_reduce_bitand(buffer: tir.Buffer) -> None:
    tir.call_intrin("handle", tir.op.Op.get("tl.warp_reduce_bitand"), buffer)


def warp_reduce_bitor(buffer: tir.Buffer) -> None:
    tir.call_intrin("handle", tir.op.Op.get("tl.warp_reduce_bitor"), buffer)
