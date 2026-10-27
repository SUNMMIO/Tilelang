"""GEMM (General Matrix Multiplication) operators exposed on the TileLang language surface."""

from __future__ import annotations

from tilelang._typing import BufferLikeType, BarrierType
from tilelang.tileop.base import GemmWarpPolicy
import tilelang.language as T
from tvm import tir
from tilelang.utils.language import (
    to_buffer_region,
    retrieve_shape,
    retrieve_stride,
    retrieve_offset,
    prim_expr_equal,
    prim_expr_equal_or_mesh_symbolic,
)
from tilelang.language.utils import (
    buffer_region_to_tile_region,
)
from tilelang.env import env as _env


def _gemm_impl(
    op_key: str,
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
    mbar: BarrierType | None = None,
) -> tir.PrimExpr:
    """Shared GEMM implementation.

    Returns a call_intrin handle for the given op key.
    """

    def legalize_arguments(arg: BufferLikeType | tir.Var) -> BufferLikeType:
        """Convert let-bound variables to their corresponding buffers.

        Args:
            arg (Union[tir.Buffer, tir.Var]): Input argument to legalize

        Returns:
            Union[tir.Buffer, tir.Var]: The legalized argument
        """
        if isinstance(arg, tir.Var) and T.has_let_value(arg):
            return T.get_let_value(arg)
        return arg

    A = legalize_arguments(A)
    B = legalize_arguments(B)
    C = legalize_arguments(C)
    mbar = legalize_arguments(mbar) if mbar is not None else None

    # Normalize A/B/C to BufferRegion for shape/stride/offset analysis
    A_region = to_buffer_region(A)
    B_region = to_buffer_region(B)
    C_region = to_buffer_region(C)

    A_shape = retrieve_shape(A_region)
    B_shape = retrieve_shape(B_region)
    C_shape = retrieve_shape(C_region)

    A_stride = retrieve_stride(A_region)
    B_stride = retrieve_stride(B_region)

    if len(C_shape) not in (2, 3):
        raise ValueError(f"T.gemm C must have rank 2 or 3, got rank {len(C_shape)} with shape {C_shape}")
    for operand, shape in (("A", A_shape), ("B", B_shape)):
        legacy_singleton = len(C_shape) == 2 and len(shape) > 3 and all(prim_expr_equal(extent, 1) for extent in shape[:-2])
        if len(shape) not in (2, 3) and not legacy_singleton:
            raise ValueError(
                f"T.gemm {operand} must have rank 2 or 3, or only singleton "
                f"leading dimensions for a rank-2 C; got rank {len(shape)} "
                f"with shape {shape}"
            )

    M, N = C_shape[-2:]
    A_M = A_shape[-1] if transpose_A else A_shape[-2]
    K = A_shape[-2] if transpose_A else A_shape[-1]
    K_B = B_shape[-1] if transpose_B else B_shape[-2]
    B_N = B_shape[-2] if transpose_B else B_shape[-1]

    def require_equal(lhs, rhs, message: str) -> None:
        if not prim_expr_equal_or_mesh_symbolic(lhs, rhs):
            raise ValueError(f"{message}: {lhs} != {rhs}")

    require_equal(A_M, M, "T.gemm A matrix row extent must match C")
    require_equal(B_N, N, "T.gemm B matrix column extent must match C")
    require_equal(K, K_B, "T.gemm reduction extents must match")

    if len(C_shape) == 3:
        C_batch = C_shape[0]
        if len(A_shape) == 3:
            require_equal(A_shape[0], C_batch, "T.gemm A batch extent must match C")
        if len(B_shape) == 3:
            require_equal(B_shape[0], C_batch, "T.gemm B batch extent must match C")
    elif len(A_shape) == 3 and len(B_shape) == 3:
        require_equal(A_shape[0], B_shape[0], "T.gemm A and B batch extents must match")

    stride_a = A_stride[-2]
    stride_b = B_stride[-2]

    A_offset = retrieve_offset(A_region)
    B_offset = retrieve_offset(B_region)
    assert A_offset[-2] == 0, "The offset of the first matrix dimension of A must be 0"
    assert B_offset[-2] == 0, "The offset of the first matrix dimension of B must be 0"
    offset_a = A_offset[-1]
    offset_b = B_offset[-1]

    if mbar is not None:
        assert isinstance(mbar, (tir.Buffer, tir.BufferLoad)), (
            f"mbar for tcgen5mma must be a tir.Buffer or tir.BufferLoad, but got {type(mbar)}"
        )
        mbar = to_buffer_region(mbar, access_type="rw")
    C_coords = [r.min for r in C_region.region]
    # Convert BufferRegion to tl.region calls for arguments
    A_arg = buffer_region_to_tile_region(A_region, "r", [r for r in A_shape])
    B_arg = buffer_region_to_tile_region(B_region, "r", [r for r in B_shape])
    C_arg = buffer_region_to_tile_region(C_region, "rw", [r for r in C_shape])
    # When mbar is None, pass a placeholder constant (0).
    # The C++ side checks if arg 16 is a BufferLoadNode before using it,
    # so a non-BufferLoad value will be correctly ignored.
    mbar_arg = mbar if mbar is not None else tir.const(0, dtype="int32")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get(op_key),
        A_arg,
        B_arg,
        C_arg,
        transpose_A,
        transpose_B,
        M,
        N,
        K,
        policy,
        clear_accum,
        stride_a,
        stride_b,
        offset_a,
        offset_b,
        k_pack,
        wg_wait,
        mbar_arg,
        C_coords[-2],
        C_coords[-1],
    )


# Public wrappers
def gemm_v1(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
    mbar: BarrierType | None = None,
) -> tir.PrimExpr:
    """GEMM v1: use op tl.gemm."""
    return _gemm_impl(
        "tl.tileop.gemm",
        A,
        B,
        C,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        k_pack,
        wg_wait,
        mbar,
    )


# experimental currently, for fast compilation
def gemm_v2(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
    mbar: BarrierType | None = None,
) -> tir.PrimExpr:
    """GEMM v2: use op tl.gemm_py."""
    return _gemm_impl(
        "tl.tileop.gemm_py",
        A,
        B,
        C,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        k_pack,
        wg_wait,
        mbar,
    )


# Default to v2; allow forcing v1 via environment variable
# gemm = gemm_v1 if _env.use_gemm_v1() else gemm_v2


def gemm(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
    mbar: BarrierType | None = None,
) -> tir.PrimExpr:
    """TileLang GEMM operator.

    Args:
        A (BufferLikeType, i.e. Buffer | BufferLoad | BufferRegion, or Var): Input buffer A.
        B (BufferLikeType): Input buffer B.
        C (BufferLikeType): Output buffer C.
        transpose_A (bool): Whether to transpose A. Defaults to False.
        transpose_B (bool): Whether to transpose B. Defaults to False.
        policy (GemmWarpPolicy): GEMM warp partition policy.
        clear_accum (bool): Whether to clear the accumulator.
        k_pack (int): Numbers of packed matrix cores, for ROCm only. Defaults to 1.
        wg_wait (int): Int identifier of the warpgroup MMA batch to wait on.. Defaults to 0.
        mbar (BarrierType, i.e. Buffer | BufferLoad, or Var, optional): Mbarrier in Blackwell. Defaults to None.

    Returns:
        tir.Call: A handle to the GEMM operation.
    """

    impl = gemm_v1 if _env.use_gemm_v1() else gemm_v2
    return impl(A, B, C, transpose_A, transpose_B, policy, clear_accum, k_pack, wg_wait, mbar)
