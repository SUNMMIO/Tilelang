"""Symbolic Sunmmio mesh dimensions exposed in the TileLang DSL."""

from __future__ import annotations

from tvm import tir
from tvm.tir import PrimExpr


def mesh_nrows() -> PrimExpr:
    """Return the symbolic number of rows in the current Sunmmio mesh."""
    return tir.call_intrin("int32", tir.op.Op.get("tl.mesh_nrows"))


def mesh_ncols() -> PrimExpr:
    """Return the symbolic number of columns in the current Sunmmio mesh."""
    return tir.call_intrin("int32", tir.op.Op.get("tl.mesh_ncols"))


def mesh_ncores() -> PrimExpr:
    """Return the symbolic number of cores in the current Sunmmio mesh."""
    return tir.call_intrin("int32", tir.op.Op.get("tl.mesh_ncores"))
