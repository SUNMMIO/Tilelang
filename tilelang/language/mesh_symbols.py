"""Symbolic Sunmmio mesh dimensions exposed in the TileLang DSL."""

from __future__ import annotations

from tvm import tir
from tvm.tir import PrimExpr

_MESH_NROWS_ATTR = "tl.sunmmio.mesh_nrows_var"
_MESH_NCOLS_ATTR = "tl.sunmmio.mesh_ncols_var"


_mesh_nrows_var = tir.SizeVar("mesh_nrows", "int32")
_mesh_ncols_var = tir.SizeVar("mesh_ncols", "int32")


def _mesh_nrows_symbol() -> PrimExpr:
    return _mesh_nrows_var


def _mesh_ncols_symbol() -> PrimExpr:
    return _mesh_ncols_var


def _current_builder():
    from tilelang.language.eager.builder import Builder

    return Builder.current()


def mesh_nrows() -> PrimExpr:
    """Return the symbolic number of rows in the current Sunmmio mesh."""
    builder = _current_builder()
    if builder is not None:
        builder.mark_sunmmio_mesh_symbols_used()
    return _mesh_nrows_symbol()


def mesh_ncols() -> PrimExpr:
    """Return the symbolic number of columns in the current Sunmmio mesh."""
    builder = _current_builder()
    if builder is not None:
        builder.mark_sunmmio_mesh_symbols_used()
    return _mesh_ncols_symbol()


def nrows() -> PrimExpr:
    """Return the symbolic number of rows in the current Sunmmio mesh."""
    return mesh_nrows()


def ncols() -> PrimExpr:
    """Return the symbolic number of columns in the current Sunmmio mesh."""
    return mesh_ncols()


def mesh_ncores() -> PrimExpr:
    """Return the symbolic number of cores in the current Sunmmio mesh."""
    return mesh_nrows() * mesh_ncols()
