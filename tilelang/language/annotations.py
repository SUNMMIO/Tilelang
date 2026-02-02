"""Annotation helpers exposed on the TileLang language surface."""
from typing import Callable

from tilelang.layout import Layout
from tilelang.tileview import TileView, make_tileview
from tvm.script.parser.tir import attr, block_attr
from tvm.tir import FloatImm

__all__ = [
    "use_swizzle",
    "annotate_layout",
    "annotate_tileview",
    "annotate_safe_value",
    "annotate_l2_hit_ratio",
]


def use_swizzle(panel_size: int, order: str = "row", enable: bool = True):
    """Annotate a kernel to use a specific threadblock swizzle pattern."""
    device_func = "rasterization2DRow" if order == "row" else "rasterization2DColumn"
    if not enable:
        return None
    return attr(None, "threadblock_swizzle_pattern", f"tl::{device_func}<{panel_size}>")


def annotate_layout(layout_map: dict):
    """Annotate the layout of the buffer."""
    _layout_map = {}
    for buffer, layout in layout_map.items():
        if isinstance(layout, Layout):
            _layout_map[buffer.data] = layout
        elif isinstance(layout, Callable):
            _layout_map[buffer.data] = Layout(buffer.shape, layout)
        else:
            raise ValueError(f"Invalid layout: {layout}")

    return block_attr({"layout_map": _layout_map})


def annotate_tileview(tileview_map: dict):
    """Annotate the tileview of the buffer.

    Parameters
    ----------
    tileview_map : dict
        A dictionary mapping buffers to their TileView specifications.
        Values can be:
        - A TileView object directly
        - A tuple of (tile_shape, index_map) to create a TileView from the buffer

    Returns
    -------
    block_attr
        A block attribute containing the tileview map.

    Examples
    --------
    >>> # Using TileView directly
    >>> annotate_tileview({A: make_tileview(A, [16, 32], [-2, -1])})

    >>> # Using tuple shorthand (tile_shape, index_map)
    >>> annotate_tileview({A: ([16, 32], [-2, -1])})
    """
    _tileview_map = {}
    for buffer, tileview in tileview_map.items():
        if isinstance(tileview, TileView):
            _tileview_map[buffer.data] = tileview
        elif isinstance(tileview, tuple) and len(tileview) == 2:
            tile_shape, index_map = tileview
            _tileview_map[buffer.data] = make_tileview(buffer, tile_shape, index_map)
        else:
            raise ValueError(
                f"Invalid tileview: {tileview}. "
                "Expected TileView or tuple of (tile_shape, index_map)"
            )

    return block_attr({"tileview_map": _tileview_map})


def annotate_safe_value(safe_value_map: dict):
    """Annotate the safe value of the buffer."""
    _safe_value_map = {}
    for buffer, safe_value in safe_value_map.items():
        _safe_value_map[buffer.data] = safe_value
    return block_attr({"safe_value_map": _safe_value_map})


def annotate_l2_hit_ratio(l2_hit_ratio_map: dict):
    """Annotate the L2 hit ratio of the buffer."""
    _l2_hit_ratio_map = {}
    for buffer, hit_ratio in l2_hit_ratio_map.items():
        assert buffer.scope() == "global", "persistent L2 can only be applied to global buffers"
        _l2_hit_ratio_map[buffer.data] = FloatImm("float32", float(hit_ratio))
    return block_attr({"l2_hit_ratio_map": _l2_hit_ratio_map})
