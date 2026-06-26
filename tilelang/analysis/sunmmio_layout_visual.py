"""Sunmmio-specific layout visualization.

Kept separate from the upstream ``layout_visual`` pass (which only understands
CUDA/ROCm/Metal ``Fragment`` layouts) so Sunmmio's ``CuteLayout`` reporting does
not have to modify an upstream file and create merge friction. ``phase.py``
dispatches to this pass when the target is Sunmmio.
"""

from tvm import tir
from tvm.tir import PyStmtExprVisitor
from tvm.tir.transform import prim_func_pass


@tir.functor.visitor
class _SunmmioLayoutVisualVisitor(PyStmtExprVisitor):
    """Print the ``CuteLayout``s inferred for a Sunmmio kernel.

    Visits both block-annotation maps written by Sunmmio layout inference:

    - ``layout_map``: on-chip SRAM (ASRAM/WSRAM/RSRAM) buffer layouts.
    - ``global_layout_map``: DRAM (MeshTensor) buffer layouts.

    Each buffer is reported once. Lowering duplicates the final maps onto nested
    blocks (``root`` and ``tilelang_root``), so reporting is deduped by buffer
    name. CuteLayouts have a one-line structural repr
    (``logical_shape / mode_shape / mode_stride / dim_levels``) and no diagram
    form, so output is textual only.
    """

    def __init__(self):
        super().__init__()
        self.reported_buffers = set()

    def visit_block_(self, op: tir.Block) -> None:
        for map_key in ("layout_map", "global_layout_map"):
            if map_key not in op.annotations:
                continue
            for key, layout in op.annotations[map_key].items():
                if key.name in self.reported_buffers:
                    continue
                self.reported_buffers.add(key.name)
                print(f"{key.name} inferenced layout:")
                print(f"  {layout}")


def SunmmioLayoutVisual(formats: str = ""):
    """Pass that prints inferred Sunmmio ``CuteLayout``s (textual only).

    ``formats`` is accepted for signature parity with the upstream
    :func:`tilelang.analysis.LayoutVisual` pass; CuteLayouts have no diagram
    form, so it is ignored.
    """

    def pass_fn(func: tir.PrimFunc, mod, ctx):
        _SunmmioLayoutVisualVisitor().visit_stmt(func.body)
        return func

    return prim_func_pass(pass_fn, opt_level=0)
