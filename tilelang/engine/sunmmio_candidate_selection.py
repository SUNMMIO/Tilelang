"""Production comparison for legal SunMMIO pipeline candidates.

This module intentionally contains no reporting, layout transformation cost, or
gem5 prediction. It compares only planner work that has the same statically
provable execution domain. Incomparable candidates return ``None`` so the
caller can preserve the established safe priority.
"""

from __future__ import annotations

from tvm import IRModule, tir


_PIPELINE_APPLIED = "tl.sunmmio.pipeline.applied"


def _as_int(value) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, tir.IntImm):
        return int(value.value)
    return None


def _annotation_int(loop: tir.For, key: str) -> int | None:
    return _as_int(loop.annotations.get(key))


def _contains_loop(root, target: tir.For) -> bool:
    found = False

    def mark(node) -> None:
        nonlocal found
        if isinstance(node, tir.For) and node.same_as(target):
            found = True

    tir.stmt_functor.post_order_visit(root, mark)
    return found


def _collect_loops(root) -> list[tir.For]:
    loops: list[tir.For] = []

    def collect_loop(node) -> None:
        if isinstance(node, tir.For):
            loops.append(node)

    tir.stmt_functor.post_order_visit(root, collect_loop)
    return loops


def _normalized_pipeline_work(
    mod: IRModule,
) -> dict[tuple[str, tuple[str, ...]], int] | None:
    """Return planner work grouped by identical dynamic outer domains.

    Nested pipeline loops are excluded because their work is already represented
    by the enclosing applied pipeline. ``None`` means the modules cannot be
    compared without guessing a dynamic extent or planner score.
    """

    groups: dict[tuple[str, tuple[str, ...]], int] = {}
    for global_var, base_func in mod.functions.items():
        if not isinstance(base_func, tir.PrimFunc):
            continue

        loops = _collect_loops(base_func.body)
        function_name = str(global_var.name_hint)

        for pipeline_loop in loops:
            if _annotation_int(pipeline_loop, _PIPELINE_APPLIED) != 1:
                continue

            has_enclosing_pipeline = any(
                not outer.same_as(pipeline_loop)
                and _contains_loop(outer.body, pipeline_loop)
                and _annotation_int(outer, _PIPELINE_APPLIED) == 1
                for outer in loops
            )
            if has_enclosing_pipeline:
                continue

            extent = _as_int(pipeline_loop.extent)
            ii = _annotation_int(pipeline_loop, "ii")
            makespan = _annotation_int(pipeline_loop, "makespan")
            if extent is None or ii is None or makespan is None:
                return None

            static_outer_factor = 1
            dynamic_outer_extents: list[str] = []
            for outer in loops:
                if outer.same_as(pipeline_loop) or not _contains_loop(outer.body, pipeline_loop):
                    continue
                if outer.kind not in {tir.ForKind.SERIAL, tir.ForKind.UNROLLED}:
                    continue
                outer_extent = _as_int(outer.extent)
                if outer_extent is None:
                    dynamic_outer_extents.append(str(outer.extent))
                else:
                    static_outer_factor *= max(0, outer_extent)

            pipeline_score = makespan + max(0, extent - 1) * ii
            key = (function_name, tuple(dynamic_outer_extents))
            groups[key] = groups.get(key, 0) + pipeline_score * static_outer_factor

    return groups


def compare_pipeline_candidate_work(lhs: IRModule, rhs: IRModule) -> int | None:
    """Compare normalized planner work for two already-legal candidates.

    Returns ``-1`` when ``lhs`` strictly dominates, ``1`` when ``rhs`` strictly
    dominates, ``0`` when all comparable work groups are equal, and ``None``
    when their execution domains or static facts are not comparable.
    """

    lhs_groups = _normalized_pipeline_work(lhs)
    rhs_groups = _normalized_pipeline_work(rhs)
    if lhs_groups is None or rhs_groups is None:
        return None
    if set(lhs_groups) != set(rhs_groups):
        return None

    deltas = [rhs_groups[key] - lhs_groups[key] for key in sorted(lhs_groups)]
    if all(delta == 0 for delta in deltas):
        return 0
    if all(delta <= 0 for delta in deltas) and any(delta < 0 for delta in deltas):
        return 1
    if all(delta >= 0 for delta in deltas) and any(delta > 0 for delta in deltas):
        return -1
    return None
