#pragma once

// Planner entrypoints for Sunmmio tile loop fusion.
//
// This stage consumes window-local planning problems and produces the chosen
// region order plus fused shared-shell tree for each window.
//
// The exact search implementation and its memo/state machinery are
// intentionally hidden from callers; the protocol surface is the window problem
// and plan.

#include "sunmmio_tile_loop_fusion_protocol.h"

namespace tvm {
namespace tl {

std::vector<SunmmioTileLoopFusionWindowPlan>
PlanSunmmioTileLoopFusionWindowProblems(
    const std::vector<SunmmioTileLoopFusionWindowProblem> &problems);

} // namespace tl
} // namespace tvm
