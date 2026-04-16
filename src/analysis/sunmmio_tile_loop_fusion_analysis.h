#pragma once

// Discovery-side entrypoints for Sunmmio tile loop fusion.
//
// This stage is responsible for:
// - finding planner-visible tile regions in lowered TIR
// - partitioning them into source-order region runs
// - building planner-ready window problems from those regions
//
// This stage does not choose schedules, format debug output, or rewrite TIR.

#include "sunmmio_tile_loop_fusion_protocol.h"

#include <tvm/tir/function.h>

namespace tvm {
namespace tl {

SunmmioTileLoopFusionProgram
BuildSunmmioTileLoopFusionProgram(const tir::PrimFunc &func);

std::vector<SunmmioTileLoopFusionWindowProblem>
BuildSunmmioTileLoopFusionWindowProblems(
    const SunmmioTileLoopFusionProgram &program);

} // namespace tl
} // namespace tvm
