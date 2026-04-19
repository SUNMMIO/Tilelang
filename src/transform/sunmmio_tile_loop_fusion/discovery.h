/*!
 * \file discovery.h
 * \brief Discovery-side entrypoints for Sunmmio tile loop fusion.
 */
#pragma once

#include "types.h"

#include <tvm/tir/function.h>

namespace tvm {
namespace tl {

/*!
 * \brief Build the discovery summary for one lowered PrimFunc.
 *
 * Finds planner-visible tile regions, analyzes their external boundaries, and
 * partitions them into source-order runs for later planning stages.
 *
 * \param func The lowered PrimFunc to analyze.
 * \return Discovery summary consumed by later dependence and planner stages.
 */
SunmmioTileLoopFusionProgram
BuildSunmmioTileLoopFusionProgram(const tir::PrimFunc &func);

/*!
 * \brief Build one planner window problem per planner-visible run.
 *
 * Normalizes region boundaries and constructs the dependence graph consumed by
 * the planner for each source-order run in the discovery summary.
 *
 * \param program The discovery summary for one PrimFunc.
 * \return Planner-facing window problems for each planner-visible run.
 */
std::vector<SunmmioTileLoopFusionWindowProblem>
BuildSunmmioTileLoopFusionWindowProblems(
    const SunmmioTileLoopFusionProgram &program);

} // namespace tl
} // namespace tvm
