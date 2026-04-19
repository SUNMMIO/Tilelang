/*!
 * \file planner.h
 * \brief Public planning entrypoints for Sunmmio tile loop fusion.
 *
 * This stage consumes window-local planning problems and produces the chosen
 * fused shared-shell tree for each window. The exact search implementation and
 * its memo/state machinery are intentionally hidden from callers; the protocol
 * surface is the window problem and final plan.
 */

#pragma once

#include "types.h"

namespace tvm {
namespace tl {

/*!
 * \brief Solve each window-local planning problem.
 *
 * \param problems Window-local semantic planning problems from discovery.
 * \return One production-ready fusion plan per input window.
 */
std::vector<SunmmioTileLoopFusionWindowPlan>
PlanSunmmioTileLoopFusionWindowProblems(
    const std::vector<SunmmioTileLoopFusionWindowProblem> &problems);

} // namespace tl
} // namespace tvm
