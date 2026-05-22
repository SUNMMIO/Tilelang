/*!
 * \file cost_model.cc
 * \brief Lexicographic score math for Sunmmio tile loop fusion planning.
 *
 * This file intentionally does only tuple arithmetic and ordering. The actual
 * byte/count deltas are produced by discovery and planner transitions; here we
 * preserve the rule that earlier score fields dominate later ones absolutely.
 */

#include "cost_model.h"

#include <limits>

namespace tvm {
namespace tl {

namespace {

// Reserve headroom below int64_t max so repeated saturating operations can use
// a stable finite sentinel for "effectively infinite" planner cost.
int64_t PlannerScoreSaturationLimit() {
  return std::numeric_limits<int64_t>::max() / 4;
}

} // namespace

int64_t SaturatingAddSunmmioTileLoopFusionPlannerCost(int64_t lhs,
                                                      int64_t rhs) {
  int64_t limit = PlannerScoreSaturationLimit();
  if (lhs >= limit || rhs >= limit) {
    return limit;
  }
  if (lhs <= -limit || rhs <= -limit) {
    return -limit;
  }
  if (rhs > 0 && lhs > limit - rhs) {
    return limit;
  }
  if (rhs < 0 && lhs < -limit - rhs) {
    return -limit;
  }
  return lhs + rhs;
}

int64_t SaturatingMulSunmmioTileLoopFusionPlannerCost(int64_t lhs,
                                                      int64_t rhs) {
  int64_t limit = PlannerScoreSaturationLimit();
  if (lhs == 0 || rhs == 0) {
    return 0;
  }
  if (lhs >= limit || rhs >= limit) {
    return limit;
  }
  if (lhs <= -limit || rhs <= -limit) {
    return -limit;
  }
  if (lhs > 0 && rhs > 0) {
    if (lhs > limit / rhs) {
      return limit;
    }
    return lhs * rhs;
  }
  if (lhs < 0 && rhs < 0) {
    if (-lhs > limit / (-rhs)) {
      return limit;
    }
    return lhs * rhs;
  }
  if (lhs < 0) {
    if (-lhs > limit / rhs) {
      return -limit;
    }
    return lhs * rhs;
  }
  if (lhs > limit / (-rhs)) {
    return -limit;
  }
  return lhs * rhs;
}

SunmmioTileLoopFusionPlannerScore AddSunmmioTileLoopFusionPlannerScores(
    const SunmmioTileLoopFusionPlannerScore &lhs,
    const SunmmioTileLoopFusionPlannerScore &rhs) {
  return {
      SaturatingAddSunmmioTileLoopFusionPlannerCost(lhs.write_cut_cost,
                                                    rhs.write_cut_cost),
      SaturatingAddSunmmioTileLoopFusionPlannerCost(lhs.shared_read_cost,
                                                    rhs.shared_read_cost),
      SaturatingAddSunmmioTileLoopFusionPlannerCost(lhs.live_range_penalty,
                                                    rhs.live_range_penalty),
      SaturatingAddSunmmioTileLoopFusionPlannerCost(lhs.reorder_penalty,
                                                    rhs.reorder_penalty),
  };
}

int CompareSunmmioTileLoopFusionPlannerScores(
    const SunmmioTileLoopFusionPlannerScore &lhs,
    const SunmmioTileLoopFusionPlannerScore &rhs) {
  // Earlier fields dominate later ones exactly. There is no scalar weighting
  // between terms: for example, any reduction in write-cut bytes beats any
  // increase in shared-read bytes or live-range penalty.
  if (lhs.write_cut_cost != rhs.write_cut_cost) {
    return lhs.write_cut_cost < rhs.write_cut_cost ? -1 : 1;
  }
  if (lhs.shared_read_cost != rhs.shared_read_cost) {
    return lhs.shared_read_cost < rhs.shared_read_cost ? -1 : 1;
  }
  if (lhs.live_range_penalty != rhs.live_range_penalty) {
    return lhs.live_range_penalty < rhs.live_range_penalty ? -1 : 1;
  }
  if (lhs.reorder_penalty != rhs.reorder_penalty) {
    return lhs.reorder_penalty < rhs.reorder_penalty ? -1 : 1;
  }
  return 0;
}

SunmmioTileLoopFusionPlannerScore
MakeInfiniteSunmmioTileLoopFusionPlannerScore() {
  int64_t inf = PlannerScoreSaturationLimit();
  return {inf, inf, inf, inf};
}

} // namespace tl
} // namespace tvm
