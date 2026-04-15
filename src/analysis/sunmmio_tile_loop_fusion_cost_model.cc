#include "sunmmio_tile_loop_fusion_cost_model.h"

#include <limits>

namespace tvm {
namespace tl {

namespace {

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
