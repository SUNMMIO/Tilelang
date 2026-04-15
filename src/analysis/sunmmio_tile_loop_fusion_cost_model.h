#pragma once

#include <cstdint>

namespace tvm {
namespace tl {

struct SunmmioTileLoopFusionPlannerScore {
  int64_t write_cut_cost{0};
  int64_t shared_read_cost{0};
  int64_t live_range_penalty{0};
  int64_t reorder_penalty{0};
};

int64_t SaturatingAddSunmmioTileLoopFusionPlannerCost(int64_t lhs, int64_t rhs);

int64_t SaturatingMulSunmmioTileLoopFusionPlannerCost(int64_t lhs, int64_t rhs);

SunmmioTileLoopFusionPlannerScore AddSunmmioTileLoopFusionPlannerScores(
    const SunmmioTileLoopFusionPlannerScore &lhs,
    const SunmmioTileLoopFusionPlannerScore &rhs);

int CompareSunmmioTileLoopFusionPlannerScores(
    const SunmmioTileLoopFusionPlannerScore &lhs,
    const SunmmioTileLoopFusionPlannerScore &rhs);

SunmmioTileLoopFusionPlannerScore
MakeInfiniteSunmmioTileLoopFusionPlannerScore();

} // namespace tl
} // namespace tvm
