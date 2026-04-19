#include <gtest/gtest.h>

#include "transform/sunmmio_tile_loop_fusion/cost_model.h"

namespace tvm {
namespace tl {
namespace {

TEST(SunmmioTileLoopFusionCostModelTest, AddsPlannerScoresComponentwise) {
  SunmmioTileLoopFusionPlannerScore lhs{1, 2, 3, 4};
  SunmmioTileLoopFusionPlannerScore rhs{10, 20, 30, 40};

  SunmmioTileLoopFusionPlannerScore sum =
      AddSunmmioTileLoopFusionPlannerScores(lhs, rhs);

  EXPECT_EQ(sum.write_cut_cost, 11);
  EXPECT_EQ(sum.shared_read_cost, 22);
  EXPECT_EQ(sum.live_range_penalty, 33);
  EXPECT_EQ(sum.reorder_penalty, 44);
}

TEST(SunmmioTileLoopFusionCostModelTest, ComparesScoresLexicographically) {
  SunmmioTileLoopFusionPlannerScore preferred{1, 0, 99, 99};
  SunmmioTileLoopFusionPlannerScore rejected{1, 1, 0, 0};

  EXPECT_LT(CompareSunmmioTileLoopFusionPlannerScores(preferred, rejected), 0);
  EXPECT_GT(CompareSunmmioTileLoopFusionPlannerScores(rejected, preferred), 0);
}

TEST(SunmmioTileLoopFusionCostModelTest, SaturatingHelpersClampInfiniteValues) {
  SunmmioTileLoopFusionPlannerScore inf =
      MakeInfiniteSunmmioTileLoopFusionPlannerScore();

  EXPECT_EQ(
      SaturatingAddSunmmioTileLoopFusionPlannerCost(inf.write_cut_cost, 1),
      inf.write_cut_cost);
  EXPECT_EQ(
      SaturatingMulSunmmioTileLoopFusionPlannerCost(inf.shared_read_cost, 2),
      inf.shared_read_cost);
}

} // namespace
} // namespace tl
} // namespace tvm
