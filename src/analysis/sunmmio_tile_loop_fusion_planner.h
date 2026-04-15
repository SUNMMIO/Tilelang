#pragma once

#include "sunmmio_tile_loop_fusion_analysis.h"
#include "sunmmio_tile_loop_fusion_cost_model.h"

#include <string>
#include <vector>

namespace tvm {
namespace tl {

// Planner-side summary types for the exact Sunmmio tile loop fusion search.
//
// A planner shell is a shared execution-prefix scope chosen by the planner.
// Regions may attach to only part of their execution loops: a rank-2 region can
// live under a shell `[i]` while keeping its inner `j` loop private.

struct SunmmioTileLoopFusionPlannerActionSummary {
  // Region scheduled by this action.
  int region_index{-1};
  // Shared execution depth kept open after closing any deeper scopes.
  int close_to_depth{0};
  // Shared execution depth used for the scheduled region after opening any new
  // shells. `open_to_depth` may be smaller than the region rank when only an
  // outer execution prefix is shared.
  int open_to_depth{0};
  // Logical execution-axis labels for each newly opened shell frame.
  std::vector<std::vector<std::string>> opened_shells;
  // Execution extents paired with `opened_shells`. These extents are part of
  // shell legality, not just a debug annotation.
  std::vector<Array<PrimExpr>> opened_shell_extents;
};

struct SunmmioTileLoopFusionPlannerTreeNode {
  // True for a shared execution shell node, false for a concrete region leaf.
  bool is_scope{false};
  // Region index for leaf nodes; `-1` for scope nodes.
  int region_index{-1};
  // Logical execution-axis labels shared by this scope node.
  std::vector<std::string> shell_axes;
  // Execution extents of the shared shell. Two regions can only share this node
  // when both the axes and these extents match.
  Array<PrimExpr> shell_extents;
  // Child scope nodes and/or region leaves in execution order.
  std::vector<SunmmioTileLoopFusionPlannerTreeNode> children;
};

struct SunmmioTileLoopFusionWindowPlanSummary {
  // Window-local region set for which this plan was built.
  std::vector<int> region_indices;
  // Chosen region order within that window.
  std::vector<int> order;
  // Final lexicographic planner score for the chosen schedule.
  SunmmioTileLoopFusionPlannerScore score;
  // Action trace reconstructed from the DP backpointers.
  std::vector<SunmmioTileLoopFusionPlannerActionSummary> actions;
  // Hierarchical fused-shell tree implied by `actions`.
  std::vector<SunmmioTileLoopFusionPlannerTreeNode> tree;
};

std::vector<SunmmioTileLoopFusionWindowPlanSummary>
PlanSunmmioTileLoopFusionWindows(
    const std::vector<TileScopeRegionSummary> &regions,
    const std::vector<TileScopeWindowSummary> &windows,
    const std::vector<TileScopeWindowGraphSummary> &graphs);

} // namespace tl
} // namespace tvm
