#pragma once

#include <tvm/ir/expr.h>
#include <tvm/ir/module.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>

#include "../support/ffi_aliases.h"

#include <cstdint>
#include <vector>

namespace tvm {
namespace tl {

// Analysis-side data model for Sunmmio tile loop fusion.
//
// A region is one lowered planner-visible `tile.scope_entry` region after the
// Sunmmio tile-op and tile-loop lowering passes.
//
// A window is one maximal consecutive sibling run of such regions in the
// lowered TIR.
//
// A legality edge records both the ordering hazard (RAW/WAR/WAW) and the
// profitability summary used by the planner within one window.

struct TileScopeSignatureSummary {
  // Number of exposed execution loops under the matched `tile.scope_entry`.
  int rank{0};
  // Extent of each interior tile axis. This is the local tile shape executed by
  // the region, not the outer execution-loop trip count.
  Array<PrimExpr> tile_shape;
  // Inverse map from logical execution-axis id to lexical execution-loop index.
  // Example: if execution axis 0 appears as the second loop, this stores `1`.
  std::vector<int> execution_axis_to_loop_index;
};

enum class TileScopeRegionKind : int {
  // Standard lowered Sunmmio tile region rooted at one `tile.scope_entry` loop.
  kScopeEntryLoop = 0,
};

struct TileScopeRegionSummary {
  // Kind of planner-visible lowered region. Kept explicit so future region
  // kinds can be added without changing the surrounding APIs.
  TileScopeRegionKind kind{TileScopeRegionKind::kScopeEntryLoop};
  // Full wrapped statement that owns the matched region in the lowered TIR.
  tir::Stmt root_stmt;
  // The actual lowered `tile.scope_entry` loop representing this region.
  tir::For scope_entry_for;
  // Stable debug name derived from the region root.
  std::string root_name;
  // Exposed execution loops directly under the scope entry, in lexical order.
  std::vector<tir::For> execution_loops;
  // Raw lowered loop variable names for `execution_loops`, used for debugging.
  std::vector<std::string> execution_loop_var_names;
  // Canonical logical axis keys for `execution_loops`, such as `i`, `j`, `k`.
  // These are used to compare regions even when the lowered loop vars differ.
  std::vector<std::string> logical_execution_axis_keys;
  // Trip counts of the exposed execution loops. These extents participate in
  // shell legality because two regions can only share a shell when the shared
  // execution axes and extents are compatible.
  Array<PrimExpr> execution_loop_extents;
  // Interior tile signature of the region body.
  TileScopeSignatureSummary sig;
  // External reads required when entering the region. Internal scratch/local
  // buffers are filtered out before reaching this summary.
  Array<tir::BufferRegion> use_in;
  // External writes produced by the region.
  Array<tir::BufferRegion> def_out;
  // For each `def_out`, the smallest shared execution depth at which that value
  // is available to later regions.
  // Example: a rank-2 reduction region may still produce a row value available
  // at execution depth 1.
  std::vector<int> available_at_execution_depths;
};

struct TileScopeWindowSummary {
  // Indices of regions that belong to one maximal consecutive planner-visible
  // sibling run in source order.
  std::vector<int> region_indices;
};

enum class TileScopeDependenceKind : int {
  kRAW = 0,
  kWAR = 1,
  kWAW = 2,
};

struct TileScopeDependenceEdgeSummary {
  // Producer region within the global region list.
  int src_region_index{-1};
  // Consumer region within the global region list.
  int dst_region_index{-1};
  // Legality hazard kind between the two regions.
  TileScopeDependenceKind kind{TileScopeDependenceKind::kRAW};
  // Overlap fragment or representative buffer region used by the edge.
  tir::BufferRegion buffer_region;
  // Required shared execution depth for internalizing this dependence.
  // rho = 1 means the edge can stay internal under a shared outer row shell;
  // rho = 2 means it needs a deeper tile shell, and so on.
  int rho{0};
  // Estimated payload cost of cutting this edge for one execution instance.
  // The planner later scales this by execution multiplicity.
  PrimExpr weight;
};

struct TileScopeWindowGraphSummary {
  // Regions participating in this window-local legality/profitability graph.
  std::vector<int> region_indices;
  // Pairwise hazards and weighted profitability summaries between those
  // regions.
  std::vector<TileScopeDependenceEdgeSummary> edges;
};

std::vector<TileScopeRegionSummary>
AnalyzeSunmmioTileLoopFusionRegions(const tir::PrimFunc &func);

std::vector<TileScopeWindowSummary>
BuildSunmmioTileLoopFusionWindows(const tir::PrimFunc &func);

std::vector<TileScopeWindowGraphSummary>
BuildSunmmioTileLoopFusionLegalityGraphs(
    const std::vector<TileScopeRegionSummary> &regions,
    const std::vector<TileScopeWindowSummary> &windows);

Map<String, ffi::Any>
DebugSunmmioTileLoopFusionAnalysisSummary(const IRModule &mod);

} // namespace tl
} // namespace tvm
