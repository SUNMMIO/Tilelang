/*!
 * \file types.h
 * \brief Shared stage-boundary data for Sunmmio tile loop fusion.
 *
 * The pass is intentionally staged as:
 *
 *   PrimFunc
 *     -> SunmmioTileLoopFusionProgram
 *     -> SunmmioTileLoopFusionWindowProblem
 *     -> SunmmioTileLoopFusionWindowPlan
 *     -> rewritten PrimFunc
 *
 * Discovery owns the program and window-problem summaries. Planner owns
 * choosing shared execution shells per window. Rewrite consumes the resulting
 * tree and re-emits TIR.
 *
 * This header only contains data exchanged across stage boundaries. Planner
 * search state and rewrite-local helper state stay in stage-local sources.
 */

#pragma once

#include "cost_model.h"

#include "../../support/ffi_aliases.h"

#include <tvm/ir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>

#include <cstdint>
#include <string>
#include <vector>

namespace tvm {
namespace tl {

/*!
 * \brief One planner-visible lowered region.
 *
 * Discovery needs a stable, immutable record for each `tile.scope_entry`
 * region so later stages never have to rediscover wrapper structure or
 * execution-loop metadata from raw TIR. This is the canonical region identity
 * used by the rest of the pipeline.
 */
struct TileScopeRegion {
  // Stable global region id within the enclosing program, in source order.
  int global_region_index{-1};
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

/*!
 * \brief One maximal consecutive source-order run of planner-visible regions.
 *
 * Discovery only needs to expose where fusion may happen inside the enclosing
 * TIR statement list. A run is the minimal faithful representation of that
 * partition: it records one consecutive region interval without pretending to
 * be a richer semantic planning object.
 */
struct TileScopeRegionRun {
  // First global region id in this run, inclusive.
  int begin_region_index{-1};
  // Number of consecutive regions in the run.
  int num_regions{0};
};

enum class TileScopeDependenceKind : int {
  kRAW = 0,
  kWAR = 1,
  kWAW = 2,
};

/*!
 * \brief One normalized buffer dimension plus its logical-axis provenance.
 *
 * Dependence analysis needs more than a normalized expression string. It must
 * know which shared execution depths each dimension depends on so `rho` can be
 * computed from the input accesses directly rather than recovered from a later
 * overlap expression.
 */
struct NormalizedBufferAccessDim {
  PrimExpr min;
  PrimExpr extent;
  std::vector<int> execution_axis_depths;
};

/*!
 * \brief One logical-axis-normalized external access.
 *
 * The planner and dependence builder both need the same semantic facts for an
 * access: the normalized region shape, the execution-depth provenance of each
 * dimension, the depth at which the value becomes available, and the payload
 * size used by the cost model.
 */
struct NormalizedBufferAccess {
  tir::BufferRegion region;
  std::vector<NormalizedBufferAccessDim> dims;
  int home_depth{0};
  int64_t payload_bytes{0};
};

/*!
 * \brief One legality/profitability dependence between two discovered regions.
 *
 * The planner should reason about precomputed hazards and costs, not about raw
 * buffer-region overlap or logical-axis inference. Discovery constructs these
 * edges once and the planner treats them as immutable facts.
 */
struct TileScopeDependenceEdge {
  // Producer region within the global region list.
  int src_region_index{-1};
  // Consumer region within the global region list.
  int dst_region_index{-1};
  // Legality hazard kind between the two regions.
  TileScopeDependenceKind kind{TileScopeDependenceKind::kRAW};
  // Source access index within the owning source region's normalized access
  // list. For RAW/WAW this indexes `def_out`; for WAR it indexes `use_in`.
  int src_access_index{-1};
  // Destination access index within the owning destination region's normalized
  // access list. For RAW/WAR this indexes `use_in`/`def_out` respectively.
  int dst_access_index{-1};
  // Required shared execution depth for internalizing this dependence.
  // rho = 1 means the edge can stay internal under a shared outer row shell;
  // rho = 2 means it needs a deeper tile shell, and so on.
  int rho{0};
  // Estimated payload cost of cutting this edge for one execution instance.
  // The planner later scales this by execution multiplicity.
  int64_t weight_bytes{0};
};

/*!
 * \brief Dependence graph for one planner window.
 *
 * This is the last analysis-shaped object before the planner-facing problem is
 * formed. Membership is owned by the surrounding window problem; this graph
 * carries only the pairwise legality/profitability relationships between those
 * regions.
 */
struct TileScopeWindowGraph {
  // Pairwise hazards and weighted profitability summaries between the owning
  // window problem's regions.
  std::vector<TileScopeDependenceEdge> edges;
};

/*!
 * \brief Logical-axis-normalized boundary information for one discovered
 * region.
 *
 * Discovery compares regions across lowered loop-variable names by mapping them
 * into a canonical logical-axis space once. The planner consumes the normalized
 * form so it never needs to understand that normalization logic or recover
 * access provenance from expressions a second time.
 */
struct NormalizedTileScopeRegion {
  std::vector<NormalizedBufferAccess> use_in;
  std::vector<NormalizedBufferAccess> def_out;
};

/*!
 * \brief Full discovery result for one PrimFunc.
 *
 * The old pipeline rediscovered planner-visible regions twice. This program
 * object makes discovery a first-class stage output so both planning and
 * rewrite can share the same canonical region list and source-order run
 * partition.
 */
struct SunmmioTileLoopFusionProgram {
  // All planner-visible regions in source order.
  std::vector<TileScopeRegion> regions;
  // Maximal source-order runs of consecutive planner-visible regions.
  std::vector<TileScopeRegionRun> region_runs;
};

/*!
 * \brief Self-contained planning problem for one discovered window.
 *
 * The planner should not know about the full PrimFunc, global statement walks,
 * or ad hoc rematerialization of normalized data. This object packages exactly
 * the per-window semantic facts that the planner needs and nothing about how
 * they were discovered.
 */
struct SunmmioTileLoopFusionWindowProblem {
  // Participating regions in source-order window order. Each region retains its
  // `global_region_index` so later stages can still refer back to the enclosing
  // program when needed.
  std::vector<TileScopeRegion> regions;
  // Normalized region boundaries aligned with `regions`.
  std::vector<NormalizedTileScopeRegion> normalized_regions;
  // Window-local legality and profitability graph for `regions`.
  TileScopeWindowGraph graph;
};

/*!
 * \brief Planner trace record for one scheduled region.
 *
 * The exact DP internally reasons in terms of close/open shell actions. This
 * record remains planner-internal state that tests may build directly, while
 * rewrite consumes the derived tree below.
 */
struct SunmmioTileLoopFusionPlannerAction {
  // Global region id scheduled by this action.
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

/*!
 * \brief Rewrite-ready shared-shell tree for one planned window.
 *
 * Rewrite consumes a structural description of fused shells rather than
 * planner state transitions. This tree is the production handoff from planning
 * to rewriting.
 */
struct SunmmioTileLoopFusionPlannerTreeNode {
  // True for a shared execution shell node, false for a concrete region leaf.
  bool is_scope{false};
  // Global region id for leaf nodes; `-1` for scope nodes.
  int region_index{-1};
  // Logical execution-axis labels shared by this scope node.
  std::vector<std::string> shell_axes;
  // Execution extents of the shared shell. Two regions can only share this node
  // when both the axes and these extents match.
  Array<PrimExpr> shell_extents;
  // Child scope nodes and/or region leaves in execution order.
  std::vector<SunmmioTileLoopFusionPlannerTreeNode> children;
};

/*!
 * \brief Final planner result for one window problem.
 *
 * This is the production contract between planning and rewrite. The tree is
 * the payload rewrite consumes; the score and source-order region ids are kept
 * for bookkeeping and test assertions.
 */
struct SunmmioTileLoopFusionWindowPlan {
  // Global region ids participating in this plan, in source-order window order.
  std::vector<int> region_indices;
  // Final lexicographic planner score for the chosen schedule.
  SunmmioTileLoopFusionPlannerScore score;
  // Hierarchical fused-shell tree emitted by the planner.
  std::vector<SunmmioTileLoopFusionPlannerTreeNode> tree;
};

} // namespace tl
} // namespace tvm
