/*!
 * \file cost_model.h
 * \brief Planner score representation and arithmetic helpers for Sunmmio tile
 * loop fusion.
 *
 * The planner compares candidate schedules lexicographically instead of
 * collapsing every objective into one scalar. This is a compile-time heuristic
 * cost model from the planner's point of view: it estimates how much reuse is
 * preserved or lost by a schedule, not a calibrated runtime or cycle model.
 *
 * The first three score terms are byte-like quantities accumulated from the
 * discovered accesses and planner state:
 * - write-cut cost estimates bytes paid when a RAW producer/consumer overlap
 *   is cut instead of kept internal under a shared shell.
 * - shared-read cost estimates bytes that must be materialized because a use
 *   is not already covered by a resident value.
 * - live-range penalty estimates bytes kept live in open shared shells.
 *
 * The final term is a structural tie-breaker counted in region inversions, not
 * bytes. There are no scalar exchange-rate weights between these terms; the
 * weighting is the lexicographic priority itself.
 */
#pragma once

#include <cstdint>

namespace tvm {
namespace tl {

/*!
 * \brief Lexicographic planner score.
 *
 * Fields are ordered by priority: first avoid cutting producer definitions,
 * then avoid extra shared reads, then minimize live resident footprint, and
 * finally prefer source-order schedules when everything else ties.
 *
 * Importantly, these are not summed with user-tunable coefficients. A unit
 * improvement in an earlier field always dominates any change in a later one.
 */
struct SunmmioTileLoopFusionPlannerScore {
  // Estimated bytes paid when a RAW overlap must escape the fused shell.
  // Discovery computes the per-instance overlap payload and planner multiplies
  // it by the execution-prefix multiplicity of that edge.
  int64_t write_cut_cost{0};
  // Estimated bytes read/materialized for uncovered uses. This is charged when
  // the planner cannot satisfy a use from a visible resident definition or
  // resident read value.
  int64_t shared_read_cost{0};
  // Estimated bytes kept live across open scopes after one action. This is the
  // sum of resident payload sizes scaled by their execution multiplicity.
  int64_t live_range_penalty{0};
  // Count of earlier source-order regions skipped over by the chosen action.
  // This is only a deterministic tie-breaker once the byte-based objectives
  // are equal.
  int64_t reorder_penalty{0};
};

/*! \brief Saturating add used by planner cost accumulation. */
int64_t SaturatingAddSunmmioTileLoopFusionPlannerCost(int64_t lhs, int64_t rhs);

/*! \brief Saturating multiply used when scaling payload cost by instance count.
 */
int64_t SaturatingMulSunmmioTileLoopFusionPlannerCost(int64_t lhs, int64_t rhs);

/*! \brief Add two planner scores componentwise with saturation. */
SunmmioTileLoopFusionPlannerScore AddSunmmioTileLoopFusionPlannerScores(
    const SunmmioTileLoopFusionPlannerScore &lhs,
    const SunmmioTileLoopFusionPlannerScore &rhs);

/*!
 * \brief Compare two planner scores lexicographically.
 *
 * There is no weighted sum or normalization across fields. The comparison is
 * a strict priority order:
 * `write_cut_cost > shared_read_cost > live_range_penalty > reorder_penalty`.
 * \return `-1`, `0`, or `1` depending on whether \p lhs is better than, equal
 * to, or worse than \p rhs.
 */
int CompareSunmmioTileLoopFusionPlannerScores(
    const SunmmioTileLoopFusionPlannerScore &lhs,
    const SunmmioTileLoopFusionPlannerScore &rhs);

/*! \brief Return the sentinel score used for impossible or pruned plans. */
SunmmioTileLoopFusionPlannerScore
MakeInfiniteSunmmioTileLoopFusionPlannerScore();

} // namespace tl
} // namespace tvm
