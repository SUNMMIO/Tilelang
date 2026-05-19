/*!
 * \file utils.h
 * \brief Shared logical-axis normalization and small helper utilities for
 * Sunmmio tile loop fusion.
 *
 * Discovery and planning both need the same canonicalization machinery for
 * execution-axis names, normalized buffer regions, and lightweight expression
 * inspection. This header keeps those helpers in one place without introducing
 * an additional stage boundary of its own.
 */
#pragma once

#include "types.h"

#include <tvm/tir/stmt_functor.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

/*! \brief Render a PrimExpr into a stable string for diagnostics and keys. */
String PrimExprToString(const PrimExpr &expr);

/*! \brief Return a printable name for one dependence kind. */
const char *DependenceKindToCString(TileScopeDependenceKind kind);

/*!
 * \brief Collect every Var referenced by the visited expression subtree.
 *
 * Discovery uses this to determine which normalized access dimensions depend on
 * which execution axes.
 */
class VarUseCollector : public tir::ExprVisitor {
public:
  std::unordered_set<const tir::VarNode *> seen_vars;

private:
  void VisitExpr_(const tir::VarNode *op) final;
};

/*!
 * \brief Replace a region's lowered execution-loop vars with canonical logical
 * axis vars such as `i`, `j`, and `k`.
 */
Map<tir::Var, PrimExpr> BuildLogicalExecutionAxisSubstitution(
    const TileScopeRegion &region,
    std::unordered_map<std::string, tir::Var> *canonical_execution_vars);

/*!
 * \brief Rewrite one BufferRegion into the shared logical-axis coordinate
 * space produced by \ref BuildLogicalExecutionAxisSubstitution.
 */
tir::BufferRegion NormalizeBufferRegionByLogicalExecutionAxes(
    const tir::BufferRegion &region, const Map<tir::Var, PrimExpr> &subst);

/*!
 * \brief Normalize each region's external use/def boundaries into a shared
 * logical-axis coordinate system and attach the planner-facing metadata used
 * by later dependence and planning stages.
 */
std::vector<NormalizedTileScopeRegion>
NormalizeRegionBoundaries(const std::vector<TileScopeRegion> &regions);

} // namespace tl
} // namespace tvm
