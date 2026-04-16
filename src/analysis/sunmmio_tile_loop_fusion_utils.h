#pragma once

// Utility helpers shared by discovery and planning.
//
// These helpers do not define stage boundaries themselves. They exist to keep
// recurring logical-axis normalization and variable-collection code out of the
// higher-level analysis and planner entrypoints.

#include "sunmmio_tile_loop_fusion_protocol.h"

#include <tvm/tir/stmt_functor.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

String PrimExprToString(const PrimExpr &expr);

const char *DependenceKindToCString(TileScopeDependenceKind kind);

class VarUseCollector : public tir::ExprVisitor {
public:
  std::unordered_set<const tir::VarNode *> seen_vars;

private:
  void VisitExpr_(const tir::VarNode *op) final;
};

Map<tir::Var, PrimExpr> BuildLogicalExecutionAxisSubstitution(
    const TileScopeRegion &region,
    std::unordered_map<std::string, tir::Var> *canonical_execution_vars);

tir::BufferRegion NormalizeBufferRegionByLogicalExecutionAxes(
    const tir::BufferRegion &region, const Map<tir::Var, PrimExpr> &subst);

std::vector<NormalizedTileScopeRegion>
NormalizeRegionBoundaries(const std::vector<TileScopeRegion> &regions);

} // namespace tl
} // namespace tvm
