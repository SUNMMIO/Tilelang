#pragma once

#include "sunmmio_tile_loop_fusion_analysis.h"

#include <tvm/tir/stmt_functor.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

struct NormalizedTileScopeRegionSummary {
  Array<tir::BufferRegion> use_in;
  Array<tir::BufferRegion> def_out;
};

String PrimExprToString(const PrimExpr &expr);

class VarUseCollector : public tir::ExprVisitor {
public:
  std::unordered_set<const tir::VarNode *> seen_vars;

private:
  void VisitExpr_(const tir::VarNode *op) final;
};

Map<tir::Var, PrimExpr> BuildLogicalExecutionAxisSubstitution(
    const TileScopeRegionSummary &region,
    std::unordered_map<std::string, tir::Var> *canonical_execution_vars);

tir::BufferRegion NormalizeBufferRegionByLogicalExecutionAxes(
    const tir::BufferRegion &region, const Map<tir::Var, PrimExpr> &subst);

std::vector<NormalizedTileScopeRegionSummary>
NormalizeRegionBoundaries(const std::vector<TileScopeRegionSummary> &regions);

} // namespace tl
} // namespace tvm
