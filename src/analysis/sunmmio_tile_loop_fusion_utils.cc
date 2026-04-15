#include "sunmmio_tile_loop_fusion_utils.h"

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

String PrimExprToString(const PrimExpr &expr) {
  std::ostringstream os;
  os << expr;
  return String(os.str());
}

void VarUseCollector::VisitExpr_(const VarNode *op) { seen_vars.insert(op); }

Map<Var, PrimExpr> BuildLogicalExecutionAxisSubstitution(
    const TileScopeRegionSummary &region,
    std::unordered_map<std::string, Var> *canonical_execution_vars) {
  Map<Var, PrimExpr> subst;
  ICHECK_EQ(region.execution_loops.size(),
            region.logical_execution_axis_keys.size())
      << "Expected one logical axis key per execution loop in region "
      << region.root_name;
  for (size_t i = 0; i < region.execution_loops.size(); ++i) {
    const For &loop = region.execution_loops[i];
    const std::string &axis_key = region.logical_execution_axis_keys[i];
    auto it = canonical_execution_vars->find(axis_key);
    if (it == canonical_execution_vars->end()) {
      it = canonical_execution_vars->emplace(axis_key, Var(axis_key)).first;
    }
    subst.Set(loop->loop_var, it->second);
  }
  return subst;
}

BufferRegion
NormalizeBufferRegionByLogicalExecutionAxes(const BufferRegion &region,
                                            const Map<Var, PrimExpr> &subst) {
  Array<Range> normalized_ranges;
  for (const Range &range : region->region) {
    normalized_ranges.push_back(Range::FromMinExtent(
        Substitute(range->min, subst), Substitute(range->extent, subst)));
  }
  return BufferRegion(region->buffer, normalized_ranges);
}

std::vector<NormalizedTileScopeRegionSummary>
NormalizeRegionBoundaries(const std::vector<TileScopeRegionSummary> &regions) {
  std::unordered_map<std::string, Var> canonical_execution_vars;
  std::vector<NormalizedTileScopeRegionSummary> normalized_regions;
  normalized_regions.reserve(regions.size());

  for (const TileScopeRegionSummary &region : regions) {
    Map<Var, PrimExpr> subst = BuildLogicalExecutionAxisSubstitution(
        region, &canonical_execution_vars);

    Array<BufferRegion> normalized_use_in;
    for (const BufferRegion &buffer_region : region.use_in) {
      normalized_use_in.push_back(
          NormalizeBufferRegionByLogicalExecutionAxes(buffer_region, subst));
    }

    Array<BufferRegion> normalized_def_out;
    for (const BufferRegion &buffer_region : region.def_out) {
      normalized_def_out.push_back(
          NormalizeBufferRegionByLogicalExecutionAxes(buffer_region, subst));
    }

    normalized_regions.push_back({normalized_use_in, normalized_def_out});
  }

  return normalized_regions;
}

} // namespace tl
} // namespace tvm
