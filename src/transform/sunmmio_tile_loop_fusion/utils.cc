/*!
 * \file utils.cc
 * \brief Shared normalization helpers for Sunmmio tile loop fusion.
 */

#include "utils.h"

#include <tvm/arith/analyzer.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

// Record which logical execution depths a normalized buffer dimension depends
// on so later dependence construction can recover rho directly from the
// normalized access description.
std::vector<int> CollectExecutionAxisDepths(
    const PrimExpr &min, const PrimExpr &extent,
    const std::unordered_map<const VarNode *, int> &depth_by_var) {
  std::vector<int> depths;
  VarUseCollector collector;
  collector(min);
  collector(extent);
  for (const VarNode *var : collector.seen_vars) {
    auto it = depth_by_var.find(var);
    if (it != depth_by_var.end()) {
      depths.push_back(it->second);
    }
  }
  std::sort(depths.begin(), depths.end());
  depths.erase(std::unique(depths.begin(), depths.end()), depths.end());
  return depths;
}

// A value's "home depth" is the shallowest shared shell where it becomes
// available or must remain visible. When explicit metadata is missing, infer it
// from the deepest execution axis referenced by the access.
int ComputeAccessHomeDepth(const std::vector<NormalizedBufferAccessDim> &dims,
                           int max_execution_rank, int fallback_rank) {
  int home_depth = 0;
  for (const NormalizedBufferAccessDim &dim : dims) {
    for (int depth : dim.execution_axis_depths) {
      home_depth = std::max(home_depth, depth);
    }
  }
  if (home_depth == 0 && max_execution_rank > 0) {
    home_depth = std::min(max_execution_rank, fallback_rank);
  }
  return home_depth;
}

int64_t ComputeAccessPayloadBytes(const BufferRegion &region,
                                  arith::Analyzer *analyzer) {
  int64_t payload = region->buffer->dtype.bytes();
  for (const Range &range : region->region) {
    PrimExpr extent = analyzer->Simplify(range->extent);
    const auto *imm = extent.as<IntImmNode>();
    if (imm == nullptr) {
      return 0;
    }
    payload *= imm->value;
  }
  return payload;
}

// Build the fully planner-facing normalized access record used by both
// dependence construction and planner preprocessing.
NormalizedBufferAccess BuildNormalizedBufferAccess(
    const BufferRegion &region, const Map<Var, PrimExpr> &subst,
    const std::unordered_map<std::string, Var> &canonical_execution_vars,
    const std::vector<std::string> &logical_execution_axis_keys,
    int home_depth_override, arith::Analyzer *analyzer) {
  BufferRegion normalized_region =
      NormalizeBufferRegionByLogicalExecutionAxes(region, subst);

  std::unordered_map<const VarNode *, int> depth_by_var;
  for (size_t i = 0; i < logical_execution_axis_keys.size(); ++i) {
    auto it = canonical_execution_vars.find(logical_execution_axis_keys[i]);
    ICHECK(it != canonical_execution_vars.end())
        << "Missing canonical execution var for logical axis "
        << logical_execution_axis_keys[i];
    depth_by_var[it->second.get()] = static_cast<int>(i) + 1;
  }

  std::vector<NormalizedBufferAccessDim> dims;
  dims.reserve(normalized_region->region.size());
  for (const Range &range : normalized_region->region) {
    PrimExpr min = analyzer->Simplify(range->min);
    PrimExpr extent = analyzer->Simplify(range->extent);
    dims.push_back(
        {min, extent, CollectExecutionAxisDepths(min, extent, depth_by_var)});
  }

  int home_depth = home_depth_override;
  if (home_depth < 0) {
    home_depth = ComputeAccessHomeDepth(
        dims, static_cast<int>(logical_execution_axis_keys.size()),
        static_cast<int>(normalized_region->region.size()));
  }

  return {normalized_region, dims, home_depth,
          ComputeAccessPayloadBytes(normalized_region, analyzer)};
}

} // namespace

String PrimExprToString(const PrimExpr &expr) {
  std::ostringstream os;
  os << expr;
  return String(os.str());
}

const char *DependenceKindToCString(TileScopeDependenceKind kind) {
  switch (kind) {
  case TileScopeDependenceKind::kRAW:
    return "RAW";
  case TileScopeDependenceKind::kWAR:
    return "WAR";
  case TileScopeDependenceKind::kWAW:
    return "WAW";
  }
  LOG(FATAL) << "Unknown TileScopeDependenceKind value";
  return "unknown";
}

void VarUseCollector::VisitExpr_(const VarNode *op) { seen_vars.insert(op); }

Map<Var, PrimExpr> BuildLogicalExecutionAxisSubstitution(
    const TileScopeRegion &region,
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

std::vector<NormalizedTileScopeRegion>
NormalizeRegionBoundaries(const std::vector<TileScopeRegion> &regions) {
  // Canonical logical-axis vars are shared across every region in the window so
  // equal accesses normalize to the same symbolic coordinate system even when
  // their lowered loop vars differ.
  std::unordered_map<std::string, Var> canonical_execution_vars;
  std::vector<NormalizedTileScopeRegion> normalized_regions;
  normalized_regions.reserve(regions.size());
  arith::Analyzer analyzer;

  for (const TileScopeRegion &region : regions) {
    Map<Var, PrimExpr> subst = BuildLogicalExecutionAxisSubstitution(
        region, &canonical_execution_vars);

    NormalizedTileScopeRegion normalized;
    normalized.use_in.reserve(region.use_in.size());
    normalized.def_out.reserve(region.def_out.size());

    for (const BufferRegion &buffer_region : region.use_in) {
      normalized.use_in.push_back(BuildNormalizedBufferAccess(
          buffer_region, subst, canonical_execution_vars,
          region.logical_execution_axis_keys, /*home_depth_override=*/-1,
          &analyzer));
    }

    for (size_t i = 0; i < region.def_out.size(); ++i) {
      int home_depth = -1;
      if (i < region.available_at_execution_depths.size()) {
        home_depth = region.available_at_execution_depths[i];
      }
      normalized.def_out.push_back(BuildNormalizedBufferAccess(
          region.def_out[i], subst, canonical_execution_vars,
          region.logical_execution_axis_keys, home_depth, &analyzer));
    }

    normalized_regions.push_back(std::move(normalized));
  }

  return normalized_regions;
}

} // namespace tl
} // namespace tvm
