#include "sunmmio_tile_loop_fusion_planner_internal.h"

#include <algorithm>
#include <sstream>
#include <unordered_map>

namespace tvm {
namespace tl {
namespace detail {

using namespace tir;

namespace {

std::vector<std::string>
ExtractLogicalExecutionAxes(const TileScopeRegion &region) {
  return region.logical_execution_axis_keys;
}

Array<PrimExpr> ExtractExecutionLoopExtents(const TileScopeRegion &region) {
  return region.execution_loop_extents;
}

std::string BufferRegionKey(const BufferRegion &region) {
  std::ostringstream os;
  os << region->buffer->name << '|';
  for (const Range &range : region->region) {
    os << PrimExprToString(range->min) << ':' << PrimExprToString(range->extent)
       << '|';
  }
  return os.str();
}

int64_t ComputeExecutionPrefixInstanceCount(const Array<PrimExpr> &extents,
                                            int depth) {
  if (depth <= 0) {
    return 1;
  }
  int64_t count = 1;
  int limit = std::min(depth, static_cast<int>(extents.size()));
  for (int i = 0; i < limit; ++i) {
    const auto *imm = extents[i].as<IntImmNode>();
    if (imm == nullptr) {
      return 1;
    }
    count = SaturatingMulSunmmioTileLoopFusionPlannerCost(count, imm->value);
  }
  return count;
}

int64_t ComputeValueInstanceCount(const Array<PrimExpr> &execution_loop_extents,
                                  int home_depth) {
  return ComputeExecutionPrefixInstanceCount(execution_loop_extents,
                                             home_depth);
}

int64_t
ComputeRawEdgeInstanceCount(const Array<PrimExpr> &src_execution_loop_extents,
                            const Array<PrimExpr> &dst_execution_loop_extents,
                            int rho) {
  return std::max(
      ComputeExecutionPrefixInstanceCount(src_execution_loop_extents, rho),
      ComputeExecutionPrefixInstanceCount(dst_execution_loop_extents, rho));
}

} // namespace

std::string RawConsumerKey(int origin_region_local_index,
                           int buffer_region_id) {
  std::ostringstream os;
  os << origin_region_local_index << ':' << buffer_region_id;
  return os.str();
}

WindowPlannerInput
BuildWindowPlannerInput(const SunmmioTileLoopFusionWindowProblem &problem) {
  WindowPlannerInput input;
  int num_regions = static_cast<int>(problem.regions.size());
  input.region_indices.reserve(num_regions);
  input.regions.reserve(num_regions);
  input.incoming_edges_by_dst.resize(num_regions);
  input.outgoing_edges_by_src.resize(num_regions);
  input.predecessor_masks.assign(num_regions, DynamicBitset(num_regions));
  input.earlier_source_masks.assign(num_regions, DynamicBitset(num_regions));

  std::unordered_map<int, int> local_index_by_global;
  std::unordered_map<std::string, int> region_id_by_key;

  auto get_region_id = [&](const BufferRegion &region) {
    std::string key = BufferRegionKey(region);
    auto it = region_id_by_key.find(key);
    if (it != region_id_by_key.end()) {
      return it->second;
    }
    int id = static_cast<int>(region_id_by_key.size());
    region_id_by_key.emplace(key, id);
    return id;
  };

  for (int local_index = 0; local_index < num_regions; ++local_index) {
    const TileScopeRegion &region = problem.regions[local_index];
    int global_region_index = region.global_region_index;
    input.region_indices.push_back(global_region_index);
    local_index_by_global[global_region_index] = local_index;

    const NormalizedTileScopeRegion &normalized =
        problem.normalized_regions[local_index];

    WindowPlannerRegionInfo info;
    info.global_region_index = global_region_index;
    info.source_order_index = local_index;
    info.logical_execution_axes = ExtractLogicalExecutionAxes(region);
    info.execution_loop_extents = ExtractExecutionLoopExtents(region);

    for (const NormalizedBufferAccess &use_access : normalized.use_in) {
      info.use_in.push_back(
          {get_region_id(use_access.region),
           static_cast<std::string>(use_access.region->buffer->name),
           use_access.region, use_access.home_depth, use_access.payload_bytes,
           ComputeValueInstanceCount(info.execution_loop_extents,
                                     use_access.home_depth)});
    }
    for (const NormalizedBufferAccess &def_access : normalized.def_out) {
      info.def_out.push_back(
          {get_region_id(def_access.region),
           static_cast<std::string>(def_access.region->buffer->name),
           def_access.region, def_access.home_depth, def_access.payload_bytes,
           ComputeValueInstanceCount(info.execution_loop_extents,
                                     def_access.home_depth)});
    }

    input.regions.push_back(std::move(info));
    for (int earlier = 0; earlier < local_index; ++earlier) {
      SetBit(&input.earlier_source_masks[local_index], earlier);
    }
  }

  input.edges.reserve(problem.graph.edges.size());
  for (const TileScopeDependenceEdge &edge : problem.graph.edges) {
    auto src_it = local_index_by_global.find(edge.src_region_index);
    auto dst_it = local_index_by_global.find(edge.dst_region_index);
    if (src_it == local_index_by_global.end() ||
        dst_it == local_index_by_global.end()) {
      continue;
    }

    WindowPlannerEdgeInfo planner_edge;
    planner_edge.src_local_index = src_it->second;
    planner_edge.dst_local_index = dst_it->second;
    planner_edge.kind = edge.kind;
    planner_edge.rho = edge.rho;
    planner_edge.weight = edge.weight_bytes;
    planner_edge.instance_count = ComputeRawEdgeInstanceCount(
        input.regions[planner_edge.src_local_index].execution_loop_extents,
        input.regions[planner_edge.dst_local_index].execution_loop_extents,
        planner_edge.rho);

    if (edge.kind == TileScopeDependenceKind::kRAW) {
      const auto &uses = input.regions[planner_edge.dst_local_index].use_in;
      ICHECK_GE(edge.dst_access_index, 0);
      ICHECK_LT(edge.dst_access_index, static_cast<int>(uses.size()));
      const PlannerBufferValueInfo &covered_use = uses[edge.dst_access_index];
      planner_edge.buffer_region_id = covered_use.buffer_region_id;
      planner_edge.buffer_name = covered_use.buffer_name;
      planner_edge.covered_use_indices.push_back(edge.dst_access_index);
    } else if (edge.kind == TileScopeDependenceKind::kWAR) {
      const auto &uses = input.regions[planner_edge.src_local_index].use_in;
      ICHECK_GE(edge.src_access_index, 0);
      ICHECK_LT(edge.src_access_index, static_cast<int>(uses.size()));
      const PlannerBufferValueInfo &source_use = uses[edge.src_access_index];
      planner_edge.buffer_region_id = source_use.buffer_region_id;
      planner_edge.buffer_name = source_use.buffer_name;
    } else {
      const auto &defs = input.regions[planner_edge.dst_local_index].def_out;
      ICHECK_GE(edge.dst_access_index, 0);
      ICHECK_LT(edge.dst_access_index, static_cast<int>(defs.size()));
      const PlannerBufferValueInfo &dst_def = defs[edge.dst_access_index];
      planner_edge.buffer_region_id = dst_def.buffer_region_id;
      planner_edge.buffer_name = dst_def.buffer_name;
    }

    int edge_index = static_cast<int>(input.edges.size());
    input.edges.push_back(planner_edge);
    input.incoming_edges_by_dst[planner_edge.dst_local_index].push_back(
        edge_index);
    input.outgoing_edges_by_src[planner_edge.src_local_index].push_back(
        edge_index);
    SetBit(&input.predecessor_masks[planner_edge.dst_local_index],
           planner_edge.src_local_index);

    if (planner_edge.kind == TileScopeDependenceKind::kRAW) {
      std::string key = RawConsumerKey(planner_edge.src_local_index,
                                       planner_edge.buffer_region_id);
      auto it = input.raw_consumer_masks_by_key.find(key);
      if (it == input.raw_consumer_masks_by_key.end()) {
        it = input.raw_consumer_masks_by_key
                 .emplace(key, DynamicBitset(num_regions))
                 .first;
      }
      SetBit(&it->second, planner_edge.dst_local_index);
    }
  }

  for (int local_index = 0; local_index < num_regions; ++local_index) {
    for (const PlannerBufferValueInfo &use_info :
         input.regions[local_index].use_in) {
      auto it = input.read_consumer_masks_by_region_id.find(
          use_info.buffer_region_id);
      if (it == input.read_consumer_masks_by_region_id.end()) {
        it = input.read_consumer_masks_by_region_id
                 .emplace(use_info.buffer_region_id, DynamicBitset(num_regions))
                 .first;
      }
      SetBit(&it->second, local_index);
    }
  }

  return input;
}

} // namespace detail
} // namespace tl
} // namespace tvm
