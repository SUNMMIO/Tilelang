#include "sunmmio_tile_loop_fusion_planner_internal.h"

#include <algorithm>
#include <sstream>
#include <unordered_map>

namespace tvm {
namespace tl {
namespace detail {

using namespace tir;

namespace {

int ComputeBufferRegionHomeDepth(
    const BufferRegion &region,
    const std::vector<std::string> &logical_execution_axis_keys) {
  std::unordered_map<std::string, int> execution_depth_by_name;
  for (size_t i = 0; i < logical_execution_axis_keys.size(); ++i) {
    execution_depth_by_name[logical_execution_axis_keys[i]] =
        static_cast<int>(i) + 1;
  }

  int home_depth = 0;
  for (const Range &range : region->region) {
    VarUseCollector collector;
    collector(range->min);
    collector(range->extent);
    for (const VarNode *var : collector.seen_vars) {
      auto it = execution_depth_by_name.find(
          static_cast<std::string>(var->name_hint));
      if (it != execution_depth_by_name.end()) {
        home_depth = std::max(home_depth, it->second);
      }
    }
  }
  if (home_depth == 0 && !logical_execution_axis_keys.empty()) {
    home_depth = std::min<int>(logical_execution_axis_keys.size(),
                               region->region.size());
  }
  return home_depth;
}

int64_t ComputeBufferRegionPayloadBytes(const BufferRegion &region) {
  int64_t payload = region->buffer->dtype.bytes();
  for (const Range &range : region->region) {
    const auto *imm = range->extent.as<IntImmNode>();
    if (imm == nullptr) {
      return 0;
    }
    payload *= imm->value;
  }
  return payload;
}

std::vector<std::string>
ExtractLogicalExecutionAxes(const TileScopeRegionSummary &region) {
  return region.logical_execution_axis_keys;
}

Array<PrimExpr>
ExtractExecutionLoopExtents(const TileScopeRegionSummary &region) {
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

WindowPlannerInput BuildWindowPlannerInput(
    const std::vector<TileScopeRegionSummary> &regions,
    const std::vector<NormalizedTileScopeRegionSummary> &normalized_regions,
    const TileScopeWindowGraphSummary &graph) {
  WindowPlannerInput input;
  input.region_indices = graph.region_indices;
  int num_regions = static_cast<int>(graph.region_indices.size());
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
    int global_region_index = graph.region_indices[local_index];
    local_index_by_global[global_region_index] = local_index;

    const TileScopeRegionSummary &region = regions[global_region_index];
    const NormalizedTileScopeRegionSummary &normalized =
        normalized_regions[global_region_index];

    WindowPlannerRegionInfo info;
    info.global_region_index = global_region_index;
    info.source_order_index = local_index;
    info.logical_execution_axes = ExtractLogicalExecutionAxes(region);
    info.execution_loop_extents = ExtractExecutionLoopExtents(region);

    for (const BufferRegion &use_region : normalized.use_in) {
      int home_depth = ComputeBufferRegionHomeDepth(
          use_region, region.logical_execution_axis_keys);
      info.use_in.push_back(
          {get_region_id(use_region),
           static_cast<std::string>(use_region->buffer->name), use_region,
           home_depth, ComputeBufferRegionPayloadBytes(use_region),
           ComputeValueInstanceCount(info.execution_loop_extents, home_depth)});
    }
    for (size_t i = 0; i < normalized.def_out.size(); ++i) {
      const BufferRegion &def_region = normalized.def_out[i];
      int home_depth = 0;
      if (i < region.available_at_execution_depths.size()) {
        home_depth = region.available_at_execution_depths[i];
      } else {
        home_depth = ComputeBufferRegionHomeDepth(
            def_region, region.logical_execution_axis_keys);
      }
      info.def_out.push_back(
          {get_region_id(def_region),
           static_cast<std::string>(def_region->buffer->name), def_region,
           home_depth, ComputeBufferRegionPayloadBytes(def_region),
           ComputeValueInstanceCount(info.execution_loop_extents, home_depth)});
    }

    input.regions.push_back(std::move(info));
    for (int earlier = 0; earlier < local_index; ++earlier) {
      SetBit(&input.earlier_source_masks[local_index], earlier);
    }
  }

  input.edges.reserve(graph.edges.size());
  for (const TileScopeDependenceEdgeSummary &edge : graph.edges) {
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
    planner_edge.buffer_region_id = get_region_id(edge.buffer_region);
    planner_edge.buffer_name =
        static_cast<std::string>(edge.buffer_region->buffer->name);
    planner_edge.rho = edge.rho;
    if (const auto *imm = edge.weight.as<IntImmNode>()) {
      planner_edge.weight = imm->value;
    } else {
      planner_edge.weight = 0;
    }
    planner_edge.instance_count = ComputeRawEdgeInstanceCount(
        input.regions[planner_edge.src_local_index].execution_loop_extents,
        input.regions[planner_edge.dst_local_index].execution_loop_extents,
        planner_edge.rho);
    if (planner_edge.kind == TileScopeDependenceKind::kRAW) {
      const auto &uses = input.regions[planner_edge.dst_local_index].use_in;
      for (size_t use_index = 0; use_index < uses.size(); ++use_index) {
        if (uses[use_index].buffer_region_id == planner_edge.buffer_region_id) {
          planner_edge.covered_use_indices.push_back(
              static_cast<int>(use_index));
        }
      }
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
