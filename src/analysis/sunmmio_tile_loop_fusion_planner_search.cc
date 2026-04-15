#include "sunmmio_tile_loop_fusion_planner_internal.h"

#include <unordered_set>

namespace tvm {
namespace tl {
namespace detail {

namespace {

std::vector<std::vector<std::string>>
CollectOpenedShells(const std::vector<std::string> &region_execution_axes,
                    int close_to_depth, int open_to_depth) {
  std::vector<std::vector<std::string>> opened_shells;
  for (int depth = close_to_depth + 1; depth <= open_to_depth; ++depth) {
    opened_shells.push_back(
        TakeExecutionAxisPrefix(region_execution_axes, depth));
  }
  return opened_shells;
}

std::vector<Array<PrimExpr>>
CollectOpenedShellExtents(const Array<PrimExpr> &region_execution_extents,
                          int close_to_depth, int open_to_depth) {
  std::vector<Array<PrimExpr>> opened_shell_extents;
  for (int depth = close_to_depth + 1; depth <= open_to_depth; ++depth) {
    opened_shell_extents.push_back(
        TakeExecutionExtentPrefix(region_execution_extents, depth));
  }
  return opened_shell_extents;
}

} // namespace

TransitionResult ApplyAction(const WindowPlannerInput &input,
                             const PlannerState &state, int region_local_index,
                             int close_to_depth, int open_to_depth) {
  TransitionResult result{state, {0, 0, 0, 0}};
  const WindowPlannerRegionInfo &region = input.regions[region_local_index];

  result.next_state.open_scopes.resize(close_to_depth);
  for (int depth = close_to_depth + 1; depth <= open_to_depth; ++depth) {
    result.next_state.open_scopes.push_back(
        {TakeExecutionAxisPrefix(region.logical_execution_axes, depth),
         TakeExecutionExtentPrefix(region.execution_loop_extents, depth),
         {}});
  }

  std::unordered_set<int> raw_covered_use_indices;
  for (int edge_index : input.incoming_edges_by_dst[region_local_index]) {
    const WindowPlannerEdgeInfo &edge = input.edges[edge_index];
    if (edge.kind != TileScopeDependenceKind::kRAW) {
      continue;
    }
    raw_covered_use_indices.insert(edge.covered_use_indices.begin(),
                                   edge.covered_use_indices.end());
    if (!HasAccessibleDefinitionResident(result.next_state.open_scopes,
                                         open_to_depth, edge.src_local_index,
                                         edge.buffer_region_id, edge.rho)) {
      AccumulatePlannerScoreTerm(&result.delta.write_cut_cost,
                                 SaturatingMulSunmmioTileLoopFusionPlannerCost(
                                     edge.weight, edge.instance_count));
    }
    if (edge.rho <= open_to_depth) {
      InstallResidentIfMissing(&result.next_state.open_scopes,
                               {ResidentValueKind::kDefinition,
                                edge.src_local_index, edge.buffer_region_id,
                                edge.buffer_name, edge.rho, edge.weight,
                                edge.instance_count});
    }
  }

  for (size_t use_index = 0; use_index < region.use_in.size(); ++use_index) {
    const PlannerBufferValueInfo &use_info = region.use_in[use_index];
    if (raw_covered_use_indices.count(static_cast<int>(use_index)) != 0) {
      continue;
    }
    if (HasAccessibleResident(result.next_state.open_scopes, open_to_depth,
                              use_info.buffer_region_id, use_info.home_depth)) {
      continue;
    }
    AccumulatePlannerScoreTerm(
        &result.delta.shared_read_cost,
        SaturatingMulSunmmioTileLoopFusionPlannerCost(use_info.payload_bytes,
                                                      use_info.instance_count));
    if (use_info.home_depth <= open_to_depth) {
      InstallResidentIfMissing(&result.next_state.open_scopes,
                               {ResidentValueKind::kRead, -1,
                                use_info.buffer_region_id, use_info.buffer_name,
                                use_info.home_depth, use_info.payload_bytes,
                                use_info.instance_count});
    }
  }

  for (const PlannerBufferValueInfo &def_info : region.def_out) {
    KillResidentsForBuffer(&result.next_state.open_scopes,
                           def_info.buffer_name);
  }
  for (const PlannerBufferValueInfo &def_info : region.def_out) {
    if (def_info.home_depth <= open_to_depth) {
      InstallResidentIfMissing(
          &result.next_state.open_scopes,
          {ResidentValueKind::kDefinition, region_local_index,
           def_info.buffer_region_id, def_info.buffer_name, def_info.home_depth,
           def_info.payload_bytes, def_info.instance_count});
    }
  }
  for (int edge_index : input.outgoing_edges_by_src[region_local_index]) {
    const WindowPlannerEdgeInfo &edge = input.edges[edge_index];
    if (edge.kind != TileScopeDependenceKind::kRAW ||
        edge.rho > open_to_depth) {
      continue;
    }
    InstallResidentIfMissing(&result.next_state.open_scopes,
                             {ResidentValueKind::kDefinition,
                              region_local_index, edge.buffer_region_id,
                              edge.buffer_name, edge.rho, edge.weight,
                              edge.instance_count});
  }

  result.next_state.scheduled_mask = state.scheduled_mask;
  SetBit(&result.next_state.scheduled_mask, region_local_index);
  PruneDeadResidents(input, &result.next_state);
  result.delta.live_range_penalty = ComputeLiveRangeDelta(result.next_state);
  result.delta.reorder_penalty = CountMissingBits(
      input.earlier_source_masks[region_local_index], state.scheduled_mask);
  return result;
}

MemoResult BuildSourceOrderFallbackPlan(const WindowPlannerInput &input) {
  MemoResult fallback;
  fallback.score = {0, 0, 0, 0};
  PlannerState state{DynamicBitset(static_cast<int>(input.regions.size())), {}};

  int scheduled_count = 0;
  while (scheduled_count < static_cast<int>(input.regions.size())) {
    bool progressed = false;
    for (int region_local_index = 0;
         region_local_index < static_cast<int>(input.regions.size());
         ++region_local_index) {
      if (TestBit(state.scheduled_mask, region_local_index)) {
        continue;
      }
      if (!ContainsAll(input.predecessor_masks[region_local_index],
                       state.scheduled_mask)) {
        continue;
      }

      TransitionResult transition =
          ApplyAction(input, state, region_local_index, 0, 0);
      fallback.score = AddSunmmioTileLoopFusionPlannerScores(fallback.score,
                                                             transition.delta);

      SunmmioTileLoopFusionPlannerActionSummary action;
      action.region_index =
          input.regions[region_local_index].global_region_index;
      action.close_to_depth = 0;
      action.open_to_depth = 0;
      fallback.actions.push_back(std::move(action));

      state = std::move(transition.next_state);
      ++scheduled_count;
      progressed = true;
      break;
    }
    ICHECK(progressed)
        << "Expected a legal next region while building fallback plan";
  }

  return fallback;
}

MemoResult SolveWindowPlan(const WindowPlannerInput &input,
                           const PlannerState &state,
                           PlannerSearchContext *context) {
  if (context->exhausted) {
    return {MakeInfiniteSunmmioTileLoopFusionPlannerScore(), {}};
  }

  std::string key = SerializePlannerState(state);
  auto it = context->memo.find(key);
  if (it != context->memo.end()) {
    return it->second;
  }
  if (context->memo.size() >= kMaxPlannerMemoEntries) {
    context->exhausted = true;
    return {MakeInfiniteSunmmioTileLoopFusionPlannerScore(), {}};
  }

  bool all_scheduled = true;
  for (int region_local_index = 0;
       region_local_index < static_cast<int>(input.regions.size());
       ++region_local_index) {
    if (!TestBit(state.scheduled_mask, region_local_index)) {
      all_scheduled = false;
      break;
    }
  }
  if (all_scheduled) {
    MemoResult done;
    done.score = {0, 0, 0, 0};
    context->memo.emplace(key, done);
    return done;
  }

  MemoResult best;
  best.score = MakeInfiniteSunmmioTileLoopFusionPlannerScore();

  for (int region_local_index = 0;
       region_local_index < static_cast<int>(input.regions.size());
       ++region_local_index) {
    if (TestBit(state.scheduled_mask, region_local_index)) {
      continue;
    }
    if (!ContainsAll(input.predecessor_masks[region_local_index],
                     state.scheduled_mask)) {
      continue;
    }

    const WindowPlannerRegionInfo &region = input.regions[region_local_index];
    for (int close_to_depth = 0;
         close_to_depth <= static_cast<int>(state.open_scopes.size());
         ++close_to_depth) {
      if (!PathMatchesExecutionPrefix(state.open_scopes, close_to_depth,
                                      region.logical_execution_axes,
                                      region.execution_loop_extents)) {
        continue;
      }
      for (int open_to_depth = close_to_depth;
           open_to_depth <=
           static_cast<int>(region.logical_execution_axes.size());
           ++open_to_depth) {
        TransitionResult transition = ApplyAction(
            input, state, region_local_index, close_to_depth, open_to_depth);
        MemoResult suffix =
            SolveWindowPlan(input, transition.next_state, context);
        if (context->exhausted) {
          return {MakeInfiniteSunmmioTileLoopFusionPlannerScore(), {}};
        }
        SunmmioTileLoopFusionPlannerScore total =
            AddSunmmioTileLoopFusionPlannerScores(transition.delta,
                                                  suffix.score);
        if (CompareSunmmioTileLoopFusionPlannerScores(total, best.score) >= 0) {
          continue;
        }

        SunmmioTileLoopFusionPlannerActionSummary action;
        action.region_index = region.global_region_index;
        action.close_to_depth = close_to_depth;
        action.open_to_depth = open_to_depth;
        action.opened_shells = CollectOpenedShells(
            region.logical_execution_axes, close_to_depth, open_to_depth);
        action.opened_shell_extents = CollectOpenedShellExtents(
            region.execution_loop_extents, close_to_depth, open_to_depth);

        best.score = total;
        best.actions.clear();
        best.actions.push_back(std::move(action));
        best.actions.insert(best.actions.end(), suffix.actions.begin(),
                            suffix.actions.end());
      }
    }
  }

  context->memo.emplace(key, best);
  return best;
}

} // namespace detail
} // namespace tl
} // namespace tvm
