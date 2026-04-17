#include "sunmmio_tile_loop_fusion_planner_internal.h"

#include <tvm/node/structural_equal.h>

#include <algorithm>
#include <memory>
#include <sstream>
#include <tuple>
#include <unordered_set>

namespace tvm {
namespace tl {
namespace planner_internal {

namespace {

std::string JoinExtents(const Array<PrimExpr> &extents) {
  std::ostringstream os;
  for (size_t i = 0; i < extents.size(); ++i) {
    if (i != 0) {
      os << '/';
    }
    os << PrimExprToString(extents[i]);
  }
  return os.str();
}

std::string JoinAxes(const std::vector<std::string> &axes) {
  std::ostringstream os;
  for (size_t i = 0; i < axes.size(); ++i) {
    if (i != 0) {
      os << '/';
    }
    os << axes[i];
  }
  return os.str();
}

bool HasFuturePotentialConsumers(const WindowPlannerInput &input,
                                 const PlannerState &state,
                                 const ResidentValueState &resident) {
  if (resident.kind == ResidentValueKind::kRead) {
    auto it =
        input.read_consumer_masks_by_region_id.find(resident.buffer_region_id);
    return it != input.read_consumer_masks_by_region_id.end() &&
           HasAnyFutureBits(it->second, state.scheduled_mask);
  }
  RawConsumerKey key{resident.origin_region_local_index,
                     resident.buffer_region_id};
  auto it = input.raw_consumer_masks_by_key.find(key);
  return it != input.raw_consumer_masks_by_key.end() &&
         HasAnyFutureBits(it->second, state.scheduled_mask);
}

bool ResidentValueLess(const ResidentValueState &lhs,
                       const ResidentValueState &rhs) {
  return std::tie(lhs.kind, lhs.origin_region_local_index, lhs.buffer_region_id,
                  lhs.home_depth, lhs.payload_bytes, lhs.instance_count) <
         std::tie(rhs.kind, rhs.origin_region_local_index, rhs.buffer_region_id,
                  rhs.home_depth, rhs.payload_bytes, rhs.instance_count);
}

struct MutablePlannerTreeNode {
  bool is_scope{false};
  int region_index{-1};
  std::vector<std::string> shell_axes;
  Array<PrimExpr> shell_extents;
  std::vector<std::shared_ptr<MutablePlannerTreeNode>> children;
};

SunmmioTileLoopFusionPlannerTreeNode
FreezeTree(const std::shared_ptr<MutablePlannerTreeNode> &node) {
  SunmmioTileLoopFusionPlannerTreeNode frozen;
  frozen.is_scope = node->is_scope;
  frozen.region_index = node->region_index;
  frozen.shell_axes = node->shell_axes;
  frozen.shell_extents = node->shell_extents;
  for (const std::shared_ptr<MutablePlannerTreeNode> &child : node->children) {
    frozen.children.push_back(FreezeTree(child));
  }
  return frozen;
}

} // namespace

void SetBit(DynamicBitset *bitset, int index) {
  ICHECK_GE(index, 0);
  ICHECK_LT(index, bitset->num_bits) << "DynamicBitset index out of bounds";
  bitset->words[index / 64] |= (uint64_t{1} << (index % 64));
}

bool TestBit(const DynamicBitset &bitset, int index) {
  ICHECK_GE(index, 0);
  ICHECK_LT(index, bitset.num_bits) << "DynamicBitset index out of bounds";
  return ((bitset.words[index / 64] >> (index % 64)) & uint64_t{1}) != 0;
}

bool ContainsAll(const DynamicBitset &required, const DynamicBitset &present) {
  ICHECK_EQ(required.num_bits, present.num_bits);
  for (size_t i = 0; i < required.words.size(); ++i) {
    if ((required.words[i] & ~present.words[i]) != 0) {
      return false;
    }
  }
  return true;
}

bool HasAnyFutureBits(const DynamicBitset &candidate,
                      const DynamicBitset &scheduled) {
  ICHECK_EQ(candidate.num_bits, scheduled.num_bits);
  for (size_t i = 0; i < candidate.words.size(); ++i) {
    if ((candidate.words[i] & ~scheduled.words[i]) != 0) {
      return true;
    }
  }
  return false;
}

int CountBits(const DynamicBitset &bitset) {
  int count = 0;
  int remaining_bits = bitset.num_bits;
  for (uint64_t word : bitset.words) {
    uint64_t masked = word;
    if (remaining_bits < 64) {
      masked &=
          (remaining_bits <= 0) ? 0 : ((uint64_t{1} << remaining_bits) - 1);
    }
    count += __builtin_popcountll(masked);
    remaining_bits -= 64;
  }
  return count;
}

int CountMissingBits(const DynamicBitset &candidate,
                     const DynamicBitset &present) {
  ICHECK_EQ(candidate.num_bits, present.num_bits);
  DynamicBitset missing(candidate.num_bits);
  for (size_t i = 0; i < candidate.words.size(); ++i) {
    missing.words[i] = candidate.words[i] & ~present.words[i];
  }
  return CountBits(missing);
}

std::string SerializeDynamicBitset(const DynamicBitset &bitset) {
  std::ostringstream os;
  os << bitset.num_bits << ':';
  for (uint64_t word : bitset.words) {
    os << word << ';';
  }
  return os.str();
}

bool SameResidentValue(const ResidentValueState &lhs,
                       const ResidentValueState &rhs) {
  return lhs.kind == rhs.kind &&
         lhs.origin_region_local_index == rhs.origin_region_local_index &&
         lhs.buffer_region_id == rhs.buffer_region_id &&
         lhs.home_depth == rhs.home_depth &&
         lhs.payload_bytes == rhs.payload_bytes &&
         lhs.instance_count == rhs.instance_count;
}

std::string SerializeResident(const ResidentValueState &resident) {
  std::ostringstream os;
  os << static_cast<int>(resident.kind) << ':'
     << resident.origin_region_local_index << ':' << resident.buffer_region_id
     << ':' << resident.buffer_name << ':' << resident.home_depth << ':'
     << resident.payload_bytes << ':' << resident.instance_count;
  return os.str();
}

std::string SerializePlannerState(const PlannerState &state) {
  std::ostringstream os;
  os << SerializeDynamicBitset(state.scheduled_mask);
  for (const OpenScopeFrame &frame : state.open_scopes) {
    os << '[' << JoinAxes(frame.shell_axes) << '@'
       << JoinExtents(frame.shell_extents) << '|';
    for (const ResidentValueState &resident : frame.residents) {
      os << SerializeResident(resident) << ',';
    }
    os << ']';
  }
  return os.str();
}

std::vector<std::string>
TakeExecutionAxisPrefix(const std::vector<std::string> &axes, int depth) {
  return std::vector<std::string>(axes.begin(), axes.begin() + depth);
}

Array<PrimExpr> TakeExecutionExtentPrefix(const Array<PrimExpr> &extents,
                                          int depth) {
  Array<PrimExpr> prefix;
  for (int i = 0; i < depth; ++i) {
    prefix.push_back(extents[i]);
  }
  return prefix;
}

bool PathMatchesExecutionPrefix(
    const std::vector<OpenScopeFrame> &open_scopes, int close_to_depth,
    const std::vector<std::string> &region_execution_axes,
    const Array<PrimExpr> &region_execution_extents) {
  if (close_to_depth > static_cast<int>(open_scopes.size()) ||
      close_to_depth > static_cast<int>(region_execution_axes.size()) ||
      close_to_depth > static_cast<int>(region_execution_extents.size())) {
    return false;
  }
  StructuralEqual equal;
  for (int depth = 1; depth <= close_to_depth; ++depth) {
    const OpenScopeFrame &frame = open_scopes[depth - 1];
    if (frame.shell_axes.size() != static_cast<size_t>(depth) ||
        frame.shell_extents.size() != static_cast<size_t>(depth)) {
      return false;
    }
    if (frame.shell_axes[depth - 1] != region_execution_axes[depth - 1]) {
      return false;
    }
    if (!equal(frame.shell_extents[depth - 1],
               region_execution_extents[depth - 1])) {
      return false;
    }
  }
  return true;
}

bool HasAccessibleResident(const std::vector<OpenScopeFrame> &open_scopes,
                           int attach_depth, int buffer_region_id,
                           int required_depth) {
  int visible_depth =
      std::min(attach_depth, static_cast<int>(open_scopes.size()));
  for (int depth = visible_depth; depth >= 1; --depth) {
    const OpenScopeFrame &frame = open_scopes[depth - 1];
    for (const ResidentValueState &resident : frame.residents) {
      if (resident.buffer_region_id == buffer_region_id &&
          resident.home_depth >= required_depth) {
        return true;
      }
    }
  }
  return false;
}

bool HasAccessibleDefinitionResident(
    const std::vector<OpenScopeFrame> &open_scopes, int attach_depth,
    int origin_region_local_index, int buffer_region_id, int required_rho) {
  int visible_depth =
      std::min(attach_depth, static_cast<int>(open_scopes.size()));
  for (int depth = visible_depth; depth >= 1; --depth) {
    const OpenScopeFrame &frame = open_scopes[depth - 1];
    for (const ResidentValueState &resident : frame.residents) {
      if (resident.kind == ResidentValueKind::kDefinition &&
          resident.origin_region_local_index == origin_region_local_index &&
          resident.buffer_region_id == buffer_region_id &&
          resident.home_depth >= required_rho) {
        return true;
      }
    }
  }
  return false;
}

void InstallResidentIfMissing(std::vector<OpenScopeFrame> *open_scopes,
                              const ResidentValueState &resident) {
  if (resident.home_depth <= 0 ||
      resident.home_depth > static_cast<int>(open_scopes->size())) {
    return;
  }
  OpenScopeFrame &frame = (*open_scopes)[resident.home_depth - 1];
  auto it = std::lower_bound(frame.residents.begin(), frame.residents.end(),
                             resident, ResidentValueLess);
  if (it != frame.residents.end() && SameResidentValue(*it, resident)) {
    return;
  }
  frame.residents.insert(it, resident);
}

void KillResidentsForBuffer(std::vector<OpenScopeFrame> *open_scopes,
                            const std::string &buffer_name) {
  for (OpenScopeFrame &frame : *open_scopes) {
    auto &residents = frame.residents;
    residents.erase(std::remove_if(residents.begin(), residents.end(),
                                   [&](const ResidentValueState &resident) {
                                     return resident.buffer_name == buffer_name;
                                   }),
                    residents.end());
  }
}

void PruneDeadResidents(const WindowPlannerInput &input, PlannerState *state) {
  for (OpenScopeFrame &frame : state->open_scopes) {
    auto &residents = frame.residents;
    residents.erase(std::remove_if(residents.begin(), residents.end(),
                                   [&](const ResidentValueState &resident) {
                                     return !HasFuturePotentialConsumers(
                                         input, *state, resident);
                                   }),
                    residents.end());
  }
}

void AccumulatePlannerScoreTerm(int64_t *field, int64_t delta) {
  *field = SaturatingAddSunmmioTileLoopFusionPlannerCost(*field, delta);
}

int64_t ComputeLiveRangeDelta(const PlannerState &state) {
  int64_t live_range_delta = 0;
  for (const OpenScopeFrame &frame : state.open_scopes) {
    for (const ResidentValueState &resident : frame.residents) {
      AccumulatePlannerScoreTerm(
          &live_range_delta,
          SaturatingMulSunmmioTileLoopFusionPlannerCost(
              resident.payload_bytes, resident.instance_count));
    }
  }
  return live_range_delta;
}

TransitionResult ApplyAction(const WindowPlannerInput &input,
                             const PlannerState &state, int region_local_index,
                             int close_to_depth, int open_to_depth) {
  ICHECK_GE(region_local_index, 0);
  ICHECK_LT(region_local_index, static_cast<int>(input.regions.size()));
  ICHECK(input.problem != nullptr);
  ICHECK_GE(close_to_depth, 0);
  ICHECK_LE(close_to_depth, static_cast<int>(state.open_scopes.size()));
  ICHECK_GE(open_to_depth, close_to_depth);

  const WindowPlannerRegionInfo &region_view =
      input.regions[region_local_index];
  const TileScopeRegion &region = input.problem->regions[region_local_index];
  ICHECK_LE(open_to_depth,
            static_cast<int>(region.logical_execution_axis_keys.size()));
  ICHECK_LE(open_to_depth,
            static_cast<int>(region.execution_loop_extents.size()));

  TransitionResult result{state, {0, 0, 0, 0}};
  result.next_state.open_scopes.resize(close_to_depth);
  for (int depth = close_to_depth + 1; depth <= open_to_depth; ++depth) {
    result.next_state.open_scopes.push_back(
        {TakeExecutionAxisPrefix(region.logical_execution_axis_keys, depth),
         TakeExecutionExtentPrefix(region.execution_loop_extents, depth),
         {}});
  }

  std::unordered_set<int> raw_covered_use_indices;
  for (int edge_index : input.incoming_edges_by_dst[region_local_index]) {
    const WindowPlannerEdgeInfo &edge = input.edges[edge_index];
    if (edge.kind != TileScopeDependenceKind::kRAW) {
      continue;
    }
    if (edge.covered_use_index >= 0) {
      raw_covered_use_indices.insert(edge.covered_use_index);
    }
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

  for (size_t use_index = 0; use_index < region_view.use_in.size();
       ++use_index) {
    const PlannerBufferValueInfo &use_info = region_view.use_in[use_index];
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

  for (const PlannerBufferValueInfo &def_info : region_view.def_out) {
    KillResidentsForBuffer(&result.next_state.open_scopes,
                           def_info.buffer_name);
  }
  for (const PlannerBufferValueInfo &def_info : region_view.def_out) {
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

      SunmmioTileLoopFusionPlannerAction action;
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
  ICHECK(input.problem != nullptr);
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

    const WindowPlannerRegionInfo &region_view =
        input.regions[region_local_index];
    const TileScopeRegion &region = input.problem->regions[region_local_index];
    for (int close_to_depth = 0;
         close_to_depth <= static_cast<int>(state.open_scopes.size());
         ++close_to_depth) {
      if (!PathMatchesExecutionPrefix(state.open_scopes, close_to_depth,
                                      region.logical_execution_axis_keys,
                                      region.execution_loop_extents)) {
        continue;
      }
      for (int open_to_depth = close_to_depth;
           open_to_depth <=
           static_cast<int>(region.logical_execution_axis_keys.size());
           ++open_to_depth) {
        TransitionResult transition = ApplyAction(
            input, state, region_local_index, close_to_depth, open_to_depth);
        // Planner score terms are nonnegative, so any suffix can only increase
        // the current transition cost. Once a complete incumbent exists, skip
        // branches whose immediate delta is already no better than that bound.
        if (CompareSunmmioTileLoopFusionPlannerScores(transition.delta,
                                                      best.score) >= 0) {
          continue;
        }
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

        SunmmioTileLoopFusionPlannerAction action;
        action.region_index = region_view.global_region_index;
        action.close_to_depth = close_to_depth;
        action.open_to_depth = open_to_depth;
        for (int depth = close_to_depth + 1; depth <= open_to_depth; ++depth) {
          action.opened_shells.push_back(TakeExecutionAxisPrefix(
              region.logical_execution_axis_keys, depth));
          action.opened_shell_extents.push_back(
              TakeExecutionExtentPrefix(region.execution_loop_extents, depth));
        }

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

std::vector<SunmmioTileLoopFusionPlannerTreeNode>
BuildPlanTree(const std::vector<SunmmioTileLoopFusionPlannerAction> &actions) {
  auto root = std::make_shared<MutablePlannerTreeNode>();
  root->is_scope = true;

  std::vector<std::shared_ptr<MutablePlannerTreeNode>> open_path;
  open_path.push_back(root);
  for (const SunmmioTileLoopFusionPlannerAction &action : actions) {
    while (static_cast<int>(open_path.size()) - 1 > action.close_to_depth) {
      open_path.pop_back();
    }
    for (size_t shell_index = 0; shell_index < action.opened_shells.size();
         ++shell_index) {
      auto scope_node = std::make_shared<MutablePlannerTreeNode>();
      scope_node->is_scope = true;
      scope_node->shell_axes = action.opened_shells[shell_index];
      if (shell_index < action.opened_shell_extents.size()) {
        scope_node->shell_extents = action.opened_shell_extents[shell_index];
      }
      open_path.back()->children.push_back(scope_node);
      open_path.push_back(scope_node);
    }

    auto region_node = std::make_shared<MutablePlannerTreeNode>();
    region_node->is_scope = false;
    region_node->region_index = action.region_index;
    open_path.back()->children.push_back(region_node);
  }

  std::vector<SunmmioTileLoopFusionPlannerTreeNode> tree;
  for (const std::shared_ptr<MutablePlannerTreeNode> &child : root->children) {
    tree.push_back(FreezeTree(child));
  }
  return tree;
}

} // namespace planner_internal
} // namespace tl
} // namespace tvm
