// Exact planning entrypoints for Sunmmio tile loop fusion.
//
// This file owns the transition from a window-local semantic problem to a
// chosen schedule. The DP/search machinery below is intentionally opaque to
// callers; the public contract is a vector of window problems in, a vector of
// window plans out.

#include "sunmmio_tile_loop_fusion_planner_internal.h"

#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace tl {

using namespace detail;

Map<String, ffi::Any> DebugSunmmioTileLoopFusionPhase1Guardrails() {
  auto make_resident = [](int64_t payload_bytes, int64_t instance_count) {
    ResidentValueState resident;
    resident.kind = ResidentValueKind::kDefinition;
    resident.origin_region_local_index = 3;
    resident.buffer_region_id = 7;
    resident.buffer_name = "debug_buffer";
    resident.home_depth = 2;
    resident.payload_bytes = payload_bytes;
    resident.instance_count = instance_count;
    return resident;
  };

  OpenScopeFrame identical_frame;
  identical_frame.residents = {make_resident(64, 8), make_resident(64, 8)};
  SortAndDedupeResidents(&identical_frame);

  OpenScopeFrame payload_distinct_frame;
  payload_distinct_frame.residents = {make_resident(64, 8),
                                      make_resident(128, 8)};
  SortAndDedupeResidents(&payload_distinct_frame);

  OpenScopeFrame instance_distinct_frame;
  instance_distinct_frame.residents = {make_resident(64, 8),
                                       make_resident(64, 16)};
  SortAndDedupeResidents(&instance_distinct_frame);

  int64_t limit =
      MakeInfiniteSunmmioTileLoopFusionPlannerScore().write_cut_cost;
  Map<String, ffi::Any> summary;
  summary.Set("planner_cost_limit", Integer(limit));
  summary.Set(
      "saturating_add_overflow",
      Integer(SaturatingAddSunmmioTileLoopFusionPlannerCost(limit - 1, 42)));
  summary.Set(
      "saturating_mul_overflow",
      Integer(SaturatingMulSunmmioTileLoopFusionPlannerCost(limit / 2 + 1, 3)));
  summary.Set("resident_dedupe_identical_count",
              Integer(static_cast<int64_t>(identical_frame.residents.size())));
  summary.Set(
      "resident_dedupe_payload_distinct_count",
      Integer(static_cast<int64_t>(payload_distinct_frame.residents.size())));
  summary.Set(
      "resident_dedupe_instance_distinct_count",
      Integer(static_cast<int64_t>(instance_distinct_frame.residents.size())));
  return summary;
}

bool DebugSunmmioTileLoopFusionCheckBitsetBounds(int num_bits, int index) {
  DynamicBitset bitset(num_bits);
  SetBit(&bitset, index);
  return TestBit(bitset, index);
}

Map<String, ffi::Any> DebugSunmmioTileLoopFusionRawCoverageAccounting() {
  WindowPlannerInput input;
  input.region_indices = {0, 1, 2};
  input.regions.resize(3);
  input.edges.resize(2);
  input.incoming_edges_by_dst.resize(3);
  input.outgoing_edges_by_src.resize(3);
  input.predecessor_masks.assign(3, DynamicBitset(3));
  input.earlier_source_masks.assign(3, DynamicBitset(3));

  for (int i = 0; i < 3; ++i) {
    input.regions[i].global_region_index = i;
    input.regions[i].source_order_index = i;
    input.regions[i].logical_execution_axes = {"i"};
    input.regions[i].execution_loop_extents = {IntImm(DataType::Int(32), 1)};
  }

  input.regions[1].use_in.push_back(
      {10, "debug_buffer", tir::BufferRegion(), 1, 64, 1});
  input.regions[1].use_in.push_back(
      {11, "debug_buffer", tir::BufferRegion(), 1, 80, 1});
  input.regions[2].use_in.push_back(
      {10, "debug_buffer", tir::BufferRegion(), 1, 64, 1});

  WindowPlannerEdgeInfo first_edge;
  first_edge.src_local_index = 0;
  first_edge.dst_local_index = 1;
  first_edge.kind = TileScopeDependenceKind::kRAW;
  first_edge.buffer_region_id = 10;
  first_edge.buffer_name = "debug_buffer";
  first_edge.rho = 1;
  first_edge.weight = 64;
  first_edge.instance_count = 1;
  first_edge.covered_use_indices = {0};
  input.edges[0] = first_edge;

  WindowPlannerEdgeInfo second_edge = first_edge;
  second_edge.dst_local_index = 2;
  input.edges[1] = second_edge;

  input.incoming_edges_by_dst[1].push_back(0);
  input.incoming_edges_by_dst[2].push_back(1);
  input.outgoing_edges_by_src[0].push_back(0);
  input.outgoing_edges_by_src[0].push_back(1);
  SetBit(&input.predecessor_masks[1], 0);
  SetBit(&input.predecessor_masks[2], 0);
  SetBit(&input.earlier_source_masks[1], 0);
  SetBit(&input.earlier_source_masks[2], 0);
  SetBit(&input.earlier_source_masks[2], 1);

  DynamicBitset raw_consumers(3);
  SetBit(&raw_consumers, 1);
  SetBit(&raw_consumers, 2);
  input.raw_consumer_masks_by_key.emplace(RawConsumerKey(0, 10), raw_consumers);

  PlannerState state{DynamicBitset(3), {}};
  TransitionResult first = ApplyAction(input, state, 1, 0, 1);
  TransitionResult second = ApplyAction(input, first.next_state, 2, 1, 1);

  Map<String, ffi::Any> summary;
  summary.Set("first_write_cut_cost", Integer(first.delta.write_cut_cost));
  summary.Set("first_shared_read_cost", Integer(first.delta.shared_read_cost));
  summary.Set("second_write_cut_cost", Integer(second.delta.write_cut_cost));
  summary.Set("second_shared_read_cost",
              Integer(second.delta.shared_read_cost));
  return summary;
}

std::vector<SunmmioTileLoopFusionWindowPlan>
PlanSunmmioTileLoopFusionWindowProblems(
    const std::vector<SunmmioTileLoopFusionWindowProblem> &problems) {
  std::vector<SunmmioTileLoopFusionWindowPlan> plans;
  plans.reserve(problems.size());

  for (const SunmmioTileLoopFusionWindowProblem &problem : problems) {
    WindowPlannerInput input = BuildWindowPlannerInput(problem);

    MemoResult best;
    if (static_cast<int>(input.regions.size()) > kMaxExactPlannerRegions) {
      best = BuildSourceOrderFallbackPlan(input);
    } else {
      PlannerState initial_state{
          DynamicBitset(static_cast<int>(input.regions.size())), {}};
      PlannerSearchContext context;
      best = SolveWindowPlan(input, initial_state, &context);
      if (context.exhausted) {
        best = BuildSourceOrderFallbackPlan(input);
      }
    }

    SunmmioTileLoopFusionWindowPlan summary;
    summary.region_indices = input.region_indices;
    summary.score = best.score;
    summary.actions = best.actions;
    for (const SunmmioTileLoopFusionPlannerAction &action : best.actions) {
      summary.order.push_back(action.region_index);
    }
    summary.tree = BuildPlanTree(best.actions);
    plans.push_back(std::move(summary));
  }

  return plans;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def(
      "tl.analysis.DebugSunmmioTileLoopFusionPhase1Guardrails",
      DebugSunmmioTileLoopFusionPhase1Guardrails);
  tvm::ffi::reflection::GlobalDef().def(
      "tl.analysis.DebugSunmmioTileLoopFusionCheckBitsetBounds",
      DebugSunmmioTileLoopFusionCheckBitsetBounds);
  tvm::ffi::reflection::GlobalDef().def(
      "tl.analysis.DebugSunmmioTileLoopFusionRawCoverageAccounting",
      DebugSunmmioTileLoopFusionRawCoverageAccounting);
}

} // namespace tl
} // namespace tvm
