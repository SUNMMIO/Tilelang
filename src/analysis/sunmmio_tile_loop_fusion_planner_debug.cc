#include "sunmmio_tile_loop_fusion_planner_internal.h"

#include <tvm/ffi/reflection/registry.h>

#include <algorithm>
#include <tuple>

namespace tvm {
namespace tl {

namespace {

void SortAndDedupeDebugResidents(planner_internal::OpenScopeFrame *frame) {
  auto &residents = frame->residents;
  std::sort(residents.begin(), residents.end(),
            [](const planner_internal::ResidentValueState &lhs,
               const planner_internal::ResidentValueState &rhs) {
              return std::tie(lhs.kind, lhs.origin_region_local_index,
                              lhs.buffer_region_id, lhs.home_depth,
                              lhs.payload_bytes, lhs.instance_count) <
                     std::tie(rhs.kind, rhs.origin_region_local_index,
                              rhs.buffer_region_id, rhs.home_depth,
                              rhs.payload_bytes, rhs.instance_count);
            });
  residents.erase(std::unique(residents.begin(), residents.end(),
                              planner_internal::SameResidentValue),
                  residents.end());
}

Map<String, ffi::Any> DebugSunmmioTileLoopFusionPhase1Guardrails() {
  auto make_resident = [](int64_t payload_bytes, int64_t instance_count) {
    planner_internal::ResidentValueState resident;
    resident.kind = planner_internal::ResidentValueKind::kDefinition;
    resident.origin_region_local_index = 3;
    resident.buffer_region_id = 7;
    resident.buffer_name = "debug_buffer";
    resident.home_depth = 2;
    resident.payload_bytes = payload_bytes;
    resident.instance_count = instance_count;
    return resident;
  };

  planner_internal::OpenScopeFrame identical_frame;
  identical_frame.residents = {make_resident(64, 8), make_resident(64, 8)};
  SortAndDedupeDebugResidents(&identical_frame);

  planner_internal::OpenScopeFrame payload_distinct_frame;
  payload_distinct_frame.residents = {make_resident(64, 8),
                                      make_resident(128, 8)};
  SortAndDedupeDebugResidents(&payload_distinct_frame);

  planner_internal::OpenScopeFrame instance_distinct_frame;
  instance_distinct_frame.residents = {make_resident(64, 8),
                                       make_resident(64, 16)};
  SortAndDedupeDebugResidents(&instance_distinct_frame);

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
  planner_internal::DynamicBitset bitset(num_bits);
  planner_internal::SetBit(&bitset, index);
  return planner_internal::TestBit(bitset, index);
}

Map<String, ffi::Any> DebugSunmmioTileLoopFusionRawCoverageAccounting() {
  SunmmioTileLoopFusionWindowProblem problem;
  problem.regions.resize(3);

  planner_internal::WindowPlannerInput input;
  input.problem = &problem;
  input.regions.resize(3);
  input.edges.resize(2);
  input.incoming_edges_by_dst.resize(3);
  input.outgoing_edges_by_src.resize(3);
  input.predecessor_masks.assign(3, planner_internal::DynamicBitset(3));
  input.earlier_source_masks.assign(3, planner_internal::DynamicBitset(3));

  for (int i = 0; i < 3; ++i) {
    problem.regions[i].global_region_index = i;
    problem.regions[i].logical_execution_axis_keys = {"i"};
    problem.regions[i].execution_loop_extents =
        Array<PrimExpr>{IntImm(DataType::Int(32), 1)};
    input.regions[i].global_region_index = i;
  }

  input.regions[1].use_in.push_back({10, "debug_buffer", 1, 64, 1});
  input.regions[1].use_in.push_back({11, "debug_buffer", 1, 80, 1});
  input.regions[2].use_in.push_back({10, "debug_buffer", 1, 64, 1});

  planner_internal::WindowPlannerEdgeInfo first_edge;
  first_edge.src_local_index = 0;
  first_edge.dst_local_index = 1;
  first_edge.kind = TileScopeDependenceKind::kRAW;
  first_edge.buffer_region_id = 10;
  first_edge.buffer_name = "debug_buffer";
  first_edge.rho = 1;
  first_edge.weight = 64;
  first_edge.instance_count = 1;
  first_edge.covered_use_index = 0;
  input.edges[0] = first_edge;

  planner_internal::WindowPlannerEdgeInfo second_edge = first_edge;
  second_edge.dst_local_index = 2;
  input.edges[1] = second_edge;

  input.incoming_edges_by_dst[1].push_back(0);
  input.incoming_edges_by_dst[2].push_back(1);
  input.outgoing_edges_by_src[0].push_back(0);
  input.outgoing_edges_by_src[0].push_back(1);
  planner_internal::SetBit(&input.predecessor_masks[1], 0);
  planner_internal::SetBit(&input.predecessor_masks[2], 0);
  planner_internal::SetBit(&input.earlier_source_masks[1], 0);
  planner_internal::SetBit(&input.earlier_source_masks[2], 0);
  planner_internal::SetBit(&input.earlier_source_masks[2], 1);

  planner_internal::DynamicBitset raw_consumers(3);
  planner_internal::SetBit(&raw_consumers, 1);
  planner_internal::SetBit(&raw_consumers, 2);
  input.raw_consumer_masks_by_key.emplace(
      planner_internal::RawConsumerKey{0, 10}, raw_consumers);

  planner_internal::PlannerState state{planner_internal::DynamicBitset(3), {}};
  planner_internal::TransitionResult first =
      planner_internal::ApplyAction(input, state, 1, 0, 1);
  planner_internal::TransitionResult second =
      planner_internal::ApplyAction(input, first.next_state, 2, 1, 1);

  Map<String, ffi::Any> summary;
  summary.Set("first_write_cut_cost", Integer(first.delta.write_cut_cost));
  summary.Set("first_shared_read_cost", Integer(first.delta.shared_read_cost));
  summary.Set("second_write_cut_cost", Integer(second.delta.write_cut_cost));
  summary.Set("second_shared_read_cost",
              Integer(second.delta.shared_read_cost));
  return summary;
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

} // namespace

} // namespace tl
} // namespace tvm
