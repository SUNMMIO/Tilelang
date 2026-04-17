#pragma once

#include "sunmmio_tile_loop_fusion_planner.h"
#include "sunmmio_tile_loop_fusion_utils.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {
namespace planner_internal {

inline constexpr int kMaxExactPlannerRegions = 15;
inline constexpr size_t kMaxPlannerMemoEntries = 200000;

struct DynamicBitset {
  int num_bits{0};
  std::vector<uint64_t> words;

  DynamicBitset() = default;
  explicit DynamicBitset(int num_bits)
      : num_bits(num_bits), words((num_bits + 63) / 64, 0) {}
};

void SetBit(DynamicBitset *bitset, int index);
bool TestBit(const DynamicBitset &bitset, int index);
bool ContainsAll(const DynamicBitset &required, const DynamicBitset &present);
bool HasAnyFutureBits(const DynamicBitset &candidate,
                      const DynamicBitset &scheduled);
int CountBits(const DynamicBitset &bitset);
int CountMissingBits(const DynamicBitset &candidate,
                     const DynamicBitset &present);
std::string SerializeDynamicBitset(const DynamicBitset &bitset);

struct PlannerBufferValueInfo {
  int buffer_region_id{-1};
  std::string buffer_name;
  int home_depth{0};
  int64_t payload_bytes{0};
  int64_t instance_count{1};
};

struct WindowPlannerRegionInfo {
  int global_region_index{-1};
  std::vector<PlannerBufferValueInfo> use_in;
  std::vector<PlannerBufferValueInfo> def_out;
};

struct WindowPlannerEdgeInfo {
  int src_local_index{-1};
  int dst_local_index{-1};
  TileScopeDependenceKind kind{TileScopeDependenceKind::kRAW};
  int buffer_region_id{-1};
  std::string buffer_name;
  int rho{0};
  int64_t weight{0};
  int64_t instance_count{1};
  int covered_use_index{-1};
};

struct RawConsumerKey {
  int origin_region_local_index{-1};
  int buffer_region_id{-1};

  bool operator==(const RawConsumerKey &other) const {
    return origin_region_local_index == other.origin_region_local_index &&
           buffer_region_id == other.buffer_region_id;
  }
};

struct RawConsumerKeyHash {
  std::size_t operator()(const RawConsumerKey &key) const {
    std::size_t seed = std::hash<int>{}(key.origin_region_local_index);
    seed ^= std::hash<int>{}(key.buffer_region_id) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
    return seed;
  }
};

struct WindowPlannerInput {
  const SunmmioTileLoopFusionWindowProblem *problem{nullptr};
  std::vector<WindowPlannerRegionInfo> regions;
  std::vector<WindowPlannerEdgeInfo> edges;
  std::vector<std::vector<int>> incoming_edges_by_dst;
  std::vector<std::vector<int>> outgoing_edges_by_src;
  std::vector<DynamicBitset> predecessor_masks;
  std::vector<DynamicBitset> earlier_source_masks;
  std::unordered_map<int, DynamicBitset> read_consumer_masks_by_region_id;
  std::unordered_map<RawConsumerKey, DynamicBitset, RawConsumerKeyHash>
      raw_consumer_masks_by_key;
};

enum class ResidentValueKind : int {
  kDefinition = 0,
  kRead = 1,
};

struct ResidentValueState {
  ResidentValueKind kind{ResidentValueKind::kDefinition};
  int origin_region_local_index{-1};
  int buffer_region_id{-1};
  std::string buffer_name;
  int home_depth{0};
  int64_t payload_bytes{0};
  int64_t instance_count{1};
};

struct OpenScopeFrame {
  std::vector<std::string> shell_axes;
  Array<PrimExpr> shell_extents;
  std::vector<ResidentValueState> residents;
};

struct PlannerState {
  DynamicBitset scheduled_mask;
  std::vector<OpenScopeFrame> open_scopes;
};

bool SameResidentValue(const ResidentValueState &lhs,
                       const ResidentValueState &rhs);
std::string SerializeResident(const ResidentValueState &resident);
std::string SerializePlannerState(const PlannerState &state);
std::vector<std::string>
TakeExecutionAxisPrefix(const std::vector<std::string> &axes, int depth);
Array<PrimExpr> TakeExecutionExtentPrefix(const Array<PrimExpr> &extents,
                                          int depth);
bool PathMatchesExecutionPrefix(
    const std::vector<OpenScopeFrame> &open_scopes, int close_to_depth,
    const std::vector<std::string> &region_execution_axes,
    const Array<PrimExpr> &region_execution_extents);
bool HasAccessibleResident(const std::vector<OpenScopeFrame> &open_scopes,
                           int attach_depth, int buffer_region_id,
                           int required_depth);
bool HasAccessibleDefinitionResident(
    const std::vector<OpenScopeFrame> &open_scopes, int attach_depth,
    int origin_region_local_index, int buffer_region_id, int required_rho);
void InstallResidentIfMissing(std::vector<OpenScopeFrame> *open_scopes,
                              const ResidentValueState &resident);
void KillResidentsForBuffer(std::vector<OpenScopeFrame> *open_scopes,
                            const std::string &buffer_name);
void PruneDeadResidents(const WindowPlannerInput &input, PlannerState *state);
void AccumulatePlannerScoreTerm(int64_t *field, int64_t delta);
int64_t ComputeLiveRangeDelta(const PlannerState &state);

struct TransitionResult {
  PlannerState next_state;
  SunmmioTileLoopFusionPlannerScore delta;
};

TransitionResult ApplyAction(const WindowPlannerInput &input,
                             const PlannerState &state, int region_local_index,
                             int close_to_depth, int open_to_depth);

struct MemoResult {
  SunmmioTileLoopFusionPlannerScore score;
  std::vector<SunmmioTileLoopFusionPlannerAction> actions;
};

struct PlannerSearchContext {
  std::unordered_map<std::string, MemoResult> memo;
  bool exhausted{false};
};

MemoResult BuildSourceOrderFallbackPlan(const WindowPlannerInput &input);
MemoResult SolveWindowPlan(const WindowPlannerInput &input,
                           const PlannerState &state,
                           PlannerSearchContext *context);
std::vector<SunmmioTileLoopFusionPlannerTreeNode>
BuildPlanTree(const std::vector<SunmmioTileLoopFusionPlannerAction> &actions);

} // namespace planner_internal
} // namespace tl
} // namespace tvm
