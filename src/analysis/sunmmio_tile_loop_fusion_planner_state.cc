#include "sunmmio_tile_loop_fusion_planner_internal.h"

#include <tvm/node/structural_equal.h>

#include <algorithm>
#include <sstream>
#include <tuple>

namespace tvm {
namespace tl {
namespace detail {

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
  std::string key = RawConsumerKey(resident.origin_region_local_index,
                                   resident.buffer_region_id);
  auto it = input.raw_consumer_masks_by_key.find(key);
  return it != input.raw_consumer_masks_by_key.end() &&
         HasAnyFutureBits(it->second, state.scheduled_mask);
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
         lhs.buffer_name == rhs.buffer_name &&
         lhs.home_depth == rhs.home_depth &&
         lhs.payload_bytes == rhs.payload_bytes &&
         lhs.instance_count == rhs.instance_count;
}

void SortAndDedupeResidents(OpenScopeFrame *frame) {
  auto &residents = frame->residents;
  std::sort(
      residents.begin(), residents.end(),
      [](const ResidentValueState &lhs, const ResidentValueState &rhs) {
        return std::tie(lhs.kind, lhs.origin_region_local_index,
                        lhs.buffer_region_id, lhs.buffer_name, lhs.home_depth,
                        lhs.payload_bytes, lhs.instance_count) <
               std::tie(rhs.kind, rhs.origin_region_local_index,
                        rhs.buffer_region_id, rhs.buffer_name, rhs.home_depth,
                        rhs.payload_bytes, rhs.instance_count);
      });
  residents.erase(
      std::unique(residents.begin(), residents.end(), SameResidentValue),
      residents.end());
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
    if (open_scopes[depth - 1].shell_axes !=
        TakeExecutionAxisPrefix(region_execution_axes, depth)) {
      return false;
    }
    if (!equal(open_scopes[depth - 1].shell_extents,
               TakeExecutionExtentPrefix(region_execution_extents, depth))) {
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
  for (const ResidentValueState &existing : frame.residents) {
    if (SameResidentValue(existing, resident)) {
      return;
    }
  }
  frame.residents.push_back(resident);
  SortAndDedupeResidents(&frame);
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

} // namespace detail
} // namespace tl
} // namespace tvm
