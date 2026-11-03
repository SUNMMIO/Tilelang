// Builds an ILP model that schedules a SunMMIO pipeline while respecting
// command dependencies, hardware-resource capacities, and SRAM bank conflicts.

#include "../op/builtin.h"
#include "../op/comm.h"
#include "../op/utils.h"
#include "../target/sunmmio/cost_model.h"
#include "../target/sunmmio/hardware_types.h"
#include "../target/sunmmio/sunmmio_async_op_info.h"
#include "common/ast_traverser.h"
#include "common/sunmmio_pipeline_utils.h"
#include "sunmmio_pipeline_planning/pipeline_diagnostic.h"
#include "sunmmio_pipeline_planning/resource_types_for_ilp.h"
#include "tvm/arith/pattern.h"
#include "tvm/ffi/reflection/registry.h"
#include "tvm/runtime/logging.h"
#include "tvm/tir/analysis.h"
#include "tvm/tir/buffer.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/function.h"
#include "tvm/tir/op.h"
#include "tvm/tir/stmt.h"
#include "tvm/tir/stmt_functor.h"
#include "tvm/tir/transform.h"

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <highs/Highs.h>
#include <highs/lp_data/HConst.h>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace bank_ilp_internal {

using namespace tir;

enum class Role : uint8_t { kConsumer, kProducer, kBoth, kUndefined };

struct BankFlipMode {
  bool wsram_flip{true};
  bool asram_flip{true};

  bool FlipForMem(int mem) const {
    ICHECK(mem == 0 || mem == 1);
    return mem == 0 ? wsram_flip : asram_flip;
  }

  int Id() const { return (wsram_flip ? 2 : 0) | (asram_flip ? 1 : 0); }
};

struct AccessInfo {
  BufferRegion region;
  bool is_write{false};
  int iter_offset{0};
  int first_offset{-1};
  int last_offset{-1};

  const Buffer &buffer() const { return region->buffer; }
};

struct CommandSpec {
  int latency{0};
  std::vector<int> resources;
  std::map<int, std::vector<std::pair<int, int>>> resource_windows;
  std::string name;
};

// A FlowSpec is one logical lifetime of data in WSRAM (mem == 0) or ASRAM
// (mem == 1).  It connects the command that produces the data to the command
// that consumes it, possibly in a later logical iteration (delta).  The
// write/read offsets and durations describe the exact bank-occupancy windows
// relative to those command starts.  resident flows enter the loop already
// live and may have a fixed physical bank; non-resident flows are assigned a
// ping/pong phase by the ILP.  precolor records a bank relation known before
// solving, while fp and initial_time carry the phase/lifetime information used
// to compare resident and loop-internal data.
struct FlowSpec {
  bool resident{false};
  int prod{-1};
  int cons{-1};
  int delta{0};
  int mem{0};
  std::string buffer_name;
  int fixed_bank{-1};
  int precolor{-1};
  int fp{1};
  int initial_time{0};
  int w_off{0};
  int w_dur{0};
  int r_off{0};
  int r_dur{0};
  int write_resource{-1};
  int read_resource{-1};
};

struct DependencySpec {
  int src{-1};
  int dst{-1};
  int delta{0};
  int lag{0};
  std::string hazard;
  std::string buffer_name;
};

using SameWriteFlowKey = std::tuple<int, int>;

SameWriteFlowKey MakeSameWriteFlowKey(const FlowSpec &flow) {
  return std::make_tuple(flow.prod, flow.mem);
}

// Complete scheduling problem for one template iteration.  P contains the N
// hardware commands; dep_edges and delta encode producer-to-consumer ordering
// across logical iterations; R/cap describe execution resources and their
// parallel capacities.  flows adds the physical SRAM-bank lifetimes that are
// not expressible as ordinary command dependencies.  The solver folds this
// infinite periodic schedule into an initiation-interval-sized time window.
struct Problem {
  int N{0};
  int Tmax{0};
  std::vector<int> R;
  std::unordered_map<int, int> cap;
  std::vector<CommandSpec> P;
  std::vector<CostEstimate> cost_estimates;
  std::string cost_model_profile{"heuristic"};
  bool preserve_absolute_cycles{false};
  int solver_scale_factor{1};
  bool solver_scaled{false};
  bool latency_gcd_enabled{false};
  int latency_gcd{1};
  bool auto_faster_selected{false};
  bool multidim{false};
  std::vector<int> command_inner_coordinates;
  bool planner_fallback{false};
  bool fallback_used{false};
  std::vector<std::pair<int, int>> dep_edges;
  std::unordered_map<long long, int> delta;
  std::vector<DependencySpec> dependencies;
  std::vector<FlowSpec> flows;
  std::vector<std::string> versioned_buffer_names;
};

struct ModelVars {
  HighsInt col_T{-1};
  std::vector<HighsInt> col_t;
  std::vector<HighsInt> col_y;
  std::vector<HighsInt> col_y_half;
  std::vector<HighsInt> col_start_parity;
  std::vector<HighsInt> col_m;
  std::vector<std::vector<HighsInt>> col_x;
  std::map<std::pair<int, int>, std::vector<HighsInt>> col_resource_a;
  std::vector<int> internal_flow_ids;
  std::vector<HighsInt> col_z;
};

struct SolveResult {
  bool ok{false};
  bool objective_optimized{false};
  bool fast_verified_schedule{false};
  int II{0};
  int makespan{0};
  int bank_slot_period{0};
  std::vector<int> t;
  std::vector<int> m;
  std::vector<int> y;
  std::vector<int> internal_flow_ids;
  std::vector<int> z_bank;
  BankFlipMode bank_flip_mode;
  bool vc_blocking_issue_modeled{true};
  int vc_blocking_issue_constraints{0};
};

struct SolutionVerifyResult {
  bool ok{true};
  bool node_time_ok{true};
  bool dependency_ok{true};
  bool resource_slot_ok{true};
  bool bank_slot_ok{true};
  bool bank_port_ok{true};
  bool multidim_identity_ok{true};
  bool raw_cycle_ok{true};
  int64_t raw_lift_unit{1};
  int64_t raw_ii{0};
  int64_t raw_makespan{0};
  std::vector<std::string> errors;
  std::map<int, std::map<int, int>> resource_slot_load;
  std::array<std::array<std::map<int, int>, 2>, 2> bank_slot_load;
  std::array<std::array<std::map<int, int>, 2>, 2> bank_port_load;
};

struct ExpandedOrderEntry {
  int iter{0};
  int id{-1};
  int absolute_start{0};
};

struct TimeWindowOrderResult {
  std::vector<ExpandedOrderEntry> prologue;
  std::vector<ExpandedOrderEntry> body;
  std::vector<ExpandedOrderEntry> epilogue;
  int steady_state_max_iter_offset{0};
};

int PositiveMod(int value, int mod);

bool CommandUsesResource(const CommandSpec &spec, int resource);

BufferRegion MaterializeBufferRegion(const BufferRegion &region,
                                     const Var &loop_var, int iter);

namespace {

// File-local helper to avoid colliding with the similarly named function in
// sunmmio_pipeline_planning.cc during final shared-library link.
bool PipelineRegionIntersect(const Region &region1, const Region &region2) {
  ICHECK(region1.size() == region2.size());
  for (size_t i = 0; i < region1.size(); ++i) {
    const Range &dim1 = region1[i];
    const Range &dim2 = region2[i];
    auto int_set1 = arith::IntSet::FromRange(dim1);
    auto int_set2 = arith::IntSet::FromRange(dim2);
    if (arith::Intersect({int_set1, int_set2}).IsNothing()) {
      return false;
    }
  }
  return true;
}

const double kInf = kHighsInf;

long long EdgeKey(int i, int j) {
  return (static_cast<long long>(i) << 32) | static_cast<unsigned int>(j);
}

int CeilDiv(int a, int b) {
  ICHECK_GT(b, 0);
  return (a + b - 1) / b;
}

int FloorDiv(int a, int b) {
  ICHECK_GT(b, 0);
  int quotient = a / b;
  int remainder = a % b;
  return remainder < 0 ? quotient - 1 : quotient;
}

int GcdInt(int a, int b) {
  a = std::abs(a);
  b = std::abs(b);
  while (b != 0) {
    int t = a % b;
    a = b;
    b = t;
  }
  return a;
}

std::vector<std::pair<int, int>>
NormalizeResourceWindows(std::vector<std::pair<int, int>> windows) {
  std::sort(windows.begin(), windows.end());
  std::vector<std::pair<int, int>> merged;
  for (const auto &window : windows) {
    ICHECK_GE(window.first, 0);
    ICHECK_GT(window.second, window.first);
    if (merged.empty() || window.first > merged.back().second) {
      merged.push_back(window);
    } else {
      merged.back().second = std::max(merged.back().second, window.second);
    }
  }
  return merged;
}

void ScaleResourceWindows(CommandSpec *spec, int64_t scale) {
  ICHECK_GT(scale, 0);
  if (spec->resource_windows.empty()) {
    return;
  }
  std::map<int, std::vector<std::pair<int, int>>> scaled;
  for (const auto &resource_entry : spec->resource_windows) {
    std::vector<std::pair<int, int>> windows;
    for (const auto &window : resource_entry.second) {
      int64_t scaled_start = static_cast<int64_t>(window.first) / scale;
      int64_t scaled_end =
          (static_cast<int64_t>(window.second) + scale - 1) / scale;
      scaled_end = std::min<int64_t>(scaled_end, spec->latency);
      ICHECK_LT(scaled_start, scaled_end);
      ICHECK_LE(scaled_end, spec->latency);
      windows.push_back(
          {static_cast<int>(scaled_start), static_cast<int>(scaled_end)});
    }
    scaled[resource_entry.first] = NormalizeResourceWindows(std::move(windows));
  }
  spec->resource_windows = std::move(scaled);
}

void ScaleAccessFrontiers(std::vector<AccessInfo> *accesses, int64_t scale,
                          int scaled_latency) {
  ICHECK_GT(scale, 0);
  for (AccessInfo &access : *accesses) {
    ICHECK_GE(access.first_offset, 0);
    ICHECK_GE(access.last_offset, access.first_offset);
    int64_t first = access.first_offset;
    int64_t last = access.last_offset;
    if (!access.is_write) {
      first /= scale;
    } else {
      first = (first + scale - 1) / scale;
    }
    last = (last + scale - 1) / scale;
    first = std::min<int64_t>(first, scaled_latency);
    last = std::min<int64_t>(last, scaled_latency);
    ICHECK_LE(first, last);
    access.first_offset = static_cast<int>(first);
    access.last_offset = static_cast<int>(last);
  }
}

int ResourceUsageDuration(const CommandSpec &spec, int resource) {
  auto it = spec.resource_windows.find(resource);
  if (it == spec.resource_windows.end()) {
    return std::max(1, spec.latency);
  }
  int duration = 0;
  for (const auto &window : it->second) {
    ICHECK_GE(window.first, 0);
    ICHECK_GT(window.second, window.first);
    ICHECK_LE(window.second, spec.latency);
    duration += window.second - window.first;
  }
  return duration;
}

std::map<int, std::vector<std::pair<int, int>>>
BuildResourceUsage(const std::vector<CommandSpec> &specs) {
  std::map<int, std::vector<std::pair<int, int>>> usage;
  for (int idx = 0; idx < static_cast<int>(specs.size()); ++idx) {
    for (int resource : specs[idx].resources) {
      usage[resource].push_back(
          {idx, ResourceUsageDuration(specs[idx], resource)});
    }
  }
  return usage;
}

std::map<int, int> BuildResourceTotals(
    const std::map<int, std::vector<std::pair<int, int>>> &usage) {
  std::map<int, int> totals;
  for (const auto &kv : usage) {
    int total = 0;
    for (const auto &item : kv.second) {
      total += item.second;
    }
    totals[kv.first] = total;
  }
  return totals;
}

std::pair<int, int> FindTargetResource(const std::map<int, int> &totals) {
  int target_resource = -1;
  int target_total = 0;
  for (const auto &kv : totals) {
    if (kv.second > target_total) {
      target_resource = kv.first;
      target_total = kv.second;
    }
  }
  return {target_resource, target_total};
}

std::vector<int> CandidateFasters(int target_total, int target_upper = 69) {
  int lower = std::max(2, CeilDiv(target_total, std::max(1, target_upper)));
  int upper = std::max(lower, CeilDiv(target_total, 10));
  std::vector<int> candidates;
  for (int faster = upper; faster >= lower; --faster) {
    candidates.push_back(faster);
  }
  return candidates;
}

std::vector<int>
TryFactorWithOptionalBumps(const std::vector<std::pair<int, int>> &items,
                           int factor) {
  std::vector<int> bump_indices;
  for (const auto &item : items) {
    int idx = item.first;
    int latency = item.second;
    if (latency % factor == 0) {
      continue;
    }
    if ((latency + 1) % factor == 0) {
      bump_indices.push_back(idx);
    } else {
      return {};
    }
  }
  return bump_indices;
}

int ScaledTotalForResource(const std::vector<CommandSpec> &specs, int resource,
                           int faster,
                           const std::unordered_set<int> &bump_indices) {
  int total = 0;
  for (int idx = 0; idx < static_cast<int>(specs.size()); ++idx) {
    if (std::find(specs[idx].resources.begin(), specs[idx].resources.end(),
                  resource) == specs[idx].resources.end()) {
      continue;
    }
    int duration = ResourceUsageDuration(specs[idx], resource);
    if (specs[idx].resource_windows.empty() && bump_indices.count(idx)) {
      duration += 1;
    }
    total += CeilDiv(std::max(1, duration), faster);
  }
  return total;
}

std::vector<int> CandidateGCDs(const std::vector<int> &latencies) {
  std::set<int, std::greater<int>> gcds;
  for (int value : latencies) {
    if (value <= 0) {
      continue;
    }
    for (int d = 1; d * d <= value; ++d) {
      if (value % d == 0) {
        gcds.insert(d);
        gcds.insert(value / d);
      }
    }
  }
  return std::vector<int>(gcds.begin(), gcds.end());
}

std::pair<int, std::vector<int>>
TryGCDWithOptionalBumps(const std::vector<std::pair<int, int>> &items,
                        int target_total, int target_upper = 69) {
  std::vector<int> latencies;
  latencies.reserve(items.size());
  for (const auto &item : items) {
    latencies.push_back(item.second);
  }
  int target_gcd_floor = std::max(1, CeilDiv(target_total, target_upper));
  int base_g = 0;
  for (int latency : latencies) {
    base_g = base_g == 0 ? std::abs(latency) : GcdInt(base_g, latency);
  }
  if (base_g >= target_gcd_floor) {
    return {base_g, {}};
  }

  for (int g : CandidateGCDs(latencies)) {
    if (g < target_gcd_floor) {
      continue;
    }
    std::vector<int> bump_indices;
    bool ok = true;
    for (const auto &item : items) {
      int idx = item.first;
      int latency = item.second;
      if (latency % g == 0) {
        continue;
      }
      if ((latency + 1) % g == 0) {
        bump_indices.push_back(idx);
      } else {
        ok = false;
        break;
      }
    }
    if (ok) {
      return {g, bump_indices};
    }
  }
  return {std::max(1, base_g), {}};
}

std::pair<int, std::vector<int>>
AutoSelectSunmmioILPFaster(const std::vector<CommandSpec> &specs,
                           int target_upper = 69) {
  // HiGHS works on integral time slots, but raw cost-model latencies can make
  // the modulo model unnecessarily large.  Scale time using the most heavily
  // occupied resource because its total latency gives the dominant lower bound
  // on II.  A latency may be increased by one only when that makes every
  // command on the bottleneck exactly divisible by the scale factor; this is a
  // conservative timing quantization, never an optimistic shortening.
  // Formally, L_r = sum_{i uses r} d_i selects r* = argmax_r L_r.  For a
  // candidate scale f and bump b_i in {0, 1}, the model uses
  // d'_i = ceil((d_i + b_i) / f), with b_i = 1 only if f divides d_i + 1.
  auto usage = BuildResourceUsage(specs);
  auto totals = BuildResourceTotals(usage);
  auto [target_resource, target_total] = FindTargetResource(totals);
  if (target_resource < 0 || target_total <= 0) {
    return {1, {}};
  }

  for (int faster : CandidateFasters(target_total, target_upper)) {
    std::vector<int> bump_vec =
        TryFactorWithOptionalBumps(usage[target_resource], faster);
    if (bump_vec.empty() && !usage[target_resource].empty()) {
      bool all_divisible = true;
      for (const auto &item : usage[target_resource]) {
        if (item.second % faster != 0) {
          all_divisible = false;
          break;
        }
      }
      if (!all_divisible) {
        continue;
      }
    }
    std::unordered_set<int> bump_indices(bump_vec.begin(), bump_vec.end());
    int target_scaled_total =
        ScaledTotalForResource(specs, target_resource, faster, bump_indices);
    // Rounding each latency independently can change which resource is the
    // bottleneck.  Reject such a factor: the search bound and model size must
    // remain anchored to the resource selected from the unscaled problem.
    bool violations = false;
    for (const auto &kv : totals) {
      int scaled_total =
          ScaledTotalForResource(specs, kv.first, faster, bump_indices);
      if (scaled_total > target_scaled_total) {
        violations = true;
        break;
      }
    }
    if (!violations) {
      return {std::max(1, faster), bump_vec};
    }
  }

  // Exact factorization is not always possible.  A common divisor preserves
  // all relative integral durations; optional +1 bumps rescue near-divisible
  // cost estimates while keeping the scaled bottleneck below target_upper.
  // The fallback chooses g >= ceil(L_r* / target_upper) and replaces each
  // latency with (d_i + b_i) / g, so the reduced model remains integral.
  auto [gcd_value, bump_vec] = TryGCDWithOptionalBumps(
      usage[target_resource], target_total, target_upper);
  return {std::max(1, gcd_value), bump_vec};
}

int GetEnvInt(const char *name, int default_value) {
  const char *raw = std::getenv(name);
  if (!raw || !*raw) {
    return default_value;
  }
  return std::atoi(raw);
}

std::string GetEnvString(const char *name) {
  const char *raw = std::getenv(name);
  if (!raw || !*raw) {
    return "";
  }
  return raw;
}

bool GetEnvBool(const char *name, bool default_value = false) {
  const char *raw = std::getenv(name);
  if (!raw || !*raw) {
    return default_value;
  }
  std::string value(raw);
  for (char &c : value) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

std::string JsonEscape(const std::string &value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (char c : value) {
    switch (c) {
    case '\\':
      escaped += "\\\\";
      break;
    case '"':
      escaped += "\\\"";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      escaped.push_back(c);
      break;
    }
  }
  return escaped;
}

void WriteJsonString(std::ostream &os, const std::string &value) {
  os << "\"" << JsonEscape(value) << "\"";
}

void WriteCostEstimateJson(std::ostream &out, const CostEstimate &estimate) {
  out << "{";
  out << "\"profile\": ";
  WriteJsonString(out, std::string(estimate.profile));
  out << ", \"model_version\": ";
  WriteJsonString(out, std::string(estimate.model_version));
  out << ", \"latency_source\": ";
  WriteJsonString(out, std::string(estimate.latency_source));
  out << ", \"fallback_reason\": ";
  WriteJsonString(out, std::string(estimate.fallback_reason));
  out << ", \"op_kind\": ";
  WriteJsonString(out, std::string(estimate.op_kind));
  out << ", \"transfer_path\": ";
  WriteJsonString(out, std::string(estimate.transfer_path));
  out << ", \"src_scope\": ";
  WriteJsonString(out, std::string(estimate.src_scope));
  out << ", \"dst_scope\": ";
  WriteJsonString(out, std::string(estimate.dst_scope));
  out << ", \"input_dtype\": ";
  WriteJsonString(out, std::string(estimate.input_dtype));
  out << ", \"output_dtype\": ";
  WriteJsonString(out, std::string(estimate.output_dtype));
  out << ", \"latency_cycles\": " << estimate.latency_cycles;
  out << ", \"issue_start_cycles\": " << estimate.issue_start_cycles;
  out << ", \"issue_finish_cycles\": " << estimate.issue_finish_cycles;
  out << ", \"bytes\": " << estimate.bytes;
  out << ", \"packet_count\": " << estimate.packet_count;
  out << ", \"path_start_delay\": " << estimate.path_start_delay;
  out << ", \"row_size\": " << estimate.row_size;
  out << ", \"col_size\": " << estimate.col_size;
  out << ", \"reduction_size\": " << estimate.reduction_size;
  out << ", \"vector_chunks\": " << estimate.vector_chunks;
  out << ", \"load_ops\": " << estimate.load_ops;
  out << ", \"store_ops\": " << estimate.store_ops;
  out << ", \"arithmetic_ops\": " << estimate.arithmetic_ops;
  out << ", \"compare_convert_ops\": " << estimate.compare_convert_ops;
  out << ", \"nonlinear_ops\": " << estimate.nonlinear_ops;
  out << ", \"reduce_ops\": " << estimate.reduce_ops;
  out << ", \"galu_ops\": " << estimate.galu_ops;
  out << ", \"exact_static_submodel\": " << estimate.exact_static_submodel;
  out << ", \"used_fallback\": " << estimate.used_fallback;
  out << ", \"solver_scale_factor\": " << estimate.solver_scale_factor;
  out << ", \"solver_latency_cycles\": " << estimate.solver_latency_cycles;
  out << ", \"solver_scaled\": " << estimate.solver_scaled;
  out << ", \"endpoint_timing_modeled\": " << estimate.endpoint_timing_modeled;
  out << ", \"resource_overlap_modeled\": "
      << estimate.resource_overlap_modeled;
  out << "}";
}

void WriteResourceWindowsJson(std::ostream &out, const CommandSpec &spec) {
  out << "{";
  bool first_resource = true;
  for (const auto &resource_entry : spec.resource_windows) {
    if (!first_resource) {
      out << ", ";
    }
    first_resource = false;
    WriteJsonString(out, std::to_string(resource_entry.first));
    out << ": [";
    for (size_t i = 0; i < resource_entry.second.size(); ++i) {
      if (i != 0) {
        out << ", ";
      }
      out << "[" << resource_entry.second[i].first << ", "
          << resource_entry.second[i].second << "]";
    }
    out << "]";
  }
  out << "}";
}

void WriteProblemJson(const Problem &prob, const std::string &path) {
  std::ofstream out(path);
  ICHECK(out.is_open()) << "Failed to open ILP problem json path: " << path;
  out << std::boolalpha;
  out << "{\n";
  out << "  \"N\": " << prob.N << ",\n";
  out << "  \"Tmax\": " << prob.Tmax << ",\n";
  out << "  \"cost_model\": {";
  out << "\"profile\": ";
  WriteJsonString(out, prob.cost_model_profile);
  out << ", \"preserve_absolute_cycles\": " << prob.preserve_absolute_cycles;
  out << ", \"solver_scale_factor\": " << prob.solver_scale_factor;
  out << ", \"solver_scaled\": " << prob.solver_scaled;
  out << ", \"latency_gcd_enabled\": " << prob.latency_gcd_enabled;
  out << ", \"latency_gcd\": " << prob.latency_gcd;
  out << ", \"auto_faster_selected\": " << prob.auto_faster_selected;
  out << ", \"multidim\": " << prob.multidim;
  out << ", \"planner_fallback\": " << prob.planner_fallback;
  out << ", \"fallback_used\": " << prob.fallback_used;
  out << "},\n";

  out << "  \"R\": [";
  for (size_t i = 0; i < prob.R.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << prob.R[i];
  }
  out << "],\n";

  std::map<int, int> sorted_cap(prob.cap.begin(), prob.cap.end());
  out << "  \"cap\": {";
  bool first_cap = true;
  for (const auto &kv : sorted_cap) {
    if (!first_cap) {
      out << ", ";
    }
    first_cap = false;
    WriteJsonString(out, std::to_string(kv.first));
    out << ": " << kv.second;
  }
  out << "},\n";

  out << "  \"commands\": {\n";
  for (int i = 0; i < prob.N; ++i) {
    out << "    ";
    WriteJsonString(out, std::to_string(i));
    out << ": {\"latency\": " << prob.P[i].latency << ", \"resources\": [";
    for (size_t j = 0; j < prob.P[i].resources.size(); ++j) {
      if (j != 0) {
        out << ", ";
      }
      out << prob.P[i].resources[j];
    }
    out << "], \"resource_windows\": ";
    WriteResourceWindowsJson(out, prob.P[i]);
    out << ", \"name\": ";
    WriteJsonString(out, prob.P[i].name);
    out << ", \"coordinate\": [0, ";
    out << (i < static_cast<int>(prob.command_inner_coordinates.size())
                ? prob.command_inner_coordinates[i]
                : -1);
    out << "]";
    out << ", \"cost_estimate\": ";
    if (i < static_cast<int>(prob.cost_estimates.size())) {
      WriteCostEstimateJson(out, prob.cost_estimates[i]);
    } else {
      out << "null";
    }
    out << "}";
    out << (i + 1 == prob.N ? "\n" : ",\n");
  }
  out << "  },\n";

  out << "  \"P\": {\n";
  for (int i = 0; i < prob.N; ++i) {
    out << "    ";
    WriteJsonString(out, std::to_string(i));
    out << ": {\"latency\": " << prob.P[i].latency << ", \"resources\": [";
    for (size_t j = 0; j < prob.P[i].resources.size(); ++j) {
      if (j != 0) {
        out << ", ";
      }
      out << prob.P[i].resources[j];
    }
    out << "], \"resource_windows\": ";
    WriteResourceWindowsJson(out, prob.P[i]);
    out << "}";
    out << (i + 1 == prob.N ? "\n" : ",\n");
  }
  out << "  },\n";

  out << "  \"dep_edges\": [";
  for (size_t i = 0; i < prob.dep_edges.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << "[" << prob.dep_edges[i].first << ", " << prob.dep_edges[i].second
        << "]";
  }
  out << "],\n";

  std::map<std::pair<int, int>, int> sorted_delta;
  for (const auto &edge : prob.dep_edges) {
    auto it = prob.delta.find(EdgeKey(edge.first, edge.second));
    if (it != prob.delta.end()) {
      sorted_delta[edge] = it->second;
    }
  }
  out << "  \"delta\": {";
  bool first_delta = true;
  for (const auto &kv : sorted_delta) {
    if (!first_delta) {
      out << ", ";
    }
    first_delta = false;
    WriteJsonString(out, std::to_string(kv.first.first) + "," +
                             std::to_string(kv.first.second));
    out << ": " << kv.second;
  }
  out << "},\n";
  out << "  \"multidim_delta\": {";
  bool first_multidim_delta = true;
  for (const auto &kv : sorted_delta) {
    if (!first_multidim_delta) {
      out << ", ";
    }
    first_multidim_delta = false;
    WriteJsonString(out, std::to_string(kv.first.first) + "," +
                             std::to_string(kv.first.second));
    int inner_delta = 0;
    if (kv.first.first <
            static_cast<int>(prob.command_inner_coordinates.size()) &&
        kv.first.second <
            static_cast<int>(prob.command_inner_coordinates.size())) {
      inner_delta = prob.command_inner_coordinates[kv.first.second] -
                    prob.command_inner_coordinates[kv.first.first];
    }
    out << ": [" << kv.second << ", " << inner_delta << "]";
  }
  out << "},\n";


  out << "  \"dependency_constraints\": [";
  for (size_t i = 0; i < prob.dependencies.size(); ++i) {
    const DependencySpec &dependency = prob.dependencies[i];
    if (i != 0) {
      out << ", ";
    }
    out << "{";
    out << "\"src\": " << dependency.src;
    out << ", \"dst\": " << dependency.dst;
    out << ", \"delta\": " << dependency.delta;
    out << ", \"lag\": " << dependency.lag;
    out << ", \"hazard\": ";
    WriteJsonString(out, dependency.hazard);
    out << ", \"buffer_name\": ";
    WriteJsonString(out, dependency.buffer_name);
    out << "}";
  }
  out << "],\n";

  std::vector<FlowSpec> sorted_flows = prob.flows;
  std::sort(
      sorted_flows.begin(), sorted_flows.end(),
      [](const FlowSpec &a, const FlowSpec &b) {
        return std::tie(a.resident, a.prod, a.cons, a.mem, a.buffer_name,
                        a.fixed_bank, a.fp, a.initial_time, a.w_off, a.w_dur,
                        a.r_off, a.r_dur, a.write_resource, a.read_resource) <
               std::tie(b.resident, b.prod, b.cons, b.mem, b.buffer_name,
                        b.fixed_bank, b.fp, b.initial_time, b.w_off, b.w_dur,
                        b.r_off, b.r_dur, b.write_resource, b.read_resource);
      });
  out << "  \"flows\": [";
  for (size_t i = 0; i < sorted_flows.size(); ++i) {
    const FlowSpec &flow = sorted_flows[i];
    if (i != 0) {
      out << ", ";
    }
    out << "{";
    out << "\"kind\": ";
    WriteJsonString(out, flow.resident ? "resident" : "internal");
    out << ", \"prod\": " << flow.prod;
    out << ", \"cons\": " << flow.cons;
    out << ", \"delta\": " << flow.delta;
    out << ", \"mem\": " << flow.mem;
    out << ", \"buffer_name\": ";
    WriteJsonString(out, flow.buffer_name);
    out << ", \"fixed_bank\": " << flow.fixed_bank;
    out << ", \"precolor\": " << flow.precolor;
    out << ", \"fp\": " << flow.fp;
    out << ", \"initial_time\": " << flow.initial_time;
    out << ", \"w_off\": " << flow.w_off;
    out << ", \"w_dur\": " << flow.w_dur;
    out << ", \"r_off\": " << flow.r_off;
    out << ", \"r_dur\": " << flow.r_dur;
    out << ", \"write_resource\": " << flow.write_resource;
    out << ", \"read_resource\": " << flow.read_resource;
    out << "}";
  }
  out << "],\n";
  out << "  \"versioned_buffers\": [";
  for (size_t i = 0; i < prob.versioned_buffer_names.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    WriteJsonString(out, prob.versioned_buffer_names[i]);
  }
  out << "]\n";
  out << "}\n";
}

void MaybeExportProblemJson(const Problem &prob, bool debug) {
  std::string export_path = GetEnvString("TL_SUNMMIO_ILP_PROBLEM_JSON");
  if (export_path.empty() && debug) {
    export_path = "body_ilp_problem.json";
  }
  if (!export_path.empty()) {
    WriteProblemJson(prob, export_path);
  }
}

std::string AddStageSuffixToPath(const std::string &path, int stage) {
  if (path.empty()) {
    return path;
  }
  std::string suffix = "_" + std::to_string(stage);
  size_t dot = path.find_last_of('.');
  size_t slash = path.find_last_of("/\\");
  if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
    return path + suffix;
  }
  return path.substr(0, dot) + suffix + path.substr(dot);
}

void MaybeExportProblemJsonForStage(const Problem &prob, bool debug,
                                    int stage) {
  std::string export_path = GetEnvString("TL_SUNMMIO_ILP_PROBLEM_JSON");
  if (export_path.empty() && debug) {
    export_path = "body_ilp_problem.json";
  }
  if (!export_path.empty()) {
    WriteProblemJson(prob, AddStageSuffixToPath(export_path, stage));
  }
}

std::string Name(int b) { return b == 0 ? "ping" : "pong"; }

std::string MemName(int mem) { return mem == 0 ? "wsram" : "asram"; }

std::string BankPhaseName(int phase) {
  return (phase & 1) == 0 ? "ping" : "pong";
}

std::string ResourceName(int r) {
  switch (r) {
  case static_cast<int>(IlpResourceType::kTensorCore):
    return "tensor_core";
  case static_cast<int>(IlpResourceType::kVectorCore):
    return "vector_core";
  case static_cast<int>(IlpResourceType::kODMA0):
    return "odma0";
  case static_cast<int>(IlpResourceType::kODMA1):
    return "odma1";
  case static_cast<int>(IlpResourceType::kWsramIn):
    return "wsram.in";
  case static_cast<int>(IlpResourceType::kWsramOut):
    return "wsram.out";
  case static_cast<int>(IlpResourceType::kAsramIn):
    return "asram.in";
  case static_cast<int>(IlpResourceType::kAsramOut):
    return "asram.out";
  // case static_cast<int>(IlpResourceType::kRsram):
  //   return "rsram";
  default:
    return "resource_" + std::to_string(r);
  }
}

enum class ConflictType : int {
  kNone = 0,
  kNeedDifferent = 1,
  kNeedSame = 2,
  kImpossible = 3,
};

int WrapBit(int start_slot, int slot) { return slot < start_slot ? 0 : 1; }

std::map<int, int> BuildSlotRhoMap(int start_slot, int duration, int ii) {
  std::map<int, int> slot_rho;
  for (int step = 0; step < duration; ++step) {
    int slot = (start_slot + step) % ii;
    slot_rho[slot] = WrapBit(start_slot, slot);
  }
  return slot_rho;
}

ConflictType AnalyzeWriteReadConflict(int write_start_slot, int write_dur,
                                      int write_parity_flip,
                                      int read_start_slot, int read_dur,
                                      int read_parity_flip, int ii, bool flip) {
  std::map<int, int> write_rho;
  for (int step = 0; step < write_dur; ++step) {
    int slot = (write_start_slot + step) % ii;
    write_rho[slot] =
        flip ? (WrapBit(write_start_slot, slot) ^ write_parity_flip) : 0;
  }

  bool saw_same = false;
  bool saw_diff = false;
  for (int step = 0; step < read_dur; ++step) {
    int slot = (read_start_slot + step) % ii;
    auto it = write_rho.find(slot);
    if (it == write_rho.end())
      continue;
    int read_rho =
        flip ? (WrapBit(read_start_slot, slot) ^ read_parity_flip) : 0;
    if (it->second == read_rho) {
      saw_same = true;
    } else {
      saw_diff = true;
    }
    if (saw_same && saw_diff)
      return ConflictType::kImpossible;
  }

  if (!saw_same && !saw_diff)
    return ConflictType::kNone;
  if (saw_same)
    return ConflictType::kNeedDifferent;
  return ConflictType::kNeedSame;
}

ConflictType MergeConflictRequirements(ConflictType lhs, ConflictType rhs) {
  if (lhs == ConflictType::kImpossible || rhs == ConflictType::kImpossible) {
    return ConflictType::kImpossible;
  }
  if (lhs == ConflictType::kNone)
    return rhs;
  if (rhs == ConflictType::kNone)
    return lhs;
  return lhs == rhs ? lhs : ConflictType::kImpossible;
}

ConflictType AnalyzePrecolorConflict(const FlowSpec &lhs, const FlowSpec &rhs) {
  if (lhs.precolor < 0 || rhs.precolor < 0 || lhs.mem != rhs.mem ||
      lhs.buffer_name != rhs.buffer_name) {
    return ConflictType::kNone;
  }
  return lhs.precolor == rhs.precolor ? ConflictType::kNeedSame
                                      : ConflictType::kNeedDifferent;
}

ConflictType AnalyzeFlowConflict(const FlowSpec &write_flow,
                                 int write_start_time,
                                 const FlowSpec &read_flow, int read_start_time,
                                 int ii, const BankFlipMode &mode) {
  ConflictType precolor = AnalyzePrecolorConflict(write_flow, read_flow);
  if (write_flow.write_resource < 0 || read_flow.read_resource < 0 ||
      write_flow.mem != read_flow.mem) {
    return precolor;
  }
  ConflictType port = AnalyzeWriteReadConflict(
      PositiveMod(write_start_time + write_flow.w_off, ii), write_flow.w_dur,
      ((write_start_time + write_flow.w_off) / ii) & 1,
      PositiveMod(read_start_time + read_flow.r_off, ii), read_flow.r_dur,
      ((read_start_time + read_flow.r_off) / ii) & 1, ii,
      mode.FlipForMem(write_flow.mem));
  return MergeConflictRequirements(port, precolor);
}

int ComputeFoldedOccupancy(int start_time, int duration, int II, int slot) {
  if (duration <= 0 || II <= 0) {
    return 0;
  }
  int rel = slot - PositiveMod(start_time, II);
  rel %= II;
  if (rel < 0) {
    rel += II;
  }
  if (rel >= duration) {
    return 0;
  }
  return CeilDiv(duration - rel, II);
}

int ComputeFoldedResourceOccupancy(const CommandSpec &spec, int resource,
                                   int start_time, int II, int slot) {
  auto it = spec.resource_windows.find(resource);
  if (it == spec.resource_windows.end()) {
    return ComputeFoldedOccupancy(start_time, spec.latency, II, slot);
  }
  int occupancy = 0;
  for (const auto &window : it->second) {
    occupancy += ComputeFoldedOccupancy(start_time + window.first,
                                        window.second - window.first, II, slot);
  }
  return occupancy;
}

SolutionVerifyResult VerifySolution(const Problem &prob,
                                    const SolveResult &sol) {
  SolutionVerifyResult result;
  if (!sol.ok) {
    result.ok = false;
    result.errors.push_back("solver returned non-ok solution");
    return result;
  }
  if (sol.II <= 0) {
    result.ok = false;
    result.node_time_ok = false;
    result.errors.push_back("II must be positive");
    return result;
  }
  ICHECK_EQ(static_cast<int>(sol.t.size()), prob.N);
  ICHECK_EQ(static_cast<int>(sol.m.size()), prob.N);
  ICHECK_EQ(static_cast<int>(sol.y.size()), prob.N);

  std::unordered_map<int, int> internal_pos;
  for (int i = 0; i < static_cast<int>(sol.internal_flow_ids.size()); ++i) {
    internal_pos[sol.internal_flow_ids[i]] = i;
  }
  ICHECK_EQ(static_cast<int>(sol.internal_flow_ids.size()),
            static_cast<int>(sol.z_bank.size()));

  auto fail = [&](bool *flag, std::string msg) {
    if (flag != nullptr) {
      *flag = false;
    }
    result.ok = false;
    result.errors.push_back(std::move(msg));
  };

  for (int i = 0; i < prob.N; ++i) {
    if (sol.m[i] < 0 || sol.m[i] >= sol.II) {
      fail(&result.node_time_ok, "node " + std::to_string(i) +
                                     " has invalid slot " +
                                     std::to_string(sol.m[i]));
    }
    if (sol.t[i] < 0) {
      fail(&result.node_time_ok, "node " + std::to_string(i) +
                                     " has negative start " +
                                     std::to_string(sol.t[i]));
    }
    if (sol.y[i] < 0) {
      fail(&result.node_time_ok, "node " + std::to_string(i) +
                                     " has negative iteration " +
                                     std::to_string(sol.y[i]));
    }
    if (sol.t[i] != sol.y[i] * sol.II + sol.m[i]) {
      fail(&result.node_time_ok,
           "node " + std::to_string(i) + " violates t=y*II+m: t=" +
               std::to_string(sol.t[i]) + " y=" + std::to_string(sol.y[i]) +
               " m=" + std::to_string(sol.m[i]));
    }
    if (PositiveMod(sol.t[i], sol.II) != sol.m[i]) {
      fail(&result.node_time_ok,
           "node " + std::to_string(i) + " has inconsistent folded slot");
    }
    if (sol.t[i] + prob.P[i].latency > sol.makespan) {
      fail(&result.node_time_ok,
           "node " + std::to_string(i) + " finishes after makespan");
    }
  }

  for (const DependencySpec &dependency : prob.dependencies) {
    int src = dependency.src;
    int dst = dependency.dst;
    int lhs = sol.t[dst] - sol.t[src];
    int rhs = dependency.lag - dependency.delta * sol.II;
    if (lhs < rhs) {
      fail(&result.dependency_ok,
           "dependency violated " + std::to_string(src) + "->" +
               std::to_string(dst) + ": lhs=" + std::to_string(lhs) +
               " rhs=" + std::to_string(rhs) +
               " delta=" + std::to_string(dependency.delta) +
               " lag=" + std::to_string(dependency.lag));
    }
  }

  for (int i = 0; i < prob.N; ++i) {
    for (int r : prob.P[i].resources) {
      for (int s = 0; s < sol.II; ++s) {
        result.resource_slot_load[r][s] +=
            ComputeFoldedResourceOccupancy(prob.P[i], r, sol.t[i], sol.II, s);
      }
    }
  }

  for (int r : prob.R) {
    int cap = prob.cap.count(r) ? prob.cap.at(r) : 1;
    for (int s = 0; s < sol.II; ++s) {
      int use = result.resource_slot_load[r].count(s)
                    ? result.resource_slot_load[r][s]
                    : 0;
      if (use > cap) {
        fail(&result.resource_slot_ok,
             "resource slot overflow " + ResourceName(r) +
                 " slot=" + std::to_string(s) + " use=" + std::to_string(use) +
                 " cap=" + std::to_string(cap));
      }
    }
  }

  for (int a = 0; a < static_cast<int>(prob.flows.size()); ++a) {
    const FlowSpec &write_flow = prob.flows[a];
    if (write_flow.write_resource < 0 || write_flow.prod < 0)
      continue;
    auto it_a = internal_pos.find(a);
    if (it_a == internal_pos.end())
      continue;
    int z_write = sol.z_bank[it_a->second];
    int write_start_time = sol.t[write_flow.prod];

    for (int b = 0; b < static_cast<int>(prob.flows.size()); ++b) {
      if (a == b)
        continue;
      const FlowSpec &read_flow = prob.flows[b];
      auto it_b = internal_pos.find(b);
      if (it_b == internal_pos.end() || read_flow.cons < 0)
        continue;
      int z_read = sol.z_bank[it_b->second];
      int read_start_time = sol.t[read_flow.cons] + read_flow.delta * sol.II;
      ConflictType conflict =
          AnalyzeFlowConflict(write_flow, write_start_time, read_flow,
                              read_start_time, sol.II, sol.bank_flip_mode);
      if (conflict == ConflictType::kNeedDifferent && z_write == z_read) {
        fail(&result.bank_port_ok,
             "bank port conflict requires different banks between flow " +
                 std::to_string(a) + " and flow " + std::to_string(b));
      } else if (conflict == ConflictType::kNeedSame && z_write != z_read) {
        fail(&result.bank_port_ok,
             "bank port conflict requires same banks between flow " +
                 std::to_string(a) + " and flow " + std::to_string(b));
      } else if (conflict == ConflictType::kImpossible) {
        fail(&result.bank_port_ok,
             "bank port conflict impossible between flow " + std::to_string(a) +
                 " and flow " + std::to_string(b));
      }
    }
  }

  if (prob.multidim) {
    if (static_cast<int>(prob.command_inner_coordinates.size()) != prob.N) {
      fail(&result.multidim_identity_ok,
           "multidim command identity count does not match N");
    } else {
      for (int i = 0; i < prob.N; ++i) {
        if (prob.command_inner_coordinates[i] < 0) {
          fail(&result.multidim_identity_ok,
               "command " + std::to_string(i) +
                   " has no inner-coordinate identity");
        }
      }
    }

    // Lift the scaled solution back to raw cost-model cycles and re-check the
    // original command latencies and dependencies.
    int64_t gcd = std::max<int64_t>(1, prob.latency_gcd);
    int64_t faster = std::max<int64_t>(1, prob.solver_scale_factor);
    if (gcd > std::numeric_limits<int64_t>::max() / faster) {
      fail(&result.raw_cycle_ok, "raw-cycle lift unit overflow");
    } else {
      result.raw_lift_unit = gcd * faster;
      if (sol.II > std::numeric_limits<int64_t>::max() /
                       result.raw_lift_unit) {
        fail(&result.raw_cycle_ok, "raw-cycle II overflow");
      } else {
        result.raw_ii =
            static_cast<int64_t>(sol.II) * result.raw_lift_unit;
      }
      std::vector<int64_t> raw_start(prob.N, 0);
      for (int i = 0; i < prob.N; ++i) {
        if (sol.t[i] > std::numeric_limits<int64_t>::max() /
                           result.raw_lift_unit) {
          fail(&result.raw_cycle_ok,
               "raw-cycle start overflow for command " + std::to_string(i));
          continue;
        }
        raw_start[i] =
            static_cast<int64_t>(sol.t[i]) * result.raw_lift_unit;
        int64_t raw_latency =
            i < static_cast<int>(prob.cost_estimates.size())
                ? std::max<int64_t>(1, prob.cost_estimates[i].latency_cycles)
                : static_cast<int64_t>(prob.P[i].latency) *
                      result.raw_lift_unit;
        int64_t covered_latency =
            static_cast<int64_t>(prob.P[i].latency) * result.raw_lift_unit;
        if (raw_latency > covered_latency) {
          fail(&result.raw_cycle_ok,
               "scaled duration does not cover raw latency for command " +
                   std::to_string(i));
        }
        result.raw_makespan =
            std::max(result.raw_makespan, raw_start[i] + raw_latency);
        if (raw_start[i] !=
            static_cast<int64_t>(sol.y[i]) * result.raw_ii +
                static_cast<int64_t>(sol.m[i]) * result.raw_lift_unit) {
          fail(&result.raw_cycle_ok,
               "raw-cycle t=y*II+m lift failed for command " +
                   std::to_string(i));
        }
      }
      for (const auto &edge : prob.dep_edges) {
        int src = edge.first;
        int dst = edge.second;
        int delta = prob.delta.at(EdgeKey(src, dst));
        int64_t raw_latency =
            src < static_cast<int>(prob.cost_estimates.size())
                ? std::max<int64_t>(
                      1, prob.cost_estimates[src].latency_cycles)
                : static_cast<int64_t>(prob.P[src].latency) *
                      result.raw_lift_unit;
        if (raw_start[dst] - raw_start[src] <
            raw_latency - static_cast<int64_t>(delta) * result.raw_ii) {
          fail(&result.raw_cycle_ok,
               "raw-cycle dependency violated " + std::to_string(src) +
                   "->" + std::to_string(dst));
        }
      }
    }
  }

  return result;
}

void WriteSolutionJson(
    const std::string &path, const Problem &prob, const SolveResult &sol,
    const SolutionVerifyResult &verify,
    const std::map<std::string, int> &runtime_bank_start_phases,
    const std::map<std::string, int> &runtime_bank_read_delta_parities,
    const std::map<std::string, std::map<int, int>>
        &runtime_bank_reader_phases) {
  std::ofstream out(path);
  ICHECK(out.is_open()) << "Failed to open ILP solution json path: " << path;
  out << std::boolalpha;

  std::unordered_map<int, int> internal_pos;
  for (int i = 0; i < static_cast<int>(sol.internal_flow_ids.size()); ++i) {
    internal_pos[sol.internal_flow_ids[i]] = i;
  }

  out << "{\n";
  out << "  \"ii\": " << sol.II << ",\n";
  out << "  \"makespan\": " << sol.makespan << ",\n";
  out << "  \"fast_verified_schedule\": " << sol.fast_verified_schedule
      << ",\n";
  out << "  \"wsram_flip\": " << sol.bank_flip_mode.wsram_flip << ",\n";
  out << "  \"asram_flip\": " << sol.bank_flip_mode.asram_flip << ",\n";
  out << "  \"vc_blocking_issue_modeled\": " << sol.vc_blocking_issue_modeled
      << ",\n";
  out << "  \"vc_blocking_issue_constraints\": "
      << sol.vc_blocking_issue_constraints << ",\n";
  out << "  \"bank_slot_period\": "
      << (sol.bank_slot_period > 0 ? sol.bank_slot_period : (2 * sol.II))
      << ",\n";
  out << "  \"nodes\": {\n";
  for (int i = 0; i < prob.N; ++i) {
    out << "    ";
    WriteJsonString(out, std::to_string(i));
    out << ": {\"start\": " << sol.t[i] << ", \"slot\": " << sol.m[i]
        << ", \"iteration\": " << sol.y[i];
    out << ", \"coordinate\": [" << sol.y[i] << ", ";
    out << (i < static_cast<int>(prob.command_inner_coordinates.size())
                ? prob.command_inner_coordinates[i]
                : -1);
    out << "], \"phases\": [";
    int phase_id = 0;
    bool first_phase = true;
    for (int resource : prob.P[i].resources) {
      std::vector<std::pair<int, int>> windows{{0, prob.P[i].latency}};
      auto windows_it = prob.P[i].resource_windows.find(resource);
      if (windows_it != prob.P[i].resource_windows.end()) {
        windows = windows_it->second;
      }
      for (const auto &window : windows) {
        if (!first_phase) {
          out << ", ";
        }
        first_phase = false;
        out << "{";
        out << "\"phase_id\": " << phase_id++;
        out << ", \"resource_name\": ";
        WriteJsonString(out, ResourceName(resource));
        out << ", \"start\": " << (sol.t[i] + window.first);
        out << ", \"end\": " << (sol.t[i] + window.second);
        out << ", \"duration\": " << (window.second - window.first);
        out << "}";
      }
    }
    out << "]}";
    out << (i + 1 == prob.N ? "\n" : ",\n");
  }
  out << "  },\n";

  out << "  \"dependencies\": [";
  for (size_t i = 0; i < prob.dependencies.size(); ++i) {
    const DependencySpec &dependency = prob.dependencies[i];
    if (i != 0) {
      out << ", ";
    }
    int lhs = sol.t[dependency.dst] - sol.t[dependency.src];
    int rhs = dependency.lag - dependency.delta * sol.II;
    out << "{";
    out << "\"src\": " << dependency.src;
    out << ", \"dst\": " << dependency.dst;
    out << ", \"delta\": " << dependency.delta;
    out << ", \"lag\": " << dependency.lag;
    out << ", \"lhs\": " << lhs;
    out << ", \"rhs\": " << rhs;
    out << ", \"slack\": " << (lhs - rhs);
    out << ", \"hazard\": ";
    WriteJsonString(out, dependency.hazard);
    out << ", \"buffer_name\": ";
    WriteJsonString(out, dependency.buffer_name);
    out << "}";
  }
  out << "],\n";

  out << "  \"flows\": [";
  bool first_flow = true;
  std::unordered_set<std::string> emitted_resident_buffers;
  for (int v = 0; v < static_cast<int>(prob.flows.size()); ++v) {
    const auto &flow = prob.flows[v];
    if (flow.resident) {
      if (!emitted_resident_buffers.insert(flow.buffer_name).second) {
        continue;
      }
      auto it_start_bank = runtime_bank_start_phases.find(flow.buffer_name);
      if (it_start_bank == runtime_bank_start_phases.end()) {
        continue;
      }
      int bank = it_start_bank->second;
      int cons_start = sol.t[flow.cons] + flow.delta * sol.II + flow.r_off;
      int cons_end = cons_start + flow.r_dur;
      int release_time = cons_end;
      if (!first_flow) {
        out << ", ";
      }
      first_flow = false;
      out << "{";
      out << "\"idx\": " << v;
      out << ", \"kind\": ";
      WriteJsonString(out, "resident");
      out << ", \"prod\": -1";
      out << ", \"prod_label\": ";
      WriteJsonString(out, "resident");
      out << ", \"cons\": " << flow.cons;
      out << ", \"delta\": " << flow.delta;
      out << ", \"buffer_name\": ";
      WriteJsonString(out, flow.buffer_name);
      out << ", \"write_resource\": -1";
      out << ", \"read_resource\": " << flow.read_resource;
      out << ", \"write_resource_name\": ";
      WriteJsonString(out, "");
      out << ", \"read_resource_name\": ";
      WriteJsonString(
          out, flow.read_resource < 0 ? "" : ResourceName(flow.read_resource));
      out << ", \"memory\": " << flow.mem;
      out << ", \"memory_name\": ";
      WriteJsonString(out, MemName(flow.mem));
      out << ", \"bank\": ";
      WriteJsonString(out, Name(bank));
      out << ", \"start_bank\": ";
      WriteJsonString(out, Name(bank));
      out << ", \"write_time\": 0";
      out << ", \"write_end\": 0";
      out << ", \"read_time\": " << cons_start;
      out << ", \"read_end\": " << cons_end;
      out << ", \"release_time\": " << release_time;
      out << ", \"write_resource\": -1";
      out << ", \"read_resource\": " << flow.read_resource;
      out << "}";
      continue;
    }

    int bank = sol.z_bank[internal_pos.at(v)];
    int cons_start = sol.t[flow.cons] + flow.delta * sol.II + flow.r_off;
    int prod_start = sol.t[flow.prod] + flow.w_off;
    int prod_end = prod_start + flow.w_dur;
    int cons_end = cons_start + flow.r_dur;
    int release_time = std::max(prod_end, cons_end);
    if (!first_flow) {
      out << ", ";
    }
    first_flow = false;
    out << "{";
    out << "\"idx\": " << v;
    out << ", \"kind\": ";
    WriteJsonString(out, flow.resident ? "resident" : "internal");
    out << ", \"prod\": " << flow.prod;
    out << ", \"prod_label\": ";
    WriteJsonString(out,
                    flow.prod >= 0 ? std::to_string(flow.prod) : "resident");
    out << ", \"cons\": " << flow.cons;
    out << ", \"delta\": " << flow.delta;
    out << ", \"buffer_name\": ";
    WriteJsonString(out, flow.buffer_name);
    out << ", \"write_resource\": " << flow.write_resource;
    out << ", \"read_resource\": " << flow.read_resource;
    out << ", \"write_resource_name\": ";
    WriteJsonString(
        out, flow.write_resource < 0 ? "" : ResourceName(flow.write_resource));
    out << ", \"read_resource_name\": ";
    WriteJsonString(
        out, flow.read_resource < 0 ? "" : ResourceName(flow.read_resource));
    out << ", \"memory\": " << flow.mem;
    out << ", \"memory_name\": ";
    WriteJsonString(out, MemName(flow.mem));
    out << ", \"bank\": ";
    WriteJsonString(out, Name(bank));
    out << ", \"start_bank\": ";
    WriteJsonString(out, Name(bank));
    out << ", \"write_time\": " << prod_start;
    out << ", \"write_end\": " << prod_end;
    out << ", \"read_time\": " << cons_start;
    out << ", \"read_end\": " << cons_end;
    out << ", \"release_time\": " << release_time;
    out << ", \"write_resource\": " << flow.write_resource;
    out << ", \"read_resource\": " << flow.read_resource;
    out << "}";
  }

  out << "],\n";

  out << "  \"verify\": {";
  out << "\"ok\": " << verify.ok;
  out << ", \"node_time_ok\": " << verify.node_time_ok;
  out << ", \"dependency_ok\": " << verify.dependency_ok;
  out << ", \"resource_slot_ok\": " << verify.resource_slot_ok;
  out << ", \"bank_slot_ok\": " << verify.bank_slot_ok;
  out << ", \"multidim_identity_ok\": " << verify.multidim_identity_ok;
  out << ", \"raw_cycle_ok\": " << verify.raw_cycle_ok;
  out << ", \"raw_lift_unit\": " << verify.raw_lift_unit;
  out << ", \"raw_ii\": " << verify.raw_ii;
  out << ", \"raw_makespan\": " << verify.raw_makespan;
  out << ", \"latency_gcd_enabled\": " << prob.latency_gcd_enabled;
  out << ", \"latency_gcd\": " << prob.latency_gcd;
  out << ", \"solver_scale_factor\": " << prob.solver_scale_factor;
  out << ", \"bank_port_ok\": " << verify.bank_port_ok;
  out << ", \"errors\": [";
  for (size_t i = 0; i < verify.errors.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    WriteJsonString(out, verify.errors[i]);
  }
  out << "]},\n";
  out << "  \"bank_load\": {}\n";
  out << "}\n";
}

void DedupAccesses(std::vector<AccessInfo> *accesses) {
  std::vector<AccessInfo> deduped;
  deduped.reserve(accesses->size());
  for (const AccessInfo &access : *accesses) {
    bool exists = false;
    for (const AccessInfo &old : deduped) {
      if (access.is_write != old.is_write ||
          access.iter_offset != old.iter_offset ||
          !access.region->buffer.same_as(old.region->buffer)) {
        continue;
      }
      if (StructuralEqual()(access.region, old.region)) {
        exists = true;
        break;
      }
    }
    if (!exists) {
      deduped.push_back(access);
    }
  }
  *accesses = std::move(deduped);
}

} // namespace

class SunmmioStmtAccessAnalyzer : public StmtExprVisitor {
public:
  explicit SunmmioStmtAccessAnalyzer(const PrimFunc &f) {
    for (const auto &kv : f->buffer_map) {
      buffer_data_to_buffer_.Set(kv.second->data, kv.second);
    }
  }

  std::vector<AccessInfo> Collect(const Stmt &stmt,
                                  const Var &pipeline_loop_var = Var()) {
    accesses_.clear();
    pipeline_loop_var_ = pipeline_loop_var;
    VisitStmt(stmt);
    DedupAccesses(&accesses_);
    return accesses_;
  }

private:
  void AddAccess(const BufferRegion &region, bool is_write) {
    accesses_.push_back(
        AccessInfo{region, is_write,
                   DetectPipelineIterOffsetFromRegion(
                       region, pipeline_loop_var_, &analyzer_)});
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    Array<Range> region;
    for (const PrimExpr &index : op->indices) {
      region.push_back(Range::FromMinExtent(index, 1));
    }
    AddAccess(BufferRegion(op->buffer, region), true);
    VisitExpr(op->value);
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      SunmmioAsyncOpInfo async_info;
      if (TryParseSunmmioAsyncODMAOp(call, &async_info)) {
        AddAccess(async_info.src, false);
        AddAccess(async_info.dst, true);
        return;
      }
      if (call->op.same_as(mma_sunmmio())) {
        AddAccess(NormalizeToBufferRegion(call->args[0]), false);
        AddAccess(NormalizeToBufferRegion(call->args[1]), false);
        BufferRegion c_region = NormalizeToBufferRegion(call->args[2]);
        AddAccess(c_region, false);
        AddAccess(c_region, true);
        if (call->args.size() == 9U) {
          AddAccess(NormalizeToBufferRegion(call->args[7]), true);
        }
        return;
      }
      if (call->op.same_as(Op::Get("tl.vector_core_in_tile_reduce"))) {
        ICHECK_GE(call->args.size(), 3U);
        AddAccess(NormalizeToBufferRegion(call->args[1]), true);
        AddAccess(NormalizeToBufferRegion(call->args[2]), false);
        return;
      }
    }
    VisitExpr(op->value);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    Array<Range> region;
    for (const PrimExpr &index : op->indices) {
      if (const auto *ramp = index.as<RampNode>()) {
        region.push_back(Range::FromMinExtent(ramp->base, ramp->lanes));
      } else {
        region.push_back(Range::FromMinExtent(index, 1));
      }
    }
    AddAccess(BufferRegion(op->buffer, region), false);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(RegionOp::Get())) {
      AddAccess(NormalizeToBufferRegion(ffi::GetRef<PrimExpr>(op)), false);
      return;
    }

    if (op->op.same_as(builtin::address_of())) {
      if (const auto *load = op->args[0].as<BufferLoadNode>()) {
        AddAccess(BufferRegion::FullRegion(load->buffer), false);
        return;
      }
      if (const auto *var_node = op->args[0].as<VarNode>()) {
        Var data_var = ffi::GetRef<Var>(var_node);
        auto it = buffer_data_to_buffer_.find(data_var);
        if (it != buffer_data_to_buffer_.end()) {
          AddAccess(BufferRegion::FullRegion((*it).second), false);
          return;
        }
      }
    }

    if (op->op.same_as(builtin::tvm_access_ptr())) {
      if (const auto *buffer_var = op->args[1].as<VarNode>()) {
        auto it = buffer_data_to_buffer_.find(ffi::GetRef<Var>(buffer_var));
        if (it != buffer_data_to_buffer_.end()) {
          AddAccess(BufferRegion::FullRegion((*it).second), false);
          return;
        }
      }
    }

    StmtExprVisitor::VisitExpr_(op);
  }

  arith::Analyzer analyzer_;
  ffi::Map<Var, Buffer> buffer_data_to_buffer_;
  Var pipeline_loop_var_;
  std::vector<AccessInfo> accesses_;
};

class SunmmioRoleMarker : public StmtVisitor {
public:
  SunmmioRoleMarker(ASTTraverser &traverser, const PrimFunc &func)
      : traverser_(traverser), access_analyzer_(func) {
    traverser_.clear();
  }

  Role GetRole(const StmtNode *stmt) const {
    auto it = map_.find(stmt);
    ICHECK(it != map_.end())
        << "Cannot find role for stmt: " << stmt->GetTypeKey();
    return it->second;
  }

  Role GetRole(const Stmt &stmt) const { return GetRole(stmt.get()); }

  std::vector<AccessInfo> GetAccesses(const Stmt &stmt,
                                      const Var &pipeline_loop_var = Var()) {
    return access_analyzer_.Collect(stmt, pipeline_loop_var);
  }

  void VisitStmt_(const EvaluateNode *op) final {
    Role role = Role::kConsumer;
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.same_as(Op::Get("tl.dma_copy"))) {
        BufferRegion src_region = NormalizeToBufferRegion(call->args[0]);
        if (IsGlobalBuffer(src_region->buffer)) {
          role = Role::kProducer;
        }
      }
    }
    SetRole(op, role);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    Role role = Role::kProducer;
    // Reuse the legacy traverser path for role classification. It is less
    // detailed than the ILP access collector but has proven stable on large
    // kernels such as flash-attention.
    traverser_.traverse_stmt(ffi::GetRef<Stmt>(op));
    auto reads = traverser_.read_buffer_regions_;
    for (const BufferRegion &read : reads) {
      if (!IsGlobalBuffer(read->buffer)) {
        role = Role::kConsumer;
        break;
      }
    }
    SetRole(op, role);
  }

  void VisitStmt_(const SeqStmtNode *op) final {
    StmtVisitor::VisitStmt_(op);
    auto role = GetRole(op->seq[0]);
    for (const Stmt &stmt : op->seq) {
      if (role != GetRole(stmt)) {
        role = Role::kBoth;
        break;
      }
    }
    SetRole(op, role);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    StmtVisitor::VisitStmt_(op);
    auto role = GetRole(op->then_case);
    if (op->else_case.defined() && role != GetRole(op->else_case.value())) {
      role = Role::kBoth;
    }
    SetRole(op, role);
  }

  void VisitStmt_(const BlockRealizeNode *op) final {
    StmtVisitor::VisitStmt_(op);
    SetRole(op, GetRole(op->block));
  }

  template <class NodeType> void HandleBodyStmt(const NodeType *op) {
    StmtVisitor::VisitStmt_(op);
    SetRole(op, GetRole(op->body));
  }

  void VisitStmt_(const ForNode *op) final { HandleBodyStmt(op); }
  void VisitStmt_(const LetStmtNode *op) final { HandleBodyStmt(op); }
  void VisitStmt_(const AttrStmtNode *op) final { HandleBodyStmt(op); }
  void VisitStmt_(const AssertStmtNode *op) final { HandleBodyStmt(op); }
  void VisitStmt_(const BlockNode *op) final { HandleBodyStmt(op); }
  void VisitStmt_(const AllocateNode *op) final { HandleBodyStmt(op); }
  void VisitStmt_(const DeclBufferNode *op) final { HandleBodyStmt(op); }

private:
  void SetRole(const StmtNode *stmt, Role role) { map_[stmt] = role; }

  std::unordered_map<const StmtNode *, Role> map_;
  ASTTraverser traverser_;
  SunmmioStmtAccessAnalyzer access_analyzer_;
};

class SunmmioExprAnalyzer : public StmtExprVisitor {
public:
  SunmmioExprAnalyzer() {}

  void Analyze(const PrimExpr &expr) {
    loop_cost_ = 0;
    load_times = 0;
    flops_ = 0;
    args_.clear();
    constants_.clear();
    vars_.clear();
    StmtExprVisitor::VisitExpr(expr);
  }

private:
  void VisitExpr_(const MulNode *op) final {
    auto a = op->a;
    auto b = op->b;
    flops_ += 1;
    if (const auto *a_int = a.as<IntImmNode>()) {
      if (const auto *b_int = b.as<IntImmNode>()) {
        return;
      }
      if (a_int->value <= 32) {
        loop_cost_ += 2;
        StmtExprVisitor::VisitExpr(op->b);
        return;
      }
    }
    if (const auto *b_int = b.as<IntImmNode>()) {
      if (b_int->value <= 32) {
        loop_cost_ += 2;
        StmtExprVisitor::VisitExpr(op->a);
        return;
      }
    }
    loop_cost_ += 4;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const SubNode *op) final {
    loop_cost_ += 4;
    flops_ += 1;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const AddNode *op) final {
    loop_cost_ += 4;
    flops_ += 1;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const MaxNode *op) final {
    loop_cost_ += 3;
    flops_ += 1;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const MinNode *op) final {
    loop_cost_ += 3;
    flops_ += 1;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const CastNode *op) final {
    loop_cost_ += 3;
    StmtExprVisitor::VisitExpr(op->value);
  }

  void VisitExpr_(const IntImmNode *op) final {
    bool insert = true;
    for (auto it : constants_) {
      if (ExprDeepEqual()(it, tvm::ffi::GetRef<PrimExpr>(op))) {
        insert = false;
        break;
      }
    }
    if (insert) {
      constants_.push_back(tvm::ffi::GetRef<PrimExpr>(op));
    }
  }

  void VisitExpr_(const FloatImmNode *op) final {
    bool insert = true;
    for (auto it : constants_) {
      if (ExprDeepEqual()(it, tvm::ffi::GetRef<PrimExpr>(op))) {
        insert = false;
        break;
      }
    }
    if (insert) {
      constants_.push_back(tvm::ffi::GetRef<PrimExpr>(op));
    }
  }

  void VisitExpr_(const VarNode *op) final {
    bool insert = true;
    for (auto it : vars_) {
      if (ExprDeepEqual()(it, tvm::ffi::GetRef<PrimExpr>(op))) {
        insert = false;
        break;
      }
    }
    if (insert) {
      vars_.push_back(tvm::ffi::GetRef<PrimExpr>(op));
    }
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(Op::Get("tir.exp2"))) {
      loop_cost_ += 10;
      flops_ += 3;
      StmtExprVisitor::VisitExpr(op->args[0]);
    } else if (op->op.same_as(Op::Get("tl.infinity"))) {
      bool insert = true;
      for (auto it : constants_) {
        if (ExprDeepEqual()(it,
                            FloatImm(DataType::Float(16),
                                     std::numeric_limits<float>::infinity()))) {
          insert = false;
          break;
        }
      }
      if (insert) {
        constants_.push_back(FloatImm(DataType::Float(16),
                                      std::numeric_limits<float>::infinity()));
      }
    } else if (op->op.same_as(Op::Get("tir.if_then_else"))) {
      bool insert = true;
      for (auto it : args_) {
        if (ExprDeepEqual()(it, op->args[0])) {
          insert = false;
          break;
        }
      }
      if (insert) {
        args_.push_back(op->args[0]);
      }
      StmtExprVisitor::VisitExpr(op->args[0]);
      StmtExprVisitor::VisitExpr(op->args[1]);
      StmtExprVisitor::VisitExpr(op->args[2]);
    } else if (op->op.same_as(Op::Get("tir.bitwise_and"))) {
      bool insert = true;
      for (auto it : args_) {
        if (ExprDeepEqual()(it, op->args[0])) {
          insert = false;
          break;
        }
      }
      if (insert) {
        args_.push_back(op->args[0]);
      }
      flops_ += 1;
      StmtExprVisitor::VisitExpr(op->args[0]);
      StmtExprVisitor::VisitExpr(op->args[1]);
    } else {
      ICHECK(0) << "Op " << op->op << " not supported now.";
    }
  }

  void VisitExpr_(const LENode *op) final {
    loop_cost_ += 3;
    flops_ += 1;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (load_times == 0) {
      load_times++;
      loop_cost_ += 14;
      flops_ += 5;
    } else {
      load_times++;
      loop_cost_ += 1;
      flops_ += 1;
    }
    for (auto arg : op->indices) {
      bool insert = true;
      for (auto it : args_) {
        if (ExprDeepEqual()(it, arg)) {
          insert = false;
          break;
        }
      }
      if (insert) {
        args_.push_back(arg);
      }
    }
  }

public:
  float loop_cost_ = 0;
  Array<PrimExpr> args_;
  Array<PrimExpr> vars_;
  Array<PrimExpr> constants_;
  int load_times = 0;
  float flops_ = 0;
};

class TemplateCommand {
public:
  int id{-1};
  std::string name;
  Stmt stmt;
  Role role{Role::kUndefined};
  DeviceType type{DeviceType::Unspecified};
  std::vector<AccessInfo> accesses;
  int inner_coordinate{-1};
  CommandSpec spec;
  CostEstimate cost_estimate;

  TemplateCommand(int id, const Stmt &stmt)
      : id(id), name("cmd_" + std::to_string(id)), stmt(stmt) {}
};

const CallNode *GetSingleBroadcastCall(const Stmt &stmt) {
  const CallNode *broadcast = nullptr;
  PostOrderVisit(stmt, [&](const ObjectRef &obj) {
    const auto *call = obj.as<CallNode>();
    if (call && call->op.same_as(Op::Get("tl.broadcast_"))) {
      ICHECK(broadcast == nullptr)
          << "A pipeline statement may contain at most one broadcast leaf";
      broadcast = call;
    }
  });
  return broadcast;
}

bool IsAllGatherBroadcast(const TemplateCommand &cmd) {
  if (cmd.cost_estimate.op_kind == "nested_pipeline") {
    return false;
  }
  const CallNode *call = GetSingleBroadcastCall(cmd.stmt);
  if (!call) {
    return false;
  }
  SunmmioAsyncOpInfo info;
  ICHECK(TryParseSunmmioAsyncODMAOp(call, &info))
      << "tl.broadcast_ expects its fixed arguments, optional src_core, "
         "and optional ODMA metadata";
  // ODMA metadata is not a source-core argument. Use the shared semantic
  // parser so resolved and unresolved broadcasts have identical ordering
  // constraints. A rooted broadcast must still remain non-collective.
  return !info.src_core.defined();
}

bool IsCopyStage(const TemplateCommand &cmd) {
  bool has_shared_write = false;
  bool has_global_read = false;
  for (const AccessInfo &access : cmd.accesses) {
    if (access.is_write && IsSunmmioSharedBuffer(access.buffer())) {
      has_shared_write = true;
    }
    if (!access.is_write && IsGlobalBuffer(access.buffer())) {
      has_global_read = true;
    }
  }
  return has_shared_write && has_global_read;
}

bool IsProducerLike(const TemplateCommand &cmd) {
  return cmd.role == Role::kProducer ||
         (cmd.role == Role::kBoth && IsCopyStage(cmd));
}

bool IsConsumerLike(const TemplateCommand &cmd) {
  return cmd.role == Role::kConsumer ||
         (cmd.role == Role::kBoth && !IsCopyStage(cmd));
}

std::string SummarizeStmtForName(const Stmt &stmt) {
  auto buffer_scope = [](const Buffer &buffer) {
    return buffer.scope().empty() ? std::string("default")
                                  : std::string(buffer.scope());
  };
  auto buffer_label = [&](const Buffer &buffer) {
    return buffer->name + "@" + buffer_scope(buffer);
  };
  auto expr_kind = [&](const PrimExpr &expr) -> std::string {
    if (expr.as<IntImmNode>()) {
      if (const auto *imm = expr.as<IntImmNode>()) {
        return imm->value == 0 ? "const0" : "const";
      }
    }
    if (expr.as<FloatImmNode>()) {
      if (const auto *imm = expr.as<FloatImmNode>()) {
        return imm->value == 0.0 ? "const0" : "const";
      }
    }
    if (const auto *load = expr.as<BufferLoadNode>()) {
      return "copy(" + load->buffer->name + ")";
    }
    if (expr.as<MaxNode>()) {
      return "max";
    }
    if (expr.as<MinNode>()) {
      return "min";
    }
    if (expr.as<AddNode>()) {
      return "add";
    }
    if (expr.as<SubNode>()) {
      return "sub";
    }
    if (expr.as<MulNode>()) {
      return "mul";
    }
    if (expr.as<DivNode>()) {
      return "div";
    }
    if (expr.as<CastNode>()) {
      return "cast";
    }
    if (const auto *call = expr.as<CallNode>()) {
      if (call->op.same_as(Op::Get("tir.exp2"))) {
        return "exp2";
      }
      if (call->op.same_as(Op::Get("tir.if_then_else"))) {
        return "if_then_else";
      }
      if (call->op.same_as(Op::Get("tir.bitwise_and"))) {
        return "bitwise_and";
      }
    }
    return expr->GetTypeKey();
  };
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    if (const auto *call = eval->value.as<CallNode>()) {
      if (call->op.same_as(Op::Get("tl.dma_copy"))) {
        BufferRegion src = NormalizeToBufferRegion(call->args[0]);
        BufferRegion dst = NormalizeToBufferRegion(call->args[1]);
        return "dma_copy(" + buffer_label(src->buffer) + "->" +
               buffer_label(dst->buffer) + ")";
      }
    }
    return "evaluate";
  }
  if (const auto *block = stmt.as<BlockRealizeNode>()) {
    if (const auto *eval = block->block->body.as<EvaluateNode>()) {
      if (const auto *call = eval->value.as<CallNode>()) {
        if (call->op.same_as(Op::Get("tl.mma_sunmmio"))) {
          BufferRegion a = NormalizeToBufferRegion(call->args[0]);
          BufferRegion b = NormalizeToBufferRegion(call->args[1]);
          BufferRegion c = NormalizeToBufferRegion(call->args[2]);
          auto static_extent = [](const BufferRegion &region,
                                  int axis) -> std::optional<int64_t> {
            int rank = static_cast<int>(region->region.size());
            int normalized_axis = axis < 0 ? rank + axis : axis;
            if (normalized_axis < 0 || normalized_axis >= rank)
              return std::nullopt;
            const auto *extent =
                region->region[normalized_axis]->extent.as<IntImmNode>();
            return extent ? std::optional<int64_t>(extent->value)
                          : std::nullopt;
          };
          auto m = static_extent(c, -2);
          auto n = static_extent(c, -1);
          auto k = static_extent(a, -1);
          if (!m || !n || !k)
            return "mma_sunmmio";
          int64_t batch = 1;
          for (const BufferRegion &region : {a, b, c}) {
            auto extent = static_extent(region, 0);
            if (region->region.size() == 3 && extent && *extent > 1)
              batch = *extent;
          }
          std::string matrix = std::to_string(*m) + "x" + std::to_string(*n) +
                               "x" + std::to_string(*k);
          if (c->region.size() == 3)
            return "mma_sunmmio(" + std::to_string(batch) + "x" + matrix + ")";
          if (batch > 1)
            return "mma_sunmmio(" + std::to_string(batch) + "x" + matrix +
                   "->" + std::to_string(*m) + "x" + std::to_string(*n) + ")";
          return "mma_sunmmio(" + matrix + ")";
        }
      }
    }
    if (block->block->name_hint == "reduce_tile_op") {
      std::string reduce_kind = "reduce";
      std::string reduce_dst;
      bool found_reduce = false;
      PostOrderVisit(block->block->body, [&](const ObjectRef &obj) {
        if (found_reduce) {
          return;
        }
        if (const auto *eval = obj.as<EvaluateNode>()) {
          if (const auto *call = eval->value.as<CallNode>()) {
            if (call->op.same_as(Op::Get("tl.vector_core_in_tile_reduce")) &&
                !call->args.empty()) {
              if (const auto *kind = call->args[0].as<StringImmNode>()) {
                reduce_kind = kind->value;
              } else {
                reduce_kind = "reduce";
              }
              if (call->args.size() >= 2) {
                BufferRegion dst = NormalizeToBufferRegion(call->args[1]);
                reduce_dst = buffer_label(dst->buffer);
              }
              found_reduce = true;
            }
          }
        }
      });
      return reduce_dst.empty()
                 ? "reduce_tile_op(" + reduce_kind + ")"
                 : "reduce_tile_op(" + reduce_kind + " -> " + reduce_dst + ")";
    }
    if (!block->block->name_hint.empty()) {
      return "block:" + block->block->name_hint;
    }
    return "block";
  }
  if (const auto *loop = stmt.as<ForNode>()) {
    std::string summary = "for";
    PostOrderVisit(loop->body, [&](const ObjectRef &obj) {
      if (summary != "for") {
        return;
      }
      if (const auto *store = obj.as<BufferStoreNode>()) {
        summary = "for store(" + buffer_label(store->buffer) +
                  " := " + expr_kind(store->value) + ")";
        return;
      }
      if (const auto *eval = obj.as<EvaluateNode>()) {
        if (const auto *call = eval->value.as<CallNode>()) {
          if (call->op.same_as(Op::Get("tl.vector_core_in_tile_reduce")) &&
              !call->args.empty()) {
            if (const auto *kind = call->args[0].as<StringImmNode>()) {
              summary = std::string("for reduce(") + kind->value + ")";
            } else {
              summary = "for reduce";
            }
            return;
          }
        }
      }
    });
    return summary;
  }
  return stmt->GetTypeKey();
}

int GetPingPongMemoryKind(const Buffer &buffer) {
  if (buffer.scope() == "shared.wsram") {
    return 0;
  }
  if (buffer.scope() == "shared.asram") {
    return 1;
  }
  return -1;
}

int GetMemoryWriteResource(int mem) {
  if (mem == 0) {
    return static_cast<int>(IlpResourceType::kWsramIn);
  }
  if (mem == 1) {
    return static_cast<int>(IlpResourceType::kAsramIn);
  }
  return -1;
}

int GetMemoryReadResource(int mem) {
  if (mem == 0) {
    return static_cast<int>(IlpResourceType::kWsramOut);
  }
  if (mem == 1) {
    return static_cast<int>(IlpResourceType::kAsramOut);
  }
  return -1;
}

bool CommandUsesResource(const CommandSpec &spec, int resource) {
  return std::find(spec.resources.begin(), spec.resources.end(), resource) !=
         spec.resources.end();
}

FlowSpec MakeInternalFlowSpec(const Problem &problem, int prod, int cons,
                              int delta, int mem,
                              const std::string &buffer_name) {
  FlowSpec flow;
  flow.resident = false;
  flow.prod = prod;
  flow.cons = cons;
  flow.delta = delta;
  flow.mem = mem;
  flow.buffer_name = buffer_name;
  flow.fixed_bank = -1;
  // Keep the initial ILP input coarse-grained: each SRAM flow currently counts
  // as one bank-capacity unit until a more precise footprint model is wired in.
  flow.fp = 1;
  flow.initial_time = 0;
  flow.w_off = 0;
  flow.w_dur = problem.P[prod].latency;
  flow.r_off = 0;
  flow.r_dur = problem.P[cons].latency;
  flow.write_resource = GetMemoryWriteResource(mem);
  flow.read_resource = GetMemoryReadResource(mem);
  return flow;
}

FlowSpec MakeResidentFlowSpec(const Problem &problem, int cons, int mem,
                              const std::string &buffer_name) {
  FlowSpec flow;
  flow.resident = true;
  flow.prod = -1;
  flow.cons = cons;
  flow.delta = 0;
  flow.mem = mem;
  flow.buffer_name = buffer_name;
  flow.fixed_bank = -1;
  flow.fp = 1;
  flow.initial_time = 0;
  flow.w_off = 0;
  flow.w_dur = 0;
  flow.r_off = 0;
  flow.r_dur = problem.P[cons].latency;
  flow.write_resource = -1;
  flow.read_resource = GetMemoryReadResource(mem);
  return flow;
}

std::string ExtractBufferNameFromCommandLabel(const std::string &name) {
  size_t arrow = name.find("->");
  if (arrow == std::string::npos) {
    return "";
  }
  size_t at = name.find('@', arrow + 2);
  if (at == std::string::npos) {
    return "";
  }
  return name.substr(arrow + 2, at - (arrow + 2));
}

int PositiveMod(int value, int mod) {
  if (mod <= 0) {
    return value;
  }
  int result = value % mod;
  if (result < 0) {
    result += mod;
  }
  return result;
}

int RuntimeVersionCount(const Buffer &buffer, int iterations) {
  if (iterations <= 0 || GetPingPongMemoryKind(buffer) >= 0) {
    return 1;
  }
  return iterations;
}

int RuntimeBankedVersionCount(const Buffer &buffer, int iterations) {
  if (iterations <= 2 || GetPingPongMemoryKind(buffer) < 0) {
    return 1;
  }
  return CeilDiv(iterations, 2);
}

struct ScheduledAccessWindow {
  int logical_iter{0};
  int command_iter{0};
  int cmd_id{-1};
  int start{0};
  int end{0};
  bool is_write{false};
  int physical_bank{-1};
  BufferRegion region;
};

struct BufferInstanceLifetime {
  int logical_iter{0};
  int command_iter{0};
  int first_write_cmd_id{-1};
  int write_bank{-1};
  int start{0};
  int end{0};
  std::vector<BufferRegion> access_regions;
  std::vector<BufferRegion> write_regions;
};

std::vector<Buffer> DetectRuntimeMultiversionBuffers(
    const std::vector<TemplateCommand> &commands,
    const std::vector<Buffer> &versioned_buffers,
    const std::vector<Buffer> &runtime_banked_buffers,
    const Var &pipeline_loop_var, const SolveResult &sol, int iterations,
    bool enable_lifetime_pruning,
    const std::map<std::string, int> &runtime_bank_start_phases,
    const std::map<std::string, int> &runtime_bank_read_delta_parities,
    const std::map<std::string, std::map<int, int>> &runtime_bank_writer_phases,
    const std::map<std::string, std::map<int, int>> &runtime_bank_reader_phases,
    const std::map<std::string, int> &runtime_bank_flip_modes) {
  // Runtime versions follow the selected logical iteration count.  In
  // particular, stage shrinking changes this value independently of the
  // schedule span ceil(makespan / II).
  iterations = std::max(iterations, 0);
  std::unordered_set<const BufferNode *> candidates;
  std::unordered_set<const BufferNode *> banked_candidates;
  for (const Buffer &buffer : runtime_banked_buffers) {
    banked_candidates.insert(buffer.get());
  }
  for (const Buffer &buffer : versioned_buffers) {
    bool is_banked = banked_candidates.count(buffer.get()) != 0;
    int version_count = is_banked
                            ? RuntimeBankedVersionCount(buffer, iterations)
                            : RuntimeVersionCount(buffer, iterations);
    if (version_count > 1) {
      candidates.insert(buffer.get());
    }
  }
  if (candidates.empty()) {
    return {};
  }
  if (!enable_lifetime_pruning) {
    std::vector<Buffer> result;
    for (const Buffer &buffer : versioned_buffers) {
      if (candidates.count(buffer.get())) {
        result.push_back(buffer);
      }
    }
    return result;
  }

  auto resolve_access_bank = [&](const Buffer &buffer, int cmd_id,
                                 int command_iter, bool is_write) {
    if (!banked_candidates.count(buffer.get())) {
      return -1;
    }
    int phase = 0;
    bool flip = true;
    auto it_flip = runtime_bank_flip_modes.find(buffer->name);
    if (it_flip != runtime_bank_flip_modes.end()) {
      flip = it_flip->second != 0;
    }
    int iter_phase = flip ? command_iter : 0;
    const auto &phase_maps =
        is_write ? runtime_bank_writer_phases : runtime_bank_reader_phases;
    auto it_buffer = phase_maps.find(buffer->name);
    if (it_buffer != phase_maps.end()) {
      auto it_phase = it_buffer->second.find(cmd_id);
      if (it_phase != it_buffer->second.end()) {
        phase = it_phase->second;
        return PositiveMod(iter_phase + phase, 2);
      }
    }
    auto it_start = runtime_bank_start_phases.find(buffer->name);
    if (it_start != runtime_bank_start_phases.end()) {
      phase = it_start->second;
    }
    if (!is_write) {
      auto it_delta = runtime_bank_read_delta_parities.find(buffer->name);
      if (it_delta != runtime_bank_read_delta_parities.end()) {
        phase += it_delta->second;
      }
    }
    return PositiveMod(iter_phase + phase, 2);
  };

  int max_iter_offset = 0;
  for (const TemplateCommand &cmd : commands) {
    for (const AccessInfo &access : cmd.accesses) {
      max_iter_offset = std::max(max_iter_offset, access.iter_offset);
    }
  }

  const int expanded_iters = std::max(2, iterations + max_iter_offset + 1);
  std::unordered_map<const BufferNode *, std::vector<ScheduledAccessWindow>>
      windows_by_buffer;
  windows_by_buffer.reserve(candidates.size());

  struct ExpandedCommand {
    int iter{0};
    const TemplateCommand *cmd{nullptr};
  };

  std::vector<int> producer_ids;
  std::vector<int> body_ids;
  producer_ids.reserve(commands.size());
  body_ids.reserve(commands.size());
  for (const TemplateCommand &cmd : commands) {
    if (IsProducerLike(cmd)) {
      producer_ids.push_back(cmd.id);
    } else {
      body_ids.push_back(cmd.id);
    }
  }

  std::vector<ExpandedCommand> expanded_commands;
  expanded_commands.reserve(expanded_iters * commands.size());
  for (int iter = 0; iter < expanded_iters; ++iter) {
    for (int id : body_ids) {
      expanded_commands.push_back(ExpandedCommand{iter, &commands[id]});
    }
    for (int id : producer_ids) {
      expanded_commands.push_back(ExpandedCommand{iter + 1, &commands[id]});
    }
  }
  std::sort(expanded_commands.begin(), expanded_commands.end(),
            [](const ExpandedCommand &a, const ExpandedCommand &b) {
              if (a.iter != b.iter) {
                return a.iter < b.iter;
              }
              return a.cmd->id < b.cmd->id;
            });

  for (const ExpandedCommand &expanded : expanded_commands) {
    const int start = sol.t[expanded.cmd->id] + expanded.iter * sol.II;
    const int end = start + expanded.cmd->spec.latency;
    for (const AccessInfo &access : expanded.cmd->accesses) {
      const BufferNode *buf = access.buffer().get();
      if (!candidates.count(buf)) {
        continue;
      }
      windows_by_buffer[buf].push_back(ScheduledAccessWindow{
          expanded.iter + access.iter_offset, expanded.iter, expanded.cmd->id,
          start, end, access.is_write,
          resolve_access_bank(access.buffer(), expanded.cmd->id, expanded.iter,
                              access.is_write),
          MaterializeBufferRegion(access.region, pipeline_loop_var,
                                  expanded.iter)});
    }
  }

  auto region_sets_intersect = [](const std::vector<BufferRegion> &lhs,
                                  const std::vector<BufferRegion> &rhs) {
    for (const BufferRegion &lhs_region : lhs) {
      for (const BufferRegion &rhs_region : rhs) {
        if (PipelineRegionIntersect(lhs_region->region, rhs_region->region)) {
          return true;
        }
      }
    }
    return false;
  };

  std::vector<Buffer> runtime_multiversion_buffers;
  for (const Buffer &buffer : versioned_buffers) {
    if (!candidates.count(buffer.get())) {
      continue;
    }

    auto it_windows = windows_by_buffer.find(buffer.get());
    if (it_windows == windows_by_buffer.end()) {
      continue;
    }

    bool is_banked = banked_candidates.count(buffer.get()) != 0;
    std::map<std::pair<int, int>, std::vector<const ScheduledAccessWindow *>>
        windows_by_instance;
    for (const ScheduledAccessWindow &window : it_windows->second) {
      int bank = is_banked ? window.physical_bank : -1;
      windows_by_instance[{window.logical_iter, bank}].push_back(&window);
    }

    std::vector<BufferInstanceLifetime> lifetimes;
    lifetimes.reserve(windows_by_instance.size());
    for (const auto &kv : windows_by_instance) {
      int first_write_start = std::numeric_limits<int>::max();
      const ScheduledAccessWindow *first_write_window = nullptr;
      for (const ScheduledAccessWindow *window : kv.second) {
        if (window->is_write) {
          if (window->start < first_write_start) {
            first_write_start = window->start;
            first_write_window = window;
          }
        }
      }
      if (first_write_start == std::numeric_limits<int>::max()) {
        continue;
      }

      BufferInstanceLifetime lifetime;
      lifetime.logical_iter = kv.first.first;
      lifetime.command_iter = first_write_window->command_iter;
      lifetime.first_write_cmd_id = first_write_window->cmd_id;
      lifetime.write_bank = kv.first.second;
      lifetime.start = first_write_start;
      lifetime.end = first_write_start;
      for (const ScheduledAccessWindow *window : kv.second) {
        if (window->end < first_write_start) {
          continue;
        }
        lifetime.end = std::max(lifetime.end, window->end);
        lifetime.access_regions.push_back(window->region);
        if (window->is_write) {
          lifetime.write_regions.push_back(window->region);
        }
      }
      if (!lifetime.write_regions.empty()) {
        lifetimes.push_back(std::move(lifetime));
      }
    }

    std::sort(
        lifetimes.begin(), lifetimes.end(),
        [](const BufferInstanceLifetime &a, const BufferInstanceLifetime &b) {
          if (a.start != b.start) {
            return a.start < b.start;
          }
          return a.logical_iter < b.logical_iter;
        });

    bool needs_runtime_multiversion = false;
    for (size_t i = 0; i < lifetimes.size() && !needs_runtime_multiversion;
         ++i) {
      for (size_t j = i + 1; j < lifetimes.size(); ++j) {
        if (lifetimes[j].start >= lifetimes[i].end) {
          break;
        }
        if (is_banked && lifetimes[i].write_bank != lifetimes[j].write_bank) {
          continue;
        }
        if (region_sets_intersect(lifetimes[i].write_regions,
                                  lifetimes[j].access_regions) ||
            region_sets_intersect(lifetimes[j].write_regions,
                                  lifetimes[i].access_regions)) {
          needs_runtime_multiversion = true;
          break;
        }
      }
    }

    if (needs_runtime_multiversion) {
      runtime_multiversion_buffers.push_back(buffer);
    }
  }

  return runtime_multiversion_buffers;
}

TimeWindowOrderResult
BuildTimeWindowOrders(const std::vector<TemplateCommand> &commands,
                      int iterations, const SolveResult &sol) {
  TimeWindowOrderResult result;
  const int stage_count = CeilDiv(sol.makespan, std::max(1, sol.II));
  const int prologue_end = std::max(0, stage_count - 1) * sol.II;
  const int body_begin = prologue_end;
  const int body_end = prologue_end + sol.II;
  const int epilogue_begin = body_end;
  const int epilogue_end = prologue_end + sol.makespan;

  int max_iter = stage_count;
  std::vector<ExpandedOrderEntry> expanded;
  expanded.reserve(max_iter * commands.size());
  for (int iter = 0; iter < max_iter; ++iter) {
    for (const TemplateCommand &cmd : commands) {
      int id = cmd.id;
      expanded.push_back(
          ExpandedOrderEntry{iter, id, sol.t[id] + iter * sol.II});
    }
  }

  std::sort(expanded.begin(), expanded.end(),
            [](const ExpandedOrderEntry &a, const ExpandedOrderEntry &b) {
              if (a.absolute_start != b.absolute_start) {
                return a.absolute_start < b.absolute_start;
              }
              if (a.iter != b.iter) {
                return a.iter < b.iter;
              }
              return a.id < b.id;
            });

  for (const ExpandedOrderEntry &entry : expanded) {
    if (entry.absolute_start < body_begin) {
      result.prologue.push_back(entry);
      continue;
    }
    if (entry.absolute_start < body_end) {
      result.body.push_back(entry);
      result.steady_state_max_iter_offset =
          std::max(result.steady_state_max_iter_offset, entry.iter);
      continue;
    }
    if (entry.absolute_start < epilogue_end) {
      result.epilogue.push_back(entry);
    }
  }
  return result;
}

BufferRegion MaterializeBufferRegion(const BufferRegion &region,
                                     const Var &loop_var, int iter) {
  if (!loop_var.defined()) {
    return region;
  }
  ffi::Map<Var, PrimExpr> vmap;
  vmap.Set(loop_var, make_const(loop_var.dtype(), iter));
  Array<Range> materialized;
  for (const Range &rng : region->region) {
    PrimExpr min = tir::Substitute(rng->min, vmap);
    PrimExpr extent = tir::Substitute(rng->extent, vmap);
    materialized.push_back(Range::FromMinExtent(min, extent));
  }
  return BufferRegion(region->buffer, materialized);
}

std::vector<Buffer>
DetectVersionedBuffers(const std::vector<TemplateCommand> &commands) {
  std::set<Buffer> used_buffers;
  std::unordered_set<const BufferNode *> consumer_used;
  std::unordered_set<const BufferNode *> producer_used;
  std::unordered_set<const BufferNode *> self_dependent_buffers;
  std::unordered_map<const BufferNode *, int> first_write_index;
  std::unordered_map<const BufferNode *, std::vector<int>> write_indexes;
  std::unordered_map<const BufferNode *, int> first_read_index;
  std::unordered_map<const BufferNode *, int> last_read_index;
  std::vector<Buffer> versioned_buffers;
  auto mark_versioned = [&](const Buffer &buffer) {
    if (std::find(versioned_buffers.begin(), versioned_buffers.end(), buffer) ==
        versioned_buffers.end()) {
      versioned_buffers.push_back(buffer);
    }
  };

  for (int i = 0; i < static_cast<int>(commands.size()); ++i) {
    bool is_producer = IsProducerLike(commands[i]);
    bool is_consumer = IsConsumerLike(commands[i]);
    std::unordered_set<const BufferNode *> reads_in_cmd;
    std::unordered_set<const BufferNode *> writes_in_cmd;
    for (const AccessInfo &access : commands[i].accesses) {
      if (IsGlobalBuffer(access.buffer())) {
        continue;
      }
      used_buffers.insert(access.buffer());
      const BufferNode *buf = access.buffer().get();
      if (access.is_write) {
        writes_in_cmd.insert(buf);
        if (is_producer) {
          producer_used.insert(buf);
        }
        if (!first_write_index.count(buf)) {
          first_write_index[buf] = i;
        }
        write_indexes[buf].push_back(i);
      } else {
        reads_in_cmd.insert(buf);
        if (is_consumer) {
          consumer_used.insert(buf);
        }
        if (!first_read_index.count(buf)) {
          first_read_index[buf] = i;
        }
        last_read_index[buf] = i;
      }
    }
    for (const BufferNode *buf : writes_in_cmd) {
      if (reads_in_cmd.count(buf)) {
        self_dependent_buffers.insert(buf);
      }
    }
  }

  for (const Buffer &buffer : used_buffers) {
    const BufferNode *buf = buffer.get();
    if (self_dependent_buffers.count(buf)) {
      continue;
    }
    auto it_w = first_write_index.find(buf);
    auto it_r = first_read_index.find(buf);
    if (it_w != first_write_index.end() && it_r != first_read_index.end() &&
        it_w->second < it_r->second) {
      mark_versioned(buffer);
      continue;
    }
    if (consumer_used.count(buf) && producer_used.count(buf)) {
      auto r = first_read_index.find(buf);
      auto w = first_write_index.find(buf);
      if (r != first_read_index.end() && w != first_write_index.end() &&
          r->second > w->second) {
        mark_versioned(buffer);
        continue;
      }
    }
    auto it_last_r = last_read_index.find(buf);
    if (it_w != first_write_index.end() && it_last_r != last_read_index.end() &&
        it_w->second < it_last_r->second &&
        IsCopyStage(commands[it_w->second])) {
      mark_versioned(buffer);
    }
  }

  bool updated = true;
  while (updated) {
    updated = false;
    for (const Buffer &buffer : used_buffers) {
      if (std::find(versioned_buffers.begin(), versioned_buffers.end(),
                    buffer) != versioned_buffers.end()) {
        continue;
      }
      const BufferNode *buf = buffer.get();
      if (self_dependent_buffers.count(buf)) {
        continue;
      }
      auto it_writes = write_indexes.find(buf);
      auto it_first_w = first_write_index.find(buf);
      auto it_first_r = first_read_index.find(buf);
      if (it_writes == write_indexes.end() || it_writes->second.empty() ||
          it_first_w == first_write_index.end() ||
          it_first_r == first_read_index.end()) {
        continue;
      }
      bool can_propagate = it_first_w->second < it_first_r->second;
      for (int idx : it_writes->second) {
        for (const AccessInfo &access : commands[idx].accesses) {
          if (access.is_write || IsGlobalBuffer(access.buffer())) {
            continue;
          }
          if (first_write_index.find(access.buffer().get()) ==
              first_write_index.end()) {
            continue;
          }
          if (std::find(versioned_buffers.begin(), versioned_buffers.end(),
                        access.buffer()) == versioned_buffers.end()) {
            can_propagate = false;
            break;
          }
        }
        if (!can_propagate) {
          break;
        }
      }
      if (can_propagate) {
        mark_versioned(buffer);
        updated = true;
      }
    }
  }
  return versioned_buffers;
}

void BuildTemplateDependencyGraph(const std::vector<TemplateCommand> &commands,
                                  int iter_mod,
                                  const std::vector<Buffer> &versioned_buffers,
                                  const Var &pipeline_loop_var,
                                  const BankFlipMode &mode, Problem *problem) {
  std::unordered_set<const BufferNode *> versioned;
  std::unordered_set<const BufferNode *> banked_versioned;
  std::unordered_set<const BufferNode *> bank_rotating_versioned;
  for (const Buffer &buffer : versioned_buffers) {
    versioned.insert(buffer.get());
    int mem = GetPingPongMemoryKind(buffer);
    if (mem >= 0) {
      banked_versioned.insert(buffer.get());
    }
    if (mem >= 0 && mode.FlipForMem(mem)) {
      bank_rotating_versioned.insert(buffer.get());
    }
  }

  problem->flows.clear();
  problem->dep_edges.clear();
  problem->delta.clear();
  problem->dependencies.clear();

  struct ExpandedCommand {
    int template_id{-1};
    int iter{-1};
    const TemplateCommand *cmd{nullptr};
  };
  enum class AccessType : uint8_t { kRead, kWrite };
  struct AccessRecord {
    BufferRegion region;
    int expanded_idx{-1};
    int access_idx{-1};
    AccessType type{AccessType::kRead};
  };

  std::map<std::pair<int, int>, int> best_delta;
  std::map<std::tuple<int, int, int, std::string, std::string>, DependencySpec>
      dependency_constraints;
  std::map<std::tuple<int, int, const BufferNode *, int, int>, int>
      flow_key_to_index;
  using ConsumerAccessKey = std::tuple<int, int, const BufferNode *, int>;
  std::set<ConsumerAccessKey> satisfied_consumer_access;

  // Pre-color distinct producer values only when one banked buffer has more
  // than one writer operation.  A color is a phase offset: physical_bank =
  // (logical_iteration + phase) % 2.  Readers inherit the phase of the write
  // that produces their value.
  std::unordered_map<const BufferNode *, std::map<int, int>> writer_phases;
  std::map<std::pair<int, int>, int> access_phases;
  for (const Buffer &buffer : versioned_buffers) {
    const BufferNode *buf = buffer.get();
    if (!banked_versioned.count(buf)) {
      continue;
    }
    std::vector<int> writer_ids;
    for (const TemplateCommand &cmd : commands) {
      bool writes_buffer = false;
      for (const AccessInfo &access : cmd.accesses) {
        writes_buffer =
            writes_buffer || (access.is_write && access.buffer().get() == buf);
      }
      if (writes_buffer) {
        writer_ids.push_back(cmd.id);
      }
    }
    if (writer_ids.size() <= 1) {
      continue;
    }
    for (size_t i = 0; i < writer_ids.size(); ++i) {
      writer_phases[buf][writer_ids[i]] = static_cast<int>(i & 1);
    }
  }

  for (const TemplateCommand &cmd : commands) {
    for (int access_idx = 0; access_idx < static_cast<int>(cmd.accesses.size());
         ++access_idx) {
      const AccessInfo &access = cmd.accesses[access_idx];
      const BufferNode *buf = access.buffer().get();
      auto phase_group = writer_phases.find(buf);
      if (phase_group == writer_phases.end()) {
        continue;
      }
      if (access.is_write) {
        access_phases[{cmd.id, access_idx}] = phase_group->second.at(cmd.id);
        continue;
      }

      int producer_phase = -1;
      int producer_iter_delta = 0;
      for (int producer_id = cmd.id - 1; producer_id >= 0 && producer_phase < 0;
           --producer_id) {
        const TemplateCommand &producer = commands[producer_id];
        for (int producer_access_idx =
                 static_cast<int>(producer.accesses.size()) - 1;
             producer_access_idx >= 0; --producer_access_idx) {
          const AccessInfo &producer_access =
              producer.accesses[producer_access_idx];
          if (!producer_access.is_write ||
              producer_access.buffer().get() != buf) {
            continue;
          }
          if (!PipelineRegionIntersect(
                  MaterializeBufferRegion(producer_access.region,
                                          pipeline_loop_var, 0)
                      ->region,
                  MaterializeBufferRegion(access.region, pipeline_loop_var, 0)
                      ->region)) {
            continue;
          }
          producer_phase = phase_group->second.at(producer_id);
          break;
        }
      }
      // A read before its producer in template order consumes the previous
      // logical iteration of the last matching writer.
      for (int producer_id = static_cast<int>(commands.size()) - 1;
           producer_id >= cmd.id && producer_phase < 0; --producer_id) {
        const TemplateCommand &producer = commands[producer_id];
        for (int producer_access_idx =
                 static_cast<int>(producer.accesses.size()) - 1;
             producer_access_idx >= 0; --producer_access_idx) {
          const AccessInfo &producer_access =
              producer.accesses[producer_access_idx];
          if (!producer_access.is_write ||
              producer_access.buffer().get() != buf) {
            continue;
          }
          if (!PipelineRegionIntersect(
                  MaterializeBufferRegion(producer_access.region,
                                          pipeline_loop_var, -1)
                      ->region,
                  MaterializeBufferRegion(access.region, pipeline_loop_var, 0)
                      ->region)) {
            continue;
          }
          producer_phase = phase_group->second.at(producer_id);
          producer_iter_delta = 1;
          break;
        }
      }
      if (producer_phase >= 0) {
        access_phases[{cmd.id, access_idx}] =
            (producer_phase +
             (bank_rotating_versioned.count(buf) ? producer_iter_delta : 0)) &
            1;
      }
    }
  }

  auto version_mod = [&](const BufferNode *buf) {
    if (iter_mod <= 0) {
      return 0;
    }
    if (bank_rotating_versioned.count(buf)) {
      // Banked buffers always rotate ping/pong every iteration. When
      // num_stages > 2 we additionally attach a runtime multiversion axis to
      // each ping/pong bank, so the full physical alias period becomes
      // 2 * ceil(num_stages / 2). For num_stages <= 2 this collapses to the
      // original ping/pong-only period 2.
      return 2 * std::max(1, CeilDiv(iter_mod, 2));
    }
    return iter_mod;
  };

  auto maybe_record_edge = [&](int src_id, int dst_id, int delta, int lag,
                               const std::string &hazard,
                               const std::string &buffer_name) {
    if (delta < 0) {
      return;
    }
    auto dependency_key =
        std::make_tuple(src_id, dst_id, delta, hazard, buffer_name);
    auto dependency_it = dependency_constraints.find(dependency_key);
    if (dependency_it == dependency_constraints.end() ||
        lag > dependency_it->second.lag) {
      dependency_constraints[dependency_key] =
          DependencySpec{src_id, dst_id, delta, lag, hazard, buffer_name};
    }
    std::pair<int, int> key{src_id, dst_id};
    auto it = best_delta.find(key);
    if (it == best_delta.end() || delta < it->second) {
      best_delta[key] = delta;
    }
  };

  auto maybe_record_flow = [&](int src_id, int dst_id, int delta,
                               const BufferNode *buf, int src_access_idx,
                               int dst_access_idx, int mem) {
    if (mem < 0 || src_id == dst_id) {
      return false;
    }
    int write_resource = GetMemoryWriteResource(mem);
    int read_resource = GetMemoryReadResource(mem);
    if (!CommandUsesResource(problem->P[src_id], write_resource) ||
        !CommandUsesResource(problem->P[dst_id], read_resource)) {
      return false;
    }
    auto flow_key =
        std::make_tuple(src_id, dst_id, buf, src_access_idx, dst_access_idx);
    auto it = flow_key_to_index.find(flow_key);
    if (it == flow_key_to_index.end()) {
      int flow_index = static_cast<int>(problem->flows.size());
      flow_key_to_index[flow_key] = flow_index;
      FlowSpec flow =
          MakeInternalFlowSpec(*problem, src_id, dst_id, delta, mem, buf->name);
      auto phase = access_phases.find({src_id, src_access_idx});
      if (phase != access_phases.end()) {
        flow.precolor = phase->second;
      }
      problem->flows.push_back(std::move(flow));
    } else {
      problem->flows[it->second].delta =
          std::min(problem->flows[it->second].delta, delta);
    }
    return true;
  };

  std::vector<int> producer_ids;
  std::vector<int> body_ids;
  producer_ids.reserve(commands.size());
  body_ids.reserve(commands.size());
  for (const TemplateCommand &cmd : commands) {
    if (IsProducerLike(cmd)) {
      producer_ids.push_back(cmd.id);
    } else {
      body_ids.push_back(cmd.id);
    }
  }

  int steady_state_iters = std::max(1, iter_mod);
  for (const Buffer &buffer : versioned_buffers) {
    steady_state_iters =
        std::max(steady_state_iters, std::max(1, version_mod(buffer.get())));
  }
  int expanded_iters = std::max(2, steady_state_iters + 1);
  std::vector<ExpandedCommand> expanded_commands;
  expanded_commands.reserve(expanded_iters * commands.size());
  for (int iter = 0; iter < expanded_iters; ++iter) {
    for (int id : body_ids) {
      expanded_commands.push_back(ExpandedCommand{id, iter, &commands[id]});
    }
    for (int id : producer_ids) {
      expanded_commands.push_back(ExpandedCommand{id, iter + 1, &commands[id]});
    }
  }
  std::sort(expanded_commands.begin(), expanded_commands.end(),
            [](const ExpandedCommand &a, const ExpandedCommand &b) {
              if (a.iter != b.iter) {
                return a.iter < b.iter;
              }
              return a.template_id < b.template_id;
            });

  auto access_version = [&](const BufferNode *buf,
                            const ExpandedCommand &command,
                            const AccessInfo &access, int access_idx) {
    int mod = version_mod(buf);
    if (mod <= 0) {
      return command.iter + access.iter_offset;
    }
    int logical_iter = command.iter + access.iter_offset;
    auto phase = access_phases.find({command.template_id, access_idx});
    if (!banked_versioned.count(buf)) {
      return PositiveMod(logical_iter, mod);
    }
    int bank_phase = phase == access_phases.end() ? 0 : phase->second;
    bool flip = bank_rotating_versioned.count(buf) != 0;
    int versions_per_bank = CeilDiv(mod, 2);
    int bank = flip ? PositiveMod(logical_iter + bank_phase, 2)
                    : PositiveMod(bank_phase, 2);
    int version_in_bank =
        flip ? PositiveMod(FloorDiv(logical_iter, 2), versions_per_bank)
             : PositiveMod(logical_iter, versions_per_bank);
    return version_in_bank * 2 + bank;
  };

  auto materialize_access = [&](const ExpandedCommand &command,
                                const AccessInfo &access) {
    return MaterializeBufferRegion(access.region, pipeline_loop_var,
                                   command.iter);
  };

  std::unordered_map<const BufferNode *, std::vector<AccessRecord>>
      buffer_access_history;
  buffer_access_history.reserve(versioned.size() + commands.size());

  struct ResidentCandidate {
    int cmd_id{-1};
    int mem{-1};
    const BufferNode *buffer{nullptr};
    int access_idx{-1};
    std::string buffer_name;
  };
  std::vector<ResidentCandidate> resident_candidates;

  for (int curr_idx = 0; curr_idx < static_cast<int>(expanded_commands.size());
       ++curr_idx) {
    const ExpandedCommand &curr_cmd = expanded_commands[curr_idx];

    for (int dst_access_idx = 0;
         dst_access_idx < static_cast<int>(curr_cmd.cmd->accesses.size());
         ++dst_access_idx) {
      const AccessInfo &dst_access = curr_cmd.cmd->accesses[dst_access_idx];
      if (dst_access.is_write) {
        continue;
      }
      const BufferNode *buf = dst_access.buffer().get();
      auto hist_it = buffer_access_history.find(buf);
      if (hist_it == buffer_access_history.end()) {
        continue;
      }
      BufferRegion dst_region = materialize_access(curr_cmd, dst_access);
      for (auto it = hist_it->second.rbegin(); it != hist_it->second.rend();
           ++it) {
        const ExpandedCommand &src_cmd = expanded_commands[it->expanded_idx];
        const AccessInfo &src_access = src_cmd.cmd->accesses[it->access_idx];
        if (versioned.count(buf) &&
            access_version(buf, src_cmd, src_access, it->access_idx) !=
                access_version(buf, curr_cmd, dst_access, dst_access_idx)) {
          continue;
        }
        if (it->type != AccessType::kWrite ||
            !PipelineRegionIntersect(dst_region->region, it->region->region)) {
          continue;
        }
        maybe_record_edge(src_cmd.template_id, curr_cmd.template_id,
                          curr_cmd.iter - src_cmd.iter,
                          src_access.last_offset - dst_access.first_offset,
                          "RAW", dst_access.buffer()->name);
        int mem = GetPingPongMemoryKind(dst_access.buffer());
        bool has_concrete_flow =
            maybe_record_flow(src_cmd.template_id, curr_cmd.template_id,
                              curr_cmd.iter - src_cmd.iter, buf, it->access_idx,
                              dst_access_idx, mem);
        if (has_concrete_flow) {
          satisfied_consumer_access.insert(
              std::make_tuple(curr_cmd.template_id, mem, buf, dst_access_idx));
        }
        break;
      }
    }

    for (int dst_access_idx = 0;
         dst_access_idx < static_cast<int>(curr_cmd.cmd->accesses.size());
         ++dst_access_idx) {
      const AccessInfo &dst_access = curr_cmd.cmd->accesses[dst_access_idx];
      if (!dst_access.is_write) {
        continue;
      }
      const BufferNode *buf = dst_access.buffer().get();
      auto hist_it = buffer_access_history.find(buf);
      if (hist_it == buffer_access_history.end()) {
        continue;
      }
      BufferRegion dst_region = materialize_access(curr_cmd, dst_access);
      for (auto it = hist_it->second.rbegin(); it != hist_it->second.rend();
           ++it) {
        const ExpandedCommand &src_cmd = expanded_commands[it->expanded_idx];
        const AccessInfo &src_access = src_cmd.cmd->accesses[it->access_idx];
        if (versioned.count(buf) &&
            access_version(buf, src_cmd, src_access, it->access_idx) !=
                access_version(buf, curr_cmd, dst_access, dst_access_idx)) {
          continue;
        }
        if (!PipelineRegionIntersect(dst_region->region, it->region->region)) {
          continue;
        }
        maybe_record_edge(src_cmd.template_id, curr_cmd.template_id,
                          curr_cmd.iter - src_cmd.iter, src_access.last_offset,
                          it->type == AccessType::kWrite ? "WAW" : "WAR",
                          dst_access.buffer()->name);
        if (it->type == AccessType::kWrite) {
          break;
        }
      }
    }

    for (int access_idx = 0;
         access_idx < static_cast<int>(curr_cmd.cmd->accesses.size());
         ++access_idx) {
      const AccessInfo &access = curr_cmd.cmd->accesses[access_idx];
      buffer_access_history[access.buffer().get()].push_back(AccessRecord{
          materialize_access(curr_cmd, access), curr_idx, access_idx,
          access.is_write ? AccessType::kWrite : AccessType::kRead});
    }
  }

  // Every core executes an all-gather broadcast and enters its participant
  // barrier before issuing the mcast.  Row and column collectives can use
  // different ODMA engines and buffers, so ordinary resource and data hazards
  // do not preserve a common barrier encounter order across cores.  Chain the
  // collectives in template order, including the loop-carried last-to-first
  // edge, while keeping their commands and bank precolor constraints separate.
  std::vector<int> all_gather_ids;
  for (const TemplateCommand &cmd : commands) {
    if (IsAllGatherBroadcast(cmd)) {
      all_gather_ids.push_back(cmd.id);
    }
  }
  for (size_t i = 1; i < all_gather_ids.size(); ++i) {
    int src = all_gather_ids[i - 1];
    maybe_record_edge(src, all_gather_ids[i], 0, problem->P[src].latency,
                      "collective_order", "");
  }
  if (all_gather_ids.size() > 1) {
    int src = all_gather_ids.back();
    maybe_record_edge(src, all_gather_ids.front(), 1, problem->P[src].latency,
                      "collective_order", "");
  }

  for (const auto &kv : dependency_constraints) {
    problem->dependencies.push_back(kv.second);
  }

  for (const auto &kv : best_delta) {
    problem->dep_edges.push_back(kv.first);
    problem->delta[EdgeKey(kv.first.first, kv.first.second)] = kv.second;
  }

  for (const TemplateCommand &cmd : commands) {
    for (int access_idx = 0; access_idx < static_cast<int>(cmd.accesses.size());
         ++access_idx) {
      const AccessInfo &access = cmd.accesses[access_idx];
      if (access.is_write)
        continue;
      int mem = GetPingPongMemoryKind(access.buffer());
      if (mem < 0 || !CommandUsesResource(problem->P[cmd.id],
                                          GetMemoryReadResource(mem))) {
        continue;
      }
      resident_candidates.push_back(
          ResidentCandidate{cmd.id, mem, access.buffer().get(), access_idx,
                            access.buffer()->name});
    }
  }

  std::set<ConsumerAccessKey> emitted_resident_access;
  for (const ResidentCandidate &candidate : resident_candidates) {
    ConsumerAccessKey key =
        std::make_tuple(candidate.cmd_id, candidate.mem, candidate.buffer,
                        candidate.access_idx);
    if (satisfied_consumer_access.count(key) != 0) {
      continue;
    }
    if (!emitted_resident_access.insert(key).second) {
      continue;
    }
    problem->flows.push_back(MakeResidentFlowSpec(
        *problem, candidate.cmd_id, candidate.mem, candidate.buffer_name));
  }
}

bool ValidateProblemGraph(const std::vector<TemplateCommand> &commands,
                          const Problem &problem) {
  if (problem.N != static_cast<int>(commands.size()) ||
      static_cast<int>(problem.P.size()) != problem.N) {
    return false;
  }
  for (int id = 0; id < problem.N; ++id) {
    if (commands[id].id != id) {
      return false;
    }
    for (const AccessInfo &access : commands[id].accesses) {
      if (!access.region.defined() || !access.buffer().defined() ||
          access.first_offset < 0 || access.last_offset < access.first_offset ||
          access.last_offset > problem.P[id].latency) {
        return false;
      }
    }
  }
  for (const auto &edge : problem.dep_edges) {
    if (edge.first < 0 || edge.first >= problem.N || edge.second < 0 ||
        edge.second >= problem.N) {
      return false;
    }
    auto delta_it = problem.delta.find(EdgeKey(edge.first, edge.second));
    if (delta_it == problem.delta.end() || delta_it->second < 0) {
      return false;
    }
  }
  for (const DependencySpec &dependency : problem.dependencies) {
    if (dependency.src < 0 || dependency.src >= problem.N ||
        dependency.dst < 0 || dependency.dst >= problem.N ||
        dependency.delta < 0 || dependency.hazard.empty()) {
      return false;
    }
  }
  for (const FlowSpec &flow : problem.flows) {
    if (flow.cons < 0 || flow.cons >= problem.N ||
        (!flow.resident && (flow.prod < 0 || flow.prod >= problem.N)) ||
        flow.delta < 0) {
      return false;
    }
  }
  return true;
}

int ResourceLowerBound(const Problem &prob) {
  int lb = 1;
  for (int r : prob.R) {
    int cap = 1;
    auto it = prob.cap.find(r);
    if (it != prob.cap.end()) {
      cap = it->second;
    }
    if (cap <= 0) {
      continue;
    }
    long long total = 0;
    for (int i = 0; i < prob.N; ++i) {
      if (std::find(prob.P[i].resources.begin(), prob.P[i].resources.end(),
                    r) != prob.P[i].resources.end()) {
        total += ResourceUsageDuration(prob.P[i], r);
      }
    }
    lb = std::max(lb, std::max(1, int((total + cap - 1) / cap)));
  }
  return lb;
}

struct FastParityDsu {
  std::vector<int> parent;
  std::vector<int> parity_to_parent;

  explicit FastParityDsu(int n) : parent(n), parity_to_parent(n, 0) {
    for (int i = 0; i < n; ++i) {
      parent[i] = i;
    }
  }

  std::pair<int, int> Find(int value) {
    if (parent[value] == value) {
      return {value, 0};
    }
    auto root = Find(parent[value]);
    parity_to_parent[value] ^= root.second;
    parent[value] = root.first;
    return {parent[value], parity_to_parent[value]};
  }

  bool Unite(int lhs, int rhs, int parity) {
    auto l = Find(lhs);
    auto r = Find(rhs);
    if (l.first == r.first) {
      return (l.second ^ r.second) == parity;
    }
    parent[l.first] = r.first;
    parity_to_parent[l.first] = l.second ^ r.second ^ parity;
    return true;
  }

  std::vector<int> Values() {
    std::vector<int> result(parent.size(), 0);
    for (int i = 0; i < static_cast<int>(parent.size()); ++i) {
      result[i] = Find(i).second & 1;
    }
    return result;
  }
};

int FastScheduleTimeUpperBound(const Problem &prob, int II) {
  int max_delta = 0;
  int max_latency = 0;
  for (const DependencySpec &dependency : prob.dependencies) {
    max_delta = std::max(max_delta, dependency.delta);
  }
  for (const CommandSpec &spec : prob.P) {
    max_latency = std::max(max_latency, spec.latency);
  }
  return prob.Tmax + max_delta * II + max_latency;
}

bool FastCommandUsesResource(const CommandSpec &spec, int resource) {
  return std::find(spec.resources.begin(), spec.resources.end(), resource) !=
         spec.resources.end();
}

bool FastDependenciesAllowCandidate(const Problem &prob,
                                    const std::vector<int> &starts, int cmd,
                                    int start, int II) {
  for (const DependencySpec &dependency : prob.dependencies) {
    int src = dependency.src;
    int dst = dependency.dst;
    if (src == cmd && dst == cmd) {
      if (dependency.lag > dependency.delta * II) {
        return false;
      }
      continue;
    }
    if (src == cmd && starts[dst] >= 0) {
      if (start + dependency.lag > starts[dst] + dependency.delta * II) {
        return false;
      }
    } else if (dst == cmd && starts[src] >= 0) {
      if (starts[src] + dependency.lag > start + dependency.delta * II) {
        return false;
      }
    }
  }
  return true;
}

bool FastResourcesAllowCandidate(const Problem &prob,
                                 const std::vector<int> &starts, int cmd,
                                 int start, int II) {
  for (int resource : prob.P[cmd].resources) {
    int cap = 1;
    auto cap_it = prob.cap.find(resource);
    if (cap_it != prob.cap.end()) {
      cap = cap_it->second;
    }
    for (int slot = 0; slot < II; ++slot) {
      int use = ComputeFoldedResourceOccupancy(prob.P[cmd], resource, start, II,
                                               slot);
      if (use == 0) {
        continue;
      }
      for (int other = 0; other < prob.N; ++other) {
        if (starts[other] < 0) {
          continue;
        }
        if (!FastCommandUsesResource(prob.P[other], resource)) {
          continue;
        }
        use += ComputeFoldedResourceOccupancy(prob.P[other], resource,
                                              starts[other], II, slot);
      }
      if (use > cap) {
        return false;
      }
    }
  }
  return true;
}

bool FastZeroDeltaPredecessorsAssigned(const Problem &prob,
                                       const std::vector<int> &starts,
                                       int cmd) {
  for (const DependencySpec &dependency : prob.dependencies) {
    int src = dependency.src;
    int dst = dependency.dst;
    if (src == dst || dst != cmd) {
      continue;
    }
    if (dependency.delta == 0 && starts[src] < 0) {
      return false;
    }
  }
  return true;
}

bool FastEarliestStart(const Problem &prob, const std::vector<int> &starts,
                       int cmd, int II, int time_ub, int *out_start) {
  int lower = 0;
  int upper = time_ub;
  for (const DependencySpec &dependency : prob.dependencies) {
    int src = dependency.src;
    int dst = dependency.dst;
    if (dst == cmd && starts[src] >= 0) {
      lower =
          std::max(lower, starts[src] + dependency.lag - dependency.delta * II);
    } else if (src == cmd && starts[dst] >= 0) {
      upper =
          std::min(upper, starts[dst] + dependency.delta * II - dependency.lag);
    }
  }
  if (upper < lower) {
    return false;
  }
  for (int start = std::max(0, lower); start <= upper; ++start) {
    if (!FastDependenciesAllowCandidate(prob, starts, cmd, start, II)) {
      continue;
    }
    if (!FastResourcesAllowCandidate(prob, starts, cmd, start, II)) {
      continue;
    }
    *out_start = start;
    return true;
  }
  return false;
}

bool FastAssignBankPhases(const Problem &prob, const std::vector<int> &starts,
                          int II, const BankFlipMode &mode,
                          std::vector<int> *z_bank) {
  FastParityDsu dsu(static_cast<int>(prob.flows.size()));
  std::map<SameWriteFlowKey, std::vector<int>> same_write_groups;
  for (int idx = 0; idx < static_cast<int>(prob.flows.size()); ++idx) {
    const FlowSpec &flow = prob.flows[idx];
    if (flow.write_resource < 0 || flow.prod < 0) {
      continue;
    }
    same_write_groups[MakeSameWriteFlowKey(flow)].push_back(idx);
  }
  for (const auto &kv : same_write_groups) {
    const std::vector<int> &flows = kv.second;
    for (size_t i = 1; i < flows.size(); ++i) {
      if (!dsu.Unite(flows[0], flows[i], 0)) {
        return false;
      }
    }
  }

  for (int a = 0; a < static_cast<int>(prob.flows.size()); ++a) {
    const FlowSpec &write_flow = prob.flows[a];
    if (write_flow.write_resource < 0 || write_flow.prod < 0) {
      continue;
    }
    int write_start_time = starts[write_flow.prod];
    for (int b = 0; b < static_cast<int>(prob.flows.size()); ++b) {
      if (a == b) {
        continue;
      }
      const FlowSpec &read_flow = prob.flows[b];
      if (read_flow.cons < 0) {
        continue;
      }
      int read_start_time = starts[read_flow.cons] + read_flow.delta * II;
      ConflictType conflict = AnalyzeFlowConflict(
          write_flow, write_start_time, read_flow, read_start_time, II, mode);
      if (conflict == ConflictType::kImpossible) {
        return false;
      }
      if (conflict == ConflictType::kNeedDifferent) {
        if (!dsu.Unite(a, b, 1)) {
          return false;
        }
      } else if (conflict == ConflictType::kNeedSame) {
        if (!dsu.Unite(a, b, 0)) {
          return false;
        }
      }
    }
  }

  *z_bank = dsu.Values();
  return true;
}

SolveResult TryFastVerifiedScheduleForMode(const Problem &prob, int II,
                                           const BankFlipMode &mode) {
  SolveResult result;
  if (prob.N <= 0 || II <= 0) {
    return result;
  }
  int time_ub = FastScheduleTimeUpperBound(prob, II);
  std::vector<int> starts(prob.N, -1);
  std::vector<bool> scheduled(prob.N, false);

  for (int placed = 0; placed < prob.N; ++placed) {
    int best_cmd = -1;
    int best_start = 0;
    bool found_ready = false;
    for (int cmd = 0; cmd < prob.N; ++cmd) {
      if (scheduled[cmd]) {
        continue;
      }
      bool ready = FastZeroDeltaPredecessorsAssigned(prob, starts, cmd);
      if (found_ready && !ready) {
        continue;
      }
      int candidate_start = 0;
      if (!FastEarliestStart(prob, starts, cmd, II, time_ub,
                             &candidate_start)) {
        continue;
      }
      if (!found_ready && ready) {
        found_ready = true;
        best_cmd = -1;
      }
      if (ready == found_ready &&
          (best_cmd < 0 || candidate_start < best_start ||
           (candidate_start == best_start && cmd < best_cmd))) {
        best_cmd = cmd;
        best_start = candidate_start;
      }
    }
    if (best_cmd < 0) {
      return {};
    }
    starts[best_cmd] = best_start;
    scheduled[best_cmd] = true;
  }

  std::vector<int> z_bank;
  if (!FastAssignBankPhases(prob, starts, II, mode, &z_bank)) {
    return {};
  }

  result.ok = true;
  result.objective_optimized = false;
  result.fast_verified_schedule = true;
  result.II = II;
  result.bank_slot_period = 2 * II;
  result.bank_flip_mode = mode;
  result.vc_blocking_issue_modeled = true;
  result.vc_blocking_issue_constraints = 0;
  result.t = starts;
  result.m.resize(prob.N, 0);
  result.y.resize(prob.N, 0);
  result.makespan = 0;
  for (int i = 0; i < prob.N; ++i) {
    result.m[i] = PositiveMod(result.t[i], II);
    result.y[i] = result.t[i] / II;
    result.makespan =
        std::max(result.makespan, result.t[i] + prob.P[i].latency);
  }
  result.internal_flow_ids.resize(prob.flows.size());
  for (int i = 0; i < static_cast<int>(prob.flows.size()); ++i) {
    result.internal_flow_ids[i] = i;
  }
  result.z_bank = std::move(z_bank);

  SolutionVerifyResult verify = VerifySolution(prob, result);
  if (!verify.ok) {
    return {};
  }
  return result;
}

int64_t FiniteHorizonScheduleCost(const SolveResult &sol,
                                  int64_t logical_extent) {
  ICHECK(sol.ok);
  ICHECK_GT(logical_extent, 0);
  return static_cast<int64_t>(sol.makespan) +
         (logical_extent - 1) * static_cast<int64_t>(sol.II);
}

SolveResult RefineV2FiniteHorizonSchedule(const Problem &prob,
                                          const SolveResult &seed,
                                          int64_t logical_extent) {
  if (!seed.ok || logical_extent <= 1 ||
      !GetEnvBool("TL_SUNMMIO_ILP_FAST_VERIFIED_SCHEDULER", true)) {
    return seed;
  }
  if (CeilDiv(seed.makespan, std::max(1, seed.II)) <= 2) {
    return seed;
  }
  int slack = GetEnvInt("TL_SUNMMIO_ILP_V2_FINITE_HORIZON_SLACK", 4);
  if (slack <= 0) {
    return seed;
  }

  SolveResult best = seed;
  int64_t best_cost = FiniteHorizonScheduleCost(best, logical_extent);
  for (int II = seed.II + 1; II <= seed.II + slack; ++II) {
    SolveResult candidate =
        TryFastVerifiedScheduleForMode(prob, II, seed.bank_flip_mode);
    if (!candidate.ok) {
      continue;
    }
    int64_t candidate_cost =
        FiniteHorizonScheduleCost(candidate, logical_extent);
    if (candidate_cost < best_cost ||
        (candidate_cost == best_cost && candidate.II < best.II)) {
      best = std::move(candidate);
      best_cost = candidate_cost;
    }
  }
  if (best.II != seed.II || best.makespan != seed.makespan) {
    LOG(INFO) << "[ILP] v2 finite-horizon refinement extent=" << logical_extent
              << " seed_ii=" << seed.II << " seed_makespan=" << seed.makespan
              << " selected_ii=" << best.II
              << " selected_makespan=" << best.makespan
              << " selected_cost=" << best_cost;
  }
  return best;
}

SolveResult TryFastVerifiedSchedule(const Problem &prob,
                                    const BankFlipMode &mode, int max_ii = -1) {
  if (!GetEnvBool("TL_SUNMMIO_ILP_FAST_VERIFIED_SCHEDULER", true)) {
    return {};
  }
  int lb = std::max(1, ResourceLowerBound(prob));
  int upper = std::max(lb, prob.Tmax);
  if (max_ii > 0) {
    upper = std::min(upper, max_ii);
  }
  int extra = GetEnvInt("TL_SUNMMIO_ILP_FAST_MAX_EXTRA_II", 80);
  if (extra >= 0) {
    upper = std::min(upper, lb + extra);
  }
  for (int II = lb; II <= upper; ++II) {
    SolveResult fast = TryFastVerifiedScheduleForMode(prob, II, mode);
    if (fast.ok) {
      LOG(INFO) << "[ILP] fast verified scheduler found II=" << II
                << " mode=" << mode.Id() << " makespan=" << fast.makespan;
      return fast;
    }
  }
  LOG(INFO) << "[ILP] fast verified scheduler found no solution in [" << lb
            << ", " << upper << "] for mode=" << mode.Id();
  return {};
}

std::string MakeSolveCacheKey(const Problem &prob, int II, bool optimize_t,
                              const BankFlipMode &mode,
                              bool model_vc_blocking_issue) {
  // Include every field that can affect BuildModel or the VC issue
  // constraints.  The cache is only an exact memoization of an identical
  // mathematical model; solver settings that merely select a search strategy
  // (for example, thread count) intentionally do not enter the key.
  std::ostringstream os;
  os << "N=" << prob.N << ";Tmax=" << prob.Tmax << ";II=" << II
     << ";opt=" << optimize_t << ";mode=" << mode.Id()
     << ";vc=" << model_vc_blocking_issue;
  os << ";R=";
  for (int resource : prob.R) {
    auto cap_it = prob.cap.find(resource);
    int capacity = cap_it == prob.cap.end() ? 1 : cap_it->second;
    os << resource << ":" << capacity << ",";
  }
  os << ";P=";
  for (const CommandSpec &spec : prob.P) {
    os << spec.latency << "[";
    for (int resource : spec.resources) {
      os << resource << ",";
    }
    os << "]{";
    for (const auto &resource_entry : spec.resource_windows) {
      os << resource_entry.first << ":";
      for (const auto &window : resource_entry.second) {
        os << window.first << "-" << window.second << ",";
      }
      os << ";";
    }
    os << "};";
  }
  os << ";E=";
  for (const DependencySpec &dependency : prob.dependencies) {
    os << dependency.src << ">" << dependency.dst << ":" << dependency.delta
       << ":" << dependency.lag << ":" << dependency.hazard.size() << ":"
       << dependency.hazard << ":" << dependency.buffer_name.size() << ":"
       << dependency.buffer_name << ",";
  }
  os << ";F=";
  for (const FlowSpec &flow : prob.flows) {
    os << flow.resident << "," << flow.prod << "," << flow.cons << ","
       << flow.delta << "," << flow.mem << "," << flow.buffer_name.size() << ":"
       << flow.buffer_name << "," << flow.fixed_bank << "," << flow.precolor
       << "," << flow.fp << "," << flow.initial_time << "," << flow.w_off << ","
       << flow.w_dur << "," << flow.r_off << "," << flow.r_dur << ","
       << flow.write_resource << "," << flow.read_resource << ";";
  }
  return os.str();
}

struct ExactSolveCache {
  std::mutex mutex;
  std::map<std::string, SolveResult> entries;
};

ExactSolveCache &GetExactSolveCache() {
  static ExactSolveCache cache;
  return cache;
}

bool LookupExactSolveCache(const std::string &key, SolveResult *result) {
  ExactSolveCache &cache = GetExactSolveCache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  auto it = cache.entries.find(key);
  if (it == cache.entries.end()) {
    return false;
  }
  *result = it->second;
  return true;
}

void StoreExactSolveCache(const std::string &key, const SolveResult &result) {
  ExactSolveCache &cache = GetExactSolveCache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  cache.entries[key] = result;
}

HighsInt AddCol(Highs &highs, double lower, double upper, double cost,
                bool is_integer) {
  HighsStatus st = highs.addCol(cost, lower, upper, 0, nullptr, nullptr);
  ICHECK(st == HighsStatus::kOk) << "addCol failed";
  HighsInt col = highs.getNumCol() - 1;
  if (is_integer) {
    highs.changeColIntegrality(col, HighsVarType::kInteger);
  }
  return col;
}

HighsStatus AddRow(Highs &highs, double lower, double upper,
                   const std::vector<HighsInt> &idx,
                   const std::vector<double> &val) {
  const HighsInt *idx_ptr = idx.empty() ? nullptr : idx.data();
  const double *val_ptr = val.empty() ? nullptr : val.data();
  return highs.addRow(lower, upper, HighsInt(idx.size()), idx_ptr, val_ptr);
}

void MergeLinearTerms(const std::vector<HighsInt> &idx,
                      const std::vector<double> &val,
                      std::vector<HighsInt> &merged_idx,
                      std::vector<double> &merged_val) {
  std::map<HighsInt, double> acc;
  for (size_t k = 0; k < idx.size(); ++k) {
    acc[idx[k]] += val[k];
  }
  merged_idx.clear();
  merged_val.clear();
  for (const auto &kv : acc) {
    if (kv.second == 0.0) {
      continue;
    }
    merged_idx.push_back(kv.first);
    merged_val.push_back(kv.second);
  }
}

void AddLeq(Highs &highs, const std::vector<HighsInt> &idx,
            const std::vector<double> &val, double rhs) {
  std::vector<HighsInt> merged_idx;
  std::vector<double> merged_val;
  MergeLinearTerms(idx, val, merged_idx, merged_val);
  if (merged_idx.empty()) {
    if (0.0 > rhs) {
      AddRow(highs, 1.0, 0.0, {}, {});
    }
    return;
  }
  AddRow(highs, -kInf, rhs, merged_idx, merged_val);
}

void AddEq(Highs &highs, const std::vector<HighsInt> &idx,
           const std::vector<double> &val, double rhs) {
  std::vector<HighsInt> merged_idx;
  std::vector<double> merged_val;
  MergeLinearTerms(idx, val, merged_idx, merged_val);
  AddRow(highs, rhs, rhs, merged_idx, merged_val);
}

void AddConditionalParity(Highs &highs, HighsInt z_write, HighsInt z_read,
                          HighsInt write_parity, HighsInt read_parity,
                          HighsInt x_prod, const std::vector<HighsInt> &x_cons,
                          int required_xor) {
  HighsInt quotient = AddCol(highs, 0, 2, 0, true);
  std::vector<HighsInt> idx{z_write,     z_read,   write_parity,
                            read_parity, quotient, x_prod};
  std::vector<double> upper{1.0, 1.0, 1.0, 1.0, -2.0, 4.0};
  std::vector<double> lower{-1.0, -1.0, -1.0, -1.0, 2.0, 4.0};
  for (HighsInt x : x_cons) {
    idx.push_back(x);
    upper.push_back(4.0);
    lower.push_back(4.0);
  }
  AddLeq(highs, idx, upper, required_xor + 8.0);
  AddLeq(highs, idx, lower, -required_xor + 8.0);
}

void AddConditionalBankRelation(Highs &highs, HighsInt z_write, HighsInt z_read,
                                HighsInt x_prod,
                                const std::vector<HighsInt> &x_cons,
                                int required_xor) {
  ICHECK(required_xor == 0 || required_xor == 1);
  std::vector<HighsInt> idx{z_write, z_read, x_prod};
  std::vector<double> first;
  std::vector<double> second;
  double first_rhs = 0.0;
  double second_rhs = 0.0;
  if (required_xor == 0) {
    first = {1.0, -1.0, 1.0};
    second = {-1.0, 1.0, 1.0};
    first_rhs = second_rhs = 2.0;
  } else {
    first = {1.0, 1.0, 1.0};
    second = {-1.0, -1.0, 1.0};
    first_rhs = 3.0;
    second_rhs = 1.0;
  }
  for (HighsInt x : x_cons) {
    idx.push_back(x);
    first.push_back(1.0);
    second.push_back(1.0);
  }
  AddLeq(highs, idx, first, first_rhs);
  AddLeq(highs, idx, second, second_rhs);
}

ModelVars BuildModel(Highs &highs, const Problem &prob, int II, bool optimize_t,
                     int threads, const BankFlipMode &mode) {
  // Variable meanings for command i and modulo slot s:
  //   t[i] = absolute start time in the representative schedule window
  //   m[i] = t[i] mod II, selected by one-hot x[i][s]
  //   y[i] = floor(t[i] / II) = 2*y_half[i] + start_parity[i]
  //   a[i][s] = number of folded copies of command i occupying slot s
  //   z[v] = ping/pong bank phase assigned to data flow v
  //   T = makespan.  Feasibility searches leave T unpriced; the final solve
  //       minimizes it for the already-minimal II.
  // Domains: t_i,y_i,yh_i,m_i,a_is,T are nonnegative integers;
  // x_is,p_i,z_v are binary, and 0 <= m_i < II.
  highs.clear();
  const bool enable_solver_log = GetEnvBool("TL_SUNMMIO_ILP_HIGHS_LOG", false);
  highs.setOptionValue("output_flag", enable_solver_log);
  highs.setOptionValue("log_to_console", enable_solver_log);
  if (enable_solver_log) {
    highs.setOptionValue("mip_report_level", 2);
  }
  highs.setOptionValue("threads", threads);
  highs.setOptionValue("parallel", "on");
  highs.changeObjectiveSense(ObjSense::kMinimize);

  int max_delta = 0;
  int max_latency = 0;
  for (const DependencySpec &dependency : prob.dependencies) {
    max_delta = std::max(max_delta, dependency.delta);
  }
  for (const auto &spec : prob.P) {
    max_latency = std::max(max_latency, spec.latency);
  }
  int time_ub = prob.Tmax + max_delta * II + max_latency;

  ModelVars vars;
  vars.col_t.resize(prob.N);
  vars.col_y.resize(prob.N);
  vars.col_y_half.resize(prob.N);
  vars.col_start_parity.resize(prob.N);
  vars.col_m.resize(prob.N);
  vars.col_x.assign(prob.N, std::vector<HighsInt>(II, -1));

  for (int v = 0; v < static_cast<int>(prob.flows.size()); ++v) {
    vars.internal_flow_ids.push_back(v);
  }

  // t, y, m are exact affine images of the integer y_half/parity/x
  // decisions below.  Keeping these derived columns continuous preserves the
  // same integer schedules while removing redundant branch-and-bound columns.
  for (int i = 0; i < prob.N; ++i) {
    vars.col_t[i] = AddCol(highs, 0, time_ub, 0, false);
    vars.col_y[i] = AddCol(highs, 0, time_ub, 0, false);
    vars.col_y_half[i] = AddCol(highs, 0, CeilDiv(time_ub, 2), 0, true);
    vars.col_start_parity[i] = AddCol(highs, 0, 1, 0, true);
    vars.col_m[i] = AddCol(highs, 0, II - 1, 0, false);
  }
  for (int i = 0; i < prob.N; ++i) {
    for (int s = 0; s < II; ++s) {
      vars.col_x[i][s] = AddCol(highs, 0, 1, 0, true);
    }
  }
  for (int i = 0; i < prob.N; ++i) {
    for (int resource : prob.P[i].resources) {
      int ub = 0;
      for (int s = 0; s < II; ++s) {
        ub = std::max(
            ub, ComputeFoldedResourceOccupancy(prob.P[i], resource, 0, II, s));
      }
      auto &columns = vars.col_resource_a[{i, resource}];
      columns.resize(II, -1);
      for (int s = 0; s < II; ++s) {
        // Resource occupancy is fixed by an equality to an
        // integer-weighted sum of the binary x[i,*] columns.
        columns[s] = AddCol(highs, 0, ub, 0, false);
      }
    }
  }
  // T is bounded below by integer completion times.  A continuous T has the
  // same exact optimum and avoids another redundant integer column.
  vars.col_T = AddCol(highs, 0, time_ub, optimize_t ? 1.0 : 0.0, false);

  // Choose exactly one modulo start slot and link all representations of the
  // same start time.  t = II*y + m linearizes modulo arithmetic; splitting y
  // into 2*y_half + parity exposes whether bank rotation has crossed an odd
  // number of initiation intervals without introducing nonlinear arithmetic.
  //   sum_s x[i,s] = 1
  //   m[i] = sum_s s*x[i,s]
  //   t[i] = II*y[i] + m[i]
  //   y[i] = 2*y_half[i] + start_parity[i]
  for (int i = 0; i < prob.N; ++i) {
    std::vector<HighsInt> idx1;
    std::vector<double> val1;
    for (int s = 0; s < II; ++s) {
      idx1.push_back(vars.col_x[i][s]);
      val1.push_back(1.0);
    }
    AddEq(highs, idx1, val1, 1.0);

    std::vector<HighsInt> idx2{vars.col_m[i]};
    std::vector<double> val2{1.0};
    for (int s = 0; s < II; ++s) {
      idx2.push_back(vars.col_x[i][s]);
      val2.push_back(-double(s));
    }
    AddEq(highs, idx2, val2, 0.0);

    AddEq(highs, {vars.col_t[i], vars.col_y[i], vars.col_m[i]},
          {1.0, -double(II), -1.0}, 0.0);
    AddEq(highs, {vars.col_y[i], vars.col_y_half[i], vars.col_start_parity[i]},
          {1.0, -2.0, -1.0}, 0.0);
  }

  // Fold every command/resource occupancy window onto the cyclic II-slot
  // calendar.  Normal commands use [0, latency); a nested summary can provide
  // several narrower windows for each physical resource.
  for (int i = 0; i < prob.N; ++i) {
    for (int resource : prob.P[i].resources) {
      const auto &columns = vars.col_resource_a.at({i, resource});
      for (int s = 0; s < II; ++s) {
        std::vector<HighsInt> idx{columns[s]};
        std::vector<double> val{1.0};
        for (int st = 0; st < II; ++st) {
          int count =
              ComputeFoldedResourceOccupancy(prob.P[i], resource, st, II, s);
          if (count != 0) {
            idx.push_back(vars.col_x[i][st]);
            val.push_back(-double(count));
          }
        }
        AddEq(highs, idx, val, 0.0);
      }
    }
  }

  // For dependency i -> j at iteration distance delta, require
  // t[i] + lag <= t[j] + delta*II.  Ordinary commands use lag=latency[i];
  // nested summaries can use producer-ready and consumer-need offsets.
  for (const DependencySpec &dependency : prob.dependencies) {
    int i = dependency.src;
    int j = dependency.dst;
    int lag = dependency.lag;
    if (i == j) {
      if (lag > dependency.delta * II) {
        AddRow(highs, 1.0, 0.0, {}, {});
      }
      continue;
    }
    AddLeq(highs, {vars.col_t[i], vars.col_t[j]}, {1.0, -1.0},
           double(dependency.delta * II - lag));
  }

  // T bounds every command completion, making minimization of T equivalent to
  // minimizing the representative window's makespan.
  //   T >= t[i] + d_i, for every command i; objective: min T.
  for (int i = 0; i < prob.N; ++i) {
    AddLeq(highs, {vars.col_t[i], vars.col_T}, {1.0, -1.0},
           double(-prob.P[i].latency));
  }

  // At every modulo slot, sum the folded occupancies of commands using a
  // physical execution resource.  This models conflicts across all overlapped
  // iterations, not merely commands visible in one template iteration.
  //   sum_{i uses r} a[i,s] <= cap[r], for every resource r and slot s.
  for (int r : prob.R) {
    int cap = prob.cap.count(r) ? prob.cap.at(r) : 1;
    for (int s = 0; s < II; ++s) {
      std::vector<HighsInt> idx;
      std::vector<double> val;
      for (int i = 0; i < prob.N; ++i) {
        if (std::find(prob.P[i].resources.begin(), prob.P[i].resources.end(),
                      r) != prob.P[i].resources.end()) {
          idx.push_back(vars.col_resource_a.at({i, r})[s]);
          val.push_back(1.0);
        }
      }
      AddLeq(highs, idx, val, double(cap));
    }
  }

  const int internal_count = static_cast<int>(vars.internal_flow_ids.size());
  std::cerr << "[ILP] bank_var_count new=" << internal_count * 2 << "\n";

  vars.col_z.resize(internal_count);
  for (int vv = 0; vv < internal_count; ++vv) {
    vars.col_z[vv] = AddCol(highs, 0, 1, 0, true);
  }
  if (internal_count > 0) {
    // Every bank constraint depends only on equality/XOR relations between
    // flow phases.  Complementing every z therefore produces the same
    // physical schedule with ping/pong labels exchanged.  Select one
    // representative of this exact two-way symmetry.
    AddEq(highs, {vars.col_z[0]}, {1.0}, 0.0);
  }

  {
    // Multiple FlowSpecs can describe reads fed by the same physical write.
    // They must share one write-bank phase; allowing independent z values
    // would assign the same produced data to two banks simultaneously.
    //   z[v] = z[w] for flows v,w with (prod[v], mem[v]) equal.
    std::map<SameWriteFlowKey, std::vector<int>> same_write_groups;
    for (int vv = 0; vv < internal_count; ++vv) {
      int fid = vars.internal_flow_ids[vv];
      const FlowSpec &flow = prob.flows[fid];
      if (flow.write_resource < 0 || flow.prod < 0)
        continue;
      same_write_groups[MakeSameWriteFlowKey(flow)].push_back(vv);
    }
    for (const auto &kv : same_write_groups) {
      const std::vector<int> &flows = kv.second;
      for (size_t i = 1; i < flows.size(); ++i) {
        AddEq(highs, {vars.col_z[flows[0]], vars.col_z[flows[i]]}, {1.0, -1.0},
              0.0);
      }
    }
  }

  // Precoloring captures bank identities already implied by resident buffers
  // or frontend metadata.  Convert pairwise implications into equality or XOR
  // constraints before considering time-dependent write/read overlap.
  //   NeedSame: z[a] - z[b] = 0; NeedDifferent: z[a] + z[b] = 1.
  for (int a = 0; a < internal_count; ++a) {
    const FlowSpec &lhs = prob.flows[vars.internal_flow_ids[a]];
    for (int b = a + 1; b < internal_count; ++b) {
      const FlowSpec &rhs = prob.flows[vars.internal_flow_ids[b]];
      ConflictType conflict = AnalyzePrecolorConflict(lhs, rhs);
      if (conflict == ConflictType::kNeedSame) {
        AddEq(highs, {vars.col_z[a], vars.col_z[b]}, {1.0, -1.0}, 0.0);
      } else if (conflict == ConflictType::kNeedDifferent) {
        AddEq(highs, {vars.col_z[a], vars.col_z[b]}, {1.0, 1.0}, 1.0);
      }
    }
  }

  auto bank_build_begin = std::chrono::steady_clock::now();
  // Compare every SRAM write lifetime with every read lifetime in the same
  // memory.  For each possible producer modulo slot, classify consumer slots:
  // overlapping accesses may require equal banks, different banks, or may be
  // impossible regardless of bank assignment.  When banks rotate across IIs,
  // the effective bank is z XOR start_parity, hence AddConditionalParity;
  // otherwise z alone determines the bank.  The x variables gate each relation
  // so it is active only for the pair of start slots selected by the solver.
  // If x[prod,sp] = x[cons,sc] = 1, enforce
  //   z_write XOR z_read XOR parity_write XOR parity_read = required_xor
  // for rotating banks, or z_write XOR z_read = required_xor otherwise.
  // AddConditionalParity introduces an integer quotient to linearize the XOR;
  // its big-M terms deactivate the equality for unselected slot pairs.
  for (int a = 0; a < internal_count; ++a) {
    const auto &write_flow = prob.flows[vars.internal_flow_ids[a]];
    if (write_flow.write_resource < 0 || write_flow.prod < 0)
      continue;
    for (int b = 0; b < internal_count; ++b) {
      if (a == b)
        continue;
      const auto &read_flow = prob.flows[vars.internal_flow_ids[b]];
      if (read_flow.read_resource < 0)
        continue;
      if (write_flow.mem != read_flow.mem)
        continue;
      bool flip = mode.FlipForMem(write_flow.mem);

      for (int prod_slot = 0; prod_slot < II; ++prod_slot) {
        std::vector<int> diff_cons_slots;
        std::vector<int> same_cons_slots;
        std::vector<int> impossible_cons_slots;
        for (int cons_slot = 0; cons_slot < II; ++cons_slot) {
          ConflictType conflict = AnalyzeWriteReadConflict(
              PositiveMod(prod_slot + write_flow.w_off, II), write_flow.w_dur,
              (prod_slot + write_flow.w_off) / II & 1,
              PositiveMod(cons_slot + read_flow.delta * II + read_flow.r_off,
                          II),
              read_flow.r_dur,
              (cons_slot + read_flow.delta * II + read_flow.r_off) / II & 1, II,
              flip);
          if (conflict == ConflictType::kNone)
            continue;
          if (conflict == ConflictType::kNeedDifferent) {
            diff_cons_slots.push_back(cons_slot);
          } else if (conflict == ConflictType::kNeedSame) {
            same_cons_slots.push_back(cons_slot);
          } else if (conflict == ConflictType::kImpossible) {
            impossible_cons_slots.push_back(cons_slot);
          }
        }

        HighsInt x_prod = vars.col_x[write_flow.prod][prod_slot];
        HighsInt z_write_ping = vars.col_z[a];
        HighsInt z_read_ping = vars.col_z[b];
        HighsInt write_parity = vars.col_start_parity[write_flow.prod];
        HighsInt read_parity = vars.col_start_parity[read_flow.cons];

        if (!diff_cons_slots.empty()) {
          std::vector<HighsInt> x_cons;
          x_cons.reserve(diff_cons_slots.size());
          for (int cons_slot : diff_cons_slots) {
            x_cons.push_back(vars.col_x[read_flow.cons][cons_slot]);
          }
          if (flip) {
            AddConditionalParity(highs, z_write_ping, z_read_ping, write_parity,
                                 read_parity, x_prod, x_cons, 1);
          } else {
            AddConditionalBankRelation(highs, z_write_ping, z_read_ping, x_prod,
                                       x_cons, 1);
          }
        }

        if (!same_cons_slots.empty()) {
          std::vector<HighsInt> x_cons;
          x_cons.reserve(same_cons_slots.size());
          for (int cons_slot : same_cons_slots) {
            x_cons.push_back(vars.col_x[read_flow.cons][cons_slot]);
          }
          if (flip) {
            AddConditionalParity(highs, z_write_ping, z_read_ping, write_parity,
                                 read_parity, x_prod, x_cons, 0);
          } else {
            AddConditionalBankRelation(highs, z_write_ping, z_read_ping, x_prod,
                                       x_cons, 0);
          }
        }

        if (!impossible_cons_slots.empty()) {
          // Forbid selecting the producer slot together with any consumer slot
          // whose physical intervals conflict even when placed on opposite
          // banks.  Since each command has one-hot x, this single inequality
          // excludes every impossible pairing collected above.
          //   x[prod,sp] + sum_{sc impossible} x[cons,sc] <= 1.
          std::vector<HighsInt> idx;
          std::vector<double> val;
          if (x_prod >= 0) {
            idx.push_back(x_prod);
            val.push_back(1.0);
          }
          for (int cons_slot : impossible_cons_slots) {
            idx.push_back(vars.col_x[read_flow.cons][cons_slot]);
            val.push_back(1.0);
          }
          AddLeq(highs, idx, val, x_prod >= 0 ? 1.0 : 0.0);
        }
      }
    }
  }
  auto bank_build_end = std::chrono::steady_clock::now();
  double bank_build_elapsed =
      std::chrono::duration<double>(bank_build_end - bank_build_begin).count();
  LOG(INFO) << "[ILP] bank_constraint_build_elapsed=" << bank_build_elapsed
            << "s";

  return vars;
}

void ApplyMipStart(Highs &highs, const Problem &prob, int II,
                   const BankFlipMode &mode, const ModelVars &vars,
                   const SolveResult *mip_start) {
  if (mip_start == nullptr || !mip_start->ok || mip_start->II != II ||
      mip_start->bank_flip_mode.Id() != mode.Id() ||
      static_cast<int>(mip_start->t.size()) != prob.N ||
      static_cast<int>(mip_start->z_bank.size()) !=
          static_cast<int>(vars.internal_flow_ids.size())) {
    return;
  }

  std::vector<HighsInt> index;
  std::vector<double> value;
  auto add = [&](HighsInt col, double val) {
    if (col >= 0) {
      index.push_back(col);
      value.push_back(val);
    }
  };

  int makespan = 0;
  for (int i = 0; i < prob.N; ++i) {
    int start = mip_start->t[i];
    if (start < 0) {
      return;
    }
    int slot = PositiveMod(start, II);
    int iteration = start / II;
    add(vars.col_t[i], start);
    add(vars.col_y[i], iteration);
    add(vars.col_y_half[i], iteration / 2);
    add(vars.col_start_parity[i], iteration & 1);
    add(vars.col_m[i], slot);
    for (int s = 0; s < II; ++s) {
      add(vars.col_x[i][s], s == slot ? 1.0 : 0.0);
      for (int resource : prob.P[i].resources) {
        add(vars.col_resource_a.at({i, resource})[s],
            ComputeFoldedResourceOccupancy(prob.P[i], resource, start, II, s));
      }
    }
    makespan = std::max(makespan, start + prob.P[i].latency);
  }
  add(vars.col_T, makespan);
  int bank_complement =
      mip_start->z_bank.empty() ? 0 : (mip_start->z_bank[0] & 1);
  for (int vv = 0; vv < static_cast<int>(vars.col_z.size()); ++vv) {
    add(vars.col_z[vv], (mip_start->z_bank[vv] & 1) ^ bank_complement);
  }

  HighsStatus status =
      highs.setSolution(HighsInt(index.size()), index.data(), value.data());
  if (status == HighsStatus::kOk) {
    LOG(INFO) << "[ILP] applied exact feasible MIP start for II=" << II
              << " entries=" << index.size();
  } else {
    // A rejected start only disables the performance hint; the exact model
    // and its optimality proof are unchanged.
    LOG(WARNING) << "[ILP] HiGHS rejected MIP start for II=" << II;
  }
}

SolveResult SolveFixedII(const Problem &prob, int II, bool optimize_t,
                         int threads, const BankFlipMode &mode,
                         const SolveResult *mip_start = nullptr) {
  bool model_vc_blocking_issue =
      tvm::transform::PassContext::Current()
          ->GetConfig<Bool>(tl::kSunmmioILPModelVCBlockingIssue, Bool(true))
          .value();
  std::string cache_key =
      MakeSolveCacheKey(prob, II, optimize_t, mode, model_vc_blocking_issue);
  SolveResult cached;
  if (LookupExactSolveCache(cache_key, &cached)) {
    LOG(INFO) << "[ILP] exact cache hit II=" << II
              << " optimize_t=" << optimize_t << " mode=" << mode.Id()
              << " feasible=" << cached.ok;
    return cached;
  }

  auto solve_begin = std::chrono::steady_clock::now();
  LOG(INFO) << "[ILP] start solve II=" << II << " optimize_t=" << optimize_t
            << " N=" << prob.N << " edges=" << prob.dependencies.size()
            << " flows=" << prob.flows.size();
  Highs highs;
  ModelVars vars = BuildModel(highs, prob, II, optimize_t, threads, mode);
  int vc_issue_constraint_count = 0;
  if (model_vc_blocking_issue) {
    std::vector<int> vc_commands;
    std::unordered_set<int> vc_set;
    for (int cmd = 0; cmd < prob.N; ++cmd) {
      if (CommandUsesResource(prob.P[cmd],
                              static_cast<int>(IlpResourceType::kVectorCore))) {
        vc_commands.push_back(cmd);
        vc_set.insert(cmd);
      }
    }
    if (!vc_commands.empty()) {
      for (int slot = 0; slot < II; ++slot) {
        std::vector<HighsInt> idx;
        std::vector<double> val;
        for (int cmd = 0; cmd < prob.N; ++cmd) {
          idx.push_back(vars.col_x[cmd][slot]);
          val.push_back(vc_set.count(cmd) ? 1.0 - double(prob.N) : 1.0);
        }
        for (int vc : vc_commands) {
          idx.push_back(vars.col_resource_a.at(
              {vc, static_cast<int>(IlpResourceType::kVectorCore)})[slot]);
          val.push_back(double(prob.N));
        }
        AddLeq(highs, idx, val, double(prob.N));
        ++vc_issue_constraint_count;
      }
    }
  }
  LOG(INFO) << "[II=" << II
            << "] VC blocking-issue constraints=" << vc_issue_constraint_count;
  ApplyMipStart(highs, prob, II, mode, vars, mip_start);
  highs.run();
  if (highs.getModelStatus() != HighsModelStatus::kOptimal) {
    auto solve_end = std::chrono::steady_clock::now();
    double elapsed =
        std::chrono::duration<double>(solve_end - solve_begin).count();
    LOG(INFO) << "[II=" << II << "] infeasible/failed, elapsed=" << elapsed
              << "s, status=" << int(highs.getModelStatus());
    if (highs.getModelStatus() == HighsModelStatus::kInfeasible) {
      StoreExactSolveCache(cache_key, {});
    }
    return {};
  }

  const HighsSolution &sol = highs.getSolution();
  SolveResult res;
  res.ok = true;
  res.objective_optimized = optimize_t;
  res.II = II;
  res.bank_slot_period = 2 * II;
  res.bank_flip_mode = mode;
  res.vc_blocking_issue_modeled = model_vc_blocking_issue;
  res.vc_blocking_issue_constraints = vc_issue_constraint_count;
  res.t.resize(prob.N);
  res.m.resize(prob.N);
  res.y.resize(prob.N);
  int computed_makespan = 0;
  for (int i = 0; i < prob.N; ++i) {
    res.t[i] = int(std::llround(sol.col_value[vars.col_t[i]]));
    res.m[i] = int(std::llround(sol.col_value[vars.col_m[i]]));
    res.y[i] = int(std::llround(sol.col_value[vars.col_y[i]]));
    computed_makespan =
        std::max(computed_makespan, res.t[i] + prob.P[i].latency);
  }
  res.makespan = optimize_t ? int(std::llround(sol.col_value[vars.col_T]))
                            : computed_makespan;
  res.internal_flow_ids = vars.internal_flow_ids;
  res.z_bank.resize(vars.internal_flow_ids.size(), 0);
  for (int vv = 0; vv < static_cast<int>(vars.internal_flow_ids.size()); ++vv) {
    res.z_bank[vv] = int(std::llround(sol.col_value[vars.col_z[vv]]));
  }
  StoreExactSolveCache(cache_key, res);
  auto solve_end = std::chrono::steady_clock::now();
  double elapsed =
      std::chrono::duration<double>(solve_end - solve_begin).count();
  LOG(INFO) << "[II=" << II << "] "
            << (optimize_t ? "solve_elapsed=" : "feasible_only_elapsed=")
            << elapsed << "s";
  return res;
}

SolveResult FindMinimalIIForMode(const Problem &prob, int threads,
                                 const BankFlipMode &mode, int max_ii = -1) {
  int lb = std::max(1, ResourceLowerBound(prob));
  int search_begin = std::max(1, lb);
  int search_end = std::max(search_begin, std::max(1, prob.Tmax));
  if (max_ii > 0) {
    search_end = std::min(search_end, max_ii);
  }
  if (search_end < search_begin) {
    return {};
  }
  if (GetEnvBool("TL_SUNMMIO_ILP_TRY_RESOURCE_LB_FIRST", false)) {
    LOG(INFO) << "[ILP] try exact optimized resource lower bound II="
              << search_begin;
    SolveResult lower_bound =
        SolveFixedII(prob, search_begin, true, threads, mode);
    if (lower_bound.ok) {
      LOG(INFO) << "[ILP] resource lower bound is feasible and globally "
                   "minimal II="
                << search_begin;
      return lower_bound;
    }
    // This exact solve proved the lower bound infeasible.  Do not ask a
    // feasibility-only model to prove the same fact again.
    ++search_begin;
    if (search_begin > search_end) {
      return {};
    }
  }
  constexpr int kInitialWindowSpan = 10;
  int best_ii = -1;
  SolveResult best_feasible;
  LOG(INFO) << "[ILP] search start=" << search_begin << " end=" << search_end
            << " lb=" << lb << " initial_window_span=" << kInitialWindowSpan;

  for (int window_l = search_begin; window_l <= search_end;
       window_l += kInitialWindowSpan + 1) {
    int window_r = std::min(search_end, window_l + kInitialWindowSpan);
    int l = window_l;
    int r = window_r;
    int window_best = -1;
    LOG(INFO) << "[ILP] search window l=" << window_l << " r=" << window_r;
    while (l <= r) {
      int mid = (l + r) / 2;
      LOG(INFO) << "[ILP] try feasible-only II=" << mid;
      SolveResult feas = SolveFixedII(prob, mid, false, threads, mode);
      if (feas.ok) {
        window_best = mid;
        best_feasible = std::move(feas);
        LOG(INFO) << "[ILP] feasible II=" << mid;
        r = mid - 1;
      } else {
        LOG(INFO) << "[ILP] infeasible II=" << mid;
        l = mid + 1;
      }
    }
    if (window_best >= 0) {
      best_ii = window_best;
      break;
    }
  }
  if (best_ii < 0) {
    LOG(INFO) << "[ILP] no feasible II found";
    return {};
  }
  LOG(INFO) << "[ILP] best feasible II=" << best_ii << " mode=" << mode.Id();
  ICHECK(best_feasible.ok && best_feasible.II == best_ii);
  return best_feasible;
}

BankFlipMode GuessBankFlipMode(const Problem &prob) {
  auto guess_for_mem = [&](int mem) {
    int write_resource = GetMemoryWriteResource(mem);
    int read_resource = GetMemoryReadResource(mem);
    int write_count = 0;
    int read_count = 0;
    int write_time = 0;
    int read_time = 0;
    for (const CommandSpec &spec : prob.P) {
      if (CommandUsesResource(spec, write_resource)) {
        ++write_count;
        write_time += ResourceUsageDuration(spec, write_resource);
      }
      if (CommandUsesResource(spec, read_resource)) {
        ++read_count;
        read_time += ResourceUsageDuration(spec, read_resource);
      }
    }
    // Bank rotation normally follows the input-side command parity.  Only let
    // output-side parity decide when output occupancy clearly dominates input.
    constexpr int kOutDominanceRatio = 2;
    bool use_output =
        read_time > kOutDominanceRatio * static_cast<long long>(write_time);
    return ((use_output ? read_count : write_count) & 1) != 0;
  };
  return {guess_for_mem(/*mem=*/0), guess_for_mem(/*mem=*/1)};
}

// The analysis set is an identity-membership set, not a stable serialization
// order: ObjectRef::operator< compares heap addresses.  Export only the required
// identities in deterministic function/AST order, without changing that set or
// the solver.  In particular, do not merge different views sharing one data Var.
class StableUsedBufferExporter : public StmtExprVisitor {
public:
  static Optional<Array<Buffer>> Export(const PrimFunc &func,
                                       const std::set<Buffer> &required) {
    StableUsedBufferExporter exporter(required);
    // buffer_map is a Map; its iteration order is not part of this proof.
    for (const Var &param : func->params) {
      auto it = func->buffer_map.find(param);
      if (it != func->buffer_map.end()) {
        exporter.Record((*it).second);
      }
    }
    exporter.VisitStmt(func->body);
    if (exporter.seen_.size() != required.size()) {
      return {};  // Never recover unknown provenance using heap order.
    }
    return exporter.ordered_;
  }

private:
  explicit StableUsedBufferExporter(const std::set<Buffer> &required)
      : required_(required) {}

  void Record(const Buffer &buffer) {
    if (required_.count(buffer) && seen_.insert(buffer.get()).second) {
      ordered_.push_back(buffer);
    }
  }

  void VisitStmt_(const BlockNode *op) final {
    // These are ordered semantic fields, not the unordered annotations Map.
    for (const Buffer &buffer : op->alloc_buffers) {
      Record(buffer);
    }
    for (const MatchBufferRegion &match : op->match_buffers) {
      Record(match->buffer);
      Record(match->source->buffer);
    }
    for (const BufferRegion &region : op->reads) {
      Record(region->buffer);
    }
    for (const BufferRegion &region : op->writes) {
      Record(region->buffer);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const DeclBufferNode *op) final {
    Record(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferRealizeNode *op) final {
    Record(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    Record(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
    // This branch's default visitor does not traverse access predicates.
    if (op->predicate.defined()) {
      VisitExpr(op->predicate.value());
    }
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    Record(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
    if (op->predicate.defined()) {
      VisitExpr(op->predicate.value());
    }
  }

  // The ordinary visitor traverses Call args and statement/loop expressions in
  // program order, but not Call/For/Block annotations or AttrStmt.node.  A Buffer
  // reachable only through such metadata intentionally remains unproven.
  const std::set<Buffer> &required_;
  std::unordered_set<const BufferNode *> seen_;
  Array<Buffer> ordered_;
};

// Converts each annotated SunMMIO pipeline loop in a PrimFunc into a periodic
// scheduling Problem, solves it with HiGHS, and returns the same TIR loop with
// prologue/body/epilogue order plus multiversion and bank-phase annotations for
// the downstream injection pass.  Unsupported or infeasible loops are returned
// with an explicit pipeline-fallback diagnostic instead of a partial schedule.
class SunmmioPipelinePlannerILP : public StmtExprMutator {
public:
  static Stmt Substitute(const PrimFunc &f, bool debug) {
    SunmmioPipelinePlannerILP planner(f, debug);
    return planner.VisitStmt(f->body);
  }

private:
  SunmmioPipelinePlannerILP(const PrimFunc &f, bool debug)
      : func_(f), traverser_(f), debug_(debug) {}

  Optional<For> FindPipelineLoop(const Stmt &stmt) {
    Optional<For> result;
    PostOrderVisit(stmt, [&](const ObjectRef &obj) {
      if (result.defined()) {
        return;
      }
      if (const auto *loop = obj.as<ForNode>()) {
        if (loop->annotations.find("num_stages") != loop->annotations.end()) {
          result = ffi::GetRef<For>(loop);
        }
      }
    });
    return result;
  }

  bool GetPipelineBodyStatements(const For &loop,
                                 Array<Stmt> *statements) const {
    Stmt current = loop->body;
    if (const auto *realize = current.as<BlockRealizeNode>()) {
      current = realize->block->body;
    }
    while (true) {
      if (const auto *seq = current.as<SeqStmtNode>()) {
        *statements = seq->seq;
        return true;
      }
      if (const auto *if_node = current.as<IfThenElseNode>()) {
        if (if_node->else_case.defined()) {
          return false;
        }
        current = if_node->then_case;
        continue;
      }
      if (const auto *let_node = current.as<LetStmtNode>()) {
        current = let_node->body;
        continue;
      }
      *statements = Array<Stmt>{current};
      return true;
    }
  }

  class PipelineNestInspector : public StmtVisitor {
  public:
    void Inspect(const Stmt &stmt) { VisitStmt(stmt); }

    int max_depth{0};
    bool dynamic_extent{false};
    bool unsupported_kind{false};
    bool indirect_nesting{false};

  private:
    void VisitStmt_(const ForNode *op) final {
      const bool pipeline = op->annotations.count("num_stages");
      if (pipeline) {
        if (pipeline_depth_ > 0 && barrier_depth_ > 0) {
          indirect_nesting = true;
        }
        ++pipeline_depth_;
        max_depth = std::max(max_depth, pipeline_depth_);
        arith::Analyzer analyzer;
        PrimExpr extent = analyzer.Simplify(op->extent);
        if (!extent.as<IntImmNode>()) {
          dynamic_extent = true;
        }
        if (op->kind != ForKind::kSerial && op->kind != ForKind::kUnrolled) {
          unsupported_kind = true;
        }
        StmtVisitor::VisitStmt_(op);
        --pipeline_depth_;
        return;
      }
      const bool separates_pipeline_loops = pipeline_depth_ > 0;
      if (separates_pipeline_loops) {
        ++barrier_depth_;
      }
      StmtVisitor::VisitStmt_(op);
      if (separates_pipeline_loops) {
        --barrier_depth_;
      }
    }

    void VisitStmt_(const IfThenElseNode *op) final {
      const bool separates_pipeline_loops = pipeline_depth_ > 0;
      if (separates_pipeline_loops) {
        ++barrier_depth_;
      }
      StmtVisitor::VisitStmt_(op);
      if (separates_pipeline_loops) {
        --barrier_depth_;
      }
    }

    int pipeline_depth_{0};
    int barrier_depth_{0};
  };

  struct IlpLoopAnalysis {
    Problem prob;
    std::vector<TemplateCommand> commands;
    std::set<Buffer> used_buffers;
    std::vector<Buffer> versioned_buffers;
    std::vector<Buffer> runtime_multiversion_buffers;
    std::vector<Buffer> runtime_banked_buffers;
    std::vector<Buffer> runtime_resident_banked_buffers;
    std::map<std::string, int> runtime_bank_start_phases;
    std::map<std::string, int> runtime_bank_read_delta_parities;
    std::map<std::string, std::map<int, int>> runtime_bank_writer_phases;
    std::map<std::string, std::map<int, int>> runtime_bank_reader_phases;
    std::map<std::string, int> runtime_bank_flip_modes;
    int iterations{0};
    bool graph_valid{false};
    bool summary_v2_applied{false};
    bool summary_data_frontiers_applied{false};
  };

  bool ShouldEnableNestedPipeline() const {
    auto pass_ctx = tvm::transform::PassContext::Current();
    auto cfg =
        pass_ctx->GetConfig<Bool>(tl::kSunmmioILPNestedPipeline, Bool(false));
    return cfg.value()->value;
  }

  bool ShouldEnableNestedSummaryV2() const {
    auto pass_ctx = tvm::transform::PassContext::Current();
    auto cfg =
        pass_ctx->GetConfig<Bool>(tl::kSunmmioILPNestedSummaryV2, Bool(false));
    return cfg.value()->value;
  }

  For StripNumStages(const For &loop) const {
    Map<String, Any> annotations = loop->annotations;
    annotations.erase("num_stages");
    For result = loop;
    result.CopyOnWrite()->annotations = annotations;
    return result;
  }

  Optional<String> FindNestedChildFailure(const Stmt &body) const {
    Optional<String> failure;
    PostOrderVisit(body, [&](const ObjectRef &obj) {
      if (failure) {
        return;
      }
      const auto *child = obj.as<ForNode>();
      if (child == nullptr || !child->annotations.count(kPipelineRequested)) {
        return;
      }
      auto applied = child->annotations.Get(kPipelineApplied);
      if (!applied || !Downcast<Bool>(applied.value())->value) {
        auto reason = child->annotations.Get(kPipelineFallbackReason);
        failure = reason ? Downcast<String>(reason.value())
                         : String("child_not_applied");
        return;
      }
      if (!child->annotations.count(
              "tl.sunmmio.pipeline.summary_raw_makespan")) {
        failure = String("missing_child_summary");
      }
    });
    return failure;
  }

  Optional<String>
  ValidateNestedSummaryV2Inputs(const Array<Stmt> &pipeline_body) const {
    if (!ShouldEnableNestedSummaryV2()) {
      return {};
    }
    for (const Stmt &stmt : pipeline_body) {
      const auto *child = stmt.as<ForNode>();
      if (child == nullptr || !child->annotations.count(
                                  "tl.sunmmio.pipeline.summary_raw_makespan")) {
        continue;
      }
      auto observed =
          child->annotations.Get("tl.sunmmio.pipeline.summary_v2_observed");
      if (!observed || !Downcast<Bool>(observed.value())->value) {
        return String("nested_summary_v2_not_observed");
      }
      if (!child->annotations.count(
              "tl.sunmmio.pipeline.summary_resource_windows_raw")) {
        return String("nested_summary_v2_missing_resource_windows");
      }
      if (!child->annotations.count(
              "tl.sunmmio.pipeline.summary_buffer_frontiers_raw")) {
        return String("nested_summary_v2_missing_buffer_frontiers");
      }
    }
    return {};
  }

  Optional<String>
  FindNestedSharedBuffers(const For &loop,
                          std::vector<Buffer> *parent_owned_buffers) const {
    Array<Stmt> body;
    if (!GetPipelineBodyStatements(loop, &body)) {
      return {};
    }

    std::set<Buffer> child_buffers;
    std::set<Buffer> child_owned_buffers;
    std::set<Buffer> outside_buffers;
    std::set<Buffer> shared_buffers;
    SunmmioStmtAccessAnalyzer access_analyzer(func_);
    for (const Stmt &stmt : body) {
      const auto *child = stmt.as<ForNode>();
      const bool is_nested_child =
          child != nullptr &&
          child->annotations.count("tl.sunmmio.pipeline.summary_raw_makespan");
      if (is_nested_child) {
        auto used_buffers = child->annotations.Get("used_buffers");
        if (!used_buffers) {
          continue;
        }
        for (const Buffer &buffer :
             Downcast<Array<Buffer>>(used_buffers.value())) {
          if (child_buffers.count(buffer)) {
            shared_buffers.insert(buffer);
          }
          child_buffers.insert(buffer);
        }
        for (const char *key :
             {"versioned_buffers", "runtime_multiversion_buffers",
              "runtime_banked_buffers"}) {
          auto owned_buffers = child->annotations.Get(key);
          if (!owned_buffers) {
            continue;
          }
          for (const Buffer &buffer :
               Downcast<Array<Buffer>>(owned_buffers.value())) {
            child_owned_buffers.insert(buffer);
          }
        }
        continue;
      }

      for (const AccessInfo &access :
           access_analyzer.Collect(stmt, loop->loop_var)) {
        Buffer buffer = access.buffer();
        if (!IsGlobalBuffer(buffer)) {
          outside_buffers.insert(buffer);
        }
      }
    }

    for (const Buffer &buffer : child_buffers) {
      if (outside_buffers.count(buffer) &&
          child_owned_buffers.count(buffer) == 0) {
        if (std::find(parent_owned_buffers->begin(),
                      parent_owned_buffers->end(),
                      buffer) == parent_owned_buffers->end()) {
          parent_owned_buffers->push_back(buffer);
        }
      } else if (outside_buffers.count(buffer)) {
        shared_buffers.insert(buffer);
      }
    }
    if (shared_buffers.empty()) {
      return {};
    }

    std::set<std::string> names;
    for (const Buffer &buffer : shared_buffers) {
      names.insert(buffer->name);
    }
    std::string detail;
    for (const std::string &name : names) {
      detail += detail.empty() ? name : "," + name;
    }
    return String(detail);
  }

  IlpLoopAnalysis AnalyzeLoop(const For &loop, const Array<Stmt> &pipeline_body,
                              int forced_iterations = -1,
                              const BankFlipMode &mode = BankFlipMode{}) {
    IlpLoopAnalysis result;
    bool export_only = GetEnvBool("TL_SUNMMIO_ILP_EXPORT_ONLY", false);
    int num_stages = -1;
    auto it = loop->annotations.find("num_stages");
    ICHECK(it != loop->annotations.end());
    const auto &any_ref = (*it).second;
    if (const auto *imm = any_ref.as<IntImmNode>()) {
      num_stages = imm->value;
    }
    ICHECK_GT(num_stages, 0);
    if (forced_iterations > 0) {
      num_stages = forced_iterations;
    }
    result.iterations = num_stages;

    ASTTraverser traverser(func_);
    SunmmioRoleMarker role_marker(traverser, func_);
    SunmmioStmtAccessAnalyzer access_analyzer(func_);
    CostModelContext cost_context = CostModel::BuildContext(func_);
    const bool gem5_profile =
        cost_context.options.profile == CostModelProfile::kGem5A4E2026;
    result.commands.reserve(pipeline_body.size());

    const bool multidim =
        loop->annotations.find("tl.sunmmio.multidim.region") !=
        loop->annotations.end();
    int multidim_inner_min = 0;
    int multidim_inner_extent = 0;
    if (multidim) {
      auto min = loop->annotations.Get("tl.sunmmio.multidim.inner_min");
      auto extent = loop->annotations.Get("tl.sunmmio.multidim.inner_extent");
      ICHECK(min.has_value() && extent.has_value());
      IntImm min_imm = Downcast<IntImm>(min.value());
      IntImm extent_imm = Downcast<IntImm>(extent.value());
      multidim_inner_min = static_cast<int>(min_imm->value);
      multidim_inner_extent = static_cast<int>(extent_imm->value);
    }

    Array<Buffer> parent_owned_buffers;
    std::unordered_set<const BufferNode *> parent_owned;
    auto parent_owned_buffers_anno =
        loop->annotations.Get("tl.sunmmio.pipeline.parent_owned_buffers");
    if (parent_owned_buffers_anno) {
      parent_owned_buffers =
          Downcast<Array<Buffer>>(parent_owned_buffers_anno.value());
      for (const Buffer &buffer : parent_owned_buffers) {
        parent_owned.insert(buffer.get());
      }
    }

    std::set<int> resource_set;
    for (int i = 0; i < static_cast<int>(pipeline_body.size()); ++i) {
      const Stmt &pipeline_stmt = pipeline_body[i];
      Stmt stmt = pipeline_stmt;
      int inner_coordinate = -1;
      if (const auto *attr = pipeline_stmt.as<AttrStmtNode>()) {
        if (attr->attr_key == "tl.sunmmio.multidim.inner_coordinate") {
          const auto *coordinate = attr->value.as<IntImmNode>();
          ICHECK(coordinate != nullptr);
          inner_coordinate = static_cast<int>(coordinate->value);
          stmt = attr->body;
        }
      }
      Optional<Any> nested_raw_makespan;
      Optional<Any> nested_resources;
      Optional<Any> nested_resource_windows;
      Optional<Any> nested_buffer_frontiers;
      if (const auto *nested_loop = stmt.as<ForNode>()) {
        nested_raw_makespan = nested_loop->annotations.Get(
            "tl.sunmmio.pipeline.summary_raw_makespan");
        nested_resources = nested_loop->annotations.Get(
            "tl.sunmmio.pipeline.summary_resources");
        nested_resource_windows = nested_loop->annotations.Get(
            "tl.sunmmio.pipeline.summary_resource_windows_raw");
        nested_buffer_frontiers = nested_loop->annotations.Get(
            "tl.sunmmio.pipeline.summary_buffer_frontiers_raw");
      }
      TemplateCommand cmd(i, stmt);
      role_marker(stmt);
      cmd.role = role_marker.GetRole(stmt);
      traverser.traverse_stmt(stmt);
      cmd.accesses = access_analyzer.Collect(stmt, loop->loop_var);
      const bool is_nested_command = nested_raw_makespan && nested_resources;
      cmd.inner_coordinate = inner_coordinate;
      if (is_nested_command) {
        // The child remains responsible for buffers used only inside its
        // pipeline. The parent keeps only accesses that connect the atomic
        // child command to sibling statements, such as clear/compute/writeback
        // through a parent-owned accumulator.
        cmd.accesses.erase(
            std::remove_if(cmd.accesses.begin(), cmd.accesses.end(),
                           [&](const AccessInfo &access) {
                             return parent_owned.count(access.buffer().get()) ==
                                    0;
                           }),
            cmd.accesses.end());
        cmd.type = DeviceType::Unspecified;
        cmd.cost_estimate.latency_cycles =
            Downcast<IntImm>(nested_raw_makespan.value())->value;
        cmd.cost_estimate.issue_finish_cycles =
            cmd.cost_estimate.latency_cycles;
        cmd.cost_estimate.profile =
            CostModel::ProfileName(cost_context.options.profile);
        cmd.cost_estimate.model_version = "nested_pipeline_summary_v1";
        cmd.cost_estimate.latency_source =
            "nested_pipeline_conservative_makespan";
        cmd.cost_estimate.op_kind = "nested_pipeline";
        cmd.cost_estimate.resource_overlap_modeled = true;
        for (const Integer &resource :
             Downcast<Array<Integer>>(nested_resources.value())) {
          cmd.spec.resources.push_back(static_cast<int>(resource->value));
        }
        if (ShouldEnableNestedSummaryV2()) {
          ICHECK(nested_resource_windows)
              << "summary v2 resource windows must be validated before "
                 "analysis";
          std::set<int> window_resources;
          const int raw_makespan = cmd.cost_estimate.latency_cycles;
          for (const Array<Integer> &window : Downcast<Array<Array<Integer>>>(
                   nested_resource_windows.value())) {
            ICHECK_EQ(window.size(), 3U);
            int resource = static_cast<int>(window[0]->value);
            int start = static_cast<int>(window[1]->value);
            int end = static_cast<int>(window[2]->value);
            ICHECK_GE(start, 0);
            ICHECK_GT(end, start);
            ICHECK_LE(end, raw_makespan);
            cmd.spec.resource_windows[resource].push_back({start, end});
            window_resources.insert(resource);
          }
          for (int resource : cmd.spec.resources) {
            ICHECK(window_resources.count(resource))
                << "summary v2 lacks a window for resource " << resource;
            cmd.spec.resource_windows[resource] = NormalizeResourceWindows(
                std::move(cmd.spec.resource_windows[resource]));
          }
          ICHECK_EQ(window_resources.size(), cmd.spec.resources.size());
          ICHECK(nested_buffer_frontiers)
              << "summary v2 buffer frontiers must be validated before "
                 "analysis";
          Map<Buffer, Array<Integer>> buffer_frontiers =
              Downcast<Map<Buffer, Array<Integer>>>(
                  nested_buffer_frontiers.value());
          std::map<std::string, Array<Integer>> buffer_frontiers_by_name;
          for (const auto &entry : buffer_frontiers) {
            buffer_frontiers_by_name[entry.first->name] = entry.second;
          }
          for (AccessInfo &access : cmd.accesses) {
            auto frontier =
                buffer_frontiers_by_name.find(access.buffer()->name);
            ICHECK(frontier != buffer_frontiers_by_name.end())
                << "summary v2 lacks buffer frontier for "
                << access.buffer()->name;
            ICHECK_EQ(frontier->second.size(), 4U);
            int first = static_cast<int>(
                frontier->second[access.is_write ? 2 : 0]->value);
            int last = static_cast<int>(
                frontier->second[access.is_write ? 3 : 1]->value);
            ICHECK_GE(first, 0);
            ICHECK_GE(last, first);
            ICHECK_LE(last, raw_makespan);
            access.first_offset = first;
            access.last_offset = last;
          }
          cmd.cost_estimate.model_version =
              "nested_pipeline_summary_v2_resource_windows_data_frontiers";
          cmd.cost_estimate.latency_source =
              "nested_pipeline_resource_windows_data_frontiers";
          result.summary_v2_applied = true;
          result.summary_data_frontiers_applied =
              result.summary_data_frontiers_applied || !cmd.accesses.empty();
        }
      } else {
        cmd.type = HardwareMapper::Map(stmt);
        cmd.cost_estimate = CostModel::Estimate(cmd.type, stmt, cost_context);
        cmd.spec.resources = BuildIlpResources(stmt, cmd.type, cmd.accesses);
      }
      ICHECK_LE(cmd.cost_estimate.latency_cycles,
                static_cast<int64_t>(std::numeric_limits<int>::max()))
          << "ILP command latency exceeds int range";
      cmd.spec.latency = static_cast<int>(cmd.cost_estimate.latency_cycles);
      cmd.spec.latency = std::max(cmd.spec.latency, 1);
      cmd.spec.name = cmd.name + ": " + SummarizeStmtForName(stmt);
      for (int resource : cmd.spec.resources) {
        resource_set.insert(resource);
      }
      if (is_nested_command) {
        for (const AccessInfo &access : cmd.accesses) {
          if (!IsGlobalBuffer(access.buffer())) {
            result.used_buffers.insert(access.buffer());
          }
        }
      } else {
        for (const BufferRegion &read : traverser.read_buffer_regions_) {
          if (!IsGlobalBuffer(read->buffer)) {
            result.used_buffers.insert(read->buffer);
          }
        }
        for (const BufferRegion &write : traverser.write_buffer_regions_) {
          if (!IsGlobalBuffer(write->buffer)) {
            result.used_buffers.insert(write->buffer);
          }
        }
      }
      result.commands.push_back(cmd);
    }

    if (multidim) {
      int first_marked = -1;
      int last_marked = -1;
      for (int i = 0; i < static_cast<int>(result.commands.size()); ++i) {
        if (result.commands[i].inner_coordinate >= 0) {
          if (first_marked < 0) {
            first_marked = i;
          }
          last_marked = i;
        }
      }
      ICHECK_GE(first_marked, 0);
      int current = multidim_inner_min;
      for (int i = 0; i < static_cast<int>(result.commands.size()); ++i) {
        if (result.commands[i].inner_coordinate >= 0) {
          current = result.commands[i].inner_coordinate;
        } else if (i > last_marked) {
          current = multidim_inner_min + multidim_inner_extent - 1;
        }
        result.commands[i].inner_coordinate = current;
      }
    }

    result.prob.N = static_cast<int>(result.commands.size());
    result.prob.P.resize(result.prob.N);
    result.prob.multidim = multidim;
    result.prob.command_inner_coordinates.reserve(result.prob.N);
    for (const TemplateCommand &cmd : result.commands) {
      result.prob.P[cmd.id] = cmd.spec;
      result.prob.command_inner_coordinates.push_back(cmd.inner_coordinate);
    }

    int faster = 0;
    bool auto_faster_selected = false;
    std::vector<int> bump_indices;
    {
      auto pass_ctx = tvm::transform::PassContext::Current();
      auto cfg = pass_ctx->GetConfig<Integer>(tl::kSunmmioFaster);
      if (cfg.defined()) {
        faster = static_cast<int>(cfg.value()->value);
      }
    }
    if (faster <= 0) {
      faster = GetEnvInt("TL_SUNMMIO_FASTER", 0);
    }
    if (faster <= 0 && !gem5_profile) {
      auto auto_selected = AutoSelectSunmmioILPFaster(result.prob.P);
      auto_faster_selected = true;
      faster = auto_selected.first;
      bump_indices = std::move(auto_selected.second);
    }
    if (faster <= 0) {
      faster = 1;
    }

    std::unordered_set<int> bump_index_set(bump_indices.begin(),
                                           bump_indices.end());
    int total_latency = 0;
    int latency_gcd = 0;
    for (int i = 0; i < static_cast<int>(result.commands.size()); ++i) {
      if (bump_index_set.count(i)) {
        result.commands[i].spec.latency += 1;
      }
      for (AccessInfo &access : result.commands[i].accesses) {
        if (access.first_offset < 0) {
          access.first_offset =
              access.is_write ? result.commands[i].spec.latency : 0;
        }
        if (access.last_offset < 0) {
          access.last_offset = result.commands[i].spec.latency;
        }
      }
      total_latency += result.commands[i].spec.latency;
      latency_gcd = latency_gcd == 0
                        ? result.commands[i].spec.latency
                        : GcdInt(latency_gcd, result.commands[i].spec.latency);
    }

    bool enable_latency_gcd = !gem5_profile || multidim;
    {
      auto pass_ctx = tvm::transform::PassContext::Current();
      auto cfg = pass_ctx->GetConfig<Bool>(tl::kSunmmioILPEnableLatencyGCD);
      if (cfg.defined()) {
        enable_latency_gcd = cfg.value()->value;
      } else if (std::getenv("TL_SUNMMIO_ILP_ENABLE_LATENCY_GCD") != nullptr) {
        enable_latency_gcd = GetEnvBool(
            "TL_SUNMMIO_ILP_ENABLE_LATENCY_GCD", enable_latency_gcd);
      }
    }
    if (latency_gcd <= 0 || !enable_latency_gcd) {
      latency_gcd = 1;
    }
    for (TemplateCommand &cmd : result.commands) {
      cmd.spec.latency /= latency_gcd;
      if (faster > 1) {
        cmd.spec.latency = CeilDiv(cmd.spec.latency, faster);
      }
      ScaleResourceWindows(&cmd.spec,
                           static_cast<int64_t>(std::max(1, latency_gcd)) *
                               static_cast<int64_t>(std::max(1, faster)));
      ScaleAccessFrontiers(&cmd.accesses,
                           static_cast<int64_t>(std::max(1, latency_gcd)) *
                               static_cast<int64_t>(std::max(1, faster)),
                           cmd.spec.latency);
      cmd.cost_estimate.solver_scale_factor = faster;
      cmd.cost_estimate.solver_latency_cycles = cmd.spec.latency;
      cmd.cost_estimate.solver_scaled = faster > 1;
    }
    result.prob.cost_model_profile =
        CostModel::ProfileName(cost_context.options.profile);
    result.prob.preserve_absolute_cycles =
        cost_context.options.preserve_absolute_cycles;
    result.prob.solver_scale_factor = faster;
    result.prob.solver_scaled = faster > 1;
    result.prob.latency_gcd_enabled = enable_latency_gcd;
    result.prob.latency_gcd = latency_gcd;
    result.prob.auto_faster_selected = auto_faster_selected;
    result.prob.planner_fallback = false;
    result.prob.fallback_used = false;
    result.prob.cost_estimates.clear();
    result.prob.cost_estimates.reserve(result.commands.size());
    for (const TemplateCommand &cmd : result.commands) {
      result.prob.cost_estimates.push_back(cmd.cost_estimate);
      result.prob.fallback_used =
          result.prob.fallback_used || cmd.cost_estimate.used_fallback;
    }
    if (gem5_profile) {
      total_latency = 0;
      for (const TemplateCommand &cmd : result.commands) {
        total_latency += cmd.spec.latency;
      }
    } else {
      total_latency /= latency_gcd;
      if (faster > 1) {
        total_latency = CeilDiv(total_latency, faster);
      }
    }

    result.prob.Tmax = total_latency + 10;
    result.prob.R.assign(resource_set.begin(), resource_set.end());
    for (int resource : result.prob.R) {
      result.prob.cap[resource] = 1;
    }
    for (const TemplateCommand &cmd : result.commands) {
      result.prob.P[cmd.id] = cmd.spec;
    }
    result.versioned_buffers = DetectVersionedBuffers(result.commands);
    for (const Buffer &buffer : parent_owned_buffers) {
      if (std::find(result.versioned_buffers.begin(),
                    result.versioned_buffers.end(),
                    buffer) == result.versioned_buffers.end()) {
        result.versioned_buffers.push_back(buffer);
      }
    }
    std::sort(
        result.versioned_buffers.begin(), result.versioned_buffers.end(),
        [](const Buffer &a, const Buffer &b) { return a->name < b->name; });
    result.runtime_multiversion_buffers.clear();
    result.runtime_banked_buffers.clear();
    result.runtime_resident_banked_buffers.clear();
    result.runtime_bank_start_phases.clear();
    result.runtime_bank_read_delta_parities.clear();
    result.runtime_bank_writer_phases.clear();
    result.runtime_bank_reader_phases.clear();
    result.runtime_bank_flip_modes.clear();
    result.prob.versioned_buffer_names.clear();
    for (const Buffer &buffer : result.versioned_buffers) {
      result.prob.versioned_buffer_names.push_back(buffer->name);
    }
    BuildTemplateDependencyGraph(result.commands, num_stages,
                                 result.versioned_buffers, loop->loop_var, mode,
                                 &result.prob);
    result.graph_valid = ValidateProblemGraph(result.commands, result.prob);
    return result;
  }

  struct StageShrinkResult {
    IlpLoopAnalysis analysis;
    SolveResult sol;
  };

  StageShrinkResult
  FindMinimalIIAcrossFlipModes(const For &loop,
                               const Array<Stmt> &pipeline_body, int threads,
                               int forced_iterations = -1) {
    IlpLoopAnalysis seed = AnalyzeLoop(loop, pipeline_body, forced_iterations);
    // Try every physical-bank rotation policy.  Both modes allocate
    // ceil(iterations / 2) versions per bank.  Flip advances the version every
    // two iterations while rotating banks; non-flip advances the version on
    // every iteration while keeping its precolored bank fixed.
    BankFlipMode guessed_mode = GuessBankFlipMode(seed.prob);
    std::vector<BankFlipMode> modes{guessed_mode};
    for (const BankFlipMode &mode :
         {BankFlipMode{true, true}, BankFlipMode{true, false},
          BankFlipMode{false, true}, BankFlipMode{false, false}}) {
      if (mode.Id() != guessed_mode.Id()) {
        modes.push_back(mode);
      }
    }
    int forced_mode = GetEnvInt("TL_SUNMMIO_ILP_FORCE_BANK_FLIP_MODE", -1);
    if (forced_mode >= 0) {
      ICHECK_LT(forced_mode, 4)
          << "TL_SUNMMIO_ILP_FORCE_BANK_FLIP_MODE must be in [0, 3]";
      modes = {BankFlipMode{(forced_mode & 2) != 0, (forced_mode & 1) != 0}};
    }

    const int global_resource_lb = ResourceLowerBound(seed.prob);
    StageShrinkResult best;
    for (size_t index = 0; index < modes.size(); ++index) {
      const BankFlipMode &mode = modes[index];
      IlpLoopAnalysis candidate_analysis =
          AnalyzeLoop(loop, pipeline_body, forced_iterations, mode);
      if (!candidate_analysis.graph_valid) {
        continue;
      }
      SolveResult candidate = TryFastVerifiedSchedule(
          candidate_analysis.prob, mode, best.sol.ok ? best.sol.II - 1 : -1);
      if (!candidate.ok && !best.sol.fast_verified_schedule) {
        if (!best.sol.ok || index == 0) {
          candidate =
              FindMinimalIIForMode(candidate_analysis.prob, threads, mode);
        } else if (best.sol.II > 1) {
          SolveResult probe = SolveFixedII(
              candidate_analysis.prob, best.sol.II - 1, false, threads, mode);
          if (probe.ok) {
            candidate = FindMinimalIIForMode(candidate_analysis.prob, threads,
                                             mode, best.sol.II - 1);
          }
        }
      }
      if (candidate.ok && (!best.sol.ok || candidate.II < best.sol.II)) {
        best.analysis = std::move(candidate_analysis);
        best.sol = std::move(candidate);
      }
      if (best.sol.ok && best.sol.II == global_resource_lb) {
        // ResourceLowerBound is independent of bank flip policy.  No
        // remaining mode can improve a solution that already attains it.
        LOG(INFO) << "[ILP] stop flip-mode search at global resource lower "
                     "bound II="
                  << global_resource_lb;
        break;
      }
    }
    if (!best.sol.ok) {
      return {std::move(seed), {}};
    }
    return best;
  }

  bool ShouldEnableStageShrink() const {
    auto pass_ctx = tvm::transform::PassContext::Current();
    auto cfg = pass_ctx->GetConfig<Bool>(tl::kSunmmioILPStageShrink);
    if (cfg.defined()) {
      return cfg.value()->value;
    }
    return false;
  }

  void PopulateRuntimeBankedBuffers(IlpLoopAnalysis *analysis) const {
    analysis->runtime_banked_buffers.clear();
    for (const Buffer &buffer : analysis->versioned_buffers) {
      if (analysis->runtime_bank_start_phases.count(buffer->name)) {
        analysis->runtime_banked_buffers.push_back(buffer);
      }
    }
  }

  void ExportStageSolutionIfRequested(const IlpLoopAnalysis &analysis,
                                      const SolveResult &sol, int stage) const {
    std::string solution_json_path =
        GetEnvString("TL_SUNMMIO_ILP_SOLUTION_JSON");
    if (solution_json_path.empty() || !sol.ok) {
      return;
    }
    IlpLoopAnalysis export_analysis = analysis;
    PopulateRuntimeBankMetadata(&export_analysis, sol);
    PopulateRuntimeBankedBuffers(&export_analysis);
    SolutionVerifyResult verify = VerifySolution(export_analysis.prob, sol);
    WriteSolutionJson(AddStageSuffixToPath(solution_json_path, stage),
                      export_analysis.prob, sol, verify,
                      export_analysis.runtime_bank_start_phases,
                      export_analysis.runtime_bank_read_delta_parities,
                      export_analysis.runtime_bank_reader_phases);
  }

  StageShrinkResult SolveWithStageShrink(const For &loop,
                                         const Array<Stmt> &pipeline_body,
                                         int threads) {
    StageShrinkResult mode_result =
        FindMinimalIIAcrossFlipModes(loop, pipeline_body, threads);
    IlpLoopAnalysis base_analysis = std::move(mode_result.analysis);
    SolveResult base_sol = std::move(mode_result.sol);
    MaybeExportProblemJsonForStage(base_analysis.prob, debug_,
                                   base_analysis.iterations);
    if (!base_sol.ok) {
      return {std::move(base_analysis), std::move(base_sol)};
    }

    int best_iterations = base_analysis.iterations;
    BankFlipMode fixed_mode = base_sol.bank_flip_mode;
    for (int candidate_iterations = base_analysis.iterations - 1;
         candidate_iterations >= 1; --candidate_iterations) {
      IlpLoopAnalysis candidate_analysis =
          AnalyzeLoop(loop, pipeline_body, candidate_iterations, fixed_mode);
      MaybeExportProblemJsonForStage(candidate_analysis.prob, debug_,
                                     candidate_iterations);
      SolveResult feas = SolveFixedII(candidate_analysis.prob, base_sol.II,
                                      false, threads, fixed_mode);
      if (!feas.ok) {
        continue;
      }
      best_iterations = candidate_iterations;
    }

    IlpLoopAnalysis final_analysis =
        AnalyzeLoop(loop, pipeline_body, best_iterations, fixed_mode);
    MaybeExportProblemJsonForStage(final_analysis.prob, debug_,
                                   final_analysis.iterations);
    SolveResult final_sol = SolveFixedII(final_analysis.prob, base_sol.II, true,
                                         threads, fixed_mode);
    ExportStageSolutionIfRequested(final_analysis, final_sol,
                                   final_analysis.iterations);
    return {std::move(final_analysis), std::move(final_sol)};
  }

  void PopulateRuntimeBankMetadata(IlpLoopAnalysis *analysis,
                                   const SolveResult &sol) const {
    std::unordered_map<int, int> internal_pos;
    for (int i = 0; i < static_cast<int>(sol.internal_flow_ids.size()); ++i) {
      internal_pos[sol.internal_flow_ids[i]] = i;
    }
    std::unordered_set<std::string> has_non_resident_flow;
    for (int fid = 0; fid < static_cast<int>(analysis->prob.flows.size());
         ++fid) {
      const auto &flow = analysis->prob.flows[fid];
      if (!flow.resident && !flow.buffer_name.empty()) {
        has_non_resident_flow.insert(flow.buffer_name);
      }
    }
    for (int fid = 0; fid < static_cast<int>(analysis->prob.flows.size());
         ++fid) {
      const auto &flow = analysis->prob.flows[fid];
      if (flow.buffer_name.empty()) {
        continue;
      }
      auto flow_pos = internal_pos.find(fid);
      if (flow_pos == internal_pos.end()) {
        continue;
      }
      // Export one phase-offset convention for every runtime bank annotation:
      //   physical_bank = (logical_iter_parity + phase_offset) % 2
      // Physical bank 0 is ping and bank 1 is pong.  The offset itself is not
      // a fixed ping/pong selection because it flips with the logical
      // iteration.
      int phase_offset = sol.z_bank[flow_pos->second];
      ICHECK(phase_offset == 0 || phase_offset == 1)
          << "ILP bank phase offset must be binary for flow " << flow.prod
          << " -> " << flow.cons;
      analysis->runtime_bank_flip_modes[flow.buffer_name] =
          sol.bank_flip_mode.FlipForMem(flow.mem) ? 1 : 0;
      bool allow_resident_to_own_bank =
          flow.resident && !has_non_resident_flow.count(flow.buffer_name);
      // Keep per-op bank metadata as the primary source of truth. A single
      // per-buffer start phase is only well-defined when either:
      //   (1) the flow is resident-only for that buffer, or
      //   (2) every contributing flow happens to agree.
      //
      // Non-resident flows of the same logical buffer can legitimately land on
      // different banks when they represent different runtime instances. Those
      // cases are consumed later through runtime_bank_writer_phases /
      // runtime_bank_reader_phases, so do not force them into a single
      // runtime_bank_start_phases entry here.
      if (!flow.resident || allow_resident_to_own_bank) {
        auto it = analysis->runtime_bank_start_phases.find(flow.buffer_name);
        if (it == analysis->runtime_bank_start_phases.end()) {
          analysis->runtime_bank_start_phases[flow.buffer_name] = phase_offset;
        } else if (it->second != phase_offset) {
          if (flow.precolor < 0) {
            LOG(WARNING)
                << "Ignoring conflicting aggregate bank phase for buffer "
                << flow.buffer_name << ": existing=" << it->second
                << " new=" << phase_offset
                << ". Per-op bank metadata will be used instead.";
          }
        }
        int delta_parity =
            sol.bank_flip_mode.FlipForMem(flow.mem) ? (flow.delta & 1) : 0;
        auto it_delta =
            analysis->runtime_bank_read_delta_parities.find(flow.buffer_name);
        if (it_delta == analysis->runtime_bank_read_delta_parities.end()) {
          analysis->runtime_bank_read_delta_parities[flow.buffer_name] =
              delta_parity;
        } else if (it_delta->second != delta_parity) {
          LOG(WARNING)
              << "Ignoring conflicting aggregate read-delta parity for buffer "
              << flow.buffer_name << ": existing=" << it_delta->second
              << " new=" << delta_parity
              << ". Per-op bank metadata will be used instead.";
        }
      } else {
        // For non-resident runtime instances, keep only per-op metadata and do
        // not force a single aggregate start phase / delta parity.
      }
      if (flow.write_resource >= 0 && flow.prod >= 0) {
        auto &writer_map =
            analysis->runtime_bank_writer_phases[flow.buffer_name];
        auto it_writer = writer_map.find(flow.prod);
        if (it_writer == writer_map.end()) {
          writer_map[flow.prod] = phase_offset;
        } else {
          ICHECK_EQ(it_writer->second, phase_offset)
              << "Conflicting writer bank phase for banked buffer "
              << flow.buffer_name << " op " << flow.prod;
        }
      }
      if (flow.cons >= 0 && flow.read_resource >= 0) {
        int reader_phase_offset =
            flow.resident || !sol.bank_flip_mode.FlipForMem(flow.mem)
                ? phase_offset
                : ((phase_offset + (flow.delta & 1)) & 1);
        ICHECK(reader_phase_offset == 0 || reader_phase_offset == 1);
        auto &reader_map =
            analysis->runtime_bank_reader_phases[flow.buffer_name];
        auto it_reader = reader_map.find(flow.cons);
        if (it_reader == reader_map.end()) {
          reader_map[flow.cons] = reader_phase_offset;
        } else {
          ICHECK_EQ(it_reader->second, reader_phase_offset)
              << "Conflicting reader bank phase for banked buffer "
              << flow.buffer_name << " op " << flow.cons;
        }
      }
    }
  }

  void PruneUnnecessaryRuntimeBanking(IlpLoopAnalysis *analysis,
                                      const SolveResult &sol) {
    auto mem_needs_pingpong = [&](int mem) {
      int write_resource = GetMemoryWriteResource(mem);
      int read_resource = GetMemoryReadResource(mem);
      if (write_resource < 0 || read_resource < 0) {
        return false;
      }
      for (int slot = 0; slot < sol.II; ++slot) {
        int write_use = 0;
        int read_use = 0;
        for (int i = 0; i < analysis->prob.N; ++i) {
          const CommandSpec &spec = analysis->prob.P[i];
          bool uses_write =
              std::find(spec.resources.begin(), spec.resources.end(),
                        write_resource) != spec.resources.end();
          bool uses_read =
              std::find(spec.resources.begin(), spec.resources.end(),
                        read_resource) != spec.resources.end();
          if (!uses_write && !uses_read) {
            continue;
          }
          if (uses_write) {
            write_use += ComputeFoldedResourceOccupancy(spec, write_resource,
                                                        sol.t[i], sol.II, slot);
          }
          if (uses_read) {
            read_use += ComputeFoldedResourceOccupancy(spec, read_resource,
                                                       sol.t[i], sol.II, slot);
          }
        }
        if (write_use > 0 && read_use > 0) {
          return true;
        }
      }
      return false;
    };

    bool keep_wsram = mem_needs_pingpong(/*mem=*/0);
    bool keep_asram = mem_needs_pingpong(/*mem=*/1);

    std::vector<Buffer> pruned_banked_buffers;
    for (const Buffer &buffer : analysis->runtime_banked_buffers) {
      int mem = GetPingPongMemoryKind(buffer);
      bool keep = (mem == 0 && keep_wsram) || (mem == 1 && keep_asram);
      if (keep) {
        pruned_banked_buffers.push_back(buffer);
      } else {
        analysis->runtime_bank_start_phases.erase(buffer->name);
        analysis->runtime_bank_read_delta_parities.erase(buffer->name);
        analysis->runtime_bank_writer_phases.erase(buffer->name);
        analysis->runtime_bank_reader_phases.erase(buffer->name);
        analysis->runtime_bank_flip_modes.erase(buffer->name);
      }
    }
    analysis->runtime_banked_buffers = std::move(pruned_banked_buffers);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    For loop = ffi::GetRef<For>(op);
    if (op->annotations.find("num_stages") == op->annotations.end()) {
      return StmtExprMutator::VisitStmt_(op);
    }

    const std::string pipeline_mode =
        op->annotations.find("tl.sunmmio.multidim.region") !=
                op->annotations.end() ? "multidim_ilp" : "ilp";

    const bool nested_enabled = ShouldEnableNestedPipeline();
    PipelineNestInspector nest;
    nest.Inspect(loop);
    const bool has_nested_pipeline = nest.max_depth > 1;
    bool planned_children_first = false;
    if (!nested_enabled && has_nested_pipeline) {
      Stmt body = this->VisitStmt(op->body);
      loop.CopyOnWrite()->body = body;
      For sequential = StripNumStages(loop);
      return MakePipelineFallback(sequential, pipeline_mode, "nested_validation",
                                  "nested_feature_disabled");
    }

    if (nested_enabled && has_nested_pipeline) {
      std::string fallback_reason;
      if (nest.max_depth > 3) {
        fallback_reason = "nested_depth_unsupported";
      } else if (nest.dynamic_extent) {
        fallback_reason = "nested_dynamic_extent_unsupported";
      } else if (nest.unsupported_kind) {
        fallback_reason = "nested_loop_kind_unsupported";
      } else if (nest.indirect_nesting) {
        fallback_reason = "nested_control_flow_unsupported";
      }
      if (!fallback_reason.empty()) {
        For sequential = Downcast<For>(StmtExprMutator::VisitStmt_(op));
        sequential = StripNumStages(sequential);
        return MakePipelineFallback(sequential, pipeline_mode, "nested_validation",
                                    fallback_reason);
      }

      // ASC-NEST phase 1: plan descendants before building the parent
      // problem. A planned child is represented in the parent as one
      // conservative command that occupies every resource used by the child
      // for the child's complete makespan.
      Stmt body = this->VisitStmt(op->body);
      loop.CopyOnWrite()->body = body;
      Optional<String> child_failure = FindNestedChildFailure(body);
      if (child_failure) {
        For sequential = StripNumStages(loop);
        return MakePipelineFallback(
            sequential, PipelineDiagnostic{false, pipeline_mode, "nested_validation",
                                           "nested_child_fallback",
                                           std::string(child_failure.value())});
      }
      std::vector<Buffer> parent_owned_buffers;
      Optional<String> shared_buffers =
          FindNestedSharedBuffers(loop, &parent_owned_buffers);
      if (!parent_owned_buffers.empty()) {
        Array<Buffer> annotation(parent_owned_buffers.begin(),
                                 parent_owned_buffers.end());
        loop.CopyOnWrite()->annotations.Set(
            "tl.sunmmio.pipeline.parent_owned_buffers", annotation);
      }
      if (shared_buffers) {
        For sequential = StripNumStages(loop);
        return MakePipelineFallback(
            sequential,
            PipelineDiagnostic{false, pipeline_mode, "nested_validation",
                               "nested_shared_buffer_unsupported",
                               std::string(shared_buffers.value())});
      }
      planned_children_first = true;
    }

    // A single logical iteration has no cross-iteration overlap to schedule.
    // Leaving it in the ILP window model can manufacture an out-of-range
    // prologue/epilogue iteration when the solved makespan spans two IIs.
    arith::Analyzer extent_analyzer;
    PrimExpr simplified_extent = extent_analyzer.Simplify(op->extent);
    const auto *extent = simplified_extent.as<IntImmNode>();
    if (extent != nullptr && extent->value <= 1) {
      For sequential = Downcast<For>(StmtExprMutator::VisitStmt_(op));
      Map<String, Any> annotations;
      for (const auto &kv : sequential->annotations) {
        if (kv.first != "num_stages") {
          annotations.Set(kv.first, kv.second);
        }
      }
      sequential.CopyOnWrite()->annotations = annotations;
      return MakePipelineFallback(sequential, pipeline_mode, "planning",
                                  "short_extent_unsupported");
    }

    Array<Stmt> pipeline_body;
    if (!GetPipelineBodyStatements(loop, &pipeline_body)) {
      return MakePipelineFallback(loop, pipeline_mode, "planning",
                                  "unsupported_statement");
    }
    for (const Stmt &stmt : pipeline_body) {
      if (!stmt.as<BlockRealizeNode>() && !stmt.as<EvaluateNode>() &&
          !stmt.as<ForNode>() && !stmt.as<AttrStmtNode>()) {
        For sequential = Downcast<For>(StmtExprMutator::VisitStmt_(op));
        Map<String, Any> annotations;
        for (const auto &kv : sequential->annotations) {
          if (kv.first != "num_stages") {
            annotations.Set(kv.first, kv.second);
          }
        }
        sequential.CopyOnWrite()->annotations = annotations;
        return MakePipelineFallback(sequential, pipeline_mode, "planning",
                                    "unsupported_statement");
      }
    }
    Optional<String> invalid_summary_v2 =
        ValidateNestedSummaryV2Inputs(pipeline_body);
    if (invalid_summary_v2) {
      For sequential = StripNumStages(loop);
      return MakePipelineFallback(
          sequential,
          PipelineDiagnostic{false, pipeline_mode, "nested_validation",
                             "nested_summary_v2_unavailable",
                             std::string(invalid_summary_v2.value())});
    }
    int threads = GetEnvInt("HIGHS_THREADS", 20);
    IlpLoopAnalysis analysis;
    SolveResult sol;
    if (ShouldEnableStageShrink()) {
      StageShrinkResult shrink_result =
          SolveWithStageShrink(loop, pipeline_body, threads);
      analysis = std::move(shrink_result.analysis);
      sol = std::move(shrink_result.sol);
    } else {
      StageShrinkResult mode_result =
          FindMinimalIIAcrossFlipModes(loop, pipeline_body, threads);
      analysis = std::move(mode_result.analysis);
      SolveResult min_ii = std::move(mode_result.sol);
      if (min_ii.ok && analysis.summary_v2_applied && extent != nullptr) {
        min_ii = RefineV2FiniteHorizonSchedule(
            analysis.prob, min_ii, static_cast<int64_t>(extent->value));
      }
      MaybeExportProblemJson(analysis.prob, debug_);
      if (min_ii.ok) {
        if (min_ii.objective_optimized || min_ii.fast_verified_schedule) {
          sol = std::move(min_ii);
        } else {
          sol = SolveFixedII(analysis.prob, min_ii.II, true, threads,
                             min_ii.bank_flip_mode, &min_ii);
        }
      }
    }
    if (!analysis.graph_valid) {
      return MakePipelineFallback(loop, pipeline_mode, "graph_validation",
                                  "incomplete_access_info");
    }
    if (GetEnvBool("TL_SUNMMIO_ILP_EXPORT_ONLY", false)) {
      return loop;
    }
    if (!sol.ok) {
      return MakePipelineFallback(loop, pipeline_mode, "planning", "ilp_infeasible");
    }
    PopulateRuntimeBankMetadata(&analysis, sol);
    std::unordered_set<std::string> resident_buffer_names;
    for (const FlowSpec &flow : analysis.prob.flows) {
      if (flow.resident && !flow.buffer_name.empty()) {
        resident_buffer_names.insert(flow.buffer_name);
      }
    }
    for (const Buffer &buffer : analysis.versioned_buffers) {
      if (analysis.runtime_bank_start_phases.count(buffer->name) ||
          analysis.runtime_bank_writer_phases.count(buffer->name) ||
          analysis.runtime_bank_reader_phases.count(buffer->name)) {
        analysis.runtime_banked_buffers.push_back(buffer);
      }
    }
    for (const Buffer &buffer : analysis.used_buffers) {
      if (!resident_buffer_names.count(buffer->name))
        continue;
      if (std::find(analysis.runtime_banked_buffers.begin(),
                    analysis.runtime_banked_buffers.end(),
                    buffer) == analysis.runtime_banked_buffers.end()) {
        analysis.runtime_banked_buffers.push_back(buffer);
      }
      analysis.runtime_resident_banked_buffers.push_back(buffer);
    }
    // Disabled per current ILP annotation semantics:
    // keep the bank-rotation decision from the solved flow model as-is, and do
    // not prune ping/pong before annotation.
    //
    // PruneUnnecessaryRuntimeBanking(&analysis, sol);
    SolutionVerifyResult verify = VerifySolution(analysis.prob, sol);
    if (!verify.ok) {
      return MakePipelineFallback(loop, pipeline_mode, "planning",
                                  "schedule_verification_failed");
    }
    std::string solution_json_path =
        GetEnvString("TL_SUNMMIO_ILP_SOLUTION_JSON");
    if (!solution_json_path.empty()) {
      WriteSolutionJson(solution_json_path, analysis.prob, sol, verify,
                        analysis.runtime_bank_start_phases,
                        analysis.runtime_bank_read_delta_parities,
                        analysis.runtime_bank_reader_phases);
    }
    if (GetEnvBool("TL_SUNMMIO_ILP_SOLVE_ONLY", false)) {
      return loop;
    }
    if (debug_) {
      LOG(INFO) << "ILP problem N=" << analysis.prob.N
                << " dependencies=" << analysis.prob.dependencies.size()
                << " flows=" << analysis.prob.flows.size()
                << " solved=" << sol.ok << " ii=" << sol.II;
    }

    Map<String, Any> annotations;
    for (const auto &kv : loop->annotations) {
      if (kv.first != "num_stages" && kv.first != "versioned_buffers") {
        annotations.Set(kv.first, kv.second);
      }
    }

    int stage_count = CeilDiv(sol.makespan, std::max(1, sol.II));
    SetPipelineAppliedAnnotations(&annotations, pipeline_mode);
    auto pass_ctx = tvm::transform::PassContext::Current();
    bool enable_lifetime_pruning =
        pass_ctx
            ->GetConfig<Bool>(tl::kSunmmioILPMultiversionLifetimePruning,
                              Bool(true))
            .value();
    analysis.runtime_multiversion_buffers = DetectRuntimeMultiversionBuffers(
        analysis.commands, analysis.versioned_buffers,
        analysis.runtime_banked_buffers, loop->loop_var, sol,
        analysis.iterations, enable_lifetime_pruning,
        analysis.runtime_bank_start_phases,
        analysis.runtime_bank_read_delta_parities,
        analysis.runtime_bank_writer_phases,
        analysis.runtime_bank_reader_phases, analysis.runtime_bank_flip_modes);
    annotations.Set("iterations", Integer(analysis.iterations));
    annotations.Set("ii", Integer(sol.II));
    annotations.Set("makespan", Integer(sol.makespan));
    annotations.Set("stage_count", Integer(stage_count));
    Array<String> prologue_orders;
    Array<String> body_orders;
    Array<String> epilogue_orders;
    auto command_priority = [&](int id) {
      const CommandSpec &spec = analysis.prob.P[id];
      if (CommandUsesResource(spec,
                              static_cast<int>(IlpResourceType::kODMA1))) {
        return 0;
      }
      if (CommandUsesResource(spec,
                              static_cast<int>(IlpResourceType::kODMA0))) {
        return 1;
      }
      // ODMA launch is asynchronous, while tmma.mm blocks the scalar issue
      // stream until the tensor command completes.  Submit same-time async
      // work first so it can overlap the blocking tensor command.
      if (CommandUsesResource(spec,
                              static_cast<int>(IlpResourceType::kTensorCore))) {
        return 2;
      }
      if (CommandUsesResource(spec,
                              static_cast<int>(IlpResourceType::kVectorCore))) {
        return 3;
      }
      return 4;
    };
    auto starts_earlier_and_resource_priority = [&](int a, int b) {
      if (sol.t[a] != sol.t[b]) {
        return sol.t[a] < sol.t[b];
      }
      int a_priority = command_priority(a);
      int b_priority = command_priority(b);
      if (a_priority != b_priority) {
        return a_priority < b_priority;
      }
      return a < b;
    };
    TimeWindowOrderResult window_orders =
        BuildTimeWindowOrders(analysis.commands, analysis.iterations, sol);
    annotations.Set("steady_state_max_iter_offset",
                    Integer(window_orders.steady_state_max_iter_offset));

    auto time_then_non_vc_first = [&](const ExpandedOrderEntry &a,
                                      const ExpandedOrderEntry &b) {
      if (a.absolute_start != b.absolute_start) {
        return a.absolute_start < b.absolute_start;
      }
      int a_priority = command_priority(a.id);
      int b_priority = command_priority(b.id);
      if (a_priority != b_priority) {
        return a_priority < b_priority;
      }
      return a.id < b.id;
    };

    std::sort(window_orders.prologue.begin(), window_orders.prologue.end(),
              time_then_non_vc_first);
    std::sort(window_orders.body.begin(), window_orders.body.end(),
              time_then_non_vc_first);
    std::sort(window_orders.epilogue.begin(), window_orders.epilogue.end(),
              time_then_non_vc_first);

    for (const ExpandedOrderEntry &entry : window_orders.prologue) {
      prologue_orders.push_back(
          String(std::to_string(entry.iter) + "-" + std::to_string(entry.id)));
    }
    for (const ExpandedOrderEntry &entry : window_orders.body) {
      body_orders.push_back(
          String(std::to_string(entry.iter) + "-" + std::to_string(entry.id)));
    }
    for (const ExpandedOrderEntry &entry : window_orders.epilogue) {
      epilogue_orders.push_back(
          String(std::to_string(entry.iter) + "-" + std::to_string(entry.id)));
    }

    annotations.Set("prologue_orders", prologue_orders);
    annotations.Set("body_orders", body_orders);
    annotations.Set("epilogue_orders", epilogue_orders);
    Array<Integer> command_start_times;
    Array<Integer> command_emit_priorities;
    for (int id = 0; id < analysis.prob.N; ++id) {
      command_start_times.push_back(Integer(sol.t[id]));
      command_emit_priorities.push_back(Integer(command_priority(id)));
    }
    // Phase 4 composes already-solved 1D schedules across an outer boundary.
    // Keep the inner solver result explicit so composition never re-solves a
    // flattened multi-dimensional problem.
    annotations.Set("tl.sunmmio.pipeline.command_start_times",
                    command_start_times);
    annotations.Set("tl.sunmmio.pipeline.command_emit_priorities",
                    command_emit_priorities);

    if (nested_enabled) {
      const int64_t raw_scale =
          static_cast<int64_t>(std::max(1, analysis.prob.latency_gcd)) *
          static_cast<int64_t>(std::max(1, analysis.prob.solver_scale_factor));
      const int64_t raw_makespan =
          static_cast<int64_t>(sol.makespan) * raw_scale;
      const int64_t raw_ii = static_cast<int64_t>(sol.II) * raw_scale;
      ICHECK_LE(raw_makespan,
                static_cast<int64_t>(std::numeric_limits<int>::max()));
      ICHECK_LE(raw_ii, static_cast<int64_t>(std::numeric_limits<int>::max()));
      Array<Integer> summary_resources;
      for (int resource : analysis.prob.R) {
        summary_resources.push_back(Integer(resource));
      }
      annotations.Set("tl.sunmmio.pipeline.region_id",
                      String("asc_nest_" + std::to_string(next_region_id_++)));
      annotations.Set("tl.sunmmio.pipeline.nested_depth",
                      Integer(nest.max_depth));
      annotations.Set("tl.sunmmio.pipeline.summary_raw_makespan",
                      Integer(raw_makespan));
      annotations.Set("tl.sunmmio.pipeline.summary_raw_ii", Integer(raw_ii));
      annotations.Set("tl.sunmmio.pipeline.summary_resources",
                      summary_resources);
      annotations.Set("tl.sunmmio.pipeline.summary_solver_scale",
                      Integer(analysis.prob.solver_scale_factor));
      annotations.Set("tl.sunmmio.pipeline.summary_latency_gcd",
                      Integer(analysis.prob.latency_gcd));

      if (ShouldEnableNestedSummaryV2()) {
        std::map<int, std::vector<std::pair<int64_t, int64_t>>>
            resource_windows;
        for (const TemplateCommand &cmd : analysis.commands) {
          for (int resource : analysis.prob.P[cmd.id].resources) {
            auto windows =
                analysis.prob.P[cmd.id].resource_windows.find(resource);
            if (windows == analysis.prob.P[cmd.id].resource_windows.end()) {
              const int64_t start =
                  static_cast<int64_t>(sol.t[cmd.id]) * raw_scale;
              const int64_t end =
                  static_cast<int64_t>(sol.t[cmd.id] +
                                       analysis.prob.P[cmd.id].latency) *
                  raw_scale;
              resource_windows[resource].push_back({start, end});
              continue;
            }
            for (const auto &window : windows->second) {
              const int64_t start =
                  static_cast<int64_t>(sol.t[cmd.id] + window.first) *
                  raw_scale;
              const int64_t end =
                  static_cast<int64_t>(sol.t[cmd.id] + window.second) *
                  raw_scale;
              resource_windows[resource].push_back({start, end});
            }
          }
        }

        Array<Array<Integer>> summary_resource_windows;
        for (auto &resource_entry : resource_windows) {
          auto &windows = resource_entry.second;
          std::sort(windows.begin(), windows.end());
          std::vector<std::pair<int64_t, int64_t>> merged;
          for (const auto &window : windows) {
            if (merged.empty() || window.first > merged.back().second) {
              merged.push_back(window);
            } else {
              merged.back().second =
                  std::max(merged.back().second, window.second);
            }
          }
          for (const auto &window : merged) {
            summary_resource_windows.push_back({Integer(resource_entry.first),
                                                Integer(window.first),
                                                Integer(window.second)});
          }
        }

        struct BufferFrontier {
          Buffer buffer;
          int64_t first_read{-1};
          int64_t last_read_end{-1};
          int64_t first_write_ready{-1};
          int64_t last_write_ready{-1};
        };
        std::map<std::string, BufferFrontier> buffer_frontiers;
        for (const TemplateCommand &cmd : analysis.commands) {
          for (const AccessInfo &access : cmd.accesses) {
            if (IsGlobalBuffer(access.buffer())) {
              continue;
            }
            const int64_t first =
                static_cast<int64_t>(sol.t[cmd.id] + access.first_offset) *
                raw_scale;
            const int64_t last =
                static_cast<int64_t>(sol.t[cmd.id] + access.last_offset) *
                raw_scale;
            BufferFrontier &frontier = buffer_frontiers[access.buffer()->name];
            frontier.buffer = access.buffer();
            if (access.is_write) {
              frontier.first_write_ready =
                  frontier.first_write_ready < 0
                      ? first
                      : std::min(frontier.first_write_ready, first);
              frontier.last_write_ready =
                  std::max(frontier.last_write_ready, last);
            } else {
              frontier.first_read = frontier.first_read < 0
                                        ? first
                                        : std::min(frontier.first_read, first);
              frontier.last_read_end = std::max(frontier.last_read_end, last);
            }
          }
        }

        Map<Buffer, Array<Integer>> summary_buffer_frontiers;
        for (const auto &entry : buffer_frontiers) {
          const BufferFrontier &frontier = entry.second;
          summary_buffer_frontiers.Set(frontier.buffer,
                                       {Integer(frontier.first_read),
                                        Integer(frontier.last_read_end),
                                        Integer(frontier.first_write_ready),
                                        Integer(frontier.last_write_ready)});
        }

        annotations.Set("tl.sunmmio.pipeline.summary_v2_requested", Bool(true));
        annotations.Set("tl.sunmmio.pipeline.summary_v2_observed", Bool(true));
        annotations.Set("tl.sunmmio.pipeline.summary_v2_applied",
                        Bool(analysis.summary_v2_applied));
        annotations.Set("tl.sunmmio.pipeline.summary_data_frontiers_applied",
                        Bool(analysis.summary_data_frontiers_applied));
        annotations.Set("tl.sunmmio.pipeline.summary_model_version",
                        String("v2_resource_windows_data_frontiers"));
        annotations.Set("tl.sunmmio.pipeline.summary_resource_windows_raw",
                        summary_resource_windows);
        annotations.Set("tl.sunmmio.pipeline.summary_buffer_frontiers_raw",
                        summary_buffer_frontiers);
      }
    }

    Optional<Array<Buffer>> used_buffers_array =
        StableUsedBufferExporter::Export(func_, analysis.used_buffers);
    if (!used_buffers_array.defined()) {
      return MakePipelineFallback(StripNumStages(loop), pipeline_mode,
                                  "planning", "used_buffer_order_unproven");
    }
    annotations.Set("used_buffers", used_buffers_array.value());
    Array<Buffer> versioned_buffers_array(analysis.versioned_buffers.begin(),
                                          analysis.versioned_buffers.end());
    annotations.Set("versioned_buffers", versioned_buffers_array);
    Array<Buffer> runtime_multiversion_buffers_array(
        analysis.runtime_multiversion_buffers.begin(),
        analysis.runtime_multiversion_buffers.end());
    annotations.Set("runtime_multiversion_buffers",
                    runtime_multiversion_buffers_array);
    Array<Buffer> runtime_banked_buffers_array(
        analysis.runtime_banked_buffers.begin(),
        analysis.runtime_banked_buffers.end());
    annotations.Set("runtime_banked_buffers", runtime_banked_buffers_array);
    Array<Buffer> runtime_resident_banked_buffers_array(
        analysis.runtime_resident_banked_buffers.begin(),
        analysis.runtime_resident_banked_buffers.end());
    annotations.Set("runtime_resident_banked_buffers",
                    runtime_resident_banked_buffers_array);
    Map<Buffer, PrimExpr> runtime_bank_start_phases;
    for (const Buffer &buffer : analysis.runtime_banked_buffers) {
      runtime_bank_start_phases.Set(
          buffer, Integer(analysis.runtime_bank_start_phases.at(buffer->name)));
    }
    annotations.Set("runtime_bank_start_phases", runtime_bank_start_phases);
    Map<Buffer, PrimExpr> runtime_bank_read_delta_parities;
    for (const Buffer &buffer : analysis.runtime_banked_buffers) {
      auto it = analysis.runtime_bank_read_delta_parities.find(buffer->name);
      if (it != analysis.runtime_bank_read_delta_parities.end()) {
        runtime_bank_read_delta_parities.Set(buffer, Integer(it->second));
      }
    }
    annotations.Set("runtime_bank_read_delta_parities",
                    runtime_bank_read_delta_parities);
    Map<Buffer, Map<Integer, PrimExpr>> runtime_bank_writer_phases;
    for (const Buffer &buffer : analysis.runtime_banked_buffers) {
      auto it = analysis.runtime_bank_writer_phases.find(buffer->name);
      if (it == analysis.runtime_bank_writer_phases.end()) {
        continue;
      }
      Map<Integer, PrimExpr> per_op;
      for (const auto &op_phase : it->second) {
        per_op.Set(Integer(op_phase.first), Integer(op_phase.second));
      }
      runtime_bank_writer_phases.Set(buffer, per_op);
    }
    annotations.Set("runtime_bank_writer_phases", runtime_bank_writer_phases);
    Map<Buffer, Map<Integer, PrimExpr>> runtime_bank_reader_phases;
    for (const Buffer &buffer : analysis.runtime_banked_buffers) {
      auto it = analysis.runtime_bank_reader_phases.find(buffer->name);
      if (it == analysis.runtime_bank_reader_phases.end()) {
        continue;
      }
      Map<Integer, PrimExpr> per_op;
      for (const auto &op_phase : it->second) {
        per_op.Set(Integer(op_phase.first), Integer(op_phase.second));
      }
      runtime_bank_reader_phases.Set(buffer, per_op);
    }
    annotations.Set("runtime_bank_reader_phases", runtime_bank_reader_phases);
    Map<Buffer, PrimExpr> runtime_bank_flip_modes;
    for (const Buffer &buffer : analysis.runtime_banked_buffers) {
      auto it = analysis.runtime_bank_flip_modes.find(buffer->name);
      if (it != analysis.runtime_bank_flip_modes.end()) {
        runtime_bank_flip_modes.Set(buffer, Integer(it->second));
      }
    }
    annotations.Set("runtime_bank_flip_modes", runtime_bank_flip_modes);

    Stmt body = planned_children_first ? loop->body : this->VisitStmt(op->body);
    For new_loop = loop;
    ForNode *loop_ptr = new_loop.CopyOnWrite();
    loop_ptr->body = body;
    loop_ptr->annotations = annotations;
    return new_loop;
  }

  PrimFunc func_;
  ASTTraverser traverser_;
  bool debug_{false};
  int next_region_id_{0};
};

tvm::transform::Pass SunmmioPipelinePlanningILP(bool debug = false) {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = SunmmioPipelinePlannerILP::Substitute(f, debug);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.SunmmioPipelinePlanningILP", {});
}

} // namespace bank_ilp_internal

tvm::transform::Pass SunmmioPipelinePlanningILP(bool debug = false) {
  return bank_ilp_internal::SunmmioPipelinePlanningILP(debug);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.SunmmioPipelinePlanningILP",
                        SunmmioPipelinePlanningILP)
      // Internal read-only test hook: exactly the production exporter, without
      // running a solver, modifying IR, or adding an alternate pipeline path.
      .def("tl.testing.SunmmioILPStableUsedBuffers",
           [](tir::PrimFunc func, Array<tir::Buffer> buffers) {
             std::set<tir::Buffer> required(buffers.begin(), buffers.end());
             return bank_ilp_internal::StableUsedBufferExporter::Export(
                 func, required);
           });
}

} // namespace tl
} // namespace tvm
