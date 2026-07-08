#include "../op/utils.h"
#include "../op/builtin.h"
#include "common/ast_traverser.h"
#include "../target/sunmmio/cost_model.h"
#include "../target/sunmmio/hardware_types.h"
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

#include <highs/Highs.h>
#include <highs/lp_data/HConst.h>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
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

struct AccessInfo {
  BufferRegion region;
  bool is_write{false};
  int iter_offset{0};

  const Buffer& buffer() const { return region->buffer; }
};

struct CommandSpec {
  int latency{0};
  std::vector<int> resources;
  std::string name;
};

struct FlowSpec {
  bool resident{false};
  int prod{-1};
  int cons{-1};
  int delta{0};
  int mem{0};
  std::string buffer_name;
  int fixed_bank{-1};
  int fp{1};
  int initial_time{0};
  int w_off{0};
  int w_dur{0};
  int r_off{0};
  int r_dur{0};
  int write_resource{-1};
  int read_resource{-1};
};

using SameWriteFlowKey = std::tuple<int, int>;

SameWriteFlowKey MakeSameWriteFlowKey(const FlowSpec& flow) {
  return std::make_tuple(flow.prod, flow.mem);
}

struct Problem {
  int N{0};
  int Tmax{0};
  std::vector<int> R;
  std::unordered_map<int, int> cap;
  std::vector<CommandSpec> P;
  std::vector<std::pair<int, int>> dep_edges;
  std::unordered_map<long long, int> delta;
  std::vector<FlowSpec> flows;
  std::vector<std::string> versioned_buffer_names;
};

struct ModelVars {
  HighsInt col_T{-1};
  std::vector<HighsInt> col_t;
  std::vector<HighsInt> col_y;
  std::vector<HighsInt> col_m;
  std::vector<std::vector<HighsInt>> col_x;
  std::vector<std::vector<HighsInt>> col_a;
  std::vector<int> internal_flow_ids;
  std::vector<HighsInt> col_z;
};

struct SolveResult {
  bool ok{false};
  int II{0};
  int makespan{0};
  int bank_slot_period{0};
  std::vector<int> t;
  std::vector<int> m;
  std::vector<int> y;
  std::vector<int> internal_flow_ids;
  std::vector<int> z_bank;
};

struct SolutionVerifyResult {
  bool ok{true};
  bool node_time_ok{true};
  bool dependency_ok{true};
  bool resource_slot_ok{true};
  bool bank_slot_ok{true};
  bool bank_port_ok{true};
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

bool CommandUsesResource(const CommandSpec& spec, int resource);

BufferRegion MaterializeBufferRegion(const BufferRegion& region,
                                     const Var& loop_var, int iter);

namespace {

// File-local helper to avoid colliding with the similarly named function in
// sunmmio_pipeline_planning.cc during final shared-library link.
bool PipelineRegionIntersect(const Region& region1, const Region& region2) {
  ICHECK(region1.size() == region2.size());
  for (size_t i = 0; i < region1.size(); ++i) {
    const Range& dim1 = region1[i];
    const Range& dim2 = region2[i];
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

std::map<int, std::vector<std::pair<int, int>>> BuildResourceUsage(
    const std::vector<CommandSpec>& specs) {
  std::map<int, std::vector<std::pair<int, int>>> usage;
  for (int idx = 0; idx < static_cast<int>(specs.size()); ++idx) {
    int latency = std::max(1, specs[idx].latency);
    for (int resource : specs[idx].resources) {
      usage[resource].push_back({idx, latency});
    }
  }
  return usage;
}

std::map<int, int> BuildResourceTotals(
    const std::map<int, std::vector<std::pair<int, int>>>& usage) {
  std::map<int, int> totals;
  for (const auto& kv : usage) {
    int total = 0;
    for (const auto& item : kv.second) {
      total += item.second;
    }
    totals[kv.first] = total;
  }
  return totals;
}

std::pair<int, int> FindTargetResource(
    const std::map<int, int>& totals) {
  int target_resource = -1;
  int target_total = 0;
  for (const auto& kv : totals) {
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

std::vector<int> TryFactorWithOptionalBumps(
    const std::vector<std::pair<int, int>>& items, int factor) {
  std::vector<int> bump_indices;
  for (const auto& item : items) {
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

int ScaledTotalForResource(
    const std::vector<CommandSpec>& specs, int resource, int faster,
    const std::unordered_set<int>& bump_indices) {
  int total = 0;
  for (int idx = 0; idx < static_cast<int>(specs.size()); ++idx) {
    if (std::find(specs[idx].resources.begin(), specs[idx].resources.end(),
                  resource) == specs[idx].resources.end()) {
      continue;
    }
    int latency = specs[idx].latency +
                  (bump_indices.count(idx) ? 1 : 0);
    total += CeilDiv(std::max(1, latency), faster);
  }
  return total;
}

std::vector<int> CandidateGCDs(const std::vector<int>& latencies) {
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

std::pair<int, std::vector<int>> TryGCDWithOptionalBumps(
    const std::vector<std::pair<int, int>>& items, int target_total,
    int target_upper = 69) {
  std::vector<int> latencies;
  latencies.reserve(items.size());
  for (const auto& item : items) {
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
    for (const auto& item : items) {
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

std::pair<int, std::vector<int>> AutoSelectSunmmioILPFaster(
    const std::vector<CommandSpec>& specs, int target_upper = 69) {
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
      for (const auto& item : usage[target_resource]) {
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
    bool violations = false;
    for (const auto& kv : totals) {
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

  auto [gcd_value, bump_vec] =
      TryGCDWithOptionalBumps(usage[target_resource], target_total,
                              target_upper);
  return {std::max(1, gcd_value), bump_vec};
}

int GetEnvInt(const char* name, int default_value) {
  const char* raw = std::getenv(name);
  if (!raw || !*raw) {
    return default_value;
  }
  return std::atoi(raw);
}

std::string GetEnvString(const char* name) {
  const char* raw = std::getenv(name);
  if (!raw || !*raw) {
    return "";
  }
  return raw;
}

bool GetEnvBool(const char* name, bool default_value = false) {
  const char* raw = std::getenv(name);
  if (!raw || !*raw) {
    return default_value;
  }
  std::string value(raw);
  for (char& c : value) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

std::string JsonEscape(const std::string& value) {
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

void WriteJsonString(std::ostream& os, const std::string& value) {
  os << "\"" << JsonEscape(value) << "\"";
}

void WriteProblemJson(const Problem& prob, const std::string& path) {
  std::ofstream out(path);
  ICHECK(out.is_open()) << "Failed to open ILP problem json path: " << path;
  out << std::boolalpha;
  out << "{\n";
  out << "  \"N\": " << prob.N << ",\n";
  out << "  \"Tmax\": " << prob.Tmax << ",\n";

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
  for (const auto& kv : sorted_cap) {
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
    out << "], \"name\": ";
    WriteJsonString(out, prob.P[i].name);
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
    out << "]}";
    out << (i + 1 == prob.N ? "\n" : ",\n");
  }
  out << "  },\n";

  out << "  \"dep_edges\": [";
  for (size_t i = 0; i < prob.dep_edges.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << "[" << prob.dep_edges[i].first << ", " << prob.dep_edges[i].second << "]";
  }
  out << "],\n";

  std::map<std::pair<int, int>, int> sorted_delta;
  for (const auto& edge : prob.dep_edges) {
    auto it = prob.delta.find(EdgeKey(edge.first, edge.second));
    if (it != prob.delta.end()) {
      sorted_delta[edge] = it->second;
    }
  }
  out << "  \"delta\": {";
  bool first_delta = true;
  for (const auto& kv : sorted_delta) {
    if (!first_delta) {
      out << ", ";
    }
    first_delta = false;
    WriteJsonString(out, std::to_string(kv.first.first) + "," +
                             std::to_string(kv.first.second));
    out << ": " << kv.second;
  }
  out << "},\n";

  std::vector<FlowSpec> sorted_flows = prob.flows;
  std::sort(sorted_flows.begin(), sorted_flows.end(),
            [](const FlowSpec& a, const FlowSpec& b) {
              return std::tie(a.resident, a.prod, a.cons, a.mem, a.buffer_name,
                              a.fixed_bank, a.fp,
                              a.initial_time, a.w_off, a.w_dur, a.r_off, a.r_dur,
                              a.write_resource, a.read_resource) <
                     std::tie(b.resident, b.prod, b.cons, b.mem, b.buffer_name,
                              b.fixed_bank, b.fp,
                              b.initial_time, b.w_off, b.w_dur, b.r_off, b.r_dur,
                              b.write_resource, b.read_resource);
            });
  out << "  \"flows\": [";
  for (size_t i = 0; i < sorted_flows.size(); ++i) {
    const FlowSpec& flow = sorted_flows[i];
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

void MaybeExportProblemJson(const Problem& prob, bool debug) {
  std::string export_path = GetEnvString("TL_SUNMMIO_ILP_PROBLEM_JSON");
  if (export_path.empty() && debug) {
    export_path = "body_ilp_problem.json";
  }
  if (!export_path.empty()) {
    WriteProblemJson(prob, export_path);
  }
}

std::string AddStageSuffixToPath(const std::string& path, int stage) {
  if (path.empty()) {
    return path;
  }
  std::string suffix = "_" + std::to_string(stage);
  size_t dot = path.find_last_of('.');
  size_t slash = path.find_last_of("/\\");
  if (dot == std::string::npos ||
      (slash != std::string::npos && dot < slash)) {
    return path + suffix;
  }
  return path.substr(0, dot) + suffix + path.substr(dot);
}

void MaybeExportProblemJsonForStage(const Problem& prob, bool debug, int stage) {
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
                                          int read_parity_flip,
                                          int ii) {
  std::map<int, int> write_rho;
  for (int step = 0; step < write_dur; ++step) {
    int slot = (write_start_slot + step) % ii;
    write_rho[slot] = WrapBit(write_start_slot, slot) ^ write_parity_flip;
  }

  bool saw_same = false;
  bool saw_diff = false;
  for (int step = 0; step < read_dur; ++step) {
    int slot = (read_start_slot + step) % ii;
    auto it = write_rho.find(slot);
    if (it == write_rho.end()) continue;
    int read_rho = WrapBit(read_start_slot, slot) ^ read_parity_flip;
    if (it->second == read_rho) {
      saw_same = true;
    } else {
      saw_diff = true;
    }
    if (saw_same && saw_diff) return ConflictType::kImpossible;
  }

  if (!saw_same && !saw_diff) return ConflictType::kNone;
  if (saw_same) return ConflictType::kNeedDifferent;
  return ConflictType::kNeedSame;
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

SolutionVerifyResult VerifySolution(const Problem& prob, const SolveResult& sol) {
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

  auto fail = [&](bool* flag, std::string msg) {
    if (flag != nullptr) {
      *flag = false;
    }
    result.ok = false;
    result.errors.push_back(std::move(msg));
  };

  for (int i = 0; i < prob.N; ++i) {
    if (sol.m[i] < 0 || sol.m[i] >= sol.II) {
      fail(&result.node_time_ok,
           "node " + std::to_string(i) + " has invalid slot " + std::to_string(sol.m[i]));
    }
    if (sol.t[i] < 0) {
      fail(&result.node_time_ok,
           "node " + std::to_string(i) + " has negative start " + std::to_string(sol.t[i]));
    }
    if (sol.y[i] < 0) {
      fail(&result.node_time_ok,
           "node " + std::to_string(i) + " has negative iteration " +
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

  for (const auto& e : prob.dep_edges) {
    int src = e.first;
    int dst = e.second;
    int delta = prob.delta.at(EdgeKey(src, dst));
    int lhs = sol.t[dst] - sol.t[src];
    int rhs = prob.P[src].latency - delta * sol.II;
    if (lhs < rhs) {
      fail(&result.dependency_ok,
           "dependency violated " + std::to_string(src) + "->" + std::to_string(dst) +
               ": lhs=" + std::to_string(lhs) + " rhs=" + std::to_string(rhs) +
               " delta=" + std::to_string(delta));
    }
  }

  for (int i = 0; i < prob.N; ++i) {
    for (int r : prob.P[i].resources) {
      for (int s = 0; s < sol.II; ++s) {
        result.resource_slot_load[r][s] +=
            ComputeFoldedOccupancy(sol.t[i], prob.P[i].latency, sol.II, s);
      }
    }
  }

  for (int r : prob.R) {
    int cap = prob.cap.count(r) ? prob.cap.at(r) : 1;
    for (int s = 0; s < sol.II; ++s) {
      int use = result.resource_slot_load[r].count(s) ? result.resource_slot_load[r][s] : 0;
      if (use > cap) {
        fail(&result.resource_slot_ok,
             "resource slot overflow " + ResourceName(r) + " slot=" + std::to_string(s) +
                 " use=" + std::to_string(use) + " cap=" + std::to_string(cap));
      }
    }
  }

  for (int a = 0; a < static_cast<int>(prob.flows.size()); ++a) {
    const auto& write_flow = prob.flows[a];
    auto it_a = internal_pos.find(a);
    if (it_a == internal_pos.end()) continue;
    int z_write = sol.z_bank[it_a->second];

    for (int b = 0; b < static_cast<int>(prob.flows.size()); ++b) {
      if (a == b) continue;
      const auto& read_flow = prob.flows[b];
      if (read_flow.read_resource < 0) continue;
      if (write_flow.mem != read_flow.mem) continue;
      auto it_b = internal_pos.find(b);
      if (it_b == internal_pos.end()) continue;
      int z_read = sol.z_bank[it_b->second];

      int write_start_time = write_flow.resident
                                 ? sol.m[read_flow.cons] + write_flow.w_off
                                 : sol.t[write_flow.prod] + write_flow.w_off;
      int write_start_slot = PositiveMod(write_start_time, sol.II);
      int write_parity_flip = (write_start_time / sol.II) & 1;
      int read_start_time = sol.t[read_flow.cons] + read_flow.delta * sol.II + read_flow.r_off;
      int read_start_slot = PositiveMod(read_start_time, sol.II);
      int read_parity_flip = (read_start_time / sol.II) & 1;
      ConflictType conflict =
          AnalyzeWriteReadConflict(write_start_slot, write_flow.w_dur, write_parity_flip,
                                   read_start_slot, read_flow.r_dur, read_parity_flip,
                                   sol.II);
      if (conflict == ConflictType::kNeedDifferent && z_write == z_read) {
        fail(&result.bank_port_ok,
             "bank port conflict requires different banks between flow " +
                 std::to_string(a) + " and flow " + std::to_string(b));
      }
      if (conflict == ConflictType::kNeedSame && z_write != z_read) {
        fail(&result.bank_port_ok,
             "bank port conflict requires same banks between flow " +
                 std::to_string(a) + " and flow " + std::to_string(b));
      }
      if (conflict == ConflictType::kImpossible) {
        fail(&result.bank_port_ok,
             "bank port conflict impossible between flow " + std::to_string(a) +
                 " and flow " + std::to_string(b));
      }
    }
  }

  return result;
}

void WriteSolutionJson(
    const std::string& path, const Problem& prob, const SolveResult& sol,
    const SolutionVerifyResult& verify,
    const std::map<std::string, int>& runtime_bank_start_phases,
    const std::map<std::string, int>& runtime_bank_read_delta_parities,
    const std::map<std::string, std::map<int, int>>& runtime_bank_reader_phases) {
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
  out << "  \"bank_slot_period\": "
      << (sol.bank_slot_period > 0 ? sol.bank_slot_period : (2 * sol.II)) << ",\n";
  out << "  \"nodes\": {\n";
  for (int i = 0; i < prob.N; ++i) {
    out << "    ";
    WriteJsonString(out, std::to_string(i));
    out << ": {\"start\": " << sol.t[i]
        << ", \"slot\": " << sol.m[i]
        << ", \"iteration\": " << sol.y[i]
        << ", \"phases\": [";
    for (size_t p = 0; p < prob.P[i].resources.size(); ++p) {
      if (p != 0) {
        out << ", ";
      }
      int resource = prob.P[i].resources[p];
      out << "{";
      out << "\"phase_id\": " << p;
      out << ", \"resource_name\": ";
      WriteJsonString(out, ResourceName(resource));
      out << ", \"start\": " << sol.t[i];
      out << ", \"end\": " << (sol.t[i] + prob.P[i].latency);
      out << ", \"duration\": " << prob.P[i].latency;
      out << "}";
    }
    out << "]}";
    out << (i + 1 == prob.N ? "\n" : ",\n");
  }
  out << "  },\n";

  out << "  \"flows\": [";
  bool first_flow = true;
  std::unordered_set<std::string> emitted_resident_buffers;
  for (int v = 0; v < static_cast<int>(prob.flows.size()); ++v) {
    const auto& flow = prob.flows[v];
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
      WriteJsonString(out, flow.read_resource < 0 ? "" : ResourceName(flow.read_resource));
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
    int prod_start =
        sol.t[flow.prod] + flow.w_off;
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
    WriteJsonString(out, flow.prod >= 0 ? std::to_string(flow.prod) : "resident");
    out << ", \"cons\": " << flow.cons;
    out << ", \"delta\": " << flow.delta;
    out << ", \"buffer_name\": ";
    WriteJsonString(out, flow.buffer_name);
    out << ", \"write_resource\": " << flow.write_resource;
    out << ", \"read_resource\": " << flow.read_resource;
    out << ", \"write_resource_name\": ";
    WriteJsonString(out, flow.write_resource < 0 ? "" : ResourceName(flow.write_resource));
    out << ", \"read_resource_name\": ";
    WriteJsonString(out, flow.read_resource < 0 ? "" : ResourceName(flow.read_resource));
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

bool IsPipelineLoopVar(const VarNode* node, const Var& pipeline_loop_var) {
  return pipeline_loop_var.defined() && node == pipeline_loop_var.get();
}

ffi::Map<Var, PrimExpr> BuildZeroSubstitutionMap(const PrimExpr& expr,
                                                 const Var& pipeline_loop_var) {
  std::unordered_set<const VarNode*> vars;
  PostOrderVisit(expr, [&](const ObjectRef& obj) {
    if (const auto* var = obj.as<VarNode>()) {
      if (!IsPipelineLoopVar(var, pipeline_loop_var)) {
        vars.insert(var);
      }
    }
  });

  ffi::Map<Var, PrimExpr> vmap;
  for (const VarNode* node : vars) {
    Var var = ffi::GetRef<Var>(node);
    vmap.Set(var, make_zero(var.dtype()));
  }
  return vmap;
}

int DetectIterOffsetFromExpr(const PrimExpr& expr, const Var& pipeline_loop_var,
                             arith::Analyzer* analyzer) {
  if (!pipeline_loop_var.defined() ||
      !UsesVar(expr, [v = pipeline_loop_var.get()](const VarNode* node) {
        return node == v;
      })) {
    return 0;
  }

  PrimExpr loop_only = expr;
  ffi::Map<Var, PrimExpr> vmap = BuildZeroSubstitutionMap(expr, pipeline_loop_var);
  if (!vmap.empty()) {
    loop_only = tir::Substitute(loop_only, vmap);
  }
  loop_only = analyzer->Simplify(loop_only);

  ffi::Array<PrimExpr> coeffs =
      arith::DetectLinearEquation(loop_only, ffi::Array<Var>{pipeline_loop_var});
  if (coeffs.size() != 2) {
    return 0;
  }

  PrimExpr coeff = analyzer->Simplify(coeffs[0]);
  PrimExpr base = analyzer->Simplify(coeffs[1]);
  const auto* coeff_int = coeff.as<IntImmNode>();
  if (coeff_int == nullptr || coeff_int->value == 0) {
    return 0;
  }

  PrimExpr unit_stride = coeff;
  PrimExpr offset_expr = analyzer->Simplify(floordiv(base, unit_stride));
  PrimExpr remainder = analyzer->Simplify(floormod(base, unit_stride));
  if (!analyzer->CanProveEqual(remainder, make_zero(remainder.dtype()))) {
    return 0;
  }

  const auto* offset_int = offset_expr.as<IntImmNode>();
  if (offset_int == nullptr) {
    return 0;
  }
  return static_cast<int>(offset_int->value);
}

int DetectIterOffsetFromRegion(const BufferRegion& region, const Var& pipeline_loop_var,
                               arith::Analyzer* analyzer) {
  int result = 0;
  bool found = false;
  for (const Range& range : region->region) {
    int dim_offset = DetectIterOffsetFromExpr(range->min, pipeline_loop_var, analyzer);
    bool uses_loop_var =
        pipeline_loop_var.defined() &&
        UsesVar(range->min, [v = pipeline_loop_var.get()](const VarNode* node) {
          return node == v;
        });
    if (!uses_loop_var) {
      continue;
    }
    if (!found) {
      result = dim_offset;
      found = true;
    } else if (result != dim_offset) {
      return 0;
    }
  }
  return found ? result : 0;
}

void DedupAccesses(std::vector<AccessInfo>* accesses) {
  std::vector<AccessInfo> deduped;
  deduped.reserve(accesses->size());
  for (const AccessInfo& access : *accesses) {
    bool exists = false;
    for (const AccessInfo& old : deduped) {
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

}  // namespace

class SunmmioStmtAccessAnalyzer : public StmtExprVisitor {
 public:
  explicit SunmmioStmtAccessAnalyzer(const PrimFunc& f) : analyzer_() {
    for (const auto& kv : f->buffer_map) {
      buffer_data_to_buffer_.Set(kv.second->data, kv.second);
    }
  }

  std::vector<AccessInfo> Collect(const Stmt& stmt,
                                  const Var& pipeline_loop_var = Var()) {
    accesses_.clear();
    pipeline_loop_var_ = pipeline_loop_var;
    VisitStmt(stmt);
    DedupAccesses(&accesses_);
    return accesses_;
  }

 private:
  void AddAccess(const BufferRegion& region, bool is_write) {
    accesses_.push_back(
        AccessInfo{region, is_write,
                   DetectIterOffsetFromRegion(region, pipeline_loop_var_, &analyzer_)});
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    Array<Range> region;
    for (const PrimExpr& index : op->indices) {
      region.push_back(Range::FromMinExtent(index, 1));
    }
    AddAccess(BufferRegion(op->buffer, region), true);
    VisitExpr(op->value);
  }

  void VisitStmt_(const EvaluateNode* op) final {
    if (const auto* call = op->value.as<CallNode>()) {
      if (call->op.same_as(dma_copy())) {
        AddAccess(NormalizeToBufferRegion(call->args[0]), false);
        AddAccess(NormalizeToBufferRegion(call->args[1]), true);
        return;
      }
      if (call->op.same_as(mma_sunmmio())) {
        AddAccess(NormalizeToBufferRegion(call->args[0]), false);
        AddAccess(NormalizeToBufferRegion(call->args[1]), false);
        BufferRegion c_region = NormalizeToBufferRegion(call->args[2]);
        AddAccess(c_region, false);
        AddAccess(c_region, true);
        return;
      }
    }
    VisitExpr(op->value);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    Array<Range> region;
    for (const PrimExpr& index : op->indices) {
      if (const auto* ramp = index.as<RampNode>()) {
        region.push_back(Range::FromMinExtent(ramp->base, ramp->lanes));
      } else {
        region.push_back(Range::FromMinExtent(index, 1));
      }
    }
    AddAccess(BufferRegion(op->buffer, region), false);
  }

  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(RegionOp::Get())) {
      AddAccess(
          NormalizeToBufferRegion(ffi::GetRef<PrimExpr>(op)),
          false);
      return;
    }

    if (op->op.same_as(builtin::address_of())) {
      if (const auto* load = op->args[0].as<BufferLoadNode>()) {
        AddAccess(BufferRegion::FullRegion(load->buffer), false);
        return;
      }
      if (const auto* var_node = op->args[0].as<VarNode>()) {
        Var data_var = ffi::GetRef<Var>(var_node);
        auto it = buffer_data_to_buffer_.find(data_var);
        if (it != buffer_data_to_buffer_.end()) {
          AddAccess(BufferRegion::FullRegion((*it).second), false);
          return;
        }
      }
    }

    if (op->op.same_as(builtin::tvm_access_ptr())) {
      if (const auto* buffer_var = op->args[1].as<VarNode>()) {
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
  SunmmioRoleMarker(ASTTraverser& traverser, const PrimFunc& func)
      : traverser_(traverser), access_analyzer_(func) {
    traverser_.clear();
  }

  Role GetRole(const StmtNode* stmt) const {
    auto it = map_.find(stmt);
    ICHECK(it != map_.end()) << "Cannot find role for stmt: "
                             << stmt->GetTypeKey();
    return it->second;
  }

  Role GetRole(const Stmt& stmt) const { return GetRole(stmt.get()); }

  std::vector<AccessInfo> GetAccesses(const Stmt& stmt,
                                      const Var& pipeline_loop_var = Var()) {
    return access_analyzer_.Collect(stmt, pipeline_loop_var);
  }

  void VisitStmt_(const EvaluateNode* op) final {
    Role role = Role::kConsumer;
    if (const auto* call = op->value.as<CallNode>()) {
      if (call->op.same_as(Op::Get("tl.dma_copy"))) {
        BufferRegion src_region = NormalizeToBufferRegion(call->args[0]);
        if (IsGlobalBuffer(src_region->buffer)) {
          role = Role::kProducer;
        }
      }
    }
    SetRole(op, role);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    Role role = Role::kProducer;
    // Reuse the legacy traverser path for role classification. It is less
    // detailed than the ILP access collector but has proven stable on large
    // kernels such as flash-attention.
    traverser_.traverse_stmt(ffi::GetRef<Stmt>(op));
    auto reads = traverser_.read_buffer_regions_;
    for (const BufferRegion& read : reads) {
      if (!IsGlobalBuffer(read->buffer)) {
        role = Role::kConsumer;
        break;
      }
    }
    SetRole(op, role);
  }

  void VisitStmt_(const SeqStmtNode* op) final {
    StmtVisitor::VisitStmt_(op);
    auto role = GetRole(op->seq[0]);
    for (const Stmt& stmt : op->seq) {
      if (role != GetRole(stmt)) {
        role = Role::kBoth;
        break;
      }
    }
    SetRole(op, role);
  }

  void VisitStmt_(const IfThenElseNode* op) final {
    StmtVisitor::VisitStmt_(op);
    auto role = GetRole(op->then_case);
    if (op->else_case.defined() && role != GetRole(op->else_case.value())) {
      role = Role::kBoth;
    }
    SetRole(op, role);
  }

  void VisitStmt_(const BlockRealizeNode* op) final {
    StmtVisitor::VisitStmt_(op);
    SetRole(op, GetRole(op->block));
  }

  template <class NodeType>
  void HandleBodyStmt(const NodeType* op) {
    StmtVisitor::VisitStmt_(op);
    SetRole(op, GetRole(op->body));
  }

  void VisitStmt_(const ForNode* op) final { HandleBodyStmt(op); }
  void VisitStmt_(const LetStmtNode* op) final { HandleBodyStmt(op); }
  void VisitStmt_(const AttrStmtNode* op) final { HandleBodyStmt(op); }
  void VisitStmt_(const AssertStmtNode* op) final { HandleBodyStmt(op); }
  void VisitStmt_(const BlockNode* op) final { HandleBodyStmt(op); }
  void VisitStmt_(const AllocateNode* op) final { HandleBodyStmt(op); }
  void VisitStmt_(const DeclBufferNode* op) final { HandleBodyStmt(op); }

 private:
  void SetRole(const StmtNode* stmt, Role role) { map_[stmt] = role; }

  std::unordered_map<const StmtNode*, Role> map_;
  ASTTraverser traverser_;
  SunmmioStmtAccessAnalyzer access_analyzer_;
};

class SunmmioExprAnalyzer : public StmtExprVisitor {
 public:
  SunmmioExprAnalyzer() {}

  void Analyze(const PrimExpr& expr) {
    loop_cost_ = 0;
    load_times = 0;
    flops_ = 0;
    args_.clear();
    constants_.clear();
    vars_.clear();
    StmtExprVisitor::VisitExpr(expr);
  }

 private:
  void VisitExpr_(const MulNode* op) final {
    auto a = op->a;
    auto b = op->b;
    flops_ += 1;
    if (const auto* a_int = a.as<IntImmNode>()) {
      if (const auto* b_int = b.as<IntImmNode>()) {
        return;
      }
      if (a_int->value <= 32) {
        loop_cost_ += 2;
        StmtExprVisitor::VisitExpr(op->b);
        return;
      }
    }
    if (const auto* b_int = b.as<IntImmNode>()) {
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

  void VisitExpr_(const SubNode* op) final {
    loop_cost_ += 4;
    flops_ += 1;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const AddNode* op) final {
    loop_cost_ += 4;
    flops_ += 1;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const MaxNode* op) final {
    loop_cost_ += 3;
    flops_ += 1;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const MinNode* op) final {
    loop_cost_ += 3;
    flops_ += 1;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const CastNode* op) final {
    loop_cost_ += 3;
    StmtExprVisitor::VisitExpr(op->value);
  }

  void VisitExpr_(const IntImmNode* op) final {
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

  void VisitExpr_(const FloatImmNode* op) final {
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

  void VisitExpr_(const VarNode* op) final {
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

  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(Op::Get("tir.exp2"))) {
      loop_cost_ += 10;
      flops_ += 3;
      StmtExprVisitor::VisitExpr(op->args[0]);
    } else if (op->op.same_as(Op::Get("tl.infinity"))) {
      bool insert = true;
      for (auto it : constants_) {
        if (ExprDeepEqual()(it, FloatImm(DataType::Float(16),
                                         std::numeric_limits<float>::infinity()))) {
          insert = false;
          break;
        }
      }
      if (insert) {
        constants_.push_back(
            FloatImm(DataType::Float(16), std::numeric_limits<float>::infinity()));
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

  void VisitExpr_(const LENode* op) final {
    loop_cost_ += 3;
    flops_ += 1;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
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
  CommandSpec spec;

  TemplateCommand(int id, const Stmt& stmt)
      : id(id), name("cmd_" + std::to_string(id)), stmt(stmt) {}
};

bool IsCopyStage(const TemplateCommand& cmd) {
  bool has_shared_write = false;
  bool has_global_read = false;
  for (const AccessInfo& access : cmd.accesses) {
    if (access.is_write && IsSunmmioSharedBuffer(access.buffer())) {
      has_shared_write = true;
    }
    if (!access.is_write && IsGlobalBuffer(access.buffer())) {
      has_global_read = true;
    }
  }
  return has_shared_write && has_global_read;
}

bool IsProducerLike(const TemplateCommand& cmd) {
  return cmd.role == Role::kProducer || (cmd.role == Role::kBoth && IsCopyStage(cmd));
}

bool IsConsumerLike(const TemplateCommand& cmd) {
  return cmd.role == Role::kConsumer || (cmd.role == Role::kBoth && !IsCopyStage(cmd));
}

std::string SummarizeStmtForName(const Stmt& stmt) {
  auto buffer_scope = [](const Buffer& buffer) {
    return buffer.scope().empty() ? std::string("default")
                                  : std::string(buffer.scope());
  };
  auto buffer_label = [&](const Buffer& buffer) {
    return buffer->name + "@" + buffer_scope(buffer);
  };
  auto expr_kind = [&](const PrimExpr& expr) -> std::string {
    if (expr.as<IntImmNode>()) {
      if (const auto* imm = expr.as<IntImmNode>()) {
        return imm->value == 0 ? "const0" : "const";
      }
    }
    if (expr.as<FloatImmNode>()) {
      if (const auto* imm = expr.as<FloatImmNode>()) {
        return imm->value == 0.0 ? "const0" : "const";
      }
    }
    if (const auto* load = expr.as<BufferLoadNode>()) {
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
    if (const auto* call = expr.as<CallNode>()) {
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
  if (const auto* eval = stmt.as<EvaluateNode>()) {
    if (const auto* call = eval->value.as<CallNode>()) {
      if (call->op.same_as(Op::Get("tl.dma_copy"))) {
        BufferRegion src = NormalizeToBufferRegion(call->args[0]);
        BufferRegion dst = NormalizeToBufferRegion(call->args[1]);
        return "dma_copy(" + buffer_label(src->buffer) + "->" +
               buffer_label(dst->buffer) + ")";
      }
    }
    return "evaluate";
  }
  if (const auto* block = stmt.as<BlockRealizeNode>()) {
    if (const auto* eval = block->block->body.as<EvaluateNode>()) {
      if (const auto* call = eval->value.as<CallNode>()) {
        if (call->op.same_as(Op::Get("tl.mma_sunmmio"))) {
          auto A = call->args[0].as<CallNode>();
          auto B = call->args[1].as<CallNode>();
          if (A && B && A->args.size() >= 4 && B->args.size() >= 4) {
            return "mma_sunmmio(" +
                   std::to_string(A->args[2].as<IntImmNode>()->value) + "x" +
                   std::to_string(B->args[3].as<IntImmNode>()->value) + "x" +
                   std::to_string(A->args[3].as<IntImmNode>()->value) + ")";
          }
          return "mma_sunmmio";
        }
      }
    }
    if (block->block->name_hint == "reduce_tile_op") {
      std::string reduce_kind = "reduce";
      std::string reduce_dst;
      bool found_reduce = false;
      PostOrderVisit(block->block->body, [&](const ObjectRef& obj) {
        if (found_reduce) {
          return;
        }
        if (const auto* eval = obj.as<EvaluateNode>()) {
          if (const auto* call = eval->value.as<CallNode>()) {
            if (call->op.same_as(Op::Get("tl.vector_core_in_tile_reduce")) &&
                !call->args.empty()) {
              if (const auto* kind = call->args[0].as<StringImmNode>()) {
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
      return reduce_dst.empty() ? "reduce_tile_op(" + reduce_kind + ")"
                                : "reduce_tile_op(" + reduce_kind + " -> " + reduce_dst + ")";
    }
    if (block->block->name_hint.size() != 0) {
      return "block:" + block->block->name_hint;
    }
    return "block";
  }
  if (const auto* loop = stmt.as<ForNode>()) {
    std::string summary = "for";
    PostOrderVisit(loop->body, [&](const ObjectRef& obj) {
      if (summary != "for") {
        return;
      }
      if (const auto* store = obj.as<BufferStoreNode>()) {
        summary = "for store(" + buffer_label(store->buffer) + " := " +
                  expr_kind(store->value) + ")";
        return;
      }
      if (const auto* eval = obj.as<EvaluateNode>()) {
        if (const auto* call = eval->value.as<CallNode>()) {
          if (call->op.same_as(Op::Get("tl.vector_core_in_tile_reduce")) &&
              !call->args.empty()) {
            if (const auto* kind = call->args[0].as<StringImmNode>()) {
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

int GetPingPongMemoryKind(const Buffer& buffer) {
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

bool CommandUsesResource(const CommandSpec& spec, int resource) {
  return std::find(spec.resources.begin(), spec.resources.end(), resource) !=
         spec.resources.end();
}

FlowSpec MakeInternalFlowSpec(const Problem& problem, int prod, int cons, int delta,
                              int mem, const std::string& buffer_name) {
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

FlowSpec MakeResidentFlowSpec(const Problem& problem, int cons, int mem,
                              const std::string& buffer_name) {
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

std::string ExtractBufferNameFromCommandLabel(const std::string& name) {
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

int RuntimeVersionCount(const Buffer& buffer, int iterations) {
  if (iterations <= 0 || GetPingPongMemoryKind(buffer) >= 0) {
    return 1;
  }
  return iterations;
}

int RuntimeBankedVersionCount(const Buffer& buffer, int iterations) {
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
    const std::vector<TemplateCommand>& commands,
    const std::vector<Buffer>& versioned_buffers,
    const std::vector<Buffer>& runtime_banked_buffers,
    const Var& pipeline_loop_var, const SolveResult& sol, int stage_count,
    const std::map<std::string, int>& runtime_bank_start_phases,
    const std::map<std::string, std::map<int, int>>& runtime_bank_writer_phases) {
  // Final runtime multiversion annotations must be derived from the selected
  // steady-state window of the optimized solution, not from the original
  // num_stages request before stage shrinking.
  stage_count = std::max(stage_count, 0);
  std::unordered_set<const BufferNode*> candidates;
  std::unordered_set<const BufferNode*> banked_candidates;
  for (const Buffer& buffer : runtime_banked_buffers) {
    banked_candidates.insert(buffer.get());
  }
  for (const Buffer& buffer : versioned_buffers) {
    int version_count =
        banked_candidates.count(buffer.get())
            ? RuntimeBankedVersionCount(buffer, stage_count)
            : RuntimeVersionCount(buffer, stage_count);
    if (version_count > 1) {
      candidates.insert(buffer.get());
    }
  }
  if (candidates.empty()) {
    return {};
  }

  int max_iter_offset = 0;
  for (const TemplateCommand& cmd : commands) {
    for (const AccessInfo& access : cmd.accesses) {
      max_iter_offset = std::max(max_iter_offset, access.iter_offset);
    }
  }

  const int expanded_iters =
      std::max(2, stage_count + max_iter_offset + 1);
  std::unordered_map<const BufferNode*, std::vector<ScheduledAccessWindow>>
      windows_by_buffer;
  windows_by_buffer.reserve(candidates.size());

  struct ExpandedCommand {
    int iter{0};
    const TemplateCommand* cmd{nullptr};
  };

  std::vector<int> producer_ids;
  std::vector<int> body_ids;
  producer_ids.reserve(commands.size());
  body_ids.reserve(commands.size());
  for (const TemplateCommand& cmd : commands) {
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
            [](const ExpandedCommand& a, const ExpandedCommand& b) {
              if (a.iter != b.iter) {
                return a.iter < b.iter;
              }
              return a.cmd->id < b.cmd->id;
            });

  for (const ExpandedCommand& expanded : expanded_commands) {
    const int start = sol.t[expanded.cmd->id] + expanded.iter * sol.II;
    const int end = start + expanded.cmd->spec.latency;
    for (const AccessInfo& access : expanded.cmd->accesses) {
      const BufferNode* buf = access.buffer().get();
      if (!candidates.count(buf)) {
        continue;
      }
      windows_by_buffer[buf].push_back(
          ScheduledAccessWindow{
              expanded.iter + access.iter_offset,
              expanded.iter,
              expanded.cmd->id,
              start,
              end,
              access.is_write,
              MaterializeBufferRegion(access.region, pipeline_loop_var,
                                      expanded.iter)});
    }
  }

  auto region_sets_intersect = [](const std::vector<BufferRegion>& lhs,
                                  const std::vector<BufferRegion>& rhs) {
    for (const BufferRegion& lhs_region : lhs) {
      for (const BufferRegion& rhs_region : rhs) {
        if (PipelineRegionIntersect(lhs_region->region, rhs_region->region)) {
          return true;
        }
      }
    }
    return false;
  };

  auto resolve_write_bank = [&](const Buffer& buffer,
                                const BufferInstanceLifetime& lifetime) {
    if (!banked_candidates.count(buffer.get())) {
      return -1;
    }
    int phase = 0;
    auto it_writer = runtime_bank_writer_phases.find(buffer->name);
    if (it_writer != runtime_bank_writer_phases.end()) {
      auto it_phase = it_writer->second.find(lifetime.first_write_cmd_id);
      if (it_phase != it_writer->second.end()) {
        phase = it_phase->second;
      }
    } else {
      auto it_start = runtime_bank_start_phases.find(buffer->name);
      if (it_start != runtime_bank_start_phases.end()) {
        phase = it_start->second;
      }
    }
    return PositiveMod(lifetime.command_iter + phase, 2);
  };

  std::vector<Buffer> runtime_multiversion_buffers;
  for (const Buffer& buffer : versioned_buffers) {
    if (!candidates.count(buffer.get())) {
      continue;
    }

    auto it_windows = windows_by_buffer.find(buffer.get());
    if (it_windows == windows_by_buffer.end()) {
      continue;
    }

    std::unordered_map<int, std::vector<const ScheduledAccessWindow*>>
        windows_by_logical_iter;
    for (const ScheduledAccessWindow& window : it_windows->second) {
      windows_by_logical_iter[window.logical_iter].push_back(&window);
    }

    std::vector<BufferInstanceLifetime> lifetimes;
    lifetimes.reserve(windows_by_logical_iter.size());
    for (const auto& kv : windows_by_logical_iter) {
      int first_write_start = std::numeric_limits<int>::max();
      const ScheduledAccessWindow* first_write_window = nullptr;
      for (const ScheduledAccessWindow* window : kv.second) {
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
      lifetime.logical_iter = kv.first;
      lifetime.command_iter =
          first_write_window == nullptr ? kv.first : first_write_window->command_iter;
      lifetime.first_write_cmd_id =
          first_write_window == nullptr ? -1 : first_write_window->cmd_id;
      lifetime.start = first_write_start;
      lifetime.end = first_write_start;
      for (const ScheduledAccessWindow* window : kv.second) {
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
        lifetime.write_bank = resolve_write_bank(buffer, lifetime);
        lifetimes.push_back(std::move(lifetime));
      }
    }

    std::sort(lifetimes.begin(), lifetimes.end(),
              [](const BufferInstanceLifetime& a,
                 const BufferInstanceLifetime& b) {
                if (a.start != b.start) {
                  return a.start < b.start;
                }
                return a.logical_iter < b.logical_iter;
              });

    bool needs_runtime_multiversion = false;
    bool is_banked = banked_candidates.count(buffer.get()) != 0;
    for (size_t i = 0; i < lifetimes.size() && !needs_runtime_multiversion; ++i) {
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

TimeWindowOrderResult BuildTimeWindowOrders(
    const std::vector<TemplateCommand>& commands, int iterations,
    const SolveResult& sol) {
  TimeWindowOrderResult result;
  const int stage_count = CeilDiv(sol.makespan, std::max(1, sol.II));
  const int prologue_end = std::max(0, stage_count - 1) * sol.II;
  const int body_begin = prologue_end;
  const int body_end = prologue_end + sol.II;
  const int epilogue_begin = body_end;
  const int epilogue_end = prologue_end+sol.makespan;

  int max_iter = stage_count;
  std::vector<ExpandedOrderEntry> expanded;  
  expanded.reserve(max_iter * commands.size());
  for (int iter = 0; iter < max_iter; ++iter) {
    for (const TemplateCommand& cmd : commands) {
      int id = cmd.id;
      expanded.push_back(ExpandedOrderEntry{iter, id, sol.t[id] + iter * sol.II});
    }
  }

  std::sort(expanded.begin(), expanded.end(),
            [](const ExpandedOrderEntry& a, const ExpandedOrderEntry& b) {
              if (a.absolute_start != b.absolute_start) {
                return a.absolute_start < b.absolute_start;
              }
              if (a.iter != b.iter) {
                return a.iter < b.iter;
              }
              return a.id < b.id;
            });

  for (const ExpandedOrderEntry& entry : expanded) {
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

BufferRegion MaterializeBufferRegion(const BufferRegion& region, const Var& loop_var,
                                     int iter) {
  if (!loop_var.defined()) {
    return region;
  }
  ffi::Map<Var, PrimExpr> vmap;
  vmap.Set(loop_var, make_const(loop_var.dtype(), iter));
  Array<Range> materialized;
  for (const Range& rng : region->region) {
    PrimExpr min = tir::Substitute(rng->min, vmap);
    PrimExpr extent = tir::Substitute(rng->extent, vmap);
    materialized.push_back(Range::FromMinExtent(min, extent));
  }
  return BufferRegion(region->buffer, materialized);
}

std::vector<Buffer> DetectVersionedBuffers(
    const std::vector<TemplateCommand>& commands) {
  std::set<Buffer> used_buffers;
  std::unordered_set<const BufferNode*> consumer_used;
  std::unordered_set<const BufferNode*> producer_used;
  std::unordered_set<const BufferNode*> self_dependent_buffers;
  std::unordered_map<const BufferNode*, int> first_write_index;
  std::unordered_map<const BufferNode*, std::vector<int>> write_indexes;
  std::unordered_map<const BufferNode*, int> first_read_index;
  std::unordered_map<const BufferNode*, int> last_read_index;
  std::vector<Buffer> versioned_buffers;
  auto mark_versioned = [&](const Buffer& buffer) {
    if (std::find(versioned_buffers.begin(), versioned_buffers.end(), buffer) ==
        versioned_buffers.end()) {
      versioned_buffers.push_back(buffer);
    }
  };

  for (int i = 0; i < static_cast<int>(commands.size()); ++i) {
    bool is_producer = IsProducerLike(commands[i]);
    bool is_consumer = IsConsumerLike(commands[i]);
    std::unordered_set<const BufferNode*> reads_in_cmd;
    std::unordered_set<const BufferNode*> writes_in_cmd;
    for (const AccessInfo& access : commands[i].accesses) {
      if (IsGlobalBuffer(access.buffer())) {
        continue;
      }
      used_buffers.insert(access.buffer());
      const BufferNode* buf = access.buffer().get();
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
    for (const BufferNode* buf : writes_in_cmd) {
      if (reads_in_cmd.count(buf)) {
        self_dependent_buffers.insert(buf);
      }
    }
  }

  for (const Buffer& buffer : used_buffers) {
    const BufferNode* buf = buffer.get();
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
      if (r != first_read_index.end() && w != first_write_index.end() && r->second > w->second) {
        mark_versioned(buffer);
        continue;
      }
    }
    auto it_last_r = last_read_index.find(buf);
    if (it_w != first_write_index.end() && it_last_r != last_read_index.end() &&
        it_w->second < it_last_r->second && IsCopyStage(commands[it_w->second])) {
      mark_versioned(buffer);
    }
  }

  bool updated = true;
  while (updated) {
    updated = false;
    for (const Buffer& buffer : used_buffers) {
      if (std::find(versioned_buffers.begin(), versioned_buffers.end(), buffer) !=
          versioned_buffers.end()) {
        continue;
      }
      const BufferNode* buf = buffer.get();
      if (self_dependent_buffers.count(buf)) {
        continue;
      }
      auto it_writes = write_indexes.find(buf);
      auto it_first_w = first_write_index.find(buf);
      auto it_first_r = first_read_index.find(buf);
      if (it_writes == write_indexes.end() || it_writes->second.empty() ||
          it_first_w == first_write_index.end() || it_first_r == first_read_index.end()) {
        continue;
      }
      bool can_propagate = it_first_w->second < it_first_r->second;
      for (int idx : it_writes->second) {
        for (const AccessInfo& access : commands[idx].accesses) {
          if (access.is_write || IsGlobalBuffer(access.buffer())) {
            continue;
          }
          if (first_write_index.find(access.buffer().get()) == first_write_index.end()) {
            continue;
          }
          if (std::find(versioned_buffers.begin(), versioned_buffers.end(), access.buffer()) ==
              versioned_buffers.end()) {
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

void BuildTemplateDependencyGraph(
    const std::vector<TemplateCommand>& commands, int iter_mod,
    const std::vector<Buffer>& versioned_buffers, const Var& pipeline_loop_var,
    Problem* problem) {
  std::unordered_set<const BufferNode*> versioned;
  std::unordered_set<const BufferNode*> bank_rotating_versioned;
  for (const Buffer& buffer : versioned_buffers) {
    versioned.insert(buffer.get());
    if (GetPingPongMemoryKind(buffer) >= 0) {
      bank_rotating_versioned.insert(buffer.get());
    }
  }

  problem->flows.clear();
  problem->dep_edges.clear();
  problem->delta.clear();

  struct ExpandedCommand {
    int template_id{-1};
    int iter{-1};
    const TemplateCommand* cmd{nullptr};
  };
  enum class AccessType : uint8_t { kRead, kWrite };
  struct AccessRecord {
    BufferRegion region;
    int expanded_idx{-1};
    int access_idx{-1};
    AccessType type{AccessType::kRead};
  };

  std::map<std::pair<int, int>, int> best_delta;
  std::map<std::tuple<int, int, const BufferNode*, int, int>, int> flow_key_to_index;
  std::set<std::pair<int, int>> satisfied_consumer_mem;

  auto version_mod = [&](const BufferNode* buf) {
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

#if 0
  /*
   * Old direct-search implementation kept for reference.
   *
   * It stays at template granularity and searches backward for the nearest
   * conflicting access while using access-level iter_offset to decide whether
   * two accesses alias the same physical version.
   *
   * The active implementation below instead expands the steady-state body
   * across `num_stages` iterations, replays RAW/WAR/WAW on the expanded
   * stream, and then compresses the result back to template-level
   * `(src_id, dst_id, delta)` edges.
   */
  auto compatible_version = [&](const BufferNode* buf, int src_iter_offset,
                                int dst_iter_offset, int delta) {
    if (!versioned.count(buf) || iter_mod <= 0) {
      return true;
    }
    return PositiveMod(src_iter_offset, iter_mod) ==
           PositiveMod(delta + dst_iter_offset, iter_mod);
  };

  auto regions_conflict = [&](const AccessInfo& src_access,
                              const AccessInfo& dst_access, int delta) {
    BufferRegion src_region =
        MaterializeBufferRegion(src_access.region, pipeline_loop_var, /*iter=*/0);
    BufferRegion dst_region =
        MaterializeBufferRegion(dst_access.region, pipeline_loop_var, /*iter=*/delta);
    return PipelineRegionIntersect(src_region->region, dst_region->region);
  };

  auto maybe_record_edge_old = [&](int src_id, int dst_id, int delta) {
    if (delta < 0) {
      return;
    }
    std::pair<int, int> key{src_id, dst_id};
    auto it = best_delta.find(key);
    if (it == best_delta.end() || delta < it->second) {
      best_delta[key] = delta;
    }
  };

  auto maybe_record_flow_old = [&](int src_id, int dst_id, const BufferNode* buf,
                                   int src_access_idx, int dst_access_idx,
                                   int mem) {
    if (mem < 0 || src_id == dst_id) {
      return;
    }
    int write_resource = GetMemoryWriteResource(mem);
    int read_resource = GetMemoryReadResource(mem);
    if (!CommandUsesResource(problem->P[src_id], write_resource) ||
        !CommandUsesResource(problem->P[dst_id], read_resource)) {
      return;
    }
    auto flow_key =
        std::make_tuple(src_id, dst_id, buf, src_access_idx, dst_access_idx);
    if (seen_flow_keys.insert(flow_key).second) {
      int delta = 0;
      auto edge_it = best_delta.find({src_id, dst_id});
      if (edge_it != best_delta.end()) {
        delta = edge_it->second;
      }
      problem->flows.push_back(
          MakeInternalFlowSpec(*problem, src_id, dst_id, delta, mem, buf->name));
    }
  };

  int max_delta_to_search = std::max(1, iter_mod);
  for (int dst_id = 0; dst_id < static_cast<int>(commands.size()); ++dst_id) {
    const TemplateCommand& dst_cmd = commands[dst_id];
    for (int dst_access_idx = 0;
         dst_access_idx < static_cast<int>(dst_cmd.accesses.size());
         ++dst_access_idx) {
      const AccessInfo& dst_access = dst_cmd.accesses[dst_access_idx];
      const BufferNode* buf = dst_access.buffer().get();

      if (!dst_access.is_write) {
        bool found = false;
        for (int delta = 0; delta <= max_delta_to_search && !found; ++delta) {
          int src_id = (delta == 0) ? (dst_id - 1)
                                    : (static_cast<int>(commands.size()) - 1);
          for (; src_id >= 0 && !found; --src_id) {
            const TemplateCommand& src_cmd = commands[src_id];
            for (int src_access_idx =
                     static_cast<int>(src_cmd.accesses.size()) - 1;
                 src_access_idx >= 0; --src_access_idx) {
              const AccessInfo& src_access = src_cmd.accesses[src_access_idx];
              if (!src_access.is_write || src_access.buffer().get() != buf) {
                continue;
              }
              if (!compatible_version(buf, src_access.iter_offset,
                                      dst_access.iter_offset, delta)) {
                continue;
              }
              if (!regions_conflict(src_access, dst_access, delta)) {
                continue;
              }
              maybe_record_edge_old(src_id, dst_id, delta);
              maybe_record_flow_old(src_id, dst_id, buf, src_access_idx,
                                    dst_access_idx,
                                    GetPingPongMemoryKind(dst_access.buffer()));
              found = true;
              break;
            }
          }
        }
        continue;
      }

      bool stop_on_prior_write = false;
      for (int delta = 0; delta <= max_delta_to_search && !stop_on_prior_write;
           ++delta) {
        int src_id = (delta == 0) ? (dst_id - 1)
                                  : (static_cast<int>(commands.size()) - 1);
        for (; src_id >= 0 && !stop_on_prior_write; --src_id) {
          const TemplateCommand& src_cmd = commands[src_id];
          for (int src_access_idx =
                   static_cast<int>(src_cmd.accesses.size()) - 1;
               src_access_idx >= 0; --src_access_idx) {
            const AccessInfo& src_access = src_cmd.accesses[src_access_idx];
            if (src_access.buffer().get() != buf) {
              continue;
            }
            if (!compatible_version(buf, src_access.iter_offset,
                                    dst_access.iter_offset, delta)) {
              continue;
            }
            if (!regions_conflict(src_access, dst_access, delta)) {
              continue;
            }
            maybe_record_edge_old(src_id, dst_id, delta);
            if (src_access.is_write) {
              stop_on_prior_write = true;
              break;
            }
          }
        }
      }
    }
  }
#endif

  auto maybe_record_edge = [&](int src_id, int dst_id, int delta) {
    if (delta < 0) {
      return;
    }
    std::pair<int, int> key{src_id, dst_id};
    auto it = best_delta.find(key);
    if (it == best_delta.end() || delta < it->second) {
      best_delta[key] = delta;
    }
  };

  auto maybe_record_flow = [&](int src_id, int dst_id, int delta,
                               const BufferNode* buf, int src_access_idx,
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
      problem->flows.push_back(
          MakeInternalFlowSpec(*problem, src_id, dst_id, delta, mem, buf->name));
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
  for (const TemplateCommand& cmd : commands) {
    if (IsProducerLike(cmd)) {
      producer_ids.push_back(cmd.id);
    } else {
      body_ids.push_back(cmd.id);
    }
  }

  int steady_state_iters = std::max(1, iter_mod);
  for (const Buffer& buffer : versioned_buffers) {
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
            [](const ExpandedCommand& a, const ExpandedCommand& b) {
              if (a.iter != b.iter) {
                return a.iter < b.iter;
              }
              return a.template_id < b.template_id;
            });

  auto access_version = [&](const BufferNode* buf,
                            const ExpandedCommand& command,
                            const AccessInfo& access) {
    int mod = version_mod(buf);
    if (mod <= 0) {
      return command.iter + access.iter_offset;
    }
    return PositiveMod(command.iter + access.iter_offset, mod);
  };

  auto materialize_access = [&](const ExpandedCommand& command,
                                const AccessInfo& access) {
    return MaterializeBufferRegion(access.region, pipeline_loop_var, command.iter);
  };

  std::unordered_map<const BufferNode*, std::vector<AccessRecord>>
      buffer_access_history;
  buffer_access_history.reserve(versioned.size() + commands.size());

  struct ResidentCandidate {
    int cmd_id{-1};
    int mem{-1};
    std::string buffer_name;
  };
  std::vector<ResidentCandidate> resident_candidates;

  for (int curr_idx = 0; curr_idx < static_cast<int>(expanded_commands.size());
       ++curr_idx) {
    const ExpandedCommand& curr_cmd = expanded_commands[curr_idx];

    for (int dst_access_idx = 0;
         dst_access_idx < static_cast<int>(curr_cmd.cmd->accesses.size());
         ++dst_access_idx) {
      const AccessInfo& dst_access = curr_cmd.cmd->accesses[dst_access_idx];
      if (dst_access.is_write) {
        continue;
      }
      const BufferNode* buf = dst_access.buffer().get();
      auto hist_it = buffer_access_history.find(buf);
      if (hist_it == buffer_access_history.end()) {
        continue;
      }
      BufferRegion dst_region = materialize_access(curr_cmd, dst_access);
      for (auto it = hist_it->second.rbegin(); it != hist_it->second.rend(); ++it) {
        const ExpandedCommand& src_cmd = expanded_commands[it->expanded_idx];
        const AccessInfo& src_access = src_cmd.cmd->accesses[it->access_idx];
        if (versioned.count(buf) &&
            access_version(buf, src_cmd, src_access) !=
                access_version(buf, curr_cmd, dst_access)) {
          continue;
        }
        if (it->type != AccessType::kWrite ||
            !PipelineRegionIntersect(dst_region->region, it->region->region)) {
          continue;
        }
        maybe_record_edge(src_cmd.template_id, curr_cmd.template_id,
                          curr_cmd.iter - src_cmd.iter);
        int mem = GetPingPongMemoryKind(dst_access.buffer());
        bool has_concrete_flow =
            maybe_record_flow(src_cmd.template_id, curr_cmd.template_id,
                              curr_cmd.iter - src_cmd.iter, buf,
                              it->access_idx, dst_access_idx, mem);
        if (has_concrete_flow) {
          satisfied_consumer_mem.insert(
              std::make_pair(curr_cmd.template_id, mem));
        }
        break;
      }
    }

    for (int dst_access_idx = 0;
         dst_access_idx < static_cast<int>(curr_cmd.cmd->accesses.size());
         ++dst_access_idx) {
      const AccessInfo& dst_access = curr_cmd.cmd->accesses[dst_access_idx];
      if (!dst_access.is_write) {
        continue;
      }
      const BufferNode* buf = dst_access.buffer().get();
      auto hist_it = buffer_access_history.find(buf);
      if (hist_it == buffer_access_history.end()) {
        continue;
      }
      BufferRegion dst_region = materialize_access(curr_cmd, dst_access);
      for (auto it = hist_it->second.rbegin(); it != hist_it->second.rend(); ++it) {
        const ExpandedCommand& src_cmd = expanded_commands[it->expanded_idx];
        const AccessInfo& src_access = src_cmd.cmd->accesses[it->access_idx];
        if (versioned.count(buf) &&
            access_version(buf, src_cmd, src_access) !=
                access_version(buf, curr_cmd, dst_access)) {
          continue;
        }
        if (!PipelineRegionIntersect(dst_region->region, it->region->region)) {
          continue;
        }
        maybe_record_edge(src_cmd.template_id, curr_cmd.template_id,
                          curr_cmd.iter - src_cmd.iter);
        if (it->type == AccessType::kWrite) {
          break;
        }
      }
    }

    for (int access_idx = 0;
         access_idx < static_cast<int>(curr_cmd.cmd->accesses.size()); ++access_idx) {
      const AccessInfo& access = curr_cmd.cmd->accesses[access_idx];
      buffer_access_history[access.buffer().get()].push_back(
          AccessRecord{materialize_access(curr_cmd, access), curr_idx,
                       access_idx,
                       access.is_write ? AccessType::kWrite
                                       : AccessType::kRead});
    }
  }

  for (const auto& kv : best_delta) {
    problem->dep_edges.push_back(kv.first);
    problem->delta[EdgeKey(kv.first.first, kv.first.second)] = kv.second;
  }

  for (const TemplateCommand& cmd : commands) {
    for (int mem = 0; mem <= 1; ++mem) {
      int read_resource = GetMemoryReadResource(mem);
      if (!CommandUsesResource(problem->P[cmd.id], read_resource)) {
        continue;
      }
      std::string buffer_name = "resident_" + MemName(mem) + "_cmd_" +
                                std::to_string(cmd.id);
      for (const AccessInfo& access : cmd.accesses) {
        if (access.is_write) {
          continue;
        }
        if (GetPingPongMemoryKind(access.buffer()) == mem) {
          buffer_name = access.buffer()->name;
          break;
        }
      }
      resident_candidates.push_back(
          ResidentCandidate{cmd.id, mem, buffer_name});
    }
  }

  std::set<std::pair<int, int>> emitted_resident_consumer_mem;
  for (const ResidentCandidate& candidate : resident_candidates) {
    if (satisfied_consumer_mem.count(
            std::make_pair(candidate.cmd_id, candidate.mem)) != 0) {
      continue;
    }
    if (!emitted_resident_consumer_mem.insert(
            std::make_pair(candidate.cmd_id, candidate.mem)).second) {
      continue;
    }
    problem->flows.push_back(
        MakeResidentFlowSpec(*problem, candidate.cmd_id, candidate.mem,
                             candidate.buffer_name));
  }

}

int ResourceLowerBound(const Problem& prob) {
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
      if (std::find(prob.P[i].resources.begin(), prob.P[i].resources.end(), r) !=
          prob.P[i].resources.end()) {
        total += prob.P[i].latency;
      }
    }
    lb = std::max(lb, std::max(1, int((total + cap - 1) / cap)));
  }
  return lb;
}

HighsInt AddCol(Highs& highs, double lower, double upper, double cost,
                bool is_integer) {
  HighsStatus st = highs.addCol(cost, lower, upper, 0, nullptr, nullptr);
  ICHECK(st == HighsStatus::kOk) << "addCol failed";
  HighsInt col = highs.getNumCol() - 1;
  if (is_integer) {
    highs.changeColIntegrality(col, HighsVarType::kInteger);
  }
  return col;
}

HighsStatus AddRow(Highs& highs, double lower, double upper,
                   const std::vector<HighsInt>& idx,
                   const std::vector<double>& val) {
  const HighsInt* idx_ptr = idx.empty() ? nullptr : idx.data();
  const double* val_ptr = val.empty() ? nullptr : val.data();
  return highs.addRow(lower, upper, HighsInt(idx.size()), idx_ptr, val_ptr);
}

void MergeLinearTerms(const std::vector<HighsInt>& idx,
                      const std::vector<double>& val,
                      std::vector<HighsInt>& merged_idx,
                      std::vector<double>& merged_val) {
  std::map<HighsInt, double> acc;
  for (size_t k = 0; k < idx.size(); ++k) {
    acc[idx[k]] += val[k];
  }
  merged_idx.clear();
  merged_val.clear();
  for (const auto& kv : acc) {
    if (kv.second == 0.0) {
      continue;
    }
    merged_idx.push_back(kv.first);
    merged_val.push_back(kv.second);
  }
}

void AddLeq(Highs& highs, const std::vector<HighsInt>& idx,
            const std::vector<double>& val, double rhs) {
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

void AddEq(Highs& highs, const std::vector<HighsInt>& idx,
           const std::vector<double>& val, double rhs) {
  std::vector<HighsInt> merged_idx;
  std::vector<double> merged_val;
  MergeLinearTerms(idx, val, merged_idx, merged_val);
  AddRow(highs, rhs, rhs, merged_idx, merged_val);
}

ModelVars BuildModel(Highs& highs, const Problem& prob, int II, bool optimize_t,
                     int threads) {
  highs.clear();
  highs.setOptionValue("output_flag", true);
  highs.setOptionValue("log_to_console", true);
  highs.setOptionValue("mip_report_level", 2);
  highs.setOptionValue("threads", threads);
  highs.setOptionValue("parallel", "on");
  highs.changeObjectiveSense(ObjSense::kMinimize);

  int max_delta = 0;
  int max_latency = 0;
  for (const auto& kv : prob.delta) {
    max_delta = std::max(max_delta, kv.second);
  }
  for (const auto& spec : prob.P) {
    max_latency = std::max(max_latency, spec.latency);
  }
  int time_ub = prob.Tmax + max_delta * II + max_latency;

  ModelVars vars;
  vars.col_t.resize(prob.N);
  vars.col_y.resize(prob.N);
  vars.col_m.resize(prob.N);
  vars.col_x.assign(prob.N, std::vector<HighsInt>(II, -1));
  vars.col_a.assign(prob.N, std::vector<HighsInt>(II, -1));

  for (int v = 0; v < static_cast<int>(prob.flows.size()); ++v) {
    vars.internal_flow_ids.push_back(v);
  }

  for (int i = 0; i < prob.N; ++i) {
    vars.col_t[i] = AddCol(highs, 0, time_ub, 0, true);
    vars.col_y[i] = AddCol(highs, 0, time_ub, 0, true);
    vars.col_m[i] = AddCol(highs, 0, II - 1, 0, true);
  }
  for (int i = 0; i < prob.N; ++i) {
    for (int s = 0; s < II; ++s) {
      vars.col_x[i][s] = AddCol(highs, 0, 1, 0, true);
    }
  }
  for (int i = 0; i < prob.N; ++i) {
    int ub = CeilDiv(prob.P[i].latency, II);
    for (int s = 0; s < II; ++s) {
      vars.col_a[i][s] = AddCol(highs, 0, ub, 0, true);
    }
  }
  vars.col_T = AddCol(highs, 0, time_ub, optimize_t ? 1.0 : 0.0, true);

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
  }

  for (int i = 0; i < prob.N; ++i) {
    int dur = prob.P[i].latency;
    for (int s = 0; s < II; ++s) {
      std::vector<HighsInt> idx{vars.col_a[i][s]};
      std::vector<double> val{1.0};
      for (int st = 0; st < II; ++st) {
        int rel = (s - st) % II;
        if (rel < 0) {
          rel += II;
        }
        int cnt = 0;
        if (rel < dur) {
          cnt = CeilDiv(dur - rel, II);
        }
        if (cnt != 0) {
          idx.push_back(vars.col_x[i][st]);
          val.push_back(-double(cnt));
        }
      }
      AddEq(highs, idx, val, 0.0);
    }
  }

  for (const auto& e : prob.dep_edges) {
    int i = e.first;
    int j = e.second;
    int d = prob.P[i].latency;
    int delta = prob.delta.at(EdgeKey(i, j));
    if (i == j) {
      if (d > delta * II) {
        AddRow(highs, 1.0, 0.0, {}, {});
      }
      continue;
    }
    AddLeq(highs, {vars.col_t[i], vars.col_t[j]}, {1.0, -1.0},
           double(delta * II - d));
  }

  for (int i = 0; i < prob.N; ++i) {
    AddLeq(highs, {vars.col_t[i], vars.col_T}, {1.0, -1.0},
           double(-prob.P[i].latency));
  }

  for (int r : prob.R) {
    int cap = prob.cap.count(r) ? prob.cap.at(r) : 1;
    for (int s = 0; s < II; ++s) {
      std::vector<HighsInt> idx;
      std::vector<double> val;
      for (int i = 0; i < prob.N; ++i) {
        if (std::find(prob.P[i].resources.begin(), prob.P[i].resources.end(), r) !=
            prob.P[i].resources.end()) {
          idx.push_back(vars.col_a[i][s]);
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

  {
    std::map<SameWriteFlowKey, std::vector<int>> same_write_groups;
    for (int vv = 0; vv < internal_count; ++vv) {
      int fid = vars.internal_flow_ids[vv];
      const FlowSpec& flow = prob.flows[fid];
      if (flow.write_resource < 0 || flow.prod < 0) continue;
      same_write_groups[MakeSameWriteFlowKey(flow)].push_back(vv);
    }
    for (const auto& kv : same_write_groups) {
      const std::vector<int>& flows = kv.second;
      for (size_t i = 1; i < flows.size(); ++i) {
        AddEq(highs, {vars.col_z[flows[0]], vars.col_z[flows[i]]},
              {1.0, -1.0}, 0.0);
      }
    }
  }

  auto bank_build_begin = std::chrono::steady_clock::now();
  for (int a = 0; a < internal_count; ++a) {
    const auto& write_flow = prob.flows[vars.internal_flow_ids[a]];
    for (int b = 0; b < internal_count; ++b) {
      if (a == b) continue;
      const auto& read_flow = prob.flows[vars.internal_flow_ids[b]];
      if (read_flow.read_resource < 0) continue;
      if (write_flow.mem != read_flow.mem) continue;

      int prod_slot_begin = write_flow.resident ? 0 : 0;
      int prod_slot_end = write_flow.resident ? II : II;
      for (int prod_slot = prod_slot_begin; prod_slot < prod_slot_end; ++prod_slot) {
        int write_start_time =
            write_flow.resident ? prod_slot + write_flow.w_off
                                : prod_slot + write_flow.w_off;
        int write_start_slot = PositiveMod(write_start_time, II);
        int write_parity_flip = (write_start_time / II) & 1;
        std::map<int, int> write_slot_rho =
            BuildSlotRhoMap(write_start_slot, write_flow.w_dur, II);
        std::vector<int> diff_cons_slots;
        std::vector<int> same_cons_slots;
        std::vector<int> impossible_cons_slots;
        for (int cons_slot = 0; cons_slot < II; ++cons_slot) {
          int read_start_slot = PositiveMod(cons_slot + read_flow.r_off, II);
          int read_parity_flip = (write_parity_flip + read_flow.delta) & 1;
          bool saw_same = false;
          bool saw_diff = false;
          for (int step = 0; step < read_flow.r_dur; ++step) {
            int slot = (read_start_slot + step) % II;
            auto it = write_slot_rho.find(slot);
            if (it == write_slot_rho.end()) continue;
            int read_rho = WrapBit(read_start_slot, slot) ^ read_parity_flip;
            if (it->second == read_rho) {
              saw_same = true;
            } else {
              saw_diff = true;
            }
            if (saw_same && saw_diff) break;
          }
          ConflictType conflict = ConflictType::kNone;
          if (saw_same && saw_diff) {
            conflict = ConflictType::kImpossible;
          } else if (saw_same) {
            conflict = ConflictType::kNeedDifferent;
          } else if (saw_diff) {
            conflict = ConflictType::kNeedSame;
          }
          if (conflict == ConflictType::kNone) continue;
          if (conflict == ConflictType::kNeedDifferent) {
            diff_cons_slots.push_back(cons_slot);
          } else if (conflict == ConflictType::kNeedSame) {
            same_cons_slots.push_back(cons_slot);
          } else if (conflict == ConflictType::kImpossible) {
            impossible_cons_slots.push_back(cons_slot);
          }
        }

        HighsInt x_prod =
            write_flow.resident ? HighsInt(-1) : vars.col_x[write_flow.prod][prod_slot];
        HighsInt z_write_ping = vars.col_z[a];
        HighsInt z_read_ping = vars.col_z[b];

        if (!diff_cons_slots.empty()) {
          std::vector<HighsInt> idx{z_write_ping, z_read_ping};
          std::vector<double> val{-1.0, -1.0};
          if (x_prod >= 0) {
            idx.insert(idx.begin(), x_prod);
            val.insert(val.begin(), 1.0);
          }
          for (int cons_slot : diff_cons_slots) {
            idx.push_back(vars.col_x[read_flow.cons][cons_slot]);
            val.push_back(1.0);
          }
          AddLeq(highs, idx, val, x_prod >= 0 ? 1.0 : 0.0);

          idx = {z_write_ping, z_read_ping};
          val = {1.0, 1.0};
          if (x_prod >= 0) {
            idx.insert(idx.begin(), x_prod);
            val.insert(val.begin(), 1.0);
          }
          for (int cons_slot : diff_cons_slots) {
            idx.push_back(vars.col_x[read_flow.cons][cons_slot]);
            val.push_back(1.0);
          }
          AddLeq(highs, idx, val, x_prod >= 0 ? 3.0 : 2.0);
        }

        if (!same_cons_slots.empty()) {
          std::vector<HighsInt> idx{z_write_ping, z_read_ping};
          std::vector<double> val{1.0, -1.0};
          if (x_prod >= 0) {
            idx.insert(idx.begin(), x_prod);
            val.insert(val.begin(), 1.0);
          }
          for (int cons_slot : same_cons_slots) {
            idx.push_back(vars.col_x[read_flow.cons][cons_slot]);
            val.push_back(1.0);
          }
          AddLeq(highs, idx, val, x_prod >= 0 ? 2.0 : 1.0);

          idx = {z_write_ping, z_read_ping};
          val = {-1.0, 1.0};
          if (x_prod >= 0) {
            idx.insert(idx.begin(), x_prod);
            val.insert(val.begin(), 1.0);
          }
          for (int cons_slot : same_cons_slots) {
            idx.push_back(vars.col_x[read_flow.cons][cons_slot]);
            val.push_back(1.0);
          }
          AddLeq(highs, idx, val, x_prod >= 0 ? 2.0 : 1.0);
        }

        if (!impossible_cons_slots.empty()) {
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
  LOG(INFO) << "[ILP] bank_constraint_build_elapsed=" << bank_build_elapsed << "s";

  return vars;
}

SolveResult SolveFixedII(const Problem& prob, int II, bool optimize_t,
                         int threads) {
  auto solve_begin = std::chrono::steady_clock::now();
  LOG(INFO) << "[ILP] start solve II=" << II
            << " optimize_t=" << optimize_t
            << " N=" << prob.N
            << " edges=" << prob.dep_edges.size()
            << " flows=" << prob.flows.size();
  Highs highs;
  ModelVars vars = BuildModel(highs, prob, II, optimize_t, threads);
  highs.run();
  if (highs.getModelStatus() != HighsModelStatus::kOptimal) {
    auto solve_end = std::chrono::steady_clock::now();
    double elapsed =
        std::chrono::duration<double>(solve_end - solve_begin).count();
    LOG(INFO) << "[II=" << II << "] infeasible/failed, elapsed=" << elapsed
              << "s, status=" << int(highs.getModelStatus());
    return {};
  }

  const HighsSolution& sol = highs.getSolution();
  SolveResult res;
  res.ok = true;
  res.II = II;
  res.bank_slot_period = 2 * II;
  if (!optimize_t) {
    auto solve_end = std::chrono::steady_clock::now();
    double elapsed =
        std::chrono::duration<double>(solve_end - solve_begin).count();
    LOG(INFO) << "[II=" << II << "] feasible_only_elapsed=" << elapsed << "s";
    return res;
  }
  res.t.resize(prob.N);
  res.m.resize(prob.N);
  res.y.resize(prob.N);
  for (int i = 0; i < prob.N; ++i) {
    res.t[i] = int(std::llround(sol.col_value[vars.col_t[i]]));
    res.m[i] = int(std::llround(sol.col_value[vars.col_m[i]]));
    res.y[i] = int(std::llround(sol.col_value[vars.col_y[i]]));
  }
  res.makespan = int(std::llround(sol.col_value[vars.col_T]));
  res.internal_flow_ids = vars.internal_flow_ids;
  res.z_bank.resize(vars.internal_flow_ids.size(), 0);
  for (int vv = 0; vv < static_cast<int>(vars.internal_flow_ids.size()); ++vv) {
    res.z_bank[vv] = int(std::llround(sol.col_value[vars.col_z[vv]]));
  }
  auto solve_end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(solve_end - solve_begin).count();
  LOG(INFO) << "[II=" << II << "] solve_elapsed=" << elapsed << "s";
  return res;
}

SolveResult FindMinimalII(const Problem& prob, int threads) {
  int lb = std::max(1, ResourceLowerBound(prob));
  int search_begin = std::max(1, lb);
  int search_end = std::max(search_begin, std::max(1, prob.Tmax));
  constexpr int kInitialWindowSpan = 10;
  int best_ii = -1;
  LOG(INFO) << "[ILP] search start=" << search_begin
            << " end=" << search_end
            << " lb=" << lb
            << " initial_window_span=" << kInitialWindowSpan;

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
      SolveResult feas = SolveFixedII(prob, mid, false, threads);
      if (feas.ok) {
        window_best = mid;
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
  LOG(INFO) << "[ILP] best feasible II=" << best_ii << ", optimize makespan";
  return SolveFixedII(prob, best_ii, true, threads);
}

class SunmmioPipelinePlannerILP : public StmtExprMutator {
 public:
  static Stmt Substitute(const PrimFunc& f, bool debug) {
    SunmmioPipelinePlannerILP planner(f, debug);
    return planner.VisitStmt(f->body);
  }

 private:
  SunmmioPipelinePlannerILP(const PrimFunc& f, bool debug)
      : func_(f), traverser_(f), debug_(debug) {}

  Optional<For> FindPipelineLoop(const Stmt& stmt) {
    Optional<For> result;
    PostOrderVisit(stmt, [&](const ObjectRef& obj) {
      if (result.defined()) {
        return;
      }
      if (const auto* loop = obj.as<ForNode>()) {
        if (loop->annotations.find("num_stages") != loop->annotations.end()) {
          result = ffi::GetRef<For>(loop);
        }
      }
    });
    return result;
  }

  const SeqStmtNode* GetPipelineBodySeq(const For& loop) {
    Stmt current = loop->body;
    if (const auto* realize = current.as<BlockRealizeNode>()) {
      current = realize->block->body;
    }
    while (true) {
      if (const auto* seq = current.as<SeqStmtNode>()) {
        return seq;
      }
      if (const auto* if_node = current.as<IfThenElseNode>()) {
        ICHECK(!if_node->else_case.defined());
        current = if_node->then_case;
        continue;
      }
      if (const auto* let_node = current.as<LetStmtNode>()) {
        current = let_node->body;
        continue;
      }
      return nullptr;
    }
  }

  struct IlpLoopAnalysis {
    Problem prob;
    std::vector<TemplateCommand> commands;
    std::set<Buffer> used_buffers;
    std::vector<Buffer> versioned_buffers;
    std::vector<Buffer> runtime_multiversion_buffers;
    std::vector<Buffer> runtime_banked_buffers;
    std::map<std::string, int> runtime_bank_start_phases;
    std::map<std::string, int> runtime_bank_read_delta_parities;
    std::map<std::string, std::map<int, int>> runtime_bank_writer_phases;
    std::map<std::string, std::map<int, int>> runtime_bank_reader_phases;
    int iterations{0};
  };

  IlpLoopAnalysis AnalyzeLoop(const For& loop,
                              const SeqStmtNode* pipeline_body_seq,
                              int forced_iterations = -1) {
    IlpLoopAnalysis result;
    bool export_only = GetEnvBool("TL_SUNMMIO_ILP_EXPORT_ONLY", false);
    int num_stages = -1;
    auto it = loop->annotations.find("num_stages");
    ICHECK(it != loop->annotations.end());
    const auto& any_ref = (*it).second;
    if (const auto* imm = any_ref.as<IntImmNode>()) {
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
    result.commands.reserve(pipeline_body_seq->seq.size());

    std::set<int> resource_set;
    for (int i = 0; i < static_cast<int>(pipeline_body_seq->seq.size()); ++i) {
      const Stmt& stmt = pipeline_body_seq->seq[i];
      TemplateCommand cmd(i, stmt);
      role_marker(stmt);
      cmd.role = role_marker.GetRole(stmt);
      traverser.traverse_stmt(stmt);
      cmd.type = HardwareMapper::Map(stmt);
      cmd.accesses = access_analyzer.Collect(stmt, loop->loop_var);
      cmd.spec.latency = static_cast<int>(std::ceil(
          CostModel::EstimateDelay(cmd.type, stmt)));
      cmd.spec.latency = std::max(cmd.spec.latency, 1);
      cmd.spec.resources = BuildIlpResources(stmt, cmd.type, cmd.accesses);
      cmd.spec.name = cmd.name + ": " + SummarizeStmtForName(stmt);
      for (int resource : cmd.spec.resources) {
        resource_set.insert(resource);
      }
      for (const BufferRegion& read : traverser.read_buffer_regions_) {
        if (!IsGlobalBuffer(read->buffer)) {
          result.used_buffers.insert(read->buffer);
        }
      }
      for (const BufferRegion& write : traverser.write_buffer_regions_) {
        if (!IsGlobalBuffer(write->buffer)) {
          result.used_buffers.insert(write->buffer);
        }
      }
      result.commands.push_back(cmd);
    }

    result.prob.N = static_cast<int>(result.commands.size());
    result.prob.P.resize(result.prob.N);
    for (const TemplateCommand& cmd : result.commands) {
      result.prob.P[cmd.id] = cmd.spec;
    }

    int faster = 0;
    std::vector<int> bump_indices;
    {
      auto pass_ctx = tvm::transform::PassContext::Current();
      auto cfg = pass_ctx->GetConfig<Integer>(tl::kSunmmioILPFaster);
      if (cfg.defined()) {
        faster = static_cast<int>(cfg.value()->value);
      }
    }
    if (faster <= 0) {
      faster = GetEnvInt("TL_SUNMMIO_ILP_FASTER", 0);
    }
    if (faster <= 0) {
      auto auto_selected = AutoSelectSunmmioILPFaster(result.prob.P);
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
      total_latency += result.commands[i].spec.latency;
      latency_gcd = latency_gcd == 0 ? result.commands[i].spec.latency
                                     : GcdInt(latency_gcd,
                                              result.commands[i].spec.latency);
    }

    if (latency_gcd <= 0) {
      latency_gcd = 1;
    }
    for (TemplateCommand& cmd : result.commands) {
      cmd.spec.latency /= latency_gcd;
      if (faster > 1) {
        cmd.spec.latency = CeilDiv(cmd.spec.latency, faster);
      }
    }
    total_latency /= latency_gcd;
    if (faster > 1) {
      total_latency = CeilDiv(total_latency, faster);
    }

    result.prob.Tmax = total_latency + 10;
    result.prob.R.assign(resource_set.begin(), resource_set.end());
    for (int resource : result.prob.R) {
      result.prob.cap[resource] = 1;
    }
    for (const TemplateCommand& cmd : result.commands) {
      result.prob.P[cmd.id] = cmd.spec;
    }
    result.versioned_buffers = DetectVersionedBuffers(result.commands);
    std::sort(result.versioned_buffers.begin(), result.versioned_buffers.end(),
              [](const Buffer& a, const Buffer& b) { return a->name < b->name; });
    result.runtime_multiversion_buffers.clear();
    result.runtime_banked_buffers.clear();
    result.runtime_bank_start_phases.clear();
    result.runtime_bank_read_delta_parities.clear();
    result.runtime_bank_writer_phases.clear();
    result.runtime_bank_reader_phases.clear();
    result.prob.versioned_buffer_names.clear();
    for (const Buffer& buffer : result.versioned_buffers) {
      result.prob.versioned_buffer_names.push_back(buffer->name);
    }
    BuildTemplateDependencyGraph(result.commands, num_stages, result.versioned_buffers,
                                 loop->loop_var, &result.prob);
    return result;
  }

  struct StageShrinkResult {
    IlpLoopAnalysis analysis;
    SolveResult sol;
  };

  bool ShouldEnableStageShrink() const {
    auto pass_ctx = tvm::transform::PassContext::Current();
    auto cfg = pass_ctx->GetConfig<Bool>(tl::kSunmmioILPStageShrink);
    if (cfg.defined()) {
      return cfg.value()->value;
    }
    return false;
  }

  void PopulateRuntimeBankedBuffers(IlpLoopAnalysis* analysis) const {
    analysis->runtime_banked_buffers.clear();
    for (const Buffer& buffer : analysis->versioned_buffers) {
      if (analysis->runtime_bank_start_phases.count(buffer->name)) {
        analysis->runtime_banked_buffers.push_back(buffer);
      }
    }
  }

  void ExportStageSolutionIfRequested(const IlpLoopAnalysis& analysis,
                                      const SolveResult& sol, int stage) const {
    std::string solution_json_path = GetEnvString("TL_SUNMMIO_ILP_SOLUTION_JSON");
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

  StageShrinkResult SolveWithStageShrink(const For& loop,
                                         const SeqStmtNode* pipeline_body_seq,
                                         int threads) {
    IlpLoopAnalysis base_analysis = AnalyzeLoop(loop, pipeline_body_seq);
    MaybeExportProblemJsonForStage(base_analysis.prob, debug_,
                                   base_analysis.iterations);
    SolveResult base_sol = FindMinimalII(base_analysis.prob, threads);
    ExportStageSolutionIfRequested(base_analysis, base_sol,
                                   base_analysis.iterations);
    if (!base_sol.ok) {
      return {std::move(base_analysis), std::move(base_sol)};
    }

    int best_iterations = base_analysis.iterations;
    for (int candidate_iterations = base_analysis.iterations - 1;
         candidate_iterations >= 1; --candidate_iterations) {
      IlpLoopAnalysis candidate_analysis =
          AnalyzeLoop(loop, pipeline_body_seq, candidate_iterations);
      MaybeExportProblemJsonForStage(candidate_analysis.prob, debug_,
                                     candidate_iterations);
      SolveResult feas =
          SolveFixedII(candidate_analysis.prob, base_sol.II, false, threads);
      if (!feas.ok) {
        continue;
      }
      SolveResult candidate_sol =
          SolveFixedII(candidate_analysis.prob, base_sol.II, true, threads);
      ExportStageSolutionIfRequested(candidate_analysis, candidate_sol,
                                     candidate_iterations);
      best_iterations = candidate_iterations;
    }

    if (best_iterations == base_analysis.iterations) {
      return {std::move(base_analysis), std::move(base_sol)};
    }

    IlpLoopAnalysis final_analysis =
        AnalyzeLoop(loop, pipeline_body_seq, best_iterations);
    MaybeExportProblemJsonForStage(final_analysis.prob, debug_,
                                   final_analysis.iterations);
    SolveResult final_sol = FindMinimalII(final_analysis.prob, threads);
    ExportStageSolutionIfRequested(final_analysis, final_sol,
                                   final_analysis.iterations);
    return {std::move(final_analysis), std::move(final_sol)};
  }

  void PopulateRuntimeBankMetadata(IlpLoopAnalysis* analysis,
                                   const SolveResult& sol) const {
    std::unordered_map<int, int> internal_pos;
    for (int i = 0; i < static_cast<int>(sol.internal_flow_ids.size()); ++i) {
      internal_pos[sol.internal_flow_ids[i]] = i;
    }
    std::unordered_set<std::string> has_non_resident_flow;
    for (int fid = 0; fid < static_cast<int>(analysis->prob.flows.size()); ++fid) {
      const auto& flow = analysis->prob.flows[fid];
      if (!flow.resident && !flow.buffer_name.empty()) {
        has_non_resident_flow.insert(flow.buffer_name);
      }
    }
    for (int fid = 0; fid < static_cast<int>(analysis->prob.flows.size()); ++fid) {
      const auto& flow = analysis->prob.flows[fid];
      if (flow.buffer_name.empty()) {
        continue;
      }
      auto flow_pos = internal_pos.find(fid);
      if (flow_pos == internal_pos.end()) {
        continue;
      }
      int bank_phase = sol.z_bank[flow_pos->second];
      bool allow_resident_to_own_bank =
          flow.resident && !has_non_resident_flow.count(flow.buffer_name);
      if (!flow.resident || allow_resident_to_own_bank) {
        auto it = analysis->runtime_bank_start_phases.find(flow.buffer_name);
        if (it == analysis->runtime_bank_start_phases.end()) {
          analysis->runtime_bank_start_phases[flow.buffer_name] = bank_phase;
        } else {
          ICHECK_EQ(it->second, bank_phase)
              << "Conflicting bank phase for runtime multiversion buffer "
              << flow.buffer_name;
        }
        int delta_parity = flow.delta & 1;
        auto it_delta =
            analysis->runtime_bank_read_delta_parities.find(flow.buffer_name);
        if (it_delta == analysis->runtime_bank_read_delta_parities.end()) {
          analysis->runtime_bank_read_delta_parities[flow.buffer_name] =
              delta_parity;
        } else {
          ICHECK_EQ(it_delta->second, delta_parity)
              << "Conflicting read delta parity for banked buffer "
              << flow.buffer_name;
        }
      }
      if (flow.write_resource >= 0 && flow.prod >= 0) {
        auto& writer_map = analysis->runtime_bank_writer_phases[flow.buffer_name];
        auto it_writer = writer_map.find(flow.prod);
        if (it_writer == writer_map.end()) {
          writer_map[flow.prod] = bank_phase;
        } else {
          ICHECK_EQ(it_writer->second, bank_phase)
              << "Conflicting writer bank phase for banked buffer "
              << flow.buffer_name << " op " << flow.prod;
        }
      }
      if (flow.cons >= 0 && flow.read_resource >= 0) {
        int reader_phase =
            flow.resident ? bank_phase : ((bank_phase + (flow.delta & 1)) & 1);
        auto& reader_map = analysis->runtime_bank_reader_phases[flow.buffer_name];
        auto it_reader = reader_map.find(flow.cons);
        if (it_reader == reader_map.end()) {
          reader_map[flow.cons] = reader_phase;
        } else {
          ICHECK_EQ(it_reader->second, reader_phase)
              << "Conflicting reader bank phase for banked buffer "
              << flow.buffer_name << " op " << flow.cons;
        }
      }
    }
  }

  void PruneUnnecessaryRuntimeBanking(IlpLoopAnalysis* analysis,
                                      const SolveResult& sol) {
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
          const CommandSpec& spec = analysis->prob.P[i];
          bool uses_write =
              std::find(spec.resources.begin(), spec.resources.end(),
                        write_resource) != spec.resources.end();
          bool uses_read =
              std::find(spec.resources.begin(), spec.resources.end(),
                        read_resource) != spec.resources.end();
          if (!uses_write && !uses_read) {
            continue;
          }
          int occ =
              ComputeFoldedOccupancy(sol.t[i], spec.latency, sol.II, slot);
          if (uses_write) {
            write_use += occ;
          }
          if (uses_read) {
            read_use += occ;
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
    for (const Buffer& buffer : analysis->runtime_banked_buffers) {
      int mem = GetPingPongMemoryKind(buffer);
      bool keep = (mem == 0 && keep_wsram) || (mem == 1 && keep_asram);
      if (keep) {
        pruned_banked_buffers.push_back(buffer);
      } else {
        analysis->runtime_bank_start_phases.erase(buffer->name);
        analysis->runtime_bank_read_delta_parities.erase(buffer->name);
        analysis->runtime_bank_writer_phases.erase(buffer->name);
        analysis->runtime_bank_reader_phases.erase(buffer->name);
      }
    }
    analysis->runtime_banked_buffers = std::move(pruned_banked_buffers);
  }

  Stmt VisitStmt_(const ForNode* op) final {
    For loop = ffi::GetRef<For>(op);
    if (op->annotations.find("num_stages") == op->annotations.end()) {
      return StmtExprMutator::VisitStmt_(op);
    }

    const SeqStmtNode* pipeline_body_seq = GetPipelineBodySeq(loop);
    ICHECK(pipeline_body_seq != nullptr) << "Pipeline body must normalize to SeqStmt.";
    int threads = GetEnvInt("HIGHS_THREADS", 20);
    IlpLoopAnalysis analysis;
    SolveResult sol;
    if (ShouldEnableStageShrink()) {
      StageShrinkResult shrink_result =
          SolveWithStageShrink(loop, pipeline_body_seq, threads);
      analysis = std::move(shrink_result.analysis);
      sol = std::move(shrink_result.sol);
    } else {
      analysis = AnalyzeLoop(loop, pipeline_body_seq);
      MaybeExportProblemJson(analysis.prob, debug_);
      sol = FindMinimalII(analysis.prob, threads);
    }
    if (GetEnvBool("TL_SUNMMIO_ILP_EXPORT_ONLY", false)) {
      return loop;
    }
    ICHECK(sol.ok) << "ILP solve failed for sunmmio pipeline planning.";
    PopulateRuntimeBankMetadata(&analysis, sol);
    for (const Buffer& buffer : analysis.versioned_buffers) {
      if (analysis.runtime_bank_start_phases.count(buffer->name)) {
        analysis.runtime_banked_buffers.push_back(buffer);
      }
    }
    // Disabled per current ILP annotation semantics:
    // keep the bank-rotation decision from the solved flow model as-is, and do
    // not prune ping/pong before annotation.
    //
    // PruneUnnecessaryRuntimeBanking(&analysis, sol);
    SolutionVerifyResult verify = VerifySolution(analysis.prob, sol);
    ICHECK(verify.ok) << "ILP solution verification failed: " << verify.errors.front();
    std::string solution_json_path = GetEnvString("TL_SUNMMIO_ILP_SOLUTION_JSON");
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
                << " dep_edges=" << analysis.prob.dep_edges.size()
                << " flows=" << analysis.prob.flows.size()
                << " solved=" << sol.ok
                << " ii=" << sol.II;
    }

    Map<String, Any> annotations;
    for (const auto& kv : op->annotations) {
      if (kv.first != "num_stages" && kv.first != "versioned_buffers") {
        annotations.Set(kv.first, kv.second);
      }
    }

    int stage_count = CeilDiv(sol.makespan, std::max(1, sol.II));
    analysis.runtime_multiversion_buffers = DetectRuntimeMultiversionBuffers(
        analysis.commands, analysis.versioned_buffers, analysis.runtime_banked_buffers,
        loop->loop_var, sol, stage_count, analysis.runtime_bank_start_phases,
        analysis.runtime_bank_writer_phases);
    annotations.Set("iterations", Integer(analysis.iterations));
    annotations.Set("ii", Integer(sol.II));
    annotations.Set("makespan", Integer(sol.makespan));
    annotations.Set("stage_count", Integer(stage_count));
    Array<String> prologue_orders;
    Array<String> body_orders;
    Array<String> epilogue_orders;
    auto starts_earlier_and_non_vc_first = [&](int a, int b) {
      if (sol.t[a] != sol.t[b]) {
        return sol.t[a] < sol.t[b];
      }
      bool a_is_vc = CommandUsesResource(
          analysis.prob.P[a], static_cast<int>(IlpResourceType::kVectorCore));
      bool b_is_vc = CommandUsesResource(
          analysis.prob.P[b], static_cast<int>(IlpResourceType::kVectorCore));
      if (a_is_vc != b_is_vc) {
        return !a_is_vc;
      }
      return a < b;
    };
    TimeWindowOrderResult window_orders =
        BuildTimeWindowOrders(analysis.commands, analysis.iterations, sol);
    annotations.Set("steady_state_max_iter_offset",
                    Integer(window_orders.steady_state_max_iter_offset));

    auto time_then_non_vc_first = [&](const ExpandedOrderEntry& a,
                                      const ExpandedOrderEntry& b) {
      if (a.absolute_start != b.absolute_start) {
        return a.absolute_start < b.absolute_start;
      }
      return starts_earlier_and_non_vc_first(a.id, b.id);
    };

    std::sort(window_orders.prologue.begin(), window_orders.prologue.end(),
              time_then_non_vc_first);
    std::sort(window_orders.body.begin(), window_orders.body.end(),
              time_then_non_vc_first);
    std::sort(window_orders.epilogue.begin(), window_orders.epilogue.end(),
              time_then_non_vc_first);

    for (const ExpandedOrderEntry& entry : window_orders.prologue) {
      prologue_orders.push_back(
          String(std::to_string(entry.iter) + "-" + std::to_string(entry.id)));
    }
    for (const ExpandedOrderEntry& entry : window_orders.body) {
      body_orders.push_back(
          String(std::to_string(entry.iter) + "-" + std::to_string(entry.id)));
    }
    for (const ExpandedOrderEntry& entry : window_orders.epilogue) {
      epilogue_orders.push_back(
          String(std::to_string(entry.iter) + "-" + std::to_string(entry.id)));
    }

    annotations.Set("prologue_orders", prologue_orders);
    annotations.Set("body_orders", body_orders);
    annotations.Set("epilogue_orders", epilogue_orders);

    Array<Buffer> used_buffers_array(analysis.used_buffers.begin(),
                                     analysis.used_buffers.end());
    annotations.Set("used_buffers", used_buffers_array);
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
    Map<Buffer, PrimExpr> runtime_bank_start_phases;
    for (const Buffer& buffer : analysis.runtime_banked_buffers) {
      runtime_bank_start_phases.Set(
          buffer, Integer(analysis.runtime_bank_start_phases.at(buffer->name)));
    }
    annotations.Set("runtime_bank_start_phases", runtime_bank_start_phases);
    Map<Buffer, PrimExpr> runtime_bank_read_delta_parities;
    for (const Buffer& buffer : analysis.runtime_banked_buffers) {
      auto it = analysis.runtime_bank_read_delta_parities.find(buffer->name);
      if (it != analysis.runtime_bank_read_delta_parities.end()) {
        runtime_bank_read_delta_parities.Set(buffer, Integer(it->second));
      }
    }
    annotations.Set("runtime_bank_read_delta_parities",
                    runtime_bank_read_delta_parities);
    Map<Buffer, Map<Integer, PrimExpr>> runtime_bank_writer_phases;
    for (const Buffer& buffer : analysis.runtime_banked_buffers) {
      auto it = analysis.runtime_bank_writer_phases.find(buffer->name);
      if (it == analysis.runtime_bank_writer_phases.end()) {
        continue;
      }
      Map<Integer, PrimExpr> per_op;
      for (const auto& op_phase : it->second) {
        per_op.Set(Integer(op_phase.first), Integer(op_phase.second));
      }
      runtime_bank_writer_phases.Set(buffer, per_op);
    }
    annotations.Set("runtime_bank_writer_phases", runtime_bank_writer_phases);
    Map<Buffer, Map<Integer, PrimExpr>> runtime_bank_reader_phases;
    for (const Buffer& buffer : analysis.runtime_banked_buffers) {
      auto it = analysis.runtime_bank_reader_phases.find(buffer->name);
      if (it == analysis.runtime_bank_reader_phases.end()) {
        continue;
      }
      Map<Integer, PrimExpr> per_op;
      for (const auto& op_phase : it->second) {
        per_op.Set(Integer(op_phase.first), Integer(op_phase.second));
      }
      runtime_bank_reader_phases.Set(buffer, per_op);
    }
    annotations.Set("runtime_bank_reader_phases", runtime_bank_reader_phases);

    Stmt body = this->VisitStmt(op->body);
    For new_loop = loop;
    ForNode* loop_ptr = new_loop.CopyOnWrite();
    loop_ptr->body = body;
    loop_ptr->annotations = annotations;
    return new_loop;
  }

  PrimFunc func_;
  ASTTraverser traverser_;
  bool debug_{false};
};

tvm::transform::Pass SunmmioPipelinePlanningILP(bool debug = false) {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule& m, PassContext ctx) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    fptr->body = SunmmioPipelinePlannerILP::Substitute(f, debug);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.SunmmioPipelinePlanningILP", {});
}

}  // namespace bank_ilp_internal

tvm::transform::Pass SunmmioPipelinePlanningILP(bool debug = false) {
  return bank_ilp_internal::SunmmioPipelinePlanningILP(debug);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.SunmmioPipelinePlanningILP",
                        SunmmioPipelinePlanningILP);
}

}  // namespace tl
}  // namespace tvm
