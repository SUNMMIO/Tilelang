/*!
 * \file sunmmio_pipeline_planning.cc
 * \brief
 * This file implements a greedy software pipeline scheduling algorithm for
 * Sunmmio. 1.1 Build a local DDG. 1.2 Identify prefetch instructions. 1.3
 * Identify multi-versioned buffers.
 * 2. Unroll DDG according to num_stages, while considering prefetch
 * instructions. 3.1 Schedule instructions in the global DDG with a b-level
 * based greedy algorithm. 3.2 Insert prefetch instructions.
 *
 * It should be noticed that: 1. Current implementation does not support
 * IF control flows. 2. Current implementation does not support tvm_access_ptr
 * call.
 */

#include "../op/builtin.h"
#include "../op/comm.h"
#include "../op/utils.h"
#include "../target/sunmmio/cost_model.h"
#include "../target/sunmmio/hardware_types.h"
#include "../target/sunmmio_utils.h"
#include "sunmmio_pipeline_planning/pipeline_diagnostic.h"
#include "sunmmio_pipeline_planning/resource_types_for_ilp.h"
#include "sunmmio_pipeline_planning/stmt_read_write_collector.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/expr.h>
#include <tvm/node/cast.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/var.h>

namespace tvm {
namespace tl {

static constexpr const char *kGreedySunmmioFaster = "tl.sunmmio_faster";
TVM_REGISTER_PASS_CONFIG_OPTION(kGreedySunmmioFaster, Integer);

using namespace tir;

/**
 * \brief AccessOverlapChecker abstracts overlap detection between buffer
 * accesses.
 *
 * The current implementation intentionally uses buffer-level equality as a
 * conservative approximation. This keeps the existing behavior stable while
 * reserving an explicit extension point for future region-level overlap
 * analysis.
 */
class AccessOverlapChecker {
public:
  /**
   * \brief Return whether two access regions should be treated as overlapping.
   */
  static bool Overlap(const BufferRegion &lhs, const BufferRegion &rhs) {
    return lhs->buffer.same_as(rhs->buffer);
  }
};

static bool HasRepeatedCollectiveDestination(const SeqStmtNode *body) {
  std::unordered_set<const BufferNode *> destinations;
  for (const Stmt &stmt : body->seq) {
    bool repeated = false;
    PostOrderVisit(stmt, [&](const ObjectRef &obj) {
      const auto *call = obj.as<CallNode>();
      if (!call || !call->op.same_as(Op::Get("tl.broadcast_"))) {
        return;
      }
      const BufferNode *destination =
          NormalizeToBufferRegion(call->args[1])->buffer.get();
      if (!destinations.insert(destination).second) {
        repeated = true;
      }
    });
    if (repeated) {
      return true;
    }
  }
  return false;
}

/**
 * \brief Pure data container representing an instruction in the pipeline.
 * It separates the AST analysis from scheduling and latency calculation.
 */
class PipelineInstruction {
public:
  int id{-1};
  int iter{-1};
  std::string name{""};
  Stmt stmt;
  DeviceType device_type{DeviceType::Unspecified};
  int execution_resource{-1};

  // True if this instruction should be placed in the prefetch queue (Shift=1)
  bool is_prefetch{false};

  // Regions read and written by this instruction
  std::vector<BufferRegion> reads;
  std::vector<BufferRegion> writes;

  // Scheduling state
  float scheduled_start{-1.0f};
  float scheduled_end{-1.0f};
  bool finished{false};

  // Pre-calculated delay (to be injected by CostModel)
  float delay{0.0f};

  PipelineInstruction(int id, int iter, Stmt stmt)
      : id(id), iter(iter), stmt(stmt),
        name(std::to_string(iter) + "-" + std::to_string(id)) {}

  void ExtractRegions(StmtReadWriteCollector &stmt_rw_collector) {
    stmt_rw_collector.clear();
    stmt_rw_collector.traverse_stmt(stmt);
    reads.assign(stmt_rw_collector.read_buffer_regions_.begin(),
                 stmt_rw_collector.read_buffer_regions_.end());
    writes.assign(stmt_rw_collector.write_buffer_regions_.begin(),
                  stmt_rw_collector.write_buffer_regions_.end());
  }

  bool operator==(const PipelineInstruction &other) const {
    return name == other.name;
  }
};

struct GreedyAccessInfo {
  BufferRegion region;
  bool is_write{false};

  Buffer buffer() const { return region->buffer; }
};

static int GetGreedyExecutionResource(const PipelineInstruction &instruction) {
  std::vector<GreedyAccessInfo> accesses;
  accesses.reserve(instruction.reads.size() + instruction.writes.size());
  for (const BufferRegion &read : instruction.reads) {
    accesses.push_back({read, false});
  }
  for (const BufferRegion &write : instruction.writes) {
    accesses.push_back({write, true});
  }
  std::vector<int> resources =
      BuildIlpResources(instruction.stmt, instruction.device_type, accesses);
  for (int resource : resources) {
    if (resource == static_cast<int>(IlpResourceType::kTensorCore) ||
        resource == static_cast<int>(IlpResourceType::kVectorCore) ||
        resource == static_cast<int>(IlpResourceType::kODMA0) ||
        resource == static_cast<int>(IlpResourceType::kODMA1)) {
      return resource;
    }
  }
  LOG(FATAL) << "No execution resource for greedy pipeline instruction "
             << instruction.name;
  return -1;
}

static int GetGreedyIssuePriority(const PipelineInstruction &instruction) {
  int resource = instruction.execution_resource;
  if (resource == static_cast<int>(IlpResourceType::kODMA1)) {
    return 0;
  }
  if (resource == static_cast<int>(IlpResourceType::kODMA0)) {
    return 1;
  }
  // DMA launch is asynchronous, while tensor commands block the scalar issue
  // stream.  Launch same-time asynchronous work before blocking computation.
  if (resource == static_cast<int>(IlpResourceType::kTensorCore)) {
    return 2;
  }
  if (resource == static_cast<int>(IlpResourceType::kVectorCore)) {
    return 3;
  }
  return 4;
}

static bool IsAllGatherInstruction(const PipelineInstruction &instruction) {
  const CallNode *broadcast = nullptr;
  PostOrderVisit(instruction.stmt, [&](const ObjectRef &obj) {
    const auto *call = obj.as<CallNode>();
    if (call && call->op.same_as(Op::Get("tl.broadcast_"))) {
      ICHECK(broadcast == nullptr)
          << "A pipeline statement may contain at most one broadcast leaf";
      broadcast = call;
    }
  });
  if (broadcast == nullptr) {
    return false;
  }
  ICHECK(broadcast->args.size() == static_cast<size_t>(kBroadcastArgCount) ||
         broadcast->args.size() == static_cast<size_t>(kBroadcastArgCount + 1))
      << "tl.broadcast_ expects its fixed arguments and optional src_core";
  return broadcast->args.size() == static_cast<size_t>(kBroadcastArgCount);
}

enum class PhysicalSramBank : int {
  ASRAMPing = 0,
  ASRAMPong = 1,
  WSRAMPing = 2,
  WSRAMPong = 3,
  Count = 4,
};

using PerCommandBankPhases =
    std::unordered_map<const BufferNode *, std::unordered_map<int, int>>;

struct GreedyBankColoring {
  PerCommandBankPhases writer_phases;
  PerCommandBankPhases reader_phases;
  std::vector<int> bits;
};

static int LookupBankPhase(const PerCommandBankPhases &phases,
                           const BufferNode *buffer, int command_id) {
  auto it_buffer = phases.find(buffer);
  if (it_buffer == phases.end())
    return 0;
  auto it_command = it_buffer->second.find(command_id);
  return it_command == it_buffer->second.end() ? 0 : it_command->second;
}

static std::vector<PhysicalSramBank> GetOccupiedSramBanks(
    const PipelineInstruction &instruction,
    const std::unordered_set<const BufferNode *> &versioned_buffers,
    int iter_mod, const PerCommandBankPhases &writer_phases,
    const PerCommandBankPhases &reader_phases) {
  std::array<bool, static_cast<int>(PhysicalSramBank::Count)> occupied{};
  int version_slot = instruction.iter;
  if (iter_mod > 0) {
    version_slot %= iter_mod;
    if (version_slot < 0) {
      version_slot += iter_mod;
    }
  }
  auto collect = [&](const BufferRegion &region, bool is_write) {
    const String &scope = region->buffer.scope();
    int phase = LookupBankPhase(is_write ? writer_phases : reader_phases,
                                region->buffer.get(), instruction.id);
    bool pong = versioned_buffers.count(region->buffer.get()) != 0 &&
                (version_slot + phase) % 2 != 0;
    if (scope == kSunmmioScopeASRAM) {
      occupied[static_cast<int>(pong ? PhysicalSramBank::ASRAMPong
                                     : PhysicalSramBank::ASRAMPing)] = true;
    } else if (scope == kSunmmioScopeWSRAM) {
      occupied[static_cast<int>(pong ? PhysicalSramBank::WSRAMPong
                                     : PhysicalSramBank::WSRAMPing)] = true;
    }
  };
  for (const BufferRegion &region : instruction.reads) {
    collect(region, false);
  }
  for (const BufferRegion &region : instruction.writes) {
    collect(region, true);
  }

  std::vector<PhysicalSramBank> result;
  for (int i = 0; i < static_cast<int>(PhysicalSramBank::Count); ++i) {
    if (occupied[i]) {
      result.push_back(static_cast<PhysicalSramBank>(i));
    }
  }
  return result;
}

/**
 * \brief A RAW dependence edge in the single-iteration local DDG.
 *
 * distance = 0 means an intra-iteration forward dependence.
 * distance > 0 means a loop-carried backward dependence.
 */
struct LocalDependencyEdge {
  int producer_instruction_id{-1};
  int consumer_instruction_id{-1};
  const BufferNode *buffer{nullptr};
  int distance{0};
};

struct TemplateOrderEdge {
  int source_instruction_id{-1};
  int target_instruction_id{-1};
  int distance{0};
};

enum class SemanticDependencyKind { kRAW, kWAR, kWAW };

struct SemanticDependencyEdge {
  int source_instruction_id{-1};
  int target_instruction_id{-1};
  const BufferNode *buffer{nullptr};
  int distance{0};
  SemanticDependencyKind kind{SemanticDependencyKind::kRAW};
};

static const char *SemanticDependencyKindName(SemanticDependencyKind kind) {
  switch (kind) {
  case SemanticDependencyKind::kRAW:
    return "RAW";
  case SemanticDependencyKind::kWAR:
    return "WAR";
  case SemanticDependencyKind::kWAW:
    return "WAW";
  }
  return "unknown";
}

/**
 * \brief Aggregated access information for one logical buffer in the local DDG.
 */
struct BufferAccessInfo {
  const BufferNode *buffer{nullptr};
  bool is_global{false};
  std::vector<int> write_instruction_indices;
  std::vector<int> read_instruction_indices;
  int first_write_index{-1};
  int first_read_index{-1};
  int last_read_index{-1};

  bool HasLoopCarriedDependence() const {
    return first_write_index != -1 && first_read_index != -1 &&
           first_read_index <= first_write_index;
  }
};

/**
 * \brief The local DDG built from a single iteration instruction sequence.
 */
struct LocalDDG {
  std::vector<LocalDependencyEdge> edges;
  std::vector<TemplateOrderEdge> ordering_edges;
  std::vector<SemanticDependencyEdge> semantic_edges;
  std::vector<std::vector<int>> forward_predecessors;
  std::vector<std::vector<int>> forward_successors;
  std::vector<std::vector<int>> backward_predecessors;
  std::vector<std::vector<int>> backward_successors;
  std::unordered_map<const BufferNode *, BufferAccessInfo> buffer_access_infos;
  std::vector<const BufferNode *> buffer_order;
};

static bool IsRuntimeBankedBuffer(const BufferNode *buffer) {
  const String &scope = tvm::ffi::GetRef<Buffer>(buffer).scope();
  return scope == kSunmmioScopeASRAM || scope == kSunmmioScopeWSRAM;
}

static std::vector<GreedyBankColoring> BuildGreedyBankColorings(
    const LocalDDG &local_ddg,
    const std::unordered_set<const BufferNode *> &versioned_buffers, int faster,
    size_t *total_candidate_count) {
  using WriterKey = std::pair<const BufferNode *, int>;
  std::map<const BufferNode *, std::set<int>> writers_by_buffer;
  for (const LocalDependencyEdge &edge : local_ddg.edges) {
    if (!versioned_buffers.count(edge.buffer) ||
        !IsRuntimeBankedBuffer(edge.buffer)) {
      continue;
    }
    writers_by_buffer[edge.buffer].insert(edge.producer_instruction_id);
  }

  // Precolors are relative bank phases.  Writers 0,2,... of one buffer must
  // stay together, writers 1,3,... must stay together, and the two classes
  // must use opposite banks.  Search one global inversion bit per buffer
  // instead of independently recoloring every writer.
  struct WriterColor {
    int variable{-1};
    int precolor{0};
  };
  std::map<WriterKey, WriterColor> writer_colors;
  int search_bits = 0;
  for (const BufferNode *buffer : local_ddg.buffer_order) {
    auto writers_it = writers_by_buffer.find(buffer);
    if (writers_it == writers_by_buffer.end()) {
      continue;
    }
    int precolor = 0;
    for (int writer_id : writers_it->second) {
      writer_colors[{buffer, writer_id}] = {search_bits, precolor};
      precolor ^= 1;
    }
    ++search_bits;
  }

  ICHECK(faster == -1 || faster > 0)
      << "tl.sunmmio_faster must be -1 or a positive coloring budget";
  ICHECK_LT(search_bits, static_cast<int>(sizeof(size_t) * 8))
      << "Too many independent greedy coloring variables: " << search_bits;
  size_t total_candidates = size_t{1} << search_bits;
  *total_candidate_count = total_candidates;
  size_t candidate_count =
      faster == -1 ? total_candidates
                   : std::min(total_candidates, static_cast<size_t>(faster));
  std::vector<GreedyBankColoring> result;
  result.reserve(candidate_count);
  for (size_t mask = 0; mask < candidate_count; ++mask) {
    GreedyBankColoring coloring;
    coloring.bits.resize(search_bits, 0);
    for (int i = 0; i < search_bits; ++i) {
      coloring.bits[i] = static_cast<int>((mask >> i) & 1);
    }
    for (const auto &[writer, color] : writer_colors) {
      coloring.writer_phases[writer.first][writer.second] =
          coloring.bits[color.variable] ^ color.precolor;
    }

    bool valid = true;
    for (const LocalDependencyEdge &edge : local_ddg.edges) {
      auto it_writer =
          writer_colors.find({edge.buffer, edge.producer_instruction_id});
      if (it_writer == writer_colors.end())
        continue;
      const WriterColor &color = it_writer->second;
      int writer_phase = coloring.bits[color.variable] ^ color.precolor;
      int reader_phase = writer_phase ^ (edge.distance & 1);
      auto &reader_map = coloring.reader_phases[edge.buffer];
      auto [it_reader, inserted] =
          reader_map.emplace(edge.consumer_instruction_id, reader_phase);
      if (!inserted && it_reader->second != reader_phase) {
        valid = false;
        break;
      }
    }
    if (valid)
      result.push_back(std::move(coloring));
  }
  if (result.empty())
    result.push_back(GreedyBankColoring{});
  return result;
}

/**
 * \brief Build the single-iteration local DDG from read/write regions.
 */
class LocalDDGBuilder {
public:
  static LocalDDG
  Build(const std::vector<PipelineInstruction> &single_iteration_instructions) {
    LocalDDG ddg;
    const int instruction_count =
        static_cast<int>(single_iteration_instructions.size());
    ddg.forward_predecessors.resize(instruction_count);
    ddg.forward_successors.resize(instruction_count);
    ddg.backward_predecessors.resize(instruction_count);
    ddg.backward_successors.resize(instruction_count);

    for (int instruction_index = 0; instruction_index < instruction_count;
         ++instruction_index) {
      const auto &instruction =
          single_iteration_instructions[instruction_index];
      for (const BufferRegion &write_region : instruction.writes) {
        const BufferNode *buffer = write_region->buffer.get();
        auto [it, inserted] = ddg.buffer_access_infos.try_emplace(buffer);
        BufferAccessInfo &access_info = it->second;
        if (inserted) {
          access_info.buffer = buffer;
          access_info.is_global = IsGlobalBuffer(write_region->buffer);
          ddg.buffer_order.push_back(buffer);
        }
        access_info.write_instruction_indices.push_back(instruction_index);
        if (access_info.first_write_index == -1) {
          access_info.first_write_index = instruction_index;
        }
      }
      for (const BufferRegion &read_region : instruction.reads) {
        const BufferNode *buffer = read_region->buffer.get();
        auto [it, inserted] = ddg.buffer_access_infos.try_emplace(buffer);
        BufferAccessInfo &access_info = it->second;
        if (inserted) {
          access_info.buffer = buffer;
          access_info.is_global = IsGlobalBuffer(read_region->buffer);
          ddg.buffer_order.push_back(buffer);
        }
        access_info.read_instruction_indices.push_back(instruction_index);
        if (access_info.first_read_index == -1) {
          access_info.first_read_index = instruction_index;
        }
        access_info.last_read_index = instruction_index;
      }
    }

    std::set<std::tuple<int, int, const BufferNode *, int>> unique_edges;
    for (int reader_index = 0; reader_index < instruction_count;
         ++reader_index) {
      const auto &reader_instruction =
          single_iteration_instructions[reader_index];
      for (const BufferRegion &read_region : reader_instruction.reads) {
        const BufferNode *buffer = read_region->buffer.get();
        auto access_info_it = ddg.buffer_access_infos.find(buffer);
        if (access_info_it == ddg.buffer_access_infos.end()) {
          continue;
        }

        const auto &access_info = access_info_it->second;
        int producer_index = -1;
        int distance = 0;

        for (int writer_index : access_info.write_instruction_indices) {
          const auto &writer_instruction =
              single_iteration_instructions[writer_index];
          bool overlaps = false;
          for (const BufferRegion &write_region : writer_instruction.writes) {
            if (AccessOverlapChecker::Overlap(write_region, read_region)) {
              overlaps = true;
              break;
            }
          }
          if (!overlaps) {
            continue;
          }

          if (writer_index < reader_index) {
            producer_index = writer_index;
            distance = 0;
          } else {
            break;
          }
        }

        if (producer_index == -1) {
          for (auto writer_it = access_info.write_instruction_indices.rbegin();
               writer_it != access_info.write_instruction_indices.rend();
               ++writer_it) {
            const int writer_index = *writer_it;
            const auto &writer_instruction =
                single_iteration_instructions[writer_index];
            bool overlaps = false;
            for (const BufferRegion &write_region : writer_instruction.writes) {
              if (AccessOverlapChecker::Overlap(write_region, read_region)) {
                overlaps = true;
                break;
              }
            }
            if (!overlaps) {
              continue;
            }
            producer_index = writer_index;
            distance = 1;
            break;
          }
        }

        if (producer_index == -1) {
          continue;
        }

        auto edge_key =
            std::make_tuple(producer_index, reader_index, buffer, distance);
        if (!unique_edges.insert(edge_key).second) {
          continue;
        }

        ddg.edges.push_back({producer_index, reader_index, buffer, distance});
        if (distance == 0) {
          ddg.forward_predecessors[reader_index].push_back(producer_index);
          ddg.forward_successors[producer_index].push_back(reader_index);
        } else {
          ddg.backward_predecessors[reader_index].push_back(producer_index);
          ddg.backward_successors[producer_index].push_back(reader_index);
        }
      }
    }

    std::set<
        std::tuple<int, int, const BufferNode *, int, SemanticDependencyKind>>
        semantic_unique;
    auto add_semantic = [&](int source, int target, const BufferNode *buffer,
                            int distance, SemanticDependencyKind kind) {
      auto key = std::make_tuple(source, target, buffer, distance, kind);
      if (semantic_unique.insert(key).second) {
        ddg.semantic_edges.push_back({source, target, buffer, distance, kind});
      }
    };
    for (const LocalDependencyEdge &edge : ddg.edges) {
      add_semantic(edge.producer_instruction_id, edge.consumer_instruction_id,
                   edge.buffer, edge.distance, SemanticDependencyKind::kRAW);
    }
    for (const BufferNode *buffer : ddg.buffer_order) {
      const BufferAccessInfo &info = ddg.buffer_access_infos.at(buffer);
      for (int reader : info.read_instruction_indices) {
        for (int writer : info.write_instruction_indices) {
          if (reader < writer) {
            add_semantic(reader, writer, buffer, 0,
                         SemanticDependencyKind::kWAR);
          }
        }
      }
      for (size_t i = 0; i < info.write_instruction_indices.size(); ++i) {
        for (size_t j = i + 1; j < info.write_instruction_indices.size(); ++j) {
          add_semantic(info.write_instruction_indices[i],
                       info.write_instruction_indices[j], buffer, 0,
                       SemanticDependencyKind::kWAW);
        }
      }
      if (!info.read_instruction_indices.empty() &&
          !info.write_instruction_indices.empty()) {
        add_semantic(info.read_instruction_indices.back(),
                     info.write_instruction_indices.front(), buffer, 1,
                     SemanticDependencyKind::kWAR);
        if (info.write_instruction_indices.size() > 1) {
          add_semantic(info.write_instruction_indices.back(),
                       info.write_instruction_indices.front(), buffer, 1,
                       SemanticDependencyKind::kWAW);
        }
      }
    }

    // All cores must encounter all-gather barriers in one common epoch order.
    // Data hazards alone cannot enforce this when collectives use different
    // buffers or ODMA directions, so preserve template order explicitly and
    // close the chain across consecutive logical iterations.
    std::vector<int> all_gather_ids;
    for (const PipelineInstruction &instruction :
         single_iteration_instructions) {
      if (IsAllGatherInstruction(instruction)) {
        all_gather_ids.push_back(instruction.id);
      }
    }
    for (size_t i = 1; i < all_gather_ids.size(); ++i) {
      int source = all_gather_ids[i - 1];
      int target = all_gather_ids[i];
      ddg.ordering_edges.push_back({source, target, 0});
    }
    if (all_gather_ids.size() > 1) {
      int source = all_gather_ids.back();
      int target = all_gather_ids.front();
      ddg.ordering_edges.push_back({source, target, 1});
    }

    return ddg;
  }
};

static bool
ValidateLocalDDG(const std::vector<PipelineInstruction> &instructions,
                 const LocalDDG &ddg) {
  const int instruction_count = static_cast<int>(instructions.size());
  std::set<std::pair<int, const BufferNode *>> reads_with_producer;
  for (const LocalDependencyEdge &edge : ddg.edges) {
    if (edge.producer_instruction_id < 0 ||
        edge.producer_instruction_id >= instruction_count ||
        edge.consumer_instruction_id < 0 ||
        edge.consumer_instruction_id >= instruction_count ||
        edge.buffer == nullptr || edge.distance < 0) {
      return false;
    }
    reads_with_producer.insert({edge.consumer_instruction_id, edge.buffer});
  }
  for (const SemanticDependencyEdge &edge : ddg.semantic_edges) {
    if (edge.source_instruction_id < 0 ||
        edge.source_instruction_id >= instruction_count ||
        edge.target_instruction_id < 0 ||
        edge.target_instruction_id >= instruction_count ||
        edge.buffer == nullptr || edge.distance < 0) {
      return false;
    }
  }
  for (const TemplateOrderEdge &edge : ddg.ordering_edges) {
    if (edge.source_instruction_id < 0 ||
        edge.source_instruction_id >= instruction_count ||
        edge.target_instruction_id < 0 ||
        edge.target_instruction_id >= instruction_count || edge.distance < 0) {
      return false;
    }
  }
  for (int id = 0; id < instruction_count; ++id) {
    for (const BufferRegion &read : instructions[id].reads) {
      if (IsGlobalBuffer(read->buffer)) {
        continue;
      }
      auto access_it = ddg.buffer_access_infos.find(read->buffer.get());
      if (access_it == ddg.buffer_access_infos.end() ||
          access_it->second.write_instruction_indices.empty()) {
        continue;
      }
      if (!reads_with_producer.count({id, read->buffer.get()})) {
        return false;
      }
    }
  }
  return true;
}

static void MaybeWriteGreedyGraphJson(
    const std::vector<PipelineInstruction> &instructions, const LocalDDG &ddg,
    const std::unordered_set<const BufferNode *> &versioned_buffers) {
  const char *path = std::getenv("TL_SUNMMIO_PIPELINE_GRAPH_JSON");
  if (path == nullptr || path[0] == '\0') {
    return;
  }
  std::ofstream out(path);
  if (!out.is_open()) {
    LOG(WARNING) << "Cannot write pipeline graph JSON to " << path;
    return;
  }
  out << "{\n  \"mode\": \"greedy\",\n  \"commands\": [\n";
  for (size_t i = 0; i < instructions.size(); ++i) {
    const PipelineInstruction &instruction = instructions[i];
    out << "    {\"id\": " << instruction.id
        << ", \"iteration_offset\": 0, \"hardware\": "
        << static_cast<int>(instruction.device_type)
        << ", \"resource\": " << instruction.execution_resource
        << ", \"reads\": [";
    for (size_t j = 0; j < instruction.reads.size(); ++j) {
      if (j != 0)
        out << ", ";
      out << "\"" << instruction.reads[j]->buffer->name << "\"";
    }
    out << "], \"writes\": [";
    for (size_t j = 0; j < instruction.writes.size(); ++j) {
      if (j != 0)
        out << ", ";
      out << "\"" << instruction.writes[j]->buffer->name << "\"";
    }
    out << "]}" << (i + 1 == instructions.size() ? "\n" : ",\n");
  }
  out << "  ],\n  \"edges\": [\n";
  size_t edge_index = 0;
  size_t edge_count = ddg.semantic_edges.size() + ddg.ordering_edges.size();
  for (const SemanticDependencyEdge &edge : ddg.semantic_edges) {
    out << "    {\"source\": " << edge.source_instruction_id
        << ", \"target\": " << edge.target_instruction_id << ", \"buffer\": \""
        << edge.buffer->name << "\", \"distance\": " << edge.distance
        << ", \"kind\": \"" << SemanticDependencyKindName(edge.kind) << "\"}"
        << (++edge_index == edge_count ? "\n" : ",\n");
  }
  for (const TemplateOrderEdge &edge : ddg.ordering_edges) {
    out << "    {\"source\": " << edge.source_instruction_id
        << ", \"target\": " << edge.target_instruction_id
        << ", \"buffer\": null, \"distance\": " << edge.distance
        << ", \"kind\": \"collective_order\"}"
        << (++edge_index == edge_count ? "\n" : ",\n");
  }
  out << "  ],\n  \"buffers\": [\n";
  for (size_t i = 0; i < ddg.buffer_order.size(); ++i) {
    const BufferNode *buffer = ddg.buffer_order[i];
    const BufferAccessInfo &access = ddg.buffer_access_infos.at(buffer);
    out << "    {\"name\": \"" << buffer->name
        << "\", \"global\": " << (access.is_global ? "true" : "false")
        << ", \"loop_carried\": "
        << (access.HasLoopCarriedDependence() ? "true" : "false")
        << ", \"versioned\": "
        << (versioned_buffers.count(buffer) ? "true" : "false")
        << ", \"banked\": "
        << (IsRuntimeBankedBuffer(buffer) ? "true" : "false")
        << ", \"classification\": \""
        << (access.is_global
                ? "global"
                : (access.HasLoopCarriedDependence() ? "loop_carried"
                                                     : "local"))
        << "\"}" << (i + 1 == ddg.buffer_order.size() ? "\n" : ",\n");
  }
  out << "  ]\n}\n";
}

static bool
VerifyScheduledWindow(const std::vector<PipelineInstruction> &expected,
                      const std::vector<PipelineInstruction> &scheduled,
                      int command_count) {
  if (expected.size() != scheduled.size())
    return false;
  std::multiset<std::pair<int, int>> expected_instances;
  std::multiset<std::pair<int, int>> scheduled_instances;
  for (const PipelineInstruction &instruction : expected) {
    if (instruction.id < 0 || instruction.id >= command_count ||
        instruction.iter < 0)
      return false;
    expected_instances.insert({instruction.iter, instruction.id});
  }
  for (const PipelineInstruction &instruction : scheduled) {
    if (instruction.id < 0 || instruction.id >= command_count ||
        instruction.iter < 0)
      return false;
    scheduled_instances.insert({instruction.iter, instruction.id});
  }
  return expected_instances == scheduled_instances;
}

static bool
VerifyGreedySchedule(const std::vector<PipelineInstruction> &expected_prologue,
                     const std::vector<PipelineInstruction> &expected_body,
                     const std::vector<PipelineInstruction> &expected_epilogue,
                     const std::vector<PipelineInstruction> &prologue,
                     const std::vector<PipelineInstruction> &body,
                     const std::vector<PipelineInstruction> &epilogue,
                     bool has_epilogue, int command_count) {
  if (!VerifyScheduledWindow(expected_prologue, prologue, command_count) ||
      !VerifyScheduledWindow(expected_body, body, command_count)) {
    return false;
  }
  return !has_epilogue ||
         VerifyScheduledWindow(expected_epilogue, epilogue, command_count);
}

static bool VerifyDynamicLogicalCoverage(
    int extent, int iterations, int command_count,
    const std::vector<PipelineInstruction> &prologue,
    const std::vector<PipelineInstruction> &body,
    const std::map<int, std::vector<PipelineInstruction>> &epilogues) {
  std::map<std::pair<int, int>, int> counts;
  auto record = [&](int base, const std::vector<PipelineInstruction> &window,
                    bool predicate_invalid) {
    for (const PipelineInstruction &instruction : window) {
      int logical_iter = base + instruction.iter;
      if (instruction.id < 0 || instruction.id >= command_count) {
        return false;
      }
      if (logical_iter < 0 || logical_iter >= extent) {
        if (predicate_invalid)
          continue;
        return false;
      }
      counts[{logical_iter, instruction.id}] += 1;
    }
    return true;
  };
  if (!record(0, prologue, false))
    return false;
  int steady_groups = std::max(0, (extent - 1) / iterations);
  for (int group = 0; group < steady_groups; ++group) {
    if (!record(group * iterations, body, false))
      return false;
  }
  int remainder = extent % iterations;
  auto epilogue_it = epilogues.find(remainder);
  if (epilogue_it == epilogues.end() ||
      !record(steady_groups * iterations, epilogue_it->second, true)) {
    return false;
  }
  for (int logical_iter = 0; logical_iter < extent; ++logical_iter) {
    for (int command = 0; command < command_count; ++command) {
      if (counts[{logical_iter, command}] != 1)
        return false;
    }
  }
  return true;
}

/**
 * \brief Identify the prefetch instruction set on top of the local DDG.
 */
class PrefetchInstructionIdentifier {
public:
  static std::vector<bool> Identify(
      const std::vector<PipelineInstruction> &single_iteration_instructions,
      const LocalDDG &local_ddg) {
    const int instruction_count =
        static_cast<int>(single_iteration_instructions.size());
    std::vector<bool> is_prefetch_instruction(instruction_count, false);
    std::deque<int> propagation_queue;

    for (int instruction_index = 0; instruction_index < instruction_count;
         ++instruction_index) {
      if (!IsPrefetchSeed(single_iteration_instructions, local_ddg,
                          instruction_index)) {
        continue;
      }
      is_prefetch_instruction[instruction_index] = true;
      propagation_queue.push_back(instruction_index);
    }

    while (!propagation_queue.empty()) {
      const int producer_instruction_index = propagation_queue.front();
      propagation_queue.pop_front();

      for (int consumer_instruction_index :
           local_ddg.forward_successors[producer_instruction_index]) {
        if (is_prefetch_instruction[consumer_instruction_index]) {
          continue;
        }
        if (!CanPropagatePrefetch(single_iteration_instructions, local_ddg,
                                  is_prefetch_instruction,
                                  consumer_instruction_index)) {
          continue;
        }
        is_prefetch_instruction[consumer_instruction_index] = true;
        propagation_queue.push_back(consumer_instruction_index);
      }
    }

    return is_prefetch_instruction;
  }

private:
  static bool IsPrefetchSeed(
      const std::vector<PipelineInstruction> &single_iteration_instructions,
      const LocalDDG &local_ddg, int instruction_index) {
    const auto &instruction = single_iteration_instructions[instruction_index];
    if (!IsPrefetchValidInstruction(single_iteration_instructions, local_ddg,
                                    instruction_index)) {
      return false;
    }
    if (!local_ddg.backward_predecessors[instruction_index].empty()) {
      return false;
    }
    if (!local_ddg.forward_predecessors[instruction_index].empty()) {
      return false;
    }
    for (const BufferRegion &read_region : instruction.reads) {
      if (!IsGlobalBuffer(read_region->buffer)) {
        return false;
      }
    }
    return true;
  }

  static bool CanPropagatePrefetch(
      const std::vector<PipelineInstruction> &single_iteration_instructions,
      const LocalDDG &local_ddg,
      const std::vector<bool> &is_prefetch_instruction, int instruction_index) {
    if (!IsPrefetchValidInstruction(single_iteration_instructions, local_ddg,
                                    instruction_index)) {
      return false;
    }
    if (!local_ddg.backward_predecessors[instruction_index].empty()) {
      return false;
    }
    if (local_ddg.forward_predecessors[instruction_index].empty()) {
      return false;
    }
    for (int producer_instruction_index :
         local_ddg.forward_predecessors[instruction_index]) {
      if (!is_prefetch_instruction[producer_instruction_index]) {
        return false;
      }
    }
    return true;
  }

  static bool IsPrefetchValidInstruction(
      const std::vector<PipelineInstruction> &single_iteration_instructions,
      const LocalDDG &local_ddg, int instruction_index) {
    const auto &instruction = single_iteration_instructions[instruction_index];
    if (!IsPrefetchCompatibleStmtKind(instruction.stmt)) {
      return false;
    }
    if (instruction.writes.empty()) {
      return false;
    }
    for (const BufferRegion &write_region : instruction.writes) {
      if (WasBufferReadBeforeInstruction(local_ddg, write_region->buffer.get(),
                                         instruction_index)) {
        return false;
      }
    }
    return true;
  }

  static bool IsPrefetchCompatibleStmtKind(const Stmt &stmt) {
    if (const auto *block_realize = stmt.as<BlockRealizeNode>()) {
      return IsPrefetchCompatibleStmtKind(block_realize->block->body);
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      return IsPrefetchCompatibleStmtKind(block->body);
    }
    if (const auto *for_loop = stmt.as<ForNode>()) {
      return IsPrefetchCompatibleStmtKind(for_loop->body);
    }
    if (const auto *let_stmt = stmt.as<LetStmtNode>()) {
      return IsPrefetchCompatibleStmtKind(let_stmt->body);
    }
    if (const auto *attr_stmt = stmt.as<AttrStmtNode>()) {
      return IsPrefetchCompatibleStmtKind(attr_stmt->body);
    }
    if (const auto *assert_stmt = stmt.as<AssertStmtNode>()) {
      return IsPrefetchCompatibleStmtKind(assert_stmt->body);
    }
    if (const auto *allocate_stmt = stmt.as<AllocateNode>()) {
      return IsPrefetchCompatibleStmtKind(allocate_stmt->body);
    }
    if (const auto *decl_buffer_stmt = stmt.as<DeclBufferNode>()) {
      return IsPrefetchCompatibleStmtKind(decl_buffer_stmt->body);
    }
    if (const auto *if_then_else = stmt.as<IfThenElseNode>()) {
      if (!IsPrefetchCompatibleStmtKind(if_then_else->then_case)) {
        return false;
      }
      if (if_then_else->else_case.defined()) {
        return IsPrefetchCompatibleStmtKind(if_then_else->else_case.value());
      }
      return true;
    }
    if (const auto *seq_stmt = stmt.as<SeqStmtNode>()) {
      for (const Stmt &child : seq_stmt->seq) {
        if (!IsPrefetchCompatibleStmtKind(child)) {
          return false;
        }
      }
      return !seq_stmt->seq.empty();
    }
    if (const auto *evaluate = stmt.as<EvaluateNode>()) {
      return IsPrefetchEvaluate(*evaluate);
    }
    return stmt.as<BufferStoreNode>() != nullptr;
  }

  static bool IsPrefetchEvaluate(const EvaluateNode &evaluate) {
    auto call = evaluate.value.as<CallNode>();
    if (!call) {
      return false;
    }
    return call->op.same_as(Op::Get("tl.dma_copy")) ||
           call->op.same_as(Op::Get("tl.broadcast_")) ||
           call->op.same_as(Op::Get("tl.sunmmio_layout_transform"));
  }

  static bool WasBufferReadBeforeInstruction(const LocalDDG &local_ddg,
                                             const BufferNode *buffer,
                                             int instruction_index) {
    auto access_info_it = local_ddg.buffer_access_infos.find(buffer);
    if (access_info_it == local_ddg.buffer_access_infos.end()) {
      return false;
    }
    const auto &access_info = access_info_it->second;
    return access_info.first_read_index != -1 &&
           access_info.first_read_index < instruction_index;
  }
};

/**
 * \brief Identify the buffers that require multiversioning on top of the local
 * DDG.
 */
class MultiversioningIdentifier {
public:
  static std::unordered_set<const BufferNode *> Identify(
      const std::vector<PipelineInstruction> &single_iteration_instructions,
      const LocalDDG &local_ddg) {
    std::unordered_set<const BufferNode *> versioned_buffers;

    // Step 1: every output buffer of a prefetch instruction is a versioning
    // seed.
    for (const auto &instruction : single_iteration_instructions) {
      if (!instruction.is_prefetch) {
        continue;
      }
      for (const BufferRegion &write_region : instruction.writes) {
        if (IsGlobalBuffer(write_region->buffer)) {
          continue;
        }
        versioned_buffers.insert(write_region->buffer.get());
      }
    }

    // Step 2: propagate versioning along the dataflow until convergence.
    bool updated = true;
    while (updated) {
      updated = false;
      for (const BufferNode *buffer : local_ddg.buffer_order) {
        auto access_info_it = local_ddg.buffer_access_infos.find(buffer);
        if (access_info_it == local_ddg.buffer_access_infos.end()) {
          continue;
        }
        const BufferAccessInfo &access_info = access_info_it->second;
        if (access_info.is_global ||
            access_info.write_instruction_indices.empty()) {
          continue;
        }
        if (versioned_buffers.count(buffer) != 0) {
          continue;
        }
        if (access_info.HasLoopCarriedDependence()) {
          continue;
        }
        if (!CanPropagateVersioning(single_iteration_instructions, local_ddg,
                                    versioned_buffers, access_info)) {
          continue;
        }
        versioned_buffers.insert(buffer);
        updated = true;
      }
    }

    return versioned_buffers;
  }

private:
  static bool CanPropagateVersioning(
      const std::vector<PipelineInstruction> &single_iteration_instructions,
      const LocalDDG &local_ddg,
      const std::unordered_set<const BufferNode *> &versioned_buffers,
      const BufferAccessInfo &access_info) {
    for (int writer_instruction_index : access_info.write_instruction_indices) {
      const auto &instruction =
          single_iteration_instructions[writer_instruction_index];
      for (const BufferRegion &read_region : instruction.reads) {
        const BufferNode *input_buffer = read_region->buffer.get();
        if (IsGlobalBuffer(read_region->buffer)) {
          continue;
        }
        if (versioned_buffers.count(input_buffer) == 0) {
          return false;
        }
      }
    }
    return true;
  }
};

/**
 * \brief The staged instruction windows prepared for later scheduling.
 */
struct PipelineStageAssembly {
  int iterations{0};
  int epilogue_iterations{-1};
  std::vector<PipelineInstruction> prologue_instructions;
  std::vector<PipelineInstruction> body_instructions;
  std::vector<PipelineInstruction> epilogue_instructions;
};

/**
 * \brief Assemble the prologue/body/epilogue instruction windows from one
 * iteration.
 */
class PipelineWindowAssembler {
public:
  static PipelineStageAssembly Assemble(
      const std::vector<PipelineInstruction> &single_iteration_instructions,
      int num_stages, const PrimExpr &loop_extent) {
    PipelineStageAssembly assembly;
    assembly.iterations = NumStagesToIterations(num_stages);

    std::vector<PipelineInstruction> prefetch_templates;
    std::vector<PipelineInstruction> compute_templates;
    for (const auto &instruction : single_iteration_instructions) {
      if (instruction.is_prefetch) {
        prefetch_templates.push_back(instruction);
      } else {
        compute_templates.push_back(instruction);
      }
    }

    for (const auto &instruction : prefetch_templates) {
      assembly.prologue_instructions.push_back(
          CloneInstructionForIteration(instruction, 0, false));
    }

    for (int iter = 0; iter < assembly.iterations; ++iter) {
      for (const auto &instruction : compute_templates) {
        assembly.body_instructions.push_back(
            CloneInstructionForIteration(instruction, iter, false));
      }
      for (const auto &instruction : prefetch_templates) {
        const bool participates_in_prefetch_queue =
            iter == assembly.iterations - 1;
        assembly.body_instructions.push_back(CloneInstructionForIteration(
            instruction, iter + 1, participates_in_prefetch_queue));
      }
    }

    int epilogue_iterations =
        TryGetConstantEpilogueIterations(loop_extent, assembly.iterations);
    assembly.epilogue_iterations = epilogue_iterations;
    if (epilogue_iterations == -1) {
      return assembly;
    }
    if (epilogue_iterations == 0) {
      epilogue_iterations = assembly.iterations;
      assembly.epilogue_iterations = epilogue_iterations;
    }

    int epilogue_iter = 0;
    for (int iter = 0; iter < epilogue_iterations - 1; ++iter) {
      for (const auto &instruction : compute_templates) {
        assembly.epilogue_instructions.push_back(
            CloneInstructionForIteration(instruction, iter, false));
      }
      for (const auto &instruction : prefetch_templates) {
        assembly.epilogue_instructions.push_back(
            CloneInstructionForIteration(instruction, iter + 1, false));
      }
      epilogue_iter += 1;
    }

    for (const auto &instruction : compute_templates) {
      assembly.epilogue_instructions.push_back(
          CloneInstructionForIteration(instruction, epilogue_iter, false));
    }

    return assembly;
  }

private:
  static int NumStagesToIterations(int num_stages) { return num_stages; }

  static int TryGetConstantEpilogueIterations(const PrimExpr &loop_extent,
                                              int iterations) {
    PrimExpr epilogue_iterations_expr = floormod(loop_extent, iterations);
    if (const auto *mod_int = epilogue_iterations_expr.as<IntImmNode>()) {
      return mod_int->value;
    }
    return -1;
  }

  static PipelineInstruction
  CloneInstructionForIteration(const PipelineInstruction &instruction, int iter,
                               bool is_prefetch) {
    PipelineInstruction cloned = instruction;
    cloned.iter = iter;
    cloned.name = std::to_string(iter) + "-" + std::to_string(cloned.id);
    cloned.is_prefetch = is_prefetch;
    cloned.scheduled_start = -1.0f;
    cloned.scheduled_end = -1.0f;
    cloned.finished = false;
    return cloned;
  }
};

class PipelineDevice {
public:
  explicit PipelineDevice(int resource) : resource(resource) {}

  void AssignInstruction(PipelineInstruction *instruction, float time) {
    current_instruction = instruction;
    busy = true;
    instruction_end_time = time + instruction->delay;
    instruction->scheduled_start = time;
    instruction->scheduled_end = instruction_end_time;
  }

  void PassTime(float time) {
    if (busy && time >= instruction_end_time) {
      current_instruction->finished = true;
      busy = false;
      current_instruction = nullptr;
      instruction_end_time = std::numeric_limits<float>::max();
    }
  }

  int resource{-1};
  bool busy{false};
  PipelineInstruction *current_instruction{nullptr};
  float instruction_end_time{std::numeric_limits<float>::max()};
};

class GlobalPipelineScheduler {
public:
  std::vector<PipelineInstruction> instructions;
  int iter_mod_{-1};
  bool debug_{false};

  GlobalPipelineScheduler() {
    devices_.push_back(
        PipelineDevice(static_cast<int>(IlpResourceType::kTensorCore)));
    devices_.push_back(
        PipelineDevice(static_cast<int>(IlpResourceType::kVectorCore)));
    devices_.push_back(
        PipelineDevice(static_cast<int>(IlpResourceType::kODMA0)));
    devices_.push_back(
        PipelineDevice(static_cast<int>(IlpResourceType::kODMA1)));
  }

  void SetVersionedBuffers(
      const std::unordered_set<const BufferNode *> &versioned_buffers) {
    versioned_buffers_ = versioned_buffers;
  }

  void SetBankColoring(const GreedyBankColoring &coloring) {
    writer_phases_ = coloring.writer_phases;
    reader_phases_ = coloring.reader_phases;
  }

  void SetTemplateOrderEdges(const std::vector<TemplateOrderEdge> &edges) {
    template_order_edges_ = edges;
  }

  void BuildDependencyGraph() {
    int instruction_count = static_cast<int>(instructions.size());
    predecessors_.assign(instruction_count, {});
    successors_.assign(instruction_count, {});
    topological_order_.resize(instruction_count);
    for (int instruction_index = 0; instruction_index < instruction_count;
         ++instruction_index) {
      topological_order_[instruction_index] = instruction_index;
    }
    std::sort(topological_order_.begin(), topological_order_.end(),
              [&](int lhs, int rhs) {
                if (instructions[lhs].iter != instructions[rhs].iter) {
                  return instructions[lhs].iter < instructions[rhs].iter;
                }
                return instructions[lhs].id < instructions[rhs].id;
              });

    enum class AccessType { kRead, kWrite };
    struct AccessRecord {
      BufferRegion region;
      int instruction_index;
      AccessType type;
      int instance_id;
    };

    std::unordered_map<const BufferNode *, std::vector<AccessRecord>>
        buffer_access_history;

    for (int ordered_index = 0; ordered_index < instruction_count;
         ++ordered_index) {
      int current_index = topological_order_[ordered_index];
      const PipelineInstruction &current_instruction =
          instructions[current_index];

      for (const BufferRegion &read_region : current_instruction.reads) {
        const BufferNode *buffer = read_region->buffer.get();
        int current_instance =
            GetAccessInstanceId(current_instruction, buffer, false);
        auto history_it = buffer_access_history.find(buffer);
        if (history_it == buffer_access_history.end()) {
          continue;
        }
        auto &history = history_it->second;
        for (auto it = history.rbegin(); it != history.rend(); ++it) {
          if (ShouldSkipVersionedAccess(buffer, current_instance,
                                        it->instance_id)) {
            continue;
          }
          if (it->type == AccessType::kWrite &&
              AccessOverlapChecker::Overlap(read_region, it->region)) {
            AddDependency(it->instruction_index, current_index);
            break;
          }
        }
      }

      for (const BufferRegion &write_region : current_instruction.writes) {
        const BufferNode *buffer = write_region->buffer.get();
        int current_instance =
            GetAccessInstanceId(current_instruction, buffer, true);
        auto history_it = buffer_access_history.find(buffer);
        if (history_it == buffer_access_history.end()) {
          continue;
        }
        auto &history = history_it->second;
        for (auto it = history.rbegin(); it != history.rend(); ++it) {
          if (ShouldSkipVersionedAccess(buffer, current_instance,
                                        it->instance_id)) {
            continue;
          }
          if (!AccessOverlapChecker::Overlap(write_region, it->region)) {
            continue;
          }
          AddDependency(it->instruction_index, current_index);
          if (it->type == AccessType::kWrite) {
            break;
          }
        }
      }

      for (const BufferRegion &read_region : current_instruction.reads) {
        buffer_access_history[read_region->buffer.get()].push_back(
            {read_region, current_index, AccessType::kRead,
             GetAccessInstanceId(current_instruction, read_region->buffer.get(),
                                 false)});
      }
      for (const BufferRegion &write_region : current_instruction.writes) {
        buffer_access_history[write_region->buffer.get()].push_back(
            {write_region, current_index, AccessType::kWrite,
             GetAccessInstanceId(current_instruction,
                                 write_region->buffer.get(), true)});
      }
    }

    std::map<std::pair<int, int>, int> instance_index;
    for (int index = 0; index < instruction_count; ++index) {
      instance_index[{instructions[index].iter, instructions[index].id}] =
          index;
    }
    for (const TemplateOrderEdge &edge : template_order_edges_) {
      for (int source_index = 0; source_index < instruction_count;
           ++source_index) {
        const PipelineInstruction &source = instructions[source_index];
        if (source.id != edge.source_instruction_id) {
          continue;
        }
        auto target = instance_index.find(
            {source.iter + edge.distance, edge.target_instruction_id});
        if (target != instance_index.end()) {
          AddDependency(source_index, target->second);
        }
      }
    }
  }

  void CalculateBottomLevels() {
    bottom_levels_.assign(instructions.size(), 0);
    for (auto it = topological_order_.rbegin(); it != topological_order_.rend();
         ++it) {
      int instruction_index = *it;
      int max_successor_level = 0;
      for (int successor_index : successors_[instruction_index]) {
        max_successor_level =
            std::max(max_successor_level, bottom_levels_[successor_index]);
      }
      bottom_levels_[instruction_index] = static_cast<int>(
          instructions[instruction_index].delay + max_successor_level);
    }
  }

  void DumpGraph(const std::string &file_name) const {
    if (!debug_) {
      return;
    }
    std::ofstream log_file(file_name, std::ios::out);
    if (!log_file.is_open()) {
      return;
    }
    log_file << "num_commands " << instructions.size() << "\n";
    log_file << "nodes\n";
    for (int instruction_index : topological_order_) {
      const auto &instruction = instructions[instruction_index];
      int bottom_level = -1;
      if (instruction_index < static_cast<int>(bottom_levels_.size())) {
        bottom_level = bottom_levels_[instruction_index];
      }
      log_file << instruction_index << " " << instruction.name << " "
               << instruction.iter << " " << instruction.id << " "
               << instruction.execution_resource << " "
               << static_cast<int>(instruction.is_prefetch) << " "
               << bottom_level << "\n";
    }
    log_file << "edges\n";
    for (int src = 0; src < static_cast<int>(successors_.size()); ++src) {
      for (int dst : successors_[src]) {
        log_file << src << " " << instructions[src].name << " -> " << dst << " "
                 << instructions[dst].name << "\n";
      }
    }
  }

  std::vector<PipelineInstruction> Schedule(const std::string &log_file_name) {
    ResetSchedulingState();
    std::ofstream log_file;
    if (debug_) {
      log_file.open(log_file_name, std::ios::out);
    }

    std::vector<PipelineInstruction *> primary_queue;
    std::vector<PipelineInstruction *> prefetch_queue;
    for (auto &instruction : instructions) {
      if (instruction.is_prefetch) {
        prefetch_queue.push_back(&instruction);
      } else {
        primary_queue.push_back(&instruction);
      }
    }

    auto primary_cmp = [&](PipelineInstruction *lhs, PipelineInstruction *rhs) {
      if (bottom_levels_[GetInstructionIndex(*lhs)] !=
          bottom_levels_[GetInstructionIndex(*rhs)]) {
        return bottom_levels_[GetInstructionIndex(*lhs)] >
               bottom_levels_[GetInstructionIndex(*rhs)];
      }
      if (lhs->iter != rhs->iter) {
        return lhs->iter < rhs->iter;
      }
      return lhs->id < rhs->id;
    };
    std::sort(primary_queue.begin(), primary_queue.end(), primary_cmp);

    float time = 0.0f;
    std::vector<PipelineInstruction> schedule;
    while (!primary_queue.empty()) {
      for (PipelineInstruction *instruction : primary_queue) {
        if (instruction->finished) {
          continue;
        }
        if (!ArePredecessorsFinished(*instruction)) {
          continue;
        }
        if (!AreBanksFree(*instruction, time)) {
          continue;
        }
        for (auto &device : devices_) {
          if (device.resource == instruction->execution_resource &&
              !device.busy) {
            device.AssignInstruction(instruction, time);
            ReserveBanks(*instruction, instruction->scheduled_end);
            schedule.push_back(*instruction);
            break;
          }
        }
      }

      float pass_time = std::numeric_limits<float>::max();
      for (const auto &device : devices_) {
        pass_time = std::min(pass_time, device.instruction_end_time - time);
      }
      time += pass_time;
      for (auto &device : devices_) {
        device.PassTime(time);
      }
      primary_queue.erase(std::remove_if(primary_queue.begin(),
                                         primary_queue.end(),
                                         [](PipelineInstruction *instruction) {
                                           return instruction->finished;
                                         }),
                          primary_queue.end());
    }

    struct Interval {
      float start;
      float end;
    };
    std::unordered_map<int, std::vector<Interval>> busy_intervals;
    std::array<std::vector<Interval>, static_cast<int>(PhysicalSramBank::Count)>
        bank_busy_intervals;
    for (const auto &instruction : schedule) {
      busy_intervals[instruction.execution_resource].push_back(
          {instruction.scheduled_start, instruction.scheduled_end});
      for (PhysicalSramBank bank :
           GetOccupiedSramBanks(instruction, versioned_buffers_, iter_mod_,
                                writer_phases_, reader_phases_)) {
        bank_busy_intervals[static_cast<int>(bank)].push_back(
            {instruction.scheduled_start, instruction.scheduled_end});
      }
    }
    for (auto &kv : busy_intervals) {
      auto &intervals = kv.second;
      std::sort(intervals.begin(), intervals.end(),
                [](const Interval &lhs, const Interval &rhs) {
                  return lhs.start < rhs.start;
                });
    }
    for (auto &intervals : bank_busy_intervals) {
      std::sort(intervals.begin(), intervals.end(),
                [](const Interval &lhs, const Interval &rhs) {
                  return lhs.start < rhs.start;
                });
    }

    std::sort(prefetch_queue.begin(), prefetch_queue.end(),
              [](PipelineInstruction *lhs, PipelineInstruction *rhs) {
                if (lhs->iter != rhs->iter) {
                  return lhs->iter < rhs->iter;
                }
                return lhs->id < rhs->id;
              });

    auto insert_interval = [](std::vector<Interval> &intervals, Interval x) {
      auto pos = std::lower_bound(intervals.begin(), intervals.end(), x,
                                  [](const Interval &lhs, const Interval &rhs) {
                                    return lhs.start < rhs.start;
                                  });
      intervals.insert(pos, x);
    };

    std::vector<int> prefetch_indices;
    prefetch_indices.reserve(prefetch_queue.size());
    for (PipelineInstruction *instruction : prefetch_queue) {
      prefetch_indices.push_back(GetInstructionIndex(*instruction));
    }
    std::vector<int> indegree(instructions.size(), 0);
    for (int instruction_index : prefetch_indices) {
      int degree = 0;
      for (int predecessor_index : predecessors_[instruction_index]) {
        if (instructions[predecessor_index].is_prefetch) {
          degree += 1;
        }
      }
      indegree[instruction_index] = degree;
    }

    std::deque<int> ready_prefetch;
    for (int instruction_index : prefetch_indices) {
      if (indegree[instruction_index] == 0) {
        ready_prefetch.push_back(instruction_index);
      }
    }

    int scheduled_prefetch = 0;
    while (!ready_prefetch.empty()) {
      int instruction_index = ready_prefetch.front();
      ready_prefetch.pop_front();
      PipelineInstruction *instruction = &instructions[instruction_index];

      float ready_time = 0.0f;
      for (int predecessor_index : predecessors_[instruction_index]) {
        if (instructions[predecessor_index].scheduled_end >= 0) {
          ready_time = std::max(ready_time,
                                instructions[predecessor_index].scheduled_end);
        }
      }

      float duration = instruction->delay;
      auto &intervals = busy_intervals[instruction->execution_resource];
      float start_time = ready_time;
      std::vector<std::vector<Interval> *> required_intervals{&intervals};
      for (PhysicalSramBank bank :
           GetOccupiedSramBanks(*instruction, versioned_buffers_, iter_mod_,
                                writer_phases_, reader_phases_)) {
        required_intervals.push_back(
            &bank_busy_intervals[static_cast<int>(bank)]);
      }
      while (!instruction->finished) {
        float next_start = start_time;
        for (const std::vector<Interval> *resource_intervals :
             required_intervals) {
          for (const Interval &interval : *resource_intervals) {
            if (start_time + duration <= interval.start) {
              break;
            }
            if (start_time < interval.end &&
                start_time + duration > interval.start) {
              next_start = std::max(next_start, interval.end);
              break;
            }
          }
        }
        if (next_start != start_time) {
          start_time = next_start;
          continue;
        }
        instruction->scheduled_start = start_time;
        instruction->scheduled_end = start_time + duration;
        instruction->finished = true;
        for (std::vector<Interval> *resource_intervals : required_intervals) {
          insert_interval(*resource_intervals, {instruction->scheduled_start,
                                                instruction->scheduled_end});
        }
        schedule.push_back(*instruction);
        scheduled_prefetch += 1;
      }
      ICHECK(instruction->finished)
          << "Failed to insert prefetch instruction " << instruction->name;

      for (int successor_index : successors_[instruction_index]) {
        if (!instructions[successor_index].is_prefetch) {
          continue;
        }
        indegree[successor_index] -= 1;
        if (indegree[successor_index] == 0) {
          ready_prefetch.push_back(successor_index);
        }
      }
    }

    ICHECK(scheduled_prefetch == static_cast<int>(prefetch_indices.size()))
        << "Cycle detected in prefetch dependency subgraph.";

    std::sort(
        schedule.begin(), schedule.end(),
        [](const PipelineInstruction &lhs, const PipelineInstruction &rhs) {
          if (lhs.scheduled_start != rhs.scheduled_start) {
            return lhs.scheduled_start < rhs.scheduled_start;
          }
          int lhs_priority = GetGreedyIssuePriority(lhs);
          int rhs_priority = GetGreedyIssuePriority(rhs);
          if (lhs_priority != rhs_priority) {
            return lhs_priority < rhs_priority;
          }
          return lhs.name < rhs.name;
        });
    if (debug_ && log_file.is_open()) {
      for (const auto &instruction : schedule) {
        log_file << (instruction.is_prefetch ? "p:" : "") << instruction.name
                 << " " << instruction.execution_resource << " "
                 << instruction.scheduled_start << " " << instruction.delay
                 << "\n";
      }
    }
    return schedule;
  }

private:
  bool ShouldSkipVersionedAccess(const BufferNode *buffer, int current_instance,
                                 int previous_instance) const {
    if (versioned_buffers_.count(buffer) == 0) {
      return false;
    }
    return previous_instance != current_instance;
  }

  int GetAccessInstanceId(const PipelineInstruction &instruction,
                          const BufferNode *buffer, bool is_write) const {
    int slot = GetVersionId(instruction);
    if (!IsRuntimeBankedBuffer(buffer))
      return slot;
    int phase = LookupBankPhase(is_write ? writer_phases_ : reader_phases_,
                                buffer, instruction.id);
    int bank = (slot + phase) & 1;
    int version = slot / 2;
    return bank * std::max(1, iter_mod_) + version;
  }

  int GetVersionId(const PipelineInstruction &instruction) const {
    if (iter_mod_ > 0) {
      return instruction.iter % iter_mod_;
    }
    return instruction.iter;
  }

  void AddDependency(int predecessor_index, int successor_index) {
    for (int existing_predecessor : predecessors_[successor_index]) {
      if (existing_predecessor == predecessor_index) {
        return;
      }
    }
    predecessors_[successor_index].push_back(predecessor_index);
    successors_[predecessor_index].push_back(successor_index);
  }

  void ResetSchedulingState() {
    for (auto &instruction : instructions) {
      instruction.finished = false;
      instruction.scheduled_start = -1.0f;
      instruction.scheduled_end = -1.0f;
    }
    for (auto &device : devices_) {
      device.busy = false;
      device.current_instruction = nullptr;
      device.instruction_end_time = std::numeric_limits<float>::max();
    }
    bank_busy_until_.fill(-1.0f);
  }

  bool AreBanksFree(const PipelineInstruction &instruction, float time) const {
    for (PhysicalSramBank bank :
         GetOccupiedSramBanks(instruction, versioned_buffers_, iter_mod_,
                              writer_phases_, reader_phases_)) {
      if (bank_busy_until_[static_cast<int>(bank)] > time) {
        return false;
      }
    }
    return true;
  }

  void ReserveBanks(const PipelineInstruction &instruction, float end_time) {
    for (PhysicalSramBank bank :
         GetOccupiedSramBanks(instruction, versioned_buffers_, iter_mod_,
                              writer_phases_, reader_phases_)) {
      bank_busy_until_[static_cast<int>(bank)] = end_time;
    }
  }

  bool ArePredecessorsFinished(const PipelineInstruction &instruction) const {
    int instruction_index = GetInstructionIndex(instruction);
    for (int predecessor_index : predecessors_[instruction_index]) {
      if (!instructions[predecessor_index].finished) {
        return false;
      }
    }
    return true;
  }

  int GetInstructionIndex(const PipelineInstruction &instruction) const {
    return static_cast<int>(&instruction - instructions.data());
  }

  std::vector<PipelineDevice> devices_;
  std::array<float, static_cast<int>(PhysicalSramBank::Count)>
      bank_busy_until_{};
  std::unordered_set<const BufferNode *> versioned_buffers_;
  PerCommandBankPhases writer_phases_;
  PerCommandBankPhases reader_phases_;
  std::vector<TemplateOrderEdge> template_order_edges_;
  std::vector<std::vector<int>> predecessors_;
  std::vector<std::vector<int>> successors_;
  std::vector<int> topological_order_;
  std::vector<int> bottom_levels_;
};

class SunmmioPipelinePlanner : public StmtExprMutator {
public:
  static Stmt Substitute(const PrimFunc &f, bool debug) {
    SunmmioPipelinePlanner substituter(f, debug);
    return substituter(f->body);
  }

  SunmmioPipelinePlanner(const PrimFunc &f, bool debug)
      : stmt_rw_collector_(f), debug_(debug) {}

  StmtReadWriteCollector stmt_rw_collector_;
  bool debug_ = false;

  Stmt VisitStmt_(const ForNode *op) final {
    // 1. Intercept the pipelined loops
    int num_stages = -1;
    if (auto ann = op->annotations.Get("num_stages")) {
      num_stages = Downcast<IntImm>(ann.value())->value;
    }
    if (num_stages <= 0) {
      return StmtExprMutator::VisitStmt_(op);
    }
    arith::Analyzer analyzer;
    PrimExpr simplified_extent = analyzer.Simplify(op->extent);
    const auto *extent = simplified_extent.as<IntImmNode>();
    if (extent != nullptr && extent->value < num_stages) {
      For fallback = Downcast<For>(StmtExprMutator::VisitStmt_(op));
      return MakePipelineFallback(fallback, "greedy", "planning",
                                  "short_extent_unsupported");
    }
    if (extent == nullptr && num_stages != 2) {
      For fallback = Downcast<For>(StmtExprMutator::VisitStmt_(op));
      return MakePipelineFallback(
          fallback,
          PipelineDiagnostic{false, "greedy", "planning",
                             "dynamic_version_count_unsupported",
                             "dynamic Greedy currently requires num_stages=2"});
    }

    // 2. Peel off the outer layers to find the true body sequence
    auto inner_stmt = op->body;
    while (true) {
      if (const auto *block = inner_stmt.as<BlockRealizeNode>()) {
        inner_stmt = block->block->body;
      } else if (const auto *ite = inner_stmt.as<IfThenElseNode>()) {
        ICHECK(!ite->else_case.defined()) << "Not supported";
        inner_stmt = ite->then_case;
      } else if (const auto *let = inner_stmt.as<LetStmtNode>()) {
        inner_stmt = let->body;
      } else {
        break;
      }
    }

    const SeqStmtNode *pipeline_body_seq = inner_stmt.as<SeqStmtNode>();
    ICHECK(pipeline_body_seq) << "Pipeline body must be a SeqStmt";
    ICHECK(op->kind == ForKind::kSerial) << "Pipeline loop must be serial";
    // 3. Stage 1: Build the PipelineInstruction containers
    std::vector<PipelineInstruction> single_iteration_instructions;

    for (size_t i = 0; i < pipeline_body_seq->seq.size(); ++i) {
      const Stmt &stmt = pipeline_body_seq->seq[i];
      if (!stmt.as<BlockRealizeNode>() && !stmt.as<EvaluateNode>() &&
          !stmt.as<ForNode>()) {
        // HardwareMapper intentionally handles hardware commands only. Scalar
        // bookkeeping stores and conditional command groups must first be
        // normalized before they can be scheduled safely.
        For fallback = Downcast<For>(StmtExprMutator::VisitStmt_(op));
        return MakePipelineFallback(fallback, "greedy", "planning",
                                    "unsupported_statement");
      }
      PipelineInstruction instruction(static_cast<int>(i), 0, stmt);
      instruction.device_type = HardwareMapper::Map(instruction.stmt);
      instruction.ExtractRegions(stmt_rw_collector_);
      instruction.execution_resource = GetGreedyExecutionResource(instruction);
      instruction.delay =
          CostModel::EstimateDelay(instruction.device_type, instruction.stmt);
      single_iteration_instructions.push_back(instruction);
    }

    if (debug_) {
      std::cout << "[Pipeline Planner] Found pipeline loop with " << num_stages
                << " stages.\n";
      std::cout << "[Pipeline Planner] Extracted "
                << single_iteration_instructions.size() << " instructions.\n";
      for (const auto &instruction : single_iteration_instructions) {
        std::cout << "  - ID: " << instruction.id
                  << ", Device: " << static_cast<int>(instruction.device_type)
                  << ", Resource: " << instruction.execution_resource
                  << ", Delay: " << instruction.delay
                  << ", Reads: " << instruction.reads.size()
                  << ", Writes: " << instruction.writes.size() << "\n";
      }
    }

    // 4. Stage 2.1: Build the local DDG for a single iteration.
    LocalDDG local_ddg = LocalDDGBuilder::Build(single_iteration_instructions);
    if (!ValidateLocalDDG(single_iteration_instructions, local_ddg)) {
      For fallback = Downcast<For>(StmtExprMutator::VisitStmt_(op));
      return MakePipelineFallback(fallback, "greedy", "graph_validation",
                                  "incomplete_access_info");
    }

    if (debug_) {
      int forward_edge_count = 0;
      int backward_edge_count = 0;
      for (const auto &edge : local_ddg.edges) {
        if (edge.distance == 0) {
          ++forward_edge_count;
        } else {
          ++backward_edge_count;
        }
      }
      std::cout << "[Pipeline Planner] Local DDG built with "
                << local_ddg.edges.size() << " RAW edges.\n";
      std::cout << "  - Forward edges (D=0): " << forward_edge_count << "\n";
      std::cout << "  - Backward edges (D>0): " << backward_edge_count << "\n";
    }

    // 5. Stage 2.2: Identify prefetch instructions on the local DDG.
    std::vector<bool> is_prefetch_instruction =
        PrefetchInstructionIdentifier::Identify(single_iteration_instructions,
                                                local_ddg);
    for (size_t instruction_index = 0;
         instruction_index < single_iteration_instructions.size();
         ++instruction_index) {
      single_iteration_instructions[instruction_index].is_prefetch =
          is_prefetch_instruction[instruction_index];
    }

    if (debug_) {
      int prefetch_instruction_count = 0;
      for (const auto &instruction : single_iteration_instructions) {
        if (!instruction.is_prefetch) {
          continue;
        }
        ++prefetch_instruction_count;
      }
      std::cout << "[Pipeline Planner] Identified "
                << prefetch_instruction_count << " prefetch instructions.\n";
      for (const auto &instruction : single_iteration_instructions) {
        if (!instruction.is_prefetch) {
          continue;
        }
        std::cout << "  - Prefetch instruction ID: " << instruction.id
                  << ", Name: " << instruction.name << "\n";
      }
    }

    // 6. Stage 2.3: Identify multiversion buffers on the local DDG.
    std::unordered_set<const BufferNode *> versioned_buffers =
        MultiversioningIdentifier::Identify(single_iteration_instructions,
                                            local_ddg);
    MaybeWriteGreedyGraphJson(single_iteration_instructions, local_ddg,
                              versioned_buffers);

    if (debug_) {
      std::cout << "[Pipeline Planner] Identified " << versioned_buffers.size()
                << " versioned buffers.\n";
      for (const BufferNode *buffer : local_ddg.buffer_order) {
        if (versioned_buffers.count(buffer) == 0) {
          continue;
        }
        std::cout << "  - Versioned buffer: " << buffer->name << "\n";
      }
    }

    // 7. Stage 3: Assemble the prologue/body/epilogue instruction windows.
    PipelineStageAssembly stage_assembly = PipelineWindowAssembler::Assemble(
        single_iteration_instructions, num_stages, op->extent);

    if (debug_) {
      int body_prefetch_instruction_count = 0;
      for (const auto &instruction : stage_assembly.body_instructions) {
        if (instruction.is_prefetch) {
          ++body_prefetch_instruction_count;
        }
      }
      std::cout << "[Pipeline Planner] Assembled stage windows.\n";
      std::cout << "  - Iterations: " << stage_assembly.iterations << "\n";
      std::cout << "  - Prologue instructions: "
                << stage_assembly.prologue_instructions.size() << "\n";
      std::cout << "  - Body instructions: "
                << stage_assembly.body_instructions.size() << "\n";
      std::cout << "  - Body prefetch instructions: "
                << body_prefetch_instruction_count << "\n";
      std::cout << "  - Epilogue instructions: "
                << stage_assembly.epilogue_instructions.size() << "\n";
      std::cout << "  - Epilogue iterations: "
                << stage_assembly.epilogue_iterations << "\n";
    }

    // 8. Stage 4: Build the global DDG and run the two-phase scheduler.
    int faster = -1;
    auto pass_ctx = tvm::transform::PassContext::Current();
    auto faster_config = pass_ctx->GetConfig<Integer>(kGreedySunmmioFaster);
    if (faster_config) {
      faster = faster_config.value()->value;
    } else if (const char *env_faster = std::getenv("TL_SUNMMIO_FASTER")) {
      faster = std::stoi(env_faster);
    }
    size_t total_coloring_candidates = 0;
    std::vector<GreedyBankColoring> coloring_candidates =
        BuildGreedyBankColorings(local_ddg, versioned_buffers, faster,
                                 &total_coloring_candidates);
    GreedyBankColoring selected_coloring = coloring_candidates.front();
    float selected_body_makespan = std::numeric_limits<float>::max();
    for (const GreedyBankColoring &candidate : coloring_candidates) {
      GlobalPipelineScheduler candidate_scheduler;
      candidate_scheduler.instructions = stage_assembly.body_instructions;
      candidate_scheduler.iter_mod_ = stage_assembly.iterations;
      candidate_scheduler.SetVersionedBuffers(versioned_buffers);
      candidate_scheduler.SetBankColoring(candidate);
      candidate_scheduler.SetTemplateOrderEdges(local_ddg.ordering_edges);
      candidate_scheduler.BuildDependencyGraph();
      candidate_scheduler.CalculateBottomLevels();
      std::vector<PipelineInstruction> candidate_schedule =
          candidate_scheduler.Schedule("");
      float makespan = 0.0f;
      for (const PipelineInstruction &instruction : candidate_schedule) {
        makespan = std::max(makespan, instruction.scheduled_end);
      }
      if (makespan < selected_body_makespan ||
          (makespan == selected_body_makespan &&
           candidate.bits < selected_coloring.bits)) {
        selected_body_makespan = makespan;
        selected_coloring = candidate;
      }
    }

    GlobalPipelineScheduler prologue_scheduler;
    prologue_scheduler.instructions = stage_assembly.prologue_instructions;
    prologue_scheduler.iter_mod_ = stage_assembly.iterations;
    prologue_scheduler.debug_ = debug_;
    prologue_scheduler.SetVersionedBuffers(versioned_buffers);
    prologue_scheduler.SetBankColoring(selected_coloring);
    prologue_scheduler.SetTemplateOrderEdges(local_ddg.ordering_edges);
    prologue_scheduler.BuildDependencyGraph();
    prologue_scheduler.CalculateBottomLevels();
    std::vector<PipelineInstruction> prologue_schedule =
        prologue_scheduler.Schedule("prologue.log");

    GlobalPipelineScheduler body_scheduler;
    body_scheduler.instructions = stage_assembly.body_instructions;
    body_scheduler.iter_mod_ = stage_assembly.iterations;
    body_scheduler.debug_ = debug_;
    body_scheduler.SetVersionedBuffers(versioned_buffers);
    body_scheduler.SetBankColoring(selected_coloring);
    body_scheduler.SetTemplateOrderEdges(local_ddg.ordering_edges);
    body_scheduler.BuildDependencyGraph();
    body_scheduler.CalculateBottomLevels();
    body_scheduler.DumpGraph("body_graph.log");
    std::vector<PipelineInstruction> body_schedule =
        body_scheduler.Schedule("body.log");

    std::vector<PipelineInstruction> epilogue_schedule;
    std::map<int, std::vector<PipelineInstruction>> dynamic_epilogue_schedules;
    if (stage_assembly.epilogue_iterations != -1) {
      GlobalPipelineScheduler epilogue_scheduler;
      epilogue_scheduler.instructions = stage_assembly.epilogue_instructions;
      epilogue_scheduler.iter_mod_ = stage_assembly.iterations;
      epilogue_scheduler.debug_ = debug_;
      epilogue_scheduler.SetVersionedBuffers(versioned_buffers);
      epilogue_scheduler.SetBankColoring(selected_coloring);
      epilogue_scheduler.SetTemplateOrderEdges(local_ddg.ordering_edges);
      epilogue_scheduler.BuildDependencyGraph();
      epilogue_scheduler.CalculateBottomLevels();
      epilogue_schedule = epilogue_scheduler.Schedule("epilogue.log");
    } else {
      for (int remainder = 0; remainder < stage_assembly.iterations;
           ++remainder) {
        int effective_remainder =
            remainder == 0 ? stage_assembly.iterations : remainder;
        PipelineStageAssembly remainder_assembly =
            PipelineWindowAssembler::Assemble(single_iteration_instructions,
                                              num_stages,
                                              Integer(effective_remainder));
        GlobalPipelineScheduler epilogue_scheduler;
        epilogue_scheduler.instructions =
            remainder_assembly.epilogue_instructions;
        epilogue_scheduler.iter_mod_ = stage_assembly.iterations;
        epilogue_scheduler.SetVersionedBuffers(versioned_buffers);
        epilogue_scheduler.SetBankColoring(selected_coloring);
        epilogue_scheduler.SetTemplateOrderEdges(local_ddg.ordering_edges);
        epilogue_scheduler.BuildDependencyGraph();
        epilogue_scheduler.CalculateBottomLevels();
        dynamic_epilogue_schedules[remainder] = epilogue_scheduler.Schedule("");
        if (!VerifyScheduledWindow(
                remainder_assembly.epilogue_instructions,
                dynamic_epilogue_schedules[remainder],
                static_cast<int>(single_iteration_instructions.size()))) {
          For fallback = Downcast<For>(StmtExprMutator::VisitStmt_(op));
          return MakePipelineFallback(
              fallback,
              PipelineDiagnostic{false, "greedy", "schedule_validation",
                                 "invalid_schedule_order",
                                 "dynamic epilogue remainder " +
                                     std::to_string(remainder)});
        }
      }
    }

    if (!VerifyGreedySchedule(
            stage_assembly.prologue_instructions,
            stage_assembly.body_instructions,
            stage_assembly.epilogue_instructions, prologue_schedule,
            body_schedule, epilogue_schedule,
            stage_assembly.epilogue_iterations != -1,
            static_cast<int>(single_iteration_instructions.size()))) {
      For fallback = Downcast<For>(StmtExprMutator::VisitStmt_(op));
      return MakePipelineFallback(
          fallback,
          PipelineDiagnostic{false, "greedy", "schedule_validation",
                             "invalid_schedule_order",
                             "scheduled window does not match assembled "
                             "logical command instances"});
    }
    if (stage_assembly.epilogue_iterations == -1) {
      int verification_extent = std::max(4, stage_assembly.iterations * 2 + 2);
      for (int extent_value = 1; extent_value <= verification_extent;
           ++extent_value) {
        if (!VerifyDynamicLogicalCoverage(
                extent_value, stage_assembly.iterations,
                static_cast<int>(single_iteration_instructions.size()),
                prologue_schedule, body_schedule, dynamic_epilogue_schedules)) {
          For fallback = Downcast<For>(StmtExprMutator::VisitStmt_(op));
          return MakePipelineFallback(
              fallback,
              PipelineDiagnostic{false, "greedy", "schedule_validation",
                                 "logical_iteration_out_of_bounds",
                                 "coverage failed for representative extent " +
                                     std::to_string(extent_value)});
        }
      }
    }

    if (debug_) {
      std::cout << "[Pipeline Planner] Scheduling finished.\n";
      std::cout << "  - Prologue orders: " << prologue_schedule.size() << "\n";
      std::cout << "  - Body orders: " << body_schedule.size() << "\n";
      std::cout << "  - Epilogue orders: " << epilogue_schedule.size() << "\n";
    }

    // 9. Stage 5: Persist the pipeline metadata onto the loop annotations.
    Map<String, Any> annotations;
    for (const auto &[key, value] : op->annotations) {
      if (key != "num_stages" && key != "versioned_buffers") {
        annotations.Set(key, value);
      }
    }
    annotations.Set("iterations", stage_assembly.iterations);
    SetPipelineAppliedAnnotations(&annotations, "greedy");
    annotations.Set("coloring_total_candidates",
                    Integer(total_coloring_candidates));
    annotations.Set("coloring_evaluated_candidates",
                    Integer(coloring_candidates.size()));

    Map<Buffer, Map<Integer, PrimExpr>> runtime_bank_writer_phases;
    for (const auto &[buffer_node, phases] : selected_coloring.writer_phases) {
      Map<Integer, PrimExpr> per_command;
      for (const auto &[command_id, phase] : phases) {
        per_command.Set(Integer(command_id), Integer(phase));
      }
      runtime_bank_writer_phases.Set(tvm::ffi::GetRef<Buffer>(buffer_node),
                                     per_command);
    }
    annotations.Set("runtime_bank_writer_phases", runtime_bank_writer_phases);

    Map<Buffer, Map<Integer, PrimExpr>> runtime_bank_reader_phases;
    for (const auto &[buffer_node, phases] : selected_coloring.reader_phases) {
      Map<Integer, PrimExpr> per_command;
      for (const auto &[command_id, phase] : phases) {
        per_command.Set(Integer(command_id), Integer(phase));
      }
      runtime_bank_reader_phases.Set(tvm::ffi::GetRef<Buffer>(buffer_node),
                                     per_command);
    }
    annotations.Set("runtime_bank_reader_phases", runtime_bank_reader_phases);

    Array<String> orders;
    for (const auto &instruction : prologue_schedule) {
      orders.push_back(instruction.name);
    }
    annotations.Set("prologue_orders", orders);

    orders = Array<String>();
    for (const auto &instruction : body_schedule) {
      orders.push_back(instruction.name);
    }
    annotations.Set("body_orders", orders);

    if (stage_assembly.epilogue_iterations != -1) {
      orders = Array<String>();
      for (const auto &instruction : epilogue_schedule) {
        orders.push_back(instruction.name);
      }
      annotations.Set("epilogue_orders", orders);
    } else {
      Map<Integer, Array<String>> dynamic_orders;
      for (const auto &[remainder, schedule] : dynamic_epilogue_schedules) {
        Array<String> remainder_orders;
        for (const PipelineInstruction &instruction : schedule) {
          remainder_orders.push_back(instruction.name);
        }
        dynamic_orders.Set(Integer(remainder), remainder_orders);
      }
      annotations.Set("dynamic_epilogue_orders", dynamic_orders);
    }

    Array<Buffer> used_buffers;
    for (const BufferNode *buffer : local_ddg.buffer_order) {
      used_buffers.push_back(tvm::ffi::GetRef<Buffer>(buffer));
    }
    annotations.Set("used_buffers", used_buffers);

    Array<Buffer> versioned_buffer_array;
    for (const BufferNode *buffer : local_ddg.buffer_order) {
      if (versioned_buffers.count(buffer) == 0) {
        continue;
      }
      versioned_buffer_array.push_back(tvm::ffi::GetRef<Buffer>(buffer));
    }
    annotations.Set("versioned_buffers", versioned_buffer_array);

    return For(op->loop_var, op->min, op->extent, op->kind, op->body,
               op->thread_binding, annotations);
  }
};

tvm::transform::Pass SunmmioPipelinePlanning(bool debug = false) {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = SunmmioPipelinePlanner::Substitute(f, debug);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.SunmmioPipelinePlanning", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.SunmmioPipelinePlanning",
                        SunmmioPipelinePlanning);
}
} // namespace tl
} // namespace tvm
