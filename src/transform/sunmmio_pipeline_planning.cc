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
#include "../target/sunmmio/sunmmio_async_op_info.h"
#include "../target/sunmmio_utils.h"
#include "sunmmio_pipeline_planning/pipeline_diagnostic.h"
#include "sunmmio_pipeline_planning/resource_types_for_ilp.h"
#include "sunmmio_pipeline_planning/stmt_read_write_collector.h"
#include "sunmmio_pipeline_planning/sunmmio_pipeline_utils.h"
#include "sunmmio_pipeline_planning/tile_residency_analysis.h"
#include "sunmmio_pipeline_planning/grouping_snapshot.h"
#include <tvm/ffi/extra/structural_hash.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
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
  CostEstimate cost_estimate;

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
  size_t semantic_args = broadcast->args.size();
  ICHECK(semantic_args > 0 &&
         ParseSunmmioOdmaUnitExpr(broadcast->args[semantic_args - 1]))
      << "tl.broadcast_ requires a resolved ODMA unit before pipeline planning";
  --semantic_args;
  ICHECK(semantic_args == static_cast<size_t>(kBroadcastArgCount) ||
         semantic_args == static_cast<size_t>(kBroadcastArgCount + 1))
      << "tl.broadcast_ expects its fixed arguments and optional src_core";
  return semantic_args == static_cast<size_t>(kBroadcastArgCount);
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
    const std::unordered_set<const BufferNode *> &versioned_buffers,
    const CostModelContext &cost_context) {
  const char *path = std::getenv("TL_SUNMMIO_PIPELINE_GRAPH_JSON");
  if (path == nullptr || path[0] == '\0') {
    return;
  }
  std::ofstream out(path);
  if (!out.is_open()) {
    LOG(WARNING) << "Cannot write pipeline graph JSON to " << path;
    return;
  }
  bool fallback_used = false;
  bool solver_scaled = false;
  int64_t solver_scale_factor = 1;
  for (const PipelineInstruction &instruction : instructions) {
    fallback_used = fallback_used || instruction.cost_estimate.used_fallback;
    solver_scaled = solver_scaled || instruction.cost_estimate.solver_scaled;
    solver_scale_factor = std::max(
        solver_scale_factor, instruction.cost_estimate.solver_scale_factor);
  }
  auto write_json_string = [&](const std::string &value) {
    out << '"';
    for (char c : value) {
      switch (c) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << c;
        break;
      }
    }
    out << '"';
  };
  auto write_cost_estimate = [&](const CostEstimate &estimate) {
    out << "{";
    out << "\"profile\": ";
    write_json_string(std::string(estimate.profile));
    out << ", \"model_version\": ";
    write_json_string(std::string(estimate.model_version));
    out << ", \"latency_source\": ";
    write_json_string(std::string(estimate.latency_source));
    out << ", \"fallback_reason\": ";
    write_json_string(std::string(estimate.fallback_reason));
    out << ", \"op_kind\": ";
    write_json_string(std::string(estimate.op_kind));
    out << ", \"transfer_path\": ";
    write_json_string(std::string(estimate.transfer_path));
    out << ", \"src_scope\": ";
    write_json_string(std::string(estimate.src_scope));
    out << ", \"dst_scope\": ";
    write_json_string(std::string(estimate.dst_scope));
    out << ", \"input_dtype\": ";
    write_json_string(std::string(estimate.input_dtype));
    out << ", \"output_dtype\": ";
    write_json_string(std::string(estimate.output_dtype));
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
    out << ", \"endpoint_timing_modeled\": "
        << estimate.endpoint_timing_modeled;
    out << ", \"resource_overlap_modeled\": "
        << estimate.resource_overlap_modeled;
    out << "}";
  };

  out << std::boolalpha;
  out << "{\n  \"mode\": \"greedy\",\n";
  out << "  \"cost_model\": {\"profile\": ";
  write_json_string(CostModel::ProfileName(cost_context.options.profile));
  out << ", \"preserve_absolute_cycles\": "
      << cost_context.options.preserve_absolute_cycles;
  out << ", \"solver_scale_factor\": " << solver_scale_factor;
  out << ", \"solver_scaled\": " << solver_scaled;
  out << ", \"latency_gcd\": 1";
  out << ", \"auto_faster_selected\": false";
  out << ", \"planner_fallback\": false";
  out << ", \"fallback_used\": " << fallback_used;
  out << "},\n  \"commands\": [\n";
  for (size_t i = 0; i < instructions.size(); ++i) {
    const PipelineInstruction &instruction = instructions[i];
    out << "    {\"id\": " << instruction.id
        << ", \"iteration_offset\": 0, \"hardware\": "
        << static_cast<int>(instruction.device_type)
        << ", \"resource\": " << instruction.execution_resource
        << ", \"static_model_cycles\": "
        << instruction.cost_estimate.latency_cycles
        << ", \"planned_duration_cycles\": "
        << instruction.cost_estimate.latency_cycles << ", \"cost_estimate\": ";
    write_cost_estimate(instruction.cost_estimate);
    out << ", \"reads\": [";
    for (size_t j = 0; j < instruction.reads.size(); ++j) {
      if (j != 0)
        out << ", ";
      write_json_string(instruction.reads[j]->buffer->name);
    }
    out << "], \"writes\": [";
    for (size_t j = 0; j < instruction.writes.size(); ++j) {
      if (j != 0)
        out << ", ";
      write_json_string(instruction.writes[j]->buffer->name);
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
    return IsSunmmioAsyncODMAOp(call);
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
      : stmt_rw_collector_(f), cost_context_(CostModel::BuildContext(f)),
        debug_(debug) {}

  StmtReadWriteCollector stmt_rw_collector_;
  CostModelContext cost_context_;
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
      instruction.cost_estimate = CostModel::Estimate(
          instruction.device_type, instruction.stmt, cost_context_);
      instruction.delay =
          static_cast<float>(instruction.cost_estimate.latency_cycles);
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
                              versioned_buffers, cost_context_);

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
    auto faster_config = pass_ctx->GetConfig<Integer>(kSunmmioFaster);
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

struct CrossOuterCollectiveDescriptor {
  int command_id{-1};
  int direction{-1};
  int64_t participant_mask{0};
  Buffer source;
  Buffer destination;
};

struct CrossOuterCollectiveAnalysis {
  bool valid{true};
  std::string reason;
  std::string detail;
  std::vector<CrossOuterCollectiveDescriptor> descriptors;
};

static bool ContainsBroadcast(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&](const ObjectRef &obj) {
    const auto *call = obj.as<CallNode>();
    if (call && call->op.same_as(Op::Get("tl.broadcast_"))) {
      found = true;
    }
  });
  return found;
}

static CrossOuterCollectiveAnalysis
AnalyzeCrossOuterCollectives(const SeqStmtNode *body) {
  CrossOuterCollectiveAnalysis result;
  for (size_t command_id = 0; command_id < body->seq.size(); ++command_id) {
    int command_collective_count = 0;
    PostOrderVisit(body->seq[command_id], [&](const ObjectRef &obj) {
      if (!result.valid) {
        return;
      }
      if (const auto *conditional = obj.as<IfThenElseNode>()) {
        if (ContainsBroadcast(tvm::ffi::GetRef<Stmt>(conditional))) {
          result.valid = false;
          result.reason = "conditional_collective";
          result.detail =
              "all-gather inside conditional control flow is unsupported";
        }
        return;
      }
      const auto *call = obj.as<CallNode>();
      if (!call || !call->op.same_as(Op::Get("tl.broadcast_"))) {
        return;
      }
      ++command_collective_count;
      if (command_collective_count > 1) {
        result.valid = false;
        result.reason = "multiple_collectives_per_command";
        result.detail = "each Core command may contain at most one all-gather";
        return;
      }
      size_t semantic_args = call->args.size();
      if (semantic_args == 0 ||
          !ParseSunmmioOdmaUnitExpr(call->args[semantic_args - 1])) {
        result.valid = false;
        result.reason = "missing_resolved_odma_unit";
        result.detail = "all-gather requires a resolved ODMA unit";
        return;
      }
      --semantic_args;
      if (semantic_args != static_cast<size_t>(kBroadcastArgCount)) {
        result.valid = false;
        result.reason = "unsupported_broadcast";
        result.detail =
            "only all-gather broadcast without explicit src_core is supported";
        return;
      }
      const auto *direction =
          call->args[kBroadcastArgDirection].as<IntImmNode>();
      const auto *participant_mask =
          call->args[kBroadcastArgMask].as<IntImmNode>();
      if (direction == nullptr ||
          (direction->value != 0 && direction->value != 1)) {
        result.valid = false;
        result.reason = "dynamic_collective_direction";
        result.detail =
            "all-gather direction must be static horizontal or vertical";
        return;
      }
      if (participant_mask == nullptr || participant_mask->value <= 0) {
        result.valid = false;
        result.reason = "dynamic_collective_mask";
        result.detail =
            "all-gather participant mask must be a positive static value";
        return;
      }
      CrossOuterCollectiveDescriptor descriptor;
      descriptor.command_id = static_cast<int>(command_id);
      descriptor.direction = static_cast<int>(direction->value);
      descriptor.participant_mask = participant_mask->value;
      descriptor.source =
          NormalizeToBufferRegion(call->args[kBroadcastArgSrc])->buffer;
      descriptor.destination =
          NormalizeToBufferRegion(call->args[kBroadcastArgDst])->buffer;
      result.descriptors.push_back(std::move(descriptor));
    });
    if (!result.valid) {
      return result;
    }
  }
  return result;
}

static bool ValidateCrossOuterCollectiveEpochOrder(
    const CrossOuterCollectiveAnalysis &analysis, int64_t ii,
    const Array<Integer> &command_start_times,
    const Array<Integer> &command_emit_priorities, std::string *detail) {
  if (analysis.descriptors.empty()) {
    return true;
  }
  ICHECK_EQ(command_start_times.size(), command_emit_priorities.size());
  auto event_key = [&](const CrossOuterCollectiveDescriptor &descriptor,
                       int64_t epoch_offset = 0) {
    ICHECK_GE(descriptor.command_id, 0);
    ICHECK_LT(descriptor.command_id,
              static_cast<int>(command_start_times.size()));
    return std::make_tuple(
        epoch_offset + command_start_times[descriptor.command_id]->value,
        command_emit_priorities[descriptor.command_id]->value,
        descriptor.command_id);
  };
  for (size_t i = 1; i < analysis.descriptors.size(); ++i) {
    if (!(event_key(analysis.descriptors[i - 1]) <
          event_key(analysis.descriptors[i]))) {
      *detail =
          "all-gather order inside one logical bk iteration is not strict";
      return false;
    }
  }
  if (!(event_key(analysis.descriptors.back()) <
        event_key(analysis.descriptors.front(), ii))) {
    *detail = "last all-gather overlaps the next logical collective epoch";
    return false;
  }
  return true;
}

static std::string
FindCrossOuterForbiddenOperation(const Stmt &body,
                                 bool allow_safe_all_gather = false) {
  std::string forbidden;
  auto inspect_name = [&forbidden](std::string name) {
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    static const std::array<const char *, 7> tokens = {
        "barrier",    "broadcast", "all_gather", "allgather",
        "all_reduce", "allreduce", "collective"};
    for (const char *token : tokens) {
      if (name.find(token) != std::string::npos) {
        forbidden = std::move(name);
        return;
      }
    }
  };
  PostOrderVisit(body, [&](const ObjectRef &obj) {
    if (!forbidden.empty()) {
      return;
    }
    const auto *call = obj.as<CallNode>();
    if (!call) {
      return;
    }
    if (allow_safe_all_gather && call->op.same_as(Op::Get("tl.broadcast_"))) {
      return;
    }
    if (const auto *op = call->op.as<OpNode>()) {
      inspect_name(op->name);
    }
    for (const PrimExpr &arg : call->args) {
      if (const auto *text = arg.as<StringImmNode>()) {
        inspect_name(text->value);
      }
    }
  });
  return forbidden;
}

class SunmmioCrossOuterPipelinePlanner : public StmtExprMutator {
public:
  SunmmioCrossOuterPipelinePlanner(const PrimFunc &func, bool debug,
                                   bool forced)
      : func_(func), debug_(debug), forced_(forced) {}

  static Stmt Substitute(const PrimFunc &func, bool debug, bool forced,
                         int *applied_count) {
    SunmmioCrossOuterPipelinePlanner planner(func, debug, forced);
    Stmt result = planner(func->body);
    *applied_count = planner.applied_count_;
    return result;
  }

private:

  Stmt Reject(const ForNode *outer, const std::string &reason,
              const std::string &detail) {
    if (forced_) {
      LOG(FATAL) << "[SunmmioPipeline][cross_outer][ForcedReject] reason="
                 << reason << " detail=" << detail;
    }
    For fallback = tvm::ffi::GetRef<For>(outer);
    Map<String, Any> annotations = fallback->annotations;
    SetCrossOuterFallbackAnnotations(&annotations, "planning", reason, detail);
    fallback.CopyOnWrite()->annotations = annotations;
    return fallback;
  }

  Stmt VisitStmt_(const ForNode *outer) final {
    const auto *inner = outer->body.as<ForNode>();
    if (inner == nullptr ||
        inner->annotations.find("num_stages") == inner->annotations.end()) {
      return StmtExprMutator::VisitStmt_(outer);
    }

    if (outer->kind != ForKind::kSerial || inner->kind != ForKind::kSerial) {
      return Reject(outer, "non_serial_loop",
                    "both outer and inner loops must be serial");
    }
    if (inner->body.as<SeqStmtNode>() == nullptr) {
      return Reject(outer, "non_direct_sequence_body",
                    "the inner loop body must be a direct SeqStmt");
    }

    arith::Analyzer analyzer;
    PrimExpr simplified_outer_min = analyzer.Simplify(outer->min);
    PrimExpr simplified_outer_extent = analyzer.Simplify(outer->extent);
    PrimExpr simplified_inner_min = analyzer.Simplify(inner->min);
    PrimExpr simplified_inner_extent = analyzer.Simplify(inner->extent);
    const auto *outer_min = simplified_outer_min.as<IntImmNode>();
    const auto *outer_extent = simplified_outer_extent.as<IntImmNode>();
    const auto *inner_min = simplified_inner_min.as<IntImmNode>();
    const auto *inner_extent = simplified_inner_extent.as<IntImmNode>();
    if (outer_min == nullptr || outer_extent == nullptr ||
        inner_min == nullptr || inner_extent == nullptr) {
      return Reject(
          outer, "dynamic_range_unsupported",
          "both loop minima and extents must be compile-time integers");
    }
    if (outer_extent->value <= 1 || inner_extent->value <= 0) {
      return Reject(
          outer, "invalid_extent",
          "outer extent must exceed one and inner extent must be positive");
    }

    auto stages_anno = inner->annotations.Get("num_stages");
    const auto *stages_imm =
        stages_anno ? stages_anno.value().as<IntImmNode>() : nullptr;
    if (stages_imm == nullptr || stages_imm->value <= 0) {
      return Reject(outer, "invalid_stage_count",
                    "num_stages must be a positive compile-time integer");
    }
    int64_t stages = stages_imm->value;
    if (inner_extent->value < stages || inner_extent->value % stages != 0) {
      return Reject(
          outer, "inner_extent_not_stage_aligned",
          "inner extent must be at least num_stages and divisible by it");
    }

    std::string forbidden = FindCrossOuterForbiddenOperation(inner->body);
    if (!forbidden.empty()) {
      return Reject(outer, "forbidden_operation", forbidden);
    }

    if (outer_extent->value >
        std::numeric_limits<int64_t>::max() / inner_extent->value) {
      return Reject(outer, "extent_overflow",
                    "outer_extent * inner_extent exceeds int64");
    }
    int64_t total_extent = outer_extent->value * inner_extent->value;

    Var logical(std::string(outer->loop_var->name_hint) +
                    "_cross_outer_logical",
                outer->loop_var.dtype());
    Map<Var, PrimExpr> substitutions;
    substitutions.Set(outer->loop_var,
                      outer->min +
                          floordiv(logical, Integer(inner_extent->value)));
    substitutions.Set(inner->loop_var,
                      inner->min +
                          floormod(logical, Integer(inner_extent->value)));
    Stmt synthetic_body = tir::Substitute(inner->body, substitutions);
    For synthetic_loop(logical, Integer(0), Integer(total_extent),
                       ForKind::kSerial, synthetic_body, inner->thread_binding,
                       inner->annotations);
    PrimFunc synthetic_func = func_;
    synthetic_func.CopyOnWrite()->body = synthetic_loop;
    Stmt planned_stmt =
        SunmmioPipelinePlanner::Substitute(synthetic_func, debug_);
    const auto *planned = planned_stmt.as<ForNode>();
    ICHECK(planned != nullptr)
        << "cross_outer synthetic planner must return a For";
    if (!planned->annotations.count("prologue_orders") ||
        !planned->annotations.count("body_orders")) {
      std::string reason = "synthetic_planning_failed";
      std::string detail;
      if (auto ann = planned->annotations.Get(kPipelineFallbackReason)) {
        reason = std::string(Downcast<String>(ann.value()));
      }
      if (auto ann = planned->annotations.Get(kPipelineFallbackDetail)) {
        detail = std::string(Downcast<String>(ann.value()));
      }
      return Reject(outer, reason, detail);
    }

    Map<String, Any> annotations = outer->annotations;
    for (const auto &[key, value] : planned->annotations) {
      if (key != "num_stages") {
        annotations.Set(key, value);
      }
    }
    SetPipelineAppliedAnnotations(&annotations, "cross_outer");
    annotations.Set(kCrossOuterMarker, Bool(true));
    annotations.Set(kCrossOuterRequested, Bool(true));
    annotations.Set(kCrossOuterOuterExtent, Integer(outer_extent->value));
    annotations.Set(kCrossOuterInnerExtent, Integer(inner_extent->value));
    annotations.erase(kCrossOuterFallbackStage);
    annotations.erase(kCrossOuterFallbackReason);
    annotations.erase(kCrossOuterFallbackDetail);
    ++applied_count_;

    return For(outer->loop_var, outer->min, outer->extent, outer->kind,
               outer->body, outer->thread_binding, annotations);
  }

  PrimFunc func_;
  bool debug_{false};
  bool forced_{false};
  int applied_count_{0};
};

class CrossOuterUnmatchedLoopAnnotator : public StmtMutator {
private:
  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtMutator::VisitStmt_(op));
    if (!op->annotations.count("num_stages")) {
      return loop;
    }
    Map<String, Any> annotations = loop->annotations;
    SetCrossOuterFallbackAnnotations(
        &annotations, "planning", "no_direct_outer_inner_nest",
        "the pipelined loop is not the direct body of a serial outer loop");
    loop.CopyOnWrite()->annotations = annotations;
    return loop;
  }
};

struct RegionCrossOuterCandidate {
  Array<Stmt> prefix;
  For core;
  Array<Stmt> suffix;
  int core_index{-1};
};

struct RegionCrossOuterAccessSummary {
  std::vector<BufferRegion> reads;
  std::vector<BufferRegion> writes;
};

struct RegionCrossOuterEventAnalysis {
  bool valid{false};
  bool boundary_overlap{false};
  std::string detail;
};

static bool IsSupportedRegionStatement(const Stmt &stmt) {
  return stmt.as<ForNode>() || stmt.as<EvaluateNode>() ||
         stmt.as<BufferStoreNode>() || stmt.as<BlockRealizeNode>() ||
         stmt.as<AttrStmtNode>() || stmt.as<LetStmtNode>() ||
         stmt.as<IfThenElseNode>();
}

static RegionCrossOuterAccessSummary
CollectRegionAccesses(const PrimFunc &func, const Array<Stmt> &statements) {
  RegionCrossOuterAccessSummary result;
  StmtReadWriteCollector collector(func);
  for (const Stmt &stmt : statements) {
    collector.traverse_stmt(stmt);
    result.reads.insert(result.reads.end(),
                        collector.read_buffer_regions_.begin(),
                        collector.read_buffer_regions_.end());
    result.writes.insert(result.writes.end(),
                         collector.write_buffer_regions_.begin(),
                         collector.write_buffer_regions_.end());
  }
  return result;
}

static bool RegionsDisjointForAdjacentOuter(const BufferRegion &source,
                                            const BufferRegion &target,
                                            const Var &outer_var,
                                            const PrimExpr &outer_min) {
  if (!source->buffer.same_as(target->buffer) ||
      source->region.size() != target->region.size()) {
    return false;
  }
  Map<Var, PrimExpr> source_substitution;
  source_substitution.Set(outer_var, outer_min);
  Map<Var, PrimExpr> target_substitution;
  target_substitution.Set(outer_var, outer_min + Integer(1));
  arith::Analyzer analyzer;
  for (size_t axis = 0; axis < source->region.size(); ++axis) {
    const Range &source_range = source->region[axis];
    const Range &target_range = target->region[axis];
    PrimExpr source_begin = analyzer.Simplify(
        tir::Substitute(source_range->min, source_substitution));
    PrimExpr source_end =
        analyzer.Simplify(source_begin + tir::Substitute(source_range->extent,
                                                         source_substitution));
    PrimExpr target_begin = analyzer.Simplify(
        tir::Substitute(target_range->min, target_substitution));
    PrimExpr target_end =
        analyzer.Simplify(target_begin + tir::Substitute(target_range->extent,
                                                         target_substitution));
    if (analyzer.CanProve(
            Or(LE(source_end, target_begin), LE(target_end, source_begin)))) {
      return true;
    }
  }
  return false;
}

static RegionCrossOuterEventAnalysis AnalyzeRegionCrossOuterEvents(
    int64_t outer_extent, int64_t inner_extent, int iterations,
    int command_count, const Array<String> &prologue_orders,
    const Array<String> &body_orders, const Array<String> &epilogue_orders) {
  RegionCrossOuterEventAnalysis result;
  if (outer_extent <= 1 || inner_extent <= 0 || iterations <= 0 ||
      command_count <= 0) {
    result.detail = "invalid event dimensions";
    return result;
  }
  int64_t total_extent = outer_extent * inner_extent;
  if (total_extent % iterations != 0) {
    result.detail = "total extent is not stage aligned";
    return result;
  }
  constexpr uint64_t kMaxRegionEvents = 16384;
  uint64_t expected = static_cast<uint64_t>(total_extent) *
                      static_cast<uint64_t>(command_count);
  if (expected > kMaxRegionEvents) {
    result.detail = "event_stream_too_large";
    return result;
  }

  std::vector<int64_t> first(outer_extent, -1);
  std::vector<int64_t> last(outer_extent, -1);
  std::unordered_set<uint64_t> covered;
  int64_t position = 0;
  auto record = [&](const String &order, int64_t base) {
    int iter = name2iter(order);
    int id = name2id(order);
    int64_t logical = base + iter;
    if (id < 0 || id >= command_count || logical < 0 ||
        logical >= total_extent) {
      std::ostringstream os;
      os << "out-of-range event " << order << " at logical " << logical;
      result.detail = os.str();
      return false;
    }
    uint64_t key =
        static_cast<uint64_t>(logical) * static_cast<uint64_t>(command_count) +
        static_cast<uint64_t>(id);
    if (!covered.insert(key).second) {
      result.detail = "duplicate core event";
      return false;
    }
    int64_t outer = logical / inner_extent;
    if (first[outer] < 0) {
      first[outer] = position;
    }
    last[outer] = position++;
    return true;
  };

  for (const String &order : prologue_orders) {
    if (!record(order, 0)) {
      return result;
    }
  }
  int64_t total_chunks = total_extent / iterations;
  for (int64_t chunk = 0; chunk + 1 < total_chunks; ++chunk) {
    for (const String &order : body_orders) {
      if (!record(order, chunk * iterations)) {
        return result;
      }
    }
  }
  for (const String &order : epilogue_orders) {
    if (!record(order, (total_chunks - 1) * iterations)) {
      return result;
    }
  }
  if (covered.size() != expected) {
    std::ostringstream os;
    os << "core coverage=" << covered.size() << " expected=" << expected;
    result.detail = os.str();
    return result;
  }
  for (int64_t outer = 0; outer < outer_extent; ++outer) {
    if (first[outer] < 0 || last[outer] < first[outer]) {
      result.detail = "outer iteration has no complete core interval";
      return result;
    }
    if (outer + 1 < outer_extent && first[outer + 1] < last[outer]) {
      result.boundary_overlap = true;
    }
  }
  result.valid = true;
  return result;
}

static RegionCrossOuterEventAnalysis
AnalyzeHierarchicalILPEvents(int64_t outer_extent, int64_t inner_extent,
                             int command_count, int64_t ii,
                             const Array<Integer> &command_start_times,
                             const Array<Integer> &command_emit_priorities) {
  RegionCrossOuterEventAnalysis result;
  if (outer_extent <= 1 || inner_extent <= 0 || command_count <= 0 || ii <= 0 ||
      command_start_times.size() != static_cast<size_t>(command_count) ||
      command_emit_priorities.size() != static_cast<size_t>(command_count)) {
    result.detail = "invalid hierarchical event dimensions";
    return result;
  }
  if (inner_extent > std::numeric_limits<int64_t>::max() / ii ||
      outer_extent >
          std::numeric_limits<int64_t>::max() / (inner_extent * ii)) {
    result.detail = "hierarchical event time overflow";
    return result;
  }
  constexpr uint64_t kMaxRegionEvents = 16384;
  uint64_t expected = static_cast<uint64_t>(outer_extent) *
                      static_cast<uint64_t>(inner_extent) *
                      static_cast<uint64_t>(command_count);
  if (expected > kMaxRegionEvents) {
    result.detail = "event_stream_too_large";
    return result;
  }

  struct EventKey {
    int64_t time;
    int priority;
    int id;
    int64_t logical;
    int64_t outer;
  };
  std::vector<EventKey> events;
  events.reserve(expected);
  int64_t outer_stride = inner_extent * ii;
  for (int64_t outer = 0; outer < outer_extent; ++outer) {
    for (int64_t inner = 0; inner < inner_extent; ++inner) {
      int64_t logical = outer * inner_extent + inner;
      for (int id = 0; id < command_count; ++id) {
        int64_t start = command_start_times[id]->value;
        int priority = command_emit_priorities[id]->value;
        events.push_back(EventKey{outer * outer_stride + inner * ii + start,
                                  priority, id, logical, outer});
      }
    }
  }
  std::sort(events.begin(), events.end(),
            [](const EventKey &lhs, const EventKey &rhs) {
              return std::tie(lhs.time, lhs.priority, lhs.id, lhs.logical) <
                     std::tie(rhs.time, rhs.priority, rhs.id, rhs.logical);
            });

  std::vector<int64_t> first(outer_extent, -1);
  std::vector<int64_t> last(outer_extent, -1);
  for (size_t position = 0; position < events.size(); ++position) {
    int64_t outer = events[position].outer;
    if (first[outer] < 0) {
      first[outer] = static_cast<int64_t>(position);
    }
    last[outer] = static_cast<int64_t>(position);
  }
  for (int64_t outer = 0; outer < outer_extent; ++outer) {
    if (first[outer] < 0 || last[outer] < first[outer]) {
      result.detail = "outer iteration has no complete core interval";
      return result;
    }
    if (outer + 1 < outer_extent && first[outer + 1] < last[outer]) {
      result.boundary_overlap = true;
    }
  }
  result.valid = true;
  return result;
}

class SunmmioRegionCrossOuterPipelinePlanner : public StmtExprMutator {
public:
  SunmmioRegionCrossOuterPipelinePlanner(const PrimFunc &func, bool debug,
                                         bool forced, std::string strategy)
      : func_(func), debug_(debug), forced_(forced),
        strategy_(strategy == "multidim_boundary" ? "hierarchical_ilp" : strategy),
        only_proved_boundary_(strategy == "multidim_boundary") {
    PostOrderVisit(func_->body, [&](const ObjectRef &obj) {
      if (const auto *block = obj.as<BlockNode>()) {
        for (const Buffer &buffer : block->alloc_buffers) {
          local_buffers_.insert(buffer.get());
        }
      }
    });
  }

  static Stmt Substitute(const PrimFunc &func, bool debug, bool forced,
                         std::string strategy, int *applied_count) {
    SunmmioRegionCrossOuterPipelinePlanner planner(func, debug, forced,
                                                   std::move(strategy));
    Stmt result = planner(func->body);
    *applied_count = planner.applied_count_;
    return result;
  }

private:
  Stmt Reject(const ForNode *outer, const std::string &reason,
              const std::string &detail) {
    if (forced_) {
      LOG(FATAL)
          << "[SunmmioPipeline][region_cross_outer][ForcedReject] reason="
          << reason << " detail=" << detail;
    }
    For fallback = tvm::ffi::GetRef<For>(outer);
    Map<String, Any> annotations = fallback->annotations;
    SetRegionCrossOuterFallbackAnnotations(&annotations, "planning", reason,
                                           detail);
    if (only_proved_boundary_) {
      // A legal region is only a candidate. If it has no useful/supported
      // schedule, preserve the solved inner 1D pipeline and report that fact.
      annotations.Set("tl.sunmmio.multidim.selected_mode", String("ilp"));
      annotations.Set("tl.sunmmio.multidim.fallback_stage", String("boundary_overlap"));
      annotations.Set("tl.sunmmio.multidim.fallback_reason", String(reason));
    }
    fallback.CopyOnWrite()->annotations = annotations;
    return fallback;
  }

  static bool ExtractCandidate(const SeqStmtNode *seq, bool accept_planned_ilp,
                               RegionCrossOuterCandidate *candidate,
                               int *core_count) {
    *core_count = 0;
    for (size_t index = 0; index < seq->seq.size(); ++index) {
      const auto *loop = seq->seq[index].as<ForNode>();
      if (loop == nullptr) {
        continue;
      }
      bool is_core =
          loop->annotations.count("num_stages") ||
          (accept_planned_ilp &&
           loop->annotations.count("tl.sunmmio.pipeline.command_start_times"));
      if (!is_core) {
        continue;
      }
      ++*core_count;
      candidate->core = tvm::ffi::GetRef<For>(loop);
      candidate->core_index = static_cast<int>(index);
    }
    if (*core_count != 1) {
      return false;
    }
    for (int index = 0; index < candidate->core_index; ++index) {
      candidate->prefix.push_back(seq->seq[index]);
    }
    for (size_t index = candidate->core_index + 1; index < seq->seq.size();
         ++index) {
      candidate->suffix.push_back(seq->seq[index]);
    }
    return true;
  }

  bool AnalyzeHazards(const ForNode *outer,
                      const RegionCrossOuterCandidate &candidate,
                      Array<Buffer> *region_versioned_buffers,
                      std::string *detail) {
    Array<Stmt> core_statements;
    core_statements.push_back(candidate.core->body);
    RegionCrossOuterAccessSummary prefix =
        CollectRegionAccesses(func_, candidate.prefix);
    RegionCrossOuterAccessSummary core =
        CollectRegionAccesses(func_, core_statements);
    RegionCrossOuterAccessSummary suffix =
        CollectRegionAccesses(func_, candidate.suffix);

    // A top-level conditional suffix may write a global buffer only on every
    // Nth logical WorkItem (for example, one full-width spill after the last
    // N tile). If the guard cannot be true in two adjacent outer iterations,
    // the apparent Suffix(o)->Suffix(o+1) WAW is unreachable. Keep this
    // deliberately narrow: exactly one suffix statement must write the
    // buffer, the statement must be an if-without-else, and Z3 must prove the
    // two adjacent guards mutually exclusive.
    std::unordered_map<const BufferNode *, int> suffix_write_stmt_count;
    std::unordered_map<const BufferNode *, bool>
        suffix_write_stmt_is_adjacent_exclusive;
    std::unordered_set<const BufferNode *> suffix_written_buffers;
    std::unordered_set<const BufferNode *> conditional_suffix_read_buffers;
    std::unordered_set<const BufferNode *> nonconditional_suffix_read_buffers;
    for (const Stmt &stmt : candidate.suffix) {
      bool conditional_no_else = false;
      bool adjacent_exclusive = false;
      if (const auto *conditional = stmt.as<IfThenElseNode>()) {
        conditional_no_else = !conditional->else_case.defined();
        if (conditional_no_else) {
          Map<Var, PrimExpr> next_iteration;
          next_iteration.Set(outer->loop_var, outer->loop_var + Integer(1));
          PrimExpr next_condition =
              tir::Substitute(conditional->condition, next_iteration);
          PrimExpr adjacent_conflict =
              tir::And(conditional->condition, next_condition);
          arith::Analyzer guard_analyzer;
          adjacent_exclusive = guard_analyzer.z3_prover.CanProve(
              tir::Not(adjacent_conflict));
        }
      }
      Array<Stmt> singleton{stmt};
      RegionCrossOuterAccessSummary stmt_accesses =
          CollectRegionAccesses(func_, singleton);
      for (const BufferRegion &read : stmt_accesses.reads) {
        if (conditional_no_else) {
          conditional_suffix_read_buffers.insert(read->buffer.get());
        } else {
          nonconditional_suffix_read_buffers.insert(read->buffer.get());
        }
      }
      std::unordered_set<const BufferNode *> unique_writes;
      for (const BufferRegion &write : stmt_accesses.writes) {
        unique_writes.insert(write->buffer.get());
        suffix_written_buffers.insert(write->buffer.get());
      }
      for (const BufferNode *buffer : unique_writes) {
        int &count = suffix_write_stmt_count[buffer];
        ++count;
        if (count == 1) {
          suffix_write_stmt_is_adjacent_exclusive[buffer] =
              adjacent_exclusive;
        } else {
          suffix_write_stmt_is_adjacent_exclusive[buffer] = false;
        }
      }
    }
    std::unordered_set<const BufferNode *> adjacent_exclusive_suffix_writes;
    for (const auto &[buffer, count] : suffix_write_stmt_count) {
      if (count == 1 && suffix_write_stmt_is_adjacent_exclusive[buffer]) {
        adjacent_exclusive_suffix_writes.insert(buffer);
      }
    }

    // A local suffix-only buffer may intentionally accumulate disjoint pieces
    // across WorkItems and be consumed by the guarded final spill. Versioning
    // such a buffer separates those pieces and is semantically wrong. Keep it
    // single-versioned; suffixes are emitted in logical order and InjectSync
    // still protects asynchronous producer/consumer reuse.
    //找到可能最后用来汇总的buffer的可能候选
    std::unordered_set<const BufferNode *> suffix_carried_buffers;
    for (const BufferNode *buffer : conditional_suffix_read_buffers) {
      if (suffix_written_buffers.count(buffer) &&
          !nonconditional_suffix_read_buffers.count(buffer)) {
        suffix_carried_buffers.insert(buffer); //表示这个buffer不能被拆成双buffer
      }
    }

    std::unordered_set<const BufferNode *> versioned;
    auto check = [&](const std::vector<BufferRegion> &source,
                     const std::vector<BufferRegion> &target,
                     const char *hazard,
                     bool allow_adjacent_exclusive_suffix_waw = false,
                     bool preserve_suffix_carried = false) {
      for (const BufferRegion &lhs : source) { //对source中的每个buffer
        for (const BufferRegion &rhs : target) { //target中的每个buffer
          if (!lhs->buffer.same_as(rhs->buffer) || //访问的不是同一个buffer
              RegionsDisjointForAdjacentOuter(lhs, rhs, outer->loop_var,
                                              outer->min)) { //是同一个buffer，但是访问区域不重叠
            continue;
          }
          const BufferNode *buffer = lhs->buffer.get();//拿到发生冲突的buffer，进行到这里就意味着发生了冲突
          if (allow_adjacent_exclusive_suffix_waw && //如果允许相邻条件互斥的Suffix WAW
              adjacent_exclusive_suffix_writes.count(buffer)) { // 这个buffer已经被证明相邻写不会同时发生
            continue; //放行
          }
          if (local_buffers_.count(buffer)) {// 判读当前buffer是不是local buffer（primfun内分配的buffer）
            if (preserve_suffix_carried &&
                suffix_carried_buffers.count(buffer)) { //跨WorkItem累加的buffer，保持单buffer
              continue;
            }
            if (versioned.insert(buffer).second) { //是否为第一次插入多版本容器
              region_versioned_buffers->push_back(lhs->buffer); //插入多版本容器
            }
            continue;
          }

          //执行到这里，意味着无法证明相邻WorkItem的buffer不重叠，拒绝跨外层优化
          std::ostringstream os;
          os << hazard << " on global buffer " << lhs->buffer->name
             << " is not provably disjoint";
          *detail = os.str();
          return false;
        }
      }
      return true;
    };
    auto check_all = [&](const RegionCrossOuterAccessSummary &source,
                         const RegionCrossOuterAccessSummary &target,
                         const char *edge,
                         bool allow_adjacent_exclusive_suffix_waw = false,
                         bool preserve_suffix_carried = false) {
      std::string raw = std::string(edge) + ":RAW";
      std::string war = std::string(edge) + ":WAR";
      std::string waw = std::string(edge) + ":WAW";
      return check(source.writes, target.reads, raw.c_str(), false,
                   preserve_suffix_carried) &&
             check(source.reads, target.writes, war.c_str(), false,
                   preserve_suffix_carried) &&
             check(source.writes, target.writes, waw.c_str(),
                   allow_adjacent_exclusive_suffix_waw,
                   preserve_suffix_carried);
    };
    return check_all(core, prefix, "Core(o)->Prefix(o+1)") &&
           check_all(suffix, prefix, "Suffix(o)->Prefix(o+1)") &&
           check_all(suffix, core, "Suffix(o)->Core(o+1)") &&
           check_all(suffix, suffix, "Suffix(o)->Suffix(o+1)", true,
                     true);
  }

  Stmt VisitStmt_(const ForNode *outer) final {
    if (only_proved_boundary_) {
      auto eligible = outer->annotations.Get("tl.sunmmio.boundary_overlap.eligible");
      auto version = outer->annotations.Get("tl.sunmmio.boundary_overlap.proof_version");
      const auto *eligible_value = eligible ? eligible.value().as<IntImmNode>() : nullptr;
      const auto *version_value = version ? version.value().as<IntImmNode>() : nullptr;
      if (!eligible_value || !eligible_value->value || !version_value ||
          version_value->value != 1) {
        return StmtExprMutator::VisitStmt_(outer);
      }
    }
    const auto *direct_inner = outer->body.as<ForNode>();
    bool hierarchical_ilp = strategy_ == "hierarchical_ilp";
    if (direct_inner && (direct_inner->annotations.count("num_stages") ||
                         (hierarchical_ilp &&
                          direct_inner->annotations.count(
                              "tl.sunmmio.pipeline.command_start_times")))) {
      return Reject(outer, "outer_body_not_seq",
                    "region mode requires a SeqStmt around the pipeline core");
    }

    const auto *seq = outer->body.as<SeqStmtNode>();
    if (seq == nullptr) {
      return StmtExprMutator::VisitStmt_(outer);
    }
    RegionCrossOuterCandidate candidate;
    int core_count = 0;
    if (!ExtractCandidate(seq, hierarchical_ilp, &candidate, &core_count)) {
      if (core_count > 1) {
        return Reject(outer, "no_unique_pipeline_core",
                      "outer region contains multiple direct pipelined loops");
      }
      return StmtExprMutator::VisitStmt_(outer);
    }
    const ForNode *inner = candidate.core.get();

    if (outer->kind != ForKind::kSerial || inner->kind != ForKind::kSerial) {
      return Reject(outer, "non_serial_loop",
                    "both outer and core loops must be serial");
    }
    if (inner->body.as<SeqStmtNode>() == nullptr) {
      return Reject(outer, "inner_body_not_seq",
                    "the pipeline core body must be a direct SeqStmt");
    }
    for (const Stmt &stmt : candidate.prefix) {
      if (!IsSupportedRegionStatement(stmt)) {
        return Reject(outer, "unsupported_region_statement",
                      "unsupported direct Prefix statement");
      }
    }
    for (const Stmt &stmt : candidate.suffix) {
      if (!IsSupportedRegionStatement(stmt)) {
        return Reject(outer, "unsupported_region_statement",
                      "unsupported direct Suffix statement");
      }
    }

    arith::Analyzer analyzer;
    PrimExpr simplified_outer_min = analyzer.Simplify(outer->min);
    PrimExpr simplified_outer_extent = analyzer.Simplify(outer->extent);
    PrimExpr simplified_inner_min = analyzer.Simplify(inner->min);
    PrimExpr simplified_inner_extent = analyzer.Simplify(inner->extent);
    const auto *outer_min = simplified_outer_min.as<IntImmNode>();
    const auto *outer_extent = simplified_outer_extent.as<IntImmNode>();
    const auto *inner_min = simplified_inner_min.as<IntImmNode>();
    const auto *inner_extent = simplified_inner_extent.as<IntImmNode>();
    if (outer_min == nullptr || outer_extent == nullptr) {
      return Reject(outer, "dynamic_outer_range",
                    "outer minimum and extent must be compile-time integers");
    }
    if (inner_min == nullptr || inner_extent == nullptr) {
      return Reject(outer, "dynamic_inner_range",
                    "core minimum and extent must be compile-time integers");
    }
    if (outer_extent->value <= 1 || inner_extent->value <= 0) {
      return Reject(
          outer, "invalid_extent",
          "outer extent must exceed one and core extent must be positive");
    }

    if (!hierarchical_ilp) {
      auto stages_anno = inner->annotations.Get("num_stages");
      const auto *stages_imm =
          stages_anno ? stages_anno.value().as<IntImmNode>() : nullptr;
      if (stages_imm == nullptr || stages_imm->value <= 0) {
        return Reject(outer, "invalid_stage_count",
                      "num_stages must be a positive compile-time integer");
      }
      int64_t stages = stages_imm->value;
      if (inner_extent->value < stages || inner_extent->value % stages != 0) {
        return Reject(
            outer, "inner_extent_not_stage_aligned",
            "core extent must be at least num_stages and divisible by it");
      }
    }
    CrossOuterCollectiveAnalysis collective_analysis;
    if (hierarchical_ilp) {
      collective_analysis =
          AnalyzeCrossOuterCollectives(inner->body.as<SeqStmtNode>());
      if (!collective_analysis.valid) {
        return Reject(outer, collective_analysis.reason,
                      collective_analysis.detail);
      }
    }
    std::string forbidden =
        FindCrossOuterForbiddenOperation(outer->body, hierarchical_ilp);
    if (!forbidden.empty()) {
      return Reject(outer, "forbidden_operation", forbidden);
    }

    if (strategy_ == "conservative") {
      PrimFunc inner_func = func_;
      inner_func.CopyOnWrite()->body = candidate.core;
      Stmt planned_stmt =
          SunmmioPipelinePlanner::Substitute(inner_func, debug_);
      const auto *planned = planned_stmt.as<ForNode>();
      if (planned == nullptr ||
          !planned->annotations.count("prologue_orders") ||
          !planned->annotations.count("body_orders")) {
        return Reject(outer, "core_planning_failed",
                      "greedy planning did not produce a core schedule");
      }
      Array<Stmt> body = seq->seq;
      body.Set(candidate.core_index, tvm::ffi::GetRef<For>(planned));
      Map<String, Any> annotations = outer->annotations;
      SetPipelineAppliedAnnotations(&annotations, "region_cross_outer");
      annotations.Set(kRegionCrossOuterMarker, Bool(true));
      annotations.Set(kRegionCrossOuterRequested, Bool(true));
      annotations.Set(kRegionCrossOuterStrategy, String(strategy_));
      annotations.Set(kRegionCrossOuterOuterExtent,
                      Integer(outer_extent->value));
      annotations.Set(kRegionCrossOuterInnerExtent,
                      Integer(inner_extent->value));
      annotations.Set(kRegionCrossOuterCoreIndex,
                      Integer(candidate.core_index));
      annotations.Set(kRegionCrossOuterPrefixCount,
                      Integer(candidate.prefix.size()));
      annotations.Set(kRegionCrossOuterSuffixCount,
                      Integer(candidate.suffix.size()));
      annotations.Set(kRegionCrossOuterBoundaryOverlap, Bool(false));
      annotations.erase(kRegionCrossOuterFallbackStage);
      annotations.erase(kRegionCrossOuterFallbackReason);
      annotations.erase(kRegionCrossOuterFallbackDetail);
      ++applied_count_;
      return For(outer->loop_var, outer->min, outer->extent, outer->kind,
                 SeqStmt::Flatten(body), outer->thread_binding, annotations);
    }

    Array<Buffer> region_versioned_buffers;
    std::string dependence_detail;
    if (!AnalyzeHazards(outer, candidate, &region_versioned_buffers,
                        &dependence_detail)) {
      return Reject(outer, "cross_iteration_dependency", dependence_detail);
    }

    if (hierarchical_ilp) {
      const Map<String, Any> &inner_annotations = inner->annotations;
      auto starts_anno =
          inner_annotations.Get("tl.sunmmio.pipeline.command_start_times");
      auto priorities_anno =
          inner_annotations.Get("tl.sunmmio.pipeline.command_emit_priorities");
      auto ii_anno = inner_annotations.Get("ii");
      auto versioned_anno = inner_annotations.Get("versioned_buffers");
      auto used_anno = inner_annotations.Get("used_buffers");
      if (!starts_anno || !priorities_anno || !ii_anno || !versioned_anno ||
          !used_anno) {
        return Reject(outer, "inner_ilp_schedule_missing",
                      "the Core must carry a complete solved 1D ILP schedule");
      }
      Array<Integer> command_start_times =
          Downcast<Array<Integer>>(starts_anno.value());
      Array<Integer> command_emit_priorities =
          Downcast<Array<Integer>>(priorities_anno.value());
      int64_t ii = Downcast<IntImm>(ii_anno.value())->value;
      std::string collective_order_detail;
      if (!ValidateCrossOuterCollectiveEpochOrder(
              collective_analysis, ii, command_start_times,
              command_emit_priorities, &collective_order_detail)) {
        return Reject(outer, "collective_epoch_order", collective_order_detail);
      }
      int command_count =
          static_cast<int>(inner->body.as<SeqStmtNode>()->seq.size());
      RegionCrossOuterEventAnalysis event_analysis =
          AnalyzeHierarchicalILPEvents(outer_extent->value, inner_extent->value,
                                       command_count, ii, command_start_times,
                                       command_emit_priorities);
      if (!event_analysis.valid) {
        std::string reason = event_analysis.detail == "event_stream_too_large"
                                 ? "event_stream_too_large"
                                 : "schedule_coverage_failed";
        return Reject(outer, reason, event_analysis.detail);
      }
      if (!event_analysis.boundary_overlap) {
        return Reject(outer, "no_boundary_overlap",
                      "the solved 1D Core has no drain/startup overlap");
      }

      Array<Buffer> core_versioned_buffers =
          Downcast<Array<Buffer>>(versioned_anno.value());
      std::unordered_set<const BufferNode *> core_versioned;
      for (const Buffer &buffer : core_versioned_buffers) {
        core_versioned.insert(buffer.get());
      }
      for (const Buffer &buffer : region_versioned_buffers) {
        if (core_versioned.count(buffer.get())) {
          return Reject(outer, "nested_versioning_unsupported",
                        std::string(buffer->name));
        }
      }
      Array<Buffer> merged_versioned_buffers = core_versioned_buffers;
      for (const Buffer &buffer : region_versioned_buffers) {
        merged_versioned_buffers.push_back(buffer);
      }
      Array<Buffer> runtime_buffers;
      if (auto runtime_anno =
              inner_annotations.Get("runtime_multiversion_buffers")) {
        runtime_buffers = Downcast<Array<Buffer>>(runtime_anno.value());
      } else {
        runtime_buffers = core_versioned_buffers;
      }
      Array<Buffer> merged_used_buffers =
          Downcast<Array<Buffer>>(used_anno.value());
      auto append_unique = [](Array<Buffer> *buffers, const Buffer &buffer) {
        if (std::find(buffers->begin(), buffers->end(), buffer) ==
            buffers->end()) {
          buffers->push_back(buffer);
        }
      };
      for (const Buffer &buffer : region_versioned_buffers) {
        append_unique(&runtime_buffers, buffer);
        append_unique(&merged_used_buffers, buffer);
      }

      Map<String, Any> annotations = outer->annotations;
      for (const auto &[key, value] : inner_annotations) {
        if (key != "num_stages" && key != "versioned_buffers" &&
            key != "runtime_multiversion_buffers" && key != "used_buffers") {
          annotations.Set(key, value);
        }
      }
      annotations.Set("versioned_buffers", merged_versioned_buffers);
      annotations.Set("runtime_multiversion_buffers", runtime_buffers);
      annotations.Set("used_buffers", merged_used_buffers);
      annotations.Set(kRegionCrossOuterCoreVersionedBuffers,
                      core_versioned_buffers);
      annotations.Set(kRegionCrossOuterRegionVersionedBuffers,
                      region_versioned_buffers);
      SetPipelineAppliedAnnotations(&annotations, "hierarchical_ilp");
      annotations.Set(kRegionCrossOuterMarker, Bool(true));
      annotations.Set(kRegionCrossOuterRequested, Bool(true));
      annotations.Set(kRegionCrossOuterStrategy, String(strategy_));
      annotations.Set(kRegionCrossOuterOuterExtent,
                      Integer(outer_extent->value));
      annotations.Set(kRegionCrossOuterInnerExtent,
                      Integer(inner_extent->value));
      annotations.Set(kRegionCrossOuterCoreIndex,
                      Integer(candidate.core_index));
      annotations.Set(kRegionCrossOuterPrefixCount,
                      Integer(candidate.prefix.size()));
      annotations.Set(kRegionCrossOuterSuffixCount,
                      Integer(candidate.suffix.size()));
      annotations.Set(kRegionCrossOuterBoundaryOverlap, Bool(true));
      if (!collective_analysis.descriptors.empty()) {
        Array<Integer> command_ids;
        Array<Integer> directions;
        Array<Integer> participant_masks;
        Array<String> source_buffers;
        Array<String> destination_buffers;
        for (const CrossOuterCollectiveDescriptor &descriptor :
             collective_analysis.descriptors) {
          command_ids.push_back(Integer(descriptor.command_id));
          directions.push_back(Integer(descriptor.direction));
          participant_masks.push_back(Integer(descriptor.participant_mask));
          source_buffers.push_back(descriptor.source->name);
          destination_buffers.push_back(descriptor.destination->name);
        }
        annotations.Set("tl.sunmmio.region_cross_outer.collective_count",
                        Integer(collective_analysis.descriptors.size()));
        annotations.Set("tl.sunmmio.region_cross_outer.collective_command_ids",
                        command_ids);
        annotations.Set("tl.sunmmio.region_cross_outer.collective_directions",
                        directions);
        annotations.Set("tl.sunmmio.region_cross_outer.collective_masks",
                        participant_masks);
        annotations.Set("tl.sunmmio.region_cross_outer.collective_sources",
                        source_buffers);
        annotations.Set("tl.sunmmio.region_cross_outer.collective_destinations",
                        destination_buffers);
        annotations.Set(
            "tl.sunmmio.region_cross_outer.collective_epoch_order_verified",
            Bool(true));
      }
      ++applied_count_;
      return For(outer->loop_var, outer->min, outer->extent, outer->kind,
                 outer->body, outer->thread_binding, annotations);
    }

    if (outer_extent->value >
        std::numeric_limits<int64_t>::max() / inner_extent->value) {
      return Reject(outer, "extent_overflow",
                    "outer_extent * inner_extent exceeds int64");
    }
    int64_t total_extent = outer_extent->value * inner_extent->value;
    Var logical(std::string(outer->loop_var->name_hint) +
                    "_region_cross_outer_logical",
                outer->loop_var.dtype());
    Map<Var, PrimExpr> substitutions;
    substitutions.Set(outer->loop_var,
                      outer->min +
                          floordiv(logical, Integer(inner_extent->value)));
    substitutions.Set(inner->loop_var,
                      inner->min +
                          floormod(logical, Integer(inner_extent->value)));
    Stmt synthetic_body = tir::Substitute(inner->body, substitutions);
    For synthetic_loop(logical, Integer(0), Integer(total_extent),
                       ForKind::kSerial, synthetic_body, inner->thread_binding,
                       inner->annotations);
    PrimFunc synthetic_func = func_;
    synthetic_func.CopyOnWrite()->body = synthetic_loop;
    Stmt planned_stmt =
        SunmmioPipelinePlanner::Substitute(synthetic_func, debug_);
    const auto *planned = planned_stmt.as<ForNode>();
    if (planned == nullptr || !planned->annotations.count("prologue_orders") ||
        !planned->annotations.count("body_orders") ||
        !planned->annotations.count("epilogue_orders")) {
      std::string reason = "core_planning_failed";
      std::string detail;
      if (planned != nullptr) {
        if (auto ann = planned->annotations.Get(kPipelineFallbackReason)) {
          reason = std::string(Downcast<String>(ann.value()));
        }
        if (auto ann = planned->annotations.Get(kPipelineFallbackDetail)) {
          detail = std::string(Downcast<String>(ann.value()));
        }
      }
      return Reject(outer, reason, detail);
    }

    int iterations =
        Downcast<IntImm>(planned->annotations.at("iterations"))->value;
    Array<String> prologue_orders =
        Downcast<Array<String>>(planned->annotations.at("prologue_orders"));
    Array<String> body_orders =
        Downcast<Array<String>>(planned->annotations.at("body_orders"));
    Array<String> epilogue_orders =
        Downcast<Array<String>>(planned->annotations.at("epilogue_orders"));
    RegionCrossOuterEventAnalysis event_analysis =
        AnalyzeRegionCrossOuterEvents(
            outer_extent->value, inner_extent->value, iterations,
            static_cast<int>(inner->body.as<SeqStmtNode>()->seq.size()),
            prologue_orders, body_orders, epilogue_orders);
    if (!event_analysis.valid) {
      std::string reason = event_analysis.detail == "event_stream_too_large"
                               ? "event_stream_too_large"
                               : "schedule_coverage_failed";
      return Reject(outer, reason, event_analysis.detail);
    }
    if (!event_analysis.boundary_overlap) {
      return Reject(outer, "no_boundary_overlap",
                    "the planned Core schedule does not interleave adjacent "
                    "outer iterations");
    }

    Array<Buffer> core_versioned_buffers =
        Downcast<Array<Buffer>>(planned->annotations.at("versioned_buffers"));
    std::unordered_set<const BufferNode *> core_versioned;
    for (const Buffer &buffer : core_versioned_buffers) {
      core_versioned.insert(buffer.get());
    }
    for (const Buffer &buffer : region_versioned_buffers) {
      if (core_versioned.count(buffer.get())) {
        return Reject(outer, "nested_versioning_unsupported",
                      std::string(buffer->name));
      }
    }
    Array<Buffer> merged_versioned_buffers = core_versioned_buffers;
    for (const Buffer &buffer : region_versioned_buffers) {
      merged_versioned_buffers.push_back(buffer);
    }

    Map<String, Any> annotations = outer->annotations;
    for (const auto &[key, value] : planned->annotations) {
      if (key != "num_stages" && key != "versioned_buffers") {
        annotations.Set(key, value);
      }
    }
    annotations.Set("versioned_buffers", merged_versioned_buffers);
    annotations.Set(kRegionCrossOuterCoreVersionedBuffers,
                    core_versioned_buffers);
    annotations.Set(kRegionCrossOuterRegionVersionedBuffers,
                    region_versioned_buffers);
    SetPipelineAppliedAnnotations(&annotations, "region_cross_outer");
    annotations.Set(kRegionCrossOuterMarker, Bool(true));
    annotations.Set(kRegionCrossOuterRequested, Bool(true));
    annotations.Set(kRegionCrossOuterStrategy, String(strategy_));
    annotations.Set(kRegionCrossOuterOuterExtent, Integer(outer_extent->value));
    annotations.Set(kRegionCrossOuterInnerExtent, Integer(inner_extent->value));
    annotations.Set(kRegionCrossOuterCoreIndex, Integer(candidate.core_index));
    annotations.Set(kRegionCrossOuterPrefixCount,
                    Integer(candidate.prefix.size()));
    annotations.Set(kRegionCrossOuterSuffixCount,
                    Integer(candidate.suffix.size()));
    annotations.Set(kRegionCrossOuterBoundaryOverlap, Bool(true));
    annotations.erase(kRegionCrossOuterFallbackStage);
    annotations.erase(kRegionCrossOuterFallbackReason);
    annotations.erase(kRegionCrossOuterFallbackDetail);
    ++applied_count_;
    return For(outer->loop_var, outer->min, outer->extent, outer->kind,
               outer->body, outer->thread_binding, annotations);
  }

  PrimFunc func_;
  bool debug_{false};
  bool forced_{false};
  std::string strategy_;
  bool only_proved_boundary_{false};
  int applied_count_{0};
  std::unordered_set<const BufferNode *> local_buffers_;
};


struct ResidencyEvent {
  residency::Proof proof;
  std::string selected{"analysis_only"};
  std::string fingerprint;
};

static std::string ResidencyJsonEscape(const std::string &value) {
  std::ostringstream result;
  for (char ch : value) {
    if (ch == '"' || ch == '\\') result << '\\' << ch;
    else if (ch == '\n') result << "\\n";
    else if (ch == '\r') result << "\\r";
    else if (ch == '\t') result << "\\t";
    else result << ch;
  }
  return result.str();
}

static void WriteResidencyEvents(const std::string &policy,
                                const std::vector<ResidencyEvent> &events) {
  const char *path = std::getenv("TL_SUNMMIO_RESIDENCY_REPORT");
  if (!path || !*path) return;
  std::ofstream out(path);
  ICHECK(out.is_open()) << "cannot write residency report: " << path;
  out << "{\n  \"schema_version\": 2,\n  \"policy\": \""
      << policy << "\",\n  \"fingerprint_scope\": \"pre_sync_tir_structure\","
      << "\n  \"cost_gate\": \"transfer_savings_prefilter_backend_schedule_required\","
      << "\n  \"candidates\": [";
  for (size_t i = 0; i < events.size(); ++i) {
    if (i) out << ",";
    const auto &event = events[i];
    const auto &proof = event.proof;
    out << "\n    {\"accepted\": " << (proof.accepted ? "true" : "false")
        << ", \"reason\": \"" << ResidencyJsonEscape(proof.reason)
        << "\", \"reuse_loop\": \"" << ResidencyJsonEscape(proof.reuse_loop)
        << "\", \"reuse_extent\": " << proof.reuse_extent
        << ", \"group_width\": " << proof.group_width
        << ", \"shared_prefix_count\": " << proof.shared_prefix_count
        << ", \"consumer_count\": " << proof.consumer_count
        << ", \"added_bytes\": " << proof.added_bytes
        << ", \"saved_transfer_bytes\": " << proof.saved_transfer_bytes
        << ", \"resource_check\": \"" << proof.resource_check
        << "\", \"selected_strategy\": \"" << event.selected
        << "\", \"schedule_fingerprint\": \"" << event.fingerprint << "\"}";
  }
  out << "\n  ]\n}\n";
}

class SunmmioResidencyAnalysisAnnotator : public StmtMutator {
public:
  SunmmioResidencyAnalysisAnnotator(const PrimFunc &func, std::string policy)
      : func_(func), policy_(std::move(policy)) {}

  std::vector<ResidencyEvent> events;
  int grouped_core_count{0};
private:
  Stmt VisitStmt_(const ForNode *op) final {
    if (op->annotations.count("num_stages"))
      return ffi::GetRef<For>(op);
    bool candidate = false;
    for (const Stmt &stmt : residency::Statements(op->body)) {
      const auto *loop = stmt.as<ForNode>();
      candidate = candidate || (loop && loop->annotations.count("num_stages"));
    }
    if (!candidate) return StmtMutator::VisitStmt_(op);
    For loop = ffi::GetRef<For>(op);
    bool slotted = op->annotations.count("tl.sunmmio.multidim.workitem_inner");
    residency::Proof proof = residency::Analyze(func_, loop, slotted, true);
    if (proof.accepted && policy_ == "auto_strict" &&
        !residency::HasMinimumReuseAmortization(proof)) {
      proof.accepted = false;
      proof.reason = "insufficient_reuse_amortization";
    }
    if (proof.accepted && proof.group_width &&
        !residency::CanSnapshotGroupingFunction(func_)) {
      proof.accepted = false;
      proof.reason = "grouping_snapshot_layout_unsupported";
    }
    events.push_back({proof, proof.accepted ? "proved_candidate" : "original",
                      std::to_string(ffi::StructuralHash::Hash(loop, true))});
    if (!proof.accepted) {
      ICHECK_NE(policy_, "forced")
          << "generic tile residency rejected: " << proof.reason;
      if (policy_ != "shadow") {
        Map<String, Any> annotations = loop->annotations;
        annotations.Set(residency::kReason, String(proof.reason));
        annotations.erase(residency::kEligible);
        loop.CopyOnWrite()->annotations = annotations;
      }
      return loop;
    }
    if (policy_ == "shadow") return loop;
    Map<String, Any> annotations = loop->annotations;
    annotations.erase(residency::kReason);
    annotations.Set(residency::kEligible, Bool(true));
    annotations.Set(residency::kStateBuffers, proof.state_buffers);
    annotations.Set(residency::kPrefixCount, Integer(proof.shared_prefix_count));
    if (proof.group_width) {
      annotations.Set(residency::kGroupWidth, Integer(proof.group_width));
      grouped_core_count += 1 + (proof.reuse_extent % proof.group_width != 0);
    }
    loop.CopyOnWrite()->annotations = annotations;
    return loop;
  }
  PrimFunc func_;
  std::string policy_;
};

class ResidencyDiagnosticStripper : public StmtMutator {
private:
  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtMutator::VisitStmt_(op));
    Map<String, Any> annotations;
    for (const auto &[key, value] : loop->annotations) {
      if (std::string(key).rfind("tl.sunmmio.residency.", 0) != 0)
        annotations.Set(key, value);
    }
    loop.CopyOnWrite()->annotations = annotations;
    return loop;
  }
};

static String ResidencyScheduleFingerprint(PrimFunc f) {
  f.CopyOnWrite()->body = ResidencyDiagnosticStripper()(f->body);
  return String(std::to_string(ffi::StructuralHash::Hash(f, true)));
}

static tvm::transform::Pass
AnalyzeSunmmioTileResidency(String policy = "auto_strict") {
  std::string value = policy;
  ICHECK(value == "off" || value == "shadow" || value == "auto_strict" ||
         value == "forced") << "invalid tile residency policy: " << value;
  auto pass_func = [=](PrimFunc f, const IRModule &, tvm::transform::PassContext) {
    if (value == "off") {
      WriteResidencyEvents(value, {});
      return f;
    }
    SunmmioResidencyAnalysisAnnotator analysis(f, value);
    Stmt body = analysis(f->body);
    if (value == "forced")
      ICHECK(!analysis.events.empty()) << "generic tile residency: no candidate";
    WriteResidencyEvents(value, analysis.events);
    if (value != "shadow") {
      PrimFunc original = f;
      f.CopyOnWrite()->body = body;
      if (analysis.grouped_core_count) {
        f = WithAttr(std::move(f), residency::kGroupingOriginal,
                     residency::SnapshotGroupingFunction(original));
        f = WithAttr(std::move(f), residency::kGroupingCoreCount,
                     Integer(analysis.grouped_core_count));
      }
    }
    return f;
  };
  return tir::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.AnalyzeSunmmioTileResidency", {});
}

class SunmmioStaticWorkItemFlattener : public StmtMutator {
public:
  explicit SunmmioStaticWorkItemFlattener(const PrimFunc &func, bool forced)
      : func_(func), forced_(forced), residency_policy_(residency::Policy()) {}

  static Stmt Substitute(const PrimFunc &func, bool forced,
                         int *requested_count, int *applied_count,
                         int *a_residency_count, int *auto_discovered_count) {
    SunmmioStaticWorkItemFlattener flattener(func, forced);
    Stmt result = flattener(func->body);
    *requested_count = flattener.requested_count_;
    *applied_count = flattener.applied_count_;
    *a_residency_count = flattener.a_residency_count_;
    *auto_discovered_count = flattener.auto_discovered_count_;
    WriteResidencyEvents(flattener.residency_policy_, flattener.events_);
    return result;
  }

private:
  static constexpr const char *kOuterMarker =
      "tl.sunmmio.multidim.workitem_outer";
  static constexpr const char *kInnerMarker =
      "tl.sunmmio.multidim.workitem_inner";
  static constexpr const char *kRequested =
      "tl.sunmmio.multidim.workitem.requested";
  static constexpr const char *kApplied =
      "tl.sunmmio.multidim.workitem.applied";
  static constexpr const char *kOuterExtent =
      "tl.sunmmio.multidim.workitem.outer_extent";
  static constexpr const char *kInnerExtent =
      "tl.sunmmio.multidim.workitem.inner_extent";
  static constexpr const char *kTotalExtent =
      "tl.sunmmio.multidim.workitem.total_extent";
  static constexpr const char *kFallbackReason =
      "tl.sunmmio.multidim.workitem.fallback_reason";
  static constexpr const char *kFallbackDetail =
      "tl.sunmmio.multidim.workitem.fallback_detail";
  static constexpr const char *kAResidencyApplied =
      "tl.sunmmio.multidim.a_residency.applied";
  static constexpr const char *kAutoDiscovered =
      "tl.sunmmio.multidim.workitem.auto_discovered";

  Stmt BuildProvedResidency(const ForNode *outer, const ForNode *inner,
                            int64_t outer_extent, int64_t inner_extent,
                            const residency::Proof &proof,
                            bool auto_discovered = false) {
    Array<Stmt> new_core_body;
    for (int i = 0; i < proof.shared_prefix_count; ++i)
      new_core_body.push_back(proof.core_statements[i]);
    auto instance = [&](const Stmt &stmt, int64_t coordinate) {
      Map<Var, PrimExpr> substitution;
      substitution.Set(inner->loop_var, inner->min + Integer(coordinate));
      return tir::Substitute(stmt, substitution);
    };
    for (int64_t i = 0; i < inner_extent; ++i)
      for (size_t j = proof.shared_prefix_count;
           j < proof.core_statements.size(); ++j)
        new_core_body.push_back(instance(proof.core_statements[j], i));
    const For &core = proof.core;
    For new_core(core->loop_var, core->min, core->extent, core->kind,
                 SeqStmt::Flatten(new_core_body), core->thread_binding,
                 core->annotations, core->step, core->span);
    Array<Stmt> new_outer_body;
    for (int64_t i = 0; i < inner_extent; ++i)
      for (const Stmt &stmt : proof.before)
        new_outer_body.push_back(instance(stmt, i));
    new_outer_body.push_back(new_core);

    // Preserve accumulator liveness through the whole K-major core.  This is
    // also part of the frozen C-style schedule's allocation behavior.
    Array<BufferRegion> accumulators;
    for (const Stmt &stmt : proof.core_statements) {
      residency::Operation operation;
      ICHECK(residency::ParseOperation(stmt, &operation));
      if (operation.transfer)
        continue;
      BufferRegion accumulator =
          NormalizeToBufferRegion(operation.call->args[2]);
      bool seen = false;
      for (const BufferRegion &old : accumulators)
        seen = seen || residency::SameRegion(old, accumulator);
      if (!seen)
        accumulators.push_back(accumulator);
    }
    for (int64_t i = 0; i < inner_extent; ++i) {
      for (const BufferRegion &accumulator : accumulators) {
        Array<PrimExpr> indices;
        for (const Range &range : accumulator->region)
          indices.push_back(range->min);
        new_outer_body.push_back(
            instance(Evaluate(BufferLoad(accumulator->buffer, indices)), i));
      }
    }
    for (int64_t i = 0; i < inner_extent; ++i)
      for (const Stmt &stmt : proof.after)
        new_outer_body.push_back(instance(stmt, i));
    Map<String, Any> annotations = outer->annotations;
    annotations.erase(kOuterMarker);
    annotations.Set(kRequested, Bool(true));
    annotations.Set(kApplied, Bool(true));
    annotations.Set(kOuterExtent, Integer(outer_extent));
    annotations.Set(kInnerExtent, Integer(inner_extent));
    annotations.Set(kTotalExtent, Integer(outer_extent));
    annotations.Set(kAResidencyApplied, Bool(true));
    if (auto_discovered)
      annotations.Set(kAutoDiscovered, Bool(true));
    else
      annotations.erase(kAutoDiscovered);
    annotations.erase(kFallbackReason);
    annotations.erase(kFallbackDetail);
    return For(outer->loop_var, outer->min, outer->extent, outer->kind,
               SeqStmt::Flatten(new_outer_body), outer->thread_binding,
               annotations, outer->step, outer->span);
  }

  std::optional<Stmt> TryAutoDiscover(const ForNode *outer) {
    // Explicit annotations remain authoritative. Forced mode preserves its
    // historical contract: it requires a user-marked candidate.
    if (forced_ || outer->annotations.count(kOuterMarker) ||
        outer->annotations.count(kInnerMarker) ||
        outer->annotations.count("num_stages"))
      return std::nullopt;

    const auto *inner = outer->body.as<ForNode>();
    if (inner == nullptr || inner->annotations.count(kOuterMarker) ||
        inner->annotations.count(kInnerMarker) ||
        inner->annotations.count("num_stages"))
      return std::nullopt;
    if (outer->kind != ForKind::kSerial || inner->kind != ForKind::kSerial ||
        outer->thread_binding.defined() || inner->thread_binding.defined() ||
        outer->loop_var.dtype() != inner->loop_var.dtype())
      return std::nullopt;

    arith::Analyzer analyzer;
    auto outer_min = residency::StaticInt(analyzer.Simplify(outer->min));
    auto outer_extent = residency::StaticInt(analyzer.Simplify(outer->extent));
    auto inner_min = residency::StaticInt(analyzer.Simplify(inner->min));
    auto inner_extent = residency::StaticInt(analyzer.Simplify(inner->extent));
    if (!outer_min || !outer_extent || !inner_min || !inner_extent ||
        *outer_extent <= 1 || *inner_extent <= 0)
      return std::nullopt;

    int pipeline_core_count = 0;
    PostOrderVisit(inner->body, [&](const ObjectRef &obj) {
      if (const auto *loop = obj.as<ForNode>())
        pipeline_core_count += loop->annotations.count("num_stages") ? 1 : 0;
    });
    if (pipeline_core_count != 1)
      return std::nullopt;

    residency::Proof proof =
        residency::Analyze(func_, ffi::GetRef<For>(inner), true);
    // Static WorkItems reuse already allocated slots.  Judge their benefit
    // across the full outer grid instead of applying the Ordinary-residency
    // expansion heuristic, which counts only consumers inside one reuse loop.
    bool amortized_static_reuse = proof.saved_transfer_bytes > 0 &&
                                  proof.shared_prefix_count > 0 &&
                                  proof.consumer_count > 0 &&
                                  proof.reuse_extent >= 2 && *outer_extent >= 2;
    if (proof.accepted && !amortized_static_reuse) {
      proof.accepted = false;
      proof.reason = "insufficient_static_workitem_amortization";
    }
    if (!proof.accepted) {
      events_.push_back({proof, "auto_workitem_rejected",
                         std::to_string(ffi::StructuralHash::Hash(
                             ffi::GetRef<For>(outer), true))});
      return std::nullopt;
    }

    ++requested_count_;
    Stmt generic = BuildProvedResidency(outer, inner, *outer_extent,
                                        *inner_extent, proof, true);
    ResidencyEvent event{proof};
    event.fingerprint =
        std::to_string(ffi::StructuralHash::Hash(generic, true));
    event.selected = "generic_residency_auto_workitem";
    events_.push_back(event);
    ++applied_count_;
    ++a_residency_count_;
    ++auto_discovered_count_;
    return generic;
  }

  Stmt Reject(const ForNode *outer, const std::string &reason,
              const std::string &detail) {
    if (forced_) {
      LOG(FATAL) << "[SunmmioPipeline][workitem_flatten][ForcedReject] reason="
                 << reason << " detail=" << detail;
    }
    For fallback = tvm::ffi::GetRef<For>(outer);
    Map<String, Any> annotations = fallback->annotations;
    annotations.Set(kRequested, Bool(true));
    annotations.Set(kApplied, Bool(false));
    annotations.Set(kFallbackReason, String(reason));
    annotations.Set(kFallbackDetail, String(detail));
    fallback.CopyOnWrite()->annotations = annotations;
    return fallback;
  }

  Stmt VisitStmt_(const ForNode *outer) final {
    if (!outer->annotations.count(kOuterMarker)) {
      if (auto discovered = TryAutoDiscover(outer))
        return discovered.value();
      return StmtMutator::VisitStmt_(outer);
    }
    ++requested_count_;

    const auto *inner = outer->body.as<ForNode>();
    if (inner == nullptr || !inner->annotations.count(kInnerMarker)) {
      return Reject(outer, "missing_direct_inner_loop",
                    "marked outer loop must directly contain the marked "
                    "inner WorkItem loop");
    }
    if (outer->kind != ForKind::kSerial || inner->kind != ForKind::kSerial) {
      return Reject(outer, "non_serial_workitem_loop",
                    "both WorkItem loops must be serial");
    }
    if (outer->thread_binding.defined() || inner->thread_binding.defined()) {
      return Reject(outer, "thread_bound_workitem_loop",
                    "WorkItem loops must not carry thread bindings");
    }
    if (outer->loop_var.dtype() != inner->loop_var.dtype()) {
      return Reject(outer, "loop_dtype_mismatch",
                    "outer and inner WorkItem coordinates must share a dtype");
    }

    arith::Analyzer analyzer;
    PrimExpr simplified_outer_min = analyzer.Simplify(outer->min);
    PrimExpr simplified_outer_extent = analyzer.Simplify(outer->extent);
    PrimExpr simplified_inner_min = analyzer.Simplify(inner->min);
    PrimExpr simplified_inner_extent = analyzer.Simplify(inner->extent);
    const auto *outer_min = simplified_outer_min.as<IntImmNode>();
    const auto *outer_extent = simplified_outer_extent.as<IntImmNode>();
    const auto *inner_min = simplified_inner_min.as<IntImmNode>();
    const auto *inner_extent = simplified_inner_extent.as<IntImmNode>();
    if (outer_min == nullptr || outer_extent == nullptr ||
        inner_min == nullptr || inner_extent == nullptr) {
      return Reject(outer, "dynamic_workitem_range",
                    "both WorkItem minima and extents must be compile-time "
                    "integers");
    }
    if (outer_extent->value <= 0 || inner_extent->value <= 0) {
      return Reject(outer, "non_positive_workitem_extent",
                    "both WorkItem extents must be positive");
    }
    if (outer_extent->value >
        std::numeric_limits<int64_t>::max() / inner_extent->value) {
      return Reject(outer, "workitem_extent_overflow",
                    "outer_extent * inner_extent exceeds int64");
    }

    int pipeline_core_count = 0;
    PostOrderVisit(inner->body, [&](const ObjectRef &obj) {
      if (const auto *loop = obj.as<ForNode>()) {
        pipeline_core_count += loop->annotations.count("num_stages") ? 1 : 0;
      }
    });
    if (pipeline_core_count != 1) {
      return Reject(outer, "non_unique_pipeline_core",
                    "the marked WorkItem nest must contain exactly one "
                    "num_stages loop");
    }

    if (residency_policy_ != "off") {
      residency::Proof proof = residency::Analyze(
          func_, ffi::GetRef<For>(inner), true);
      // A single outer WorkItem cannot amortize a cross-WorkItem residency
      // transform, even if an internal loop exposes reuse.
      if (outer_extent->value <= 1) {
        proof.accepted = false;
        proof.reason = "outer_extent_too_short";
      }
      ResidencyEvent event{proof};
      if (residency_policy_ == "shadow") {
        event.selected = "analysis_only";
        events_.push_back(event);
      } else if (proof.accepted) {
        Stmt generic = BuildProvedResidency(
            outer, inner, outer_extent->value, inner_extent->value, proof);
        event.fingerprint = std::to_string(ffi::StructuralHash::Hash(generic, true));
        For chosen = Downcast<For>(generic);
        Map<String, Any> annotations = chosen->annotations;
        annotations.Set(residency::kApplied, Bool(true));
        chosen.CopyOnWrite()->annotations = annotations;
        event.selected = "generic_residency";
        events_.push_back(event);
        ++applied_count_;
        ++a_residency_count_;
        return chosen;
      } else {
        event.selected = "original_order_flatten";
        events_.push_back(event);
        ICHECK(residency_policy_ != "forced" ||
               outer_extent->value <= 1)
            << "generic tile residency rejected: " << proof.reason;
        // A rejected proof falls through to the ordinary safe flattening path.
      }
    }

    int64_t total_extent = outer_extent->value * inner_extent->value;
    Var logical(std::string(outer->loop_var->name_hint) +
                    "_multidim_workitem",
                outer->loop_var.dtype());
    Map<Var, PrimExpr> substitutions;
    substitutions.Set(
        outer->loop_var,
        outer->min + floordiv(logical, Integer(inner_extent->value)));
    substitutions.Set(
        inner->loop_var,
        inner->min + floormod(logical, Integer(inner_extent->value)));
    Stmt flattened_body = tir::Substitute(inner->body, substitutions);

    Map<String, Any> annotations = outer->annotations;
    annotations.erase(kOuterMarker);
    annotations.Set(kRequested, Bool(true));
    annotations.Set(kApplied, Bool(true));
    annotations.Set(kOuterExtent, Integer(outer_extent->value));
    annotations.Set(kInnerExtent, Integer(inner_extent->value));
    annotations.Set(kTotalExtent, Integer(total_extent));
    annotations.erase(kFallbackReason);
    annotations.erase(kFallbackDetail);
    ++applied_count_;

    return For(logical, Integer(0), Integer(total_extent), ForKind::kSerial,
               flattened_body, outer->thread_binding, annotations);
  }

  PrimFunc func_;
  bool forced_{false};
  std::string residency_policy_;
  std::vector<ResidencyEvent> events_;
  int requested_count_{0};
  int applied_count_{0};
  int a_residency_count_{0};
  int auto_discovered_count_{0};
};

class RegionCrossOuterUnmatchedLoopAnnotator : public StmtMutator {
private:
  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtMutator::VisitStmt_(op));
    if (!op->annotations.count("num_stages")) {
      return loop;
    }
    Map<String, Any> annotations = loop->annotations;
    SetRegionCrossOuterFallbackAnnotations(
        &annotations, "planning", "no_unique_pipeline_core",
        "the pipelined loop is not the unique direct Core of an outer SeqStmt");
    loop.CopyOnWrite()->annotations = annotations;
    return loop;
  }
};

static tvm::transform::Pass
FlattenSunmmioStaticWorkItems(String policy = "auto") {
  using namespace tir::transform;
  std::string policy_value = policy;
  ICHECK(policy_value == "auto" || policy_value == "forced")
      << "static WorkItem flatten policy must be 'auto' or 'forced', got "
      << policy_value;
  bool forced = policy_value == "forced";
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    int requested_count = 0;
    int applied_count = 0;
    int a_residency_count = 0;
    int auto_discovered_count = 0;
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = SunmmioStaticWorkItemFlattener::Substitute(
        f, forced, &requested_count, &applied_count, &a_residency_count,
        &auto_discovered_count);
    if (forced && requested_count == 0) {
      LOG(FATAL) << "[SunmmioPipeline][workitem_flatten][ForcedReject] "
                    "reason=no_marked_workitem_nest";
    }
    ICHECK_LE(applied_count, requested_count);
    auto propagate_applied_key = [&](const String &applied_key) {
      f = WithAttr(std::move(f), applied_key, Bool(true));
      Array<String> device_attr_keys;
      if (auto existing =
              f->GetAttr<Array<String>>(attr::kDeviceFuncAttrKeys)) {
        device_attr_keys = existing.value();
      }
      if (std::find(device_attr_keys.begin(), device_attr_keys.end(),
                    applied_key) == device_attr_keys.end()) {
        device_attr_keys.push_back(applied_key);
      }
      f = WithAttr(std::move(f), attr::kDeviceFuncAttrKeys, device_attr_keys);
    };
    if (applied_count > a_residency_count) {
      propagate_applied_key("tl.sunmmio.multidim.workitem_flatten.applied");
    }
    if (a_residency_count > 0) {
      propagate_applied_key("tl.sunmmio.multidim.a_residency.applied");
    }
    if (auto_discovered_count > 0) {
      f = WithAttr(std::move(f), "tl.sunmmio.multidim.workitem.auto_discovered",
                   Integer(auto_discovered_count));
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.FlattenSunmmioStaticWorkItems",
                            {});
}

tvm::transform::Pass SunmmioPipelinePlanning(bool debug = false) {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = SunmmioPipelinePlanner::Substitute(f, debug);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.SunmmioPipelinePlanning", {});
}

static tvm::transform::Pass
SunmmioCrossOuterPipelinePlanning(bool debug = false, String policy = "auto") {
  using namespace tir::transform;
  std::string policy_value = policy;
  ICHECK(policy_value == "auto" || policy_value == "forced")
      << "cross_outer policy must be 'auto' or 'forced', got " << policy_value;
  bool forced = policy_value == "forced";
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    bool has_requested_loop = false;
    PostOrderVisit(f->body, [&](const ObjectRef &obj) {
      if (const auto *loop = obj.as<ForNode>()) {
        has_requested_loop =
            has_requested_loop || loop->annotations.count("num_stages");
      }
    });
    int applied_count = 0;
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = SunmmioCrossOuterPipelinePlanner::Substitute(f, debug, forced,
                                                              &applied_count);
    bool has_cross_outer_fallback = false;
    PostOrderVisit(fptr->body, [&](const ObjectRef &obj) {
      if (const auto *loop = obj.as<ForNode>()) {
        has_cross_outer_fallback =
            has_cross_outer_fallback ||
            loop->annotations.count(kCrossOuterFallbackReason);
      }
    });
    if (!forced && has_requested_loop && applied_count == 0 &&
        !has_cross_outer_fallback) {
      CrossOuterUnmatchedLoopAnnotator annotator;
      fptr->body = annotator(fptr->body);
    }
    if (forced && has_requested_loop && applied_count == 0) {
      LOG(FATAL) << "[SunmmioPipeline][cross_outer][ForcedReject] "
                    "reason=no_supported_outer_inner_nest";
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0,
                            "tl.SunmmioCrossOuterPipelinePlanning", {});
}

static tvm::transform::Pass
SunmmioRegionCrossOuterPipelinePlanning(bool debug = false,
                                        String policy = "auto",
                                        String strategy = "continuous") {
  using namespace tir::transform;
  std::string policy_value = policy;
  std::string strategy_value = strategy;
  ICHECK(policy_value == "auto" || policy_value == "forced")
      << "region_cross_outer policy must be 'auto' or 'forced', got "
      << policy_value;
  ICHECK(strategy_value == "conservative" || strategy_value == "continuous" ||
         strategy_value == "hierarchical_ilp" || strategy_value == "multidim_boundary")
      << "region_cross_outer strategy must be 'conservative', 'continuous', "
         "'hierarchical_ilp', or 'multidim_boundary', got "
      << strategy_value;
  bool forced = policy_value == "forced";
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    bool has_requested_loop = false;
    PostOrderVisit(f->body, [&](const ObjectRef &obj) {
      if (const auto *loop = obj.as<ForNode>()) {
        if (strategy_value == "multidim_boundary") {
          auto eligible = loop->annotations.Get("tl.sunmmio.boundary_overlap.eligible");
          const auto *value = eligible ? eligible.value().as<IntImmNode>() : nullptr;
          has_requested_loop = has_requested_loop || (value && value->value);
          return;
        }
        has_requested_loop = has_requested_loop ||
                             loop->annotations.count("num_stages") ||
                             (strategy_value == "hierarchical_ilp" &&
                              loop->annotations.count(
                                  "tl.sunmmio.pipeline.command_start_times"));
      }
    });
    int applied_count = 0;
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = SunmmioRegionCrossOuterPipelinePlanner::Substitute(
        f, debug, forced, strategy_value, &applied_count);
    if (strategy_value == "multidim_boundary" && applied_count > 0) {
      f = WithAttr(std::move(f), "tl.sunmmio.boundary_overlap.planned", Bool(true));
      fptr = f.CopyOnWrite();
    }
    bool has_region_fallback = false;
    PostOrderVisit(fptr->body, [&](const ObjectRef &obj) {
      if (const auto *loop = obj.as<ForNode>()) {
        has_region_fallback =
            has_region_fallback ||
            loop->annotations.count(kRegionCrossOuterFallbackReason);
      }
    });
    if (strategy_value != "multidim_boundary" && !forced && has_requested_loop && applied_count == 0 &&
        !has_region_fallback) {
      RegionCrossOuterUnmatchedLoopAnnotator annotator;
      fptr->body = annotator(fptr->body);
    }
    if (forced && has_requested_loop && applied_count == 0) {
      LOG(FATAL) << "[SunmmioPipeline][region_cross_outer][ForcedReject] "
                    "reason=no_supported_region";
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0,
                            "tl.SunmmioRegionCrossOuterPipelinePlanning", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnalyzeSunmmioTileResidency",
                        AnalyzeSunmmioTileResidency);
  refl::GlobalDef().def("tl.transform.SunmmioResidencyScheduleFingerprint",
                        ResidencyScheduleFingerprint);
  refl::GlobalDef().def("tl.transform.FlattenSunmmioStaticWorkItems",
                        FlattenSunmmioStaticWorkItems);
  refl::GlobalDef().def("tl.transform.SunmmioPipelinePlanning",
                        SunmmioPipelinePlanning);
  refl::GlobalDef().def("tl.transform.SunmmioCrossOuterPipelinePlanning",
                        SunmmioCrossOuterPipelinePlanning);
  refl::GlobalDef().def("tl.transform.SunmmioRegionCrossOuterPipelinePlanning",
                        SunmmioRegionCrossOuterPipelinePlanning);
}
} // namespace tl
} // namespace tvm
