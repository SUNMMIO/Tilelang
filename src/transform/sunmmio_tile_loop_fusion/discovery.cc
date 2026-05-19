/*!
 * \file discovery.cc
 * \brief Discovery and problem-building for Sunmmio tile loop fusion.
 */

#include "discovery.h"
#include "utils.h"

#include "../../op/builtin.h"
#include "../../op/copy.h"
#include "../../op/reduce.h"
#include "../../op/utils.h"
#include "../common/attr.h"

#include <tvm/arith/analyzer.h>
#include <tvm/arith/pattern.h>
#include <tvm/node/structural_equal.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

bool IsAnnotatedOne(const ForNode *loop, const char *key) {
  auto it = loop->annotations.find(key);
  if (it == loop->annotations.end()) {
    return false;
  }
  if (const auto *imm = (*it).second.as<IntImmNode>()) {
    return imm->value == 1;
  }
  return true;
}

bool IsTileScopeEntry(const ForNode *loop) {
  return IsAnnotatedOne(loop, attr::tile_scope_entry);
}

bool IsTileExecutionLoop(const ForNode *loop) {
  return loop->annotations.count(attr::tile_execution_axis) != 0;
}

int GetIntAnnotation(const ForNode *loop, const char *key) {
  auto it = loop->annotations.find(key);
  if (it == loop->annotations.end()) {
    return -1;
  }
  const auto *imm = (*it).second.as<IntImmNode>();
  ICHECK(imm) << "Expected integer annotation `" << key << "`, but got "
              << (*it).second;
  return static_cast<int>(imm->value);
}

bool IsPlannerPrivateBuffer(const Buffer &buffer) {
  std::string scope = buffer.scope();
  if (scope.empty() || scope == "global") {
    return false;
  }
  return scope.rfind("shared", 0) != 0;
}

struct PlannerVisibleRegionMatch {
  Stmt root_stmt;
  For scope_entry_for;
};

struct PlannerVisibleMatchResult {
  PlannerVisibleRegionMatch match;
  int end_index{-1};
};

class VisibleBufferCollector : public StmtExprVisitor {
public:
  Map<Var, Buffer> buffers;

private:
  void AddBuffer(const Buffer &buffer) { buffers.Set(buffer->data, buffer); }

  void VisitStmt_(const BlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers) {
      AddBuffer(buffer);
    }
    for (const MatchBufferRegion &match_buffer : op->match_buffers) {
      AddBuffer(match_buffer->buffer);
      AddBuffer(match_buffer->source->buffer);
    }
    for (const BufferRegion &region : op->reads) {
      AddBuffer(region->buffer);
    }
    for (const BufferRegion &region : op->writes) {
      AddBuffer(region->buffer);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    AddBuffer(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferRealizeNode *op) final {
    AddBuffer(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    AddBuffer(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }
};

class LocalBufferCollector : public StmtVisitor {
public:
  std::unordered_set<const VarNode *> local_buffers;

private:
  void AddBuffer(const Buffer &buffer) {
    local_buffers.insert(buffer->data.get());
  }

  void VisitStmt_(const BlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers) {
      AddBuffer(buffer);
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferRealizeNode *op) final {
    AddBuffer(op->buffer);
    StmtVisitor::VisitStmt_(op);
  }
};

class RegionLoopCollector : public StmtVisitor {
public:
  std::vector<For> execution_loops;

private:
  void VisitStmt_(const ForNode *op) final {
    if (IsTileExecutionLoop(op)) {
      execution_loops.push_back(ffi::GetRef<For>(op));
    }
    StmtVisitor::VisitStmt_(op);
  }
};

Map<Var, Buffer> CollectVisibleBuffers(const PrimFunc &func) {
  Map<Var, Buffer> buffers;
  for (const auto &kv : func->buffer_map) {
    buffers.Set(kv.first, kv.second);
  }

  VisibleBufferCollector collector;
  collector(func->body);
  for (const auto &kv : collector.buffers) {
    buffers.Set(kv.first, kv.second);
  }
  return buffers;
}

std::unordered_set<const VarNode *> CollectLocalBufferVars(const Stmt &stmt) {
  LocalBufferCollector collector;
  collector(stmt);
  return collector.local_buffers;
}

bool IsLocallyAllocatedBuffer(
    const Buffer &buffer,
    const std::unordered_set<const VarNode *> &local_buffer_vars) {
  return local_buffer_vars.count(buffer->data.get()) != 0;
}

bool SameBufferRegion(const BufferRegion &lhs, const BufferRegion &rhs) {
  return lhs->buffer.same_as(rhs->buffer) &&
         StructuralEqual()(lhs->region, rhs->region);
}

std::optional<BufferRegion> NormalizeRegionArgument(const PrimExpr &arg) {
  try {
    return NormalizeToBufferRegion(arg);
  } catch (const tvm::Error &) {
    return std::nullopt;
  }
}

class OpaqueBuiltinAccessCollector : public StmtExprVisitor {
public:
  void Collect(const Stmt &stmt) { VisitStmt(stmt); }

  Array<BufferRegion> reads;
  Array<BufferRegion> writes;
  Array<BufferRegion> write_only_writes;

private:
  void VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call != nullptr && call->op.same_as(vector_core_in_tile_reduce()) &&
        call->args.size() >= 3) {
      if (std::optional<BufferRegion> dst =
              NormalizeRegionArgument(call->args[1])) {
        writes.push_back(*dst);
        write_only_writes.push_back(*dst);
      }
      if (std::optional<BufferRegion> src =
              NormalizeRegionArgument(call->args[2])) {
        reads.push_back(*src);
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

std::optional<For> FindWrappedTileScopeEntryLoop(const Stmt &stmt) {
  // Discovery accepts thin AttrStmt/LetStmt/Block wrappers around a
  // `tile.scope_entry` loop so rewrite can later preserve that wrapper shape.
  if (const auto *loop = stmt.as<ForNode>()) {
    if (IsTileScopeEntry(loop)) {
      return ffi::GetRef<For>(loop);
    }
    return std::nullopt;
  }

  if (const auto *block_realize = stmt.as<BlockRealizeNode>()) {
    return FindWrappedTileScopeEntryLoop(block_realize->block);
  }

  if (const auto *block = stmt.as<BlockNode>()) {
    if (block->init.defined()) {
      return std::nullopt;
    }
    if (const auto *seq = block->body.as<SeqStmtNode>()) {
      if (seq->seq.size() != 1) {
        return std::nullopt;
      }
      return FindWrappedTileScopeEntryLoop(seq->seq[0]);
    }
    return FindWrappedTileScopeEntryLoop(block->body);
  }

  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    return FindWrappedTileScopeEntryLoop(attr->body);
  }

  if (const auto *let = stmt.as<LetStmtNode>()) {
    return FindWrappedTileScopeEntryLoop(let->body);
  }

  return std::nullopt;
}

std::optional<PlannerVisibleMatchResult>
MatchPlannerVisibleRegion(const Array<Stmt> &seq, int start_index) {
  if (std::optional<For> loop =
          FindWrappedTileScopeEntryLoop(seq[start_index])) {
    PlannerVisibleRegionMatch match;
    match.root_stmt = seq[start_index];
    match.scope_entry_for = *loop;
    return PlannerVisibleMatchResult{match, start_index};
  }
  return std::nullopt;
}

class PlannerVisibleProgramCollector : public StmtVisitor {
public:
  void Collect(const PrimFunc &func) { VisitStmt(func->body); }

  std::vector<PlannerVisibleRegionMatch> region_matches;
  std::vector<TileScopeRegionRun> region_runs;

private:
  void Flush(std::vector<int> *run) {
    if (!run->empty()) {
      ICHECK_EQ(run->back() - run->front() + 1, static_cast<int>(run->size()));
      region_runs.push_back({run->front(), static_cast<int>(run->size())});
      run->clear();
    }
  }

  void VisitStmt_(const SeqStmtNode *op) final {
    // A planner-visible run is a maximal consecutive source-order interval of
    // matched regions inside the same enclosing SeqStmt.
    std::vector<int> current_run;
    int index = 0;
    while (index < static_cast<int>(op->seq.size())) {
      std::optional<PlannerVisibleMatchResult> match =
          MatchPlannerVisibleRegion(op->seq, index);
      if (match) {
        int region_index = static_cast<int>(region_matches.size());
        region_matches.push_back(match->match);
        current_run.push_back(region_index);
        index = match->end_index + 1;
        continue;
      }
      Flush(&current_run);
      VisitStmt(op->seq[index]);
      ++index;
    }
    Flush(&current_run);
  }

  void VisitStmt_(const ForNode *op) final {
    if (IsTileScopeEntry(op)) {
      int region_index = static_cast<int>(region_matches.size());
      PlannerVisibleRegionMatch match;
      match.root_stmt = ffi::GetRef<For>(op);
      match.scope_entry_for = ffi::GetRef<For>(op);
      region_matches.push_back(match);
      region_runs.push_back({region_index, 1});
      return;
    }
    StmtVisitor::VisitStmt_(op);
  }
};

Array<BufferRegion> DedupeExternalRegions(
    const Array<BufferRegion> &regions,
    const std::unordered_set<const VarNode *> &local_buffer_vars = {}) {
  // Discovery only keeps externally visible regions. Local scratch buffers and
  // planner-private scopes never cross stage boundaries.
  Array<BufferRegion> result;
  for (const BufferRegion &region : regions) {
    if (IsPlannerPrivateBuffer(region->buffer) ||
        IsLocallyAllocatedBuffer(region->buffer, local_buffer_vars)) {
      continue;
    }

    bool duplicate = false;
    for (const BufferRegion &existing : result) {
      if (SameBufferRegion(existing, region)) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) {
      result.push_back(region);
    }
  }
  return result;
}

Array<BufferRegion>
RemoveMatchingRegions(const Array<BufferRegion> &regions,
                      const Array<BufferRegion> &to_remove) {
  Array<BufferRegion> result;
  for (const BufferRegion &region : regions) {
    bool remove = false;
    for (const BufferRegion &candidate : to_remove) {
      if (SameBufferRegion(region, candidate)) {
        remove = true;
        break;
      }
    }
    if (!remove) {
      result.push_back(region);
    }
  }
  return result;
}

Array<PrimExpr>
BuildExecutionLoopExtents(const std::vector<For> &execution_loops,
                          arith::Analyzer *analyzer) {
  Array<PrimExpr> execution_loop_extents;
  for (const For &loop : execution_loops) {
    execution_loop_extents.push_back(analyzer->Simplify(loop->extent));
  }
  return execution_loop_extents;
}

std::vector<int> GetExecutionDomainAxes(const For &scope_entry_for, int rank) {
  auto it = scope_entry_for->annotations.find(attr::tile_execution_domain_axes);
  if (it == scope_entry_for->annotations.end()) {
    std::vector<int> identity(rank);
    for (int axis = 0; axis < rank; ++axis) {
      identity[axis] = axis;
    }
    return identity;
  }

  Array<PrimExpr> array = Downcast<Array<PrimExpr>>((*it).second);

  std::vector<int> execution_domain_axes;
  execution_domain_axes.reserve(array.size());
  for (const PrimExpr &item : array) {
    const auto *imm = item.as<IntImmNode>();
    ICHECK(imm) << "Expected integer tile.execution_domain_axes entry, but got "
                << item;
    execution_domain_axes.push_back(static_cast<int>(imm->value));
  }
  return execution_domain_axes;
}

std::string MakeLogicalAxisKey(int axis) {
  static const char *kAxisNames[] = {"i", "j", "k", "l", "m", "n", "o", "p"};
  if (axis >= 0 &&
      axis < static_cast<int>(sizeof(kAxisNames) / sizeof(kAxisNames[0]))) {
    return kAxisNames[axis];
  }
  return "axis" + std::to_string(axis);
}

std::vector<std::string>
BuildLogicalExecutionAxisKeys(const For &scope_entry_for,
                              const std::vector<For> &execution_loops) {
  std::vector<int> execution_domain_axes = GetExecutionDomainAxes(
      scope_entry_for, static_cast<int>(execution_loops.size()));

  std::vector<std::string> logical_axis_keys;
  logical_axis_keys.reserve(execution_loops.size());
  for (const For &loop : execution_loops) {
    int execution_axis =
        GetIntAnnotation(loop.get(), attr::tile_execution_axis);
    int logical_axis = execution_axis;
    if (execution_axis >= 0 &&
        execution_axis < static_cast<int>(execution_domain_axes.size())) {
      logical_axis = execution_domain_axes[execution_axis];
    }
    logical_axis_keys.push_back(MakeLogicalAxisKey(logical_axis));
  }
  return logical_axis_keys;
}

int ComputeAvailableExecutionDepth(const BufferRegion &region,
                                   const std::vector<For> &execution_loops) {
  std::unordered_map<const VarNode *, int> execution_depth_by_var;
  for (size_t i = 0; i < execution_loops.size(); ++i) {
    execution_depth_by_var[execution_loops[i]->loop_var.get()] =
        static_cast<int>(i) + 1;
  }

  int avail_depth = 0;
  for (const Range &range : region->region) {
    VarUseCollector collector;
    collector(range->min);
    collector(range->extent);
    for (const VarNode *var : collector.seen_vars) {
      auto it = execution_depth_by_var.find(var);
      if (it != execution_depth_by_var.end()) {
        avail_depth = std::max(avail_depth, it->second);
      }
    }
  }
  return avail_depth;
}

std::vector<std::string>
ExtractExecutionLoopVarNames(const std::vector<For> &execution_loops) {
  std::vector<std::string> axis_names;
  axis_names.reserve(execution_loops.size());
  for (const For &loop : execution_loops) {
    axis_names.push_back(static_cast<std::string>(loop->loop_var->name_hint));
  }
  return axis_names;
}

struct ActiveUseInfo {
  int region_index{-1};
  int access_index{-1};
  const NormalizedBufferAccess *access{nullptr};
};

struct ActiveDefInfo {
  int region_index{-1};
  int access_index{-1};
  const NormalizedBufferAccess *access{nullptr};
};

struct OverlapFacts {
  int rho{0};
  int64_t weight_bytes{0};
};

std::optional<BufferRegion> IntersectBufferRegions(const BufferRegion &lhs,
                                                   const BufferRegion &rhs,
                                                   arith::Analyzer *analyzer) {
  // Only compare accesses to the same normalized buffer with the same rank.
  // Then intersect each dimension independently and reject the pair only when
  // some per-dimension overlap extent is provably empty.
  if (!lhs->buffer.same_as(rhs->buffer) ||
      lhs->region.size() != rhs->region.size()) {
    return std::nullopt;
  }

  Array<Range> intersection;
  for (size_t i = 0; i < lhs->region.size(); ++i) {
    const Range &lhs_range = lhs->region[i];
    const Range &rhs_range = rhs->region[i];

    PrimExpr start =
        analyzer->Simplify(tvm::max(lhs_range->min, rhs_range->min));
    PrimExpr end =
        analyzer->Simplify(tvm::min(lhs_range->min + lhs_range->extent,
                                    rhs_range->min + rhs_range->extent));
    PrimExpr extent = analyzer->Simplify(end - start);

    if (analyzer->CanProve(extent <= 0)) {
      return std::nullopt;
    }
    intersection.push_back(Range::FromMinExtent(start, extent));
  }

  return BufferRegion(lhs->buffer, intersection);
}

int ComputeRequiredSharedPrefixDepth(const NormalizedBufferAccess &src_access,
                                     const NormalizedBufferAccess &dst_access) {
  // `rho` is the minimum shared execution depth needed to keep this overlap
  // internal. If either access boundary depends on depth d, the value instance
  // can vary across iterations at d, so any shallower shared shell is too weak.
  // Prefix axis/extent compatibility is enforced later by the planner.
  ICHECK_EQ(src_access.dims.size(), dst_access.dims.size());
  int rho = 0;
  for (size_t i = 0; i < src_access.dims.size(); ++i) {
    for (int depth : src_access.dims[i].execution_axis_depths) {
      rho = std::max(rho, depth);
    }
    for (int depth : dst_access.dims[i].execution_axis_depths) {
      rho = std::max(rho, depth);
    }
  }
  return rho;
}

int64_t ComputeEdgeWeightBytes(const NormalizedBufferAccess &src_access,
                               const NormalizedBufferAccess &dst_access,
                               const BufferRegion &exact_overlap_region,
                               TileScopeDependenceKind kind,
                               arith::Analyzer *analyzer) {
  // Only RAW edges carry a direct cut cost today. Measure that cost in bytes
  // of overlap payload that would need to escape the fused shell if the edge is
  // cut. The factor of 2 models a spill/reload pair: once out of the producer
  // and once back into the consumer. When the symbolic overlap extent is not
  // constant, fall back to the minimum static per-dimension extents.
  if (kind != TileScopeDependenceKind::kRAW) {
    return 0;
  }

  ICHECK_EQ(src_access.dims.size(), dst_access.dims.size());
  ICHECK_EQ(exact_overlap_region->region.size(), src_access.dims.size());

  int64_t element_count = 1;
  for (size_t i = 0; i < exact_overlap_region->region.size(); ++i) {
    PrimExpr exact_extent =
        analyzer->Simplify(exact_overlap_region->region[i]->extent);
    if (const auto *imm = exact_extent.as<IntImmNode>()) {
      element_count *= imm->value;
      continue;
    }

    const auto *src_imm = src_access.dims[i].extent.as<IntImmNode>();
    const auto *dst_imm = dst_access.dims[i].extent.as<IntImmNode>();
    if (src_imm == nullptr || dst_imm == nullptr) {
      return 0;
    }
    element_count *= std::min(src_imm->value, dst_imm->value);
  }

  int64_t bytes = exact_overlap_region->buffer->dtype.bytes();
  return 2 * element_count * bytes;
}

std::optional<OverlapFacts>
ComputeOverlapFacts(const NormalizedBufferAccess &src_access,
                    const NormalizedBufferAccess &dst_access,
                    TileScopeDependenceKind kind, arith::Analyzer *analyzer) {
  std::optional<BufferRegion> exact_overlap =
      IntersectBufferRegions(src_access.region, dst_access.region, analyzer);
  if (!exact_overlap.has_value()) {
    return std::nullopt;
  }

  OverlapFacts facts;
  facts.rho = ComputeRequiredSharedPrefixDepth(src_access, dst_access);
  facts.weight_bytes = ComputeEdgeWeightBytes(
      src_access, dst_access, exact_overlap.value(), kind, analyzer);
  return facts;
}

TileScopeDependenceEdge
MakeDependenceEdge(int src_region_index, int dst_region_index,
                   TileScopeDependenceKind kind, int src_access_index,
                   int dst_access_index, const OverlapFacts &facts) {
  return {src_region_index,  dst_region_index, kind,
          src_access_index,  dst_access_index, facts.rho,
          facts.weight_bytes};
}

TileScopeRegion
AnalyzeOneTileScopeRegion(const PlannerVisibleRegionMatch &match,
                          const Map<Var, Buffer> &visible_buffers) {
  const For &scope_entry_for = match.scope_entry_for;
  RegionLoopCollector loop_collector;
  loop_collector(scope_entry_for);

  arith::Analyzer analyzer;

  Array<PrimExpr> execution_loop_extents =
      BuildExecutionLoopExtents(loop_collector.execution_loops, &analyzer);

  Stmt boundary_stmt = scope_entry_for;
  if (!loop_collector.execution_loops.empty()) {
    boundary_stmt = loop_collector.execution_loops.back()->body;
  }

  Block block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{},
              /*name_hint=*/"sunmmio_tile_scope_analysis",
              /*body=*/boundary_stmt);
  Array<Array<BufferRegion>> access =
      GetBlockReadWriteRegion(block, visible_buffers);
  OpaqueBuiltinAccessCollector opaque_access_collector;
  opaque_access_collector.Collect(boundary_stmt);

  Array<BufferRegion> raw_use_in = access[0];
  for (const BufferRegion &region : opaque_access_collector.reads) {
    raw_use_in.push_back(region);
  }
  Array<BufferRegion> raw_def_out = access[1];
  for (const BufferRegion &region : opaque_access_collector.writes) {
    raw_def_out.push_back(region);
  }

  std::unordered_set<const VarNode *> local_buffer_vars =
      CollectLocalBufferVars(match.root_stmt);

  Array<BufferRegion> use_in =
      DedupeExternalRegions(raw_use_in, local_buffer_vars);
  use_in =
      RemoveMatchingRegions(use_in, opaque_access_collector.write_only_writes);
  Array<BufferRegion> def_out =
      DedupeExternalRegions(raw_def_out, local_buffer_vars);

  std::vector<int> available_at_execution_depths;
  available_at_execution_depths.reserve(def_out.size());
  for (const BufferRegion &region : def_out) {
    available_at_execution_depths.push_back(
        ComputeAvailableExecutionDepth(region, loop_collector.execution_loops));
  }

  TileScopeRegion summary;
  summary.root_stmt = match.root_stmt;
  summary.scope_entry_for = scope_entry_for;
  summary.root_name =
      static_cast<std::string>(scope_entry_for->loop_var->name_hint);
  summary.execution_loops = loop_collector.execution_loops;
  summary.execution_loop_var_names =
      ExtractExecutionLoopVarNames(loop_collector.execution_loops);
  summary.logical_execution_axis_keys = BuildLogicalExecutionAxisKeys(
      scope_entry_for, loop_collector.execution_loops);
  summary.execution_loop_extents = execution_loop_extents;
  summary.use_in = use_in;
  summary.def_out = def_out;
  summary.available_at_execution_depths = available_at_execution_depths;
  return summary;
}

} // namespace

static std::vector<TileScopeWindowGraph> BuildWindowGraphs(
    const std::vector<TileScopeRegionRun> &region_runs,
    const std::vector<NormalizedTileScopeRegion> &normalized_regions) {
  std::vector<TileScopeWindowGraph> graphs;
  graphs.reserve(region_runs.size());
  arith::Analyzer analyzer;

  for (const TileScopeRegionRun &run : region_runs) {
    TileScopeWindowGraph graph;

    /* ---- Sweep one run in source order and keep the last relevant uses/defs
     * per buffer. Reads produce RAW edges from currently active defs. Writes
     * produce WAR/WAW edges against currently active reads/defs and then kill
     * the overlapping older accesses they supersede. ---- */
    std::unordered_map<const BufferNode *, std::vector<ActiveUseInfo>>
        active_uses;
    std::unordered_map<const BufferNode *, std::vector<ActiveDefInfo>>
        active_defs;

    for (int region_offset = 0; region_offset < run.num_regions;
         ++region_offset) {
      int region_index = run.begin_region_index + region_offset;
      const NormalizedTileScopeRegion &normalized =
          normalized_regions[region_index];

      std::vector<ActiveUseInfo> current_uses;
      current_uses.reserve(normalized.use_in.size());
      for (size_t use_index = 0; use_index < normalized.use_in.size();
           ++use_index) {
        const NormalizedBufferAccess &use_access = normalized.use_in[use_index];
        const BufferNode *buffer = use_access.region->buffer.get();
        auto defs_it = active_defs.find(buffer);
        if (defs_it != active_defs.end()) {
          for (const ActiveDefInfo &active_def : defs_it->second) {
            std::optional<OverlapFacts> facts =
                ComputeOverlapFacts(*active_def.access, use_access,
                                    TileScopeDependenceKind::kRAW, &analyzer);
            if (facts.has_value()) {
              graph.edges.push_back(MakeDependenceEdge(
                  active_def.region_index, region_index,
                  TileScopeDependenceKind::kRAW, active_def.access_index,
                  static_cast<int>(use_index), facts.value()));
            }
          }
        }
        current_uses.push_back(
            {region_index, static_cast<int>(use_index), &use_access});
      }

      std::vector<ActiveDefInfo> current_defs;
      current_defs.reserve(normalized.def_out.size());
      for (size_t def_index = 0; def_index < normalized.def_out.size();
           ++def_index) {
        const NormalizedBufferAccess &def_access =
            normalized.def_out[def_index];
        const BufferNode *buffer = def_access.region->buffer.get();

        auto &reads = active_uses[buffer];
        std::vector<ActiveUseInfo> surviving_reads;
        surviving_reads.reserve(reads.size());
        for (const ActiveUseInfo &active_use : reads) {
          std::optional<OverlapFacts> facts =
              ComputeOverlapFacts(*active_use.access, def_access,
                                  TileScopeDependenceKind::kWAR, &analyzer);
          if (!facts.has_value()) {
            surviving_reads.push_back(active_use);
            continue;
          }
          graph.edges.push_back(MakeDependenceEdge(
              active_use.region_index, region_index,
              TileScopeDependenceKind::kWAR, active_use.access_index,
              static_cast<int>(def_index), facts.value()));
        }
        reads = std::move(surviving_reads);

        auto &defs = active_defs[buffer];
        std::vector<ActiveDefInfo> surviving_defs;
        surviving_defs.reserve(defs.size());
        for (const ActiveDefInfo &active_def : defs) {
          std::optional<OverlapFacts> facts =
              ComputeOverlapFacts(*active_def.access, def_access,
                                  TileScopeDependenceKind::kWAW, &analyzer);
          if (!facts.has_value()) {
            surviving_defs.push_back(active_def);
            continue;
          }
          graph.edges.push_back(MakeDependenceEdge(
              active_def.region_index, region_index,
              TileScopeDependenceKind::kWAW, active_def.access_index,
              static_cast<int>(def_index), facts.value()));
        }
        defs = std::move(surviving_defs);

        current_defs.push_back(
            {region_index, static_cast<int>(def_index), &def_access});
      }

      for (const ActiveUseInfo &active_use : current_uses) {
        active_uses[active_use.access->region->buffer.get()].push_back(
            active_use);
      }
      for (const ActiveDefInfo &active_def : current_defs) {
        active_defs[active_def.access->region->buffer.get()].push_back(
            active_def);
      }
    }

    graphs.push_back(std::move(graph));
  }

  return graphs;
}

SunmmioTileLoopFusionProgram
BuildSunmmioTileLoopFusionProgram(const PrimFunc &func) {
  // First discover every planner-visible region and every maximal source-order
  // run where loop fusion may later occur.
  PlannerVisibleProgramCollector collector;
  collector.Collect(func);
  Map<Var, Buffer> visible_buffers = CollectVisibleBuffers(func);

  SunmmioTileLoopFusionProgram program;
  program.region_runs = collector.region_runs;
  program.regions.reserve(collector.region_matches.size());
  for (size_t region_index = 0; region_index < collector.region_matches.size();
       ++region_index) {
    TileScopeRegion region = AnalyzeOneTileScopeRegion(
        collector.region_matches[region_index], visible_buffers);
    region.global_region_index = static_cast<int>(region_index);
    program.regions.push_back(std::move(region));
  }
  return program;
}

std::vector<SunmmioTileLoopFusionWindowProblem>
BuildSunmmioTileLoopFusionWindowProblems(
    const SunmmioTileLoopFusionProgram &program) {
  // Then normalize access boundaries into logical-axis space and attach the
  // per-run dependence graph consumed by the planner.
  std::vector<NormalizedTileScopeRegion> normalized_regions =
      NormalizeRegionBoundaries(program.regions);
  std::vector<TileScopeWindowGraph> graphs =
      BuildWindowGraphs(program.region_runs, normalized_regions);

  std::vector<SunmmioTileLoopFusionWindowProblem> problems;
  problems.reserve(program.region_runs.size());
  for (size_t run_index = 0; run_index < program.region_runs.size();
       ++run_index) {
    const TileScopeRegionRun &run = program.region_runs[run_index];

    SunmmioTileLoopFusionWindowProblem problem;
    problem.graph = graphs[run_index];
    problem.regions.reserve(run.num_regions);
    problem.normalized_regions.reserve(run.num_regions);
    for (int region_offset = 0; region_offset < run.num_regions;
         ++region_offset) {
      int global_region_index = run.begin_region_index + region_offset;
      problem.regions.push_back(program.regions[global_region_index]);
      problem.normalized_regions.push_back(
          normalized_regions[global_region_index]);
    }
    problems.push_back(std::move(problem));
  }
  return problems;
}

} // namespace tl
} // namespace tvm
