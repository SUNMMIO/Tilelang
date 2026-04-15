#include "sunmmio_tile_loop_fusion_analysis.h"
#include "sunmmio_tile_loop_fusion_planner.h"
#include "sunmmio_tile_loop_fusion_utils.h"

#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/reduce.h"
#include "../op/utils.h"
#include "../transform/common/attr.h"

#include <tvm/arith/analyzer.h>
#include <tvm/arith/pattern.h>
#include <tvm/ffi/reflection/registry.h>
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

bool IsTileInterior(const ForNode *loop) {
  return IsAnnotatedOne(loop, attr::tile_interior);
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

std::vector<int> BuildIdentityExecMap(int rank) {
  std::vector<int> exec_map;
  exec_map.reserve(rank);
  for (int axis = 0; axis < rank; ++axis) {
    exec_map.push_back(axis);
  }
  return exec_map;
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

struct PlannerVisibleProgramSummary {
  std::vector<PlannerVisibleRegionMatch> region_matches;
  std::vector<TileScopeWindowSummary> windows;
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
  std::vector<For> interior_loops;

private:
  void VisitStmt_(const ForNode *op) final {
    if (IsTileExecutionLoop(op)) {
      execution_loops.push_back(ffi::GetRef<For>(op));
    }
    if (IsTileInterior(op)) {
      interior_loops.push_back(ffi::GetRef<For>(op));
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

struct OpaqueBuiltinAccessSummary {
  Array<BufferRegion> reads;
  Array<BufferRegion> writes;
  Array<BufferRegion> write_only_writes;
};

class OpaqueBuiltinAccessCollector : public StmtExprVisitor {
public:
  OpaqueBuiltinAccessSummary summary;

private:
  void VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call != nullptr && call->op.same_as(vector_core_in_tile_reduce()) &&
        call->args.size() >= 3) {
      if (std::optional<BufferRegion> dst =
              NormalizeRegionArgument(call->args[1])) {
        summary.writes.push_back(*dst);
        summary.write_only_writes.push_back(*dst);
      }
      if (std::optional<BufferRegion> src =
              NormalizeRegionArgument(call->args[2])) {
        summary.reads.push_back(*src);
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

OpaqueBuiltinAccessSummary CollectOpaqueBuiltinAccesses(const Stmt &stmt) {
  OpaqueBuiltinAccessCollector collector;
  collector(stmt);
  return collector.summary;
}

const CallNode *GetTileOperatorCall(const Stmt &stmt, const Op &op) {
  const auto *eval = stmt.as<EvaluateNode>();
  if (eval == nullptr) {
    return nullptr;
  }
  const auto *call = eval->value.as<CallNode>();
  if (call == nullptr || !call->op.same_as(op)) {
    return nullptr;
  }
  return call;
}

std::optional<For> FindWrappedTileScopeEntryLoop(const Stmt &stmt) {
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
  PlannerVisibleProgramSummary summary;

private:
  void Flush(std::vector<int> *run) {
    if (!run->empty()) {
      summary.windows.push_back({*run});
      run->clear();
    }
  }

  void VisitStmt_(const SeqStmtNode *op) final {
    std::vector<int> current_run;
    int index = 0;
    while (index < static_cast<int>(op->seq.size())) {
      std::optional<PlannerVisibleMatchResult> match =
          MatchPlannerVisibleRegion(op->seq, index);
      if (match) {
        int region_index = static_cast<int>(summary.region_matches.size());
        summary.region_matches.push_back(match->match);
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
      int region_index = static_cast<int>(summary.region_matches.size());
      PlannerVisibleRegionMatch match;
      match.root_stmt = ffi::GetRef<For>(op);
      match.scope_entry_for = ffi::GetRef<For>(op);
      summary.region_matches.push_back(match);
      summary.windows.push_back({{region_index}});
      return;
    }
    StmtVisitor::VisitStmt_(op);
  }
};

PlannerVisibleProgramSummary
CollectPlannerVisibleProgram(const PrimFunc &func) {
  PlannerVisibleProgramCollector collector;
  collector(func->body);
  return collector.summary;
}

Array<BufferRegion> DedupeExternalRegions(
    const Array<BufferRegion> &regions,
    const std::unordered_set<const VarNode *> &local_buffer_vars = {}) {
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

Array<PrimExpr> BuildTileShape(const std::vector<For> &interior_loops) {
  int max_axis = -1;
  for (const For &loop : interior_loops) {
    max_axis = std::max(max_axis,
                        GetIntAnnotation(loop.get(), attr::tile_interior_axis));
  }

  Array<PrimExpr> tile_shape;
  if (max_axis < 0) {
    return tile_shape;
  }

  std::vector<PrimExpr> extents(max_axis + 1);
  std::vector<bool> present(max_axis + 1, false);
  for (const For &loop : interior_loops) {
    int axis = GetIntAnnotation(loop.get(), attr::tile_interior_axis);
    if (axis >= 0 && !present[axis]) {
      extents[axis] = loop->extent;
      present[axis] = true;
    }
  }

  for (int axis = 0; axis <= max_axis; ++axis) {
    ICHECK(present[axis]) << "Missing tile.interior_axis=" << axis
                          << " under tile.scope_entry loop "
                          << interior_loops.front()->loop_var;
    tile_shape.push_back(extents[axis]);
  }
  return tile_shape;
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

std::vector<int>
BuildExecutionAxisToLoopIndex(const std::vector<For> &execution_loops) {
  int max_axis = -1;
  for (const For &loop : execution_loops) {
    max_axis = std::max(
        max_axis, GetIntAnnotation(loop.get(), attr::tile_execution_axis));
  }

  if (max_axis < 0) {
    return {};
  }

  std::vector<int> exec_map(max_axis + 1, -1);
  for (size_t prefix_index = 0; prefix_index < execution_loops.size();
       ++prefix_index) {
    int axis = GetIntAnnotation(execution_loops[prefix_index].get(),
                                attr::tile_execution_axis);
    ICHECK(axis >= 0)
        << "Missing tile.execution_axis on exposed execution loop";
    ICHECK(axis < static_cast<int>(exec_map.size()));
    exec_map[axis] = static_cast<int>(prefix_index);
  }

  for (size_t axis = 0; axis < exec_map.size(); ++axis) {
    ICHECK_GE(exec_map[axis], 0) << "Execution axis " << axis
                                 << " is missing from the tile scope prefix";
  }
  return exec_map;
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

const char *DependenceKindToCString(TileScopeDependenceKind kind) {
  switch (kind) {
  case TileScopeDependenceKind::kRAW:
    return "RAW";
  case TileScopeDependenceKind::kWAR:
    return "WAR";
  case TileScopeDependenceKind::kWAW:
    return "WAW";
  }
  LOG(FATAL) << "Unknown TileScopeDependenceKind value";
  return "unknown";
}

struct ActiveUseInfo {
  int region_index{-1};
  BufferRegion raw_region;
  BufferRegion normalized_region;
};

struct ActiveDefInfo {
  int region_index{-1};
  BufferRegion raw_region;
  BufferRegion normalized_region;
};

std::optional<BufferRegion> IntersectBufferRegions(const BufferRegion &lhs,
                                                   const BufferRegion &rhs,
                                                   arith::Analyzer *analyzer) {
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

std::unordered_map<std::string, int>
BuildLogicalAxisDepthMap(const TileScopeRegionSummary &region) {
  std::unordered_map<std::string, int> depth_by_axis;
  for (size_t i = 0; i < region.logical_execution_axis_keys.size(); ++i) {
    depth_by_axis[region.logical_execution_axis_keys[i]] =
        static_cast<int>(i) + 1;
  }
  return depth_by_axis;
}

int ComputeRequiredSharedPrefixDepth(const BufferRegion &overlap_region,
                                     const TileScopeRegionSummary &src_region,
                                     const TileScopeRegionSummary &dst_region) {
  std::unordered_map<std::string, int> src_depth =
      BuildLogicalAxisDepthMap(src_region);
  std::unordered_map<std::string, int> dst_depth =
      BuildLogicalAxisDepthMap(dst_region);

  int rho = 0;
  for (const Range &range : overlap_region->region) {
    VarUseCollector collector;
    collector(range->min);
    collector(range->extent);
    for (const VarNode *var : collector.seen_vars) {
      std::string axis_name = static_cast<std::string>(var->name_hint);
      auto src_it = src_depth.find(axis_name);
      auto dst_it = dst_depth.find(axis_name);
      if (src_it != src_depth.end() && dst_it != dst_depth.end()) {
        rho = std::max(rho, std::max(src_it->second, dst_it->second));
      }
    }
  }
  return rho;
}

struct LogicalBufferDimSummary {
  std::optional<std::string> axis_key;
  int64_t extent{0};
  PrimExpr min;
};

std::optional<std::vector<LogicalBufferDimSummary>>
DescribeLogicalBufferRegion(const BufferRegion &region,
                            const TileScopeRegionSummary &owner,
                            arith::Analyzer *analyzer) {
  if (owner.execution_loops.size() !=
      owner.logical_execution_axis_keys.size()) {
    return std::nullopt;
  }

  std::unordered_map<const VarNode *, std::string> axis_key_by_var;
  for (size_t i = 0; i < owner.execution_loops.size(); ++i) {
    axis_key_by_var[owner.execution_loops[i]->loop_var.get()] =
        owner.logical_execution_axis_keys[i];
  }

  std::vector<LogicalBufferDimSummary> dims;
  dims.reserve(region->region.size());
  for (const Range &range : region->region) {
    PrimExpr extent = analyzer->Simplify(range->extent);
    const auto *imm = extent.as<IntImmNode>();
    if (imm == nullptr) {
      return std::nullopt;
    }

    std::optional<std::string> axis_key;
    VarUseCollector collector;
    collector(range->min);
    collector(range->extent);
    for (const VarNode *var : collector.seen_vars) {
      auto it = axis_key_by_var.find(var);
      if (it == axis_key_by_var.end()) {
        continue;
      }
      if (axis_key && *axis_key != it->second) {
        return std::nullopt;
      }
      axis_key = it->second;
    }

    dims.push_back({axis_key, imm->value, range->min});
  }
  return dims;
}

std::optional<BufferRegion> InferLogicalOverlapRegion(
    const BufferRegion &lhs, const TileScopeRegionSummary &lhs_owner,
    const BufferRegion &rhs, const TileScopeRegionSummary &rhs_owner,
    arith::Analyzer *analyzer) {
  if (!lhs->buffer.same_as(rhs->buffer) ||
      lhs->region.size() != rhs->region.size()) {
    return std::nullopt;
  }

  std::optional<std::vector<LogicalBufferDimSummary>> lhs_dims =
      DescribeLogicalBufferRegion(lhs, lhs_owner, analyzer);
  std::optional<std::vector<LogicalBufferDimSummary>> rhs_dims =
      DescribeLogicalBufferRegion(rhs, rhs_owner, analyzer);
  if (!lhs_dims || !rhs_dims) {
    return std::nullopt;
  }

  Array<Range> overlap;
  for (size_t i = 0; i < lhs_dims->size(); ++i) {
    const LogicalBufferDimSummary &lhs_dim = (*lhs_dims)[i];
    const LogicalBufferDimSummary &rhs_dim = (*rhs_dims)[i];
    if (lhs_dim.axis_key != rhs_dim.axis_key) {
      return std::nullopt;
    }
    if (!lhs_dim.axis_key.has_value() &&
        !StructuralEqual()(lhs_dim.min, rhs_dim.min)) {
      return std::nullopt;
    }

    PrimExpr min = lhs_dim.axis_key.has_value()
                       ? PrimExpr(Var(lhs_dim.axis_key.value()))
                       : analyzer->Simplify(lhs_dim.min);
    int64_t extent = std::min(lhs_dim.extent, rhs_dim.extent);
    if (extent <= 0) {
      return std::nullopt;
    }
    overlap.push_back(Range::FromMinExtent(min, Integer(extent)));
  }

  return BufferRegion(lhs->buffer, overlap);
}

int64_t EdgeWeightAsInt64(const PrimExpr &weight) {
  if (const auto *imm = weight.as<IntImmNode>()) {
    return imm->value;
  }
  return -1;
}

TileScopeDependenceEdgeSummary MakeDependenceEdge(
    int src_region_index, int dst_region_index, TileScopeDependenceKind kind,
    const BufferRegion &overlap_region,
    const TileScopeRegionSummary &src_region,
    const TileScopeRegionSummary &dst_region, arith::Analyzer *analyzer);

std::optional<TileScopeDependenceEdgeSummary> MakeDependenceEdgeWithFallback(
    int src_region_index, int dst_region_index, TileScopeDependenceKind kind,
    const std::optional<BufferRegion> &exact_overlap_region,
    const BufferRegion &raw_src_region, const BufferRegion &raw_dst_region,
    const TileScopeRegionSummary &src_region,
    const TileScopeRegionSummary &dst_region, arith::Analyzer *analyzer) {
  std::optional<TileScopeDependenceEdgeSummary> exact_edge;
  if (exact_overlap_region.has_value()) {
    exact_edge = MakeDependenceEdge(src_region_index, dst_region_index, kind,
                                    exact_overlap_region.value(), src_region,
                                    dst_region, analyzer);
  }

  std::optional<BufferRegion> fallback_overlap = InferLogicalOverlapRegion(
      raw_src_region, src_region, raw_dst_region, dst_region, analyzer);
  std::optional<TileScopeDependenceEdgeSummary> fallback_edge;
  if (fallback_overlap.has_value()) {
    fallback_edge = MakeDependenceEdge(src_region_index, dst_region_index, kind,
                                       fallback_overlap.value(), src_region,
                                       dst_region, analyzer);
  }

  if (!exact_edge) {
    return fallback_edge;
  }
  if (!fallback_edge) {
    return exact_edge;
  }

  bool fallback_has_more_rho =
      fallback_edge->rho > exact_edge->rho && exact_edge->rho == 0;
  bool fallback_has_more_weight = EdgeWeightAsInt64(fallback_edge->weight) >
                                      EdgeWeightAsInt64(exact_edge->weight) &&
                                  EdgeWeightAsInt64(exact_edge->weight) == 0;
  if (fallback_has_more_rho || fallback_has_more_weight) {
    return fallback_edge;
  }
  return exact_edge;
}

PrimExpr ComputeEdgeWeight(const BufferRegion &overlap_region,
                           TileScopeDependenceKind kind,
                           arith::Analyzer *analyzer) {
  if (kind != TileScopeDependenceKind::kRAW) {
    return Integer(0);
  }

  int64_t element_count = 1;
  for (const Range &range : overlap_region->region) {
    PrimExpr extent = analyzer->Simplify(range->extent);
    const auto *imm = extent.as<IntImmNode>();
    if (imm == nullptr) {
      return Integer(0);
    }
    element_count *= imm->value;
  }

  int64_t bytes = overlap_region->buffer->dtype.bytes();
  return Integer(2 * element_count * bytes);
}

TileScopeDependenceEdgeSummary MakeDependenceEdge(
    int src_region_index, int dst_region_index, TileScopeDependenceKind kind,
    const BufferRegion &overlap_region,
    const TileScopeRegionSummary &src_region,
    const TileScopeRegionSummary &dst_region, arith::Analyzer *analyzer) {
  return {
      src_region_index,
      dst_region_index,
      kind,
      overlap_region,
      ComputeRequiredSharedPrefixDepth(overlap_region, src_region, dst_region),
      ComputeEdgeWeight(overlap_region, kind, analyzer)};
}

TileScopeRegionSummary
AnalyzeOneTileScopeRegion(const PlannerVisibleRegionMatch &match,
                          const Map<Var, Buffer> &visible_buffers) {
  const For &scope_entry_for = match.scope_entry_for;
  RegionLoopCollector loop_collector;
  loop_collector(scope_entry_for);

  arith::Analyzer analyzer;

  TileScopeSignatureSummary sig;
  sig.rank = static_cast<int>(loop_collector.execution_loops.size());
  sig.tile_shape = BuildTileShape(loop_collector.interior_loops);
  sig.execution_axis_to_loop_index =
      BuildExecutionAxisToLoopIndex(loop_collector.execution_loops);
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
  OpaqueBuiltinAccessSummary opaque_access =
      CollectOpaqueBuiltinAccesses(boundary_stmt);

  Array<BufferRegion> raw_use_in = access[0];
  for (const BufferRegion &region : opaque_access.reads) {
    raw_use_in.push_back(region);
  }
  Array<BufferRegion> raw_def_out = access[1];
  for (const BufferRegion &region : opaque_access.writes) {
    raw_def_out.push_back(region);
  }

  std::unordered_set<const VarNode *> local_buffer_vars =
      CollectLocalBufferVars(match.root_stmt);

  Array<BufferRegion> use_in =
      DedupeExternalRegions(raw_use_in, local_buffer_vars);
  use_in = RemoveMatchingRegions(use_in, opaque_access.write_only_writes);
  Array<BufferRegion> def_out =
      DedupeExternalRegions(raw_def_out, local_buffer_vars);

  std::vector<int> available_at_execution_depths;
  available_at_execution_depths.reserve(def_out.size());
  for (const BufferRegion &region : def_out) {
    available_at_execution_depths.push_back(
        ComputeAvailableExecutionDepth(region, loop_collector.execution_loops));
  }

  TileScopeRegionSummary summary;
  summary.kind = TileScopeRegionKind::kScopeEntryLoop;
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
  summary.sig = sig;
  summary.use_in = use_in;
  summary.def_out = def_out;
  summary.available_at_execution_depths = available_at_execution_depths;
  return summary;
}

Map<String, ffi::Any> BufferRegionToDebugObject(const BufferRegion &region) {
  Map<String, ffi::Any> result;
  Array<String> mins;
  Array<String> extents;
  for (const Range &range : region->region) {
    mins.push_back(PrimExprToString(range->min));
    extents.push_back(PrimExprToString(range->extent));
  }
  result.Set("buffer", region->buffer->name);
  result.Set("mins", mins);
  result.Set("extents", extents);
  return result;
}

Map<String, ffi::Any>
TileScopeRegionToDebugObject(const TileScopeRegionSummary &region) {
  Map<String, ffi::Any> result;

  Array<String> execution_loop_vars;
  for (const std::string &axis : region.execution_loop_var_names) {
    execution_loop_vars.push_back(String(axis));
  }

  Array<String> logical_execution_axes;
  for (const std::string &axis : region.logical_execution_axis_keys) {
    logical_execution_axes.push_back(String(axis));
  }

  Array<String> execution_loop_extents;
  for (const PrimExpr &expr : region.execution_loop_extents) {
    execution_loop_extents.push_back(PrimExprToString(expr));
  }

  Array<String> tile_shape;
  for (const PrimExpr &expr : region.sig.tile_shape) {
    tile_shape.push_back(PrimExprToString(expr));
  }

  Array<Integer> execution_axis_to_loop_index;
  for (int loop_index : region.sig.execution_axis_to_loop_index) {
    execution_axis_to_loop_index.push_back(Integer(loop_index));
  }

  Array<Map<String, ffi::Any>> use_in;
  for (const BufferRegion &buffer_region : region.use_in) {
    use_in.push_back(BufferRegionToDebugObject(buffer_region));
  }

  Array<Map<String, ffi::Any>> def_out;
  for (const BufferRegion &buffer_region : region.def_out) {
    def_out.push_back(BufferRegionToDebugObject(buffer_region));
  }

  Array<Integer> available_at_execution_depths;
  for (int depth : region.available_at_execution_depths) {
    available_at_execution_depths.push_back(Integer(depth));
  }

  result.Set("kind", String("tile_scope_entry"));
  result.Set("root_name", String(region.root_name));
  result.Set("root_loop_var", String(region.root_name));
  result.Set("execution_loop_vars", execution_loop_vars);
  result.Set("logical_execution_axes", logical_execution_axes);
  result.Set("execution_loop_extents", execution_loop_extents);
  result.Set("sig_rank", Integer(region.sig.rank));
  result.Set("sig_tile_shape", tile_shape);
  result.Set("sig_execution_axis_to_loop_index", execution_axis_to_loop_index);
  result.Set("use_in", use_in);
  result.Set("def_out", def_out);
  result.Set("available_at_execution_depths", available_at_execution_depths);
  return result;
}

Map<String, ffi::Any> TileScopeDependenceEdgeToDebugObject(
    const TileScopeDependenceEdgeSummary &edge) {
  Map<String, ffi::Any> result;
  result.Set("src", Integer(edge.src_region_index));
  result.Set("dst", Integer(edge.dst_region_index));
  result.Set("kind", String(DependenceKindToCString(edge.kind)));
  result.Set("buffer_region", BufferRegionToDebugObject(edge.buffer_region));
  result.Set("rho", Integer(edge.rho));
  if (const auto *imm = edge.weight.as<IntImmNode>()) {
    result.Set("w", Integer(imm->value));
  } else {
    result.Set("w", PrimExprToString(edge.weight));
  }
  return result;
}

Map<String, ffi::Any>
TileScopeWindowGraphToDebugObject(const TileScopeWindowGraphSummary &graph) {
  Map<String, ffi::Any> result;
  Array<Integer> region_indices;
  for (int region_index : graph.region_indices) {
    region_indices.push_back(Integer(region_index));
  }

  Array<Map<String, ffi::Any>> edges;
  for (const TileScopeDependenceEdgeSummary &edge : graph.edges) {
    edges.push_back(TileScopeDependenceEdgeToDebugObject(edge));
  }

  result.Set("region_indices", region_indices);
  result.Set("edges", edges);
  return result;
}

Map<String, ffi::Any>
PlannerScoreToDebugObject(const SunmmioTileLoopFusionPlannerScore &score) {
  Map<String, ffi::Any> result;
  result.Set("write_cut_cost", Integer(score.write_cut_cost));
  result.Set("shared_read_cost", Integer(score.shared_read_cost));
  result.Set("live_range_penalty", Integer(score.live_range_penalty));
  result.Set("reorder_penalty", Integer(score.reorder_penalty));
  return result;
}

Map<String, ffi::Any> PlannerActionToDebugObject(
    const SunmmioTileLoopFusionPlannerActionSummary &action) {
  Map<String, ffi::Any> result;
  result.Set("region_index", Integer(action.region_index));
  result.Set("close_to_depth", Integer(action.close_to_depth));
  result.Set("open_to_depth", Integer(action.open_to_depth));

  Array<Array<String>> opened_shells;
  for (const std::vector<std::string> &shell_axes : action.opened_shells) {
    Array<String> shell;
    for (const std::string &axis : shell_axes) {
      shell.push_back(String(axis));
    }
    opened_shells.push_back(shell);
  }
  result.Set("opened_shells", opened_shells);

  Array<Array<String>> opened_shell_extents;
  for (const Array<PrimExpr> &shell_extents : action.opened_shell_extents) {
    Array<String> extents;
    for (const PrimExpr &extent : shell_extents) {
      extents.push_back(PrimExprToString(extent));
    }
    opened_shell_extents.push_back(extents);
  }
  result.Set("opened_shell_extents", opened_shell_extents);
  return result;
}

Map<String, ffi::Any>
PlannerTreeNodeToDebugObject(const SunmmioTileLoopFusionPlannerTreeNode &node) {
  Map<String, ffi::Any> result;
  result.Set("is_scope", Bool(node.is_scope));
  result.Set("region_index", Integer(node.region_index));

  Array<String> shell_axes;
  for (const std::string &axis : node.shell_axes) {
    shell_axes.push_back(String(axis));
  }
  result.Set("shell_axes", shell_axes);

  Array<String> shell_extents;
  for (const PrimExpr &extent : node.shell_extents) {
    shell_extents.push_back(PrimExprToString(extent));
  }
  result.Set("shell_extents", shell_extents);

  Array<Map<String, ffi::Any>> children;
  for (const SunmmioTileLoopFusionPlannerTreeNode &child : node.children) {
    children.push_back(PlannerTreeNodeToDebugObject(child));
  }
  result.Set("children", children);
  return result;
}

Map<String, ffi::Any>
PlannerWindowToDebugObject(const SunmmioTileLoopFusionWindowPlanSummary &plan) {
  Map<String, ffi::Any> result;

  Array<Integer> region_indices;
  for (int region_index : plan.region_indices) {
    region_indices.push_back(Integer(region_index));
  }
  result.Set("region_indices", region_indices);

  Array<Integer> order;
  for (int region_index : plan.order) {
    order.push_back(Integer(region_index));
  }
  result.Set("order", order);
  result.Set("score", PlannerScoreToDebugObject(plan.score));

  Array<Map<String, ffi::Any>> actions;
  for (const SunmmioTileLoopFusionPlannerActionSummary &action : plan.actions) {
    actions.push_back(PlannerActionToDebugObject(action));
  }
  result.Set("actions", actions);

  Array<Map<String, ffi::Any>> tree;
  for (const SunmmioTileLoopFusionPlannerTreeNode &node : plan.tree) {
    tree.push_back(PlannerTreeNodeToDebugObject(node));
  }
  result.Set("tree", tree);
  return result;
}

PrimFunc LookupMainPrimFunc(const IRModule &mod) {
  BaseFunc func = mod->Lookup("main");
  const auto *prim = func.as<PrimFuncNode>();
  ICHECK(prim) << "Expected `main` to be a PrimFunc";
  return ffi::GetRef<PrimFunc>(prim);
}

} // namespace

std::vector<TileScopeRegionSummary>
AnalyzeSunmmioTileLoopFusionRegions(const PrimFunc &func) {
  PlannerVisibleProgramSummary visible_program =
      CollectPlannerVisibleProgram(func);

  Map<Var, Buffer> visible_buffers = CollectVisibleBuffers(func);
  std::vector<TileScopeRegionSummary> regions;
  regions.reserve(visible_program.region_matches.size());
  for (const PlannerVisibleRegionMatch &match :
       visible_program.region_matches) {
    regions.push_back(AnalyzeOneTileScopeRegion(match, visible_buffers));
  }
  return regions;
}

std::vector<TileScopeWindowSummary>
BuildSunmmioTileLoopFusionWindows(const PrimFunc &func) {
  return CollectPlannerVisibleProgram(func).windows;
}

std::vector<TileScopeWindowGraphSummary>
BuildSunmmioTileLoopFusionLegalityGraphs(
    const std::vector<TileScopeRegionSummary> &regions,
    const std::vector<TileScopeWindowSummary> &windows) {
  std::vector<NormalizedTileScopeRegionSummary> normalized_regions =
      NormalizeRegionBoundaries(regions);

  std::vector<TileScopeWindowGraphSummary> graphs;
  graphs.reserve(windows.size());
  arith::Analyzer analyzer;

  for (const TileScopeWindowSummary &window : windows) {
    TileScopeWindowGraphSummary graph;
    graph.region_indices = window.region_indices;

    std::unordered_map<const BufferNode *, std::vector<ActiveUseInfo>>
        active_uses;
    std::unordered_map<const BufferNode *, std::vector<ActiveDefInfo>>
        active_defs;

    for (int region_index : graph.region_indices) {
      const TileScopeRegionSummary &region = regions[region_index];
      const NormalizedTileScopeRegionSummary &normalized =
          normalized_regions[region_index];

      std::vector<ActiveUseInfo> current_uses;
      current_uses.reserve(normalized.use_in.size());
      for (size_t use_index = 0; use_index < normalized.use_in.size();
           ++use_index) {
        const BufferRegion &use_region = normalized.use_in[use_index];
        const BufferRegion &raw_use_region = region.use_in[use_index];
        const BufferNode *buffer = use_region->buffer.get();
        auto defs_it = active_defs.find(buffer);
        if (defs_it != active_defs.end()) {
          for (const ActiveDefInfo &active_def : defs_it->second) {
            std::optional<BufferRegion> overlap = IntersectBufferRegions(
                active_def.normalized_region, use_region, &analyzer);
            std::optional<TileScopeDependenceEdgeSummary> edge =
                MakeDependenceEdgeWithFallback(
                    active_def.region_index, region_index,
                    TileScopeDependenceKind::kRAW, overlap,
                    active_def.raw_region, raw_use_region,
                    regions[active_def.region_index], region, &analyzer);
            if (edge.has_value()) {
              graph.edges.push_back(edge.value());
            }
          }
        }
        current_uses.push_back({region_index, raw_use_region, use_region});
      }

      std::vector<ActiveDefInfo> current_defs;
      current_defs.reserve(normalized.def_out.size());
      for (size_t def_index = 0; def_index < normalized.def_out.size();
           ++def_index) {
        const BufferRegion &def_region = normalized.def_out[def_index];
        const BufferRegion &raw_def_region = region.def_out[def_index];
        const BufferNode *buffer = def_region->buffer.get();

        auto &reads = active_uses[buffer];
        std::vector<ActiveUseInfo> surviving_reads;
        surviving_reads.reserve(reads.size());
        for (const ActiveUseInfo &active_use : reads) {
          std::optional<BufferRegion> overlap = IntersectBufferRegions(
              active_use.normalized_region, def_region, &analyzer);
          std::optional<TileScopeDependenceEdgeSummary> edge =
              MakeDependenceEdgeWithFallback(
                  active_use.region_index, region_index,
                  TileScopeDependenceKind::kWAR, overlap, active_use.raw_region,
                  raw_def_region, regions[active_use.region_index], region,
                  &analyzer);
          if (!edge.has_value()) {
            surviving_reads.push_back(active_use);
            continue;
          }
          graph.edges.push_back(edge.value());
        }
        reads = std::move(surviving_reads);

        auto &defs = active_defs[buffer];
        std::vector<ActiveDefInfo> surviving_defs;
        surviving_defs.reserve(defs.size());
        for (const ActiveDefInfo &active_def : defs) {
          std::optional<BufferRegion> overlap = IntersectBufferRegions(
              active_def.normalized_region, def_region, &analyzer);
          std::optional<TileScopeDependenceEdgeSummary> edge =
              MakeDependenceEdgeWithFallback(
                  active_def.region_index, region_index,
                  TileScopeDependenceKind::kWAW, overlap, active_def.raw_region,
                  raw_def_region, regions[active_def.region_index], region,
                  &analyzer);
          if (!edge.has_value()) {
            surviving_defs.push_back(active_def);
            continue;
          }
          graph.edges.push_back(edge.value());
        }
        defs = std::move(surviving_defs);

        current_defs.push_back({region_index, raw_def_region, def_region});
      }

      for (const ActiveUseInfo &active_use : current_uses) {
        active_uses[active_use.normalized_region->buffer.get()].push_back(
            active_use);
      }
      for (const ActiveDefInfo &active_def : current_defs) {
        active_defs[active_def.normalized_region->buffer.get()].push_back(
            active_def);
      }
    }

    graphs.push_back(std::move(graph));
  }

  return graphs;
}

Map<String, ffi::Any>
DebugSunmmioTileLoopFusionAnalysisSummary(const IRModule &mod) {
  PrimFunc func = LookupMainPrimFunc(mod);
  std::vector<TileScopeRegionSummary> regions =
      AnalyzeSunmmioTileLoopFusionRegions(func);
  std::vector<TileScopeWindowSummary> windows =
      BuildSunmmioTileLoopFusionWindows(func);
  std::vector<TileScopeWindowGraphSummary> graphs =
      BuildSunmmioTileLoopFusionLegalityGraphs(regions, windows);
  std::vector<SunmmioTileLoopFusionWindowPlanSummary> plans =
      PlanSunmmioTileLoopFusionWindows(regions, windows, graphs);

  Map<String, ffi::Any> summary;
  Array<Integer> window_lengths;
  for (const auto &window : windows) {
    window_lengths.push_back(
        Integer(static_cast<int>(window.region_indices.size())));
  }

  Array<Map<String, ffi::Any>> region_summaries;
  for (const TileScopeRegionSummary &region : regions) {
    region_summaries.push_back(TileScopeRegionToDebugObject(region));
  }

  Array<Map<String, ffi::Any>> graph_summaries;
  for (const TileScopeWindowGraphSummary &graph : graphs) {
    graph_summaries.push_back(TileScopeWindowGraphToDebugObject(graph));
  }

  Array<Map<String, ffi::Any>> plan_summaries;
  for (const SunmmioTileLoopFusionWindowPlanSummary &plan : plans) {
    plan_summaries.push_back(PlannerWindowToDebugObject(plan));
  }

  summary.Set("region_count", Integer(static_cast<int>(regions.size())));
  summary.Set("window_count", Integer(static_cast<int>(windows.size())));
  summary.Set("window_lengths", window_lengths);
  summary.Set("regions", region_summaries);
  summary.Set("graphs", graph_summaries);
  summary.Set("plans", plan_summaries);
  return summary;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def(
      "tl.analysis.DebugSunmmioTileLoopFusionAnalysisSummary",
      DebugSunmmioTileLoopFusionAnalysisSummary);
}

} // namespace tl
} // namespace tvm
