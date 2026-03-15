/*!
 * \file analyze_pipelined_buffer_dependency.cc
 * \brief Analyze intra-iteration and inter-iteration RAW dependencies for
 *        buffers used inside pipelined loops.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <functional>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pipelined_buffer_dependency.h"
#include "../op/builtin.h"
#include "../op/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

constexpr const char *kStateBuffersAttr = "tl_buffer_dependency_state_buffers";
constexpr const char *kChannelBuffersAttr =
    "tl_buffer_dependency_channel_buffers";
constexpr const char *kIntraRawEdgesAttr = "tl_buffer_dependency_intra_raw";
constexpr const char *kInterRawEdgesAttr = "tl_buffer_dependency_inter_raw";
constexpr const char *kMixedRoleBuffersAttr =
    "tl_buffer_dependency_mixed_role_buffers";
constexpr const char *kMixedRoleDetailsAttr =
    "tl_buffer_dependency_mixed_role_details";
constexpr const char *kPartialOverwriteHazardsAttr =
    "tl_buffer_dependency_partial_overwrite_hazards";
constexpr const char *kCoveredRewritesAttr =
    "tl_buffer_dependency_covered_rewrites";
constexpr const char *kUnknownEffectsAttr =
    "tl_buffer_dependency_unknown_effects";

std::string EffectKindToString(CallEffectKind effect_kind) {
  switch (effect_kind) {
  case CallEffectKind::kPure:
    return "pure";
  case CallEffectKind::kReadState:
    return "read_state";
  case CallEffectKind::kUpdateState:
    return "update_state";
  case CallEffectKind::kControlJump:
    return "control_jump";
  case CallEffectKind::kSpecialCallArg:
    return "special_call_arg";
  case CallEffectKind::kEmbedInfo:
    return "embed_info";
  case CallEffectKind::kExprAnnotation:
    return "expr_annotation";
  default:
    return "unknown";
  }
}

std::string CallOpName(const CallNode *op) {
  if (auto opt = op->op.as<Op>()) {
    std::string name = opt.value()->name;
    if (name == "tir.call_extern" && !op->args.empty()) {
      if (const auto *func_name = op->args[0].as<StringImmNode>()) {
        name += ":" + func_name->value;
      }
    }
    return name;
  }
  std::ostringstream os;
  os << op->op;
  return os.str();
}

CallEffectKind GetCallEffectKind(const CallNode *op) {
  static auto op_call_effect = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
  if (auto opt = op->op.as<Op>()) {
    return static_cast<CallEffectKind>(op_call_effect[opt.value()]->value);
  }
  return CallEffectKind::kOpaque;
}

bool MayConflict(const Region &region1, const Region &region2) {
  ICHECK_EQ(region1.size(), region2.size());
  for (size_t i = 0; i < region1.size(); ++i) {
    auto int_set1 = arith::IntSet::FromRange(region1[i]);
    auto int_set2 = arith::IntSet::FromRange(region2[i]);
    if (arith::Intersect({int_set1, int_set2}).IsNothing()) {
      return false;
    }
  }
  return true;
}

std::string RegionToString(const BufferRegion &region) {
  std::ostringstream os;
  os << region;
  return os.str();
}

Array<PrimExpr> ToStringImmArray(const std::set<std::string> &values) {
  Array<PrimExpr> result;
  for (const std::string &value : values) {
    result.push_back(StringImm(value));
  }
  return result;
}

class BufferAccessCollector : public StmtExprVisitor {
public:
  explicit BufferAccessCollector(
      const Map<Var, Buffer> &buffer_data_to_buffer,
      const Map<Var, arith::IntSet> &loop_var_domains,
      const Map<Var, PrimExpr> &let_bindings,
      bool record_unknown_stmt_calls = false)
      : buffer_data_to_buffer_(buffer_data_to_buffer),
        loop_var_domains_(loop_var_domains),
        let_bindings_(let_bindings),
        record_unknown_stmt_calls_(record_unknown_stmt_calls) {}

  Array<BufferRegion> GetReads() const { return reads_; }
  Array<BufferRegion> GetWrites() const { return writes_; }
  Array<String> GetUnknownEffects() const { return unknown_effects_; }

private:
  PrimExpr SubstituteBindings(PrimExpr expr) const {
    PrimExpr remapped = tir::Substitute(expr, let_bindings_);
    while (!remapped.same_as(expr)) {
      expr = remapped;
      remapped = tir::Substitute(expr, let_bindings_);
    }
    return expr;
  }

  Range RelaxRange(const Range &range, const PrimExpr &shape) const {
    Range substituted = Range::FromMinExtent(SubstituteBindings(range->min),
                                             SubstituteBindings(range->extent));
    arith::IntSet relaxed = arith::EvalSet(Array<Range>{substituted},
                                           loop_var_domains_)[0];
    return relaxed.CoverRange(Range::FromMinExtent(0, shape));
  }

  Array<Range> RelaxIndicesToRegion(const Buffer &buffer,
                                    const Array<PrimExpr> &indices) const {
    ICHECK_EQ(buffer->shape.size(), indices.size());
    Array<Range> region;
    for (size_t i = 0; i < indices.size(); ++i) {
      const PrimExpr &index = indices[i];
      if (const auto *ramp = index.as<RampNode>()) {
        ICHECK(ramp->stride.as<IntImmNode>())
            << "Only constant-stride Ramp is supported in dependency analysis";
        ICHECK_EQ(ramp->stride.as<IntImmNode>()->value, 1)
            << "Only stride-1 Ramp is supported in dependency analysis";
        Range pointwise =
            Range::FromMinExtent(SubstituteBindings(ramp->base), ramp->lanes);
        arith::IntSet relaxed =
            arith::EvalSet(Array<Range>{pointwise}, loop_var_domains_)[0];
        region.push_back(
            relaxed.CoverRange(Range::FromMinExtent(0, buffer->shape[i])));
      } else {
        PrimExpr substituted = SubstituteBindings(index);
        arith::IntSet relaxed = arith::EvalSet(substituted, loop_var_domains_);
        region.push_back(
            relaxed.CoverRange(Range::FromMinExtent(0, buffer->shape[i])));
      }
    }
    return region;
  }

  BufferRegion RelaxBufferRegion(const BufferRegion &region) const {
    Array<Range> relaxed_region;
    ICHECK_EQ(region->buffer->shape.size(), region->region.size());
    for (size_t i = 0; i < region->region.size(); ++i) {
      relaxed_region.push_back(
          RelaxRange(region->region[i], region->buffer->shape[i]));
    }
    return BufferRegion(region->buffer, relaxed_region);
  }

  void AddUnique(Array<BufferRegion> *regions, const BufferRegion &region) {
    StructuralEqual equal;
    for (const BufferRegion &existing : *regions) {
      if (existing->buffer.same_as(region->buffer) &&
          equal(existing->region, region->region)) {
        return;
      }
    }
    regions->push_back(region);
  }

  void AddRead(const BufferRegion &region) { AddUnique(&reads_, region); }
  void AddWrite(const BufferRegion &region) { AddUnique(&writes_, region); }
  void AddUnknownEffect(const String &detail) {
    for (const String &existing : unknown_effects_) {
      if (existing == detail) {
        return;
      }
    }
    unknown_effects_.push_back(detail);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (op->predicate) {
      VisitExpr(op->predicate.value());
    }
    VisitExpr(op->value);
    AddWrite(BufferRegion(op->buffer, RelaxIndicesToRegion(op->buffer, op->indices)));
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    AddRead(BufferRegion(op->buffer, RelaxIndicesToRegion(op->buffer, op->indices)));
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(dma_copy())) {
      ICHECK_EQ(op->args.size(), 2U);
      AddRead(RelaxBufferRegion(NormalizeToBufferRegion(op->args[0])));
      AddWrite(RelaxBufferRegion(NormalizeToBufferRegion(op->args[1])));
      return;
    }
    if (op->op.same_as(mma_sunmmio())) {
      ICHECK_GE(op->args.size(), 6U);
      AddRead(RelaxBufferRegion(NormalizeToBufferRegion(op->args[0])));
      AddRead(RelaxBufferRegion(NormalizeToBufferRegion(op->args[1])));
      BufferRegion accum = RelaxBufferRegion(NormalizeToBufferRegion(op->args[2]));
      AddWrite(accum);
      arith::Analyzer analyzer;
      if (!analyzer.CanProveEqual(op->args[5], Bool(true))) {
        AddRead(accum);
      }
      return;
    }
    if (op->op.same_as(RegionOp::Get())) {
      // tl.region is metadata carried by intrinsics, not an actual read.
      return;
    }
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      if (const auto *buffer_var = op->args[1].as<VarNode>()) {
        auto it =
            buffer_data_to_buffer_.find(tvm::ffi::GetRef<Var>(buffer_var));
        if (it != buffer_data_to_buffer_.end()) {
          BufferRegion full = RelaxBufferRegion(BufferRegion::FullRegion((*it).second));
          int rw_mask = 1;
          if (const auto *mask_imm = op->args[4].as<IntImmNode>()) {
            rw_mask = static_cast<int>(mask_imm->value);
          }
          if (rw_mask & 1) {
            AddRead(full);
          }
          if (rw_mask & 2) {
            AddWrite(full);
          }
          return;
        }
      }
      if (record_unknown_stmt_calls_) {
        AddUnknownEffect(String(CallOpName(op) + " effect=opaque"));
      }
    }
    if (record_unknown_stmt_calls_) {
      CallEffectKind effect_kind = GetCallEffectKind(op);
      if (effect_kind > CallEffectKind::kReadState) {
        AddUnknownEffect(String(CallOpName(op) + " effect=" +
                                EffectKindToString(effect_kind)));
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Var, arith::IntSet> loop_var_domains_;
  Map<Var, PrimExpr> let_bindings_;
  bool record_unknown_stmt_calls_{false};
  Array<BufferRegion> reads_;
  Array<BufferRegion> writes_;
  Array<String> unknown_effects_;
};

struct UnknownEffectInfo {
  String detail;
};

struct StatementAccessInfo {
  int index{0};
  Stmt stmt;
  Array<BufferRegion> reads;
  Array<BufferRegion> writes;
  std::vector<UnknownEffectInfo> unknown_effects;
};

struct DependenceEdge {
  int src_index{0};
  int dst_index{0};
  int distance{0};
  BufferRegion write_region;
  BufferRegion read_region;
};

struct ReachingDef {
  int stmt_index{0};
  int distance{0};
  BufferRegion region;
};

struct AnalysisSummary {
  std::unordered_map<const BufferNode *, Buffer> written_internal_buffers;
  std::unordered_map<const BufferNode *, std::vector<BufferRegion>>
      state_regions_by_buffer;
  std::unordered_map<const BufferNode *, std::vector<BufferRegion>>
      write_regions_by_buffer;
  std::vector<BufferDependencyEdge> edges;
  std::set<std::string> edge_keys;
  std::vector<BufferDependencyPattern> patterns;
  std::set<std::string> pattern_keys;
};

struct PendingPartialOverwrite {
  ReachingDef carried_def;
  BufferRegion overwrite_region;
  int overwrite_op{-1};
};

struct AnalysisContext {
  Map<Var, arith::IntSet> loop_var_domains;
  Map<Var, PrimExpr> let_bindings;
};

class AnalyzePipelinedBufferDependencyRewriter : public StmtExprMutator {
public:
  static PrimFunc Run(PrimFunc f) {
    auto rewriter = AnalyzePipelinedBufferDependencyRewriter();
    for (const auto &[_, buffer] : f->buffer_map) {
      rewriter.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

private:
  BufferRegion ShiftRegion(const BufferRegion &region, const Var &loop_var,
                           int delta, arith::Analyzer *analyzer) const {
    if (delta == 0) {
      return region;
    }
    Map<Var, PrimExpr> subs;
    subs.Set(loop_var, loop_var + make_const(loop_var.dtype(), delta));
    Array<Range> shifted_region;
    for (const Range &range : region->region) {
      shifted_region.push_back(Range::FromMinExtent(
          analyzer->Simplify(tir::Substitute(range->min, subs)),
          analyzer->Simplify(tir::Substitute(range->extent, subs))));
    }
    return BufferRegion(region->buffer, shifted_region);
  }

  bool Overlaps(const BufferRegion &lhs, const BufferRegion &rhs,
                arith::Analyzer *analyzer) const {
    if (!lhs->buffer.same_as(rhs->buffer)) {
      return false;
    }
    return MayConflict(lhs->region, rhs->region);
  }

  std::optional<StatementAccessInfo> CollectStatementAccess(const Stmt &stmt,
                                                            int index,
                                                            const AnalysisContext &ctx) const {
    BufferAccessCollector collector(buffer_data_to_buffer_, ctx.loop_var_domains,
                                    ctx.let_bindings,
                                    stmt.as<EvaluateNode>() != nullptr);
    collector(stmt);
    if (collector.GetReads().empty() && collector.GetWrites().empty() &&
        collector.GetUnknownEffects().empty()) {
      return std::nullopt;
    }
    StatementAccessInfo info;
    info.index = index;
    info.stmt = stmt;
    info.reads = collector.GetReads();
    info.writes = collector.GetWrites();
    for (const String &detail : collector.GetUnknownEffects()) {
      info.unknown_effects.push_back(UnknownEffectInfo{detail});
    }
    return info;
  }

  bool SameRegion(const BufferRegion &lhs, const BufferRegion &rhs) const {
    StructuralEqual equal;
    return lhs->buffer.same_as(rhs->buffer) && equal(lhs->region, rhs->region);
  }

  bool SameRegionEquivalent(const BufferRegion &lhs,
                            const BufferRegion &rhs) const {
    return SameRegion(lhs, rhs) ||
           (lhs->buffer.same_as(rhs->buffer) &&
            RegionToString(lhs) == RegionToString(rhs));
  }

  bool RangeContains(const Range &cover, const Range &target,
                     arith::Analyzer *analyzer) const {
    PrimExpr cover_min = analyzer->Simplify(cover->min);
    PrimExpr target_min = analyzer->Simplify(target->min);
    PrimExpr cover_max = analyzer->Simplify(cover->min + cover->extent);
    PrimExpr target_max = analyzer->Simplify(target->min + target->extent);
    return analyzer->CanProve(cover_min <= target_min) &&
           analyzer->CanProve(target_max <= cover_max);
  }

  bool RegionContains(const BufferRegion &cover, const BufferRegion &target,
                      arith::Analyzer *analyzer) const {
    if (!cover->buffer.same_as(target->buffer) ||
        cover->region.size() != target->region.size()) {
      return false;
    }
    for (size_t i = 0; i < cover->region.size(); ++i) {
      if (!RangeContains(cover->region[i], target->region[i], analyzer)) {
        return false;
      }
    }
    return true;
  }

  void AddUniqueDef(std::vector<ReachingDef> *defs, const ReachingDef &def) const {
    for (const ReachingDef &existing : *defs) {
      if (existing.stmt_index == def.stmt_index &&
          existing.distance == def.distance &&
          SameRegion(existing.region, def.region)) {
        return;
      }
    }
    defs->push_back(def);
  }

  void AddUniqueRegion(std::vector<BufferRegion> *regions,
                       const BufferRegion &region) const {
    StructuralEqual equal;
    for (const BufferRegion &existing : *regions) {
      if (existing->buffer.same_as(region->buffer) &&
          equal(existing->region, region->region)) {
        return;
      }
    }
    regions->push_back(region);
  }

  void AddUniqueRegionToMap(
      std::unordered_map<const BufferNode *, std::vector<BufferRegion>> *map,
      const BufferRegion &region) const {
    AddUniqueRegion(&(*map)[region->buffer.get()], region);
  }

  void AddUniqueEdge(AnalysisSummary *summary,
                     const BufferDependencyEdge &edge) const {
    std::ostringstream os;
    os << edge->dep_kind << "|" << edge->distance->value << "|"
       << RegionToString(edge->src_region) << "|"
       << RegionToString(edge->dst_region) << "|";
    if (edge->src_effect_id) {
      os << edge->src_effect_id.value()->value;
    }
    os << "|";
    if (edge->dst_effect_id) {
      os << edge->dst_effect_id.value()->value;
    }
    if (summary->edge_keys.insert(os.str()).second) {
      summary->edges.push_back(edge);
    }
  }

  void AddUniquePattern(AnalysisSummary *summary,
                        const BufferDependencyPattern &pattern) const {
    std::ostringstream os;
    os << pattern->kind << "|";
    if (pattern->buffer.defined()) {
      os << pattern->buffer->name;
    } else {
      os << "<none>";
    }
    for (const BufferRegion &region : pattern->regions) {
      os << "|" << RegionToString(region);
    }
    for (const Integer &effect_id : pattern->effect_ids) {
      os << "|" << effect_id->value;
    }
    if (pattern->detail.size() != 0) {
      os << "|" << pattern->detail;
    }
    if (summary->pattern_keys.insert(os.str()).second) {
      summary->patterns.push_back(pattern);
    }
  }

  BufferDependencyEdge MakeEdge(const DependenceEdge &edge) const {
    return BufferDependencyEdge(
        String(buffer_dependency::kDepKindRAW), Integer(edge.distance),
        edge.write_region->buffer, edge.write_region, edge.read_region,
        Integer(edge.src_index), Integer(edge.dst_index));
  }

  BufferDependencyPattern MakePattern(String kind, Buffer buffer,
                                      Array<BufferRegion> regions,
                                      Array<Integer> effect_ids = {},
                                      String detail = "") const {
    return BufferDependencyPattern(std::move(kind), std::move(buffer),
                                   std::move(regions), std::move(effect_ids),
                                   std::move(detail));
  }

  std::string FormatEdge(const BufferDependencyEdge &edge) const {
    std::ostringstream os;
    os << edge->buffer->name << ": d=" << edge->distance->value;
    if (edge->src_effect_id) {
      os << " writer_op=" << edge->src_effect_id.value()->value;
    }
    if (edge->dst_effect_id) {
      os << " reader_op=" << edge->dst_effect_id.value()->value;
    }
    os << " write=" << RegionToString(edge->src_region)
       << " read=" << RegionToString(edge->dst_region);
    return os.str();
  }

  std::string FormatPattern(const BufferDependencyPattern &pattern) const {
    std::ostringstream os;
    if (pattern->kind == buffer_dependency::kPatternUnknownEffect) {
      if (pattern->effect_ids.size() >= 1) {
        os << "op_id=" << pattern->effect_ids[0]->value << " ";
      }
      os << pattern->detail;
      return os.str();
    }
    if (pattern->kind == buffer_dependency::kPatternCoveredRewrite) {
      os << pattern->buffer->name << ": carried_d=1";
      if (pattern->effect_ids.size() >= 1) {
        os << " writer_op=" << pattern->effect_ids[0]->value;
      }
      if (pattern->effect_ids.size() >= 2) {
        os << " overwrite_op=" << pattern->effect_ids[1]->value;
      }
      if (pattern->regions.size() >= 1) {
        os << " carried=" << RegionToString(pattern->regions[0]);
      }
      if (pattern->regions.size() >= 2) {
        os << " cover=" << RegionToString(pattern->regions[1]);
      }
      os << " relation=covered";
      return os.str();
    }
    if (pattern->kind == buffer_dependency::kPatternPartialOverwriteRemainderRead) {
      os << pattern->buffer->name << ": carried_d=1";
      if (pattern->effect_ids.size() >= 1) {
        os << " writer_op=" << pattern->effect_ids[0]->value;
      }
      if (pattern->effect_ids.size() >= 2) {
        os << " overwrite_op=" << pattern->effect_ids[1]->value;
      }
      if (pattern->effect_ids.size() >= 3) {
        os << " read_op=" << pattern->effect_ids[2]->value;
      }
      if (pattern->regions.size() >= 1) {
        os << " carried=" << RegionToString(pattern->regions[0]);
      }
      if (pattern->regions.size() >= 2) {
        os << " overwrite=" << RegionToString(pattern->regions[1]);
      }
      if (pattern->regions.size() >= 3) {
        os << " read=" << RegionToString(pattern->regions[2]);
      }
      return os.str();
    }
    if (pattern->kind == buffer_dependency::kPatternMixedRoleRegions) {
      os << pattern->buffer->name;
      if (pattern->regions.size() >= 1) {
        os << ": state_region=" << RegionToString(pattern->regions[0]);
      }
      if (pattern->regions.size() >= 2) {
        os << " local_region=" << RegionToString(pattern->regions[1]);
      }
      os << " relation=disjoint";
      return os.str();
    }
    if (pattern->buffer.defined()) {
      os << pattern->buffer->name << ": kind=" << pattern->kind;
    } else {
      os << "kind=" << pattern->kind;
    }
    return os.str();
  }

  void MaterializePartialOverwriteHazards(
      const std::vector<PendingPartialOverwrite> &pending_hazards,
      const BufferRegion &read_region, int read_op, AnalysisSummary *summary,
      arith::Analyzer *analyzer) const {
    for (const PendingPartialOverwrite &hazard : pending_hazards) {
      if (!hazard.carried_def.region->buffer.same_as(read_region->buffer)) {
        continue;
      }
      if (!Overlaps(hazard.carried_def.region, read_region, analyzer)) {
        continue;
      }
      if (Overlaps(hazard.overwrite_region, read_region, analyzer)) {
        continue;
      }
      // `partial_overwrite_remainder_read` is only emitted once we have the
      // full witness:
      //   1. a loop-carried reaching definition from the previous iteration,
      //   2. a current-iteration write that overlaps only part of that region,
      //   3. a later read that touches the carried region but not the
      //      overwritten part.
      //
      // This avoids treating every partial overlap as unsupported. The pattern
      // is only recorded when some untouched remainder is actually observed by
      // a later read.
      AddUniquePattern(
          summary,
          MakePattern(
              String(buffer_dependency::kPatternPartialOverwriteRemainderRead),
              read_region->buffer,
              {hazard.carried_def.region, hazard.overwrite_region, read_region},
              {Integer(hazard.carried_def.stmt_index),
               Integer(hazard.overwrite_op), Integer(read_op)}));
    }
  }

  void KillOverlappingDefs(std::vector<ReachingDef> *defs,
                           const BufferRegion &write_region,
                           arith::Analyzer *analyzer,
                           AnalysisSummary *summary = nullptr,
                           int overwrite_op = -1,
                           std::vector<PendingPartialOverwrite> *pending_hazards =
                               nullptr) const {
    std::vector<ReachingDef> kept;
    kept.reserve(defs->size());
    for (const ReachingDef &def : *defs) {
      if (!Overlaps(def.region, write_region, analyzer)) {
        kept.push_back(def);
        continue;
      }
      if (summary != nullptr && pending_hazards != nullptr && def.distance > 0) {
        // `covered_rewrite` means the current write fully contains the
        // loop-carried reaching definition from the previous iteration.
        //
        // Example:
        //   carried def at loop header:  acc[0:4]
        //   current nested-loop write:   acc[i] for i in [0, 4)
        // After relaxing the leaf write through the active inner loop domains,
        // `write_region` becomes `acc[0:4]`, so we can prove that the carried
        // value is completely rewritten in the current iteration.
        //
        // This is intentionally different from
        // `partial_overwrite_remainder_read`: if we can only prove partial
        // overlap, we defer the hazard until a later read actually observes
        // the untouched remainder of the carried region.
        if (RegionContains(write_region, def.region, analyzer)) {
          AddUniquePattern(
              summary,
              MakePattern(String(buffer_dependency::kPatternCoveredRewrite),
                          write_region->buffer,
                          {def.region, write_region},
                          {Integer(def.stmt_index), Integer(overwrite_op)}));
        } else if (!SameRegionEquivalent(def.region, write_region)) {
          pending_hazards->push_back(
              PendingPartialOverwrite{def, write_region, overwrite_op});
        }
      }
    }
    *defs = std::move(kept);
  }

  void ApplyStatementWrites(std::vector<ReachingDef> *defs,
                            const StatementAccessInfo &info,
                            arith::Analyzer *analyzer,
                            AnalysisSummary *summary = nullptr,
                            std::vector<PendingPartialOverwrite> *pending_hazards =
                                nullptr) const {
    for (const BufferRegion &write_region : info.writes) {
      KillOverlappingDefs(defs, write_region, analyzer, summary, info.index,
                          pending_hazards);
    }
    for (const BufferRegion &write_region : info.writes) {
      AddUniqueDef(defs, ReachingDef{info.index, 0, write_region});
    }
  }

  void MergeDefs(std::vector<ReachingDef> *dst,
                 const std::vector<ReachingDef> &src) const {
    for (const ReachingDef &def : src) {
      AddUniqueDef(dst, def);
    }
  }

  void WithScopedBuffers(const Array<Buffer> &buffers,
                         const std::function<void()> &body) {
    for (const Buffer &buffer : buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    body();
    for (const Buffer &buffer : buffers) {
      buffer_data_to_buffer_.erase(buffer->data);
    }
  }

  void ProcessLeafStmt(const Stmt &stmt, std::vector<ReachingDef> *defs,
                       int *effect_index, AnalysisSummary *summary,
                       arith::Analyzer *analyzer,
                       std::vector<PendingPartialOverwrite> *pending_hazards,
                       const AnalysisContext &ctx) {
    std::optional<StatementAccessInfo> maybe_info =
        CollectStatementAccess(stmt, *effect_index, ctx);
    if (!maybe_info.has_value()) {
      return;
    }
    const StatementAccessInfo &info = maybe_info.value();
    ++(*effect_index);
    ProcessAccessInfo(info, defs, summary, analyzer, pending_hazards);
  }

  std::optional<StatementAccessInfo> CollectExpressionAccess(
      const PrimExpr &expr, int index, const AnalysisContext &ctx) const {
    BufferAccessCollector collector(buffer_data_to_buffer_, ctx.loop_var_domains,
                                    ctx.let_bindings);
    collector(expr);
    if (collector.GetReads().empty() && collector.GetWrites().empty() &&
        collector.GetUnknownEffects().empty()) {
      return std::nullopt;
    }
    StatementAccessInfo info;
    info.index = index;
    info.reads = collector.GetReads();
    info.writes = collector.GetWrites();
    for (const String &detail : collector.GetUnknownEffects()) {
      info.unknown_effects.push_back(UnknownEffectInfo{detail});
    }
    return info;
  }

  void ProcessLeafExpr(const PrimExpr &expr, std::vector<ReachingDef> *defs,
                       int *effect_index, AnalysisSummary *summary,
                       arith::Analyzer *analyzer,
                       std::vector<PendingPartialOverwrite> *pending_hazards,
                       const AnalysisContext &ctx) {
    if (!expr.defined()) {
      return;
    }
    std::optional<StatementAccessInfo> maybe_info =
        CollectExpressionAccess(expr, *effect_index, ctx);
    if (!maybe_info.has_value()) {
      return;
    }
    ++(*effect_index);
    ProcessAccessInfo(maybe_info.value(), defs, summary, analyzer,
                      pending_hazards);
  }

  void ProcessExprArray(const Array<PrimExpr> &exprs,
                        std::vector<ReachingDef> *defs, int *effect_index,
                        AnalysisSummary *summary, arith::Analyzer *analyzer,
                        std::vector<PendingPartialOverwrite> *pending_hazards,
                        const AnalysisContext &ctx) {
    for (const PrimExpr &expr : exprs) {
      ProcessLeafExpr(expr, defs, effect_index, summary, analyzer,
                      pending_hazards, ctx);
    }
  }

  void ProcessAccessInfo(const StatementAccessInfo &info,
                         std::vector<ReachingDef> *defs,
                         AnalysisSummary *summary,
                         arith::Analyzer *analyzer,
                         std::vector<PendingPartialOverwrite> *pending_hazards) {
    if (summary != nullptr) {
      for (const UnknownEffectInfo &unknown_effect : info.unknown_effects) {
        AddUniquePattern(
            summary,
            MakePattern(String(buffer_dependency::kPatternUnknownEffect),
                        Buffer(), {}, {Integer(info.index)},
                        unknown_effect.detail));
      }
      for (const BufferRegion &read_region : info.reads) {
        MaterializePartialOverwriteHazards(*pending_hazards, read_region,
                                           info.index, summary, analyzer);
        for (const DependenceEdge &edge :
             FindReachingDefs(*defs, read_region, info.index, analyzer)) {
          BufferDependencyEdge dep_edge = MakeEdge(edge);
          AddUniqueEdge(summary, dep_edge);
          if (edge.distance > 0 && !IsGlobalBuffer(edge.write_region->buffer)) {
            AddUniqueRegionToMap(&summary->state_regions_by_buffer,
                                 edge.write_region);
          }
        }
      }
      for (const BufferRegion &write_region : info.writes) {
        if (!IsGlobalBuffer(write_region->buffer)) {
          summary->written_internal_buffers[write_region->buffer.get()] =
              write_region->buffer;
          AddUniqueRegionToMap(&summary->write_regions_by_buffer, write_region);
        }
      }
    }
    ApplyStatementWrites(defs, info, analyzer, summary, pending_hazards);
  }

  Array<Buffer> CollectScopedBuffers(const BlockNode *block) const {
    Array<Buffer> scoped = block->alloc_buffers;
    for (const MatchBufferRegion &match_buffer : block->match_buffers) {
      scoped.push_back(match_buffer->buffer);
    }
    return scoped;
  }

  void AnalyzeStmt(const Stmt &stmt, std::vector<ReachingDef> *defs,
                   int *effect_index, AnalysisSummary *summary,
                   arith::Analyzer *analyzer,
                   std::vector<PendingPartialOverwrite> *pending_hazards,
                   AnalysisContext *ctx) {
    if (!stmt.defined()) {
      return;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &child : seq->seq) {
        AnalyzeStmt(child, defs, effect_index, summary, analyzer,
                    pending_hazards, ctx);
      }
      return;
    }
    if (const auto *block_realize = stmt.as<BlockRealizeNode>()) {
      std::vector<Var> bound_vars;
      for (size_t i = 0; i < block_realize->block->iter_vars.size(); ++i) {
        Var iter_var = block_realize->block->iter_vars[i]->var;
        ctx->let_bindings.Set(iter_var, block_realize->iter_values[i]);
        bound_vars.push_back(iter_var);
      }
      ProcessLeafExpr(block_realize->predicate, defs, effect_index, summary,
                      analyzer, pending_hazards, *ctx);
      AnalyzeStmt(block_realize->block, defs, effect_index, summary, analyzer,
                  pending_hazards, ctx);
      for (const Var &iter_var : bound_vars) {
        ctx->let_bindings.erase(iter_var);
      }
      return;
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      WithScopedBuffers(CollectScopedBuffers(block), [&]() {
        if (block->init) {
          AnalyzeStmt(block->init.value(), defs, effect_index, summary, analyzer,
                      pending_hazards, ctx);
        }
        AnalyzeStmt(block->body, defs, effect_index, summary, analyzer,
                    pending_hazards, ctx);
      });
      return;
    }
    if (const auto *buffer_realize = stmt.as<BufferRealizeNode>()) {
      ProcessLeafExpr(buffer_realize->condition, defs, effect_index, summary,
                      analyzer, pending_hazards, *ctx);
      WithScopedBuffers({buffer_realize->buffer}, [&]() {
        AnalyzeStmt(buffer_realize->body, defs, effect_index, summary, analyzer,
                    pending_hazards, ctx);
      });
      return;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      ProcessLeafExpr(attr->value, defs, effect_index, summary, analyzer,
                      pending_hazards, *ctx);
      AnalyzeStmt(attr->body, defs, effect_index, summary, analyzer,
                  pending_hazards, ctx);
      return;
    }
    if (const auto *assert_stmt = stmt.as<AssertStmtNode>()) {
      ProcessLeafExpr(assert_stmt->condition, defs, effect_index, summary,
                      analyzer, pending_hazards, *ctx);
      ProcessLeafExpr(assert_stmt->message, defs, effect_index, summary,
                      analyzer, pending_hazards, *ctx);
      AnalyzeStmt(assert_stmt->body, defs, effect_index, summary, analyzer,
                  pending_hazards, ctx);
      return;
    }
    if (const auto *let_stmt = stmt.as<LetStmtNode>()) {
      ProcessLeafExpr(let_stmt->value, defs, effect_index, summary, analyzer,
                      pending_hazards, *ctx);
      ctx->let_bindings.Set(let_stmt->var, let_stmt->value);
      AnalyzeStmt(let_stmt->body, defs, effect_index, summary, analyzer,
                  pending_hazards, ctx);
      ctx->let_bindings.erase(let_stmt->var);
      return;
    }
    if (const auto *allocate = stmt.as<AllocateNode>()) {
      ProcessExprArray(allocate->extents, defs, effect_index, summary, analyzer,
                       pending_hazards, *ctx);
      ProcessLeafExpr(allocate->condition, defs, effect_index, summary,
                      analyzer, pending_hazards, *ctx);
      AnalyzeStmt(allocate->body, defs, effect_index, summary, analyzer,
                  pending_hazards, ctx);
      return;
    }
    if (const auto *allocate_const = stmt.as<AllocateConstNode>()) {
      ProcessExprArray(allocate_const->extents, defs, effect_index, summary,
                       analyzer, pending_hazards, *ctx);
      AnalyzeStmt(allocate_const->body, defs, effect_index, summary, analyzer,
                  pending_hazards, ctx);
      return;
    }
    if (const auto *decl_buffer = stmt.as<DeclBufferNode>()) {
      WithScopedBuffers({decl_buffer->buffer}, [&]() {
        AnalyzeStmt(decl_buffer->body, defs, effect_index, summary, analyzer,
                    pending_hazards, ctx);
      });
      return;
    }
    if (const auto *for_node = stmt.as<ForNode>()) {
      ProcessLeafExpr(for_node->min, defs, effect_index, summary, analyzer,
                      pending_hazards, *ctx);
      ProcessLeafExpr(for_node->extent, defs, effect_index, summary, analyzer,
                      pending_hazards, *ctx);
      if (for_node->step.defined()) {
        ProcessLeafExpr(for_node->step.value(), defs, effect_index, summary,
                        analyzer, pending_hazards, *ctx);
      }
      ctx->loop_var_domains.Set(
          for_node->loop_var,
          arith::IntSet::FromRange(
              Range::FromMinExtent(for_node->min, for_node->extent)));
      AnalyzeStmt(for_node->body, defs, effect_index, summary, analyzer,
                  pending_hazards, ctx);
      ctx->loop_var_domains.erase(for_node->loop_var);
      return;
    }
    if (const auto *while_node = stmt.as<WhileNode>()) {
      // Best-effort handling for dynamic loops: record reads in the entry
      // condition, analyze one body iteration, then re-evaluate the condition
      // on the body-exit defs before merging with the zero-iteration path.
      ProcessLeafExpr(while_node->condition, defs, effect_index, summary,
                      analyzer, pending_hazards, *ctx);
      std::vector<ReachingDef> incoming_defs = *defs;
      std::vector<ReachingDef> body_defs = incoming_defs;
      AnalyzeStmt(while_node->body, &body_defs, effect_index, summary, analyzer,
                  pending_hazards, ctx);
      ProcessLeafExpr(while_node->condition, &body_defs, effect_index, summary,
                      analyzer, pending_hazards, *ctx);
      *defs = std::move(incoming_defs);
      MergeDefs(defs, body_defs);
      return;
    }
    if (const auto *if_then_else = stmt.as<IfThenElseNode>()) {
      ProcessLeafExpr(if_then_else->condition, defs, effect_index, summary,
                      analyzer, pending_hazards, *ctx);
      std::vector<ReachingDef> incoming_defs = *defs;
      std::vector<ReachingDef> then_defs = incoming_defs;
      AnalyzeStmt(if_then_else->then_case, &then_defs, effect_index, summary,
                  analyzer, pending_hazards, ctx);
      if (if_then_else->else_case.defined()) {
        std::vector<ReachingDef> else_defs = incoming_defs;
        AnalyzeStmt(if_then_else->else_case.value(), &else_defs, effect_index,
                    summary, analyzer, pending_hazards, ctx);
        *defs = std::move(then_defs);
        MergeDefs(defs, else_defs);
      } else {
        *defs = std::move(incoming_defs);
        MergeDefs(defs, then_defs);
      }
      return;
    }
    ProcessLeafStmt(stmt, defs, effect_index, summary, analyzer,
                    pending_hazards, *ctx);
  }

  std::vector<ReachingDef>
  ComputeLoopExitDefs(const Stmt &pipeline_body_root) {
    std::vector<ReachingDef> defs;
    arith::Analyzer analyzer;
    int effect_index = 0;
    std::vector<PendingPartialOverwrite> pending_hazards;
    AnalysisContext ctx;
    AnalyzeStmt(pipeline_body_root, &defs, &effect_index, nullptr, &analyzer,
                &pending_hazards, &ctx);
    return defs;
  }

  void FinalizePatterns(AnalysisSummary *summary,
                        arith::Analyzer *analyzer) const {
    for (const auto &[buffer_node, state_regions] :
         summary->state_regions_by_buffer) {
      auto write_it = summary->write_regions_by_buffer.find(buffer_node);
      if (write_it == summary->write_regions_by_buffer.end()) {
        continue;
      }
      for (const BufferRegion &state_region : state_regions) {
        for (const BufferRegion &write_region : write_it->second) {
          if (SameRegionEquivalent(state_region, write_region)) {
            continue;
          }
          // `mixed_role_regions` captures a buffer-level partition:
          // one region participates in loop-carried RAW dependence, while a
          // different non-overlapping written region remains iteration-local.
          //
          // Example:
          //   mix[1] is read across iterations  -> state region
          //   mix[0] is only freshly written    -> channel region
          //
          // This is a structural fact about the buffer's regions. Downstream
          // passes can decide whether they support such mixed-role buffers.
          if (!Overlaps(state_region, write_region, analyzer)) {
            AddUniquePattern(
                summary,
                MakePattern(String(buffer_dependency::kPatternMixedRoleRegions),
                            write_region->buffer,
                            {state_region, write_region}));
          }
        }
      }
    }
  }

  std::vector<ReachingDef>
  BuildLoopHeaderDefs(const std::vector<ReachingDef> &exit_defs,
                      const Var &loop_var) const {
    std::vector<ReachingDef> header_defs;
    header_defs.reserve(exit_defs.size());
    arith::Analyzer analyzer;
    for (const ReachingDef &def : exit_defs) {
      AddUniqueDef(&header_defs,
                   ReachingDef{def.stmt_index,
                               1,
                               ShiftRegion(def.region, loop_var, -1, &analyzer)});
    }
    return header_defs;
  }

  std::vector<DependenceEdge>
  FindReachingDefs(const std::vector<ReachingDef> &defs,
                   const BufferRegion &read_region, int dst_index,
                   arith::Analyzer *analyzer) const {
    std::vector<DependenceEdge> edges;
    for (const ReachingDef &def : defs) {
      if (Overlaps(def.region, read_region, analyzer)) {
        edges.push_back(DependenceEdge{def.stmt_index, dst_index, def.distance,
                                       def.region, read_region});
      }
    }
    return edges;
  }

  Array<BufferRegion> BuildChannelRegions(
      const std::vector<BufferRegion> &write_regions,
      const std::vector<BufferRegion> &state_regions,
      arith::Analyzer *analyzer) const {
    Array<BufferRegion> channel_regions;
    for (const BufferRegion &write_region : write_regions) {
      bool is_state = false;
      for (const BufferRegion &state_region : state_regions) {
        if (SameRegionEquivalent(write_region, state_region) ||
            RegionContains(state_region, write_region, analyzer)) {
          is_state = true;
          break;
        }
      }
      if (!is_state) {
        channel_regions.push_back(write_region);
      }
    }
    return channel_regions;
  }

  BufferDependencyAnalysis BuildAnalysisResult(
      const AnalysisSummary &summary, arith::Analyzer *analyzer) const {
    Array<BufferDependencyInfo> infos;
    for (const auto &[buffer_node, buffer] : summary.written_internal_buffers) {
      Array<BufferRegion> state_regions;
      if (auto it = summary.state_regions_by_buffer.find(buffer_node);
          it != summary.state_regions_by_buffer.end()) {
        for (const BufferRegion &region : it->second) {
          state_regions.push_back(region);
        }
      }
      Array<BufferRegion> channel_regions;
      if (auto it = summary.write_regions_by_buffer.find(buffer_node);
          it != summary.write_regions_by_buffer.end()) {
        const std::vector<BufferRegion> empty_regions;
        const std::vector<BufferRegion> &buffer_state_regions =
            summary.state_regions_by_buffer.count(buffer_node)
                ? summary.state_regions_by_buffer.at(buffer_node)
                : empty_regions;
        channel_regions =
            BuildChannelRegions(it->second, buffer_state_regions, analyzer);
      }
      infos.push_back(
          BufferDependencyInfo(buffer, std::move(state_regions),
                               std::move(channel_regions)));
    }

    Array<BufferDependencyEdge> edges;
    for (const BufferDependencyEdge &edge : summary.edges) {
      edges.push_back(edge);
    }
    Array<BufferDependencyPattern> patterns;
    for (const BufferDependencyPattern &pattern : summary.patterns) {
      patterns.push_back(pattern);
    }
    return BufferDependencyAnalysis(std::move(infos), std::move(edges),
                                    std::move(patterns));
  }

  Array<PrimExpr> DebugStateBuffers(const BufferDependencyAnalysis &analysis) const {
    std::set<std::string> buffers;
    for (const BufferDependencyInfo &info : analysis->buffers) {
      if (!info->state_regions.empty()) {
        buffers.insert(info->buffer->name);
      }
    }
    return ToStringImmArray(buffers);
  }

  Array<PrimExpr> DebugChannelBuffers(const BufferDependencyAnalysis &analysis) const {
    std::set<std::string> buffers;
    for (const BufferDependencyInfo &info : analysis->buffers) {
      if (info->state_regions.empty() && !info->channel_regions.empty()) {
        buffers.insert(info->buffer->name);
      }
    }
    return ToStringImmArray(buffers);
  }

  Array<PrimExpr> DebugEdges(const BufferDependencyAnalysis &analysis,
                             int distance) const {
    std::set<std::string> edges;
    for (const BufferDependencyEdge &edge : analysis->edges) {
      if (edge->dep_kind == buffer_dependency::kDepKindRAW &&
          edge->distance->value == distance) {
        edges.insert(FormatEdge(edge));
      }
    }
    return ToStringImmArray(edges);
  }

  Array<PrimExpr> DebugPatternBuffers(const BufferDependencyAnalysis &analysis,
                                      const String &kind) const {
    std::set<std::string> buffers;
    for (const BufferDependencyPattern &pattern : analysis->patterns) {
      if (pattern->kind == kind && pattern->buffer.defined()) {
        buffers.insert(pattern->buffer->name);
      }
    }
    return ToStringImmArray(buffers);
  }

  Array<PrimExpr> DebugPatternDetails(const BufferDependencyAnalysis &analysis,
                                      const String &kind) const {
    std::set<std::string> details;
    for (const BufferDependencyPattern &pattern : analysis->patterns) {
      if (pattern->kind == kind) {
        details.insert(FormatPattern(pattern));
      }
    }
    return ToStringImmArray(details);
  }

  Map<String, Any> BuildAnalysisAnnotations(const ForNode *op) {
    Stmt pipeline_body_root = op->body;
    Array<Buffer> scoped_buffers;
    if (const auto *realize = op->body.as<BlockRealizeNode>()) {
      scoped_buffers = CollectScopedBuffers(realize->block.get());
    } else if (const auto *block = op->body.as<BlockNode>()) {
      scoped_buffers = CollectScopedBuffers(block);
    }

    AnalysisSummary summary;
    for (const Buffer &buffer : scoped_buffers) {
      if (buffer.defined() && !IsGlobalBuffer(buffer)) {
        summary.written_internal_buffers[buffer.get()] = buffer;
      }
    }

    std::vector<ReachingDef> current_defs =
        BuildLoopHeaderDefs(ComputeLoopExitDefs(pipeline_body_root),
                            op->loop_var);
    arith::Analyzer analyzer;
    int effect_index = 0;
    std::vector<PendingPartialOverwrite> pending_hazards;
    AnalysisContext ctx;
    AnalyzeStmt(pipeline_body_root, &current_defs, &effect_index, &summary,
                &analyzer, &pending_hazards, &ctx);
    FinalizePatterns(&summary, &analyzer);
    BufferDependencyAnalysis analysis =
        BuildAnalysisResult(summary, &analyzer);

    Map<String, Any> annotations;
    for (const auto &[key, value] : op->annotations) {
      annotations.Set(key, value);
    }
    annotations.Set(attr::kBufferDependencyAnalysis, analysis);
    annotations.Set(kStateBuffersAttr, DebugStateBuffers(analysis));
    annotations.Set(kChannelBuffersAttr, DebugChannelBuffers(analysis));
    annotations.Set(kIntraRawEdgesAttr, DebugEdges(analysis, 0));
    annotations.Set(kInterRawEdgesAttr, DebugEdges(analysis, 1));
    annotations.Set(
        kMixedRoleBuffersAttr,
        DebugPatternBuffers(analysis,
                            String(buffer_dependency::kPatternMixedRoleRegions)));
    annotations.Set(
        kMixedRoleDetailsAttr,
        DebugPatternDetails(
            analysis, String(buffer_dependency::kPatternMixedRoleRegions)));
    annotations.Set(
        kPartialOverwriteHazardsAttr,
        DebugPatternDetails(
            analysis,
            String(buffer_dependency::kPatternPartialOverwriteRemainderRead)));
    annotations.Set(
        kCoveredRewritesAttr,
        DebugPatternDetails(analysis,
                            String(buffer_dependency::kPatternCoveredRewrite)));
    annotations.Set(
        kUnknownEffectsAttr,
        DebugPatternDetails(analysis,
                            String(buffer_dependency::kPatternUnknownEffect)));
    return annotations;
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    for (const Buffer &buffer : CollectScopedBuffers(op)) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    for (const Buffer &buffer : CollectScopedBuffers(op)) {
      buffer_data_to_buffer_.erase(buffer->data);
    }
    return stmt;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    std::vector<Buffer> pipeline_alloc_buffers;
    if (const auto *realize = op->body.as<BlockRealizeNode>()) {
      for (const Buffer &buffer : CollectScopedBuffers(realize->block.get())) {
        buffer_data_to_buffer_.Set(buffer->data, buffer);
        pipeline_alloc_buffers.push_back(buffer);
      }
    } else if (const auto *block = op->body.as<BlockNode>()) {
      for (const Buffer &buffer : CollectScopedBuffers(block)) {
        buffer_data_to_buffer_.Set(buffer->data, buffer);
        pipeline_alloc_buffers.push_back(buffer);
      }
    }

    For loop = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    auto num_stages_anno = loop->annotations.Get("num_stages");
    if (num_stages_anno && loop->kind == ForKind::kSerial) {
      for (const Buffer &buffer : pipeline_alloc_buffers) {
        buffer_data_to_buffer_.Set(buffer->data, buffer);
      }
      loop.CopyOnWrite()->annotations = BuildAnalysisAnnotations(loop.get());
    }

    for (const Buffer &buffer : pipeline_alloc_buffers) {
      buffer_data_to_buffer_.erase(buffer->data);
    }
    return loop;
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
};

} // namespace

using namespace tir::transform;

tvm::transform::Pass AnalyzePipelinedBufferDependency() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return AnalyzePipelinedBufferDependencyRewriter::Run(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AnalyzePipelinedBufferDependency", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnalyzePipelinedBufferDependency",
                        AnalyzePipelinedBufferDependency);
}

} // namespace tl
} // namespace tvm