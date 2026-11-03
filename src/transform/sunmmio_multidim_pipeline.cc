#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/module.h>
#include <tvm/ir/op.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../layout/cute_layout.h"
#include "../layout/utils.h"
#include "../op/region.h"
#include "sunmmio_pipeline_planning/boundary_overlap_analysis.h"
#include "sunmmio_pipeline_planning/grouping_snapshot.h"
#include "sunmmio_pipeline_planning/pipeline_diagnostic.h"
#include "sunmmio_pipeline_planning/tile_residency_analysis.h"
#include "tir/transforms/ir_utils.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;
using ffi::Array;

namespace {

constexpr const char *kMultidimRegion = "tl.sunmmio.multidim.region";
constexpr const char *kMultidimRank = "tl.sunmmio.multidim.rank";
constexpr const char *kMultidimOuterVar = "tl.sunmmio.multidim.outer_var";
constexpr const char *kMultidimInnerVar = "tl.sunmmio.multidim.inner_var";
constexpr const char *kMultidimOuterMin = "tl.sunmmio.multidim.outer_min";
constexpr const char *kMultidimOuterExtent = "tl.sunmmio.multidim.outer_extent";
constexpr const char *kMultidimInnerMin = "tl.sunmmio.multidim.inner_min";
constexpr const char *kMultidimInnerExtent = "tl.sunmmio.multidim.inner_extent";
constexpr const char *kMultidimInnerCoordinate =
    "tl.sunmmio.multidim.inner_coordinate";
constexpr const char *kMultidimOriginalLoop =
    "tl.sunmmio.multidim.original_loop";

constexpr const char *kMultidimSelectedMode =
    "tl.sunmmio.multidim.selected_mode";
constexpr const char *kMultidimFallbackStage =
    "tl.sunmmio.multidim.fallback_stage";
constexpr const char *kMultidimFallbackReason =
    "tl.sunmmio.multidim.fallback_reason";
// Analysis identities belong to the actual selected core, never to a Var
// referenced from an enclosing loop that injection may later remove/rename.
constexpr const char *kMultidimCoreDecisionId =
    "tl.sunmmio.multidim.core_decision_id";
constexpr const char *kMultidimCandidateId = "tl.sunmmio.multidim.candidate_id";
constexpr const char *kMultidimCandidateLoop =
    "tl.sunmmio.multidim.candidate_loop";
constexpr const char *kPipelineSuppressArtifacts =
    "tl.sunmmio.pipeline.suppress_artifacts";
constexpr const char *kMultidimPredictedScore =
    "tl.sunmmio.multidim.predicted_score";
constexpr const char *kOneDimPredictedScore =
    "tl.sunmmio.multidim.one_dim_predicted_score";
constexpr const char *kMultidimMinGainPercent =
    "tl.sunmmio.multidim.min_gain_percent";
constexpr const char *kMultidimStrategy = "tl.sunmmio.multidim.strategy";
constexpr const char *kHierarchicalStrategy = "hierarchical_v2";
constexpr const char *kHierarchicalInner =
    "tl.sunmmio.multidim.hierarchical_inner";
constexpr const char *kHierarchicalSummaryValid =
    "tl.sunmmio.multidim.inner_summary_valid";
constexpr const char *kHierarchicalInnerII = "tl.sunmmio.multidim.inner_ii";
constexpr const char *kHierarchicalInnerMakespan =
    "tl.sunmmio.multidim.inner_makespan";
constexpr const char *kHierarchicalInnerStageCount =
    "tl.sunmmio.multidim.inner_stage_count";
constexpr const char *kHierarchicalInnerPhase =
    "tl.sunmmio.multidim.inner_phase";
constexpr const char *kHierarchicalOuterPhase =
    "tl.sunmmio.multidim.outer_phase";
constexpr const char *kHierarchicalOuterBuilt =
    "tl.sunmmio.multidim.outer_pipeline_built";
constexpr const char *kHierarchicalPrefetchBeforePrefix =
    "tl.sunmmio.multidim.prefetch_before_prefix";
constexpr const char *kHierarchicalOuterVersions =
    "tl.sunmmio.multidim.outer_versions";
constexpr const char *kInterchangeStrategy = "interchange_reuse_v1";
constexpr const char *kInterchangeApplied =
    "tl.sunmmio.multidim.interchange.applied";
constexpr const char *kInterchangeSharedPrefixCount =
    "tl.sunmmio.multidim.interchange.shared_prefix_count";
constexpr const char *kInterchangeOuterVersions =
    "tl.sunmmio.multidim.interchange.outer_versions";
constexpr const char *kInterchangeKMin =
    "tl.sunmmio.multidim.interchange.k_min";
constexpr const char *kInterchangeKExtent =
    "tl.sunmmio.multidim.interchange.k_extent";
constexpr const char *kInterchangeAccumulatorVersions =
    "tl.sunmmio.multidim.interchange.accumulator_versions";

For MarkOneDimensionalFallback(const For &loop, const std::string &stage,
                               const std::string &reason) {
  Map<String, Any> annotations = loop->annotations;
  annotations.Set(kMultidimSelectedMode, String("ilp"));
  annotations.Set(kMultidimFallbackStage, String(stage));
  annotations.Set(kMultidimFallbackReason, String(reason));
  For fallback = loop;
  fallback.CopyOnWrite()->annotations = annotations;
  return fallback;
}

std::optional<int64_t> AsStaticInt(const PrimExpr &expr) {
  arith::Analyzer analyzer;
  PrimExpr simplified = analyzer.Simplify(expr);
  if (const auto *imm = simplified.as<IntImmNode>()) {
    return imm->value;
  }
  return std::nullopt;
}

bool HasNumStages(const ForNode *loop) {
  return loop->annotations.find("num_stages") != loop->annotations.end();
}

bool ContainsBarrier(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&](const ObjectRef &obj) {
    if (found) {
      return;
    }
    const auto *call = obj.as<CallNode>();
    if (call == nullptr) {
      return;
    }
    const auto *op = call->op.as<OpNode>();
    if (op != nullptr &&
        std::string(op->name).find("barrier") != std::string::npos) {
      found = true;
    }
  });
  return found;
}

bool InterchangeEnabled() {
  const char *raw = std::getenv("TL_SUNMMIO_MULTIDIM_INTERCHANGE");
  return raw != nullptr && std::atoi(raw) != 0;
}

struct InterchangeBufferTouches {
  std::unordered_map<const BufferNode *, Buffer> local_buffers;
  std::unordered_map<const BufferNode *, Buffer> written_local_buffers;
};

InterchangeBufferTouches CollectInterchangeBufferTouches(const Stmt &stmt) {
  InterchangeBufferTouches result;
  auto record = [&](const Buffer &buffer, bool is_write) {
    String scope = buffer.scope();
    if (!scope.empty() && scope != "global") {
      result.local_buffers.emplace(buffer.get(), buffer);
      if (is_write) {
        result.written_local_buffers.emplace(buffer.get(), buffer);
      }
    }
  };
  PostOrderVisit(stmt, [&](const ObjectRef &obj) {
    if (const auto *store = obj.as<BufferStoreNode>()) {
      record(store->buffer, true);
      return;
    }
    if (const auto *load = obj.as<BufferLoadNode>()) {
      record(load->buffer, false);
      return;
    }
    if (const auto *call = obj.as<CallNode>()) {
      if (call->op.same_as(RegionOp::Get())) {
        RegionOp region(call->args);
        std::optional<int64_t> access_mask =
            AsStaticInt(region->GetAccessMask());
        // Unknown masks are conservatively treated as writes. A region with
        // write access cannot be hoisted when the same local buffer is written
        // again later in the logical inner body.
        bool is_write = !access_mask.has_value() || ((*access_mask & 2) != 0);
        record(region->GetBuffer(), is_write);
      }
    }
  });
  return result;
}

bool WritesBufferWrittenLater(const Array<Stmt> &statements, int index) {
  InterchangeBufferTouches current =
      CollectInterchangeBufferTouches(statements[index]);
  if (current.written_local_buffers.empty()) {
    return false;
  }
  for (int later_index = index + 1;
       later_index < static_cast<int>(statements.size()); ++later_index) {
    InterchangeBufferTouches later =
        CollectInterchangeBufferTouches(statements[later_index]);
    for (const auto &entry : current.written_local_buffers) {
      if (later.written_local_buffers.count(entry.first) != 0) {
        return true;
      }
    }
  }
  return false;
}

bool UsesVar(const Stmt &stmt, const Var &var) {
  bool found = false;
  PostOrderVisit(stmt, [&](const ObjectRef &obj) {
    if (!found) {
      if (const auto *candidate = obj.as<VarNode>()) {
        found = candidate == var.get();
      }
    }
  });
  return found;
}

bool TouchesAnyBuffer(const Stmt &stmt,
                      const std::unordered_set<const BufferNode *> &buffers) {
  InterchangeBufferTouches touches = CollectInterchangeBufferTouches(stmt);
  for (const auto &entry : touches.local_buffers) {
    if (buffers.count(entry.first) != 0) {
      return true;
    }
  }
  return false;
}

bool ContainsUnsupportedInterchangeControl(const Stmt &stmt) {
  bool unsupported = false;
  PostOrderVisit(stmt, [&](const ObjectRef &obj) {
    if (obj.as<WhileNode>() || obj.as<IfThenElseNode>()) {
      unsupported = true;
    }
  });
  return unsupported;
}

class DirectNestedLoopCollector : public StmtVisitor {
public:
  static std::vector<For> Collect(const Stmt &body) {
    DirectNestedLoopCollector collector;
    collector(body);
    return collector.loops_;
  }

private:
  void VisitStmt_(const ForNode *op) final {
    loops_.push_back(ffi::GetRef<For>(op));
    // Do not descend through a loop: only the nearest nested loop belongs to
    // the current outer-loop candidate.
  }

  std::vector<For> loops_;
};

class StripPipelineAnnotations : public StmtMutator {
private:
  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtMutator::VisitStmt_(op));
    Map<String, Any> annotations;
    for (const auto &kv : loop->annotations) {
      if (kv.first != "num_stages" && kv.first != kMultidimOriginalLoop) {
        annotations.Set(kv.first, kv.second);
      }
    }
    loop.CopyOnWrite()->annotations = annotations;
    return loop;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == kMultidimInnerCoordinate) {
      return VisitStmt(op->body);
    }
    return StmtMutator::VisitStmt_(op);
  }
};
class MarkPipelineArtifactsSuppressed : public StmtMutator {
private:
  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtMutator::VisitStmt_(op));
    if (HasNumStages(loop.get())) {
      Map<String, Any> annotations = loop->annotations;
      annotations.Set(kPipelineSuppressArtifacts, Bool(true));
      loop.CopyOnWrite()->annotations = annotations;
    }
    return loop;
  }
};

class StripCandidatePrivateAnnotations : public StmtMutator {
private:
  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtMutator::VisitStmt_(op));
    Map<String, Any> annotations;
    for (const auto &kv : loop->annotations) {
      if (kv.first != kPipelineSuppressArtifacts &&
          kv.first != kMultidimCandidateLoop) {
        annotations.Set(kv.first, kv.second);
      }
    }
    loop.CopyOnWrite()->annotations = annotations;
    return loop;
  }
};

class ReplaceLoop : public StmtMutator {
public:
  ReplaceLoop(const ForNode *target, Stmt replacement)
      : target_(target), replacement_(std::move(replacement)) {}

private:
  Stmt VisitStmt_(const ForNode *op) final {
    if (op == target_) {
      return replacement_;
    }
    return StmtMutator::VisitStmt_(op);
  }

  const ForNode *target_;
  Stmt replacement_;
};

struct RegionEvent {
  bool accepted{false};
  std::string outer;
  std::string inner;
  std::optional<int64_t> outer_min;
  std::optional<int64_t> outer_extent;
  std::optional<int64_t> inner_min;
  std::optional<int64_t> inner_extent;
  std::string reason;
  int64_t source_outer_extent{0};
  int64_t group_count{0};
  bool tail{false};
  int64_t candidate_id{-1};
  int64_t core_decision_id{-1};
  std::string fallback_stage;
  std::string outer_min_expr, outer_extent_expr, inner_min_expr,
      inner_extent_expr;
};

// This proof is separate from tile residency. An accepted candidate retains
// the original outer/core order and still needs the RegionCrossOuter planner's
// hazard, collective-epoch and event-coverage checks after 1D planning.
class BoundaryOverlapAnnotator : public StmtMutator {
public:
  BoundaryOverlapAnnotator(const PrimFunc &func, std::string policy)
      : func_(func), policy_(std::move(policy)) {}

private:
  Stmt VisitStmt_(const ForNode *op) final {
    if (HasNumStages(op))
      return ffi::GetRef<For>(op);
    auto reason = op->annotations.Get(residency::kReason);
    if (!reason.has_value() ||
        Downcast<String>(reason.value()) != "no_invariant_transfer_prefix")
      return StmtMutator::VisitStmt_(op);
    For loop = ffi::GetRef<For>(op);
    boundary_overlap::Proof proof = boundary_overlap::Analyze(func_, loop);
    if (policy_ == "shadow")
      return loop;
    Map<String, Any> annotations = loop->annotations;
    annotations.erase(boundary_overlap::kEligible);
    annotations.erase(boundary_overlap::kProofVersion);
    annotations.erase(boundary_overlap::kReadOnlyBuffers);
    annotations.erase(boundary_overlap::kStateBuffers);
    if (proof.accepted) {
      annotations.Set(boundary_overlap::kEligible, Bool(true));
      annotations.Set(boundary_overlap::kProofVersion, Integer(1));
      annotations.Set(boundary_overlap::kReadOnlyBuffers,
                      proof.read_only_buffers);
      annotations.Set(boundary_overlap::kStateBuffers, proof.state_buffers);
      annotations.erase(boundary_overlap::kReason);
      loop.CopyOnWrite()->body =
          SeqStmt::Flatten(residency::Statements(op->body));
    } else {
      ICHECK_NE(policy_, "forced")
          << "boundary overlap rejected: " << proof.reason;
      annotations.Set(boundary_overlap::kReason, String(proof.reason));
    }
    // Keep residency::kReason: failure of that optimization remains a fact.
    loop.CopyOnWrite()->annotations = annotations;
    return loop;
  }
  PrimFunc func_;
  std::string policy_;
};

class MultidimRegionNormalizer : public StmtExprMutator {
public:
  explicit MultidimRegionNormalizer(std::string policy)
      : policy_(std::move(policy)) {
    ICHECK(policy_ == "forced" || policy_ == "auto")
        << "tl.sunmmio_multidim_policy must be 'forced' or 'auto'";
  }

  const std::vector<RegionEvent> &events() const { return events_; }

private:
  struct CoreDecision {
    int64_t candidate_id;
    int64_t core_decision_id;
    std::string stage;
    std::string reason;
  };

  For MarkCore(const For &core, const CoreDecision &decision) {
    For result =
        MarkOneDimensionalFallback(core, decision.stage, decision.reason);
    auto annotations = result->annotations;
    annotations.Set(kMultidimCandidateId, Integer(decision.candidate_id));
    annotations.Set(kMultidimCoreDecisionId,
                    Integer(decision.core_decision_id));
    result.CopyOnWrite()->annotations = annotations;
    return result;
  }

  Stmt RejectCores(const For &outer, const std::vector<For> &cores,
                   const std::string &stage, const std::string &reason) {
    ICHECK(!cores.empty());
    // A failed generic residency proof is an authoritative 1D boundary even
    // under forced policy.  Only region discovery has the forced-error
    // contract.
    if (stage == "region_discovery") {
      ICHECK_NE(policy_, "forced")
          << "multidim_ilp forced region rejected: " << reason
          << " outer=" << outer->loop_var
          << " inner=" << cores.front()->loop_var;
    }

    const int64_t candidate_id = next_candidate_id_++;
    auto expression = [](const PrimExpr &expr) {
      std::ostringstream stream;
      stream << expr;
      return stream.str();
    };
    for (const For &core : cores) {
      ICHECK(!claimed_cores_.count(core->loop_var.get()))
          << "one source pipeline core cannot own multiple rejection decisions";
      CoreDecision decision{candidate_id, next_core_decision_id_++, stage,
                            reason};
      claimed_cores_.emplace(core->loop_var.get(), decision);
      RegionEvent event;
      event.outer = outer->loop_var->name_hint;
      event.inner = core->loop_var->name_hint;
      event.outer_min = AsStaticInt(outer->min);
      event.outer_extent = AsStaticInt(outer->extent);
      event.inner_min = AsStaticInt(core->min);
      event.inner_extent = AsStaticInt(core->extent);
      event.reason = reason;
      event.candidate_id = candidate_id;
      event.core_decision_id = decision.core_decision_id;
      event.fallback_stage = stage;
      event.outer_min_expr = expression(outer->min);
      event.outer_extent_expr = expression(outer->extent);
      event.inner_min_expr = expression(core->min);
      event.inner_extent_expr = expression(core->extent);
      events_.push_back(std::move(event));
    }

    if (outer.same_as(cores.front())) {
      ICHECK_EQ(cores.size(), 1U);
      For result = MarkCore(outer, claimed_cores_.at(outer->loop_var.get()));
      claimed_cores_.erase(outer->loop_var.get());
      return result;
    }

    // Protect only the exact cores already covered by this rejected candidate.
    // Other nearest-loop subtrees still need their own independent decisions.
    Stmt body = VisitStmt(outer->body);
    for (const For &core : cores)
      claimed_cores_.erase(core->loop_var.get());

    For fallback = outer;
    fallback.CopyOnWrite()->body = body;
    Map<String, Any> annotations = fallback->annotations;
    if (stage == "region_discovery") {
      annotations.Set(kMultidimRank, Integer(2));
      annotations.Set(kMultidimOuterVar, String(outer->loop_var->name_hint));
      annotations.Set(kMultidimInnerVar,
                      String(cores.front()->loop_var->name_hint));
    }
    // Keep the rejected candidate as diagnostics; a selection belongs to each
    // concrete core, not to every descendant of this enclosing loop.
    annotations.Set(kMultidimFallbackStage, String(stage));
    annotations.Set(kMultidimFallbackReason, String(reason));
    annotations.erase(kMultidimSelectedMode);
    annotations.erase(kMultidimCoreDecisionId);
    annotations.erase(kMultidimCandidateId);
    fallback.CopyOnWrite()->annotations = annotations;
    return fallback;
  }

  Stmt Reject(const For &outer, const For &inner, const std::string &reason) {
    return RejectCores(outer, {inner}, "region_discovery", reason);
  }

  std::optional<Array<Stmt>> InnerStatements(const For &inner,
                                             std::string *reason) const {
    Stmt body = inner->body;
    if (const auto *realize = body.as<BlockRealizeNode>()) {
      if (!realize->block->alloc_buffers.empty() ||
          !realize->block->match_buffers.empty() ||
          realize->block->init.defined()) {
        *reason = "inner_block_state_unsupported";
        return std::nullopt;
      }
      body = realize->block->body;
    }
    if (const auto *seq = body.as<SeqStmtNode>())
      return seq->seq;
    return Array<Stmt>{body};
  }

  Stmt NormalizeHierarchical(const For &outer, const For &inner,
                             int64_t outer_min, int64_t outer_extent,
                             int64_t inner_min, int64_t inner_extent,
                             int64_t num_stages) {
    For marked_inner = inner;
    Map<String, Any> inner_annotations = marked_inner->annotations;
    inner_annotations.Set(kHierarchicalInner, Bool(true));
    marked_inner.CopyOnWrite()->annotations = inner_annotations;

    Stmt outer_body =
        ReplaceLoop(inner.get(), std::move(marked_inner))(outer->body);
    For normalized = outer;
    auto *node = normalized.CopyOnWrite();
    node->body = outer_body;
    Map<String, Any> annotations = node->annotations;
    annotations.Set(kMultidimRegion, Bool(true));
    annotations.Set(kMultidimStrategy, String(kHierarchicalStrategy));
    annotations.Set(kMultidimRank, Integer(2));
    annotations.Set(kMultidimOuterVar, String(outer->loop_var->name_hint));
    annotations.Set(kMultidimInnerVar, String(inner->loop_var->name_hint));
    annotations.Set(kMultidimOuterMin, Integer(outer_min));
    annotations.Set(kMultidimOuterExtent, Integer(outer_extent));
    annotations.Set(kMultidimInnerMin, Integer(inner_min));
    annotations.Set(kMultidimInnerExtent, Integer(inner_extent));
    annotations.Set(kMultidimOriginalLoop, outer);
    annotations.Set("tl.sunmmio.multidim.requested_stages",
                    Integer(num_stages));
    node->annotations = annotations;

    events_.push_back(RegionEvent{true, outer->loop_var->name_hint,
                                  inner->loop_var->name_hint, outer_min,
                                  outer_extent, inner_min, inner_extent, ""});
    return normalized;
  }

  Stmt NormalizeInterchange(const For &outer, const For &inner,
                            int64_t outer_min, int64_t outer_extent,
                            int64_t inner_min, int64_t inner_extent,
                            int64_t num_stages) {
    bool generic = outer->annotations.count(residency::kEligible);
    if (policy_ != "forced" && !generic) {
      return Reject(outer, inner, "interchange_requires_forced_policy");
    }
    int64_t group_width = 0;
    if (auto width = outer->annotations.Get(residency::kGroupWidth))
      group_width = Downcast<IntImm>(width.value())->value;
    bool grouped = generic && group_width >= 2 && group_width <= 8 &&
                   group_width < outer_extent;
    if (outer_extent < 2 || (outer_extent > 8 && !grouped)) {
      return Reject(outer, inner, "interchange_outer_extent_out_of_range");
    }
    if (!generic && ContainsUnsupportedInterchangeControl(outer->body)) {
      return Reject(outer, inner, "interchange_control_flow_unsupported");
    }

    Stmt canonical_body =
        generic ? SeqStmt::Flatten(residency::Statements(outer->body))
                : outer->body;
    const auto *body = canonical_body.as<SeqStmtNode>();
    if (body == nullptr) {
      return Reject(outer, inner, "interchange_outer_body_not_sequence");
    }

    int inner_index = -1;
    for (int i = 0; i < static_cast<int>(body->seq.size()); ++i) {
      if (body->seq[i].same_as(inner)) {
        inner_index = i;
        break;
      }
    }
    if (inner_index < 0 ||
        (!generic && (inner_index == 0 ||
                      inner_index + 1 >= static_cast<int>(body->seq.size())))) {
      return Reject(outer, inner, "interchange_prefix_or_suffix_missing");
    }

    Array<Stmt> prefix;
    Array<Stmt> suffix;
    for (int i = 0; i < inner_index; ++i) {
      prefix.push_back(body->seq[i]);
    }
    for (int i = inner_index + 1; i < static_cast<int>(body->seq.size()); ++i) {
      suffix.push_back(body->seq[i]);
    }

    InterchangeBufferTouches prefix_touches =
        CollectInterchangeBufferTouches(SeqStmt::Flatten(prefix));
    InterchangeBufferTouches inner_touches =
        CollectInterchangeBufferTouches(inner->body);
    InterchangeBufferTouches suffix_touches =
        CollectInterchangeBufferTouches(SeqStmt::Flatten(suffix));

    std::vector<Buffer> accumulator_candidates;
    for (const auto &entry : prefix_touches.local_buffers) {
      const BufferNode *node = entry.first;
      const Buffer &buffer = entry.second;
      if (inner_touches.local_buffers.count(node) != 0 &&
          suffix_touches.local_buffers.count(node) != 0 &&
          buffer.scope() == "shared.rsram") {
        accumulator_candidates.push_back(buffer);
      }
    }
    if (generic) {
      accumulator_candidates.clear();
      Array<Buffer> states = Downcast<Array<Buffer>>(
          outer->annotations.Get(residency::kStateBuffers).value());
      for (const Buffer &state : states)
        accumulator_candidates.push_back(state);
    }
    std::sort(accumulator_candidates.begin(), accumulator_candidates.end(),
              [](const Buffer &lhs, const Buffer &rhs) {
                return std::string(lhs->name) < std::string(rhs->name);
              });
    if (accumulator_candidates.empty()) {
      return Reject(outer, inner, "interchange_accumulator_missing");
    }

    Array<Buffer> accumulators;
    for (const Buffer &buffer : accumulator_candidates) {
      accumulators.push_back(buffer);
    }

    For normalized = outer;
    if (generic)
      normalized.CopyOnWrite()->body = canonical_body;
    Map<String, Any> annotations = normalized->annotations;
    annotations.Set(kMultidimRegion, Bool(true));
    annotations.Set(kMultidimStrategy, String(kInterchangeStrategy));
    annotations.Set(kMultidimRank, Integer(2));
    annotations.Set(kMultidimOuterVar, String(outer->loop_var->name_hint));
    annotations.Set(kMultidimInnerVar, String(inner->loop_var->name_hint));
    annotations.Set(kMultidimOuterMin, Integer(outer_min));
    annotations.Set(kMultidimOuterExtent, Integer(outer_extent));
    annotations.Set(kMultidimInnerMin, Integer(0));
    annotations.Set(kMultidimInnerExtent, Integer(outer_extent));
    annotations.Set(kInterchangeKMin, Integer(inner_min));
    annotations.Set(kInterchangeKExtent, Integer(inner_extent));
    annotations.Set(kMultidimOriginalLoop, outer);
    annotations.Set("tl.sunmmio.multidim.requested_stages",
                    Integer(num_stages));
    annotations.Set("iterations",
                    Integer(grouped ? group_width : outer_extent));
    annotations.Set("used_buffers", accumulators);
    annotations.Set("versioned_buffers", accumulators);
    annotations.Set("runtime_multiversion_buffers", accumulators);
    normalized.CopyOnWrite()->annotations = annotations;

    auto record_group = [&](int64_t lanes, int64_t count, bool tail) {
      RegionEvent event{true,
                        outer->loop_var->name_hint,
                        inner->loop_var->name_hint,
                        outer_min,
                        lanes,
                        0,
                        lanes,
                        ""};
      if (grouped) {
        event.source_outer_extent = outer_extent;
        event.group_count = count;
        event.tail = tail;
      }
      events_.push_back(std::move(event));
    };
    if (grouped) {
      record_group(group_width, outer_extent / group_width, false);
      if (outer_extent % group_width)
        record_group(outer_extent % group_width, 1, true);
    } else {
      record_group(outer_extent, 1, false);
    }
    return normalized;
  }

  Stmt Normalize(const For &outer, const For &inner, int64_t outer_min,
                 int64_t outer_extent, int64_t inner_min, int64_t inner_extent,
                 int64_t num_stages) {
    if (InterchangeEnabled() ||
        outer->annotations.count(residency::kEligible)) {
      return NormalizeInterchange(outer, inner, outer_min, outer_extent,
                                  inner_min, inner_extent, num_stages);
    }

    if (const char *raw = std::getenv("TL_SUNMMIO_MULTIDIM_HIERARCHICAL")) {
      if (std::atoi(raw) != 0) {
        return NormalizeHierarchical(outer, inner, outer_min, outer_extent,
                                     inner_min, inner_extent, num_stages);
      }
    }
    std::string reason;
    std::optional<Array<Stmt>> inner_statements =
        InnerStatements(inner, &reason);
    if (!inner_statements.has_value()) {
      return Reject(outer, inner, reason);
    }

    Array<Stmt> expanded;
    for (int64_t offset = 0; offset < inner_extent; ++offset) {
      PrimExpr coordinate =
          make_const(inner->loop_var.dtype(), inner_min + offset);
      Map<Var, PrimExpr> substitution;
      substitution.Set(inner->loop_var, coordinate);
      for (const Stmt &stmt : inner_statements.value()) {
        Stmt instance = Substitute(stmt, substitution);
        instance = AttrStmt(inner->loop_var, kMultidimInnerCoordinate,
                            Integer(inner_min + offset), instance);
        expanded.push_back(instance);
      }
    }

    Stmt expanded_inner = SeqStmt::Flatten(expanded);
    Stmt outer_body =
        ReplaceLoop(inner.get(), std::move(expanded_inner))(outer->body);

    For normalized = outer;
    auto *node = normalized.CopyOnWrite();
    node->body = outer_body;
    Map<String, Any> annotations = node->annotations;
    annotations.Set("num_stages", Integer(num_stages));
    annotations.Set(kMultidimRegion, Bool(true));
    annotations.Set(kMultidimRank, Integer(2));
    annotations.Set(kMultidimOuterVar, String(outer->loop_var->name_hint));
    annotations.Set(kMultidimInnerVar, String(inner->loop_var->name_hint));
    annotations.Set(kMultidimOuterMin, Integer(outer_min));
    annotations.Set(kMultidimOuterExtent, Integer(outer_extent));
    annotations.Set(kMultidimInnerMin, Integer(inner_min));
    annotations.Set(kMultidimInnerExtent, Integer(inner_extent));
    annotations.Set(kMultidimOriginalLoop, outer);
    node->annotations = annotations;

    events_.push_back(RegionEvent{true, outer->loop_var->name_hint,
                                  inner->loop_var->name_hint, outer_min,
                                  outer_extent, inner_min, inner_extent, ""});
    return normalized;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    For outer = ffi::GetRef<For>(op);
    if (auto claimed = claimed_cores_.find(op->loop_var.get());
        claimed != claimed_cores_.end()) {
      ICHECK(HasNumStages(op));
      return MarkCore(outer, claimed->second);
    }

    if (boundary_overlap::Enabled() &&
        op->annotations.count(boundary_overlap::kEligible)) {
      // Never re-enter legacy interchange. Preserve the natural region and
      // plan its inner core; the separate proved-only region planner is the
      // sole consumer of this alternative optimization's authorization.
      return outer;
    }

    if (auto rejected = op->annotations.Get(residency::kReason)) {
      // A failed generic proof is a region boundary for *all* interchange
      // paths, including the historical environment-controlled path.
      std::vector<For> cores;
      if (HasNumStages(op)) {
        cores.push_back(outer);
      } else {
        for (const For &nested : DirectNestedLoopCollector::Collect(op->body))
          if (HasNumStages(nested.get()))
            cores.push_back(nested);
      }
      if (!cores.empty())
        return RejectCores(outer, cores, "generic_residency",
                           Downcast<String>(rejected.value()));
      // Unknown ownership remains an opaque fallback boundary.  Do not infer
      // a core binding by name or broadcast this decision to descendants.
      return MarkOneDimensionalFallback(outer, "generic_residency",
                                        Downcast<String>(rejected.value()));
    }

    if (HasNumStages(op)) {
      // A pipeline loop reaching this point had no legal static outer region.
      return Reject(outer, outer, "missing_static_outer_region");
    }

    std::vector<For> nested = DirectNestedLoopCollector::Collect(op->body);
    std::vector<For> pipeline_loops;
    for (const For &loop : nested) {
      if (HasNumStages(loop.get())) {
        pipeline_loops.push_back(loop);
      }
    }
    if (pipeline_loops.empty()) {
      return StmtExprMutator::VisitStmt_(op);
    }
    if (pipeline_loops.size() != 1) {
      return RejectCores(outer, pipeline_loops, "region_discovery",
                         "multiple_inner_pipeline_loops");
    }

    For inner = pipeline_loops.front();
    if (outer->kind != ForKind::kSerial || inner->kind != ForKind::kSerial) {
      return Reject(outer, inner, "non_serial_loop_kind");
    }
    std::optional<int64_t> outer_min = AsStaticInt(outer->min);
    std::optional<int64_t> outer_extent = AsStaticInt(outer->extent);
    std::optional<int64_t> inner_min = AsStaticInt(inner->min);
    std::optional<int64_t> inner_extent = AsStaticInt(inner->extent);
    if (!outer_min || !outer_extent || !inner_min || !inner_extent) {
      return Reject(outer, inner, "dynamic_or_non_affine_extent");
    }
    if (*outer_extent <= 1) {
      return Reject(outer, inner, "outer_extent_too_short");
    }
    int64_t max_inner_extent = 64;
    if (const char *raw = std::getenv("TL_SUNMMIO_MULTIDIM_MAX_INNER_EXTENT")) {
      max_inner_extent = std::max<int64_t>(1, std::atoll(raw));
    }
    if (*inner_extent <= 0 || *inner_extent > max_inner_extent) {
      return Reject(outer, inner, "inner_extent_out_of_range");
    }
    if (ContainsBarrier(outer->body)) {
      return Reject(outer, inner, "barrier_region_boundary");
    }

    auto stages = inner->annotations.Get("num_stages");
    if (!stages.has_value()) {
      return Reject(outer, inner, "invalid_num_stages");
    }
    IntImm stage_imm = Downcast<IntImm>(stages.value());
    if (stage_imm->value <= 0) {
      return Reject(outer, inner, "invalid_num_stages");
    }

    return Normalize(outer, inner, *outer_min, *outer_extent, *inner_min,
                     *inner_extent, stage_imm->value);
  }

  std::string policy_;
  std::vector<RegionEvent> events_;
  std::unordered_map<const VarNode *, CoreDecision> claimed_cores_;
  int64_t next_candidate_id_{0};
  int64_t next_core_decision_id_{0};
};

std::optional<int64_t> AnnotationInt(const For &loop, const char *key);

class InterchangeVersionAccessRewriter : public StmtExprMutator {
public:
  using VersionMap = std::unordered_map<const BufferNode *, Array<Buffer>>;

  InterchangeVersionAccessRewriter(const VersionMap &versions, int version)
      : versions_(versions), version_(version) {}

private:
  bool IsVersioned(const Buffer &buffer) const {
    return versions_.count(buffer.get()) != 0;
  }

  Buffer VersionBuffer(const Buffer &buffer) const {
    auto it = versions_.find(buffer.get());
    ICHECK(it != versions_.end())
        << "interchange accumulator has no independent versions";
    ICHECK_GE(version_, 0);
    ICHECK_LT(version_, static_cast<int>(it->second.size()));
    return it->second[version_];
  }

  BufferRegion RewriteRegion(const BufferRegion &region) const {
    if (!IsVersioned(region->buffer)) {
      return region;
    }
    return BufferRegion(VersionBuffer(region->buffer), region->region);
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    block.CopyOnWrite()->reads.MutateByApply(
        [this](const BufferRegion &region) { return RewriteRegion(region); });
    block.CopyOnWrite()->writes.MutateByApply(
        [this](const BufferRegion &region) { return RewriteRegion(region); });
    return block;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    if (!IsVersioned(store->buffer)) {
      return store;
    }
    return BufferStore(VersionBuffer(store->buffer), store->value,
                       store->indices, store->predicate, store->span);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    if (!IsVersioned(load->buffer)) {
      return load;
    }
    return BufferLoad(VersionBuffer(load->buffer), load->indices,
                      load->predicate, load->span);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (!op->op.same_as(RegionOp::Get())) {
      return StmtExprMutator::VisitExpr_(op);
    }
    Call call = ffi::GetRef<Call>(op);
    RegionOp region(call->args);
    Buffer original = region->GetBuffer();
    if (!IsVersioned(original)) {
      return StmtExprMutator::VisitExpr_(op);
    }

    Array<Range> ranges = region->GetRanges();
    Array<PrimExpr> mins;
    Array<PrimExpr> args;
    for (const Range &range : ranges) {
      mins.push_back(VisitExpr(range->min));
    }
    args.push_back(BufferLoad(VersionBuffer(original), mins));
    args.push_back(region->GetAccessMask());
    for (const Range &range : ranges) {
      args.push_back(VisitExpr(range->extent));
    }
    return Call(call->dtype, call->op, args, call->annotations, call->span);
  }

  const VersionMap &versions_;
  int version_;
};

class InterchangeLoopBuilder : public StmtMutator {
public:
  using VersionMap = InterchangeVersionAccessRewriter::VersionMap;

  explicit InterchangeLoopBuilder(std::string policy)
      : policy_(std::move(policy)) {
    ICHECK(policy_ == "forced" || policy_ == "auto");
  }

  void RewriteFunctionLayoutAttrs(PrimFunc &f) const {
    auto layout_map = f->GetAttr<Map<Buffer, Layout>>(attr::kLayoutMap);
    if (layout_map.has_value()) {
      f = WithAttr(std::move(f), attr::kLayoutMap,
                   RewriteLayoutMap(layout_map.value()));
    }
  }

private:
  Stmt Fail(const std::string &reason) const {
    ICHECK(false) << "multidim interchange construction failed: " << reason;
    return Stmt();
  }

  std::optional<Array<Stmt>> GetInnerStatements(const For &inner,
                                                bool generic) const {
    Stmt body = inner->body;
    if (const auto *realize = body.as<BlockRealizeNode>()) {
      if (!realize->block->alloc_buffers.empty() ||
          !realize->block->match_buffers.empty() ||
          realize->block->init.defined()) {
        return std::nullopt;
      }
      body = realize->block->body;
    }
    if (generic)
      return residency::Statements(body);
    if (const auto *seq = body.as<SeqStmtNode>())
      return seq->seq;
    return Array<Stmt>{body};
  }

  Buffer MakeAccumulatorVersion(const Buffer &buffer, int version) const {
    std::string suffix = "_v" + std::to_string(version);
    Var data(std::string(buffer->data->name_hint) + suffix,
             buffer->data->type_annotation);
    return Buffer(data, buffer->dtype, buffer->shape, buffer->strides,
                  buffer->elem_offset,
                  String(std::string(buffer->name) + suffix),
                  buffer->data_alignment, buffer->offset_factor,
                  buffer->buffer_type, buffer->axis_separators, buffer->span);
  }

  void EnsureAccumulatorVersions(const Array<Buffer> &accumulators,
                                 int versions) {
    ICHECK_GT(versions, 0);
    for (const Buffer &buffer : accumulators) {
      auto it = accumulator_versions_.find(buffer.get());
      if (it != accumulator_versions_.end()) {
        ICHECK_EQ(static_cast<int>(it->second.size()), versions)
            << "interchange accumulator reused with a different version count";
        continue;
      }
      Array<Buffer> buffers;
      for (int version = 0; version < versions; ++version) {
        buffers.push_back(MakeAccumulatorVersion(buffer, version));
      }
      accumulator_versions_.emplace(buffer.get(), std::move(buffers));
    }
  }

  Map<Buffer, Layout>
  RewriteLayoutMap(const Map<Buffer, Layout> &layout_map) const {
    arith::Analyzer analyzer;
    Map<Buffer, Layout> rewritten;
    for (const auto &[buffer, layout] : layout_map) {
      auto versions = accumulator_versions_.find(buffer.get());
      if (versions == accumulator_versions_.end()) {
        rewritten.Set(buffer, layout);
        continue;
      }
      for (const Buffer &version : versions->second) {
        Optional<Layout> derived =
            DeriveLayoutLikeForDType(layout, version->shape, version->dtype,
                                     Optional<Array<Integer>>(), &analyzer);
        ICHECK(derived.defined())
            << "failed to derive layout for independent interchange "
               "accumulator "
            << version->name;
        rewritten.Set(version, derived.value());
      }
    }
    return rewritten;
  }

  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    BlockRealize realize = Downcast<BlockRealize>(StmtMutator::VisitStmt_(op));
    Block block = realize->block;

    if (block->annotations.count(attr::kLayoutMap)) {
      auto value = block->annotations.Get(attr::kLayoutMap);
      if (auto layout_map = value.value().as<Map<Buffer, Layout>>()) {
        Map<String, Any> annotations = block->annotations;
        annotations.Set(attr::kLayoutMap, RewriteLayoutMap(layout_map.value()));
        block.CopyOnWrite()->annotations = annotations;
      }
    }

    Array<Buffer> alloc_buffers;
    bool changed = false;
    for (const Buffer &buffer : block->alloc_buffers) {
      auto versions = accumulator_versions_.find(buffer.get());
      if (versions == accumulator_versions_.end()) {
        alloc_buffers.push_back(buffer);
        continue;
      }
      changed = true;
      for (const Buffer &version : versions->second) {
        alloc_buffers.push_back(version);
      }
    }
    if (changed) {
      block.CopyOnWrite()->alloc_buffers = alloc_buffers;
    }
    realize.CopyOnWrite()->block = block;
    return realize;
  }

  Stmt MakeInstance(const Stmt &stmt, const Var &outer_var, PrimExpr coordinate,
                    int version) const {
    Map<Var, PrimExpr> substitution;
    arith::Analyzer analyzer;
    substitution.Set(outer_var, analyzer.Simplify(coordinate));
    Stmt instance = Substitute(stmt, substitution);
    return InterchangeVersionAccessRewriter(accumulator_versions_,
                                            version)(instance);
  }

  Stmt MarkCoordinate(const Var &inner_var, int coordinate,
                      const Stmt &stmt) const {
    return AttrStmt(inner_var, kMultidimInnerCoordinate, Integer(coordinate),
                    stmt);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    auto strategy = op->annotations.Get(kMultidimStrategy);
    if (!strategy.has_value() ||
        Downcast<String>(strategy.value()) != kInterchangeStrategy) {
      return StmtMutator::VisitStmt_(op);
    }

    For original = ffi::GetRef<For>(op);
    auto outer_min = AnnotationInt(original, kMultidimOuterMin);
    auto outer_extent = AnnotationInt(original, kMultidimOuterExtent);
    auto buffers_value =
        original->annotations.Get("runtime_multiversion_buffers");
    if (!outer_min || !outer_extent || !buffers_value.has_value()) {
      return Fail("interchange_metadata_missing");
    }
    Array<Buffer> accumulators = Downcast<Array<Buffer>>(buffers_value.value());
    if (accumulators.empty()) {
      return Fail("interchange_accumulator_missing_after_prepare");
    }
    auto grouped_width = AnnotationInt(original, residency::kGroupWidth);
    int64_t pool_width = grouped_width.value_or(*outer_extent);
    ICHECK_GE(pool_width, 2);
    ICHECK_LE(pool_width, 8);
    EnsureAccumulatorVersions(accumulators, pool_width);

    // Create version buffers before descending so enclosing and nested block
    // metadata can be rewritten during the normal post-order traversal.
    For loop = Downcast<For>(StmtMutator::VisitStmt_(op));

    const auto *body = loop->body.as<SeqStmtNode>();
    if (body == nullptr) {
      return Fail("interchange_outer_body_not_sequence_after_prepare");
    }

    int inner_index = -1;
    For inner;
    for (int i = 0; i < static_cast<int>(body->seq.size()); ++i) {
      if (const auto *candidate = body->seq[i].as<ForNode>()) {
        if (HasNumStages(candidate)) {
          if (inner_index >= 0) {
            return Fail("interchange_multiple_pipeline_loops");
          }
          inner_index = i;
          inner = ffi::GetRef<For>(candidate);
        }
      }
    }
    bool generic = loop->annotations.count(residency::kEligible);
    if (inner_index < 0 ||
        (!generic && (inner_index == 0 ||
                      inner_index + 1 >= static_cast<int>(body->seq.size())))) {
      return Fail("interchange_prefix_or_suffix_missing_after_prepare");
    }

    std::optional<Array<Stmt>> inner_statements =
        GetInnerStatements(inner, generic);
    if (!inner_statements.has_value()) {
      return Fail("interchange_inner_block_state_unsupported");
    }

    std::unordered_set<const BufferNode *> accumulator_set;
    for (const Buffer &buffer : accumulators) {
      accumulator_set.insert(buffer.get());
    }

    int shared_prefix_count = 0;
    for (int i = 0; i < static_cast<int>(inner_statements.value().size());
         ++i) {
      const Stmt &stmt = inner_statements.value()[i];
      bool dependent = UsesVar(stmt, loop->loop_var) ||
                       TouchesAnyBuffer(stmt, accumulator_set) ||
                       WritesBufferWrittenLater(inner_statements.value(), i);
      if (dependent) {
        break;
      }
      ++shared_prefix_count;
    }
    if (generic) {
      int proved_count =
          Downcast<IntImm>(
              loop->annotations.Get(residency::kPrefixCount).value())
              ->value;
      ICHECK_EQ(proved_count, shared_prefix_count)
          << "residency proof/prefix builder disagreement";
    }
    if (shared_prefix_count <= 0 ||
        shared_prefix_count >=
            static_cast<int>(inner_statements.value().size())) {
      return Fail("interchange_shared_prefix_not_found");
    }

    auto build_group = [&](PrimExpr group_base, int64_t lanes) -> Stmt {
      Array<Stmt> new_inner_body;
      for (int i = 0; i < shared_prefix_count; ++i) {
        new_inner_body.push_back(
            MarkCoordinate(inner->loop_var, 0, inner_statements.value()[i]));
      }
      for (int version = 0; version < lanes; ++version) {
        PrimExpr coordinate =
            group_base + make_const(loop->loop_var.dtype(), version);
        for (int i = shared_prefix_count;
             i < static_cast<int>(inner_statements.value().size()); ++i) {
          Stmt instance = MakeInstance(inner_statements.value()[i],
                                       loop->loop_var, coordinate, version);
          new_inner_body.push_back(
              MarkCoordinate(inner->loop_var, version, instance));
        }
      }

      auto k_min = AsStaticInt(inner->min);
      auto k_extent = AsStaticInt(inner->extent);
      ICHECK(k_min && k_extent) << "interchange K loop must remain static";
      Map<String, Any> inner_annotations = inner->annotations;
      inner_annotations.Set(kMultidimRegion, Bool(true));
      inner_annotations.Set(kMultidimStrategy, String(kInterchangeStrategy));
      inner_annotations.Set(kMultidimRank, Integer(2));
      inner_annotations.Set(kMultidimOuterVar,
                            String(loop->loop_var->name_hint));
      inner_annotations.Set(kMultidimInnerVar,
                            String(inner->loop_var->name_hint));
      inner_annotations.Set(kMultidimOuterMin, Integer(*outer_min));
      inner_annotations.Set(kMultidimOuterExtent, Integer(lanes));
      // The planner's command coordinate is the statically unrolled outer Tile.
      inner_annotations.Set(kMultidimInnerMin, Integer(0));
      inner_annotations.Set(kMultidimInnerExtent, Integer(lanes));
      inner_annotations.Set(kInterchangeKMin, Integer(*k_min));
      inner_annotations.Set(kInterchangeKExtent, Integer(*k_extent));
      inner_annotations.Set(kInterchangeApplied, Bool(true));
      if (generic)
        inner_annotations.Set(residency::kApplied, Bool(true));
      inner_annotations.Set(kInterchangeSharedPrefixCount,
                            Integer(shared_prefix_count));
      inner_annotations.Set(kInterchangeOuterVersions, Integer(lanes));
      if (grouped_width) {
        inner_annotations.Set(residency::kGroupedCore, Bool(true));
        inner_annotations.Set(residency::kGroupWidth, Integer(pool_width));
        inner_annotations.Set("tl.sunmmio.residency.group_base", group_base);
        inner_annotations.Set("tl.sunmmio.residency.source_outer_extent",
                              Integer(*outer_extent));
      }
      Array<Buffer> independent_accumulators;
      for (const Buffer &buffer : accumulators) {
        const auto &versions = accumulator_versions_.at(buffer.get());
        for (int version = 0; version < lanes; ++version) {
          independent_accumulators.push_back(versions[version]);
        }
      }
      inner_annotations.Set(kInterchangeAccumulatorVersions,
                            independent_accumulators);
      inner_annotations.Set(kMultidimSelectedMode, String("multidim_ilp"));

      For interchanged = inner;
      interchanged.CopyOnWrite()->body = SeqStmt::Flatten(new_inner_body);
      interchanged.CopyOnWrite()->annotations = inner_annotations;

      Array<Stmt> result;
      for (int version = 0; version < lanes; ++version) {
        PrimExpr coordinate =
            group_base + make_const(loop->loop_var.dtype(), version);
        for (int i = 0; i < inner_index; ++i) {
          result.push_back(
              MakeInstance(body->seq[i], loop->loop_var, coordinate, version));
        }
      }
      result.push_back(interchanged);
      for (int version = 0; version < lanes; ++version) {
        PrimExpr coordinate =
            group_base + make_const(loop->loop_var.dtype(), version);
        for (int i = inner_index + 1; i < static_cast<int>(body->seq.size());
             ++i) {
          result.push_back(
              MakeInstance(body->seq[i], loop->loop_var, coordinate, version));
        }
      }
      return SeqStmt::Flatten(result);
    };

    PrimExpr minimum = make_const(loop->loop_var.dtype(), *outer_min);
    if (!grouped_width)
      return build_group(minimum, *outer_extent);
    int64_t full_groups = *outer_extent / pool_width;
    int64_t tail_lanes = *outer_extent % pool_width;
    Var group(std::string(loop->loop_var->name_hint) + "_group",
              loop->loop_var.dtype());
    PrimExpr base = minimum + group * make_const(group.dtype(), pool_width);
    Array<Stmt> groups;
    groups.push_back(For(group, make_zero(group.dtype()),
                         make_const(group.dtype(), full_groups),
                         ForKind::kSerial, build_group(base, pool_width)));
    if (tail_lanes) {
      groups.push_back(build_group(
          minimum + make_const(group.dtype(), full_groups * pool_width),
          tail_lanes));
    }
    // Both syntactic cores must have distinct binder identities even though
    // their accumulator pool is intentionally shared sequentially.
    return ConvertSSA(SeqStmt::Flatten(groups));
  }

  std::string policy_;
  VersionMap accumulator_versions_;
};

void WriteRegionEvents(const std::vector<RegionEvent> &events) {
  const char *path = std::getenv("TL_SUNMMIO_MULTIDIM_REGION_JSON");
  if (path == nullptr || *path == '\0') {
    return;
  }
  std::ofstream out(path);
  ICHECK(out.is_open()) << "cannot open multidim region JSON: " << path;
  auto number = [](const std::optional<int64_t> &value) {
    return value ? std::to_string(*value) : std::string("null");
  };
  auto quoted = [](const std::string &value) {
    std::ostringstream stream;
    stream << '"';
    for (unsigned char ch : value) {
      if (ch == '"' || ch == '\\')
        stream << '\\' << ch;
      else if (ch == '\n')
        stream << "\\n";
      else if (ch == '\r')
        stream << "\\r";
      else if (ch == '\t')
        stream << "\\t";
      else if (ch < 0x20) {
        const char *digits = "0123456789abcdef";
        stream << "\\u00" << digits[ch >> 4] << digits[ch & 15];
      } else
        stream << ch;
    }
    stream << '"';
    return stream.str();
  };
  out << "{\n  \"schema_version\": 1,\n  \"regions\": [";
  for (size_t i = 0; i < events.size(); ++i) {
    const RegionEvent &event = events[i];
    if (i != 0) {
      out << ",";
    }
    out << "\n    {\"accepted\": " << (event.accepted ? "true" : "false")
        << ", \"outer_var\": " << quoted(event.outer)
        << ", \"inner_var\": " << quoted(event.inner)
        << ", \"outer_min\": " << number(event.outer_min)
        << ", \"outer_extent\": " << number(event.outer_extent)
        << ", \"inner_min\": " << number(event.inner_min)
        << ", \"inner_extent\": " << number(event.inner_extent)
        << ", \"source_outer_extent\": " << event.source_outer_extent
        << ", \"group_count\": " << event.group_count
        << ", \"tail\": " << (event.tail ? "true" : "false")
        << ", \"reason\": " << quoted(event.reason);
    if (!event.accepted) {
      out << ", \"candidate_id\": " << event.candidate_id
          << ", \"core_decision_id\": " << event.core_decision_id
          << ", \"fallback_stage\": " << quoted(event.fallback_stage)
          << ", \"outer_min_expr\": " << quoted(event.outer_min_expr)
          << ", \"outer_extent_expr\": " << quoted(event.outer_extent_expr)
          << ", \"inner_min_expr\": " << quoted(event.inner_min_expr)
          << ", \"inner_extent_expr\": " << quoted(event.inner_extent_expr);
    }
    out << "}";
  }
  out << "\n  ]\n}\n";
}

class MultidimValidationMutator : public StmtMutator {
public:
  MultidimValidationMutator(std::string policy, std::string stage)
      : policy_(std::move(policy)), stage_(std::move(stage)) {}

  bool failed() const { return failed_; }
  const std::string &reason() const { return reason_; }

private:
  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtMutator::VisitStmt_(op));
    auto mode = loop->annotations.Get(kPipelineMode);
    bool multidim_mode =
        mode.has_value() && Downcast<String>(mode.value()) == "multidim_ilp";
    auto region = loop->annotations.Get(kMultidimRegion);
    bool is_applied = false;
    if (region.has_value() || multidim_mode) {
      auto applied = loop->annotations.Get(kPipelineApplied);
      is_applied =
          applied.has_value() && Downcast<Bool>(applied.value())->value;
      bool has_cross_outer_overlap = true;
      if (stage_ == "planning" && is_applied) {
        auto strategy = loop->annotations.Get(kMultidimStrategy);
        auto summary_valid = loop->annotations.Get(kHierarchicalSummaryValid);
        bool hierarchical_summary =
            strategy.has_value() &&
            Downcast<String>(strategy.value()) == kHierarchicalStrategy &&
            summary_valid.has_value() &&
            Downcast<Bool>(summary_valid.value())->value;
        if (!hierarchical_summary) {
          auto stage_count = loop->annotations.Get("stage_count");
          auto max_iter_offset =
              loop->annotations.Get("steady_state_max_iter_offset");
          has_cross_outer_overlap = false;
          if (stage_count.has_value() && max_iter_offset.has_value()) {
            IntImm stage_count_imm = Downcast<IntImm>(stage_count.value());
            IntImm max_iter_offset_imm =
                Downcast<IntImm>(max_iter_offset.value());
            has_cross_outer_overlap =
                stage_count_imm->value >= 2 && max_iter_offset_imm->value > 0;
          }
        }
      }
      if (!is_applied || !has_cross_outer_overlap) {
        failed_ = true;
        if (!has_cross_outer_overlap) {
          reason_ = "no_cross_outer_overlap";
        } else {
          auto fallback = loop->annotations.Get(kPipelineFallbackReason);
          reason_ = fallback.has_value()
                        ? std::string(Downcast<String>(fallback.value()))
                        : stage_ + "_not_applied";
        }

        auto original = loop->annotations.Get(kMultidimOriginalLoop);
        if (policy_ == "auto" && original.has_value()) {
          For restored = Downcast<For>(original.value());
          return MarkOneDimensionalFallback(restored, stage_, reason_);
        }
      }
      if (policy_ == "auto" && stage_ == "planning" && is_applied &&
          has_cross_outer_overlap) {
        auto original = loop->annotations.Get(kMultidimOriginalLoop);
        if (original.has_value()) {
          For restored = Downcast<For>(original.value());
          restored.CopyOnWrite()->body =
              MarkPipelineArtifactsSuppressed()(restored->body);
          Map<String, Any> annotations = restored->annotations;
          annotations.Set(kMultidimCandidateLoop, loop);
          restored.CopyOnWrite()->annotations = annotations;
          return restored;
        }
      }
    }

    Map<String, Any> annotations;
    for (const auto &kv : loop->annotations) {
      const bool keep_original_for_inject = stage_ == "planning" && is_applied;
      if (kv.first != kMultidimOriginalLoop || keep_original_for_inject) {
        annotations.Set(kv.first, kv.second);
      }
    }
    loop.CopyOnWrite()->annotations = annotations;
    return loop;
  }

  std::string policy_;
  std::string stage_;
  bool failed_{false};
  std::string reason_;
};

std::optional<int64_t> AnnotationInt(const For &loop, const char *key) {
  auto value = loop->annotations.Get(key);
  if (!value.has_value()) {
    return std::nullopt;
  }
  return Downcast<IntImm>(value.value())->value;
}

std::optional<For> FindAppliedOneDimensionalLoop(const Stmt &stmt) {
  std::optional<For> result;
  PostOrderVisit(stmt, [&](const ObjectRef &obj) {
    if (result.has_value()) {
      return;
    }
    const auto *node = obj.as<ForNode>();
    if (node == nullptr) {
      return;
    }
    For loop = ffi::GetRef<For>(node);
    auto mode = loop->annotations.Get(kPipelineMode);
    auto applied = loop->annotations.Get(kPipelineApplied);
    if (!mode.has_value() || !applied.has_value()) {
      return;
    }
    if (Downcast<String>(mode.value()) == "ilp" &&
        Downcast<Bool>(applied.value())->value &&
        loop->annotations.Get("ii").has_value() &&
        loop->annotations.Get("makespan").has_value()) {
      result = loop;
    }
  });
  return result;
}

class HierarchicalSummaryBuilder : public StmtMutator {
public:
  explicit HierarchicalSummaryBuilder(std::string policy)
      : policy_(std::move(policy)) {
    ICHECK(policy_ == "forced" || policy_ == "auto");
  }

private:
  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtMutator::VisitStmt_(op));
    auto strategy = loop->annotations.Get(kMultidimStrategy);
    if (!strategy.has_value() ||
        Downcast<String>(strategy.value()) != kHierarchicalStrategy) {
      return loop;
    }

    std::optional<For> inner = FindAppliedOneDimensionalLoop(loop->body);
    if (!inner.has_value()) {
      ICHECK_NE(policy_, "forced")
          << "hierarchical multidim inner 1D planning failed";
      auto original = loop->annotations.Get(kMultidimOriginalLoop);
      if (original.has_value()) {
        return MarkOneDimensionalFallback(Downcast<For>(original.value()),
                                          "inner_summary",
                                          "inner_pipeline_not_applied");
      }
      return MarkOneDimensionalFallback(loop, "inner_summary",
                                        "inner_pipeline_not_applied");
    }

    auto ii = AnnotationInt(inner.value(), "ii");
    auto makespan = AnnotationInt(inner.value(), "makespan");
    auto stage_count = AnnotationInt(inner.value(), "stage_count");
    if (!ii || !makespan || !stage_count) {
      ICHECK_NE(policy_, "forced")
          << "hierarchical multidim inner summary is incomplete";
      auto original = loop->annotations.Get(kMultidimOriginalLoop);
      if (original.has_value()) {
        return MarkOneDimensionalFallback(Downcast<For>(original.value()),
                                          "inner_summary",
                                          "inner_summary_incomplete");
      }
      return MarkOneDimensionalFallback(loop, "inner_summary",
                                        "inner_summary_incomplete");
    }

    Map<String, Any> annotations = loop->annotations;
    annotations.Set(kPipelineRequested, Bool(true));
    annotations.Set(kPipelineApplied, Bool(true));
    annotations.Set(kPipelineMode, String("multidim_ilp"));
    annotations.Set(kMultidimSelectedMode, String("multidim_ilp"));
    annotations.Set(kHierarchicalSummaryValid, Bool(true));
    annotations.Set(kHierarchicalInnerII, Integer(*ii));
    annotations.Set(kHierarchicalInnerMakespan, Integer(*makespan));
    annotations.Set(kHierarchicalInnerStageCount, Integer(*stage_count));
    auto inner_extent = AnnotationInt(loop, kMultidimInnerExtent);
    if (inner_extent.has_value()) {
      int64_t tile_score =
          *makespan + std::max<int64_t>(0, *inner_extent - 1) * *ii;
      // Keep auto conservative until measured outer-phase latency is
      // available.  Forced mode still builds and benchmarks the outer
      // pipeline, while auto compares this no-overlap upper bound to 1D.
      annotations.Set("ii", Integer(tile_score));
      annotations.Set("makespan", Integer(tile_score));
      annotations.Set("stage_count", Integer(2));
      annotations.Set("steady_state_max_iter_offset", Integer(1));
    }
    loop.CopyOnWrite()->annotations = annotations;
    return loop;
  }

  std::string policy_;
};

class HierarchicalOuterPipelineBuilder : public StmtMutator {
public:
  explicit HierarchicalOuterPipelineBuilder(std::string policy)
      : policy_(std::move(policy)) {
    ICHECK(policy_ == "forced" || policy_ == "auto");
  }

private:
  Stmt Reject(const For &loop, const std::string &reason) const {
    ICHECK_NE(policy_, "forced")
        << "hierarchical outer pipeline construction failed: " << reason;
    auto original = loop->annotations.Get(kMultidimOriginalLoop);
    if (original.has_value()) {
      return MarkOneDimensionalFallback(Downcast<For>(original.value()),
                                        "outer_pipeline", reason);
    }
    return MarkOneDimensionalFallback(loop, "outer_pipeline", reason);
  }

  void AppendLocalBuffer(const Buffer &buffer, Array<Buffer> *buffers,
                         std::unordered_set<const BufferNode *> *seen) const {
    String scope = buffer.scope();
    if (scope.empty() || scope == "global" || seen->count(buffer.get())) {
      return;
    }
    seen->insert(buffer.get());
    buffers->push_back(buffer);
  }

  Array<Buffer> CollectLocalBuffers(const Array<Stmt> &phases) const {
    Array<Buffer> buffers;
    std::unordered_set<const BufferNode *> seen;
    for (const Stmt &phase : phases) {
      PostOrderVisit(phase, [&](const ObjectRef &obj) {
        if (const auto *store = obj.as<BufferStoreNode>()) {
          AppendLocalBuffer(store->buffer, &buffers, &seen);
          return;
        }
        if (const auto *load = obj.as<BufferLoadNode>()) {
          AppendLocalBuffer(load->buffer, &buffers, &seen);
          return;
        }
        const auto *call = obj.as<CallNode>();
        if (call != nullptr && call->op.same_as(RegionOp::Get())) {
          RegionOp region(call->args);
          AppendLocalBuffer(region->GetBuffer(), &buffers, &seen);
        }
      });
    }
    return buffers;
  }

  struct BufferTouchSummary {
    // Buffer objects are logical views. Distinct views may still alias the
    // same physical allocation, so motion must be keyed by storage root.
    std::unordered_set<const VarNode *> storage_roots;
    bool reorderable{true};
  };

  BufferTouchSummary CollectBufferTouches(const Stmt &stmt,
                                          bool validate_prefix) const {
    BufferTouchSummary summary;
    PostOrderVisit(stmt, [&](const ObjectRef &obj) {
      if (const auto *store = obj.as<BufferStoreNode>()) {
        summary.storage_roots.insert(store->buffer->data.get());
        return;
      }
      if (const auto *load = obj.as<BufferLoadNode>()) {
        summary.storage_roots.insert(load->buffer->data.get());
        return;
      }
      if (const auto *call = obj.as<CallNode>()) {
        if (call->op.same_as(RegionOp::Get())) {
          RegionOp region(call->args);
          summary.storage_roots.insert(region->GetBuffer()->data.get());
        } else if (validate_prefix) {
          // Keep the motion conservative. Prefix calls may have effects that
          // are not represented by buffer regions.
          summary.reorderable = false;
        }
        return;
      }
      if (validate_prefix &&
          (obj.as<AssertStmtNode>() || obj.as<WhileNode>() ||
           obj.as<AllocateNode>() || obj.as<AllocateConstNode>() ||
           obj.as<BufferRealizeNode>())) {
        summary.reorderable = false;
      }
    });
    return summary;
  }

  bool CanMovePrefetchBeforePrefix(const Array<Stmt> &prefix,
                                   const Stmt &prefetch) const {
    if (prefix.empty()) {
      return false;
    }
    BufferTouchSummary prefetch_summary = CollectBufferTouches(prefetch, false);
    if (prefetch_summary.storage_roots.empty()) {
      return false;
    }
    for (const Stmt &stmt : prefix) {
      BufferTouchSummary prefix_summary = CollectBufferTouches(stmt, true);
      if (!prefix_summary.reorderable) {
        return false;
      }
      for (const VarNode *root : prefix_summary.storage_roots) {
        if (prefetch_summary.storage_roots.count(root) != 0) {
          return false;
        }
      }
    }
    return true;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtMutator::VisitStmt_(op));
    auto strategy = loop->annotations.Get(kMultidimStrategy);
    if (!strategy.has_value() ||
        Downcast<String>(strategy.value()) != kHierarchicalStrategy) {
      return loop;
    }
    auto summary = loop->annotations.Get(kHierarchicalSummaryValid);
    if (!summary.has_value() || !Downcast<Bool>(summary.value())->value) {
      return Reject(loop, "inner_summary_missing");
    }

    const auto *body = loop->body.as<SeqStmtNode>();
    if (body == nullptr) {
      return Reject(loop, "outer_body_not_sequence");
    }

    Array<Stmt> prefix;
    Array<Stmt> suffix;
    Optional<Stmt> inner_prefetch;
    Optional<Stmt> inner_compute;
    bool between_phases = false;
    std::string malformed_reason;
    for (const Stmt &stmt : body->seq) {
      if (const auto *attr = stmt.as<AttrStmtNode>()) {
        if (attr->attr_key == kHierarchicalInnerPhase) {
          const auto *phase = attr->value.as<IntImmNode>();
          if (phase == nullptr) {
            malformed_reason = "inner_phase_not_constant";
            break;
          }
          if (phase->value == 0 && !inner_prefetch.defined() &&
              !inner_compute.defined()) {
            inner_prefetch = attr->body;
            between_phases = true;
            continue;
          }
          if (phase->value == 1 && inner_prefetch.defined() &&
              !inner_compute.defined()) {
            inner_compute = attr->body;
            between_phases = false;
            continue;
          }
          malformed_reason = "inner_phase_order_invalid";
          break;
        }
      }
      if (!inner_prefetch.defined()) {
        prefix.push_back(stmt);
      } else if (!inner_compute.defined() || between_phases) {
        malformed_reason = "unmarked_statement_between_inner_phases";
        break;
      } else {
        suffix.push_back(stmt);
      }
    }

    if (!malformed_reason.empty()) {
      return Reject(loop, malformed_reason);
    }
    if (!inner_prefetch.defined() || !inner_compute.defined()) {
      return Reject(loop, "inner_phase_markers_missing");
    }
    if (suffix.empty()) {
      return Reject(loop, "outer_store_phase_missing");
    }

    bool prefetch_before_prefix = false;
    if (const char *raw =
            std::getenv("TL_SUNMMIO_MULTIDIM_PREFETCH_BEFORE_PREFIX")) {
      prefetch_before_prefix =
          std::atoi(raw) != 0 &&
          CanMovePrefetchBeforePrefix(prefix, inner_prefetch.value());
    }
    Array<Stmt> prefetch_parts;
    if (prefetch_before_prefix) {
      prefetch_parts.push_back(inner_prefetch.value());
    }
    for (const Stmt &stmt : prefix) {
      prefetch_parts.push_back(stmt);
    }
    if (!prefetch_before_prefix) {
      prefetch_parts.push_back(inner_prefetch.value());
    }
    Stmt prefetch_phase =
        AttrStmt(loop->loop_var, kHierarchicalOuterPhase, Integer(0),
                 SeqStmt::Flatten(prefetch_parts));
    Stmt compute_phase = AttrStmt(loop->loop_var, kHierarchicalOuterPhase,
                                  Integer(1), inner_compute.value());
    Stmt store_phase = AttrStmt(loop->loop_var, kHierarchicalOuterPhase,
                                Integer(2), SeqStmt::Flatten(suffix));
    Array<Stmt> phases{prefetch_phase, compute_phase, store_phase};
    Array<Buffer> local_buffers = CollectLocalBuffers(phases);
    if (local_buffers.empty()) {
      return Reject(loop, "outer_local_buffers_missing");
    }

    auto inner_ii = AnnotationInt(loop, kHierarchicalInnerII);
    auto inner_makespan = AnnotationInt(loop, kHierarchicalInnerMakespan);
    auto inner_extent = AnnotationInt(loop, kMultidimInnerExtent);
    if (!inner_ii || !inner_makespan || !inner_extent) {
      return Reject(loop, "inner_summary_incomplete");
    }
    int64_t tile_score =
        *inner_makespan + std::max<int64_t>(0, *inner_extent - 1) * *inner_ii;

    int outer_versions = 2;
    if (const char *raw = std::getenv("TL_SUNMMIO_MULTIDIM_OUTER_VERSIONS")) {
      int parsed = std::atoi(raw);
      ICHECK_GE(parsed, 2)
          << "TL_SUNMMIO_MULTIDIM_OUTER_VERSIONS must be at least 2";
      ICHECK_LE(parsed, 4)
          << "TL_SUNMMIO_MULTIDIM_OUTER_VERSIONS must not exceed 4";
      outer_versions = parsed;
    }

    Map<String, Any> annotations = loop->annotations;
    annotations.Set(kHierarchicalOuterBuilt, Bool(true));
    annotations.Set(kHierarchicalPrefetchBeforePrefix,
                    Bool(prefetch_before_prefix));
    annotations.Set(kHierarchicalOuterVersions, Integer(outer_versions));
    annotations.Set(kPipelineRequested, Bool(true));
    annotations.Set(kPipelineApplied, Bool(true));
    annotations.Set(kPipelineMode, String("multidim_ilp"));
    annotations.Set(kMultidimSelectedMode, String("multidim_ilp"));
    annotations.Set("iterations", Integer(outer_versions));
    annotations.Set("ii", Integer(tile_score));
    annotations.Set("makespan", Integer(tile_score));
    annotations.Set("stage_count", Integer(2));
    annotations.Set("steady_state_max_iter_offset", Integer(1));
    annotations.Set("prologue_orders", Array<String>{String("0-0")});
    annotations.Set("body_orders",
                    Array<String>{String("1-0"), String("0-1"), String("0-2")});
    annotations.Set("epilogue_orders",
                    Array<String>{String("1-1"), String("1-2")});
    annotations.Set("used_buffers", local_buffers);
    annotations.Set("versioned_buffers", local_buffers);
    annotations.Set("runtime_multiversion_buffers", local_buffers);

    loop.CopyOnWrite()->body = SeqStmt(phases);
    loop.CopyOnWrite()->annotations = annotations;
    return loop;
  }

  std::string policy_;
};

class MultidimCandidateSelector : public StmtMutator {
public:
  explicit MultidimCandidateSelector(std::string policy)
      : policy_(std::move(policy)) {
    ICHECK(policy_ == "forced" || policy_ == "auto");
  }

private:
  For AnnotateSelection(const For &input, const std::string &selected_mode,
                        int64_t multidim_score, int64_t one_dim_score,
                        int min_gain_percent,
                        const std::string &fallback_reason =
                            "predicted_not_better_than_1d") const {
    For selected = Downcast<For>(StripCandidatePrivateAnnotations()(input));
    Map<String, Any> annotations = selected->annotations;
    annotations.Set(kMultidimSelectedMode, String(selected_mode));
    annotations.Set(kMultidimPredictedScore, Integer(multidim_score));
    annotations.Set(kOneDimPredictedScore, Integer(one_dim_score));
    annotations.Set(kMultidimMinGainPercent, Integer(min_gain_percent));
    if (selected_mode == "ilp") {
      annotations.Set(kMultidimFallbackStage, String("candidate_selection"));
      annotations.Set(kMultidimFallbackReason, String(fallback_reason));
    } else {
      annotations.erase(kMultidimFallbackStage);
      annotations.erase(kMultidimFallbackReason);
    }
    selected.CopyOnWrite()->annotations = annotations;
    return selected;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtMutator::VisitStmt_(op));
    auto candidate_value = loop->annotations.Get(kMultidimCandidateLoop);
    if (!candidate_value.has_value()) {
      return loop;
    }
    ICHECK_EQ(policy_, "auto");
    For multidim = Downcast<For>(candidate_value.value());
    std::optional<For> one_dim = FindAppliedOneDimensionalLoop(loop->body);
    if (!one_dim.has_value()) {
      return AnnotateSelection(loop, "ilp", -1, -1, 2,
                               "one_dim_cost_unavailable");
    }

    auto multidim_ii = AnnotationInt(multidim, "ii");
    auto multidim_makespan = AnnotationInt(multidim, "makespan");
    auto one_dim_ii = AnnotationInt(one_dim.value(), "ii");
    auto one_dim_makespan = AnnotationInt(one_dim.value(), "makespan");
    auto outer_extent = AnnotationInt(multidim, kMultidimOuterExtent);
    auto inner_extent = AnnotationInt(multidim, kMultidimInnerExtent);
    if (!multidim_ii || !multidim_makespan || !one_dim_ii ||
        !one_dim_makespan || !outer_extent || !inner_extent) {
      return AnnotateSelection(loop, "ilp", -1, -1, 2,
                               "incomplete_candidate_cost");
    }

    int min_gain_percent = 2;
    if (const char *raw = std::getenv("TL_SUNMMIO_MULTIDIM_MIN_GAIN_PERCENT")) {
      min_gain_percent = std::clamp(std::atoi(raw), 0, 99);
    }
    auto checked_score = [](int64_t makespan, int64_t extent,
                            int64_t ii) -> std::optional<int64_t> {
      __int128 value = static_cast<__int128>(makespan) +
                       static_cast<__int128>(std::max<int64_t>(0, extent - 1)) *
                           static_cast<__int128>(ii);
      if (value < std::numeric_limits<int64_t>::min() ||
          value > std::numeric_limits<int64_t>::max()) {
        return std::nullopt;
      }
      return static_cast<int64_t>(value);
    };
    std::optional<int64_t> multidim_score =
        checked_score(*multidim_makespan, *outer_extent, *multidim_ii);
    std::optional<int64_t> one_dim_tile_score =
        checked_score(*one_dim_makespan, *inner_extent, *one_dim_ii);
    __int128 one_dim_wide = one_dim_tile_score
                                ? static_cast<__int128>(*outer_extent) *
                                      static_cast<__int128>(*one_dim_tile_score)
                                : static_cast<__int128>(-1);
    if (!multidim_score || !one_dim_tile_score || one_dim_wide < 0 ||
        one_dim_wide > std::numeric_limits<int64_t>::max()) {
      return AnnotateSelection(loop, "ilp", -1, -1, min_gain_percent,
                               "candidate_cost_overflow");
    }
    int64_t one_dim_score = static_cast<int64_t>(one_dim_wide);
    // Require the multidimensional candidate to save at least the configured
    // percentage. Score construction and comparison both use wide arithmetic.
    bool choose_multidim =
        static_cast<__int128>(*multidim_score) * 100 <=
        static_cast<__int128>(one_dim_score) * (100 - min_gain_percent);
    return choose_multidim
               ? AnnotateSelection(multidim, "multidim_ilp", *multidim_score,
                                   one_dim_score, min_gain_percent)
               : AnnotateSelection(loop, "ilp", *multidim_score, one_dim_score,
                                   min_gain_percent);
  }

  std::string policy_;
};

} // namespace

tvm::transform::Pass AnalyzeSunmmioBoundaryOverlap(String policy) {
  using namespace tir::transform;
  std::string value = policy;
  ICHECK(value == "off" || value == "shadow" || value == "auto" ||
         value == "forced")
      << "invalid boundary overlap policy: " << value;
  auto pass_func = [=](PrimFunc f, const IRModule &, PassContext) {
    if (value == "off" || !boundary_overlap::Enabled())
      return f;
    BoundaryOverlapAnnotator annotator(f, value);
    PrimFunc result = f;
    result.CopyOnWrite()->body = annotator(f->body);
    return result;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AnalyzeSunmmioBoundaryOverlap",
                            {});
}

tvm::transform::Pass SunmmioMultidimRegion(String policy) {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &, PassContext) {
    MultidimRegionNormalizer normalizer{std::string(policy)};
    PrimFunc result = f;
    result.CopyOnWrite()->body = normalizer(f->body);
    WriteRegionEvents(normalizer.events());
    return result;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.SunmmioMultidimRegion", {});
}

tvm::transform::Pass ApplySunmmioMultidimInterchange(String policy) {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &, PassContext) {
    bool proved_generic = false;
    PostOrderVisit(f->body, [&](const ObjectRef &node) {
      if (const auto *loop = node.as<ForNode>())
        proved_generic =
            proved_generic || loop->annotations.count(residency::kEligible);
    });
    if ((!InterchangeEnabled() || std::string(policy) != "forced") &&
        !proved_generic) {
      return f;
    }
    try {
      InterchangeLoopBuilder builder{std::string(policy)};
      PrimFunc result = f;
      result.CopyOnWrite()->body = builder(f->body);
      builder.RewriteFunctionLayoutAttrs(result);
      return result;
    } catch (const std::exception &error) {
      auto original = f->GetAttr<String>(residency::kGroupingOriginal);
      if (std::string(policy) != "auto" || !original)
        throw;
      LOG(WARNING) << "[SunmmioResidency][AtomicFallback] interchange: "
                   << error.what();
      return WithAttr(residency::RestoreGroupingFunction(original.value()),
                      "tl.sunmmio.residency.grouping_fallback",
                      String("interchange_exception"));
    }
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ApplySunmmioMultidimInterchange",
                            {});
}

tvm::transform::Pass BuildSunmmioHierarchicalSummary(String policy) {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &, PassContext) {
    HierarchicalSummaryBuilder builder{std::string(policy)};
    PrimFunc result = f;
    result.CopyOnWrite()->body = builder(f->body);
    return result;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.BuildSunmmioHierarchicalSummary",
                            {});
}

tvm::transform::Pass BuildSunmmioHierarchicalOuterPipeline(String policy) {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &, PassContext) {
    HierarchicalOuterPipelineBuilder builder{std::string(policy)};
    PrimFunc result = f;
    result.CopyOnWrite()->body = builder(f->body);
    return result;
  };
  return CreatePrimFuncPass(pass_func, 0,
                            "tl.BuildSunmmioHierarchicalOuterPipeline", {});
}

tvm::transform::Pass ValidateSunmmioMultidimPipeline(String policy,
                                                     String stage) {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &, PassContext) {
    MultidimValidationMutator validator{std::string(policy),
                                        std::string(stage)};
    PrimFunc result = f;
    result.CopyOnWrite()->body = validator(f->body);
    ICHECK(!(std::string(policy) == "forced" && validator.failed()))
        << "multidim_ilp forced validation failed at " << stage << ": "
        << validator.reason();
    if (auto original = f->GetAttr<String>(residency::kGroupingOriginal)) {
      bool failed = validator.failed();
      std::string reason = validator.reason();
      if (std::string(stage) == "planning" ||
          std::string(stage) == "artifact") {
        int cores = 0;
        PostOrderVisit(result->body, [&](const ObjectRef &node) {
          if (const auto *loop = node.as<ForNode>()) {
            if (!loop->annotations.count(residency::kGroupedCore))
              return;
            ++cores;
            auto applied = loop->annotations.Get(kPipelineApplied);
            failed =
                failed || !applied || !Downcast<Bool>(applied.value())->value;
          }
        });
        auto expected = f->GetAttr<Integer>(residency::kGroupingCoreCount);
        failed = failed || !expected || cores != expected.value()->value;
        if (reason.empty())
          reason = "grouped_core_not_planned";
      } else {
        auto fallback = PipelineFallbackValidator::FindDisallowed(result->body);
        failed = failed || fallback.has_value() ||
                 !result->HasNonzeroAttr("tl.sunmmio.multidim.applied");
        if (reason.empty())
          reason = "grouped_injection_failed";
      }
      if (failed) {
        ICHECK_NE(std::string(policy), "forced")
            << "grouped residency validation failed: " << reason;
        LOG(WARNING) << "[SunmmioResidency][AtomicFallback] " << stage << ": "
                     << reason;
        // Restore the complete function, including buffer allocations and
        // layout maps, not just the core loop. The caller replans ordinary1D.
        return WithAttr(residency::RestoreGroupingFunction(original.value()),
                        "tl.sunmmio.residency.grouping_fallback",
                        String(reason));
      }
      if (std::string(stage) == "inject") {
        result = WithoutAttr(std::move(result),
                             String(residency::kGroupingOriginal));
        result = WithoutAttr(std::move(result),
                             String(residency::kGroupingCoreCount));
      }
    }
    return result;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ValidateSunmmioMultidimPipeline",
                            {});
}
tvm::transform::Pass SelectSunmmioMultidimPipeline(String policy) {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &, PassContext) {
    MultidimCandidateSelector selector{std::string(policy)};
    PrimFunc result = f;
    result.CopyOnWrite()->body = selector(f->body);
    return result;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.SelectSunmmioMultidimPipeline",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.transform.AnalyzeSunmmioBoundaryOverlap",
           AnalyzeSunmmioBoundaryOverlap)
      .def("tl.transform.ApplySunmmioMultidimInterchange",
           ApplySunmmioMultidimInterchange)
      .def("tl.transform.SunmmioMultidimRegion", SunmmioMultidimRegion)
      .def("tl.transform.BuildSunmmioHierarchicalSummary",
           BuildSunmmioHierarchicalSummary)
      .def("tl.transform.BuildSunmmioHierarchicalOuterPipeline",
           BuildSunmmioHierarchicalOuterPipeline)
      .def("tl.transform.ValidateSunmmioMultidimPipeline",
           ValidateSunmmioMultidimPipeline)
      .def("tl.transform.SelectSunmmioMultidimPipeline",
           SelectSunmmioMultidimPipeline);
}

} // namespace tl
} // namespace tvm
