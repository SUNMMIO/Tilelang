/*!
 * \file tile_residency_analysis.h
 * \brief Bounded, side-effect-aware proof for reuse-axis tile residency.
 *
 * This is an analysis only.  It neither changes synchronization nor assumes
 * that fewer transfers imply a measured speedup.  Existing buffer allocation,
 * backend verification, and the pipeline cost model remain mandatory.
 */
#ifndef TVM_TL_TRANSFORM_TILE_RESIDENCY_ANALYSIS_H_
#define TVM_TL_TRANSFORM_TILE_RESIDENCY_ANALYSIS_H_

#include "../../target/sunmmio/sunmmio_async_op_info.h"
#include "stmt_read_write_collector.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <optional>
#include <utility>
#include <map>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>
#include <tvm/arith/analyzer.h>
#include <tvm/node/structural_equal.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tl {
namespace residency {

using namespace tir;

constexpr const char *kEligible = "tl.sunmmio.residency.generic_eligible";
constexpr const char *kApplied = "tl.sunmmio.residency.generic_applied";
constexpr const char *kStateBuffers = "tl.sunmmio.residency.state_buffers";
constexpr const char *kPrefixCount = "tl.sunmmio.residency.shared_prefix_count";
constexpr const char *kReason = "tl.sunmmio.residency.reject_reason";
constexpr const char *kGroupWidth = "tl.sunmmio.residency.group_width";
constexpr const char *kGroupedCore = "tl.sunmmio.residency.grouped_core";
constexpr const char *kGroupingOriginal =
    "tl.sunmmio.residency.grouping_original_func";
constexpr const char *kGroupingCoreCount =
    "tl.sunmmio.residency.grouping_core_count";

inline std::string Policy() {
  const char *raw = std::getenv("TL_SUNMMIO_RESIDENCY_POLICY");
  std::string policy = raw && *raw ? raw : "auto_strict";
  ICHECK(policy == "off" || policy == "shadow" || policy == "auto_strict" ||
         policy == "forced") << "invalid tile residency policy: " << policy;
  return policy;
}

inline int64_t Budget(const char *key, int64_t fallback) {
  const char *raw = std::getenv(key);
  if (!raw || !*raw) return fallback;
  char *end = nullptr;
  int64_t value = std::strtoll(raw, &end, 10);
  ICHECK(end && !*end && value >= 0) << "invalid residency budget " << key;
  return value;
}

inline bool Uses(const ObjectRef &object, const Var &var) {
  bool found = false;
  PostOrderVisit(object, [&](const ObjectRef &node) {
    if (const auto *candidate = node.as<VarNode>())
      found = found || candidate == var.get();
  });
  return found;
}

inline std::optional<int64_t> StaticInt(const PrimExpr &expr) {
  arith::Analyzer analyzer;
  PrimExpr value = analyzer.Simplify(expr);
  if (const auto *imm = value.as<IntImmNode>()) return imm->value;
  return std::nullopt;
}

inline bool SameStorage(const Buffer &a, const Buffer &b) {
  return a->data.same_as(b->data);
}

inline bool RegionUses(const BufferRegion &region, const Var &var) {
  for (const Range &range : region->region)
    if (Uses(range->min, var) || Uses(range->extent, var)) return true;
  for (const PrimExpr &dim : region->buffer->shape)
    if (Uses(dim, var)) return true;
  for (const PrimExpr &stride : region->buffer->strides)
    if (Uses(stride, var)) return true;
  return Uses(region->buffer->elem_offset, var);
}

inline bool SameRegion(const BufferRegion &a, const BufferRegion &b) {
  if (!SameStorage(a->buffer, b->buffer) ||
      a->region.size() != b->region.size()) return false;
  arith::Analyzer analyzer;
  for (size_t i = 0; i < a->region.size(); ++i) {
    if (!analyzer.CanProveEqual(a->region[i]->min, b->region[i]->min) ||
        !analyzer.CanProveEqual(a->region[i]->extent, b->region[i]->extent))
      return false;
  }
  return true;
}

inline Array<Stmt> Statements(const Stmt &stmt) {
  Stmt flat = SeqStmt::Flatten(Array<Stmt>{stmt});
  if (const auto *seq = flat.as<SeqStmtNode>()) return seq->seq;
  return Array<Stmt>{flat};
}

inline std::optional<int64_t> Bytes(const Buffer &buffer) {
  int64_t bytes = (buffer->dtype.bits() * buffer->dtype.lanes() + 7) / 8;
  for (const PrimExpr &dim : buffer->shape) {
    auto n = StaticInt(dim);
    if (!n || *n <= 0 || bytes > std::numeric_limits<int64_t>::max() / *n)
      return std::nullopt;
    bytes *= *n;
  }
  return bytes;
}

struct Operation {
  Stmt stmt;
  Call call;
  bool transfer{false};
  SunmmioAsyncOpInfo async;
  std::vector<BufferRegion> reads;
  std::vector<BufferRegion> writes;
};

using ScalarSet = std::set<const VarNode *>;

// Small metadata buffers are not automatically pure: prove a dominating
// assignment, local ownership, no alias/escape and no candidate-region writes.
class DominatingScalarDefinitions : public StmtVisitor {
public:
  explicit DominatingScalarDefinitions(const ForNode *target) : target_(target) {}
  ScalarSet at_entry;
private:
  void VisitStmt_(const ForNode *op) final {
    if (op == target_) {
      // TIR is a DAG: the same For object can appear at distinct statement
      // positions. A later initialized occurrence cannot justify an earlier
      // uninitialized one. The proof must hold at every occurrence.
      if (!seen_target_) at_entry = defined_;
      else {
        for (auto it = at_entry.begin(); it != at_entry.end();) {
          if (!defined_.count(*it)) it = at_entry.erase(it);
          else ++it;
        }
      }
      seen_target_ = true;
      return;
    }
    ScalarSet before = defined_;
    VisitStmt(op->body);
    defined_ = std::move(before);  // A preceding loop may execute zero times.
  }
  void VisitStmt_(const WhileNode *op) final {
    ScalarSet before = defined_;
    VisitStmt(op->body);
    defined_ = std::move(before);
  }
  void VisitStmt_(const IfThenElseNode *op) final {
    ScalarSet before = defined_;
    VisitStmt(op->then_case);
    ScalarSet then_defs = defined_;
    defined_ = before;
    if (op->else_case.defined()) VisitStmt(op->else_case.value());
    for (auto it = defined_.begin(); it != defined_.end();) {
      if (!then_defs.count(*it)) it = defined_.erase(it);
      else ++it;
    }
  }
  void VisitStmt_(const BlockRealizeNode *op) final {
    ScalarSet before = defined_;
    VisitStmt(op->block->body);
    if (!is_one(op->predicate)) defined_ = std::move(before);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    if (op->buffer.scope() == "local.var" && op->indices.size() == 1 &&
        is_zero(op->indices[0]) &&
        (!op->predicate.defined() || is_one(op->predicate.value())))
      defined_.insert(op->buffer->data.get());
  }
  const ForNode *target_;
  ScalarSet defined_;
  bool seen_target_{false};
};

class ScalarEscapeCheck : public StmtExprVisitor {
public:
  explicit ScalarEscapeCheck(ScalarSet candidates) : candidates_(std::move(candidates)) {}
  ScalarSet escaped;
private:
  void VisitExpr_(const VarNode *op) final {
    if (candidates_.count(op)) escaped.insert(op);
  }
  void VisitExpr_(const CallNode *op) final {
    const auto *name = op->op.as<OpNode>();
    bool known = name && (name->name == "tl.tileop.region" ||
                          name->name == "tl.odma_unit" ||
                          name->name == "tl.dma_copy" ||
                          name->name == "tl.broadcast_" ||
                          name->name == "tl.mma_sunmmio");
    if (!known) {
      // address_of(BufferLoad(...)) escapes without a raw data Var argument.
      PostOrderVisit(ffi::GetRef<Call>(op), [&](const ObjectRef &node) {
        if (const auto *load = node.as<BufferLoadNode>())
          if (candidates_.count(load->buffer->data.get()))
            escaped.insert(load->buffer->data.get());
      });
    }
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == "volatile_scope") {
      if (const auto *data = op->node.as<VarNode>())
        if (candidates_.count(data)) escaped.insert(data);
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  ScalarSet candidates_;
};

inline ScalarSet ProveReadonlyScalars(const PrimFunc &func, const For &reuse) {
  ScalarSet candidates, allocated, invalid;
  std::map<const VarNode *, const BufferNode *> views;
  auto record = [&](const Buffer &buffer) {
    if (buffer.scope() != "local.var" || !buffer->dtype.is_int() ||
        buffer->dtype.lanes() != 1 || buffer->shape.size() != 1 ||
        !is_one(buffer->shape[0]) || !is_zero(buffer->elem_offset) ||
        (!buffer->strides.empty() &&
         (buffer->strides.size() != 1 || !is_one(buffer->strides[0])))) return;
    candidates.insert(buffer->data.get());
  };
  PostOrderVisit(reuse->body, [&](const ObjectRef &node) {
    if (const auto *load = node.as<BufferLoadNode>()) record(load->buffer);
  });
  if (candidates.empty()) return candidates;
  PostOrderVisit(func->body, [&](const ObjectRef &node) {
    auto view = [&](const Buffer &buffer) {
      if (!candidates.count(buffer->data.get())) return;
      auto it = views.emplace(buffer->data.get(), buffer.get());
      if (!it.second && it.first->second != buffer.get()) invalid.insert(buffer->data.get());
    };
    if (const auto *block = node.as<BlockNode>()) {
      for (const Buffer &buffer : block->alloc_buffers) {
        allocated.insert(buffer->data.get());
        view(buffer);
      }
      for (const MatchBufferRegion &match : block->match_buffers) {
        // Match-buffer aliases can introduce a different data Var for the
        // same storage. This bounded proof does not chase alias chains.
        invalid.insert(match->source->buffer->data.get());
        invalid.insert(match->buffer->data.get());
      }
    }
    if (const auto *load = node.as<BufferLoadNode>()) view(load->buffer);
    if (const auto *store = node.as<BufferStoreNode>()) view(store->buffer);
    if (const auto *call = node.as<CallNode>()) {
      SunmmioAsyncOpInfo info;
      // An asynchronous write issued before entry may still be in flight.
      if (TryParseSunmmioAsyncODMAOp(call, &info)) invalid.insert(info.dst->buffer->data.get());
      else if (call->op.same_as(mma_sunmmio()) &&
               (call->args.size() == 7U || call->args.size() == 9U)) {
        invalid.insert(NormalizeToBufferRegion(call->args[2])->buffer->data.get());
        if (call->args.size() == 9U)
          invalid.insert(NormalizeToBufferRegion(call->args[7])->buffer->data.get());
      }
    }
  });
  for (const Var &parameter : func->params) invalid.insert(parameter.get());
  PostOrderVisit(reuse->body, [&](const ObjectRef &node) {
    if (const auto *store = node.as<BufferStoreNode>())
      invalid.insert(store->buffer->data.get());
    if (const auto *call = node.as<CallNode>()) {
      SunmmioAsyncOpInfo info;
      if (TryParseSunmmioAsyncODMAOp(call, &info)) invalid.insert(info.dst->buffer->data.get());
      else if (call->op.same_as(mma_sunmmio()) &&
               (call->args.size() == 7U || call->args.size() == 9U)) {
        invalid.insert(NormalizeToBufferRegion(call->args[2])->buffer->data.get());
        if (call->args.size() == 9U)
          invalid.insert(NormalizeToBufferRegion(call->args[7])->buffer->data.get());
      }
    }
  });
  DominatingScalarDefinitions definitions(reuse.get());
  definitions(func->body);
  ScalarEscapeCheck escape(candidates);
  escape(func->body);
  for (auto it = candidates.begin(); it != candidates.end();) {
    if (!allocated.count(*it) || !definitions.at_entry.count(*it) ||
        invalid.count(*it) || escape.escaped.count(*it)) it = candidates.erase(it);
    else ++it;
  }
  return candidates;
}

class CommandArgumentCheck : public ExprVisitor {
public:
  explicit CommandArgumentCheck(const ScalarSet *readonly = nullptr) : readonly_(readonly) {}
  bool pure{true};
  std::vector<BufferRegion> scalar_reads;
private:
  void VisitExpr_(const BufferLoadNode *load) final {
    if (!readonly_ || !readonly_->count(load->buffer->data.get()) ||
        load->indices.size() != 1 || !is_zero(load->indices[0]) ||
        (load->predicate.defined() && !is_one(load->predicate.value()))) {
      pure = false;
      return;
    }
    scalar_reads.push_back(BufferRegion::FullRegion(load->buffer));
  }
  void VisitExpr_(const CallNode *call) final {
    const auto *op = call->op.as<OpNode>();
    if (!op) { pure = false; return; }
    if (op->name == "tl.tileop.region") {
      if (call->args.empty()) { pure = false; return; }
      const auto *base = call->args[0].as<BufferLoadNode>();
      if (!base) { pure = false; return; }
      // A region's base BufferLoad describes an address, not a scalar read.
      // Indices/extents can read only explicitly proved immutable metadata.
      for (const PrimExpr &index : base->indices) VisitExpr(index);
      for (size_t i = 1; i < call->args.size(); ++i) VisitExpr(call->args[i]);
    } else if (op->name == "tl.odma_unit") {
      for (const PrimExpr &arg : call->args) VisitExpr(arg);
    } else {
      pure = false;
    }
  }
  const ScalarSet *readonly_;
};

inline bool PureCommandArguments(const CallNode *call,
                                 const ScalarSet *readonly = nullptr,
                                 std::vector<BufferRegion> *scalar_reads = nullptr) {
  CommandArgumentCheck check(readonly);
  for (const PrimExpr &argument : call->args) check(argument);
  if (scalar_reads) *scalar_reads = check.scalar_reads;
  return check.pure;
}

// Only transparent blocks may surround a core command.  In particular an
// if/else with one matching call is not an unconditional command.
inline bool ParseOperation(const Stmt &statement, Operation *result,
                            const ScalarSet *readonly = nullptr) {
  Stmt leaf = statement;
  while (const auto *realize = leaf.as<BlockRealizeNode>()) {
    if (!is_one(realize->predicate) || !realize->iter_values.empty() ||
        !realize->block->alloc_buffers.empty() ||
        !realize->block->match_buffers.empty() ||
        realize->block->init.defined()) return false;
    leaf = realize->block->body;
  }
  const auto *eval = leaf.as<EvaluateNode>();
  const auto *call = eval ? eval->value.as<CallNode>() : nullptr;
  if (!call || !PureCommandArguments(call, readonly, &result->reads)) return false;
  result->stmt = statement;
  result->call = ffi::GetRef<Call>(call);
  if (TryParseSunmmioAsyncODMAOp(call, &result->async)) {
    result->transfer = true;
    result->reads.push_back(result->async.src);
    result->writes.push_back(result->async.dst);
    if (result->async.kind == SunmmioAsyncOpKind::kBroadcast) {
      if (!result->async.direction.defined() ||
          !StaticInt(result->async.direction) ||
          !result->async.mask.defined() || !StaticInt(result->async.mask))
        return false;
    }
    return true;
  }
  if (!call->op.same_as(mma_sunmmio()) ||
      (call->args.size() != 7U && call->args.size() != 9U)) return false;
  for (int i = 0; i < 3; ++i)
    result->reads.push_back(NormalizeToBufferRegion(call->args[i]));
  result->writes.push_back(NormalizeToBufferRegion(call->args[2]));
  if (call->args.size() == 9U)
    result->writes.push_back(NormalizeToBufferRegion(call->args[7]));
  return true;
}

// Prefix/suffix may contain normal scalar/vector stores, pure loops and
// data-only conditions.  Unknown calls, allocation, collective epochs and
// control exits are never moved across the reduction core.
class BoundaryCheck : public StmtExprVisitor {
public:
  explicit BoundaryCheck(const ScalarSet *readonly = nullptr) : readonly_(readonly) {}
  bool valid{true};
  void VisitExpr_(const CallNode *op) final {
    // Pure arithmetic is safe to move with its explicitly collected buffer
    // reads. Do not classify the whole expression as kPure: BufferLoad is
    // kReadState even when the enclosing math operator has no side effects.
    static auto effects = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
    auto builtin = op->op.as<Op>();
    if (!builtin.has_value() || !effects.count(builtin.value()) ||
        op->dtype.is_handle()) {
      valid = false;
      return;
    }
    int64_t effect = effects[builtin.value()]->value;
    if (effect != static_cast<int>(CallEffectKind::kPure) &&
        effect != static_cast<int>(CallEffectKind::kExprAnnotation)) {
      valid = false;
      return;
    }
    for (const PrimExpr &arg : op->args) {
      if (arg.dtype().is_handle()) {
        valid = false;
        return;
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const EvaluateNode *op) final {
    SunmmioAsyncOpInfo info;
    const auto *call = op->value.as<CallNode>();
    if (TryParseSunmmioAsyncODMAOp(call, &info)) {
      if (info.kind == SunmmioAsyncOpKind::kBroadcast ||
          !PureCommandArguments(call, readonly_)) valid = false;
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const ForNode *op) final {
    if ((op->kind != ForKind::kSerial && op->kind != ForKind::kVectorized &&
         op->kind != ForKind::kUnrolled) || op->thread_binding.defined() ||
        op->annotations.count("num_stages")) valid = false;
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const BlockRealizeNode *op) final {
    if (!is_one(op->predicate) || !op->iter_values.empty() ||
        !op->block->alloc_buffers.empty() ||
        !op->block->match_buffers.empty() || op->block->init.defined())
      valid = false;
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const IfThenElseNode *op) final {
    bool has_load = false;
    PostOrderVisit(op->condition, [&](const ObjectRef &node) {
      has_load = has_load || node.as<BufferLoadNode>() || node.as<CallNode>();
    });
    if (has_load) valid = false;
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const WhileNode *) final { valid = false; }
  void VisitStmt_(const AllocateNode *) final { valid = false; }
  void VisitStmt_(const AttrStmtNode *) final { valid = false; }
  void VisitStmt_(const AssertStmtNode *) final { valid = false; }
private:
  const ScalarSet *readonly_;
};

struct Proof {
  bool accepted{false};
  std::string reason;
  std::string reuse_loop;
  int64_t reuse_extent{0};
  int64_t group_width{0};
  int shared_prefix_count{0};
  int consumer_count{0};
  int core_index{-1};
  int64_t added_bytes{0};
  int64_t saved_transfer_bytes{0};
  std::string resource_check;
  Array<Buffer> state_buffers;
  Array<Stmt> before;
  Array<Stmt> after;
  Array<Stmt> core_statements;
  For core;
};

// This is an Auto profitability prefilter, not a legality proof.  A candidate
// needs enough eliminated reuse instances to repay each invariant transfer in
// the shared prefix across the available consumers.  Complete whole-kernel
// costing still happens later; forced/shadow modes may inspect weaker cases.
inline bool HasMinimumReuseAmortization(const Proof &proof) {
  if (proof.reuse_extent < 2 || proof.shared_prefix_count <= 0 ||
      proof.consumer_count <= 0)
    return false;
  int64_t groups =
      proof.group_width ? 1 + (proof.reuse_extent - 1) / proof.group_width : 1;
  int64_t saved_instances = proof.reuse_extent - groups;
  if (saved_instances <= 0) return false;
  int64_t required_instances =
      1 + (proof.shared_prefix_count - 1) / proof.consumer_count;
  return saved_instances >= required_instances;
}

inline bool ContainsStorage(const Array<Buffer> &buffers, const Buffer &buffer) {
  for (const Buffer &item : buffers)
    if (SameStorage(item, buffer)) return true;
  return false;
}

inline bool IsSlot(const BufferRegion &region, const For &reuse) {
  if (region->region.empty() || region->buffer->shape.empty()) return false;
  arith::Analyzer analyzer;
  auto extent = StaticInt(region->region[0]->extent);
  auto capacity = StaticInt(region->buffer->shape[0]);
  auto count = StaticInt(reuse->extent);
  return extent && *extent == 1 && capacity && count &&
         analyzer.CanProveEqual(region->region[0]->min,
                               reuse->loop_var - reuse->min) &&
         *capacity >= *count;
}


inline std::optional<BufferRegion>
CollectiveCoverage(const SunmmioAsyncOpInfo &info, const PrimFunc &func) {
  BufferRegion full = BufferRegion::FullRegion(info.dst->buffer);
  if (SameRegion(info.dst, full)) return full;
  // A single-source broadcast does not prove that all rank slots are filled.
  auto target = func->GetAttr<Target>("target");
  if (info.src_core.defined() || !target.has_value()) return std::nullopt;
  auto direction = StaticInt(info.direction);
  auto mask = StaticInt(info.mask);
  if (!direction || (*direction != 0 && *direction != 1) || !mask)
    return std::nullopt;
  SunmmioMeshConfig mesh = GetSunmmioMeshConfig(target.value());
  int width = *direction == 0 ? mesh.ncol : mesh.nrow;
  if (width < 2 || width >= 63 || *mask != ((int64_t{1} << width) - 1))
    return std::nullopt;
  Var core_id;
  PostOrderVisit(func->body, [&](const ObjectRef &node) {
    if (const auto *attr = node.as<AttrStmtNode>()) {
      if (attr->attr_key == "thread_extent") {
        if (const auto *iter = attr->node.as<IterVarNode>())
          if (iter->thread_tag == "blockIdx.x") core_id = iter->var;
      }
    }
  });
  if (!core_id.defined()) return std::nullopt;
  arith::Analyzer analyzer;
  int shard_axis_count = 0;
  for (size_t i = 0; i < info.dst->region.size(); ++i) {
    const Range &range = info.dst->region[i];
    if (analyzer.CanProveEqual(range->min, 0) &&
        analyzer.CanProveEqual(range->extent, info.dst->buffer->shape[i]))
      continue;
    auto chunk = StaticInt(range->extent);
    if (!chunk || *chunk <= 0 || ++shard_axis_count != 1 ||
        !analyzer.CanProveEqual(info.dst->buffer->shape[i], range->extent * width))
      return std::nullopt;
    PrimExpr rank = *direction == 0
                       ? floormod(core_id, Integer(mesh.ncol))
                       : floordiv(core_id, Integer(mesh.ncol));
    if (!analyzer.CanProveEqual(range->min, rank * range->extent))
      return std::nullopt;
  }
  return shard_axis_count == 1 ? std::optional<BufferRegion>(full) : std::nullopt;
}

class StateEscapeCheck : public StmtExprVisitor {
public:
  StateEscapeCheck(const ForNode *region, const Array<Buffer> &states)
      : region_(region), states_(states) {}
  bool escapes{false};
private:
  void VisitStmt_(const ForNode *op) final {
    if (op != region_) StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const BufferLoadNode *op) final {
    escapes = escapes || ContainsStorage(states_, op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    escapes = escapes || ContainsStorage(states_, op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }
  const ForNode *region_;
  Array<Buffer> states_;
};

class FullInitializationCheck : public StmtVisitor {
public:
  explicit FullInitializationCheck(const Buffer &buffer) : buffer_(buffer) {}
  bool full{false};
private:
  void VisitStmt_(const ForNode *op) final {
    if (op->step.has_value() && !is_one(op->step.value())) return;
    domains_[op->loop_var.get()] = {op->min, op->extent};
    StmtVisitor::VisitStmt_(op);
    domains_.erase(op->loop_var.get());
  }
  void VisitStmt_(const IfThenElseNode *) final {}
  bool CoversDimension(const PrimExpr &index, const PrimExpr &shape,
                       std::set<const VarNode *> *used_axes) const {
    auto size = StaticInt(shape);
    if (!size || *size <= 0) return false;
    arith::Analyzer analyzer;
    Map<Var, PrimExpr> zeros;
    Map<Var, PrimExpr> singletons;
    for (const auto &[axis, domain] : domains_) {
      auto minimum = StaticInt(domain.first);
      auto extent = StaticInt(domain.second);
      if (!minimum || *minimum != 0 || !extent || *extent <= 0) return false;
      Var var = ffi::GetRef<Var>(axis);
      zeros.Set(var, make_const(var.dtype(), 0));
      if (*extent == 1) singletons.Set(var, make_const(var.dtype(), 0));
    }
    PrimExpr normalized = analyzer.Simplify(Substitute(index, singletons));
    PrimExpr base = analyzer.Simplify(Substitute(normalized, zeros));
    if (!is_zero(base)) return false;
    PrimExpr reconstructed = base;
    struct Digit { int64_t stride; int64_t extent; const VarNode *axis; };
    std::vector<Digit> digits;
    for (const auto &[axis, domain] : domains_) {
      auto extent = StaticInt(domain.second);
      if (*extent == 1) continue;
      Var var = ffi::GetRef<Var>(axis);
      Map<Var, PrimExpr> unit = zeros;
      unit.Set(var, make_const(var.dtype(), 1));
      auto stride = StaticInt(analyzer.Simplify(Substitute(normalized, unit)));
      if (!stride || *stride < 0) return false;
      if (*stride == 0) continue;
      reconstructed = reconstructed + var * make_const(var.dtype(), *stride);
      digits.push_back({*stride, *extent, axis});
    }
    // Equality rejects nonlinear expressions that merely match at 0 and 1.
    if (!analyzer.CanProveEqual(normalized, reconstructed)) return false;
    std::sort(digits.begin(), digits.end(),
              [](const Digit &a, const Digit &b) { return a.stride < b.stride; });
    int64_t covered = 1;
    for (const Digit &digit : digits) {
      // Exact mixed-radix strides prove no gaps/duplicates. Axes cannot be
      // shared across dimensions (which would only clear a diagonal).
      if (digit.stride != covered || !used_axes->insert(digit.axis).second ||
          covered > std::numeric_limits<int64_t>::max() / digit.extent)
        return false;
      covered *= digit.extent;
    }
    return covered == *size;
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    const auto *float_value = op->value.as<FloatImmNode>();
    bool zero_value = is_zero(op->value) || (float_value && float_value->value == 0.0);
    if (!SameStorage(op->buffer, buffer_) || !zero_value ||
        (op->predicate.defined() && !is_one(op->predicate.value())) ||
        op->indices.size() != buffer_->shape.size()) return;
    std::set<const VarNode *> axes;
    for (size_t i = 0; i < op->indices.size(); ++i) {
      if (!CoversDimension(op->indices[i], buffer_->shape[i], &axes)) return;
    }
    full = true;
  }
  Buffer buffer_;
  std::map<const VarNode *, std::pair<PrimExpr, PrimExpr>> domains_;
};

inline Proof Analyze(const PrimFunc &func, const For &reuse, bool slotted,
                     bool allow_grouping = false) {
  Proof proof;
  proof.reuse_loop = reuse->loop_var->name_hint;
  auto reject = [&](const char *reason) {
    proof.reason = reason;
    return proof;
  };
  auto minimum = StaticInt(reuse->min);
  auto extent = StaticInt(reuse->extent);
  if (!minimum || !extent) return reject("dynamic_reuse_range");
  proof.reuse_extent = *extent;
  if (*extent < 2) return reject("reuse_extent_too_short");
  int64_t expanded_extent = *extent;
  int64_t reuse_budget = Budget("TL_SUNMMIO_RESIDENCY_MAX_REUSE_EXTENT", 8);
  if (*extent > reuse_budget) {
    int64_t group_width = Budget("TL_SUNMMIO_RESIDENCY_GROUP_WIDTH", 2);
    if (!allow_grouping || slotted || group_width < 2 || group_width > 8 ||
        group_width > reuse_budget)
      return reject("expansion_budget");
    if (*extent > std::numeric_limits<int32_t>::max())
      return reject("grouping_loop_range_unsupported");
    // All semantic checks below still inspect the ORIGINAL complete region.
    // Only the simultaneously live output-state expansion is bounded here.
    proof.group_width = group_width;
    expanded_extent = group_width;
  }
  if (reuse->step.has_value() && !is_one(reuse->step.value())) return reject("non_unit_loop_step");
  if (reuse->kind != ForKind::kSerial || reuse->thread_binding.defined())
    return reject("non_serial_reuse_loop");
  auto body = Statements(reuse->body);
  for (size_t i = 0; i < body.size(); ++i) {
    const auto *loop = body[i].as<ForNode>();
    if (loop && loop->annotations.count("num_stages")) {
      if (proof.core.defined()) return reject("non_unique_pipeline_core");
      proof.core = ffi::GetRef<For>(loop);
      proof.core_index = static_cast<int>(i);
    }
  }
  if (!proof.core.defined()) return reject("missing_direct_pipeline_core");
  const For &core = proof.core;
  auto k_min = StaticInt(core->min);
  auto k_extent = StaticInt(core->extent);
  if (!k_min || !k_extent || *k_extent < 1 ||
      Uses(core->min, reuse->loop_var) || Uses(core->extent, reuse->loop_var))
    return reject("dynamic_core_range");
  if (core->step.has_value() && !is_one(core->step.value())) return reject("non_unit_loop_step");
  if (core->kind != ForKind::kSerial || core->thread_binding.defined())
    return reject("non_serial_core_loop");
  for (int i = 0; i < proof.core_index; ++i) proof.before.push_back(body[i]);
  for (size_t i = proof.core_index + 1; i < body.size(); ++i)
    proof.after.push_back(body[i]);
  ScalarSet readonly_scalars = ProveReadonlyScalars(func, reuse);
  BoundaryCheck boundary(&readonly_scalars);
  for (const Stmt &stmt : proof.before) boundary(stmt);
  for (const Stmt &stmt : proof.after) boundary(stmt);
  if (!boundary.valid) return reject("prefix_or_suffix_side_effect");

  proof.core_statements = Statements(core->body);
  std::vector<Operation> operations;
  for (const Stmt &stmt : proof.core_statements) {
    Operation operation;
    if (!ParseOperation(stmt, &operation, &readonly_scalars))
      return reject("core_control_or_side_effect");
    if (operation.transfer &&
        operation.async.kind == SunmmioAsyncOpKind::kBroadcast) {
      auto coverage = CollectiveCoverage(operation.async, func);
      if (!coverage) return reject("collective_coverage_unproven");
      operation.writes[0] = coverage.value();
    }
    operations.push_back(std::move(operation));
  }
  if (operations.empty()) return reject("empty_pipeline_core");
  for (const Operation &operation : operations) {
    for (const BufferRegion &write : operation.writes)
      if (write->buffer.scope().empty() || write->buffer.scope() == "global")
        return reject("core_global_side_effect");
  }

  // Different Buffer views of one allocation are not treated as independent.
  // Reject ambiguous views instead of relying on BufferNode/name equality.
  std::map<const VarNode *, const BufferNode *> views;
  bool ambiguous_alias = false;
  auto record_view = [&](const Buffer &buffer) {
    auto entry = views.emplace(buffer->data.get(), buffer.get());
    if (!entry.second && entry.first->second != buffer.get())
      ambiguous_alias = true;
  };
  PostOrderVisit(reuse->body, [&](const ObjectRef &node) {
    if (const auto *load = node.as<BufferLoadNode>()) record_view(load->buffer);
    if (const auto *store = node.as<BufferStoreNode>()) record_view(store->buffer);
  });
  if (ambiguous_alias) return reject("ambiguous_buffer_alias");

  for (const Operation &operation : operations) {
    if (operation.transfer) continue;
    ++proof.consumer_count;
    for (const BufferRegion &region : operation.writes) {
      if (region->buffer.scope() != "shared.rsram")
        return reject("reduction_storage_unsupported");
      if (slotted ? !IsSlot(region, reuse) : RegionUses(region, reuse->loop_var))
        return reject("reduction_state_not_independent");
      if (!ContainsStorage(proof.state_buffers, region->buffer))
        proof.state_buffers.push_back(region->buffer);
    }
    if (Uses(operation.call->args[5], reuse->loop_var) ||
        (operation.call->args.size() == 9U &&
         Uses(operation.call->args[8], reuse->loop_var)))
      return reject("reduction_predicate_depends_on_reuse");
    arith::Analyzer analyzer;
    PrimExpr clear = operation.call->args[5];
    if (!is_zero(clear) &&
        !analyzer.CanProveEqual(clear, core->loop_var == core->min))
      return reject("clear_not_first_reduction");
    if (operation.call->args.size() == 9U &&
        !analyzer.CanProveEqual(operation.call->args[8],
                               core->loop_var == core->min + core->extent - 1))
      return reject("output_not_last_reduction");
  }
  if (proof.consumer_count == 0) return reject("no_mma_consumer");

  // Share only an invariant leading transfer chain whose writes are not
  // overwritten later.  This handles stage->ASRAM and stage-only reuse when
  // BF16 legalization rewrites ASRAM between two MMA half-passes.
  for (size_t i = 0; i < operations.size(); ++i) {
    const Operation &operation = operations[i];
    if (!operation.transfer || Uses(operation.stmt, reuse->loop_var)) break;
    bool conflict = false;
    for (const BufferRegion &write : operation.writes) {
      if (ContainsStorage(proof.state_buffers, write->buffer)) conflict = true;
      for (size_t j = i + 1; j < operations.size(); ++j)
        for (const BufferRegion &later : operations[j].writes)
          if (SameStorage(write->buffer, later->buffer)) conflict = true;
    }
    if (conflict) break;
    ++proof.shared_prefix_count;
  }
  if (proof.shared_prefix_count == 0 ||
      proof.shared_prefix_count == static_cast<int>(operations.size()))
    return reject("no_invariant_transfer_prefix");

  StmtReadWriteCollector boundary_rw(func);
  boundary_rw.traverse_stmt(SeqStmt::Flatten(proof.before));
  auto before_reads = boundary_rw.read_buffer_regions_;
  auto before_writes = boundary_rw.write_buffer_regions_;
  boundary_rw.clear();
  boundary_rw.traverse_stmt(SeqStmt::Flatten(proof.after));
  auto after_reads = boundary_rw.read_buffer_regions_;
  auto after_writes = boundary_rw.write_buffer_regions_;
  bool global_read = false, global_write = false;
  for (const Operation &operation : operations)
    for (const BufferRegion &read : operation.reads)
      global_read = global_read || read->buffer.scope().empty() ||
                    read->buffer.scope() == "global";
  for (const BufferRegion &write : after_writes)
    global_write = global_write || write->buffer.scope().empty() ||
                     write->buffer.scope() == "global";
  if (global_read && global_write) {
    bool noalias = func->HasNonzeroAttr("tir.noalias");
    auto nonrestrict = func->GetAttr<Array<Var>>("tl.non_restrict_params");
    if (!noalias ||
        (nonrestrict.has_value() && !nonrestrict.value().empty()))
      return reject("global_alias_unproven");
  }

  // Prefix is initialization only.  Moving all N initializers before K is
  // safe only when they write independent state and read no shared state.
  if (!before_reads.empty()) return reject("prefix_reads_shared_state");
  for (const BufferRegion &write : before_writes) {
    if (!ContainsStorage(proof.state_buffers, write->buffer))
      return reject("prefix_writes_non_state");
    if (slotted && !IsSlot(write, reuse))
      return reject("prefix_cross_slot_dependency");
  }

  // Suffix is delayed, not reordered among N instances.  It may use a shared
  // epilogue scratch buffer but cannot feed back into any core input/state.
  for (const BufferRegion &write : after_writes) {
    if (ContainsStorage(proof.state_buffers, write->buffer)) {
      if (slotted && !IsSlot(write, reuse))
        return reject("suffix_cross_slot_dependency");
      continue;  // Epilogue updates this N's independent state/version.
    }
    for (const Operation &operation : operations) {
      for (const BufferRegion &read : operation.reads)
        if (SameStorage(write->buffer, read->buffer))
          return reject("producer_source_modified");
      for (const BufferRegion &core_write : operation.writes)
        if (SameStorage(write->buffer, core_write->buffer))
          return reject("suffix_overwrites_core_state");
    }
  }
  for (const BufferRegion &read : after_reads) {
    if (ContainsStorage(proof.state_buffers, read->buffer)) {
      if (slotted && !IsSlot(read, reuse))
        return reject("suffix_cross_slot_dependency");
      continue;
    }
    for (const Operation &operation : operations)
      for (const BufferRegion &write : operation.writes)
        if (SameStorage(read->buffer, write->buffer))
          return reject("suffix_reads_ephemeral_core_tile");
  }

  // A resident source must be invariant in value, not just in its address.
  for (int i = 0; i < proof.shared_prefix_count; ++i) {
    for (const BufferRegion &read : operations[i].reads) {
      for (size_t j = i; j < operations.size(); ++j)
        for (const BufferRegion &write : operations[j].writes)
          if (SameStorage(read->buffer, write->buffer))
            return reject("producer_source_modified");
    }
    auto bytes = Bytes(operations[i].async.dst->buffer);
    int64_t groups = proof.group_width ? 1 + (*extent - 1) / proof.group_width : 1;
    int64_t saved_instances = *extent - groups;
    if (!bytes || *bytes > std::numeric_limits<int64_t>::max() / saved_instances ||
        proof.saved_transfer_bytes > std::numeric_limits<int64_t>::max() -
                                          *bytes * saved_instances)
      return reject("transfer_size_overflow");
    proof.saved_transfer_bytes += *bytes * saved_instances;
  }

  // Every local operand must be defined in this K step, or be an independent
  // accumulator.  Incoming read-only staging buffers are allowed only if no
  // statement in the entire reuse region writes their allocation.
  std::vector<BufferRegion> definitions;
  for (const Operation &operation : operations) {
    if (operation.transfer && ContainsStorage(proof.state_buffers,
                                               operation.async.dst->buffer))
      return reject("transfer_overwrites_reduction_state");
    for (const BufferRegion &read : operation.reads) {
      if (ContainsStorage(proof.state_buffers, read->buffer)) {
        if (slotted ? !IsSlot(read, reuse) : RegionUses(read, reuse->loop_var))
          return reject("reduction_state_not_independent");
        continue;
      }
      bool written_here = false;
      for (const Operation &other : operations)
        for (const BufferRegion &write : other.writes)
          written_here = written_here || SameStorage(read->buffer, write->buffer);
      if (!written_here) continue;
      bool defined = false;
      for (const BufferRegion &definition : definitions)
        defined = defined || SameRegion(read, definition);
      if (!defined) return reject("cross_iteration_scratch_dependency");
    }
    for (const BufferRegion &write : operation.writes) {
      // The newest write kills earlier aliases even if the new region is
      // smaller.  Partial coverage is rejected conservatively.
      definitions.erase(std::remove_if(definitions.begin(), definitions.end(),
        [&](const BufferRegion &old) { return SameStorage(old->buffer, write->buffer); }),
        definitions.end());
      definitions.push_back(write);
    }
  }

  for (const Operation &operation : operations) {
    if (operation.transfer || !is_zero(operation.call->args[5])) continue;
    Buffer acc = operation.writes[0]->buffer;
    FullInitializationCheck initialization(acc);
    for (const Stmt &stmt : proof.before) initialization(stmt);
    if (!initialization.full) return reject("accumulator_not_fully_initialized");
  }

  if (!slotted) {
    std::set<const VarNode *> allocated;
    PostOrderVisit(func->body, [&](const ObjectRef &node) {
      if (const auto *block = node.as<BlockNode>())
        for (const Buffer &buffer : block->alloc_buffers)
          allocated.insert(buffer->data.get());
    });
    for (const Buffer &buffer : proof.state_buffers) {
      if (!allocated.count(buffer->data.get()))
        return reject("state_not_local_allocation");
      auto bytes = Bytes(buffer);
      if (!bytes || *bytes > std::numeric_limits<int64_t>::max() / (expanded_extent - 1) ||
          proof.added_bytes > std::numeric_limits<int64_t>::max() - *bytes * (expanded_extent - 1))
        return reject("allocation_size_overflow");
      proof.added_bytes += *bytes * (expanded_extent - 1);
    }
    StateEscapeCheck escape(reuse.get(), proof.state_buffers);
    escape(func->body);
    if (escape.escapes) return reject("reduction_state_escapes_region");
    if (proof.added_bytes > Budget("TL_SUNMMIO_RESIDENCY_MAX_ADDED_BYTES", 1048576))
      return reject("allocation_budget");
    proof.resource_check = "bounded_expansion_backend_required";
  } else {
    proof.resource_check = "existing_allocations";
  }
  proof.accepted = true;
  proof.reason = "proved";
  return proof;
}

} // namespace residency
} // namespace tl
} // namespace tvm
#endif
