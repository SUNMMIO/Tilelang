/*!
 * \file sunmmio_mma_clear_fusion.h
 * \brief Conservative, opt-in full +0 fill / first-MMA clear fusion.
 *
 * Internal to LegalizeSunmmioGemm. This is a shared backend optimization, not
 * a Multidim-only optimization. Preserve the logical K domain and ordering;
 * do not fuse output casts. Unsupported proof forms are left unchanged.
 */
#ifndef TILELANG_TRANSFORM_SUNMMIO_MMA_CLEAR_FUSION_H_
#define TILELANG_TRANSFORM_SUNMMIO_MMA_CLEAR_FUSION_H_

#include <tvm/arith/analyzer.h>
#include <tvm/node/structural_equal.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt_functor.h>

#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../target/sunmmio_utils.h"

namespace tvm {
namespace tl {
namespace sunmmio_mma_clear_fusion {

using namespace tir;

inline std::string Name(const CallNode *call) {
  const auto *op = call->op.as<OpNode>();
  return op ? std::string(op->name) : "<external>";
}

inline bool IsGemm(const CallNode *call) {
  return Name(call) == "tl.tileop.gemm" || Name(call) == "tl.tileop.gemm_py";
}

inline bool IsTransfer(const CallNode *call) {
  const std::string name = Name(call);
  return name == "tl.tileop.copy" || name == "tl.tileop.comm_allgather" ||
         name == "tl.dma_copy" || name == "tl.broadcast_" ||
         name == "tl.sunmmio_layout_transform" || name == "tl.sunmmio_transpose";
}

inline std::optional<int64_t> Integer(const PrimExpr &expr) {
  arith::Analyzer analyzer;
  PrimExpr value = analyzer.Simplify(expr);
  if (const auto *imm = value.as<IntImmNode>()) return imm->value;
  return std::nullopt;
}

inline bool IsInteger(const PrimExpr &expr, int64_t expected) {
  auto value = Integer(expr);
  return value && *value == expected;
}

struct Region {
  Buffer buffer;
  Array<PrimExpr> starts;
  Array<PrimExpr> extents;
};

// Do not use NormalizeToBufferRegion here: unsupported Ramps must reject the
// optional optimization, rather than reach that utility's fatal assertions.
inline std::optional<Region> DecodeRegion(const PrimExpr &expr) {
  if (const auto *call = expr.as<CallNode>()) {
    if (Name(call) != "tl.tileop.region" || call->args.size() < 2 ||
        !call->annotations.empty()) return std::nullopt;
    const auto *load = call->args[0].as<BufferLoadNode>();
    if (!load || load->predicate.defined() ||
        call->args.size() != load->indices.size() + 2) return std::nullopt;
    for (const PrimExpr &index : load->indices)
      if (index.as<RampNode>()) return std::nullopt;
    Array<PrimExpr> extents;
    for (size_t i = 2; i < call->args.size(); ++i) extents.push_back(call->args[i]);
    return Region{load->buffer, load->indices, extents};
  }
  const auto *load = expr.as<BufferLoadNode>();
  if (!load || load->predicate.defined()) return std::nullopt;
  Array<PrimExpr> starts, extents;
  bool has_ramp = false;
  for (const PrimExpr &index : load->indices) {
    if (const auto *ramp = index.as<RampNode>()) {
      const auto *stride = ramp->stride.as<IntImmNode>();
      const auto *lanes = ramp->lanes.as<IntImmNode>();
      if (!stride || stride->value != 1 || !lanes || lanes->value <= 0)
        return std::nullopt;
      starts.push_back(ramp->base);
      extents.push_back(ramp->lanes);
      has_ramp = true;
    } else {
      starts.push_back(index);
      extents.push_back(IntImm(index.dtype(), 1));
    }
  }
  if (!has_ramp) return std::nullopt;
  return Region{load->buffer, starts, extents};
}

inline bool IsFullRankTwo(const Region &region) {
  if (region.buffer->shape.size() != 2 || region.starts.size() != 2 ||
      region.extents.size() != 2) return false;
  for (size_t i = 0; i < 2; ++i) {
    auto shape = Integer(region.buffer->shape[i]);
    if (!shape || *shape <= 0 || !IsInteger(region.starts[i], 0) ||
        !IsInteger(region.extents[i], *shape)) return false;
  }
  return true;
}

inline bool IsPositiveZero(const PrimExpr &expr) {
  if (const auto *imm = expr.as<IntImmNode>()) return imm->value == 0;
  if (const auto *imm = expr.as<FloatImmNode>())
    return imm->value == 0.0 && !std::signbit(imm->value);
  return false;
}

class FunctionSafety : public StmtExprVisitor {
public:
  explicit FunctionSafety(Buffer accumulator) : accumulator_(accumulator) {}
  bool valid{true};
  bool locally_owned{false};
  size_t same_storage_gemms{0};

private:
  Buffer accumulator_;

  void InspectBuffer(const Buffer &buffer) {
    if (buffer->data.same_as(accumulator_->data) && !buffer.same_as(accumulator_))
      valid = false;
    for (const PrimExpr &shape : buffer->shape)
      if (!Integer(shape)) valid = false;
    for (const PrimExpr &stride : buffer->strides)
      if (!Integer(stride)) valid = false;
    if (!Integer(buffer->elem_offset)) valid = false;
  }

  void VisitStmt_(const BlockNode *op) final {
    if (!op->match_buffers.empty()) valid = false;
    for (const Buffer &buffer : op->alloc_buffers)
      if (buffer.same_as(accumulator_)) locally_owned = true;
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const LetStmtNode *op) final {
    if (op->var.dtype().is_handle() || op->value.dtype().is_handle()) valid = false;
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const LetNode *op) final {
    if (op->var.dtype().is_handle() || op->value.dtype().is_handle()) valid = false;
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const VarNode *op) final {
    if (op == accumulator_->data.get()) valid = false;
  }
  void VisitExpr_(const ProducerLoadNode *) final { valid = false; }
  void VisitExpr_(const BufferLoadNode *op) final {
    InspectBuffer(op->buffer);
    if (op->predicate.defined()) valid = false;
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    InspectBuffer(op->buffer);
    if (op->predicate.defined()) valid = false;
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const CallNode *op) final {
    const std::string name = Name(op);
    if (!op->annotations.empty() || name == "tir.call_pure_extern" ||
        name == "tir.call_pure_packed" || name == "tir.call_llvm_pure_intrin")
      valid = false;
    if (IsGemm(op)) {
      if (op->args.size() < 3) {
        valid = false;
      } else if (auto region = DecodeRegion(op->args[2])) {
        if (region->buffer->data.same_as(accumulator_->data)) ++same_storage_gemms;
      } else {
        valid = false;
      }
    } else if (!IsTransfer(op) && name != "tl.tileop.region" &&
               name != "tl.tileop.fill" && name != "tl.odma_unit") {
      static auto effects = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
      auto builtin = op->op.as<Op>();
      if (!builtin.has_value() || !effects.count(builtin.value()) ||
          op->dtype.is_handle()) {
        valid = false;
      } else {
        int64_t effect = effects[builtin.value()]->value;
        if (effect != static_cast<int>(CallEffectKind::kPure) &&
            effect != static_cast<int>(CallEffectKind::kExprAnnotation)) valid = false;
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }
};

class HiddenAccumulatorAccess : public StmtExprVisitor {
public:
  explicit HiddenAccumulatorAccess(Var data) : data_(data) {}
  bool found{false};

private:
  Var data_;
  void VisitExpr_(const VarNode *op) final {
    if (op == data_.get()) found = true;
  }
  void VisitExpr_(const BufferLoadNode *op) final {
    if (op->buffer->data.same_as(data_)) found = true;
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    if (op->buffer->data.same_as(data_)) found = true;
    StmtExprVisitor::VisitStmt_(op);
  }
};

struct Plan {
  Buffer accumulator;
  Call gemm;
  For loop;
  bool dual_stripe;
  int64_t second_offset;
};

inline bool LayoutIsReviewed(Target target, DataType dtype) {
  // The diagnostic's reviewed target contract is a 32 x 32 layout tile.
  // Do not mistake that shape contract for a dtype equivalence proof.
  if (dtype != DataType::BFloat(16) && dtype != DataType::Float(32)) return false;
  auto shape = GetSunmmioLayoutBlockShape(target, dtype);
  return shape.size() == 2 && IsInteger(shape[0], 32) && IsInteger(shape[1], 32);
}

inline std::optional<Plan> Prove(const PrimFunc &func, const Stmt &init,
                                 const Stmt &next, Target target,
                                 const SunmmioTileProcessorConfig &cfg) {
  const auto *eval = init.as<EvaluateNode>();
  const auto *fill = eval ? eval->value.as<CallNode>() : nullptr;
  if (!fill || Name(fill) != "tl.tileop.fill" || fill->args.size() != 2 ||
      !IsPositiveZero(fill->args[1])) return std::nullopt;
  auto region = DecodeRegion(fill->args[0]);
  if (!region || !IsFullRankTwo(*region)) return std::nullopt;
  Buffer c = region->buffer;
  if (c.scope() != "shared.rsram" || c->dtype != DataType::Float(32) ||
      !c->strides.empty() || !IsInteger(c->elem_offset, 0)) return std::nullopt;
  const auto *loop = next.as<ForNode>();
  if (!loop || loop->kind != ForKind::kSerial || loop->thread_binding.defined() ||
      !Integer(loop->min) || !Integer(loop->extent) || *Integer(loop->extent) <= 0 ||
      (loop->step.has_value() && !IsInteger(loop->step.value(), 1))) return std::nullopt;
  Array<Stmt> body;
  if (const auto *sequence = loop->body.as<SeqStmtNode>()) body = sequence->seq;
  else body.push_back(loop->body);
  const CallNode *gemm = nullptr;
  for (const Stmt &stmt : body) {
    const auto *item = stmt.as<EvaluateNode>();
    const auto *call = item ? item->value.as<CallNode>() : nullptr;
    if (!call) return std::nullopt;
    if (IsGemm(call)) {
      if (gemm) return std::nullopt;
      gemm = call;
    }
  }
  if (!gemm || (gemm->args.size() != 19 && gemm->args.size() != 20)) return std::nullopt;
  auto gc = DecodeRegion(gemm->args[2]);
  if (!gc || !gc->buffer.same_as(c) || !IsFullRankTwo(*gc) ||
      !IsInteger(gemm->args[9], 0) || !IsInteger(gemm->args[17], 0) ||
      !IsInteger(gemm->args[18], 0) ||
      (gemm->args.size() == 20 && !IsInteger(gemm->args[19], 0))) return std::nullopt;
  auto m = Integer(gemm->args[5]), n = Integer(gemm->args[6]), k = Integer(gemm->args[7]);
  if (!m || !n || !k || *k <= 0 || !IsInteger(c->shape[0], *m) ||
      !IsInteger(c->shape[1], *n)) return std::nullopt;
  auto a = DecodeRegion(gemm->args[0]), b = DecodeRegion(gemm->args[1]);
  if (!a || !b || a->buffer->shape.size() != 2 || b->buffer->shape.size() != 2 ||
      a->buffer->data.same_as(c->data) || b->buffer->data.same_as(c->data) ||
      a->buffer.scope() != "shared.asram" || b->buffer.scope() != "shared.wsram" ||
      // Runtime evidence is limited to BF16 inputs and FP32 accumulation.
      // Other dtypes keep the original compilation path, without fusion.
      a->buffer->dtype != DataType::BFloat(16) || b->buffer->dtype != DataType::BFloat(16) ||
      !LayoutIsReviewed(target, a->buffer->dtype) || !LayoutIsReviewed(target, c->dtype) ||
      cfg.bf16_gemm_single_pass_max_rows != 16) return std::nullopt;
  if (!func->HasNonzeroAttr("tir.noalias")) return std::nullopt;
  auto nonrestrict = func->GetAttr<Array<Var>>("tl.non_restrict_params");
  if (nonrestrict.has_value() && !nonrestrict.value().empty()) return std::nullopt;
  FunctionSafety safety(c);
  safety(func->body);
  // Match the diagnostic's post-legalization whole-function stripe-count
  // restriction before changing anything (including multiple candidates).
  if (!safety.valid || !safety.locally_owned || safety.same_storage_gemms != 1)
    return std::nullopt;
  for (const Stmt &stmt : body) {
    const auto *call = stmt.as<EvaluateNode>()->value.as<CallNode>();
    if (call != gemm && !IsTransfer(call)) return std::nullopt;
    HiddenAccumulatorAccess hidden(c->data);
    for (size_t i = 0; i < call->args.size(); ++i)
      if (call != gemm || i != 2) hidden(call->args[i]);
    if (hidden.found) return std::nullopt;
  }
  return Plan{c, tvm::ffi::GetRef<Call>(gemm), tvm::ffi::GetRef<For>(loop),
              a->buffer->dtype == DataType::BFloat(16) && *m > 16, 2048};
}

class Rewriter : public StmtExprMutator {
public:
  Rewriter(PrimFunc func, Target target, const SunmmioTileProcessorConfig &cfg)
      : func_(func), target_(target), cfg_(cfg) {}
  std::vector<Plan> plans;

private:
  PrimFunc func_;
  Target target_;
  const SunmmioTileProcessorConfig &cfg_;

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    // Recurse first, as in the diagnostic. A parent candidate requires only
    // direct calls, so it cannot absorb a nested candidate's K loop.
    Stmt visited = StmtExprMutator::VisitStmt_(op);
    const auto *sequence = visited.as<SeqStmtNode>();
    if (!sequence) return visited;
    Array<Stmt> output;
    bool changed = false;
    for (size_t i = 0; i < sequence->seq.size(); ++i) {
      if (i + 1 < sequence->seq.size()) {
        auto plan = Prove(func_, sequence->seq[i], sequence->seq[i + 1], target_, cfg_);
        if (plan) {
          Call call = plan->gemm;
          Array<PrimExpr> args = call->args;
          args.Set(9, plan->loop->loop_var == plan->loop->min);
          Call replacement(call->dtype, call->op, args, call->annotations, call->span);
          class ReplaceCall : public StmtExprMutator {
          public:
            ReplaceCall(Call old_call, Call new_call) : old_(old_call), new_(new_call) {}
          private:
            Call old_, new_;
            PrimExpr VisitExpr_(const CallNode *node) final {
              if (node == old_.get()) return new_;
              return StmtExprMutator::VisitExpr_(node);
            }
          } replace(call, replacement);
          For loop = plan->loop;
          loop.CopyOnWrite()->body = replace(loop->body);
          output.push_back(loop);
          plans.push_back(*plan);
          ++i;
          changed = true;
          continue;
        }
      }
      output.push_back(sequence->seq[i]);
    }
    if (!changed) return visited;
    return output.size() == 1 ? output[0] : SeqStmt(output, sequence->span);
  }
};

class StripeCheck : public StmtExprVisitor {
public:
  explicit StripeCheck(const Plan &plan) : plan_(plan) {}
  bool valid{true};
  std::vector<int64_t> offsets;
private:
  const Plan &plan_;
  void VisitExpr_(const CallNode *op) final {
    if (IsGemm(op)) {
      if (op->args.size() < 3) { valid = false; return; }
      auto region = DecodeRegion(op->args[2]);
      if (!region) { valid = false; return; }
      if (region->buffer->data.same_as(plan_.accumulator->data)) {
        if (op->args.size() != 19 && op->args.size() != 20) { valid = false; return; }
        auto offset = op->args.size() == 20 ? Integer(op->args[19]) : std::optional<int64_t>(0);
        if (!offset || !StructuralEqual()(op->args[9], plan_.loop->loop_var == plan_.loop->min)) {
          valid = false;
          return;
        }
        offsets.push_back(*offset);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }
};

inline bool VerifyLegalized(const PrimFunc &func, const std::vector<Plan> &plans) {
  if (!VerifySSA(func)) return false;
  for (const Plan &plan : plans) {
    StripeCheck check(plan);
    check(func->body);
    const std::vector<int64_t> expected = plan.dual_stripe
        ? std::vector<int64_t>{0, plan.second_offset} : std::vector<int64_t>{0};
    if (!check.valid || check.offsets != expected) return false;
  }
  return true;
}

} // namespace sunmmio_mma_clear_fusion
} // namespace tl
} // namespace tvm
#endif // TILELANG_TRANSFORM_SUNMMIO_MMA_CLEAR_FUSION_H_
