/*!
 * \file legalize_negative_index.cc
 * \brief Legalize negative indices in buffer load/store expressions.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>
#include <variant>
#include <vector>

#include "../op/builtin.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "arith/ir_visitor_with_analyzer.h"
#include "common/mx_view.h"

namespace tvm {
namespace tl {

using namespace tir;
using arith::IRVisitorWithAnalyzer;

enum class IndexSignState { kNonNegative, kNegative, kUnknown };

using BufferAccessVariant =
    std::variant<const BufferLoadNode *, const BufferStoreNode *>;
using LoadStore2StateMap =
    std::unordered_map<BufferAccessVariant, std::vector<IndexSignState>>;

namespace {

template <typename FThen, typename FElse>
void VisitReachableBranches(arith::Analyzer *analyzer,
                            const PrimExpr &condition, FThen visit_then,
                            FElse visit_else) {
  PrimExpr predicate = condition;
  if (const auto *call = predicate.as<CallNode>()) {
    if (call->op.same_as(builtin::likely()))
      predicate = call->args[0];
  }
  PrimExpr negated = analyzer->rewrite_simplify(Not(predicate));
  // Prove reachability in the enclosing context. Adding an impossible branch
  // constraint first can make IntSet's modulo analysis operate on infinity.
  bool visit_true = !analyzer->CanProve(negated);
  bool visit_false = !analyzer->CanProve(predicate);
  if (visit_true) {
    With<arith::ConstraintContext> context(analyzer, predicate);
    visit_then();
  }
  if (visit_false) {
    With<arith::ConstraintContext> context(analyzer, negated);
    visit_else();
  }
}

} // namespace

class MXCopyIndexNormalizer : public arith::IRMutatorWithAnalyzer {
public:
  explicit MXCopyIndexNormalizer(arith::Analyzer *analyzer)
      : arith::IRMutatorWithAnalyzer(analyzer) {}

private:
  Stmt VisitStmt_(const ForNode *op) final {
    if (analyzer_->CanProve(op->extent <= 0))
      return Evaluate(0);
    return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    PrimExpr condition = VisitExpr(op->condition);
    Optional<Stmt> then_case, else_case;
    VisitReachableBranches(
        analyzer_, condition, [&] { then_case = VisitStmt(op->then_case); },
        [&] {
          if (op->else_case)
            else_case = VisitStmt(op->else_case.value());
        });
    if (!then_case)
      return else_case.value_or(Evaluate(0));
    PrimExpr predicate = condition;
    if (auto call = predicate.as<CallNode>();
        call && call->op.same_as(builtin::likely()))
      predicate = call->args[0];
    if (analyzer_->CanProve(predicate))
      return then_case.value();
    return IfThenElse(condition, then_case.value(), else_case, op->span);
  }

  PrimExpr VisitExpr_(const SelectNode *op) final {
    // Both values are evaluated; neither inherits a branch predicate.
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::if_then_else())) {
      PrimExpr condition = VisitExpr(op->args[0]);
      Optional<PrimExpr> yes, no;
      VisitReachableBranches(
          analyzer_, condition, [&] { yes = VisitExpr(op->args[1]); },
          [&] { no = VisitExpr(op->args[2]); });
      if (!yes)
        return no.value();
      if (!no)
        return yes.value();
      return Call(op->dtype, op->op, {condition, yes.value(), no.value()},
                  op->annotations, op->span);
    }
    if (!op->op.same_as(Op::Get(kMXDeferredCopy)))
      return arith::IRMutatorWithAnalyzer::VisitExpr_(op);
    // Strict copy validates the original bounds without Python-style wrapping.
    if (!is_zero(op->args[2]))
      return ffi::GetRef<Call>(op);
    auto args = op->args;
    for (int side : {0, 1}) {
      auto operand = Downcast<Call>(args[side]);
      ICHECK(operand->op.same_as(mx_copy_operand()));
      auto fields = operand->args;
      auto load = Downcast<BufferLoad>(fields[1]);
      auto indices = load->indices;
      size_t rank = indices.size();
      bool has_ends = is_one(fields[0]);
      size_t count = has_ends ? 2 * rank : rank;
      Array<Integer> done;
      if (auto flags = operand->annotations.Get("tl.mx_normalized_bounds"))
        done = Downcast<Array<Integer>>(flags.value());
      else
        for (size_t i = 0; i < count; ++i)
          done.push_back(Integer(0));
      ICHECK_EQ(done.size(), count);
      for (size_t i = 0; i < count; ++i) {
        if (done[i]->value)
          continue;
        size_t axis = i % rank;
        PrimExpr bound = i < rank ? indices[axis] : fields[2 + axis];
        if (analyzer_->CanProve(bound < 0)) {
          bound = analyzer_->Simplify(bound + load->buffer->shape[axis]);
          if (i < rank)
            indices.Set(axis, bound);
          else
            fields.Set(2 + axis, bound);
          done.Set(i, Integer(1));
        } else if (analyzer_->CanProve(bound >= 0)) {
          done.Set(i, Integer(1));
        }
      }
      fields.Set(1, BufferLoad(load->buffer, indices, load->predicate));
      auto *node = operand.CopyOnWrite();
      node->args = fields;
      // Only constant completion flags live in metadata; coordinates and
      // shape expressions remain operands of the descriptor.
      node->annotations.Set("tl.mx_normalized_bounds", done);
      args.Set(side, operand);
    }
    return Call(op->dtype, op->op, args, op->annotations, op->span);
  }
};

PrimFunc NormalizeMXCopyIndices(PrimFunc func) {
  bool has_deferred_copy = false;
  PostOrderVisit(func->body, [&](const ObjectRef &obj) {
    if (auto call = obj.as<CallNode>())
      has_deferred_copy |= call->op.same_as(Op::Get(kMXDeferredCopy));
  });
  if (!has_deferred_copy)
    return func;
  arith::Analyzer analyzer;
  MXCopyIndexNormalizer normalizer(&analyzer);
  func.CopyOnWrite()->body = normalizer(func->body);
  return func;
}

class NegativeIndexAnalyzer : public IRVisitorWithAnalyzer {
public:
  explicit NegativeIndexAnalyzer(LoadStore2StateMap *result)
      : result_(result) {}

private:
  void VisitStmt_(const IfThenElseNode *op) final {
    VisitExpr(op->condition);
    VisitReachableBranches(
        &analyzer_, op->condition, [&] { VisitStmt(op->then_case); },
        [&] {
          if (op->else_case)
            VisitStmt(op->else_case.value());
        });
  }

  void VisitExpr_(const CallNode *op) final {
    // The contextual MX normalizer handles descriptor bounds as a pair.
    if (op->op.same_as(mx_copy_operand()))
      return;
    if (!op->op.same_as(builtin::if_then_else())) {
      IRVisitorWithAnalyzer::VisitExpr_(op);
      return;
    }
    VisitExpr(op->args[0]);
    VisitReachableBranches(
        &analyzer_, op->args[0], [&] { VisitExpr(op->args[1]); },
        [&] { VisitExpr(op->args[2]); });
  }

  std::vector<IndexSignState> ProcessIdx(const ffi::Array<PrimExpr> &indices,
                                         ffi::String buffer_name) {
    std::vector<IndexSignState> states;
    states.reserve(indices.size());

    for (size_t i = 0; i < indices.size(); ++i) {
      PrimExpr simplified = analyzer_.Simplify(indices[i]);
      IndexSignState state = IndexSignState::kUnknown;

      // Handle scalar indices with the standard analyzer
      if (simplified.dtype().lanes() == 1) {
        if (analyzer_.CanProve(simplified >= 0))
          state = IndexSignState::kNonNegative;
        else if (analyzer_.CanProve(simplified < 0))
          state = IndexSignState::kNegative;
        else
          DLOG(WARNING)
              << "LegalizeNegativeIndex: cannot prove non-negative index "
              << simplified << " for buffer " << buffer_name << " (axis " << i
              << ", index " + indices[i]->Script() + ").";
      }
      // Vector indices: try to reason about non-negativity/negativity
      // Common patterns are Ramp(base, stride, lanes) and Broadcast(value,
      // lanes).
      else if (const auto *ramp = simplified.as<RampNode>()) {
        // Compute a safe lower/upper bound for the vector lanes
        // lower_bound = base_min + min(0, stride_min) * (lanes - 1)
        // upper_bound = base_max + max(0, stride_max) * (lanes - 1)
        auto base_bound = analyzer_.const_int_bound(ramp->base);
        auto stride_bound = analyzer_.const_int_bound(ramp->stride);
        int lanes = *as_const_int(ramp->lanes);

        int64_t base_min = base_bound->min_value;
        int64_t base_max = base_bound->max_value;
        int64_t s_min = stride_bound->min_value;
        int64_t s_max = stride_bound->max_value;

        // Guard against overflow is not strictly necessary here because
        // bounds may be +/-inf represented by sentinel values.
        int64_t lower = base_min;
        if (s_min < 0)
          lower += s_min * (lanes - 1);
        int64_t upper = base_max;
        if (s_max > 0)
          upper += s_max * (lanes - 1);

        if (lower >= 0)
          state = IndexSignState::kNonNegative;
        else if (upper < 0)
          state = IndexSignState::kNegative;
        else
          DLOG(WARNING)
              << "LegalizeNegativeIndex: cannot prove non-negative index "
              << simplified << " for buffer " << buffer_name << " (axis " << i
              << ", index " + indices[i]->Script() + ").";
      } else if (const auto *broadcast = simplified.as<BroadcastNode>()) {
        auto v = analyzer_.Simplify(broadcast->value);
        if (analyzer_.CanProve(v >= 0))
          state = IndexSignState::kNonNegative;
        else if (analyzer_.CanProve(v < 0))
          state = IndexSignState::kNegative;
        else {
          // Try const bound if proof unavailable
          auto vb = analyzer_.const_int_bound(v);
          if (vb->min_value >= 0)
            state = IndexSignState::kNonNegative;
          else if (vb->max_value < 0)
            state = IndexSignState::kNegative;
          else
            DLOG(WARNING)
                << "LegalizeNegativeIndex: cannot prove non-negative index "
                << simplified << " for buffer " << buffer_name << " (axis " << i
                << ", index " + indices[i]->Script() + ").";
        }
      }
      states.push_back(state);
    }

    return std::move(states);
  }

  bool NeedRecord(const std::vector<IndexSignState> &states) {
    return std::any_of(states.begin(), states.end(),
                       [](const IndexSignState &state) {
                         return state == IndexSignState::kUnknown ||
                                state == IndexSignState::kNegative;
                       });
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    std::vector<IndexSignState> states =
        ProcessIdx(op->indices, op->buffer->name);

    if (NeedRecord(states))
      (*result_)[op] = std::move(states);

    IRVisitorWithAnalyzer::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    std::vector<IndexSignState> states =
        ProcessIdx(op->indices, op->buffer->name);

    if (NeedRecord(states))
      (*result_)[op] = std::move(states);

    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

private:
  LoadStore2StateMap *result_;
};

class NegativeIndexRewriter : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Apply(PrimFunc func, const LoadStore2StateMap &states) {
    arith::Analyzer analyzer;
    NegativeIndexRewriter rewriter(&analyzer, states);
    PrimFuncNode *func_node = func.CopyOnWrite();
    func_node->body = rewriter.VisitStmt(func_node->body);
    return func;
  }

private:
  NegativeIndexRewriter(arith::Analyzer *analyzer,
                        const LoadStore2StateMap &states)
      : arith::IRMutatorWithAnalyzer(analyzer), states_(states) {}

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    PrimExpr condition = VisitExpr(op->condition);
    Stmt then_case = op->then_case;
    ffi::Optional<Stmt> else_case = op->else_case;
    VisitReachableBranches(
        analyzer_, condition, [&] { then_case = VisitStmt(then_case); },
        [&] {
          if (else_case)
            else_case = VisitStmt(else_case.value());
        });
    if (condition.same_as(op->condition) && then_case.same_as(op->then_case) &&
        else_case.same_as(op->else_case))
      return ffi::GetRef<Stmt>(op);
    auto n = CopyOnWrite(op);
    n->condition = std::move(condition);
    n->then_case = std::move(then_case);
    n->else_case = std::move(else_case);
    return Stmt(n);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (!op->op.same_as(builtin::if_then_else()))
      return arith::IRMutatorWithAnalyzer::VisitExpr_(op);
    PrimExpr condition = VisitExpr(op->args[0]);
    PrimExpr true_value = op->args[1];
    PrimExpr false_value = op->args[2];
    VisitReachableBranches(
        analyzer_, condition, [&] { true_value = VisitExpr(true_value); },
        [&] { false_value = VisitExpr(false_value); });
    if (condition.same_as(op->args[0]) && true_value.same_as(op->args[1]) &&
        false_value.same_as(op->args[2]))
      return ffi::GetRef<PrimExpr>(op);
    return Call(op->dtype, op->op, {condition, true_value, false_value},
                op->annotations, op->span);
  }

  PrimExpr VisitExpr_(const SelectNode *op) final {
    // Select evaluates both values. Match the analysis traversal and avoid
    // introducing mutually exclusive branch constraints during rewriting.
    return StmtExprMutator::VisitExpr_(op);
  }

  ffi::Array<PrimExpr> UpdateIdx(const ffi::Array<PrimExpr> &indices,
                                 const ffi::Array<PrimExpr> &buffer_shape,
                                 const std::vector<IndexSignState> &state_vec) {
    ICHECK_EQ(state_vec.size(), indices.size())
        << "State vector size mismatch for buffer load/store indices ("
        << indices << ")";
    ffi::Array<PrimExpr> new_indices = indices;
    for (size_t i = 0; i < indices.size(); ++i) {
      if (state_vec[i] != IndexSignState::kNegative)
        continue;
      new_indices.Set(i, analyzer_->Simplify(buffer_shape[i] + indices[i]));
    }
    return new_indices;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load =
        Downcast<BufferLoad>(arith::IRMutatorWithAnalyzer::VisitExpr_(op));

    auto it = states_.find(op);
    if (it == states_.end())
      return load;

    auto indices = UpdateIdx(load->indices, load->buffer->shape, it->second);
    return BufferLoad(load->buffer, indices, load->predicate);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store =
        Downcast<BufferStore>(arith::IRMutatorWithAnalyzer::VisitStmt_(op));

    auto it = states_.find(op);
    if (it == states_.end())
      return store;

    auto indices = UpdateIdx(store->indices, store->buffer->shape, it->second);
    return BufferStore(store->buffer, store->value, indices, store->predicate);
  }

private:
  const LoadStore2StateMap &states_;
};

PrimFunc LegalizeNegativeIndex(PrimFunc func) {
  if (!func->body.defined()) {
    return func;
  }

  func = NormalizeMXCopyIndices(std::move(func));

  LoadStore2StateMap states;
  NegativeIndexAnalyzer analyzer(&states);
  analyzer(func->body);
  if (states.empty()) {
    return func;
  }

  return NegativeIndexRewriter::Apply(std::move(func), states);
}

tvm::transform::Pass LegalizeNegativeIndexPass() {
  using namespace tir::transform;
  auto pass_func = [](PrimFunc f, const IRModule &, PassContext) {
    return LegalizeNegativeIndex(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LegalizeNegativeIndex", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LegalizeNegativeIndex",
                        LegalizeNegativeIndexPass);
}

} // namespace tl
} // namespace tvm
