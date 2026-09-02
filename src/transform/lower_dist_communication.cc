/*!
 * \file tl/transform/lower_dist_communication.cc
 * \brief Plan completion expectations and lower Rank communication leaves.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../op/builtin.h"
#include "../op/utils.h"
#include "dist_transform_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;
using namespace dist_transform;

namespace {

class DistExpectationPlanner : public DistRouteMutatorBase {
public:
  using DistRouteMutatorBase::DistRouteMutatorBase;

  static PrimFunc Rewrite(PrimFunc func) {
    auto context = GetDistPassContext(func);
    if (!context) {
      return func;
    }
    arith::Analyzer analyzer;
    DistExpectationPlanner rewriter(&analyzer, context.value().target,
                                    context.value().world_size,
                                    context.value().rank_id);
    Stmt body = rewriter.VisitStmt(func->body);
    if (!body.same_as(func->body)) {
      func.CopyOnWrite()->body = std::move(body);
    }
    return func;
  }

private:
  struct SignalExpectation {
    PrimExpr signal;
    PrimExpr current_core;
    std::vector<std::vector<PrimExpr>> deltas;
    std::vector<std::vector<std::optional<std::pair<int64_t, int64_t>>>>
        senders;
  };

  int64_t RequireResolvedInt(const PrimExpr &expr, const char *name) {
    PrimExpr simplified = analyzer_->Simplify(expr);
    const auto *imm = simplified.as<IntImmNode>();
    ICHECK(imm) << "Cannot resolve " << name << " in static route: " << expr;
    return imm->value;
  }

  bool ExprEqual(const PrimExpr &lhs, const PrimExpr &rhs) {
    return lhs.same_as(rhs) || analyzer_->CanProve(lhs == rhs);
  }

  PrimExpr BuildDelta(std::vector<std::vector<PrimExpr>> deltas,
                      const PrimExpr &current_core) {
    for (auto &rank_deltas : deltas) {
      for (PrimExpr &delta : rank_deltas) {
        delta = analyzer_->Simplify(delta);
      }
    }
    PrimExpr uniform = deltas[0][0];
    bool all_equal = true;
    for (const auto &rank_deltas : deltas) {
      for (const PrimExpr &delta : rank_deltas) {
        all_equal &= ExprEqual(delta, uniform);
      }
    }
    if (all_equal) {
      return uniform;
    }
    PrimExpr current_row = CurrentRow(current_core);

    bool ranks_equal = true;
    for (int64_t rank = 1; rank < world_size_; ++rank) {
      for (int64_t row = 0; row < mesh_nrows_; ++row) {
        ranks_equal &= ExprEqual(deltas[rank][row], deltas[0][row]);
      }
    }
    if (ranks_equal) {
      PrimExpr delta = I32(0);
      for (int64_t row = mesh_nrows_ - 1; row >= 0; --row) {
        if (!is_zero(deltas[0][row])) {
          delta = Select(current_row == I32(row), deltas[0][row], delta);
        }
      }
      return analyzer_->Simplify(delta);
    }

    bool rows_equal = true;
    for (const auto &rank_deltas : deltas) {
      for (int64_t row = 1; row < mesh_nrows_; ++row) {
        rows_equal &= ExprEqual(rank_deltas[row], rank_deltas[0]);
      }
    }
    if (rows_equal) {
      PrimExpr delta = I32(0);
      for (int64_t rank = world_size_ - 1; rank >= 0; --rank) {
        if (!is_zero(deltas[rank][0])) {
          delta = Select(rank_id_ == I32(rank), deltas[rank][0], delta);
        }
      }
      return analyzer_->Simplify(delta);
    }

    PrimExpr delta = I32(0);
    for (int64_t rank = world_size_ - 1; rank >= 0; --rank) {
      for (int64_t row = mesh_nrows_ - 1; row >= 0; --row) {
        PrimExpr endpoint_delta = deltas[rank][row];
        if (is_zero(endpoint_delta)) {
          continue;
        }
        PrimExpr condition =
            And(rank_id_ == I32(rank), current_row == I32(row));
        delta = Select(condition, endpoint_delta, delta);
      }
    }
    return analyzer_->Simplify(delta);
  }

  SignalExpectation &GetExpectation(
      std::unordered_map<const VarNode *, SignalExpectation> *expectations,
      const PrimExpr &signal, const PrimExpr &current_core) {
    const auto *signal_var = signal.as<VarNode>();
    ICHECK(signal_var) << "T.dist peer operation expects a signal Var";
    auto it = expectations->find(signal_var);
    if (it == expectations->end()) {
      std::vector<std::vector<PrimExpr>> deltas(
          world_size_, std::vector<PrimExpr>(mesh_nrows_, I32(0)));
      std::vector<std::vector<std::optional<std::pair<int64_t, int64_t>>>>
          senders(world_size_,
                  std::vector<std::optional<std::pair<int64_t, int64_t>>>(
                      mesh_nrows_));
      it = expectations
               ->emplace(signal_var, SignalExpectation{signal, current_core,
                                                       deltas, senders})
               .first;
    }
    return it->second;
  }

  void AddDelta(SignalExpectation *expectation, int64_t dst_rank,
                int64_t dst_row, int64_t src_rank, int64_t src_row,
                const PrimExpr &predicate) {
    ICHECK_GE(dst_rank, 0);
    ICHECK_LT(dst_rank, world_size_);
    ICHECK_GE(dst_row, 0);
    ICHECK_LT(dst_row, mesh_nrows_);
    if (!analyzer_->CanProve(Not(predicate))) {
      auto &sender = expectation->senders[dst_rank][dst_row];
      std::pair<int64_t, int64_t> endpoint{src_rank, src_row};
      ICHECK(!sender || sender.value() == endpoint)
          << "T.dist signal " << expectation->signal
          << " has multiple physical senders for receiver endpoint (rank="
          << dst_rank << ", row=" << dst_row
          << "). Use one independent signal per sender";
      sender = endpoint;
    }
    PrimExpr contribution = Select(predicate, I32(1), I32(0));
    expectation->deltas[dst_rank][dst_row] = analyzer_->Simplify(
        expectation->deltas[dst_rank][dst_row] + contribution);
  }

  void AccumulatePeerCall(
      const CallNode *call, const PrimExpr &source_predicate,
      std::unordered_map<const VarNode *, SignalExpectation> *expectations) {
    if (call->op.same_as(DistPeerPutOp::Get())) {
      ICHECK_EQ(call->args.size(), 5U);
      SignalExpectation &expectation =
          GetExpectation(expectations, call->args[3], call->args[4]);
      Var current_core = Downcast<Var>(call->args[4]);
      for (int64_t src_rank = 0; src_rank < world_size_; ++src_rank) {
        for (int64_t src_row = 0; src_row < mesh_nrows_; ++src_row) {
          Map<Var, PrimExpr> substitution =
              MakeSourceSubstitution(current_core, src_row, src_rank);
          int64_t dst_rank = RequireResolvedInt(
              Substitute(call->args[2], substitution), "peer destination rank");
          PrimExpr predicate =
              analyzer_->Simplify(Substitute(source_predicate, substitution));
          AddDelta(&expectation, dst_rank, src_row, src_rank, src_row,
                   predicate);
        }
      }
      return;
    }

    if (call->op.same_as(DistRoutedPeerPutOp::Get())) {
      ICHECK_GE(call->args.size(), 5U);
      SignalExpectation &expectation =
          GetExpectation(expectations, call->args[1], call->args[2]);
      Var current_core = Downcast<Var>(call->args[2]);
      std::vector<PeerRouteEntry> routes = ParsePeerRouteTable(call->args[0]);
      for (int64_t src_rank = 0; src_rank < world_size_; ++src_rank) {
        for (const PeerRouteEntry &route : routes) {
          int64_t peer_row = RequireResolvedInt(route.peer_row, "peer row");
          Map<Var, PrimExpr> substitution =
              MakeSourceSubstitution(current_core, peer_row, src_rank);
          int64_t dst_rank =
              RequireResolvedInt(Substitute(route.dst_rank, substitution),
                                 "routed peer destination rank");
          PrimExpr predicate =
              analyzer_->Simplify(Substitute(source_predicate, substitution));
          AddDelta(&expectation, dst_rank, peer_row, src_rank, peer_row,
                   predicate);
        }
      }
    }
  }

  class ConditionalPeerCollector : public StmtExprVisitor {
  public:
    ConditionalPeerCollector(
        DistExpectationPlanner *planner, PrimExpr predicate,
        std::unordered_map<const VarNode *, SignalExpectation> *expectations)
        : planner_(planner), predicate_(std::move(predicate)),
          expectations_(expectations) {}

  private:
    void VisitStmt_(const IfThenElseNode *op) final {
      PrimExpr old_predicate = predicate_;
      predicate_ =
          planner_->analyzer_->Simplify(And(old_predicate, op->condition));
      VisitStmt(op->then_case);
      if (op->else_case) {
        predicate_ = planner_->analyzer_->Simplify(
            And(old_predicate, Not(op->condition)));
        VisitStmt(op->else_case.value());
      }
      predicate_ = old_predicate;
    }

    void VisitStmt_(const ForNode *op) final {
      ++loop_depth_;
      StmtExprVisitor::VisitStmt_(op);
      --loop_depth_;
    }

    void VisitStmt_(const EvaluateNode *op) final {
      const auto *call = op->value.as<CallNode>();
      if (call && (call->op.same_as(DistPeerPutOp::Get()) ||
                   call->op.same_as(DistRoutedPeerPutOp::Get()))) {
        ICHECK_EQ(loop_depth_, 0)
            << "Rank/column-conditioned T.dist peer operations inside loops "
               "are not supported yet";
        planner_->AccumulatePeerCall(call, predicate_, expectations_);
        return;
      }
      StmtExprVisitor::VisitStmt_(op);
    }

    DistExpectationPlanner *planner_;
    PrimExpr predicate_;
    std::unordered_map<const VarNode *, SignalExpectation> *expectations_;
    int loop_depth_{0};
  };

  Array<Stmt> MakeMarkers(
      std::unordered_map<const VarNode *, SignalExpectation> expectations) {
    Array<Stmt> markers;
    std::vector<SignalExpectation> ordered;
    ordered.reserve(expectations.size());
    for (auto &[_, expectation] : expectations) {
      ordered.push_back(std::move(expectation));
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const SignalExpectation &lhs, const SignalExpectation &rhs) {
                return lhs.signal.as<VarNode>()->name_hint <
                       rhs.signal.as<VarNode>()->name_hint;
              });
    for (SignalExpectation &expectation : ordered) {
      PrimExpr delta =
          BuildDelta(std::move(expectation.deltas), expectation.current_core);
      markers.push_back(Evaluate(Call(DataType::Handle(), dist_expect(),
                                      {expectation.signal, delta})));
    }
    return markers;
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    if (suppress_injection_) {
      return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    std::unordered_map<const VarNode *, SignalExpectation> expectations;
    ConditionalPeerCollector collector(this, const_true(), &expectations);
    collector(tvm::ffi::GetRef<Stmt>(op));
    if (expectations.empty()) {
      return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    suppress_injection_ = true;
    Stmt guarded_operations = arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    suppress_injection_ = false;
    Array<Stmt> statements = MakeMarkers(std::move(expectations));
    statements.push_back(guarded_operations);
    return SeqStmt::Flatten(statements);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (suppress_injection_) {
      return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    const auto *call = op->value.as<CallNode>();
    if (!call || (!call->op.same_as(DistPeerPutOp::Get()) &&
                  !call->op.same_as(DistRoutedPeerPutOp::Get()))) {
      return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    std::unordered_map<const VarNode *, SignalExpectation> expectations;
    AccumulatePeerCall(call, const_true(), &expectations);
    Array<Stmt> statements = MakeMarkers(std::move(expectations));
    statements.push_back(tvm::ffi::GetRef<Stmt>(op));
    return SeqStmt::Flatten(statements);
  }

  bool suppress_injection_{false};
};

class LowerDistPrimitiveMutator : public StmtExprMutator {
public:
  explicit LowerDistPrimitiveMutator(int mesh_ncols)
      : mesh_ncols_(mesh_ncols) {}

private:
  struct SignalInfo {
    PrimExpr kind;
    PrimExpr index;
    Buffer expected;
    Buffer generation;
    bool used{false};
  };

  Stmt VisitStmt_(const LetStmtNode *let_node) final {
    const auto *signal_call_node = let_node->value.as<CallNode>();
    if (signal_call_node && signal_call_node->op.same_as(dist_signal_decl())) {
      ICHECK(false) << "tl.dist_signal_decl must be resolved before "
                       "LowerDistCommunication";
    }
    if (!signal_call_node || !signal_call_node->op.same_as(dist_signal()) ||
        signal_call_node->args.size() != 2U) {
      return StmtExprMutator::VisitStmt_(let_node);
    }

    std::string expect_name = let_node->var->name_hint + "_expect";
    std::string generation_name = let_node->var->name_hint + "_generation";
    Buffer expected = decl_buffer({IntImm(DataType::Int(32), 1)},
                                  DataType::UInt(8), expect_name, "local.var");
    Buffer generation =
        decl_buffer({IntImm(DataType::Int(32), 1)}, DataType::UInt(8),
                    generation_name, "local.var");
    SignalInfo info{signal_call_node->args[0], signal_call_node->args[1],
                    expected, generation};
    signal_info_.emplace(let_node->var.get(), std::move(info));
    Stmt body = VisitStmt(let_node->body);
    bool used = signal_info_.at(let_node->var.get()).used;
    signal_info_.erase(let_node->var.get());
    if (!used) {
      return body;
    }

    Stmt scoped = DeclBuffer(expected, std::move(body));
    scoped = DeclBuffer(generation, std::move(scoped));
    Map<String, ffi::Any> annotations;
    annotations.Set(tl::attr::kLocalVarInit, IntImm(DataType::UInt(8), 0));
    scoped = Allocate(expected->data, expected->dtype, expected->shape,
                      const_true(), std::move(scoped), annotations);
    return Allocate(generation->data, generation->dtype, generation->shape,
                    const_true(), std::move(scoped), annotations);
  }

  PrimExpr VisitExpr_(const CallNode *call_node) final {
    PrimExpr rewritten = StmtExprMutator::VisitExpr_(call_node);
    const auto *call = rewritten.as<CallNode>();
    ICHECK(call);
    if (call->op.same_as(DistPutOp::Get())) {
      ICHECK(false) << "Logical T.dist.put must be processed by "
                       "LowerDistRouting before LowerDistCommunication";
    }
    if (call->op.same_as(DistPeerPutOp::Get())) {
      ICHECK_EQ(call->args.size(), 5U);
      SignalInfo &signal = LookupSignal(call->args[3]);
      signal.used = true;
      Array<PrimExpr> args{call->args[0], call->args[1],
                           call->args[2], signal.kind,
                           signal.index,  BufferLoad(signal.generation, {0})};
      return Call(call->dtype, dist_put_(), args, call->annotations,
                  call->span);
    }
    if (call->op.same_as(DistRoutedPeerPutOp::Get())) {
      ICHECK(false) << "T.dist.routed_peer_put must appear as an Evaluate "
                       "statement";
    }
    if (call->op.same_as(DistWaitSignalOp::Get())) {
      ICHECK_EQ(call->args.size(), 2U);
      SignalInfo &signal = LookupSignal(call->args[0]);
      signal.used = true;
      Array<PrimExpr> args{signal.kind, signal.index,
                           BufferLoad(signal.expected, {0}), call->args[1]};
      return Call(call->dtype, dist_wait_signal_(), args, call->annotations,
                  call->span);
    }
    if (call->op.same_as(dist_expect())) {
      ICHECK_EQ(call->args.size(), 2U);
      SignalInfo &signal = LookupSignal(call->args[0]);
      signal.used = true;
      Array<PrimExpr> args{BufferLoad(signal.expected, {0}), call->args[1]};
      return Call(call->dtype, dist_expect_(), args, call->annotations,
                  call->span);
    }
    return rewritten;
  }

  Stmt VisitStmt_(const EvaluateNode *evaluate_node) final {
    const auto *call = evaluate_node->value.as<CallNode>();
    if (call && call->op.same_as(dist_wait_all())) {
      ICHECK_GE(call->args.size(), 2U);
      Array<Stmt> waits;
      for (size_t index = 1; index < call->args.size(); ++index) {
        SignalInfo &signal = LookupSignal(call->args[index]);
        signal.used = true;
        Array<PrimExpr> args{signal.kind, signal.index,
                             BufferLoad(signal.expected, {0}), call->args[0]};
        waits.push_back(Evaluate(Call(DataType::Handle(), dist_wait_signal_(),
                                      args, call->annotations, call->span)));
      }
      return SeqStmt::Flatten(waits);
    }
    if (!call || !call->op.same_as(DistRoutedPeerPutOp::Get())) {
      return StmtExprMutator::VisitStmt_(evaluate_node);
    }
    ICHECK_GE(call->args.size(), 5U);
    ICHECK_EQ((call->args.size() - 3U) % 2U, 0U);
    std::vector<PeerRouteEntry> routes = ParsePeerRouteTable(call->args[0]);
    ICHECK_EQ(routes.size(), (call->args.size() - 3U) / 2U);
    SignalInfo &signal = LookupSignal(call->args[1]);
    signal.used = true;
    PrimExpr current_row =
        floordiv(call->args[2], IntImm(DataType::Int(32), mesh_ncols_));
    Array<Stmt> sends;
    for (size_t index = 0; index < routes.size(); ++index) {
      const PeerRouteEntry &route = routes[index];
      Array<PrimExpr> args{call->args[3 + index * 2],
                           call->args[4 + index * 2],
                           route.dst_rank,
                           signal.kind,
                           signal.index,
                           BufferLoad(signal.generation, {0})};
      Stmt send = Evaluate(Call(DataType::Handle(), dist_put_(), args));
      sends.push_back(IfThenElse(current_row == route.peer_row, send));
    }
    return SeqStmt::Flatten(sends);
  }

  SignalInfo &LookupSignal(const PrimExpr &signal) {
    const auto *var = signal.as<VarNode>();
    ICHECK(var) << "T.dist primitive expected a signal Var, got " << signal;
    auto it = signal_info_.find(var);
    ICHECK(it != signal_info_.end())
        << "Cannot find T.dist.signal definition for " << signal;
    return it->second;
  }

  std::unordered_map<const VarNode *, SignalInfo> signal_info_;
  int mesh_ncols_{1};
};

PrimFunc Run(PrimFunc func) {
  if (!IsMultiRank(func)) {
    return func;
  }
  DistOpDetector detector(/*high_level=*/true);
  if (!detector.Detect(func->body)) {
    return func;
  }
  func = DistExpectationPlanner::Rewrite(std::move(func));
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  ICHECK(target && TargetIsSunmmio(target.value()));
  int mesh_ncols = GetSunmmioMeshConfig(target.value()).ncol;
  Stmt body = LowerDistPrimitiveMutator(mesh_ncols)(func->body);
  if (!body.same_as(func->body)) {
    func.CopyOnWrite()->body = std::move(body);
  }
  return func;
}

} // namespace

tvm::transform::Pass LowerDistCommunication() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) {
    return Run(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerDistCommunication", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerDistCommunication",
                        LowerDistCommunication);
}

} // namespace tl
} // namespace tvm
