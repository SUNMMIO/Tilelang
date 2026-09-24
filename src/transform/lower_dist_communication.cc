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
#include <unordered_set>
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
  struct SignalPlan {
    const DistSignalKindInfo *kind;
    bool auto_expected;
  };

  bool IsAutoExpected(const PrimExpr &mode_expr) const {
    std::string mode = RequireStringImm(mode_expr, "resolved expect mode");
    ICHECK(mode == "auto" || mode == "manual");
    return mode == "auto";
  }

  PrimExpr PeerSignal(const CallNode *call) const {
    if (call->op.same_as(dist_signal_put())) {
      ICHECK_EQ(call->args.size(), 3U);
      return call->args[0];
    }
    if (call->op.same_as(DistPeerPutOp::Get())) {
      ICHECK_EQ(call->args.size(), 5U);
      return call->args[3];
    }
    ICHECK(call->op.same_as(DistRoutedPeerPutOp::Get()));
    ICHECK_GE(call->args.size(), 5U);
    return call->args[1];
  }

  bool ShouldInferExpected(const CallNode *call) const {
    PrimExpr signal = PeerSignal(call);
    if (const auto *owner = signal.as<VarNode>()) {
      auto plan = signal_kinds_.find(owner);
      ICHECK(plan != signal_kinds_.end());
      return plan->second.auto_expected;
    }
    const auto *ref = signal.as<CallNode>();
    ICHECK(ref);
    // These references carry an explicit user/protocol expectation source,
    // including auto all-gather groups whose completion supplies the delta.
    if (ref->op.same_as(dist_signal_route()) ||
        ref->op.same_as(dist_signal_group_route())) {
      return false;
    }
    ICHECK(ref->op.same_as(dist_signal_ref()));
    ICHECK_EQ(ref->args.size(), 2U);
    auto plan = signal_group_kinds_.find(ref->args[0].as<VarNode>());
    ICHECK(plan != signal_group_kinds_.end());
    return plan->second.auto_expected;
  }

  struct SignalExpectation {
    PrimExpr signal;
    std::string sort_key;
    PrimExpr current_core;
    const DistSignalKindInfo *kind;
    std::vector<std::vector<PrimExpr>> deltas;
    std::vector<std::vector<std::optional<std::pair<int64_t, int64_t>>>>
        senders;
  };

  struct SignalExpectationKey {
    const VarNode *owner;
    int64_t member_index;

    bool operator==(const SignalExpectationKey &other) const {
      return owner == other.owner && member_index == other.member_index;
    }
  };

  struct SignalExpectationKeyHash {
    size_t operator()(const SignalExpectationKey &key) const {
      return std::hash<const VarNode *>()(key.owner) ^
             (std::hash<int64_t>()(key.member_index) << 1);
    }
  };

  using SignalExpectationMap =
      std::unordered_map<SignalExpectationKey, SignalExpectation,
                         SignalExpectationKeyHash>;

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

  SignalExpectation &GetExpectation(SignalExpectationMap *expectations,
                                    const PrimExpr &signal,
                                    const PrimExpr &current_core) {
    const VarNode *owner = signal.as<VarNode>();
    int64_t member_index = -1;
    const DistSignalKindInfo *kind = nullptr;
    std::string sort_key;
    if (owner) {
      auto kind_it = signal_kinds_.find(owner);
      ICHECK(kind_it != signal_kinds_.end())
          << "Cannot find resolved T.dist.signal kind for " << signal;
      kind = kind_it->second.kind;
      sort_key = owner->name_hint;
    } else {
      const auto *ref = signal.as<CallNode>();
      ICHECK(ref && ref->op.same_as(dist_signal_ref()))
          << "T.dist peer operation expects a signal or static SignalList "
             "member";
      ICHECK_EQ(ref->args.size(), 2U);
      owner = ref->args[0].as<VarNode>();
      ICHECK(owner);
      member_index = RequireIntImm(ref->args[1], "signal-group index");
      auto kind_it = signal_group_kinds_.find(owner);
      ICHECK(kind_it != signal_group_kinds_.end())
          << "Cannot find resolved T.dist.signals kind for " << signal;
      kind = kind_it->second.kind;
      sort_key = owner->name_hint + ":" + std::to_string(member_index);
    }
    SignalExpectationKey key{owner, member_index};
    auto it = expectations->find(key);
    if (it == expectations->end()) {
      std::vector<std::vector<PrimExpr>> deltas(
          world_size_, std::vector<PrimExpr>(mesh_nrows_, I32(0)));
      std::vector<std::vector<std::optional<std::pair<int64_t, int64_t>>>>
          senders(world_size_,
                  std::vector<std::optional<std::pair<int64_t, int64_t>>>(
                      mesh_nrows_));
      it = expectations
               ->emplace(key, SignalExpectation{signal, sort_key, current_core,
                                                kind, deltas, senders})
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
    if (!expectation->kind->allow_multi_sender &&
        !analyzer_->CanProve(Not(predicate))) {
      auto &sender = expectation->senders[dst_rank][dst_row];
      std::pair<int64_t, int64_t> endpoint{src_rank, src_row};
      ICHECK(!sender || sender.value() == endpoint)
          << "T.dist signal " << expectation->signal
          << " has multiple physical senders for receiver endpoint (rank="
          << dst_rank << ", row=" << dst_row << ") with kind "
          << expectation->kind->name
          << ". Use one independent signal per sender";
      sender = endpoint;
    }
    PrimExpr contribution = Select(predicate, I32(1), I32(0));
    expectation->deltas[dst_rank][dst_row] = analyzer_->Simplify(
        expectation->deltas[dst_rank][dst_row] + contribution);
  }

  void AccumulatePeerCall(const CallNode *call,
                          const PrimExpr &source_predicate,
                          SignalExpectationMap *expectations) {
    if (!ShouldInferExpected(call)) {
      return;
    }
    size_t core_arg = call->op.same_as(DistPeerPutOp::Get()) ? 4U : 2U;
    ValidateAutoPredicate(source_predicate, call->args[core_arg]);
    if (call->op.same_as(dist_signal_put())) {
      ICHECK_EQ(call->args.size(), 3U);
      SignalExpectation &expectation =
          GetExpectation(expectations, call->args[0], call->args[2]);
      Var current_core = Downcast<Var>(call->args[2]);
      for (int64_t src_rank = 0; src_rank < world_size_; ++src_rank) {
        for (int64_t src_row = 0; src_row < mesh_nrows_; ++src_row) {
          Map<Var, PrimExpr> substitution =
              MakeSourceSubstitution(current_core, src_row, src_rank);
          int64_t dst_rank =
              RequireResolvedInt(Substitute(call->args[1], substitution),
                                 "signal destination rank");
          PrimExpr predicate =
              analyzer_->Simplify(Substitute(source_predicate, substitution));
          ICHECK(dst_rank != src_rank || analyzer_->CanProve(Not(predicate)))
              << "T.dist.put_signal only supports a remote peer Rank; source "
                 "Rank "
              << src_rank << " targets itself";
          AddDelta(&expectation, dst_rank, src_row, src_rank, src_row,
                   predicate);
        }
      }
      return;
    }
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
    ConditionalPeerCollector(DistExpectationPlanner *planner,
                             PrimExpr predicate,
                             SignalExpectationMap *expectations)
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

    void VisitStmt_(const WhileNode *op) final {
      ++loop_depth_;
      StmtExprVisitor::VisitStmt_(op);
      --loop_depth_;
    }

    void VisitStmt_(const EvaluateNode *op) final {
      const auto *call = op->value.as<CallNode>();
      if (call && (call->op.same_as(dist_signal_put()) ||
                   call->op.same_as(DistPeerPutOp::Get()) ||
                   call->op.same_as(DistRoutedPeerPutOp::Get()))) {
        if (!planner_->ShouldInferExpected(call)) {
          return;
        }
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
    SignalExpectationMap *expectations_;
    int loop_depth_{0};
  };

  class ReceiverWaitDetector : public StmtExprVisitor {
  public:
    bool Detect(const Stmt &stmt) {
      VisitStmt(stmt);
      return found_;
    }

    static bool IsWait(const CallNode *call) {
      return call && (call->op.same_as(dist_wait_signal()) ||
                      call->op.same_as(dist_wait_signal_delta()) ||
                      call->op.same_as(dist_wait_all()) ||
                      call->op.same_as(dist_wait_signals()) ||
                      call->op.same_as(dist_wait_any()) ||
                      call->op.same_as(dist_wait_completion_all()));
    }

  private:
    void VisitExpr_(const CallNode *call) final {
      found_ |= IsWait(call);
      StmtExprVisitor::VisitExpr_(call);
    }

    bool found_{false};
  };

  class MutableConditionDetector : public ExprVisitor {
  public:
    bool Detect(const PrimExpr &expr) {
      VisitExpr(expr);
      return found_;
    }

  private:
    void VisitExpr_(const BufferLoadNode *op) final { found_ = true; }

    bool found_{false};
  };

  // Hoisting a branch's deltas must not make an earlier wait observe a later
  // send on the same resource. Prove only flat communication sequences; do
  // not infer phases through nested control flow or completion snapshots.
  class ConditionalPhaseValidator : public StmtExprVisitor {
  public:
    explicit ConditionalPhaseValidator(DistExpectationPlanner *planner)
        : planner_(planner) {}

    void Validate(const Stmt &branch) { VisitStmt(branch); }

  private:
    SignalExpectationKey SignalKey(const PrimExpr &signal) const {
      if (const auto *owner = signal.as<VarNode>()) {
        ICHECK(planner_->signal_kinds_.count(owner) ||
               planner_->signal_group_kinds_.count(owner));
        return {owner, -1};
      }
      const auto *ref = signal.as<CallNode>();
      ICHECK(ref && ref->op.same_as(dist_signal_ref()) &&
             ref->args.size() == 2U && ref->args[0].as<VarNode>() &&
             ref->args[1].as<IntImmNode>())
          << "Cannot prove safe T.dist if/else expectation hoisting for a "
             "dynamic signal member; split the condition into phases";
      return {ref->args[0].as<VarNode>(), ref->args[1].as<IntImmNode>()->value};
    }

    void CheckFlat() const {
      ICHECK_EQ(control_depth_, 0)
          << "Cannot prove safe T.dist if/else expectation hoisting through "
             "nested communication; split the condition into phases";
    }

    void VisitStmt_(const IfThenElseNode *op) final {
      ++control_depth_;
      StmtExprVisitor::VisitStmt_(op);
      --control_depth_;
    }

    void VisitStmt_(const ForNode *op) final {
      ++control_depth_;
      StmtExprVisitor::VisitStmt_(op);
      --control_depth_;
    }

    void VisitStmt_(const WhileNode *op) final {
      ++control_depth_;
      StmtExprVisitor::VisitStmt_(op);
      --control_depth_;
    }

    void VisitExpr_(const CallNode *call) final {
      if (call->op.same_as(dist_signal_put()) ||
          call->op.same_as(DistPeerPutOp::Get()) ||
          call->op.same_as(DistRoutedPeerPutOp::Get())) {
        CheckFlat();
        if (planner_->ShouldInferExpected(call)) {
          PrimExpr signal = planner_->PeerSignal(call);
          SignalExpectationKey sent = SignalKey(signal);
          for (const SignalExpectationKey &waited : waits_) {
            ICHECK(waited.owner != sent.owner ||
                   (waited.member_index >= 0 &&
                    waited.member_index != sent.member_index))
                << "Cannot hoist T.dist if/else expectation: signal " << signal
                << " is sent after a wait on the same signal; split the "
                   "condition into phases";
          }
        }
      } else if (call->op.same_as(dist_wait_signal()) ||
                 call->op.same_as(dist_wait_all())) {
        CheckFlat();
        ICHECK_EQ(call->args.size(), 1U);
        waits_.push_back(SignalKey(call->args[0]));
      } else {
        ICHECK(!ReceiverWaitDetector::IsWait(call) &&
               !call->op.same_as(dist_completion()) &&
               !call->op.same_as(dist_completion_has_pending()) &&
               !call->op.same_as(tir::builtin::ret()) &&
               !call->op.same_as(tir::builtin::break_loop()) &&
               !call->op.same_as(tir::builtin::continue_loop()))
            << "Cannot prove safe T.dist if/else expectation hoisting with "
               "completion, dynamic/manual waits, or early exits; split "
               "the condition into phases";
      }
      StmtExprVisitor::VisitExpr_(call);
    }

    DistExpectationPlanner *planner_;
    int control_depth_{0};
    std::vector<SignalExpectationKey> waits_;
  };

  class BranchConditionValidator : public ExprVisitor {
  public:
    explicit BranchConditionValidator(
        std::unordered_set<const VarNode *> known_vars)
        : known_vars_(std::move(known_vars)) {}

    bool Check(const PrimExpr &condition) {
      VisitExpr(condition);
      return valid_;
    }

  private:
    void VisitExpr_(const VarNode *op) final {
      valid_ &= known_vars_.count(op) != 0;
    }
    void VisitExpr_(const BufferLoadNode *) final { valid_ = false; }
    void VisitExpr_(const CallNode *) final { valid_ = false; }

    std::unordered_set<const VarNode *> known_vars_;
    bool valid_{true};
  };

  void ValidateAutoPredicate(const PrimExpr &predicate,
                             const PrimExpr &current_core) {
    std::unordered_set<const VarNode *> known_vars = loop_vars_;
    known_vars.insert(rank_id_.get());
    known_vars.insert(Downcast<Var>(current_core).get());
    // Validate the full path to each auto send, including nested conditions.
    // Other kernel arguments may differ between Rank launches.
    ICHECK(BranchConditionValidator(std::move(known_vars))
               .Check(analyzer_->Simplify(predicate)))
        << "Cannot infer automatic T.dist expectation from a sender-local "
           "condition; use expect='manual' with receiver-provided expected "
           "deltas. Auto conditions must depend only on Rank/core and proven "
           "loop variables";
  }

  Array<Stmt> MakeMarkers(SignalExpectationMap expectations) {
    Array<Stmt> markers;
    std::vector<SignalExpectation> ordered;
    ordered.reserve(expectations.size());
    for (auto &[_, expectation] : expectations) {
      ordered.push_back(std::move(expectation));
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const SignalExpectation &lhs, const SignalExpectation &rhs) {
                return lhs.sort_key < rhs.sort_key;
              });
    for (SignalExpectation &expectation : ordered) {
      PrimExpr delta =
          BuildDelta(std::move(expectation.deltas), expectation.current_core);
      markers.push_back(Evaluate(Call(DataType::Handle(), dist_expect(),
                                      {expectation.signal, delta})));
    }
    return markers;
  }

  Stmt VisitStmt_(const LetStmtNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call && call->op.same_as(dist_signal_group())) {
      ICHECK_EQ(call->args.size(), 4U);
      const DistSignalKindInfo &kind = RequireDistSignalKindInfo(
          call->args[0], "resolved signal-group kind");
      signal_group_kinds_.emplace(
          op->var.get(), SignalPlan{&kind, IsAutoExpected(call->args[3])});
      Stmt body = VisitStmt(op->body);
      signal_group_kinds_.erase(op->var.get());
      return LetStmt(op->var, op->value, body, op->span);
    }
    if (!call || !call->op.same_as(dist_signal())) {
      return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    ICHECK_EQ(call->args.size(), 3U);
    const DistSignalKindInfo &kind =
        RequireDistSignalKindInfo(call->args[0], "resolved signal kind");
    signal_kinds_.emplace(op->var.get(),
                          SignalPlan{&kind, IsAutoExpected(call->args[2])});
    Stmt body = VisitStmt(op->body);
    signal_kinds_.erase(op->var.get());
    return LetStmt(op->var, op->value, body, op->span);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    if (suppress_injection_) {
      return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    SignalExpectationMap expectations;
    ConditionalPeerCollector collector(this, const_true(), &expectations);
    collector(tvm::ffi::GetRef<Stmt>(op));
    if (expectations.empty()) {
      return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    if (op->else_case) {
      ConditionalPhaseValidator(this).Validate(op->then_case);
      ConditionalPhaseValidator(this).Validate(op->else_case.value());
    } else if (const auto *seq = op->then_case.as<SeqStmtNode>()) {
      Array<Stmt> phases;
      Array<Stmt> phase;
      for (const Stmt &stmt : seq->seq) {
        bool contains_wait = ReceiverWaitDetector().Detect(stmt);
        const auto *evaluate = stmt.as<EvaluateNode>();
        const auto *call = evaluate ? evaluate->value.as<CallNode>() : nullptr;
        ICHECK(!contains_wait || ReceiverWaitDetector::IsWait(call))
            << "T.dist receiver wait nested within a conditional send phase "
               "cannot be safely planned; split the condition into phases";
        phase.push_back(stmt);
        if (contains_wait) {
          phases.push_back(SeqStmt::Flatten(phase));
          phase.clear();
        }
      }
      if (!phase.empty()) {
        phases.push_back(SeqStmt::Flatten(phase));
      }
      if (phases.size() > 1) {
        ICHECK(!MutableConditionDetector().Detect(op->condition))
            << "T.dist conditional send phases require a stable condition; "
               "split buffer-dependent conditions explicitly around waits";
        Array<Stmt> ordered;
        for (const Stmt &body : phases) {
          ordered.push_back(VisitStmt(IfThenElse(op->condition, body)));
        }
        return SeqStmt::Flatten(ordered);
      }
    } else {
      const auto *evaluate = op->then_case.as<EvaluateNode>();
      const auto *call = evaluate ? evaluate->value.as<CallNode>() : nullptr;
      ICHECK(!ReceiverWaitDetector().Detect(op->then_case) ||
             ReceiverWaitDetector::IsWait(call))
          << "T.dist receiver wait nested within a conditional send phase "
             "cannot be safely planned; split the condition into phases";
    }
    suppress_injection_ = true;
    Stmt guarded_operations = arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    suppress_injection_ = false;
    Array<Stmt> statements = MakeMarkers(std::move(expectations));
    statements.push_back(guarded_operations);
    return SeqStmt::Flatten(statements);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // PlanDistSignals has already checked loop uniformity for auto sends.
    loop_vars_.insert(op->loop_var.get());
    Stmt stmt = arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    loop_vars_.erase(op->loop_var.get());
    return stmt;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (suppress_injection_) {
      return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    const auto *call = op->value.as<CallNode>();
    if (!call || (!call->op.same_as(dist_signal_put()) &&
                  !call->op.same_as(DistPeerPutOp::Get()) &&
                  !call->op.same_as(DistRoutedPeerPutOp::Get()))) {
      return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    SignalExpectationMap expectations;
    AccumulatePeerCall(call, const_true(), &expectations);
    Array<Stmt> statements = MakeMarkers(std::move(expectations));
    statements.push_back(tvm::ffi::GetRef<Stmt>(op));
    return SeqStmt::Flatten(statements);
  }

  bool suppress_injection_{false};
  std::unordered_set<const VarNode *> loop_vars_;
  std::unordered_map<const VarNode *, SignalPlan> signal_kinds_;
  std::unordered_map<const VarNode *, SignalPlan> signal_group_kinds_;
};

struct GenerationLayout {
  bool has_send{false};
  bool dense{false};
  bool has_compiler_index{false};
  bool row_domain{false};
};

class DistGenerationLayoutCollector : public StmtExprVisitor {
public:
  DistGenerationLayoutCollector(const VarNode *owner, bool is_group,
                                int mesh_ncols)
      : owner_(owner), is_group_(is_group), mesh_ncols_(mesh_ncols) {}

  GenerationLayout Collect(const Stmt &stmt) {
    VisitStmt(stmt);
    return {has_send_, dense_, has_compiler_index_, row_domain_};
  }

private:
  struct Endpoint {
    PrimExpr rank;
    PrimExpr row;
  };

  struct SignalMember {
    bool matches{false};
    bool dynamic{false};
    int64_t index{-1};
  };

  SignalMember ResolveMember(const PrimExpr &signal) {
    if (const auto *var = signal.as<VarNode>()) {
      return {var == owner_ && !is_group_, false, -1};
    }
    const auto *ref = signal.as<CallNode>();
    if (!ref) {
      return {};
    }
    if (ref->op.same_as(dist_signal_route())) {
      ICHECK_EQ(ref->args.size(), 2U);
      return ResolveMember(ref->args[0]);
    }
    if (!is_group_ || (!ref->op.same_as(dist_signal_ref()) &&
                       !ref->op.same_as(dist_signal_group_route()))) {
      return {};
    }
    const auto *group = ref->args[0].as<VarNode>();
    if (group != owner_) {
      return {};
    }
    if (ref->op.same_as(dist_signal_group_route())) {
      return {true, true, -1};
    }
    ICHECK_EQ(ref->args.size(), 2U);
    if (const auto *index = ref->args[1].as<IntImmNode>()) {
      return {true, false, index->value};
    }
    ICHECK(ref->args[1].dtype().is_int());
    return {true, true, -1};
  }

  bool EqualEndpoint(const Endpoint &lhs, const Endpoint &rhs) {
    return structural_equal_(lhs.rank, rhs.rank) &&
           structural_equal_(lhs.row, rhs.row);
  }

  void Record(const PrimExpr &signal, PrimExpr dst_rank, PrimExpr dst_row) {
    SignalMember member = ResolveMember(signal);
    if (!member.matches) {
      return;
    }
    has_send_ = true;
    if (member.dynamic) {
      dense_ = true;
      has_compiler_index_ = true;
      return;
    }
    int64_t key = is_group_ ? member.index : -1;
    Endpoint endpoint{std::move(dst_rank), std::move(dst_row)};
    auto it = first_endpoint_.find(key);
    if (it == first_endpoint_.end()) {
      first_endpoint_.emplace(key, std::move(endpoint));
    } else if (!EqualEndpoint(it->second, endpoint)) {
      dense_ = true;
      row_domain_ |= !structural_equal_(it->second.row, endpoint.row);
    }
  }

  void VisitExpr_(const CallNode *call) final {
    if (call->op.same_as(dist_signal_put())) {
      ICHECK_EQ(call->args.size(), 3U);
      Record(call->args[0], call->args[1],
             floordiv(call->args[2], IntImm(DataType::Int(32), mesh_ncols_)));
    } else if (call->op.same_as(DistPeerPutOp::Get())) {
      ICHECK_EQ(call->args.size(), 5U);
      Record(call->args[3], call->args[2],
             floordiv(call->args[4], IntImm(DataType::Int(32), mesh_ncols_)));
    } else if (call->op.same_as(DistRoutedPeerPutOp::Get())) {
      ICHECK_GE(call->args.size(), 5U);
      for (const PeerRouteEntry &route : ParsePeerRouteTable(call->args[0])) {
        Record(call->args[1], route.dst_rank, route.peer_row);
      }
    }
    StmtExprVisitor::VisitExpr_(call);
  }

  const VarNode *owner_;
  bool is_group_{false};
  int mesh_ncols_{1};
  bool has_send_{false};
  bool dense_{false};
  bool has_compiler_index_{false};
  bool row_domain_{false};
  StructuralEqual structural_equal_;
  std::unordered_map<int64_t, Endpoint> first_endpoint_;
};

class LowerDistPrimitiveMutator : public StmtExprMutator {
public:
  LowerDistPrimitiveMutator(int64_t world_size, int mesh_nrows, int mesh_ncols,
                            Var rank_id)
      : world_size_(world_size), mesh_nrows_(mesh_nrows),
        mesh_ncols_(mesh_ncols), rank_id_(std::move(rank_id)) {}

private:
  struct SignalInfo {
    PrimExpr kind;
    PrimExpr index;
    const DistSignalKindInfo *kind_info;
    Buffer expected;
    Buffer generation;
    bool dense_generation{false};
    bool generation_row_domain{false};
    bool used{false};
  };

  struct SignalGroupInfo {
    PrimExpr kind;
    PrimExpr base_index;
    int64_t count;
    const DistSignalKindInfo *kind_info;
    Buffer expected;
    Buffer generation;
    bool dense_generation{false};
    bool has_compiler_generation_index{false};
    bool generation_row_domain{false};
    std::string expect_mode;
    Buffer auto_completion_deltas;
    bool used{false};
  };

  struct CompletionInfo {
    PrimExpr kind;
    PrimExpr base_index;
    int64_t count;
    Buffer expect;
    Buffer pending;
    Buffer pending_count;
    bool used{false};
  };

  struct SignalSelection {
    PrimExpr kind;
    PrimExpr index;
    PrimExpr generation;
    PrimExpr predicate;
  };

  struct SignalWaitSelection {
    PrimExpr kind;
    PrimExpr index;
    PrimExpr expected;
  };

  Stmt VisitStmt_(const LetStmtNode *let_node) final {
    const auto *signal_call_node = let_node->value.as<CallNode>();
    if (signal_call_node &&
        (signal_call_node->op.same_as(dist_signal_decl()) ||
         signal_call_node->op.same_as(dist_signal_group_decl()))) {
      ICHECK(false) << "T.dist signal declarations must be resolved before "
                       "LowerDistCommunication";
    }
    if (signal_call_node && signal_call_node->op.same_as(dist_signal_group())) {
      ICHECK_EQ(signal_call_node->args.size(), 4U);
      PrimExpr kind = signal_call_node->args[0];
      int64_t base_index =
          RequireIntImm(signal_call_node->args[1], "signal-group base index");
      int64_t count =
          RequireIntImm(signal_call_node->args[2], "signal-group count");
      const DistSignalKindInfo *kind_info =
          &RequireDistSignalKindInfo(kind, "signal-group kind");
      GenerationLayout generation_layout =
          DistGenerationLayoutCollector(let_node->var.get(), /*is_group=*/true,
                                        mesh_ncols_)
              .Collect(let_node->body);
      class CompletionDomainCollector : public StmtExprVisitor {
      public:
        explicit CompletionDomainCollector(const VarNode *group)
            : group_(group) {}

        void VisitExpr_(const CallNode *call) final {
          if (call->op.same_as(dist_completion()) &&
              call->args[0].as<VarNode>() == group_) {
            std::string mode =
                RequireStringImm(call->args.back(), "completion mode");
            uses_row_domain |= mode == "row";
            uses_auto_completion |= mode == "auto";
          }
          StmtExprVisitor::VisitExpr_(call);
        }

        bool uses_row_domain{false};
        bool uses_auto_completion{false};

      private:
        const VarNode *group_;
      } completion_domain(let_node->var.get());
      completion_domain(let_node->body);
      bool needs_generation =
          generation_layout.has_send &&
          kind_info->update_mode != DistSignalUpdateMode::kIncrement;
      bool generation_row_domain = generation_layout.has_compiler_index
                                       ? completion_domain.uses_row_domain
                                       : generation_layout.row_domain;
      PrimExpr count_imm = IntImm(DataType::Int(32), count);
      std::string base_name = let_node->var->name_hint;
      Buffer expected = decl_buffer({count_imm}, kind_info->state_dtype,
                                    base_name + "_expect", "local");
      int64_t generation_factor = 1;
      if (generation_layout.dense) {
        generation_factor =
            world_size_ * (generation_row_domain ? mesh_nrows_ : 1);
      }
      PrimExpr generation_count =
          IntImm(DataType::Int(32), count * generation_factor);
      Buffer generation;
      if (needs_generation) {
        generation = decl_buffer({generation_count}, kind_info->state_dtype,
                                 base_name + "_generation", "local");
      }
      std::string expect_mode =
          RequireStringImm(signal_call_node->args[3], "signal-group expect");
      Buffer auto_completion_deltas;
      if (completion_domain.uses_auto_completion) {
        auto_completion_deltas =
            decl_buffer({count_imm}, DataType::Int(32),
                        base_name + "_completion_deltas", "local");
      }
      signal_group_info_.emplace(
          let_node->var.get(),
          SignalGroupInfo{
              kind, IntImm(DataType::Int(32), base_index), count, kind_info,
              expected, generation, generation_layout.dense,
              generation_layout.has_compiler_index, generation_row_domain,
              std::move(expect_mode), auto_completion_deltas});
      Stmt body = VisitStmt(let_node->body);
      bool used = signal_group_info_.at(let_node->var.get()).used;
      signal_group_info_.erase(let_node->var.get());
      if (!used) {
        return body;
      }

      Var expect_init_index(base_name + "_expect_init", DataType::Int(32));
      Var generation_init_index(base_name + "_generation_init",
                                DataType::Int(32));
      Stmt initialize_expect =
          For(expect_init_index, IntImm(DataType::Int(32), 0), count_imm,
              ForKind::kSerial,
              BufferStore(expected, make_zero(kind_info->state_dtype),
                          {expect_init_index}));
      Array<Stmt> statements{initialize_expect};
      if (auto_completion_deltas.defined()) {
        Var delta_index(base_name + "_completion_delta_init",
                        DataType::Int(32));
        statements.push_back(
            For(delta_index, IntImm(DataType::Int(32), 0), count_imm,
                ForKind::kSerial,
                BufferStore(auto_completion_deltas,
                            IntImm(DataType::Int(32), 0), {delta_index})));
      }
      if (generation.defined()) {
        statements.push_back(
            For(generation_init_index, IntImm(DataType::Int(32), 0),
                generation_count, ForKind::kSerial,
                BufferStore(generation, make_zero(kind_info->state_dtype),
                            {generation_init_index})));
      }
      statements.push_back(body);
      Stmt scoped = SeqStmt::Flatten(statements);
      scoped = DeclBuffer(expected, std::move(scoped));
      if (auto_completion_deltas.defined()) {
        scoped = DeclBuffer(auto_completion_deltas, std::move(scoped));
      }
      if (generation.defined()) {
        scoped = DeclBuffer(generation, std::move(scoped));
      }
      scoped = Allocate(expected->data, expected->dtype, expected->shape,
                        const_true(), std::move(scoped));
      if (auto_completion_deltas.defined()) {
        scoped = Allocate(
            auto_completion_deltas->data, auto_completion_deltas->dtype,
            auto_completion_deltas->shape, const_true(), std::move(scoped));
      }
      if (generation.defined()) {
        scoped = Allocate(generation->data, generation->dtype,
                          generation->shape, const_true(), std::move(scoped));
      }
      return scoped;
    }
    if (signal_call_node && signal_call_node->op.same_as(dist_completion())) {
      ICHECK(signal_call_node->args.size() == 2U ||
             signal_call_node->args.size() == 3U);
      SignalGroupInfo &group = LookupSignalGroup(signal_call_node->args[0]);
      group.used = true;
      std::string mode =
          RequireStringImm(signal_call_node->args.back(), "completion mode");
      Array<PrimExpr> deltas =
          ResolveCompletionDeltas(group, signal_call_node, mode);
      // Auto P2P already emitted dist_expect; protocol/manual deltas advance
      // the persistent expected state while creating this phase snapshot.
      bool advance_expected = mode != "auto";
      PrimExpr count_imm = IntImm(DataType::Int(32), group.count);
      std::string base_name = let_node->var->name_hint;
      Buffer expect = decl_buffer({count_imm}, group.kind_info->state_dtype,
                                  base_name + "_expect", "local");
      Buffer pending = decl_buffer({count_imm}, DataType::UInt(8),
                                   base_name + "_pending", "local");
      Buffer pending_count =
          decl_buffer({IntImm(DataType::Int(32), 1)}, DataType::Int(32),
                      base_name + "_pending_count", "local.var");
      CompletionInfo completion{group.kind, group.base_index, group.count,
                                expect,     pending,          pending_count};
      completion_info_.emplace(let_node->var.get(), std::move(completion));
      Stmt body = VisitStmt(let_node->body);
      completion_info_.erase(let_node->var.get());

      PrimExpr group_expect = FullRegion(group.expected, /*access_mask=*/3);
      PrimExpr completion_expect = FullRegion(expect, /*access_mask=*/2);
      PrimExpr pending_region = FullRegion(pending, /*access_mask=*/2);
      PrimExpr pending_count_load = BufferLoad(pending_count, {0});
      Array<PrimExpr> init_args{group_expect, completion_expect, pending_region,
                                pending_count_load, Bool(advance_expected)};
      for (const PrimExpr &delta : deltas) {
        init_args.push_back(delta);
      }
      Stmt initialize = Evaluate(
          Call(DataType::Handle(), dist_completion_init_(), init_args));
      Array<Stmt> statements{initialize};
      if (mode == "auto") {
        for (int64_t index = 0; index < group.count; ++index) {
          statements.push_back(BufferStore(group.auto_completion_deltas,
                                           IntImm(DataType::Int(32), 0),
                                           {IntImm(DataType::Int(32), index)}));
        }
      }
      statements.push_back(body);
      Stmt scoped = SeqStmt::Flatten(statements);
      scoped = DeclBuffer(expect, std::move(scoped));
      scoped = DeclBuffer(pending, std::move(scoped));
      scoped = DeclBuffer(pending_count, std::move(scoped));
      Map<String, ffi::Any> annotations;
      annotations.Set(tl::attr::kLocalVarInit, IntImm(DataType::Int(32), 0));
      scoped = Allocate(pending_count->data, pending_count->dtype,
                        pending_count->shape, const_true(), std::move(scoped),
                        annotations);
      scoped = Allocate(expect->data, expect->dtype, expect->shape,
                        const_true(), std::move(scoped));
      return Allocate(pending->data, pending->dtype, pending->shape,
                      const_true(), std::move(scoped));
    }
    if (!signal_call_node || !signal_call_node->op.same_as(dist_signal()) ||
        signal_call_node->args.size() != 3U) {
      return StmtExprMutator::VisitStmt_(let_node);
    }

    const DistSignalKindInfo &kind_info = RequireDistSignalKindInfo(
        signal_call_node->args[0], "resolved signal kind");
    GenerationLayout generation_layout =
        DistGenerationLayoutCollector(let_node->var.get(), /*is_group=*/false,
                                      mesh_ncols_)
            .Collect(let_node->body);
    bool needs_generation =
        generation_layout.has_send &&
        kind_info.update_mode != DistSignalUpdateMode::kIncrement;
    std::string expect_name = let_node->var->name_hint + "_expect";
    std::string generation_name = let_node->var->name_hint + "_generation";
    Buffer expected =
        decl_buffer({IntImm(DataType::Int(32), 1)}, kind_info.state_dtype,
                    expect_name, "local.var");
    int64_t generation_extent =
        generation_layout.dense
            ? world_size_ * (generation_layout.row_domain ? mesh_nrows_ : 1)
            : 1;
    Buffer generation;
    if (needs_generation) {
      generation = decl_buffer({IntImm(DataType::Int(32), generation_extent)},
                               kind_info.state_dtype, generation_name,
                               generation_layout.dense ? "local" : "local.var");
    }
    SignalInfo info{signal_call_node->args[0],
                    signal_call_node->args[1],
                    &kind_info,
                    expected,
                    generation,
                    generation_layout.dense,
                    generation_layout.row_domain};
    signal_info_.emplace(let_node->var.get(), std::move(info));
    Stmt body = VisitStmt(let_node->body);
    bool used = signal_info_.at(let_node->var.get()).used;
    signal_info_.erase(let_node->var.get());
    if (!used) {
      return body;
    }

    Stmt scoped = DeclBuffer(expected, std::move(body));
    Map<String, ffi::Any> annotations;
    annotations.Set(tl::attr::kLocalVarInit, make_zero(kind_info.state_dtype));
    scoped = Allocate(expected->data, expected->dtype, expected->shape,
                      const_true(), std::move(scoped), annotations);
    if (!generation.defined()) {
      return scoped;
    }
    if (!generation_layout.dense) {
      scoped = DeclBuffer(generation, std::move(scoped));
      return Allocate(generation->data, generation->dtype, generation->shape,
                      const_true(), std::move(scoped), annotations);
    }
    Var generation_init_index(generation_name + "_init", DataType::Int(32));
    Stmt initialize_generation =
        For(generation_init_index, IntImm(DataType::Int(32), 0),
            IntImm(DataType::Int(32), generation_extent), ForKind::kSerial,
            BufferStore(generation, make_zero(kind_info.state_dtype),
                        {generation_init_index}));
    scoped = SeqStmt::Flatten(Array<Stmt>{initialize_generation, scoped});
    scoped = DeclBuffer(generation, std::move(scoped));
    return Allocate(generation->data, generation->dtype, generation->shape,
                    const_true(), std::move(scoped));
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
      PrimExpr dst_row =
          floordiv(call->args[4], IntImm(DataType::Int(32), mesh_ncols_));
      SignalSelection signal =
          ResolveSignalSelection(call->args[3], call->args[2], dst_row);
      Array<PrimExpr> args{call->args[0], call->args[1], call->args[2],
                           signal.kind,   signal.index,  signal.generation};
      return Call(call->dtype, dist_put_(), args, call->annotations,
                  call->span);
    }
    if (call->op.same_as(DistRoutedPeerPutOp::Get())) {
      ICHECK(false) << "T.dist.routed_peer_put must appear as an Evaluate "
                       "statement";
    }
    if (call->op.same_as(dist_wait_signal())) {
      ICHECK_EQ(call->args.size(), 1U);
      SignalWaitSelection signal = ResolveSignalWait(call->args[0]);
      Array<PrimExpr> args{signal.kind, signal.index, signal.expected};
      return Call(call->dtype, dist_wait_signal_(), args, call->annotations,
                  call->span);
    }
    if (call->op.same_as(dist_expect())) {
      ICHECK_EQ(call->args.size(), 2U);
      SignalWaitSelection signal = ResolveSignalWait(call->args[0]);
      Array<PrimExpr> args{signal.expected, call->args[1]};
      return Call(call->dtype, dist_expect_(), args, call->annotations,
                  call->span);
    }
    if (call->op.same_as(dist_completion_has_pending())) {
      ICHECK_EQ(call->args.size(), 1U);
      CompletionInfo &completion = LookupCompletion(call->args[0]);
      completion.used = true;
      return BufferLoad(completion.pending_count, {0}) > 0;
    }
    if (call->op.same_as(dist_wait_any())) {
      ICHECK_EQ(call->args.size(), 1U);
      CompletionInfo &completion = LookupCompletion(call->args[0]);
      completion.used = true;
      return MakeWaitAny(completion, call->span);
    }
    return rewritten;
  }

  Stmt VisitStmt_(const EvaluateNode *evaluate_node) final {
    const auto *call = evaluate_node->value.as<CallNode>();
    if (call && call->op.same_as(dist_expect())) {
      PrimExpr lowered = VisitExpr(evaluate_node->value);
      const auto *ref = call->args[0].as<CallNode>();
      if (!ref || !ref->op.same_as(dist_signal_ref())) {
        return Evaluate(lowered);
      }
      SignalGroupInfo &group = LookupSignalGroup(ref->args[0]);
      if (!group.auto_completion_deltas.defined()) {
        return Evaluate(lowered);
      }
      int64_t index = RequireIntImm(
          ref->args[1], "auto-completion signal-group member index");
      ICHECK_GE(index, 0);
      ICHECK_LT(index, group.count);
      PrimExpr offset = IntImm(DataType::Int(32), index);
      PrimExpr accumulated = BufferLoad(group.auto_completion_deltas, {offset});
      Stmt update = BufferStore(
          group.auto_completion_deltas,
          accumulated + Cast(DataType::Int(32), call->args[1]), {offset});
      return SeqStmt::Flatten(Array<Stmt>{Evaluate(lowered), update});
    }
    if (call && call->op.same_as(dist_batch_begin())) {
      ICHECK(call->args.empty());
      return Evaluate(Call(DataType::Handle(), dist_batch_begin_(), {},
                           call->annotations, call->span));
    }
    if (call && call->op.same_as(dist_submit())) {
      ICHECK(call->args.empty());
      return Evaluate(Call(DataType::Handle(), dist_submit_(), {},
                           call->annotations, call->span));
    }
    if (call && call->op.same_as(dist_signal_put())) {
      ICHECK_EQ(call->args.size(), 3U);
      PrimExpr dst_row =
          floordiv(call->args[2], IntImm(DataType::Int(32), mesh_ncols_));
      SignalSelection signal =
          ResolveSignalSelection(call->args[0], call->args[1], dst_row);
      return Evaluate(
          Call(DataType::Handle(), dist_signal_put_(),
               {call->args[1], signal.kind, signal.index, signal.generation},
               call->annotations, call->span));
    }
    if (call && call->op.same_as(dist_wait_signal_delta())) {
      ICHECK_EQ(call->args.size(), 2U);
      SignalWaitSelection signal = ResolveSignalWait(call->args[0]);
      Stmt update = Evaluate(Call(DataType::Handle(), dist_expect_(),
                                  {signal.expected, call->args[1]},
                                  call->annotations, call->span));
      Stmt wait = Evaluate(Call(DataType::Handle(), dist_wait_signal_(),
                                {signal.kind, signal.index, signal.expected},
                                call->annotations, call->span));
      return SeqStmt::Flatten(Array<Stmt>{update, wait});
    }
    if (call && call->op.same_as(DistPeerPutOp::Get()) &&
        call->args[3].as<CallNode>() &&
        (call->args[3].as<CallNode>()->op.same_as(dist_signal_route()) ||
         call->args[3].as<CallNode>()->op.same_as(dist_signal_group_route()))) {
      const auto *signal_ref = call->args[3].as<CallNode>();
      PrimExpr lowered = VisitExpr(evaluate_node->value);
      PrimExpr active = signal_ref->op.same_as(dist_signal_route())
                            ? signal_ref->args[1]
                            : signal_ref->args[3];
      return IfThenElse(active, Evaluate(lowered));
    }
    if (call && call->op.same_as(dist_wait_all())) {
      ICHECK_EQ(call->args.size(), 1U);
      SignalGroupInfo &group = LookupSignalGroup(call->args[0]);
      group.used = true;
      Array<Stmt> waits;
      for (int64_t index = 0; index < group.count; ++index) {
        PrimExpr offset = IntImm(DataType::Int(32), index);
        Array<PrimExpr> args{group.kind, group.base_index + offset,
                             BufferLoad(group.expected, {offset})};
        waits.push_back(Evaluate(Call(DataType::Handle(), dist_wait_signal_(),
                                      args, call->annotations, call->span)));
      }
      return SeqStmt::Flatten(waits);
    }
    if (call && call->op.same_as(dist_wait_signals())) {
      ICHECK_EQ(call->args.size(), 2U);
      SignalGroupInfo &group = LookupSignalGroup(call->args[0]);
      group.used = true;
      BufferRegion deltas = NormalizeToBufferRegion(call->args[1]);
      ICHECK_EQ(deltas->region.size(), 1U);
      ICHECK_EQ(
          RequireIntImm(deltas->region[0]->extent, "expected_deltas extent"),
          group.count);
      Array<Stmt> updates;
      Array<Stmt> waits;
      for (int64_t index = 0; index < group.count; ++index) {
        PrimExpr offset = IntImm(DataType::Int(32), index);
        PrimExpr delta_index = deltas->region[0]->min + offset;
        PrimExpr expected = BufferLoad(group.expected, {offset});
        PrimExpr delta = BufferLoad(deltas->buffer, {delta_index});
        updates.push_back(
            Evaluate(Call(DataType::Handle(), dist_expect_(), {expected, delta},
                          call->annotations, call->span)));
        Array<PrimExpr> args{group.kind, group.base_index + offset, expected};
        waits.push_back(Evaluate(Call(DataType::Handle(), dist_wait_signal_(),
                                      args, call->annotations, call->span)));
      }
      for (const Stmt &wait : waits) {
        updates.push_back(wait);
      }
      return SeqStmt::Flatten(updates);
    }
    if (call && call->op.same_as(dist_wait_completion_all())) {
      ICHECK_EQ(call->args.size(), 1U);
      CompletionInfo &completion = LookupCompletion(call->args[0]);
      completion.used = true;
      Var index("dist_wait_any_index", DataType::Int(32));
      Stmt wait =
          LetStmt(index, MakeWaitAny(completion, call->span), Evaluate(0));
      return While(BufferLoad(completion.pending_count, {0}) > 0, wait);
    }
    if (!call || !call->op.same_as(DistRoutedPeerPutOp::Get())) {
      return StmtExprMutator::VisitStmt_(evaluate_node);
    }
    ICHECK_GE(call->args.size(), 5U);
    ICHECK_EQ((call->args.size() - 3U) % 2U, 0U);
    std::vector<PeerRouteEntry> routes = ParsePeerRouteTable(call->args[0]);
    ICHECK_EQ(routes.size(), (call->args.size() - 3U) / 2U);
    PrimExpr current_row =
        floordiv(call->args[2], IntImm(DataType::Int(32), mesh_ncols_));
    Array<Stmt> sends;
    for (size_t index = 0; index < routes.size(); ++index) {
      const PeerRouteEntry &route = routes[index];
      SignalSelection signal =
          ResolveSignalSelection(call->args[1], route.dst_rank, route.peer_row);
      Array<PrimExpr> args{call->args[3 + index * 2],
                           call->args[4 + index * 2],
                           route.dst_rank,
                           signal.kind,
                           signal.index,
                           signal.generation};
      Stmt send = Evaluate(Call(DataType::Handle(), dist_put_(), args));
      sends.push_back(IfThenElse(
          And(current_row == route.peer_row, signal.predicate), send));
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

  SignalGroupInfo &LookupSignalGroup(const PrimExpr &group) {
    const auto *var = group.as<VarNode>();
    ICHECK(var) << "T.dist primitive expected a SignalList Var, got " << group;
    auto it = signal_group_info_.find(var);
    ICHECK(it != signal_group_info_.end())
        << "Cannot find T.dist.signals definition for " << group;
    return it->second;
  }

  CompletionInfo &LookupCompletion(const PrimExpr &completion) {
    const auto *var = completion.as<VarNode>();
    ICHECK(var) << "T.dist primitive expected a DistCompletion Var, got "
                << completion;
    auto it = completion_info_.find(var);
    ICHECK(it != completion_info_.end())
        << "Cannot find T.dist completion definition for " << completion;
    return it->second;
  }

  PrimExpr FullRegion(const Buffer &buffer, int access_mask) {
    Array<Range> region;
    for (const PrimExpr &extent : buffer->shape) {
      region.push_back(
          Range::FromMinExtent(IntImm(DataType::Int(32), 0), extent));
    }
    return MakeRegionExpr(buffer, region, access_mask);
  }

  PrimExpr DestinationIndex(const PrimExpr &dst_rank, const PrimExpr &dst_row,
                            bool row_domain) {
    if (!row_domain) {
      return dst_rank;
    }
    return dst_rank * IntImm(DataType::Int(32), mesh_nrows_) + dst_row;
  }

  PrimExpr GenerationValue(const SignalInfo &info, const PrimExpr &dst_rank,
                           const PrimExpr &dst_row) {
    if (info.kind_info->update_mode == DistSignalUpdateMode::kIncrement) {
      return make_zero(info.kind_info->state_dtype);
    }
    ICHECK(info.generation.defined());
    PrimExpr offset =
        info.dense_generation
            ? DestinationIndex(dst_rank, dst_row, info.generation_row_domain)
            : IntImm(DataType::Int(32), 0);
    return BufferLoad(info.generation, {offset});
  }

  PrimExpr GenerationValue(const SignalGroupInfo &group,
                           const PrimExpr &signal_index,
                           const PrimExpr &dst_rank, const PrimExpr &dst_row) {
    if (group.kind_info->update_mode == DistSignalUpdateMode::kIncrement) {
      return make_zero(group.kind_info->state_dtype);
    }
    ICHECK(group.generation.defined());
    PrimExpr offset = signal_index;
    if (group.dense_generation) {
      int64_t destination_count =
          world_size_ * (group.generation_row_domain ? mesh_nrows_ : 1);
      offset = signal_index * IntImm(DataType::Int(32), destination_count) +
               DestinationIndex(dst_rank, dst_row, group.generation_row_domain);
    }
    return BufferLoad(group.generation, {offset});
  }

  SignalSelection ResolveSignalSelection(const PrimExpr &signal,
                                         const PrimExpr &dst_rank,
                                         const PrimExpr &dst_row) {
    if (const auto *var = signal.as<VarNode>()) {
      SignalInfo &info = LookupSignal(tvm::ffi::GetRef<Var>(var));
      info.used = true;
      return {info.kind, info.index, GenerationValue(info, dst_rank, dst_row),
              const_true()};
    }
    const auto *ref = signal.as<CallNode>();
    if (ref && ref->op.same_as(dist_signal_route())) {
      ICHECK_EQ(ref->args.size(), 2U);
      SignalSelection selection =
          ResolveSignalSelection(ref->args[0], dst_rank, dst_row);
      selection.predicate = And(selection.predicate, ref->args[1]);
      return selection;
    }
    ICHECK(ref && (ref->op.same_as(dist_signal_ref()) ||
                   ref->op.same_as(dist_signal_group_route())))
        << "T.dist primitive expected a signal reference, got " << signal;
    SignalGroupInfo &group = LookupSignalGroup(ref->args[0]);
    group.used = true;
    PrimExpr signal_index = ref->args[1];
    ICHECK(signal_index.dtype().is_int());
    if (ref->op.same_as(dist_signal_ref())) {
      ICHECK_EQ(ref->args.size(), 2U);
      return {group.kind, group.base_index + signal_index,
              GenerationValue(group, signal_index, dst_rank, dst_row),
              const_true()};
    }
    ICHECK_EQ(ref->args.size(), 4U);
    ICHECK(group.has_compiler_generation_index);
    ICHECK(ref->args[2].dtype().is_int());
    PrimExpr generation =
        group.kind_info->update_mode == DistSignalUpdateMode::kIncrement
            ? make_zero(group.kind_info->state_dtype)
            : BufferLoad(group.generation, {ref->args[2]});
    return {group.kind, group.base_index + signal_index, generation,
            ref->args[3]};
  }

  SignalWaitSelection ResolveSignalWait(const PrimExpr &signal) {
    if (const auto *var = signal.as<VarNode>()) {
      SignalInfo &info = LookupSignal(tvm::ffi::GetRef<Var>(var));
      info.used = true;
      return {info.kind, info.index, BufferLoad(info.expected, {0})};
    }
    const auto *ref = signal.as<CallNode>();
    ICHECK(ref && ref->op.same_as(dist_signal_ref()))
        << "T.dist wait expected a signal or static SignalList member, got "
        << signal;
    ICHECK_EQ(ref->args.size(), 2U);
    SignalGroupInfo &group = LookupSignalGroup(ref->args[0]);
    group.used = true;
    PrimExpr index = ref->args[1];
    ICHECK(index.dtype().is_int());
    return {group.kind, group.base_index + index,
            BufferLoad(group.expected, {index})};
  }

  Array<PrimExpr> ResolveCompletionDeltas(SignalGroupInfo &group,
                                          const CallNode *call,
                                          const std::string &mode) {
    Array<PrimExpr> deltas;
    deltas.reserve(group.count);
    if (mode == "auto") {
      ICHECK_EQ(call->args.size(), 2U);
      ICHECK_EQ(group.expect_mode, "auto");
      ICHECK(group.auto_completion_deltas.defined());
      for (int64_t index = 0; index < group.count; ++index) {
        deltas.push_back(BufferLoad(group.auto_completion_deltas,
                                    {IntImm(DataType::Int(32), index)}));
      }
      return deltas;
    }
    if (mode == "all_gather") {
      ICHECK_EQ(call->args.size(), 2U);
      ICHECK_EQ(group.expect_mode, "auto");
      ICHECK_EQ(group.count, world_size_);
      for (int64_t index = 0; index < group.count; ++index) {
        deltas.push_back(Cast(DataType::Int(32),
                              rank_id_ != IntImm(rank_id_.dtype(), index)));
      }
      return deltas;
    }

    ICHECK_EQ(call->args.size(), 3U);
    ICHECK_EQ(group.expect_mode, "manual");
    BufferRegion source = NormalizeToBufferRegion(call->args[1]);
    if (mode == "manual") {
      ICHECK_EQ(source->region.size(), 1U);
      ICHECK_EQ(RequireIntImm(source->region[0]->extent,
                              "completion expected_deltas extent"),
                group.count);
      for (int64_t index = 0; index < group.count; ++index) {
        PrimExpr source_index =
            source->region[0]->min + IntImm(DataType::Int(32), index);
        deltas.push_back(BufferLoad(source->buffer, {source_index}));
      }
      return deltas;
    }

    ICHECK(mode == "rank" || mode == "row")
        << "Unsupported T.dist completion mode " << mode;
    bool row_domain = mode == "row";
    ICHECK_EQ(source->region.size(), row_domain ? 2U : 1U);
    int64_t row_count = row_domain ? mesh_nrows_ : 1;
    ICHECK_EQ(group.count, world_size_ * row_count);
    for (int64_t index = 0; index < group.count; ++index) {
      int64_t source_rank = row_domain ? index / row_count : index;
      int64_t source_row = row_domain ? index % row_count : 0;
      Array<PrimExpr> indices{source->region[0]->min +
                              IntImm(DataType::Int(32), source_rank)};
      if (row_domain) {
        indices.push_back(source->region[1]->min +
                          IntImm(DataType::Int(32), source_row));
      }
      PrimExpr count = BufferLoad(source->buffer, indices);
      PrimExpr active = And(rank_id_ != IntImm(rank_id_.dtype(), source_rank),
                            count > make_zero(count.dtype()));
      deltas.push_back(Cast(DataType::Int(32), active));
    }
    return deltas;
  }

  PrimExpr MakeWaitAny(CompletionInfo &completion, Span span) {
    return Call(DataType::Int(32), dist_wait_any_(),
                {completion.kind, completion.base_index,
                 IntImm(DataType::Int(32), completion.count),
                 FullRegion(completion.expect, /*access_mask=*/1),
                 FullRegion(completion.pending, /*access_mask=*/3),
                 BufferLoad(completion.pending_count, {0})},
                {}, span);
  }

  std::unordered_map<const VarNode *, SignalInfo> signal_info_;
  std::unordered_map<const VarNode *, SignalGroupInfo> signal_group_info_;
  std::unordered_map<const VarNode *, CompletionInfo> completion_info_;
  int64_t world_size_{1};
  int mesh_nrows_{1};
  int mesh_ncols_{1};
  Var rank_id_;
};

PrimFunc Run(PrimFunc func) {
  if (!IsMultiRank(func)) {
    return func;
  }
  DistOpDetector detector(/*high_level=*/true);
  if (!detector.Detect(func->body)) {
    return func;
  }
  auto context = GetDistPassContext(func);
  ICHECK(context);
  func = DistExpectationPlanner::Rewrite(std::move(func));
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  ICHECK(target && TargetIsSunmmio(target.value()));
  auto mesh = GetSunmmioMeshConfig(target.value());
  Stmt body =
      LowerDistPrimitiveMutator(context.value().world_size, mesh.nrow,
                                mesh.ncol, context.value().rank_id)(func->body);
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
