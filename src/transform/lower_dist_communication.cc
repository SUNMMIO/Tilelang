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
      kind = kind_it->second;
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
      kind = kind_it->second;
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
    if (call->op.same_as(DistPeerPutOp::Get())) {
      ICHECK_EQ(call->args.size(), 5U);
      if (const auto *signal_ref = call->args[3].as<CallNode>()) {
        if (signal_ref->op.same_as(dist_signal_route()) ||
            signal_ref->op.same_as(dist_signal_group_route())) {
          return;
        }
        ICHECK(signal_ref->op.same_as(dist_signal_ref()));
      }
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
      if (const auto *signal_ref = call->args[1].as<CallNode>()) {
        if (signal_ref->op.same_as(dist_signal_route()) ||
            signal_ref->op.same_as(dist_signal_group_route())) {
          return;
        }
        ICHECK(signal_ref->op.same_as(dist_signal_ref()));
      }
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
    SignalExpectationMap *expectations_;
    int loop_depth_{0};
  };

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
      ICHECK_EQ(call->args.size(), 3U);
      const DistSignalKindInfo &kind = RequireDistSignalKindInfo(
          call->args[0], "resolved signal-group kind");
      signal_group_kinds_.emplace(op->var.get(), &kind);
      Stmt body = VisitStmt(op->body);
      signal_group_kinds_.erase(op->var.get());
      return LetStmt(op->var, op->value, body, op->span);
    }
    if (!call || !call->op.same_as(dist_signal())) {
      return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    ICHECK_EQ(call->args.size(), 2U);
    const DistSignalKindInfo &kind =
        RequireDistSignalKindInfo(call->args[0], "resolved signal kind");
    signal_kinds_.emplace(op->var.get(), &kind);
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
    SignalExpectationMap expectations;
    AccumulatePeerCall(call, const_true(), &expectations);
    Array<Stmt> statements = MakeMarkers(std::move(expectations));
    statements.push_back(tvm::ffi::GetRef<Stmt>(op));
    return SeqStmt::Flatten(statements);
  }

  bool suppress_injection_{false};
  std::unordered_map<const VarNode *, const DistSignalKindInfo *> signal_kinds_;
  std::unordered_map<const VarNode *, const DistSignalKindInfo *>
      signal_group_kinds_;
};

class LowerDistPrimitiveMutator : public StmtExprMutator {
public:
  LowerDistPrimitiveMutator(int mesh_nrows, int mesh_ncols, Var rank_id)
      : mesh_nrows_(mesh_nrows), mesh_ncols_(mesh_ncols),
        rank_id_(std::move(rank_id)) {}

private:
  struct SignalInfo {
    PrimExpr kind;
    PrimExpr index;
    Buffer expected;
    Buffer generation;
    bool used{false};
  };

  struct SignalGroupInfo {
    PrimExpr kind;
    PrimExpr base_index;
    int64_t count;
    DataType state_dtype;
    Buffer expected;
    Buffer generation;
    bool used{false};
  };

  struct CompletionInfo {
    PrimExpr kind;
    PrimExpr base_index;
    int64_t count;
    Buffer expect;
    Buffer pending;
    Buffer pending_count;
    PrimExpr dst;
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
      ICHECK_EQ(signal_call_node->args.size(), 3U);
      PrimExpr kind = signal_call_node->args[0];
      int64_t base_index =
          RequireIntImm(signal_call_node->args[1], "signal-group base index");
      int64_t count =
          RequireIntImm(signal_call_node->args[2], "signal-group count");
      const DistSignalKindInfo *kind_info =
          &RequireDistSignalKindInfo(kind, "signal-group kind");
      class GroupDomainCollector : public StmtExprVisitor {
      public:
        explicit GroupDomainCollector(const VarNode *group) : group_(group) {}

        void VisitExpr_(const CallNode *call) final {
          if (call->op.same_as(dist_completion()) &&
              call->args[0].as<VarNode>() == group_ &&
              RequireStringImm(call->args[3], "completion domain") == "row") {
            uses_row_domain = true;
          }
          StmtExprVisitor::VisitExpr_(call);
        }

        bool uses_row_domain{false};

      private:
        const VarNode *group_;
      } domain_collector(let_node->var.get());
      domain_collector(let_node->body);
      int64_t generation_factor =
          domain_collector.uses_row_domain ? mesh_nrows_ : 1;
      PrimExpr count_imm = IntImm(DataType::Int(32), count);
      std::string base_name = let_node->var->name_hint;
      Buffer expected = decl_buffer({count_imm}, kind_info->state_dtype,
                                    base_name + "_expect", "local");
      PrimExpr generation_count =
          IntImm(DataType::Int(32), count * generation_factor);
      Buffer generation =
          decl_buffer({generation_count}, kind_info->state_dtype,
                      base_name + "_generation", "local");
      signal_group_info_.emplace(
          let_node->var.get(),
          SignalGroupInfo{kind, IntImm(DataType::Int(32), base_index), count,
                          kind_info->state_dtype, expected, generation});
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
      Stmt initialize_generation =
          For(generation_init_index, IntImm(DataType::Int(32), 0),
              generation_count, ForKind::kSerial,
              BufferStore(generation, make_zero(kind_info->state_dtype),
                          {generation_init_index}));
      Stmt scoped = SeqStmt::Flatten(
          Array<Stmt>{initialize_expect, initialize_generation, body});
      scoped = DeclBuffer(expected, std::move(scoped));
      scoped = DeclBuffer(generation, std::move(scoped));
      scoped = Allocate(expected->data, expected->dtype, expected->shape,
                        const_true(), std::move(scoped));
      return Allocate(generation->data, generation->dtype, generation->shape,
                      const_true(), std::move(scoped));
    }
    if (signal_call_node && signal_call_node->op.same_as(dist_completion())) {
      ICHECK_EQ(signal_call_node->args.size(), 4U);
      SignalGroupInfo &group = LookupSignalGroup(signal_call_node->args[0]);
      group.used = true;
      PrimExpr count_imm = IntImm(DataType::Int(32), group.count);
      std::string base_name = let_node->var->name_hint;
      Buffer expect = decl_buffer({count_imm}, group.state_dtype,
                                  base_name + "_expect", "local");
      Buffer pending = decl_buffer({count_imm}, DataType::UInt(8),
                                   base_name + "_pending", "local");
      Buffer pending_count =
          decl_buffer({IntImm(DataType::Int(32), 1)}, DataType::Int(32),
                      base_name + "_pending_count", "local.var");
      CompletionInfo completion{
          group.kind,    group.base_index,         group.count, expect, pending,
          pending_count, signal_call_node->args[2]};
      completion_info_.emplace(let_node->var.get(), std::move(completion));
      Stmt body = VisitStmt(let_node->body);
      completion_info_.erase(let_node->var.get());

      PrimExpr group_expect = FullRegion(group.expected, /*access_mask=*/3);
      PrimExpr completion_expect = FullRegion(expect, /*access_mask=*/2);
      PrimExpr pending_region = FullRegion(pending, /*access_mask=*/2);
      PrimExpr pending_count_load = BufferLoad(pending_count, {0});
      Stmt initialize = Evaluate(Call(
          DataType::Handle(), dist_completion_init_(),
          {group_expect, completion_expect, pending_region, pending_count_load,
           signal_call_node->args[1], signal_call_node->args[3], rank_id_}));
      Stmt scoped = SeqStmt::Flatten(Array<Stmt>{initialize, body});
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
        signal_call_node->args.size() != 2U) {
      return StmtExprMutator::VisitStmt_(let_node);
    }

    const DistSignalKindInfo &kind_info = RequireDistSignalKindInfo(
        signal_call_node->args[0], "resolved signal kind");
    std::string expect_name = let_node->var->name_hint + "_expect";
    std::string generation_name = let_node->var->name_hint + "_generation";
    Buffer expected =
        decl_buffer({IntImm(DataType::Int(32), 1)}, kind_info.state_dtype,
                    expect_name, "local.var");
    Buffer generation =
        decl_buffer({IntImm(DataType::Int(32), 1)}, kind_info.state_dtype,
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
    annotations.Set(tl::attr::kLocalVarInit, make_zero(kind_info.state_dtype));
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
      SignalSelection signal = ResolveSignalSelection(call->args[3]);
      Array<PrimExpr> args{call->args[0], call->args[1], call->args[2],
                           signal.kind,   signal.index,  signal.generation};
      return Call(call->dtype, dist_put_(), args, call->annotations,
                  call->span);
    }
    if (call->op.same_as(DistRoutedPeerPutOp::Get())) {
      ICHECK(false) << "T.dist.routed_peer_put must appear as an Evaluate "
                       "statement";
    }
    if (call->op.same_as(DistWaitSignalOp::Get())) {
      ICHECK_EQ(call->args.size(), 2U);
      SignalWaitSelection signal = ResolveSignalWait(call->args[0]);
      Array<PrimExpr> args{signal.kind, signal.index, signal.expected,
                           call->args[1]};
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
    if (call && call->op.same_as(dist_signal_put())) {
      ICHECK_EQ(call->args.size(), 3U);
      SignalSelection signal = ResolveSignalSelection(call->args[0]);
      return Evaluate(
          Call(DataType::Handle(), dist_signal_put_(),
               {call->args[1], signal.kind, signal.index, signal.generation},
               call->annotations, call->span));
    }
    if (call && call->op.same_as(dist_wait_barrier())) {
      ICHECK_EQ(call->args.size(), 1U);
      SignalWaitSelection signal = ResolveSignalWait(call->args[0]);
      return Evaluate(Call(DataType::Handle(), dist_wait_barrier_(),
                           {signal.kind, signal.index, signal.expected},
                           call->annotations, call->span));
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
      ICHECK_EQ(call->args.size(), 2U);
      SignalGroupInfo &group = LookupSignalGroup(call->args[1]);
      group.used = true;
      Array<Stmt> waits;
      for (int64_t index = 0; index < group.count; ++index) {
        PrimExpr offset = IntImm(DataType::Int(32), index);
        Array<PrimExpr> args{group.kind, group.base_index + offset,
                             BufferLoad(group.expected, {offset}),
                             call->args[0]};
        waits.push_back(Evaluate(Call(DataType::Handle(), dist_wait_signal_(),
                                      args, call->annotations, call->span)));
      }
      return SeqStmt::Flatten(waits);
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
    SignalSelection signal = ResolveSignalSelection(call->args[1]);
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

  SignalSelection ResolveSignalSelection(const PrimExpr &signal) {
    if (const auto *var = signal.as<VarNode>()) {
      SignalInfo &info = LookupSignal(tvm::ffi::GetRef<Var>(var));
      info.used = true;
      return {info.kind, info.index, BufferLoad(info.generation, {0}),
              const_true()};
    }
    const auto *ref = signal.as<CallNode>();
    if (ref && ref->op.same_as(dist_signal_route())) {
      ICHECK_EQ(ref->args.size(), 2U);
      SignalSelection selection = ResolveSignalSelection(ref->args[0]);
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
              BufferLoad(group.generation, {signal_index}), const_true()};
    }
    ICHECK_EQ(ref->args.size(), 4U);
    PrimExpr generation_index = ref->args[2];
    ICHECK(generation_index.dtype().is_int());
    return {group.kind, group.base_index + signal_index,
            BufferLoad(group.generation, {generation_index}), ref->args[3]};
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

  PrimExpr MakeWaitAny(CompletionInfo &completion, Span span) {
    return Call(DataType::Int(32), dist_wait_any_(),
                {completion.kind, completion.base_index,
                 IntImm(DataType::Int(32), completion.count),
                 FullRegion(completion.expect, /*access_mask=*/1),
                 FullRegion(completion.pending, /*access_mask=*/3),
                 BufferLoad(completion.pending_count, {0}), completion.dst},
                {}, span);
  }

  std::unordered_map<const VarNode *, SignalInfo> signal_info_;
  std::unordered_map<const VarNode *, SignalGroupInfo> signal_group_info_;
  std::unordered_map<const VarNode *, CompletionInfo> completion_info_;
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
  Stmt body = LowerDistPrimitiveMutator(mesh.nrow, mesh.ncol,
                                        context.value().rank_id)(func->body);
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
