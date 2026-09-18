/*!
 * \file tl/transform/lower_dist_routing.cc
 * \brief Validate, plan, and lower Rank-level P2P routes.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/transform.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "../op/comm.h"
#include "../op/copy.h"
#include "../op/utils.h"
#include "dist_transform_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;
using namespace dist_transform;

namespace {

class DistRankRoutedPutNormalizer : public StmtExprMutator {
public:
  static PrimFunc Rewrite(PrimFunc func) {
    auto context = GetDistPassContext(func);
    if (!context) {
      return func;
    }
    DistRankRoutedPutNormalizer rewriter(context.value().world_size,
                                         context.value().rank_id);
    Stmt body = rewriter(func->body);
    if (!body.same_as(func->body)) {
      func.CopyOnWrite()->body = std::move(body);
    }
    return func;
  }

private:
  DistRankRoutedPutNormalizer(int64_t world_size, Var rank_id)
      : world_size_(world_size), rank_id_(std::move(rank_id)) {}

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call_node = op->value.as<CallNode>();
    if (!call_node || !call_node->op.same_as(dist_rank_routed_put())) {
      return StmtExprMutator::VisitStmt_(op);
    }
    ICHECK_EQ(call_node->args.size(), 6U);
    PrimExpr src_rank = call_node->args[3];
    Array<PrimExpr> args{call_node->args[0], call_node->args[1],
                         call_node->args[2], call_node->args[4],
                         call_node->args[5]};
    Stmt routed = Evaluate(Call(DataType::Handle(), dist_routed_put(), args,
                                call_node->annotations, call_node->span));
    if (src_rank.same_as(rank_id_)) {
      return routed;
    }
    int64_t src_rank_value =
        RequireIntImm(src_rank, "T.dist.routed_put src_rank");
    ICHECK_GE(src_rank_value, 0);
    ICHECK_LT(src_rank_value, world_size_)
        << "T.dist.routed_put src_rank is outside [0, " << world_size_
        << "): " << src_rank_value;
    return IfThenElse(rank_id_ == src_rank, routed);
  }

  int64_t world_size_;
  Var rank_id_;
};

class DistLocalRouteLowerer : public DistRouteMutatorBase {
public:
  using DistRouteMutatorBase::DistRouteMutatorBase;

  static PrimFunc Rewrite(PrimFunc func) {
    auto context = GetDistPassContext(func);
    if (!context) {
      return func;
    }
    arith::Analyzer analyzer;
    DistLocalRouteLowerer rewriter(&analyzer, context.value().target,
                                   context.value().world_size,
                                   context.value().rank_id);
    Stmt body = rewriter.VisitStmt(func->body);
    if (!body.same_as(func->body)) {
      func.CopyOnWrite()->body = std::move(body);
    }
    return func;
  }

private:
  struct Locality {
    bool has_local{false};
    bool has_remote{false};
  };

  Stmt VisitStmt_(const BlockNode *op) final {
    alloc_buffer_stack_.emplace_back();
    Block block = Downcast<Block>(arith::IRMutatorWithAnalyzer::VisitStmt_(op));
    Array<Buffer> new_buffers = alloc_buffer_stack_.back();
    alloc_buffer_stack_.pop_back();
    if (!new_buffers.empty()) {
      Array<Buffer> alloc_buffers = block->alloc_buffers;
      for (const Buffer &buffer : new_buffers) {
        alloc_buffers.push_back(buffer);
      }
      block.CopyOnWrite()->alloc_buffers = std::move(alloc_buffers);
    }
    return block;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call_node = op->value.as<CallNode>();
    if (!call_node) {
      return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    Call call = tvm::ffi::GetRef<Call>(call_node);
    if (call->op.same_as(DistPutOp::Get())) {
      return LowerPut(call);
    }
    if (call->op.same_as(dist_routed_put())) {
      return LowerRoutedPut(call);
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Locality ValidateAndClassify(const PrimExpr &dst_rank,
                               const PrimExpr &dst_row, const Var &current_core,
                               std::optional<int64_t> fixed_src_row,
                               const char *op_name) {
    Locality result;
    int64_t row_begin = fixed_src_row.value_or(0);
    int64_t row_end = fixed_src_row ? row_begin + 1 : mesh_nrows_;
    for (int64_t src_rank = 0; src_rank < world_size_; ++src_rank) {
      for (int64_t src_row = row_begin; src_row < row_end; ++src_row) {
        PrimExpr resolved_rank =
            RewriteForSource(dst_rank, current_core, src_row, src_rank);
        PrimExpr resolved_row =
            RewriteForSource(dst_row, current_core, src_row, src_rank);
        ValidateResolvedEndpoint(resolved_rank, resolved_row, op_name);
        bool is_local = analyzer_->CanProve(resolved_rank == I32(src_rank));
        bool is_remote =
            analyzer_->CanProve(Not(resolved_rank == I32(src_rank)));
        ICHECK(is_local || is_remote)
            << op_name
            << " cannot determine whether destination Rank is "
               "local for source Rank "
            << src_rank << ": " << resolved_rank;
        result.has_local |= is_local;
        result.has_remote |= is_remote;
      }
    }
    return result;
  }

  Stmt GuardCopyToRow(Stmt copy, const PrimExpr &current_core,
                      const PrimExpr &owner_core) {
    PrimExpr condition =
        analyzer_->Simplify(CurrentRow(current_core) == CurrentRow(owner_core));
    if (analyzer_->CanProve(condition)) {
      return copy;
    }
    return IfThenElse(condition, copy);
  }

  BufferRegion CreateLocalDestinationStaging(const BufferRegion &destination) {
    ICHECK(!alloc_buffer_stack_.empty());
    std::string name = destination->buffer->name + "_dist_local_stage_" +
                       std::to_string(local_stage_counter_++);
    Buffer stage = MakeCompactBufferLike(
        destination->buffer, destination->region, kSunmmioScopeRSRAM, name);
    alloc_buffer_stack_.back().push_back(stage);
    return BufferRegion(stage, MakeCompactRegion(destination->region));
  }

  Stmt MakeLocalTransfer(const PrimExpr &src, const PrimExpr &dst,
                         const PrimExpr &current_core, const PrimExpr &src_core,
                         const PrimExpr &dst_core) {
    BufferRegion dst_region = NormalizeToBufferRegion(dst);
    Stmt copy = Evaluate(Call(DataType::Handle(), Copy::Get(), {src, dst}));
    Stmt guarded_copy = GuardCopyToRow(copy, current_core, src_core);

    PrimExpr same_row =
        analyzer_->Simplify(CurrentRow(src_core) == CurrentRow(dst_core));
    if (analyzer_->CanProve(same_row)) {
      return guarded_copy;
    }

    Stmt cross_row;
    if (IsDramScope(dst_region->buffer.scope())) {
      BufferRegion staging = CreateLocalDestinationStaging(dst_region);
      PrimExpr stage_write = MakeRegionExpr(staging->buffer, staging->region,
                                            /*access_mask=*/2);
      PrimExpr stage_read = MakeRegionExpr(staging->buffer, staging->region,
                                           /*access_mask=*/1);
      Stmt comm_put =
          Evaluate(Call(DataType::Handle(), PutOp::Get(),
                        {src, stage_write, I32(-1), src_core, dst_core}));
      Stmt write_back =
          Evaluate(Call(DataType::Handle(), Copy::Get(), {stage_read, dst}));
      cross_row = SeqStmt::Flatten(Array<Stmt>{
          comm_put, GuardCopyToRow(write_back, current_core, dst_core)});
    } else {
      cross_row = Evaluate(Call(DataType::Handle(), PutOp::Get(),
                                {src, dst, I32(-1), src_core, dst_core}));
    }
    if (analyzer_->CanProve(Not(same_row))) {
      return cross_row;
    }
    return IfThenElse(same_row, guarded_copy, cross_row);
  }

  Stmt SelectLocalOrRemote(const Locality &locality,
                           const PrimExpr &local_condition, Stmt local,
                           Stmt remote) {
    if (!locality.has_local) {
      return remote;
    }
    if (!locality.has_remote) {
      return local;
    }
    return IfThenElse(analyzer_->Simplify(local_condition), local, remote);
  }

  Stmt GuardDynamicSignal(const PrimExpr &signal, Stmt stmt) {
    const auto *ref = signal.as<CallNode>();
    if (!ref) {
      return stmt;
    }
    if (ref->op.same_as(dist_signal_ref())) {
      return stmt;
    }
    if (ref->op.same_as(dist_signal_route())) {
      ICHECK_EQ(ref->args.size(), 2U);
      return IfThenElse(ref->args[1], std::move(stmt));
    }
    ICHECK(ref->op.same_as(dist_signal_group_route()));
    ICHECK_EQ(ref->args.size(), 4U);
    return IfThenElse(ref->args[3], std::move(stmt));
  }

  Stmt LowerPut(const Call &call) {
    ICHECK_EQ(call->args.size(), 6U);
    const auto *current_core_node = call->args[5].as<VarNode>();
    ICHECK(current_core_node);
    Var current_core = tvm::ffi::GetRef<Var>(current_core_node);
    ValidateRouteExpression(call->args[2], "dst_rank");
    ValidateRouteExpression(call->args[3], "dst_row");
    Locality locality = ValidateAndClassify(
        call->args[2], call->args[3], current_core, std::nullopt, "T.dist.put");
    if (!locality.has_local) {
      return Evaluate(call);
    }
    bool all_local_routes_are_peer = true;
    for (int64_t src_rank = 0; src_rank < world_size_; ++src_rank) {
      for (int64_t src_row = 0; src_row < mesh_nrows_; ++src_row) {
        PrimExpr resolved_rank =
            RewriteForSource(call->args[2], current_core, src_row, src_rank);
        if (!analyzer_->CanProve(resolved_rank == I32(src_rank))) {
          continue;
        }
        PrimExpr resolved_row =
            RewriteForSource(call->args[3], current_core, src_row, src_rank);
        all_local_routes_are_peer &=
            analyzer_->CanProve(resolved_row == I32(src_row));
      }
    }

    Stmt local;
    if (all_local_routes_are_peer) {
      local = MakeLocalTransfer(call->args[0], call->args[1], current_core,
                                current_core, current_core);
    } else {
      Array<Stmt> transfers;
      PrimExpr current_col = CurrentCol(current_core);
      for (int64_t src_row = 0; src_row < mesh_nrows_; ++src_row) {
        PrimExpr src = RewriteForSource(call->args[0], current_core, src_row);
        PrimExpr dst = RewriteForSource(call->args[1], current_core, src_row);
        PrimExpr dst_row =
            RewriteForSource(call->args[3], current_core, src_row);
        PrimExpr src_core = I32(src_row * mesh_ncols_) + current_col;
        PrimExpr dst_core = dst_row * I32(mesh_ncols_) + current_col;
        transfers.push_back(
            MakeLocalTransfer(src, dst, current_core, src_core, dst_core));
      }
      local = SeqStmt::Flatten(transfers);
    }
    Stmt lowered = SelectLocalOrRemote(locality, call->args[2] == rank_id_,
                                       local, Evaluate(call));
    return GuardDynamicSignal(call->args[4], std::move(lowered));
  }

  bool SameRoute(const NormalRouteEntry &lhs, const NormalRouteEntry &rhs,
                 const Var &current_core) {
    if (lhs.origin_src_row != rhs.origin_src_row) {
      return false;
    }
    for (int64_t src_rank = 0; src_rank < world_size_; ++src_rank) {
      PrimExpr lhs_rank = RewriteForSource(lhs.dst_rank, current_core,
                                           lhs.origin_src_row, src_rank);
      PrimExpr rhs_rank = RewriteForSource(rhs.dst_rank, current_core,
                                           rhs.origin_src_row, src_rank);
      PrimExpr lhs_row = RewriteForSource(lhs.dst_row, current_core,
                                          lhs.origin_src_row, src_rank);
      PrimExpr rhs_row = RewriteForSource(rhs.dst_row, current_core,
                                          rhs.origin_src_row, src_rank);
      if (!analyzer_->CanProve(lhs_rank == rhs_rank) ||
          !analyzer_->CanProve(lhs_row == rhs_row)) {
        return false;
      }
    }
    return true;
  }

  Stmt LowerRoutedPut(const Call &call) {
    ICHECK_EQ(call->args.size(), 5U);
    const auto *current_core_node = call->args[4].as<VarNode>();
    ICHECK(current_core_node);
    Var current_core = tvm::ffi::GetRef<Var>(current_core_node);
    std::vector<NormalRouteEntry> routes = ParseNormalRouteTable(call->args[2]);
    Array<Stmt> statements;
    for (size_t index = 0; index < routes.size(); ++index) {
      const NormalRouteEntry &route = routes[index];
      ValidateRouteExpression(route.dst_rank, "dst_rank");
      ValidateRouteExpression(route.dst_row, "dst_row");
      ICHECK_GE(route.origin_src_row, 0);
      ICHECK_LT(route.origin_src_row, mesh_nrows_)
          << "T.dist.routed_put source row is outside [0, " << mesh_nrows_
          << "): " << route.origin_src_row;
      for (size_t previous = 0; previous < index; ++previous) {
        ICHECK(!SameRoute(routes[previous], route, current_core))
            << "T.dist.routed_put contains a duplicate static route at entry "
            << index;
      }

      Locality locality =
          ValidateAndClassify(route.dst_rank, route.dst_row, current_core,
                              route.origin_src_row, "T.dist.routed_put");
      PrimExpr entry =
          Call(DataType::Handle(), dist_route(),
               {I32(route.origin_src_row), route.dst_rank, route.dst_row});
      PrimExpr table = Call(DataType::Handle(), dist_route_table(), {entry});
      Stmt remote = Evaluate(Call(
          DataType::Handle(), dist_routed_put(),
          {call->args[0], call->args[1], table, call->args[3], current_core}));
      if (!locality.has_local) {
        statements.push_back(remote);
        continue;
      }

      PrimExpr src =
          RewriteForSource(call->args[0], current_core, route.origin_src_row);
      PrimExpr dst =
          RewriteForSource(call->args[1], current_core, route.origin_src_row);
      PrimExpr dst_rank =
          RewriteForSource(route.dst_rank, current_core, route.origin_src_row);
      PrimExpr dst_row =
          RewriteForSource(route.dst_row, current_core, route.origin_src_row);
      PrimExpr current_col = CurrentCol(current_core);
      PrimExpr src_core = I32(route.origin_src_row * mesh_ncols_) + current_col;
      PrimExpr dst_core = dst_row * I32(mesh_ncols_) + current_col;
      Stmt local =
          MakeLocalTransfer(src, dst, current_core, src_core, dst_core);
      statements.push_back(
          SelectLocalOrRemote(locality, dst_rank == rank_id_, local, remote));
    }
    return SeqStmt::Flatten(statements);
  }

  int local_stage_counter_{0};
  std::vector<Array<Buffer>> alloc_buffer_stack_;
};

class DistRoutePlanner : public DistRouteMutatorBase {
public:
  using DistRouteMutatorBase::DistRouteMutatorBase;

  static PrimFunc Rewrite(PrimFunc func) {
    auto context = GetDistPassContext(func);
    if (!context) {
      return func;
    }
    arith::Analyzer analyzer;
    DistRoutePlanner rewriter(&analyzer, context.value().target,
                              context.value().world_size,
                              context.value().rank_id);
    Stmt body = rewriter.VisitStmt(func->body);
    if (!body.same_as(func->body)) {
      func.CopyOnWrite()->body = std::move(body);
    }
    return func;
  }

private:
  Stmt VisitStmt_(const ForNode *op) final {
    ++loop_depth_;
    Stmt result = arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    --loop_depth_;
    return result;
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    PrimExpr condition = VisitExpr(op->condition);
    PrimExpr old_predicate = route_predicate_;
    route_predicate_ = analyzer_->Simplify(And(old_predicate, condition));
    Stmt then_case = VisitStmt(op->then_case);
    Optional<Stmt> else_case;
    if (op->else_case) {
      route_predicate_ =
          analyzer_->Simplify(And(old_predicate, Not(condition)));
      else_case = VisitStmt(op->else_case.value());
    }
    route_predicate_ = old_predicate;
    return IfThenElse(condition, then_case, else_case, op->span);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (!call || !call->op.same_as(DistPutOp::Get())) {
      return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    return LowerLogicalPut(tvm::ffi::GetRef<Call>(call));
  }

  std::vector<NormalRouteEntry>
  BuildRoutes(const Call &call, const Var &current_core, bool *all_peer) {
    std::vector<NormalRouteEntry> routes;
    *all_peer = true;
    for (int64_t src_rank = 0; src_rank < world_size_; ++src_rank) {
      for (int64_t src_row = 0; src_row < mesh_nrows_; ++src_row) {
        PrimExpr dst_rank =
            RewriteForSource(call->args[2], current_core, src_row, src_rank);
        PrimExpr dst_row =
            RewriteForSource(call->args[3], current_core, src_row, src_rank);
        ValidateResolvedEndpoint(dst_rank, dst_row);
        *all_peer &= analyzer_->CanProve(dst_row == I32(src_row));
      }
    }
    for (int64_t src_row = 0; src_row < mesh_nrows_; ++src_row) {
      routes.push_back(NormalRouteEntry{
          src_row, RewriteForSource(call->args[2], current_core, src_row),
          RewriteForSource(call->args[3], current_core, src_row)});
    }
    return routes;
  }

  Stmt MakePeerPut(const PrimExpr &src, const PrimExpr &dst,
                   const PrimExpr &dst_rank, const PrimExpr &signal,
                   const PrimExpr &current_core) {
    return Evaluate(Call(DataType::Handle(), DistPeerPutOp::Get(),
                         {src, dst, dst_rank, signal, current_core}));
  }

  Stmt LowerLogicalPut(const Call &call) {
    ICHECK_EQ(call->args.size(), 6U);
    const auto *current_core_node = call->args[5].as<VarNode>();
    ICHECK(current_core_node)
        << "T.dist.put current_core must be the T.Kernel block binding";
    Var current_core = tvm::ffi::GetRef<Var>(current_core_node);
    ValidateRouteExpression(call->args[2], "dst_rank");
    ValidateRouteExpression(call->args[3], "dst_row");

    bool all_peer = false;
    std::vector<NormalRouteEntry> routes =
        BuildRoutes(call, current_core, &all_peer);
    const auto *signal_ref = call->args[4].as<CallNode>();
    bool is_dynamic_route =
        signal_ref && (signal_ref->op.same_as(dist_signal_route()) ||
                       signal_ref->op.same_as(dist_signal_group_route()));
    if (!all_peer || !is_dynamic_route) {
      ICHECK(IsRowInvariantPredicate(route_predicate_, current_core))
          << "Cross-row T.dist.put is guarded by a row-dependent condition. "
             "Use T.dist.routed_put to select source rows explicitly; only "
             "Rank- or column-uniform outer conditions are supported";
    }
    ICHECK(all_peer || loop_depth_ == 0)
        << "Cross-row T.dist.put inside loops is not supported until routed "
           "staging lifetime waits are implemented";
    if (all_peer) {
      return MakePeerPut(call->args[0], call->args[1], call->args[2],
                         call->args[4], current_core);
    }

    Array<PrimExpr> entries;
    for (const NormalRouteEntry &route : routes) {
      entries.push_back(
          Call(DataType::Handle(), dist_route(),
               {I32(route.origin_src_row), route.dst_rank, route.dst_row}));
    }
    PrimExpr table = Call(DataType::Handle(), dist_route_table(), entries);
    return Evaluate(Call(
        DataType::Handle(), dist_routed_put(),
        {call->args[0], call->args[1], table, call->args[4], current_core}));
  }

  int loop_depth_{0};
  PrimExpr route_predicate_{const_true()};
};

class LowerPhysicalDistRoutesMutator : public DistRouteMutatorBase {
public:
  using DistRouteMutatorBase::DistRouteMutatorBase;

  static PrimFunc Rewrite(PrimFunc func) {
    auto context = GetDistPassContext(func);
    if (!context) {
      return func;
    }
    arith::Analyzer analyzer;
    LowerPhysicalDistRoutesMutator rewriter(&analyzer, context.value().target,
                                            context.value().world_size,
                                            context.value().rank_id);
    Stmt body = rewriter.VisitStmt(func->body);
    if (!body.same_as(func->body)) {
      func.CopyOnWrite()->body = std::move(body);
    }
    return func;
  }

private:
  Stmt VisitStmt_(const BlockNode *op) final {
    alloc_buffer_stack_.emplace_back();
    Block block = Downcast<Block>(arith::IRMutatorWithAnalyzer::VisitStmt_(op));
    Array<Buffer> new_buffers = alloc_buffer_stack_.back();
    alloc_buffer_stack_.pop_back();
    if (!new_buffers.empty()) {
      Array<Buffer> alloc_buffers = block->alloc_buffers;
      for (const Buffer &buffer : new_buffers) {
        alloc_buffers.push_back(buffer);
      }
      block.CopyOnWrite()->alloc_buffers = std::move(alloc_buffers);
    }
    return block;
  }

  Stmt VisitStmt_(const LetStmtNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call && (call->op.same_as(dist_signal_decl()) ||
                 call->op.same_as(dist_signal_group_decl()))) {
      ICHECK(false) << "T.dist signal declarations must be resolved before "
                       "LowerDistRouting";
    }
    if (call && call->op.same_as(dist_signal_group())) {
      ICHECK_EQ(call->args.size(), 3U);
      const DistSignalKindInfo &group_kind =
          RequireDistSignalKindInfo(call->args[0], "signal-group kind");
      ICHECK_GE(RequireIntImm(call->args[1], "signal-group base index"), 0);
      ICHECK_GT(RequireIntImm(call->args[2], "signal-group count"), 0);
      group_signal_kinds_.emplace(op->var.get(), &group_kind);
      Stmt body = VisitStmt(op->body);
      group_signal_kinds_.erase(op->var.get());
      return LetStmt(op->var, op->value, body, op->span);
    }
    if (!call || !call->op.same_as(dist_signal())) {
      return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    const DistSignalKindInfo &kind =
        RequireDistSignalKindInfo(call->args[0], "resolved signal kind");
    signal_kinds_.emplace(op->var.get(), &kind);
    signal_indices_.emplace(
        op->var.get(), RequireIntImm(call->args[1], "resolved signal index"));
    Stmt body = VisitStmt(op->body);
    signal_kinds_.erase(op->var.get());
    signal_indices_.erase(op->var.get());
    return LetStmt(op->var, op->value, body, op->span);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    ++loop_depth_;
    Stmt result = arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    --loop_depth_;
    return result;
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    PrimExpr condition = VisitExpr(op->condition);
    PrimExpr old_predicate = route_predicate_;
    route_predicate_ = analyzer_->Simplify(And(old_predicate, condition));
    Stmt then_case = VisitStmt(op->then_case);
    Optional<Stmt> else_case;
    if (op->else_case) {
      route_predicate_ =
          analyzer_->Simplify(And(old_predicate, Not(condition)));
      else_case = VisitStmt(op->else_case.value());
    }
    route_predicate_ = old_predicate;
    return IfThenElse(condition, then_case, else_case, op->span);
  }

  BufferRegion CreateStaging(const BufferRegion &source, int64_t src_row) {
    ICHECK(!alloc_buffer_stack_.empty());
    std::string name = source->buffer->name + "_dist_route_stage_" +
                       std::to_string(route_counter_) + "_" +
                       std::to_string(src_row);
    Buffer stage = MakeCompactBufferLike(source->buffer, source->region,
                                         kSunmmioScopeRSRAM, name);
    alloc_buffer_stack_.back().push_back(stage);
    return BufferRegion(stage, MakeCompactRegion(source->region));
  }

  BufferRegion CreatePredicateStaging(int64_t src_row) {
    ICHECK(!alloc_buffer_stack_.empty());
    std::string name = "dist_route_active_" + std::to_string(route_counter_) +
                       "_" + std::to_string(src_row);
    Buffer stage =
        decl_buffer({I32(1)}, DataType::UInt(8), name, kSunmmioScopeRSRAM);
    alloc_buffer_stack_.back().push_back(stage);
    return BufferRegion(stage, {Range::FromMinExtent(I32(0), I32(1))});
  }

  Stmt LowerDynamicRoutedPut(const Call &call, const Var &current_core,
                             const CallNode *signal_ref) {
    bool group_route = signal_ref->op.same_as(dist_signal_group_route());
    ICHECK(group_route || signal_ref->op.same_as(dist_signal_route()));
    ICHECK_EQ(signal_ref->args.size(), group_route ? 4U : 2U);
    std::vector<NormalRouteEntry> routes = ParseNormalRouteTable(call->args[2]);
    Array<Stmt> statements;
    PrimExpr current_row = CurrentRow(current_core);
    PrimExpr current_col = CurrentCol(current_core);
    for (const NormalRouteEntry &route : routes) {
      int64_t src_row = route.origin_src_row;
      PrimExpr route_src =
          RewriteForSource(call->args[0], current_core, src_row);
      PrimExpr route_dst =
          RewriteForSource(call->args[1], current_core, src_row);
      PrimExpr peer_src = route_src;
      PrimExpr active = group_route ? signal_ref->args[3] : signal_ref->args[1];
      PrimExpr peer_active = RewriteForSource(active, current_core, src_row);
      if (!analyzer_->CanProve(route.dst_row == I32(src_row))) {
        ICHECK(analyzer_->CanProve(Not(route.dst_row == I32(src_row))))
            << "A route must be statically peer or cross-row after source-row "
               "substitution";
        BufferRegion source = NormalizeToBufferRegion(route_src);
        BufferRegion staging = CreateStaging(source, src_row);
        PrimExpr stage_write = MakeRegionExpr(staging->buffer, staging->region,
                                              /*access_mask=*/2);
        peer_src = MakeRegionExpr(staging->buffer, staging->region,
                                  /*access_mask=*/1);
        PrimExpr src_core = I32(src_row * mesh_ncols_) + current_col;
        PrimExpr egress_core = route.dst_row * I32(mesh_ncols_) + current_col;
        statements.push_back(Evaluate(
            Call(DataType::Handle(), PutOp::Get(),
                 {route_src, stage_write, I32(-1), src_core, egress_core})));

        BufferRegion active_stage = CreatePredicateStaging(src_row);
        PrimExpr active_write = MakeRegionExpr(
            active_stage->buffer, active_stage->region, /*access_mask=*/2);
        PrimExpr active_read = MakeRegionExpr(
            active_stage->buffer, active_stage->region, /*access_mask=*/1);
        statements.push_back(IfThenElse(
            current_row == I32(src_row),
            BufferStore(active_stage->buffer,
                        Cast(DataType::UInt(8), peer_active), {I32(0)})));
        statements.push_back(Evaluate(
            Call(DataType::Handle(), PutOp::Get(),
                 {active_read, active_write, I32(-1), src_core, egress_core})));
        peer_active = BufferLoad(active_stage->buffer, {I32(0)}) !=
                      IntImm(DataType::UInt(8), 0);
      }

      PrimExpr signal;
      if (group_route) {
        signal =
            Call(DataType::Handle(), dist_signal_group_route(),
                 {signal_ref->args[0],
                  RewriteForSource(signal_ref->args[1], current_core, src_row),
                  RewriteForSource(signal_ref->args[2], current_core, src_row),
                  peer_active});
      } else {
        signal = Call(DataType::Handle(), dist_signal_route(),
                      {signal_ref->args[0], peer_active});
      }
      PrimExpr peer_entry = Call(DataType::Handle(), dist_peer_route(),
                                 {route.dst_row, route.dst_rank});
      PrimExpr peer_table =
          Call(DataType::Handle(), dist_peer_route_table(), {peer_entry});
      statements.push_back(Evaluate(
          Call(DataType::Handle(), DistRoutedPeerPutOp::Get(),
               {peer_table, signal, current_core, peer_src, route_dst})));
    }
    ++route_counter_;
    return SeqStmt::Flatten(statements);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (!call || !call->op.same_as(dist_routed_put())) {
      return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    ICHECK_EQ(call->args.size(), 5U);
    const auto *current_core_node = call->args[4].as<VarNode>();
    ICHECK(current_core_node);
    Var current_core = tvm::ffi::GetRef<Var>(current_core_node);
    ICHECK(loop_depth_ == 0)
        << "T.dist.routed_put inside loops is not supported until routed "
           "staging lifetime waits are implemented";
    ICHECK(IsRowInvariantPredicate(route_predicate_, current_core))
        << "T.dist.routed_put must be guarded by a condition that is uniform "
           "across rows. Move source-row selection into the route table";
    const DistSignalKindInfo *signal_kind = LookupSignalKind(call->args[3]);
    ICHECK(signal_kind->update_mode != DistSignalUpdateMode::kMemory)
        << signal_kind->name
        << " signal does not support cross-row T.dist.put yet";

    if (const auto *signal_ref = call->args[3].as<CallNode>()) {
      if (signal_ref->op.same_as(dist_signal_route()) ||
          signal_ref->op.same_as(dist_signal_group_route())) {
        return LowerDynamicRoutedPut(tvm::ffi::GetRef<Call>(call), current_core,
                                     signal_ref);
      }
      ICHECK(signal_ref->op.same_as(dist_signal_ref()));
    }

    std::vector<NormalRouteEntry> routes = ParseNormalRouteTable(call->args[2]);
    Array<Stmt> local_puts;
    Array<PrimExpr> peer_entries;
    Array<PrimExpr> peer_operands;
    for (const NormalRouteEntry &route : routes) {
      int64_t src_row = route.origin_src_row;
      PrimExpr route_src =
          RewriteForSource(call->args[0], current_core, src_row);
      PrimExpr route_dst =
          RewriteForSource(call->args[1], current_core, src_row);
      PrimExpr peer_src = route_src;
      if (analyzer_->CanProve(route.dst_row == I32(src_row))) {
        // The origin row is already the physical peer row.
      } else {
        ICHECK(analyzer_->CanProve(Not(route.dst_row == I32(src_row))))
            << "A route must be statically peer or cross-row after source-row "
               "substitution";
        BufferRegion source = NormalizeToBufferRegion(route_src);
        BufferRegion staging = CreateStaging(source, src_row);
        PrimExpr stage_write = MakeRegionExpr(staging->buffer, staging->region,
                                              /*access_mask=*/2);
        peer_src = MakeRegionExpr(staging->buffer, staging->region,
                                  /*access_mask=*/1);
        PrimExpr current_col = CurrentCol(current_core);
        PrimExpr src_core = I32(src_row * mesh_ncols_) + current_col;
        PrimExpr egress_core = route.dst_row * I32(mesh_ncols_) + current_col;
        local_puts.push_back(Evaluate(
            Call(DataType::Handle(), PutOp::Get(),
                 {route_src, stage_write, I32(-1), src_core, egress_core})));
      }
      peer_entries.push_back(Call(DataType::Handle(), dist_peer_route(),
                                  {route.dst_row, route.dst_rank}));
      peer_operands.push_back(peer_src);
      peer_operands.push_back(route_dst);
    }
    PrimExpr peer_table =
        Call(DataType::Handle(), dist_peer_route_table(), peer_entries);
    Array<PrimExpr> args{peer_table, call->args[3], current_core};
    for (const PrimExpr &operand : peer_operands) {
      args.push_back(operand);
    }
    local_puts.push_back(
        Evaluate(Call(DataType::Handle(), DistRoutedPeerPutOp::Get(), args)));
    ++route_counter_;
    return SeqStmt::Flatten(local_puts);
  }

  int route_counter_{0};
  int loop_depth_{0};
  PrimExpr route_predicate_{const_true()};
  std::vector<Array<Buffer>> alloc_buffer_stack_;
  std::unordered_map<const VarNode *, const DistSignalKindInfo *> signal_kinds_;
  std::unordered_map<const VarNode *, int64_t> signal_indices_;
  std::unordered_map<const VarNode *, const DistSignalKindInfo *>
      group_signal_kinds_;

  const DistSignalKindInfo *LookupSignalKind(const PrimExpr &signal) {
    if (const auto *var = signal.as<VarNode>()) {
      auto it = signal_kinds_.find(var);
      ICHECK(it != signal_kinds_.end());
      return it->second;
    }
    const auto *ref = signal.as<CallNode>();
    ICHECK(ref);
    if (ref->op.same_as(dist_signal_route())) {
      ICHECK_EQ(ref->args.size(), 2U);
      return LookupSignalKind(ref->args[0]);
    }
    ICHECK(ref->op.same_as(dist_signal_ref()) ||
           ref->op.same_as(dist_signal_group_route()));
    const auto *group_var = ref->args[0].as<VarNode>();
    ICHECK(group_var);
    auto it = group_signal_kinds_.find(group_var);
    ICHECK(it != group_signal_kinds_.end());
    return it->second;
  }
};

PrimFunc Run(PrimFunc func) {
  if (!IsMultiRank(func)) {
    return func;
  }
  DistOpDetector detector(/*high_level=*/true);
  if (!detector.Detect(func->body)) {
    return func;
  }
  func = DistRankRoutedPutNormalizer::Rewrite(std::move(func));
  func = DistLocalRouteLowerer::Rewrite(std::move(func));
  func = DistRoutePlanner::Rewrite(std::move(func));
  return LowerPhysicalDistRoutesMutator::Rewrite(std::move(func));
}

} // namespace

tvm::transform::Pass LowerDistRouting() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) {
    return Run(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerDistRouting", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerDistRouting", LowerDistRouting);
}

} // namespace tl
} // namespace tvm
