/*!
 * \file tl/transform/plan_dist_signals.cc
 * \brief Plan and validate Rank-level communication signal resources.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../op/utils.h"
#include "dist_transform_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;
using namespace dist_transform;

namespace {

// Temporary admission rule until pipeline planning models distributed effects.
// Keep it independent of resource collection: submit/wait-only loops also
// need rejection, even when the function has no signal declarations.
void ValidateNoDistOpsInPipelinedLoops(const PrimFunc &func) {
  class Validator : public StmtExprVisitor {
  private:
    void VisitStmt_(const ForNode *op) final {
      auto stages = op->annotations.Get("num_stages");
      bool pipelined = false;
      if (stages) {
        const auto *count = stages.value().as<IntImmNode>();
        ICHECK(count)
            << "T.Pipelined num_stages must be a compile-time integer";
        pipelined = count->value > 0;
      }
      pipeline_depth_ += pipelined;
      StmtExprVisitor::VisitStmt_(op);
      pipeline_depth_ -= pipelined;
    }

    void VisitExpr_(const CallNode *call) final {
      ICHECK(pipeline_depth_ == 0 ||
             (!IsHighDistOp(call) && !IsLeafDistOp(call)))
          << "T.dist operations are not supported inside "
             "T.Pipelined(num_stages > 0); move communication, submit and "
             "waits outside the pipelined loop. Found "
          << call->op;
      StmtExprVisitor::VisitExpr_(call);
    }

    int pipeline_depth_{0};
  };
  Validator()(func->body);
}

void ValidateSignalArgs(const CallNode *call, size_t kind_arg,
                        size_t index_arg) {
  const DistSignalKindInfo &kind =
      RequireDistSignalKindInfo(call->args[kind_arg], "signal kind");
  int64_t index = RequireIntImm(call->args[index_arg], "signal index");
  ICHECK_GE(index, 0);
  if (kind.capacity >= 0) {
    ICHECK_LT(index, kind.capacity)
        << kind.name << " signal index must be in [0, " << kind.capacity
        << "), got " << index;
  }
}

void ValidateStaticRegion(const BufferRegion &region, const char *op_name,
                          const char *operand_name, bool allow_dram) {
  bool valid_scope = region->buffer.scope() == kSunmmioScopeRSRAM ||
                     (allow_dram && IsDramScope(region->buffer.scope()));
  ICHECK(valid_scope) << op_name << " " << operand_name
                      << " must use shared.rsram"
                      << (allow_dram ? " or global/DRAM" : "") << ", got "
                      << region->buffer.scope();
  for (const Range &range : region->region) {
    ICHECK(range->extent.as<IntImmNode>())
        << op_name << " " << operand_name
        << " region must have static extents, got " << region;
  }
}

void ValidateStaticTransfer(const BufferRegion &src, const BufferRegion &dst,
                            const char *op_name) {
  ValidateStaticRegion(src, op_name, "source", /*allow_dram=*/true);
  ValidateStaticRegion(dst, op_name, "destination", /*allow_dram=*/true);
}

int64_t StaticElementCount(const BufferRegion &region) {
  int64_t count = 1;
  for (const Range &range : region->region) {
    count *= range->extent.as<IntImmNode>()->value;
  }
  return count;
}

const char *DestinationScopeName(DistSignalScope scope) {
  switch (scope) {
  case DistSignalScope::kSram:
    return "shared.rsram";
  case DistSignalScope::kDram:
    return "global/DRAM";
  }
  return "unknown";
}

DistSignalScope ClassifyDestinationScope(const BufferRegion &region,
                                         const char *op_name) {
  const ffi::String &scope = region->buffer.scope();
  if (scope == kSunmmioScopeRSRAM) {
    return DistSignalScope::kSram;
  }
  if (IsDramScope(scope)) {
    return DistSignalScope::kDram;
  }
  ICHECK(false) << op_name
                << " destination must use shared.rsram or global/DRAM, got "
                << scope;
  return DistSignalScope::kSram;
}

struct DistSignalRecord {
  Var var;
  std::string requested_kind;
  std::string requested_expect;
  int64_t logical_id;
  std::optional<DistSignalScope> destination_scope;
  bool has_payload_use{false};
  bool has_signal_only_use{false};
  bool requires_increment{false};
  int64_t generation_cost{1};
  bool has_plain_wait{false};
  bool has_manual_expect{false};
  bool has_protocol_expect{false};
  bool has_auto_completion{false};
  bool used{false};
  std::string resolved_expect;
  const DistSignalKindInfo *resolved_kind{nullptr};
  int64_t resolved_index{-1};
};

struct DistSignalGroupRecord {
  Var var;
  std::string requested_kind;
  std::string requested_expect;
  int64_t logical_id;
  int64_t count;
  std::optional<DistSignalScope> destination_scope;
  bool has_payload_use{false};
  bool has_signal_only_use{false};
  bool requires_increment{false};
  int64_t generation_cost{1};
  bool has_plain_wait{false};
  bool has_manual_expect{false};
  bool has_protocol_expect{false};
  bool has_auto_completion{false};
  bool used{false};
  std::string resolved_expect;
  const DistSignalKindInfo *resolved_kind{nullptr};
  int64_t resolved_base{-1};
};

class DistSignalUseCollector : public StmtExprVisitor {
public:
  void Collect(const Stmt &stmt) { VisitStmt(stmt); }

  std::vector<DistSignalRecord> &Records() { return records_; }
  std::vector<DistSignalGroupRecord> &Groups() { return groups_; }

private:
  void VisitStmt_(const LetStmtNode *let_node) final {
    const auto *call = let_node->value.as<CallNode>();
    if (call && call->op.same_as(dist_signal_group_decl())) {
      ICHECK_EQ(call->args.size(), 4U);
      std::string requested_kind =
          RequireStringImm(call->args[0], "requested signal-group kind");
      std::string requested_expect =
          RequireStringImm(call->args[3], "requested signal-group expect mode");
      int64_t logical_id =
          RequireIntImm(call->args[1], "logical signal-group id");
      int64_t count = RequireIntImm(call->args[2], "signal-group count");
      ICHECK(requested_kind == kAutoSignalKind ||
             FindDistSignalKindInfo(requested_kind))
          << "Unsupported requested T.dist signal-group kind "
          << requested_kind;
      ICHECK_GE(logical_id, 0);
      ICHECK_GT(count, 0);
      ICHECK(requested_expect == "infer" || requested_expect == "auto" ||
             requested_expect == "manual")
          << "Unsupported T.dist signal-group expect mode " << requested_expect;
      ICHECK(!logical_ids_.count(logical_id))
          << "Duplicate logical T.dist signal id " << logical_id;
      size_t group_index = groups_.size();
      groups_.push_back(DistSignalGroupRecord{
          let_node->var, requested_kind, requested_expect, logical_id, count});
      logical_ids_.emplace(logical_id, group_index);
      active_groups_.emplace(let_node->var.get(), group_index);
      VisitStmt(let_node->body);
      active_groups_.erase(let_node->var.get());
      return;
    }
    if (!call || !call->op.same_as(dist_signal_decl())) {
      StmtExprVisitor::VisitStmt_(let_node);
      return;
    }
    ICHECK_EQ(call->args.size(), 3U);
    std::string requested_kind =
        RequireStringImm(call->args[0], "requested signal kind");
    std::string requested_expect =
        RequireStringImm(call->args[2], "requested signal expect mode");
    int64_t logical_id = RequireIntImm(call->args[1], "logical signal id");
    ICHECK(requested_kind == kAutoSignalKind ||
           FindDistSignalKindInfo(requested_kind))
        << "Unsupported requested T.dist signal kind " << requested_kind;
    ICHECK_GE(logical_id, 0);
    ICHECK(requested_expect == "infer" || requested_expect == "auto" ||
           requested_expect == "manual")
        << "Unsupported T.dist signal expect mode " << requested_expect;
    ICHECK(!logical_ids_.count(logical_id))
        << "Duplicate logical T.dist signal id " << logical_id;

    size_t record_index = records_.size();
    records_.push_back(DistSignalRecord{let_node->var, requested_kind,
                                        requested_expect, logical_id});
    logical_ids_.emplace(logical_id, record_index);
    active_.emplace(let_node->var.get(), record_index);
    VisitStmt(let_node->body);
    active_.erase(let_node->var.get());
  }

  void VisitExpr_(const CallNode *call) final {
    if (call->op.same_as(DistPutOp::Get())) {
      ICHECK_EQ(call->args.size(), 6U);
      RegisterUse(call->args[4], NormalizeToBufferRegion(call->args[1]),
                  "T.dist.put");
    } else if (call->op.same_as(dist_wait_signal())) {
      ICHECK_EQ(call->args.size(), 1U);
      RegisterSignalWait(call->args[0], "T.dist.wait_signal");
    } else if (call->op.same_as(dist_wait_signal_delta())) {
      ICHECK_EQ(call->args.size(), 2U);
      RegisterSignalExpectedWait(call->args[0], call->args[1]);
    } else if (call->op.same_as(dist_rank_routed_put())) {
      ICHECK_EQ(call->args.size(), 6U);
      RegisterUse(call->args[4], NormalizeToBufferRegion(call->args[1]),
                  "T.dist.routed_put");
    } else if (call->op.same_as(dist_routed_put())) {
      ICHECK_EQ(call->args.size(), 5U);
      RegisterUse(call->args[3], NormalizeToBufferRegion(call->args[1]),
                  "T.dist.routed_put");
    } else if (call->op.same_as(dist_wait_all())) {
      ICHECK_EQ(call->args.size(), 1U);
      RegisterGroupWait(call->args[0], "T.dist.wait_all");
    } else if (call->op.same_as(dist_wait_signals())) {
      ICHECK_EQ(call->args.size(), 2U);
      RegisterGroupExpectedWait(call->args[0], call->args[1]);
    } else if (call->op.same_as(dist_completion())) {
      ICHECK(call->args.size() == 2U || call->args.size() == 3U);
      std::string mode =
          RequireStringImm(call->args.back(), "completion expect mode");
      if (mode == "manual") {
        ICHECK_EQ(call->args.size(), 3U);
        RegisterGroupExpectedWait(call->args[0], call->args[1]);
      } else if (mode == "rank" || mode == "row") {
        ICHECK_EQ(call->args.size(), 3U);
        RegisterProtocolExpect(call->args[0], "T.dist.all_to_allv");
      } else {
        ICHECK_EQ(call->args.size(), 2U);
        ICHECK(mode == "auto" || mode == "all_gather")
            << "Unsupported T.dist completion mode " << mode;
        RegisterAutoCompletion(call->args[0]);
      }
    } else if (call->op.same_as(dist_signal_put())) {
      ICHECK_EQ(call->args.size(), 3U);
      RegisterSignalOnlyPut(call->args[0], "T.dist.put_signal");
    } else if (call->op.same_as(dist_expect())) {
      ICHECK_EQ(call->args.size(), 2U);
      RegisterProtocolExpect(call->args[0], "T.dist protocol expect");
    }
    StmtExprVisitor::VisitExpr_(call);
  }

  void RegisterUse(const PrimExpr &signal, const BufferRegion &destination,
                   const char *op_name) {
    if (const auto *ref = signal.as<CallNode>()) {
      if (ref->op.same_as(dist_signal_route())) {
        ICHECK_EQ(ref->args.size(), 2U);
        ICHECK(ref->args[1].dtype().is_bool());
        RegisterUse(ref->args[0], destination, op_name);
        return;
      }
      ICHECK(ref->op.same_as(dist_signal_ref()) ||
             ref->op.same_as(dist_signal_group_route()))
          << op_name << " expected a signal or signal-group reference, got "
          << signal;
      if (ref->op.same_as(dist_signal_ref())) {
        ICHECK_EQ(ref->args.size(), 2U);
        ICHECK(ref->args[1].dtype().is_int());
        const auto *group_var = ref->args[0].as<VarNode>();
        ICHECK(group_var);
        auto group = active_groups_.find(group_var);
        ICHECK(group != active_groups_.end());
        if (const auto *index = ref->args[1].as<IntImmNode>()) {
          ICHECK_GE(index->value, 0);
          ICHECK_LT(index->value, groups_[group->second].count);
        }
      } else {
        ICHECK_EQ(ref->args.size(), 4U);
        ICHECK(ref->args[1].dtype().is_int());
        ICHECK(ref->args[2].dtype().is_int());
        ICHECK(ref->args[3].dtype().is_bool());
      }
      RegisterGroupUse(ref->args[0], destination, op_name);
      return;
    }
    const auto *var = signal.as<VarNode>();
    ICHECK(var) << op_name << " expected a signal Var, got " << signal;
    auto it = active_.find(var);
    ICHECK(it != active_.end())
        << op_name << " cannot find the corresponding T.dist.signal for "
        << signal;
    DistSignalRecord &record = records_[it->second];
    DistSignalScope scope = ClassifyDestinationScope(destination, op_name);
    if (record.destination_scope && record.destination_scope.value() != scope) {
      ICHECK(false) << "T.dist.signal " << record.var
                    << " is used with inconsistent destination scopes: "
                    << DestinationScopeName(record.destination_scope.value())
                    << " and " << DestinationScopeName(scope);
    }
    record.destination_scope = scope;
    record.has_payload_use = true;
    record.used = true;
  }

  void RegisterGroupUse(const PrimExpr &group, const BufferRegion &destination,
                        const char *op_name) {
    const auto *group_var = group.as<VarNode>();
    ICHECK(group_var) << op_name << " expected a SignalList handle, got "
                      << group;
    auto active = active_groups_.find(group_var);
    ICHECK(active != active_groups_.end())
        << op_name << " cannot find the corresponding T.dist.signals for "
        << group;
    DistSignalGroupRecord &record = groups_[active->second];
    DistSignalScope scope = ClassifyDestinationScope(destination, op_name);
    if (record.destination_scope && record.destination_scope.value() != scope) {
      ICHECK(false) << "T.dist.signals " << record.var
                    << " is used with inconsistent destination scopes";
    }
    record.destination_scope = scope;
    record.has_payload_use = true;
    record.used = true;
  }

  void RegisterSignalOnlyPut(const PrimExpr &signal, const char *op_name) {
    if (const auto *ref = signal.as<CallNode>()) {
      ICHECK(ref->op.same_as(dist_signal_ref()))
          << op_name << " requires a single signal or SignalList member";
      ICHECK_EQ(ref->args.size(), 2U);
      int64_t index = RequireIntImm(ref->args[1], "signal-group index");
      const auto *group_var = ref->args[0].as<VarNode>();
      ICHECK(group_var);
      auto active = active_groups_.find(group_var);
      ICHECK(active != active_groups_.end());
      DistSignalGroupRecord &record = groups_[active->second];
      ICHECK_GE(index, 0);
      ICHECK_LT(index, record.count);
      record.has_signal_only_use = true;
      record.used = true;
      return;
    }

    const auto *var = signal.as<VarNode>();
    ICHECK(var) << op_name << " expected a signal Var";
    auto active = active_.find(var);
    ICHECK(active != active_.end())
        << op_name << " cannot find the corresponding T.dist.signal";
    DistSignalRecord &record = records_[active->second];
    record.has_signal_only_use = true;
    record.used = true;
  }

  void RegisterSignalWait(const PrimExpr &signal, const char *op_name) {
    if (const auto *ref = signal.as<CallNode>()) {
      ICHECK(ref->op.same_as(dist_signal_ref()))
          << op_name << " requires a single signal or SignalList member";
      ICHECK_EQ(ref->args.size(), 2U);
      ICHECK(ref->args[1].dtype().is_int());
      const auto *group_var = ref->args[0].as<VarNode>();
      ICHECK(group_var);
      auto active = active_groups_.find(group_var);
      ICHECK(active != active_groups_.end());
      DistSignalGroupRecord &record = groups_[active->second];
      if (const auto *index = ref->args[1].as<IntImmNode>()) {
        ICHECK_GE(index->value, 0);
        ICHECK_LT(index->value, record.count);
      }
      record.has_plain_wait = true;
      record.used = true;
      return;
    }
    const auto *var = signal.as<VarNode>();
    ICHECK(var) << op_name << " expected a signal Var";
    auto active = active_.find(var);
    ICHECK(active != active_.end())
        << op_name << " cannot find the corresponding T.dist.signal";
    records_[active->second].has_plain_wait = true;
    records_[active->second].used = true;
  }

  void RegisterGroupWait(const PrimExpr &group, const char *op_name) {
    const auto *group_var = group.as<VarNode>();
    ICHECK(group_var) << op_name << " expected a SignalList handle";
    auto active = active_groups_.find(group_var);
    ICHECK(active != active_groups_.end())
        << op_name << " cannot find the corresponding T.dist.signals";
    groups_[active->second].has_plain_wait = true;
    groups_[active->second].used = true;
  }

  void RegisterSignalExpectedWait(const PrimExpr &signal,
                                  const PrimExpr &expected_delta) {
    ICHECK(expected_delta.dtype().is_int() || expected_delta.dtype().is_uint())
        << "T.dist.wait_signal expected_delta must have integer dtype";
    if (const auto *ref = signal.as<CallNode>()) {
      ICHECK(ref->op.same_as(dist_signal_ref()))
          << "T.dist.wait_signal expected a SignalList member";
      ICHECK_EQ(ref->args.size(), 2U);
      ICHECK(ref->args[1].dtype().is_int());
      const auto *group_var = ref->args[0].as<VarNode>();
      ICHECK(group_var);
      auto active = active_groups_.find(group_var);
      ICHECK(active != active_groups_.end());
      DistSignalGroupRecord &record = groups_[active->second];
      if (const auto *index = ref->args[1].as<IntImmNode>()) {
        ICHECK_GE(index->value, 0);
        ICHECK_LT(index->value, record.count);
      }
      record.has_manual_expect = true;
      record.used = true;
      return;
    }
    const auto *var = signal.as<VarNode>();
    ICHECK(var) << "T.dist.wait_signal expected a signal handle";
    auto active = active_.find(var);
    ICHECK(active != active_.end());
    records_[active->second].has_manual_expect = true;
    records_[active->second].used = true;
  }

  void RegisterGroupExpectedWait(const PrimExpr &group,
                                 const PrimExpr &expected_deltas) {
    const auto *group_var = group.as<VarNode>();
    ICHECK(group_var) << "T.dist.wait_signals expected a SignalList handle";
    auto active = active_groups_.find(group_var);
    ICHECK(active != active_groups_.end())
        << "T.dist.wait_signals cannot find the corresponding T.dist.signals";
    DistSignalGroupRecord &record = groups_[active->second];
    BufferRegion deltas = NormalizeToBufferRegion(expected_deltas);
    ICHECK(
        (deltas->buffer->dtype.is_int() || deltas->buffer->dtype.is_uint()) &&
        deltas->buffer->dtype.bits() == 32)
        << "T.dist.wait_signals expected_deltas must use int32 or uint32";
    ICHECK_EQ(deltas->region.size(), 1U)
        << "T.dist.wait_signals expected_deltas must be one-dimensional";
    ICHECK_EQ(
        RequireIntImm(deltas->region[0]->extent, "expected_deltas extent"),
        record.count)
        << "T.dist.wait_signals expected_deltas extent must equal the "
           "SignalList count";
    record.used = true;
    record.has_manual_expect = true;
  }

  void RegisterAutoCompletion(const PrimExpr &group) {
    const auto *group_var = group.as<VarNode>();
    ICHECK(group_var) << "T.dist.completion expected a SignalList handle";
    auto active = active_groups_.find(group_var);
    ICHECK(active != active_groups_.end())
        << "T.dist.completion cannot find the corresponding T.dist.signals";
    DistSignalGroupRecord &record = groups_[active->second];
    record.has_auto_completion = true;
    record.used = true;
  }

  void RegisterProtocolExpect(const PrimExpr &signal, const char *op_name) {
    if (const auto *ref = signal.as<CallNode>()) {
      ICHECK(ref->op.same_as(dist_signal_ref()))
          << op_name << " expected a signal or static SignalList member";
      const auto *group_var = ref->args[0].as<VarNode>();
      ICHECK(group_var);
      auto active = active_groups_.find(group_var);
      ICHECK(active != active_groups_.end());
      groups_[active->second].has_protocol_expect = true;
      groups_[active->second].used = true;
      return;
    }
    const auto *var = signal.as<VarNode>();
    ICHECK(var) << op_name << " expected a signal handle";
    if (auto active = active_.find(var); active != active_.end()) {
      records_[active->second].has_protocol_expect = true;
      records_[active->second].used = true;
      return;
    }
    auto group = active_groups_.find(var);
    ICHECK(group != active_groups_.end())
        << op_name << " cannot find the corresponding signal declaration";
    groups_[group->second].has_protocol_expect = true;
    groups_[group->second].used = true;
  }

  std::vector<DistSignalRecord> records_;
  std::vector<DistSignalGroupRecord> groups_;
  std::unordered_map<const VarNode *, size_t> active_;
  std::unordered_map<const VarNode *, size_t> active_groups_;
  std::unordered_map<int64_t, size_t> logical_ids_;
};

struct SignalTopologyKey {
  const VarNode *owner;
  int64_t member;

  bool operator==(const SignalTopologyKey &other) const {
    return owner == other.owner && member == other.member;
  }
};

struct SignalTopologyKeyHash {
  size_t operator()(const SignalTopologyKey &key) const {
    return std::hash<const VarNode *>()(key.owner) ^
           (std::hash<int64_t>()(key.member) << 1);
  }
};

struct SignalTopologyInfo {
  std::map<int64_t, std::set<int64_t>> receiver_senders;
  std::map<int64_t, std::set<int64_t>> sender_destinations;
};

class ExprVarUseDetector : public ExprVisitor {
public:
  explicit ExprVarUseDetector(const VarNode *target) : target_(target) {}

  bool Detect(const PrimExpr &expr) {
    VisitExpr(expr);
    return found_;
  }

private:
  void VisitExpr_(const VarNode *op) final { found_ |= op == target_; }

  const VarNode *target_;
  bool found_{false};
};

class DistLoopDependencyDetector : public StmtExprVisitor {
public:
  explicit DistLoopDependencyDetector(const VarNode *loop_var)
      : loop_var_(loop_var) {}

  bool Detect(const Stmt &stmt) {
    VisitStmt(stmt);
    return found_;
  }

private:
  void VisitStmt_(const LetStmtNode *op) final {
    Map<Var, PrimExpr> old_bindings = let_bindings_;
    let_bindings_.Set(op->var, Substitute(op->value, let_bindings_));
    VisitStmt(op->body);
    let_bindings_ = std::move(old_bindings);
  }

  void VisitExpr_(const CallNode *call) final {
    if (call->op.same_as(DistPutOp::Get())) {
      ICHECK_EQ(call->args.size(), 6U);
      found_ |= ExprVarUseDetector(loop_var_).Detect(
          Substitute(call->args[2], let_bindings_));
      found_ |= ExprVarUseDetector(loop_var_).Detect(
          Substitute(call->args[3], let_bindings_));
    }
    if (call->op.same_as(dist_signal_group_route())) {
      ICHECK_EQ(call->args.size(), 4U);
      PrimExpr member = Substitute(call->args[1], let_bindings_);
      found_ |= ExprVarUseDetector(loop_var_).Detect(member);
      if (found_) {
        return;
      }
    }
    if (call->op.same_as(dist_signal_ref())) {
      ICHECK_EQ(call->args.size(), 2U);
      PrimExpr member = Substitute(call->args[1], let_bindings_);
      found_ |= ExprVarUseDetector(loop_var_).Detect(member);
      if (found_) {
        return;
      }
    }
    StmtExprVisitor::VisitExpr_(call);
  }

  const VarNode *loop_var_;
  bool found_{false};
  Map<Var, PrimExpr> let_bindings_;
};

class DistSignalTopologyAnalyzer : public DistRouteMutatorBase {
public:
  DistSignalTopologyAnalyzer(arith::Analyzer *analyzer, Target target,
                             int64_t world_size, Var rank_id,
                             const std::vector<DistSignalRecord> &records,
                             const std::vector<DistSignalGroupRecord> &groups)
      : DistRouteMutatorBase(analyzer, std::move(target), world_size,
                             std::move(rank_id)) {
    for (size_t index = 0; index < records.size(); ++index) {
      single_indices_.emplace(records[index].var.get(), index);
      if (records[index].resolved_expect == "auto") {
        auto_signals_.insert(records[index].var.get());
      }
    }
    for (size_t index = 0; index < groups.size(); ++index) {
      group_indices_.emplace(groups[index].var.get(), index);
      group_counts_.emplace(groups[index].var.get(), groups[index].count);
      if (groups[index].resolved_expect == "auto") {
        auto_signals_.insert(groups[index].var.get());
      }
    }
  }

  void Analyze(const Stmt &stmt, std::vector<DistSignalRecord> *records,
               std::vector<DistSignalGroupRecord> *groups) {
    VisitStmt(stmt);
    int64_t endpoint_count = world_size_ * mesh_nrows_;
    for (DistSignalRecord &record : *records) {
      const SignalTopologyInfo *info = FindTopology({record.var.get(), -1});
      if (!info) {
        continue;
      }
      record.requires_increment = RequiresIncrement(*info);
      record.generation_cost = GenerationCost(*info, endpoint_count);
    }
    for (DistSignalGroupRecord &group : *groups) {
      int64_t total_cost = 0;
      for (int64_t member = 0; member < group.count; ++member) {
        const SignalTopologyInfo *info =
            FindTopology({group.var.get(), member});
        if (!info) {
          total_cost += 1;
          continue;
        }
        group.requires_increment |= RequiresIncrement(*info);
        total_cost += GenerationCost(*info, endpoint_count);
      }
      group.generation_cost = std::max<int64_t>(total_cost, 1);
    }
  }

private:
  // Kernel arguments and buffer contents are local to each endpoint. Without
  // a uniformity contract, only constants and proven-uniform enclosing loop
  // variables may determine an automatic expectation loop's bounds.
  class UniformLoopExpr : public ExprVisitor {
  public:
    explicit UniformLoopExpr(const std::unordered_set<const VarNode *> &vars)
        : vars_(vars) {}

    bool Check(const PrimExpr &expr) {
      VisitExpr(expr);
      return uniform_;
    }

  private:
    void VisitExpr_(const VarNode *op) final {
      uniform_ &= vars_.count(op) != 0;
    }
    void VisitExpr_(const BufferLoadNode *) final { uniform_ = false; }
    void VisitExpr_(const CallNode *) final { uniform_ = false; }

    const std::unordered_set<const VarNode *> &vars_;
    bool uniform_{true};
  };

  bool HasLoopExit(const Stmt &body) const {
    bool found = false;
    PostOrderVisit(body, [&](const ObjectRef &node) {
      if (const auto *call = node.as<CallNode>()) {
        found |= call->op.same_as(tir::builtin::ret()) ||
                 call->op.same_as(tir::builtin::break_loop()) ||
                 call->op.same_as(tir::builtin::continue_loop());
      }
    });
    return found;
  }

  struct ResolvedSignal {
    SignalTopologyKey key;
    PrimExpr active;
    bool allows_sender_dynamic{false};
  };

  const SignalTopologyInfo *FindTopology(const SignalTopologyKey &key) const {
    auto it = topology_.find(key);
    return it == topology_.end() ? nullptr : &it->second;
  }

  bool RequiresIncrement(const SignalTopologyInfo &info) const {
    for (const auto &[_, senders] : info.receiver_senders) {
      if (senders.size() > 1U) {
        return true;
      }
    }
    return false;
  }

  int64_t GenerationCost(const SignalTopologyInfo &info,
                         int64_t endpoint_count) const {
    for (const auto &[_, destinations] : info.sender_destinations) {
      if (destinations.size() > 1U) {
        return endpoint_count;
      }
    }
    return 1;
  }

  ResolvedSignal ResolveSignal(const PrimExpr &signal,
                               const Map<Var, PrimExpr> &substitution) {
    if (const auto *var = signal.as<VarNode>()) {
      ICHECK(single_indices_.count(var))
          << "Cannot find T.dist.signal declaration for " << signal;
      return {{var, -1}, const_true(), false};
    }
    const auto *ref = signal.as<CallNode>();
    ICHECK(ref) << "Expected a T.dist signal reference, got " << signal;
    if (ref->op.same_as(dist_signal_route())) {
      ICHECK_EQ(ref->args.size(), 2U);
      ResolvedSignal resolved = ResolveSignal(ref->args[0], substitution);
      resolved.active = analyzer_->Simplify(
          And(resolved.active, Substitute(ref->args[1], substitution)));
      resolved.allows_sender_dynamic = true;
      return resolved;
    }
    ICHECK(ref->op.same_as(dist_signal_ref()) ||
           ref->op.same_as(dist_signal_group_route()))
        << "Expected a T.dist signal reference, got " << signal;
    const auto *owner = ref->args[0].as<VarNode>();
    ICHECK(owner && group_indices_.count(owner));
    PrimExpr member_expr = ResolveExpr(ref->args[1], substitution);
    int64_t member = RequireIntImm(member_expr, "signal-group member");
    ICHECK_GE(member, 0);
    ICHECK_LT(member, group_counts_.at(owner));
    PrimExpr active = const_true();
    bool allows_sender_dynamic = false;
    if (ref->op.same_as(dist_signal_group_route())) {
      ICHECK_EQ(ref->args.size(), 4U);
      active = ResolveExpr(ref->args[3], substitution);
      allows_sender_dynamic = true;
    } else {
      ICHECK_EQ(ref->args.size(), 2U);
    }
    return {{owner, member}, active, allows_sender_dynamic};
  }

  void RecordSend(const PrimExpr &signal, const PrimExpr &dst_rank_expr,
                  const PrimExpr &dst_row_expr, const Var &current_core,
                  int64_t src_rank, int64_t src_row,
                  const PrimExpr &extra_predicate = const_true()) {
    Map<Var, PrimExpr> substitution =
        MakeSourceSubstitution(current_core, src_row, src_rank);
    PrimExpr dst_rank = ResolveExpr(dst_rank_expr, substitution);
    PrimExpr dst_row = ResolveExpr(dst_row_expr, substitution);
    ICHECK(!RouteExprBufferLoadDetector().Detect(dst_rank) &&
           !RouteExprBufferLoadDetector().Detect(dst_row))
        << "T.dist signal destination cannot depend on BufferLoad";
    ValidateResolvedEndpoint(dst_rank, dst_row, "T.dist signal topology");
    int64_t dst_rank_value = RequireIntImm(dst_rank, "signal dst_rank");
    int64_t dst_row_value = RequireIntImm(dst_row, "signal dst_row");
    if (dst_rank_value == src_rank) {
      return;
    }

    PrimExpr source_predicate = ResolveExpr(predicate_, substitution);
    PrimExpr route_predicate = analyzer_->Simplify(
        And(source_predicate, ResolveExpr(extra_predicate, substitution)));
    if (analyzer_->CanProve(Not(route_predicate))) {
      return;
    }

    ResolvedSignal resolved = ResolveSignal(signal, substitution);
    if (!resolved.allows_sender_dynamic) {
      ICHECK(!RouteExprBufferLoadDetector().Detect(source_predicate))
          << "T.dist signal update condition depends on sender-local "
             "BufferLoad; the receiver cannot derive its expected value";
    }
    PrimExpr predicate =
        analyzer_->Simplify(And(route_predicate, resolved.active));
    if (!resolved.allows_sender_dynamic) {
      ICHECK(!RouteExprBufferLoadDetector().Detect(predicate))
          << "T.dist signal update condition depends on sender-local "
             "BufferLoad; the receiver cannot derive its expected value";
    }
    if (analyzer_->CanProve(Not(predicate))) {
      return;
    }

    ICHECK(unproven_loop_depth_ == 0 ||
           !auto_signals_.count(resolved.key.owner))
        << "T.dist expect='auto' cannot prove matching send/receive loop "
           "iterations; use expect='manual' and provide receiver expected "
           "deltas (or protocol recv_counts). Runtime/local bounds, While, "
           "and loops with early exits are not inferred";

    int64_t sender = src_rank * mesh_nrows_ + src_row;
    int64_t receiver = dst_rank_value * mesh_nrows_ + dst_row_value;
    SignalTopologyInfo &info = topology_[resolved.key];
    info.receiver_senders[receiver].insert(sender);
    info.sender_destinations[sender].insert(receiver);
  }

  void AnalyzePut(const CallNode *call) {
    ICHECK_EQ(call->args.size(), 6U);
    Var current_core = Downcast<Var>(call->args[5]);
    for (int64_t src_rank = 0; src_rank < world_size_; ++src_rank) {
      for (int64_t src_row = 0; src_row < mesh_nrows_; ++src_row) {
        RecordSend(call->args[4], call->args[2], call->args[3], current_core,
                   src_rank, src_row);
      }
    }
  }

  void AnalyzeSignalPut(const CallNode *call) {
    ICHECK_EQ(call->args.size(), 3U);
    Var current_core = Downcast<Var>(call->args[2]);
    for (int64_t src_rank = 0; src_rank < world_size_; ++src_rank) {
      for (int64_t src_row = 0; src_row < mesh_nrows_; ++src_row) {
        RecordSend(call->args[0], call->args[1], I32(src_row), current_core,
                   src_rank, src_row);
      }
    }
  }

  void AnalyzeRoutedPut(const CallNode *call, bool has_src_rank) {
    ICHECK_EQ(call->args.size(), has_src_rank ? 6U : 5U);
    size_t signal_arg = has_src_rank ? 4U : 3U;
    size_t core_arg = has_src_rank ? 5U : 4U;
    Var current_core = Downcast<Var>(call->args[core_arg]);
    std::vector<NormalRouteEntry> routes = ParseNormalRouteTable(call->args[2]);
    for (int64_t src_rank = 0; src_rank < world_size_; ++src_rank) {
      for (const NormalRouteEntry &route : routes) {
        PrimExpr active = const_true();
        if (has_src_rank) {
          active = I32(src_rank) == call->args[3];
        }
        RecordSend(call->args[signal_arg], route.dst_rank, route.dst_row,
                   current_core, src_rank, route.origin_src_row, active);
      }
    }
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    PrimExpr old_predicate = predicate_;
    predicate_ = analyzer_->Simplify(And(old_predicate, op->condition));
    VisitStmt(op->then_case);
    if (op->else_case) {
      predicate_ = analyzer_->Simplify(And(old_predicate, Not(op->condition)));
      VisitStmt(op->else_case.value());
    }
    predicate_ = old_predicate;
    return tvm::ffi::GetRef<Stmt>(op);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    PrimExpr min_expr = analyzer_->Simplify(Substitute(op->min, let_bindings_));
    PrimExpr extent_expr =
        analyzer_->Simplify(Substitute(op->extent, let_bindings_));
    bool uniform = UniformLoopExpr(uniform_loop_vars_).Check(min_expr) &&
                   UniformLoopExpr(uniform_loop_vars_).Check(extent_expr) &&
                   !HasLoopExit(op->body);
    unproven_loop_depth_ += !uniform;
    if (uniform) {
      uniform_loop_vars_.insert(op->loop_var.get());
    }
    if (!DistLoopDependencyDetector(op->loop_var.get()).Detect(op->body)) {
      VisitStmt(op->body);
    } else {
      int64_t min = RequireIntImm(min_expr, "dynamic signal-index loop min");
      int64_t extent =
          RequireIntImm(extent_expr, "dynamic signal-index loop extent");
      ICHECK_GE(extent, 0);
      Map<Var, PrimExpr> old_bindings = let_bindings_;
      for (int64_t offset = 0; offset < extent; ++offset) {
        let_bindings_.Set(op->loop_var,
                          IntImm(op->loop_var.dtype(), min + offset));
        VisitStmt(op->body);
        let_bindings_ = old_bindings;
      }
    }
    uniform_loop_vars_.erase(op->loop_var.get());
    unproven_loop_depth_ -= !uniform;
    return tvm::ffi::GetRef<Stmt>(op);
  }

  Stmt VisitStmt_(const WhileNode *op) final {
    ++unproven_loop_depth_;
    VisitStmt(op->body);
    --unproven_loop_depth_;
    return tvm::ffi::GetRef<Stmt>(op);
  }

  Stmt VisitStmt_(const LetStmtNode *op) final {
    Map<Var, PrimExpr> old_bindings = let_bindings_;
    let_bindings_.Set(
        op->var, analyzer_->Simplify(Substitute(op->value, let_bindings_)));
    VisitStmt(op->body);
    let_bindings_ = std::move(old_bindings);
    return tvm::ffi::GetRef<Stmt>(op);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (!call) {
      return tvm::ffi::GetRef<Stmt>(op);
    }
    if (call->op.same_as(DistPutOp::Get())) {
      AnalyzePut(call);
    } else if (call->op.same_as(dist_signal_put())) {
      AnalyzeSignalPut(call);
    } else if (call->op.same_as(dist_rank_routed_put())) {
      AnalyzeRoutedPut(call, true);
    } else if (call->op.same_as(dist_routed_put())) {
      AnalyzeRoutedPut(call, false);
    }
    return tvm::ffi::GetRef<Stmt>(op);
  }

  std::unordered_map<const VarNode *, size_t> single_indices_;
  std::unordered_map<const VarNode *, size_t> group_indices_;
  std::unordered_map<const VarNode *, int64_t> group_counts_;
  std::unordered_set<const VarNode *> auto_signals_;
  std::unordered_set<const VarNode *> uniform_loop_vars_;
  int unproven_loop_depth_{0};
  std::unordered_map<SignalTopologyKey, SignalTopologyInfo,
                     SignalTopologyKeyHash>
      topology_;
  PrimExpr predicate_{const_true()};
  Map<Var, PrimExpr> let_bindings_;

  PrimExpr ResolveExpr(const PrimExpr &expr,
                       const Map<Var, PrimExpr> &source_substitution) {
    return analyzer_->Simplify(
        Substitute(Substitute(expr, let_bindings_), source_substitution));
  }
};

class DistSignalPlanRewriter : public StmtExprMutator {
public:
  explicit DistSignalPlanRewriter(
      const std::unordered_map<const VarNode *, DistSignalRecord> &plans,
      const std::unordered_map<const VarNode *, DistSignalGroupRecord>
          &group_plans)
      : plans_(plans), group_plans_(group_plans) {}

private:
  Stmt VisitStmt_(const LetStmtNode *let_node) final {
    const auto *call = let_node->value.as<CallNode>();
    if (call && call->op.same_as(dist_signal_group_decl())) {
      auto it = group_plans_.find(let_node->var.get());
      ICHECK(it != group_plans_.end());
      Stmt body = VisitStmt(let_node->body);
      const DistSignalGroupRecord &plan = it->second;
      if (!plan.used) {
        return body;
      }
      Array<PrimExpr> args{
          StringImm(plan.resolved_kind->name),
          IntImm(DataType::Int(32), plan.resolved_base),
          IntImm(DataType::Int(32), plan.count),
          StringImm(plan.resolved_expect),
      };
      PrimExpr value = Call(call->dtype, dist_signal_group(), args,
                            call->annotations, call->span);
      return LetStmt(let_node->var, value, body, let_node->span);
    }
    if (!call || !call->op.same_as(dist_signal_decl())) {
      return StmtExprMutator::VisitStmt_(let_node);
    }
    auto it = plans_.find(let_node->var.get());
    ICHECK(it != plans_.end());
    Stmt body = VisitStmt(let_node->body);
    const DistSignalRecord &plan = it->second;
    if (!plan.used) {
      return body;
    }
    Array<PrimExpr> args{
        StringImm(plan.resolved_kind->name),
        IntImm(DataType::Int(32), plan.resolved_index),
        StringImm(plan.resolved_expect),
    };
    PrimExpr value =
        Call(call->dtype, dist_signal(), args, call->annotations, call->span);
    return LetStmt(let_node->var, value, body, let_node->span);
  }

  const std::unordered_map<const VarNode *, DistSignalRecord> &plans_;
  const std::unordered_map<const VarNode *, DistSignalGroupRecord>
      &group_plans_;
};

PrimFunc PlanSignalResources(PrimFunc func) {
  if (func->GetAttr<Map<ffi::String, Integer>>(kDistSignalCountsAttr)) {
    return func;
  }
  if (!IsMultiRank(func)) {
    return func;
  }

  DistSignalUseCollector collector;
  collector.Collect(func->body);
  std::vector<DistSignalRecord> &records = collector.Records();
  std::vector<DistSignalGroupRecord> &groups = collector.Groups();
  if (records.empty() && groups.empty()) {
    return func;
  }

  auto legalize_expect = [](auto *record, const char *resource_name) {
    ICHECK(!(record->has_manual_expect && record->has_protocol_expect))
        << resource_name
        << " mixes user-provided expected_deltas with protocol-managed "
           "expectation";
    bool has_external_expect =
        record->has_manual_expect || record->has_protocol_expect;
    ICHECK(
        !(record->has_auto_completion && record->requested_expect == "manual"))
        << resource_name
        << " uses an auto-managed completion with expect='manual'";
    if (record->requested_expect == "infer") {
      record->resolved_expect = has_external_expect ? "manual" : "auto";
    } else {
      record->resolved_expect = record->requested_expect;
    }
    ICHECK(!(record->resolved_expect == "auto" && has_external_expect))
        << resource_name
        << " explicitly requests expect='auto' but its expectation is "
           "provided by a wait or communication protocol";
    ICHECK(!(record->resolved_expect == "manual" && record->has_plain_wait &&
             !record->has_protocol_expect))
        << resource_name
        << " uses expect='manual' with wait_signal/wait_all but provides no "
           "expected delta";
  };
  for (DistSignalRecord &record : records) {
    legalize_expect(&record, "T.dist.signal");
  }
  for (DistSignalGroupRecord &group : groups) {
    legalize_expect(&group, "T.dist.signals");
  }

  auto context = GetDistPassContext(func);
  ICHECK(context);
  arith::Analyzer topology_arith;
  DistSignalTopologyAnalyzer topology(&topology_arith, context.value().target,
                                      context.value().world_size,
                                      context.value().rank_id, records, groups);
  topology.Analyze(func->body, &records, &groups);

  std::array<int64_t, static_cast<size_t>(DistSignalKind::kCount)> counts{};
  auto allocate = [&](DistSignalRecord &record, const DistSignalKindInfo &kind,
                      bool inferred) {
    ICHECK(record.used);
    ICHECK(record.destination_scope);
    ICHECK(kind.scope == record.destination_scope.value())
        << "T.dist.signal " << record.var << " explicitly requests "
        << kind.name << " but its destination scope is "
        << DestinationScopeName(record.destination_scope.value());
    size_t kind_index = DistSignalKindIndex(kind.kind);
    if (kind.capacity >= 0 && counts[kind_index] >= kind.capacity) {
      ICHECK(false) << kind.name << " signal capacity exceeded: maximum is "
                    << kind.capacity
                    << (inferred ? "; this multi-sender signal requires INC "
                                   "and cannot spill to VALUE or MEMORY"
                                 : "");
    }
    record.resolved_kind = &kind;
    record.resolved_index = counts[kind_index]++;
  };

  auto allocate_group = [&](DistSignalGroupRecord &group,
                            const DistSignalKindInfo &kind, bool inferred) {
    ICHECK(group.used);
    ICHECK(group.destination_scope);
    ICHECK(kind.scope == group.destination_scope.value())
        << "T.dist.signals " << group.var << " explicitly requests "
        << kind.name << " but its destination scope is "
        << DestinationScopeName(group.destination_scope.value());
    size_t kind_index = DistSignalKindIndex(kind.kind);
    int64_t group_size = group.count;
    if (kind.capacity >= 0 && counts[kind_index] + group_size > kind.capacity) {
      ICHECK(false) << kind.name
                    << " contiguous signal-group capacity exceeded: needs "
                    << group_size << " entries at base " << counts[kind_index]
                    << ", maximum capacity is " << kind.capacity
                    << (inferred ? "; this multi-sender group requires INC "
                                   "and cannot spill to VALUE or MEMORY"
                                 : "");
    }
    group.resolved_kind = &kind;
    group.resolved_base = counts[kind_index];
    counts[kind_index] += group_size;
  };

  struct Candidate {
    bool is_group;
    size_t index;
    int64_t logical_id;
    int64_t weight;
    int64_t generation_cost;
    bool mandatory_increment;
  };

  std::vector<Candidate> candidates;
  for (size_t index = 0; index < records.size(); ++index) {
    if (records[index].used) {
      candidates.push_back(Candidate{false, index, records[index].logical_id, 1,
                                     records[index].generation_cost,
                                     records[index].requires_increment});
    }
  }
  for (size_t index = 0; index < groups.size(); ++index) {
    if (groups[index].used) {
      candidates.push_back(Candidate{
          true, index, groups[index].logical_id, groups[index].count,
          groups[index].generation_cost, groups[index].requires_increment});
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &lhs, const Candidate &rhs) {
              return lhs.logical_id < rhs.logical_id;
            });

  auto requested_kind = [&](const Candidate &candidate) -> const std::string & {
    return candidate.is_group ? groups[candidate.index].requested_kind
                              : records[candidate.index].requested_kind;
  };
  auto candidate_scope =
      [&](const Candidate &candidate) -> std::optional<DistSignalScope> {
    return candidate.is_group ? groups[candidate.index].destination_scope
                              : records[candidate.index].destination_scope;
  };
  auto set_candidate_scope = [&](const Candidate &candidate,
                                 DistSignalScope scope) {
    if (candidate.is_group) {
      groups[candidate.index].destination_scope = scope;
    } else {
      records[candidate.index].destination_scope = scope;
    }
  };
  auto kind_for =
      [&](DistSignalScope scope,
          DistSignalUpdateMode update_mode) -> const DistSignalKindInfo & {
    DistSignalKind kind;
    if (update_mode == DistSignalUpdateMode::kIncrement) {
      kind = scope == DistSignalScope::kSram ? DistSignalKind::kSramFlagregInc
                                             : DistSignalKind::kDramFlagregInc;
    } else if (update_mode == DistSignalUpdateMode::kValue) {
      kind = scope == DistSignalScope::kSram
                 ? DistSignalKind::kSramFlagregValue
                 : DistSignalKind::kDramFlagregValue;
    } else {
      kind = scope == DistSignalScope::kSram ? DistSignalKind::kSramMemory
                                             : DistSignalKind::kDramMemory;
    }
    return DistSignalKindInfos()[DistSignalKindIndex(kind)];
  };
  auto fits = [&](const Candidate &candidate, const DistSignalKindInfo &kind) {
    return kind.capacity < 0 ||
           counts[DistSignalKindIndex(kind.kind)] + candidate.weight <=
               kind.capacity;
  };
  auto assign_candidate = [&](const Candidate &candidate,
                              const DistSignalKindInfo &kind, bool inferred) {
    if (candidate.mandatory_increment) {
      ICHECK(kind.update_mode == DistSignalUpdateMode::kIncrement)
          << "T.dist signal " << candidate.logical_id
          << " has multiple physical senders and requires an INC flagreg, got "
          << kind.name;
    }
    if (!candidate_scope(candidate)) {
      set_candidate_scope(candidate, kind.scope);
    }
    if (candidate.is_group) {
      allocate_group(groups[candidate.index], kind, inferred);
    } else {
      allocate(records[candidate.index], kind, inferred);
    }
  };
  auto choose_flexible_scope =
      [&](const Candidate &candidate,
          DistSignalUpdateMode update_mode) -> std::optional<DistSignalScope> {
    const DistSignalKindInfo &sram =
        kind_for(DistSignalScope::kSram, update_mode);
    const DistSignalKindInfo &dram =
        kind_for(DistSignalScope::kDram, update_mode);
    bool sram_fits = fits(candidate, sram);
    bool dram_fits = fits(candidate, dram);
    if (!sram_fits && !dram_fits) {
      return std::nullopt;
    }
    if (sram_fits && !dram_fits) {
      return DistSignalScope::kSram;
    }
    if (!sram_fits && dram_fits) {
      return DistSignalScope::kDram;
    }
    int64_t sram_remaining =
        sram.capacity < 0
            ? std::numeric_limits<int64_t>::max()
            : sram.capacity - counts[DistSignalKindIndex(sram.kind)];
    int64_t dram_remaining =
        dram.capacity < 0
            ? std::numeric_limits<int64_t>::max()
            : dram.capacity - counts[DistSignalKindIndex(dram.kind)];
    return dram_remaining > sram_remaining ? DistSignalScope::kDram
                                           : DistSignalScope::kSram;
  };

  for (const Candidate &candidate : candidates) {
    if (requested_kind(candidate) == kAutoSignalKind) {
      continue;
    }
    const DistSignalKindInfo *kind =
        FindDistSignalKindInfo(requested_kind(candidate));
    ICHECK(kind);
    assign_candidate(candidate, *kind, /*inferred=*/false);
  }

  for (const Candidate &candidate : candidates) {
    if (requested_kind(candidate) != kAutoSignalKind ||
        !candidate.mandatory_increment || !candidate_scope(candidate)) {
      continue;
    }
    const DistSignalKindInfo &kind = kind_for(
        candidate_scope(candidate).value(), DistSignalUpdateMode::kIncrement);
    assign_candidate(candidate, kind, /*inferred=*/true);
  }
  for (const Candidate &candidate : candidates) {
    if (requested_kind(candidate) != kAutoSignalKind ||
        !candidate.mandatory_increment || candidate_scope(candidate)) {
      continue;
    }
    std::optional<DistSignalScope> scope =
        choose_flexible_scope(candidate, DistSignalUpdateMode::kIncrement);
    ICHECK(scope) << "No INC flagreg capacity remains for mandatory multi-"
                     "sender T.dist signal "
                  << candidate.logical_id;
    assign_candidate(candidate,
                     kind_for(scope.value(), DistSignalUpdateMode::kIncrement),
                     /*inferred=*/true);
  }

  std::vector<Candidate> flexible;
  for (const Candidate &candidate : candidates) {
    if (requested_kind(candidate) == kAutoSignalKind &&
        !candidate.mandatory_increment) {
      flexible.push_back(candidate);
    }
  }
  std::sort(flexible.begin(), flexible.end(),
            [&](const Candidate &lhs, const Candidate &rhs) {
              bool lhs_fixed = candidate_scope(lhs).has_value();
              bool rhs_fixed = candidate_scope(rhs).has_value();
              if (lhs_fixed != rhs_fixed) {
                return lhs_fixed > rhs_fixed;
              }
              int64_t lhs_score = lhs.generation_cost * rhs.weight;
              int64_t rhs_score = rhs.generation_cost * lhs.weight;
              if (lhs_score != rhs_score) {
                return lhs_score > rhs_score;
              }
              return lhs.logical_id < rhs.logical_id;
            });

  std::vector<Candidate> spilled;
  for (const Candidate &candidate : flexible) {
    std::optional<DistSignalScope> scope = candidate_scope(candidate);
    if (scope) {
      const DistSignalKindInfo &kind =
          kind_for(scope.value(), DistSignalUpdateMode::kIncrement);
      if (fits(candidate, kind)) {
        assign_candidate(candidate, kind, /*inferred=*/true);
        continue;
      }
    } else {
      scope =
          choose_flexible_scope(candidate, DistSignalUpdateMode::kIncrement);
      if (scope) {
        assign_candidate(
            candidate,
            kind_for(scope.value(), DistSignalUpdateMode::kIncrement),
            /*inferred=*/true);
        continue;
      }
    }
    spilled.push_back(candidate);
  }

  for (const Candidate &candidate : spilled) {
    std::optional<DistSignalScope> scope = candidate_scope(candidate);
    if (!scope) {
      scope = choose_flexible_scope(candidate, DistSignalUpdateMode::kValue);
    }
    if (scope) {
      const DistSignalKindInfo &value =
          kind_for(scope.value(), DistSignalUpdateMode::kValue);
      if (fits(candidate, value)) {
        assign_candidate(candidate, value, /*inferred=*/true);
        continue;
      }
    }
    DistSignalScope memory_scope =
        candidate_scope(candidate).value_or(DistSignalScope::kSram);
    assign_candidate(candidate,
                     kind_for(memory_scope, DistSignalUpdateMode::kMemory),
                     /*inferred=*/true);
  }

  std::unordered_map<const VarNode *, DistSignalRecord> plans;
  for (const DistSignalRecord &record : records) {
    plans.emplace(record.var.get(), record);
  }
  std::unordered_map<const VarNode *, DistSignalGroupRecord> group_plans;
  for (const DistSignalGroupRecord &group : groups) {
    group_plans.emplace(group.var.get(), group);
  }
  Stmt body = DistSignalPlanRewriter(plans, group_plans)(func->body);
  if (!body.same_as(func->body)) {
    func.CopyOnWrite()->body = std::move(body);
  }
  Map<ffi::String, Integer> signal_counts;
  for (const DistSignalKindInfo &info : DistSignalKindInfos()) {
    signal_counts.Set(info.name,
                      Integer(counts[DistSignalKindIndex(info.kind)]));
  }
  func = WithAttr(std::move(func), kDistSignalCountsAttr, signal_counts);
  return func;
}

class DistCommunicationValidator : public StmtExprVisitor {
public:
  void Validate(const PrimFunc &func) {
    auto world_size = func->GetAttr<Integer>(kDistWorldSizeAttr);
    int64_t world_size_value = world_size ? world_size.value()->value : 1;
    ICHECK_GT(world_size_value, 1)
        << "T.dist communication op appears in a single-Rank kernel: "
           "world_size=1 disables Rank communication";
    if (auto signal_counts =
            func->GetAttr<Map<ffi::String, Integer>>(kDistSignalCountsAttr)) {
      ICHECK_EQ(signal_counts.value().size(), DistSignalKindInfos().size());
      for (const DistSignalKindInfo &info : DistSignalKindInfos()) {
        Optional<Integer> count = signal_counts.value().Get(info.name);
        ICHECK(count) << "Missing T.dist signal count for " << info.name;
        ICHECK_GE(count.value()->value, 0);
        if (info.capacity >= 0) {
          ICHECK_LE(count.value()->value, info.capacity);
        }
      }
    }
    VisitStmt(func->body);
  }

private:
  void VisitExpr_(const CallNode *call) final {
    if (call->op.same_as(dist_signal_decl())) {
      ICHECK(false) << "tl.dist_signal_decl must be resolved by "
                       "PlanDistSignals before validation";
    } else if (call->op.same_as(dist_signal_group_decl())) {
      ICHECK(false) << "tl.dist_signal_group_decl must be resolved by "
                       "PlanDistSignals before validation";
    } else if (call->op.same_as(dist_signal())) {
      ICHECK_EQ(call->args.size(), 3U);
      ValidateSignalArgs(call, 0, 1);
      std::string expect =
          RequireStringImm(call->args[2], "resolved signal expect mode");
      ICHECK(expect == "auto" || expect == "manual");
      ICHECK(call->dtype.is_handle());
    } else if (call->op.same_as(dist_signal_group())) {
      ICHECK_EQ(call->args.size(), 4U);
      const DistSignalKindInfo &kind =
          RequireDistSignalKindInfo(call->args[0], "signal-group kind");
      int64_t base = RequireIntImm(call->args[1], "signal-group base index");
      int64_t count = RequireIntImm(call->args[2], "signal-group count");
      ICHECK_GE(base, 0);
      ICHECK_GT(count, 0);
      if (kind.capacity >= 0) {
        ICHECK_LE(base + count, kind.capacity);
      }
      std::string expect =
          RequireStringImm(call->args[3], "resolved signal-group expect mode");
      ICHECK(expect == "auto" || expect == "manual");
    } else if (call->op.same_as(dist_signal_ref())) {
      ICHECK_EQ(call->args.size(), 2U);
      ICHECK(call->args[0].dtype().is_handle());
      ICHECK(call->args[1].dtype().is_int());
    } else if (call->op.same_as(dist_signal_route())) {
      ICHECK_EQ(call->args.size(), 2U);
      ICHECK(call->args[0].dtype().is_handle());
      ICHECK(call->args[1].dtype().is_bool());
    } else if (call->op.same_as(dist_signal_group_route())) {
      ICHECK_EQ(call->args.size(), 4U);
      ICHECK(call->args[0].dtype().is_handle());
      ICHECK(call->args[1].dtype().is_int());
      ICHECK(call->args[2].dtype().is_int());
      ICHECK(call->args[3].dtype().is_bool());
    } else if (call->op.same_as(dist_signal_put())) {
      ICHECK_EQ(call->args.size(), 3U);
      ICHECK(call->args[0].dtype().is_handle());
      ICHECK(call->args[1].dtype().is_int());
      ICHECK(call->args[2].dtype().is_int());
    } else if (call->op.same_as(DistPutOp::Get())) {
      ValidatePut(call);
    } else if (call->op.same_as(dist_wait_signal())) {
      ValidateWaitSignal(call);
    } else if (call->op.same_as(dist_wait_signal_delta())) {
      ICHECK_EQ(call->args.size(), 2U);
      ICHECK(call->args[0].dtype().is_handle());
      ICHECK(call->args[1].dtype().is_int() || call->args[1].dtype().is_uint());
    } else if (call->op.same_as(dist_rank_routed_put())) {
      ValidateRankRoutedPut(call);
    } else if (call->op.same_as(dist_routed_put())) {
      ValidateRoutedPut(call);
    } else if (call->op.same_as(dist_wait_all())) {
      ValidateWaitAll(call);
    } else if (call->op.same_as(dist_wait_signals())) {
      ICHECK_EQ(call->args.size(), 2U);
      ICHECK(call->args[0].dtype().is_handle());
      ValidateStaticRegion(NormalizeToBufferRegion(call->args[1]),
                           "T.dist.wait_signals", "expected_deltas",
                           /*allow_dram=*/true);
    } else if (call->op.same_as(dist_wait_send())) {
      ICHECK_EQ(call->args.size(), 0);
    } else if (call->op.same_as(dist_completion())) {
      ICHECK(call->args.size() == 2U || call->args.size() == 3U);
      ICHECK(call->args[0].dtype().is_handle());
      std::string mode =
          RequireStringImm(call->args.back(), "completion expect mode");
      ICHECK(mode == "auto" || mode == "manual" || mode == "all_gather" ||
             mode == "rank" || mode == "row")
          << "Unsupported T.dist completion mode " << mode;
      if (call->args.size() == 3U) {
        ValidateStaticRegion(
            NormalizeToBufferRegion(call->args[1]), "T.dist.completion",
            mode == "manual" ? "expected_deltas" : "recv_counts",
            /*allow_dram=*/true);
      }
    } else if (call->op.same_as(dist_completion_has_pending()) ||
               call->op.same_as(dist_wait_any()) ||
               call->op.same_as(dist_wait_completion_all())) {
      ICHECK_EQ(call->args.size(), 1U);
      ICHECK(call->args[0].dtype().is_handle());
    }
    StmtExprVisitor::VisitExpr_(call);
  }

  void ValidatePut(const CallNode *call) {
    ICHECK_EQ(call->args.size(), 6);
    BufferRegion src = NormalizeToBufferRegion(call->args[0]);
    BufferRegion dst = NormalizeToBufferRegion(call->args[1]);
    ValidateStaticTransfer(src, dst, "T.dist.put");
    ICHECK(src->buffer->dtype == dst->buffer->dtype)
        << "T.dist.put source and destination dtypes must match";
    ICHECK_EQ(StaticElementCount(src), StaticElementCount(dst))
        << "T.dist.put source and destination regions must contain the same "
           "number of elements";
    ICHECK(call->args[2].dtype().is_int())
        << "T.dist.put dst_rank must have integer dtype";
    ICHECK(call->args[3].dtype().is_int())
        << "T.dist.put dst_row must have integer dtype";
    ICHECK(call->args[4].dtype().is_handle())
        << "T.dist.put signal must be a signal handle";
    ICHECK(call->args[5].dtype().is_int())
        << "T.dist.put current_core must have integer dtype";
  }

  void ValidateWaitAll(const CallNode *call) {
    ICHECK_EQ(call->args.size(), 1U)
        << "T.dist.wait_all requires one SignalList";
    ICHECK(call->args[0].dtype().is_handle())
        << "T.dist.wait_all argument must be a SignalList handle";
  }

  void ValidateWaitSignal(const CallNode *call) {
    ICHECK_EQ(call->args.size(), 1);
    ICHECK(call->args[0].dtype().is_handle())
        << "T.dist.wait_signal signal must be a signal handle";
  }

  void ValidateRoutedPut(const CallNode *call) {
    ICHECK_EQ(call->args.size(), 5U);
    BufferRegion src = NormalizeToBufferRegion(call->args[0]);
    BufferRegion dst = NormalizeToBufferRegion(call->args[1]);
    ValidateStaticTransfer(src, dst, "T.dist.routed_put");
    ICHECK(src->buffer->dtype == dst->buffer->dtype);
    ICHECK_EQ(StaticElementCount(src), StaticElementCount(dst));
    ICHECK(call->args[3].dtype().is_handle());
    ICHECK(call->args[4].dtype().is_int());
    std::vector<NormalRouteEntry> routes = ParseNormalRouteTable(call->args[2]);
    ICHECK(!routes.empty()) << "T.dist.routed_put route table cannot be empty";
  }

  void ValidateRankRoutedPut(const CallNode *call) {
    ICHECK_EQ(call->args.size(), 6U);
    BufferRegion src = NormalizeToBufferRegion(call->args[0]);
    BufferRegion dst = NormalizeToBufferRegion(call->args[1]);
    ValidateStaticTransfer(src, dst, "T.dist.routed_put");
    ICHECK(src->buffer->dtype == dst->buffer->dtype);
    ICHECK_EQ(StaticElementCount(src), StaticElementCount(dst));
    ICHECK(call->args[3].dtype().is_int())
        << "T.dist.routed_put src_rank must have integer dtype";
    ICHECK(call->args[4].dtype().is_handle());
    ICHECK(call->args[5].dtype().is_int());
    std::vector<NormalRouteEntry> routes = ParseNormalRouteTable(call->args[2]);
    ICHECK(!routes.empty()) << "T.dist.routed_put route table cannot be empty";
  }
};

// Only completions used for per-member progress own their SignalList. Other
// collectives may accumulate on the same group before a final group wait.
class DistCompletionReuseValidator : public StmtVisitor {
public:
  void Validate(const PrimFunc &func) {
    progress_uses_ = ProgressUseCollector().Collect(func->body);
    VisitStmt(func->body);
  }

private:
  // A null completion means control flow joined two different owners: neither
  // can be proven to release the group on every path.
  using ActiveGroups = std::unordered_map<const VarNode *, const VarNode *>;

  class ProgressUseCollector : public StmtExprVisitor {
  public:
    std::unordered_set<const VarNode *> Collect(const Stmt &body) {
      VisitStmt(body);
      return std::move(progress_uses_);
    }

  private:
    void VisitExpr_(const CallNode *call) final {
      if (call->op.same_as(dist_wait_any()) ||
          call->op.same_as(dist_completion_has_pending())) {
        ICHECK_EQ(call->args.size(), 1U);
        const auto *completion = call->args[0].as<VarNode>();
        ICHECK(completion) << "T.dist wait_any/has_pending requires a "
                              "completion handle";
        progress_uses_.insert(completion);
      }
      StmtExprVisitor::VisitExpr_(call);
    }

    std::unordered_set<const VarNode *> progress_uses_;
  };

  class WaitAnyLoopBody : public StmtVisitor {
  public:
    explicit WaitAnyLoopBody(const VarNode *completion)
        : completion_(completion) {}

    bool Drains(const Stmt &body) {
      VisitStmt(body);
      return !has_control_flow_ && wait_count_ == 1;
    }

  private:
    void VisitStmt_(const LetStmtNode *op) final {
      if (const auto *call = op->value.as<CallNode>()) {
        if (call->op.same_as(dist_wait_any())) {
          ++wait_count_;
          if (call->args.size() != 1U ||
              call->args[0].as<VarNode>() != completion_) {
            has_control_flow_ = true;
          }
        }
      }
      StmtVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const EvaluateNode *op) final {
      if (const auto *call = op->value.as<CallNode>()) {
        has_control_flow_ |= call->op.same_as(dist_wait_any()) ||
                             call->op.same_as(dist_wait_completion_all()) ||
                             call->op.same_as(tir::builtin::break_loop()) ||
                             call->op.same_as(tir::builtin::continue_loop()) ||
                             call->op.same_as(tir::builtin::ret());
      }
    }

    void VisitStmt_(const IfThenElseNode *) final { has_control_flow_ = true; }
    void VisitStmt_(const ForNode *) final { has_control_flow_ = true; }
    void VisitStmt_(const WhileNode *) final { has_control_flow_ = true; }

    const VarNode *completion_;
    int wait_count_{0};
    bool has_control_flow_{false};
  };

  void VisitStmt_(const LetStmtNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (!call || !call->op.same_as(dist_completion())) {
      StmtVisitor::VisitStmt_(op);
      return;
    }
    ICHECK(!call->args.empty());
    const auto *group = call->args[0].as<VarNode>();
    ICHECK(group) << "T.dist completion requires a SignalList handle";
    ICHECK(!active_groups_.count(group))
        << "T.dist SignalList previous completion must be drained before "
           "creating another completion; use T.dist.wait_all(completion) "
           "or while T.dist.has_pending(completion): "
           "T.dist.wait_any(completion)";
    if (progress_uses_.count(op->var.get())) {
      active_groups_.emplace(group, op->var.get());
    }
    completion_groups_.emplace(op->var.get(), group);
    VisitStmt(op->body);
    completion_groups_.erase(op->var.get());
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.same_as(dist_wait_completion_all())) {
        ICHECK_EQ(call->args.size(), 1U);
        const auto *completion = call->args[0].as<VarNode>();
        auto owner = completion_groups_.find(completion);
        if (owner != completion_groups_.end()) {
          auto active = active_groups_.find(owner->second);
          if (active != active_groups_.end() && active->second == completion) {
            active_groups_.erase(active);
          }
        }
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    ActiveGroups incoming = active_groups_;
    VisitStmt(op->then_case);
    ActiveGroups merged = active_groups_;
    active_groups_ = incoming;
    if (op->else_case) {
      VisitStmt(op->else_case.value());
    }
    for (const auto &[group, completion] : active_groups_) {
      auto [it, inserted] = merged.emplace(group, completion);
      if (!inserted && it->second != completion) {
        it->second = nullptr;
      }
    }
    active_groups_ = std::move(merged);
  }

  void VisitStmt_(const ForNode *op) final {
    ActiveGroups incoming = active_groups_;
    VisitStmt(op->body);
    CheckLoopBody(incoming);
    for (const auto &[group, completion] : incoming) {
      auto [it, inserted] = active_groups_.emplace(group, completion);
      if (!inserted && it->second != completion) {
        it->second = nullptr;
      }
    }
  }

  void VisitStmt_(const WhileNode *op) final {
    const auto *condition = op->condition.as<CallNode>();
    const VarNode *drained_group = nullptr;
    if (condition && condition->op.same_as(dist_completion_has_pending()) &&
        condition->args.size() == 1U) {
      const auto *completion = condition->args[0].as<VarNode>();
      auto owner = completion_groups_.find(completion);
      auto active = owner == completion_groups_.end()
                        ? active_groups_.end()
                        : active_groups_.find(owner->second);
      if (active != active_groups_.end() && active->second == completion &&
          WaitAnyLoopBody(completion).Drains(op->body)) {
        drained_group = owner->second;
      }
    }
    ActiveGroups incoming = active_groups_;
    VisitStmt(op->body);
    CheckLoopBody(incoming);
    for (const auto &[group, completion] : incoming) {
      auto [it, inserted] = active_groups_.emplace(group, completion);
      if (!inserted && it->second != completion) {
        it->second = nullptr;
      }
    }
    if (drained_group) {
      auto active = active_groups_.find(drained_group);
      if (active != active_groups_.end() &&
          active->second == condition->args[0].as<VarNode>()) {
        active_groups_.erase(active);
      }
    }
  }

  void CheckLoopBody(const ActiveGroups &incoming) const {
    for (const auto &[group, completion] : active_groups_) {
      auto before = incoming.find(group);
      ICHECK(before != incoming.end() && before->second == completion)
          << "T.dist SignalList completion created inside a loop must be "
             "drained in that iteration before the next one";
    }
  }

  ActiveGroups active_groups_;
  std::unordered_set<const VarNode *> progress_uses_;
  std::unordered_map<const VarNode *, const VarNode *> completion_groups_;
};

PrimFunc Run(PrimFunc func) {
  ValidateNoDistOpsInPipelinedLoops(func);
  DistOpDetector detector(/*high_level=*/true);
  if (!detector.Detect(func->body)) {
    return func;
  }
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  ICHECK(target && TargetIsSunmmio(target.value()))
      << "T.dist operations only support the SunMMIO target";
  func = PlanSignalResources(std::move(func));
  DistCommunicationValidator().Validate(func);
  DistCompletionReuseValidator().Validate(func);
  return func;
}

} // namespace

tvm::transform::Pass PlanDistSignals() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) {
    return Run(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.PlanDistSignals", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.PlanDistSignals", PlanDistSignals);
}

} // namespace tl
} // namespace tvm
