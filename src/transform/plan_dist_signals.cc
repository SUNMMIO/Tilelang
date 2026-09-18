/*!
 * \file tl/transform/plan_dist_signals.cc
 * \brief Plan and validate Rank-level communication signal resources.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/transform.h>

#include <array>
#include <unordered_map>
#include <vector>

#include "../op/utils.h"
#include "dist_transform_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;
using namespace dist_transform;

namespace {

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
  int64_t logical_id;
  std::optional<DistSignalScope> destination_scope;
  bool used{false};
  const DistSignalKindInfo *resolved_kind{nullptr};
  int64_t resolved_index{-1};
};

struct DistSignalGroupRecord {
  Var var;
  std::string requested_kind;
  int64_t logical_id;
  int64_t count;
  std::optional<DistSignalScope> destination_scope;
  bool used{false};
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
      ICHECK_EQ(call->args.size(), 3U);
      std::string requested_kind =
          RequireStringImm(call->args[0], "requested signal-group kind");
      int64_t logical_id =
          RequireIntImm(call->args[1], "logical signal-group id");
      int64_t count = RequireIntImm(call->args[2], "signal-group count");
      ICHECK(requested_kind == kAutoSignalKind ||
             FindDistSignalKindInfo(requested_kind))
          << "Unsupported requested T.dist signal-group kind "
          << requested_kind;
      ICHECK_GE(logical_id, 0);
      ICHECK_GT(count, 0);
      ICHECK(!logical_ids_.count(logical_id))
          << "Duplicate logical T.dist signal id " << logical_id;
      size_t group_index = groups_.size();
      groups_.push_back(DistSignalGroupRecord{let_node->var, requested_kind,
                                              logical_id, count});
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
    ICHECK_EQ(call->args.size(), 2U);
    std::string requested_kind =
        RequireStringImm(call->args[0], "requested signal kind");
    int64_t logical_id = RequireIntImm(call->args[1], "logical signal id");
    ICHECK(requested_kind == kAutoSignalKind ||
           FindDistSignalKindInfo(requested_kind))
        << "Unsupported requested T.dist signal kind " << requested_kind;
    ICHECK_GE(logical_id, 0);
    ICHECK(!logical_ids_.count(logical_id))
        << "Duplicate logical T.dist signal id " << logical_id;

    size_t record_index = records_.size();
    records_.push_back(
        DistSignalRecord{let_node->var, requested_kind, logical_id});
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
    } else if (call->op.same_as(DistWaitSignalOp::Get())) {
      ICHECK_EQ(call->args.size(), 2U);
      RegisterUse(call->args[0], NormalizeToBufferRegion(call->args[1]),
                  "T.dist.wait_signal");
    } else if (call->op.same_as(dist_rank_routed_put())) {
      ICHECK_EQ(call->args.size(), 6U);
      RegisterUse(call->args[4], NormalizeToBufferRegion(call->args[1]),
                  "T.dist.routed_put");
    } else if (call->op.same_as(dist_routed_put())) {
      ICHECK_EQ(call->args.size(), 5U);
      RegisterUse(call->args[3], NormalizeToBufferRegion(call->args[1]),
                  "T.dist.routed_put");
    } else if (call->op.same_as(dist_wait_all())) {
      ICHECK_EQ(call->args.size(), 2U);
      BufferRegion destination = NormalizeToBufferRegion(call->args[0]);
      RegisterGroupUse(call->args[1], destination, "T.dist.wait_all");
    } else if (call->op.same_as(dist_completion())) {
      ICHECK_EQ(call->args.size(), 4U);
      RegisterGroupUse(call->args[0], NormalizeToBufferRegion(call->args[2]),
                       "T.dist.all_to_allv");
    } else if (call->op.same_as(dist_signal_put())) {
      ICHECK_EQ(call->args.size(), 3U);
      RegisterSyncUse(call->args[0], "T.dist.barrier");
    } else if (call->op.same_as(dist_wait_barrier())) {
      ICHECK_EQ(call->args.size(), 1U);
      RegisterSyncUse(call->args[0], "T.dist.barrier");
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
        int64_t index = RequireIntImm(ref->args[1], "signal-group index");
        const auto *group_var = ref->args[0].as<VarNode>();
        ICHECK(group_var);
        auto group = active_groups_.find(group_var);
        ICHECK(group != active_groups_.end());
        ICHECK_GE(index, 0);
        ICHECK_LT(index, groups_[group->second].count);
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
    record.used = true;
  }

  void RegisterSyncUse(const PrimExpr &signal, const char *op_name) {
    if (const auto *ref = signal.as<CallNode>()) {
      ICHECK(ref->op.same_as(dist_signal_ref()))
          << op_name
          << " requires a single signal or an explicit INC "
             "SignalList member";
      ICHECK_EQ(ref->args.size(), 2U);
      int64_t index = RequireIntImm(ref->args[1], "signal-group index");
      const auto *group_var = ref->args[0].as<VarNode>();
      ICHECK(group_var);
      auto active = active_groups_.find(group_var);
      ICHECK(active != active_groups_.end());
      DistSignalGroupRecord &record = groups_[active->second];
      ICHECK_GE(index, 0);
      ICHECK_LT(index, record.count);
      ICHECK(record.requested_kind != kAutoSignalKind)
          << op_name << " does not accept an automatic SignalList member";
      const DistSignalKindInfo *kind =
          FindDistSignalKindInfo(record.requested_kind);
      ICHECK(kind && kind->update_mode == DistSignalUpdateMode::kIncrement)
          << op_name << " requires an INC flagreg signal";
      if (record.destination_scope && record.destination_scope != kind->scope) {
        ICHECK(false) << op_name << " signal has inconsistent scopes";
      }
      record.destination_scope = kind->scope;
      record.used = true;
      return;
    }

    const auto *var = signal.as<VarNode>();
    ICHECK(var) << op_name << " expected a signal Var";
    auto active = active_.find(var);
    ICHECK(active != active_.end())
        << op_name << " cannot find the corresponding T.dist.signal";
    DistSignalRecord &record = records_[active->second];
    DistSignalScope scope = DistSignalScope::kSram;
    if (record.requested_kind != kAutoSignalKind) {
      const DistSignalKindInfo *kind =
          FindDistSignalKindInfo(record.requested_kind);
      ICHECK(kind && kind->update_mode == DistSignalUpdateMode::kIncrement)
          << op_name << " requires an automatic or INC flagreg signal";
      scope = kind->scope;
    }
    if (record.destination_scope && record.destination_scope != scope) {
      ICHECK(false) << op_name << " signal has inconsistent scopes";
    }
    record.destination_scope = scope;
    record.used = true;
  }

  std::vector<DistSignalRecord> records_;
  std::vector<DistSignalGroupRecord> groups_;
  std::unordered_map<const VarNode *, size_t> active_;
  std::unordered_map<const VarNode *, size_t> active_groups_;
  std::unordered_map<int64_t, size_t> logical_ids_;
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

  DistSignalUseCollector collector;
  collector.Collect(func->body);
  std::vector<DistSignalRecord> &records = collector.Records();
  std::vector<DistSignalGroupRecord> &groups = collector.Groups();
  if (records.empty() && groups.empty()) {
    return func;
  }

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
                    << (inferred ? "; automatic signal planning does not "
                                   "silently fall back to VALUE or MEMORY"
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
                    << (inferred ? "; automatic group planning will use "
                                   "MEMORY only when VALUE capacity is "
                                   "insufficient"
                                 : "");
    }
    group.resolved_kind = &kind;
    group.resolved_base = counts[kind_index];
    counts[kind_index] += group_size;
  };

  for (DistSignalGroupRecord &group : groups) {
    if (!group.used) {
      continue;
    }
    if (group.requested_kind != kAutoSignalKind) {
      allocate_group(group, *FindDistSignalKindInfo(group.requested_kind),
                     /*inferred=*/false);
    }
  }

  for (DistSignalRecord &record : records) {
    if (record.used && record.requested_kind != kAutoSignalKind) {
      allocate(record, *FindDistSignalKindInfo(record.requested_kind),
               /*inferred=*/false);
    }
  }
  for (DistSignalGroupRecord &group : groups) {
    if (!group.used || group.resolved_kind) {
      continue;
    }
    DistSignalKind value_kind =
        group.destination_scope.value() == DistSignalScope::kSram
            ? DistSignalKind::kSramFlagregValue
            : DistSignalKind::kDramFlagregValue;
    const DistSignalKindInfo &value_info =
        DistSignalKindInfos()[DistSignalKindIndex(value_kind)];
    int64_t remaining =
        value_info.capacity - counts[DistSignalKindIndex(value_kind)];
    if (group.count <= remaining) {
      allocate_group(group, value_info, /*inferred=*/true);
      continue;
    }
    DistSignalKind memory_kind =
        group.destination_scope.value() == DistSignalScope::kSram
            ? DistSignalKind::kSramMemory
            : DistSignalKind::kDramMemory;
    allocate_group(group,
                   DistSignalKindInfos()[DistSignalKindIndex(memory_kind)],
                   /*inferred=*/true);
  }
  for (DistSignalRecord &record : records) {
    if (record.used && record.requested_kind == kAutoSignalKind) {
      DistSignalKind kind =
          record.destination_scope.value() == DistSignalScope::kSram
              ? DistSignalKind::kSramFlagregInc
              : DistSignalKind::kDramFlagregInc;
      allocate(record, DistSignalKindInfos()[DistSignalKindIndex(kind)],
               /*inferred=*/true);
    }
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
      ICHECK_EQ(call->args.size(), 2);
      ValidateSignalArgs(call, 0, 1);
      ICHECK(call->dtype.is_handle());
    } else if (call->op.same_as(dist_signal_group())) {
      ICHECK_EQ(call->args.size(), 3U);
      const DistSignalKindInfo &kind =
          RequireDistSignalKindInfo(call->args[0], "signal-group kind");
      int64_t base = RequireIntImm(call->args[1], "signal-group base index");
      int64_t count = RequireIntImm(call->args[2], "signal-group count");
      ICHECK_GE(base, 0);
      ICHECK_GT(count, 0);
      if (kind.capacity >= 0) {
        ICHECK_LE(base + count, kind.capacity);
      }
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
    } else if (call->op.same_as(dist_wait_barrier())) {
      ICHECK_EQ(call->args.size(), 1U);
      ICHECK(call->args[0].dtype().is_handle());
    } else if (call->op.same_as(DistPutOp::Get())) {
      ValidatePut(call);
    } else if (call->op.same_as(DistWaitSignalOp::Get())) {
      ValidateWaitSignal(call);
    } else if (call->op.same_as(dist_rank_routed_put())) {
      ValidateRankRoutedPut(call);
    } else if (call->op.same_as(dist_routed_put())) {
      ValidateRoutedPut(call);
    } else if (call->op.same_as(dist_wait_all())) {
      ValidateWaitAll(call);
    } else if (call->op.same_as(dist_wait_send())) {
      ICHECK_EQ(call->args.size(), 0);
    } else if (call->op.same_as(dist_completion())) {
      ICHECK_EQ(call->args.size(), 4U);
      ICHECK(call->args[0].dtype().is_handle());
      ValidateStaticRegion(NormalizeToBufferRegion(call->args[1]),
                           "T.dist.all_to_allv", "recv_counts",
                           /*allow_dram=*/true);
      ValidateStaticRegion(NormalizeToBufferRegion(call->args[2]),
                           "T.dist.all_to_allv", "destination",
                           /*allow_dram=*/true);
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
    ICHECK_EQ(call->args.size(), 2U)
        << "T.dist.wait_all requires a destination and one SignalList";
    BufferRegion dst = NormalizeToBufferRegion(call->args[0]);
    ValidateStaticRegion(dst, "T.dist.wait_all", "destination",
                         /*allow_dram=*/true);
    ICHECK(call->args[1].dtype().is_handle())
        << "T.dist.wait_all second argument must be a SignalList handle";
  }

  void ValidateWaitSignal(const CallNode *call) {
    ICHECK_EQ(call->args.size(), 2);
    ICHECK(call->args[0].dtype().is_handle())
        << "T.dist.wait_signal signal must be a signal handle";
    BufferRegion dst = NormalizeToBufferRegion(call->args[1]);
    ValidateStaticRegion(dst, "T.dist.wait_signal", "destination",
                         /*allow_dram=*/true);
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

PrimFunc Run(PrimFunc func) {
  DistOpDetector detector(/*high_level=*/true);
  if (!detector.Detect(func->body)) {
    return func;
  }
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  ICHECK(target && TargetIsSunmmio(target.value()))
      << "T.dist operations only support the SunMMIO target";
  func = PlanSignalResources(std::move(func));
  DistCommunicationValidator().Validate(func);
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
