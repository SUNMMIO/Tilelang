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

class DistSignalUseCollector : public StmtExprVisitor {
public:
  void Collect(const Stmt &stmt) { VisitStmt(stmt); }

  std::vector<DistSignalRecord> &Records() { return records_; }

private:
  void VisitStmt_(const LetStmtNode *let_node) final {
    const auto *call = let_node->value.as<CallNode>();
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
      ICHECK_GE(call->args.size(), 2U);
      BufferRegion destination = NormalizeToBufferRegion(call->args[0]);
      for (size_t index = 1; index < call->args.size(); ++index) {
        RegisterUse(call->args[index], destination, "T.dist.wait_all");
      }
    }
    StmtExprVisitor::VisitExpr_(call);
  }

  void RegisterUse(const PrimExpr &signal, const BufferRegion &destination,
                   const char *op_name) {
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

  std::vector<DistSignalRecord> records_;
  std::unordered_map<const VarNode *, size_t> active_;
  std::unordered_map<int64_t, size_t> logical_ids_;
};

class DistSignalPlanRewriter : public StmtExprMutator {
public:
  explicit DistSignalPlanRewriter(
      const std::unordered_map<const VarNode *, DistSignalRecord> &plans)
      : plans_(plans) {}

private:
  Stmt VisitStmt_(const LetStmtNode *let_node) final {
    const auto *call = let_node->value.as<CallNode>();
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
};

PrimFunc PlanSignalResources(PrimFunc func) {
  if (func->GetAttr<Map<ffi::String, Integer>>(kDistSignalCountsAttr)) {
    return func;
  }

  DistSignalUseCollector collector;
  collector.Collect(func->body);
  std::vector<DistSignalRecord> &records = collector.Records();
  if (records.empty()) {
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

  for (DistSignalRecord &record : records) {
    if (record.used && record.requested_kind != kAutoSignalKind) {
      allocate(record, *FindDistSignalKindInfo(record.requested_kind),
               /*inferred=*/false);
    }
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
  Stmt body = DistSignalPlanRewriter(plans)(func->body);
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
    } else if (call->op.same_as(dist_signal())) {
      ICHECK_EQ(call->args.size(), 2);
      ValidateSignalArgs(call, 0, 1);
      ICHECK(call->dtype.is_handle());
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
    ICHECK_GE(call->args.size(), 2U)
        << "T.dist.wait_all requires a destination and at least one signal";
    BufferRegion dst = NormalizeToBufferRegion(call->args[0]);
    ValidateStaticRegion(dst, "T.dist.wait_all", "destination",
                         /*allow_dram=*/true);
    for (size_t index = 1; index < call->args.size(); ++index) {
      ICHECK(call->args[index].dtype().is_handle())
          << "T.dist.wait_all arguments after destination must be signal "
             "handles";
    }
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
