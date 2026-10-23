/*!
 * \file tl/transform/lower_dist_collectives.cc
 * \brief Lower high-level Rank collectives to logical P2P operations.
 */

#include <tvm/arith/pattern.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/transform.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "../op/copy.h"
#include "../op/dist_collective.h"
#include "../op/reduce.h"
#include "../op/utils.h"
#include "dist_transform_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;
using namespace dist_transform;

namespace {

class DistCollectiveDetector : public StmtExprVisitor {
public:
  bool Detect(const Stmt &stmt) {
    VisitStmt(stmt);
    return found_;
  }

private:
  void VisitExpr_(const CallNode *call) final {
    if (call->op.same_as(DistAllgatherOp::Get()) ||
        call->op.same_as(DistAlltoallOp::Get()) ||
        call->op.same_as(DistAllreduceOp::Get()) ||
        call->op.same_as(DistAlltoallvOp::Get()) ||
        call->op.same_as(dist_barrier()) ||
        call->op.same_as(dist_barrier_arrive())) {
      found_ = true;
      return;
    }
    StmtExprVisitor::VisitExpr_(call);
  }

  bool found_{false};
};

// One all-reduce call site owns one receive buffer. Even independent signals
// cannot stop a faster Rank from overwriting it in the next loop iteration.
class DistAllreduceLoopValidator : public StmtExprVisitor {
public:
  void Validate(const Stmt &body) { VisitStmt(body); }

private:
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

  void VisitExpr_(const CallNode *call) final {
    ICHECK(loop_depth_ == 0 || !call->op.same_as(DistAllreduceOp::Get()))
        << "T.dist.all_reduce inside For/While is not supported: receive "
           "staging may be overwritten before all receivers have consumed "
           "it. Use independent static all_reduce call sites and signals";
    StmtExprVisitor::VisitExpr_(call);
  }

  int loop_depth_{0};
};

// An aggregate collective owns one receiver signal for one invocation. Check
// this before expansion: the many sends produced by one collective are not
// resource reuse. Per-source SignalList collectives remain reusable, but must
// not also write a member reserved by an aggregate collective.
class DistCollectiveSignalValidator : public StmtExprVisitor {
public:
  void Validate(const Stmt &body) { VisitStmt(body); }

private:
  struct Resource {
    bool is_group;
    std::vector<bool> aggregate;
    std::vector<bool> other_writes;
  };

  PrimExpr Resolve(const PrimExpr &expr) {
    return analyzer_.Simplify(Substitute(expr, bindings_));
  }

  std::optional<PrimExpr> CollectiveSignal(const CallNode *call) const {
    if (call->op.same_as(DistAllgatherOp::Get()) ||
        call->op.same_as(DistAlltoallOp::Get()) ||
        call->op.same_as(DistAllreduceOp::Get())) {
      ICHECK_EQ(call->args.size(), 5U);
      return call->args[3];
    }
    if (call->op.same_as(DistAlltoallvOp::Get())) {
      ICHECK_EQ(call->args.size(), 7U);
      return call->args[5];
    }
    return std::nullopt;
  }

  bool IsAggregate(const PrimExpr &signal) {
    PrimExpr resolved = Resolve(signal);
    if (const auto *var = resolved.as<VarNode>()) {
      auto it = resources_.find(var);
      ICHECK(it != resources_.end())
          << "Cannot find collective signal " << signal;
      return !it->second.is_group;
    }
    const auto *ref = resolved.as<CallNode>();
    ICHECK(ref && ref->op.same_as(dist_signal_ref()));
    return true;
  }

  bool HasAggregateCollective(const Stmt &body) {
    bool found = false;
    PostOrderVisit(body, [&](const ObjectRef &node) {
      if (const auto *call = node.as<CallNode>()) {
        auto signal = CollectiveSignal(call);
        if (signal && IsAggregate(signal.value())) {
          found = true;
        }
      }
    });
    return found;
  }

  void RegisterWrite(const PrimExpr &signal, bool aggregate) {
    PrimExpr resolved = Resolve(signal);
    const VarNode *owner = resolved.as<VarNode>();
    std::optional<int64_t> member;
    if (!owner) {
      const auto *ref = resolved.as<CallNode>();
      ICHECK(ref) << "Cannot resolve T.dist signal resource " << signal;
      if (ref->op.same_as(dist_signal_route())) {
        RegisterWrite(ref->args[0], aggregate);
        return;
      }
      ICHECK(ref->op.same_as(dist_signal_ref()) ||
             ref->op.same_as(dist_signal_group_route()));
      owner = ref->args[0].as<VarNode>();
      PrimExpr index = Resolve(ref->args[1]);
      if (const auto *imm = index.as<IntImmNode>()) {
        member = imm->value;
      } else {
        ICHECK(!aggregate)
            << "Cannot prove exclusive collective signal member " << signal
            << "; use statically distinct members for each invocation";
        // Unknown P2P indices may overlap any aggregate-owned group member.
      }
    }
    auto it = resources_.find(owner);
    ICHECK(it != resources_.end()) << "Cannot find T.dist signal " << signal;
    Resource &resource = it->second;
    int64_t begin = member.value_or(0);
    int64_t end = member ? begin + 1 : resource.aggregate.size();
    ICHECK_GE(begin, 0);
    ICHECK_LE(end, static_cast<int64_t>(resource.aggregate.size()));
    for (int64_t index = begin; index < end; ++index) {
      ICHECK(!resource.aggregate[index])
          << "Collective aggregate signal " << signal
          << " must be dedicated to one collective invocation; it is reused "
             "by another collective or signal-writing operation";
      if (aggregate) {
        ICHECK(!resource.other_writes[index])
            << "Collective aggregate signal " << signal
            << " must be dedicated to one collective invocation; another "
               "operation also writes this signal";
        resource.aggregate[index] = true;
      } else {
        resource.other_writes[index] = true;
      }
    }
  }

  void VisitStmt_(const LetStmtNode *op) final {
    const auto *call = op->value.as<CallNode>();
    bool group = call && (call->op.same_as(dist_signal_group_decl()) ||
                          call->op.same_as(dist_signal_group()));
    bool single = call && (call->op.same_as(dist_signal_decl()) ||
                           call->op.same_as(dist_signal()));
    if (group || single) {
      int64_t count =
          group ? RequireIntImm(call->args[2], "signal-group count") : 1;
      ICHECK_GT(count, 0);
      resources_.emplace(op->var.get(),
                         Resource{group, std::vector<bool>(count, false),
                                  std::vector<bool>(count, false)});
      resource_count_ += count;
      VisitStmt(op->body);
      resource_count_ -= count;
      resources_.erase(op->var.get());
      return;
    }
    VisitExpr(op->value);
    Map<Var, PrimExpr> previous = bindings_;
    bindings_.Set(op->var, Resolve(op->value));
    VisitStmt(op->body);
    bindings_ = std::move(previous);
  }

  void VisitStmt_(const ForNode *op) final {
    if (!HasAggregateCollective(op->body)) {
      // Ordinary writes need only a conservative member footprint, not a
      // count. Unknown members are treated as possibly touching the group.
      VisitStmt(op->body);
      return;
    }
    PrimExpr min = Resolve(op->min);
    PrimExpr extent = Resolve(op->extent);
    const auto *start = min.as<IntImmNode>();
    const auto *count = extent.as<IntImmNode>();
    ICHECK(start && count && count->value >= 0 &&
           count->value <= resource_count_)
        << "Cannot prove exclusive collective signal use across loop "
           "iterations; use statically bounded distinct members or the "
           "supported per-source signals= interface";
    Map<Var, PrimExpr> previous = bindings_;
    for (int64_t offset = 0; offset < count->value; ++offset) {
      bindings_.Set(op->loop_var,
                    IntImm(op->loop_var.dtype(), start->value + offset));
      VisitStmt(op->body);
      bindings_ = previous;
    }
  }

  void VisitStmt_(const WhileNode *op) final {
    ICHECK(!HasAggregateCollective(op->body))
        << "Cannot prove exclusive collective signal use across While "
           "iterations; use the supported per-source signals= interface "
           "or independent static collective invocations";
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *call) final {
    auto signal = CollectiveSignal(call);
    if (signal) {
      RegisterWrite(signal.value(), IsAggregate(signal.value()));
    } else if (call->op.same_as(DistPutOp::Get()) ||
               call->op.same_as(dist_rank_routed_put())) {
      RegisterWrite(call->args[4], false);
    } else if (call->op.same_as(dist_routed_put())) {
      RegisterWrite(call->args[3], false);
    } else if (call->op.same_as(dist_signal_put()) ||
               call->op.same_as(dist_barrier()) ||
               call->op.same_as(dist_barrier_arrive())) {
      RegisterWrite(call->args[0], false);
    }
    StmtExprVisitor::VisitExpr_(call);
  }

  arith::Analyzer analyzer_;
  Map<Var, PrimExpr> bindings_;
  std::unordered_map<const VarNode *, Resource> resources_;
  int64_t resource_count_{0};
};

class DistCollectiveLowerer : public StmtExprMutator {
public:
  DistCollectiveLowerer(int64_t world_size, Var rank_id, int mesh_nrows,
                        int mesh_ncols)
      : world_size_(world_size), rank_id_(std::move(rank_id)),
        mesh_nrows_(mesh_nrows), mesh_ncols_(mesh_ncols) {}

private:
  PrimExpr I32(int64_t value) const { return IntImm(DataType::Int(32), value); }

  Map<String, ObjectRef> CompilerRegionAnnotations() const {
    Map<String, ObjectRef> annotations;
    annotations.Set(kDistCompilerRegionAttr, Integer(1));
    return annotations;
  }

  int64_t RequireStaticExtent(const Range &range, const char *op_name,
                              const char *name) const {
    const auto *extent = range->extent.as<IntImmNode>();
    ICHECK(extent) << op_name << " " << name
                   << " extent must be a compile-time integer, got "
                   << range->extent;
    ICHECK_GT(extent->value, 0)
        << op_name << " " << name << " extent must be positive";
    return extent->value;
  }

  void ValidateConstantOffsets(const Array<Range> &region, const char *op_name,
                               const char *operand_name) const {
    arith::Analyzer analyzer;
    for (const Range &range : region) {
      PrimExpr offset = analyzer.Simplify(range->min);
      ICHECK(offset.as<IntImmNode>())
          << op_name << " " << operand_name
          << " region offsets must be compile-time constants; collective "
             "endpoint slots are assigned by the compiler, got "
          << range->min;
    }
  }

  struct OffsetLoop {
    Var var;
    PrimExpr extent;
    bool supported;
  };

  PrimExpr ResolveOffset(const PrimExpr &expr) const {
    return offset_analyzer_.Simplify(Substitute(expr, offset_bindings_));
  }

  bool IsUniformOffsetExpr(const PrimExpr &expr) const {
    bool valid = true;
    PostOrderVisit(expr, [&](const ObjectRef &node) {
      if (const auto *var = node.as<VarNode>()) {
        bool loop_var = false;
        for (const auto &loop : offset_loops_)
          loop_var |= loop.var.get() == var && loop.supported;
        valid &= loop_var;
      } else if (node.as<BufferLoadNode>() || node.as<CallNode>()) {
        valid = false;
      }
    });
    return valid;
  }

  // Only newly admitted symbolic offsets use these checks. Existing constant
  // region behavior (including ROW collectives) remains unchanged.
  void ValidateLoopRegions(const std::vector<BufferRegion> &regions,
                           const PrimExpr &signal, bool row_domain,
                           const char *op_name, int endpoint_dims,
                           int gather_axis = -2) const {
    bool symbolic = false;
    for (const auto &region : regions)
      for (const auto &range : region->region)
        symbolic |= !ResolveOffset(range->min).as<IntImmNode>();
    if (!symbolic)
      return;

    ICHECK(!row_domain) << op_name
                        << " loop region offsets only support the RANK domain";
    ICHECK(!offset_loops_.empty() && offset_while_depth_ == 0)
        << op_name << " region offsets must be compile-time constants or "
        << "affine expressions of statically bounded serial loops";
    ICHECK(!IsSingleSignalResource(signal))
        << op_name << " loop region offsets require per-source signals=";
    Array<Var> variables;
    for (const auto &loop : offset_loops_) {
      ICHECK(loop.supported)
          << op_name << " loop region offsets require statically bounded "
          << "serial loops without early exits";
      variables.push_back(loop.var);
    }
    for (const PrimExpr &condition : offset_conditions_) {
      ICHECK(IsUniformOffsetExpr(condition))
          << op_name << " loop region offsets require Rank/core-uniform "
          << "control flow";
    }

    std::vector<Array<PrimExpr>> destination_coefficients;
    for (size_t operand = 0; operand < regions.size(); ++operand) {
      const BufferRegion &region = regions[operand];
      const char *name = operand == 0   ? "source"
                         : operand == 1 ? "destination"
                         : operand == 2 ? "send_counts"
                                        : "recv_counts";
      ICHECK_EQ(region->region.size(), region->buffer->shape.size());
      for (size_t dim = 0; dim < region->region.size(); ++dim) {
        const Range &range = region->region[dim];
        PrimExpr min = ResolveOffset(range->min);
        int64_t extent = RequireStaticExtent(range, op_name, name);
        ICHECK(IsUniformOffsetExpr(min))
            << op_name << " " << name
            << " region offsets must be compile-time constants or "
               "Rank/core-uniform affine loop expressions, got "
            << min;
        Array<PrimExpr> coefficients =
            arith::DetectLinearEquation(min, variables);
        ICHECK_EQ(coefficients.size(), variables.size() + 1)
            << op_name << " " << name << " loop region offset must be affine";
        for (const PrimExpr &coefficient : coefficients) {
          ICHECK(offset_analyzer_.Simplify(coefficient).as<IntImmNode>())
              << op_name << " loop offset coefficients must be constants";
        }
        ICHECK(offset_analyzer_.CanProve(min >= 0) &&
               offset_analyzer_.CanProve(min + range->extent <=
                                         region->buffer->shape[dim]))
            << op_name << " cannot prove " << name << " region is in bounds "
            << "for buffer " << region->buffer->name << " at dimension " << dim
            << ": min=" << min << ", extent=" << extent;

        // Counts are scalar accesses. Payload checks use the actual per-peer
        // tile extent, not the whole collective's endpoint/gather dimension.
        if (operand < 2) {
          int64_t tile_extent = extent;
          if (gather_axis == -2) {
            if (static_cast<int>(dim) < endpoint_dims)
              tile_extent = 1;
          } else if (operand == 1) {
            if (gather_axis < 0 && dim == 0)
              tile_extent = 1;
            else if (static_cast<int>(dim) == gather_axis)
              tile_extent /= world_size_;
          }
          ICHECK_GT(tile_extent, 0);
          ICHECK(offset_analyzer_.CanProveEqual(floormod(min, tile_extent), 0))
              << op_name << " " << name << " loop region min at dimension "
              << dim << " must align to per-peer tile extent " << tile_extent;
          ICHECK(offset_analyzer_.CanProveEqual(
              floormod(region->buffer->shape[dim], tile_extent), 0))
              << op_name << " per-peer tile extent must divide buffer shape "
              << "at dimension " << dim;
        }
        if (operand == 1)
          destination_coefficients.push_back(coefficients);
      }
    }

    // Sufficient affine separation test: for the first differing loop index,
    // at least one destination axis separates the complete inner-loop spans.
    // This covers separate axes and flattened expert/block slots without
    // unrolling the kernel or accepting ring-buffer reuse.
    const auto &destination = regions[1];
    for (size_t loop = 0; loop < offset_loops_.size(); ++loop) {
      if (offset_analyzer_.CanProve(offset_loops_[loop].extent <= 1))
        continue;
      bool separated = false;
      for (size_t dim = 0; dim < destination_coefficients.size(); ++dim) {
        const auto &coefficients = destination_coefficients[dim];
        PrimExpr inner_span = destination->region[dim]->extent;
        for (size_t inner = loop + 1; inner < offset_loops_.size(); ++inner)
          inner_span +=
              abs(coefficients[inner]) * (offset_loops_[inner].extent - 1);
        separated |=
            offset_analyzer_.CanProve(abs(coefficients[loop]) >= inner_span);
      }
      ICHECK(separated)
          << op_name << " cannot prove independent destination slots across "
          << "loop " << offset_loops_[loop].var
          << "; receive-buffer reuse requires a consumption protocol";
    }
  }

  void ValidateLoopCompletion(const LetStmtNode *op) const {
    const auto *call = op->value.as<CallNode>();
    size_t operands = call->op.same_as(DistAlltoallvOp::Get()) ? 4 : 2;
    bool symbolic = false;
    for (size_t operand = 0; operand < operands; ++operand) {
      BufferRegion region = NormalizeToBufferRegion(call->args[operand]);
      for (const Range &range : region->region)
        symbolic |= !ResolveOffset(range->min).as<IntImmNode>();
    }
    if (!symbolic)
      return;

    class IterationWaits : public StmtVisitor {
    public:
      explicit IterationWaits(Var completion)
          : completion_(std::move(completion)) {}
      bool received{false};
      bool sent{false};

    private:
      void VisitStmt_(const EvaluateNode *op) final {
        const auto *call = op->value.as<CallNode>();
        if (!call)
          return;
        if (call->op.same_as(dist_wait_completion_all()) &&
            call->args[0].same_as(completion_))
          received = true;
        sent |= call->op.same_as(dist_wait_send());
      }
      void VisitStmt_(const IfThenElseNode *) final {}
      void VisitStmt_(const ForNode *) final {}
      void VisitStmt_(const BlockRealizeNode *op) final {
        if (is_one(op->predicate))
          StmtVisitor::VisitStmt_(op);
      }
      void VisitStmt_(const WhileNode *op) final {
        const auto *condition = op->condition.as<CallNode>();
        if (!condition ||
            !condition->op.same_as(dist_completion_has_pending()) ||
            !condition->args[0].same_as(completion_))
          return;
        int waits = 0;
        bool exits = false;
        PostOrderVisit(op->body, [&](const ObjectRef &node) {
          if (const auto *call = node.as<CallNode>()) {
            waits += call->op.same_as(dist_wait_any()) &&
                     call->args[0].same_as(completion_);
            exits |= call->op.same_as(builtin::ret()) ||
                     call->op.same_as(builtin::break_loop()) ||
                     call->op.same_as(builtin::continue_loop());
          }
          exits |= node.as<IfThenElseNode>() || node.as<ForNode>() ||
                   node.as<WhileNode>();
          if (const auto *block = node.as<BlockRealizeNode>())
            exits |= !is_one(block->predicate);
        });
        received |= waits == 1 && !exits;
      }
      Var completion_;
    } waits(op->var);
    waits(op->body);
    ICHECK(waits.received && waits.sent)
        << "Collective loop region offsets require the completion to be "
           "fully drained and T.dist.wait() in the same iteration";
  }

  void ValidateSignal(const PrimExpr &signal, const char *op_name) const {
    ICHECK(signal.dtype().is_handle())
        << op_name << " signal must be a signal handle";
    const VarNode *signal_var = signal.as<VarNode>();
    const std::string *kind_ptr = nullptr;
    if (signal_var) {
      auto active = active_signal_kinds_.find(signal_var);
      ICHECK(active != active_signal_kinds_.end())
          << op_name << " cannot find the corresponding T.dist.signal";
      kind_ptr = &active->second;
    } else {
      const auto *ref = signal.as<CallNode>();
      ICHECK(ref && ref->op.same_as(dist_signal_ref()))
          << op_name << " signal must reference T.dist.signal or a static "
          << "T.dist.signals member";
      ICHECK_EQ(ref->args.size(), 2U);
      signal_var = ref->args[0].as<VarNode>();
      ICHECK(signal_var);
      auto active = active_signal_group_kinds_.find(signal_var);
      ICHECK(active != active_signal_group_kinds_.end())
          << op_name << " cannot find the corresponding T.dist.signals";
      ICHECK(ref->args[1].dtype().is_int())
          << op_name << " signal-group index must have integer dtype";
      auto size = active_signal_group_sizes_.find(signal_var);
      ICHECK(size != active_signal_group_sizes_.end());
      if (const auto *index = ref->args[1].as<IntImmNode>()) {
        ICHECK_GE(index->value, 0);
        ICHECK_LT(index->value, size->second);
      }
      kind_ptr = &active->second;
    }
    const std::string &kind = *kind_ptr;
    if (kind != kAutoSignalKind) {
      const DistSignalKindInfo *kind_info = FindDistSignalKindInfo(kind);
      ICHECK(kind_info &&
             kind_info->update_mode == DistSignalUpdateMode::kIncrement)
          << op_name << " requires an automatic or INC flagreg signal, got "
          << kind;
    }
  }

  bool ValidateAllgather(const DistAllgatherOpNode *op) const {
    ICHECK(op->src->dtype == op->dst->dtype)
        << "T.dist.all_gather source and destination dtypes must match";
    ICHECK(!op->src->data.same_as(op->dst->data))
        << "T.dist.all_gather does not support overlapping source and "
           "destination storage";
    ICHECK(op->current_core.dtype().is_int())
        << "T.dist.all_gather current_core must have integer dtype";
    ValidateLoopRegions({BufferRegion(op->src, op->src_range),
                         BufferRegion(op->dst, op->dst_range)},
                        op->signal, false, "T.dist.all_gather", 0, op->axis);

    size_t src_rank = op->src_range.size();
    size_t dst_rank = op->dst_range.size();
    ICHECK(op->axis == -1 ||
           (src_rank > 0 &&
            (op->axis == 0 || op->axis == static_cast<int>(src_rank) - 1)))
        << "T.dist.all_gather only supports axis=None, axis=0, or axis=-1";

    for (size_t dim = 0; dim < src_rank; ++dim) {
      RequireStaticExtent(op->src_range[dim], "T.dist.all_gather", "source");
    }
    for (size_t dim = 0; dim < dst_rank; ++dim) {
      RequireStaticExtent(op->dst_range[dim], "T.dist.all_gather",
                          "destination");
    }

    if (op->axis < 0) {
      ICHECK_EQ(dst_rank, src_rank + 1)
          << "T.dist.all_gather destination rank must be source rank + 1 "
             "when axis=None";
      ICHECK_EQ(RequireStaticExtent(op->dst_range[0], "T.dist.all_gather",
                                    "destination"),
                world_size_)
          << "T.dist.all_gather destination leading extent must equal "
             "world_size";
      for (size_t dim = 0; dim < src_rank; ++dim) {
        ICHECK_EQ(RequireStaticExtent(op->src_range[dim], "T.dist.all_gather",
                                      "source"),
                  RequireStaticExtent(op->dst_range[dim + 1],
                                      "T.dist.all_gather", "destination"))
            << "T.dist.all_gather source and destination slot shapes must "
               "match";
      }
    } else {
      ICHECK_EQ(dst_rank, src_rank)
          << "T.dist.all_gather source and destination ranks must match when "
             "axis is specified";
      for (size_t dim = 0; dim < src_rank; ++dim) {
        int64_t src_extent = RequireStaticExtent(op->src_range[dim],
                                                 "T.dist.all_gather", "source");
        int64_t expected = dim == static_cast<size_t>(op->axis)
                               ? world_size_ * src_extent
                               : src_extent;
        ICHECK_EQ(RequireStaticExtent(op->dst_range[dim], "T.dist.all_gather",
                                      "destination"),
                  expected)
            << "T.dist.all_gather destination shape does not match source "
               "shape and world_size";
      }
    }

    bool single_signal = IsSingleSignalResource(op->signal);
    if (single_signal) {
      ValidateSignal(op->signal, "T.dist.all_gather");
      return true;
    }
    const auto *group_var = op->signal.as<VarNode>();
    ICHECK(group_var)
        << "T.dist.all_gather signals must reference T.dist.signals";
    auto group = active_signal_group_sizes_.find(group_var);
    ICHECK(group != active_signal_group_sizes_.end())
        << "T.dist.all_gather cannot find its T.dist.signals declaration";
    ICHECK_EQ(group->second, world_size_)
        << "T.dist.all_gather requires one signal per source Rank";
    return false;
  }

  bool IsRowDomain(const DistAlltoallOpNode *op) const {
    ICHECK(op->domain == "rank" || op->domain == "row")
        << "T.dist.all_to_all domain must be 'rank' or 'row', got "
        << op->domain;
    return op->domain == "row";
  }

  bool ValidateAlltoall(const DistAlltoallOpNode *op) const {
    ICHECK(op->src->dtype == op->dst->dtype)
        << "T.dist.all_to_all source and destination dtypes must match";
    ICHECK(!op->src->data.same_as(op->dst->data))
        << "T.dist.all_to_all does not support overlapping source and "
           "destination storage";
    ICHECK(op->current_core.dtype().is_int())
        << "T.dist.all_to_all current_core must have integer dtype";
    ValidateLoopRegions({BufferRegion(op->src, op->src_range),
                         BufferRegion(op->dst, op->dst_range)},
                        op->signal, IsRowDomain(op), "T.dist.all_to_all", 1);

    bool row_domain = IsRowDomain(op);
    size_t endpoint_dims = row_domain ? 2U : 1U;
    ICHECK_EQ(op->src_range.size(), op->dst_range.size())
        << "T.dist.all_to_all source and destination ranks must match";
    ICHECK_GE(op->src_range.size(), endpoint_dims)
        << "T.dist.all_to_all domain=" << op->domain << " requires "
        << endpoint_dims << " leading endpoint dimensions";

    for (size_t dim = 0; dim < op->src_range.size(); ++dim) {
      int64_t src_extent = RequireStaticExtent(op->src_range[dim],
                                               "T.dist.all_to_all", "source");
      int64_t dst_extent = RequireStaticExtent(
          op->dst_range[dim], "T.dist.all_to_all", "destination");
      ICHECK_EQ(src_extent, dst_extent)
          << "T.dist.all_to_all source and destination shapes must match";
    }
    ICHECK_EQ(
        RequireStaticExtent(op->src_range[0], "T.dist.all_to_all", "source"),
        world_size_)
        << "T.dist.all_to_all leading Rank extent must equal world_size";
    if (row_domain) {
      ICHECK_EQ(
          RequireStaticExtent(op->src_range[1], "T.dist.all_to_all", "source"),
          mesh_nrows_)
          << "T.dist.all_to_all ROW domain leading row extent must equal "
             "mesh_nrows";
    }
    bool single_signal = IsSingleSignalResource(op->signal);
    if (single_signal) {
      ValidateSignal(op->signal, "T.dist.all_to_all");
      return true;
    }
    const auto *group_var = op->signal.as<VarNode>();
    ICHECK(group_var)
        << "T.dist.all_to_all signals must reference T.dist.signals";
    auto group = active_signal_group_sizes_.find(group_var);
    ICHECK(group != active_signal_group_sizes_.end())
        << "T.dist.all_to_all cannot find its signal declaration";
    int64_t endpoint_count = world_size_ * (row_domain ? mesh_nrows_ : 1);
    ICHECK_EQ(group->second, endpoint_count)
        << "T.dist.all_to_all requires one signal per source endpoint";
    return false;
  }

  void ValidateAllreduce(const DistAllreduceOpNode *op) const {
    ICHECK(op->reduce_type == "sum" || op->reduce_type == "max" ||
           op->reduce_type == "min")
        << "T.dist.all_reduce reduce_type must be 'sum', 'max', or 'min', "
           "got "
        << op->reduce_type;
    ICHECK(op->src->dtype == op->dst->dtype)
        << "T.dist.all_reduce source and destination dtypes must match";
    ICHECK(!op->src->data.same_as(op->dst->data))
        << "T.dist.all_reduce does not support overlapping source and "
           "destination storage";
    ICHECK_EQ(op->src_range.size(), op->dst_range.size())
        << "T.dist.all_reduce source and destination ranks must match";
    ICHECK_GT(op->src_range.size(), 0U)
        << "T.dist.all_reduce does not support scalar buffers";
    ValidateConstantOffsets(op->src_range, "T.dist.all_reduce", "source");
    ValidateConstantOffsets(op->dst_range, "T.dist.all_reduce", "destination");
    for (size_t dim = 0; dim < op->src_range.size(); ++dim) {
      ICHECK_EQ(RequireStaticExtent(op->src_range[dim], "T.dist.all_reduce",
                                    "source"),
                RequireStaticExtent(op->dst_range[dim], "T.dist.all_reduce",
                                    "destination"))
          << "T.dist.all_reduce source and destination shapes must match";
    }
    auto valid_scope = [](const Buffer &buffer) {
      const ffi::String &scope = buffer.scope();
      return scope == "shared" || scope == "shared.dyn" ||
             scope == kSunmmioScopeRSRAM;
    };
    ICHECK(valid_scope(op->src) && valid_scope(op->dst))
        << "T.dist.all_reduce source and destination must use RSRAM or "
           "generic shared scope";
    ICHECK(op->current_core.dtype().is_int())
        << "T.dist.all_reduce current_core must have integer dtype";
    ValidateSignal(op->signal, "T.dist.all_reduce");
  }

  bool IsRowDomain(const DistAlltoallvOpNode *op) const {
    ICHECK(op->domain == "rank" || op->domain == "row")
        << "T.dist.all_to_allv domain must be 'rank' or 'row', got "
        << op->domain;
    return op->domain == "row";
  }

  bool IsSingleSignalResource(const PrimExpr &resource) const {
    if (const auto *ref = resource.as<CallNode>()) {
      ICHECK(ref->op.same_as(dist_signal_ref()))
          << "T.dist collective signal resource must reference "
             "T.dist.signal or T.dist.signals";
      return true;
    }
    const auto *var = resource.as<VarNode>();
    ICHECK(var) << "T.dist collective signal resource must be a handle";
    if (active_signal_kinds_.count(var)) {
      return true;
    }
    ICHECK(active_signal_group_sizes_.count(var))
        << "T.dist collective cannot find its signal declaration";
    return false;
  }

  bool ValidateAlltoallv(const DistAlltoallvOpNode *op) const {
    ICHECK(op->src->dtype == op->dst->dtype)
        << "T.dist.all_to_allv source and destination dtypes must match";
    ICHECK(!op->src->data.same_as(op->dst->data))
        << "T.dist.all_to_allv does not support overlapping source and "
           "destination storage";
    ICHECK(op->current_core.dtype().is_int())
        << "T.dist.all_to_allv current_core must have integer dtype";
    ICHECK(
        (op->send_counts->dtype.is_int() || op->send_counts->dtype.is_uint()) &&
        op->send_counts->dtype.bits() == 32)
        << "T.dist.all_to_allv send_counts must use int32 or uint32";
    ICHECK(op->recv_counts->dtype == op->send_counts->dtype)
        << "T.dist.all_to_allv send_counts and recv_counts must have the "
           "same dtype";
    ValidateLoopRegions({BufferRegion(op->src, op->src_range),
                         BufferRegion(op->dst, op->dst_range),
                         BufferRegion(op->send_counts, op->send_counts_range),
                         BufferRegion(op->recv_counts, op->recv_counts_range)},
                        op->signal_resource, IsRowDomain(op),
                        "T.dist.all_to_allv", 1);

    bool row_domain = IsRowDomain(op);
    size_t endpoint_dims = row_domain ? 2U : 1U;
    ICHECK_EQ(op->src_range.size(), op->dst_range.size());
    ICHECK_GE(op->src_range.size(), endpoint_dims + 1U)
        << "T.dist.all_to_allv requires a capacity dimension after endpoint "
           "dimensions";
    for (size_t dim = 0; dim < op->src_range.size(); ++dim) {
      ICHECK_EQ(RequireStaticExtent(op->src_range[dim], "T.dist.all_to_allv",
                                    "source"),
                RequireStaticExtent(op->dst_range[dim], "T.dist.all_to_allv",
                                    "destination"))
          << "T.dist.all_to_allv source and destination shapes must match";
    }
    ICHECK_EQ(
        RequireStaticExtent(op->src_range[0], "T.dist.all_to_allv", "source"),
        world_size_);
    if (row_domain) {
      ICHECK_EQ(
          RequireStaticExtent(op->src_range[1], "T.dist.all_to_allv", "source"),
          mesh_nrows_);
    }

    ICHECK_EQ(op->send_counts_range.size(), endpoint_dims)
        << "T.dist.all_to_allv send_counts rank must match its endpoint "
           "domain";
    ICHECK_EQ(op->recv_counts_range.size(), endpoint_dims)
        << "T.dist.all_to_allv recv_counts rank must match its endpoint "
           "domain";
    for (size_t dim = 0; dim < endpoint_dims; ++dim) {
      int64_t expected = dim == 0 ? world_size_ : mesh_nrows_;
      ICHECK_EQ(RequireStaticExtent(op->send_counts_range[dim],
                                    "T.dist.all_to_allv", "send_counts"),
                expected);
      ICHECK_EQ(RequireStaticExtent(op->recv_counts_range[dim],
                                    "T.dist.all_to_allv", "recv_counts"),
                expected);
    }

    bool single_signal = IsSingleSignalResource(op->signal_resource);
    if (single_signal) {
      ValidateSignal(op->signal_resource, "T.dist.all_to_allv");
      return true;
    }

    const auto *group_var = op->signal_resource.as<VarNode>();
    ICHECK(group_var)
        << "T.dist.all_to_allv signals must reference T.dist.signals";
    auto group = active_signal_group_sizes_.find(group_var);
    ICHECK(group != active_signal_group_sizes_.end())
        << "T.dist.all_to_allv cannot find its T.dist.signals declaration";
    int64_t endpoint_count = world_size_ * (row_domain ? mesh_nrows_ : 1);
    ICHECK_EQ(group->second, endpoint_count)
        << "T.dist.all_to_allv requires one signal per source endpoint";
    return false;
  }

  Array<Range> DestinationSlot(const DistAllgatherOpNode *op,
                               const PrimExpr &source_rank) const {
    Array<Range> ranges;
    ranges.reserve(op->dst_range.size());
    for (size_t dim = 0; dim < op->dst_range.size(); ++dim) {
      const Range &dst_range = op->dst_range[dim];
      if (op->axis < 0 && dim == 0) {
        ranges.push_back(
            Range::FromMinExtent(dst_range->min + source_rank, I32(1)));
      } else if (op->axis >= 0 && dim == static_cast<size_t>(op->axis)) {
        PrimExpr slot_extent = op->src_range[dim]->extent;
        ranges.push_back(Range::FromMinExtent(
            dst_range->min + source_rank * slot_extent, slot_extent));
      } else {
        ranges.push_back(dst_range);
      }
    }
    return ranges;
  }

  struct AllgatherLowering {
    Stmt schedule;
    std::optional<PrimExpr> completion;
  };

  AllgatherLowering LowerAllgather(const DistAllgatherOpNode *op) const {
    bool single_signal = ValidateAllgather(op);
    PrimExpr src = MakeRegionExpr(op->src, op->src_range, /*access_mask=*/1);
    PrimExpr dst_slot = MakeRegionExpr(op->dst, DestinationSlot(op, rank_id_),
                                       /*access_mask=*/2);

    Array<Stmt> schedule;
    schedule.push_back(
        Evaluate(Call(DataType::Handle(), Copy::Get(), {src, dst_slot})));

    PrimExpr dst_row = floordiv(op->current_core, I32(mesh_ncols_));
    for (int64_t offset = 1; offset < world_size_; ++offset) {
      PrimExpr dst_rank = floormod(rank_id_ + I32(offset), I32(world_size_));
      PrimExpr signal = op->signal;
      if (!single_signal) {
        PrimExpr generation_index = rank_id_ * I32(world_size_) + dst_rank;
        signal = Call(DataType::Handle(), dist_signal_group_route(),
                      {op->signal, rank_id_, generation_index, const_true()});
      }
      schedule.push_back(Evaluate(
          Call(DataType::Handle(), DistPutOp::Get(),
               {src, dst_slot, dst_rank, dst_row, signal, op->current_core},
               CompilerRegionAnnotations())));
    }

    if (single_signal) {
      return {SeqStmt::Flatten(schedule), std::nullopt};
    }
    PrimExpr completion = Call(DataType::Handle(), dist_completion(),
                               {op->signal, StringImm("all_gather")});
    return {SeqStmt::Flatten(schedule), completion};
  }

  BufferRegion CreateAllreduceGather(const DistAllreduceOpNode *op) {
    ICHECK(!alloc_buffer_stack_.empty());
    Array<Range> region;
    region.reserve(op->src_range.size() + 1U);
    region.push_back(Range::FromMinExtent(I32(0), I32(world_size_)));
    for (const Range &range : op->src_range) {
      region.push_back(Range::FromMinExtent(I32(0), range->extent));
    }
    std::string name =
        "dist_allreduce_gather_" + std::to_string(allreduce_counter_++);
    Buffer buffer =
        MakeCompactBufferLike(op->src, region, kSunmmioScopeRSRAM, name);
    alloc_buffer_stack_.back().push_back(buffer);
    return BufferRegion(buffer, std::move(region));
  }

  Stmt LowerAllreduce(const DistAllreduceOpNode *op) {
    ValidateAllreduce(op);
    BufferRegion gather = CreateAllreduceGather(op);
    Array<Range> local_slot;
    local_slot.reserve(gather->region.size());
    local_slot.push_back(Range::FromMinExtent(rank_id_, I32(1)));
    for (size_t dim = 1; dim < gather->region.size(); ++dim) {
      local_slot.push_back(gather->region[dim]);
    }

    PrimExpr src = MakeRegionExpr(op->src, op->src_range, /*access_mask=*/1);
    PrimExpr gather_slot =
        MakeRegionExpr(gather->buffer, local_slot, /*access_mask=*/2);
    Array<Stmt> schedule{
        Evaluate(Call(DataType::Handle(), Copy::Get(), {src, gather_slot}))};

    PrimExpr current_row = floordiv(op->current_core, I32(mesh_ncols_));
    for (int64_t offset = 1; offset < world_size_; ++offset) {
      PrimExpr dst_rank = floormod(rank_id_ + I32(offset), I32(world_size_));
      schedule.push_back(Evaluate(Call(DataType::Handle(), DistPutOp::Get(),
                                       {src, gather_slot, dst_rank, current_row,
                                        op->signal, op->current_core},
                                       CompilerRegionAnnotations())));
    }

    PrimExpr gather_region =
        MakeRegionExpr(gather->buffer, gather->region, /*access_mask=*/1);
    schedule.push_back(Evaluate(Call(DataType::Handle(), dist_submit(), {})));
    schedule.push_back(
        Evaluate(Call(DataType::Handle(), dist_wait_signal(), {op->signal})));
    schedule.push_back(Evaluate(
        Call(DataType::Handle(), ReduceOp::Get(),
             {gather_region,
              MakeRegionExpr(op->dst, op->dst_range, /*access_mask=*/2),
              StringImm(op->reduce_type), I32(0), Bool(true)})));
    schedule.push_back(
        Evaluate(Call(DataType::Handle(), dist_wait_send(), {})));
    return SeqStmt::Flatten(schedule);
  }

  Array<Range> EndpointSlot(const Array<Range> &region, const PrimExpr &rank,
                            const PrimExpr &row, bool row_domain) const {
    Array<Range> ranges;
    ranges.reserve(region.size());
    for (size_t dim = 0; dim < region.size(); ++dim) {
      const Range &range = region[dim];
      if (dim == 0) {
        ranges.push_back(Range::FromMinExtent(range->min + rank, I32(1)));
      } else if (row_domain && dim == 1) {
        ranges.push_back(Range::FromMinExtent(range->min + row, I32(1)));
      } else {
        ranges.push_back(range);
      }
    }
    return ranges;
  }

  struct AlltoallLowering {
    Stmt schedule;
    std::optional<PrimExpr> completion;
  };

  AlltoallLowering LowerAlltoall(const DistAlltoallOpNode *op) const {
    bool single_signal = ValidateAlltoall(op);
    bool row_domain = IsRowDomain(op);
    PrimExpr current_row = floordiv(op->current_core, I32(mesh_ncols_));
    PrimExpr source_index = EndpointIndex(rank_id_, current_row, row_domain);
    int64_t endpoint_count = world_size_ * (row_domain ? mesh_nrows_ : 1);
    Array<Range> dst_slot_range =
        EndpointSlot(op->dst_range, rank_id_, current_row, row_domain);
    PrimExpr dst_slot =
        MakeRegionExpr(op->dst, dst_slot_range, /*access_mask=*/2);

    Array<Stmt> schedule;
    for (int64_t offset = 0; offset < world_size_; ++offset) {
      PrimExpr dst_rank = floormod(rank_id_ + I32(offset), I32(world_size_));
      int64_t row_count = row_domain ? mesh_nrows_ : 1;
      for (int64_t dst_row_value = 0; dst_row_value < row_count;
           ++dst_row_value) {
        PrimExpr dst_row = row_domain ? I32(dst_row_value) : current_row;
        Array<Range> src_slot_range =
            EndpointSlot(op->src_range, dst_rank, dst_row, row_domain);
        PrimExpr src =
            MakeRegionExpr(op->src, src_slot_range, /*access_mask=*/1);
        PrimExpr signal = op->signal;
        if (!single_signal) {
          PrimExpr destination_index =
              EndpointIndex(dst_rank, dst_row, row_domain);
          PrimExpr generation_index =
              source_index * I32(endpoint_count) + destination_index;
          signal =
              Call(DataType::Handle(), dist_signal_group_route(),
                   {op->signal, source_index, generation_index, const_true()});
        }
        schedule.push_back(Evaluate(
            Call(DataType::Handle(), DistPutOp::Get(),
                 {src, dst_slot, dst_rank, dst_row, signal, op->current_core},
                 CompilerRegionAnnotations())));
      }
    }

    if (single_signal)
      return {SeqStmt::Flatten(schedule), std::nullopt};
    PrimExpr completion =
        Call(DataType::Handle(), dist_completion(),
             {op->signal,
              StringImm(row_domain ? "all_to_all_row" : "all_to_all_rank")});
    return {SeqStmt::Flatten(schedule), completion};
  }

  Stmt LowerBarrierArrive(const CallNode *call, const char *op_name) const {
    ICHECK_EQ(call->args.size(), 2U);
    ValidateSignal(call->args[0], op_name);
    ICHECK(call->args[1].dtype().is_int())
        << op_name << " current_core must have integer dtype";

    Array<Stmt> schedule;
    for (int64_t offset = 1; offset < world_size_; ++offset) {
      PrimExpr dst_rank = floormod(rank_id_ + I32(offset), I32(world_size_));
      schedule.push_back(
          Evaluate(Call(DataType::Handle(), dist_signal_put(),
                        {call->args[0], dst_rank, call->args[1]})));
    }
    return SeqStmt::Flatten(schedule);
  }

  Stmt LowerBarrier(const CallNode *call) const {
    Stmt arrive = LowerBarrierArrive(call, "T.dist.barrier");
    Stmt submit = Evaluate(Call(DataType::Handle(), dist_submit(), {}));
    Stmt wait =
        Evaluate(Call(DataType::Handle(), dist_wait_signal(), {call->args[0]}));
    return SeqStmt::Flatten(Array<Stmt>{arrive, submit, wait});
  }

  PrimExpr EndpointIndex(const PrimExpr &rank, const PrimExpr &row,
                         bool row_domain) const {
    return row_domain ? rank * I32(mesh_nrows_) + row : rank;
  }

  PrimExpr CountLoad(const Buffer &buffer, const Array<Range> &region,
                     const PrimExpr &rank, const PrimExpr &row,
                     bool row_domain) const {
    Array<PrimExpr> indices{region[0]->min + rank};
    if (row_domain) {
      indices.push_back(region[1]->min + row);
    }
    return BufferLoad(buffer, indices);
  }

  PrimExpr AggregateRecvDelta(const DistAlltoallvOpNode *op,
                              bool row_domain) const {
    PrimExpr delta = I32(0);
    int64_t row_count = row_domain ? mesh_nrows_ : 1;
    for (int64_t src_rank = 0; src_rank < world_size_; ++src_rank) {
      for (int64_t src_row = 0; src_row < row_count; ++src_row) {
        PrimExpr count = CountLoad(op->recv_counts, op->recv_counts_range,
                                   I32(src_rank), I32(src_row), row_domain);
        PrimExpr active =
            And(rank_id_ != I32(src_rank), count > make_zero(count.dtype()));
        delta = delta + Cast(DataType::Int(32), active);
      }
    }
    return delta;
  }

  struct AlltoallvLowering {
    Stmt schedule;
    std::optional<PrimExpr> completion;
  };

  AlltoallvLowering LowerAlltoallv(const DistAlltoallvOpNode *op) const {
    bool single_signal = ValidateAlltoallv(op);
    bool row_domain = IsRowDomain(op);
    PrimExpr current_row = floordiv(op->current_core, I32(mesh_ncols_));
    PrimExpr source_index = EndpointIndex(rank_id_, current_row, row_domain);
    int64_t endpoint_count = world_size_ * (row_domain ? mesh_nrows_ : 1);
    Array<Range> dst_slot_range =
        EndpointSlot(op->dst_range, rank_id_, current_row, row_domain);
    PrimExpr dst_slot =
        MakeRegionExpr(op->dst, dst_slot_range, /*access_mask=*/2);

    Array<Stmt> schedule;
    if (single_signal) {
      schedule.push_back(Evaluate(
          Call(DataType::Handle(), dist_expect(),
               {op->signal_resource, AggregateRecvDelta(op, row_domain)})));
    }
    for (int64_t offset = 0; offset < world_size_; ++offset) {
      PrimExpr dst_rank = floormod(rank_id_ + I32(offset), I32(world_size_));
      int64_t row_count = row_domain ? mesh_nrows_ : 1;
      for (int64_t dst_row_value = 0; dst_row_value < row_count;
           ++dst_row_value) {
        PrimExpr dst_row = row_domain ? I32(dst_row_value) : current_row;
        PrimExpr destination_index =
            EndpointIndex(dst_rank, dst_row, row_domain);
        PrimExpr generation_index =
            source_index * I32(endpoint_count) + destination_index;
        PrimExpr count = CountLoad(op->send_counts, op->send_counts_range,
                                   dst_rank, dst_row, row_domain);
        Array<Range> src_slot_range =
            EndpointSlot(op->src_range, dst_rank, dst_row, row_domain);
        PrimExpr src =
            MakeRegionExpr(op->src, src_slot_range, /*access_mask=*/1);
        PrimExpr active = count > make_zero(count.dtype());
        PrimExpr signal =
            single_signal ? Call(DataType::Handle(), dist_signal_route(),
                                 {op->signal_resource, active})
                          : Call(DataType::Handle(), dist_signal_group_route(),
                                 {op->signal_resource, source_index,
                                  generation_index, active});
        schedule.push_back(Evaluate(
            Call(DataType::Handle(), DistPutOp::Get(),
                 {src, dst_slot, dst_rank, dst_row, signal, op->current_core},
                 CompilerRegionAnnotations())));
      }
    }
    if (single_signal) {
      return {SeqStmt::Flatten(schedule), std::nullopt};
    }
    PrimExpr recv_counts = MakeRegionExpr(
        op->recv_counts, op->recv_counts_range, /*access_mask=*/1);
    PrimExpr completion =
        Call(DataType::Handle(), dist_completion(),
             {op->signal_resource, recv_counts, StringImm(op->domain)});
    return {SeqStmt::Flatten(schedule), completion};
  }

  Stmt VisitStmt_(const LetStmtNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call && call->op.same_as(DistAlltoallOp::Get())) {
      DistAlltoallOp alltoall(call->args, call->annotations);
      AlltoallLowering lowered = LowerAlltoall(alltoall.operator->());
      ICHECK(lowered.completion)
          << "Single-signal T.dist.all_to_all must be used as a statement";
      ValidateLoopCompletion(op);
      Stmt body = VisitStmt(op->body);
      Stmt completion =
          LetStmt(op->var, lowered.completion.value(), body, op->span);
      return SeqStmt::Flatten(Array<Stmt>{lowered.schedule, completion});
    }
    if (call && call->op.same_as(DistAllgatherOp::Get())) {
      DistAllgatherOp allgather(call->args, call->annotations);
      AllgatherLowering lowered = LowerAllgather(allgather.operator->());
      ICHECK(lowered.completion)
          << "SignalList T.dist.all_gather must produce a completion";
      ValidateLoopCompletion(op);
      Stmt body = VisitStmt(op->body);
      Stmt completion =
          LetStmt(op->var, lowered.completion.value(), body, op->span);
      return SeqStmt::Flatten(Array<Stmt>{lowered.schedule, completion});
    }
    if (call && call->op.same_as(DistAlltoallvOp::Get())) {
      DistAlltoallvOp alltoallv(call->args, call->annotations);
      AlltoallvLowering lowered = LowerAlltoallv(alltoallv.operator->());
      ICHECK(lowered.completion)
          << "Single-signal T.dist.all_to_allv must be used as a statement";
      ValidateLoopCompletion(op);
      Stmt body = VisitStmt(op->body);
      Stmt completion =
          LetStmt(op->var, lowered.completion.value(), body, op->span);
      return SeqStmt::Flatten(Array<Stmt>{lowered.schedule, completion});
    }
    if (call && call->op.same_as(dist_signal_group_decl())) {
      ICHECK_EQ(call->args.size(), 4U);
      int64_t count = RequireIntImm(call->args[2], "signal-group count");
      ICHECK_GT(count, 0) << "T.dist.signals group cannot be empty";
      std::string kind = RequireStringImm(call->args[0], "signal-group kind");
      active_signal_group_sizes_.emplace(op->var.get(), count);
      active_signal_group_kinds_.emplace(op->var.get(), std::move(kind));
      Stmt body = VisitStmt(op->body);
      active_signal_group_sizes_.erase(op->var.get());
      active_signal_group_kinds_.erase(op->var.get());
      return LetStmt(op->var, op->value, body, op->span);
    }
    bool is_signal = call && (call->op.same_as(dist_signal_decl()) ||
                              call->op.same_as(dist_signal()));
    if (!is_signal) {
      Map<Var, PrimExpr> previous = offset_bindings_;
      if (SideEffect(op->value) <= CallEffectKind::kPure)
        offset_bindings_.Set(op->var, ResolveOffset(op->value));
      Stmt result = StmtExprMutator::VisitStmt_(op);
      offset_bindings_ = std::move(previous);
      return result;
    }

    ICHECK(call->args.size() == 2U || call->args.size() == 3U);
    std::string kind = RequireStringImm(call->args[0], "signal kind");
    active_signal_kinds_.emplace(op->var.get(), std::move(kind));
    Stmt body = VisitStmt(op->body);
    active_signal_kinds_.erase(op->var.get());
    return LetStmt(op->var, op->value, body, op->span);
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    alloc_buffer_stack_.emplace_back();
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
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

  Stmt VisitStmt_(const ForNode *op) final {
    PrimExpr min = ResolveOffset(op->min);
    PrimExpr extent = ResolveOffset(op->extent);
    const auto *count = extent.as<IntImmNode>();
    bool has_exit = false;
    PostOrderVisit(op->body, [&](const ObjectRef &node) {
      if (const auto *call = node.as<CallNode>())
        has_exit |= call->op.same_as(builtin::ret()) ||
                    call->op.same_as(builtin::break_loop()) ||
                    call->op.same_as(builtin::continue_loop());
    });
    bool supported = op->kind == ForKind::kSerial && min.as<IntImmNode>() &&
                     count && count->value >= 0 && !has_exit;
    offset_loops_.push_back({op->loop_var, extent, supported});
    if (supported && count->value > 0)
      offset_analyzer_.Bind(op->loop_var, Range::FromMinExtent(min, extent));
    Stmt result = StmtExprMutator::VisitStmt_(op);
    offset_loops_.pop_back();
    return result;
  }

  Stmt VisitStmt_(const WhileNode *op) final {
    ++offset_while_depth_;
    Stmt result = StmtExprMutator::VisitStmt_(op);
    --offset_while_depth_;
    return result;
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    PrimExpr condition = ResolveOffset(op->condition);
    offset_conditions_.push_back(condition);
    Stmt result = StmtExprMutator::VisitStmt_(op);
    offset_conditions_.pop_back();
    return result;
  }

  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    offset_conditions_.push_back(ResolveOffset(op->predicate));
    Stmt result = StmtExprMutator::VisitStmt_(op);
    offset_conditions_.pop_back();
    return result;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (!call || !call->op.same_as(DistAllgatherOp::Get())) {
      if (call && call->op.same_as(DistAlltoallOp::Get())) {
        DistAlltoallOp alltoall(call->args, call->annotations);
        AlltoallLowering lowered = LowerAlltoall(alltoall.operator->());
        ICHECK(!lowered.completion)
            << "SignalList T.dist.all_to_all must bind its completion result";
        return lowered.schedule;
      }
      if (call && call->op.same_as(DistAllreduceOp::Get())) {
        DistAllreduceOp allreduce(call->args, call->annotations);
        return LowerAllreduce(allreduce.operator->());
      }
      if (call && call->op.same_as(DistAlltoallvOp::Get())) {
        DistAlltoallvOp alltoallv(call->args, call->annotations);
        AlltoallvLowering lowered = LowerAlltoallv(alltoallv.operator->());
        ICHECK(!lowered.completion)
            << "SignalList T.dist.all_to_allv must produce a completion";
        return lowered.schedule;
      }
      if (call && call->op.same_as(dist_barrier())) {
        return LowerBarrier(call);
      }
      if (call && call->op.same_as(dist_barrier_arrive())) {
        return LowerBarrierArrive(call, "T.dist.barrier_arrive");
      }
      return StmtExprMutator::VisitStmt_(op);
    }
    DistAllgatherOp allgather(call->args, call->annotations);
    AllgatherLowering lowered = LowerAllgather(allgather.operator->());
    ICHECK(!lowered.completion)
        << "SignalList T.dist.all_gather must bind its completion result";
    return lowered.schedule;
  }

  int64_t world_size_;
  Var rank_id_;
  int mesh_nrows_;
  int mesh_ncols_;
  std::unordered_map<const VarNode *, std::string> active_signal_kinds_;
  std::unordered_map<const VarNode *, int64_t> active_signal_group_sizes_;
  std::unordered_map<const VarNode *, std::string> active_signal_group_kinds_;
  std::vector<Array<Buffer>> alloc_buffer_stack_;
  int allreduce_counter_{0};
  mutable arith::Analyzer offset_analyzer_;
  Map<Var, PrimExpr> offset_bindings_;
  std::vector<OffsetLoop> offset_loops_;
  std::vector<PrimExpr> offset_conditions_;
  int offset_while_depth_{0};
};

PrimFunc Run(PrimFunc func) {
  if (!DistCollectiveDetector().Detect(func->body)) {
    return func;
  }

  auto world_size = func->GetAttr<Integer>(kDistWorldSizeAttr);
  ICHECK(world_size && world_size.value()->value > 1)
      << "world_size=1 disables Rank communication; "
         "T.dist collectives require world_size > 1";
  auto context = GetDistPassContext(func);
  ICHECK(context);
  DistAllreduceLoopValidator().Validate(func->body);
  DistCollectiveSignalValidator().Validate(func->body);
  auto mesh = GetSunmmioMeshConfig(context.value().target);
  DistCollectiveLowerer lowerer(context.value().world_size,
                                context.value().rank_id, mesh.nrow, mesh.ncol);
  Stmt body = lowerer(func->body);
  if (!body.same_as(func->body)) {
    func.CopyOnWrite()->body = std::move(body);
  }
  return func;
}

} // namespace

tvm::transform::Pass LowerDistCollectives() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) {
    return Run(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerDistCollectives", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerDistCollectives",
                        LowerDistCollectives);
}

} // namespace tl
} // namespace tvm
