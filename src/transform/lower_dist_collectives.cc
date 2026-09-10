/*!
 * \file tl/transform/lower_dist_collectives.cc
 * \brief Lower high-level Rank collectives to logical P2P operations.
 */

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
        call->op.same_as(dist_barrier())) {
      found_ = true;
      return;
    }
    StmtExprVisitor::VisitExpr_(call);
  }

  bool found_{false};
};

class DistCollectiveLowerer : public StmtExprMutator {
public:
  DistCollectiveLowerer(int64_t world_size, Var rank_id, int mesh_nrows,
                        int mesh_ncols)
      : world_size_(world_size), rank_id_(std::move(rank_id)),
        mesh_nrows_(mesh_nrows), mesh_ncols_(mesh_ncols) {}

private:
  PrimExpr I32(int64_t value) const { return IntImm(DataType::Int(32), value); }

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
      int64_t index = RequireIntImm(ref->args[1], "signal-group index");
      auto size = active_signal_group_sizes_.find(signal_var);
      ICHECK(size != active_signal_group_sizes_.end());
      ICHECK_GE(index, 0);
      ICHECK_LT(index, size->second);
      kind_ptr = &active->second;
      ICHECK(*kind_ptr != kAutoSignalKind)
          << op_name << " requires a single automatic signal or an explicitly "
          << "INC SignalList; automatic SignalList planning selects VALUE or "
          << "MEMORY";
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

  void ValidateAllgather(const DistAllgatherOpNode *op) const {
    ICHECK(op->src->dtype == op->dst->dtype)
        << "T.dist.all_gather source and destination dtypes must match";
    ICHECK(!op->src->data.same_as(op->dst->data))
        << "T.dist.all_gather does not support overlapping source and "
           "destination storage";
    ICHECK(op->current_core.dtype().is_int())
        << "T.dist.all_gather current_core must have integer dtype";

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

    ValidateSignal(op->signal, "T.dist.all_gather");
  }

  bool IsRowDomain(const DistAlltoallOpNode *op) const {
    ICHECK(op->domain == "rank" || op->domain == "row")
        << "T.dist.all_to_all domain must be 'rank' or 'row', got "
        << op->domain;
    return op->domain == "row";
  }

  void ValidateAlltoall(const DistAlltoallOpNode *op) const {
    ICHECK(op->src->dtype == op->dst->dtype)
        << "T.dist.all_to_all source and destination dtypes must match";
    ICHECK(!op->src->data.same_as(op->dst->data))
        << "T.dist.all_to_all does not support overlapping source and "
           "destination storage";
    ICHECK(op->current_core.dtype().is_int())
        << "T.dist.all_to_all current_core must have integer dtype";

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
    ValidateSignal(op->signal, "T.dist.all_to_all");
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
          << "T.dist.all_to_allv signal resource must reference "
             "T.dist.signal or T.dist.signals";
      return true;
    }
    const auto *var = resource.as<VarNode>();
    ICHECK(var) << "T.dist.all_to_allv signal resource must be a handle";
    if (active_signal_kinds_.count(var)) {
      return true;
    }
    ICHECK(active_signal_group_sizes_.count(var))
        << "T.dist.all_to_allv cannot find its signal declaration";
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

  Stmt LowerAllgather(const DistAllgatherOpNode *op) const {
    ValidateAllgather(op);
    PrimExpr src = MakeRegionExpr(op->src, op->src_range, /*access_mask=*/1);
    PrimExpr dst_slot = MakeRegionExpr(op->dst, DestinationSlot(op, rank_id_),
                                       /*access_mask=*/2);

    Array<Stmt> schedule;
    schedule.push_back(
        Evaluate(Call(DataType::Handle(), Copy::Get(), {src, dst_slot})));

    PrimExpr dst_row = floordiv(op->current_core, I32(mesh_ncols_));
    for (int64_t offset = 1; offset < world_size_; ++offset) {
      PrimExpr dst_rank = floormod(rank_id_ + I32(offset), I32(world_size_));
      schedule.push_back(Evaluate(Call(
          DataType::Handle(), DistPutOp::Get(),
          {src, dst_slot, dst_rank, dst_row, op->signal, op->current_core})));
    }

    schedule.push_back(
        Evaluate(Call(DataType::Handle(), dist_wait_send(), {})));
    return SeqStmt::Flatten(schedule);
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
                                        op->signal, op->current_core})));
    }

    PrimExpr gather_region =
        MakeRegionExpr(gather->buffer, gather->region, /*access_mask=*/1);
    schedule.push_back(Evaluate(
        Call(DataType::Handle(), DistWaitSignalOp::Get(),
             {op->signal, MakeRegionExpr(gather->buffer, gather->region,
                                         /*access_mask=*/2)})));
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

  Stmt LowerAlltoall(const DistAlltoallOpNode *op) const {
    ValidateAlltoall(op);
    bool row_domain = IsRowDomain(op);
    PrimExpr current_row = floordiv(op->current_core, I32(mesh_ncols_));
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
        schedule.push_back(Evaluate(Call(
            DataType::Handle(), DistPutOp::Get(),
            {src, dst_slot, dst_rank, dst_row, op->signal, op->current_core})));
      }
    }

    schedule.push_back(
        Evaluate(Call(DataType::Handle(), dist_wait_send(), {})));
    return SeqStmt::Flatten(schedule);
  }

  Stmt LowerBarrier(const CallNode *call) const {
    ICHECK_EQ(call->args.size(), 2U);
    ValidateSignal(call->args[0], "T.dist.barrier");
    ICHECK(call->args[1].dtype().is_int())
        << "T.dist.barrier current_core must have integer dtype";

    Array<Stmt> schedule{
        Evaluate(Call(DataType::Handle(), dist_wait_send(), {}))};
    schedule.push_back(Evaluate(Call(DataType::Handle(), dist_expect(),
                                     {call->args[0], I32(world_size_ - 1)})));
    for (int64_t offset = 1; offset < world_size_; ++offset) {
      PrimExpr dst_rank = floormod(rank_id_ + I32(offset), I32(world_size_));
      schedule.push_back(
          Evaluate(Call(DataType::Handle(), dist_signal_put(),
                        {call->args[0], dst_rank, call->args[1]})));
    }
    schedule.push_back(Evaluate(
        Call(DataType::Handle(), dist_wait_barrier(), {call->args[0]})));
    return SeqStmt::Flatten(schedule);
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
            row_domain ? current_row * I32(endpoint_count) + destination_index
                       : destination_index;
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
        schedule.push_back(Evaluate(Call(
            DataType::Handle(), DistPutOp::Get(),
            {src, dst_slot, dst_rank, dst_row, signal, op->current_core})));
      }
    }
    // Temporary source-lifetime barrier; a dedicated pass will insert sender
    // waits.
    schedule.push_back(
        Evaluate(Call(DataType::Handle(), dist_wait_send(), {})));

    if (single_signal) {
      return {SeqStmt::Flatten(schedule), std::nullopt};
    }
    PrimExpr recv_counts = MakeRegionExpr(
        op->recv_counts, op->recv_counts_range, /*access_mask=*/1);
    PrimExpr dst = MakeRegionExpr(op->dst, op->dst_range, /*access_mask=*/2);
    PrimExpr completion =
        Call(DataType::Handle(), dist_completion(),
             {op->signal_resource, recv_counts, dst, StringImm(op->domain)});
    return {SeqStmt::Flatten(schedule), completion};
  }

  Stmt VisitStmt_(const LetStmtNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call && call->op.same_as(DistAlltoallvOp::Get())) {
      DistAlltoallvOp alltoallv(call->args, call->annotations);
      AlltoallvLowering lowered = LowerAlltoallv(alltoallv.operator->());
      ICHECK(lowered.completion)
          << "Single-signal T.dist.all_to_allv must be used as a statement";
      Stmt body = VisitStmt(op->body);
      Stmt completion =
          LetStmt(op->var, lowered.completion.value(), body, op->span);
      return SeqStmt::Flatten(Array<Stmt>{lowered.schedule, completion});
    }
    if (call && call->op.same_as(dist_signal_group_decl())) {
      ICHECK_EQ(call->args.size(), 3U);
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
      return StmtExprMutator::VisitStmt_(op);
    }

    ICHECK_EQ(call->args.size(), 2U);
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

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (!call || !call->op.same_as(DistAllgatherOp::Get())) {
      if (call && call->op.same_as(DistAlltoallOp::Get())) {
        DistAlltoallOp alltoall(call->args, call->annotations);
        return LowerAlltoall(alltoall.operator->());
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
      return StmtExprMutator::VisitStmt_(op);
    }
    DistAllgatherOp allgather(call->args, call->annotations);
    return LowerAllgather(allgather.operator->());
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
