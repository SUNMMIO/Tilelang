/*!
 * \file cost_model.cc
 * \brief Implementation of CostModel.
 */

#include "cost_model.h"
#include "../../layout/cute_layout.h"
#include "../../op/builtin.h"
#include "../../op/comm.h"
#include "../../op/utils.h"
#include "../sunmmio_utils.h"
#include "sunmmio_async_op_info.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

struct ODMACost {
  enum Value : int {
    kStartupLatency = 100,
    kDmaBytesPerCycle = 1024,
    kLayoutTransformStartupLatency = 10,
    kLayoutTransformBytesPerCycle = 512,
    kTransposeStartupLatency = 10,
    kTransposeBytesPerCycle = 512,
    kBroadcastHorizontalBytesPerCycle = 1024,
    kBroadcastVerticalBytesPerCycle = 2048,
  };
};

struct VectorCoreCost {
  enum Value : int {
    kMulFlops = 1,
    kAddSubFlops = 1,
    kMinMaxFlops = 1,
    kCastFlops = 1,
    kExp2Flops = 3,
    kBitwiseAndFlops = 1,
    kCompareFlops = 1,
    kFirstLoadFlops = 5,
    kRepeatedLoadFlops = 1,
    kBufferStoreBaseFlops = 5,
    kConstantMaterializeFlops = 1,
    kReduceMinMaxFlops = 1,
    kReduceSumFlops = 3,
    kFlopsPerCycle = 250,
  };
};

constexpr int kMXFP8CustomTypeCode = DataType::kCustomBegin;
constexpr int kMXFP4CustomTypeCode = DataType::kCustomBegin + 1;

bool IsMXFP8DType(DataType dtype) {
  return dtype.code() == kMXFP8CustomTypeCode && dtype.bits() == 8 &&
         dtype.lanes() == 1;
}

bool IsMXFP4DType(DataType dtype) {
  return dtype.code() == kMXFP4CustomTypeCode && dtype.bits() == 4 &&
         dtype.lanes() == 1;
}

bool IsMXDType(DataType dtype) {
  return IsMXFP8DType(dtype) || IsMXFP4DType(dtype);
}

bool IsSupportedA4ETensorCorePair(DataType activation, DataType weight) {
  if (activation.is_bfloat16()) {
    return weight.is_bfloat16() || IsMXDType(weight);
  }
  return IsMXDType(activation) && IsMXDType(weight);
}

static bool ContainsExpr(const tvm::ffi::Array<PrimExpr> &exprs,
                         const PrimExpr &expr) {
  for (const PrimExpr &it : exprs) {
    if (ExprDeepEqual()(it, expr)) {
      return true;
    }
  }
  return false;
}

static void PushUniqueExpr(tvm::ffi::Array<PrimExpr> *exprs,
                           const PrimExpr &expr) {
  if (!ContainsExpr(*exprs, expr)) {
    exprs->push_back(expr);
  }
}

static bool IsImmediateScalar(const PrimExpr &expr) {
  return expr.as<IntImmNode>() || expr.as<FloatImmNode>();
}

} // namespace

/**
 * \brief Lightweight expression analyzer for VectorCore flop estimation.
 *
 * This helper only extracts local information from one expression tree:
 * - flop contribution of the expression itself
 * - constants that may need materialization
 * - reusable address expressions collected from BufferLoad indices
 *
 * It does not implement global common-subexpression reuse. Any reuse policy is
 * applied later at stmt level in a deliberately conservative way.
 */
class SunmmioExprAnalyzerV2 : public StmtExprVisitor {
public:
  SunmmioExprAnalyzerV2() {}

  void Analyze(const PrimExpr &expr) {
    load_times = 0;
    flops_ = 0;
    args_.clear();
    constants_.clear();
    StmtExprVisitor::VisitExpr(expr);
  }

  /** Reusable address expressions charged later through `once`. */
  tvm::ffi::Array<PrimExpr> args_;
  tvm::ffi::Array<PrimExpr> constants_;
  int load_times = 0;
  float flops_ = 0;

private:
  void VisitExpr_(const MulNode *op) final {
    if (IsImmediateScalar(op->a) && IsImmediateScalar(op->b)) {
      return;
    }
    flops_ += VectorCoreCost::kMulFlops;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const SubNode *op) final {
    flops_ += VectorCoreCost::kAddSubFlops;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const AddNode *op) final {
    flops_ += VectorCoreCost::kAddSubFlops;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const MaxNode *op) final {
    flops_ += VectorCoreCost::kMinMaxFlops;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const MinNode *op) final {
    flops_ += VectorCoreCost::kMinMaxFlops;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const CastNode *op) final {
    flops_ += VectorCoreCost::kCastFlops;
    StmtExprVisitor::VisitExpr(op->value);
  }

  void VisitExpr_(const IntImmNode *op) final {
    PushUniqueExpr(&constants_, tvm::ffi::GetRef<PrimExpr>(op));
  }

  void VisitExpr_(const FloatImmNode *op) final {
    PushUniqueExpr(&constants_, tvm::ffi::GetRef<PrimExpr>(op));
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(Op::Get("tir.exp2"))) {
      flops_ += VectorCoreCost::kExp2Flops;
      StmtExprVisitor::VisitExpr(op->args[0]);
    } else if (op->op.same_as(Op::Get("tir.if_then_else"))) {
      StmtExprVisitor::VisitExpr(op->args[0]);
      StmtExprVisitor::VisitExpr(op->args[1]);
      StmtExprVisitor::VisitExpr(op->args[2]);
    } else if (op->op.same_as(Op::Get("tl.infinity"))) {
      PushUniqueExpr(
          &constants_,
          FloatImm(op->dtype, std::numeric_limits<float>::infinity()));
    } else if (op->op.same_as(Op::Get("tir.bitwise_and"))) {
      flops_ += VectorCoreCost::kBitwiseAndFlops;
      StmtExprVisitor::VisitExpr(op->args[0]);
      StmtExprVisitor::VisitExpr(op->args[1]);
    } else {
      ICHECK(0) << "Op " << op->op << " not supported now.";
    }
  }

  void VisitExpr_(const LENode *op) final {
    flops_ += VectorCoreCost::kCompareFlops;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const LTNode *op) final {
    flops_ += VectorCoreCost::kCompareFlops;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const GENode *op) final {
    flops_ += VectorCoreCost::kCompareFlops;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const GTNode *op) final {
    flops_ += VectorCoreCost::kCompareFlops;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const EQNode *op) final {
    flops_ += VectorCoreCost::kCompareFlops;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const NENode *op) final {
    flops_ += VectorCoreCost::kCompareFlops;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (load_times == 0) {
      load_times++;
      flops_ += VectorCoreCost::kFirstLoadFlops;
    } else {
      load_times++;
      flops_ += VectorCoreCost::kRepeatedLoadFlops;
    }
    for (auto arg : op->indices) {
      PushUniqueExpr(&args_, arg);
    }
  }
};

float CostModel::EstimateTensorCoreDelayHeuristic(const tir::Stmt &stmt) {
  auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
  const auto *block = stmt.as<BlockRealizeNode>();
  ICHECK(block) << "TensorCore command must be a BlockRealizeNode: " << stmt;

  auto body = block->block->body;
  const auto *eval = body.as<EvaluateNode>();
  ICHECK(eval) << "TensorCore block body must be an EvaluateNode: " << stmt;

  const auto *call = eval->value.as<CallNode>();
  ICHECK(call) << "TensorCore evaluate body must be a CallNode: " << stmt;
  ICHECK(call->op.same_as(Op::Get("tl.mma_sunmmio")))
      << "TensorCore command must call tl.mma_sunmmio: " << stmt;

  BufferRegion a_region = NormalizeToBufferRegion(call->args[0]);
  BufferRegion b_region = NormalizeToBufferRegion(call->args[1]);
  BufferRegion c_region = NormalizeToBufferRegion(call->args[2]);
  DataType input_dtype = a_region->buffer->dtype;
  DataType rhs_dtype = b_region->buffer->dtype;
  DataType output_dtype = c_region->buffer->dtype;

  ICHECK(IsSupportedA4ETensorCorePair(input_dtype, rhs_dtype))
      << "Unsupported TensorCore input dtypes in current cost model: "
      << input_dtype << " x " << rhs_dtype << "; " << stmt;

  auto static_extent = [&](const BufferRegion &region, int axis,
                           const char *operand) {
    int rank = static_cast<int>(region->region.size());
    ICHECK(rank == 2 || rank == 3) << "TensorCore " << operand
                                   << " region must have rank 2 or 3: " << stmt;
    int normalized_axis = axis < 0 ? rank + axis : axis;
    const auto *extent =
        region->region[normalized_axis]->extent.as<IntImmNode>();
    ICHECK(extent) << "TensorCore " << operand
                   << " region extent must be IntImm on axis "
                   << normalized_axis << ": " << stmt;
    return static_cast<int>(extent->value);
  };

  int row_size = static_extent(c_region, -2, "output");
  int col_size = static_extent(c_region, -1, "output");
  int acc_size = static_extent(a_region, -1, "activation");
  int batch_size = 1;
  for (const auto &[region, operand] :
       {std::pair{a_region, "activation"}, std::pair{b_region, "weight"},
        std::pair{c_region, "output"}}) {
    if (region->region.size() != 3)
      continue;
    int extent = static_extent(region, 0, operand);
    if (extent <= 1)
      continue;
    ICHECK(batch_size == 1 || batch_size == extent)
        << "TensorCore batch extents must match: " << stmt;
    batch_size = extent;
  }

  /**
   * The effective inner-loop gap is bounded by the activation path's reduction
   * throughput and a minimum block count implied by the tile shape. The
   * accumulator dtype contributes a small fixed latency difference between
   * bf16 accumulation and fp32 accumulation.
   */
  int min_blk_num = (row_size <= 16 && col_size <= 32) ? 1 : 4;
  int reduction_elems_per_cycle = IsMXDType(input_dtype) ? 128 : 32;
  int gap =
      std::max(ceil_div(acc_size, reduction_elems_per_cycle), min_blk_num);
  int st_dtype_delay = 0;
  if (output_dtype.is_bfloat16()) {
    st_dtype_delay = 8;
  } else if (output_dtype.is_float() && output_dtype.bits() == 32) {
    st_dtype_delay = 7;
  } else {
    ICHECK(0) << "Unsupported TensorCore accumulator dtype in cost model: "
              << output_dtype;
  }

  /**
   * The final TensorCore delay is modeled as:
   * - fixed frontend/setup terms (11 + 5 + 5)
   * - dtype-dependent accumulator term
   * - tile compute term scaled by ceil(m / 16) * ceil(n / 32) * gap
   * - fixed tail latency (70)
   */
  float delay = 11 + 5 + 5 + st_dtype_delay +
                ceil_div(row_size, 16) * ceil_div(col_size, 32) * gap + 70;
  return batch_size * delay;
}

float CostModel::EstimateODMADelayHeuristic(const tir::Stmt &stmt) {
  auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
  const auto *eval = stmt.as<EvaluateNode>();
  ICHECK(eval) << "ODMA command must be an EvaluateNode: " << stmt;

  const auto *call = eval->value.as<CallNode>();
  ICHECK(call) << "ODMA evaluate body must be a CallNode: " << stmt;

  if (call->op.same_as(Op::Get("tl.dma_copy"))) {
    BufferRegion src_region = NormalizeToBufferRegion(call->args[0]);
    int total_elements = 1;
    for (const Range &range : src_region->region) {
      ICHECK(range->extent.as<IntImmNode>())
          << "ODMA dma_copy source extents must be IntImm: " << stmt;
      total_elements *= range->extent.as<IntImmNode>()->value;
    }
    int bytes_per_element = ceil_div(src_region->buffer->dtype.bits() *
                                         src_region->buffer->dtype.lanes(),
                                     8);
    int total_bytes = total_elements * bytes_per_element;
    /** Model dma_copy as a fixed startup cost plus 1 KB/cycle payload transfer.
     */
    return ODMACost::kStartupLatency +
           ceil_div(total_bytes, ODMACost::kDmaBytesPerCycle);
  } else if (call->op.same_as(Op::Get("tl.sunmmio_layout_transform"))) {
    BufferRegion src_region = NormalizeToBufferRegion(call->args[0]);
    int total_elements = 1;
    for (const Range &range : src_region->region) {
      ICHECK(range->extent.as<IntImmNode>())
          << "ODMA sunmmio_layout_transform source extents must be IntImm: "
          << stmt;
      total_elements *= range->extent.as<IntImmNode>()->value;
    }
    int bytes_per_element = ceil_div(src_region->buffer->dtype.bits() *
                                         src_region->buffer->dtype.lanes(),
                                     8);
    int total_bytes = total_elements * bytes_per_element;
    /** Model layout transform as a lighter startup plus 512 B/cycle payload
     * transfer. */
    return ODMACost::kLayoutTransformStartupLatency +
           ceil_div(total_bytes, ODMACost::kLayoutTransformBytesPerCycle);
  } else if (call->op.same_as(Op::Get("tl.sunmmio_transpose"))) {
    BufferRegion src_region = NormalizeToBufferRegion(call->args[0]);
    int total_elements = 1;
    for (const Range &range : src_region->region) {
      ICHECK(range->extent.as<IntImmNode>())
          << "ODMA sunmmio_transpose source extents must be IntImm: " << stmt;
      total_elements *= range->extent.as<IntImmNode>()->value;
    }
    int bytes_per_element = ceil_div(src_region->buffer->dtype.bits() *
                                         src_region->buffer->dtype.lanes(),
                                     8);
    int total_bytes = total_elements * bytes_per_element;
    return ODMACost::kTransposeStartupLatency +
           ceil_div(total_bytes, ODMACost::kTransposeBytesPerCycle);
  } else if (call->op.same_as(Op::Get("tl.broadcast_"))) {
    ICHECK(call->args[kBroadcastArgDirection].as<IntImmNode>())
        << "ODMA broadcast direction must be IntImm: " << stmt;

    BufferRegion src_region =
        NormalizeToBufferRegion(call->args[kBroadcastArgSrc]);
    int broadcast_elements = 1;
    for (const Range &range : src_region->region) {
      ICHECK(range->extent.as<IntImmNode>())
          << "ODMA broadcast source extents must be IntImm: " << stmt;
      broadcast_elements *= range->extent.as<IntImmNode>()->value;
    }
    int direction = call->args[kBroadcastArgDirection].as<IntImmNode>()->value;
    int bytes_per_element = ceil_div(src_region->buffer->dtype.bits() *
                                         src_region->buffer->dtype.lanes(),
                                     8);
    int total_bytes = broadcast_elements * bytes_per_element;

    /**
     * Broadcast cost is modeled with a common startup cost and a direction-
     * dependent payload bandwidth:
     * - horizontal: 1 KB/cycle
     * - vertical: 2 KB/cycle
     */
    int bandwidth_bytes_per_cycle = 0;
    if (direction == 0) {
      bandwidth_bytes_per_cycle = ODMACost::kBroadcastHorizontalBytesPerCycle;
    } else if (direction == 1) {
      bandwidth_bytes_per_cycle = ODMACost::kBroadcastVerticalBytesPerCycle;
    } else {
      ICHECK(0) << "Unsupported broadcast direction in cost model: "
                << direction;
    }
    return ODMACost::kStartupLatency +
           ceil_div(total_bytes, bandwidth_bytes_per_cycle);
  }
  ICHECK(0) << "Unsupported ODMA op in cost model: " << call->op;
  return 0.0f;
}

struct VectorStmtFlopCost {
  float repeated{0.0f};
  float once{0.0f};
};

/**
 * \brief Recursive stmt-level analyzer for VectorCore flop cost.
 *
 * The current reuse policy is intentionally conservative:
 * - constants may be materialized once and reused later
 * - reusable expressions extracted from one expr are deduplicated only within
 *   that expr
 * - explicit BufferStore indices are deduplicated only within one store
 * - generic cross-stmt expression reuse is not modeled yet
 */
class VectorCoreFlopAnalyzer {
public:
  float Analyze(const Stmt &stmt) {
    VectorStmtFlopCost cost = AnalyzeStmt(stmt);
    return cost.once + cost.repeated;
  }

private:
  /**
   * \brief Analyze one whole expression and split its cost into repeated/once.
   *
   * Repeated cost models the arithmetic/load work that still belongs to the
   * expression execution itself. Once cost is reserved for values that can be
   * materialized once and reused later, including:
   * - constants that are first seen here
   * - reusable address expressions extracted from BufferLoad indices in this
   *   expression
   */
  VectorStmtFlopCost AnalyzeExpr(const PrimExpr &expr) {
    SunmmioExprAnalyzerV2 analyzer;
    analyzer.Analyze(expr);

    VectorStmtFlopCost cost;
    cost.repeated = analyzer.flops_;
    for (const PrimExpr &constant : analyzer.constants_) {
      if (!ContainsExpr(materialized_constants_, constant)) {
        materialized_constants_.push_back(constant);
        cost.once += VectorCoreCost::kConstantMaterializeFlops;
      }
    }

    tvm::ffi::Array<PrimExpr> reusable_exprs;
    for (const PrimExpr &arg : analyzer.args_) {
      PushUniqueExpr(&reusable_exprs, arg);
    }
    for (const PrimExpr &reusable_expr : reusable_exprs) {
      VectorStmtFlopCost reusable_cost = AnalyzeExpr(reusable_expr);
      cost.once += reusable_cost.once + reusable_cost.repeated;
    }
    return cost;
  }

  /**
   * \brief Estimate flop cost for one BufferStore.
   *
   * Reuse policy for the current implementation:
   * - the whole store value expression is analyzed through `AnalyzeExpr`
   * - constants are charged through `once` when first materialized
   * - locally deduplicated explicit store indices are charged through `once`
   * - ordinary expression subtrees are still charged locally through
   *   `repeated`, without cross-store common-subexpression reuse
   */
  VectorStmtFlopCost AnalyzeBufferStore(const BufferStoreNode &store) {
    VectorStmtFlopCost cost = AnalyzeExpr(store.value);
    cost.repeated += VectorCoreCost::kBufferStoreBaseFlops;

    tvm::ffi::Array<PrimExpr> index_exprs;
    for (const PrimExpr &index : store.indices) {
      PushUniqueExpr(&index_exprs, index);
    }
    for (const PrimExpr &index_expr : index_exprs) {
      VectorStmtFlopCost index_cost = AnalyzeExpr(index_expr);
      cost.once += index_cost.once + index_cost.repeated;
    }
    return cost;
  }

  static float AnalyzeEvaluate(const EvaluateNode &eval) {
    const auto *call = eval.value.as<CallNode>();
    ICHECK(call) << "VectorCore Evaluate must wrap a CallNode.";
    ICHECK(call->op == Op::Get("tl.vector_core_in_tile_reduce"))
        << "An intrinsic on VectorCore should be "
           "tl.vector_core_in_tile_reduce.";
    const auto *reduce_kind = call->args[0].as<StringImmNode>();
    ICHECK(reduce_kind)
        << "VectorCore Evaluate call must carry a StringImm reduce kind.";
    if (reduce_kind->value == "max" || reduce_kind->value == "min") {
      return VectorCoreCost::kReduceMinMaxFlops;
    }
    if (reduce_kind->value == "sum") {
      return VectorCoreCost::kReduceSumFlops;
    }
    ICHECK(0) << "Unsupported VectorCore reduce kind: " << reduce_kind->value;
    return 0.0f;
  }

  VectorStmtFlopCost AnalyzeStmt(const Stmt &stmt) {
    if (const auto *for_node = stmt.as<ForNode>()) {
      ICHECK(for_node->extent.as<IntImmNode>())
          << "VectorCore loop extent must be IntImm: " << stmt;
      VectorStmtFlopCost body_cost = AnalyzeStmt(for_node->body);
      return {body_cost.repeated * for_node->extent.as<IntImmNode>()->value,
              body_cost.once};
    }
    if (const auto *block_node = stmt.as<BlockRealizeNode>()) {
      return AnalyzeStmt(block_node->block->body);
    }
    if (const auto *seq_stmt = stmt.as<SeqStmtNode>()) {
      VectorStmtFlopCost cost;
      for (const Stmt &child : seq_stmt->seq) {
        VectorStmtFlopCost child_cost = AnalyzeStmt(child);
        cost.repeated += child_cost.repeated;
        cost.once += child_cost.once;
      }
      return cost;
    }
    if (const auto *if_stmt = stmt.as<IfThenElseNode>()) {
      // The condition is evaluated whenever the surrounding stmt executes,
      // so
      // its cost belongs to the current repeated context. Without branch
      // probability information, conservatively charge the more expensive
      // branch together with the condition.
      VectorStmtFlopCost cond_cost = AnalyzeExpr(if_stmt->condition);
      VectorStmtFlopCost then_cost = AnalyzeStmt(if_stmt->then_case);
      float branch_cost = then_cost.once + then_cost.repeated;
      if (if_stmt->else_case.defined()) {
        VectorStmtFlopCost else_cost = AnalyzeStmt(if_stmt->else_case.value());
        branch_cost =
            std::max(branch_cost, else_cost.once + else_cost.repeated);
      }
      return {cond_cost.repeated + branch_cost, cond_cost.once};
    }
    if (const auto *store = stmt.as<BufferStoreNode>()) {
      return AnalyzeBufferStore(*store);
    }
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      return {AnalyzeEvaluate(*eval), 0.0f};
    }
    ICHECK(0) << "Unsupported VectorCore stmt in cost model: " << stmt;
    return {0.0f, 0.0f};
  }

  tvm::ffi::Array<PrimExpr> materialized_constants_;
};

float CostModel::EstimateVectorCoreDelayHeuristic(const tir::Stmt &stmt) {
  auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
  /**
   * VectorCore delay is modeled from flop workload only. The recursive
   * analyzer walks both plain BufferStore loops and reduce blocks, and applies
   * only a conservative reuse policy:
   * - constants may be materialized once
   * - each expr may contribute reusable address expressions through `once`
   * - per-store index expressions are deduplicated locally
   * - generic cross-stmt expression reuse is intentionally not modeled yet
   */
  VectorCoreFlopAnalyzer analyzer;
  float flops_cost = analyzer.Analyze(stmt);
  return ceil_div(flops_cost, VectorCoreCost::kFlopsPerCycle);
}

float CostModel::EstimateDelayHeuristic(DeviceType device_type, const tir::Stmt &stmt) {
  if (device_type == DeviceType::TensorCore) {
    return EstimateTensorCoreDelayHeuristic(stmt);
  } else if (device_type == DeviceType::ODMA) {
    return EstimateODMADelayHeuristic(stmt);
  } else if (device_type == DeviceType::VectorCore) {
    return EstimateVectorCoreDelayHeuristic(stmt);
  }
  return 0.0f;
}

namespace {

constexpr const char *kHeuristicProfileName = "heuristic";
constexpr const char *kHeuristicModelVersion = "heuristic_v1";
constexpr const char *kGem5A4EProfileName = "gem5_a4e_2026";
constexpr const char *kGem5A4EModelVersion = "gem5_a4e_2026@a7ecd393";
constexpr int64_t kGem5TensorFirstResultDelay = 34;
constexpr int64_t kGem5TensorCompletionDelay = 3;
constexpr int64_t kGem5ODMALocalCompletionDelay = 3;
constexpr int64_t kGem5ODMADramToRsramCompletionExtra = 1;
constexpr int64_t kGem5ODMALinkCompletionDelay = 4;
constexpr int64_t kGem5ODMARsramToLinkAsramCompletionExtra = 1;
constexpr int64_t kGem5ODMAMultiDescriptorFetchCycles = 29;
constexpr int64_t kGem5ODMADirectCommandFrontendCycles = 2;
constexpr int64_t kGem5VLinkLastCompletionCycles = 14;
constexpr int64_t kGem5HLinkLastCompletionCycles = 40;
constexpr int64_t kA4ERegisterBits = 4096;

int64_t CeilDivI64(int64_t value, int64_t divisor) {
  ICHECK_GT(divisor, 0);
  return value / divisor + (value % divisor != 0);
}

ffi::String DTypeName(DataType dtype) {
  return runtime::DLDataTypeToString(dtype);
}

bool CheckedMulU64(uint64_t lhs, uint64_t rhs, uint64_t *out) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
    return false;
  }
  *out = lhs * rhs;
  return true;
}

bool CheckedAddU64(uint64_t lhs, uint64_t rhs, uint64_t *out) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
    return false;
  }
  *out = lhs + rhs;
  return true;
}

Optional<Layout> FindBufferLayout(const CostModelContext &context,
                                  const Buffer &buffer) {
  if (context.layout_map.count(buffer)) {
    return context.layout_map[buffer];
  }
  if (context.global_layout_map.count(buffer)) {
    return context.global_layout_map[buffer];
  }
  return Optional<Layout>();
}

enum class A4ETransformLayoutKind {
  kUnknown,
  kRowMajor,
  kZZ,
  kMXRowMajor,
  kMXZZ,
};

enum class A4ETransformKind {
  kUnknown,
  kBlockwiseToRowmajor,
  kRowmajorToBlockwise,
  kBlockscaleBlockwiseToRowmajor,
  kBlockscaleRowmajorToBlockwise,
  kRowmajorPadding,
  kRowmajorUnpadding,
};

const char *TransformKindName(A4ETransformKind kind) {
  switch (kind) {
  case A4ETransformKind::kBlockwiseToRowmajor:
    return "blockwise_to_rowmajor";
  case A4ETransformKind::kRowmajorToBlockwise:
    return "rowmajor_to_blockwise";
  case A4ETransformKind::kBlockscaleBlockwiseToRowmajor:
    return "blockscale_blockwise_to_rowmajor";
  case A4ETransformKind::kBlockscaleRowmajorToBlockwise:
    return "blockscale_rowmajor_to_blockwise";
  case A4ETransformKind::kRowmajorPadding:
    return "rowmajor_padding";
  case A4ETransformKind::kRowmajorUnpadding:
    return "rowmajor_unpadding";
  default:
    return "unknown";
  }
}

bool IsDenseRowMajorLike(const Layout &layout) {
  const auto *cute = layout.as<CuteLayoutNode>();
  if (!cute) {
    return false;
  }
  Array<Integer> levels = cute->GetDimLevels();
  for (const Integer &level : levels) {
    if (level.IntValue() != 1) {
      return false;
    }
  }
  Array<PrimExpr> covered = cute->GetCoveredShape();
  Array<PrimExpr> strides = cute->GetModeStride();
  if (covered.size() != strides.size() || covered.empty()) {
    return false;
  }
  arith::Analyzer analyzer;
  PrimExpr expected = Integer(1);
  for (int i = static_cast<int>(covered.size()) - 1; i >= 0; --i) {
    if (!analyzer.CanProveEqual(strides[i], expected)) {
      return false;
    }
    expected = analyzer.Simplify(expected * covered[i]);
  }
  return true;
}

A4ETransformLayoutKind ClassifyTransformLayout(const Layout &layout,
                                               DataType dtype) {
  if (sunmmio::IsMXDType(dtype)) {
    auto analysis = sunmmio::AnalyzeMXLayout(layout, dtype);
    if (!analysis.has_value()) {
      return A4ETransformLayoutKind::kUnknown;
    }
    if (analysis->kind == sunmmio::MXLayoutKind::kRowMajor) {
      return A4ETransformLayoutKind::kMXRowMajor;
    }
    if (analysis->kind == sunmmio::MXLayoutKind::kMXZZ) {
      return A4ETransformLayoutKind::kMXZZ;
    }
    return A4ETransformLayoutKind::kUnknown;
  }
  if (IsDenseRowMajorLike(layout)) {
    return A4ETransformLayoutKind::kRowMajor;
  }
  if (sunmmio::IsZZLike(layout)) {
    auto block = sunmmio::GetZZBlockShape(layout);
    if (block.has_value() && block->height == 32 && block->width == 32) {
      return A4ETransformLayoutKind::kZZ;
    }
  }
  return A4ETransformLayoutKind::kUnknown;
}

bool GetTransformPhysicalShape(const BufferRegion &region, const Layout &layout,
                               std::vector<uint64_t> *shape,
                               std::string *reason) {
  const auto *cute = layout.as<CuteLayoutNode>();
  if (!cute || cute->GetLogicalShape().size() != region->region.size()) {
    *reason = "layout_region_rank_mismatch";
    return false;
  }
  Array<PrimExpr> logical = cute->GetLogicalShape();
  Array<PrimExpr> covered = cute->GetCoveredShape();
  for (size_t i = 0; i < region->region.size(); ++i) {
    const auto *extent = region->region[i]->extent.as<IntImmNode>();
    const auto *logical_dim = logical[i].as<IntImmNode>();
    const auto *covered_dim = covered[i].as<IntImmNode>();
    if (!extent || !logical_dim || !covered_dim) {
      *reason = "dynamic_transform_shape";
      return false;
    }
    if (extent->value <= 0) {
      *reason = "non_positive_transform_shape";
      return false;
    }
    if (extent->value == 1) {
      continue;
    }
    int64_t physical = extent->value == logical_dim->value ? covered_dim->value
                                                           : extent->value;
    if (physical <= 0) {
      *reason = "non_positive_transform_shape";
      return false;
    }
    shape->push_back(static_cast<uint64_t>(physical));
  }
  if (shape->size() != 2) {
    *reason = "transform_requires_two_tiled_dims";
    return false;
  }
  return true;
}

struct A4ETransformPlan {
  A4ETransformKind kind{A4ETransformKind::kUnknown};
  uint64_t rows{0};
  uint64_t cols{0};
  uint64_t descriptors{0};
  uint64_t packets{0};
  uint64_t issue_finish{0};
  uint64_t latency{0};
  uint64_t descriptor_fetch_cycles{0};
};

bool BuildA4ETransformPlan(const SunmmioAsyncOpInfo &info,
                           const CostModelContext &context,
                           A4ETransformPlan *plan, std::string *reason) {
  Optional<Layout> src_layout = FindBufferLayout(context, info.src->buffer);
  Optional<Layout> dst_layout = FindBufferLayout(context, info.dst->buffer);
  if (!src_layout.defined() || !dst_layout.defined()) {
    *reason = "missing_layout_or_descriptor_mapping";
    return false;
  }
  if ((info.src->buffer.scope() != "shared.rsram" &&
       info.src->buffer.scope() != "local") ||
      (info.dst->buffer.scope() != "shared.rsram" &&
       info.dst->buffer.scope() != "local")) {
    *reason = "layout_transform_requires_rsram_to_rsram";
    return false;
  }
  if (info.src->buffer->dtype != info.dst->buffer->dtype) {
    *reason = "layout_transform_dtype_mismatch";
    return false;
  }

  DataType dtype = info.src->buffer->dtype;
  A4ETransformLayoutKind src_kind =
      ClassifyTransformLayout(src_layout.value(), dtype);
  A4ETransformLayoutKind dst_kind =
      ClassifyTransformLayout(dst_layout.value(), dtype);
  if (src_kind == A4ETransformLayoutKind::kUnknown ||
      dst_kind == A4ETransformLayoutKind::kUnknown) {
    *reason = "unsupported_transform_layout_kind";
    return false;
  }

  std::vector<uint64_t> src_shape;
  std::vector<uint64_t> dst_shape;
  if (!GetTransformPhysicalShape(info.src, src_layout.value(), &src_shape,
                                 reason) ||
      !GetTransformPhysicalShape(info.dst, dst_layout.value(), &dst_shape,
                                 reason)) {
    return false;
  }

  const std::vector<uint64_t> *plan_shape = nullptr;
  if (src_kind == A4ETransformLayoutKind::kZZ &&
      dst_kind == A4ETransformLayoutKind::kRowMajor) {
    plan->kind = A4ETransformKind::kBlockwiseToRowmajor;
    plan_shape = &dst_shape;
  } else if (src_kind == A4ETransformLayoutKind::kRowMajor &&
             dst_kind == A4ETransformLayoutKind::kZZ) {
    plan->kind = A4ETransformKind::kRowmajorToBlockwise;
    plan_shape = &src_shape;
  } else if (src_kind == A4ETransformLayoutKind::kMXZZ &&
             dst_kind == A4ETransformLayoutKind::kMXRowMajor) {
    plan->kind = A4ETransformKind::kBlockscaleBlockwiseToRowmajor;
    plan_shape = &dst_shape;
  } else if (src_kind == A4ETransformLayoutKind::kMXRowMajor &&
             dst_kind == A4ETransformLayoutKind::kMXZZ) {
    plan->kind = A4ETransformKind::kBlockscaleRowmajorToBlockwise;
    plan_shape = &src_shape;
  } else if (src_kind == A4ETransformLayoutKind::kRowMajor &&
             dst_kind == A4ETransformLayoutKind::kRowMajor &&
             src_shape[0] == dst_shape[0] && src_shape[1] != dst_shape[1]) {
    if (src_shape[1] < dst_shape[1]) {
      plan->kind = A4ETransformKind::kRowmajorPadding;
      plan_shape = &src_shape;
    } else {
      plan->kind = A4ETransformKind::kRowmajorUnpadding;
      plan_shape = &dst_shape;
    }
  } else {
    *reason = "unsupported_transform_layout_pair";
    return false;
  }

  plan->rows = (*plan_shape)[0];
  plan->cols = (*plan_shape)[1];
  plan->descriptors =
      static_cast<uint64_t>(CeilDivI64(static_cast<int64_t>(plan->rows), 32));
  if (plan->rows == 0 || plan->cols == 0 || plan->descriptors == 0) {
    *reason = "non_positive_transform_shape";
    return false;
  }

  uint64_t packet_count = 0;
  if (plan->kind == A4ETransformKind::kBlockwiseToRowmajor ||
      plan->kind == A4ETransformKind::kRowmajorUnpadding) {
    if (!(dtype.is_bfloat16() || (dtype.is_float() && dtype.bits() == 32))) {
      *reason = "unsupported_unquantized_transform_dtype";
      return false;
    }
    uint64_t element_bytes = dtype.is_bfloat16() ? 2 : 4;
    uint64_t slice_elements = 64 / element_bytes;
    if (!CheckedMulU64(plan->rows,
                       static_cast<uint64_t>(
                           CeilDivI64(static_cast<int64_t>(plan->cols),
                                      static_cast<int64_t>(slice_elements))),
                       &packet_count)) {
      *reason = "transform_packet_overflow";
      return false;
    }
  } else if (plan->kind == A4ETransformKind::kRowmajorToBlockwise ||
             plan->kind == A4ETransformKind::kRowmajorPadding) {
    if (!(dtype.is_bfloat16() || (dtype.is_float() && dtype.bits() == 32))) {
      *reason = "unsupported_unquantized_transform_dtype";
      return false;
    }
    uint64_t total_elements = 0;
    uint64_t total_bytes = 0;
    if (!CheckedMulU64(plan->rows, plan->cols, &total_elements) ||
        !CheckedMulU64(total_elements, dtype.is_bfloat16() ? 2 : 4,
                       &total_bytes)) {
      *reason = "transform_packet_overflow";
      return false;
    }
    packet_count = static_cast<uint64_t>(
        CeilDivI64(static_cast<int64_t>(total_bytes), 64));
  } else {
    // A4E MXFP4/MXFP8 issue one data read and one scale read for each
    // logical 32-element block of every row.  The comp-core Gem5 path has no
    // additional FIFO bubbles for these dtypes.
    if (!sunmmio::IsMXDType(dtype) || plan->cols % 32 != 0) {
      *reason = "unsupported_blockscale_transform_shape_or_dtype";
      return false;
    }
    uint64_t blocks = 0;
    if (!CheckedMulU64(plan->rows, plan->cols / 32, &blocks) ||
        !CheckedMulU64(blocks, 2, &packet_count)) {
      *reason = "transform_packet_overflow";
      return false;
    }
  }
  if (packet_count == 0 || packet_count > static_cast<uint64_t>(INT64_MAX)) {
    *reason = "transform_packet_overflow";
    return false;
  }

  plan->packets = packet_count;
  // Gem5 starts each descriptor's local issue counter at zero.  Under the
  // same no-contention endpoint assumption used by the other ODMA submodels,
  // a following descriptor starts one cycle after completion.  Therefore:
  //   latency = packets + 3 * descriptors - 1.
  uint64_t descriptor_tail = 0;
  uint64_t latency = 0;
  if (!CheckedMulU64(plan->descriptors, 3, &descriptor_tail) ||
      !CheckedAddU64(packet_count, descriptor_tail, &latency) || latency == 0) {
    *reason = "transform_latency_overflow";
    return false;
  }
  latency -= 1;
  plan->latency = latency;
  plan->issue_finish = latency - kGem5ODMALocalCompletionDelay;
  if (plan->descriptors > 1) {
    // Descriptor mode reads one 32-byte descriptor as two 16-byte comp-core
    // DTCM reads: 16 + 13 = 29 cycles.  Existing ODMA costs are referenced at
    // dmaProc(), so this frontend cost is reported separately.
    plan->descriptor_fetch_cycles = kGem5ODMAMultiDescriptorFetchCycles;
  }
  return true;
}

bool GetStaticRegionBytes(const BufferRegion &region, uint64_t *bytes,
                          std::string *reason) {
  uint64_t elements = 1;
  for (const Range &range : region->region) {
    const auto *extent = range->extent.as<IntImmNode>();
    if (!extent) {
      *reason = "dynamic_extent";
      return false;
    }
    if (extent->value <= 0) {
      *reason = "non_positive_extent";
      return false;
    }
    uint64_t next = 0;
    if (!CheckedMulU64(elements, static_cast<uint64_t>(extent->value), &next)) {
      *reason = "element_count_overflow";
      return false;
    }
    elements = next;
  }
  // MX storage includes scale, reserved, padding, and memory-space-dependent
  // softening.  Its byte count is a property of the final materialized
  // transfer slots, not just logical dtype bits.
  if (sunmmio::IsMXDType(region->buffer->dtype)) {
    *reason = "mx_transfer_requires_descriptor_plan";
    return false;
  }
  uint64_t bit_width = static_cast<uint64_t>(region->buffer->dtype.bits()) *
                       region->buffer->dtype.lanes();
  if (bit_width == 0 || bit_width % 8 != 0) {
    // Sub-byte storage may be padded independently for each materialized slot.
    // A logical bit count is therefore not a reliable physical transfer size.
    *reason = "subbyte_transfer_requires_descriptor_plan";
    return false;
  }
  uint64_t total_bits = 0;
  uint64_t rounded_bits = 0;
  if (!CheckedMulU64(elements, bit_width, &total_bits) ||
      !CheckedAddU64(total_bits, 7, &rounded_bits)) {
    *reason = "byte_count_overflow";
    return false;
  }
  *bytes = rounded_bits / 8;
  if (*bytes == 0 || *bytes > static_cast<uint64_t>(INT64_MAX)) {
    *reason = *bytes == 0 ? "zero_bytes" : "byte_count_overflow";
    return false;
  }
  return true;
}

bool GetStaticTiledMatrixShape(const BufferRegion &region, uint64_t *dim0,
                               uint64_t *dim1, std::string *reason) {
  std::vector<uint64_t> tiled_extents;
  for (const Range &range : region->region) {
    const auto *extent = range->extent.as<IntImmNode>();
    if (!extent) {
      *reason = "dynamic_tensor_shape";
      return false;
    }
    if (extent->value <= 0) {
      *reason = "non_positive_tensor_shape";
      return false;
    }
    if (extent->value != 1) {
      tiled_extents.push_back(static_cast<uint64_t>(extent->value));
    }
  }
  if (tiled_extents.size() != 2) {
    *reason = "tensor_requires_two_tiled_dims";
    return false;
  }
  *dim0 = tiled_extents[0];
  *dim1 = tiled_extents[1];
  return true;
}

CostEstimate MakeGem5Estimate(const char *op_kind, CostEstimateStatus status) {
  CostEstimate estimate;
  estimate.status = status;
  estimate.profile = kGem5A4EProfileName;
  estimate.model_version = kGem5A4EModelVersion;
  estimate.op_kind = op_kind;
  return estimate;
}

CostEstimate UnsupportedGem5(const char *op_kind, const std::string &reason) {
  CostEstimate estimate =
      MakeGem5Estimate(op_kind, CostEstimateStatus::kUnsupported);
  estimate.fallback_reason = reason;
  return estimate;
}

int TensorOperandDTypeId(DataType dtype) {
  if (dtype.is_bfloat16()) {
    return 1;
  }
  if (sunmmio::IsMXDType(dtype) && dtype.bits() == 8) {
    return 2;
  }
  if (sunmmio::IsMXDType(dtype) && dtype.bits() == 4) {
    return 8;
  }
  return -1;
}

int TensorAccumulatorDTypeId(DataType dtype) {
  if (dtype.is_float() && dtype.bits() == 32) {
    return 0;
  }
  if (dtype.is_bfloat16()) {
    return 1;
  }
  return -1;
}

uint64_t TensorDTypeBytes(int dtype_id) {
  if (dtype_id == 0) {
    return 4;
  }
  if (dtype_id == 1) {
    return 2;
  }
  return 1;
}

uint64_t TensorBackpressureExtraCycles(int act_dtype, int ld_dtype,
                                       int st_dtype, uint64_t row_col,
                                       uint64_t acc) {
  const bool act_bf16 = act_dtype == 1;
  const bool ld_bf16 = ld_dtype == 1;
  const bool st_bf16 = st_dtype == 1;
  // Preserve gem5's division-before-multiplication semantics.
  uint64_t store_times = row_col / 1024 * TensorDTypeBytes(act_dtype);

  if (!act_bf16) {
    if (ld_bf16 && st_bf16 && acc == 128)
      return store_times / 7 * 2;
    if (ld_bf16 && !st_bf16 && acc == 512)
      return store_times / 4 * 2;
    if (!ld_bf16 && st_bf16 && acc == 512)
      return store_times / 5 * 4;
    if (!ld_bf16 && !st_bf16 && acc == 512)
      return store_times / 3 * 6;
    return 0;
  }
  if (st_bf16)
    return 0;
  if (ld_bf16 && acc == 128)
    return store_times / 7 * 2;
  if (!ld_bf16 && acc == 128)
    return store_times / 7 * 4;
  return 0;
}

CostModelProfile ParseProfile(const ffi::String &profile) {
  if (profile == "gem5_a4e_2026") {
    return CostModelProfile::kGem5A4E2026;
  }
  if (profile == "heuristic" || profile.empty()) {
    return CostModelProfile::kHeuristic;
  }
  LOG(FATAL) << "Unsupported SunMMIO cost model profile: " << profile;
  return CostModelProfile::kHeuristic;
}

} // namespace

const char *CostModel::ProfileName(CostModelProfile profile) {
  return profile == CostModelProfile::kGem5A4E2026 ? kGem5A4EProfileName
                                                   : kHeuristicProfileName;
}

const char *CostModel::ModelVersion(CostModelProfile profile) {
  return profile == CostModelProfile::kGem5A4E2026 ? kGem5A4EModelVersion
                                                   : kHeuristicModelVersion;
}

const char *CostModel::StatusName(CostEstimateStatus status) {
  switch (status) {
  case CostEstimateStatus::kExactStatic:
    return "exact_static";
  case CostEstimateStatus::kApproximate:
    return "approximate";
  case CostEstimateStatus::kFallback:
    return "fallback";
  case CostEstimateStatus::kUnsupported:
    return "unsupported";
  }
  LOG(FATAL) << "Unknown CostEstimateStatus: " << static_cast<int>(status);
  return "unsupported";
}

CostModelContext CostModel::BuildContext(const tir::PrimFunc &func) {
  CostModelContext context;
  auto pass_ctx = tvm::transform::PassContext::Current();
  auto profile = pass_ctx->GetConfig<ffi::String>(kSunmmioCostModel);
  if (profile.has_value()) {
    context.options.profile = ParseProfile(profile.value());
  } else if (const char *env = std::getenv("TL_SUNMMIO_COST_MODEL")) {
    context.options.profile = ParseProfile(ffi::String(env));
  }

  context.options.fallback_to_heuristic =
      pass_ctx->GetConfig<Bool>(kSunmmioCostModelFallback, Bool(true)).value();
  context.options.strict_calibration =
      pass_ctx->GetConfig<Bool>(kSunmmioCostModelStrict, Bool(false)).value();
  if (context.options.strict_calibration) {
    ICHECK(context.options.profile == CostModelProfile::kGem5A4E2026)
        << kSunmmioCostModelStrict << " requires " << kSunmmioCostModel
        << "=gem5_a4e_2026";
    context.options.fallback_to_heuristic = false;
  }
  context.target = func->GetAttr<Target>(tvm::attr::kTarget);
  if (auto layout_map =
          func->GetAttr<ffi::Map<Buffer, Layout>>(attr::kLayoutMap)) {
    context.layout_map = layout_map.value();
  }
  if (auto global_layout_map =
          func->GetAttr<ffi::Map<Buffer, Layout>>(attr::kGlobalLayoutMap)) {
    context.global_layout_map = global_layout_map.value();
  }
  return context;
}

CostEstimate
CostModel::EstimateTensorCoreDelayGem5A4E(const tir::Stmt &stmt,
                                          const CostModelContext &context) {
  (void)context;
  const auto *block = stmt.as<BlockRealizeNode>();
  if (!block)
    return UnsupportedGem5("tensor_mma", "expected_block_realize");
  const auto *eval = block->block->body.as<EvaluateNode>();
  if (!eval)
    return UnsupportedGem5("tensor_mma", "expected_evaluate_body");
  const auto *call = eval->value.as<CallNode>();
  if (!call || !call->op.same_as(mma_sunmmio()) || call->args.size() < 6)
    return UnsupportedGem5("tensor_mma", "invalid_mma_intrinsic");
  const auto *a = call->args[0].as<CallNode>();
  const auto *b = call->args[1].as<CallNode>();
  const auto *c = call->args[2].as<CallNode>();
  if (!a || !b || !c || a->args.size() < 4 || b->args.size() < 4 ||
      c->args.size() < 4)
    return UnsupportedGem5("tensor_mma", "invalid_region_encoding");
  const auto *trans_a_imm = call->args[3].as<IntImmNode>();
  const auto *trans_b_imm = call->args[4].as<IntImmNode>();
  const auto *clear_imm = call->args[5].as<IntImmNode>();
  if (!trans_a_imm || !trans_b_imm)
    return UnsupportedGem5("tensor_mma", "dynamic_tensor_transpose");
  if (!clear_imm)
    return UnsupportedGem5("tensor_mma", "dynamic_clear_accum");
  if (trans_a_imm->value != 0)
    return UnsupportedGem5("tensor_mma", "tensor_trans_a_unsupported");

  BufferRegion a_region = NormalizeToBufferRegion(call->args[0]);
  BufferRegion b_region = NormalizeToBufferRegion(call->args[1]);
  BufferRegion c_region = NormalizeToBufferRegion(call->args[2]);
  if (a_region->buffer.scope() != "shared.asram")
    return UnsupportedGem5("tensor_mma", "activation_requires_asram");
  if (b_region->buffer.scope() != "shared.wsram")
    return UnsupportedGem5("tensor_mma", "weight_requires_wsram");
  if (c_region->buffer.scope() != "shared.rsram")
    return UnsupportedGem5("tensor_mma", "accumulator_requires_rsram");
  std::string shape_reason;
  uint64_t row = 0;
  uint64_t acc = 0;
  uint64_t weight_dim0 = 0;
  uint64_t weight_dim1 = 0;
  uint64_t store_row = 0;
  uint64_t col = 0;
  if (!GetStaticTiledMatrixShape(a_region, &row, &acc, &shape_reason) ||
      !GetStaticTiledMatrixShape(b_region, &weight_dim0, &weight_dim1,
                                 &shape_reason) ||
      !GetStaticTiledMatrixShape(c_region, &store_row, &col, &shape_reason)) {
    return UnsupportedGem5("tensor_mma", shape_reason);
  }
  if (store_row != row)
    return UnsupportedGem5("tensor_mma", "tensor_store_row_mismatch");
  const bool trans_b = trans_b_imm->value != 0;
  if ((!trans_b && (weight_dim0 != acc || weight_dim1 != col)) ||
      (trans_b && (weight_dim0 != col || weight_dim1 != acc))) {
    return UnsupportedGem5("tensor_mma", "tensor_weight_shape_mismatch");
  }
  DataType act_type = a_region->buffer->dtype;
  DataType rhs_type = b_region->buffer->dtype;
  DataType st_type = c_region->buffer->dtype;
  int act_dtype = TensorOperandDTypeId(act_type);
  int rhs_dtype = TensorOperandDTypeId(rhs_type);
  int st_dtype = TensorAccumulatorDTypeId(st_type);
  if (act_dtype < 0)
    return UnsupportedGem5("tensor_mma", "unsupported_activation_dtype");
  if (rhs_dtype < 0)
    return UnsupportedGem5("tensor_mma", "unsupported_weight_dtype");
  if (st_dtype < 0)
    return UnsupportedGem5("tensor_mma", "unsupported_accumulator_dtype");
  const bool act_bf16 = act_dtype == 1;
  const bool rhs_bf16 = rhs_dtype == 1;
  // NPU-IR permits BF16 x (BF16 or MX) and MX x MX.  MX x BF16 is not a
  // legal TensorCore combination.
  if (!act_bf16 && rhs_bf16) {
    return UnsupportedGem5("tensor_mma",
                           "unsupported_activation_weight_combination");
  }

  if (row % 32 != 0)
    return UnsupportedGem5("tensor_mma", "tensor_row_not_multiple_of_32");
  if (col < 32 || col % 32 != 0)
    return UnsupportedGem5("tensor_mma", "tensor_col_not_multiple_of_32");
  const bool has_mx_operand = act_dtype != 1 || rhs_dtype != 1;
  const uint64_t acc_multiple = has_mx_operand ? 64 : 32;
  if (acc % acc_multiple != 0) {
    return UnsupportedGem5("tensor_mma", has_mx_operand
                                             ? "mx_acc_not_multiple_of_64"
                                             : "bf16_acc_not_multiple_of_32");
  }
  // NPU-IR lowers a logical BF16 activation tile as two tc.mma operations
  // and writes half of its logical row extent to TC_ROW_SIZE.
  if (act_dtype == 1) {
    if ((row & 1U) != 0)
      return UnsupportedGem5("tensor_mma", "bf16_tensor_row_not_even");
    row /= 2;
  }
  bool preload_result = clear_imm->value == 0;
  int ld_dtype = st_dtype;
  uint64_t min_blk_st =
      st_dtype == 1 ? (act_dtype == 1 ? 2 : 3) : (act_dtype == 1 ? 3 : 7);
  uint64_t min_blk = min_blk_st;
  if (preload_result) {
    uint64_t min_blk_ld =
        ld_dtype == 1 ? (act_dtype == 1 ? 2 : 3) : (act_dtype == 1 ? 3 : 7);
    min_blk += min_blk_ld;
  }
  if ((act_dtype == 1 && row <= 16 && col <= 32) ||
      (act_dtype != 1 && row <= 32 && col <= 32)) {
    min_blk = 1;
  } else {
    min_blk = std::max<uint64_t>(min_blk, 4);
  }
  uint64_t gap = act_dtype == 1
                     ? std::max<uint64_t>(CeilDivI64(acc, 32), min_blk)
                     : std::max<uint64_t>(CeilDivI64(acc, 64), min_blk);
  uint64_t row_tiles = CeilDivI64(row, act_dtype == 1 ? 16 : 32);
  uint64_t col_tiles = CeilDivI64(col, 32);
  uint64_t tile_count = 0;
  if (!CheckedMulU64(row_tiles, col_tiles, &tile_count) ||
      !CheckedMulU64(tile_count, gap, &tile_count) ||
      tile_count > static_cast<uint64_t>(INT64_MAX)) {
    return UnsupportedGem5("tensor_mma", "tensor_latency_overflow");
  }
  uint64_t delay =
      kGem5TensorCompletionDelay + tile_count + kGem5TensorFirstResultDelay;
  if (act_dtype != 1 && (acc / 64) >= min_blk)
    delay += 1;
  if (preload_result) {
    uint64_t row_col = 0;
    if (!CheckedMulU64(row, col, &row_col)) {
      return UnsupportedGem5("tensor_mma", "tensor_latency_overflow");
    }
    delay += TensorBackpressureExtraCycles(act_dtype, ld_dtype, st_dtype,
                                           row_col, acc);
  }
  if (delay > static_cast<uint64_t>(INT64_MAX))
    return UnsupportedGem5("tensor_mma", "tensor_latency_overflow");

  CostEstimate estimate =
      MakeGem5Estimate("tensor_mma", CostEstimateStatus::kExactStatic);
  estimate.latency_cycles = static_cast<int64_t>(delay);
  estimate.issue_finish_cycles = estimate.latency_cycles;
  estimate.latency_source = "TensorCore::calculateOpDelayCycles";
  estimate.input_dtype = DTypeName(act_type);
  estimate.weight_dtype = DTypeName(rhs_type);
  estimate.output_dtype = DTypeName(st_type);
  estimate.timing_origin = "TensorCore::calculateOpDelayCycles_entry";
  estimate.timing_end = "tensor_completion";
  estimate.command_frontend_modeled = false;
  estimate.row_size = static_cast<int64_t>(row);
  estimate.col_size = static_cast<int64_t>(col);
  estimate.reduction_size = static_cast<int64_t>(acc);
  return estimate;
}

CostEstimate
CostModel::EstimateODMADelayGem5A4E(const tir::Stmt &stmt,
                                    const CostModelContext &context) {
  SunmmioAsyncOpInfo info;
  if (!TryParseSunmmioAsyncODMAStmt(stmt, &info))
    return UnsupportedGem5("odma", "unrecognized_async_odma_op");
  uint64_t bytes = 0;
  std::string reason;
  // Layout-transform packetization is derived from the materialized layout
  // plan, not from logical dtype bits.  In particular, MX transfers carry data,
  // scale, reserved, and padded storage and must reach BuildA4ETransformPlan.
  // For byte-addressable non-MX transforms, keep the logical byte count as a
  // useful diagnostic even though it does not drive the latency formula.
  if (info.kind != SunmmioAsyncOpKind::kLayoutTransform ||
      !sunmmio::IsMXDType(info.src->buffer->dtype)) {
    if (!GetStaticRegionBytes(info.src, &bytes, &reason))
      return UnsupportedGem5("odma", reason);
    if (bytes > static_cast<uint64_t>(INT64_MAX))
      return UnsupportedGem5("odma", "transfer_size_overflow");
  }

  CostEstimate estimate =
      MakeGem5Estimate("odma", CostEstimateStatus::kApproximate);
  estimate.bytes = static_cast<int64_t>(bytes);
  estimate.src_scope =
      IsGlobalBuffer(info.src->buffer) ? "global" : info.src->buffer.scope();
  estimate.dst_scope =
      IsGlobalBuffer(info.dst->buffer) ? "global" : info.dst->buffer.scope();
  estimate.input_dtype = DTypeName(info.src->buffer->dtype);
  estimate.output_dtype = DTypeName(info.dst->buffer->dtype);
  estimate.odma_channel = info.odma_channel_hint;
  estimate.odma_channel_mask = info.odma_channel_mask;
  estimate.endpoint_timing_modeled = false;
  estimate.timing_origin = "ODMA::dmaStart_entry";
  estimate.timing_end = "odma_completion";
  estimate.command_frontend_modeled = false;

  if (info.kind == SunmmioAsyncOpKind::kLayoutTransform) {
    if (context.layout_map.empty() && context.global_layout_map.empty()) {
      return UnsupportedGem5("layout_transform", "missing_layout_context");
    }
    A4ETransformPlan plan;
    if (!BuildA4ETransformPlan(info, context, &plan, &reason)) {
      return UnsupportedGem5("layout_transform", reason);
    }
    estimate.op_kind = "layout_transform";
    estimate.transfer_path = "rsram_to_rsram_layout_transform";
    estimate.transform_kind = TransformKindName(plan.kind);
    estimate.row_size = static_cast<int64_t>(plan.rows);
    estimate.col_size = static_cast<int64_t>(plan.cols);
    estimate.packet_count = static_cast<int64_t>(plan.packets);
    estimate.descriptor_count = static_cast<int64_t>(plan.descriptors);
    estimate.descriptor_fetch_cycles =
        static_cast<int64_t>(plan.descriptor_fetch_cycles);
    uint64_t frontend_cycles = kGem5ODMADirectCommandFrontendCycles;
    if (!CheckedAddU64(frontend_cycles, plan.descriptor_fetch_cycles,
                       &frontend_cycles)) {
      return UnsupportedGem5("layout_transform", "command_frontend_overflow");
    }
    uint64_t issue_finish = 0;
    uint64_t latency = 0;
    if (!CheckedAddU64(plan.issue_finish, frontend_cycles, &issue_finish) ||
        !CheckedAddU64(plan.latency, frontend_cycles, &latency) ||
        issue_finish > static_cast<uint64_t>(INT64_MAX) ||
        latency > static_cast<uint64_t>(INT64_MAX)) {
      return UnsupportedGem5("layout_transform", "command_latency_overflow");
    }
    estimate.command_frontend_cycles = static_cast<int64_t>(frontend_cycles);
    estimate.issue_start_cycles = static_cast<int64_t>(frontend_cycles);
    estimate.issue_finish_cycles = static_cast<int64_t>(issue_finish);
    estimate.issue_interval_cycles = 1;
    estimate.endpoint_completion_cycles = kGem5ODMALocalCompletionDelay;
    estimate.latency_cycles = static_cast<int64_t>(latency);
    estimate.latency_source =
        plan.kind == A4ETransformKind::kBlockwiseToRowmajor
            ? "ODMA::procUnqtBlk2RowLSram2LSram"
        : plan.kind == A4ETransformKind::kRowmajorToBlockwise
            ? "ODMA::procUnqtRow2BlkLSram2LSram"
        : plan.kind == A4ETransformKind::kBlockscaleBlockwiseToRowmajor
            ? "ODMA::procQtBlk2RowLSram2LSram"
        : plan.kind == A4ETransformKind::kBlockscaleRowmajorToBlockwise
            ? "ODMA::procQtRow2BlkLSram2LSram"
        : plan.kind == A4ETransformKind::kRowmajorPadding
            ? "ODMA::procUnqtRow2RowPadLSram2LSram"
            : "ODMA::procUnqtRow2RowUnpadLSram2LSram";
    estimate.status = CostEstimateStatus::kExactStatic;
    estimate.endpoint_timing_modeled = true;
    estimate.descriptor_plan_modeled = true;
    estimate.command_frontend_modeled = true;
    return estimate;
  }

  if (info.kind == SunmmioAsyncOpKind::kTranspose) {
    if (info.src->buffer.scope() != "shared.rsram" ||
        info.dst->buffer.scope() != "shared.rsram")
      return UnsupportedGem5("transpose", "transpose_requires_rsram");
    if (info.src->buffer->dtype != info.dst->buffer->dtype)
      return UnsupportedGem5("transpose", "transpose_dtype_mismatch");
    if (info.src->region.size() != 2 || info.dst->region.size() != 2)
      return UnsupportedGem5("transpose", "transpose_requires_rank_2");
    const auto *m = info.src->region[0]->extent.as<IntImmNode>();
    const auto *n = info.src->region[1]->extent.as<IntImmNode>();
    const auto *dst_m = info.dst->region[0]->extent.as<IntImmNode>();
    const auto *dst_n = info.dst->region[1]->extent.as<IntImmNode>();
    if (!m || !n || !dst_m || !dst_n)
      return UnsupportedGem5("transpose", "dynamic_transpose_shape");
    if (dst_m->value != n->value || dst_n->value != m->value)
      return UnsupportedGem5("transpose", "transpose_dst_shape_mismatch");
    if (m->value <= 0 || n->value <= 0 || m->value % 32 != 0 ||
        n->value % 32 != 0)
      return UnsupportedGem5("transpose", "transpose_shape_not_32_aligned");
    int64_t block_size = info.src->buffer->dtype.is_bfloat16() ? 2048
                         : info.src->buffer->dtype.is_float() &&
                                 info.src->buffer->dtype.bits() == 32
                             ? 4096
                             : 0;
    if (block_size == 0)
      return UnsupportedGem5("transpose", "unsupported_transpose_dtype");
    uint64_t blocks = 0;
    if (!CheckedMulU64(static_cast<uint64_t>(m->value / 32),
                       static_cast<uint64_t>(n->value / 32), &blocks))
      return UnsupportedGem5("transpose", "transpose_packet_overflow");
    uint64_t slices = static_cast<uint64_t>(block_size / 256);
    uint64_t packets = 0;
    if (!CheckedMulU64(blocks, slices, &packets) ||
        !CheckedMulU64(packets, slices, &packets) || packets == 0)
      return UnsupportedGem5("transpose", "transpose_packet_overflow");
    uint64_t first_issue = blocks * 4 + 8;
    uint64_t last_issue = first_issue + packets - 1;
    if (last_issue >
        static_cast<uint64_t>(INT64_MAX - kGem5ODMADirectCommandFrontendCycles -
                              kGem5ODMALocalCompletionDelay))
      return UnsupportedGem5("transpose", "transpose_latency_overflow");
    estimate.op_kind = "transpose";
    estimate.transfer_path = "rsram_to_rsram_transpose";
    estimate.transform_kind = "transpose";
    estimate.path_start_delay = static_cast<int64_t>(first_issue);
    estimate.packet_count = static_cast<int64_t>(packets);
    estimate.descriptor_count = 1;
    estimate.command_frontend_cycles = kGem5ODMADirectCommandFrontendCycles;
    estimate.issue_start_cycles = static_cast<int64_t>(first_issue) +
                                  kGem5ODMADirectCommandFrontendCycles;
    estimate.issue_finish_cycles =
        static_cast<int64_t>(last_issue) + kGem5ODMADirectCommandFrontendCycles;
    estimate.issue_interval_cycles = 1;
    estimate.endpoint_completion_cycles = kGem5ODMALocalCompletionDelay;
    estimate.latency_cycles =
        estimate.issue_finish_cycles + estimate.endpoint_completion_cycles;
    estimate.latency_source = "ODMA::procTransposeLSram2LSram";
    estimate.status = CostEstimateStatus::kExactStatic;
    estimate.endpoint_timing_modeled = true;
    estimate.descriptor_plan_modeled = true;
    estimate.command_frontend_modeled = true;
    return estimate;
  }

  if (info.odma_channel_hint < 0) {
    return UnsupportedGem5(
        info.kind == SunmmioAsyncOpKind::kBroadcast ? "broadcast" : "dma_copy",
        "no_legal_a4e_odma_channel");
  }

  if (info.src->buffer->dtype != info.dst->buffer->dtype) {
    return UnsupportedGem5(
        info.kind == SunmmioAsyncOpKind::kBroadcast ? "broadcast" : "dma_copy",
        "dtype_conversion_requires_descriptor_plan");
  }

  const bool src_global = IsGlobalBuffer(info.src->buffer);
  const bool dst_global = IsGlobalBuffer(info.dst->buffer);
  const ffi::String &src_scope = info.src->buffer.scope();
  const ffi::String &dst_scope = info.dst->buffer.scope();
  const bool src_rsram = src_scope == "shared.rsram" || src_scope == "local";
  const bool dst_rsram = dst_scope == "shared.rsram" || dst_scope == "local";
  const bool dst_asram = dst_scope == "shared.asram";
  const bool dst_wsram = dst_scope == "shared.wsram";
  const int64_t packet_count = CeilDivI64(static_cast<int64_t>(bytes), 1024);
  const int64_t tail = std::max<int64_t>(packet_count - 1, 0);
  int64_t start = 0;
  int64_t endpoint_completion = 0;
  int64_t latency = 0;

  if (info.kind == SunmmioAsyncOpKind::kDmaCopy) {
    estimate.op_kind = "dma_copy";
    endpoint_completion = kGem5ODMALocalCompletionDelay;
    if (src_global && !dst_global) {
      start = 67;
      if (dst_rsram) {
        endpoint_completion += kGem5ODMADramToRsramCompletionExtra;
      }
      estimate.transfer_path = "dram_to_local_sram";
    } else if (src_rsram && dst_rsram) {
      start = 7;
      estimate.transfer_path = "rsram_to_rsram_local";
    } else if (src_rsram && dst_asram &&
               !info.src->buffer->dtype.is_bfloat16()) {
      start = 8;
      estimate.transfer_path = "rsram_to_asram_non_bf16_local";
    } else if (src_rsram && (dst_asram || dst_wsram)) {
      start = 10 + static_cast<int64_t>(bytes / 8192);
      estimate.transfer_path =
          dst_asram ? "rsram_to_asram_bf16_local" : "rsram_to_wsram_local";
    } else if (src_rsram && dst_global) {
      start = 7;
      estimate.transfer_path = "rsram_to_dram";
    } else {
      return UnsupportedGem5("dma_copy", "unsupported_local_copy_path");
    }
    latency = start + tail;
    estimate.latency_source = "ODMA copy issue stream + endpoint completion";
  } else if (info.kind == SunmmioAsyncOpKind::kBroadcast) {
    estimate.op_kind = "broadcast";
    endpoint_completion = kGem5ODMALinkCompletionDelay;
    if (info.path_kind == SunmmioODMAPathKind::kUnknown)
      return UnsupportedGem5("broadcast", "dynamic_or_invalid_direction");
    if (src_global) {
      start = 95 + static_cast<int64_t>(bytes / (dst_rsram ? 155 : 1024));
      estimate.transfer_path =
          dst_rsram ? "dram_to_link_rsram" : "dram_to_link_asram_wsram";
    } else if (src_rsram && dst_asram) {
      start =
          16 + static_cast<int64_t>(
                   bytes / (info.src->buffer->dtype.is_bfloat16() ? 700 : 224));
      endpoint_completion += kGem5ODMARsramToLinkAsramCompletionExtra;
      estimate.transfer_path = info.src->buffer->dtype.is_bfloat16()
                                   ? "rsram_to_link_asram_bf16"
                                   : "rsram_to_link_asram_non_bf16";
    } else if (src_rsram && dst_wsram) {
      start = 16 + static_cast<int64_t>(bytes / 1024);
      estimate.transfer_path = "rsram_to_link_wsram";
    } else if (src_rsram && dst_rsram) {
      start = 10 + static_cast<int64_t>(bytes / 175);
      estimate.transfer_path = "rsram_to_link_rsram";
    } else {
      return UnsupportedGem5("broadcast", "unsupported_link_copy_path");
    }
    latency = start + tail;
    estimate.latency_source =
        "ODMA link copy issue stream + endpoint completion";
  } else {
    return UnsupportedGem5("odma", "unsupported_odma_kind");
  }

  int64_t issue_finish = latency + kGem5ODMADirectCommandFrontendCycles;
  latency = issue_finish + endpoint_completion;
  estimate.command_frontend_cycles = kGem5ODMADirectCommandFrontendCycles;
  estimate.path_start_delay = start;
  estimate.packet_count = packet_count;
  // The common two-cycle dmaStart -> dmaProc path is always present.  The
  // TIR operation does not yet expose the final AddressMaterialization slots,
  // so descriptor count and the possible extra 29-cycle fetch remain unknown.
  estimate.descriptor_count = 0;
  estimate.issue_start_cycles = start + kGem5ODMADirectCommandFrontendCycles;
  estimate.issue_finish_cycles = issue_finish;
  estimate.issue_interval_cycles = 1;
  estimate.endpoint_completion_cycles = endpoint_completion;
  estimate.latency_cycles = latency;
  // Direction selects HLink/VLink, but both copy_async and multicast may
  // lower to multiple AddressMaterialization slots and may carry Vmode or
  // strided semantics.  Until the final descriptor plan is shared, neither
  // operation is an exact static submodel.
  estimate.descriptor_plan_modeled = false;
  estimate.endpoint_timing_modeled = endpoint_completion > 0;
  if (info.kind == SunmmioAsyncOpKind::kBroadcast) {
    estimate.link_completion_cycles =
        info.path_kind == SunmmioODMAPathKind::kHLink
            ? kGem5HLinkLastCompletionCycles
            : kGem5VLinkLastCompletionCycles;
    // Gem5 starts this delay when the receiving link observes LINK_LAST.
    // TIR does not expose the network-arrival point, so adding it to the
    // local ODMA completion would incorrectly assume the two are serial.
    estimate.link_completion_modeled = false;
    estimate.timing_end = "local_odma_completion_before_link_last";
  }
  return estimate;
}

namespace {

struct VectorLatencyStats {
  int64_t cycles{0};
  int64_t chunks{0};
  int64_t load_ops{0};
  int64_t store_ops{0};
  int64_t arithmetic_ops{0};
  int64_t compare_convert_ops{0};
  int64_t nonlinear_ops{0};
  int64_t reduce_ops{0};
  int64_t galu_ops{0};
  std::string reason;
};

class Gem5VectorLatencyAnalyzer {
public:
  explicit Gem5VectorLatencyAnalyzer(int64_t register_bits)
      : register_bits_(register_bits) {}

  bool Analyze(const Stmt &stmt, VectorLatencyStats *stats) {
    stats_ = stats;
    return AnalyzeStmt(stmt, 1);
  }

private:
  int64_t Chunks(int64_t multiplicity, DataType dtype) {
    if (multiplicity <= 0 || dtype.bits() <= 0 || dtype.lanes() <= 0 ||
        register_bits_ <= 0) {
      stats_->reason = "invalid_vector_width";
      return -1;
    }
    if (multiplicity > INT64_MAX / dtype.bits() / dtype.lanes()) {
      stats_->reason = "vector_width_overflow";
      return -1;
    }
    int64_t bits = multiplicity * dtype.bits() * dtype.lanes();
    return CeilDivI64(bits, register_bits_);
  }

  bool AddCost(int64_t latency, int64_t chunks, int64_t *counter) {
    if (chunks < 0 || latency < 0 ||
        (chunks != 0 && latency > INT64_MAX / chunks) ||
        stats_->cycles > INT64_MAX - latency * chunks ||
        stats_->chunks > INT64_MAX - chunks || *counter > INT64_MAX - chunks) {
      stats_->reason = "vector_latency_overflow";
      return false;
    }
    stats_->cycles += latency * chunks;
    stats_->chunks += chunks;
    *counter += chunks;
    return true;
  }

  bool AddForExpr(DataType dtype, int64_t multiplicity, int64_t latency,
                  int64_t *counter) {
    int64_t chunks = Chunks(multiplicity, dtype);
    return chunks >= 0 && AddCost(latency, chunks, counter);
  }

  bool AnalyzeBinary(const PrimExpr &a, const PrimExpr &b, DataType dtype,
                     int64_t multiplicity, bool arithmetic) {
    int64_t *counter = arithmetic ? &stats_->arithmetic_ops : &stats_->galu_ops;
    int64_t latency = arithmetic && dtype.is_float() ? 4 : 3;
    return AddForExpr(dtype, multiplicity, latency, counter) &&
           AnalyzeExpr(a, multiplicity) && AnalyzeExpr(b, multiplicity);
  }

  bool AnalyzeCompare(const PrimExpr &a, const PrimExpr &b,
                      int64_t multiplicity) {
    return AddForExpr(a.dtype(), multiplicity, 3,
                      &stats_->compare_convert_ops) &&
           AnalyzeExpr(a, multiplicity) && AnalyzeExpr(b, multiplicity);
  }

  bool AnalyzeExpr(const PrimExpr &expr, int64_t multiplicity) {
    if (expr.as<IntImmNode>() || expr.as<FloatImmNode>() ||
        expr.as<VarNode>() || expr.as<StringImmNode>()) {
      return true;
    }
    if (const auto *op = expr.as<BufferLoadNode>()) {
      return AddForExpr(op->dtype, multiplicity, 12, &stats_->load_ops);
    }
    if (const auto *op = expr.as<AddNode>())
      return AnalyzeBinary(op->a, op->b, op->dtype, multiplicity, true);
    if (const auto *op = expr.as<SubNode>())
      return AnalyzeBinary(op->a, op->b, op->dtype, multiplicity, true);
    if (const auto *op = expr.as<MulNode>())
      return AnalyzeBinary(op->a, op->b, op->dtype, multiplicity, true);
    if (const auto *op = expr.as<DivNode>())
      return AnalyzeBinary(op->a, op->b, op->dtype, multiplicity, true);
    if (const auto *op = expr.as<FloorDivNode>())
      return AnalyzeBinary(op->a, op->b, op->dtype, multiplicity, false);
    if (const auto *op = expr.as<FloorModNode>())
      return AnalyzeBinary(op->a, op->b, op->dtype, multiplicity, false);
    if (const auto *op = expr.as<ModNode>())
      return AnalyzeBinary(op->a, op->b, op->dtype, multiplicity, false);
    if (const auto *op = expr.as<MinNode>())
      return AnalyzeBinary(op->a, op->b, op->dtype, multiplicity, true);
    if (const auto *op = expr.as<MaxNode>())
      return AnalyzeBinary(op->a, op->b, op->dtype, multiplicity, true);
    if (const auto *op = expr.as<CastNode>()) {
      return AddForExpr(op->dtype, multiplicity, 3,
                        &stats_->compare_convert_ops) &&
             AnalyzeExpr(op->value, multiplicity);
    }
    if (const auto *op = expr.as<LTNode>())
      return AnalyzeCompare(op->a, op->b, multiplicity);
    if (const auto *op = expr.as<LENode>())
      return AnalyzeCompare(op->a, op->b, multiplicity);
    if (const auto *op = expr.as<GTNode>())
      return AnalyzeCompare(op->a, op->b, multiplicity);
    if (const auto *op = expr.as<GENode>())
      return AnalyzeCompare(op->a, op->b, multiplicity);
    if (const auto *op = expr.as<EQNode>())
      return AnalyzeCompare(op->a, op->b, multiplicity);
    if (const auto *op = expr.as<NENode>())
      return AnalyzeCompare(op->a, op->b, multiplicity);
    if (const auto *op = expr.as<AndNode>())
      return AnalyzeBinary(op->a, op->b, op->dtype, multiplicity, false);
    if (const auto *op = expr.as<OrNode>())
      return AnalyzeBinary(op->a, op->b, op->dtype, multiplicity, false);
    if (const auto *op = expr.as<NotNode>()) {
      return AddForExpr(op->dtype, multiplicity, 3, &stats_->galu_ops) &&
             AnalyzeExpr(op->a, multiplicity);
    }
    if (const auto *op = expr.as<SelectNode>()) {
      return AnalyzeExpr(op->condition, multiplicity) &&
             AnalyzeExpr(op->true_value, multiplicity) &&
             AnalyzeExpr(op->false_value, multiplicity);
    }
    if (const auto *op = expr.as<LetNode>()) {
      return AnalyzeExpr(op->value, multiplicity) &&
             AnalyzeExpr(op->body, multiplicity);
    }
    if (const auto *op = expr.as<BroadcastNode>()) {
      return AddForExpr(op->dtype, multiplicity, 1, &stats_->galu_ops) &&
             AnalyzeExpr(op->value, multiplicity);
    }
    if (const auto *op = expr.as<RampNode>()) {
      return AddForExpr(op->dtype, multiplicity, 1, &stats_->galu_ops) &&
             AnalyzeExpr(op->base, multiplicity) &&
             AnalyzeExpr(op->stride, multiplicity);
    }
    if (const auto *call = expr.as<CallNode>()) {
      const auto *op_node = call->op.as<OpNode>();
      std::string name = op_node ? op_node->name : "";
      if (name == "tl.infinity")
        return true;
      int64_t latency = 0;
      int64_t *counter = nullptr;
      if (name == "tir.exp2" || name == "tir.exp" || name == "tir.log" ||
          name == "tir.tanh" || name == "tir.sigmoid") {
        latency = 11;
        counter = &stats_->nonlinear_ops;
      } else if (name == "tir.bitwise_and" || name == "tir.bitwise_or" ||
                 name == "tir.bitwise_xor" || name == "tir.shift_left" ||
                 name == "tir.shift_right") {
        latency = 3;
        counter = &stats_->galu_ops;
      } else if (name == "tir.if_then_else") {
        for (const PrimExpr &arg : call->args) {
          if (!AnalyzeExpr(arg, multiplicity))
            return false;
        }
        return true;
      } else {
        stats_->reason = "unknown_vector_call:" + name;
        return false;
      }
      if (!AddForExpr(call->dtype, multiplicity, latency, counter))
        return false;
      for (const PrimExpr &arg : call->args) {
        if (!AnalyzeExpr(arg, multiplicity))
          return false;
      }
      return true;
    }
    stats_->reason = std::string("unknown_vector_expr:") + expr->GetTypeKey();
    return false;
  }

  bool AnalyzeStmt(const Stmt &stmt, int64_t multiplicity) {
    if (const auto *op = stmt.as<ForNode>()) {
      const auto *extent = op->extent.as<IntImmNode>();
      if (!extent) {
        stats_->reason = "dynamic_vector_loop_extent";
        return false;
      }
      if (extent->value <= 0 || multiplicity > INT64_MAX / extent->value) {
        stats_->reason = "invalid_vector_loop_extent";
        return false;
      }
      return AnalyzeStmt(op->body, multiplicity * extent->value);
    }
    if (const auto *op = stmt.as<BlockRealizeNode>())
      return AnalyzeStmt(op->block->body, multiplicity);
    if (const auto *op = stmt.as<SeqStmtNode>()) {
      for (const Stmt &child : op->seq) {
        if (!AnalyzeStmt(child, multiplicity))
          return false;
      }
      return true;
    }
    if (const auto *op = stmt.as<IfThenElseNode>()) {
      if (!AnalyzeExpr(op->condition, multiplicity) ||
          !AnalyzeStmt(op->then_case, multiplicity))
        return false;
      return !op->else_case.defined() ||
             AnalyzeStmt(op->else_case.value(), multiplicity);
    }
    if (const auto *op = stmt.as<BufferStoreNode>()) {
      return AnalyzeExpr(op->value, multiplicity) &&
             AddForExpr(op->buffer->dtype, multiplicity, 2, &stats_->store_ops);
    }
    if (const auto *op = stmt.as<EvaluateNode>()) {
      if (const auto *call = op->value.as<CallNode>();
          call && call->op.same_as(vector_core_in_tile_reduce())) {
        const auto *kind = call->args[0].as<StringImmNode>();
        if (!kind) {
          stats_->reason = "dynamic_reduce_kind";
          return false;
        }
        int64_t latency = kind->value == "sum"                             ? 14
                          : (kind->value == "min" || kind->value == "max") ? 2
                                                                           : 0;
        if (latency == 0) {
          stats_->reason = "unknown_reduce_kind";
          return false;
        }
        return AddForExpr(call->dtype, multiplicity, latency,
                          &stats_->reduce_ops);
      }
      return AnalyzeExpr(op->value, multiplicity);
    }
    if (const auto *op = stmt.as<LetStmtNode>()) {
      return AnalyzeExpr(op->value, multiplicity) &&
             AnalyzeStmt(op->body, multiplicity);
    }
    if (const auto *op = stmt.as<AttrStmtNode>())
      return AnalyzeStmt(op->body, multiplicity);
    if (const auto *op = stmt.as<AllocateNode>())
      return AnalyzeStmt(op->body, multiplicity);
    stats_->reason = std::string("unknown_vector_stmt:") + stmt->GetTypeKey();
    return false;
  }

  int64_t register_bits_{kA4ERegisterBits};
  VectorLatencyStats *stats_{nullptr};
};

ffi::Map<ffi::String, ffi::String>
EstimateToStringMap(const CostEstimate &estimate) {
  ffi::Map<ffi::String, ffi::String> result;
  auto set_int = [&](const char *key, int64_t value) {
    result.Set(key, ffi::String(std::to_string(value)));
  };
  result.Set("status", ffi::String(CostModel::StatusName(estimate.status)));
  result.Set("profile", estimate.profile);
  result.Set("model_version", estimate.model_version);
  result.Set("latency_source", estimate.latency_source);
  result.Set("fallback_reason", estimate.fallback_reason);
  result.Set("op_kind", estimate.op_kind);
  result.Set("transfer_path", estimate.transfer_path);
  result.Set("src_scope", estimate.src_scope);
  result.Set("dst_scope", estimate.dst_scope);
  result.Set("input_dtype", estimate.input_dtype);
  result.Set("weight_dtype", estimate.weight_dtype);
  result.Set("output_dtype", estimate.output_dtype);
  result.Set("transform_kind", estimate.transform_kind);
  result.Set("timing_origin", estimate.timing_origin);
  result.Set("timing_end", estimate.timing_end);
  set_int("latency_cycles", estimate.latency_cycles);
  set_int("issue_start_cycles", estimate.issue_start_cycles);
  set_int("issue_finish_cycles", estimate.issue_finish_cycles);
  set_int("bytes", estimate.bytes);
  set_int("packet_count", estimate.packet_count);
  set_int("descriptor_count", estimate.descriptor_count);
  set_int("descriptor_fetch_cycles", estimate.descriptor_fetch_cycles);
  set_int("command_frontend_cycles", estimate.command_frontend_cycles);
  set_int("path_start_delay", estimate.path_start_delay);
  set_int("endpoint_completion_cycles", estimate.endpoint_completion_cycles);
  set_int("link_completion_cycles", estimate.link_completion_cycles);
  set_int("issue_interval_cycles", estimate.issue_interval_cycles);
  set_int("odma_channel", estimate.odma_channel);
  set_int("odma_channel_mask", estimate.odma_channel_mask);
  set_int("row_size", estimate.row_size);
  set_int("col_size", estimate.col_size);
  set_int("reduction_size", estimate.reduction_size);
  set_int("vector_chunks", estimate.vector_chunks);
  set_int("load_ops", estimate.load_ops);
  set_int("store_ops", estimate.store_ops);
  set_int("arithmetic_ops", estimate.arithmetic_ops);
  set_int("compare_convert_ops", estimate.compare_convert_ops);
  set_int("nonlinear_ops", estimate.nonlinear_ops);
  set_int("reduce_ops", estimate.reduce_ops);
  set_int("galu_ops", estimate.galu_ops);
  set_int("exact_static_submodel", estimate.exact_static_submodel ? 1 : 0);
  set_int("used_fallback", estimate.used_fallback ? 1 : 0);
  set_int("solver_scale_factor", estimate.solver_scale_factor);
  set_int("solver_latency_cycles", estimate.solver_latency_cycles);
  set_int("solver_scaled", estimate.solver_scaled ? 1 : 0);
  set_int("endpoint_timing_modeled", estimate.endpoint_timing_modeled ? 1 : 0);
  set_int("resource_overlap_modeled",
          estimate.resource_overlap_modeled ? 1 : 0);
  set_int("command_frontend_modeled",
          estimate.command_frontend_modeled ? 1 : 0);
  set_int("descriptor_plan_modeled", estimate.descriptor_plan_modeled ? 1 : 0);
  set_int("link_completion_modeled", estimate.link_completion_modeled ? 1 : 0);
  return result;
}

} // namespace

CostEstimate
CostModel::EstimateVectorCoreDelayGem5A4E(const tir::Stmt &stmt,
                                          const CostModelContext &context) {
  if (!context.target.defined())
    return UnsupportedGem5("vector", "missing_target_register_bits");
  int64_t register_bits =
      GetSunmmioTileProcessorConfig(context.target).register_bits;
  if (register_bits <= 0)
    return UnsupportedGem5("vector", "invalid_target_register_bits");
  VectorLatencyStats stats;
  Gem5VectorLatencyAnalyzer analyzer(register_bits);
  if (!analyzer.Analyze(stmt, &stats))
    return UnsupportedGem5("vector", stats.reason);

  CostEstimate estimate =
      MakeGem5Estimate("vector", CostEstimateStatus::kApproximate);
  estimate.latency_cycles = stats.cycles;
  estimate.issue_finish_cycles = stats.cycles;
  // A complete set of supported TIR nodes is still a conservative sum rather
  // than a cycle-accurate multi-FU schedule.  It is nevertheless a successful
  // gem5 profile estimate, not a heuristic fallback.
  estimate.latency_source = "BaseMinorCPU vector FU table; conservative_sum";
  estimate.timing_origin = "vector_tir_region_entry";
  estimate.timing_end = "conservative_fu_sum_end";
  estimate.vector_chunks = stats.chunks;
  estimate.load_ops = stats.load_ops;
  estimate.store_ops = stats.store_ops;
  estimate.arithmetic_ops = stats.arithmetic_ops;
  estimate.compare_convert_ops = stats.compare_convert_ops;
  estimate.nonlinear_ops = stats.nonlinear_ops;
  estimate.reduce_ops = stats.reduce_ops;
  estimate.galu_ops = stats.galu_ops;
  return estimate;
}

CostEstimate CostModel::Estimate(DeviceType device_type, const tir::Stmt &stmt,
                                 const CostModelContext &context) {
  if (context.options.strict_calibration &&
      context.options.profile != CostModelProfile::kGem5A4E2026) {
    LOG(FATAL) << kSunmmioCostModelStrict << " requires " << kSunmmioCostModel
               << "=gem5_a4e_2026";
  }
  CostEstimate estimate;
  if (context.options.profile == CostModelProfile::kHeuristic) {
    estimate.status = CostEstimateStatus::kApproximate;
    estimate.latency_cycles = static_cast<int64_t>(
        std::ceil(EstimateDelayHeuristic(device_type, stmt)));
    estimate.issue_finish_cycles = estimate.latency_cycles;
    estimate.profile = ProfileName(context.options.profile);
    estimate.model_version = ModelVersion(context.options.profile);
    estimate.latency_source = "legacy_tilelang_heuristic";
  } else if (device_type == DeviceType::TensorCore) {
    estimate = EstimateTensorCoreDelayGem5A4E(stmt, context);
  } else if (device_type == DeviceType::ODMA) {
    estimate = EstimateODMADelayGem5A4E(stmt, context);
  } else if (device_type == DeviceType::VectorCore) {
    estimate = EstimateVectorCoreDelayGem5A4E(stmt, context);
  } else {
    estimate = UnsupportedGem5("unspecified", "unsupported_device_type");
  }

  if (context.options.profile == CostModelProfile::kGem5A4E2026 &&
      estimate.status != CostEstimateStatus::kUnsupported &&
      estimate.latency_cycles <= 0) {
    estimate.status = CostEstimateStatus::kUnsupported;
    estimate.fallback_reason = "non_positive_latency";
  }

  if (context.options.profile == CostModelProfile::kGem5A4E2026 &&
      estimate.status == CostEstimateStatus::kUnsupported) {
    if (context.options.strict_calibration ||
        !context.options.fallback_to_heuristic) {
      LOG(FATAL) << kGem5A4EProfileName << " cost model cannot estimate "
                 << estimate.op_kind << ": " << estimate.fallback_reason;
    }
    ffi::String reason = estimate.fallback_reason;
    ffi::String op_kind = estimate.op_kind;
    estimate = MakeGem5Estimate("unspecified", CostEstimateStatus::kFallback);
    estimate.latency_cycles = static_cast<int64_t>(
        std::ceil(EstimateDelayHeuristic(device_type, stmt)));
    estimate.issue_finish_cycles = estimate.latency_cycles;
    estimate.latency_source = "legacy_tilelang_heuristic";
    estimate.fallback_reason = reason;
    estimate.op_kind = op_kind;
  }

  if (context.options.strict_calibration &&
      estimate.status != CostEstimateStatus::kExactStatic) {
    LOG(FATAL) << "Strict Gem5 calibration requires an exact static submodel; "
               << estimate.op_kind << " is sourced from "
               << estimate.latency_source;
  }

  estimate.exact_static_submodel =
      estimate.status == CostEstimateStatus::kExactStatic;
  estimate.used_fallback = estimate.status == CostEstimateStatus::kFallback;
  // Solver scaling belongs to the planner.  Keep these legacy diagnostics in
  // raw-cycle form until a planner deliberately overwrites them.
  estimate.solver_scale_factor = 1;
  estimate.solver_latency_cycles = estimate.latency_cycles;
  estimate.solver_scaled = false;
  return estimate;
}

float CostModel::EstimateDelay(DeviceType device_type, const tir::Stmt &stmt) {
  return EstimateDelayHeuristic(device_type, stmt);
}

float CostModel::EstimateDelay(DeviceType device_type, const tir::Stmt &stmt,
                               const CostModelOptions &options) {
  CostModelContext context;
  context.options = options;
  return static_cast<float>(
      Estimate(device_type, stmt, context).latency_cycles);
}

float CostModel::EstimateDelay(DeviceType device_type, const tir::Stmt &stmt,
                               const CostModelContext &context) {
  return static_cast<float>(
      Estimate(device_type, stmt, context).latency_cycles);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tl.sunmmio.EstimateCostForTesting",
      [](int device_type, Stmt stmt, ffi::String profile,
         bool fallback_to_heuristic) {
        CostModelContext context;
        context.options.profile = ParseProfile(profile);
        context.options.fallback_to_heuristic = fallback_to_heuristic;
        context.options.strict_calibration =
            tvm::transform::PassContext::Current()
                ->GetConfig<Bool>(kSunmmioCostModelStrict, Bool(false))
                .value();
        context.target = Target::Current(true);
        return EstimateToStringMap(CostModel::Estimate(
            static_cast<DeviceType>(device_type), stmt, context));
      });
  refl::GlobalDef().def(
      "tl.sunmmio.EstimateCostWithLayoutsForTesting",
      [](int device_type, Stmt stmt, ffi::Map<Buffer, Layout> layout_map,
         ffi::Map<Buffer, Layout> global_layout_map, ffi::String profile,
         bool fallback_to_heuristic) {
        CostModelContext context;
        context.options.profile = ParseProfile(profile);
        context.options.fallback_to_heuristic = fallback_to_heuristic;
        context.options.strict_calibration =
            tvm::transform::PassContext::Current()
                ->GetConfig<Bool>(kSunmmioCostModelStrict, Bool(false))
                .value();
        context.layout_map = std::move(layout_map);
        context.global_layout_map = std::move(global_layout_map);
        context.target = Target::Current(true);
        return EstimateToStringMap(CostModel::Estimate(
            static_cast<DeviceType>(device_type), stmt, context));
      });
}

} // namespace tl
} // namespace tvm
