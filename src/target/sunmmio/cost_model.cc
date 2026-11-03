/*!
 * \file cost_model.cc
 * \brief Implementation of CostModel.
 */

#include "cost_model.h"
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
    if (op->op.same_as(Op::Get("tir.exp2")) ||
        op->op.same_as(Op::Get("tir.exp"))) {
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
  BufferRegion d_region = call->args.size() == 9U
                              ? NormalizeToBufferRegion(call->args[7])
                              : c_region;
  DataType input_dtype = a_region->buffer->dtype;
  DataType rhs_dtype = b_region->buffer->dtype;
  DataType output_dtype = d_region->buffer->dtype;

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
    if (const auto *let_stmt = stmt.as<LetStmtNode>()) {
      VectorStmtFlopCost value_cost = AnalyzeExpr(let_stmt->value);
      VectorStmtFlopCost body_cost = AnalyzeStmt(let_stmt->body);
      return {value_cost.repeated + body_cost.repeated,
              value_cost.once + body_cost.once};
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

float CostModel::EstimateDelayHeuristic(DeviceType device_type,
                                        const tir::Stmt &stmt) {
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

constexpr int64_t kGem5TensorFirstResultDelay = 34;
constexpr int64_t kGem5TensorCompletionDelay = 3;
constexpr int64_t kGem5ODMALocalCompletionDelay = 3;
constexpr int64_t kGem5ODMADramToRsramCompletionExtra = 1;
constexpr int64_t kGem5ODMALinkCompletionDelay = 4;
constexpr int64_t kGem5ODMARsramToLinkAsramCompletionExtra = 1;
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
  uint64_t bit_width = static_cast<uint64_t>(region->buffer->dtype.bits()) *
                       region->buffer->dtype.lanes();
  uint64_t bytes_per_element = (bit_width + 7) / 8;
  if (!CheckedMulU64(elements, bytes_per_element, bytes)) {
    *reason = "byte_count_overflow";
    return false;
  }
  if (*bytes == 0 || *bytes > static_cast<uint64_t>(INT64_MAX)) {
    *reason = *bytes == 0 ? "zero_bytes" : "byte_count_overflow";
    return false;
  }
  return true;
}

CostEstimate UnsupportedGem5(const char *op_kind, const std::string &reason) {
  CostEstimate estimate;
  estimate.profile = "gem5_a4e_2026";
  estimate.model_version = "gem5_a4e_2026@a7ecd393";
  estimate.op_kind = op_kind;
  estimate.fallback_reason = reason;
  return estimate;
}

int TensorDTypeId(DataType dtype) {
  if (dtype.is_float() && dtype.bits() == 32) {
    return 0;
  }
  if (dtype.is_bfloat16()) {
    return 1;
  }
  if (dtype.bits() == 8) {
    return 2;
  }
  if (dtype.bits() == 6) {
    return 4;
  }
  if (dtype.bits() == 4) {
    return 8;
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
                                       int st_dtype, uint64_t row, uint64_t col,
                                       uint64_t acc) {
  const bool act_bf16 = act_dtype == 1;
  const bool ld_bf16 = ld_dtype == 1;
  const bool st_bf16 = st_dtype == 1;
  uint64_t row_col = 0;
  ICHECK(CheckedMulU64(row, col, &row_col));
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
  return profile == CostModelProfile::kGem5A4E2026 ? "gem5_a4e_2026"
                                                   : "heuristic";
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
  bool default_preserve =
      context.options.profile == CostModelProfile::kGem5A4E2026;
  context.options.preserve_absolute_cycles =
      pass_ctx
          ->GetConfig<Bool>(kSunmmioCostModelPreserveCycles,
                            Bool(default_preserve))
          .value();
  context.explicit_solver_scale = pass_ctx->GetConfig<Integer>(kSunmmioFaster);
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
  const auto *row_imm = a->args[2].as<IntImmNode>();
  const auto *acc_imm = a->args[3].as<IntImmNode>();
  // Match NPU-IR getTcInitConfig(): TC_COL_SIZE comes from the store tile,
  // not the weight tile, which is transposed by qkv_summa_gemm.
  const auto *col_imm = c->args[3].as<IntImmNode>();
  const auto *clear_imm = call->args[5].as<IntImmNode>();
  if (!row_imm || !acc_imm || !col_imm)
    return UnsupportedGem5("tensor_mma", "dynamic_tensor_shape");
  if (row_imm->value <= 0 || col_imm->value <= 0 || acc_imm->value <= 0)
    return UnsupportedGem5("tensor_mma", "non_positive_tensor_shape");

  BufferRegion a_region = NormalizeToBufferRegion(call->args[0]);
  BufferRegion b_region = NormalizeToBufferRegion(call->args[1]);
  BufferRegion c_region = NormalizeToBufferRegion(call->args[2]);
  BufferRegion d_region = call->args.size() == 9U
                              ? NormalizeToBufferRegion(call->args[7])
                              : c_region;
  DataType act_type = a_region->buffer->dtype;
  DataType rhs_type = b_region->buffer->dtype;
  DataType st_type = c_region->buffer->dtype;
  if (call->args.size() == 9U) {
    const auto *store_output_imm = call->args[8].as<IntImmNode>();
    if (store_output_imm && store_output_imm->value != 0) {
      st_type = d_region->buffer->dtype;
    }
  }
  if (act_type != rhs_type)
    return UnsupportedGem5("tensor_mma", "mismatched_input_dtype");
  int act_dtype = TensorDTypeId(act_type);
  int st_dtype = TensorDTypeId(st_type);
  if (act_dtype < 0 || st_dtype < 0)
    return UnsupportedGem5("tensor_mma", "unsupported_tensor_dtype");

  uint64_t row = static_cast<uint64_t>(row_imm->value);
  uint64_t col = static_cast<uint64_t>(col_imm->value);
  uint64_t acc = static_cast<uint64_t>(acc_imm->value);
  // NPU-IR lowers a logical BF16 activation tile as two tc.mma operations
  // and writes half of its logical row extent to TC_ROW_SIZE.
  if (act_dtype == 1) {
    if ((row & 1U) != 0)
      return UnsupportedGem5("tensor_mma", "bf16_tensor_row_not_even");
    row /= 2;
  }
  // A pipelined reduction commonly uses clear_accum=(bk == 0).  The planner
  // assigns one latency to the command template before bk is substituted, so
  // use the conservative accumulate latency for a dynamic clear condition.
  // Injection later specializes every logical bk to a constant boolean.
  bool preload_result = clear_imm == nullptr || clear_imm->value == 0;
  int ld_dtype = TensorDTypeId(c_region->buffer->dtype);
  if (ld_dtype < 0)
    return UnsupportedGem5("tensor_mma", "unsupported_accumulator_dtype");
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
    delay += TensorBackpressureExtraCycles(act_dtype, ld_dtype, st_dtype, row,
                                           col, acc);
  }
  if (delay > static_cast<uint64_t>(INT64_MAX))
    return UnsupportedGem5("tensor_mma", "tensor_latency_overflow");

  CostEstimate estimate;
  estimate.latency_cycles = static_cast<int64_t>(delay);
  estimate.issue_finish_cycles = estimate.latency_cycles;
  estimate.exact_static_submodel = true;
  estimate.profile = "gem5_a4e_2026";
  estimate.model_version = "gem5_a4e_2026@a7ecd393";
  estimate.latency_source = "TensorCore::calculateOpDelayCycles";
  estimate.op_kind = "tensor_mma";
  estimate.input_dtype = DTypeName(act_type);
  estimate.output_dtype = DTypeName(st_type);
  estimate.row_size = static_cast<int64_t>(row);
  estimate.col_size = static_cast<int64_t>(col);
  estimate.reduction_size = static_cast<int64_t>(acc);
  return estimate;
}

CostEstimate
CostModel::EstimateODMADelayGem5A4E(const tir::Stmt &stmt,
                                    const CostModelContext &context) {
  (void)context;
  SunmmioAsyncOpInfo info;
  if (!TryParseSunmmioAsyncODMAStmt(stmt, &info))
    return UnsupportedGem5("odma", "unrecognized_async_odma_op");
  uint64_t bytes = 0;
  std::string reason;
  if (!GetStaticRegionBytes(info.src, &bytes, &reason))
    return UnsupportedGem5("odma", reason);

  CostEstimate estimate;
  estimate.profile = "gem5_a4e_2026";
  estimate.model_version = "gem5_a4e_2026@a7ecd393";
  estimate.bytes = static_cast<int64_t>(bytes);
  estimate.src_scope =
      IsGlobalBuffer(info.src->buffer) ? "global" : info.src->buffer.scope();
  estimate.dst_scope =
      IsGlobalBuffer(info.dst->buffer) ? "global" : info.dst->buffer.scope();
  estimate.input_dtype = DTypeName(info.src->buffer->dtype);
  estimate.output_dtype = DTypeName(info.dst->buffer->dtype);
  estimate.endpoint_timing_modeled = false;

  if (info.kind == SunmmioAsyncOpKind::kLayoutTransform) {
    return UnsupportedGem5("layout_transform",
                           context.layout_map.empty() &&
                                   context.global_layout_map.empty()
                               ? "missing_layout_context"
                               : "missing_layout_or_descriptor_mapping");
  }

  if (info.kind == SunmmioAsyncOpKind::kTranspose) {
    if (info.src->buffer.scope() != "shared.rsram" ||
        info.dst->buffer.scope() != "shared.rsram")
      return UnsupportedGem5("transpose", "transpose_requires_rsram");
    if (info.src->region.size() != 2)
      return UnsupportedGem5("transpose", "transpose_requires_rank_2");
    const auto *m = info.src->region[0]->extent.as<IntImmNode>();
    const auto *n = info.src->region[1]->extent.as<IntImmNode>();
    if (!m || !n)
      return UnsupportedGem5("transpose", "dynamic_transpose_shape");
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
    if (last_issue > static_cast<uint64_t>(INT64_MAX))
      return UnsupportedGem5("transpose", "transpose_latency_overflow");
    estimate.op_kind = "transpose";
    estimate.transfer_path = "rsram_to_rsram_transpose";
    estimate.path_start_delay = static_cast<int64_t>(first_issue);
    estimate.packet_count = static_cast<int64_t>(packets);
    estimate.issue_start_cycles = static_cast<int64_t>(first_issue);
    estimate.issue_finish_cycles = static_cast<int64_t>(last_issue);
    estimate.latency_cycles = estimate.issue_finish_cycles;
    estimate.latency_source = "ODMA::procTransposeLSram2LSram";
    estimate.exact_static_submodel = true;
    return estimate;
  }

  const bool src_global = IsGlobalBuffer(info.src->buffer);
  const bool dst_global = IsGlobalBuffer(info.dst->buffer);
  const ffi::String &src_scope = info.src->buffer.scope();
  const ffi::String &dst_scope = info.dst->buffer.scope();
  const bool src_rsram = src_scope == "shared.rsram" || src_scope == "local";
  const bool dst_rsram = dst_scope == "shared.rsram" || dst_scope == "local";
  const bool dst_asram = dst_scope == "shared.asram";
  const bool dst_wsram = dst_scope == "shared.wsram";
  const bool src_local_sram =
      src_rsram || src_scope == "shared.asram" || src_scope == "shared.wsram";
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
    } else if (src_local_sram && dst_global) {
      start = 7;
      estimate.transfer_path = "local_sram_to_dram";
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

  int64_t issue_finish = latency;
  latency += endpoint_completion;
  estimate.path_start_delay = start;
  estimate.packet_count = packet_count;
  estimate.issue_start_cycles = start;
  estimate.issue_finish_cycles = issue_finish;
  estimate.latency_cycles = latency;
  estimate.exact_static_submodel = true;
  estimate.endpoint_timing_modeled = endpoint_completion > 0;
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
  result.Set("profile", estimate.profile);
  result.Set("model_version", estimate.model_version);
  result.Set("latency_source", estimate.latency_source);
  result.Set("fallback_reason", estimate.fallback_reason);
  result.Set("op_kind", estimate.op_kind);
  result.Set("transfer_path", estimate.transfer_path);
  result.Set("src_scope", estimate.src_scope);
  result.Set("dst_scope", estimate.dst_scope);
  result.Set("input_dtype", estimate.input_dtype);
  result.Set("output_dtype", estimate.output_dtype);
  set_int("latency_cycles", estimate.latency_cycles);
  set_int("issue_start_cycles", estimate.issue_start_cycles);
  set_int("issue_finish_cycles", estimate.issue_finish_cycles);
  set_int("bytes", estimate.bytes);
  set_int("packet_count", estimate.packet_count);
  set_int("path_start_delay", estimate.path_start_delay);
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

  CostEstimate estimate;
  estimate.latency_cycles = stats.cycles;
  estimate.issue_finish_cycles = stats.cycles;
  estimate.exact_static_submodel = false;
  // A complete set of supported TIR nodes is still a conservative sum rather
  // than a cycle-accurate multi-FU schedule.  It is nevertheless a successful
  // gem5 profile estimate, not a heuristic fallback.
  estimate.profile = "gem5_a4e_2026";
  estimate.model_version = "gem5_a4e_2026@a7ecd393";
  estimate.latency_source = "BaseMinorCPU vector FU table; conservative_sum";
  estimate.op_kind = "vector";
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
  CostEstimate estimate;
  if (context.options.profile == CostModelProfile::kHeuristic) {
    estimate.latency_cycles = static_cast<int64_t>(
        std::ceil(EstimateDelayHeuristic(device_type, stmt)));
    estimate.issue_finish_cycles = estimate.latency_cycles;
    estimate.profile = "heuristic";
    estimate.model_version = "heuristic_v1";
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
      estimate.latency_cycles <= 0) {
    if (!context.options.fallback_to_heuristic) {
      LOG(FATAL) << "gem5_a4e_2026 cost model cannot estimate "
                 << estimate.op_kind << ": " << estimate.fallback_reason;
    }
    ffi::String reason = estimate.fallback_reason;
    ffi::String op_kind = estimate.op_kind;
    estimate = CostEstimate{};
    estimate.latency_cycles = static_cast<int64_t>(
        std::ceil(EstimateDelayHeuristic(device_type, stmt)));
    estimate.issue_finish_cycles = estimate.latency_cycles;
    estimate.profile = "gem5_a4e_2026";
    estimate.model_version = "gem5_a4e_2026@a7ecd393";
    estimate.latency_source = "legacy_tilelang_heuristic";
    estimate.fallback_reason = reason;
    estimate.op_kind = op_kind;
    estimate.used_fallback = true;
  }

  estimate.solver_latency_cycles = estimate.latency_cycles;
  if (context.explicit_solver_scale.defined() &&
      context.explicit_solver_scale.value()->value > 1) {
    estimate.solver_scale_factor = context.explicit_solver_scale.value()->value;
    estimate.solver_latency_cycles =
        CeilDivI64(estimate.latency_cycles, estimate.solver_scale_factor);
    estimate.solver_scaled = true;
  }
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
        context.explicit_solver_scale =
            tvm::transform::PassContext::Current()->GetConfig<Integer>(
                kSunmmioFaster);
        context.target = Target::Current(true);
        return EstimateToStringMap(CostModel::Estimate(
            static_cast<DeviceType>(device_type), stmt, context));
      });
}

} // namespace tl
} // namespace tvm
