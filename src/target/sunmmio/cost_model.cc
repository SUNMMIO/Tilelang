/*!
 * \file cost_model.cc
 * \brief Implementation of CostModel.
 */

#include "cost_model.h"
#include "../../op/comm.h"
#include "../../op/utils.h"

#include <algorithm>
#include <limits>
#include <tvm/ffi/container/array.h>
#include <tvm/runtime/logging.h>
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

float CostModel::EstimateTensorCoreDelay(const tir::Stmt &stmt) {
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

  const auto *A = call->args[0].as<CallNode>();
  const auto *B = call->args[1].as<CallNode>();
  ICHECK(A) << "TensorCore lhs fragment must be a CallNode: " << stmt;
  ICHECK(B) << "TensorCore rhs fragment must be a CallNode: " << stmt;
  ICHECK(A->args[2].as<IntImmNode>())
      << "TensorCore row size must be IntImm: " << stmt;
  ICHECK(B->args[3].as<IntImmNode>())
      << "TensorCore col size must be IntImm: " << stmt;
  ICHECK(A->args[3].as<IntImmNode>())
      << "TensorCore reduction size must be IntImm: " << stmt;

  BufferRegion a_region = NormalizeToBufferRegion(call->args[0]);
  BufferRegion b_region = NormalizeToBufferRegion(call->args[1]);
  BufferRegion c_region = NormalizeToBufferRegion(call->args[2]);
  DataType input_dtype = a_region->buffer->dtype;
  DataType rhs_dtype = b_region->buffer->dtype;
  DataType output_dtype = c_region->buffer->dtype;

  ICHECK(input_dtype.is_bfloat16())
      << "Only bf16 TensorCore input is supported in current cost model: "
      << stmt;
  ICHECK(rhs_dtype == input_dtype)
      << "TensorCore input dtypes must match for current cost model: " << stmt;

  auto row_size = A->args[2].as<IntImmNode>()->value;
  auto col_size = B->args[3].as<IntImmNode>()->value;
  auto acc_size = A->args[3].as<IntImmNode>()->value;

  /**
   * For bf16 TensorCore MMA, the effective inner-loop gap is bounded by both
   * the reduction depth and a minimum block count implied by the tile shape.
   * The accumulator dtype contributes a small fixed latency difference between
   * bf16 accumulation and fp32 accumulation.
   */
  int min_blk_num = (row_size <= 16 && col_size <= 32) ? 1 : 4;
  int gap = std::max(ceil_div(acc_size, 32), min_blk_num);
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
  return delay;
}

float CostModel::EstimateODMADelay(const tir::Stmt &stmt) {
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
      // The condition is evaluated whenever the surrounding stmt executes, so
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
    ICHECK(0) << "Unsupported VectorCore stmt shape in cost model: " << stmt;
    return {0.0f, 0.0f};
  }

  tvm::ffi::Array<PrimExpr> materialized_constants_;
};

float CostModel::EstimateVectorCoreDelay(const tir::Stmt &stmt) {
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

float CostModel::EstimateDelay(DeviceType device_type, const tir::Stmt &stmt) {
  if (device_type == DeviceType::TensorCore) {
    return EstimateTensorCoreDelay(stmt);
  } else if (device_type == DeviceType::ODMA) {
    return EstimateODMADelay(stmt);
  } else if (device_type == DeviceType::VectorCore) {
    return EstimateVectorCoreDelay(stmt);
  }
  return 0.0f;
}

} // namespace tl
} // namespace tvm
