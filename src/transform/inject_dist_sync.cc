/*!
 * \file tl/transform/inject_dist_sync.cc
 * \brief Materialize Rank communication generation updates after scheduling.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/transform.h>

#include "../op/utils.h"
#include "dist_transform_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;
using namespace dist_transform;

namespace {

class InjectDistSignalAdvance : public StmtExprMutator {
public:
  static PrimFunc Run(PrimFunc func) {
    InjectDistSignalAdvance injector;
    Stmt body = injector(func->body);
    if (!body.same_as(func->body)) {
      func.CopyOnWrite()->body = std::move(body);
    }
    return func;
  }

private:
  Stmt VisitStmt_(const EvaluateNode *evaluate_node) final {
    const auto *call = evaluate_node->value.as<CallNode>();
    if (!call) {
      return StmtExprMutator::VisitStmt_(evaluate_node);
    }
    if (call->op.same_as(dist_put_())) {
      ICHECK_EQ(call->args.size(), 6U)
          << "tl.dist_put_ must have its stable peer signature before "
             "InjectDistSync";
      BufferLoad generation = RequireLocalState(call->args[5], "generation");
      PrimExpr next = Cast(generation->dtype,
                           generation + make_const(generation->dtype, 1));
      Stmt advance = BufferStore(generation->buffer, next, generation->indices);
      return SeqStmt::Flatten(
          Array<Stmt>{advance, tvm::ffi::GetRef<Stmt>(evaluate_node)});
    }
    if (call->op.same_as(dist_signal_put_())) {
      ICHECK_EQ(call->args.size(), 4U);
      BufferLoad generation = RequireLocalState(call->args[3], "generation");
      PrimExpr next = Cast(generation->dtype,
                           generation + make_const(generation->dtype, 1));
      Stmt advance = BufferStore(generation->buffer, next, generation->indices);
      return SeqStmt::Flatten(
          Array<Stmt>{advance, tvm::ffi::GetRef<Stmt>(evaluate_node)});
    }
    if (call->op.same_as(dist_expect_())) {
      ICHECK_EQ(call->args.size(), 2U);
      BufferLoad expected = RequireLocalState(call->args[0], "expected");
      PrimExpr delta = Cast(expected->dtype, call->args[1]);
      PrimExpr next = Cast(expected->dtype, expected + delta);
      return BufferStore(expected->buffer, next, expected->indices);
    }
    if (call->op.same_as(dist_completion_init_())) {
      return InitializeCompletion(call);
    }
    if (call->op.same_as(dist_wait_any_())) {
      Var index("dist_wait_any_index", DataType::Int(32));
      return LetStmt(index, evaluate_node->value,
                     ConsumeWaitAny(call, index, Evaluate(0)));
    }
    return StmtExprMutator::VisitStmt_(evaluate_node);
  }

  Stmt VisitStmt_(const LetStmtNode *let_node) final {
    const auto *call = let_node->value.as<CallNode>();
    if (!call || !call->op.same_as(dist_wait_any_())) {
      return StmtExprMutator::VisitStmt_(let_node);
    }
    ICHECK_EQ(call->args.size(), 7U);
    Stmt body = VisitStmt(let_node->body);
    return LetStmt(let_node->var, let_node->value,
                   ConsumeWaitAny(call, let_node->var, body), let_node->span);
  }

  Stmt ConsumeWaitAny(const CallNode *call, const PrimExpr &index, Stmt body) {
    ICHECK_EQ(call->args.size(), 7U);
    // args[6] is compiler-only destination metadata. Completion state is
    // entirely materialized from expect/pending state in this pass.
    BufferRegion pending = NormalizeToBufferRegion(call->args[4]);
    BufferLoad pending_count =
        RequireLocalState(call->args[5], "pending count");
    PrimExpr pending_index = pending->region[0]->min + index;
    return SeqStmt::Flatten(Array<Stmt>{
        BufferStore(pending->buffer, make_zero(pending->buffer->dtype),
                    {pending_index}),
        BufferStore(pending_count->buffer,
                    pending_count - make_const(pending_count->dtype, 1),
                    pending_count->indices),
        body,
    });
  }

  Stmt InitializeCompletion(const CallNode *call) {
    ICHECK_EQ(call->args.size(), 7U);
    BufferRegion signal_expect = NormalizeToBufferRegion(call->args[0]);
    BufferRegion completion_expect = NormalizeToBufferRegion(call->args[1]);
    BufferRegion pending = NormalizeToBufferRegion(call->args[2]);
    BufferLoad pending_count =
        RequireLocalState(call->args[3], "pending count");
    BufferRegion recv_counts = NormalizeToBufferRegion(call->args[4]);
    std::string domain = RequireStringImm(call->args[5], "completion domain");
    ICHECK(domain == "rank" || domain == "row");
    PrimExpr rank_id = call->args[6];

    ICHECK_EQ(signal_expect->region.size(), 1U);
    ICHECK_EQ(completion_expect->region.size(), 1U);
    ICHECK_EQ(pending->region.size(), 1U);
    const auto *count_imm = signal_expect->region[0]->extent.as<IntImmNode>();
    ICHECK(count_imm);
    int64_t endpoint_count = count_imm->value;
    const auto *completion_count =
        completion_expect->region[0]->extent.as<IntImmNode>();
    const auto *pending_extent = pending->region[0]->extent.as<IntImmNode>();
    ICHECK(completion_count && pending_extent);
    ICHECK_EQ(completion_count->value, endpoint_count);
    ICHECK_EQ(pending_extent->value, endpoint_count);

    bool row_domain = domain == "row";
    ICHECK_EQ(recv_counts->region.size(), row_domain ? 2U : 1U);
    int64_t row_count = 1;
    if (row_domain) {
      const auto *row_extent = recv_counts->region[1]->extent.as<IntImmNode>();
      ICHECK(row_extent);
      row_count = row_extent->value;
    }

    Array<Stmt> statements;
    PrimExpr total = IntImm(DataType::Int(32), 0);
    for (int64_t index = 0; index < endpoint_count; ++index) {
      int64_t source_rank = row_domain ? index / row_count : index;
      int64_t source_row = row_domain ? index % row_count : 0;
      Array<PrimExpr> count_indices{recv_counts->region[0]->min +
                                    IntImm(DataType::Int(32), source_rank)};
      if (row_domain) {
        count_indices.push_back(recv_counts->region[1]->min +
                                IntImm(DataType::Int(32), source_row));
      }
      PrimExpr recv_count = BufferLoad(recv_counts->buffer, count_indices);
      PrimExpr active = And(rank_id != IntImm(rank_id.dtype(), source_rank),
                            recv_count > make_zero(recv_count.dtype()));
      PrimExpr signal_index =
          signal_expect->region[0]->min + IntImm(DataType::Int(32), index);
      PrimExpr completion_index =
          completion_expect->region[0]->min + IntImm(DataType::Int(32), index);
      PrimExpr pending_index =
          pending->region[0]->min + IntImm(DataType::Int(32), index);
      PrimExpr next = Cast(signal_expect->buffer->dtype,
                           BufferLoad(signal_expect->buffer, {signal_index}) +
                               Cast(signal_expect->buffer->dtype, active));
      statements.push_back(
          BufferStore(completion_expect->buffer, next, {completion_index}));
      statements.push_back(
          BufferStore(signal_expect->buffer,
                      BufferLoad(completion_expect->buffer, {completion_index}),
                      {signal_index}));
      statements.push_back(BufferStore(pending->buffer,
                                       Cast(pending->buffer->dtype, active),
                                       {pending_index}));
      total = total + Cast(DataType::Int(32), active);
    }
    statements.push_back(
        BufferStore(pending_count->buffer, total, pending_count->indices));
    return SeqStmt::Flatten(statements);
  }

  BufferLoad RequireLocalState(const PrimExpr &expr, const char *name) {
    const auto *load = expr.as<BufferLoadNode>();
    ICHECK(load) << "T.dist " << name << " state must be a BufferLoad, got "
                 << expr;
    ICHECK((load->dtype.is_int() || load->dtype.is_uint()) &&
           (load->dtype.bits() == 8 || load->dtype.bits() == 32))
        << "T.dist " << name << " state must use an 8- or 32-bit integer dtype";
    return tvm::ffi::GetRef<BufferLoad>(load);
  }
};

PrimFunc Run(PrimFunc func) {
  if (!IsMultiRank(func)) {
    return func;
  }
  DistOpDetector detector(/*high_level=*/false);
  if (!detector.Detect(func->body)) {
    return func;
  }
  return InjectDistSignalAdvance::Run(std::move(func));
}

} // namespace

tvm::transform::Pass InjectDistSync() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) {
    return Run(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectDistSync", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectDistSync", InjectDistSync);
}

} // namespace tl
} // namespace tvm
