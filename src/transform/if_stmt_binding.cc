/*!
 * \file if_stmt_binding.cc
 * \brief Bind the If Stmt to each Stmt in SeqStmt
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tir;

class IfStmtBindingRewriter : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc &f) {
    auto rewriter = IfStmtBindingRewriter();
    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

private:
  IfStmtBindingRewriter() = default;

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    auto condition = op->condition;
    auto then_case = VisitStmt(op->then_case);
    Optional<Stmt> else_case = op->else_case;
    if (else_case.defined()) {
      return tvm::ffi::GetRef<Stmt>(op);
    }
    ICHECK(then_case.defined()) << "then_case must be defined";
    ICHECK(!else_case.defined()) << "else_case must be undefined";

    const auto *seq_stmt = then_case.as<SeqStmtNode>();
    if (!seq_stmt || seq_stmt->seq.size() <= 1) {
      return IfThenElse(condition, then_case, Stmt(), op->span);
    }

    PrimExpr guard = condition;
    Optional<Var> condition_snapshot;
    // Repeated guards must preserve the original if's single evaluation.
    // Only replay pure conditions: this TIR implementation does not have the
    // upstream TIRX access analysis needed to prove read-only conditions
    // stable.
    if (SideEffect(condition) > CallEffectKind::kPure) {
      condition_snapshot = Var("if_condition", condition.dtype());
      guard = condition_snapshot.value();
    }

    Array<Stmt> guarded_stmts;
    for (const Stmt &stmt : seq_stmt->seq) {
      guarded_stmts.push_back(IfThenElse(guard, stmt, Stmt(), op->span));
    }
    Stmt body = SeqStmt(std::move(guarded_stmts), seq_stmt->span);
    if (condition_snapshot.defined()) {
      return LetStmt(condition_snapshot.value(), condition, body, op->span);
    }
    return body;
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> seq;
    for (auto stmt : op->seq) {
      seq.push_back(VisitStmt(stmt));
    }
    return SeqStmt(std::move(seq));
  }
};

using namespace tir::transform;
tvm::transform::Pass IfStmtBinding() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return IfStmtBindingRewriter::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.IfStmtBinding", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.IfStmtBinding", IfStmtBinding);
}

} // namespace tl
} // namespace tvm
