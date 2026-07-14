/*!
 * \file substitute_with_buffer_predicates.cc
 * \brief Predicate-aware wrappers around TIR variable substitution.
 */

#include "substitute_with_buffer_predicates.h"

#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

#include <utility>

namespace tvm {
namespace tl {
namespace {

using namespace tir;

class BufferPredicateSubstituter : public StmtExprMutator {
public:
  explicit BufferPredicateSubstituter(
      const ffi::Map<Var, PrimExpr> &substitutions)
      : substitutions_(substitutions) {}

  Stmt Rewrite(const Stmt &stmt) {
    return VisitStmt(tir::Substitute(stmt, substitutions_));
  }

  PrimExpr Rewrite(const PrimExpr &expr) {
    return VisitExpr(tir::Substitute(expr, substitutions_));
  }

private:
  ffi::Optional<PrimExpr>
  RewritePredicate(const ffi::Optional<PrimExpr> &predicate) {
    if (!predicate.defined()) {
      return std::nullopt;
    }
    PrimExpr substituted = tir::Substitute(predicate.value(), substitutions_);
    return VisitExpr(substituted);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    ffi::Optional<PrimExpr> predicate = RewritePredicate(op->predicate);
    if (predicate.same_as(load->predicate)) {
      return load;
    }
    load.CopyOnWrite()->predicate = std::move(predicate);
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    ffi::Optional<PrimExpr> predicate = RewritePredicate(op->predicate);
    if (predicate.same_as(store->predicate)) {
      return store;
    }
    store.CopyOnWrite()->predicate = std::move(predicate);
    return store;
  }

  ffi::Map<Var, PrimExpr> substitutions_;
};

} // namespace

tir::Stmt SubstituteWithBufferPredicates(
    const tir::Stmt &stmt, const ffi::Map<tir::Var, PrimExpr> &substitutions) {
  return BufferPredicateSubstituter(substitutions).Rewrite(stmt);
}

PrimExpr SubstituteWithBufferPredicates(
    const PrimExpr &expr, const ffi::Map<tir::Var, PrimExpr> &substitutions) {
  return BufferPredicateSubstituter(substitutions).Rewrite(expr);
}

} // namespace tl
} // namespace tvm
