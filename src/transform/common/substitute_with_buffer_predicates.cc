/*!
 * \file substitute_with_buffer_predicates.cc
 * \brief Compatibility wrappers around TIR variable substitution.
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
      const ffi::Map<Var, PrimExpr> &substitutions,
      bool rewrite_annotations = false)
      : substitutions_(substitutions),
        rewrite_annotations_(rewrite_annotations) {}

  Stmt Rewrite(const Stmt &stmt) {
    return VisitStmt(tir::Substitute(stmt, substitutions_));
  }

  PrimExpr Rewrite(const PrimExpr &expr) {
    return VisitExpr(tir::Substitute(expr, substitutions_));
  }

private:
  struct AnnotationRewriteResult {
    ffi::Any value;
    bool changed{false};
  };

  AnnotationRewriteResult RewriteAnnotationValue(const ffi::Any &value) {
    if (auto expr = value.as<PrimExpr>()) {
      PrimExpr rewritten = Rewrite(expr.value());
      return {rewritten, !rewritten.same_as(expr.value())};
    }

    if (auto array = value.as<ffi::Array<PrimExpr>>()) {
      ffi::Array<PrimExpr> rewritten_array;
      bool changed = false;
      for (const PrimExpr &item : array.value()) {
        PrimExpr rewritten = Rewrite(item);
        rewritten_array.push_back(rewritten);
        changed = changed || !rewritten.same_as(item);
      }
      return {changed ? ffi::Any(rewritten_array) : value, changed};
    }

    if (auto array = value.as<ffi::Array<ffi::Any>>()) {
      ffi::Array<ffi::Any> rewritten_array;
      bool changed = false;
      for (const ffi::Any &item : array.value()) {
        AnnotationRewriteResult rewritten = RewriteAnnotationValue(item);
        rewritten_array.push_back(rewritten.value);
        changed = changed || rewritten.changed;
      }
      return {changed ? ffi::Any(rewritten_array) : value, changed};
    }

    if (auto map = value.as<ffi::Map<ffi::String, PrimExpr>>()) {
      ffi::Map<ffi::String, PrimExpr> rewritten_map;
      bool changed = false;
      for (const auto &[key, map_value] : map.value()) {
        PrimExpr rewritten = Rewrite(map_value);
        rewritten_map.Set(key, rewritten);
        changed = changed || !rewritten.same_as(map_value);
      }
      return {changed ? ffi::Any(rewritten_map) : value, changed};
    }

    if (auto map = value.as<ffi::Map<ffi::String, ffi::Any>>()) {
      ffi::Map<ffi::String, ffi::Any> rewritten_map;
      bool changed = false;
      for (const auto &[key, map_value] : map.value()) {
        AnnotationRewriteResult rewritten = RewriteAnnotationValue(map_value);
        rewritten_map.Set(key, rewritten.value);
        changed = changed || rewritten.changed;
      }
      return {changed ? ffi::Any(rewritten_map) : value, changed};
    }

    if (auto map = value.as<ffi::Map<ffi::Any, ffi::Any>>()) {
      ffi::Map<ffi::Any, ffi::Any> rewritten_map;
      bool changed = false;
      for (const auto &[key, map_value] : map.value()) {
        AnnotationRewriteResult rewritten_key = RewriteAnnotationValue(key);
        AnnotationRewriteResult rewritten_value =
            RewriteAnnotationValue(map_value);
        rewritten_map.Set(rewritten_key.value, rewritten_value.value);
        changed = changed || rewritten_key.changed || rewritten_value.changed;
      }
      return {changed ? ffi::Any(rewritten_map) : value, changed};
    }

    return {value, false};
  }

  ffi::Map<ffi::String, ffi::Any>
  RewriteAnnotations(const ffi::Map<ffi::String, ffi::Any> &annotations,
                     bool *changed) {
    ffi::Map<ffi::String, ffi::Any> rewritten_annotations;
    for (const auto &[key, value] : annotations) {
      AnnotationRewriteResult rewritten = RewriteAnnotationValue(value);
      rewritten_annotations.Set(key, rewritten.value);
      *changed = *changed || rewritten.changed;
    }
    return *changed ? rewritten_annotations : annotations;
  }

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

  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    if (!rewrite_annotations_ || loop->annotations.empty()) {
      return loop;
    }
    bool changed = false;
    auto annotations = RewriteAnnotations(loop->annotations, &changed);
    if (changed) {
      loop.CopyOnWrite()->annotations = std::move(annotations);
    }
    return loop;
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    if (!rewrite_annotations_ || block->annotations.empty()) {
      return block;
    }
    bool changed = false;
    auto annotations = RewriteAnnotations(block->annotations, &changed);
    if (changed) {
      block.CopyOnWrite()->annotations = std::move(annotations);
    }
    return block;
  }

  ffi::Map<Var, PrimExpr> substitutions_;
  bool rewrite_annotations_{false};
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

tir::Stmt SubstituteWithAnnotationsAndBufferPredicates(
    const tir::Stmt &stmt, const ffi::Map<tir::Var, PrimExpr> &substitutions) {
  return BufferPredicateSubstituter(substitutions,
                                    /*rewrite_annotations=*/true)
      .Rewrite(stmt);
}

} // namespace tl
} // namespace tvm
