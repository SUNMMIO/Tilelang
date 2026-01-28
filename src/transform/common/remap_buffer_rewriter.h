/*!
 * \file remap_buffer_rewriter.h
 */
#ifndef TVM_TL_REAMP_BUFFER_REWRITER_H_
#define TVM_TL_REAMP_BUFFER_REWRITER_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>
#include <unordered_map>
#include <vector>

#include "arith/ir_mutator_with_analyzer.h"

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief A class that rewrites buffer references in a statement based on a
 * given buffer remapping.
 *
 * This class is used to update buffer references in a statement after buffer
 * transformations have been applied. It specifically handles the remapping of
 * padding annotations.
 */
class RemapBufferRewriter : public arith::IRMutatorWithAnalyzer {
public:
  /*!
   * \brief Substitute buffer references in a statement based on a given buffer
   * remapping. \param stmt The statement to rewrite. \param buffer_remap A map
   * from old buffers to new buffers. \return The rewritten statement.
   */
  static Stmt Substitute(const Stmt &stmt, Map<Buffer, Buffer> buffer_remap) {
    arith::Analyzer analyzer;
    RemapBufferRewriter substituter(&analyzer);
    substituter.buffer_remap_ = std::move(buffer_remap);
    return substituter.VisitStmt(stmt);
  }

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  Stmt VisitStmt_(const BlockNode *op) final {
    if (op->annotations.count(attr::kSafeValueMap)) {
      return RewritePaddingMap(op);
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  /*!
   * \brief Rewrite the padding map annotation of a block.
   * \param op The block node to rewrite.
   * \return The rewritten block.
   */
  Stmt RewritePaddingMap(const BlockNode *op) {
    auto safe_value_map = op->annotations.Get(attr::kSafeValueMap);
    if (!safe_value_map) {
      LOG(FATAL) << "Padding map annotation is missing";
    }

    Map<Var, Var> var_remap = CreateVarRemap();
    Map<Var, PrimExpr> new_safe_value_map = RemapPaddingMap(
        Downcast<Map<Var, PrimExpr>>(safe_value_map.value()), var_remap);

    auto block = Downcast<Block>(IRMutatorWithAnalyzer::VisitStmt_(op));
    auto block_ptr = block.CopyOnWrite();
    block_ptr->annotations.Set(attr::kSafeValueMap, new_safe_value_map);
    return block;
  }

  /*!
   * \brief Create a mapping from old variables to new variables based on buffer
   * remapping. \return A map from old variables to new variables.
   */
  Map<Var, Var> CreateVarRemap() const {
    Map<Var, Var> var_remap;
    for (const auto &[buffer, buffer_remap] : buffer_remap_) {
      var_remap.Set(buffer->data, buffer_remap->data);
    }
    return var_remap;
  }

  /*!
   * \brief Remap the padding map using the variable remapping.
   * \param safe_value_map The original padding map.
   * \param var_remap The variable remapping.
   * \return The remapped padding map.
   */
  Map<Var, PrimExpr> RemapPaddingMap(const Map<Var, PrimExpr> &safe_value_map,
                                     const Map<Var, Var> &var_remap) const {
    Map<Var, PrimExpr> new_safe_value_map;
    for (const auto &[var, padding] : safe_value_map) {
      if (var_remap.count(var)) {
        new_safe_value_map.Set(var_remap.at(var), padding);
      } else {
        new_safe_value_map.Set(var, padding);
      }
    }
    return new_safe_value_map;
  }

  Map<Buffer, Buffer> buffer_remap_;
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_THREAD_BOUND_KEY_H_
