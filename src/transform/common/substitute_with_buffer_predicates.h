/*!
 * \file substitute_with_buffer_predicates.h
 * \brief Predicate-aware wrappers around TIR variable substitution.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_SUBSTITUTE_WITH_BUFFER_PREDICATES_H_
#define TVM_TL_TRANSFORM_COMMON_SUBSTITUTE_WITH_BUFFER_PREDICATES_H_

#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tl {

/*!
 * \brief Substitute variables throughout a statement, including predicates
 * attached to BufferLoad and BufferStore nodes.
 *
 * TVM's generic TIR visitors and mutators currently traverse buffer indices
 * and values but not buffer-level predicates. This compatibility helper first
 * applies TVM's standard substitution, preserving its buffer and attribute
 * remapping semantics, then repairs predicates recursively with the same map.
 */
tir::Stmt SubstituteWithBufferPredicates(
    const tir::Stmt &stmt, const ffi::Map<tir::Var, PrimExpr> &substitutions);

/*!
 * \brief Expression overload of SubstituteWithBufferPredicates.
 */
PrimExpr SubstituteWithBufferPredicates(
    const PrimExpr &expr, const ffi::Map<tir::Var, PrimExpr> &substitutions);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_SUBSTITUTE_WITH_BUFFER_PREDICATES_H_
