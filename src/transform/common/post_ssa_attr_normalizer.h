#pragma once

#include <tvm/ir/module.h>

namespace tvm {
namespace tl {

/*!
 * \brief Normalize TileLang-owned function attrs after TVM's ConvertSSA pass.
 *
 * TVM's SSA conversion owns the function body, params, and buffer metadata.
 * TileLang-owned attrs may also contain TIR Vars that must stay in the same
 * post-SSA scope. This helper repairs only attrs with explicit TileLang
 * semantics, leaving TVM attrs untouched.
 */
IRModule NormalizePostSSAAttrs(const IRModule &before_ssa, IRModule after_ssa);

} // namespace tl
} // namespace tvm
