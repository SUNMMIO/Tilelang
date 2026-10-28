#ifndef TVM_TL_TRANSFORM_COMMON_MX_VIEW_H_
#define TVM_TL_TRANSFORM_COMMON_MX_VIEW_H_

#include "../../layout/layout.h"
#include "../../op/builtin.h"
#include <tvm/tir/stmt_functor.h>
#include <unordered_map>

namespace tvm::tl {

inline constexpr const char *kMXParent = "tl.mx_parent";
inline constexpr const char *kMXDeclaration = "tl.mx_declaration";
inline constexpr const char *kMXFieldLayout = "tl.mx_field_layout";
inline constexpr const char *kMXParentLayout = "tl.mx_parent_layout";
inline constexpr const char *kMXFieldOffset = "tl.mx_field_offset";
inline constexpr const char *kMXDeferredCopy = "tl.mx_deferred_copy";

struct MXViewInfo {
  tir::Buffer parent;
  int field;
  tir::Call binding;
};

using MXViewMap = std::unordered_map<const tir::VarNode *, MXViewInfo>;

bool IsMXFieldHandle(const PrimExpr &expr);
bool HasMXShapePlaceholder(const PrimExpr &expr);
MXViewMap CollectMXViews(const tir::Stmt &body);
ffi::Map<tir::Var, PrimExpr> CollectMXViewLets(const tir::Stmt &body);
tir::Var MXStorageRoot(const tir::Buffer &buffer, const MXViewMap &views);
bool ReadsMXField(const PrimExpr &expr, const MXViewMap &views,
                  const ffi::Map<tir::Var, PrimExpr> &lets = {});
void CheckMXViewPipeline(const tir::PrimFunc &func);

// Resolves shape expressions and metadata together, preserving Buffer identity.
tir::PrimFunc ResolveMXViewShapes(
    tir::PrimFunc func,
    const ffi::Map<tir::Var, ffi::Array<PrimExpr>> &scale_shapes,
    const ffi::Map<tir::Var, Layout> &field_layouts,
    const ffi::Map<tir::Var, Integer> &field_offsets);

// Repairs parent descriptors after scope rewriting and SSA cloning.
tir::PrimFunc NormalizeMXViewBindings(tir::PrimFunc func);
void ValidateMXViewDeclarations(const tir::PrimFunc &func);
void CheckMXViewResolved(const tir::PrimFunc &func);
// Normalizes each proven-negative copy boundary once, in its lexical context.
tir::PrimFunc NormalizeMXCopyIndices(tir::PrimFunc func);
tir::Call RewriteMXCopyAnnotations(
    tir::Call call, const std::function<PrimExpr(const PrimExpr &)> &rewrite);

} // namespace tvm::tl
#endif
