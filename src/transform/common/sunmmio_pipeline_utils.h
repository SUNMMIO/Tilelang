#ifndef SUNMMIO_PIPELINE_UTILS_H
#define SUNMMIO_PIPELINE_UTILS_H

#include <string>
#include <unordered_set>

#include <tvm/arith/analyzer.h>
#include <tvm/arith/pattern.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tl {
inline int name2iter(const std::string &name) {
  return std::stoi(name.substr(0, name.find('-')));
}

inline int name2id(const std::string &name) {
  return std::stoi(name.substr(name.find('-') + 1));
}

inline ffi::Map<tir::Var, PrimExpr>
BuildPipelineIterZeroSubstitutionMap(const PrimExpr &expr,
                                     const tir::Var &pipeline_loop_var) {
  std::unordered_set<const tir::VarNode *> vars;
  tir::PostOrderVisit(expr, [&](const ObjectRef &obj) {
    if (const auto *var = obj.as<tir::VarNode>()) {
      if (!pipeline_loop_var.defined() || var != pipeline_loop_var.get()) {
        vars.insert(var);
      }
    }
  });

  ffi::Map<tir::Var, PrimExpr> vmap;
  for (const tir::VarNode *node : vars) {
    tir::Var var = ffi::GetRef<tir::Var>(node);
    vmap.Set(var, tir::make_zero(var.dtype()));
  }
  return vmap;
}

inline int DetectPipelineIterOffsetFromExpr(const PrimExpr &expr,
                                            const tir::Var &pipeline_loop_var,
                                            arith::Analyzer *analyzer) {
  if (!pipeline_loop_var.defined() ||
      !tir::UsesVar(expr,
                    [v = pipeline_loop_var.get()](const tir::VarNode *node) {
                      return node == v;
                    })) {
    return 0;
  }

  PrimExpr loop_only = expr;
  ffi::Map<tir::Var, PrimExpr> vmap =
      BuildPipelineIterZeroSubstitutionMap(expr, pipeline_loop_var);
  if (!vmap.empty()) {
    loop_only = tir::Substitute(loop_only, vmap);
  }
  loop_only = analyzer->Simplify(loop_only);

  ffi::Array<PrimExpr> coeffs = arith::DetectLinearEquation(
      loop_only, ffi::Array<tir::Var>{pipeline_loop_var});
  if (coeffs.size() != 2) {
    return 0;
  }

  PrimExpr coeff = analyzer->Simplify(coeffs[0]);
  PrimExpr base = analyzer->Simplify(coeffs[1]);
  const auto *coeff_int = coeff.as<IntImmNode>();
  if (coeff_int == nullptr || coeff_int->value == 0) {
    return 0;
  }

  PrimExpr offset_expr = analyzer->Simplify(floordiv(base, coeff));
  PrimExpr remainder = analyzer->Simplify(floormod(base, coeff));
  if (!analyzer->CanProveEqual(remainder, tir::make_zero(remainder.dtype()))) {
    return 0;
  }

  const auto *offset_int = offset_expr.as<IntImmNode>();
  return offset_int == nullptr ? 0 : static_cast<int>(offset_int->value);
}

inline int DetectPipelineIterOffsetFromRegion(const tir::BufferRegion &region,
                                              const tir::Var &pipeline_loop_var,
                                              arith::Analyzer *analyzer) {
  int result = 0;
  bool found = false;
  for (const Range &range : region->region) {
    bool uses_loop_var =
        pipeline_loop_var.defined() &&
        tir::UsesVar(range->min,
                     [v = pipeline_loop_var.get()](const tir::VarNode *node) {
                       return node == v;
                     });
    if (!uses_loop_var) {
      continue;
    }
    int dim_offset = DetectPipelineIterOffsetFromExpr(
        range->min, pipeline_loop_var, analyzer);
    if (!found) {
      result = dim_offset;
      found = true;
    } else if (result != dim_offset) {
      return 0;
    }
  }
  return found ? result : 0;
}

} // namespace tl
} // namespace tvm
#endif
