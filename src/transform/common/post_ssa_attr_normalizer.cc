#include "post_ssa_attr_normalizer.h"

#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <vector>

namespace tvm {
namespace tl {

using namespace ffi;
namespace tir = tvm::tir;

namespace {

constexpr const char *kTensorMeta = "tensor_meta";

struct AttrRemapResult {
  Any value;
  bool changed{false};
};

void AddUniqueVar(std::vector<tir::Var> *vars, const tir::Var &var) {
  for (const tir::Var &existing : *vars) {
    if (existing.same_as(var)) {
      return;
    }
  }
  vars->push_back(var);
}

std::vector<tir::Var> CollectVars(const PrimExpr &expr) {
  std::vector<tir::Var> vars;
  tir::PostOrderVisit(expr, [&](const ObjectRef &node) {
    if (const auto *var = node.as<tir::VarNode>()) {
      AddUniqueVar(&vars, ffi::GetRef<tir::Var>(var));
    }
  });
  return vars;
}

void AddVarRemap(const tir::Var &old_var, const tir::Var &new_var,
                 Map<tir::Var, PrimExpr> *var_remap) {
  if (old_var.same_as(new_var)) {
    return;
  }
  if (var_remap->count(old_var)) {
    return;
  }
  var_remap->Set(old_var, new_var);
}

void CollectVarRemapFromExpr(const PrimExpr &old_expr, const PrimExpr &new_expr,
                             Map<tir::Var, PrimExpr> *var_remap) {
  std::vector<tir::Var> old_vars = CollectVars(old_expr);
  std::vector<tir::Var> new_vars = CollectVars(new_expr);
  if (old_vars.empty() || new_vars.empty()) {
    return;
  }

  for (const tir::Var &old_var : old_vars) {
    std::vector<tir::Var> candidates;
    for (const tir::Var &new_var : new_vars) {
      if (old_var->name_hint == new_var->name_hint &&
          old_var.dtype() == new_var.dtype()) {
        candidates.push_back(new_var);
      }
    }
    if (candidates.size() == 1) {
      AddVarRemap(old_var, candidates[0], var_remap);
    }
  }

  if (old_vars.size() == new_vars.size()) {
    for (size_t i = 0; i < old_vars.size(); ++i) {
      if (old_vars[i].dtype() == new_vars[i].dtype()) {
        AddVarRemap(old_vars[i], new_vars[i], var_remap);
      }
    }
  }
}

void CollectVarRemapFromBuffer(const tir::Buffer &old_buffer,
                               const tir::Buffer &new_buffer,
                               Map<tir::Var, PrimExpr> *var_remap) {
  AddVarRemap(old_buffer->data, new_buffer->data, var_remap);

  if (old_buffer->shape.size() == new_buffer->shape.size()) {
    for (size_t i = 0; i < old_buffer->shape.size(); ++i) {
      CollectVarRemapFromExpr(old_buffer->shape[i], new_buffer->shape[i],
                              var_remap);
    }
  }
  if (old_buffer->strides.size() == new_buffer->strides.size()) {
    for (size_t i = 0; i < old_buffer->strides.size(); ++i) {
      CollectVarRemapFromExpr(old_buffer->strides[i], new_buffer->strides[i],
                              var_remap);
    }
  }
  if (old_buffer->elem_offset.defined() && new_buffer->elem_offset.defined()) {
    CollectVarRemapFromExpr(old_buffer->elem_offset, new_buffer->elem_offset,
                            var_remap);
  }
}

Map<tir::Var, PrimExpr> DerivePostSSAVarRemap(const tir::PrimFunc &old_func,
                                              const tir::PrimFunc &new_func) {
  Map<tir::Var, PrimExpr> var_remap;
  size_t num_params =
      std::min(old_func->params.size(), new_func->params.size());
  for (size_t i = 0; i < num_params; ++i) {
    const tir::Var &old_param = old_func->params[i];
    const tir::Var &new_param = new_func->params[i];
    if (old_param.dtype() == new_param.dtype()) {
      AddVarRemap(old_param, new_param, &var_remap);
    }

    auto old_buffer_it = old_func->buffer_map.find(old_param);
    auto new_buffer_it = new_func->buffer_map.find(new_param);
    if (old_buffer_it != old_func->buffer_map.end() &&
        new_buffer_it != new_func->buffer_map.end()) {
      CollectVarRemapFromBuffer((*old_buffer_it).second,
                                (*new_buffer_it).second, &var_remap);
    }
  }
  return var_remap;
}

AttrRemapResult RemapTensorMetaAny(const Any &value,
                                   const Map<tir::Var, PrimExpr> &var_remap) {
  if (auto expr = value.as<PrimExpr>()) {
    PrimExpr remapped = tir::Substitute(expr.value(), var_remap);
    return AttrRemapResult{remapped, !remapped.same_as(expr.value())};
  }

  if (auto array = value.as<Array<Any>>()) {
    Array<Any> remapped_array;
    bool changed = false;
    for (const Any &item : array.value()) {
      AttrRemapResult remapped_item = RemapTensorMetaAny(item, var_remap);
      remapped_array.push_back(remapped_item.value);
      changed = changed || remapped_item.changed;
    }
    return AttrRemapResult{changed ? Any(remapped_array) : value, changed};
  }

  if (auto map = value.as<Map<Any, Any>>()) {
    Map<Any, Any> remapped_map;
    bool changed = false;
    for (const auto &[key, map_value] : map.value()) {
      AttrRemapResult remapped_key = RemapTensorMetaAny(key, var_remap);
      AttrRemapResult remapped_value = RemapTensorMetaAny(map_value, var_remap);
      remapped_map.Set(remapped_key.value, remapped_value.value);
      changed = changed || remapped_key.changed || remapped_value.changed;
    }
    return AttrRemapResult{changed ? Any(remapped_map) : value, changed};
  }

  return AttrRemapResult{value, false};
}

tir::PrimFunc NormalizeTensorMetaAfterSSA(const tir::PrimFunc &old_func,
                                          tir::PrimFunc new_func) {
  if (!new_func->attrs.defined()) {
    return new_func;
  }
  auto tensor_meta_it = new_func->attrs->dict.find(kTensorMeta);
  if (tensor_meta_it == new_func->attrs->dict.end()) {
    return new_func;
  }

  Map<tir::Var, PrimExpr> var_remap = DerivePostSSAVarRemap(old_func, new_func);
  if (var_remap.empty()) {
    return new_func;
  }

  AttrRemapResult remapped =
      RemapTensorMetaAny((*tensor_meta_it).second, var_remap);
  if (!remapped.changed) {
    return new_func;
  }
  return WithAttrs(std::move(new_func),
                   {{String(kTensorMeta), remapped.value}});
}

} // namespace

IRModule NormalizePostSSAAttrs(const IRModule &before_ssa, IRModule after_ssa) {
  IRModule updates = IRModule(Map<GlobalVar, BaseFunc>({}));
  for (const auto &[gvar, base_func] : after_ssa->functions) {
    auto new_func_opt = base_func.as<tir::PrimFunc>();
    if (!new_func_opt || !before_ssa->functions.count(gvar)) {
      continue;
    }
    auto old_func_opt = before_ssa->functions[gvar].as<tir::PrimFunc>();
    if (!old_func_opt) {
      continue;
    }

    tir::PrimFunc new_func = new_func_opt.value();
    tir::PrimFunc normalized =
        NormalizeTensorMetaAfterSSA(old_func_opt.value(), new_func);
    if (!normalized.same_as(new_func)) {
      updates->Add(gvar, normalized);
    }
  }
  if (!updates->functions.empty()) {
    after_ssa.CopyOnWrite()->Update(updates);
  }
  return after_ssa;
}

} // namespace tl
} // namespace tvm
