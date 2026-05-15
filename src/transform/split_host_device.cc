/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file split_host_device.cc
 * \brief Split device function from host.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/global_var_supply.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <functional>
#include <optional>
#include <utility>

#include "../layout/cute_layout.h"
#include "../layout/layout.h"
#include "../op/builtin.h"
#include "common/assume.h"
#include "tir/analysis/var_use_def_analysis.h"
#include "tvm/node/cast.h"
#include "tvm/runtime/logging.h"
#include "tvm/tir/stmt.h"

namespace tvm {
namespace tl {
using namespace ffi;
namespace tir = tvm::tir;

// This pass traverses the AST, split the target function into host part and
// device part and copies all assume attribute statements to the device side.

// 1. Traverse AST and collect all assume statements into host_assumes_.
// 2. Until the first AttrStmtNode with tvm::attr::kTarget.
// 3. Call SplitDeviceFunc, which will create a new device function and replace
//    the original body with a call to that function.
class HostDeviceSplitter : public tir::StmtMutator {
public:
  explicit HostDeviceSplitter(IRModule *device_mod,
                              std::function<GlobalVar()> var_supply)
      : device_mod_(device_mod), var_supply_(std::move(var_supply)) {}

  void SetNonRestrictParams(Optional<Array<tir::Var>> params) {
    for (auto param : params.value()) {
      non_restrict_params_.push_back(param);
    }
  }

  void SetExtraDeviceFuncAttrs(Map<String, Any> attrs) {
    extra_device_func_attrs_ = std::move(attrs);
  }

  tir::Stmt VisitStmt_(const tir::AttrStmtNode *op) final {
    if (op->attr_key == tvm::attr::kTarget) {
      found_device_region_ = true;
      auto device_target = op->node.as<tvm::Target>().value().WithoutHost();
      return SplitDeviceFunc(op->body, device_target);
    } else if (op->attr_key == tir::attr::tilelang_assume) {
      // NOTE(chaofan): the assumes collected here must be in host-side.
      //    This is because when the collector reaches the split region,
      //    it will start to split and return. For safety, we add a check here.
      ICHECK(!found_device_region_)
          << "Assumes collection should not be in device region.";
      // We first push back the outside assume, then visit the child.
      // So when moving assumes to device side, we need to do the building
      // process in a reverse order.
      host_assumes_.push_back(op);
    }
    return tir::StmtMutator::VisitStmt_(op);
  }

  tir::Stmt VisitStmt_(const tir::EvaluateNode *op) final {
    auto stmt = GetRef<tir::Stmt>(op);
    // There should be no assume in evaluate form after InjectAssumes.
    ICHECK(!IsAssumeInEvaluateForm(stmt))
        << "Unexpected assume in evaluate form. Please run InjectAssumes pass "
           "first.";
    return tir::StmtMutator::VisitStmt_(op);
  }

  tir::Stmt ForceSplit(tir::Stmt body, tvm::Target device_target) {
    return SplitDeviceFunc(std::move(body), std::move(device_target));
  }

  bool found_device_region() const { return found_device_region_; }

private:
  bool found_device_region_{false};
  Array<tir::Var> non_restrict_params_;

  // Wrap body with assumes, substituting variables in assumes with the
  // corresponding variables in the device body based on name_hint matching.
  // This substitution is necessary because host-side assume variables may be
  // different Var objects from device-side parameters, even if they have the
  // same name. We always perform substitution to ensure ConvertSSA sees
  // consistent variable references.
  Stmt wrapBodyWithHostSideAssumes(
      Stmt body, const std::unordered_map<std::string, tir::Var> &name_to_var) {
    // Build substitution map: assume_var -> body_var
    // Always substitute if we find a matching name, regardless of whether
    // it's the same object. This ensures ConvertSSA treats them as the same
    // variable.
    auto substitute_func =
        [&name_to_var](const tir::Var &var) -> Optional<PrimExpr> {
      auto it = name_to_var.find(var->name_hint);
      if (it != name_to_var.end()) {
        return it->second;
      }
      return Optional<PrimExpr>();
    };

    for (auto it = host_assumes_.rbegin(); it != host_assumes_.rend(); ++it) {
      // Substitute variables in the assume condition
      PrimExpr original_node = Downcast<PrimExpr>((*it)->node);
      PrimExpr substituted_node =
          tir::Substitute(original_node, substitute_func);
      body = AttrStmt(substituted_node, tir::attr::tilelang_assume,
                      (*it)->value, body);
    }
    return body;
  }

  static Layout RemapLayout(const Layout &layout,
                            const Map<tir::Var, PrimExpr> &var_remap) {
    auto substitute = [&](const PrimExpr &expr) {
      return tir::Substitute(expr, var_remap);
    };

    Array<PrimExpr> input_size = layout->InputShape().Map(
        [&](const PrimExpr &expr) { return substitute(expr); });
    Array<PrimExpr> forward_index = layout->GetForwardIndex().Map(
        [&](const PrimExpr &expr) { return substitute(expr); });

    if (auto cute = layout.as<CuteLayoutNode>()) {
      Array<PrimExpr> logical_shape = cute->GetLogicalShape().Map(
          [&](const PrimExpr &expr) { return substitute(expr); });
      Array<PrimExpr> mode_shape = cute->GetModeShape().Map(
          [&](const PrimExpr &expr) { return substitute(expr); });
      Array<PrimExpr> mode_stride = cute->GetModeStride().Map(
          [&](const PrimExpr &expr) { return substitute(expr); });
      return CuteLayout(logical_shape, mode_shape, mode_stride,
                        cute->GetDimLevels());
    }

    if (auto fragment = layout.as<FragmentNode>()) {
      Fragment remapped(input_size, forward_index,
                        substitute(fragment->GetForwardThread()),
                        substitute(fragment->ReplicateExtent()), std::nullopt);
      Range thread_range = fragment->ThreadRange();
      if (thread_range.defined()) {
        remapped = remapped->BindThreadRange(Range(
            substitute(thread_range->min), substitute(thread_range->extent)));
      }
      return remapped;
    }

    return Layout(input_size, forward_index);
  }

  Any RemapLayoutMapAttr(
      const String &key, const Any &value,
      const Map<tir::Var, PrimExpr> &var_remap,
      const std::function<tir::Buffer(const tir::Buffer &)> &remap_buffer) {
    if (auto layout_map = value.as<Map<tir::Buffer, Layout>>()) {
      Map<tir::Buffer, Layout> remapped_layout_map;
      for (const auto &[buffer, layout] : layout_map.value()) {
        remapped_layout_map.Set(remap_buffer(buffer),
                                RemapLayout(layout, var_remap));
      }
      return remapped_layout_map;
    }

    if (auto layout_map = value.as<Map<tir::Var, Layout>>()) {
      Map<tir::Var, Layout> remapped_layout_map;
      for (const auto &[var, layout] : layout_map.value()) {
        tir::Var remapped_var = var;
        if (var_remap.count(var)) {
          remapped_var = Downcast<tir::Var>(var_remap[var]);
        }
        remapped_layout_map.Set(remapped_var, RemapLayout(layout, var_remap));
      }
      return remapped_layout_map;
    }

    LOG(FATAL) << "Unsupported `" << key
               << "` attr type for device function propagation: "
               << value.GetTypeKey();
    return Any();
  }

  struct SimpleAttrRemapResult {
    Any value;
    bool changed{false};
  };

  static std::optional<SimpleAttrRemapResult> TryRemapSimpleScalarAttrValue(
      const Any &value, const Map<tir::Var, PrimExpr> &var_remap,
      const std::function<tir::Buffer(const tir::Buffer &)> &remap_buffer) {
    if (auto var = value.as<tir::Var>()) {
      tir::Var remapped_var = var.value();
      if (var_remap.count(var.value())) {
        remapped_var = Downcast<tir::Var>(var_remap[var.value()]);
      }
      return SimpleAttrRemapResult{remapped_var,
                                   !remapped_var.same_as(var.value())};
    }

    if (auto buffer = value.as<tir::Buffer>()) {
      tir::Buffer remapped_buffer = remap_buffer(buffer.value());
      return SimpleAttrRemapResult{remapped_buffer,
                                   !remapped_buffer.same_as(buffer.value())};
    }

    if (auto expr = value.as<PrimExpr>()) {
      PrimExpr remapped_expr = tir::Substitute(expr.value(), var_remap);
      return SimpleAttrRemapResult{remapped_expr,
                                   !remapped_expr.same_as(expr.value())};
    }

    return std::nullopt;
  }

  static std::optional<SimpleAttrRemapResult> TryRemapSimpleArrayAttrValue(
      const Any &value, const Map<tir::Var, PrimExpr> &var_remap,
      const std::function<tir::Buffer(const tir::Buffer &)> &remap_buffer) {
    if (auto array = value.as<Array<Any>>()) {
      Array<Any> remapped_array;
      bool changed = false;
      for (const Any &item : array.value()) {
        auto remapped_item =
            TryRemapSimpleScalarAttrValue(item, var_remap, remap_buffer);
        if (remapped_item) {
          remapped_array.push_back(remapped_item->value);
          changed = changed || remapped_item->changed;
        } else {
          remapped_array.push_back(item);
        }
      }
      if (changed) {
        return SimpleAttrRemapResult{remapped_array, true};
      }
      return SimpleAttrRemapResult{value, false};
    }

    return std::nullopt;
  }

  static std::optional<SimpleAttrRemapResult> TryRemapSimpleMapAttrValue(
      const Any &value, const Map<tir::Var, PrimExpr> &var_remap,
      const std::function<tir::Buffer(const tir::Buffer &)> &remap_buffer) {
    if (auto map = value.as<Map<Any, Any>>()) {
      Map<Any, Any> remapped_map;
      bool changed = false;
      for (const auto &[key, map_value] : map.value()) {
        auto remapped_key =
            TryRemapSimpleScalarAttrValue(key, var_remap, remap_buffer);
        auto remapped_value =
            TryRemapSimpleScalarAttrValue(map_value, var_remap, remap_buffer);
        const Any &new_key = remapped_key ? remapped_key->value : key;
        const Any &new_value =
            remapped_value ? remapped_value->value : map_value;
        remapped_map.Set(new_key, new_value);
        changed = changed || (remapped_key && remapped_key->changed) ||
                  (remapped_value && remapped_value->changed);
      }
      if (changed) {
        return SimpleAttrRemapResult{remapped_map, true};
      }
      return SimpleAttrRemapResult{value, false};
    }

    return std::nullopt;
  }

  static Any RemapSimpleAttrValue(
      const Any &value, const Map<tir::Var, PrimExpr> &var_remap,
      const std::function<tir::Buffer(const tir::Buffer &)> &remap_buffer) {
    if (auto scalar =
            TryRemapSimpleScalarAttrValue(value, var_remap, remap_buffer)) {
      return scalar->value;
    }
    if (auto array =
            TryRemapSimpleArrayAttrValue(value, var_remap, remap_buffer)) {
      return array->value;
    }
    if (auto map = TryRemapSimpleMapAttrValue(value, var_remap, remap_buffer)) {
      return map->value;
    }
    return value;
  }

  // Remap parameters. Special handling is applied to complex types based on
  // specific keys. Primitive types (Var, Buffer, PrimExpr) are substituted
  // directly. Container types (List, Map) are also remapped if they contain
  // these primitive types. For Maps, supports replacing only the keys or only
  // the values.
  Any RemapDeviceFuncAttr(
      const String &key, const Any &value,
      const Map<tir::Var, PrimExpr> &var_remap,
      const std::function<tir::Buffer(const tir::Buffer &)> &remap_buffer) {
    if (key == tl::attr::kLayoutMap || key == tl::attr::kGlobalLayoutMap) {
      return RemapLayoutMapAttr(key, value, var_remap, remap_buffer);
    }
    return RemapSimpleAttrValue(value, var_remap, remap_buffer);
  }

  tir::Stmt SplitDeviceFunc(tir::Stmt body, tvm::Target device_target) {
    // First, analyze undefined variables in the device body
    auto [old_params, buffers_to_declare] =
        [&]() -> std::tuple<Array<tir::Var>, Array<tir::Buffer>> {
      tir::VarUseDefAnalyzer use_def(/*defined_vars=*/{},
                                     /*visit_thread_extent=*/true);
      use_def(body);

      // Sort first by variable type, then by variable name
      std::vector<tir::Var> params{use_def.undefined_.begin(),
                                   use_def.undefined_.end()};
      std::sort(params.begin(), params.end(),
                [](const tir::Var &a, const tir::Var &b) {
                  auto sort_key = [](const tir::Var &var) {
                    return std::tuple{
                        !var->dtype.is_handle(),
                        var->name_hint,
                    };
                  };
                  return sort_key(a) < sort_key(b);
                });
      return {params, use_def.undefined_buffers_};
    }();

    // Create new parameter variables for the device function to avoid sharing
    // Var objects with the host function. This prevents ConvertSSA from
    // incorrectly renaming variables when it processes multiple functions.
    Array<tir::Var> params;
    Map<tir::Var, PrimExpr> var_remap;
    std::unordered_map<std::string, tir::Var> name_to_var;
    for (const auto &old_var : old_params) {
      tir::Var new_var(old_var->name_hint, old_var->type_annotation);
      params.push_back(new_var);
      var_remap.Set(old_var, new_var);
      name_to_var[old_var->name_hint] = new_var;
    }

    // Substitute old variables with new ones in the body
    body = tir::Substitute(body, var_remap);

    // Remap buffers to use new variables.  Keep a cache so DeclBuffer and
    // function attributes that reference the same old Buffer also share the
    // same remapped Buffer.
    Map<tir::Buffer, tir::Buffer> buffer_remap;
    std::function<tir::Buffer(const tir::Buffer &)> remap_buffer =
        [&](const tir::Buffer &buf) -> tir::Buffer {
      if (buffer_remap.count(buf)) {
        return buffer_remap[buf];
      }

      auto new_shape = buf->shape.Map(
          [&](const PrimExpr &e) { return tir::Substitute(e, var_remap); });
      auto new_strides = buf->strides.Map(
          [&](const PrimExpr &e) { return tir::Substitute(e, var_remap); });
      auto new_elem_offset = tir::Substitute(buf->elem_offset, var_remap);
      auto new_data = var_remap.count(buf->data)
                          ? Downcast<tir::Var>(var_remap[buf->data])
                          : buf->data;

      if (new_data.same_as(buf->data) && new_shape.same_as(buf->shape) &&
          new_strides.same_as(buf->strides) &&
          new_elem_offset.same_as(buf->elem_offset)) {
        return buf;
      }

      tir::Buffer new_buf(new_data, buf->dtype, new_shape, new_strides,
                          new_elem_offset, buf->name, buf->data_alignment,
                          buf->offset_factor, buf->buffer_type,
                          buf->axis_separators, buf->span);
      buffer_remap.Set(buf, new_buf);
      return new_buf;
    };

    Array<tir::Buffer> new_buffers_to_declare;
    for (const auto &buf : buffers_to_declare) {
      new_buffers_to_declare.push_back(remap_buffer(buf));
    }
    buffers_to_declare = new_buffers_to_declare;

    // CodeGenCPU is used for some device-side targets, such as
    // "ext_dev", and expects to be able to return a int32_t status
    // code.

    bool can_propagate_errors = [&]() {
      auto kind = device_target->GetTargetDeviceType();
      return kind == kDLCPU || kind == kDLExtDev || kind == kDLHexagon;
    }();
    IntImm success(DataType::Int(32), 0);
    Type kernel_ret_type;
    if (can_propagate_errors) {
      kernel_ret_type = PrimType(DataType::Int(32));
      body = tir::SeqStmt::Flatten(body, tir::Evaluate(ret(success)));
    } else {
      kernel_ret_type = VoidType();
    }

    // Declare necessary buffers for the device side.
    for (tir::Buffer buf : buffers_to_declare) {
      body = tir::DeclBuffer(buf, std::move(body));
    }

    // Copy assumes from host-side to device-side, with variable substitution.
    // This must be done after DeclBuffer so that assumes are at the outermost
    // level of the function body. This ensures ConvertSSA correctly identifies
    // that assume variables refer to function parameters.
    body = wrapBodyWithHostSideAssumes(body, name_to_var);

    // Remap non_restrict_params to use new parameter variables
    Array<tir::Var> remapped_non_restrict_params;
    for (const auto &old_var : non_restrict_params_) {
      if (var_remap.count(old_var)) {
        remapped_non_restrict_params.push_back(
            Downcast<tir::Var>(var_remap[old_var]));
      } else {
        remapped_non_restrict_params.push_back(old_var);
      }
    }

    tir::PrimFunc device_func(params, body, kernel_ret_type);
    device_func = WithAttrs(
        std::move(device_func),
        {{tvm::attr::kTarget, device_target},
         {tir::attr::kNoAlias, true},
         {tir::attr::kIsGlobalFunc, true},
         {tl::attr::kNonRestrictParams, remapped_non_restrict_params}});

    Map<String, Any> remapped_extra_device_func_attrs;
    for (const auto &[key, value] : extra_device_func_attrs_) {
      remapped_extra_device_func_attrs.Set(
          key, RemapDeviceFuncAttr(key, value, var_remap, remap_buffer));
    }
    device_func =
        WithAttrs(std::move(device_func), remapped_extra_device_func_attrs);

    GlobalVar kernel_symbol_global = var_supply_();
    (*device_mod_)->Add(kernel_symbol_global, device_func);
    // Use old_params as call arguments (host-side variables)
    Array<PrimExpr> args =
        old_params.Map([](const tir::Var &var) -> PrimExpr { return var; });

    if (can_propagate_errors) {
      tir::Var kernel_error_code("kernel_error_code", success->dtype);
      tir::Call kernel_call(success->dtype, kernel_symbol_global, args);
      tir::AssertStmt assert_success(
          kernel_error_code == success,
          tir::StringImm("Error executing compute kernel"), tir::Evaluate(0));
      tir::LetStmt let_check(kernel_error_code, kernel_call, assert_success);

      return let_check;

    } else {
      return tir::Evaluate(
          tir::Call(DataType::Void(), kernel_symbol_global, args));
    }
  }

  // target ir module
  IRModule *device_mod_;
  // Generate new GlobalVar for the kernel
  std::function<GlobalVar()> var_supply_;
  // Collect assumes in host side
  Array<const tir::AttrStmtNode *> host_assumes_;
  Map<String, Any> extra_device_func_attrs_;
};

tir::PrimFunc SplitHostDevice(tir::PrimFunc func, IRModule *device_mod,
                              std::function<GlobalVar()> var_supply) {
  HostDeviceSplitter splitter(device_mod, std::move(var_supply));
  // 保存需要保留到device_function的属性
  Map<String, Any> extra_device_func_attrs;
  if (func->attrs.defined()) {
    if (auto opt_keys =
            func->GetAttr<Array<String>>(tl::attr::kDeviceFuncAttrKeys)) {
      for (const String &key : opt_keys.value()) {
        if (key == tl::attr::kDeviceFuncAttrKeys) {
          continue;
        }
        if (auto it = func->attrs->dict.find(key);
            it != func->attrs->dict.end()) {
          extra_device_func_attrs.Set(key, (*it).second);
        }
      }
    }
  }
  splitter.SetExtraDeviceFuncAttrs(std::move(extra_device_func_attrs));

  // Propagate non-restrict parameter list from host func to device kernels
  if (auto opt = func->GetAttr<Array<tir::Var>>(tl::attr::kNonRestrictParams)) {
    splitter.SetNonRestrictParams(opt.value());
    // Remove the attribute from host-side PrimFunc; it only matters for device
    // codegen.
    func = tvm::WithoutAttr(std::move(func), tl::attr::kNonRestrictParams);
  }

  if (auto body = splitter(func->body); !body.same_as(func->body)) {
    func.CopyOnWrite()->body = body;
  } else if (!splitter.found_device_region()) {
    if (auto target = func->GetAttr<Target>(tvm::attr::kTarget)) {
      auto device_target = target.value().WithoutHost();
      if (device_target.defined() &&
          func->HasNonzeroAttr(tir::attr::kIsEntryFunc) &&
          tir::is_no_op(func->body)) {
        if (auto forced = splitter.ForceSplit(func->body, device_target);
            !forced.same_as(func->body)) {
          func.CopyOnWrite()->body = forced;
        }
      }
    }
  }
  if (func->attrs.defined() &&
      func->attrs->dict.count(tl::attr::kDeviceFuncAttrKeys)) {
    func = tvm::WithoutAttr(std::move(func), tl::attr::kDeviceFuncAttrKeys);
  }
  return func;
}

namespace transform {

tvm::transform::Pass SplitHostDevice() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext ctx) {
    tvm::GlobalVarSupply global_var_supply(mod);

    IRModule device_mod = IRModule(Map<GlobalVar, BaseFunc>({}));
    IRModule updates = IRModule(Map<GlobalVar, BaseFunc>({}));

    for (const auto &[gvar, base_func] : mod->functions) {
      if (auto opt = base_func.as<tir::PrimFunc>()) {
        tir::PrimFunc func = opt.value();

        auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
        auto name_prefix = global_symbol.value_or(gvar->name_hint);
        auto kernel_name = name_prefix + "_kernel";
        auto var_supply = [&global_var_supply, &kernel_name]() -> GlobalVar {
          return global_var_supply->FreshGlobal(kernel_name, false);
        };

        func = ::tvm::tl::SplitHostDevice(std::move(func), &device_mod,
                                          var_supply);
        if (!func.same_as(base_func)) {
          updates->Add(gvar, func);
        }
      }
    }
    mod->Update(updates);
    mod->Update(device_mod);
    return tir::transform::ConvertSSA()(mod);
  };

  return tvm::transform::CreateModulePass(pass_func, 0, "tl.SplitHostDevice",
                                          {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.SplitHostDevice", SplitHostDevice);
}

} // namespace transform
} // namespace tl
} // namespace tvm
