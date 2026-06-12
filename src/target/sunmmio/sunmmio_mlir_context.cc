#include "sunmmio_mlir_context.h"

#include "../../layout/cute_layout.h"

#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <limits>

namespace tvm {
namespace codegen {

SunmmioMlirContext::SunmmioMlirContext() : builder(&mlir_ctx) {}

namespace {

bool IsGlobalScope(const tir::Buffer &buffer) {
  return buffer.scope().empty() || buffer.scope() == "global";
}

ffi::Optional<tl::Layout>
LookupLayoutInMap(const SunmmioMlirContext::TirLayoutMap &layout_map,
                  const tir::Buffer &buffer) {
  if (layout_map.count(buffer)) {
    return layout_map[buffer];
  }
  for (const auto &kv : layout_map) {
    const tir::Buffer &candidate = kv.first;
    if (candidate->data.same_as(buffer->data) ||
        candidate->name == buffer->name) {
      return kv.second;
    }
  }
  return ffi::Optional<tl::Layout>();
}

} // namespace

void SunmmioMlirContext::ClearLayoutScopes() {
  layout_map_stack.clear();
  global_layout_map_stack.clear();
}

void SunmmioMlirContext::PushLayoutScope(
    const TirLayoutMap &layout_map, const TirLayoutMap &global_layout_map) {
  layout_map_stack.push_back(layout_map);
  global_layout_map_stack.push_back(global_layout_map);
}

void SunmmioMlirContext::PopLayoutScope() {
  ICHECK(!layout_map_stack.empty());
  ICHECK(!global_layout_map_stack.empty());
  layout_map_stack.pop_back();
  global_layout_map_stack.pop_back();
}

ffi::Optional<tl::Layout>
SunmmioMlirContext::LookupLayout(const tir::Buffer &buffer) const {
  auto lookup_stack =
      [&](const std::vector<TirLayoutMap> &stack) -> ffi::Optional<tl::Layout> {
    for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
      auto layout = LookupLayoutInMap(*it, buffer);
      if (layout.defined()) {
        return layout;
      }
    }
    return ffi::Optional<tl::Layout>();
  };

  if (IsGlobalScope(buffer)) {
    auto global_layout = lookup_stack(global_layout_map_stack);
    if (global_layout.defined()) {
      return global_layout;
    }
    return lookup_stack(layout_map_stack);
  }

  auto local_layout = lookup_stack(layout_map_stack);
  if (local_layout.defined()) {
    return local_layout;
  }
  return lookup_stack(global_layout_map_stack);
}

void SunmmioMlirContext::ApplyLayoutToType(const tir::Buffer &buffer,
                                           SunMMIOType *type) const {
  auto layout_opt = LookupLayout(buffer);
  if (!layout_opt.defined()) {
    return;
  }

  const tl::Layout &layout = layout_opt.value();
  const auto *cute = layout.as<tl::CuteLayoutNode>();
  ICHECK(cute) << "SunMMIO SUVM codegen expects CuteLayout for buffer "
               << buffer->name << ", got " << layout->DebugOutput();

  arith::Analyzer analyzer;
  type->layout_hshape.clear();
  type->layout_hstride.clear();
  type->layout_dim_levels.clear();

  for (const PrimExpr &dim : cute->GetModeShape()) {
    type->layout_hshape.push_back(analyzer.Simplify(dim));
  }
  for (const PrimExpr &stride : cute->GetModeStride()) {
    type->layout_hstride.push_back(analyzer.Simplify(stride));
  }
  for (const Integer &level : cute->GetDimLevels()) {
    int64_t value = level.IntValue();
    ICHECK_GT(value, 0) << "SunMMIO SUVM layout dim level must be > 0";
    ICHECK_LE(value, static_cast<int64_t>(std::numeric_limits<uint8_t>::max()))
        << "SunMMIO SUVM layout dim level does not fit uint8_t: " << value;
    type->layout_dim_levels.push_back(static_cast<uint8_t>(value));
  }
}

void SunmmioMlirContext::Clear() {
  mlir_value_table_stack.clear();
  token_by_id.clear();
  barrier_by_mask.clear();
  for_stack.clear();
  if_stack.clear();
  while_stack.clear();
  control_flow_stack.clear();
  ClearLayoutScopes();
  module = nullptr;
}

} // namespace codegen
} // namespace tvm
