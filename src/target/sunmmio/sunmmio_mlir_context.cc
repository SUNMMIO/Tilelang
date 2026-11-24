#include "sunmmio_mlir_context.h"

#include "../../layout/cute_layout.h"

#include "npuir/Dialect/SUVM/IR/Interfaces.h"
#include "npuir/Dialect/SUVM/IR/Ops.h"
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <limits>

namespace tvm {
namespace codegen {

SunmmioMlirContext::SunmmioMlirContext() : builder(&mlir_ctx) {}

SunmmioMlirContext::PendingSyncState
SunmmioMlirContext::MergePendingSyncStates(const PendingSyncState &lhs,
                                           const PendingSyncState &rhs) {
  PendingSyncState merged = lhs;
  for (const auto &[op, units] : rhs.operations) {
    merged.operations[op] |= units;
  }
  return merged;
}

void SunmmioMlirContext::AddPendingSyncOperation(mlir::Operation *op,
                                                 SyncUnitMask units) {
  ICHECK(op) << "Cannot record a null pending synchronization operation";
  pending_sync_state.operations[op] |= units;
}

SunmmioMlirContext::SyncUnitMask SunmmioMlirContext::PendingSyncUnits() const {
  SyncUnitMask units = kNoSyncUnits;
  for (const auto &entry : pending_sync_state.operations) {
    units |= entry.second;
  }
  return units;
}

bool SunmmioMlirContext::HasPendingSyncUnits(SyncUnitMask units) const {
  return (PendingSyncUnits() & units) != kNoSyncUnits;
}

void SunmmioMlirContext::CompletePendingSyncUnits(SyncUnitMask units) {
  for (auto it = pending_sync_state.operations.begin();
       it != pending_sync_state.operations.end();) {
    if ((units & it->second) == it->second) {
      it = pending_sync_state.operations.erase(it);
    } else {
      ++it;
    }
  }
}

void SunmmioMlirContext::CompleteConsumedTileLoads(mlir::Operation *consumer) {
  if (!consumer || !mlir::isa<mlir::suvm::TileCompute>(consumer)) {
    return;
  }
  // A direct TileCompute data dependency completes the corresponding tile load.
  // Shape-only operations do not implement this interface and remain pending.
  for (mlir::Value operand : consumer->getOperands()) {
    auto load = operand.getDefiningOp<mlir::suvm::TileLoadOp>();
    if (load && operand == load.getResult()) {
      pending_sync_state.operations.erase(load.getOperation());
    }
  }
}

namespace {

bool IsGlobalScope(const tir::Buffer &buffer) {
  return buffer.scope().empty() || buffer.scope() == "global";
}

// A declared buffer and its backing buffer share a data variable. Adapt the
// inherited layout only for a full, contiguous alias at the same origin.
bool CanAdaptDataAliasLayout(const tir::Buffer &source,
                             const tl::Layout &layout,
                             const tir::Buffer &view) {
  if (source->dtype != view->dtype || view->shape.empty())
    return false;

  arith::Analyzer analyzer;
  const auto source_shape = layout->InputShape();
  if (source_shape.empty() ||
      !tl::IsSameLayout(layout, tl::sunmmio::MakeRowMajor(source_shape),
                        &analyzer) ||
      !analyzer.CanProveEqual(source->elem_offset, 0) ||
      !analyzer.CanProveEqual(view->elem_offset, 0))
    return false;

  const DataType i64 = DataType::Int(64);
  // A metadata-free DRAM buffer can inherit a row-major layout even when
  // its explicit TIR strides include padding. Singleton axes never advance.
  if (!source->strides.empty()) {
    PrimExpr expected_stride = tir::make_const(i64, 1);
    for (size_t i = source->shape.size(); i > 0; --i) {
      const PrimExpr &extent = source->shape[i - 1];
      if (!analyzer.CanProveEqual(tir::Cast(i64, source->strides[i - 1]),
                                  expected_stride) &&
          !analyzer.CanProveEqual(extent, 1))
        return false;
      expected_stride = expected_stride * tir::Cast(i64, extent);
    }
  }

  PrimExpr view_size = tir::make_const(i64, 1);
  for (size_t i = view->shape.size(); i > 0; --i) {
    if (!view->strides.empty() &&
        !analyzer.CanProveEqual(tir::Cast(i64, view->strides[i - 1]),
                                view_size))
      return false;
    view_size = view_size * tir::Cast(i64, view->shape[i - 1]);
  }
  PrimExpr source_size = tir::make_const(i64, 1);
  for (const PrimExpr &extent : source_shape)
    source_size = source_size * tir::Cast(i64, extent);
  return analyzer.CanProveEqual(source_size, view_size);
}

ffi::Optional<tl::Layout>
LookupLayoutInMap(const SunmmioMlirContext::TirLayoutMap &layout_map,
                  const tir::Buffer &buffer) {
  if (layout_map.count(buffer)) {
    return layout_map[buffer];
  }
  ffi::Optional<tl::Layout> data_match;
  for (const auto &[candidate, layout] : layout_map) {
    if (!candidate->data.same_as(buffer->data))
      continue;
    if (!IsGlobalScope(buffer))
      return layout;
    if (layout->InputDim() == buffer->shape.size()) {
      if (!data_match.defined())
        data_match = layout;
      continue;
    }
    // The map can list a same-rank alias before its backing buffer. Every
    // rank-changing match must be compatible with the declared view.
    if (!CanAdaptDataAliasLayout(candidate, layout, buffer))
      return layout;
    data_match = tl::sunmmio::MakeRowMajor(buffer->shape);
  }
  if (data_match.defined())
    return data_match;
  ffi::Optional<tl::Layout> name_match;
  int name_match_count = 0;
  for (const auto &kv : layout_map) {
    const tir::Buffer &candidate = kv.first;
    if (candidate->name == buffer->name) {
      name_match = kv.second;
      ++name_match_count;
    }
  }
  ICHECK_LE(name_match_count, 1)
      << "SunMMIO layout lookup found multiple buffers named `" << buffer->name
      << "`; use data-var or exact buffer mapping to disambiguate";
  if (name_match.defined()) {
    return name_match;
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
  ClearFunctionState();
  ClearLayoutScopes();
  module = nullptr;
}

} // namespace codegen
} // namespace tvm
