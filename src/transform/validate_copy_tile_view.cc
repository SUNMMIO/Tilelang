/*!
 * \file validate_copy_tile_view.cc
 * \brief Validate copy regions that must be representable as Sunmmio tile
 * views.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../layout/cute_layout.h"
#include "../layout/layout.h"
#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/operator.h"
#include "../op/utils.h"
#include "../target/utils.h"
#include "common/attr.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

namespace {

class ValidateCopyTileViewPass : public StmtExprVisitor {
public:
  static PrimFunc Run(PrimFunc f) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsSunmmio(target.value())) {
      return f;
    }

    ValidateCopyTileViewPass validator;
    validator.VisitStmt(f->body);
    return f;
  }

private:
  struct DimLayoutInfo {
    bool has_layout{false};
    bool has_dynamic_outer_mode{false};
    int64_t coalesced_extent{1};
    int64_t inner_static_mode_shape{1};
  };

  struct ScopeFrame {
    LayoutMap layout_map;
    LayoutMap global_layout_map;
  };

  void VisitStmt_(const BlockNode *op) final {
    PushScope(op);
    StmtExprVisitor::VisitStmt_(op);
    scope_stack_.pop_back();
  }

  void VisitExpr_(const CallNode *op) final {
    if (IsCopyTileOp(op)) {
      ValidateCopyCall(op, "tl.tileop.copy", /*arg_base=*/0);
    } else if (IsDmaCopyOp(op)) {
      ValidateCopyCall(op, "tl.dma_copy", /*arg_base=*/0);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  static bool IsCopyTileOp(const CallNode *call) {
    return call->op.same_as(Copy::Get());
  }

  static bool IsDmaCopyOp(const CallNode *call) {
    return call->op.same_as(dma_copy());
  }

  void PushScope(const BlockNode *block) {
    ScopeFrame frame;
    if (!scope_stack_.empty()) {
      frame = scope_stack_.back();
    }

    CollectLayoutAnnotation(block, attr::kLayoutMap, &frame.layout_map);
    CollectLayoutAnnotation(block, attr::kGlobalLayoutMap,
                            &frame.global_layout_map);
    scope_stack_.push_back(std::move(frame));
  }

  void CollectLayoutAnnotation(const BlockNode *block, const char *attr_key,
                               LayoutMap *dst) {
    if (!block->annotations.count(attr_key)) {
      return;
    }

    auto layout_map_obj = block->annotations.Get(attr_key).value();
    if (auto layout_map = layout_map_obj.as<LayoutMap>()) {
      for (const auto &[buffer, layout] : layout_map.value()) {
        dst->Set(buffer, layout);
      }
      return;
    }

    if (auto layout_map = layout_map_obj.as<Map<Var, Layout>>()) {
      std::vector<Buffer> block_buffers;
      for (const Buffer &buffer : block->alloc_buffers) {
        AddBufferIfMissing(&block_buffers, buffer);
      }
      for (const BufferRegion &region : block->reads) {
        AddBufferIfMissing(&block_buffers, region->buffer);
      }
      for (const BufferRegion &region : block->writes) {
        AddBufferIfMissing(&block_buffers, region->buffer);
      }
      for (const MatchBufferRegion &match_buffer : block->match_buffers) {
        AddBufferIfMissing(&block_buffers, match_buffer->buffer);
      }

      for (const auto &[buffer_var, layout] : layout_map.value()) {
        bool found = false;
        for (const Buffer &buffer : block_buffers) {
          if (buffer->data.same_as(buffer_var)) {
            dst->Set(buffer, layout);
            found = true;
          }
        }
        ICHECK(found) << attr_key
                      << " annotation references unknown buffer var "
                      << buffer_var << ".";
      }
      return;
    }

    LOG(FATAL) << "Unsupported " << attr_key << " annotation type.";
  }

  static void AddBufferIfMissing(std::vector<Buffer> *buffers,
                                 const Buffer &buffer) {
    for (const Buffer &existing : *buffers) {
      if (existing.same_as(buffer)) {
        return;
      }
    }
    buffers->push_back(buffer);
  }

  void ValidateCopyCall(const CallNode *call, const char *op_name,
                        size_t arg_base) {
    ICHECK_GE(call->args.size(), arg_base + 2)
        << op_name << " expects at least src and dst region arguments";

    BufferRegion src = NormalizeToBufferRegion(call->args[arg_base]);
    BufferRegion dst = NormalizeToBufferRegion(call->args[arg_base + 1]);

    ValidateRegionCanFormTileView(src, op_name, "src");
    ValidateRegionCanFormTileView(dst, op_name, "dst");
  }

  void ValidateRegionCanFormTileView(const BufferRegion &region,
                                     const char *op_name,
                                     const char *operand_name) {
    ICHECK(region.defined())
        << op_name << " " << operand_name << " region is undefined";
    ICHECK(region->buffer.defined())
        << op_name << " " << operand_name << " buffer is undefined";
    ICHECK(!region->region.empty())
        << op_name << " " << operand_name
        << " region must have at least one dimension";
    ICHECK_EQ(region->region.size(), region->buffer->shape.size())
        << op_name << " " << operand_name << " region rank "
        << region->region.size() << " does not match buffer "
        << region->buffer->name << " rank " << region->buffer->shape.size();

    Layout layout = LookupLayout(region->buffer, op_name, operand_name);
    std::vector<int> tiled_dims;
    for (size_t dim = 0; dim < region->region.size(); ++dim) {
      const Range &range = region->region[dim];
      ICHECK(range.defined())
          << op_name << " " << operand_name << " range at dim " << dim
          << " is undefined for buffer " << region->buffer->name;
      ICHECK(range->extent.defined())
          << op_name << " " << operand_name << " extent at dim " << dim
          << " is undefined for buffer " << region->buffer->name;

      if (analyzer_.CanProveEqual(range->extent, 1)) {
        continue;
      }

      tiled_dims.push_back(static_cast<int>(dim));
      ValidateTiledDim(region, layout, dim, op_name, operand_name);
    }

    ICHECK(tiled_dims.size() == 1 || tiled_dims.size() == 2)
        << op_name << " " << operand_name << " region for buffer "
        << region->buffer->name << " must form a 1D or 2D tile_view, but got "
        << tiled_dims.size() << " non-unit region extents.";
  }

  std::optional<int64_t> TryGetStaticInt(const PrimExpr &expr) {
    PrimExpr simplified = analyzer_.Simplify(expr);
    if (const auto *imm = simplified.as<IntImmNode>()) {
      return imm->value;
    }
    return std::nullopt;
  }

  bool CanProveDivisible(const PrimExpr &value, const PrimExpr &divisor) {
    PrimExpr simplified_divisor = analyzer_.Simplify(divisor);
    if (analyzer_.CanProveEqual(simplified_divisor, 0)) {
      return false;
    }
    PrimExpr remainder =
        analyzer_.Simplify(floormod(value, simplified_divisor));
    return analyzer_.CanProveEqual(remainder, 0);
  }

  void ValidateTiledDim(const BufferRegion &region, const Layout &layout,
                        size_t dim, const char *op_name,
                        const char *operand_name) {
    const Range &range = region->region[dim];
    const PrimExpr &buffer_shape = region->buffer->shape[dim];
    const PrimExpr &region_min = range->min;
    const PrimExpr &region_extent = range->extent;

    std::optional<int64_t> static_extent = TryGetStaticInt(region_extent);
    ICHECK(static_extent.has_value() && static_extent.value() > 0)
        << op_name << " " << operand_name << " region extent at dim " << dim
        << " for buffer " << region->buffer->name
        << " must be a positive compile-time constant, but got extent="
        << region_extent << ".";

    DimLayoutInfo info = GetDimLayoutInfo(layout, dim, region->buffer);

    ICHECK(!analyzer_.CanProve(region_min < make_zero(region_min.dtype())))
        << op_name << " " << operand_name << " region min at dim " << dim
        << " for buffer " << region->buffer->name
        << " must be non-negative, but got min=" << region_min << ".";
    ICHECK(CanProveDivisible(region_min, region_extent))
        << op_name << " " << operand_name << " region min at dim " << dim
        << " for buffer " << region->buffer->name
        << " must align to region extent " << region_extent
        << ", but got min=" << region_min << ".";
    if (info.has_dynamic_outer_mode) {
      ICHECK(!analyzer_.CanProve(region_min + region_extent > buffer_shape))
          << op_name << " " << operand_name << " region at dim " << dim
          << " for buffer " << region->buffer->name
          << " must stay within buffer shape " << buffer_shape
          << ", but got min=" << region_min << " and extent=" << region_extent
          << ".";
    } else {
      ICHECK(analyzer_.CanProve(region_min + region_extent <= buffer_shape))
          << op_name << " " << operand_name << " region at dim " << dim
          << " for buffer " << region->buffer->name
          << " must stay within buffer shape " << buffer_shape
          << ", but got min=" << region_min << " and extent=" << region_extent
          << ".";
    }

    if (info.has_dynamic_outer_mode) {
      ICHECK_EQ(static_extent.value() % info.inner_static_mode_shape, 0)
          << op_name << " " << operand_name << " region extent at dim " << dim
          << " for buffer " << region->buffer->name
          << " must be a multiple of dynamic layout inner static mode shape "
          << info.inner_static_mode_shape
          << ", but got extent=" << region_extent << ".";
      return;
    }

    ICHECK(CanProveDivisible(buffer_shape, region_extent))
        << op_name << " " << operand_name << " region extent at dim " << dim
        << " for buffer " << region->buffer->name
        << " must divide buffer shape " << buffer_shape
        << ", but got extent=" << region_extent << ".";

    if (info.coalesced_extent < static_extent.value()) {
      ICHECK_EQ(static_extent.value() % info.coalesced_extent, 0)
          << op_name << " " << operand_name << " region extent at dim " << dim
          << " for buffer " << region->buffer->name
          << " must be compatible with coalesced extent "
          << info.coalesced_extent << ", but got extent=" << region_extent
          << ".";
    } else {
      ICHECK_EQ(info.coalesced_extent % static_extent.value(), 0)
          << op_name << " " << operand_name << " region extent at dim " << dim
          << " for buffer " << region->buffer->name
          << " must be compatible with coalesced extent "
          << info.coalesced_extent << ", but got extent=" << region_extent
          << ".";
    }
  }

  DimLayoutInfo GetDimLayoutInfo(const Layout &layout, size_t dim,
                                 const Buffer &buffer) {
    DimLayoutInfo info;
    info.has_layout = true;

    if (const auto *cute = layout.as<CuteLayoutNode>()) {
      Array<PrimExpr> mode_shapes =
          cute->GetModeShapeOfDim(static_cast<int>(dim));
      Array<PrimExpr> mode_strides =
          cute->GetModeStrideOfDim(static_cast<int>(dim));
      ICHECK_EQ(mode_shapes.size(), mode_strides.size())
          << "CuteLayout mode shape/stride mismatch for buffer " << buffer->name
          << " dim " << dim << ".";
      ICHECK(!mode_shapes.empty()) << "CuteLayout has no modes for buffer "
                                   << buffer->name << " dim " << dim << ".";

      const auto *inner_shape = mode_shapes[0].as<IntImmNode>();
      ICHECK(inner_shape) << "CuteLayout inner mode shape for buffer "
                          << buffer->name << " dim " << dim
                          << " must be static, but got " << mode_shapes[0]
                          << ".";
      info.inner_static_mode_shape = inner_shape->value;

      const bool outer_dynamic =
          mode_shapes.size() > 1 &&
          !mode_shapes[mode_shapes.size() - 1].as<IntImmNode>();
      info.has_dynamic_outer_mode = outer_dynamic;
      info.coalesced_extent =
          ComputeDimCoalescedExtent(mode_shapes, mode_strides);
      return info;
    }

    std::optional<int64_t> shape_value = TryGetStaticInt(buffer->shape[dim]);
    ICHECK(shape_value.has_value() && shape_value.value() > 0)
        << "Non-CuteLayout buffer " << buffer->name << " dim " << dim
        << " requires static positive buffer.shape, but got "
        << buffer->shape[dim] << ".";
    info.coalesced_extent = shape_value.value();
    info.inner_static_mode_shape = shape_value.value();
    return info;
  }

  int64_t ComputeDimCoalescedExtent(const Array<PrimExpr> &mode_shapes,
                                    const Array<PrimExpr> &mode_strides) {
    std::vector<std::pair<int64_t, int64_t>> static_modes;
    for (size_t i = 0; i < mode_shapes.size(); ++i) {
      const auto *shape = mode_shapes[i].as<IntImmNode>();
      const auto *stride = mode_strides[i].as<IntImmNode>();
      if (!shape || !stride) {
        break;
      }
      if (shape->value == 1) {
        continue;
      }
      static_modes.push_back({shape->value, stride->value});
    }

    ICHECK(!static_modes.empty())
        << "Cannot compute coalesced extent from layout with no static modes.";
    std::sort(static_modes.begin(), static_modes.end(),
              [](const auto &a, const auto &b) { return a.second < b.second; });

    int64_t coalesced_extent = static_modes[0].first;
    int64_t running_stride = static_modes[0].second;
    for (size_t i = 1; i < static_modes.size(); ++i) {
      if (coalesced_extent * running_stride != static_modes[i].second) {
        break;
      }
      coalesced_extent *= static_modes[i].first;
    }
    return coalesced_extent;
  }

  Layout LookupLayout(const Buffer &buffer, const char *op_name,
                      const char *operand_name) const {
    if (!scope_stack_.empty()) {
      const ScopeFrame &frame = scope_stack_.back();
      if (auto layout = LookupLayoutInMap(frame.layout_map, buffer)) {
        return layout.value();
      }
      if (auto layout = LookupLayoutInMap(frame.global_layout_map, buffer)) {
        return layout.value();
      }
    }

    ICHECK(false) << op_name << " " << operand_name << " buffer "
                  << buffer->name
                  << " requires layout_map/global_layout_map metadata for "
                     "tile_view validation.";
    return Layout();
  }

  std::optional<Layout> LookupLayoutInMap(const LayoutMap &layout_map,
                                          const Buffer &buffer) const {
    if (layout_map.count(buffer)) {
      return layout_map[buffer];
    }

    for (const auto &kv : layout_map) {
      const Buffer &candidate = kv.first;
      if (candidate->data.same_as(buffer->data) ||
          candidate->data->name_hint == buffer->data->name_hint) {
        return kv.second;
      }
    }
    return std::nullopt;
  }

  arith::Analyzer analyzer_;
  std::vector<ScopeFrame> scope_stack_;
};

} // namespace

tvm::transform::Pass ValidateCopyTileView() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return ValidateCopyTileViewPass::Run(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ValidateCopyTileView", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ValidateCopyTileView",
                        ValidateCopyTileView);
}

} // namespace tl
} // namespace tvm
