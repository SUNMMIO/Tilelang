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

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/operator.h"
#include "../op/utils.h"
#include "../target/utils.h"
#include "../tileview/tileview.h"
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
  struct ScopeFrame {
    Map<Var, TileView> tileview_map;
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

    if (block->annotations.count(attr::kTileViewMap)) {
      frame.tileview_map = block->annotations.at(attr::kTileViewMap)
                               .as<Map<Var, TileView>>()
                               .value();
    }
    scope_stack_.push_back(std::move(frame));
  }

  void ValidateCopyCall(const CallNode *call, const char *op_name,
                        size_t arg_base) {
    ICHECK_GE(call->args.size(), arg_base + 2)
        << op_name << " expects at least src and dst region arguments";

    BufferRegion src = NormalizeToBufferRegion(call->args[arg_base]);
    BufferRegion dst = NormalizeToBufferRegion(call->args[arg_base + 1]);

    Optional<TileView> src_tileview = LookupTileView(src->buffer);
    Optional<TileView> dst_tileview = LookupTileView(dst->buffer);
    Optional<TileView> src_validation_tileview =
        src_tileview.defined() ? src_tileview : dst_tileview;
    Optional<TileView> dst_validation_tileview =
        dst_tileview.defined() ? dst_tileview : src_tileview;

    ValidateRegionCanFormTileView(src, src_validation_tileview, op_name, "src");
    ValidateRegionCanFormTileView(dst, dst_validation_tileview, op_name, "dst");
    ValidateTileViewMetadata(src, src_tileview, op_name, "src");
    ValidateTileViewMetadata(dst, dst_tileview, op_name, "dst");
  }

  void ValidateRegionCanFormTileView(const BufferRegion &region,
                                     const Optional<TileView> &maybe_tileview,
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

    std::unordered_map<size_t, PrimExpr> tile_size_by_dim =
        BuildTileSizeByDim(region->buffer, maybe_tileview);
    for (size_t dim = 0; dim < region->region.size(); ++dim) {
      const Range &range = region->region[dim];
      ICHECK(range.defined())
          << op_name << " " << operand_name << " range at dim " << dim
          << " is undefined for buffer " << region->buffer->name;
      ICHECK(range->extent.defined())
          << op_name << " " << operand_name << " extent at dim " << dim
          << " is undefined for buffer " << region->buffer->name;

      const PrimExpr &shape = region->buffer->shape[dim];
      ValidateMinAlignmentToTileSize(range->min, tile_size_by_dim, dim, region,
                                     op_name, operand_name);
      ValidateStaticMinIsShapeFactor(range->min, shape, region, dim, op_name,
                                     operand_name);
      ValidateExtentIsShapeFactor(range->extent, shape, region, dim, op_name,
                                  operand_name);
    }
  }

  void ValidateTileViewMetadata(const BufferRegion &region, const char *op_name,
                                const char *operand_name) {
    ValidateTileViewMetadata(region, LookupTileView(region->buffer), op_name,
                             operand_name);
  }

  void ValidateTileViewMetadata(const BufferRegion &region,
                                const Optional<TileView> &maybe_tileview,
                                const char *op_name, const char *operand_name) {
    if (!maybe_tileview.defined()) {
      // Explicit TileView metadata is optional for raw copy/dma_copy regions.
      // The backend can construct a tile view directly from region extents.
      return;
    }

    TileView tileview = maybe_tileview.value();
    ICHECK(tileview->TileDim() == 1 || tileview->TileDim() == 2)
        << op_name << " " << operand_name << " buffer " << region->buffer->name
        << " has unsupported TileView rank " << tileview->TileDim()
        << "; Sunmmio copy tile views must be 1D or 2D.";
  }

  std::unordered_map<size_t, PrimExpr>
  BuildTileSizeByDim(const Buffer &buffer,
                     const Optional<TileView> &maybe_tileview) {
    std::unordered_map<size_t, PrimExpr> tile_size_by_dim;
    if (!maybe_tileview.defined()) {
      return tile_size_by_dim;
    }

    TileView tileview = maybe_tileview.value();
    Array<PrimExpr> tile_shape = tileview->TileShape();
    Array<PrimExpr> index_map = tileview->IndexMap();
    ICHECK_EQ(tile_shape.size(), index_map.size())
        << "TileView for buffer " << buffer->name
        << " has mismatched tile_shape and index_map ranks";

    int buffer_rank = static_cast<int>(buffer->shape.size());
    for (size_t tile_axis = 0; tile_axis < index_map.size(); ++tile_axis) {
      int mapped_dim =
          NormalizeMappedDim(index_map[tile_axis], buffer_rank, buffer->name);
      tile_size_by_dim.emplace(static_cast<size_t>(mapped_dim),
                               tile_shape[tile_axis]);
    }
    return tile_size_by_dim;
  }

  static int NormalizeMappedDim(const PrimExpr &expr, int ndim,
                                const std::string &buffer_name) {
    const auto *imm = expr.as<IntImmNode>();
    ICHECK(imm) << "TileView index_map entries must be IntImm, but got " << expr
                << " for buffer " << buffer_name;
    int mapped_dim = static_cast<int>(imm->value);
    if (mapped_dim < 0) {
      mapped_dim += ndim;
    }
    ICHECK(mapped_dim >= 0 && mapped_dim < ndim)
        << "TileView index_map entry " << expr
        << " is out of bounds for buffer " << buffer_name << " with rank "
        << ndim;
    return mapped_dim;
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

  void ValidateMinAlignmentToTileSize(
      const PrimExpr &min,
      const std::unordered_map<size_t, PrimExpr> &tile_size_by_dim, size_t dim,
      const BufferRegion &region, const char *op_name,
      const char *operand_name) {
    auto it = tile_size_by_dim.find(dim);
    if (it == tile_size_by_dim.end()) {
      return;
    }

    const PrimExpr &tile_size = it->second;
    ICHECK(CanProveDivisible(min, tile_size))
        << op_name << " " << operand_name << " region min at dim " << dim
        << " for buffer " << region->buffer->name << " must align to TileView "
        << "tile size " << tile_size << ", but got min=" << min << ".";
  }

  void ValidateStaticMinIsShapeFactor(const PrimExpr &min,
                                      const PrimExpr &shape,
                                      const BufferRegion &region, size_t dim,
                                      const char *op_name,
                                      const char *operand_name) {
    if (analyzer_.CanProveEqual(min, 0)) {
      return;
    }

    std::optional<int64_t> min_value = TryGetStaticInt(min);
    if (!min_value.has_value()) {
      return;
    }

    ICHECK_GT(min_value.value(), 0)
        << op_name << " " << operand_name << " region min at dim " << dim
        << " for buffer " << region->buffer->name
        << " must be non-negative, but got min=" << min << ".";
    ICHECK(CanProveDivisible(shape, Integer(min_value.value())))
        << op_name << " " << operand_name << " region min at dim " << dim
        << " for buffer " << region->buffer->name
        << " must be zero or a factor of buffer shape " << shape
        << ", but got min=" << min << ".";
  }

  void ValidateExtentIsShapeFactor(const PrimExpr &extent,
                                   const PrimExpr &shape,
                                   const BufferRegion &region, size_t dim,
                                   const char *op_name,
                                   const char *operand_name) {
    ICHECK(analyzer_.CanProveGreaterEqual(extent, 1))
        << op_name << " " << operand_name << " region extent at dim " << dim
        << " for buffer " << region->buffer->name
        << " must be positive, but got extent=" << extent << ".";
    ICHECK(CanProveDivisible(shape, extent))
        << op_name << " " << operand_name << " region extent at dim " << dim
        << " for buffer " << region->buffer->name
        << " must be a factor of buffer shape " << shape
        << ", but got extent=" << extent << ".";
  }

  Optional<TileView> LookupTileView(const Buffer &buffer) const {
    if (scope_stack_.empty()) {
      return Optional<TileView>();
    }

    const ScopeFrame &frame = scope_stack_.back();
    if (frame.tileview_map.count(buffer->data)) {
      return frame.tileview_map[buffer->data];
    }

    // Some passes remap Buffer objects while preserving the user-facing buffer
    // name.  Keep a conservative name-based fallback so validation can still
    // see manual TileView hints after those remaps.
    for (const auto &kv : frame.tileview_map) {
      const Var &var = kv.first;
      if (var->name_hint == buffer->data->name_hint) {
        return kv.second;
      }
    }
    return Optional<TileView>();
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
