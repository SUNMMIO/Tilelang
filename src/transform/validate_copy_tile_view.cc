/*!
 * \file validate_copy_tile_view.cc
 * \brief Validate copy regions that must be representable as Sunmmio tile views.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <string>
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
    LayoutMap layout_map;
    LayoutMap global_layout_map;
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

    if (block->annotations.count(attr::kLayoutMap)) {
      frame.layout_map = block->annotations.at(attr::kLayoutMap)
                             .as<LayoutMap>()
                             .value();
    }
    if (block->annotations.count(attr::kGlobalLayoutMap)) {
      frame.global_layout_map = block->annotations.at(attr::kGlobalLayoutMap)
                                    .as<LayoutMap>()
                                    .value();
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

    ValidateRegionCanFormTileView(src, op_name, "src");
    ValidateRegionCanFormTileView(dst, op_name, "dst");
    ValidateTileViewMetadata(src, op_name, "src");
    ValidateTileViewMetadata(dst, op_name, "dst");
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

    std::vector<size_t> non_unit_extent_dims;
    for (size_t dim = 0; dim < region->region.size(); ++dim) {
      const Range &range = region->region[dim];
      ICHECK(range.defined())
          << op_name << " " << operand_name << " range at dim " << dim
          << " is undefined for buffer " << region->buffer->name;
      ICHECK(range->extent.defined())
          << op_name << " " << operand_name << " extent at dim " << dim
          << " is undefined for buffer " << region->buffer->name;

      if (!analyzer_.CanProveEqual(range->extent, 1)) {
        non_unit_extent_dims.push_back(dim);
      }
    }

    // TODO(copy-tile-view): after the exact tile alignment rule is finalized,
    // check region mins/extents against the buffer layout here.  The collected
    // non-unit extent dims are the candidate tile axes; the final rule should
    // decide whether to reject ranks other than 1D/2D here or leave them to
    // non-tile DMA fallback paths.  At this point SunmmioLayoutInference has
    // attached layout_map/global_layout_map to block annotations, so this pass
    // has access to both the region and its layout.
    (void)non_unit_extent_dims;
  }

  void ValidateTileViewMetadata(const BufferRegion &region, const char *op_name,
                                const char *operand_name) {
    Optional<TileView> maybe_tileview = LookupTileView(region->buffer);
    if (!maybe_tileview.defined()) {
      // Explicit TileView metadata is optional for raw copy/dma_copy regions.
      // The backend can construct a tile view directly from region extents.
      return;
    }

    TileView tileview = maybe_tileview.value();
    ICHECK(tileview->TileDim() == 1 || tileview->TileDim() == 2)
        << op_name << " " << operand_name << " buffer "
        << region->buffer->name
        << " has unsupported TileView rank " << tileview->TileDim()
        << "; Sunmmio copy tile views must be 1D or 2D.";
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

}  // namespace

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

}  // namespace tl
}  // namespace tvm
