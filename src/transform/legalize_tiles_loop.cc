#include <unordered_map>
#include <unordered_set>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../support/ffi_aliases.h"
#include "../tileview/tileview.h"
#include "tiles_loop_attr.h"

namespace tvm {
namespace tl {

using namespace tir;

/* ============================================================
 * TileView Collector
 *
 * Collect block-level:
 *   block.annotations["tileview_map"]
 *     : Map<Var, TileView>
 * ============================================================ */
class TileViewCollector : public StmtExprVisitor {
public:
  using TileViewMap =
      std::unordered_map<Var, TileView, ObjectPtrHash, ObjectPtrEqual>;

  static TileViewMap Collect(const PrimFunc &f) {
    TileViewCollector collector;
    collector(f->body);
    return std::move(collector.tileviews_);
  }

private:
  void VisitStmt_(const BlockNode *block) final {
    auto it = block->annotations.find(attr::kTileViewMap);
    if (it != block->annotations.end()) {
      auto tv_map = Downcast<Map<Var, TileView>>((*it).second);
      for (const auto &kv : tv_map) {
        auto res = tileviews_.emplace(kv.first, kv.second);
        ICHECK(res.second) << "Duplicate TileView for buffer " << kv.first;
      }
    }
    StmtExprVisitor::VisitStmt_(block);
  }

private:
  TileViewMap tileviews_;
};

/* ============================================================
 * SharedBufferCollector
 *
 * Collect all buffer->data Vars used inside a loop body
 * ============================================================ */
class SharedBufferCollector : public StmtExprVisitor {
public:
  using BufferSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

  static BufferSet Collect(const Stmt &stmt) {
    SharedBufferCollector collector;
    collector(stmt);
    return std::move(collector.buffers_);
  }

private:
  void VisitExpr_(const BufferLoadNode *op) final {
    buffers_.insert(op->buffer->data);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    buffers_.insert(op->buffer->data);
    StmtExprVisitor::VisitStmt_(op);
  }

private:
  BufferSet buffers_;
};

/* ============================================================
 * LegalizeTilesLoopRewriter
 *
 * Rewrite tile-level For loops:
 *   for ... in T.Tiles(...)
 * into:
 *   extent := TileView::TiledBufferShape()[tile_dim]
 *
 * Assumptions:
 * - Tile loop nesting order == TileView dimension order
 * - TileView already validated semantic correctness
 * ============================================================ */
class LegalizeTilesLoopRewriter : public StmtExprMutator {
public:
  using TileViewMap =
      std::unordered_map<Var, TileView, ObjectPtrHash, ObjectPtrEqual>;

  static PrimFunc Rewrite(PrimFunc f) {
    LegalizeTilesLoopRewriter rewriter;
    rewriter.tileviews_ = TileViewCollector::Collect(f);

    if (rewriter.tileviews_.empty()) {
      return f;
    }

    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

private:
  Stmt VisitStmt_(const ForNode *loop) final {
    // Only rewrite tile-level loops
    if (!loop->annotations.count(attr::tile_level_loop)) {
      return StmtExprMutator::VisitStmt_(loop);
    }

    // ---- Stage Check (Idempotency) ----
    int stage = static_cast<int>(TileLoopStage::kInitial);

    auto stage_it = loop->annotations.find(attr::kTileLoopStage);
    if (stage_it != loop->annotations.end()) {
      stage = Downcast<Integer>((*stage_it).second)->value;
    }

    if (stage >= static_cast<int>(TileLoopStage::kLegalized)) {
      // Already legalized → skip
      LOG(INFO) << "[Legalize tiles loop] tile loop stage is: " << stage
                << ", so skip. ";
      return StmtExprMutator::VisitStmt_(loop);
    }

    // Must be associated with a tiled buffer
    auto buf_it = loop->annotations.find(attr::tiled_buffer);
    if (buf_it == loop->annotations.end()) {
      return StmtExprMutator::VisitStmt_(loop);
    }

    Var buffer_data = Downcast<Var>((*buf_it).second);

    auto tv_it = tileviews_.find(buffer_data);
    if (tv_it == tileviews_.end()) {
      return StmtExprMutator::VisitStmt_(loop);
    }

    // ---- Collect all used buffers inside loop body ----
    auto used_buffers = SharedBufferCollector::Collect(loop->body);

    if (used_buffers.empty()) {
      return StmtExprMutator::VisitStmt_(loop);
    }

    // ---- Validate TileViews ----
    const TileView *ref_tv = nullptr;

    for (const Var &buf : used_buffers) {
      auto it = tileviews_.find(buf);

      if (it == tileviews_.end()) {
        LOG(FATAL) << "Buffer " << buf
                   << " used inside tile loop but has no TileView.";
      }

      if (!ref_tv) {
        ref_tv = &(it->second);
      } else {
        ICHECK(ref_tv->get()->IsEqual(it->second.get()))
            << "Inconsistent TileView inside tile loop.";
      }
    }

    ICHECK(ref_tv) << "Internal error: ref_tv should not be null.";

    const TileView &tv = *ref_tv;

    // Enter tile loop (depth == tile dimension)
    int dim = tile_loop_depth_++;
    Stmt new_body = VisitStmt(loop->body);
    tile_loop_depth_--;

    Array<PrimExpr> tiled_shape = tv->TiledBufferShape();

    ICHECK(dim < static_cast<int>(tiled_shape.size()))
        << "Tile loop depth exceeds tiled buffer rank";

    // Rewrite loop
    For new_for = ffi::GetRef<For>(loop);
    auto *n = new_for.CopyOnWrite();
    n->extent = tiled_shape[dim];
    n->body = new_body;

    // Attach normalized loop annotations
    n->annotations.Set(attr::tile_new_shape, tiled_shape);
    n->annotations.Set(attr::tile_tile_size, tv->TileShape());
    n->annotations.Set(attr::tile_dim_map, tv->IndexMap());
    n->annotations.Set(attr::kTileLoopStage,
                       Integer(static_cast<int>(TileLoopStage::kLegalized)));
    // ---- Determine whether this loop is a tile execution dimension ----
    int buf_ndim = static_cast<int>(tv->BufferShape().size());
    bool is_tile_execution = false;

    for (const PrimExpr &pe : tv->IndexMap()) {
      const auto *imm = pe.as<IntImmNode>();
      ICHECK(imm) << "index_map must contain IntImm";

      int mapped_dim = static_cast<int>(imm->value);
      if (mapped_dim < 0) {
        mapped_dim += buf_ndim;
      }

      if (mapped_dim == dim) {
        is_tile_execution = true;
        break;
      }
    }

    if (is_tile_execution) {
      n->annotations.Set(attr::tile_execution_loop, Integer(1));
    }
    return new_for;
  }

private:
  TileViewMap tileviews_;
  int tile_loop_depth_{0};
};

/* ============================================================
 * Pass Registration
 * ============================================================ */
using namespace tir::transform;

tvm::transform::Pass LegalizeTilesLoop() {
  auto pass_func = [](PrimFunc f, const IRModule &, const PassContext &) {
    return LegalizeTilesLoopRewriter::Rewrite(std::move(f));
  };

  return CreatePrimFuncPass(pass_func,
                            /*opt_level=*/0, "tl.LegalizeTilesLoop", {});
}

/* ============================================================
 * FFI Registration
 * ============================================================ */
TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def("tl.transform.LegalizeTilesLoop",
                                        LegalizeTilesLoop);
}

} // namespace tl
} // namespace tvm
