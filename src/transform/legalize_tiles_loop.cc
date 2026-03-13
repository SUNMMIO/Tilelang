#include <unordered_map>
#include <unordered_set>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../layout/layout.h"
#include "../support/ffi_aliases.h"
#include "../tileview/tileview.h"
#include "common/attr.h"

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
  using BufferDataMap =
      std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual>;

  static PrimFunc Rewrite(PrimFunc f) {
    LegalizeTilesLoopRewriter rewriter;
    rewriter.tileviews_ = TileViewCollector::Collect(f);

    // Collect buffer info from PrimFunc params
    for (const auto &[_, buffer] : f->buffer_map) {
      rewriter.buffer_data_to_buffer_[buffer->data] = buffer;
    }

    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

private:
  /* ---- Collect buffer and layout info from Block nodes ---- */
  Stmt VisitStmt_(const BlockNode *block) final {
    // Collect buffer data -> Buffer mapping from alloc_buffers and
    // match_buffers
    for (const auto &buffer : block->alloc_buffers) {
      buffer_data_to_buffer_[buffer->data] = buffer;
    }
    for (const auto &match_buffer : block->match_buffers) {
      buffer_data_to_buffer_[match_buffer->buffer->data] = match_buffer->buffer;
    }

    // Collect layout_map from block annotation
    if (block->annotations.count(attr::kLayoutMap)) {
      auto layout_map = Downcast<Map<Buffer, Layout>>(
          block->annotations.at(attr::kLayoutMap));
      for (const auto &[buffer, layout] : layout_map) {
        layout_map_.Set(buffer, layout);
      }
    }

    return StmtExprMutator::VisitStmt_(block);
  }

  /* ---- Check if buffer has a layout in layout_map_ ---- */
  bool HasLayout(const Buffer &buffer) const {
    return layout_map_.count(buffer) > 0;
  }

  /* ---- Infer TileView for a buffer when no manual annotation exists ----
   *
   * All buffers inside T.Tiles are on-chip SRAM, so scope is not a
   * distinguishing factor.  The only question is whether the buffer has
   * a layout in layout_map_ (set by LayoutInference), which indicates
   * that the tile unit processes it in 2-D blocks rather than row-by-row.
   *
   * Rules:
   *   1D buffer                → tile_shape=(32,),    index_map=(-1,)
   *   2D+ buffer WITH layout   → tile_shape=(H, 32),  index_map=(-2,-1)
   *       where H = largest of {32,16,8,1} dividing dim[-2]
   *   2D+ buffer WITHOUT layout→ tile_shape=(1, 32),   index_map=(-2,-1)
   * ------------------------------------------------------------------ */
  TileView InferTileView(const Var &buffer_data) {
    auto buf_it = buffer_data_to_buffer_.find(buffer_data);
    ICHECK(buf_it != buffer_data_to_buffer_.end())
        << "Cannot find buffer for data var " << buffer_data
        << " during TileView inference.";

    const Buffer &buffer = buf_it->second;
    int ndim = static_cast<int>(buffer->shape.size());

    if (ndim == 1) {
      // 1D buffer: tile_shape={32}, index_map={-1}
      Array<PrimExpr> buffer_shape = buffer->shape;
      Array<PrimExpr> tile_shape = {Integer(32)};
      Array<PrimExpr> index_map = {Integer(-1)};
      return makeTileView(buffer_shape, tile_shape, index_map);
    }

    // 2D+ buffer
    Array<PrimExpr> buffer_shape = buffer->shape;

    int tile_h = 1;
    if (HasLayout(buffer)) {
      // TODO: Further verify the layout (blockwise)
      // Buffer has a layout from LayoutInference — tile unit processes
      // it in 2-D blocks.  Pick the largest tile height that evenly
      // divides the second-to-last dimension.
      const auto *dim_val = buffer->shape[ndim - 2].as<IntImmNode>();
      if (dim_val) {
        int64_t dim_size = dim_val->value;
        for (int candidate : {32, 16, 8, 1}) {
          if (dim_size % candidate == 0) {
            tile_h = candidate;
            break;
          }
        }
      }
      // Dynamic dim: tile_h stays 1
    }

    Array<PrimExpr> tile_shape = {Integer(tile_h), Integer(32)};
    Array<PrimExpr> index_map = {Integer(-2), Integer(-1)};
    return makeTileView(buffer_shape, tile_shape, index_map);
  }

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
      // Already legalized -> skip
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

    // ---- Reject nested T.Tiles from different buffers ----
    // Nesting separate T.Tiles calls (e.g., T.Tiles(A) inside T.Tiles(B))
    // is not supported.  Loops from a single T.Tiles call share the same
    // tiled_buffer and nest naturally; only cross-buffer nesting is banned.
    if (tile_loop_depth_ > 0 && active_tiled_buffer_.defined() &&
        !buffer_data.same_as(active_tiled_buffer_)) {
      LOG(FATAL) << "Nested T.Tiles from different buffers is not supported. "
                 << "Outer T.Tiles uses buffer " << active_tiled_buffer_
                 << ", but inner T.Tiles uses buffer " << buffer_data << ". "
                 << "Access multiple buffers within a single T.Tiles scope "
                 << "instead.";
    }

    // ---- Local tileview map for this tile loop scope ----
    TileViewMap local_tileviews;

    // Look up primary buffer's TileView: manual annotation first, then infer
    auto tv_it = tileviews_.find(buffer_data);
    TileView primary_tv;
    if (tv_it != tileviews_.end()) {
      primary_tv = tv_it->second;
    } else {
      // Auto-infer TileView for the primary buffer
      primary_tv = InferTileView(buffer_data);
      LOG(INFO) << "[Legalize tiles loop] Auto-inferred TileView for primary "
                   "buffer: "
                << buffer_data;
    }
    local_tileviews[buffer_data] = primary_tv;

    // ---- Collect all used buffers inside loop body ----
    auto used_buffers = SharedBufferCollector::Collect(loop->body);

    if (used_buffers.empty()) {
      return StmtExprMutator::VisitStmt_(loop);
    }

    // ---- Resolve TileViews for all used buffers ----
    for (const Var &buf : used_buffers) {
      if (local_tileviews.count(buf))
        continue;

      auto it = tileviews_.find(buf);
      if (it != tileviews_.end()) {
        // Manual annotation exists
        local_tileviews[buf] = it->second;
      } else {
        // Auto-infer
        local_tileviews[buf] = InferTileView(buf);
        LOG(INFO) << "[Legalize tiles loop] Auto-inferred TileView for "
                     "buffer: "
                  << buf;
      }
    }

    // ---- Validate TileView compatibility across buffers ----
    // 1) Same-rank buffers must have equal TileViews.
    // 2) Cross-rank buffers must share the same tile width (last dim of
    //    tile_shape) so that broadcast is straightforward.
    {
      size_t primary_rank = primary_tv->TileDim();
      Array<PrimExpr> primary_tile = primary_tv->TileShape();
      PrimExpr primary_width = primary_tile[primary_tile.size() - 1];

      for (const auto &[buf, tv] : local_tileviews) {
        size_t rank = tv->TileDim();
        Array<PrimExpr> tile = tv->TileShape();
        PrimExpr width = tile[tile.size() - 1];

        if (rank == primary_rank) {
          // Same rank: full equality check
          ICHECK(primary_tv->IsEqual(tv.get()))
              << "Inconsistent TileView inside tile loop: buffer " << buf
              << " has same tile rank (" << rank
              << ") as primary buffer but different TileView.";
        } else {
          // Cross rank: tile width must match for broadcast
          const auto *pw = primary_width.as<IntImmNode>();
          const auto *w = width.as<IntImmNode>();
          ICHECK(pw && w && pw->value == w->value)
              << "Tile width mismatch inside tile loop: primary buffer has "
                 "tile width "
              << primary_width << " but buffer " << buf << " has tile width "
              << width << ". Cross-rank buffers must share tile width for "
              << "broadcast.";
        }
      }
    }

    // Use the primary buffer's TileView for loop structure
    const TileView &tv = primary_tv;

    // Enter tile loop (depth == tile dimension)
    int dim = tile_loop_depth_++;
    Var prev_active_buffer = active_tiled_buffer_;
    active_tiled_buffer_ = buffer_data;
    Stmt new_body = VisitStmt(loop->body);
    active_tiled_buffer_ = prev_active_buffer;
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
  BufferDataMap buffer_data_to_buffer_;
  Map<Buffer, Layout> layout_map_;
  int tile_loop_depth_{0};
  Var active_tiled_buffer_; // tracks the tiled_buffer of the innermost T.Tiles
                            // scope
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
