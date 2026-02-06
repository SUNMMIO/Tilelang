#include <unordered_map>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../support/ffi_aliases.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace attr {
// ---- block-level ----
constexpr const char *tile_view = "tile_view";
constexpr const char *tile_size = "tile_size";
constexpr const char *dim_map = "dim_map";
constexpr const char *new_shape = "new_shape";

// ---- loop-level ----
constexpr const char *tile_level_loop = "tile_level_loop";
constexpr const char *tiled_buffer = "tiled_buffer";

// ---- added by this pass ----
constexpr const char *tile_execution = "tile_execution";
constexpr const char *tile_new_shape = "tile_new_shape";
constexpr const char *tile_tile_size = "tile.tile_size";
constexpr const char *tile_dim_map = "tile.dim_map";
} // namespace attr

/* ============================================================
 * Collector
 *
 * Collect block-level tile_view:
 *   Map<Var (buffer.data), Map<String, ObjectRef>>
 * ============================================================ */
class TileViewCollector : public StmtExprVisitor {
public:
  using TileViewMap = std::unordered_map<Var, Map<String, ObjectRef>,
                                         ObjectPtrHash, ObjectPtrEqual>;

  /*! \brief Entry point */
  static TileViewMap Collect(const PrimFunc &f) {
    TileViewCollector collector;
    collector(f->body);
    return std::move(collector.tile_views_);
  }

private:
  /*! \brief Collect tile_view annotations from BlockNode */
  void VisitStmt_(const BlockNode *block) final {
    auto it = block->annotations.find(attr::tile_view);
    if (it != block->annotations.end()) {
      auto tile_view = Downcast<Map<Var, Map<String, ObjectRef>>>((*it).second);

      for (const auto &kv : tile_view) {
        // kv.first  : buffer.data (Var)
        // kv.second : {"tile_size", "dim_map", "new_shape"}

        auto res = tile_views_.emplace(kv.first, kv.second);
        ICHECK(res.second) << "Duplicate tile_view for buffer " << kv.first;
      }
    }
    StmtExprVisitor::VisitStmt_(block);
  }

private:
  TileViewMap tile_views_;
};

/* ============================================================
 * Rewriter
 *
 * Rewrite tile-level For loops:
 *   extent := new_shape[tile_dim]
 * ============================================================ */
class LegalizeTilesLoopRewriter : public StmtExprMutator {
public:
  using TileViewMap = std::unordered_map<Var, Map<String, ObjectRef>,
                                         ObjectPtrHash, ObjectPtrEqual>;

  /*! \brief Entry point */
  static PrimFunc Rewrite(PrimFunc f) {
    LegalizeTilesLoopRewriter rewriter;
    rewriter.tile_views_ = TileViewCollector::Collect(f);
    LOG(INFO) << "Collected " << rewriter.tile_views_.size() << " tile_view(s)"
              << " in LegalizeTilesLoopRewriter." << std::endl;
    // Fast path: no tile_view, nothing to do
    if (rewriter.tile_views_.empty()) {
      return f;
    }

    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

private:
  /*! \brief Rewrite tile-level for loops */
  Stmt VisitStmt_(const ForNode *loop) final {
    // Only care about tile-level loops
    if (!loop->annotations.count(attr::tile_level_loop)) {
      return StmtExprMutator::VisitStmt_(loop);
    }

    // Must have tiled_buffer
    auto buf_it = loop->annotations.find(attr::tiled_buffer);
    if (buf_it == loop->annotations.end()) {
      return StmtExprMutator::VisitStmt_(loop);
    }

    Var buffer_data = Downcast<Var>((*buf_it).second);
    LOG(INFO) << "Legalizing tile loop for buffer " << buffer_data;
    auto view_it = tile_views_.find(buffer_data);
    if (view_it == tile_views_.end()) {
      // No tile_view for this buffer
      return StmtExprMutator::VisitStmt_(loop);
    }

    // Enter tile loop (MVP assumption: nesting order == tile dim order)
    int dim = tile_loop_depth_++;
    Stmt new_body = VisitStmt(loop->body);
    tile_loop_depth_--;

    const Map<String, ObjectRef> &view = view_it->second;

    ObjectRef obj = view.at(attr::new_shape);
    auto arr = Downcast<Array<ObjectRef>>(obj);

    LOG(INFO) << "new_shape raw size: " << arr.size();

    for (int i = 0; i < arr.size(); i++) {
      LOG(INFO) << "  new_shape[" << i << "] type: " << arr[i]->GetTypeKey();
    }

    auto new_shape_opt = obj.as<Array<PrimExpr>>();
    LOG(INFO) << "tile_view.new_shape must be Array<PrimExpr>, we getted "
              << obj->GetTypeKey();

    Array<PrimExpr> new_shape = new_shape_opt.value();

    // auto new_shape =
    //     Downcast<Array<PrimExpr>>(view.at(attr::new_shape));
    LOG(INFO) << " - Retrieved tile_view for buffer " << buffer_data
              << " at tile dim " << dim << " with new_shape size "
              << new_shape.size();
    ICHECK(dim < static_cast<int>(new_shape.size()))
        << "Tile loop depth exceeds new_shape rank";
    LOG(INFO) << " - Setting loop extent to new_shape[" << dim
              << "] = " << new_shape[dim];
    // Rewrite loop
    For new_for = ffi::GetRef<For>(loop);
    auto *n = new_for.CopyOnWrite();
    n->extent = new_shape[dim];
    n->body = new_body;

    // Attach normalized annotations for later passes
    n->annotations.Set(attr::tile_execution, Integer(1));
    n->annotations.Set(attr::tile_new_shape, new_shape);
    n->annotations.Set(attr::tile_tile_size,
                       Downcast<Array<PrimExpr>>(view.at(attr::tile_size)));
    n->annotations.Set(attr::tile_dim_map,
                       Downcast<Array<PrimExpr>>(view.at(attr::dim_map)));

    return new_for;
  }

private:
  /*! \brief Collected tile_view info */
  TileViewMap tile_views_;

  /*! \brief Tile loop nesting depth (MVP) */
  int tile_loop_depth_{0};
};

using namespace tir::transform;

tvm::transform::Pass LegalizeTilesLoop() {
  auto pass_func = [](PrimFunc f, const IRModule &,
                      const PassContext &) -> PrimFunc {
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
