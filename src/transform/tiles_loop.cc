#include <unordered_set>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../support/ffi_aliases.h"
#include "../tileview/tileview.h"
#include "common/attr.h"

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * ------------------------------------------------------------
 * StageUpdateRewriter
 *
 * After structural rewrite succeeds, we update all loops
 * belonging to the same tiled_buffer scope.
 *
 * This guarantees:
 *  - Multi-level loops are handled
 *  - Outer ancestors are updated
 *  - Multiple Tiles do not interfere
 *  - Idempotency preserved
 * ------------------------------------------------------------
 */
class StageUpdateRewriter : public StmtExprMutator {
public:
  explicit StageUpdateRewriter(
      const std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> &buffers)
      : buffers_(buffers) {}

private:
  Stmt VisitStmt_(const ForNode *loop) final {
    For f = Downcast<For>(StmtExprMutator::VisitStmt_(loop));

    auto it = f->annotations.find(attr::tiled_buffer);
    if (it != f->annotations.end()) {
      Var buf = Downcast<Var>((*it).second);
      if (buffers_.count(buf)) {
        f.CopyOnWrite()->annotations.Set(
            attr::kTileLoopStage,
            Integer(static_cast<int>(TileLoopStage::kTiled)));
      }
    }

    return f;
  }

  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> buffers_;
};

/*!
 * ------------------------------------------------------------
 * TilesLoopRewriter
 *
 * Lower:
 *   for i
 *     for j
 *       body
 *
 * Into:
 *   for i
 *     for j
 *       for ki
 *         for kj (vectorized)
 *           body
 *
 * Only for stage == kLegalized.
 * After rewrite, stage → kTiled (by buffer-group update).
 * ------------------------------------------------------------
 */
class TilesLoopRewriter : public StmtExprMutator {
public:
  static PrimFunc Rewrite(PrimFunc f) {
    TilesLoopRewriter rewriter;
    f.CopyOnWrite()->body = rewriter(f->body);

    // If any tiles loop was modified, update stage by semantic grouping
    // When visiting tiles for node, wo collect all tiled_buffers and
    // update stage based on these buffers.
    if (!rewriter.modified_tile_buffers_.empty()) {
      StageUpdateRewriter updater(rewriter.modified_tile_buffers_);
      f.CopyOnWrite()->body = updater(f->body);
    }

    return f;
  }

private:
  bool IsTilesScope(const ForNode *loop) const {
    return loop->annotations.count(attr::tile_execution_loop);
  }

  bool IsSerialFor(const ForNode *loop) const {
    return loop->kind == ForKind::kSerial;
  }

  Optional<Array<PrimExpr>> GetTileSize(const ForNode *loop) const {
    auto it = loop->annotations.find(attr::tile_tile_size);
    if (it == loop->annotations.end()) {
      return std::nullopt;
    }
    return Downcast<Array<PrimExpr>>((*it).second);
  }

  Stmt UpdateBody(const ForNode *loop, Stmt new_body) {
    if (new_body.same_as(loop->body)) {
      return ffi::GetRef<For>(loop);
    }
    For f = ffi::GetRef<For>(loop);
    f.CopyOnWrite()->body = new_body;
    return f;
  }

  Stmt VisitStmt_(const ForNode *loop) final {
    // Post-order traversal
    Stmt new_body = VisitStmt(loop->body);

    // Only care about tile execution loops
    if (!IsTilesScope(loop)) {
      return UpdateBody(loop, new_body);
    }

    // ---- Stage gate ----
    int stage = static_cast<int>(TileLoopStage::kInitial);
    auto stage_it = loop->annotations.find(attr::kTileLoopStage);
    if (stage_it != loop->annotations.end()) {
      stage = Downcast<Integer>((*stage_it).second)->value;
    }

    if (stage != static_cast<int>(TileLoopStage::kLegalized)) {
      return UpdateBody(loop, new_body);
    }

    // ---- Structural match ----
    const ForNode *inner = new_body.as<ForNode>();
    if (!inner) {
      return UpdateBody(loop, new_body);
    }

    if (!IsSerialFor(loop) || !IsSerialFor(inner)) {
      return UpdateBody(loop, new_body);
    }

    if (!IsTilesScope(inner)) {
      return UpdateBody(loop, new_body);
    }

    auto tile_size_opt = GetTileSize(loop);
    if (!tile_size_opt.defined()) {
      return UpdateBody(loop, new_body);
    }

    Array<PrimExpr> tile_size = tile_size_opt.value();
    ICHECK_EQ(tile_size.size(), 2) << "TilesLoop expects exactly 2D tile_size";

    // ---- Perform lowering ----
    Var ti = loop->loop_var;
    Var tj = inner->loop_var;

    Var ki("ki");
    Var kj("kj");

    Map<Var, PrimExpr> vmap;
    vmap.Set(ti, ti * tile_size[0] + ki);
    vmap.Set(tj, tj * tile_size[1] + kj);

    Stmt tiled_body = Substitute(inner->body, vmap);

    tiled_body = For(kj, 0, tile_size[1], ForKind::kVectorized, tiled_body);
    tiled_body = For(ki, 0, tile_size[0], ForKind::kSerial, tiled_body);

    For new_inner = ffi::GetRef<For>(inner);
    new_inner.CopyOnWrite()->body = tiled_body;

    For new_outer = ffi::GetRef<For>(loop);
    new_outer.CopyOnWrite()->body = new_inner;

    // ---- Record modified tile buffer (semantic grouping anchor) ----
    auto buf_it = loop->annotations.find(attr::tiled_buffer);
    ICHECK(buf_it != loop->annotations.end());
    Var tiled_buf = Downcast<Var>((*buf_it).second);

    modified_tile_buffers_.insert(tiled_buf);

    return new_outer;
  }

private:
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> modified_tile_buffers_;
};

/*!
 * ------------------------------------------------------------
 * Pass Registration
 * ------------------------------------------------------------
 */
using namespace tir::transform;

Pass TilesLoop() {
  auto pass_func = [](PrimFunc f, const IRModule &,
                      const PassContext &) -> PrimFunc {
    return TilesLoopRewriter::Rewrite(std::move(f));
  };

  return CreatePrimFuncPass(pass_func,
                            /*opt_level=*/0, "tl.TilesLoop", {});
}

/*!
 * ------------------------------------------------------------
 * FFI Registration
 * ------------------------------------------------------------
 */
TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def("tl.transform.TilesLoop", TilesLoop);
}

} // namespace tl
} // namespace tvm