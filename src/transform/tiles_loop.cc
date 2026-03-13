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
 *   for i        (tile.scope_entry=1)
 *     for j
 *       for ki   (tile.interior, axis=0)
 *         for kj (tile.interior, axis=1, vectorized)
 *           body
 *
 * Only for stage == kLegalized.
 * After rewrite, stage → kTiled (by buffer-group update).
 *
 * The 2D path is tolerant of non-trivial nesting between the
 * outer and inner tile.execution loops (LetStmt, AttrStmt, etc.
 * may appear in between).
 * ------------------------------------------------------------
 */
class TilesLoopRewriter : public StmtExprMutator {
public:
  static PrimFunc Rewrite(PrimFunc f) {
    TilesLoopRewriter rewriter;
    f.CopyOnWrite()->body = rewriter(f->body);

    // If any tiles loop was modified, update stage by semantic grouping
    // When visiting tiles for node, we collect all tiled_buffers and
    // update stage based on these buffers.
    if (!rewriter.modified_tile_buffers_.empty()) {
      StageUpdateRewriter updater(rewriter.modified_tile_buffers_);
      f.CopyOnWrite()->body = updater(f->body);
    }

    return f;
  }

private:
  // ---- Predicates ----

  static bool IsTilesScope(const ForNode *loop) {
    return loop->annotations.count(attr::tile_execution_loop);
  }

  static bool IsSerialFor(const ForNode *loop) {
    return loop->kind == ForKind::kSerial;
  }

  static Optional<Array<PrimExpr>> GetTileSize(const ForNode *loop) {
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

  // ---- Helpers for tolerant nesting ----

  /*!
   * \brief Search for the inner tile.execution ForNode by peeling through
   *        transparent wrapper statements (LetStmt, AttrStmt, AssertStmt).
   *
   * This makes the 2D lowering path tolerant of intermediate IR that
   * earlier passes may have inserted between the outer and inner tile
   * execution loops.
   *
   * \param body The statement to search within.
   * \param tiled_buf Only match loops tied to this buffer.
   * \return The inner ForNode if found, nullptr otherwise.
   */
  static const ForNode *PeelToInnerTileExecLoop(const Stmt &body,
                                                const Var &tiled_buf) {
    if (const auto *f = body.as<ForNode>()) {
      if (f->annotations.count(attr::tile_execution_loop)) {
        auto buf_it = f->annotations.find(attr::tiled_buffer);
        if (buf_it != f->annotations.end() &&
            Downcast<Var>((*buf_it).second).same_as(tiled_buf)) {
          return f;
        }
      }
      // A ForNode that isn't the right tile.execution — stop peeling
      return nullptr;
    }
    // Peel through transparent single-child wrappers
    if (const auto *let = body.as<LetStmtNode>()) {
      return PeelToInnerTileExecLoop(let->body, tiled_buf);
    }
    if (const auto *attr_stmt = body.as<AttrStmtNode>()) {
      return PeelToInnerTileExecLoop(attr_stmt->body, tiled_buf);
    }
    if (const auto *assert_stmt = body.as<AssertStmtNode>()) {
      return PeelToInnerTileExecLoop(assert_stmt->body, tiled_buf);
    }
    // SeqStmt, IfThenElse, etc. — don't descend
    return nullptr;
  }

  /*!
   * \brief Replace a specific ForNode within a statement tree.
   *
   * Used to swap the inner tile.execution loop with its rewritten
   * version, preserving any wrapper statements in between.
   */
  class InnerLoopReplacer : public StmtExprMutator {
  public:
    InnerLoopReplacer(const ForNode *target, Stmt replacement)
        : target_(target), replacement_(std::move(replacement)) {}

    Stmt VisitStmt_(const ForNode *loop) final {
      if (loop == target_)
        return replacement_;
      return StmtExprMutator::VisitStmt_(loop);
    }

  private:
    const ForNode *target_;
    Stmt replacement_;
  };

  /*!
   * \brief Build a For loop annotated as a tile interior loop.
   * \param loop_var  The intra-tile iteration variable (ki or kj).
   * \param extent    tile_size along this axis.
   * \param kind      kSerial for non-last axes, kVectorized for last axis.
   * \param body      The loop body.
   * \param axis      Which axis of the tile shape (0, 1, ...).
   * \param tiled_buf The tiled buffer this loop belongs to.
   */
  static For MakeTileInteriorLoop(Var loop_var, PrimExpr extent, ForKind kind,
                                  Stmt body, int axis, const Var &tiled_buf) {
    For loop(loop_var, Integer(0), extent, kind, body);
    auto *n = loop.CopyOnWrite();
    n->annotations.Set(attr::tile_interior, Integer(1));
    n->annotations.Set(attr::tile_interior_axis, Integer(axis));
    n->annotations.Set(attr::tiled_buffer, tiled_buf);
    return loop;
  }

  // ---- Main rewrite ----

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

    auto tile_size_opt = GetTileSize(loop);
    if (!tile_size_opt.defined()) {
      return UpdateBody(loop, new_body);
    }

    Array<PrimExpr> tile_size = tile_size_opt.value();
    ICHECK(tile_size.size() == 1 || tile_size.size() == 2)
        << "TilesLoop expects 1D or 2D tile_size, got " << tile_size.size();

    // ---- Record modified tile buffer (semantic grouping anchor) ----
    auto buf_it = loop->annotations.find(attr::tiled_buffer);
    ICHECK(buf_it != loop->annotations.end());
    Var tiled_buf = Downcast<Var>((*buf_it).second);
    modified_tile_buffers_.insert(tiled_buf);

    if (tile_size.size() == 1) {
      return Rewrite1D(loop, new_body, tile_size, tiled_buf);
    }
    return Rewrite2D(loop, new_body, tile_size, tiled_buf);
  }

  /*!
   * \brief 1D lowering:
   *   for ti → for ti (scope_entry) { for ki (interior, vec) { body } }
   */
  Stmt Rewrite1D(const ForNode *loop, Stmt new_body,
                 const Array<PrimExpr> &tile_size, const Var &tiled_buf) {
    if (!IsSerialFor(loop)) {
      return UpdateBody(loop, new_body);
    }

    Var ti = loop->loop_var;
    Var ki("ki");

    Map<Var, PrimExpr> vmap;
    vmap.Set(ti, ti * tile_size[0] + ki);

    Stmt tiled_body = Substitute(new_body, vmap);

    // Create annotated interior loop
    For ki_loop =
        MakeTileInteriorLoop(ki, tile_size[0], ForKind::kVectorized, tiled_body,
                             /*axis=*/0, tiled_buf);

    // Update outer loop: set scope_entry
    For new_outer = ffi::GetRef<For>(loop);
    auto *n = new_outer.CopyOnWrite();
    n->body = ki_loop;
    n->annotations.Set(attr::tile_scope_entry, Integer(1));
    return new_outer;
  }

  /*!
   * \brief 2D lowering:
   *   for ti { [wrappers...] for tj { body } }
   *   →
   *   for ti (scope_entry) { [wrappers...] for tj {
   *       for ki (interior, axis=0) { for kj (interior, axis=1, vec) { body } }
   *   } }
   *
   * Tolerant of LetStmt/AttrStmt/AssertStmt between ti and tj.
   */
  Stmt Rewrite2D(const ForNode *loop, Stmt new_body,
                 const Array<PrimExpr> &tile_size, const Var &tiled_buf) {
    // Find inner tile.execution loop, peeling through wrappers
    const ForNode *inner = PeelToInnerTileExecLoop(new_body, tiled_buf);
    if (!inner) {
      return UpdateBody(loop, new_body);
    }

    if (!IsSerialFor(loop) || !IsSerialFor(inner)) {
      return UpdateBody(loop, new_body);
    }

    Var ti = loop->loop_var;
    Var tj = inner->loop_var;

    Var ki("ki");
    Var kj("kj");

    Map<Var, PrimExpr> vmap;
    vmap.Set(ti, ti * tile_size[0] + ki);
    vmap.Set(tj, tj * tile_size[1] + kj);

    Stmt tiled_body = Substitute(inner->body, vmap);

    // Create annotated interior loops
    For kj_loop =
        MakeTileInteriorLoop(kj, tile_size[1], ForKind::kVectorized, tiled_body,
                             /*axis=*/1, tiled_buf);
    For ki_loop =
        MakeTileInteriorLoop(ki, tile_size[0], ForKind::kSerial, kj_loop,
                             /*axis=*/0, tiled_buf);

    // Replace inner loop's body with the tiled loops
    For new_inner = ffi::GetRef<For>(inner);
    new_inner.CopyOnWrite()->body = ki_loop;

    // Replace inner loop in the body tree (tolerant of wrapper stmts)
    Stmt replaced_body;
    if (new_body.as<ForNode>() == inner) {
      // Fast path: inner loop is the direct body
      replaced_body = new_inner;
    } else {
      // General path: inner loop is behind wrapper stmts
      replaced_body = InnerLoopReplacer(inner, new_inner)(new_body);
    }

    // Update outer loop: set scope_entry
    For new_outer = ffi::GetRef<For>(loop);
    auto *n = new_outer.CopyOnWrite();
    n->body = replaced_body;
    n->annotations.Set(attr::tile_scope_entry, Integer(1));
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
