#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../support/ffi_aliases.h"

#include <tvm/runtime/logging.h>

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Annotation keys used by Tiles lowering pipeline.
 *
 * NOTE:
 * - tile.execution : loops corresponding to the index map(index_map=(-2, -1))
 * - tile.tile_size : 2D tile size, e.g. (32, 32)
 */
namespace attr {
constexpr const char *tile_execution = "tile.execution";
constexpr const char *tile_tile_size = "tile.tile_size";
} // namespace attr

/*!
 * \brief TilesLoopRewriter
 *
 * This pass performs the lowering of T.Tiles() into
 *   serial + vectorized inner loops.
 *
 * Design principles:
 * ------------------
 * 1. Post-order traversal:
 *    - Always visit loop body first.
 *    - Structural decisions are made only after the body is stable.
 *
 * 2. Scope gating via annotation:
 *    - Only loops marked with `tile_execution` are considered.
 *    - Other loops (e.g. T.parallel, normal serial loops) are ignored.
 *
 * 3. Structural matching (not annotation-driven semantics):
 *    - Actual lowering only happens when we see:
 *
 *        for i (serial, annotation="tile.execution..."):
 *          for j (serial, annotation="tile.execution..."):
 *            BODY
 *
 *    - i, j are assumed to be the 2D tile-execution axes.
 * 4. construct and insert two new ForNodes.
 */
class TilesLoopRewriter : public StmtExprMutator {
public:
  static PrimFunc Rewrite(PrimFunc f) {
    LOG(INFO) << "[TilesLoop] Start rewriting PrimFunc";
    TilesLoopRewriter rewriter;
    f.CopyOnWrite()->body = rewriter(f->body);
    LOG(INFO) << "[TilesLoop] Finished rewriting PrimFunc";
    return f;
  }

private:
  /*! \brief Check whether a loop belongs to T.Tiles() scope */
  bool IsTilesScope(const ForNode *loop) const {
    return loop->annotations.count(attr::tile_execution);
  }

  /*! \brief Check whether a loop is a serial loop */
  bool IsSerialFor(const ForNode *loop) const {
    return loop->kind == ForKind::kSerial;
  }

  /*! \brief Read tile size annotation if present */
  Optional<Array<PrimExpr>> GetTileSize(const ForNode *loop) const {
    auto it = loop->annotations.find(attr::tile_tile_size);
    if (it == loop->annotations.end()) {
      return std::nullopt;
    }
    return Downcast<Array<PrimExpr>>((*it).second);
  }

  /*!
   * \brief Update loop body while preserving other fields.
   *
   * This helper avoids unnecessary CopyOnWrite when body is unchanged.
   */
  Stmt UpdateBody(const ForNode *loop, Stmt new_body) {
    if (new_body.same_as(loop->body)) {
      return ffi::GetRef<For>(loop);
    }
    For f = ffi::GetRef<For>(loop);
    f.CopyOnWrite()->body = new_body;
    return f;
  }

  /*!
   * \brief Visit ForNode (post-order).
   *
   * Execution order:
   *   1. Recursively visit body.
   *   2. Gate by tile_execution.
   *   3. Try to match 2D tile pattern.
   *   4. construct two new ForNodes if matched.
   */
  Stmt VisitStmt_(const ForNode *loop) final {
    // ------------------------------------------------------------
    // (1) Post-order: first visit the body
    // ------------------------------------------------------------
    Stmt new_body = VisitStmt(loop->body);

    // ------------------------------------------------------------
    // (2) Scope gate: only care about T.Tiles() loops
    // ------------------------------------------------------------
    if (!IsTilesScope(loop)) {
      // Not a Tiles loop: just propagate rewritten body
      return UpdateBody(loop, new_body);
    }

    LOG(INFO) << "[TilesLoop] Visiting tile loop: "
              << loop->loop_var->name_hint;

    // ------------------------------------------------------------
    // (3) Structural pattern matching:
    //     for i:
    //       for j:
    //         BODY
    // ------------------------------------------------------------
    const ForNode *inner = new_body.as<ForNode>();
    if (!inner) {
      // Body is not a loop → cannot form a 2D tile
      LOG(INFO) << "[TilesLoop]  Body is not a ForNode, skip lowering";
      return UpdateBody(loop, new_body);
    }

    if (!IsSerialFor(loop) || !IsSerialFor(inner)) {
      // Only handle serial-serial pattern
      LOG(INFO) << "[TilesLoop]  Non-serial loop detected, skip lowering";
      return UpdateBody(loop, new_body);
    }

    if (!IsTilesScope(inner)) {
      // Inner loop must also belong to Tiles scope
      LOG(INFO)
          << "[TilesLoop]  Inner loop is not tile_execution, skip lowering";
      return UpdateBody(loop, new_body);
    }

    // ------------------------------------------------------------
    // (4) Read tile size (must exist and be 2D)
    // ------------------------------------------------------------
    auto tile_size_opt = GetTileSize(loop);
    if (!tile_size_opt.defined()) {
      return UpdateBody(loop, new_body);
    }

    Array<PrimExpr> tile_size = tile_size_opt.value();
    ICHECK_EQ(tile_size.size(), 2) << "TilesLoop expects exactly 2D tile_size";

    LOG(INFO) << "[TilesLoop]  Performing 2D tile lowering";

    // ------------------------------------------------------------
    // (5) Perform tile lowering, construct and insert new ForNodes
    // ------------------------------------------------------------
    Var ti = loop->loop_var;
    Var tj = inner->loop_var;

    // Tile-inner loop variables
    Var ki("ki");
    Var kj("kj");

    // Index substitution:
    //   i -> i * Ts0 + ki
    //   j -> j * Ts1 + kj
    Map<Var, PrimExpr> vmap;
    vmap.Set(ti, ti * tile_size[0] + ki);
    vmap.Set(tj, tj * tile_size[1] + kj);

    // Apply substitution to the original tile body
    Stmt tiled_body = Substitute(inner->body, vmap);

    // Construct inner tile loops:
    //   ki : serial
    //   kj : vectorized
    tiled_body = For(kj, 0, tile_size[1], ForKind::kVectorized, tiled_body);

    tiled_body = For(ki, 0, tile_size[0], ForKind::kSerial, tiled_body);

    // Replace the original inner loop body
    For new_inner = ffi::GetRef<For>(inner);
    new_inner.CopyOnWrite()->body = tiled_body;
    // Replace the original outer loop body
    For new_outer = ffi::GetRef<For>(loop);
    new_outer.CopyOnWrite()->body = new_inner;

    LOG(INFO) << "[TilesLoop]  Tile lowering done at loop: "
              << loop->loop_var->name_hint;

    return new_outer;
  }
};

using namespace tir::transform;

/*!
 * \brief Create TilesLoop pass.
 */
Pass TilesLoop() {
  auto pass_func = [](PrimFunc f, const IRModule &,
                      const PassContext &) -> PrimFunc {
    return TilesLoopRewriter::Rewrite(std::move(f));
  };

  return CreatePrimFuncPass(pass_func,
                            /*opt_level=*/0, "tl.TilesLoop", {});
}

/*!
 * \brief FFI registration.
 */
TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def("tl.transform.TilesLoop", TilesLoop);
}

} // namespace tl
} // namespace tvm
