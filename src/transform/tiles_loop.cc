#include <optional>
#include <unordered_map>
#include <unordered_set>

#include <tvm/arith/analyzer.h>
#include <tvm/arith/pattern.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../support/ffi_aliases.h"
#include "../tileview/tileview.h"
#include "common/attr.h"

namespace tvm {
namespace tl {

using namespace tir;

class FinalTileAttrCleanupRewriter : public StmtExprMutator {
public:
  static PrimFunc Rewrite(PrimFunc f) {
    FinalTileAttrCleanupRewriter rewriter;
    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

private:
  Stmt VisitStmt_(const ForNode *loop) final {
    For f = Downcast<For>(StmtExprMutator::VisitStmt_(loop));
    auto *n = f.CopyOnWrite();
    bool is_scope_root = n->annotations.count(attr::kTileDomain);

    n->annotations.erase(attr::kTileLoopStage);
    n->annotations.erase(attr::tile_level_loop);

    if (!is_scope_root) {
      n->annotations.erase(attr::tile_tile_size);
      n->annotations.erase(attr::tile_execution_domain_axes);
    }
    return f;
  }
};

class TileAccessRewriter : public StmtExprMutator {
public:
  TileAccessRewriter(const std::vector<Var> &exec_vars,
                     const std::vector<Var> &interior_vars,
                     const Array<PrimExpr> &tile_size)
      : exec_vars_(exec_vars), interior_vars_(interior_vars),
        tile_size_(tile_size) {
    ICHECK_EQ(exec_vars_.size(), interior_vars_.size());
    ICHECK_EQ(exec_vars_.size(), tile_size_.size());

    for (size_t i = 0; i < exec_vars_.size(); ++i) {
      exec_var_nodes_.insert(exec_vars_[i].get());
      subst_map_.emplace(exec_vars_[i],
                         exec_vars_[i] * tile_size_[i] + interior_vars_[i]);
    }
  }

  Stmt Rewrite(const Stmt &stmt) { return VisitStmt(stmt); }

private:
  bool UsesExecVar(const PrimExpr &expr) const {
    return UsesVar(expr, [this](const VarNode *node) {
      return exec_var_nodes_.count(node) != 0;
    });
  }

  PrimExpr NormalizeAccessIndex(const PrimExpr &index) {
    if (!UsesExecVar(index)) {
      return analyzer_.Simplify(index);
    }

    // LegalizeTilesLoop already proved that every execution-bound access is
    // affine, unit-coefficient, and tile-aligned. Lowering only needs to
    // recover the matched execution axis and fold the offset into tile space.
    Array<Var> exec_vars;
    for (const Var &var : exec_vars_) {
      exec_vars.push_back(var);
    }

    Array<PrimExpr> coeffs = arith::DetectLinearEquation(index, exec_vars);
    ICHECK(!coeffs.empty()) << "Internal error: TilesLoop expected "
                               "LegalizeTilesLoop to pre-validate "
                               "an affine tile access index, but got "
                            << index << ".";

    int matched_axis = -1;
    PrimExpr base = analyzer_.Simplify(coeffs[coeffs.size() - 1]);

    for (size_t i = 0; i < exec_vars_.size(); ++i) {
      PrimExpr coeff = analyzer_.Simplify(coeffs[i]);
      PrimExpr zero = make_zero(coeff.dtype());
      PrimExpr one = make_const(coeff.dtype(), 1);

      if (analyzer_.CanProve(coeff == zero)) {
        continue;
      }
      ICHECK(analyzer_.CanProve(coeff == one))
          << "Internal error: TilesLoop expected a legalized tile access index "
             "to use tile execution vars with unit coefficient, but got "
             "coefficient "
          << coeff << " in " << index << ".";
      ICHECK_EQ(matched_axis, -1)
          << "Internal error: TilesLoop expected a legalized tile access index "
             "to depend on at most one tile execution var, but got "
          << index << ".";
      matched_axis = static_cast<int>(i);
    }

    ICHECK_GE(matched_axis, 0)
        << "Internal error: TilesLoop expected a legalized tile access index "
           "to bind one tile execution axis, but no matching axis was found "
           "for "
        << index << ".";

    PrimExpr tile_extent = tile_size_[matched_axis];
    PrimExpr remainder = analyzer_.Simplify(floormod(base, tile_extent));
    ICHECK(analyzer_.CanProve(remainder == make_zero(remainder.dtype())))
        << "Internal error: TilesLoop expected LegalizeTilesLoop to "
           "pre-normalize "
           "tile access offsets, but got offset "
        << base << " with non-zero intra-tile remainder for index " << index
        << ".";

    PrimExpr tile_offset = analyzer_.Simplify(floordiv(base, tile_extent));
    PrimExpr tile_coord =
        analyzer_.Simplify(exec_vars_[matched_axis] + tile_offset);
    return tile_coord * tile_extent + interior_vars_[matched_axis];
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    Var var = ffi::GetRef<Var>(op);
    auto it = subst_map_.find(var);
    if (it != subst_map_.end()) {
      return it->second;
    }
    return var;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    Array<PrimExpr> indices;
    for (const PrimExpr &index : op->indices) {
      indices.push_back(NormalizeAccessIndex(index));
    }
    return BufferLoad(op->buffer, indices, op->predicate, op->span);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    Array<PrimExpr> indices;
    for (const PrimExpr &index : op->indices) {
      indices.push_back(NormalizeAccessIndex(index));
    }
    PrimExpr value = VisitExpr(op->value);
    Optional<PrimExpr> predicate = op->predicate;
    if (predicate.defined()) {
      predicate = VisitExpr(predicate.value());
    }
    return BufferStore(op->buffer, value, indices, predicate, op->span);
  }

  arith::Analyzer analyzer_;
  std::vector<Var> exec_vars_;
  std::vector<Var> interior_vars_;
  Array<PrimExpr> tile_size_;
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> subst_map_;
  std::unordered_set<const VarNode *> exec_var_nodes_;
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
 *
 * The 2D path is tolerant of non-trivial nesting between the
 * outer and inner tile.execution_axis loops (LetStmt, AttrStmt, etc.
 * may appear in between).
 * ------------------------------------------------------------
 */
class TilesLoopRewriter : public StmtExprMutator {
public:
  static PrimFunc Rewrite(PrimFunc f) {
    TilesLoopRewriter rewriter;
    f.CopyOnWrite()->body = rewriter(f->body);
    f = FinalTileAttrCleanupRewriter::Rewrite(std::move(f));
    return f;
  }

private:
  // ---- Predicates ----

  static bool IsTileExecutionLoop(const ForNode *loop) {
    return loop->annotations.count(attr::tile_execution_axis);
  }

  static bool IsTilesRoot(const ForNode *loop) {
    return loop->annotations.count(attr::kTileDomain);
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

  static Optional<Array<PrimExpr>> GetExecutionDomainAxes(const ForNode *loop) {
    auto it = loop->annotations.find(attr::tile_execution_domain_axes);
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

  struct ScopeExecInfo {
    std::vector<const ForNode *> chain;
    std::vector<const ForNode *> axis_to_loop;
    const ForNode *outermost_exec{nullptr};
    const ForNode *innermost_exec{nullptr};
  };

  static int NormalizeDomainAxis(const PrimExpr &expr, int rank) {
    const auto *imm = expr.as<IntImmNode>();
    ICHECK(imm) << "tile.execution_domain_axes entries must be IntImm, but got "
                << expr;
    int mapped_dim = static_cast<int>(imm->value);
    if (mapped_dim < 0) {
      mapped_dim += rank;
    }
    ICHECK(mapped_dim >= 0 && mapped_dim < rank)
        << "tile.execution_domain_axes entry " << expr
        << " is out of bounds for rank " << rank << ".";
    return mapped_dim;
  }

  static std::vector<const ForNode *>
  CollectTileLoopChain(const ForNode *root) {
    std::vector<const ForNode *> loops;
    const ForNode *current = root;
    while (current != nullptr &&
           current->annotations.count(attr::tile_level_loop)) {
      loops.push_back(current);
      const auto *next = current->body.as<ForNode>();
      if (next == nullptr || !next->annotations.count(attr::tile_level_loop)) {
        break;
      }
      current = next;
    }
    return loops;
  }

  static std::optional<ScopeExecInfo>
  AnalyzeScopeExecution(const ForNode *root) {
    ScopeExecInfo info;
    info.chain = CollectTileLoopChain(root);
    int domain_rank = static_cast<int>(info.chain.size());

    // Decode the explicit execution-axis mapping emitted by
    // LegalizeTilesLoop. TilesLoop should not be inferring this again.
    auto execution_domain_axes_opt = GetExecutionDomainAxes(root);
    ICHECK(execution_domain_axes_opt.defined())
        << "TilesLoop requires tile.execution_domain_axes on the T.Tiles scope "
           "root.";

    Array<PrimExpr> execution_domain_axes = execution_domain_axes_opt.value();
    info.axis_to_loop.resize(execution_domain_axes.size(), nullptr);

    for (int axis = 0; axis < static_cast<int>(execution_domain_axes.size());
         ++axis) {
      int mapped_dim =
          NormalizeDomainAxis(execution_domain_axes[axis], domain_rank);
      const ForNode *mapped_loop = info.chain[mapped_dim];
      ICHECK(IsTileExecutionLoop(mapped_loop))
          << "tile.execution_domain_axes references loop dim " << mapped_dim
          << " that is not marked as an execution loop.";
      auto axis_it = mapped_loop->annotations.find(attr::tile_execution_axis);
      ICHECK(axis_it != mapped_loop->annotations.end())
          << "TilesLoop requires tile.execution_axis on every execution loop.";
      int loop_axis = Downcast<Integer>((*axis_it).second)->value;
      ICHECK_EQ(loop_axis, axis)
          << "Execution loop for domain axis " << mapped_dim
          << " carries tile.execution_axis=" << loop_axis
          << " but was planned as axis " << axis << ".";
      info.axis_to_loop[axis] = mapped_loop;
    }

    for (const ForNode *loop : info.chain) {
      if (!IsTileExecutionLoop(loop)) {
        continue;
      }
      if (info.outermost_exec == nullptr) {
        info.outermost_exec = loop;
      }
      info.innermost_exec = loop;
    }

    return info;
  }

  /*!
   * \brief Replace a specific ForNode within a statement tree.
   *
   * Used to swap the inner tile.execution loop with its rewritten
   * version, preserving any wrapper statements in between.
   */
  class LoopReplacer : public StmtExprMutator {
  public:
    LoopReplacer(const ForNode *target, Stmt replacement)
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
   */
  static For MakeTileInteriorLoop(Var loop_var, PrimExpr extent, ForKind kind,
                                  Stmt body, int axis) {
    For loop(loop_var, Integer(0), extent, kind, body);
    auto *n = loop.CopyOnWrite();
    n->annotations.Set(attr::tile_interior, Integer(1));
    n->annotations.Set(attr::tile_interior_axis, Integer(axis));
    return loop;
  }

  // ---- Main rewrite ----

  Stmt VisitStmt_(const ForNode *loop) final {
    // Post-order traversal
    Stmt new_body = VisitStmt(loop->body);

    if (!IsTilesRoot(loop)) {
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

    if (tile_size.size() == 1) {
      return Rewrite1D(loop, new_body, tile_size);
    }
    return Rewrite2D(loop, new_body, tile_size);
  }

  /*!
   * \brief 1D lowering:
   *   for ti → for ti (scope_entry) { for ki (interior, vec) { body } }
   */
  Stmt Rewrite1D(const ForNode *loop, Stmt new_body,
                 const Array<PrimExpr> &tile_size) {
    For scope_root = ffi::GetRef<For>(loop);
    scope_root.CopyOnWrite()->body = new_body;

    auto scope_info_opt = AnalyzeScopeExecution(scope_root.get());
    if (!scope_info_opt.has_value()) {
      return scope_root;
    }
    ScopeExecInfo &scope_info = scope_info_opt.value();
    const ForNode *exec_loop = scope_info.axis_to_loop[0];
    if (exec_loop == nullptr || !IsSerialFor(exec_loop)) {
      return scope_root;
    }

    Var ti = exec_loop->loop_var;
    Var ki("ki");
    TileAccessRewriter access_rewriter({ti}, {ki}, tile_size);
    Stmt tiled_body = access_rewriter.Rewrite(exec_loop->body);

    // Create annotated interior loop
    For ki_loop =
        MakeTileInteriorLoop(ki, tile_size[0], ForKind::kVectorized, tiled_body,
                             /*axis=*/0);

    For new_exec = ffi::GetRef<For>(exec_loop);
    auto *n = new_exec.CopyOnWrite();
    n->body = ki_loop;
    n->annotations.Set(attr::tile_scope_entry, Integer(1));

    if (exec_loop == scope_root.get()) {
      return new_exec;
    }

    scope_root.CopyOnWrite()->body =
        LoopReplacer(exec_loop, new_exec)(scope_root->body);
    return scope_root;
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
                 const Array<PrimExpr> &tile_size) {
    For scope_root = ffi::GetRef<For>(loop);
    scope_root.CopyOnWrite()->body = new_body;

    auto scope_info_opt = AnalyzeScopeExecution(scope_root.get());
    if (!scope_info_opt.has_value()) {
      return scope_root;
    }
    ScopeExecInfo &scope_info = scope_info_opt.value();

    const ForNode *outermost_exec = scope_info.outermost_exec;
    const ForNode *innermost_exec = scope_info.innermost_exec;
    const ForNode *axis0_loop = scope_info.axis_to_loop[0];
    const ForNode *axis1_loop = scope_info.axis_to_loop[1];

    if (outermost_exec == nullptr || innermost_exec == nullptr ||
        axis0_loop == nullptr || axis1_loop == nullptr ||
        !IsSerialFor(axis0_loop) || !IsSerialFor(axis1_loop)) {
      return scope_root;
    }

    Var ti = axis0_loop->loop_var;
    Var tj = axis1_loop->loop_var;

    Var ki("ki");
    Var kj("kj");
    TileAccessRewriter access_rewriter({ti, tj}, {ki, kj}, tile_size);
    Stmt tiled_body = access_rewriter.Rewrite(innermost_exec->body);

    // Create annotated interior loops
    For kj_loop =
        MakeTileInteriorLoop(kj, tile_size[1], ForKind::kVectorized, tiled_body,
                             /*axis=*/1);
    For ki_loop =
        MakeTileInteriorLoop(ki, tile_size[0], ForKind::kSerial, kj_loop,
                             /*axis=*/0);

    // Replace the innermost execution loop body with the tiled loops.
    For new_inner_exec = ffi::GetRef<For>(innermost_exec);
    new_inner_exec.CopyOnWrite()->body = ki_loop;

    Stmt replaced_exec_body = LoopReplacer(innermost_exec, new_inner_exec)(
        ffi::GetRef<For>(outermost_exec)->body);

    For new_outer_exec = ffi::GetRef<For>(outermost_exec);
    auto *n = new_outer_exec.CopyOnWrite();
    n->body = replaced_exec_body;
    n->annotations.Set(attr::tile_scope_entry, Integer(1));

    if (outermost_exec == scope_root.get()) {
      return new_outer_exec;
    }

    scope_root.CopyOnWrite()->body =
        LoopReplacer(outermost_exec, new_outer_exec)(scope_root->body);
    return scope_root;
  }
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
