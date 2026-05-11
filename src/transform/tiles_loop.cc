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
                     const Array<PrimExpr> &tile_size,
                     Optional<PrimExpr> tile_predicate)
      : exec_vars_(exec_vars), interior_vars_(interior_vars),
        tile_size_(tile_size), tile_predicate_(std::move(tile_predicate)) {
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
  struct ExecutionIndexInfo {
    int matched_axis;
    PrimExpr base;
  };

  struct NormalizedAccessIndex {
    PrimExpr index;
    std::optional<int> matched_axis;
  };

  bool UsesExecVar(const PrimExpr &expr) const {
    return UsesVar(expr, [this](const VarNode *node) {
      return exec_var_nodes_.count(node) != 0;
    });
  }

  std::optional<ExecutionIndexInfo>
  AnalyzeExecutionIndex(const PrimExpr &index) {
    if (!UsesExecVar(index)) {
      return std::nullopt;
    }

    Array<Var> exec_vars;
    for (const Var &var : exec_vars_) {
      exec_vars.push_back(var);
    }

    Array<PrimExpr> coeffs = arith::DetectLinearEquation(index, exec_vars);
    ICHECK(!coeffs.empty())
        << "Internal error: TilesLoop expected LegalizeTilesLoop to "
           "pre-validate an affine tile access index, but got "
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
    return ExecutionIndexInfo{matched_axis, base};
  }

  NormalizedAccessIndex NormalizeAccessIndex(const PrimExpr &index) {
    std::optional<ExecutionIndexInfo> execution_index =
        AnalyzeExecutionIndex(index);
    if (!execution_index.has_value()) {
      return {analyzer_.Simplify(index), std::nullopt};
    }

    // LegalizeTilesLoop already proved that every execution-bound access is
    // affine, unit-coefficient, and tile-aligned. Lowering only needs to
    // recover the matched execution axis and fold the offset into tile space.
    int matched_axis = execution_index->matched_axis;
    PrimExpr base = execution_index->base;
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
    return {tile_coord * tile_extent + interior_vars_[matched_axis],
            matched_axis};
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    Var var = ffi::GetRef<Var>(op);
    auto it = subst_map_.find(var);
    if (it != subst_map_.end()) {
      return it->second;
    }
    return var;
  }

  Optional<PrimExpr> RewritePredicate(Optional<PrimExpr> predicate) {
    if (predicate.defined()) {
      predicate = VisitExpr(predicate.value());
    }
    if (!tile_predicate_.defined()) {
      return predicate;
    }
    if (!predicate.defined()) {
      return tile_predicate_;
    }
    return And(predicate.value(), tile_predicate_.value());
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    Optional<PrimExpr> predicate = RewritePredicate(op->predicate);

    if (op->indices.size() == 1) {
      NormalizedAccessIndex normalized = NormalizeAccessIndex(op->indices[0]);
      Array<PrimExpr> indices = {normalized.index};
      return BufferLoad(op->buffer, indices, predicate, op->span);
    }

    Array<PrimExpr> indices;
    for (const PrimExpr &index : op->indices) {
      indices.push_back(NormalizeAccessIndex(index).index);
    }
    return BufferLoad(op->buffer, indices, predicate, op->span);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    Array<PrimExpr> indices;
    for (const PrimExpr &index : op->indices) {
      indices.push_back(NormalizeAccessIndex(index).index);
    }
    PrimExpr value = VisitExpr(op->value);
    Optional<PrimExpr> predicate = RewritePredicate(op->predicate);
    return BufferStore(op->buffer, value, indices, predicate, op->span);
  }

  arith::Analyzer analyzer_;
  std::vector<Var> exec_vars_;
  std::vector<Var> interior_vars_;
  Array<PrimExpr> tile_size_;
  Optional<PrimExpr> tile_predicate_;
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

  static Optional<Array<PrimExpr>> GetTileDomain(const ForNode *loop) {
    auto it = loop->annotations.find(attr::kTileDomain);
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

  static Optional<PrimExpr> Conjoin(Optional<PrimExpr> predicate,
                                    PrimExpr clause) {
    return predicate.defined() ? And(predicate.value(), clause) : clause;
  }

  struct TilePredicateInfo {
    // No access_predicate: every tile is in bounds; emit the unpredicated body.
    // access_predicate without full_tile_predicate: emit only the predicated
    // body. Both predicates: emit scalar full/tail control flow.
    Optional<PrimExpr> access_predicate;
    Optional<PrimExpr> full_tile_predicate;
  };

  TilePredicateInfo MakeTilePredicateInfo(
      const std::vector<Var> &exec_vars, const std::vector<Var> &interior_vars,
      const std::vector<const ForNode *> &axis_loops,
      const Array<PrimExpr> &tile_size, const Array<PrimExpr> &domain,
      const Array<PrimExpr> &execution_domain_axes) {
    ICHECK_EQ(exec_vars.size(), interior_vars.size());
    ICHECK_EQ(exec_vars.size(), axis_loops.size());
    ICHECK_EQ(exec_vars.size(), tile_size.size());
    ICHECK_EQ(exec_vars.size(), execution_domain_axes.size());

    TilePredicateInfo info;
    bool may_emit_full_tile_branch = true;

    int domain_rank = static_cast<int>(domain.size());
    for (size_t axis = 0; axis < exec_vars.size(); ++axis) {
      int domain_axis =
          NormalizeDomainAxis(execution_domain_axes[axis], domain_rank);
      const ForNode *axis_loop = axis_loops[axis];
      ICHECK(axis_loop != nullptr)
          << "TilesLoop expected one execution loop per tile axis.";

      PrimExpr tile_extent = tile_size[axis];
      PrimExpr domain_extent = domain[domain_axis];

      // If the last planned tile is still inside the logical domain, this axis
      // cannot create out-of-bounds accesses.
      PrimExpr last_tile_end = analyzer_.Simplify(
          (axis_loop->min + axis_loop->extent) * tile_extent);
      if (analyzer_.CanProve(last_tile_end <= domain_extent)) {
        continue;
      }

      // This axis has a tail tile. Every access in the tile body must be
      // guarded by the per-element logical coordinate on this axis.
      PrimExpr logical_index =
          exec_vars[axis] * tile_extent + interior_vars[axis];
      info.access_predicate =
          Conjoin(info.access_predicate, logical_index < domain_extent);

      if (!may_emit_full_tile_branch) {
        continue;
      }

      // If the first tile in this loop is already partial, there is no
      // unpredicated full-tile case for the whole tile body.
      PrimExpr first_tile_end =
          analyzer_.Simplify((axis_loop->min + Integer(1)) * tile_extent);
      if (analyzer_.CanProve(first_tile_end > domain_extent)) {
        may_emit_full_tile_branch = false;
        info.full_tile_predicate = std::nullopt;
        continue;
      }

      // Otherwise, this axis contributes one scalar condition for choosing
      // between the unpredicated full-tile body and the predicated tail body.
      PrimExpr current_tile_end =
          analyzer_.Simplify(exec_vars[axis] * tile_extent + tile_extent);
      info.full_tile_predicate =
          Conjoin(info.full_tile_predicate, current_tile_end <= domain_extent);
    }

    if (!info.access_predicate.defined()) {
      return info;
    }

    if (!may_emit_full_tile_branch) {
      info.full_tile_predicate = std::nullopt;
      return info;
    }

    ICHECK(info.full_tile_predicate.defined())
        << "TilesLoop expected a scalar full-tile predicate when some full "
           "tiles exist.";
    PrimExpr predicate = analyzer_.Simplify(info.full_tile_predicate.value());
    if (analyzer_.CanProve(predicate)) {
      return TilePredicateInfo{};
    }
    if (analyzer_.CanProve(!predicate)) {
      info.full_tile_predicate = std::nullopt;
      return info;
    }
    info.full_tile_predicate = predicate;
    return info;
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

  template <typename BuildBody>
  static Stmt BuildFullOrTailTileBody(const TilePredicateInfo &predicate_info,
                                      BuildBody build_body) {
    if (!predicate_info.access_predicate.defined()) {
      return build_body(std::nullopt);
    }
    Stmt tail_body = build_body(predicate_info.access_predicate);
    if (!predicate_info.full_tile_predicate.defined()) {
      return tail_body;
    }
    return IfThenElse(predicate_info.full_tile_predicate.value(),
                      build_body(std::nullopt), tail_body);
  }

  static Stmt Build1DInteriorBody(Var ki, const Array<PrimExpr> &tile_size,
                                  Stmt body) {
    return MakeTileInteriorLoop(ki, tile_size[0], ForKind::kVectorized, body,
                                /*axis=*/0);
  }

  static Stmt Build2DInteriorBody(Var ki, Var kj,
                                  const Array<PrimExpr> &tile_size, Stmt body) {
    For kj_loop =
        MakeTileInteriorLoop(kj, tile_size[1], ForKind::kVectorized, body,
                             /*axis=*/1);
    return MakeTileInteriorLoop(ki, tile_size[0], ForKind::kSerial, kj_loop,
                                /*axis=*/0);
  }

  arith::Analyzer analyzer_;

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
    Array<PrimExpr> domain = GetTileDomain(scope_root.get()).value();
    Array<PrimExpr> execution_domain_axes =
        GetExecutionDomainAxes(scope_root.get()).value();
    TilePredicateInfo predicate_info = MakeTilePredicateInfo(
        {ti}, {ki}, {exec_loop}, tile_size, domain, execution_domain_axes);
    auto build_body = [&](Optional<PrimExpr> tile_predicate) {
      TileAccessRewriter access_rewriter({ti}, {ki}, tile_size,
                                         std::move(tile_predicate));
      return Build1DInteriorBody(ki, tile_size,
                                 access_rewriter.Rewrite(exec_loop->body));
    };

    For new_exec = ffi::GetRef<For>(exec_loop);
    auto *n = new_exec.CopyOnWrite();
    n->body = BuildFullOrTailTileBody(predicate_info, build_body);
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
    Array<PrimExpr> domain = GetTileDomain(scope_root.get()).value();
    Array<PrimExpr> execution_domain_axes =
        GetExecutionDomainAxes(scope_root.get()).value();
    TilePredicateInfo predicate_info =
        MakeTilePredicateInfo({ti, tj}, {ki, kj}, {axis0_loop, axis1_loop},
                              tile_size, domain, execution_domain_axes);
    auto build_body = [&](Optional<PrimExpr> tile_predicate) {
      TileAccessRewriter access_rewriter({ti, tj}, {ki, kj}, tile_size,
                                         std::move(tile_predicate));
      return Build2DInteriorBody(ki, kj, tile_size,
                                 access_rewriter.Rewrite(innermost_exec->body));
    };

    // Replace the innermost execution loop body with the tiled loops.
    For new_inner_exec = ffi::GetRef<For>(innermost_exec);
    new_inner_exec.CopyOnWrite()->body =
        BuildFullOrTailTileBody(predicate_info, build_body);

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
