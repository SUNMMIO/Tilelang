#include <algorithm>
#include <unordered_map>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../layout/layout.h"
#include "../support/ffi_aliases.h"
#include "../tileview/tileview.h"
#include "../tileview/tileview_planner.h"
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

class NestedTilesScopeDetector : public StmtExprVisitor {
public:
  static bool Exists(const Stmt &stmt) {
    NestedTilesScopeDetector detector;
    detector(stmt);
    return detector.found_;
  }

private:
  void VisitStmt_(const ForNode *loop) final {
    if (loop->annotations.count(attr::kTileDomain)) {
      found_ = true;
      return;
    }
    StmtExprVisitor::VisitStmt_(loop);
  }

  bool found_{false};
};

/* ============================================================
 * LegalizeTilesLoopRewriter
 *
 * Rewrite tile-level For loops using a TileView plan solved by
 * the shared tileview planner.
 * ============================================================ */
class LegalizeTilesLoopRewriter : public StmtExprMutator {
public:
  static PrimFunc Rewrite(PrimFunc f) {
    LegalizeTilesLoopRewriter rewriter;
    rewriter.tileviews_ = TileViewCollector::Collect(f);
    rewriter.tile_processor_config_ =
        GetSunmmioTileProcessorConfig(f->GetAttr<Target>("target"));

    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

private:
  static void AddBufferIfMissing(std::vector<Buffer> *buffers,
                                 const Buffer &buffer) {
    auto it = std::find_if(
        buffers->begin(), buffers->end(),
        [&](const Buffer &candidate) { return candidate.same_as(buffer); });
    if (it == buffers->end()) {
      buffers->push_back(buffer);
    }
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

  /* ---- Collect buffer and layout info from Block nodes ---- */
  Stmt VisitStmt_(const BlockNode *block) final {
    // Collect layout_map from block annotation
    if (block->annotations.count(attr::kLayoutMap)) {
      auto layout_map_obj = block->annotations.Get(attr::kLayoutMap).value();
      if (auto layout_map = layout_map_obj.as<Map<Buffer, Layout>>()) {
        for (const auto &[buffer, layout] : layout_map.value()) {
          layout_map_.Set(buffer, layout);
        }
      } else if (auto layout_map = layout_map_obj.as<Map<Var, Layout>>()) {
        std::vector<Buffer> block_buffers;
        for (const Buffer &buffer : block->alloc_buffers) {
          AddBufferIfMissing(&block_buffers, buffer);
        }
        for (const BufferRegion &region : block->reads) {
          AddBufferIfMissing(&block_buffers, region->buffer);
        }
        for (const BufferRegion &region : block->writes) {
          AddBufferIfMissing(&block_buffers, region->buffer);
        }
        for (const MatchBufferRegion &match_buffer : block->match_buffers) {
          AddBufferIfMissing(&block_buffers, match_buffer->buffer);
        }

        for (const auto &[buffer_var, layout] : layout_map.value()) {
          bool found = false;
          for (const Buffer &buffer : block_buffers) {
            if (buffer->data.same_as(buffer_var)) {
              layout_map_.Set(buffer, layout);
              found = true;
            }
          }
          ICHECK(found)
              << "layout_map annotation references unknown buffer var "
              << buffer_var << ".";
        }
      } else {
        LOG(FATAL)
            << "Unsupported layout_map annotation type in LegalizeTilesLoop.";
      }
    }

    return StmtExprMutator::VisitStmt_(block);
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

    bool starts_scope = loop->annotations.count(attr::kTileDomain);
    if (starts_scope) {
      ICHECK(!in_active_scope_) << "Nested T.Tiles scopes are not supported.";

      Array<PrimExpr> domain =
          Downcast<Array<PrimExpr>>(loop->annotations.at(attr::kTileDomain));
      ICHECK(!NestedTilesScopeDetector::Exists(loop->body))
          << "Nested T.Tiles scopes are not supported.";
      auto scope_loops = CollectTileLoopChain(loop);
      ICHECK_GE(scope_loops.size(), domain.size())
          << "T.Tiles scope loop rank does not cover the declared domain rank.";
      scope_loops.resize(domain.size());
      auto accesses = CollectBufferAccesses(loop->body);
      active_tileview_plan_ =
          PlanTileViewsForTilesScope(domain, scope_loops, accesses, tileviews_,
                                     layout_map_, tile_processor_config_);
      active_scope_depth_ = 0;
      in_active_scope_ = true;
    }

    ICHECK(in_active_scope_)
        << "Tile loop encountered without an active T.Tiles domain root.";

    // Enter tile loop (depth == tile dimension)
    int dim = active_scope_depth_++;
    Stmt new_body = VisitStmt(loop->body);
    active_scope_depth_--;

    const TileViewPlan &plan = active_tileview_plan_;
    const TileView &tv = plan.execution_tileview;
    Array<PrimExpr> tiled_shape = tv->TiledBufferShape();

    ICHECK(dim < static_cast<int>(tiled_shape.size()))
        << "Tile loop depth exceeds tiled buffer rank";

    // Rewrite loop
    For new_for = ffi::GetRef<For>(loop);
    auto *n = new_for.CopyOnWrite();
    n->extent = tiled_shape[dim];
    n->body = new_body;

    // Attach normalized loop annotations
    n->annotations.Set(attr::tile_tile_size, tv->TileShape());
    n->annotations.Set(attr::kTileLoopStage,
                       Integer(static_cast<int>(TileLoopStage::kLegalized)));
    // ---- Determine whether this logical domain loop carries an execution axis
    // If plan.execution_domain_axes[k] == dim, then:
    //   - this loop carries execution axis k
    //   - tile.tile_size[k] applies to this logical domain axis
    // Example: execution_domain_axes=[1,0] means tile_size[0] belongs to the
    // second logical loop axis, and tile_size[1] belongs to the first.
    auto axis_it = std::find(plan.execution_domain_axes.begin(),
                             plan.execution_domain_axes.end(), dim);
    bool is_tile_execution = axis_it != plan.execution_domain_axes.end();

    if (is_tile_execution) {
      n->annotations.Set(attr::tile_execution_axis,
                         Integer(static_cast<int>(std::distance(
                             plan.execution_domain_axes.begin(), axis_it))));
    }

    if (starts_scope) {
      // Scope-level execution axis -> logical domain axis mapping.
      Array<PrimExpr> execution_domain_axes;
      for (int axis : plan.execution_domain_axes) {
        execution_domain_axes.push_back(Integer(axis));
      }
      n->annotations.Set(attr::tile_execution_domain_axes,
                         execution_domain_axes);
    }

    if (starts_scope) {
      in_active_scope_ = false;
      active_tileview_plan_ = TileViewPlan();
      active_scope_depth_ = 0;
    }
    return new_for;
  }

private:
  TileViewMap tileviews_;
  Map<Buffer, Layout> layout_map_;
  SunmmioTileProcessorConfig tile_processor_config_{
      GetSunmmioTileProcessorConfig(ffi::Optional<Target>())};
  bool in_active_scope_{false};
  TileViewPlan active_tileview_plan_;
  int active_scope_depth_{0};
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
