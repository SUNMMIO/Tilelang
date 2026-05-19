/*!
 * \file pass.cc
 * \brief Entry point and rewrite stage for the Sunmmio tile loop fusion pass.
 *
 * \details
 * This pass solves one specific problem in lowered Sunmmio TIR: after tile
 * lowering, many adjacent `tile.scope_entry` regions still execute as separate
 * loop nests even when they touch the same logical execution prefix and could
 * profitably share outer loops. Leaving them separate cuts producer/consumer
 * reuse across region boundaries, forces extra materialization of intermediate
 * values, and misses opportunities to keep data alive inside shared shells.
 *
 * Intuitively, the pass tries to turn a source-order run of compatible tile
 * regions into a fused tree of shared execution shells. Instead of replaying
 * each region as a standalone loop nest, it discovers which outer execution
 * loops can be shared, chooses an order and shell structure that preserves the
 * most useful reuse, and then rewrites the original statement interval into
 * that fused tree.
 *
 * The planning problem is formulated window-by-window. Discovery first builds a
 * dependence graph over each maximal source-order run of planner-visible tile
 * regions. The planner then solves a dynamic-programming search over partial
 * schedules: each action chooses the next region plus how much of the current
 * shared-shell stack to keep or reopen. The objective is a lexicographic cost
 * model rather than a single weighted scalar, so the solver first minimizes cut
 * RAW reuse, then uncovered shared reads, then live resident footprint, and
 * finally source-order inversions.
 *
 * Structurally, the pass is split into four stages:
 * 1. Discovery: find planner-visible tile regions and partition them into
 *    source-order windows.
 * 2. Problem building: normalize external accesses and construct the per-window
 *    dependence graph.
 * 3. Planning: solve each window as a shared-shell scheduling problem and emit
 *    a rewrite-ready tree.
 * 4. Rewrite: replace each original window in the TIR with the chosen fused
 *    shell tree.
 *
 * This file contains stage 4 plus the top-level pass entry that orchestrates
 * the full pipeline.
 */

#include "discovery.h"
#include "planner.h"
#include "utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/node/structural_equal.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {
namespace {

using namespace tir;

/*!
 * \brief Replace one specific scope-entry loop with a prebuilt leaf body.
 *
 * Region leaves keep their original wrapper structure, but when a planner tree
 * places a region under shared outer loops we must cut away the shared prefix
 * from the original scope-entry loop nest and splice in only the private
 * execution suffix.
 */
class ScopeEntryReplacer : public StmtExprMutator {
public:
  ScopeEntryReplacer(const For &target, const Stmt &replacement)
      : target_(target.get()), replacement_(replacement) {}

  Stmt Rewrite(const Stmt &stmt) { return VisitStmt(stmt); }

private:
  Stmt VisitStmt_(const ForNode *op) final {
    if (op == target_) {
      return replacement_;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  const ForNode *target_;
  Stmt replacement_;
};

Map<Var, PrimExpr>
BuildSharedLoopVarSubst(const TileScopeRegion &region,
                        const std::vector<Var> &shared_loop_vars) {
  Map<Var, PrimExpr> subst;
  int limit = std::min(static_cast<int>(region.execution_loops.size()),
                       static_cast<int>(shared_loop_vars.size()));
  for (int i = 0; i < limit; ++i) {
    subst.Set(region.execution_loops[i]->loop_var, shared_loop_vars[i]);
  }
  return subst;
}

bool AttrReferencesExecutionLoops(const AttrStmt &attr,
                                  const TileScopeRegion &region) {
  std::unordered_set<const VarNode *> execution_vars;
  for (const For &loop : region.execution_loops) {
    execution_vars.insert(loop->loop_var.get());
  }

  VarUseCollector collector;
  if (const auto *node_expr = attr->node.as<PrimExprNode>()) {
    collector(ffi::GetRef<PrimExpr>(node_expr));
  }
  collector(attr->value);
  for (const VarNode *var : collector.seen_vars) {
    if (execution_vars.count(var) != 0) {
      return true;
    }
  }
  return false;
}

// AttrStmt wrappers that do not depend on execution-loop vars can be hoisted
// once around a fused shell instead of being duplicated on every child leaf.
std::vector<AttrStmt>
ExtractLeadingHoistableAttrPrefix(const TileScopeRegion &region) {
  std::vector<AttrStmt> attrs;
  Stmt current = region.root_stmt;
  while (!current.same_as(region.scope_entry_for)) {
    if (const auto *attr = current.as<AttrStmtNode>()) {
      AttrStmt attr_stmt = ffi::GetRef<AttrStmt>(attr);
      if (AttrReferencesExecutionLoops(attr_stmt, region)) {
        break;
      }
      attrs.push_back(attr_stmt);
      current = attr->body;
      continue;
    }
    break;
  }
  return attrs;
}

Stmt StripLeadingHoistableAttrPrefix(const TileScopeRegion &region,
                                     int strip_count) {
  Stmt current = region.root_stmt;
  for (int i = 0; i < strip_count; ++i) {
    const auto *attr = current.as<AttrStmtNode>();
    ICHECK(attr != nullptr)
        << "Expected hoistable AttrStmt prefix while rewriting region "
        << region.root_name;
    current = attr->body;
  }
  return current;
}

bool SameHoistableAttrFrame(const AttrStmt &lhs, const AttrStmt &rhs) {
  return lhs->attr_key == rhs->attr_key &&
         StructuralEqual()(lhs->node, rhs->node) &&
         StructuralEqual()(lhs->value, rhs->value);
}

std::vector<int>
CollectLeafRegionIndices(const SunmmioTileLoopFusionPlannerTreeNode &node) {
  if (!node.is_scope) {
    return {node.region_index};
  }
  std::vector<int> indices;
  for (const SunmmioTileLoopFusionPlannerTreeNode &child : node.children) {
    std::vector<int> child_indices = CollectLeafRegionIndices(child);
    indices.insert(indices.end(), child_indices.begin(), child_indices.end());
  }
  return indices;
}

std::vector<AttrStmt> CollectCommonHoistableAttrPrefix(
    const SunmmioTileLoopFusionPlannerTreeNode &node,
    const std::vector<TileScopeRegion> &regions, int already_hoisted_attrs) {
  std::vector<int> leaf_region_indices = CollectLeafRegionIndices(node);
  if (leaf_region_indices.empty()) {
    return {};
  }

  std::vector<AttrStmt> common =
      ExtractLeadingHoistableAttrPrefix(regions[leaf_region_indices.front()]);
  if (already_hoisted_attrs > static_cast<int>(common.size())) {
    return {};
  }
  common.erase(common.begin(), common.begin() + already_hoisted_attrs);

  for (size_t leaf_index = 1;
       leaf_index < leaf_region_indices.size() && !common.empty();
       ++leaf_index) {
    std::vector<AttrStmt> attrs = ExtractLeadingHoistableAttrPrefix(
        regions[leaf_region_indices[leaf_index]]);
    if (already_hoisted_attrs > static_cast<int>(attrs.size())) {
      return {};
    }
    attrs.erase(attrs.begin(), attrs.begin() + already_hoisted_attrs);

    size_t prefix_len = 0;
    while (prefix_len < common.size() && prefix_len < attrs.size() &&
           SameHoistableAttrFrame(common[prefix_len], attrs[prefix_len])) {
      ++prefix_len;
    }
    common.resize(prefix_len);
  }
  return common;
}

Stmt ReapplyAttrPrefix(const std::vector<AttrStmt> &attrs, Stmt body) {
  for (auto it = attrs.rbegin(); it != attrs.rend(); ++it) {
    body =
        AttrStmt((*it)->node, (*it)->attr_key, (*it)->value, body, (*it)->span);
  }
  return body;
}

// A leaf keeps only the execution suffix below the shared shell depth. The
// outer shared loops are rebuilt by the enclosing scope node and substituted
// into the region body here.
Stmt BuildPrivateExecutionSuffix(const TileScopeRegion &region,
                                 int shared_depth,
                                 const std::vector<Var> &shared_loop_vars) {
  Map<Var, PrimExpr> subst = BuildSharedLoopVarSubst(region, shared_loop_vars);
  Stmt body = region.execution_loops.empty()
                  ? region.scope_entry_for->body
                  : region.execution_loops.back()->body;
  body = Substitute(body, subst);

  for (int i = static_cast<int>(region.execution_loops.size()) - 1;
       i >= shared_depth; --i) {
    const For &loop = region.execution_loops[i];
    body = For(loop->loop_var, Substitute(loop->min, subst),
               Substitute(loop->extent, subst), loop->kind, body,
               loop->thread_binding, loop->annotations);
  }
  return body;
}

Stmt BuildRegionLeaf(const TileScopeRegion &region, int shared_depth,
                     const std::vector<Var> &shared_loop_vars,
                     int hoisted_attr_prefix_count) {
  Stmt stripped_root =
      StripLeadingHoistableAttrPrefix(region, hoisted_attr_prefix_count);
  if (shared_depth <= 0) {
    return stripped_root;
  }
  Stmt replacement =
      BuildPrivateExecutionSuffix(region, shared_depth, shared_loop_vars);
  return ScopeEntryReplacer(region.scope_entry_for, replacement)
      .Rewrite(stripped_root);
}

int FindFirstLeafRegionIndex(const SunmmioTileLoopFusionPlannerTreeNode &node) {
  if (!node.is_scope) {
    return node.region_index;
  }
  for (const SunmmioTileLoopFusionPlannerTreeNode &child : node.children) {
    int result = FindFirstLeafRegionIndex(child);
    if (result >= 0) {
      return result;
    }
  }
  return -1;
}

// Materialize one planner-tree sibling list into a flat SeqStmt while
// preserving the planner-chosen execution order among scope nodes and leaves.
Stmt BuildPlannedStmtList(
    const std::vector<SunmmioTileLoopFusionPlannerTreeNode> &nodes,
    int parent_depth, const std::vector<Var> &shared_loop_vars,
    const std::vector<TileScopeRegion> &regions, int hoisted_attr_prefix_count);

// Emit one shared-shell node from the planner tree. The representative region
// contributes the concrete loop objects for the shared prefix; planner legality
// guarantees every child under this node agrees on that shared prefix shape.
Stmt BuildScopeNode(const SunmmioTileLoopFusionPlannerTreeNode &node,
                    int parent_depth, const std::vector<Var> &shared_loop_vars,
                    const std::vector<TileScopeRegion> &regions,
                    int hoisted_attr_prefix_count) {
  int scope_depth = static_cast<int>(node.shell_axes.size());
  ICHECK_GT(scope_depth, parent_depth);
  int representative_region_index = FindFirstLeafRegionIndex(node);
  ICHECK_GE(representative_region_index, 0);
  const TileScopeRegion &representative = regions[representative_region_index];

  std::vector<AttrStmt> common_attrs = CollectCommonHoistableAttrPrefix(
      node, regions, hoisted_attr_prefix_count);

  std::vector<Var> next_shared_loop_vars = shared_loop_vars;
  for (int depth = parent_depth; depth < scope_depth; ++depth) {
    const For &loop = representative.execution_loops[depth];
    next_shared_loop_vars.push_back(
        Var(loop->loop_var->name_hint, loop->loop_var.dtype()));
  }

  Stmt body = BuildPlannedStmtList(
      node.children, scope_depth, next_shared_loop_vars, regions,
      hoisted_attr_prefix_count + static_cast<int>(common_attrs.size()));
  for (int depth = scope_depth - 1; depth >= parent_depth; --depth) {
    const For &loop = representative.execution_loops[depth];
    std::vector<Var> outer_shared_loop_vars(
        next_shared_loop_vars.begin(), next_shared_loop_vars.begin() + depth);
    Map<Var, PrimExpr> outer_subst =
        BuildSharedLoopVarSubst(representative, outer_shared_loop_vars);
    body = For(next_shared_loop_vars[depth], Substitute(loop->min, outer_subst),
               Substitute(loop->extent, outer_subst), loop->kind, body,
               loop->thread_binding, loop->annotations);
  }
  return ReapplyAttrPrefix(common_attrs, body);
}

Stmt BuildPlannedStmtList(
    const std::vector<SunmmioTileLoopFusionPlannerTreeNode> &nodes,
    int parent_depth, const std::vector<Var> &shared_loop_vars,
    const std::vector<TileScopeRegion> &regions,
    int hoisted_attr_prefix_count) {
  Array<Stmt> stmts;
  stmts.reserve(nodes.size());
  for (const SunmmioTileLoopFusionPlannerTreeNode &node : nodes) {
    if (node.is_scope) {
      stmts.push_back(BuildScopeNode(node, parent_depth, shared_loop_vars,
                                     regions, hoisted_attr_prefix_count));
    } else {
      ICHECK_GE(node.region_index, 0);
      stmts.push_back(BuildRegionLeaf(regions[node.region_index], parent_depth,
                                      shared_loop_vars,
                                      hoisted_attr_prefix_count));
    }
  }
  return SeqStmt::Flatten(stmts);
}

// Rewrite works window-by-window: match one discovered source-order run, then
// replace that exact statement interval with the planner-produced tree.
struct WindowRewriteSpec {
  TileScopeRegionRun source_region_run;
  std::vector<SunmmioTileLoopFusionPlannerTreeNode> tree;
};

class SunmmioTileLoopFusionRewriter : public StmtExprMutator {
public:
  SunmmioTileLoopFusionRewriter(const std::vector<TileScopeRegion> &regions,
                                const std::vector<WindowRewriteSpec> &windows)
      : regions_(regions), windows_(windows) {}

  Stmt Rewrite(const Stmt &stmt) { return VisitStmt(stmt); }

private:
  // Discovery records the original source-order statement run for each window.
  // Rewrite only substitutes when the current SeqStmt still contains that
  // exact region interval unchanged.
  bool MatchWindowAt(const Array<Stmt> &seq, int start,
                     const WindowRewriteSpec &window) const {
    if (start + window.source_region_run.num_regions >
        static_cast<int>(seq.size())) {
      return false;
    }
    for (int region_offset = 0;
         region_offset < window.source_region_run.num_regions;
         ++region_offset) {
      int region_index =
          window.source_region_run.begin_region_index + region_offset;
      if (!seq[start + region_offset].same_as(
              regions_[region_index].root_stmt)) {
        return false;
      }
    }
    return true;
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> rewritten;
    int i = 0;
    while (i < static_cast<int>(op->seq.size())) {
      bool matched = false;
      for (const WindowRewriteSpec &window : windows_) {
        if (!MatchWindowAt(op->seq, i, window)) {
          continue;
        }
        rewritten.push_back(
            BuildPlannedStmtList(window.tree, 0, {}, regions_, 0));
        i += window.source_region_run.num_regions;
        matched = true;
        break;
      }
      if (!matched) {
        rewritten.push_back(StmtExprMutator::VisitStmt(op->seq[i]));
        ++i;
      }
    }
    return SeqStmt::Flatten(rewritten);
  }

  const std::vector<TileScopeRegion> &regions_;
  const std::vector<WindowRewriteSpec> &windows_;
};

/*!
 * \brief Apply the chosen planner trees back onto one PrimFunc.
 *
 * Windows whose plan tree is empty are true no-ops and are skipped. Every
 * non-empty window is rewritten by matching its original source-order run and
 * replacing that interval with the rebuilt fused shell tree.
 */
PrimFunc RewriteSunmmioTileLoopFusion(
    PrimFunc func, const SunmmioTileLoopFusionProgram &program,
    const std::vector<SunmmioTileLoopFusionWindowPlan> &plans) {
  ICHECK_EQ(program.region_runs.size(), plans.size())
      << "Expected one rewrite plan per fusion window, but got "
      << program.region_runs.size() << " windows and " << plans.size()
      << " plans.";

  std::vector<WindowRewriteSpec> rewrite_windows;
  rewrite_windows.reserve(program.region_runs.size());
  for (size_t i = 0; i < program.region_runs.size(); ++i) {
    if (plans[i].tree.empty()) {
      continue;
    }
    rewrite_windows.push_back({program.region_runs[i], plans[i].tree});
  }
  if (rewrite_windows.empty()) {
    return func;
  }

  PrimFuncNode *node = func.CopyOnWrite();
  node->body = SunmmioTileLoopFusionRewriter(program.regions, rewrite_windows)
                   .Rewrite(node->body);
  return func;
}

} // namespace

/*!
 * \brief Module pass entrypoint for Sunmmio tile loop fusion.
 */
tvm::transform::Pass SunmmioTileLoopFusion() {
  auto pass_func = [](IRModule mod, const tvm::transform::PassContext &) {
    for (const auto &kv : mod->functions) {
      if (const auto *prim = kv.second.as<tir::PrimFuncNode>()) {
        tir::PrimFunc func = ffi::GetRef<tir::PrimFunc>(prim);
        // Stage 1: discover planner-visible regions and source-order windows.
        SunmmioTileLoopFusionProgram program =
            BuildSunmmioTileLoopFusionProgram(func);
        // Stage 2: build one window-local planning problem per discovered run.
        std::vector<SunmmioTileLoopFusionWindowProblem> problems =
            BuildSunmmioTileLoopFusionWindowProblems(program);
        // Stage 3: choose a fused shared-shell tree for each window.
        std::vector<SunmmioTileLoopFusionWindowPlan> plans =
            PlanSunmmioTileLoopFusionWindowProblems(problems);
        // Stage 4: splice the chosen trees back into the original TIR.
        func = RewriteSunmmioTileLoopFusion(func, program, plans);
        mod->Add(kv.first, func, true);
      }
    }
    return mod;
  };

  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "tl.SunmmioTileLoopFusion", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def("tl.transform.SunmmioTileLoopFusion",
                                        SunmmioTileLoopFusion);
}

} // namespace tl
} // namespace tvm
