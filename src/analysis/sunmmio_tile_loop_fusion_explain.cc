// Explain/debug projection layer for Sunmmio tile loop fusion.
//
// This file formats typed discovery, dependence, and planner outputs into the
// narrow FFI schemas consumed by Python tests and debugging tools. It must not
// contain production analysis or scheduling logic.

#include "sunmmio_tile_loop_fusion_explain.h"

#include "sunmmio_tile_loop_fusion_analysis.h"
#include "sunmmio_tile_loop_fusion_planner.h"
#include "sunmmio_tile_loop_fusion_utils.h"

#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

PrimFunc LookupMainPrimFunc(const IRModule &mod) {
  BaseFunc func = mod->Lookup("main");
  const auto *prim = func.as<PrimFuncNode>();
  ICHECK(prim) << "Expected `main` to be a PrimFunc";
  return ffi::GetRef<PrimFunc>(prim);
}

std::vector<int>
CollectGlobalRegionIndices(const std::vector<TileScopeRegion> &regions) {
  std::vector<int> region_indices;
  region_indices.reserve(regions.size());
  for (const TileScopeRegion &region : regions) {
    region_indices.push_back(region.global_region_index);
  }
  return region_indices;
}

Map<String, ffi::Any> BufferRegionToExplainObject(const BufferRegion &region) {
  Map<String, ffi::Any> result;
  Array<String> mins;
  Array<String> extents;
  for (const Range &range : region->region) {
    mins.push_back(PrimExprToString(range->min));
    extents.push_back(PrimExprToString(range->extent));
  }
  result.Set("buffer", region->buffer->name);
  result.Set("mins", mins);
  result.Set("extents", extents);
  return result;
}

Map<String, ffi::Any>
TileScopeRegionRunToExplainObject(const TileScopeRegionRun &run) {
  Map<String, ffi::Any> result;
  result.Set("begin_region_index", Integer(run.begin_region_index));
  result.Set("num_regions", Integer(run.num_regions));
  return result;
}

Map<String, ffi::Any>
TileScopeRegionToExplainObject(const TileScopeRegion &region) {
  Map<String, ffi::Any> result;

  Array<String> execution_loop_vars;
  for (const std::string &axis : region.execution_loop_var_names) {
    execution_loop_vars.push_back(String(axis));
  }

  Array<String> logical_execution_axes;
  for (const std::string &axis : region.logical_execution_axis_keys) {
    logical_execution_axes.push_back(String(axis));
  }

  Array<String> execution_loop_extents;
  for (const PrimExpr &expr : region.execution_loop_extents) {
    execution_loop_extents.push_back(PrimExprToString(expr));
  }

  Array<Map<String, ffi::Any>> use_in;
  for (const BufferRegion &buffer_region : region.use_in) {
    use_in.push_back(BufferRegionToExplainObject(buffer_region));
  }

  Array<Map<String, ffi::Any>> def_out;
  for (const BufferRegion &buffer_region : region.def_out) {
    def_out.push_back(BufferRegionToExplainObject(buffer_region));
  }

  Array<Integer> available_at_execution_depths;
  for (int depth : region.available_at_execution_depths) {
    available_at_execution_depths.push_back(Integer(depth));
  }

  result.Set("kind", String("tile_scope_entry"));
  result.Set("root_name", String(region.root_name));
  result.Set("root_loop_var", String(region.root_name));
  result.Set("execution_loop_vars", execution_loop_vars);
  result.Set("logical_execution_axes", logical_execution_axes);
  result.Set("execution_loop_extents", execution_loop_extents);
  result.Set("use_in", use_in);
  result.Set("def_out", def_out);
  result.Set("available_at_execution_depths", available_at_execution_depths);
  return result;
}

Map<String, ffi::Any>
TileScopeDependenceEdgeToExplainObject(const TileScopeDependenceEdge &edge) {
  Map<String, ffi::Any> result;
  result.Set("src", Integer(edge.src_region_index));
  result.Set("dst", Integer(edge.dst_region_index));
  result.Set("kind", String(DependenceKindToCString(edge.kind)));
  result.Set("src_access_index", Integer(edge.src_access_index));
  result.Set("dst_access_index", Integer(edge.dst_access_index));
  result.Set("rho", Integer(edge.rho));
  result.Set("weight_bytes", Integer(edge.weight_bytes));

  ICHECK(edge.debug_overlap_region.has_value())
      << "Expected debug overlap region for explained dependence edge";
  const BufferRegion &overlap_region = edge.debug_overlap_region.value();
  result.Set("buffer", overlap_region->buffer->name);
  result.Set("debug_overlap_region",
             BufferRegionToExplainObject(overlap_region));
  return result;
}

Map<String, ffi::Any>
TileScopeWindowGraphToExplainObject(const std::vector<int> &region_indices,
                                    const TileScopeWindowGraph &graph) {
  Map<String, ffi::Any> result;

  Array<Integer> graph_region_indices;
  for (int region_index : region_indices) {
    graph_region_indices.push_back(Integer(region_index));
  }

  Array<Map<String, ffi::Any>> edges;
  for (const TileScopeDependenceEdge &edge : graph.edges) {
    edges.push_back(TileScopeDependenceEdgeToExplainObject(edge));
  }

  result.Set("region_indices", graph_region_indices);
  result.Set("edges", edges);
  return result;
}

Map<String, ffi::Any>
PlannerScoreToExplainObject(const SunmmioTileLoopFusionPlannerScore &score) {
  Map<String, ffi::Any> result;
  result.Set("write_cut_cost", Integer(score.write_cut_cost));
  result.Set("shared_read_cost", Integer(score.shared_read_cost));
  result.Set("live_range_penalty", Integer(score.live_range_penalty));
  result.Set("reorder_penalty", Integer(score.reorder_penalty));
  return result;
}

Map<String, ffi::Any>
PlannerActionToExplainObject(const SunmmioTileLoopFusionPlannerAction &action) {
  Map<String, ffi::Any> result;
  result.Set("region_index", Integer(action.region_index));
  result.Set("close_to_depth", Integer(action.close_to_depth));
  result.Set("open_to_depth", Integer(action.open_to_depth));

  Array<Array<String>> opened_shells;
  for (const std::vector<std::string> &shell_axes : action.opened_shells) {
    Array<String> shell;
    for (const std::string &axis : shell_axes) {
      shell.push_back(String(axis));
    }
    opened_shells.push_back(shell);
  }
  result.Set("opened_shells", opened_shells);

  Array<Array<String>> opened_shell_extents;
  for (const Array<PrimExpr> &shell_extents : action.opened_shell_extents) {
    Array<String> extents;
    for (const PrimExpr &extent : shell_extents) {
      extents.push_back(PrimExprToString(extent));
    }
    opened_shell_extents.push_back(extents);
  }
  result.Set("opened_shell_extents", opened_shell_extents);
  return result;
}

Map<String, ffi::Any> PlannerTreeNodeToExplainObject(
    const SunmmioTileLoopFusionPlannerTreeNode &node) {
  Map<String, ffi::Any> result;
  result.Set("is_scope", Bool(node.is_scope));
  result.Set("region_index", Integer(node.region_index));

  Array<String> shell_axes;
  for (const std::string &axis : node.shell_axes) {
    shell_axes.push_back(String(axis));
  }
  result.Set("shell_axes", shell_axes);

  Array<String> shell_extents;
  for (const PrimExpr &extent : node.shell_extents) {
    shell_extents.push_back(PrimExprToString(extent));
  }
  result.Set("shell_extents", shell_extents);

  Array<Map<String, ffi::Any>> children;
  for (const SunmmioTileLoopFusionPlannerTreeNode &child : node.children) {
    children.push_back(PlannerTreeNodeToExplainObject(child));
  }
  result.Set("children", children);
  return result;
}

Map<String, ffi::Any>
PlannerWindowToExplainObject(const SunmmioTileLoopFusionWindowPlan &plan) {
  Map<String, ffi::Any> result;

  Array<Integer> region_indices;
  for (int region_index : plan.region_indices) {
    region_indices.push_back(Integer(region_index));
  }
  result.Set("region_indices", region_indices);

  Array<Integer> order;
  for (int region_index : plan.order) {
    order.push_back(Integer(region_index));
  }
  result.Set("order", order);
  result.Set("score", PlannerScoreToExplainObject(plan.score));

  Array<Map<String, ffi::Any>> actions;
  for (const SunmmioTileLoopFusionPlannerAction &action : plan.actions) {
    actions.push_back(PlannerActionToExplainObject(action));
  }
  result.Set("actions", actions);

  Array<Map<String, ffi::Any>> tree;
  for (const SunmmioTileLoopFusionPlannerTreeNode &node : plan.tree) {
    tree.push_back(PlannerTreeNodeToExplainObject(node));
  }
  result.Set("tree", tree);
  return result;
}

Map<String, ffi::Any>
ExplainDiscoverySummary(const SunmmioTileLoopFusionProgram &program) {
  Map<String, ffi::Any> summary;

  Array<Map<String, ffi::Any>> region_runs;
  Array<Integer> region_run_lengths;
  for (const TileScopeRegionRun &run : program.region_runs) {
    region_runs.push_back(TileScopeRegionRunToExplainObject(run));
    region_run_lengths.push_back(Integer(run.num_regions));
  }

  Array<Map<String, ffi::Any>> regions;
  for (const TileScopeRegion &region : program.regions) {
    regions.push_back(TileScopeRegionToExplainObject(region));
  }

  summary.Set("region_count",
              Integer(static_cast<int>(program.regions.size())));
  summary.Set("region_run_count",
              Integer(static_cast<int>(program.region_runs.size())));
  summary.Set("region_run_lengths", region_run_lengths);
  summary.Set("region_runs", region_runs);
  summary.Set("regions", regions);
  return summary;
}

Map<String, ffi::Any> ExplainDependenceSummary(
    const SunmmioTileLoopFusionProgram &program,
    const std::vector<SunmmioTileLoopFusionWindowProblem> &problems) {
  Map<String, ffi::Any> summary;

  Array<Integer> region_run_lengths;
  for (const TileScopeRegionRun &run : program.region_runs) {
    region_run_lengths.push_back(Integer(run.num_regions));
  }

  Array<Map<String, ffi::Any>> graphs;
  for (const SunmmioTileLoopFusionWindowProblem &problem : problems) {
    graphs.push_back(TileScopeWindowGraphToExplainObject(
        CollectGlobalRegionIndices(problem.regions), problem.graph));
  }

  summary.Set("region_run_count",
              Integer(static_cast<int>(program.region_runs.size())));
  summary.Set("region_run_lengths", region_run_lengths);
  summary.Set("graphs", graphs);
  return summary;
}

Map<String, ffi::Any>
ExplainPlanSummary(const std::vector<SunmmioTileLoopFusionWindowPlan> &plans) {
  Map<String, ffi::Any> summary;

  Array<Map<String, ffi::Any>> explained_plans;
  for (const SunmmioTileLoopFusionWindowPlan &plan : plans) {
    explained_plans.push_back(PlannerWindowToExplainObject(plan));
  }

  summary.Set("plan_count", Integer(static_cast<int>(plans.size())));
  summary.Set("plans", explained_plans);
  return summary;
}

} // namespace

Map<String, ffi::Any>
ExplainSunmmioTileLoopFusionDiscovery(const IRModule &mod) {
  PrimFunc func = LookupMainPrimFunc(mod);
  SunmmioTileLoopFusionProgram program =
      BuildSunmmioTileLoopFusionProgram(func);
  return ExplainDiscoverySummary(program);
}

Map<String, ffi::Any>
ExplainSunmmioTileLoopFusionDependence(const IRModule &mod) {
  PrimFunc func = LookupMainPrimFunc(mod);
  SunmmioTileLoopFusionProgram program =
      BuildSunmmioTileLoopFusionProgram(func);
  std::vector<SunmmioTileLoopFusionWindowProblem> problems =
      BuildSunmmioTileLoopFusionWindowProblems(program);
  return ExplainDependenceSummary(program, problems);
}

Map<String, ffi::Any> ExplainSunmmioTileLoopFusionPlan(const IRModule &mod) {
  PrimFunc func = LookupMainPrimFunc(mod);
  SunmmioTileLoopFusionProgram program =
      BuildSunmmioTileLoopFusionProgram(func);
  std::vector<SunmmioTileLoopFusionWindowProblem> problems =
      BuildSunmmioTileLoopFusionWindowProblems(program);
  std::vector<SunmmioTileLoopFusionWindowPlan> plans =
      PlanSunmmioTileLoopFusionWindowProblems(problems);
  return ExplainPlanSummary(plans);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def(
      "tl.analysis.ExplainSunmmioTileLoopFusionDiscovery",
      ExplainSunmmioTileLoopFusionDiscovery);
  tvm::ffi::reflection::GlobalDef().def(
      "tl.analysis.ExplainSunmmioTileLoopFusionDependence",
      ExplainSunmmioTileLoopFusionDependence);
  tvm::ffi::reflection::GlobalDef().def(
      "tl.analysis.ExplainSunmmioTileLoopFusionPlan",
      ExplainSunmmioTileLoopFusionPlan);
}

} // namespace tl
} // namespace tvm
