#include "sunmmio_tile_loop_fusion_planner_internal.h"

#include <memory>

namespace tvm {
namespace tl {
namespace detail {

namespace {

struct MutablePlannerTreeNode {
  bool is_scope{false};
  int region_index{-1};
  std::vector<std::string> shell_axes;
  Array<PrimExpr> shell_extents;
  std::vector<std::shared_ptr<MutablePlannerTreeNode>> children;
};

SunmmioTileLoopFusionPlannerTreeNode
FreezeTree(const std::shared_ptr<MutablePlannerTreeNode> &node) {
  SunmmioTileLoopFusionPlannerTreeNode frozen;
  frozen.is_scope = node->is_scope;
  frozen.region_index = node->region_index;
  frozen.shell_axes = node->shell_axes;
  frozen.shell_extents = node->shell_extents;
  for (const std::shared_ptr<MutablePlannerTreeNode> &child : node->children) {
    frozen.children.push_back(FreezeTree(child));
  }
  return frozen;
}

} // namespace

std::vector<SunmmioTileLoopFusionPlannerTreeNode> BuildPlanTree(
    const std::vector<SunmmioTileLoopFusionPlannerActionSummary> &actions) {
  auto root = std::make_shared<MutablePlannerTreeNode>();
  root->is_scope = true;

  std::vector<std::shared_ptr<MutablePlannerTreeNode>> open_path;
  open_path.push_back(root);
  for (const SunmmioTileLoopFusionPlannerActionSummary &action : actions) {
    while (static_cast<int>(open_path.size()) - 1 > action.close_to_depth) {
      open_path.pop_back();
    }
    for (size_t shell_index = 0; shell_index < action.opened_shells.size();
         ++shell_index) {
      auto scope_node = std::make_shared<MutablePlannerTreeNode>();
      scope_node->is_scope = true;
      scope_node->shell_axes = action.opened_shells[shell_index];
      if (shell_index < action.opened_shell_extents.size()) {
        scope_node->shell_extents = action.opened_shell_extents[shell_index];
      }
      open_path.back()->children.push_back(scope_node);
      open_path.push_back(scope_node);
    }

    auto region_node = std::make_shared<MutablePlannerTreeNode>();
    region_node->is_scope = false;
    region_node->region_index = action.region_index;
    open_path.back()->children.push_back(region_node);
  }

  std::vector<SunmmioTileLoopFusionPlannerTreeNode> tree;
  for (const std::shared_ptr<MutablePlannerTreeNode> &child : root->children) {
    tree.push_back(FreezeTree(child));
  }
  return tree;
}

} // namespace detail
} // namespace tl
} // namespace tvm
