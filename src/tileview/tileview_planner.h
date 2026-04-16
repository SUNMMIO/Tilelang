/*!
 * \file tileview/tileview_planner.h
 * \brief TileView planning helpers for T.Tiles scopes.
 */

#ifndef TVM_TL_TILEVIEW_TILEVIEW_PLANNER_H_
#define TVM_TL_TILEVIEW_TILEVIEW_PLANNER_H_

#include <unordered_map>
#include <vector>

#include <tvm/tir/buffer.h>
#include <tvm/tir/stmt.h>

#include "../layout/layout.h"
#include "../target/sunmmio_utils.h"
#include "tileview.h"

namespace tvm {
namespace tl {

using namespace tir;

using TileViewMap =
    std::unordered_map<Var, TileView, ObjectPtrHash, ObjectPtrEqual>;

struct BufferAccessRecord {
  Buffer buffer;
  Array<PrimExpr> indices;
  bool is_store{false};
};

struct TileViewPlan {
  TileView execution_tileview;
  std::vector<int> execution_domain_axes;
};

std::vector<BufferAccessRecord> CollectBufferAccesses(const Stmt &stmt);

/*!
 * \brief Solve the common execution TileView for an isolated T.Tiles scope.
 *
 * The planner derives feasible TileView candidates from each buffer access,
 * intersects those candidates across the scope, and returns the selected
 * execution TileView for legalization and lowering.
 */
TileViewPlan PlanTileViewsForTilesScope(
    const Array<PrimExpr> &domain,
    const std::vector<const ForNode *> &scope_loops,
    const std::vector<BufferAccessRecord> &accesses,
    const TileViewMap &manual_tileviews, const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &tile_processor_config);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TILEVIEW_TILEVIEW_PLANNER_H_
