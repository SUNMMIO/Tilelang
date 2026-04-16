/*!
 * \file tileview/tileview_planner_common.h
 * \brief Shared legality and enumeration helpers for TileView planning.
 */

#ifndef TVM_TL_TILEVIEW_TILEVIEW_PLANNER_COMMON_H_
#define TVM_TL_TILEVIEW_TILEVIEW_PLANNER_COMMON_H_

#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/tir/buffer.h>

#include "../layout/layout.h"
#include "../target/sunmmio_utils.h"
#include "tileview.h"

namespace tvm {
namespace tl {

using namespace tir;

/// Coarse layout classes that drive Sunmmio tile legality rules.
enum class LayoutClass {
  /// Buffers with an explicit layout entry use the modeled 32x32 blockwise
  /// rules.
  kBlockwise32x32,
  /// Buffers without a layout entry are treated as plain row-major buffers.
  kRowMajor,
};

/*!
 * \brief Canonical trailing tile pattern shared by the generic and reduction
 * planners.
 *
 * `mapped_dims` records which trailing buffer dimensions participate in
 * execution, while `tile_shape` stores the corresponding static tile extents in
 * the same order. The pattern itself is purely structural; planner-specific
 * code is responsible for checking whether loop bindings or region offsets can
 * realize it.
 */
struct TrailingTilePattern {
  /// Buffer dimensions covered by the execution tile, expressed as normalized
  /// indices.
  std::vector<int> mapped_dims;
  /// Static tile extents for each mapped dimension.
  std::vector<int> tile_shape;
};

/*!
 * \brief Return the integer value of a static `PrimExpr`, or `fallback`
 * otherwise.
 *
 * The planners use this when a legality rule requires a compile-time constant
 * but still wants a caller-controlled default for dynamic expressions.
 */
int64_t GetStaticIntValue(const PrimExpr &expr, int64_t fallback = -1);

/*!
 * \brief Classify a buffer as blockwise or row-major for TileView planning.
 *
 * Today the classification is intentionally simple: presence in `layout_map`
 * means the buffer follows Sunmmio blockwise rules, otherwise the buffer is
 * treated as row-major.
 */
LayoutClass GetLayoutClass(const Buffer &buffer,
                           const Map<Buffer, Layout> &layout_map);

/*!
 * \brief Return the scalar element bit-width used by the buffer.
 *
 * TileView planning models register capacity in elements, so the planners first
 * normalize Sunmmio register width against the buffer element type.
 */
int GetElementBits(const Buffer &buffer);

/*!
 * \brief Return the modeled Sunmmio register capacity in buffer elements.
 *
 * This converts `config.register_bits` into an element count using the buffer
 * dtype and checks that the modeled register width is divisible by the scalar
 * element size.
 */
int GetCapacityElems(const Buffer &buffer,
                     const SunmmioTileProcessorConfig &config);

/*!
 * \brief Check whether `value` is provably divisible by `factor`.
 *
 * Both planners use this helper for buffer-shape divisibility, loop-offset
 * alignment, and region-min / region-extent alignment checks.
 */
bool CanProveDivisible(arith::Analyzer *analyzer, const PrimExpr &value,
                       int factor);

/*!
 * \brief Require that a tile offset is aligned to `factor`.
 *
 * This is the checked variant of `CanProveDivisible`. It is used when
 * validating a manual TileView or a strict planner path that must emit a
 * precise diagnostics message tied to the original `index` and `buffer`.
 */
void RequireDivisible(arith::Analyzer *analyzer, const PrimExpr &value,
                      int factor, const PrimExpr &index, const Buffer &buffer);

/*!
 * \brief Convert an index-map entry into a normalized non-negative buffer
 * dimension.
 *
 * TileView index maps in this code path are expected to use `IntImm` entries
 * and may use negative indexing for trailing dimensions. This helper
 * canonicalizes them so the planners can compare mappings structurally.
 */
int NormalizeMappedDim(const PrimExpr &expr, int ndim);

/*!
 * \brief Check whether a TileView uses the canonical trailing-dimension
 * mapping.
 *
 * The shared planner core only reasons about trailing 1D or trailing 2D
 * execution patterns. This helper rejects manual TileViews that remap execution
 * onto arbitrary interior dimensions.
 */
bool HasTrailingIndexMap(const TileView &tv, int exec_rank);

/*!
 * \brief Return the total number of elements in a static tile shape.
 *
 * This overload is used when a planner already stores tile extents as plain
 * integers.
 */
int TileElements(const std::vector<int> &tile_shape);

/*!
 * \brief Return the total number of elements in a static TileView tile shape.
 *
 * This overload is used when a tile shape is still represented as `PrimExpr`s.
 * It requires every extent to be a positive compile-time constant.
 */
int TileElements(const Array<PrimExpr> &tile_shape);

/*!
 * \brief Convert a plain integer tile shape into the canonical `PrimExpr` form.
 *
 * This is mainly used when lifting a shared `TrailingTilePattern` into a full
 * `TileView`.
 */
Array<PrimExpr> MakeTileShapeExpr(const std::vector<int> &tile_shape);

/*!
 * \brief Build the canonical trailing index map for a tile rank.
 *
 * For example, a trailing rank-2 TileView on an `N`-D buffer maps to `{-2,
 * -1}`. Both planners use this form when constructing inferred or projected
 * TileViews.
 */
Array<PrimExpr> MakeCanonicalIndexMap(int buffer_rank, int tile_rank);

/*!
 * \brief Materialize a canonical trailing TileView from a shared tile pattern.
 *
 * The returned TileView preserves the original buffer/domain shape and uses the
 * standard trailing index map implied by `pattern`.
 */
TileView MakeTrailingTileView(const Array<PrimExpr> &buffer_shape,
                              const TrailingTilePattern &pattern);

/*!
 * \brief Enumerate all inferred trailing tile patterns legal for a buffer.
 *
 * This is the main shared legality enumerator. It applies the modeled Sunmmio
 * layout rules once, producing the trailing 1D/2D patterns that the generic and
 * reduction planners later filter with their own loop-binding or
 * region-specific constraints.
 */
std::vector<TrailingTilePattern> EnumerateInferredTrailingTilePatterns(
    const Buffer &buffer, int exec_rank, const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &config, arith::Analyzer *analyzer);

/*!
 * \brief Validate a manual trailing TileView and normalize it into a shared
 * pattern.
 *
 * This routine checks the structural legality that is common to all planners:
 * rank, trailing index-map shape, static tile extents, buffer-shape
 * divisibility, Sunmmio register-capacity limits, and blockwise-vs-row-major
 * width/height rules. Planner-specific checks such as loop binding
 * compatibility or region alignment remain in the caller.
 */
TrailingTilePattern ValidateManualTrailingTileView(
    const Buffer &buffer, const TileView &manual_tv, int exec_rank,
    const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &config, arith::Analyzer *analyzer,
    const char *usage, bool enforce_blockwise_width_for_rank1);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TILEVIEW_TILEVIEW_PLANNER_COMMON_H_
