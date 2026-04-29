/*!
 * \file layout/cute_layout.h
 * \brief CuTe-style structural layout representation.
 *
 * CuteLayoutNode preserves structural layout metadata (mode_shape,
 * mode_stride, dim_levels) as the source of truth. The affine
 * forward_index is derived and cached in the LayoutNode base class
 * for backward compatibility.
 */

#ifndef TVM_TL_LAYOUT_CUTE_LAYOUT_H_
#define TVM_TL_LAYOUT_CUTE_LAYOUT_H_

#include <optional>

#include "layout.h"

namespace tvm {
namespace tl {

using namespace tir;

class CuteLayout;

/*!
 * \brief Structural memory-layout node following CuTe conventions.
 *
 * Fields:
 *   logical_shape_  – the logical buffer shape (one entry per rank)
 *   mode_shape_     – per-mode shape extents (innermost-first within each dim)
 *   mode_stride_    – per-mode strides aligned with mode_shape
 *   dim_levels_     – number of modes belonging to each logical dimension
 *
 * Invariant:
 *   sum(dim_levels_) == mode_shape_.size() == mode_stride_.size()
 *   dim_levels_.size() == logical_shape_.size()
 *   logical_shape_[d] <= product(mode_shape_ modes belonging to d)
 */
class CuteLayoutNode : public LayoutNode {
public:
  CuteLayoutNode() = default;
  CuteLayoutNode(Array<PrimExpr> logical_shape, Array<PrimExpr> mode_shape,
                 Array<PrimExpr> mode_stride, Array<Integer> dim_levels);

  /*! \brief Return the logical buffer shape. */
  Array<PrimExpr> GetLogicalShape() const { return logical_shape_; }
  /*! \brief Return the flattened mode shapes. */
  Array<PrimExpr> GetModeShape() const { return mode_shape_; }
  /*! \brief Return the flattened mode strides. */
  Array<PrimExpr> GetModeStride() const { return mode_stride_; }
  /*! \brief Return the dim-levels array. */
  Array<Integer> GetDimLevels() const { return dim_levels_; }

  /*! \brief Return the mode shapes for a single logical dimension. */
  Array<PrimExpr> GetModeShapeOfDim(int dim) const;
  /*! \brief Return the mode strides for a single logical dimension. */
  Array<PrimExpr> GetModeStrideOfDim(int dim) const;

  /*!
   * \brief Return the covered physical extent per logical dimension.
   *
   * For each dimension d, this is the product of all mode shapes
   * belonging to d.  May be larger than logical_shape_[d] when the
   * logical extent is not an exact multiple of the block size.
   */
  Array<PrimExpr> GetCoveredShape() const;

  /*!
   * \brief Return the physical allocation size.
   *
   * This is the maximum addressable offset + 1 over the full covered
   * domain.  For padded blocked layouts this may be larger than the
   * product of logical_shape_.
   */
  PrimExpr GetStorageSize() const;

  /*!
   * \brief Structural equality comparison.
   *
   * Returns true iff both layouts have the same dim_levels, mode_shape,
   * and mode_stride (after symbolic simplification).
   */
  bool SameLayout(const CuteLayoutNode *other,
                  arith::Analyzer *analyzer = nullptr) const;

  // --- LayoutNode overrides ------------------------------------------------

  Array<PrimExpr> GetForwardVars() const final;
  Array<PrimExpr> Forward(const Array<PrimExpr> &vars) const final;

  /*!
   * \brief Not supported for Sunmmio in the first implementation.
   * Throws LOG(FATAL) if called.
   */
  Layout Inverse() const final;

  /*!
   * \brief Not supported for Sunmmio in the first implementation.
   * Throws LOG(FATAL) if called.
   */
  Layout Reshape(const Array<PrimExpr> &shape, arith::Analyzer *analyzer,
                 const PrimExpr rescale_num = Integer(1),
                 const PrimExpr rescale_den = Integer(1)) const final;

  std::string DebugOutput() const final;

  bool IsEqual(const LayoutNode *other, bool skip_index = false) const final;

  static void RegisterReflection();

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.CuteLayout", CuteLayoutNode,
                                    LayoutNode);
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;

private:
  /*! \brief Compute the flat mode offset for a given logical dimension. */
  int ModeOffset(int dim) const;

  Array<PrimExpr> logical_shape_;
  Array<PrimExpr> mode_shape_;
  Array<PrimExpr> mode_stride_;
  Array<Integer> dim_levels_;
};

/*!
 * \brief CuteLayout reference class.
 */
class CuteLayout : public Layout {
public:
  TVM_DLL CuteLayout(Array<PrimExpr> logical_shape, Array<PrimExpr> mode_shape,
                     Array<PrimExpr> mode_stride, Array<Integer> dim_levels);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(CuteLayout, Layout,
                                             CuteLayoutNode);
};

// ---------------------------------------------------------------------------
// Contiguous tile step analysis
// ---------------------------------------------------------------------------

/*!
 * \brief One step of the contiguous tile envelope.
 *
 * ComputeContiguousTileSteps walks a CuteLayout's coalesced stride
 * structure and emits a sequence of steps describing the maximal
 * contiguous tile.  Each step says "dimension `dim` varies by `extent`
 * elements at this position in the physical ordering."
 *
 * The product of all step extents equals the total contiguous element
 * count.  The planner reads these steps to enumerate legal tile shapes
 * without layout-kind branching.
 */
struct ContiguousStep {
  int dim;    ///< Logical dimension index.
  int extent; ///< Max contiguous extent in this step.
};

/*!
 * \brief Compute the contiguous tile envelope of a CuteLayout.
 *
 * Algorithm:
 *  1. Extract concrete inner modes per logical dimension (stop at the
 *     first symbolic shape/stride).
 *  2. Coalesce within each dimension (merge modes where
 *     shape * stride == next_stride).
 *  3. Pool all modes as (shape, stride, dim), sort by stride ascending.
 *  4. Walk while stride == running_product, emitting ContiguousStep
 *     entries.  Stop at the first stride gap.
 *
 * Returns an empty vector when the layout is not a CuteLayout
 * (caller should apply row-major fallback).
 */
std::vector<ContiguousStep> ComputeContiguousTileSteps(const Layout &layout);

// ---------------------------------------------------------------------------
// Free-standing layout relation APIs
// ---------------------------------------------------------------------------

/*!
 * \brief Exact structural layout comparison.
 *
 * If both layouts are CuteLayoutNode, compares dim_levels, mode_shape,
 * and mode_stride element-wise.  Otherwise falls back to expression-based
 * comparison via LayoutNode::IsEqual.
 */
bool IsSameLayout(const Layout &lhs, const Layout &rhs,
                  arith::Analyzer *analyzer = nullptr);

/*!
 * \brief Build a layout for dst_shape using src as the structural template.
 *
 * Two-pass algorithm:
 *
 * **Pass 1 — mode structure.**  Blocked (multi-mode) src dimensions are
 * placed on the dst dims given by axis_map: inner modes are preserved
 * verbatim, the outermost mode is recomputed via ceildiv to cover the
 * new logical extent.  All other dst dims receive single-level modes
 * (one mode element equal to the logical extent).
 *
 * **Pass 2 — stride ordering.**  The physical stride order of the src
 * layout (recovered via RecoverPhysicalOrder) is projected onto the new
 * modes, so the layout *kind* (row-major, col-major, ZZ, ZN, …) is
 * faithfully preserved without explicit kind detection.
 *
 * Rank-changing is handled by back-aligning single-level src dims to
 * non-blocked dst dims:
 *   - dst has more non-blocked dims than src → excess dst dims get
 *     outermost (largest-stride) row-major modes (broadcast-like).
 *   - src has more single-level dims than dst → excess src dims are
 *     dropped (reduce-like).
 *
 * Returns NullOpt when src is not a CuteLayout, or when dst_rank is
 * too small to accommodate all blocked src dims.
 *
 * \param axis_map  One entry per blocked src dim (in dim-index order);
 *                  axis_map[i] = which dst dim receives that structure.
 *                  NullOpt defaults to the last N axes of dst.
 */
Optional<Layout>
DeriveLayoutLike(const Layout &src, Array<PrimExpr> dst_shape,
                 Optional<Array<Integer>> axis_map = Optional<Array<Integer>>(),
                 arith::Analyzer *analyzer = nullptr);

/*!
 * \brief Same layout kind, possibly for different logical shapes.
 */
bool IsLayoutMatch(const Layout &lhs, const Layout &rhs,
                   arith::Analyzer *analyzer = nullptr);

// ---------------------------------------------------------------------------
// Sunmmio named layout constructors
// ---------------------------------------------------------------------------

namespace sunmmio {

/*!
 * \brief Check if a layout is "ZZ-like": has blocked dims with row-major
 * inner ordering.  Covers ZZ, ZZZ, and NZZ layouts.
 *
 * Structural invariant: at least 2 blocked dims (nlevels > 1), and among
 * the last two blocked dims, the higher-indexed dim's innermost mode has
 * a smaller stride (row-major ordering at the innermost level).
 *
 * Returns false for ZN (col-major inner), row-major (no blocked dims),
 * or non-CuteLayout.
 */
bool IsZZLike(const Layout &layout);

/*!
 * \brief Block dimensions of a ZZ-like layout.
 *
 * Extracted from the innermost mode shapes of the last two blocked
 * dimensions.  `height` corresponds to the lower-indexed blocked dim
 * and `width` to the higher-indexed one.
 */
struct ZZBlockShape {
  int height;
  int width;
};

/*!
 * \brief Extract the block shape from a ZZ-like CuteLayout.
 *
 * For a ZZ layout built with block_shape = {bh, bw}, this returns
 * {bh, bw}.  Works for ZZ, ZZZ, and NZZ layouts — the innermost
 * mode shape of each blocked dimension is the block extent regardless
 * of the number of tiling levels.
 *
 * Returns std::nullopt when the layout is not ZZ-like or the innermost
 * mode shapes are not compile-time constants.
 */
std::optional<ZZBlockShape> GetZZBlockShape(const Layout &layout);

Layout MakeRowMajor(Array<PrimExpr> shape);

Layout MakeZZ(Array<PrimExpr> shape, Array<Integer> axes,
              Array<PrimExpr> block_shape);

Layout MakeZN(Array<PrimExpr> shape, Array<Integer> axes,
              Array<PrimExpr> block_shape);

Layout MakeZZZ(Array<PrimExpr> shape, Array<Integer> axes,
               Array<PrimExpr> block_shape, Array<PrimExpr> cluster_shape);

Layout MakeNZZ(Array<PrimExpr> shape, Array<Integer> axes,
               Array<PrimExpr> block_shape, Array<PrimExpr> cluster_shape);

} // namespace sunmmio

} // namespace tl
} // namespace tvm

#endif // TVM_TL_LAYOUT_CUTE_LAYOUT_H_
