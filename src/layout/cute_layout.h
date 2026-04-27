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
 * Preserves dim_levels and fixed inner modes; recomputes shape-dependent
 * outer modes via ceildiv.
 *
 * \param axis_map  Maps source logical dims to destination logical dims.
 *                  NullOpt means identity.
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
