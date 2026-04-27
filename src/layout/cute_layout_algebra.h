/*!
 * \file layout/cute_layout_algebra.h
 * \brief CuTe layout algebra for composing (shape, stride) layouts.
 *
 * This implements the CuTe layout algebra (complement, composition,
 * logicalProduct, zippedProduct) using TVM PrimExpr.  It is used
 * internally by the Sunmmio layout constructors to compute mode_shape
 * and mode_stride — it does NOT replace or interact with TVM's
 * Layout/LayoutNode system.
 *
 * Ported from NPU-IR's Layout.cpp and inspired by NVIDIA CUTLASS CuTe.
 */

#ifndef TVM_TL_LAYOUT_CUTE_LAYOUT_ALGEBRA_H_
#define TVM_TL_LAYOUT_CUTE_LAYOUT_ALGEBRA_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <cassert>
#include <vector>

namespace tvm {
namespace tl {
namespace cute {

using namespace tir;

// ---------------------------------------------------------------------------
// ExprTuple — recursive PrimExpr container (analogous to CuTe's IntTuple)
// ---------------------------------------------------------------------------

struct ExprTuple {
  bool isTuple = false;
  PrimExpr value;
  std::vector<ExprTuple> tuple;

  ExprTuple() : isTuple(false), value(make_const(DataType::Int(32), 1)) {}

  /*! \brief Construct a scalar ExprTuple from a PrimExpr. */
  explicit ExprTuple(PrimExpr v) : isTuple(false), value(std::move(v)) {}

  /*! \brief Construct a scalar ExprTuple from an int64_t. */
  explicit ExprTuple(int64_t v)
      : isTuple(false), value(make_const(DataType::Int(32), v)) {}

  /*! \brief Construct a tuple ExprTuple from nested elements. */
  explicit ExprTuple(std::vector<ExprTuple> elems)
      : isTuple(true), value(make_const(DataType::Int(32), 1)),
        tuple(std::move(elems)) {}

  /*! \brief Build a flat ExprTuple from leaf PrimExpr values. */
  static ExprTuple fromLeafValues(const std::vector<PrimExpr> &values) {
    if (values.size() == 1)
      return ExprTuple(values[0]);
    std::vector<ExprTuple> elems;
    elems.reserve(values.size());
    for (const auto &v : values)
      elems.emplace_back(v);
    return ExprTuple(std::move(elems));
  }

  bool isScalar() const { return !isTuple; }

  size_t size() const { return isTuple ? tuple.size() : 1; }
};

// ---------------------------------------------------------------------------
// CuteAlgebraLayout — lightweight (shape, stride) pair for algebraic ops
// ---------------------------------------------------------------------------

/*!
 * \brief CuTe (shape, stride) layout for algebraic composition.
 *
 * This is NOT tl::Layout (LayoutNode).  It is a lightweight computation
 * structure used only during layout construction via the CuTe algebra.
 */
struct CuteAlgebraLayout {
  ExprTuple shape;
  ExprTuple stride;

  CuteAlgebraLayout() = default;

  /*! \brief Construct from scalar shape and stride. */
  CuteAlgebraLayout(PrimExpr shape, PrimExpr stride)
      : shape(ExprTuple(std::move(shape))),
        stride(ExprTuple(std::move(stride))) {}

  /*! \brief Construct from int64 shape and stride. */
  CuteAlgebraLayout(int64_t shape, int64_t stride)
      : shape(ExprTuple(shape)), stride(ExprTuple(stride)) {}

  /*! \brief Construct from flat shape/stride vectors. */
  CuteAlgebraLayout(const std::vector<PrimExpr> &shapes,
                    const std::vector<PrimExpr> &strides)
      : shape(ExprTuple::fromLeafValues(shapes)),
        stride(ExprTuple::fromLeafValues(strides)) {}

  /*! \brief Construct from nested ExprTuples. */
  CuteAlgebraLayout(ExprTuple shape, ExprTuple stride)
      : shape(std::move(shape)), stride(std::move(stride)) {}

  /*! \brief Construct from shape only, deriving contiguous strides. */
  explicit CuteAlgebraLayout(ExprTuple shape);

  /*! \brief Construct by packing multiple layouts as modes. */
  explicit CuteAlgebraLayout(const std::vector<CuteAlgebraLayout> &layouts);

  /*! \brief Number of top-level modes. */
  size_t rank() const { return shape.isTuple ? shape.tuple.size() : 1; }

  /*! \brief Extract a single mode as a CuteAlgebraLayout. */
  CuteAlgebraLayout mode(size_t i) const;

  /*! \brief Total number of elements (product of all shapes). */
  PrimExpr size() const;

  /*! \brief Maximum linear index + 1. */
  PrimExpr cosize() const;
};

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/*! \brief Flatten an ExprTuple into a vector of leaf PrimExprs. */
std::vector<PrimExpr> flattenExprTuple(const ExprTuple &value);

/*! \brief Product of all elements in an ExprTuple. */
PrimExpr product(const ExprTuple &value);

/*! \brief Running prefix product of an ExprTuple. */
ExprTuple prefixProduct(const ExprTuple &value, PrimExpr init);

/*! \brief Extract a concrete int64_t from a PrimExpr, with ICHECK. */
int64_t toConcrete(const PrimExpr &expr);

// ---------------------------------------------------------------------------
// Core CuTe algebra operations
// ---------------------------------------------------------------------------

/*! \brief Merge compatible consecutive modes. */
CuteAlgebraLayout coalesce(const CuteAlgebraLayout &layout);

/*! \brief Compute the complement (free index space) of a layout. */
CuteAlgebraLayout complement(const CuteAlgebraLayout &layout, int64_t maxIdx);

/*! \brief Compose two layouts. */
CuteAlgebraLayout composition(const CuteAlgebraLayout &layoutA,
                              const CuteAlgebraLayout &layoutB,
                              bool byMode = false);

/*! \brief Logical product: (A, composition(complement(A, ...), B)). */
CuteAlgebraLayout logicalProduct(const CuteAlgebraLayout &layoutA,
                                 const CuteAlgebraLayout &layoutB,
                                 bool byMode = false);

/*! \brief Zipped product: tileUnzip(logicalProduct(A, B), B). */
CuteAlgebraLayout zippedProduct(const CuteAlgebraLayout &layoutA,
                                const CuteAlgebraLayout &layoutB);

/*! \brief Logical divide: composition(A, (B, complement(B, A.size()))). */
CuteAlgebraLayout logicalDivide(const CuteAlgebraLayout &layoutA,
                                const CuteAlgebraLayout &layoutB,
                                bool byMode = false);

} // namespace cute
} // namespace tl
} // namespace tvm

#endif // TVM_TL_LAYOUT_CUTE_LAYOUT_ALGEBRA_H_
