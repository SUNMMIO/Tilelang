/*!
 * \file layout/cute_layout_algebra.cc
 * \brief CuTe layout algebra implementation.
 *
 * Ported from NPU-IR's Layout.cpp with PrimExpr support.
 */

#include "cute_layout_algebra.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/op.h>

#include <algorithm>
#include <numeric>

namespace tvm {
namespace tl {
namespace cute {

using namespace tir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

int64_t toConcrete(const PrimExpr &expr) {
  auto *imm = expr.as<IntImmNode>();
  ICHECK(imm) << "CuTe algebra requires a concrete integer, got: " << expr;
  return imm->value;
}

static PrimExpr makeInt(int64_t v) { return make_const(DataType::Int(32), v); }

static int64_t ceilDivInt(int64_t a, int64_t b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// ExprTuple utilities
// ---------------------------------------------------------------------------

std::vector<PrimExpr> flattenExprTuple(const ExprTuple &value) {
  std::vector<PrimExpr> result;
  std::function<void(const ExprTuple &)> flatten = [&](const ExprTuple &t) {
    if (!t.isTuple) {
      result.push_back(t.value);
      return;
    }
    for (const auto &elem : t.tuple)
      flatten(elem);
  };
  flatten(value);
  return result;
}

PrimExpr product(const ExprTuple &value) {
  if (!value.isTuple)
    return value.value;
  PrimExpr result = makeInt(1);
  for (const auto &elem : value.tuple) {
    result = result * product(elem);
  }
  return result;
}

ExprTuple prefixProduct(const ExprTuple &value, PrimExpr init) {
  if (!value.isTuple)
    return ExprTuple(std::move(init));
  std::vector<ExprTuple> result;
  result.reserve(value.tuple.size());
  PrimExpr running = std::move(init);
  for (const auto &elem : value.tuple) {
    result.push_back(prefixProduct(elem, running));
    running = running * product(elem);
  }
  return ExprTuple(std::move(result));
}

/*! \brief Compute crd2idx: coordinate → linear index. */
static PrimExpr crd2idx(const ExprTuple &crd, const ExprTuple &shape,
                        const ExprTuple &stride) {
  if (crd.isTuple) {
    ICHECK(shape.isTuple && stride.isTuple);
    ICHECK_EQ(crd.tuple.size(), shape.tuple.size());
    ICHECK_EQ(crd.tuple.size(), stride.tuple.size());
    PrimExpr result = makeInt(0);
    for (size_t i = 0; i < crd.tuple.size(); ++i) {
      result = result + crd2idx(crd.tuple[i], shape.tuple[i], stride.tuple[i]);
    }
    return result;
  }

  PrimExpr crdValue = crd.value;
  if (shape.isTuple) {
    ICHECK(stride.isTuple && shape.tuple.size() == stride.tuple.size());
    PrimExpr result = makeInt(0);
    for (size_t i = 0; i + 1 < shape.tuple.size(); ++i) {
      PrimExpr shapeProduct = product(shape.tuple[i]);
      result = result + crd2idx(ExprTuple(FloorMod(crdValue, shapeProduct)),
                                shape.tuple[i], stride.tuple[i]);
      crdValue = FloorDiv(crdValue, shapeProduct);
    }
    return result + crd2idx(ExprTuple(crdValue), shape.tuple.back(),
                            stride.tuple.back());
  }

  ICHECK(!stride.isTuple);
  return crdValue * stride.value;
}

// ---------------------------------------------------------------------------
// zip2By / tileUnzip — structural regrouping
// ---------------------------------------------------------------------------

static ExprTuple zip2By(const ExprTuple &value, const ExprTuple &guide) {
  if (guide.isTuple) {
    ICHECK(value.isTuple && value.tuple.size() >= guide.tuple.size());
    std::vector<ExprTuple> split;
    split.reserve(guide.tuple.size());
    for (size_t i = 0; i < guide.tuple.size(); ++i) {
      split.push_back(zip2By(value.tuple[i], guide.tuple[i]));
    }

    std::vector<ExprTuple> firstModes;
    firstModes.reserve(guide.tuple.size());
    for (const auto &s : split) {
      ICHECK(s.isTuple && s.tuple.size() == 2);
      firstModes.push_back(s.tuple[0]);
    }

    std::vector<ExprTuple> secondModes;
    secondModes.reserve(value.tuple.size());
    for (const auto &s : split) {
      secondModes.push_back(s.tuple[1]);
    }
    for (size_t i = guide.tuple.size(); i < value.tuple.size(); ++i) {
      secondModes.push_back(value.tuple[i]);
    }

    return ExprTuple(std::vector<ExprTuple>{ExprTuple(std::move(firstModes)),
                                            ExprTuple(std::move(secondModes))});
  }

  ICHECK(value.isTuple && value.tuple.size() == 2);
  return value;
}

static CuteAlgebraLayout tileUnzip(const CuteAlgebraLayout &layout,
                                   const CuteAlgebraLayout &tiler) {
  return CuteAlgebraLayout(zip2By(layout.shape, tiler.shape),
                           zip2By(layout.stride, tiler.stride));
}

// ---------------------------------------------------------------------------
// CuteAlgebraLayout constructors and methods
// ---------------------------------------------------------------------------

CuteAlgebraLayout::CuteAlgebraLayout(ExprTuple shape_in)
    : shape(std::move(shape_in)) {
  stride = prefixProduct(this->shape, makeInt(1));
}

CuteAlgebraLayout::CuteAlgebraLayout(
    const std::vector<CuteAlgebraLayout> &layouts) {
  std::vector<ExprTuple> shapes;
  std::vector<ExprTuple> strides;
  shapes.reserve(layouts.size());
  strides.reserve(layouts.size());
  for (const auto &l : layouts) {
    shapes.push_back(l.shape);
    strides.push_back(l.stride);
  }
  shape = ExprTuple(std::move(shapes));
  stride = ExprTuple(std::move(strides));
}

CuteAlgebraLayout CuteAlgebraLayout::mode(size_t i) const {
  if (shape.isTuple) {
    ICHECK(stride.isTuple && i < shape.tuple.size() && i < stride.tuple.size());
    return CuteAlgebraLayout(shape.tuple[i], stride.tuple[i]);
  }
  ICHECK_EQ(i, 0);
  return *this;
}

PrimExpr CuteAlgebraLayout::size() const { return product(shape); }

PrimExpr CuteAlgebraLayout::cosize() const {
  return crd2idx(ExprTuple(size() - makeInt(1)), shape, stride) + makeInt(1);
}

// ---------------------------------------------------------------------------
// coalesce
// ---------------------------------------------------------------------------

CuteAlgebraLayout coalesce(const CuteAlgebraLayout &layout) {
  auto flatShape = flattenExprTuple(layout.shape);
  auto flatStride = flattenExprTuple(layout.stride);
  ICHECK_EQ(flatShape.size(), flatStride.size());

  std::vector<int64_t> resultShape{1};
  std::vector<int64_t> resultStride{0};

  for (size_t i = 0; i < flatShape.size(); ++i) {
    auto *shapeImm = flatShape[i].as<IntImmNode>();
    auto *strideImm = flatStride[i].as<IntImmNode>();

    // If either is symbolic, we cannot merge — keep as separate mode.
    if (!shapeImm || !strideImm) {
      // Flush the current concrete accumulator as-is, then add the
      // symbolic mode as a standalone entry.  We cannot coalesce across
      // symbolic boundaries.
      //
      // For our use case (block/tiler layouts), all values are concrete,
      // so this path should not be hit.  We add it for safety.
      ICHECK(shapeImm && strideImm)
          << "CuTe coalesce expects concrete modes, got shape=" << flatShape[i]
          << " stride=" << flatStride[i];
      continue;
    }

    int64_t s = shapeImm->value;
    int64_t d = strideImm->value;

    if (s == 1)
      continue;

    if (resultShape.back() == 1) {
      resultShape.back() = s;
      resultStride.back() = d;
      continue;
    }

    if (resultShape.back() * resultStride.back() == d) {
      resultShape.back() *= s;
      continue;
    }

    resultShape.push_back(s);
    resultStride.push_back(d);
  }

  std::vector<PrimExpr> rs, rd;
  rs.reserve(resultShape.size());
  rd.reserve(resultStride.size());
  for (size_t i = 0; i < resultShape.size(); ++i) {
    rs.push_back(makeInt(resultShape[i]));
    rd.push_back(makeInt(resultStride[i]));
  }

  if (rs.size() == 1) {
    return CuteAlgebraLayout(rs[0], rd[0]);
  }
  return CuteAlgebraLayout(rs, rd);
}

// ---------------------------------------------------------------------------
// complement
// ---------------------------------------------------------------------------

CuteAlgebraLayout complement(const CuteAlgebraLayout &layout, int64_t maxIdx) {
  auto flatStride = flattenExprTuple(layout.stride);
  auto flatShape = flattenExprTuple(layout.shape);
  ICHECK_EQ(flatStride.size(), flatShape.size());

  // Build (stride, shape) pairs and sort by concrete stride.
  std::vector<std::pair<int64_t, int64_t>> pairs;
  pairs.reserve(flatShape.size());
  for (size_t i = 0; i < flatShape.size(); ++i) {
    pairs.emplace_back(toConcrete(flatStride[i]), toConcrete(flatShape[i]));
  }
  std::sort(pairs.begin(), pairs.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });

  std::vector<int64_t> resultShape;
  std::vector<int64_t> resultStride;
  int64_t currentIdx = 1;

  for (const auto &[stride, shape] : pairs) {
    if (stride == 0 || shape == 1)
      continue;
    ICHECK(currentIdx <= shape * stride)
        << "complement: currentIdx=" << currentIdx
        << " > shape*stride=" << shape * stride;
    resultShape.push_back(stride / currentIdx);
    resultStride.push_back(currentIdx);
    currentIdx = shape * stride;
  }

  resultShape.push_back(ceilDivInt(maxIdx, currentIdx));
  resultStride.push_back(currentIdx);

  // Convert to PrimExpr and coalesce.
  std::vector<PrimExpr> rs, rd;
  rs.reserve(resultShape.size());
  rd.reserve(resultStride.size());
  for (size_t i = 0; i < resultShape.size(); ++i) {
    rs.push_back(makeInt(resultShape[i]));
    rd.push_back(makeInt(resultStride[i]));
  }

  CuteAlgebraLayout result(rs, rd);
  return coalesce(result);
}

// ---------------------------------------------------------------------------
// composition
// ---------------------------------------------------------------------------

CuteAlgebraLayout composition(const CuteAlgebraLayout &layoutA,
                              const CuteAlgebraLayout &layoutB, bool byMode) {
  if (byMode) {
    ICHECK(layoutA.rank() >= layoutB.rank());
    std::vector<CuteAlgebraLayout> modes;
    modes.reserve(layoutA.rank());
    for (size_t i = 0; i < layoutB.rank(); ++i) {
      modes.push_back(composition(layoutA.mode(i), layoutB.mode(i), false));
    }
    for (size_t i = layoutB.rank(); i < layoutA.rank(); ++i) {
      modes.push_back(layoutA.mode(i));
    }
    return CuteAlgebraLayout(modes);
  }

  if (layoutB.shape.isTuple) {
    std::vector<CuteAlgebraLayout> modes;
    modes.reserve(layoutB.rank());
    for (size_t i = 0; i < layoutB.rank(); ++i) {
      modes.push_back(composition(layoutA, layoutB.mode(i), false));
    }
    return CuteAlgebraLayout(modes);
  }

  ICHECK(!layoutB.stride.isTuple);
  int64_t bStride = toConcrete(layoutB.stride.value);
  if (bStride == 0) {
    return CuteAlgebraLayout(layoutB.shape.value, makeInt(0));
  }

  int64_t restShape = toConcrete(layoutB.shape.value);
  int64_t restStride = bStride;

  CuteAlgebraLayout flatA = coalesce(layoutA);
  auto flatAShape = flattenExprTuple(flatA.shape);
  auto flatAStride = flattenExprTuple(flatA.stride);

  std::vector<int64_t> resultShape;
  std::vector<int64_t> resultStride;

  for (size_t i = 0; i + 1 < flatAShape.size(); ++i) {
    int64_t currShape = toConcrete(flatAShape[i]);
    int64_t currStride = toConcrete(flatAStride[i]);

    if (currShape % restStride != 0 && restStride % currShape != 0) {
      return CuteAlgebraLayout(int64_t(1), int64_t(0));
    }

    int64_t newShape =
        std::min(std::max(int64_t(1), currShape / restStride), restShape);
    if (restShape % newShape != 0) {
      return CuteAlgebraLayout(int64_t(1), int64_t(0));
    }

    if (newShape != 1) {
      resultShape.push_back(newShape);
      resultStride.push_back(restStride * currStride);
    }

    restShape /= newShape;
    restStride = ceilDivInt(restStride, currShape);
  }

  if (restShape != 1 || resultShape.empty()) {
    resultShape.push_back(restShape);
    resultStride.push_back(restStride * toConcrete(flatAStride.back()));
  }

  std::vector<PrimExpr> rs, rd;
  rs.reserve(resultShape.size());
  rd.reserve(resultStride.size());
  for (size_t i = 0; i < resultShape.size(); ++i) {
    rs.push_back(makeInt(resultShape[i]));
    rd.push_back(makeInt(resultStride[i]));
  }

  if (rs.size() == 1) {
    return CuteAlgebraLayout(rs[0], rd[0]);
  }
  return CuteAlgebraLayout(rs, rd);
}

// ---------------------------------------------------------------------------
// logicalProduct
// ---------------------------------------------------------------------------

CuteAlgebraLayout logicalProduct(const CuteAlgebraLayout &layoutA,
                                 const CuteAlgebraLayout &layoutB,
                                 bool byMode) {
  if (byMode) {
    ICHECK(layoutA.rank() >= layoutB.rank());
    std::vector<CuteAlgebraLayout> modes;
    modes.reserve(layoutA.rank());
    for (size_t i = 0; i < layoutB.rank(); ++i) {
      modes.push_back(logicalProduct(layoutA.mode(i), layoutB.mode(i), false));
    }
    for (size_t i = layoutB.rank(); i < layoutA.rank(); ++i) {
      modes.push_back(layoutA.mode(i));
    }
    return CuteAlgebraLayout(modes);
  }

  // logicalProduct(A, B) = (A, composition(complement(A, A.size * B.cosize),
  // B))
  int64_t aSize = toConcrete(layoutA.size());
  int64_t bCosize = toConcrete(layoutB.cosize());

  auto comp = complement(layoutA, aSize * bCosize);
  auto composed = composition(comp, layoutB, false);

  std::vector<CuteAlgebraLayout> modes{layoutA, composed};
  return CuteAlgebraLayout(modes);
}

// ---------------------------------------------------------------------------
// zippedProduct
// ---------------------------------------------------------------------------

CuteAlgebraLayout zippedProduct(const CuteAlgebraLayout &layoutA,
                                const CuteAlgebraLayout &layoutB) {
  return tileUnzip(logicalProduct(layoutA, layoutB, true), layoutB);
}

// ---------------------------------------------------------------------------
// logicalDivide
// ---------------------------------------------------------------------------

CuteAlgebraLayout logicalDivide(const CuteAlgebraLayout &layoutA,
                                const CuteAlgebraLayout &layoutB, bool byMode) {
  if (byMode) {
    ICHECK(layoutA.rank() >= layoutB.rank());
    std::vector<CuteAlgebraLayout> modes;
    modes.reserve(layoutA.rank());
    for (size_t i = 0; i < layoutB.rank(); ++i) {
      modes.push_back(logicalDivide(layoutA.mode(i), layoutB.mode(i), false));
    }
    for (size_t i = layoutB.rank(); i < layoutA.rank(); ++i) {
      modes.push_back(layoutA.mode(i));
    }
    return CuteAlgebraLayout(modes);
  }

  // Non-byMode: composition(A, (B, complement(B, A.size())))
  int64_t aSize = toConcrete(layoutA.size());
  auto comp = complement(layoutB, aSize);
  std::vector<CuteAlgebraLayout> inner{layoutB, comp};
  return composition(layoutA, CuteAlgebraLayout(inner), false);
}

} // namespace cute
} // namespace tl
} // namespace tvm
