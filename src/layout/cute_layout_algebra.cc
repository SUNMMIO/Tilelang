/*!
 * \file layout/cute_layout_algebra.cc
 * \brief CuTe layout algebra implementation.
 *
 * Ported from NPU-IR's Layout.cpp with PrimExpr support.
 */

#include "cute_layout_algebra.h"

#include <tvm/arith/analyzer.h>
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

bool tryConstInt(const PrimExpr &expr, int64_t *out) {
  if (const auto *imm = expr.as<IntImmNode>()) {
    *out = imm->value;
    return true;
  }
  arith::Analyzer analyzer;
  PrimExpr simplified = analyzer.Simplify(expr);
  if (const auto *imm = simplified.as<IntImmNode>()) {
    *out = imm->value;
    return true;
  }
  return false;
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

  // Sentinel (shape 1, stride 0) — matches the original accumulator seed.
  std::vector<PrimExpr> resultShape{makeInt(1)};
  std::vector<PrimExpr> resultStride{makeInt(0)};

  for (size_t i = 0; i < flatShape.size(); ++i) {
    int64_t s;
    bool shapeConst = tryConstInt(flatShape[i], &s);

    // Drop trivial (shape-1) modes.  A symbolic shape can never be proven to
    // be 1, so it is always kept.
    if (shapeConst && s == 1)
      continue;

    int64_t backShape;
    bool backShapeConst = tryConstInt(resultShape.back(), &backShape);

    // Replace the sentinel / a leading shape-1 mode in place.
    if (backShapeConst && backShape == 1) {
      resultShape.back() = flatShape[i];
      resultStride.back() = flatStride[i];
      continue;
    }

    // Merge into the previous mode only when contiguity is provable (all
    // concrete): back.shape * back.stride == this.stride.  Symbolic modes
    // cannot be merged and fall through to a standalone entry below.
    int64_t backStride, d;
    if (shapeConst && backShapeConst &&
        tryConstInt(resultStride.back(), &backStride) &&
        tryConstInt(flatStride[i], &d) && backShape * backStride == d) {
      resultShape.back() = makeInt(backShape * s);
      continue;
    }

    resultShape.push_back(flatShape[i]);
    resultStride.push_back(flatStride[i]);
  }

  if (resultShape.size() == 1) {
    return CuteAlgebraLayout(resultShape[0], resultStride[0]);
  }
  return CuteAlgebraLayout(resultShape, resultStride);
}

// ---------------------------------------------------------------------------
// complement
// ---------------------------------------------------------------------------

CuteAlgebraLayout complement(const CuteAlgebraLayout &layout, PrimExpr maxIdx) {
  auto flatStride = flattenExprTuple(layout.stride);
  auto flatShape = flattenExprTuple(layout.shape);
  ICHECK_EQ(flatStride.size(), flatShape.size());

  // Build (stride, shape) pairs and sort by stride.  The input layout is
  // always a tiler/block layout, so these are compile-time constants; sorting
  // modes by stride is inherently a concrete operation.
  std::vector<std::pair<int64_t, int64_t>> pairs;
  pairs.reserve(flatShape.size());
  for (size_t i = 0; i < flatShape.size(); ++i) {
    pairs.emplace_back(toConcrete(flatStride[i]), toConcrete(flatShape[i]));
  }
  std::sort(pairs.begin(), pairs.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });

  std::vector<PrimExpr> resultShape;
  std::vector<PrimExpr> resultStride;
  int64_t currentIdx = 1;

  for (const auto &[stride, shape] : pairs) {
    if (stride == 0 || shape == 1)
      continue;
    ICHECK(currentIdx <= shape * stride)
        << "complement: currentIdx=" << currentIdx
        << " > shape*stride=" << shape * stride;
    resultShape.push_back(makeInt(stride / currentIdx));
    resultStride.push_back(makeInt(currentIdx));
    currentIdx = shape * stride;
  }

  // The free-dimension extent is the only possibly-symbolic mode: it carries
  // the (dynamic) total domain extent maxIdx.  ceildiv folds to an IntImm for
  // concrete maxIdx and stays symbolic otherwise.
  resultShape.push_back(tvm::ceildiv(maxIdx, makeInt(currentIdx)));
  resultStride.push_back(makeInt(currentIdx));

  return coalesce(CuteAlgebraLayout(resultShape, resultStride));
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
  PrimExpr restStride = layoutB.stride.value;
  int64_t bStrideConst;
  if (tryConstInt(restStride, &bStrideConst) && bStrideConst == 0) {
    return CuteAlgebraLayout(layoutB.shape.value, makeInt(0));
  }

  PrimExpr restShape = layoutB.shape.value; // may be symbolic

  CuteAlgebraLayout flatA = coalesce(layoutA);
  auto flatAShape = flattenExprTuple(flatA.shape);
  auto flatAStride = flattenExprTuple(flatA.stride);

  std::vector<PrimExpr> resultShape;
  std::vector<PrimExpr> resultStride;

  // Splitting a multi-mode left operand across B's stride is inherently
  // concrete (it makes modular divisibility decisions).  Every layout-
  // construction path feeds a single-mode A here (the loop is skipped), so it
  // only runs for the concrete algebra (e.g. zippedProduct); require concrete
  // operands with a clear message via toConcrete.
  if (flatAShape.size() > 1) {
    int64_t restShapeC = toConcrete(restShape);
    int64_t restStrideC = toConcrete(restStride);
    for (size_t i = 0; i + 1 < flatAShape.size(); ++i) {
      int64_t currShape = toConcrete(flatAShape[i]);
      int64_t currStride = toConcrete(flatAStride[i]);

      if (currShape % restStrideC != 0 && restStrideC % currShape != 0) {
        return CuteAlgebraLayout(int64_t(1), int64_t(0));
      }

      int64_t newShape =
          std::min(std::max(int64_t(1), currShape / restStrideC), restShapeC);
      if (restShapeC % newShape != 0) {
        return CuteAlgebraLayout(int64_t(1), int64_t(0));
      }

      if (newShape != 1) {
        resultShape.push_back(makeInt(newShape));
        resultStride.push_back(makeInt(restStrideC * currStride));
      }

      restShapeC /= newShape;
      restStrideC = ceilDivInt(restStrideC, currShape);
    }
    restShape = makeInt(restShapeC);
    restStride = makeInt(restStrideC);
  }

  // Final (outermost) mode — pure arithmetic that supports a symbolic
  // restShape and a symbolic left-operand stride.
  int64_t restShapeConst;
  bool restIsOne =
      tryConstInt(restShape, &restShapeConst) && restShapeConst == 1;
  if (!restIsOne || resultShape.empty()) {
    resultShape.push_back(restShape);
    resultStride.push_back(restStride * flatAStride.back());
  }

  if (resultShape.size() == 1) {
    return CuteAlgebraLayout(resultShape[0], resultStride[0]);
  }
  return CuteAlgebraLayout(resultShape, resultStride);
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
  // B)).  size()/cosize() are pure products, passed straight to complement as
  // its (possibly symbolic) maxIdx — no concreteness required here.
  PrimExpr aSize = layoutA.size();
  PrimExpr bCosize = layoutB.cosize();

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

  // Non-byMode: composition(A, (B, complement(B, A.size()))).  A.size() is a
  // pure product passed to complement as maxIdx, so a dynamic extent flows
  // through here unchanged.
  PrimExpr aSize = layoutA.size();
  auto comp = complement(layoutB, aSize);
  std::vector<CuteAlgebraLayout> inner{layoutB, comp};
  return composition(layoutA, CuteAlgebraLayout(inner), false);
}

} // namespace cute
} // namespace tl
} // namespace tvm
