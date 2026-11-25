// Copyright (c) Tile-AI Corporation.
// Finite-width half-open ranges shared by local unit and sender
// synchronization.
#ifndef TVM_TL_TARGET_SUNMMIO_ACCESS_RANGE_H_
#define TVM_TL_TARGET_SUNMMIO_ACCESS_RANGE_H_

#include "../support/finite_width_integer.h"
#include "../tileview/tileview_planner_common.h"
#include "../transform/common/attr.h"
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tl {

// Both passes see the same interior axes and integer loop domains when
// translating a logical BufferStore into its physical RSRAM writeback.
class SunmmioAccessContext {
public:
  arith::Analyzer analyzer;
  FiniteWidthInteger::Definitions definitions;
  void Enter(const tir::ForNode *op) {
    FiniteWidthInteger facts(&analyzer, 1, &definitions);
    auto lo = facts.Analyze(op->min), count = facts.Analyze(op->extent),
         type = facts.Analyze(op->loop_var);
    auto hi = lo.range.hi + count.range.hi - 1;
    if (lo.valid && count.valid && count.range.hi > 0 &&
        lo.range.lo >= type.range.lo && hi <= type.range.hi &&
        lo.range.lo > INT64_MIN && hi < INT64_MAX)
      analyzer.const_int_bound.Update(
          op->loop_var, arith::ConstIntBound(static_cast<int64_t>(lo.range.lo),
                                             static_cast<int64_t>(hi)));
    bool scope = op->annotations.count(attr::kTileDomain);
    tile_depth_ += scope;
    bool interior =
        tile_depth_ && op->annotations.count(attr::tile_interior_axis);
    if (interior)
      interiors_[op->loop_var.get()] = GetStaticIntValue(op->extent);
  }
  void Leave(const tir::ForNode *op) {
    interiors_.erase(op->loop_var.get());
    tile_depth_ -= op->annotations.count(attr::kTileDomain);
  }
  std::vector<int64_t> LogicalShape(const Array<PrimExpr> &indices) const {
    std::vector<int64_t> logical;
    if (tile_depth_) {
      for (const auto &index : indices) {
        // Only narrow a store whose interior coordinate is a unit-stride
        // term. Ambiguous expressions conservatively cover the whole buffer.
        const VarNode *axis = nullptr;
        bool ambiguous = false;
        PostOrderVisit(index, [&](const ObjectRef &n) {
          if (auto *v = n.as<VarNode>(); v && interiors_.count(v)) {
            ambiguous |= axis != nullptr;
            axis = v;
          }
        });
        int64_t extent = 0;
        if (axis) {
          const auto *add = index.as<AddNode>();
          bool direct = index.get() == axis ||
                        (add && (add->a.get() == axis || add->b.get() == axis));
          if (add && direct) {
            const PrimExpr &offset = add->a.get() == axis ? add->b : add->a;
            PostOrderVisit(offset, [&](const ObjectRef &n) {
              ambiguous |= n.get() == axis;
            });
          }
          extent = (direct && !ambiguous) ? interiors_.at(axis) : -1;
        }
        logical.push_back(extent);
      }
      if (std::none_of(logical.begin(), logical.end(),
                       [](int64_t n) { return n != 0; }))
        logical.assign(indices.size(),
                       -1); // Unknown tile-to-buffer mapping.
    }
    return logical;
  }

private:
  int tile_depth_{0};
  std::unordered_map<const tir::VarNode *, int64_t> interiors_;
};

struct SunmmioAccessRange {
  Range envelope; // Across the bound domains.
  Range current;  // At this evaluation, or the envelope when unknown.
};

inline SunmmioAccessRange AnalyzeSunmmioAccessRange(FiniteWidthInteger *facts,
                                                    const PrimExpr &min,
                                                    const PrimExpr &length,
                                                    const PrimExpr &covered) {
  auto start = facts->Analyze(min), extent = facts->Analyze(length);
  auto end = start.range.hi + extent.range.hi;
  auto size = FiniteWidthInteger::Literal(covered);
  auto i64 = [](FiniteWidthInteger::Wide n) {
    return IntImm(DataType::Int(64), static_cast<int64_t>(n));
  };
  Range bounds = Range::FromMinExtent(0, covered);
  if (start.valid && extent.valid && size && start.range.lo >= 0 &&
      extent.range.lo >= 0 && end <= *size && end < INT64_MAX)
    bounds =
        Range::FromMinExtent(i64(start.range.lo), i64(end - start.range.lo));
  if (start.valid && extent.valid && start.range.lo > INT64_MIN / 4 &&
      end < INT64_MAX / 4 && extent.range.lo > 0 &&
      extent.range.lo == extent.range.hi)
    return {bounds, Range::FromMinExtent(cast(DataType::Int(64), start.expr),
                                         i64(extent.range.lo))};
  return {bounds, bounds};
}
} // namespace tl
} // namespace tvm
#endif
