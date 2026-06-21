/*!
 * \file layout/cute_layout.cc
 * \brief CuTe-style structural layout implementation.
 */

#include "cute_layout.h"
#include "cute_layout_algebra.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <map>
#include <numeric>
#include <sstream>

namespace tvm {
namespace tl {

using namespace tir;

// ---------------------------------------------------------------------------
// CuteLayoutNode helpers
// ---------------------------------------------------------------------------

int CuteLayoutNode::ModeOffset(int dim) const {
  int offset = 0;
  for (int d = 0; d < dim; ++d) {
    offset += dim_levels_[d].IntValue();
  }
  return offset;
}

// ---------------------------------------------------------------------------
// CuteLayoutNode constructor
// ---------------------------------------------------------------------------

CuteLayoutNode::CuteLayoutNode(Array<PrimExpr> logical_shape,
                               Array<PrimExpr> mode_shape,
                               Array<PrimExpr> mode_stride,
                               Array<Integer> dim_levels) {
  // --- Validation ---
  int total_modes = 0;
  for (size_t d = 0; d < dim_levels.size(); ++d) {
    total_modes += dim_levels[d].IntValue();
  }
  ICHECK_EQ(dim_levels.size(), logical_shape.size())
      << "dim_levels length must match logical_shape rank";
  ICHECK_EQ(static_cast<size_t>(total_modes), mode_shape.size())
      << "sum(dim_levels) must equal mode_shape length";
  ICHECK_EQ(mode_shape.size(), mode_stride.size())
      << "mode_shape and mode_stride must have the same length";

  // Invariant: within each dimension, only the outermost mode may be
  // symbolic (non-IntImm).  Inner modes must be concrete so that
  // RecoverPhysicalOrder and DeriveLayoutLike can work generically.
  {
    int off = 0;
    for (size_t d = 0; d < dim_levels.size(); ++d) {
      int nlevels = dim_levels[d].IntValue();
      for (int i = 0; i < nlevels - 1; ++i) {
        ICHECK(mode_shape[off + i].as<IntImmNode>())
            << "CuteLayout: inner mode " << i << " of dimension " << d
            << " must be a concrete integer, but got " << mode_shape[off + i]
            << ". Only the outermost mode (index " << (nlevels - 1)
            << ") may be symbolic.";
      }
      off += nlevels;
    }
  }

  // Store structural fields.
  logical_shape_ = logical_shape;
  mode_shape_ = mode_shape;
  mode_stride_ = mode_stride;
  dim_levels_ = dim_levels;

  // --- Derive affine forward_index from structural fields ---
  //   For each logical dim d, decompose the logical index into
  //   mixed-radix mode indices using the mode shapes, then
  //   compute dot-product with mode strides.

  Array<PrimExpr> input_vars;
  for (size_t d = 0; d < logical_shape.size(); ++d) {
    input_vars.push_back(InputPlaceholder(d));
  }

  PrimExpr total_offset = make_zero(DataType::Int(32));
  int mode_idx = 0;

  for (size_t d = 0; d < logical_shape.size(); ++d) {
    int num_modes = dim_levels[d].IntValue();
    PrimExpr logical_var = input_vars[d];

    // Decompose: modes are stored innermost-first per CuTe convention.
    // For modes [m0, m1, ..., mk-1] (inner to outer), the logical
    // index I decomposes as:
    //   h_0     = I mod m0           (innermost)
    //   h_1     = (I / m0) mod m1
    //   ...
    //   h_{k-1} = I / (m0 * ... * m_{k-2})  (outermost, remainder)
    //
    // We iterate from innermost (mode_idx) to outermost (mode_idx + num_modes -
    // 1).
    PrimExpr rem = logical_var;
    for (int i = mode_idx; i < mode_idx + num_modes; ++i) {
      PrimExpr h_index;
      if (i == mode_idx + num_modes - 1) {
        // Outermost mode: just the remainder
        h_index = rem;
      } else {
        h_index = FloorMod(rem, mode_shape[i]);
        rem = FloorDiv(rem, mode_shape[i]);
      }
      total_offset = total_offset + h_index * mode_stride[i];
    }

    mode_idx += num_modes;
  }

  // Initialize base class fields.
  input_size_ = logical_shape;
  arith::Analyzer analyzer;
  UpdateAnalyzer(&analyzer);
  forward_index_ = {analyzer.Simplify(total_offset)};
}

// ---------------------------------------------------------------------------
// CuteLayout reference class constructor
// ---------------------------------------------------------------------------

CuteLayout::CuteLayout(Array<PrimExpr> logical_shape,
                       Array<PrimExpr> mode_shape, Array<PrimExpr> mode_stride,
                       Array<Integer> dim_levels) {
  auto n = tvm::ffi::make_object<CuteLayoutNode>(
      std::move(logical_shape), std::move(mode_shape), std::move(mode_stride),
      std::move(dim_levels));
  data_ = std::move(n);
}

// ---------------------------------------------------------------------------
// Per-dimension accessors
// ---------------------------------------------------------------------------

Array<PrimExpr> CuteLayoutNode::GetModeShapeOfDim(int dim) const {
  ICHECK_GE(dim, 0);
  ICHECK_LT(static_cast<size_t>(dim), dim_levels_.size());
  int start = ModeOffset(dim);
  int count = dim_levels_[dim].IntValue();
  Array<PrimExpr> result;
  for (int i = start; i < start + count; ++i) {
    result.push_back(mode_shape_[i]);
  }
  return result;
}

Array<PrimExpr> CuteLayoutNode::GetModeStrideOfDim(int dim) const {
  ICHECK_GE(dim, 0);
  ICHECK_LT(static_cast<size_t>(dim), dim_levels_.size());
  int start = ModeOffset(dim);
  int count = dim_levels_[dim].IntValue();
  Array<PrimExpr> result;
  for (int i = start; i < start + count; ++i) {
    result.push_back(mode_stride_[i]);
  }
  return result;
}

// ---------------------------------------------------------------------------
// GetCoveredShape / GetStorageSize
// ---------------------------------------------------------------------------

Array<PrimExpr> CuteLayoutNode::GetCoveredShape() const {
  Array<PrimExpr> covered;
  int mode_idx = 0;
  for (size_t d = 0; d < dim_levels_.size(); ++d) {
    int num_modes = dim_levels_[d].IntValue();
    PrimExpr prod = make_const(DataType::Int(32), 1);
    for (int i = 0; i < num_modes; ++i) {
      prod = prod * mode_shape_[mode_idx + i];
    }
    covered.push_back(prod);
    mode_idx += num_modes;
  }
  return covered;
}

PrimExpr CuteLayoutNode::GetStorageSize() const {
  // Storage size = max offset + 1 over the covered domain.
  // For a layout with mode_shape[i] and mode_stride[i], this is:
  //   sum_i ( (mode_shape[i] - 1) * mode_stride[i] ) + 1
  PrimExpr max_offset = make_zero(DataType::Int(32));
  for (size_t i = 0; i < mode_shape_.size(); ++i) {
    max_offset =
        max_offset +
        (mode_shape_[i] - make_const(DataType::Int(32), 1)) * mode_stride_[i];
  }
  return max_offset + make_const(DataType::Int(32), 1);
}

// ---------------------------------------------------------------------------
// SameLayout
// ---------------------------------------------------------------------------

bool CuteLayoutNode::SameLayout(const CuteLayoutNode *other,
                                arith::Analyzer *analyzer) const {
  if (!other)
    return false;
  if (dim_levels_.size() != other->dim_levels_.size())
    return false;
  for (size_t i = 0; i < dim_levels_.size(); ++i) {
    if (dim_levels_[i].IntValue() != other->dim_levels_[i].IntValue())
      return false;
  }
  if (mode_shape_.size() != other->mode_shape_.size())
    return false;

  // Compare mode_shape and mode_stride element-wise.
  arith::Analyzer local_analyzer;
  arith::Analyzer *ana = analyzer ? analyzer : &local_analyzer;

  for (size_t i = 0; i < mode_shape_.size(); ++i) {
    PrimExpr diff_shape = ana->Simplify(mode_shape_[i] - other->mode_shape_[i]);
    if (!is_zero(diff_shape))
      return false;
    PrimExpr diff_stride =
        ana->Simplify(mode_stride_[i] - other->mode_stride_[i]);
    if (!is_zero(diff_stride))
      return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// LayoutNode overrides
// ---------------------------------------------------------------------------

Array<PrimExpr> CuteLayoutNode::GetForwardVars() const {
  // Delegate to base; forward_index_ uses InputPlaceholder vars.
  Array<PrimExpr> vars;
  for (size_t i = 0; i < InputDim(); i++) {
    vars.push_back(InputPlaceholder(i));
  }
  return vars;
}

Array<PrimExpr> CuteLayoutNode::Forward(const Array<PrimExpr> &vars) const {
  // Delegate to base class logic.
  if (vars.empty())
    return forward_index_;
  ICHECK_GE(vars.size(), InputDim());
  Map<Var, PrimExpr> vmap;
  for (size_t i = 0; i < InputDim(); i++) {
    vmap.Set(Downcast<Var>(InputPlaceholder(i)), vars[i]);
  }
  Array<PrimExpr> result;
  for (const auto &e : forward_index_) {
    result.push_back(Substitute(e, vmap));
  }
  return result;
}

Layout CuteLayoutNode::Inverse() const {
  LOG(FATAL) << "CuteLayout does not support Inverse(). "
             << "Construct a new layout explicitly for the inverse mapping.";
  return Layout();
}

Layout CuteLayoutNode::Reshape(const Array<PrimExpr> &shape,
                               arith::Analyzer *analyzer,
                               const PrimExpr rescale_num,
                               const PrimExpr rescale_den) const {
  LOG(FATAL) << "CuteLayout does not support Reshape(). "
             << "Construct a new layout explicitly for the aliased buffer.";
  return Layout();
}

std::string CuteLayoutNode::DebugOutput() const {
  std::stringstream ss;
  ss << "CuteLayout(logical_shape=" << logical_shape_
     << ", mode_shape=" << mode_shape_ << ", mode_stride=" << mode_stride_
     << ", dim_levels=" << dim_levels_ << ")";
  return ss.str();
}

bool CuteLayoutNode::IsEqual(const LayoutNode *other, bool skip_index) const {
  // Try structural comparison first.
  if (auto *cute_other = dynamic_cast<const CuteLayoutNode *>(other)) {
    return SameLayout(cute_other);
  }
  // Fall back to base-class expression-based comparison.
  return LayoutNode::IsEqual(other, skip_index);
}

// ---------------------------------------------------------------------------
// RegisterReflection
// ---------------------------------------------------------------------------

void CuteLayoutNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<CuteLayoutNode>()
      .def_ro("logical_shape", &CuteLayoutNode::logical_shape_)
      .def_ro("mode_shape", &CuteLayoutNode::mode_shape_)
      .def_ro("mode_stride", &CuteLayoutNode::mode_stride_)
      .def_ro("dim_levels", &CuteLayoutNode::dim_levels_)
      .def("_DebugOutput", &CuteLayoutNode::DebugOutput);
}

// ---------------------------------------------------------------------------
// RecoverPhysicalOrder — extract mode permutation from strides
// ---------------------------------------------------------------------------

/*!
 * \brief Decompose a stride into (#symbolic factors, concrete coefficient).
 *
 * Layout strides are products of mode shapes (running prefix products), so a
 * stride is a Mul-tree of IntImm and symbolic (dynamic count) factors.  We
 * fold the concrete leaves into a coefficient and count the symbolic ones.
 */
static void CollectStrideFactors(const PrimExpr &e, int *sym_count,
                                 int64_t *coeff) {
  if (const auto *mul = e.as<MulNode>()) {
    CollectStrideFactors(mul->a, sym_count, coeff);
    CollectStrideFactors(mul->b, sym_count, coeff);
    return;
  }
  int64_t c;
  if (cute::tryConstInt(e, &c)) {
    *coeff *= c;
  } else {
    ++(*sym_count);
  }
}

/*!
 * \brief Extract the physical ordering (permutation) of modes from strides.
 *
 * For a contiguous CuteLayout, the strides encode the physical ordering:
 * the mode with the smallest stride is physically innermost, etc.
 *
 * The strides form a single nested product chain (each outer stride equals an
 * inner stride times an inner shape), so even when several are symbolic they
 * are totally ordered by the key (#symbolic factors, concrete coefficient):
 * the innermost has the fewest symbolic factors / smallest coefficient.  For
 * all-concrete layouts this degrades to sorting by stride value.
 *
 * \return A permutation vector: physical_order[0] is the innermost mode
 *         index, physical_order[n-1] is the outermost.
 */
static std::vector<int> RecoverPhysicalOrder(const CuteLayoutNode *layout) {
  auto strides = layout->GetModeStride();
  int n = strides.size();

  std::vector<std::pair<int, int64_t>> keys(n);
  for (int i = 0; i < n; ++i) {
    int sym_count = 0;
    int64_t coeff = 1;
    CollectStrideFactors(strides[i], &sym_count, &coeff);
    keys[i] = {sym_count, coeff};
  }

  std::vector<int> order(n);
  for (int i = 0; i < n; ++i)
    order[i] = i;

  // Innermost first: fewer symbolic factors, then smaller concrete coefficient.
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    if (keys[a].first != keys[b].first)
      return keys[a].first < keys[b].first;
    return keys[a].second < keys[b].second;
  });
  return order;
}

// ---------------------------------------------------------------------------
// Free-standing layout relation APIs
// ---------------------------------------------------------------------------

bool IsSameLayout(const Layout &lhs, const Layout &rhs,
                  arith::Analyzer *analyzer) {
  if (!lhs.defined() || !rhs.defined())
    return false;
  auto *lhs_cute = lhs.as<CuteLayoutNode>();
  auto *rhs_cute = rhs.as<CuteLayoutNode>();
  if (lhs_cute && rhs_cute) {
    return lhs_cute->SameLayout(rhs_cute, analyzer);
  }
  // Fallback: expression-based comparison.
  return lhs->IsEqual(rhs.get());
}

Optional<Layout> DeriveLayoutLike(const Layout &src, Array<PrimExpr> dst_shape,
                                  Optional<Array<Integer>> axis_map,
                                  arith::Analyzer *analyzer) {
  auto *src_cute = src.as<CuteLayoutNode>();
  if (!src_cute)
    return Optional<Layout>();

  size_t src_rank = src_cute->GetDimLevels().size();
  size_t dst_rank = dst_shape.size();

  // --- Identify multi-mode (blocked) source dimensions. ---
  std::vector<int> blocked_src_dims;
  for (size_t d = 0; d < src_rank; ++d) {
    if (src_cute->GetDimLevels()[d].IntValue() > 1)
      blocked_src_dims.push_back(d);
  }
  int num_blocked = static_cast<int>(blocked_src_dims.size());

  // --- Resolve axis_map ---
  //
  // axis_map controls where each blocked (multi-mode) source dimension
  // lands in the destination layout.  It has one entry per blocked src
  // dim: axis_map[i] = the dst dim that receives the i-th blocked src
  // dim's mode structure.
  //
  // Three cases:
  //
  //  1. Explicit axis_map — use it verbatim.
  //     Example: 4D src with ZZ on dims [1,3], axis_map = {0, 2}
  //              → dst dims 0 and 2 get the blocked structure.
  //
  //  2. axis_map is NullOpt, src_rank == dst_rank (same-rank derivation,
  //     e.g. sharding a 4D tensor into a 4D sharded tensor).
  //     Blocked dims keep their original positions (identity mapping).
  //     Example: src blocked_dims = [1, 3] → target_axes = [1, 3].
  //     This is the common case for MeshTensor sharding where the
  //     tensor rank does not change and the block structure must stay
  //     on the same logical axes.
  //
  //  3. axis_map is NullOpt, src_rank != dst_rank (rank-changing
  //     derivation, e.g. reducing a 3D batched tensor to 2D).
  //     Blocked dims are packed onto the last N axes of dst.
  //     Example: 3D src with ZZ on dims [1, 2], dst is 2D
  //              → target_axes = [0, 1] (last 2 of 2).
  //     This is natural for reductions where leading batch dims are
  //     dropped and the blocked spatial dims fill the output.
  //
  // If dst doesn't have enough dims to hold all blocked dims, we
  // cannot derive a meaningful layout — return NullOpt.
  std::vector<int> target_axes;
  if (axis_map.defined()) {
    // Case 1: explicit mapping.
    for (auto &v : axis_map.value()) {
      target_axes.push_back(v.IntValue());
    }
    ICHECK_EQ(static_cast<int>(target_axes.size()), num_blocked)
        << "axis_map must have one entry per multi-mode source dimension";
  } else {
    if (num_blocked > static_cast<int>(dst_rank)) {
      // Not enough dst dims to place all blocked structure.
      return Optional<Layout>();
    }
    if (src_rank == dst_rank) {
      // Case 2: same rank — identity mapping.
      for (int i = 0; i < num_blocked; ++i) {
        target_axes.push_back(blocked_src_dims[i]);
      }
    } else {
      // Case 3: rank change — pack into last N dst axes.
      for (int i = 0; i < num_blocked; ++i) {
        target_axes.push_back(static_cast<int>(dst_rank) - num_blocked + i);
      }
    }
  }

  // Reverse lookup: dst_dim → index into blocked_src_dims.
  std::unordered_map<int, int> dst_to_blocked_idx;
  for (size_t i = 0; i < target_axes.size(); ++i) {
    dst_to_blocked_idx[target_axes[i]] = i;
  }

  // --- Pass 1: build new mode_shape and dim_levels ---
  //
  // For each dst dim: if it maps to a blocked src dim, preserve inner
  // (fixed) modes and recompute the outermost mode via ceildiv.
  // Otherwise, create a single-level mode (stride assigned in Pass 2).

  Array<PrimExpr> new_mode_shape;
  Array<Integer> new_dim_levels;

  for (size_t dst_d = 0; dst_d < dst_rank; ++dst_d) {
    PrimExpr dst_ext = dst_shape[dst_d];
    auto it = dst_to_blocked_idx.find(dst_d);

    if (it != dst_to_blocked_idx.end()) {
      // Blocked dst dim — derive mode structure from corresponding src dim.
      int src_d = blocked_src_dims[it->second];
      int nlevels = src_cute->GetDimLevels()[src_d].IntValue();
      new_dim_levels.push_back(Integer(nlevels));

      PrimExpr inner_product = make_const(DataType::Int(32), 1);
      Array<PrimExpr> dim_shapes = src_cute->GetModeShapeOfDim(src_d);
      for (int i = 0; i < nlevels - 1; ++i) {
        new_mode_shape.push_back(dim_shapes[i]);
        inner_product = inner_product * dim_shapes[i];
      }
      new_mode_shape.push_back(ceildiv(dst_ext, inner_product));
    } else {
      // Non-blocked dst dim — single-level row-major.
      new_dim_levels.push_back(Integer(1));
      new_mode_shape.push_back(dst_ext);
    }
  }

  // --- Pass 2: recover physical ordering and assign contiguous strides ---
  //
  // We map ALL src modes (blocked + single-level) to new mode indices,
  // then filter RecoverPhysicalOrder through that map.  This preserves
  // the physical stride ordering for every layout kind (row-major,
  // column-major, ZZ, ZN, ...) without explicit kind detection.
  //
  // Single-level src dims are matched to non-blocked dst dims, aligned
  // from the back: last single-level src dim → last remaining dst dim.
  // Truly excess dst dims (when dst_rank > mapped count) get fresh
  // row-major modes at outermost.

  auto full_order = RecoverPhysicalOrder(src_cute);

  // Collect single-level src dims and non-blocked dst dims.
  std::vector<int> single_src_dims;
  for (size_t d = 0; d < src_rank; ++d) {
    if (src_cute->GetDimLevels()[d].IntValue() == 1)
      single_src_dims.push_back(d);
  }
  std::vector<int> nonblocked_dst_dims;
  for (size_t dst_d = 0; dst_d < dst_rank; ++dst_d) {
    if (dst_to_blocked_idx.find(dst_d) == dst_to_blocked_idx.end())
      nonblocked_dst_dims.push_back(dst_d);
  }

  // Align single-level src dims to non-blocked dst dims from the back.
  // E.g. single_src=[0] nonblocked_dst=[0,1] → src dim 0 maps to dst dim 1,
  //       dst dim 0 is excess (unmapped).
  // E.g. single_src=[0,1,2] nonblocked_dst=[0] → src dim 2 maps to dst dim 0,
  //       src dims 0,1 are dropped (reduced).
  std::unordered_map<int, int> single_src_to_dst;
  {
    int ns = static_cast<int>(single_src_dims.size());
    int nd = static_cast<int>(nonblocked_dst_dims.size());
    int count = std::min(ns, nd);
    for (int i = 0; i < count; ++i) {
      single_src_to_dst[single_src_dims[ns - count + i]] =
          nonblocked_dst_dims[nd - count + i];
    }
  }

  // Build src_mode → new_mode mapping for ALL mapped modes.
  std::unordered_map<int, int> src_mode_to_new;
  std::vector<int> unmapped_new_indices; // excess dst dims with no src
  int new_idx = 0;

  for (size_t dst_d = 0; dst_d < dst_rank; ++dst_d) {
    auto it = dst_to_blocked_idx.find(dst_d);
    if (it != dst_to_blocked_idx.end()) {
      // Blocked dst dim — map all modes from the corresponding src dim.
      int src_d = blocked_src_dims[it->second];
      int nlevels = src_cute->GetDimLevels()[src_d].IntValue();
      int src_mode_start = 0;
      for (int d2 = 0; d2 < src_d; ++d2)
        src_mode_start += src_cute->GetDimLevels()[d2].IntValue();
      for (int i = 0; i < nlevels; ++i)
        src_mode_to_new[src_mode_start + i] = new_idx++;
    } else {
      // Check if a single-level src dim maps here.
      bool found = false;
      for (auto &kv : single_src_to_dst) {
        if (kv.second == static_cast<int>(dst_d)) {
          // Map the single src mode to this new mode index.
          int src_d = kv.first;
          int src_mode_start = 0;
          for (int d2 = 0; d2 < src_d; ++d2)
            src_mode_start += src_cute->GetDimLevels()[d2].IntValue();
          src_mode_to_new[src_mode_start] = new_idx++;
          found = true;
          break;
        }
      }
      if (!found) {
        // Excess dst dim — no corresponding src dim.
        unmapped_new_indices.push_back(new_idx++);
      }
    }
  }

  // Filter RecoverPhysicalOrder through the mapping (covers ALL mapped modes).
  // Unmapped (excess) dst dims get fresh row-major order at outermost.
  std::vector<int> order;
  for (int idx : full_order) {
    if (src_mode_to_new.count(idx))
      order.push_back(src_mode_to_new[idx]);
  }
  // Excess dims: last dst dim innermost (row-major convention).
  for (auto rit = unmapped_new_indices.rbegin();
       rit != unmapped_new_indices.rend(); ++rit)
    order.push_back(*rit);

  ICHECK_EQ(order.size(), new_mode_shape.size());

  Array<PrimExpr> new_mode_stride(new_mode_shape.size(), PrimExpr());
  PrimExpr running = make_const(DataType::Int(32), 1);
  for (int idx : order) {
    new_mode_stride.Set(idx, running);
    running = running * new_mode_shape[idx];
  }

  Array<PrimExpr> new_logical_shape(dst_shape.begin(), dst_shape.end());

  return CuteLayout(new_logical_shape, new_mode_shape, new_mode_stride,
                    new_dim_levels);
}

Layout TryCanonicalizeToRowMajor(const Layout &layout) {
  const auto *cute = layout.as<CuteLayoutNode>();
  if (!cute)
    return layout; // non-CuteLayout: nothing to coalesce

  Array<PrimExpr> logical_shape = cute->GetLogicalShape();
  Array<Integer> dim_levels = cute->GetDimLevels();

  Array<PrimExpr> mode_shape, mode_stride;
  Array<Integer> new_levels;
  for (size_t d = 0; d < dim_levels.size(); ++d) {
    Array<PrimExpr> shapes = cute->GetModeShapeOfDim(d);
    Array<PrimExpr> strides = cute->GetModeStrideOfDim(d);
    int nlevels = dim_levels[d].IntValue();

    // Per-dimension coalesce (storage order preserved -- never sort, that would
    // change the index->offset map): drop static size-1 modes, and merge into
    // the previous mode when both are static and contiguous.
    std::vector<std::pair<PrimExpr, PrimExpr>> modes; // (shape, stride)
    for (int i = 0; i < nlevels; ++i) {
      const auto *cur_sh = shapes[i].as<IntImmNode>();
      if (cur_sh && cur_sh->value == 1)
        continue; // drop a degenerate (size-1) mode
      if (!modes.empty() && cur_sh) {
        const auto *pm = modes.back().first.as<IntImmNode>();
        const auto *ps = modes.back().second.as<IntImmNode>();
        const auto *cs = strides[i].as<IntImmNode>();
        if (pm && ps && cs && pm->value * ps->value == cs->value) {
          modes.back().first =
              IntImm(modes.back().first.dtype(), pm->value * cur_sh->value);
          continue; // merge: inner.shape * inner.stride == outer.stride
        }
      }
      modes.push_back({shapes[i], strides[i]});
    }
    if (modes.empty()) // all modes were size-1: a logical extent-1 dim
      modes.push_back(
          {IntImm(shapes[0].dtype(), 1), IntImm(strides[0].dtype(), 0)});

    new_levels.push_back(Integer(static_cast<int>(modes.size())));
    for (const auto &[sh, st] : modes) {
      mode_shape.push_back(sh);
      mode_stride.push_back(st);
    }
  }
  Layout coalesced =
      CuteLayout(logical_shape, mode_shape, mode_stride, new_levels);
  // Coalescing preserves the byte map but rewrites the mode structure that
  // DeriveLayoutLike reads as the layout kind, so a partial coalesce would
  // change how IsLayoutMatch shape-adapts the layout. Commit only when all
  // structure is gone (plain row-major); keep anything still blocked or
  // padded unchanged.
  if (IsSameLayout(coalesced, sunmmio::MakeRowMajor(logical_shape)))
    return coalesced;
  return layout;
}

bool IsLayoutMatch(const Layout &lhs, const Layout &rhs,
                   arith::Analyzer *analyzer) {
  // Compare on canonical forms so representations describing the
  // same byte map agree -- e.g. a single-block ZZ canonicalizes to row-major
  // and matches a row-major buffer. Checked in both directions:
  // DeriveLayoutLike rebuilds the template at the target's shape with tight
  // strides, dropping the template's padding, so a single direction only sees
  // padding on rhs; both directions give each layout a turn as rhs (so a tight
  // and an alignment- padded row-major do not match), while unpadded layouts of
  // different shapes still match.
  Layout cl = TryCanonicalizeToRowMajor(lhs),
         cr = TryCanonicalizeToRowMajor(rhs);
  // Identical coalesced forms => identical byte map. Covers the reflexive and
  // same-shape cases, including over-covered dims (covered > logical, e.g. a ZZ
  // whose logical extent is below the block size) that DeriveLayoutLike would
  // otherwise tighten and miss.
  if (IsSameLayout(cl, cr, analyzer))
    return true;
  // Different logical shapes, same kind: re-derive each onto the other's shape.
  auto fwd = DeriveLayoutLike(cl, cr->InputShape(), Optional<Array<Integer>>(),
                              analyzer);
  auto bwd = DeriveLayoutLike(cr, cl->InputShape(), Optional<Array<Integer>>(),
                              analyzer);
  return fwd.defined() && IsSameLayout(fwd.value(), cr, analyzer) &&
         bwd.defined() && IsSameLayout(bwd.value(), cl, analyzer);
}

// ---------------------------------------------------------------------------
// ComputeContiguousTileSteps — stride-walk algorithm
// ---------------------------------------------------------------------------

std::vector<ContiguousStep> ComputeContiguousTileSteps(const Layout &layout) {
  auto *cute = layout.as<CuteLayoutNode>();
  if (!cute)
    return {}; // non-CuteLayout: caller handles row-major fallback

  auto dim_levels = cute->GetDimLevels();
  int ndim = static_cast<int>(dim_levels.size());

  // Step 1 & 2: extract concrete inner modes per dim, coalesce within each.
  struct Mode {
    int64_t shape;
    int64_t stride;
    int dim;
  };
  std::vector<Mode> modes;

  for (int d = 0; d < ndim; ++d) {
    auto shapes = cute->GetModeShapeOfDim(d);
    auto strides = cute->GetModeStrideOfDim(d);
    int nlevels = dim_levels[d].IntValue();

    // Collect innermost consecutive static modes, stopping at the first
    // symbolic shape or stride.
    std::vector<std::pair<int64_t, int64_t>> dim_modes;
    for (int i = 0; i < nlevels; ++i) {
      auto *s = shapes[i].as<IntImmNode>();
      auto *st = strides[i].as<IntImmNode>();
      if (!s || !st)
        break; // stop at symbolic boundary
      if (s->value == 1)
        continue; // skip trivial modes
      dim_modes.push_back({s->value, st->value});
    }

    // Coalesce: sort by stride ascending within this dim, then merge
    // where shape * stride == next_stride.
    std::sort(dim_modes.begin(), dim_modes.end(),
              [](const auto &a, const auto &b) { return a.second < b.second; });

    std::vector<std::pair<int64_t, int64_t>> coalesced;
    for (const auto &[s, st] : dim_modes) {
      if (!coalesced.empty() &&
          coalesced.back().first * coalesced.back().second == st) {
        coalesced.back().first *= s;
      } else {
        coalesced.push_back({s, st});
      }
    }

    for (const auto &[s, st] : coalesced) {
      modes.push_back({s, st, d});
    }
  }

  // Step 3: sort all modes by stride ascending.
  std::sort(modes.begin(), modes.end(),
            [](const Mode &a, const Mode &b) { return a.stride < b.stride; });

  // Step 4: walk while contiguous — stride must equal running product.
  std::vector<ContiguousStep> steps;
  int64_t running = 1;
  for (const auto &m : modes) {
    if (m.stride < running)
      continue; // already covered by a prior mode
    if (m.stride > running)
      break; // gap — end of contiguous region
    steps.push_back({m.dim, static_cast<int>(m.shape)});
    running *= m.shape;
  }

  return steps;
}

// ---------------------------------------------------------------------------
// Sunmmio named constructors
// ---------------------------------------------------------------------------

namespace sunmmio {

bool IsZZLike(const Layout &layout) {
  auto *cute = layout.as<CuteLayoutNode>();
  if (!cute)
    return false;

  auto dim_levels = cute->GetDimLevels();

  // Collect blocked dims (nlevels > 1).
  std::vector<int> blocked_dims;
  for (size_t d = 0; d < dim_levels.size(); ++d) {
    if (dim_levels[d].IntValue() > 1)
      blocked_dims.push_back(d);
  }
  if (blocked_dims.size() < 2)
    return false;

  // Last two blocked dims correspond to (ax0, ax1) in the Make* constructors.
  int ax0 = blocked_dims[blocked_dims.size() - 2];
  int ax1 = blocked_dims[blocked_dims.size() - 1];

  // Innermost mode stride for each blocked dim.
  PrimExpr stride_ax0 = cute->GetModeStrideOfDim(ax0)[0];
  PrimExpr stride_ax1 = cute->GetModeStrideOfDim(ax1)[0];

  // ZZ-like: ax1's innermost stride < ax0's innermost stride (row-major inner).
  auto *s0 = stride_ax0.as<IntImmNode>();
  auto *s1 = stride_ax1.as<IntImmNode>();
  if (s0 && s1)
    return s1->value < s0->value;

  // Fallback: stride 1 on ax1 is the common ZZ-like pattern.
  if (s1 && s1->value == 1)
    return true;

  return false;
}

std::optional<ZZBlockShape> GetZZBlockShape(const Layout &layout) {
  if (!IsZZLike(layout))
    return std::nullopt;

  auto *cute = layout.as<CuteLayoutNode>();
  ICHECK(cute);

  auto dim_levels = cute->GetDimLevels();

  // Find the last two blocked dims (same logic as IsZZLike).
  std::vector<int> blocked_dims;
  for (size_t d = 0; d < dim_levels.size(); ++d) {
    if (dim_levels[d].IntValue() > 1)
      blocked_dims.push_back(d);
  }
  ICHECK_GE(blocked_dims.size(), 2);
  int ax0 = blocked_dims[blocked_dims.size() - 2];
  int ax1 = blocked_dims[blocked_dims.size() - 1];

  // Innermost mode shape of each blocked dim is the block extent.
  PrimExpr shape_ax0 = cute->GetModeShapeOfDim(ax0)[0];
  PrimExpr shape_ax1 = cute->GetModeShapeOfDim(ax1)[0];

  auto *s0 = shape_ax0.as<IntImmNode>();
  auto *s1 = shape_ax1.as<IntImmNode>();
  if (!s0 || !s1)
    return std::nullopt;

  return ZZBlockShape{static_cast<int>(s0->value), static_cast<int>(s1->value)};
}

static PrimExpr makeInt(int64_t v) { return make_const(DataType::Int(32), v); }

Layout MakeRowMajor(Array<PrimExpr> shape) {
  Array<PrimExpr> mode_shape;
  Array<PrimExpr> mode_stride;
  Array<Integer> dim_levels;

  // Row-major strides: last dim has stride 1, previous dims have
  // stride = product of all subsequent dims.
  int rank = shape.size();
  std::vector<PrimExpr> strides(rank);
  strides[rank - 1] = make_const(DataType::Int(32), 1);
  for (int i = rank - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  for (int d = 0; d < rank; ++d) {
    mode_shape.push_back(shape[d]);
    mode_stride.push_back(strides[d]);
    dim_levels.push_back(Integer(1));
  }

  return CuteLayout(shape, mode_shape, mode_stride, dim_levels);
}

Layout MakeAlignedRowMajor(Array<PrimExpr> shape, DataType dtype,
                           int align_bytes) {
  int rank = shape.size();
  if (rank < 1)
    return MakeRowMajor(shape);

  // Number of elements that fill one alignment unit (e.g. 64B = 32 bf16 or
  // 128 fp4 elements). Bit arithmetic keeps sub-byte dtypes exact.
  int elem_bits = dtype.bits();
  ICHECK_GT(elem_bits, 0) << "MakeAlignedRowMajor requires a positive element "
                             "bit-width, but got dtype "
                          << dtype;
  int align_elems = align_bytes * 8 / elem_bits;
  if (align_elems < 1)
    align_elems = 1;

  Array<PrimExpr> mode_shape;
  Array<PrimExpr> mode_stride;
  Array<Integer> dim_levels;

  // Pad the innermost (contiguous) extent to a multiple of align_elems and
  // store it as the covered extent in mode_shape; logical_shape keeps the true
  // extent.
  for (int d = 0; d < rank; ++d)
    mode_shape.push_back(shape[d]);
  mode_shape.Set(rank - 1, ceildiv(shape[rank - 1], makeInt(align_elems)) *
                               makeInt(align_elems));

  // Dense row-major strides over the PADDED mode_shape, so alignment propagates
  // to every outer dimension automatically.
  std::vector<PrimExpr> strides(rank);
  strides[rank - 1] = makeInt(1);
  for (int i = rank - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * mode_shape[i + 1];
  }

  for (int d = 0; d < rank; ++d) {
    mode_stride.push_back(strides[d]);
    dim_levels.push_back(Integer(1));
  }

  return CuteLayout(shape, mode_shape, mode_stride, dim_levels);
}

/*!
 * \brief Build a row-major CuteAlgebraLayout from a logical shape.
 *
 * Creates a rank-N layout where each dimension is a scalar mode with
 * row-major strides (last dimension has stride 1).
 */
static cute::CuteAlgebraLayout
makeRowMajorAlgebra(const Array<PrimExpr> &shape) {
  int rank = shape.size();
  std::vector<PrimExpr> strides(rank);
  strides[rank - 1] = makeInt(1);
  for (int i = rank - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  std::vector<cute::CuteAlgebraLayout> modes;
  modes.reserve(rank);
  for (int i = 0; i < rank; ++i) {
    modes.emplace_back(shape[i], strides[i]);
  }
  return cute::CuteAlgebraLayout(modes);
}

/*!
 * \brief Build a rank-N tiler for logicalDivide.
 *
 * The tiler has block_shape values on tiled axes (ax0, ax1) and 1 on
 * all other axes.  Each mode is a scalar (shape, stride=1) layout.
 */
static cute::CuteAlgebraLayout buildTiler(int rank, int ax0, int ax1,
                                          const Array<PrimExpr> &block_shape) {
  std::vector<cute::CuteAlgebraLayout> modes;
  modes.reserve(rank);
  for (int d = 0; d < rank; ++d) {
    if (d == ax0)
      modes.emplace_back(block_shape[0], makeInt(1));
    else if (d == ax1)
      modes.emplace_back(block_shape[1], makeInt(1));
    else
      modes.emplace_back(makeInt(1), makeInt(1));
  }
  return cute::CuteAlgebraLayout(modes);
}

/*!
 * \brief Convert per-dimension mode shapes to a CuteLayout with
 *        contiguous physical strides.
 *
 * \param dim_modes  Per-dimension mode shapes.
 *                   Tiled dims: [BM, QM] (2-level) or [BM, TM, GM] (3-level).
 *                   Non-tiled dims: [D] (single level).
 * \param shape      Full logical shape.
 * \param ax0, ax1   Indices of the two tiled dimensions.
 * \param level_col_major  Per-level col-major flag, indexed from innermost
 *                         (level 0) to outermost.  Length equals the number
 *                         of levels in tiled dims.
 *                         false (Z/row-major): ax1 first, ax0 second.
 *                         true  (N/col-major): ax0 first, ax1 second.
 */
static Layout toCuteLayout(const std::vector<std::vector<PrimExpr>> &dim_modes,
                           Array<PrimExpr> shape, int ax0, int ax1,
                           const std::vector<bool> &level_col_major) {
  int rank = shape.size();
  size_t tiled_levels = dim_modes[ax0].size();
  ICHECK_EQ(dim_modes[ax1].size(), tiled_levels)
      << "Tiled axes must have the same number of levels";
  ICHECK_EQ(level_col_major.size(), tiled_levels)
      << "level_col_major size must match number of tiled levels";

  // Build physical mode ordering (innermost → outermost).
  struct ModeRef {
    int dim;
    size_t level;
  };
  std::vector<ModeRef> order;

  // Tiled modes by level.
  for (size_t lv = 0; lv < tiled_levels; ++lv) {
    if (level_col_major[lv]) {
      // N (col-major): ax0 innermost.
      order.push_back({ax0, lv});
      order.push_back({ax1, lv});
    } else {
      // Z (row-major): ax1 innermost.
      order.push_back({ax1, lv});
      order.push_back({ax0, lv});
    }
  }

  // Non-tiled dims: row-major (rightmost dim is innermost).
  for (int d = rank - 1; d >= 0; --d) {
    if (d != ax0 && d != ax1)
      order.push_back({d, 0});
  }

  // Compute contiguous strides from ordering.
  std::map<std::pair<int, size_t>, PrimExpr> stride_map;
  PrimExpr running = makeInt(1);
  for (const auto &ref : order) {
    stride_map[{ref.dim, ref.level}] = running;
    running = running * dim_modes[ref.dim][ref.level];
  }

  // Assemble mode_shape, mode_stride, dim_levels.
  Array<PrimExpr> out_mode_shape;
  Array<PrimExpr> out_mode_stride;
  Array<Integer> dim_levels;

  for (int d = 0; d < rank; ++d) {
    for (size_t lv = 0; lv < dim_modes[d].size(); ++lv) {
      out_mode_shape.push_back(dim_modes[d][lv]);
      out_mode_stride.push_back(stride_map[{d, lv}]);
    }
    dim_levels.push_back(Integer(dim_modes[d].size()));
  }

  return CuteLayout(shape, out_mode_shape, out_mode_stride, dim_levels);
}

/*!
 * \brief Iteratively divide the full layout by a chain of tilers.
 *
 * Given tiler_chain = [block_shape, cluster_shape, ...], performs:
 *   Stage 0: logicalDivide(full_row_major, block_tiler, byMode=true)
 *   Stage k>0: for each tiled dim, logicalDivide(outer_{k-1}, tiler_k)
 *
 * At each stage, logicalDivide produces (tiler_part, remainder) per dim.
 * The tiler_part represents the partition size (fixed), and the remainder
 * represents the count of partitions (derived from the shape).
 *
 * For the first divide, the tiler_part IS the innermost block.
 * For subsequent divides, the tiler_part is placed at the middle level
 * (fixed cluster size) and the remainder is placed at the outermost
 * level (derived grid count):
 *   [block, tiler_1, count_1, tiler_2, count_2, ...]
 *
 * For tiler_chain = [block_shape]:
 *   per tiled dim: [BM, QM]              (2 levels)
 *
 * For tiler_chain = [block_shape, cluster_shape]:
 *   per tiled dim: [BM, CM, GM]          (3 levels)
 *   where CM = cluster_shape (fixed), GM = ceil(QM/CM) = remainder (derived)
 *
 * Non-tiled dims always produce a single level: [D].
 *
 * The outermost mode is always the derived (remainder) mode, making it
 * natural for DeriveLayoutLike to recompute and for symbolic shapes.
 */
static std::vector<std::vector<PrimExpr>>
iterativeDivide(const Array<PrimExpr> &shape, int ax0, int ax1,
                const std::vector<Array<PrimExpr>> &tiler_chain) {
  ICHECK(!tiler_chain.empty());
  int rank = shape.size();

  // Stage 0: divide full row-major layout by the first tiler.
  auto full = makeRowMajorAlgebra(shape);
  auto first_tiler = buildTiler(rank, ax0, ax1, tiler_chain[0]);
  auto divided = cute::logicalDivide(full, first_tiler, /*byMode=*/true);

  // Per-dim state: collect mode shapes and track the current outer layout.
  // After stage 0, tiled dims have mode(0) = tiler_part, mode(1) = outer.
  std::vector<std::vector<PrimExpr>> dim_modes(rank);
  // Track the current outer (remainder) layout per tiled dim.
  std::vector<cute::CuteAlgebraLayout> outer(rank);

  for (int d = 0; d < rank; ++d) {
    if (d == ax0 || d == ax1) {
      auto mode_d = divided.mode(d); // shape (BM, QM)
      auto inner_flat = cute::flattenExprTuple(mode_d.mode(0).shape);
      ICHECK_EQ(inner_flat.size(), 1);
      dim_modes[d].push_back(inner_flat[0]); // BM
      outer[d] = mode_d.mode(1);             // QM layout
    } else {
      // Non-tiled: logicalDivide gives (1, D). Take D.
      auto flat = cute::flattenExprTuple(divided.mode(d).shape);
      ICHECK_GE(flat.size(), 1);
      dim_modes[d].push_back(flat.back()); // D
    }
  }

  // Subsequent stages: divide the outer part by each additional tiler.
  for (size_t stage = 1; stage < tiler_chain.size(); ++stage) {
    const auto &tiler_shapes = tiler_chain[stage];
    ICHECK_EQ(tiler_shapes.size(), 2);

    for (int d = 0; d < rank; ++d) {
      if (d != ax0 && d != ax1)
        continue; // Non-tiled: nothing to do.

      int ti = (d == ax0) ? 0 : 1;
      cute::CuteAlgebraLayout tiler_d(tiler_shapes[ti], makeInt(1));
      auto stage_result = cute::logicalDivide(outer[d], tiler_d, false);
      // mode(0) = tiler_part (partition size), mode(1) = remainder (count)

      auto tiler_flat = cute::flattenExprTuple(stage_result.mode(0).shape);
      auto remainder_flat = cute::flattenExprTuple(stage_result.mode(1).shape);
      ICHECK_EQ(tiler_flat.size(), 1);
      ICHECK_EQ(remainder_flat.size(), 1);

      // Place tiler (cluster, fixed) at middle, remainder (grid, derived)
      // outermost.
      dim_modes[d].push_back(tiler_flat[0]);     // CM (cluster size, fixed)
      dim_modes[d].push_back(remainder_flat[0]); // GM (grid count, derived)

      // Update outer for potential further stages.
      outer[d] = stage_result.mode(1);
    }
  }

  // If only one tiler, append the outer (remainder) as the last level.
  if (tiler_chain.size() == 1) {
    for (int d = 0; d < rank; ++d) {
      if (d == ax0 || d == ax1) {
        auto outer_flat = cute::flattenExprTuple(outer[d].shape);
        ICHECK_EQ(outer_flat.size(), 1);
        dim_modes[d].push_back(outer_flat[0]); // QM
      }
    }
  }

  return dim_modes;
}

Layout MakeZZ(Array<PrimExpr> shape, Array<Integer> axes,
              Array<PrimExpr> block_shape) {
  ICHECK_EQ(axes.size(), 2) << "MakeZZ requires exactly 2 axes";
  ICHECK_EQ(block_shape.size(), 2);
  int ax0 = axes[0].IntValue(), ax1 = axes[1].IntValue();

  auto dim_modes = iterativeDivide(shape, ax0, ax1, {block_shape});

  // ZZ: row-major at both levels.
  return toCuteLayout(dim_modes, shape, ax0, ax1, {false, false});
}

Layout MakeZN(Array<PrimExpr> shape, Array<Integer> axes,
              Array<PrimExpr> block_shape) {
  ICHECK_EQ(axes.size(), 2) << "MakeZN requires exactly 2 axes";
  ICHECK_EQ(block_shape.size(), 2);
  int ax0 = axes[0].IntValue(), ax1 = axes[1].IntValue();

  auto dim_modes = iterativeDivide(shape, ax0, ax1, {block_shape});

  // ZN: col-major at block level (innermost), row-major at tiler level.
  return toCuteLayout(dim_modes, shape, ax0, ax1, {true, false});
}

Layout MakeZZZ(Array<PrimExpr> shape, Array<Integer> axes,
               Array<PrimExpr> block_shape, Array<PrimExpr> cluster_shape) {
  ICHECK_EQ(axes.size(), 2);
  ICHECK_EQ(block_shape.size(), 2);
  ICHECK_EQ(cluster_shape.size(), 2);
  int ax0 = axes[0].IntValue(), ax1 = axes[1].IntValue();

  auto dim_modes =
      iterativeDivide(shape, ax0, ax1, {block_shape, cluster_shape});

  // ZZZ: row-major at all 3 levels.
  return toCuteLayout(dim_modes, shape, ax0, ax1, {false, false, false});
}

Layout MakeNZZ(Array<PrimExpr> shape, Array<Integer> axes,
               Array<PrimExpr> block_shape, Array<PrimExpr> cluster_shape) {
  ICHECK_EQ(axes.size(), 2);
  ICHECK_EQ(block_shape.size(), 2);
  ICHECK_EQ(cluster_shape.size(), 2);
  int ax0 = axes[0].IntValue(), ax1 = axes[1].IntValue();

  auto dim_modes =
      iterativeDivide(shape, ax0, ax1, {block_shape, cluster_shape});

  // NZZ: row-major at block and tile levels, col-major at group level.
  return toCuteLayout(dim_modes, shape, ax0, ax1, {false, false, true});
}

} // namespace sunmmio

// ---------------------------------------------------------------------------
// FFI registration
// ---------------------------------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  CuteLayoutNode::RegisterReflection();

  refl::GlobalDef()
      .def("tl.make_cute_layout",
           [](Array<PrimExpr> logical_shape, Array<PrimExpr> mode_shape,
              Array<PrimExpr> mode_stride, Array<Integer> dim_levels) {
             return CuteLayout(logical_shape, mode_shape, mode_stride,
                               dim_levels);
           })
      .def("tl.CuteLayout_logical_shape",
           [](CuteLayout layout) { return layout->GetLogicalShape(); })
      .def("tl.CuteLayout_mode_shape",
           [](CuteLayout layout) { return layout->GetModeShape(); })
      .def("tl.CuteLayout_mode_stride",
           [](CuteLayout layout) { return layout->GetModeStride(); })
      .def("tl.CuteLayout_dim_levels",
           [](CuteLayout layout) { return layout->GetDimLevels(); })
      .def("tl.CuteLayout_covered_shape",
           [](CuteLayout layout) { return layout->GetCoveredShape(); })
      .def("tl.CuteLayout_storage_size",
           [](CuteLayout layout) { return layout->GetStorageSize(); })
      .def("tl.CuteLayout_same_layout",
           [](CuteLayout lhs, CuteLayout rhs) {
             return lhs->SameLayout(rhs.get());
           })
      .def("tl.IsSameLayout",
           [](Layout lhs, Layout rhs) { return IsSameLayout(lhs, rhs); })
      .def("tl.DeriveLayoutLike",
           [](Layout src, Array<PrimExpr> dst_shape,
              Optional<Array<Integer>> axis_map) {
             return DeriveLayoutLike(src, dst_shape, axis_map);
           })
      .def("tl.IsLayoutMatch",
           [](Layout lhs, Layout rhs) { return IsLayoutMatch(lhs, rhs); })
      .def("tl.TryCanonicalizeToRowMajor",
           [](Layout layout) { return TryCanonicalizeToRowMajor(layout); })
      .def("tl.sunmmio.make_row_major",
           [](Array<PrimExpr> shape) { return sunmmio::MakeRowMajor(shape); })
      .def("tl.sunmmio.make_aligned_row_major",
           [](Array<PrimExpr> shape, DataType dtype, int align_bytes) {
             return sunmmio::MakeAlignedRowMajor(shape, dtype, align_bytes);
           })
      .def("tl.sunmmio.make_zz",
           [](Array<PrimExpr> shape, Array<Integer> axes,
              Array<PrimExpr> block_shape) {
             return sunmmio::MakeZZ(shape, axes, block_shape);
           })
      .def("tl.sunmmio.make_zn",
           [](Array<PrimExpr> shape, Array<Integer> axes,
              Array<PrimExpr> block_shape) {
             return sunmmio::MakeZN(shape, axes, block_shape);
           })
      .def("tl.sunmmio.make_zzz",
           [](Array<PrimExpr> shape, Array<Integer> axes,
              Array<PrimExpr> block_shape, Array<PrimExpr> cluster_shape) {
             return sunmmio::MakeZZZ(shape, axes, block_shape, cluster_shape);
           })
      .def("tl.sunmmio.make_nzz",
           [](Array<PrimExpr> shape, Array<Integer> axes,
              Array<PrimExpr> block_shape, Array<PrimExpr> cluster_shape) {
             return sunmmio::MakeNZZ(shape, axes, block_shape, cluster_shape);
           })
      // CuTe layout algebra FFI
      .def("tl.cute.zipped_product",
           [](Array<PrimExpr> a_shapes, Array<PrimExpr> a_strides,
              Array<PrimExpr> b_shapes, Array<PrimExpr> b_strides) {
             std::vector<PrimExpr> as(a_shapes.begin(), a_shapes.end());
             std::vector<PrimExpr> ad(a_strides.begin(), a_strides.end());
             std::vector<PrimExpr> bs(b_shapes.begin(), b_shapes.end());
             std::vector<PrimExpr> bd(b_strides.begin(), b_strides.end());
             cute::CuteAlgebraLayout a(as, ad);
             cute::CuteAlgebraLayout b(bs, bd);
             auto result = cute::zippedProduct(a, b);
             auto flatShape = cute::flattenExprTuple(result.shape);
             auto flatStride = cute::flattenExprTuple(result.stride);
             Array<PrimExpr> out_shapes(flatShape.begin(), flatShape.end());
             Array<PrimExpr> out_strides(flatStride.begin(), flatStride.end());
             return Array<Array<PrimExpr>>{out_shapes, out_strides};
           })
      .def(
          "tl.cute.complement",
          [](Array<PrimExpr> shapes, Array<PrimExpr> strides, int64_t max_idx) {
            std::vector<PrimExpr> s(shapes.begin(), shapes.end());
            std::vector<PrimExpr> d(strides.begin(), strides.end());
            cute::CuteAlgebraLayout layout(s, d);
            auto result = cute::complement(
                layout, make_const(DataType::Int(32), max_idx));
            auto flatShape = cute::flattenExprTuple(result.shape);
            auto flatStride = cute::flattenExprTuple(result.stride);
            Array<PrimExpr> out_shapes(flatShape.begin(), flatShape.end());
            Array<PrimExpr> out_strides(flatStride.begin(), flatStride.end());
            return Array<Array<PrimExpr>>{out_shapes, out_strides};
          })
      .def("tl.cute.coalesce",
           [](Array<PrimExpr> shapes, Array<PrimExpr> strides) {
             std::vector<PrimExpr> s(shapes.begin(), shapes.end());
             std::vector<PrimExpr> d(strides.begin(), strides.end());
             cute::CuteAlgebraLayout layout(s, d);
             auto result = cute::coalesce(layout);
             auto flatShape = cute::flattenExprTuple(result.shape);
             auto flatStride = cute::flattenExprTuple(result.stride);
             Array<PrimExpr> out_shapes(flatShape.begin(), flatShape.end());
             Array<PrimExpr> out_strides(flatStride.begin(), flatStride.end());
             return Array<Array<PrimExpr>>{out_shapes, out_strides};
           })
      .def("tl.cute.logical_divide",
           [](Array<PrimExpr> a_shapes, Array<PrimExpr> a_strides,
              Array<PrimExpr> b_shapes, Array<PrimExpr> b_strides) {
             std::vector<PrimExpr> as(a_shapes.begin(), a_shapes.end());
             std::vector<PrimExpr> ad(a_strides.begin(), a_strides.end());
             std::vector<PrimExpr> bs(b_shapes.begin(), b_shapes.end());
             std::vector<PrimExpr> bd(b_strides.begin(), b_strides.end());
             cute::CuteAlgebraLayout a(as, ad);
             cute::CuteAlgebraLayout b(bs, bd);
             auto result = cute::logicalDivide(a, b, /*byMode=*/false);
             auto flatShape = cute::flattenExprTuple(result.shape);
             auto flatStride = cute::flattenExprTuple(result.stride);
             Array<PrimExpr> out_shapes(flatShape.begin(), flatShape.end());
             Array<PrimExpr> out_strides(flatStride.begin(), flatStride.end());
             return Array<Array<PrimExpr>>{out_shapes, out_strides};
           })
      .def("tl.ComputeContiguousTileSteps", [](Layout layout) {
        auto steps = ComputeContiguousTileSteps(layout);
        // Return as Array<Array<Integer>>: [[dim0, extent0], [dim1,
        // extent1], ...]
        Array<Array<Integer>> result;
        for (const auto &step : steps) {
          result.push_back({Integer(step.dim), Integer(step.extent)});
        }
        return result;
      });
}

} // namespace tl
} // namespace tvm
