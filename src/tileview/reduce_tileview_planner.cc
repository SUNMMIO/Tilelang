/*!
 * \file tileview/reduce_tileview_planner.cc
 * \brief TileView planning helpers specialized for Sunmmio reductions.
 */

#include "reduce_tileview_planner.h"

#include <algorithm>
#include <utility>

#include <tvm/runtime/logging.h>
#include <tvm/tir/op.h>

#include "tileview_planner_common.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

struct ReduceTileCandidate {
  TileView src_tileview;
  TileView dst_tileview;
  std::vector<int> execution_domain_axes;
  int reduce_tile_axis{-1};
  int reduce_tile_extent{1}; // tile extent on the reduce dim (1 if not tiled)
  int src_tile_elems{1};
  int dst_tile_elems{1};
};

Array<PrimExpr> RegionExtents(const BufferRegion &region) {
  Array<PrimExpr> extents;
  extents.reserve(region->region.size());
  for (const Range &range : region->region) {
    extents.push_back(range->extent);
  }
  return extents;
}

std::vector<int> BuildSrcDimToDstDim(int src_rank, int dst_rank,
                                     int reduce_dim) {
  std::vector<int> mapping(src_rank, -1);
  if (dst_rank == src_rank) {
    for (int i = 0; i < src_rank; ++i) {
      mapping[i] = (i == reduce_dim) ? -1 : i;
    }
    return mapping;
  }

  ICHECK_EQ(dst_rank, src_rank - 1)
      << "Sunmmio reduction expects dst rank to equal src rank or src rank - "
         "1, but got src rank "
      << src_rank << " and dst rank " << dst_rank << ".";
  for (int i = 0; i < src_rank; ++i) {
    if (i == reduce_dim) {
      mapping[i] = -1;
    } else if (i < reduce_dim) {
      mapping[i] = i;
    } else {
      mapping[i] = i - 1;
    }
  }
  return mapping;
}

void ValidateReduceRegions(const BufferRegion &src_region,
                           const BufferRegion &dst_region, int reduce_dim,
                           arith::Analyzer *analyzer) {
  int src_rank = static_cast<int>(src_region->region.size());
  int dst_rank = static_cast<int>(dst_region->region.size());
  ICHECK(reduce_dim >= 0 && reduce_dim < src_rank)
      << "Reduction axis " << reduce_dim << " is out of bounds for src rank "
      << src_rank << ".";
  ICHECK(dst_rank == src_rank || dst_rank == src_rank - 1)
      << "Sunmmio reduction expects dst rank to equal src rank or src rank - "
         "1, but got src rank "
      << src_rank << " and dst rank " << dst_rank << ".";

  std::vector<int> src_dim_to_dst_dim =
      BuildSrcDimToDstDim(src_rank, dst_rank, reduce_dim);
  for (int src_dim = 0; src_dim < src_rank; ++src_dim) {
    int dst_dim = src_dim_to_dst_dim[src_dim];
    if (dst_dim < 0) {
      if (dst_rank == src_rank) {
        ICHECK(is_one(analyzer->Simplify(dst_region->region[src_dim]->extent)))
            << "keepdim reduction expects dst extent 1 on reduced dim "
            << reduce_dim << ", but got " << dst_region->region[src_dim]->extent
            << ".";
      }
      continue;
    }

    ICHECK(analyzer->CanProveEqual(src_region->region[src_dim]->extent,
                                   dst_region->region[dst_dim]->extent))
        << "Reduction src/dst region extents must match on surviving dims, "
           "but src dim "
        << src_dim << " has extent " << src_region->region[src_dim]->extent
        << " while dst dim " << dst_dim << " has extent "
        << dst_region->region[dst_dim]->extent << ".";
  }
}

bool IsCompatibleProjectedTileView(const Optional<TileView> &maybe_manual_tv,
                                   const TileView &projected_tv,
                                   arith::Analyzer *analyzer) {
  if (!maybe_manual_tv.defined()) {
    return true;
  }

  TileView manual_tv = maybe_manual_tv.value();
  int manual_rank = static_cast<int>(manual_tv->TileDim());
  int projected_rank = static_cast<int>(projected_tv->TileDim());
  if (manual_rank != projected_rank) {
    return false;
  }

  int manual_buf_rank = static_cast<int>(manual_tv->BufferShape().size());
  int projected_buf_rank = static_cast<int>(projected_tv->BufferShape().size());
  if (manual_buf_rank != projected_buf_rank) {
    return false;
  }

  for (int i = 0; i < manual_rank; ++i) {
    if (!analyzer->CanProveEqual(manual_tv->TileShape()[i],
                                 projected_tv->TileShape()[i])) {
      return false;
    }
    if (NormalizeMappedDim(manual_tv->IndexMap()[i], manual_buf_rank) !=
        NormalizeMappedDim(projected_tv->IndexMap()[i], projected_buf_rank)) {
      return false;
    }
  }
  return true;
}

ReduceTileCandidate MakeCandidate(const Array<PrimExpr> &source_domain,
                                  const Array<PrimExpr> &dst_domain,
                                  const std::vector<int> &src_dim_to_dst_dim,
                                  int reduce_dim,
                                  const TrailingTilePattern &src_pattern,
                                  arith::Analyzer *analyzer) {
  int dst_rank = static_cast<int>(dst_domain.size());
  int tile_rank = static_cast<int>(src_pattern.tile_shape.size());
  Array<PrimExpr> src_tile_shape = MakeTileShapeExpr(src_pattern.tile_shape);

  ReduceTileCandidate candidate;
  candidate.execution_domain_axes = src_pattern.mapped_dims;
  candidate.src_tileview = MakeTrailingTileView(source_domain, src_pattern);
  candidate.src_tile_elems = TileElements(src_pattern.tile_shape);

  Array<PrimExpr> dst_tile_shape;
  std::vector<int> dst_exec_axes;
  for (int axis = 0; axis < tile_rank; ++axis) {
    int src_dim = src_pattern.mapped_dims[axis];
    if (src_dim == reduce_dim) {
      candidate.reduce_tile_axis = axis;
      candidate.reduce_tile_extent = src_pattern.tile_shape[axis];
      continue;
    }
    int dst_dim = src_dim_to_dst_dim[src_dim];
    ICHECK_GE(dst_dim, 0) << "Surviving tiled source dim " << src_dim
                          << " does not map to any destination dim.";
    dst_exec_axes.push_back(dst_dim);
    dst_tile_shape.push_back(src_tile_shape[axis]);
  }

  candidate.dst_tileview = makeTileView(
      dst_domain, dst_tile_shape,
      MakeCanonicalIndexMap(dst_rank, static_cast<int>(dst_exec_axes.size())));
  candidate.dst_tile_elems = TileElements(dst_tile_shape);
  return candidate;
}

TrailingTilePattern ValidateManualSrcTilePattern(
    const BufferRegion &src_region, const TileView &manual_tv,
    const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &config, arith::Analyzer *analyzer) {
  int src_rank = static_cast<int>(src_region->region.size());
  TrailingTilePattern pattern = ValidateManualTrailingTileView(
      src_region->buffer, manual_tv, src_rank == 1 ? 1 : 2, layout_map, config,
      analyzer, "Manual src TileView for Sunmmio reduction");

  for (size_t axis = 0; axis < pattern.tile_shape.size(); ++axis) {
    int src_dim = pattern.mapped_dims[axis];
    int tile_extent = pattern.tile_shape[axis];
    ICHECK(CanProveDivisible(analyzer, src_region->region[src_dim]->min,
                             tile_extent))
        << "Manual src TileView extent " << tile_extent
        << " requires the source region offset on dim " << src_dim
        << " to be aligned, but got " << src_region->region[src_dim]->min
        << ".";
  }
  return pattern;
}

std::vector<ReduceTileCandidate> EnumerateInferredCandidates(
    const BufferRegion &src_region, const Array<PrimExpr> &source_domain,
    const Array<PrimExpr> &dst_domain,
    const std::vector<int> &src_dim_to_dst_dim, int reduce_dim,
    const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &config, arith::Analyzer *analyzer) {
  std::vector<ReduceTileCandidate> candidates;
  int exec_rank = static_cast<int>(source_domain.size()) == 1 ? 1 : 2;
  for (const TrailingTilePattern &pattern :
       EnumerateInferredTrailingTilePatterns(src_region->buffer, exec_rank,
                                             layout_map, config, analyzer)) {
    bool aligned = true;
    for (size_t axis = 0; axis < pattern.tile_shape.size(); ++axis) {
      int src_dim = pattern.mapped_dims[axis];
      int tile_extent = pattern.tile_shape[axis];
      if (!CanProveDivisible(analyzer, src_region->region[src_dim]->min,
                             tile_extent)) {
        aligned = false;
        break;
      }
    }
    if (!aligned) {
      continue;
    }

    candidates.push_back(MakeCandidate(source_domain, dst_domain,
                                       src_dim_to_dst_dim, reduce_dim, pattern,
                                       analyzer));
  }
  return candidates;
}

std::vector<ReduceTileCandidate> EnumerateManualCandidates(
    const BufferRegion &src_region, const BufferRegion &dst_region,
    int reduce_dim, const TileView &manual_tv,
    const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &config, arith::Analyzer *analyzer) {
  Array<PrimExpr> source_domain = RegionExtents(src_region);
  Array<PrimExpr> dst_domain = RegionExtents(dst_region);
  int src_rank = static_cast<int>(source_domain.size());
  std::vector<int> src_dim_to_dst_dim = BuildSrcDimToDstDim(
      src_rank, static_cast<int>(dst_domain.size()), reduce_dim);
  TrailingTilePattern pattern = ValidateManualSrcTilePattern(
      src_region, manual_tv, layout_map, config, analyzer);
  return {MakeCandidate(source_domain, dst_domain, src_dim_to_dst_dim,
                        reduce_dim, pattern, analyzer)};
}

} // namespace

ReduceTileViewPlan
PlanReduceTileViews(const BufferRegion &src_region,
                    const BufferRegion &dst_region, int reduce_dim,
                    const ReduceTileViewHints &hints,
                    const Map<Buffer, Layout> &layout_map,
                    const SunmmioTileProcessorConfig &tile_processor_config,
                    arith::Analyzer *analyzer) {
  ICHECK(analyzer != nullptr)
      << "Reduction TileView planning requires a valid analyzer.";

  ValidateReduceRegions(src_region, dst_region, reduce_dim, analyzer);

  Array<PrimExpr> source_domain = RegionExtents(src_region);
  Array<PrimExpr> dst_domain = RegionExtents(dst_region);
  int src_rank = static_cast<int>(source_domain.size());
  int dst_rank = static_cast<int>(dst_domain.size());
  std::vector<int> src_dim_to_dst_dim =
      BuildSrcDimToDstDim(src_rank, dst_rank, reduce_dim);

  std::vector<ReduceTileCandidate> candidates;
  if (hints.src_tileview.defined()) {
    candidates = EnumerateManualCandidates(
        src_region, dst_region, reduce_dim, hints.src_tileview.value(),
        layout_map, tile_processor_config, analyzer);
  } else {
    candidates = EnumerateInferredCandidates(
        src_region, source_domain, dst_domain, src_dim_to_dst_dim, reduce_dim,
        layout_map, tile_processor_config, analyzer);
  }

  std::vector<ReduceTileCandidate> compatible_candidates;
  compatible_candidates.reserve(candidates.size());
  for (const ReduceTileCandidate &candidate : candidates) {
    if (IsCompatibleProjectedTileView(hints.dst_tileview,
                                      candidate.dst_tileview, analyzer)) {
      compatible_candidates.push_back(candidate);
    }
  }

  ICHECK(!compatible_candidates.empty())
      << "Cannot infer a legal Sunmmio reduction TileView plan for src buffer "
      << src_region->buffer->name << " and dst buffer "
      << dst_region->buffer->name
      << ". The source candidates are incompatible with the reduction "
         "projection and any manual dst TileView hint.";

  // Score candidates to minimize total tile-unit dispatches.
  //
  // The Sunmmio reduction algorithm (see reduce.cc MakeSunmmioTileReduce)
  // accumulates element-wise across K/t_K iterations, then calls a single
  // hardware in-tile reduction at the end of each spatial position:
  //
  //   N_total = (S_spatial / dst_tile_elems) * (ceildiv(K, t_K) + 1)
  //
  // where t_K = reduce_tile_extent (1 if reduce dim is not tiled).
  // Since S_spatial is constant across candidates, the ranking is
  // determined by the score:
  //
  //   score = (ceildiv(K, t_K) + 1) / dst_tile_elems     (lower is better)
  //
  // When K is statically known (IntImm), we compare scores exactly via
  // cross-multiplication.  When K is dynamic, we fall back to a proxy:
  //   1. src_tile_elems ↑  (dominates for large K)
  //   2. reduce_tile_extent ↓  (breaks ties; fewer hardware reductions)
  //
  // Tiebreaks: simpler tile rank, then deterministic axis ordering.

  // Extract static K from source_domain[reduce_dim] if possible.
  int64_t K_static = -1;
  {
    const auto *imm = source_domain[reduce_dim].as<IntImmNode>();
    if (imm) {
      K_static = imm->value;
    }
  }

  std::sort(
      compatible_candidates.begin(), compatible_candidates.end(),
      [K_static](const ReduceTileCandidate &a, const ReduceTileCandidate &b) {
        if (K_static > 0) {
          // Static K: exact score comparison.
          // a better iff (ceildiv(K, a.t_K) + 1) / a.dst <
          //              (ceildiv(K, b.t_K) + 1) / b.dst
          int64_t lhs_steps =
              (K_static + a.reduce_tile_extent - 1) / a.reduce_tile_extent + 1;
          int64_t rhs_steps =
              (K_static + b.reduce_tile_extent - 1) / b.reduce_tile_extent + 1;
          int64_t lhs = lhs_steps * static_cast<int64_t>(b.dst_tile_elems);
          int64_t rhs = rhs_steps * static_cast<int64_t>(a.dst_tile_elems);
          if (lhs != rhs)
            return lhs < rhs;
        } else {
          // Dynamic K: proxy ordering.
          if (a.src_tile_elems != b.src_tile_elems)
            return a.src_tile_elems > b.src_tile_elems;
          if (a.reduce_tile_extent != b.reduce_tile_extent)
            return a.reduce_tile_extent < b.reduce_tile_extent;
        }
        // Deterministic tiebreaks.
        if (a.src_tileview->TileDim() != b.src_tileview->TileDim())
          return a.src_tileview->TileDim() < b.src_tileview->TileDim();
        return a.execution_domain_axes < b.execution_domain_axes;
      });

  const ReduceTileCandidate &best = compatible_candidates.front();
  return {source_domain,      best.src_tileview,
          best.dst_tileview,  best.execution_domain_axes,
          src_dim_to_dst_dim, best.reduce_tile_axis};
}

} // namespace tl
} // namespace tvm
