/*!
 * \file tileview/tileview_planner.cc
 * \brief TileView planning helpers for T.Tiles scopes.
 */

#include "tileview_planner.h"

#include <algorithm>
#include <unordered_set>
#include <utility>

#include <tvm/arith/analyzer.h>
#include <tvm/arith/pattern.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../transform/common/attr.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

enum class LayoutClass {
  kBlockwise32x32,
  kRowMajor,
};

struct IndexBinding {
  bool uses_loop_var{false};
  int domain_axis{-1};
  PrimExpr offset{Integer(0)};
};

struct AccessTileCandidate {
  TileView tileview;
  std::vector<int> tiled_domain_axes;
  std::unordered_set<int> non_tiled_domain_axes;
  std::vector<int> tile_shape;
};

struct AccessInfo {
  Buffer buffer;
  Array<PrimExpr> indices;
  std::vector<IndexBinding> bindings;
  std::vector<AccessTileCandidate> candidates;
};

struct ExecutionPlanCandidate {
  std::vector<int> execution_domain_axes;
  std::vector<int> tile_shape;
};

class BufferAccessCollector : public StmtExprVisitor {
public:
  static std::vector<BufferAccessRecord> Collect(const Stmt &stmt) {
    BufferAccessCollector collector;
    collector(stmt);
    return std::move(collector.accesses_);
  }

private:
  void VisitExpr_(const BufferLoadNode *op) final {
    accesses_.push_back({op->buffer, op->indices, /*is_store=*/false});
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    accesses_.push_back({op->buffer, op->indices, /*is_store=*/true});
    StmtExprVisitor::VisitStmt_(op);
  }

  std::vector<BufferAccessRecord> accesses_;
};

int64_t GetStaticIntValue(const PrimExpr &expr, int64_t fallback = -1) {
  if (const auto *imm = expr.as<IntImmNode>()) {
    return imm->value;
  }
  return fallback;
}

LayoutClass GetLayoutClass(const Buffer &buffer,
                           const Map<Buffer, Layout> &layout_map) {
  return layout_map.count(buffer) ? LayoutClass::kBlockwise32x32
                                  : LayoutClass::kRowMajor;
}

int GetElementBits(const Buffer &buffer) {
  ICHECK_EQ(buffer->dtype.lanes(), 1)
      << "T.Tiles currently expects scalar element dtypes, but buffer "
      << buffer->name << " uses lanes=" << buffer->dtype.lanes() << ".";
  return buffer->dtype.bits();
}

int GetCapacityElems(const Buffer &buffer,
                     const SunmmioTileProcessorConfig &config) {
  int element_bits = GetElementBits(buffer);
  ICHECK_GT(element_bits, 0)
      << "T.Tiles requires a positive element bit-width for buffer "
      << buffer->name << ".";
  ICHECK_EQ(config.register_bits % element_bits, 0)
      << "Sunmmio tile register size " << config.register_bits
      << " is not divisible by element bit-width " << element_bits
      << " for buffer " << buffer->name << ".";
  return config.register_bits / element_bits;
}

int TileElements(const std::vector<int> &tile_shape) {
  int elems = 1;
  for (int extent : tile_shape) {
    elems *= extent;
  }
  return elems;
}

bool CanProveDivisible(arith::Analyzer *analyzer, const PrimExpr &value,
                       int factor) {
  PrimExpr remainder = analyzer->Simplify(floormod(value, Integer(factor)));
  return analyzer->CanProve(remainder == make_zero(remainder.dtype()));
}

void RequireDivisible(arith::Analyzer *analyzer, const PrimExpr &value,
                      int factor, const PrimExpr &index, const Buffer &buffer) {
  ICHECK(CanProveDivisible(analyzer, value, factor))
      << "Tile access offset " << value << " is not divisible by tile size "
      << factor << " in index " << index << " for buffer " << buffer->name
      << ".";
}

bool HasTrailingIndexMap(const TileView &tv, int exec_rank) {
  if (static_cast<int>(tv->TileDim()) != exec_rank) {
    return false;
  }

  int buf_ndim = static_cast<int>(tv->BufferShape().size());
  int first_exec_dim = buf_ndim - exec_rank;
  for (int i = 0; i < exec_rank; ++i) {
    const auto *imm = tv->IndexMap()[i].as<IntImmNode>();
    if (imm == nullptr) {
      return false;
    }
    int mapped_dim = static_cast<int>(imm->value);
    if (mapped_dim < 0) {
      mapped_dim += buf_ndim;
    }
    if (mapped_dim != first_exec_dim + i) {
      return false;
    }
  }
  return true;
}

int NormalizeMappedDim(const PrimExpr &expr, int ndim) {
  const auto *imm = expr.as<IntImmNode>();
  ICHECK(imm) << "TileView index_map entries must be IntImm, but got " << expr;
  int mapped_dim = static_cast<int>(imm->value);
  if (mapped_dim < 0) {
    mapped_dim += ndim;
  }
  ICHECK(mapped_dim >= 0 && mapped_dim < ndim)
      << "TileView index_map entry " << expr << " is out of bounds for rank "
      << ndim << ".";
  return mapped_dim;
}

IndexBinding
AnalyzeIndexBinding(const PrimExpr &index, const Array<Var> &loop_vars,
                    const std::unordered_set<const VarNode *> &loop_var_nodes,
                    arith::Analyzer *analyzer) {
  if (!UsesVar(index, [&loop_var_nodes](const VarNode *node) {
        return loop_var_nodes.count(node) != 0;
      })) {
    return {false, -1, analyzer->Simplify(index)};
  }

  Array<PrimExpr> coeffs = arith::DetectLinearEquation(index, loop_vars);
  ICHECK(!coeffs.empty())
      << "T.Tiles access index must be affine in the tile loop vars, but got "
      << index << ".";

  int matched_axis = -1;
  PrimExpr base = analyzer->Simplify(coeffs[coeffs.size() - 1]);
  for (int i = 0; i < static_cast<int>(loop_vars.size()); ++i) {
    PrimExpr coeff = analyzer->Simplify(coeffs[i]);
    PrimExpr zero = make_zero(coeff.dtype());
    PrimExpr one = make_const(coeff.dtype(), 1);
    if (analyzer->CanProve(coeff == zero)) {
      continue;
    }
    ICHECK(analyzer->CanProve(coeff == one))
        << "T.Tiles access index must use a tile loop var with unit "
           "coefficient, "
           "but got coefficient "
        << coeff << " in " << index << ".";
    ICHECK_EQ(matched_axis, -1) << "T.Tiles access index may depend on at most "
                                   "one tile loop var, but got "
                                << index << ".";
    matched_axis = i;
  }

  ICHECK_GE(matched_axis, 0) << "T.Tiles access index uses tile loop vars, but "
                                "no matching axis was found "
                                "for "
                             << index << ".";
  return {true, matched_axis, base};
}

std::unordered_set<int>
CollectNonTiledDomainAxes(const std::vector<IndexBinding> &bindings,
                          const std::vector<int> &tiled_dims) {
  std::unordered_set<int> tiled_dim_set(tiled_dims.begin(), tiled_dims.end());
  std::unordered_set<int> non_tiled_axes;
  for (int dim = 0; dim < static_cast<int>(bindings.size()); ++dim) {
    if (tiled_dim_set.count(dim) != 0) {
      continue;
    }
    const IndexBinding &binding = bindings[dim];
    if (binding.uses_loop_var) {
      non_tiled_axes.insert(binding.domain_axis);
    }
  }
  return non_tiled_axes;
}

bool SameIntVector(const std::vector<int> &lhs, const std::vector<int> &rhs) {
  return lhs == rhs;
}

void AddRank1Candidate(std::vector<AccessTileCandidate> *candidates,
                       const Buffer &buffer, const Array<PrimExpr> &indices,
                       const std::vector<IndexBinding> &bindings,
                       int mapped_dim, int tile_width,
                       arith::Analyzer *analyzer, bool strict_checks,
                       TileView tv = TileView()) {
  const IndexBinding &binding = bindings[mapped_dim];
  if (!binding.uses_loop_var) {
    ICHECK(!strict_checks)
        << "1D TileView inside T.Tiles must bind to a tile loop var for buffer "
        << buffer->name << ".";
    return;
  }

  if (!CanProveDivisible(analyzer, buffer->shape[mapped_dim], tile_width)) {
    ICHECK(!strict_checks) << "Buffer dimension " << mapped_dim << " of buffer "
                           << buffer->name << " is not divisible by tile width "
                           << tile_width << ".";
    return;
  }

  if (strict_checks) {
    RequireDivisible(analyzer, binding.offset, tile_width, indices[mapped_dim],
                     buffer);
  } else if (!CanProveDivisible(analyzer, binding.offset, tile_width)) {
    return;
  }

  if (!tv.defined()) {
    tv = makeTileView(
        buffer->shape, {Integer(tile_width)},
        {Integer(mapped_dim - static_cast<int>(buffer->shape.size()))});
  }

  candidates->push_back({tv,
                         {binding.domain_axis},
                         CollectNonTiledDomainAxes(bindings, {mapped_dim}),
                         {tile_width}});
}

void AddRank2Candidate(std::vector<AccessTileCandidate> *candidates,
                       const Buffer &buffer, const Array<PrimExpr> &indices,
                       const std::vector<IndexBinding> &bindings,
                       int mapped_height_dim, int mapped_width_dim,
                       int tile_height, int tile_width,
                       arith::Analyzer *analyzer, bool strict_checks,
                       TileView tv = TileView()) {
  const IndexBinding &height_binding = bindings[mapped_height_dim];
  const IndexBinding &width_binding = bindings[mapped_width_dim];

  if (!height_binding.uses_loop_var || !width_binding.uses_loop_var) {
    ICHECK(!strict_checks) << "Tiled buffer dimensions of buffer "
                           << buffer->name
                           << " must bind to tile loop vars inside T.Tiles.";
    return;
  }
  if (height_binding.domain_axis == width_binding.domain_axis) {
    ICHECK(!strict_checks) << "The tiled dimensions of buffer " << buffer->name
                           << " cannot bind to the same tile loop var.";
    return;
  }

  if (!CanProveDivisible(analyzer, buffer->shape[mapped_width_dim],
                         tile_width)) {
    ICHECK(!strict_checks) << "Buffer width dimension " << mapped_width_dim
                           << " of buffer " << buffer->name
                           << " is not divisible by tile width " << tile_width
                           << ".";
    return;
  }

  if (strict_checks) {
    RequireDivisible(analyzer, width_binding.offset, tile_width,
                     indices[mapped_width_dim], buffer);
  } else if (!CanProveDivisible(analyzer, width_binding.offset, tile_width)) {
    return;
  }

  if (strict_checks) {
    ICHECK(CanProveDivisible(analyzer, buffer->shape[mapped_height_dim],
                             tile_height))
        << "Buffer height dimension " << mapped_height_dim << " of buffer "
        << buffer->name << " is not divisible by tile height " << tile_height
        << ".";
  } else if (!CanProveDivisible(analyzer, buffer->shape[mapped_height_dim],
                                tile_height)) {
    return;
  }

  if (strict_checks) {
    RequireDivisible(analyzer, height_binding.offset, tile_height,
                     indices[mapped_height_dim], buffer);
  } else if (!CanProveDivisible(analyzer, height_binding.offset, tile_height)) {
    return;
  }

  if (!tv.defined()) {
    int ndim = static_cast<int>(buffer->shape.size());
    tv = makeTileView(
        buffer->shape, {Integer(tile_height), Integer(tile_width)},
        {Integer(mapped_height_dim - ndim), Integer(mapped_width_dim - ndim)});
  }

  candidates->push_back(
      {tv,
       {height_binding.domain_axis, width_binding.domain_axis},
       CollectNonTiledDomainAxes(bindings,
                                 {mapped_height_dim, mapped_width_dim}),
       {tile_height, tile_width}});
}

void EnumerateBlockwiseCandidates(std::vector<AccessTileCandidate> *candidates,
                                  const Buffer &buffer,
                                  const Array<PrimExpr> &indices,
                                  const std::vector<IndexBinding> &bindings,
                                  int exec_rank,
                                  const SunmmioTileProcessorConfig &config,
                                  arith::Analyzer *analyzer) {
  int ndim = static_cast<int>(buffer->shape.size());
  int capacity_elems = GetCapacityElems(buffer, config);

  if (ndim >= 1 && CanProveDivisible(analyzer, buffer->shape[ndim - 1],
                                     config.block_width)) {
    AddRank1Candidate(candidates, buffer, indices, bindings, ndim - 1,
                      config.block_width, analyzer, /*strict_checks=*/false);
  }

  if (exec_rank != 2 || ndim < 2) {
    return;
  }

  int max_height =
      std::min(config.block_height, capacity_elems / config.block_width);
  for (int tile_height = 1; tile_height <= max_height; ++tile_height) {
    AddRank2Candidate(candidates, buffer, indices, bindings, ndim - 2, ndim - 1,
                      tile_height, config.block_width, analyzer,
                      /*strict_checks=*/false);
  }
}

void EnumerateRowMajorCandidates(std::vector<AccessTileCandidate> *candidates,
                                 const Buffer &buffer,
                                 const Array<PrimExpr> &indices,
                                 const std::vector<IndexBinding> &bindings,
                                 int exec_rank,
                                 const SunmmioTileProcessorConfig &config,
                                 arith::Analyzer *analyzer) {
  int ndim = static_cast<int>(buffer->shape.size());
  int capacity_elems = GetCapacityElems(buffer, config);

  if (ndim >= 1) {
    for (int tile_width = 1; tile_width <= capacity_elems; ++tile_width) {
      AddRank1Candidate(candidates, buffer, indices, bindings, ndim - 1,
                        tile_width, analyzer, /*strict_checks=*/false);
    }
  }

  if (exec_rank != 2 || ndim < 2) {
    return;
  }

  int64_t row_width = GetStaticIntValue(buffer->shape[ndim - 1]);
  if (row_width <= 0) {
    return;
  }

  // A single-row contiguous slice is always legal in row-major layout as long
  // as its width fits both the physical row and the register capacity.
  int max_single_row_width =
      std::min(capacity_elems, static_cast<int>(row_width));
  for (int tile_width = 1; tile_width <= max_single_row_width; ++tile_width) {
    AddRank2Candidate(candidates, buffer, indices, bindings, ndim - 2, ndim - 1,
                      /*tile_height=*/1, tile_width, analyzer,
                      /*strict_checks=*/false);
  }

  // Multi-row row-major tiles remain conservative for now: they must cover a
  // full physical row so the whole tile stays one contiguous interval.
  if (row_width > capacity_elems) {
    return;
  }

  int max_height = capacity_elems / static_cast<int>(row_width);
  for (int tile_height = 2; tile_height <= max_height; ++tile_height) {
    AddRank2Candidate(candidates, buffer, indices, bindings, ndim - 2, ndim - 1,
                      tile_height, static_cast<int>(row_width), analyzer,
                      /*strict_checks=*/false);
  }
}

std::vector<AccessTileCandidate> EnumerateManualCandidates(
    const Buffer &buffer, const Array<PrimExpr> &indices,
    const std::vector<IndexBinding> &bindings, const TileView &manual_tv,
    int exec_rank, const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &config, arith::Analyzer *analyzer) {
  std::vector<AccessTileCandidate> candidates;
  int tv_rank = static_cast<int>(manual_tv->TileDim());
  int ndim = static_cast<int>(indices.size());
  int capacity_elems = GetCapacityElems(buffer, config);
  LayoutClass layout_class = GetLayoutClass(buffer, layout_map);

  if (exec_rank == 1) {
    ICHECK_EQ(tv_rank, 1) << "1D T.Tiles domain is incompatible with buffer "
                          << buffer->name << ", which requires a " << tv_rank
                          << "D TileView.";
  } else {
    ICHECK(tv_rank == 1 || tv_rank == 2)
        << "T.Tiles currently supports only 1D or trailing-2D execution "
           "TileViews, but buffer "
        << buffer->name << " uses rank " << tv_rank << ".";
  }

  if (tv_rank == 1) {
    ICHECK(HasTrailingIndexMap(manual_tv, /*exec_rank=*/1))
        << "1D TileView inside T.Tiles must target the trailing buffer "
           "dimension "
           "for buffer "
        << buffer->name << ".";
    int manual_width = GetStaticIntValue(manual_tv->TileShape()[0]);
    ICHECK_GT(manual_width, 0)
        << "1D TileView inside T.Tiles must use a positive static tile width "
           "for buffer "
        << buffer->name << ".";

    if (layout_class == LayoutClass::kBlockwise32x32 && ndim >= 2) {
      ICHECK_EQ(manual_width, config.block_width)
          << "Blockwise TileView inside T.Tiles must use trailing width "
          << config.block_width << " for buffer " << buffer->name << ".";
    } else {
      ICHECK_LE(manual_width, capacity_elems)
          << "Manual TileView width " << manual_width
          << " exceeds the Sunmmio register capacity of " << capacity_elems
          << " elements for buffer " << buffer->name << ".";
    }

    int mapped_dim = NormalizeMappedDim(manual_tv->IndexMap()[0], ndim);
    AddRank1Candidate(&candidates, buffer, indices, bindings, mapped_dim,
                      manual_width, analyzer, /*strict_checks=*/true,
                      manual_tv);
    return candidates;
  }

  ICHECK(HasTrailingIndexMap(manual_tv, /*exec_rank=*/2))
      << "2D TileView inside T.Tiles must target the trailing two buffer "
         "dimensions for buffer "
      << buffer->name << ".";
  int manual_height = GetStaticIntValue(manual_tv->TileShape()[0]);
  int manual_width = GetStaticIntValue(manual_tv->TileShape()[1]);
  ICHECK_GT(manual_height, 0)
      << "2D TileView height inside T.Tiles must be a positive static integer "
         "for buffer "
      << buffer->name << ".";
  ICHECK_GT(manual_width, 0)
      << "2D TileView width inside T.Tiles must be a positive static integer "
         "for buffer "
      << buffer->name << ".";

  if (layout_class == LayoutClass::kBlockwise32x32) {
    ICHECK_EQ(manual_width, config.block_width)
        << "Blockwise TileView inside T.Tiles must use trailing width "
        << config.block_width << " for buffer " << buffer->name << ".";
    ICHECK_LE(manual_height * manual_width, capacity_elems)
        << "Manual TileView shape (" << manual_height << ", " << manual_width
        << ") exceeds the Sunmmio register capacity of " << capacity_elems
        << " elements for buffer " << buffer->name << ".";
    ICHECK_LE(manual_height, config.block_height)
        << "Blockwise TileView height " << manual_height
        << " exceeds the modeled block height " << config.block_height
        << " for buffer " << buffer->name << ".";
  } else {
    int64_t row_width = GetStaticIntValue(buffer->shape[ndim - 1]);
    ICHECK_GT(row_width, 0)
        << "Manual row-major rank-2 TileView requires a static trailing row "
           "width for buffer "
        << buffer->name << ".";
    ICHECK_LE(manual_width, row_width)
        << "Manual row-major rank-2 TileView width " << manual_width
        << " exceeds trailing buffer dimension " << row_width << " for buffer "
        << buffer->name << ".";
    if (manual_height > 1) {
      ICHECK_EQ(manual_width, row_width)
          << "Manual multi-row row-major rank-2 TileView width must match the "
             "trailing buffer dimension "
          << row_width << " for buffer " << buffer->name << ".";
    }
    ICHECK_LE(manual_height * manual_width, capacity_elems)
        << "Manual TileView shape (" << manual_height << ", " << manual_width
        << ") exceeds the Sunmmio register capacity of " << capacity_elems
        << " elements for buffer " << buffer->name << ".";
  }

  int mapped_height_dim = NormalizeMappedDim(manual_tv->IndexMap()[0], ndim);
  int mapped_width_dim = NormalizeMappedDim(manual_tv->IndexMap()[1], ndim);
  AddRank2Candidate(&candidates, buffer, indices, bindings, mapped_height_dim,
                    mapped_width_dim, manual_height, manual_width, analyzer,
                    /*strict_checks=*/true, manual_tv);
  return candidates;
}

std::vector<AccessTileCandidate> EnumerateInferredCandidates(
    const Buffer &buffer, const Array<PrimExpr> &indices,
    const std::vector<IndexBinding> &bindings, int exec_rank,
    const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &config, arith::Analyzer *analyzer) {
  std::vector<AccessTileCandidate> candidates;
  if (GetLayoutClass(buffer, layout_map) == LayoutClass::kBlockwise32x32) {
    EnumerateBlockwiseCandidates(&candidates, buffer, indices, bindings,
                                 exec_rank, config, analyzer);
  } else {
    EnumerateRowMajorCandidates(&candidates, buffer, indices, bindings,
                                exec_rank, config, analyzer);
  }
  return candidates;
}

std::vector<AccessTileCandidate> EnumerateAccessTileCandidates(
    const BufferAccessRecord &access, const std::vector<IndexBinding> &bindings,
    int exec_rank, const TileViewMap &manual_tileviews,
    const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &config, arith::Analyzer *analyzer) {
  auto manual_it = manual_tileviews.find(access.buffer->data);
  if (manual_it != manual_tileviews.end()) {
    return EnumerateManualCandidates(access.buffer, access.indices, bindings,
                                     manual_it->second, exec_rank, layout_map,
                                     config, analyzer);
  }

  return EnumerateInferredCandidates(access.buffer, access.indices, bindings,
                                     exec_rank, layout_map, config, analyzer);
}

bool UsesExecutionAxisInNonTiledDims(const AccessTileCandidate &candidate,
                                     const std::vector<int> &exec_domain_axes) {
  for (int exec_domain_axis : exec_domain_axes) {
    if (candidate.non_tiled_domain_axes.count(exec_domain_axis) != 0) {
      return true;
    }
  }
  return false;
}

bool Supports1DPlan(const AccessTileCandidate &candidate) {
  return candidate.tiled_domain_axes.size() == 1 &&
         candidate.tile_shape.size() == 1 &&
         candidate.tiled_domain_axes[0] == 0 &&
         !UsesExecutionAxisInNonTiledDims(candidate, {0});
}

bool Supports1DPlan(const AccessTileCandidate &candidate, int tile_extent) {
  return Supports1DPlan(candidate) && candidate.tile_shape[0] == tile_extent;
}

bool Supports2DPlan(const AccessTileCandidate &candidate,
                    const ExecutionPlanCandidate &plan) {
  if (UsesExecutionAxisInNonTiledDims(candidate, plan.execution_domain_axes)) {
    return false;
  }

  if (candidate.tile_shape.size() == 2) {
    return candidate.tiled_domain_axes.size() == 2 &&
           candidate.tiled_domain_axes[0] == plan.execution_domain_axes[0] &&
           candidate.tiled_domain_axes[1] == plan.execution_domain_axes[1] &&
           SameIntVector(candidate.tile_shape, plan.tile_shape);
  }

  if (candidate.tile_shape.size() != 1 ||
      candidate.tiled_domain_axes.size() != 1) {
    return false;
  }

  int bound_axis = candidate.tiled_domain_axes[0];
  if (bound_axis == plan.execution_domain_axes[0]) {
    return candidate.tile_shape[0] == plan.tile_shape[0];
  }
  if (bound_axis == plan.execution_domain_axes[1]) {
    return candidate.tile_shape[0] == plan.tile_shape[1];
  }
  return false;
}

std::vector<ExecutionPlanCandidate>
CollectRank2PlanCandidates(const AccessInfo &access) {
  std::vector<ExecutionPlanCandidate> plans;
  for (const auto &candidate : access.candidates) {
    if (candidate.tile_shape.size() != 2) {
      continue;
    }
    bool exists = std::any_of(
        plans.begin(), plans.end(), [&](const ExecutionPlanCandidate &plan) {
          return SameIntVector(plan.execution_domain_axes,
                               candidate.tiled_domain_axes) &&
                 SameIntVector(plan.tile_shape, candidate.tile_shape);
        });
    if (!exists) {
      plans.push_back({candidate.tiled_domain_axes, candidate.tile_shape});
    }
  }
  return plans;
}

} // namespace

std::vector<BufferAccessRecord> CollectBufferAccesses(const Stmt &stmt) {
  return BufferAccessCollector::Collect(stmt);
}

TileViewPlan PlanTileViewsForTilesScope(
    const Array<PrimExpr> &domain,
    const std::vector<const ForNode *> &scope_loops,
    const std::vector<BufferAccessRecord> &accesses,
    const TileViewMap &manual_tileviews, const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &tile_processor_config) {
  ICHECK(!domain.empty()) << "T.Tiles domain must be non-empty.";
  ICHECK_EQ(domain.size(), scope_loops.size())
      << "T.Tiles scope loop rank does not match the declared domain rank.";
  ICHECK(!accesses.empty()) << "T.Tiles scope must access at least one buffer.";

  int domain_rank = static_cast<int>(domain.size());
  int exec_rank = domain_rank == 1 ? 1 : 2;

  arith::Analyzer analyzer;

  // Phase 1: materialize the raw T.Tiles loop vars so each access index can
  // be analyzed as an affine expression over the logical tile domain.
  Array<Var> loop_vars;
  std::unordered_set<const VarNode *> loop_var_nodes;
  for (const ForNode *loop : scope_loops) {
    loop_vars.push_back(loop->loop_var);
    loop_var_nodes.insert(loop->loop_var.get());
  }

  // Phase 2: analyze each access and enumerate the full set of feasible
  // tileview candidates implied by that access pattern.
  std::vector<AccessInfo> analyzed_accesses;
  analyzed_accesses.reserve(accesses.size());
  for (const auto &access : accesses) {
    std::vector<IndexBinding> bindings;
    bindings.reserve(access.indices.size());
    for (const PrimExpr &index : access.indices) {
      bindings.push_back(
          AnalyzeIndexBinding(index, loop_vars, loop_var_nodes, &analyzer));
    }

    std::vector<AccessTileCandidate> candidates = EnumerateAccessTileCandidates(
        access, bindings, exec_rank, manual_tileviews, layout_map,
        tile_processor_config, &analyzer);
    ICHECK(!candidates.empty())
        << "Cannot infer any feasible TileView candidate for access to buffer "
        << access.buffer->name << " with indices " << access.indices << ".";
    analyzed_accesses.push_back({access.buffer, access.indices,
                                 std::move(bindings), std::move(candidates)});
  }

  // Phase 3: solve the common execution plan by intersecting the feasible
  // candidate set of every access. For 1D domains we choose the largest common
  // contiguous segment. For 2D+ domains we intersect the exact rank-2 plan
  // candidates and pick the densest legal execution tile.
  if (exec_rank == 1) {
    std::vector<int> plan_extents;
    for (const auto &access : analyzed_accesses) {
      for (const auto &candidate : access.candidates) {
        if (Supports1DPlan(candidate)) {
          plan_extents.push_back(candidate.tile_shape[0]);
        }
      }
    }

    ICHECK(!plan_extents.empty())
        << "Cannot infer any feasible 1D execution tileview for T.Tiles domain "
        << domain << ".";

    std::sort(plan_extents.begin(), plan_extents.end());
    plan_extents.erase(std::unique(plan_extents.begin(), plan_extents.end()),
                       plan_extents.end());
    std::sort(plan_extents.begin(), plan_extents.end(), std::greater<int>());

    for (int tile_extent : plan_extents) {
      bool all_supported = true;
      for (const auto &access : analyzed_accesses) {
        bool access_supported =
            std::any_of(access.candidates.begin(), access.candidates.end(),
                        [&](const AccessTileCandidate &candidate) {
                          return Supports1DPlan(candidate, tile_extent);
                        });
        if (!access_supported) {
          all_supported = false;
          break;
        }
      }

      if (all_supported) {
        return {makeTileView(domain, {Integer(tile_extent)}, {Integer(-1)}),
                {0}};
      }
    }

    LOG(FATAL) << "Cannot infer a common 1D execution tileview for T.Tiles "
               << "domain " << domain
               << ". The per-access feasible tileview sets do not intersect.";
    return {TileView(), {}};
  }

  std::vector<ExecutionPlanCandidate> plan_candidates;
  bool saw_rank2_candidates = false;
  for (const auto &access : analyzed_accesses) {
    std::vector<ExecutionPlanCandidate> access_plan_candidates =
        CollectRank2PlanCandidates(access);
    if (access_plan_candidates.empty()) {
      continue;
    }

    if (!saw_rank2_candidates) {
      plan_candidates = std::move(access_plan_candidates);
      saw_rank2_candidates = true;
      continue;
    }

    std::vector<ExecutionPlanCandidate> common_plan_candidates;
    for (const auto &plan_candidate : plan_candidates) {
      bool supported = std::any_of(
          access_plan_candidates.begin(), access_plan_candidates.end(),
          [&](const ExecutionPlanCandidate &access_plan_candidate) {
            return SameIntVector(plan_candidate.execution_domain_axes,
                                 access_plan_candidate.execution_domain_axes) &&
                   SameIntVector(plan_candidate.tile_shape,
                                 access_plan_candidate.tile_shape);
          });
      if (supported) {
        common_plan_candidates.push_back(plan_candidate);
      }
    }
    plan_candidates = std::move(common_plan_candidates);
  }

  ICHECK(saw_rank2_candidates)
      << "2D T.Tiles requires at least one access with a feasible rank-2 "
         "TileView candidate.";
  ICHECK(!plan_candidates.empty())
      << "Cannot infer a common execution tileview for T.Tiles domain "
      << domain
      << ". The rank-2 access candidates do not share a common axis binding "
         "and tile shape.";

  std::sort(
      plan_candidates.begin(), plan_candidates.end(),
      [](const ExecutionPlanCandidate &lhs, const ExecutionPlanCandidate &rhs) {
        int lhs_elems = TileElements(lhs.tile_shape);
        int rhs_elems = TileElements(rhs.tile_shape);
        if (lhs_elems != rhs_elems) {
          return lhs_elems > rhs_elems;
        }
        if (lhs.tile_shape[0] != rhs.tile_shape[0]) {
          return lhs.tile_shape[0] > rhs.tile_shape[0];
        }
        if (lhs.tile_shape[1] != rhs.tile_shape[1]) {
          return lhs.tile_shape[1] > rhs.tile_shape[1];
        }
        return lhs.execution_domain_axes < rhs.execution_domain_axes;
      });

  for (const auto &plan_candidate : plan_candidates) {
    const auto &exec_domain_axes = plan_candidate.execution_domain_axes;
    int tile_height = plan_candidate.tile_shape[0];
    int tile_width = plan_candidate.tile_shape[1];

    if (!CanProveDivisible(&analyzer, domain[exec_domain_axes[0]],
                           tile_height) ||
        !CanProveDivisible(&analyzer, domain[exec_domain_axes[1]],
                           tile_width)) {
      continue;
    }

    bool all_supported = true;
    for (const auto &access : analyzed_accesses) {
      bool access_supported =
          std::any_of(access.candidates.begin(), access.candidates.end(),
                      [&](const AccessTileCandidate &candidate) {
                        return Supports2DPlan(candidate, plan_candidate);
                      });
      if (!access_supported) {
        all_supported = false;
        break;
      }
    }

    if (all_supported) {
      return {makeTileView(domain, {Integer(tile_height), Integer(tile_width)},
                           {Integer(exec_domain_axes[0] - domain_rank),
                            Integer(exec_domain_axes[1] - domain_rank)}),
              exec_domain_axes};
    }
  }

  LOG(FATAL) << "Cannot infer a common execution tileview for T.Tiles domain "
             << domain
             << ". The per-access feasible tileview sets do not intersect.";
  return {TileView(), {}};
}

} // namespace tl
} // namespace tvm
