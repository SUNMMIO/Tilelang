/*!
 * \file tileview/tileview_planner.cc
 * \brief TileView planning helpers for T.Tiles scopes.
 */

#include "tileview_planner.h"

#include <algorithm>
#include <optional>
#include <unordered_set>
#include <utility>

#include <tvm/arith/analyzer.h>
#include <tvm/arith/pattern.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../transform/common/attr.h"
#include "tileview_planner_common.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

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
  std::vector<AccessTileCandidate> strict_candidates;
  std::vector<AccessTileCandidate> relaxed_candidates;
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
           "coefficient, but got coefficient "
        << coeff << " in " << index << ".";
    ICHECK_EQ(matched_axis, -1)
        << "T.Tiles access index may depend on at most one tile loop var, but "
        << "got " << index << ".";
    matched_axis = i;
  }

  ICHECK_GE(matched_axis, 0)
      << "T.Tiles access index uses tile loop vars, but no matching axis was "
      << "found for " << index << ".";
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

  if (strict_checks) {
    RequireDivisible(analyzer, width_binding.offset, tile_width,
                     indices[mapped_width_dim], buffer);
  } else if (!CanProveDivisible(analyzer, width_binding.offset, tile_width)) {
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

void AddCandidateFromPattern(std::vector<AccessTileCandidate> *candidates,
                             const Buffer &buffer,
                             const Array<PrimExpr> &indices,
                             const std::vector<IndexBinding> &bindings,
                             const TrailingTilePattern &pattern,
                             arith::Analyzer *analyzer, bool strict_checks) {
  TileView tv = MakeTrailingTileView(buffer->shape, pattern);
  if (pattern.tile_shape.size() == 1) {
    AddRank1Candidate(candidates, buffer, indices, bindings,
                      pattern.mapped_dims[0], pattern.tile_shape[0], analyzer,
                      strict_checks, tv);
    return;
  }

  ICHECK_EQ(pattern.tile_shape.size(), 2)
      << "T.Tiles currently supports only 1D or 2D trailing TileView patterns.";
  AddRank2Candidate(candidates, buffer, indices, bindings,
                    pattern.mapped_dims[0], pattern.mapped_dims[1],
                    pattern.tile_shape[0], pattern.tile_shape[1], analyzer,
                    strict_checks, tv);
}

std::vector<AccessTileCandidate> EnumerateManualCandidates(
    const Buffer &buffer, const Array<PrimExpr> &indices,
    const std::vector<IndexBinding> &bindings, const TileView &manual_tv,
    int exec_rank, const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &config, arith::Analyzer *analyzer) {
  std::vector<AccessTileCandidate> candidates;
  TrailingTilePattern pattern = ValidateManualTrailingTileView(
      buffer, manual_tv, exec_rank, layout_map, config, analyzer,
      "TileView inside T.Tiles");
  AddCandidateFromPattern(&candidates, buffer, indices, bindings, pattern,
                          analyzer, /*strict_checks=*/true);
  return candidates;
}

std::vector<AccessTileCandidate> EnumerateInferredCandidates(
    const Buffer &buffer, const Array<PrimExpr> &indices,
    const std::vector<IndexBinding> &bindings, int exec_rank,
    const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &config, arith::Analyzer *analyzer,
    AlignmentMode alignment_mode = AlignmentMode::kStrict) {
  std::vector<AccessTileCandidate> candidates;
  for (const TrailingTilePattern &pattern :
       EnumerateInferredTrailingTilePatterns(
           buffer, exec_rank, layout_map, config, analyzer, alignment_mode)) {
    AddCandidateFromPattern(&candidates, buffer, indices, bindings, pattern,
                            analyzer, /*strict_checks=*/false);
  }
  return candidates;
}

std::vector<AccessTileCandidate> EnumerateAccessTileCandidates(
    const BufferAccessRecord &access, const std::vector<IndexBinding> &bindings,
    int exec_rank, const TileViewMap &manual_tileviews,
    const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &config, arith::Analyzer *analyzer,
    AlignmentMode alignment_mode = AlignmentMode::kStrict) {
  auto manual_it = manual_tileviews.find(access.buffer->data);
  if (manual_it != manual_tileviews.end()) {
    return EnumerateManualCandidates(access.buffer, access.indices, bindings,
                                     manual_it->second, exec_rank, layout_map,
                                     config, analyzer);
  }

  return EnumerateInferredCandidates(access.buffer, access.indices, bindings,
                                     exec_rank, layout_map, config, analyzer,
                                     alignment_mode);
}

bool IsEligibleRank1LoadForRelaxedAlignmentSearch(
    const BufferAccessRecord &access, const std::vector<IndexBinding> &bindings,
    int exec_rank) {
  if (access.is_store || exec_rank != 2 || access.buffer->shape.size() != 1 ||
      access.indices.size() != 1 || bindings.size() != 1) {
    return false;
  }

  const IndexBinding &binding = bindings[0];
  return binding.uses_loop_var && binding.domain_axis >= 0;
}

const std::vector<AccessTileCandidate> &
CandidatesForSearch(const AccessInfo &access, AlignmentMode mode) {
  return mode == AlignmentMode::kRelaxed ? access.relaxed_candidates
                                         : access.strict_candidates;
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
CollectRank2PlanCandidates(const std::vector<AccessTileCandidate> &candidates) {
  std::vector<ExecutionPlanCandidate> plans;
  for (const auto &candidate : candidates) {
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

std::optional<TileViewPlan>
TrySelectPlan(const Array<PrimExpr> &domain,
              const std::vector<AccessInfo> &accesses, int domain_rank,
              int exec_rank, arith::Analyzer *analyzer, AlignmentMode mode,
              bool fail_on_error) {
  for (const auto &access : accesses) {
    if (CandidatesForSearch(access, mode).empty()) {
      if (fail_on_error) {
        LOG(FATAL)
            << "Cannot infer any feasible TileView candidate for access to "
               "buffer "
            << access.buffer->name << " with indices " << access.indices << ".";
      }
      return std::nullopt;
    }
  }

  if (exec_rank == 1) {
    std::vector<int> plan_extents;
    for (const auto &access : accesses) {
      for (const auto &candidate : CandidatesForSearch(access, mode)) {
        if (Supports1DPlan(candidate)) {
          plan_extents.push_back(candidate.tile_shape[0]);
        }
      }
    }

    if (plan_extents.empty()) {
      if (fail_on_error) {
        LOG(FATAL) << "Cannot infer any feasible 1D execution tileview for "
                      "T.Tiles domain "
                   << domain << ".";
      }
      return std::nullopt;
    }

    std::sort(plan_extents.begin(), plan_extents.end());
    plan_extents.erase(std::unique(plan_extents.begin(), plan_extents.end()),
                       plan_extents.end());
    std::sort(plan_extents.begin(), plan_extents.end(), std::greater<int>());

    for (int tile_extent : plan_extents) {
      bool all_supported = true;
      for (const auto &access : accesses) {
        const auto &candidates = CandidatesForSearch(access, mode);
        bool access_supported =
            std::any_of(candidates.begin(), candidates.end(),
                        [&](const AccessTileCandidate &candidate) {
                          return Supports1DPlan(candidate, tile_extent);
                        });
        if (!access_supported) {
          all_supported = false;
          break;
        }
      }

      if (all_supported) {
        return TileViewPlan{
            makeTileView(domain, {Integer(tile_extent)}, {Integer(-1)}), {0}};
      }
    }

    if (fail_on_error) {
      LOG(FATAL) << "Cannot infer a common 1D execution tileview for T.Tiles "
                 << "domain " << domain
                 << ". The per-access feasible tileview sets do not intersect.";
    }
    return std::nullopt;
  }

  std::vector<ExecutionPlanCandidate> plan_candidates;
  bool saw_rank2_candidates = false;
  for (const auto &access : accesses) {
    std::vector<ExecutionPlanCandidate> access_plan_candidates =
        CollectRank2PlanCandidates(CandidatesForSearch(access, mode));
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

  if (!saw_rank2_candidates) {
    if (fail_on_error) {
      LOG(FATAL)
          << "2D T.Tiles requires at least one access with a feasible rank-2 "
             "TileView candidate.";
    }
    return std::nullopt;
  }

  if (plan_candidates.empty()) {
    if (fail_on_error) {
      LOG(FATAL)
          << "Cannot infer a common execution tileview for T.Tiles domain "
          << domain
          << ". The rank-2 access candidates do not share a common axis "
             "binding and tile shape.";
    }
    return std::nullopt;
  }

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

    bool all_supported = true;
    for (const auto &access : accesses) {
      const auto &candidates = CandidatesForSearch(access, mode);
      bool access_supported =
          std::any_of(candidates.begin(), candidates.end(),
                      [&](const AccessTileCandidate &candidate) {
                        return Supports2DPlan(candidate, plan_candidate);
                      });
      if (!access_supported) {
        all_supported = false;
        break;
      }
    }

    if (all_supported) {
      return TileViewPlan{
          makeTileView(domain, {Integer(tile_height), Integer(tile_width)},
                       {Integer(exec_domain_axes[0] - domain_rank),
                        Integer(exec_domain_axes[1] - domain_rank)}),
          exec_domain_axes};
    }
  }

  if (fail_on_error) {
    LOG(FATAL) << "Cannot infer a common execution tileview for T.Tiles domain "
               << domain
               << ". The per-access feasible tileview sets do not intersect.";
  }
  return std::nullopt;
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

  Array<Var> loop_vars;
  std::unordered_set<const VarNode *> loop_var_nodes;
  for (const ForNode *loop : scope_loops) {
    loop_vars.push_back(loop->loop_var);
    loop_var_nodes.insert(loop->loop_var.get());
  }

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
    std::vector<AccessTileCandidate> relaxed_candidates = candidates;
    if (IsEligibleRank1LoadForRelaxedAlignmentSearch(access, bindings,
                                                     exec_rank)) {
      relaxed_candidates = EnumerateAccessTileCandidates(
          access, bindings, exec_rank, manual_tileviews, layout_map,
          tile_processor_config, &analyzer, AlignmentMode::kRelaxed);
    }

    analyzed_accesses.push_back({access.buffer, access.indices,
                                 std::move(candidates),
                                 std::move(relaxed_candidates)});
  }

  if (auto plan = TrySelectPlan(domain, analyzed_accesses, domain_rank,
                                exec_rank, &analyzer, AlignmentMode::kStrict,
                                /*fail_on_error=*/false)) {
    return plan.value();
  }

  if (auto plan = TrySelectPlan(domain, analyzed_accesses, domain_rank,
                                exec_rank, &analyzer, AlignmentMode::kRelaxed,
                                /*fail_on_error=*/false)) {
    return plan.value();
  }

  TrySelectPlan(domain, analyzed_accesses, domain_rank, exec_rank, &analyzer,
                AlignmentMode::kRelaxed, /*fail_on_error=*/true);
  return {TileView(), {}};
}

} // namespace tl
} // namespace tvm
