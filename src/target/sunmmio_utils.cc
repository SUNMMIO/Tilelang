/*!
 * \file tl/target/sunmmio_utils.cc
 * \brief Centralized Sunmmio device-model helpers used by passes and analysis.
 */

#include "sunmmio_utils.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/op.h>

#include "../layout/cute_layout.h"
#include "../op/builtin.h"
#include "../op/comm.h"
#include "../tileview/tileview_planner_common.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

std::string IntVectorToString(const std::vector<int64_t> &values) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << values[i];
  }
  os << "]";
  return os.str();
}

int64_t StaticIntOrNegative(const PrimExpr &expr, arith::Analyzer *analyzer) {
  PrimExpr simplified = analyzer->Simplify(expr);
  const auto *imm = simplified.as<IntImmNode>();
  if (!imm || imm->value <= 0) {
    return -1;
  }
  return imm->value;
}

std::vector<int64_t> GetScalarAccessLayoutExtents(
    const tir::Buffer &buffer, const ffi::Optional<tl::Layout> &layout,
    arith::Analyzer *analyzer, std::string *failure_reason) {
  ffi::Array<PrimExpr> extents;
  if (layout.defined()) {
    const auto *cute = layout.value().as<tl::CuteLayoutNode>();
    if (!cute) {
      if (failure_reason != nullptr) {
        *failure_reason = "Sunmmio scalar tile access planning expects a "
                          "CuteLayout when a layout annotation is present.";
      }
      return {};
    }
    extents = cute->GetCoveredShape();
  } else {
    extents = buffer->shape;
  }

  std::vector<int64_t> static_extents;
  static_extents.reserve(extents.size());
  for (const PrimExpr &extent : extents) {
    static_extents.push_back(StaticIntOrNegative(extent, analyzer));
  }
  return static_extents;
}

int64_t TileElementCount(const std::vector<int64_t> &shape) {
  int64_t elems = 1;
  for (int64_t extent : shape) {
    elems *= extent;
  }
  return elems;
}

bool TileTransferSizeIsLegal(const tir::Buffer &buffer, DataType dtype,
                             const std::vector<int64_t> &tile_shape,
                             const tl::SunmmioTileProcessorConfig &config) {
  if (buffer.scope() != kSunmmioScopeRSRAM && buffer.scope() != "rsram") {
    return true;
  }
  if (config.rsram_align_bytes <= 0) {
    return true;
  }
  int64_t tile_bits =
      TileElementCount(tile_shape) * static_cast<int64_t>(dtype.bits());
  if (tile_bits % 8 != 0) {
    return false;
  }
  int64_t tile_bytes = tile_bits / 8;
  return tile_bytes >= config.rsram_align_bytes &&
         tile_bytes % config.rsram_align_bytes == 0;
}

bool CandidateExtentsAreLegal(const tl::TrailingTilePattern &pattern,
                              const std::vector<int64_t> &layout_extents,
                              std::string *failure_reason) {
  for (size_t i = 0; i < pattern.mapped_dims.size(); ++i) {
    int dim = pattern.mapped_dims[i];
    int64_t extent = layout_extents[dim];
    int64_t tile_extent = pattern.tile_shape[i];
    if (extent < 0) {
      continue;
    }
    if (tile_extent > extent) {
      if (failure_reason != nullptr) {
        std::ostringstream os;
        os << "tile extent " << tile_extent << " exceeds layout extent "
           << extent << " at buffer dim " << dim << ".";
        *failure_reason = os.str();
      }
      return false;
    }
    if (extent % tile_extent != 0) {
      if (failure_reason != nullptr) {
        std::ostringstream os;
        os << "layout extent " << extent << " is not divisible by tile extent "
           << tile_extent << " at buffer dim " << dim << ".";
        *failure_reason = os.str();
      }
      return false;
    }
  }
  return true;
}

bool HasTrailingRowMajorContiguousOrder(
    const tir::Buffer &buffer,
    const ffi::Map<tir::Buffer, tl::Layout> &layout_map,
    std::string *failure_reason) {
  std::vector<tl::ContiguousStep> steps =
      tl::GetBufferContiguousSteps(buffer, layout_map);
  if (steps.empty()) {
    if (failure_reason != nullptr) {
      *failure_reason = "cannot recover a contiguous trailing layout step.";
    }
    return false;
  }
  int width_dim = static_cast<int>(buffer->shape.size()) - 1;
  if (steps.front().dim != width_dim) {
    if (failure_reason != nullptr) {
      std::ostringstream os;
      os << "the innermost contiguous layout step is dim " << steps.front().dim
         << ", not the trailing width dim " << width_dim
         << "; suvm.tile.load/store require row-major tile views.";
      *failure_reason = os.str();
    }
    return false;
  }
  return true;
}

std::optional<tl::TrailingTilePattern>
SelectScalarTilePattern(const tir::Buffer &buffer, int tile_rank,
                        DataType dtype,
                        const ffi::Map<tir::Buffer, tl::Layout> &layout_map,
                        const std::vector<int64_t> &layout_extents,
                        const tl::SunmmioTileProcessorConfig &config,
                        arith::Analyzer *analyzer, std::string *last_reason) {
  std::vector<tl::TrailingTilePattern> legal_patterns;
  for (const tl::TrailingTilePattern &pattern :
       tl::EnumerateInferredTrailingTilePatterns(
           buffer, tile_rank, layout_map, config, analyzer,
           {tl::TileExtentPolicy::kRegisterBounded,
            tl::AlignmentMode::kStrict})) {
    if (static_cast<int>(pattern.tile_shape.size()) != tile_rank) {
      continue;
    }
    std::vector<int64_t> tile_shape(pattern.tile_shape.begin(),
                                    pattern.tile_shape.end());
    if (!TileTransferSizeIsLegal(buffer, dtype, tile_shape, config)) {
      if (last_reason != nullptr) {
        std::ostringstream os;
        os << "tile shape " << IntVectorToString(tile_shape)
           << " does not satisfy the " << config.rsram_align_bytes
           << "-byte RSRAM vector-transfer size/alignment constraint.";
        *last_reason = os.str();
      }
      continue;
    }
    if (!CandidateExtentsAreLegal(pattern, layout_extents, last_reason)) {
      continue;
    }
    legal_patterns.push_back(pattern);
  }

  if (legal_patterns.empty()) {
    return std::nullopt;
  }

  std::sort(legal_patterns.begin(), legal_patterns.end(),
            [](const tl::TrailingTilePattern &lhs,
               const tl::TrailingTilePattern &rhs) {
              int64_t lhs_elems = tl::TileElements(lhs.tile_shape);
              int64_t rhs_elems = tl::TileElements(rhs.tile_shape);
              if (lhs_elems != rhs_elems) {
                return lhs_elems > rhs_elems;
              }
              return lhs.tile_shape > rhs.tile_shape;
            });
  return legal_patterns.front();
}

} // namespace

std::optional<SunmmioScalarTileAccessPlan> PlanSunmmioScalarTileAccess(
    const tir::Buffer &buffer, const ffi::Array<PrimExpr> &indices,
    DataType dtype, const ffi::Optional<tl::Layout> &layout,
    const tl::SunmmioTileProcessorConfig &config, std::string *failure_reason) {
  int buffer_rank = static_cast<int>(buffer->shape.size());
  if (buffer_rank < 1) {
    if (failure_reason != nullptr) {
      *failure_reason = "scalar tile access requires a rank >= 1 buffer.";
    }
    return std::nullopt;
  }
  if (static_cast<int>(indices.size()) != buffer_rank) {
    if (failure_reason != nullptr) {
      std::ostringstream os;
      os << "indices rank " << indices.size()
         << " does not match source buffer rank " << buffer_rank << ".";
      *failure_reason = os.str();
    }
    return std::nullopt;
  }

  arith::Analyzer analyzer;
  ffi::Map<tir::Buffer, tl::Layout> layout_map;
  if (layout.defined()) {
    layout_map.Set(buffer, layout.value());
  }

  std::string local_reason;
  std::vector<int64_t> layout_extents =
      GetScalarAccessLayoutExtents(buffer, layout, &analyzer, &local_reason);
  if (layout_extents.empty()) {
    if (failure_reason != nullptr) {
      *failure_reason = local_reason;
    }
    return std::nullopt;
  }
  if (static_cast<int>(layout_extents.size()) != buffer_rank) {
    if (failure_reason != nullptr) {
      std::ostringstream os;
      os << "layout covered rank " << layout_extents.size()
         << " does not match buffer rank " << buffer_rank << ".";
      *failure_reason = os.str();
    }
    return std::nullopt;
  }

  if (!HasTrailingRowMajorContiguousOrder(buffer, layout_map, &local_reason)) {
    if (failure_reason != nullptr) {
      *failure_reason = local_reason;
    }
    return std::nullopt;
  }

  std::vector<int> ranks_to_try;
  if (buffer_rank >= 2) {
    ranks_to_try.push_back(2);
  }
  ranks_to_try.push_back(1);

  std::optional<tl::TrailingTilePattern> selected;
  for (int tile_rank : ranks_to_try) {
    selected = SelectScalarTilePattern(buffer, tile_rank, dtype, layout_map,
                                       layout_extents, config, &analyzer,
                                       &local_reason);
    if (selected.has_value()) {
      break;
    }
  }

  if (!selected.has_value()) {
    if (failure_reason != nullptr) {
      std::ostringstream os;
      os << "cannot infer a legal trailing rank-1/rank-2 tile_view";
      if (!local_reason.empty()) {
        os << ": " << local_reason;
      }
      os << ". Use an aligned-row-major staging layout when logical extents "
            "are "
            "not compatible with 64B RSRAM vector transfers.";
      *failure_reason = os.str();
    }
    return std::nullopt;
  }

  SunmmioScalarTileAccessPlan plan;
  plan.tile_shape.assign(selected->tile_shape.begin(),
                         selected->tile_shape.end());
  plan.tiled_dims.assign(selected->mapped_dims.begin(),
                         selected->mapped_dims.end());
  plan.partition_indices.reserve(buffer_rank);
  std::unordered_map<int, int64_t> tiled_dim_to_extent;
  for (size_t i = 0; i < selected->mapped_dims.size(); ++i) {
    tiled_dim_to_extent[selected->mapped_dims[i]] = selected->tile_shape[i];
  }

  for (int dim = 0; dim < buffer_rank; ++dim) {
    auto it = tiled_dim_to_extent.find(dim);
    if (it == tiled_dim_to_extent.end()) {
      plan.partition_indices.push_back(indices[dim]);
      continue;
    }
    PrimExpr tile_extent = IntImm(indices[dim].dtype(), it->second);
    plan.partition_indices.push_back(
        analyzer.Simplify(floordiv(indices[dim], tile_extent)));
    plan.local_indices.push_back(
        analyzer.Simplify(floormod(indices[dim], tile_extent)));
  }

  return plan;
}

tir::Region
GetSunmmioScalarTileAccessRegion(const tir::Buffer &buffer,
                                 const ffi::Array<PrimExpr> &indices,
                                 const ffi::Map<tir::Buffer, Layout> &layouts,
                                 const SunmmioTileProcessorConfig &config) {
  tir::Region region;
  for (const PrimExpr &index : indices) {
    region.push_back(Range::FromMinExtent(index, 1));
  }
  if (buffer.scope() != kSunmmioScopeRSRAM ||
      std::any_of(indices.begin(), indices.end(), [](const PrimExpr &index) {
        return index.dtype().lanes() != 1;
      })) {
    return region;
  }
  ffi::Optional<Layout> layout;
  for (const auto &entry : layouts) {
    if (entry.first->data.same_as(buffer->data)) {
      layout = entry.second;
      break;
    }
  }
  std::string reason;
  auto plan = PlanSunmmioScalarTileAccess(buffer, indices, buffer->dtype,
                                          layout, config, &reason);
  if (!plan) {
    // Without a legal scalar access plan, retain a conservative storage bound.
    auto shape = buffer->shape;
    if (layout.defined()) {
      if (const auto *cute = layout.value().as<CuteLayoutNode>()) {
        shape = cute->GetCoveredShape();
      }
    }
    for (size_t dim = 0; dim < region.size(); ++dim) {
      region.Set(dim, Range::FromMinExtent(0, shape[dim]));
    }
    return region;
  }
  arith::Analyzer analyzer;
  for (size_t i = 0; i < plan->tiled_dims.size(); ++i) {
    int dim = plan->tiled_dims[i];
    int64_t extent = plan->tile_shape[i];
    region.Set(dim,
               Range::FromMinExtent(
                   analyzer.Simplify(floordiv(indices[dim], Integer(extent)) *
                                     Integer(extent)),
                   Integer(extent)));
  }
  return region;
}

const char *StringifySunmmioOdmaUnit(SunmmioOdmaUnit unit) {
  switch (unit) {
  case SunmmioOdmaUnit::kOdma0:
    return "odma0";
  case SunmmioOdmaUnit::kOdma1:
    return "odma1";
  }
  LOG(FATAL) << "Unknown Sunmmio ODMA unit";
  TVM_FFI_UNREACHABLE();
}

PrimExpr MakeSunmmioOdmaUnitExpr(SunmmioOdmaUnit unit) {
  return tir::Call(DataType::Handle(), odma_unit(),
                   {StringImm(StringifySunmmioOdmaUnit(unit))});
}

std::optional<SunmmioOdmaUnit> ParseSunmmioOdmaUnitExpr(const PrimExpr &expr) {
  const auto *call = expr.as<tir::CallNode>();
  if (!call || !call->op.same_as(odma_unit())) {
    return std::nullopt;
  }
  ICHECK_EQ(call->args.size(), 1U)
      << "tl.odma_unit expects exactly one argument";
  const auto *unit = call->args[0].as<StringImmNode>();
  ICHECK(unit) << "tl.odma_unit expects a StringImm argument";
  if (unit->value == "odma0") {
    return SunmmioOdmaUnit::kOdma0;
  }
  if (unit->value == "odma1") {
    return SunmmioOdmaUnit::kOdma1;
  }
  LOG(FATAL) << "tl.odma_unit expects odma0 or odma1, but got " << unit->value;
  TVM_FFI_UNREACHABLE();
}

std::optional<SunmmioOdmaUnit> GetSunmmioOdmaUnit(const tir::CallNode *call) {
  std::optional<SunmmioOdmaUnit> result;
  for (const PrimExpr &arg : call->args) {
    std::optional<SunmmioOdmaUnit> unit = ParseSunmmioOdmaUnitExpr(arg);
    if (!unit) {
      continue;
    }
    ICHECK(!result.has_value())
        << "Sunmmio transfer carries more than one tl.odma_unit argument";
    result = unit;
  }
  return result;
}

namespace {

enum class SunmmioMemoryScope {
  kGlobal,
  kASRAM,
  kWSRAM,
  kRSRAM,
  kUnsupported,
};

enum class SunmmioDTypePolicy {
  kSameDType,
  kConversionAllowed,
};

struct SunmmioDirectTransferRule {
  SunmmioTransferMechanism mechanism;
  SunmmioMemoryScope src_scope;
  SunmmioMemoryScope dst_scope;
  SunmmioDTypePolicy dtype_policy;
};

// Temporary high-level projection of the A4E capabilities defined by NPU-IR's
// DeviceArchitecture.td. Keep this table at TileLang's abstraction level: it
// describes direct transfer outcomes without reproducing channel wiring,
// alignment, descriptor, or layout constraints. Once NPU-IR exposes a stable
// capability API, SupportsSunmmioDirectTransfer can delegate to it without
// changing TileLang callers.
constexpr SunmmioDirectTransferRule kSunmmioA4EDirectTransferRules[] = {
    {SunmmioTransferMechanism::kLocalDma, SunmmioMemoryScope::kGlobal,
     SunmmioMemoryScope::kRSRAM, SunmmioDTypePolicy::kSameDType},
    {SunmmioTransferMechanism::kLocalDma, SunmmioMemoryScope::kGlobal,
     SunmmioMemoryScope::kWSRAM, SunmmioDTypePolicy::kSameDType},
    {SunmmioTransferMechanism::kLocalDma, SunmmioMemoryScope::kRSRAM,
     SunmmioMemoryScope::kGlobal, SunmmioDTypePolicy::kSameDType},
    {SunmmioTransferMechanism::kLocalDma, SunmmioMemoryScope::kRSRAM,
     SunmmioMemoryScope::kASRAM, SunmmioDTypePolicy::kSameDType},
    {SunmmioTransferMechanism::kLocalDma, SunmmioMemoryScope::kRSRAM,
     SunmmioMemoryScope::kWSRAM, SunmmioDTypePolicy::kSameDType},
    {SunmmioTransferMechanism::kLocalDma, SunmmioMemoryScope::kRSRAM,
     SunmmioMemoryScope::kRSRAM, SunmmioDTypePolicy::kSameDType},
    {SunmmioTransferMechanism::kTile, SunmmioMemoryScope::kRSRAM,
     SunmmioMemoryScope::kRSRAM, SunmmioDTypePolicy::kConversionAllowed},
    {SunmmioTransferMechanism::kHLink, SunmmioMemoryScope::kRSRAM,
     SunmmioMemoryScope::kASRAM, SunmmioDTypePolicy::kSameDType},
    {SunmmioTransferMechanism::kHLink, SunmmioMemoryScope::kRSRAM,
     SunmmioMemoryScope::kRSRAM, SunmmioDTypePolicy::kSameDType},
    {SunmmioTransferMechanism::kVLink, SunmmioMemoryScope::kGlobal,
     SunmmioMemoryScope::kWSRAM, SunmmioDTypePolicy::kSameDType},
    {SunmmioTransferMechanism::kVLink, SunmmioMemoryScope::kGlobal,
     SunmmioMemoryScope::kRSRAM, SunmmioDTypePolicy::kSameDType},
    {SunmmioTransferMechanism::kVLink, SunmmioMemoryScope::kRSRAM,
     SunmmioMemoryScope::kWSRAM, SunmmioDTypePolicy::kSameDType},
    {SunmmioTransferMechanism::kVLink, SunmmioMemoryScope::kRSRAM,
     SunmmioMemoryScope::kRSRAM, SunmmioDTypePolicy::kSameDType},
};

SunmmioMemoryScope ParseSunmmioMemoryScope(const ffi::String &scope) {
  if (scope.empty() || scope == "global") {
    return SunmmioMemoryScope::kGlobal;
  }
  if (scope == kSunmmioScopeASRAM) {
    return SunmmioMemoryScope::kASRAM;
  }
  if (scope == kSunmmioScopeWSRAM) {
    return SunmmioMemoryScope::kWSRAM;
  }
  if (scope == kSunmmioScopeRSRAM) {
    return SunmmioMemoryScope::kRSRAM;
  }
  return SunmmioMemoryScope::kUnsupported;
}

SunmmioTileProcessorConfig MakeSunmmioA4EConfig() {
  return {/*register_bits=*/4096,
          /*block_height=*/32,
          /*block_width=*/32,
          /*rsram_align_bytes=*/64,
          /*asram_bank_stripe_bytes=*/1024,
          /*bf16_gemm_single_pass_max_rows=*/16,
          // NPU-IR DeviceArchitecture.td: A4E_Hlink/VlinkWriteRsram.
          /*rsram_mcast_transfer_bytes=*/1024};
}

ffi::Array<PrimExpr> MakeSunmmioA4EBlockShape(DataType dtype) {
  // A4E uses a (32, 32) block for every element dtype so that buffers of
  // different dtypes share the same tile shapes in elements.
  (void)dtype;
  return {tir::make_const(DataType::Int(32), 32),
          tir::make_const(DataType::Int(32), 32)};
}

SunmmioMeshConfig MakeSunmmioA4EMeshConfig() {
  return {/*nrow=*/4, /*ncol=*/4};
}

Target NormalizeTarget(Target target) {
  if (target.defined() && target->GetHost()) {
    return target.WithoutHost();
  }
  return target;
}

std::optional<int> ParsePositiveIntegerMattr(Target target,
                                             const char *attr_prefix) {
  auto mattr = target->GetAttr<tvm::ffi::Array<tvm::ffi::String>>("mattr");
  if (!mattr.has_value()) {
    return std::nullopt;
  }

  std::optional<int> parsed_value;
  const std::string prefix(attr_prefix);
  for (const auto &attr : mattr.value()) {
    const std::string value = attr;
    if (value.rfind(prefix, 0) != 0) {
      continue;
    }

    const std::string suffix = value.substr(prefix.size());
    int candidate = 0;
    size_t parsed_chars = 0;
    try {
      candidate = std::stoi(suffix, &parsed_chars);
    } catch (const std::invalid_argument &) {
      candidate = -1;
    } catch (const std::out_of_range &) {
      candidate = -1;
    }

    ICHECK_GT(parsed_chars, 0U) << "Invalid Sunmmio target attribute '" << value
                                << "': expected a positive integer suffix.";
    ICHECK_EQ(parsed_chars, suffix.size())
        << "Invalid Sunmmio target attribute '" << value
        << "': expected only digits after '" << prefix << "'.";
    ICHECK_GT(candidate, 0) << "Invalid Sunmmio target attribute '" << value
                            << "': expected a positive integer suffix.";

    if (parsed_value.has_value()) {
      ICHECK_EQ(parsed_value.value(), candidate)
          << "Conflicting Sunmmio target attributes for '" << prefix
          << "' in mattr.";
    }
    parsed_value = candidate;
  }

  return parsed_value;
}

} // namespace

SunmmioTileProcessorConfig
GetSunmmioTileProcessorConfig(ffi::Optional<Target> target) {
  if (!target.defined()) {
    return MakeSunmmioA4EConfig();
  }
  return GetSunmmioTileProcessorConfig(target.value());
}

SunmmioTileProcessorConfig GetSunmmioTileProcessorConfig(Target target) {
  target = NormalizeTarget(target);

  if (!target.defined() || !TargetIsSunmmio(target)) {
    // T.Tiles is currently only modeled for Sunmmio tile processors. Until the
    // broader target contract is tightened, keep the existing A4E defaults as a
    // conservative fallback so target-less unit tests continue to work.
    return MakeSunmmioA4EConfig();
  }

  auto mcpu = target->GetAttr<tvm::ffi::String>("mcpu");
  if (!mcpu.has_value() || mcpu.value() == "sunmmio-a4e") {
    return MakeSunmmioA4EConfig();
  }

  LOG(WARNING) << "Unknown Sunmmio device model '" << mcpu.value()
               << "' when querying tile-processor config. Falling back to the "
                  "sunmmio-a4e defaults.";
  return MakeSunmmioA4EConfig();
}

ffi::Array<PrimExpr> GetSunmmioLayoutBlockShape(ffi::Optional<Target> target,
                                                DataType dtype) {
  if (!target.defined()) {
    return MakeSunmmioA4EBlockShape(dtype);
  }
  return GetSunmmioLayoutBlockShape(target.value(), dtype);
}

ffi::Array<PrimExpr> GetSunmmioLayoutBlockShape(Target target, DataType dtype) {
  target = NormalizeTarget(target);

  if (!target.defined() || !TargetIsSunmmio(target)) {
    // Keep target-less tests and non-Sunmmio analysis fallbacks aligned with
    // the current A4E device model until the broader target contract is strict.
    return MakeSunmmioA4EBlockShape(dtype);
  }

  auto mcpu = target->GetAttr<tvm::ffi::String>("mcpu");
  if (!mcpu.has_value() || mcpu.value() == "sunmmio-a4e") {
    return MakeSunmmioA4EBlockShape(dtype);
  }

  LOG(WARNING) << "Unknown Sunmmio device model '" << mcpu.value()
               << "' when querying layout block shape. Falling back to the "
                  "sunmmio-a4e defaults.";
  return MakeSunmmioA4EBlockShape(dtype);
}

SunmmioMeshConfig GetSunmmioMeshConfig(ffi::Optional<Target> target) {
  if (!target.defined()) {
    return MakeSunmmioA4EMeshConfig();
  }
  return GetSunmmioMeshConfig(target.value());
}

SunmmioMeshConfig GetSunmmioMeshConfig(Target target) {
  target = NormalizeTarget(target);

  if (!target.defined()) {
    return MakeSunmmioA4EMeshConfig();
  }

  ICHECK(TargetIsSunmmio(target))
      << "Sunmmio mesh config is only defined for Sunmmio targets.";

  auto nrow = ParsePositiveIntegerMattr(target, "device_mesh_nrow_");
  auto ncol = ParsePositiveIntegerMattr(target, "device_mesh_ncol_");

  ICHECK(nrow.has_value()) << "Sunmmio target is missing required mattr entry "
                              "'device_mesh_nrow_<positive-int>'.";
  ICHECK(ncol.has_value()) << "Sunmmio target is missing required mattr entry "
                              "'device_mesh_ncol_<positive-int>'.";

  return {/*nrow=*/nrow.value(), /*ncol=*/ncol.value()};
}

bool SupportsSunmmioDirectTransfer(Target target,
                                   SunmmioTransferMechanism mechanism,
                                   ffi::String src_scope, DataType src_dtype,
                                   ffi::String dst_scope, DataType dst_dtype) {
  target = NormalizeTarget(target);
  if (!target.defined() || !TargetIsSunmmio(target)) {
    return false;
  }

  auto mcpu = target->GetAttr<ffi::String>("mcpu");
  if (mcpu.has_value() && mcpu.value() != "sunmmio-a4e") {
    return false;
  }

  // MX is a data+scale storage container, not a vector register element type.
  // Moving an MX container uses DMA; quantization is an explicit operation.
  if (mechanism == SunmmioTransferMechanism::kTile &&
      (sunmmio::IsMXDType(src_dtype) || sunmmio::IsMXDType(dst_dtype))) {
    return false;
  }

  SunmmioMemoryScope parsed_src_scope = ParseSunmmioMemoryScope(src_scope);
  SunmmioMemoryScope parsed_dst_scope = ParseSunmmioMemoryScope(dst_scope);
  for (const SunmmioDirectTransferRule &rule : kSunmmioA4EDirectTransferRules) {
    if (rule.mechanism != mechanism || rule.src_scope != parsed_src_scope ||
        rule.dst_scope != parsed_dst_scope) {
      continue;
    }
    return rule.dtype_policy == SunmmioDTypePolicy::kConversionAllowed ||
           src_dtype == dst_dtype;
  }
  return false;
}

bool SupportsSunmmioDirectCopy(Target target, ffi::String src_scope,
                               DataType src_dtype, ffi::String dst_scope,
                               DataType dst_dtype) {
  return SupportsSunmmioDirectTransfer(target, SunmmioTransferMechanism::kTile,
                                       src_scope, src_dtype, dst_scope,
                                       dst_dtype) ||
         SupportsSunmmioDirectTransfer(
             target, SunmmioTransferMechanism::kLocalDma, src_scope, src_dtype,
             dst_scope, dst_dtype);
}

bool SupportsSunmmioDirectCommunication(
    Target target, CommunicationDirections directions, ffi::String src_scope,
    DataType src_dtype, ffi::String dst_scope, DataType dst_dtype) {
  auto supports = [&](SunmmioTransferMechanism mechanism) {
    return SupportsSunmmioDirectTransfer(target, mechanism, src_scope,
                                         src_dtype, dst_scope, dst_dtype);
  };
  switch (directions) {
  case CommunicationDirections::kHorizontal:
    return supports(SunmmioTransferMechanism::kHLink);
  case CommunicationDirections::kVertical:
    return supports(SunmmioTransferMechanism::kVLink);
  case CommunicationDirections::kHorizontalAndVertical:
    return supports(SunmmioTransferMechanism::kHLink) &&
           supports(SunmmioTransferMechanism::kVLink);
  case CommunicationDirections::kNone:
    return false;
  }
  return false;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.target.GetSunmmioLayoutBlockShape",
                        static_cast<ffi::Array<PrimExpr> (*)(Target, DataType)>(
                            GetSunmmioLayoutBlockShape));
  // Expose annotation-key constants used by the Sunmmio bf16 GEMM legalization
  // pass so the Python frontend can refer to the same string without
  // duplicating the literal.
  refl::GlobalDef().def("tl.target.GetAttrSrcOffsetByte",
                        []() { return tvm::ffi::String(kAttrSrcOffsetByte); });
  refl::GlobalDef().def("tl.target.GetFieldAccOffsetByte",
                        []() { return tvm::ffi::String(kFieldAccOffsetByte); });
}

} // namespace tl
} // namespace tvm
