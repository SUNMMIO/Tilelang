/*!
 * \file tl/target/sunmmio_utils.cc
 * \brief Centralized Sunmmio device-model helpers used by passes and analysis.
 */

#include "sunmmio_utils.h"

#include <stdexcept>
#include <string>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/op.h>

#include "../op/comm.h"
#include "utils.h"

namespace tvm {
namespace tl {

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
          /*bf16_gemm_single_pass_max_rows=*/16};
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
