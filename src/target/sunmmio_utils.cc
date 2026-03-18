/*!
 * \file tl/target/sunmmio_utils.cc
 * \brief Centralized Sunmmio device-model helpers used by passes and analysis.
 */

#include "sunmmio_utils.h"

#include <stdexcept>
#include <string>

#include <tvm/runtime/logging.h>

#include "utils.h"

namespace tvm {
namespace tl {

namespace {

SunmmioTileProcessorConfig MakeSunmmioA4EConfig() {
  return {/*register_bits=*/4096, /*block_height=*/32, /*block_width=*/32};
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

} // namespace tl
} // namespace tvm
