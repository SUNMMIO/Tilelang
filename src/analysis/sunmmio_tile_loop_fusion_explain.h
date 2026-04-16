#pragma once

// Explain/projection entrypoints for Sunmmio tile loop fusion.
//
// This layer adapts typed production objects into narrow FFI-friendly schemas
// used by Python tests and debugging tools. It must not participate in
// discovery, dependence inference, or planning.

#include <tvm/ir/module.h>

#include "../support/ffi_aliases.h"

namespace tvm {
namespace tl {

Map<String, ffi::Any>
ExplainSunmmioTileLoopFusionDiscovery(const IRModule &mod);

Map<String, ffi::Any>
ExplainSunmmioTileLoopFusionDependence(const IRModule &mod);

Map<String, ffi::Any> ExplainSunmmioTileLoopFusionPlan(const IRModule &mod);

} // namespace tl
} // namespace tvm
