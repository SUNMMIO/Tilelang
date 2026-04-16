/*!
 * \file tl/target/sunmmio_utils.h
 * \brief Centralized Sunmmio device-model helpers used by passes and analysis.
 */

#ifndef TVM_TL_TARGET_SUNMMIO_UTILS_H_
#define TVM_TL_TARGET_SUNMMIO_UTILS_H_

#include <optional>
#include <vector>

#include <tvm/target/target.h>

namespace tvm {
namespace tl {

struct SunmmioTileProcessorConfig {
  int register_bits;
  int block_height;
  int block_width;
};

struct SunmmioMeshConfig {
  int nrow;
  int ncol;
};

SunmmioTileProcessorConfig
GetSunmmioTileProcessorConfig(ffi::Optional<Target> target);
SunmmioTileProcessorConfig GetSunmmioTileProcessorConfig(Target target);
SunmmioMeshConfig GetSunmmioMeshConfig(ffi::Optional<Target> target);
SunmmioMeshConfig GetSunmmioMeshConfig(Target target);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TARGET_SUNMMIO_UTILS_H_
