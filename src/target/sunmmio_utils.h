/*!
 * \file tl/target/sunmmio_utils.h
 * \brief Centralized Sunmmio device-model helpers used by passes and analysis.
 */

#ifndef TVM_TL_TARGET_SUNMMIO_UTILS_H_
#define TVM_TL_TARGET_SUNMMIO_UTILS_H_

#include <optional>
#include <vector>

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>

namespace tvm {
namespace tl {

// ---------------------------------------------------------------------------
// Sunmmio on-chip SRAM scope identifiers
// ---------------------------------------------------------------------------
constexpr const char *kSunmmioScopeASRAM = "shared.asram";
constexpr const char *kSunmmioScopeWSRAM = "shared.wsram";
constexpr const char *kSunmmioScopeRSRAM = "shared.rsram";

struct SunmmioTileProcessorConfig {
  int register_bits;
  int block_height;
  int block_width;
  /// Minimum byte-alignment for RSRAM vector memory accesses.
  int rsram_align_bytes;
};

struct SunmmioMeshConfig {
  int nrow;
  int ncol;
};

SunmmioTileProcessorConfig
GetSunmmioTileProcessorConfig(ffi::Optional<Target> target);
SunmmioTileProcessorConfig GetSunmmioTileProcessorConfig(Target target);
ffi::Array<PrimExpr> GetSunmmioLayoutBlockShape(ffi::Optional<Target> target,
                                                DataType dtype);
ffi::Array<PrimExpr> GetSunmmioLayoutBlockShape(Target target, DataType dtype);
SunmmioMeshConfig GetSunmmioMeshConfig(ffi::Optional<Target> target);
SunmmioMeshConfig GetSunmmioMeshConfig(Target target);

/*!
 * \brief Check whether a buffer scope is one of the Sunmmio on-chip SRAM
 * scopes.
 */
inline bool IsSunmmioSramScope(const ffi::String &scope) {
  return scope == kSunmmioScopeASRAM || scope == kSunmmioScopeWSRAM ||
         scope == kSunmmioScopeRSRAM;
}

/*!
 * \brief Convert an RSRAM byte-alignment requirement into element count.
 */
inline int GetSunmmioRsramAlignmentElems(int rsram_align_bytes,
                                         DataType dtype) {
  if (rsram_align_bytes <= 0) {
    return 1;
  }
  ICHECK_EQ(dtype.lanes(), 1)
      << "Sunmmio RSRAM alignment expects scalar element dtypes, but got "
      << dtype << ".";
  int element_bits = dtype.bits();
  int align_bits = rsram_align_bytes * 8;
  if (align_bits <= element_bits) {
    return 1;
  }
  ICHECK_EQ(align_bits % element_bits, 0)
      << "RSRAM alignment " << rsram_align_bytes
      << " bytes is not divisible by element bit-width " << element_bits << ".";
  return align_bits / element_bits;
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TARGET_SUNMMIO_UTILS_H_
