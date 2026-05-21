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

// ---------------------------------------------------------------------------
// Op annotation keys used by the Sunmmio bf16 GEMM legalization pass.
// ---------------------------------------------------------------------------
// Annotation key on CopyNode / AllgatherOpNode whose value is an IntImm
// giving the byte offset added to the source pointer at codegen. Set by the
// LegalizeSunmmioGemm pass to re-stage south-bound A data into the
// destination's north bank for the second bf16 TC half-pass. The value
// propagates to the leaf intrinsic (tl.dma_copy / tl.broadcast_) as a
// trailing positional arg.
constexpr const char *kAttrSrcOffsetByte = "src_offset_byte";

// Reflection field name for the Gemm/GemmPy node member accOffsetByte_. It
// is the Python-visible name registered via def_ro and consumed via
// getattr(node, kFieldAccOffsetByte) in GemmBase. The Sunmmio bf16 GEMM
// legalization pass sets this field on the cloned second-pass Gemm to
// select the second stripe-parity's starting row in the RSRAM accumulator.
constexpr const char *kFieldAccOffsetByte = "accOffsetByte";

struct SunmmioTileProcessorConfig {
  int register_bits;
  int block_height;
  int block_width;
  /// Minimum byte-alignment for RSRAM vector memory accesses.
  int rsram_align_bytes;
  /// ASRAM north/south bank stripe width in bytes. The bf16 tensor core can
  /// only read from the north bank, so legalization of bf16 GEMM duplicates
  /// the A-operand writer with a source-pointer offset of this many bytes so
  /// that what previously landed in destination south now lands in north.
  int asram_bank_stripe_bytes;
  /// Largest bf16 GEMM row count (M-extent) the bf16 tensor core consumes
  /// from the ASRAM north bank in a single pass. A bf16 GEMM whose row count
  /// does not exceed this fits entirely in the north bank and needs no
  /// two-pass legalization.
  int bf16_gemm_single_pass_max_rows;
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
