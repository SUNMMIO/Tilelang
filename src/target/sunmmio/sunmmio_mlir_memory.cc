#include "sunmmio_mlir_memory.h"

namespace tvm {
namespace codegen {

SunmmioMlirMemory::SunmmioMlirMemory(SunmmioMlirContext &ctx) : ctx_(ctx) {}

SunMMIOValue
SunmmioMlirMemory::Alloc(const std::string &result_name,
                         const SunMMIOType &memref_type,
                         const std::vector<SunMMIOValue> &dyn_extents,
                         const std::string &scope_name, DataType dtype) {
  (void)dyn_extents;
  (void)scope_name;
  SunMMIOValue out{dtype, result_name, memref_type};
  ctx_.value_symbol_table[result_name] = out;
  return out;
}

SunMMIOValue SunmmioMlirMemory::Load(const std::string &result_name,
                                     const std::string &buffer_handle,
                                     const std::vector<SunMMIOValue> &indices,
                                     const SunMMIOType &memref_type,
                                     DataType dtype,
                                     const SunMMIOType &result_type) {
  (void)buffer_handle;
  (void)indices;
  (void)memref_type;
  SunMMIOValue out{dtype, result_name, result_type};
  ctx_.value_symbol_table[result_name] = out;
  return out;
}

void SunmmioMlirMemory::Store(const SunMMIOValue &value,
                              const std::string &buffer_handle,
                              const std::vector<SunMMIOValue> &indices,
                              const SunMMIOType &memref_type) {
  (void)value;
  (void)buffer_handle;
  (void)indices;
  (void)memref_type;
}

} // namespace codegen
} // namespace tvm
