#ifndef TVM_TL_TARGET_SUNMMIO_MLIR_MEMORY_H_
#define TVM_TL_TARGET_SUNMMIO_MLIR_MEMORY_H_

#include "sunmmio_mlir_context.h"

namespace tvm {
namespace codegen {

class SunmmioMlirMemory {
 public:
  explicit SunmmioMlirMemory(SunmmioMlirContext& ctx);

  SunMMIOValue Alloc(const std::string& result_name,
                     const SunMMIOType& memref_type,
                     const std::vector<SunMMIOValue>& dyn_extents,
                     const std::string& scope_name, DataType dtype);

  SunMMIOValue Load(const std::string& result_name,
                    const std::string& buffer_handle,
                    const std::vector<SunMMIOValue>& indices,
                    const SunMMIOType& memref_type, DataType dtype,
                    const SunMMIOType& result_type);

  void Store(const SunMMIOValue& value, const std::string& buffer_handle,
             const std::vector<SunMMIOValue>& indices,
             const SunMMIOType& memref_type);

 private:
  SunmmioMlirContext& ctx_;
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TL_TARGET_SUNMMIO_MLIR_MEMORY_H_
