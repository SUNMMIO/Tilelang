#ifndef TVM_TL_TARGET_SUNMMIO_MLIR_CALL_H_
#define TVM_TL_TARGET_SUNMMIO_MLIR_CALL_H_

#include "sunmmio_mlir_context.h"

namespace tvm {
namespace codegen {

class SunmmioMlirCall {
public:
  explicit SunmmioMlirCall(SunmmioMlirContext &ctx);

  SunMMIOValue Call(const std::string &result_name, const std::string &callee,
                    const std::vector<SunMMIOValue> &operands,
                    const std::vector<std::string> &string_args,
                    const std::string &category, DataType ret_dtype,
                    const SunMMIOType &ret_type);

private:
  SunmmioMlirContext &ctx_;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_SUNMMIO_MLIR_CALL_H_
