#include "sunmmio_mlir_call.h"

namespace tvm {
namespace codegen {

SunmmioMlirCall::SunmmioMlirCall(SunmmioMlirContext& ctx) : ctx_(ctx) {}

SunMMIOValue SunmmioMlirCall::Call(const std::string& result_name,
                                   const std::string& callee,
                                   const std::vector<SunMMIOValue>& operands,
                                   const std::vector<std::string>& string_args,
                                   const std::string& category,
                                   DataType ret_dtype,
                                   const SunMMIOType& ret_type) {
  (void)callee;
  (void)operands;
  (void)string_args;
  (void)category;
  SunMMIOValue out{ret_dtype, result_name, ret_type};
  if (!result_name.empty()) {
    ctx_.value_symbol_table[result_name] = out;
  }
  return out;
}

}  // namespace codegen
}  // namespace tvm
