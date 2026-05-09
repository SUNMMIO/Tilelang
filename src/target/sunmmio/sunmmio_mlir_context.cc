#include "sunmmio_mlir_context.h"

namespace tvm {
namespace codegen {

SunmmioMlirContext::SunmmioMlirContext() : builder(&mlir_ctx) {}

void SunmmioMlirContext::Clear() {
  mlir_value_table_stack.clear();
  token_by_id.clear();
  for_stack.clear();
  if_stack.clear();
  control_flow_stack.clear();
  module = nullptr;
}

} // namespace codegen
} // namespace tvm
