#include "sunmmio_mlir_context.h"

namespace tvm {
namespace codegen {

SunmmioMlirContext::SunmmioMlirContext() : builder(&mlir_ctx) {}

void SunmmioMlirContext::Clear() {
  mlir_value_table_stack.clear();
  for_stack.clear();
  module = nullptr;
}

} // namespace codegen
} // namespace tvm
