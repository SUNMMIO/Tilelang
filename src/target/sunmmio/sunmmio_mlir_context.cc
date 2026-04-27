#include "sunmmio_mlir_context.h"

namespace tvm {
namespace codegen {

SunmmioMlirContext::SunmmioMlirContext() : builder(&mlir_ctx) {}

void SunmmioMlirContext::Clear() {
  module_open = false;
  function_open = false;
  current_function.clear();
  insertion_point_stack.clear();
  value_symbol_table.clear();
  buffer_symbol_table.clear();
  mlir_value_table.clear();
  module = nullptr;
}

} // namespace codegen
} // namespace tvm
