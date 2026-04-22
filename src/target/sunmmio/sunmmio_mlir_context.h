#ifndef TVM_TL_TARGET_SUNMMIO_MLIR_CONTEXT_H_
#define TVM_TL_TARGET_SUNMMIO_MLIR_CONTEXT_H_

#include "codegen_sunmmio.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace codegen {

struct SunmmioMlirContext {
  bool module_open{false};
  bool function_open{false};
  std::string current_function;

  std::vector<std::string> insertion_point_stack;
  std::unordered_map<std::string, SunMMIOValue> value_symbol_table;
  std::unordered_map<std::string, BufferBinding> buffer_symbol_table;

  void Clear();
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_SUNMMIO_MLIR_CONTEXT_H_
