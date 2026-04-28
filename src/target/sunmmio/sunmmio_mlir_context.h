#ifndef TVM_TL_TARGET_SUNMMIO_MLIR_CONTEXT_H_
#define TVM_TL_TARGET_SUNMMIO_MLIR_CONTEXT_H_

#include "sunmmio_mlir_type.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace codegen {

struct SunmmioMlirContext {
  SunmmioMlirContext();

  mlir::MLIRContext mlir_ctx;
  mlir::OpBuilder builder;
  mlir::OwningOpRef<mlir::ModuleOp> module;

  using MLIRValueTable = std::unordered_map<std::string, mlir::Value>;
  std::vector<MLIRValueTable> mlir_value_table_stack;
  struct ForFrame {
    mlir::scf::ForOp op;
    ffi::Map<ffi::String, ffi::Any> annotations;
  };
  std::vector<ForFrame> for_stack;

  const ffi::Map<ffi::String, ffi::Any> *CurrentForAnnotations() const {
    if (for_stack.empty()) {
      return nullptr;
    }
    return &for_stack.back().annotations;
  }

  void ClearMLIRValueScopes() { mlir_value_table_stack.clear(); }

  void PushMLIRValueScope() { mlir_value_table_stack.emplace_back(); }

  void PopMLIRValueScope() {
    if (!mlir_value_table_stack.empty()) {
      mlir_value_table_stack.pop_back();
    }
  }

  mlir::Value LookupMLIRValue(const std::string &name) const {
    for (auto it = mlir_value_table_stack.rbegin();
         it != mlir_value_table_stack.rend(); ++it) {
      auto vit = it->find(name);
      if (vit != it->end()) {
        return vit->second;
      }
    }
    return mlir::Value();
  }

  void BindMLIRValue(const std::string &name, mlir::Value v) {
    if (mlir_value_table_stack.empty()) {
      mlir_value_table_stack.emplace_back();
    }
    mlir_value_table_stack.back()[name] = v;
  }

  void Clear();
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_SUNMMIO_MLIR_CONTEXT_H_
