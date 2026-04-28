#ifndef TVM_TL_TARGET_SUNMMIO_MLIR_FUNCTION_H_
#define TVM_TL_TARGET_SUNMMIO_MLIR_FUNCTION_H_

#include "sunmmio_mlir_context.h"
#include "sunmmio_mlir_type.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace tvm {
namespace codegen {

class SunmmioMlirFunction {
public:
  explicit SunmmioMlirFunction(SunmmioMlirContext &ctx);

  void BeginModule();
  void EndModule();

  void BeginFunction(const std::string &name,
                     const std::vector<BuilderArg> &args);
  void EndFunction();
  void EmitReturn();

  void BeginFor(const std::string &iv, const SunMMIOValue &lb,
                const SunMMIOValue &ub, const SunMMIOValue &step,
                const ffi::Map<ffi::String, ffi::Any> &annotations);
  void EndFor();

  void BeginIf(const SunMMIOValue &cond);
  void BeginElse();
  void EndIf();

  void EmitAssert(const SunMMIOValue &cond, const std::string &msg_text);

private:
  struct IfFrame {
    mlir::scf::IfOp op;
    bool in_else{false};
  };

  SunmmioMlirContext &ctx_;
  SunmmioMlirType type_;
  mlir::func::FuncOp current_func_;
  std::vector<IfFrame> if_stack_;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_SUNMMIO_MLIR_FUNCTION_H_
