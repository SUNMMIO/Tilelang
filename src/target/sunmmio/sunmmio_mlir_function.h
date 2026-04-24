#ifndef TVM_TL_TARGET_SUNMMIO_MLIR_FUNCTION_H_
#define TVM_TL_TARGET_SUNMMIO_MLIR_FUNCTION_H_

#include "sunmmio_mlir_context.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"

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
                const SunMMIOValue &ub, const SunMMIOValue &step);
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

  mlir::Location Loc() const;
  mlir::Type MapType(const SunMMIOType &type) const;
  mlir::Type MapElementType(DataType dtype) const;
  mlir::Value EnsureI1(mlir::Value v);
  mlir::Value EnsureIndex(mlir::Value v);
  mlir::Value ResolveValueOrCreatePlaceholder(const SunMMIOValue &v,
                                              mlir::Type expected_type);

  SunmmioMlirContext &ctx_;
  mlir::func::FuncOp current_func_;
  std::vector<mlir::scf::ForOp> for_stack_;
  std::vector<IfFrame> if_stack_;
  std::unordered_map<std::string, mlir::Value> mlir_value_table_;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_SUNMMIO_MLIR_FUNCTION_H_
