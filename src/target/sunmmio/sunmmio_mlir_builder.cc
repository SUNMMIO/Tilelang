#include "sunmmio_mlir_builder.h"

#include "sunmmio_mlir_call.h"
#include "sunmmio_mlir_expr.h"
#include "sunmmio_mlir_function.h"
#include "sunmmio_mlir_memory.h"

namespace tvm {
namespace codegen {

SuvmSunmmioBuilder::SuvmSunmmioBuilder()
    : function_(std::make_unique<SunmmioMlirFunction>(ctx_)),
      expr_(std::make_unique<SunmmioMlirExpr>(ctx_)),
      memory_(std::make_unique<SunmmioMlirMemory>(ctx_)),
      call_(std::make_unique<SunmmioMlirCall>(ctx_)) {}

SuvmSunmmioBuilder::~SuvmSunmmioBuilder() = default;

void SuvmSunmmioBuilder::Init() { Clear(); }

void SuvmSunmmioBuilder::Clear() { ctx_.Clear(); }

std::string SuvmSunmmioBuilder::Finish() {
  // TODO(@sunmmio-mlir): serialize real NPU-IR/MLIR module.
  return "module {\n  // TODO: SuvmSunmmioBuilder NPU-IR lowering\n}\n";
}

void SuvmSunmmioBuilder::BeginModule() { function_->BeginModule(); }
void SuvmSunmmioBuilder::EndModule() { function_->EndModule(); }

void SuvmSunmmioBuilder::BeginFunction(const std::string& name,
                                       const std::vector<BuilderArg>& args) {
  function_->BeginFunction(name, args);
}

void SuvmSunmmioBuilder::EndFunction() { function_->EndFunction(); }
void SuvmSunmmioBuilder::EmitReturn() { function_->EmitReturn(); }

SunMMIOValue SuvmSunmmioBuilder::ConstantInt(const std::string& result_name,
                                             int64_t v,
                                             const SunMMIOType& type,
                                             DataType dtype) {
  return expr_->ConstantInt(result_name, v, type, dtype);
}

SunMMIOValue SuvmSunmmioBuilder::ConstantFloat(const std::string& result_name,
                                               const std::string& literal,
                                               const SunMMIOType& type,
                                               DataType dtype) {
  return expr_->ConstantFloat(result_name, literal, type, dtype);
}

SunMMIOValue SuvmSunmmioBuilder::Cast(const std::string& result_name,
                                      const SunMMIOValue& v,
                                      const SunMMIOType& dst_type,
                                      DataType dst_dtype) {
  return expr_->Cast(result_name, v, dst_type, dst_dtype);
}

SunMMIOValue SuvmSunmmioBuilder::Binary(const std::string& result_name,
                                        BinaryOp op, ArithmeticFlavor flavor,
                                        const SunMMIOValue& a,
                                        const SunMMIOValue& b,
                                        const SunMMIOType& result_type,
                                        DataType dtype) {
  return expr_->Binary(result_name, op, flavor, a, b, result_type, dtype);
}

SunMMIOValue SuvmSunmmioBuilder::Compare(const std::string& result_name,
                                         CompareOp op, CompareDomain domain,
                                         const SunMMIOValue& a,
                                         const SunMMIOValue& b,
                                         const SunMMIOType& operand_type) {
  return expr_->Compare(result_name, op, domain, a, b, operand_type);
}

SunMMIOValue SuvmSunmmioBuilder::Select(const std::string& result_name,
                                        const SunMMIOValue& cond,
                                        const SunMMIOValue& tv,
                                        const SunMMIOValue& fv,
                                        const SunMMIOType& result_type,
                                        DataType dtype) {
  return expr_->Select(result_name, cond, tv, fv, result_type, dtype);
}

SunMMIOValue SuvmSunmmioBuilder::Alloc(const std::string& result_name,
                                       const SunMMIOType& memref_type,
                                       const std::vector<SunMMIOValue>& dyn_extents,
                                       const std::string& scope_name,
                                       DataType dtype) {
  return memory_->Alloc(result_name, memref_type, dyn_extents, scope_name, dtype);
}

SunMMIOValue SuvmSunmmioBuilder::Load(const std::string& result_name,
                                      const std::string& buffer_handle,
                                      const std::vector<SunMMIOValue>& indices,
                                      const SunMMIOType& memref_type,
                                      DataType dtype,
                                      const SunMMIOType& result_type) {
  return memory_->Load(result_name, buffer_handle, indices, memref_type, dtype,
                       result_type);
}

void SuvmSunmmioBuilder::Store(const SunMMIOValue& value,
                               const std::string& buffer_handle,
                               const std::vector<SunMMIOValue>& indices,
                               const SunMMIOType& memref_type) {
  memory_->Store(value, buffer_handle, indices, memref_type);
}

SunMMIOValue SuvmSunmmioBuilder::Call(const std::string& result_name,
                                      const std::string& callee,
                                      const std::vector<SunMMIOValue>& operands,
                                      const std::vector<std::string>& string_args,
                                      const std::string& category,
                                      DataType ret_dtype,
                                      const SunMMIOType& ret_type) {
  return call_->Call(result_name, callee, operands, string_args, category,
                     ret_dtype, ret_type);
}

SunMMIOValue SuvmSunmmioBuilder::Ramp(const std::string& result_name,
                                      const SunMMIOValue& base,
                                      const SunMMIOValue& stride, int lanes,
                                      const SunMMIOType& elem_type,
                                      const SunMMIOType& vec_type,
                                      DataType dtype) {
  return expr_->Ramp(result_name, base, stride, lanes, elem_type, vec_type,
                     dtype);
}

SunMMIOValue SuvmSunmmioBuilder::Broadcast(const std::string& result_name,
                                           const SunMMIOValue& scalar,
                                           int lanes,
                                           const SunMMIOType& scalar_type,
                                           const SunMMIOType& vec_type,
                                           DataType dtype) {
  return expr_->Broadcast(result_name, scalar, lanes, scalar_type, vec_type,
                          dtype);
}

void SuvmSunmmioBuilder::BeginFor(const std::string& iv, const SunMMIOValue& lb,
                                  const SunMMIOValue& ub,
                                  const SunMMIOValue& step) {
  function_->BeginFor(iv, lb, ub, step);
}

void SuvmSunmmioBuilder::EndFor() { function_->EndFor(); }

void SuvmSunmmioBuilder::BeginIf(const SunMMIOValue& cond) {
  function_->BeginIf(cond);
}

void SuvmSunmmioBuilder::BeginElse() { function_->BeginElse(); }

void SuvmSunmmioBuilder::EndIf() { function_->EndIf(); }

void SuvmSunmmioBuilder::EmitAssert(const SunMMIOValue& cond,
                                    const std::string& msg_text) {
  function_->EmitAssert(cond, msg_text);
}

}  // namespace codegen
}  // namespace tvm
