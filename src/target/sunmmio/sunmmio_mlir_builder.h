#ifndef TVM_TL_TARGET_SUNMMIO_MLIR_BUILDER_H_
#define TVM_TL_TARGET_SUNMMIO_MLIR_BUILDER_H_

#include "codegen_sunmmio.h"
#include "sunmmio_mlir_context.h"

#include <memory>

namespace tvm {
namespace codegen {

class SunmmioMlirFunction;
class SunmmioMlirExpr;
class SunmmioMlirMemory;
class SunmmioMlirCall;

class SuvmSunmmioBuilder final : public SunMMIOBuilder {
public:
  SuvmSunmmioBuilder();
  ~SuvmSunmmioBuilder() final;

  void Init() final;
  void Clear() final;
  std::string Finish() final;

  void BeginModule() final;
  void EndModule() final;

  void BeginFunction(const std::string &name,
                     const std::vector<BuilderArg> &args) final;
  void EndFunction() final;
  void EmitReturn() final;

  SunMMIOValue ConstantInt(const std::string &result_name, int64_t v,
                           const SunMMIOType &type, DataType dtype) final;
  SunMMIOValue ConstantFloat(const std::string &result_name,
                             const std::string &literal,
                             const SunMMIOType &type, DataType dtype) final;

  SunMMIOValue Cast(const std::string &result_name, const SunMMIOValue &v,
                    const SunMMIOType &dst_type, DataType dst_dtype) final;

  SunMMIOValue Binary(const std::string &result_name, BinaryOp op,
                      ArithmeticFlavor flavor, const SunMMIOValue &a,
                      const SunMMIOValue &b, const SunMMIOType &result_type,
                      DataType dtype) final;

  SunMMIOValue Compare(const std::string &result_name, CompareOp op,
                       CompareDomain domain, const SunMMIOValue &a,
                       const SunMMIOValue &b,
                       const SunMMIOType &operand_type) final;

  SunMMIOValue Select(const std::string &result_name, const SunMMIOValue &cond,
                      const SunMMIOValue &tv, const SunMMIOValue &fv,
                      const SunMMIOType &result_type, DataType dtype) final;

  SunMMIOValue Alloc(const std::string &result_name,
                     const SunMMIOType &memref_type,
                     const std::vector<SunMMIOValue> &dyn_extents,
                     const std::string &scope_name, DataType dtype) final;

  SunMMIOValue Load(const std::string &result_name,
                    const std::string &buffer_handle,
                    const std::vector<SunMMIOValue> &indices,
                    const SunMMIOType &memref_type, DataType dtype,
                    const SunMMIOType &result_type) final;

  void Store(const SunMMIOValue &value, const std::string &buffer_handle,
             const std::vector<SunMMIOValue> &indices,
             const SunMMIOType &memref_type) final;

  SunMMIOValue Call(const std::string &result_name, const std::string &callee,
                    const std::vector<SunMMIOValue> &operands,
                    const std::vector<std::string> &string_args,
                    const std::string &category, DataType ret_dtype,
                    const SunMMIOType &ret_type) final;

  SunMMIOValue Ramp(const std::string &result_name, const SunMMIOValue &base,
                    const SunMMIOValue &stride, int lanes,
                    const SunMMIOType &elem_type, const SunMMIOType &vec_type,
                    DataType dtype) final;

  SunMMIOValue Broadcast(const std::string &result_name,
                         const SunMMIOValue &scalar, int lanes,
                         const SunMMIOType &scalar_type,
                         const SunMMIOType &vec_type, DataType dtype) final;

  void BeginFor(const std::string &iv, const SunMMIOValue &lb,
                const SunMMIOValue &ub, const SunMMIOValue &step,
                const ffi::Map<ffi::String, ffi::Any> &annotations,
                const std::vector<int64_t> &live_out_token_ids) final;
  void EndFor() final;

  void BeginIf(const SunMMIOValue &cond,
               const std::vector<int64_t> &live_out_token_ids) final;
  void BeginElse() final;
  void EndIf() final;

  void EmitAssert(const SunMMIOValue &cond, const std::string &msg_text) final;

  void PushLayoutScope(const TirLayoutMap &layout_map,
                       const TirLayoutMap &global_layout_map) final;
  void PopLayoutScope() final;
  ffi::Optional<tl::Layout> LookupLayout(const tir::Buffer &buffer) const final;
  void ApplyLayoutToType(const tir::Buffer &buffer,
                         SunMMIOType *type) const final;

private:
  SunmmioMlirContext ctx_;
  std::unique_ptr<SunmmioMlirFunction> function_;
  std::unique_ptr<SunmmioMlirExpr> expr_;
  std::unique_ptr<SunmmioMlirMemory> memory_;
  std::unique_ptr<SunmmioMlirCall> call_;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_SUNMMIO_MLIR_BUILDER_H_
