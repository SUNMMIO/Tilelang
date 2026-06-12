#ifndef TVM_TL_TARGET_SUNMMIO_MLIR_EXPR_H_
#define TVM_TL_TARGET_SUNMMIO_MLIR_EXPR_H_

#include "sunmmio_mlir_context.h"

namespace tvm {
namespace codegen {

class SunmmioMlirExpr {
public:
  explicit SunmmioMlirExpr(SunmmioMlirContext &ctx);

  SunMMIOValue ConstantInt(const std::string &result_name, int64_t v,
                           const SunMMIOType &type, DataType dtype);
  SunMMIOValue ConstantFloat(const std::string &result_name,
                             const std::string &literal,
                             const SunMMIOType &type, DataType dtype);
  SunMMIOValue Cast(const std::string &result_name, const SunMMIOValue &v,
                    const SunMMIOType &dst_type, DataType dst_dtype);

  SunMMIOValue Binary(const std::string &result_name, BinaryOp op,
                      ArithmeticFlavor flavor, const SunMMIOValue &a,
                      const SunMMIOValue &b, const SunMMIOType &result_type,
                      DataType dtype);

  SunMMIOValue Compare(const std::string &result_name, CompareOp op,
                       CompareDomain domain, const SunMMIOValue &a,
                       const SunMMIOValue &b, const SunMMIOType &operand_type);

  SunMMIOValue Select(const std::string &result_name, const SunMMIOValue &cond,
                      const SunMMIOValue &tv, const SunMMIOValue &fv,
                      const SunMMIOType &result_type, DataType dtype);

  SunMMIOValue Ramp(const std::string &result_name, const SunMMIOValue &base,
                    const SunMMIOValue &stride, int lanes,
                    const SunMMIOType &elem_type, const SunMMIOType &vec_type,
                    DataType dtype);

  SunMMIOValue Broadcast(const std::string &result_name,
                         const SunMMIOValue &scalar, int lanes,
                         const SunMMIOType &scalar_type,
                         const SunMMIOType &vec_type, DataType dtype);

private:
  SunmmioMlirContext &ctx_;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_SUNMMIO_MLIR_EXPR_H_
