#include "sunmmio_mlir_expr.h"

namespace tvm {
namespace codegen {

SunmmioMlirExpr::SunmmioMlirExpr(SunmmioMlirContext &ctx) : ctx_(ctx) {}

SunMMIOValue SunmmioMlirExpr::ConstantInt(const std::string &result_name,
                                          int64_t v, const SunMMIOType &type,
                                          DataType dtype) {
  (void)v;
  SunMMIOValue out{dtype, result_name, type};
  ctx_.value_symbol_table[result_name] = out;
  return out;
}

SunMMIOValue SunmmioMlirExpr::ConstantFloat(const std::string &result_name,
                                            const std::string &literal,
                                            const SunMMIOType &type,
                                            DataType dtype) {
  (void)literal;
  SunMMIOValue out{dtype, result_name, type};
  ctx_.value_symbol_table[result_name] = out;
  return out;
}

SunMMIOValue SunmmioMlirExpr::Cast(const std::string &result_name,
                                   const SunMMIOValue &v,
                                   const SunMMIOType &dst_type,
                                   DataType dst_dtype) {
  (void)v;
  SunMMIOValue out{dst_dtype, result_name, dst_type};
  ctx_.value_symbol_table[result_name] = out;
  return out;
}

SunMMIOValue SunmmioMlirExpr::Binary(const std::string &result_name,
                                     BinaryOp op, ArithmeticFlavor flavor,
                                     const SunMMIOValue &a,
                                     const SunMMIOValue &b,
                                     const SunMMIOType &result_type,
                                     DataType dtype) {
  (void)op;
  (void)flavor;
  (void)a;
  (void)b;
  SunMMIOValue out{dtype, result_name, result_type};
  ctx_.value_symbol_table[result_name] = out;
  return out;
}

SunMMIOValue SunmmioMlirExpr::Compare(const std::string &result_name,
                                      CompareOp op, CompareDomain domain,
                                      const SunMMIOValue &a,
                                      const SunMMIOValue &b,
                                      const SunMMIOType &operand_type) {
  (void)op;
  (void)domain;
  (void)a;
  (void)b;
  (void)operand_type;
  SunMMIOType bool_type{SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}};
  SunMMIOValue out{DataType::Bool(), result_name, bool_type};
  ctx_.value_symbol_table[result_name] = out;
  return out;
}

SunMMIOValue SunmmioMlirExpr::Select(const std::string &result_name,
                                     const SunMMIOValue &cond,
                                     const SunMMIOValue &tv,
                                     const SunMMIOValue &fv,
                                     const SunMMIOType &result_type,
                                     DataType dtype) {
  (void)cond;
  (void)tv;
  (void)fv;
  SunMMIOValue out{dtype, result_name, result_type};
  ctx_.value_symbol_table[result_name] = out;
  return out;
}

SunMMIOValue SunmmioMlirExpr::Ramp(const std::string &result_name,
                                   const SunMMIOValue &base,
                                   const SunMMIOValue &stride, int lanes,
                                   const SunMMIOType &elem_type,
                                   const SunMMIOType &vec_type,
                                   DataType dtype) {
  (void)base;
  (void)stride;
  (void)lanes;
  (void)elem_type;
  SunMMIOValue out{dtype, result_name, vec_type};
  ctx_.value_symbol_table[result_name] = out;
  return out;
}

SunMMIOValue SunmmioMlirExpr::Broadcast(const std::string &result_name,
                                        const SunMMIOValue &scalar, int lanes,
                                        const SunMMIOType &scalar_type,
                                        const SunMMIOType &vec_type,
                                        DataType dtype) {
  (void)scalar;
  (void)lanes;
  (void)scalar_type;
  SunMMIOValue out{dtype, result_name, vec_type};
  ctx_.value_symbol_table[result_name] = out;
  return out;
}

} // namespace codegen
} // namespace tvm
