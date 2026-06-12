#include "sunmmio_mlir_expr.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Value.h"

namespace tvm {
namespace codegen {

namespace {

mlir::Location MapMlirLoc(SunmmioMlirContext &ctx) {
  return SunmmioMlirType(ctx).Loc();
}

mlir::Type MapMlirType(SunmmioMlirContext &ctx, const SunMMIOType &type) {
  return SunmmioMlirType(ctx).MapType(type);
}

mlir::arith::CmpFPredicate GetCmpFloatPredicate(CompareOp op) {
  switch (op) {
  case CompareOp::kEQ:
    return mlir::arith::CmpFPredicate::OEQ;
  case CompareOp::kNE:
    return mlir::arith::CmpFPredicate::ONE;
  case CompareOp::kLT:
    return mlir::arith::CmpFPredicate::OLT;
  case CompareOp::kLE:
    return mlir::arith::CmpFPredicate::OLE;
  case CompareOp::kGT:
    return mlir::arith::CmpFPredicate::OGT;
  case CompareOp::kGE:
    return mlir::arith::CmpFPredicate::OGE;
  }
  LOG(FATAL) << "Unsupported float compare op";
  throw;
}

mlir::arith::CmpIPredicate GetCmpIntegerPredicate(CompareOp op,
                                                  CompareDomain domain) {
  if (op == CompareOp::kEQ) {
    return mlir::arith::CmpIPredicate::eq;
  }
  if (op == CompareOp::kNE) {
    return mlir::arith::CmpIPredicate::ne;
  }
  if (domain == CompareDomain::kUnsignedInt) {
    switch (op) {
    case CompareOp::kLT:
      return mlir::arith::CmpIPredicate::ult;
    case CompareOp::kLE:
      return mlir::arith::CmpIPredicate::ule;
    case CompareOp::kGT:
      return mlir::arith::CmpIPredicate::ugt;
    case CompareOp::kGE:
      return mlir::arith::CmpIPredicate::uge;
    default:
      break;
    }
  }
  switch (op) {
  case CompareOp::kLT:
    return mlir::arith::CmpIPredicate::slt;
  case CompareOp::kLE:
    return mlir::arith::CmpIPredicate::sle;
  case CompareOp::kGT:
    return mlir::arith::CmpIPredicate::sgt;
  case CompareOp::kGE:
    return mlir::arith::CmpIPredicate::sge;
  default:
    break;
  }
  LOG(FATAL) << "Unsupported integer compare op";
  throw;
}

mlir::Value GetCastOp(SunmmioMlirContext &ctx, mlir::Value src_value,
                      const SunMMIOValue &src, const SunMMIOType &dst_type,
                      DataType dst_dtype) {
  mlir::Location loc = MapMlirLoc(ctx);
  mlir::Type dst_mlir_type = MapMlirType(ctx, dst_type);

  if (src.type.kind == dst_type.kind && src.dtype == dst_dtype &&
      src.type.lanes == dst_type.lanes) {
    return src_value;
  }
  if (dst_dtype.is_bool()) {
    return SunmmioMlirType(ctx).EnsureI1(src_value);
  }
  if (src.type.kind == SunMMIOType::Kind::kIndex ||
      dst_type.kind == SunMMIOType::Kind::kIndex) {
    return mlir::arith::IndexCastOp::create(ctx.builder, loc, dst_mlir_type,
                                            src_value)
        .getResult();
  }
  if (dst_dtype.is_float() || dst_dtype.is_bfloat16()) {
    if (src.dtype.is_float() || src.dtype.is_bfloat16()) {
      if (dst_dtype.bits() > src.dtype.bits()) {
        return mlir::arith::ExtFOp::create(ctx.builder, loc, dst_mlir_type,
                                           src_value, nullptr)
            .getResult();
      }
      return mlir::arith::TruncFOp::create(ctx.builder, loc, dst_mlir_type,
                                           src_value)
          .getResult();
    }
    if (src.dtype.is_uint() || src.dtype.is_bool()) {
      return mlir::arith::UIToFPOp::create(ctx.builder, loc, dst_mlir_type,
                                           src_value)
          .getResult();
    }
    return mlir::arith::SIToFPOp::create(ctx.builder, loc, dst_mlir_type,
                                         src_value)
        .getResult();
  }
  if (src.dtype.is_float() || src.dtype.is_bfloat16()) {
    if (dst_dtype.is_uint()) {
      return mlir::arith::FPToUIOp::create(ctx.builder, loc, dst_mlir_type,
                                           src_value)
          .getResult();
    }
    return mlir::arith::FPToSIOp::create(ctx.builder, loc, dst_mlir_type,
                                         src_value)
        .getResult();
  }
  if (dst_dtype.bits() > src.dtype.bits()) {
    if (src.dtype.is_uint() || src.dtype.is_bool()) {
      return mlir::arith::ExtUIOp::create(ctx.builder, loc, dst_mlir_type,
                                          src_value)
          .getResult();
    }
    return mlir::arith::ExtSIOp::create(ctx.builder, loc, dst_mlir_type,
                                        src_value)
        .getResult();
  }
  if (dst_dtype.bits() < src.dtype.bits()) {
    return mlir::arith::TruncIOp::create(ctx.builder, loc, dst_mlir_type,
                                         src_value)
        .getResult();
  }
  return mlir::arith::BitcastOp::create(ctx.builder, loc, dst_mlir_type,
                                        src_value)
      .getResult();
}

} // namespace

SunmmioMlirExpr::SunmmioMlirExpr(SunmmioMlirContext &ctx) : ctx_(ctx) {}

SunMMIOValue SunmmioMlirExpr::ConstantInt(const std::string &result_name,
                                          int64_t v, const SunMMIOType &type,
                                          DataType dtype) {
  ICHECK(ctx_.module)
      << "MLIR module must be initialized before lowering Sunmmio expressions";

  mlir::Type value_type = MapMlirType(ctx_, type);
  mlir::TypedAttr value_attr;
  if (value_type.isIndex()) {
    value_attr = ctx_.builder.getIndexAttr(v);
  } else {
    value_attr = ctx_.builder.getIntegerAttr(value_type, v);
  }

  auto constant_op = mlir::arith::ConstantOp::create(
      ctx_.builder, MapMlirLoc(ctx_), value_attr);
  mlir::Value constant_value = constant_op.getResult();

  ctx_.BindMLIRValue(result_name, constant_value);

  SunMMIOValue out{dtype, result_name, type};
  return out;
}

SunMMIOValue SunmmioMlirExpr::ConstantFloat(const std::string &result_name,
                                            const std::string &literal,
                                            const SunMMIOType &type,
                                            DataType dtype) {
  ICHECK(ctx_.module)
      << "MLIR module must be initialized before lowering Sunmmio expressions";

  double parsed_value = std::stod(literal);
  mlir::TypedAttr value_attr;
  mlir::Type value_type = MapMlirType(ctx_, type);
  value_attr = ctx_.builder.getFloatAttr(value_type, parsed_value);
  auto constant_op = mlir::arith::ConstantOp::create(
      ctx_.builder, MapMlirLoc(ctx_), value_attr);
  mlir::Value constant_value = constant_op.getResult();

  ctx_.BindMLIRValue(result_name, constant_value);

  SunMMIOValue out{dtype, result_name, type};
  return out;
}

SunMMIOValue SunmmioMlirExpr::Cast(const std::string &result_name,
                                   const SunMMIOValue &v,
                                   const SunMMIOType &dst_type,
                                   DataType dst_dtype) {
  ICHECK(ctx_.module)
      << "MLIR module must be initialized before lowering Sunmmio expressions";

  // mlir::Value src_value = ctx_.LookupMLIRValue(v.value);
  // ICHECK(src_value) << "Missing MLIR source value in Cast for `" << v.value
  //                   << "` while lowering result `" << result_name << "`";
  mlir::Value src_value =
      ctx_.LookupOrCreateFakeValue(v, "fake_missing_cast_src");

  /**
   * Materialize explicit arith casts so later MLIR lowering always sees a
   * concrete SSA value instead of a symbolic-only cast category.
   */
  mlir::Value cast_value = GetCastOp(ctx_, src_value, v, dst_type, dst_dtype);

  ctx_.BindMLIRValue(result_name, cast_value);
  SunMMIOValue out{dst_dtype, result_name, dst_type};
  return out;
}

SunMMIOValue SunmmioMlirExpr::Binary(const std::string &result_name,
                                     BinaryOp op, ArithmeticFlavor flavor,
                                     const SunMMIOValue &a,
                                     const SunMMIOValue &b,
                                     const SunMMIOType &result_type,
                                     DataType dtype) {
  ICHECK(ctx_.module)
      << "MLIR module must be initialized before lowering Sunmmio expressions";

  // mlir::Value lhs = ctx_.LookupMLIRValue(a.value);
  // mlir::Value rhs = ctx_.LookupMLIRValue(b.value);
  // ICHECK(lhs) << "Missing MLIR lhs value in Binary for `" << a.value
  //             << "` while lowering result `" << result_name << "`";
  // ICHECK(rhs) << "Missing MLIR rhs value in Binary for `" << b.value
  //             << "` while lowering result `" << result_name << "`";
  mlir::Value lhs = ctx_.LookupOrCreateFakeValue(a, "fake_missing_binary_lhs");
  mlir::Value rhs = ctx_.LookupOrCreateFakeValue(b, "fake_missing_binary_rhs");

  mlir::Location loc = MapMlirLoc(ctx_);
  mlir::Type result_mlir_type = MapMlirType(ctx_, result_type);
  mlir::Value binary_value;

  /**
   * Mirror the text backend opcode mapping with explicit arith ops so scalar
   * and vector values share the same binary semantics in MLIR form.
   */
  switch (op) {
  case BinaryOp::kAdd:
    if (flavor == ArithmeticFlavor::kFloat) {
      binary_value = mlir::arith::AddFOp::create(ctx_.builder, loc,
                                                 result_mlir_type, lhs, rhs)
                         .getResult();
    } else {
      binary_value = mlir::arith::AddIOp::create(ctx_.builder, loc,
                                                 result_mlir_type, lhs, rhs)
                         .getResult();
    }
    break;
  case BinaryOp::kSub:
    if (flavor == ArithmeticFlavor::kFloat) {
      binary_value = mlir::arith::SubFOp::create(ctx_.builder, loc,
                                                 result_mlir_type, lhs, rhs)
                         .getResult();
    } else {
      binary_value = mlir::arith::SubIOp::create(ctx_.builder, loc,
                                                 result_mlir_type, lhs, rhs)
                         .getResult();
    }
    break;
  case BinaryOp::kMul:
    if (flavor == ArithmeticFlavor::kFloat) {
      binary_value = mlir::arith::MulFOp::create(ctx_.builder, loc,
                                                 result_mlir_type, lhs, rhs)
                         .getResult();
    } else {
      binary_value = mlir::arith::MulIOp::create(ctx_.builder, loc,
                                                 result_mlir_type, lhs, rhs)
                         .getResult();
    }
    break;
  case BinaryOp::kDiv:
    if (flavor == ArithmeticFlavor::kFloat) {
      binary_value = mlir::arith::DivFOp::create(ctx_.builder, loc,
                                                 result_mlir_type, lhs, rhs)
                         .getResult();
    } else if (flavor == ArithmeticFlavor::kUnsignedInt) {
      binary_value = mlir::arith::DivUIOp::create(ctx_.builder, loc,
                                                  result_mlir_type, lhs, rhs)
                         .getResult();
    } else {
      binary_value = mlir::arith::DivSIOp::create(ctx_.builder, loc,
                                                  result_mlir_type, lhs, rhs)
                         .getResult();
    }
    break;
  case BinaryOp::kMod:
    if (flavor == ArithmeticFlavor::kFloat) {
      binary_value = mlir::arith::RemFOp::create(ctx_.builder, loc,
                                                 result_mlir_type, lhs, rhs)
                         .getResult();
    } else if (flavor == ArithmeticFlavor::kUnsignedInt) {
      binary_value = mlir::arith::RemUIOp::create(ctx_.builder, loc,
                                                  result_mlir_type, lhs, rhs)
                         .getResult();
    } else {
      binary_value = mlir::arith::RemSIOp::create(ctx_.builder, loc,
                                                  result_mlir_type, lhs, rhs)
                         .getResult();
    }
    break;
  case BinaryOp::kMin:
    if (flavor == ArithmeticFlavor::kFloat) {
      binary_value = mlir::arith::MinimumFOp::create(ctx_.builder, loc,
                                                     result_mlir_type, lhs, rhs)
                         .getResult();
    } else if (flavor == ArithmeticFlavor::kUnsignedInt) {
      binary_value = mlir::arith::MinUIOp::create(ctx_.builder, loc,
                                                  result_mlir_type, lhs, rhs)
                         .getResult();
    } else {
      binary_value = mlir::arith::MinSIOp::create(ctx_.builder, loc,
                                                  result_mlir_type, lhs, rhs)
                         .getResult();
    }
    break;
  case BinaryOp::kMax:
    if (flavor == ArithmeticFlavor::kFloat) {
      binary_value = mlir::arith::MaximumFOp::create(ctx_.builder, loc,
                                                     result_mlir_type, lhs, rhs)
                         .getResult();
    } else if (flavor == ArithmeticFlavor::kUnsignedInt) {
      binary_value = mlir::arith::MaxUIOp::create(ctx_.builder, loc,
                                                  result_mlir_type, lhs, rhs)
                         .getResult();
    } else {
      binary_value = mlir::arith::MaxSIOp::create(ctx_.builder, loc,
                                                  result_mlir_type, lhs, rhs)
                         .getResult();
    }
    break;
  case BinaryOp::kAnd:
    binary_value = mlir::arith::AndIOp::create(ctx_.builder, loc,
                                               result_mlir_type, lhs, rhs)
                       .getResult();
    break;
  case BinaryOp::kOr:
    binary_value = mlir::arith::OrIOp::create(ctx_.builder, loc,
                                              result_mlir_type, lhs, rhs)
                       .getResult();
    break;
  case BinaryOp::kXor:
    binary_value = mlir::arith::XOrIOp::create(ctx_.builder, loc,
                                               result_mlir_type, lhs, rhs)
                       .getResult();
    break;
  case BinaryOp::kShl:
    binary_value = mlir::arith::ShLIOp::create(ctx_.builder, loc,
                                               result_mlir_type, lhs, rhs)
                       .getResult();
    break;
  case BinaryOp::kShr:
    if (flavor == ArithmeticFlavor::kUnsignedInt) {
      binary_value = mlir::arith::ShRUIOp::create(ctx_.builder, loc,
                                                  result_mlir_type, lhs, rhs)
                         .getResult();
    } else {
      binary_value = mlir::arith::ShRSIOp::create(ctx_.builder, loc,
                                                  result_mlir_type, lhs, rhs)
                         .getResult();
    }
    break;
  }

  ctx_.BindMLIRValue(result_name, binary_value);
  SunMMIOValue out{dtype, result_name, result_type};
  return out;
}

SunMMIOValue SunmmioMlirExpr::Compare(const std::string &result_name,
                                      CompareOp op, CompareDomain domain,
                                      const SunMMIOValue &a,
                                      const SunMMIOValue &b,
                                      const SunMMIOType &operand_type) {
  ICHECK(ctx_.module)
      << "MLIR module must be initialized before lowering Sunmmio expressions";

  // mlir::Value lhs = ctx_.LookupMLIRValue(a.value);
  // mlir::Value rhs = ctx_.LookupMLIRValue(b.value);
  // ICHECK(lhs) << "Missing MLIR lhs value in Compare for `" << a.value
  //             << "` while lowering result `" << result_name << "`";
  // ICHECK(rhs) << "Missing MLIR rhs value in Compare for `" << b.value
  //             << "` while lowering result `" << result_name << "`";
  mlir::Value lhs = ctx_.LookupOrCreateFakeValue(a, "fake_missing_compare_lhs");
  mlir::Value rhs = ctx_.LookupOrCreateFakeValue(b, "fake_missing_compare_rhs");

  mlir::Location loc = MapMlirLoc(ctx_);
  mlir::Type bool_mlir_type = MapMlirType(
      ctx_, SunMMIOType{SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}});
  mlir::Value compare_value;

  if (domain == CompareDomain::kFloat) {
    compare_value =
        mlir::arith::CmpFOp::create(ctx_.builder, loc, bool_mlir_type,
                                    GetCmpFloatPredicate(op), lhs, rhs)
            .getResult();
  } else {
    compare_value = mlir::arith::CmpIOp::create(
                        ctx_.builder, loc, bool_mlir_type,
                        GetCmpIntegerPredicate(op, domain), lhs, rhs)
                        .getResult();
  }

  ctx_.BindMLIRValue(result_name, compare_value);
  SunMMIOType bool_type{SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}};
  return SunMMIOValue{DataType::Bool(), result_name, bool_type};
}

SunMMIOValue SunmmioMlirExpr::Select(const std::string &result_name,
                                     const SunMMIOValue &cond,
                                     const SunMMIOValue &tv,
                                     const SunMMIOValue &fv,
                                     const SunMMIOType &result_type,
                                     DataType dtype) {
  ICHECK(ctx_.module)
      << "MLIR module must be initialized before lowering Sunmmio expressions";

  // mlir::Value cond_value = ctx_.LookupMLIRValue(cond.value);
  // mlir::Value true_value = ctx_.LookupMLIRValue(tv.value);
  // mlir::Value false_value = ctx_.LookupMLIRValue(fv.value);
  // ICHECK(cond_value) << "Missing MLIR condition value in Select for `"
  //                    << cond.value << "` while lowering result `" <<
  //                    result_name
  //                    << "`";
  // ICHECK(true_value) << "Missing MLIR true value in Select for `" << tv.value
  //                    << "` while lowering result `" << result_name << "`";
  // ICHECK(false_value) << "Missing MLIR false value in Select for `" <<
  // fv.value
  //                     << "` while lowering result `" << result_name << "`";
  mlir::Value cond_value =
      ctx_.LookupOrCreateFakeValue(cond, "fake_missing_select_cond");
  mlir::Value true_value =
      ctx_.LookupOrCreateFakeValue(tv, "fake_missing_select_true");
  mlir::Value false_value =
      ctx_.LookupOrCreateFakeValue(fv, "fake_missing_select_false");

  mlir::Location loc = MapMlirLoc(ctx_);
  mlir::Type result_mlir_type = MapMlirType(ctx_, result_type);

  /**
   * Materialize the expression-form conditional with `arith.select` so the
   * chosen value remains in SSA form without introducing control-flow regions.
   */
  mlir::Value select_value =
      mlir::arith::SelectOp::create(ctx_.builder, loc, result_mlir_type,
                                    cond_value, true_value, false_value)
          .getResult();

  ctx_.BindMLIRValue(result_name, select_value);
  SunMMIOValue out{dtype, result_name, result_type};
  return out;
}

SunMMIOValue SunmmioMlirExpr::Ramp(const std::string &result_name,
                                   const SunMMIOValue &base,
                                   const SunMMIOValue &stride, int lanes,
                                   const SunMMIOType &elem_type,
                                   const SunMMIOType &vec_type,
                                   DataType dtype) {
  ICHECK(ctx_.module)
      << "MLIR module must be initialized before lowering Sunmmio expressions";
  mlir::TypedAttr value_attr = ctx_.builder.getIntegerAttr(
      mlir::Type::getFromOpaquePointer(
          ctx_.builder.getF32Type().getAsOpaquePointer()),
      0);

  auto fake_op = mlir::arith::ConstantOp::create(
      ctx_.builder, SunmmioMlirType(ctx_).MakeDebugLoc("fake_ramp"),
      value_attr);
  fake_op->setAttr("sunmmio.fake", ctx_.builder.getStringAttr("ramp"));
  mlir::Value ramp_value = fake_op.getResult();
  ctx_.BindMLIRValue(result_name, ramp_value);
  return SunMMIOValue{dtype, result_name, vec_type};
}

SunMMIOValue SunmmioMlirExpr::Broadcast(const std::string &result_name,
                                        const SunMMIOValue &scalar, int lanes,
                                        const SunMMIOType &scalar_type,
                                        const SunMMIOType &vec_type,
                                        DataType dtype) {
  (void)scalar;
  (void)lanes;
  (void)scalar_type;
  ICHECK(ctx_.module)
      << "MLIR module must be initialized before lowering Sunmmio expressions";
  mlir::TypedAttr value_attr = ctx_.builder.getIntegerAttr(
      mlir::Type::getFromOpaquePointer(
          ctx_.builder.getF32Type().getAsOpaquePointer()),
      0);
  auto fake_op = mlir::arith::ConstantOp::create(
      ctx_.builder, SunmmioMlirType(ctx_).MakeDebugLoc("fake_broadcast"),
      value_attr);
  fake_op->setAttr("sunmmio.fake", ctx_.builder.getStringAttr("broadcast"));
  mlir::Value broadcast_value = fake_op.getResult();
  ctx_.BindMLIRValue(result_name, broadcast_value);
  return SunMMIOValue{dtype, result_name, vec_type};
}

} // namespace codegen
} // namespace tvm
