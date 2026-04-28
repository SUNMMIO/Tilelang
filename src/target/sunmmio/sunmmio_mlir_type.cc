#include "sunmmio_mlir_type.h"

#include "sunmmio_mlir_context.h"

namespace tvm {
namespace codegen {

SunmmioMlirType::SunmmioMlirType(SunmmioMlirContext &ctx) : ctx_(ctx) {}

mlir::Location SunmmioMlirType::Loc() const {
  return ctx_.builder.getUnknownLoc();
}

mlir::Type SunmmioMlirType::MapElementType(DataType dtype) const {
  if (dtype.is_bool()) {
    return mlir::Type::getFromOpaquePointer(
        ctx_.builder.getI1Type().getAsOpaquePointer());
  }
  if (dtype.is_float()) {
    if (dtype.bits() == 16) {
      return mlir::Type::getFromOpaquePointer(
          ctx_.builder.getF16Type().getAsOpaquePointer());
    }
    if (dtype.bits() == 32) {
      return mlir::Type::getFromOpaquePointer(
          ctx_.builder.getF32Type().getAsOpaquePointer());
    }
    if (dtype.bits() == 64) {
      return mlir::Type::getFromOpaquePointer(
          ctx_.builder.getF64Type().getAsOpaquePointer());
    }
  }
  if (dtype.is_bfloat16()) {
    return mlir::Type::getFromOpaquePointer(
        ctx_.builder.getBF16Type().getAsOpaquePointer());
  }
  if (dtype.is_int() || dtype.is_uint()) {
    return mlir::Type::getFromOpaquePointer(
        ctx_.builder.getIntegerType(dtype.bits()).getAsOpaquePointer());
  }
  return mlir::Type::getFromOpaquePointer(
      ctx_.builder.getIntegerType(64).getAsOpaquePointer());
}

mlir::Type SunmmioMlirType::MapType(const SunMMIOType &type) const {
  switch (type.kind) {
  case SunMMIOType::Kind::kScalar:
    return MapElementType(type.dtype);
  case SunMMIOType::Kind::kIndex:
    return mlir::Type::getFromOpaquePointer(
        ctx_.builder.getIndexType().getAsOpaquePointer());
  case SunMMIOType::Kind::kHandle:
    return mlir::Type::getFromOpaquePointer(
        ctx_.builder.getIntegerType(64).getAsOpaquePointer());
  case SunMMIOType::Kind::kVector: {
    mlir::Type elem = MapElementType(type.dtype);
    return mlir::Type::getFromOpaquePointer(
        mlir::VectorType::get({type.lanes}, elem).getAsOpaquePointer());
  }
  case SunMMIOType::Kind::kMemRef: {
    mlir::Type elem = MapElementType(type.dtype);
    std::vector<int64_t> shape;
    shape.reserve(type.shape.size());
    for (const PrimExpr &dim : type.shape) {
      const auto *imm = dim.as<tvm::tir::IntImmNode>();
      if (imm) {
        shape.push_back(static_cast<int64_t>(imm->value));
      } else {
        shape.push_back(mlir::ShapedType::kDynamic);
      }
    }
    return mlir::Type::getFromOpaquePointer(
        mlir::MemRefType::get(shape, elem).getAsOpaquePointer());
  }
  case SunMMIOType::Kind::kUnknown:
  default:
    return mlir::Type::getFromOpaquePointer(
        ctx_.builder.getIntegerType(64).getAsOpaquePointer());
  }
}

mlir::Value SunmmioMlirType::EnsureI1(mlir::Value v) {
  if (!v) {
    return mlir::arith::ConstantIntOp::create(ctx_.builder, Loc(), 1, 1);
  }
  mlir::Type ty = v.getType();
  if (ty.isInteger(1)) {
    return v;
  }
  if (auto int_ty = mlir::dyn_cast<mlir::IntegerType>(ty)) {
    auto zero = mlir::arith::ConstantIntOp::create(ctx_.builder, Loc(), 0,
                                                   int_ty.getWidth());
    return mlir::arith::CmpIOp::create(ctx_.builder, Loc(),
                                       mlir::arith::CmpIPredicate::ne, v, zero);
  }
  if (ty.isIndex()) {
    auto zero = mlir::arith::ConstantIndexOp::create(ctx_.builder, Loc(), 0);
    return mlir::arith::CmpIOp::create(ctx_.builder, Loc(),
                                       mlir::arith::CmpIPredicate::ne, v, zero);
  }
  if (auto float_ty = mlir::dyn_cast<mlir::FloatType>(ty)) {
    mlir::Type float_type =
        mlir::Type::getFromOpaquePointer(float_ty.getAsOpaquePointer());
    mlir::Value zero =
        mlir::arith::ConstantOp::create(
            ctx_.builder, Loc(), ctx_.builder.getFloatAttr(float_type, 0.0))
            .getResult();
    return mlir::arith::CmpFOp::create(
        ctx_.builder, Loc(), mlir::arith::CmpFPredicate::ONE, v, zero);
  }
  return mlir::arith::ConstantIntOp::create(ctx_.builder, Loc(), 1, 1);
}

mlir::Value SunmmioMlirType::EnsureIndex(mlir::Value v) {
  if (!v) {
    return mlir::arith::ConstantIndexOp::create(ctx_.builder, Loc(), 0);
  }
  if (v.getType().isIndex()) {
    return v;
  }
  if (mlir::isa<mlir::IntegerType>(v.getType())) {
    return mlir::arith::IndexCastOp::create(ctx_.builder, Loc(),
                                            ctx_.builder.getIndexType(), v);
  }
  return mlir::arith::ConstantIndexOp::create(ctx_.builder, Loc(), 0);
}

mlir::Value
SunmmioMlirType::ResolveValueOrCreatePlaceholder(const SunMMIOValue &v,
                                                 mlir::Type expected_type) {
  if (!v.value.empty()) {
    mlir::Value existing = ctx_.LookupMLIRValue(v.value);
    if (existing) {
      return existing;
    }
  }

  mlir::Type ty = expected_type ? expected_type : MapType(v.type);
  if (ty.isIndex()) {
    return mlir::arith::ConstantIndexOp::create(ctx_.builder, Loc(), 0);
  }
  if (ty.isInteger(1)) {
    return mlir::arith::ConstantIntOp::create(ctx_.builder, Loc(), 1, 1);
  }
  if (auto int_ty = mlir::dyn_cast<mlir::IntegerType>(ty)) {
    return mlir::arith::ConstantIntOp::create(ctx_.builder, Loc(), 0,
                                              int_ty.getWidth());
  }
  if (auto float_ty = mlir::dyn_cast<mlir::FloatType>(ty)) {
    mlir::Type float_type =
        mlir::Type::getFromOpaquePointer(float_ty.getAsOpaquePointer());
    return mlir::arith::ConstantOp::create(
               ctx_.builder, Loc(), ctx_.builder.getFloatAttr(float_type, 0.0))
        .getResult();
  }
  return mlir::Value();
}

} // namespace codegen
} // namespace tvm
