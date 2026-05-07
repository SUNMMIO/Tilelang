#include "sunmmio_mlir_type.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "npuir/Dialect/SUVM/IR/Attributes.h"
#include "npuir/Dialect/SUVM/IR/Types.h"
#include "sunmmio_mlir_context.h"
#include "tvm/runtime/logging.h"

namespace tvm {
namespace codegen {

std::vector<int64_t> ExtractShape(const SunMMIOType &type) {
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
  return shape;
}

std::vector<int64_t> BuildRowMajorStrides(llvm::ArrayRef<int64_t> shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
    if (shape[static_cast<size_t>(i + 1)] == mlir::ShapedType::kDynamic ||
        strides[static_cast<size_t>(i + 1)] == mlir::ShapedType::kDynamic) {
      strides[static_cast<size_t>(i)] = mlir::ShapedType::kDynamic;
      continue;
    }
    strides[static_cast<size_t>(i)] =
        shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
  }
  return strides;
}

std::vector<uint8_t> BuildFlatDimLevels(size_t rank) {
  return std::vector<uint8_t>(rank, 1);
}

mlir::suvm::MemorySpaceAttr MapMemorySpaceAttr(mlir::MLIRContext *ctx,
                                               const std::string &space) {
  using mlir::suvm::MemorySpace;
  if (space.empty() || space == "global") {
    return mlir::suvm::MemorySpaceAttr::get(ctx, MemorySpace::global);
  }
  if (space == "shared.asram" || space == "asram") {
    return mlir::suvm::MemorySpaceAttr::get(ctx, MemorySpace::asram);
  }
  if (space == "shared.wsram" || space == "wsram") {
    return mlir::suvm::MemorySpaceAttr::get(ctx, MemorySpace::wsram);
  }
  if (space == "shared.rsram" || space == "rsram") {
    return mlir::suvm::MemorySpaceAttr::get(ctx, MemorySpace::rsram);
  }
  LOG(FATAL) << "Unsupported SUVM memory space: " << space;
  TVM_FFI_UNREACHABLE();
}

void ValidateLayout(const SunMMIOType &type,
                    llvm::ArrayRef<int64_t> layout_hshape,
                    llvm::ArrayRef<int64_t> layout_hstride,
                    llvm::ArrayRef<uint8_t> dim_levels) {
  ICHECK_EQ(layout_hshape.size(), layout_hstride.size())
      << "SunMMIOType layout_hshape/layout_hstride size mismatch: "
      << layout_hshape.size() << " vs " << layout_hstride.size();
  ICHECK_EQ(dim_levels.size(), type.shape.size())
      << "SunMMIOType layout_dim_levels rank mismatch: " << dim_levels.size()
      << " vs " << type.shape.size();

  size_t expected_flattened_size = 0;
  for (uint8_t level : dim_levels) {
    ICHECK_GT(level, 0) << "SunMMIOType layout_dim_levels must be > 0";
    expected_flattened_size += static_cast<size_t>(level);
  }
  ICHECK_EQ(layout_hshape.size(), expected_flattened_size)
      << "SunMMIOType flattened layout size mismatch: got "
      << layout_hshape.size() << ", expected " << expected_flattened_size;
}

SunmmioMlirType::SunmmioMlirType(SunmmioMlirContext &ctx) : ctx_(ctx) {}

mlir::Location SunmmioMlirType::Loc() const {
  return ctx_.builder.getUnknownLoc();
}

mlir::Location SunmmioMlirType::MakeDebugLoc(const std::string &tag) const {
  return mlir::NameLoc::get(ctx_.builder.getStringAttr(tag),
                            ctx_.builder.getUnknownLoc());
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
    std::vector<int64_t> shape = ExtractShape(type);
    return mlir::Type::getFromOpaquePointer(
        mlir::MemRefType::get(shape, elem).getAsOpaquePointer());
  }
  case SunMMIOType::Kind::kMemTensor: {
    mlir::Type elem = MapElementType(type.dtype);
    std::vector<int64_t> shape = ExtractShape(type);

    std::vector<int64_t> layout_hshape =
        type.layout_hshape.empty() ? shape : type.layout_hshape;
    std::vector<int64_t> layout_hstride =
        type.layout_hstride.empty() ? BuildRowMajorStrides(layout_hshape)
                                    : type.layout_hstride;
    std::vector<uint8_t> dim_levels = type.layout_dim_levels.empty()
                                          ? BuildFlatDimLevels(shape.size())
                                          : type.layout_dim_levels;

    ValidateLayout(type, layout_hshape, layout_hstride, dim_levels);

    mlir::suvm::LayoutAttr layout = mlir::suvm::LayoutAttr::get(
        &ctx_.mlir_ctx, layout_hshape, layout_hstride, dim_levels);
    mlir::suvm::MemorySpaceAttr memory_scope =
        MapMemorySpaceAttr(&ctx_.mlir_ctx, type.memory_scope);
    mlir::suvm::MemTensorType memtensor = mlir::suvm::MemTensorType::get(
        shape, elem, layout, memory_scope, type.byte_offset);
    return mlir::Type::getFromOpaquePointer(memtensor.getAsOpaquePointer());
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
