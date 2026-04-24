#include "sunmmio_mlir_function.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "npuir/Dialect/SUVM/IR/Dialect.h"

namespace tvm {
namespace codegen {

SunmmioMlirFunction::SunmmioMlirFunction(SunmmioMlirContext &ctx) : ctx_(ctx) {}

mlir::Location SunmmioMlirFunction::Loc() const {
  return ctx_.builder.getUnknownLoc();
}

mlir::Type SunmmioMlirFunction::MapElementType(DataType dtype) const {
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

mlir::Type SunmmioMlirFunction::MapType(const SunMMIOType &type) const {
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

mlir::Value SunmmioMlirFunction::EnsureI1(mlir::Value v) {
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

mlir::Value SunmmioMlirFunction::EnsureIndex(mlir::Value v) {
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
SunmmioMlirFunction::ResolveValueOrCreatePlaceholder(const SunMMIOValue &v,
                                                     mlir::Type expected_type) {
  auto it = mlir_value_table_.find(v.value);
  if (it != mlir_value_table_.end()) {
    return it->second;
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

void SunmmioMlirFunction::BeginModule() {
  ctx_.module_open = true;
  ctx_.mlir_ctx.getOrLoadDialect<mlir::suvm::SUVMDialect>();
  ctx_.mlir_ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx_.mlir_ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  ctx_.mlir_ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx_.mlir_ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  ctx_.mlir_ctx.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  ctx_.mlir_ctx.getOrLoadDialect<mlir::affine::AffineDialect>();

  ctx_.module = mlir::ModuleOp::create(ctx_.builder.getUnknownLoc());
  ctx_.builder.setInsertionPointToEnd(ctx_.module->getBody());
  current_func_ = mlir::func::FuncOp();
  for_stack_.clear();
  if_stack_.clear();
  mlir_value_table_.clear();
}

void SunmmioMlirFunction::EndModule() {
  ctx_.module_open = false;
  if (failed(mlir::verify(*ctx_.module))) {
    ctx_.module->emitError("Module verification failed");
  }
}

void SunmmioMlirFunction::BeginFunction(const std::string &name,
                                        const std::vector<BuilderArg> &args) {
  ctx_.function_open = true;
  ctx_.current_function = name;

  mlir::SmallVector<mlir::Type, 8> arg_types;
  arg_types.reserve(args.size());
  for (const BuilderArg &arg : args) {
    arg_types.push_back(MapType(arg.type));
  }
  mlir::FunctionType func_type =
      ctx_.builder.getFunctionType(arg_types, mlir::TypeRange{});
  mlir::func::FuncOp func = mlir::func::FuncOp::create(Loc(), name, func_type);
  ctx_.module->push_back(func);
  current_func_ = func;

  mlir_value_table_.clear();
  for_stack_.clear();
  if_stack_.clear();

  mlir::Block *entry = func.addEntryBlock();
  ctx_.builder.setInsertionPointToStart(entry);
  for (int i = 0, e = static_cast<int>(args.size()); i < e; ++i) {
    mlir_value_table_[args[i].name] = entry->getArgument(i);
  }
}

void SunmmioMlirFunction::EndFunction() {
  ctx_.function_open = false;
  ctx_.current_function.clear();
  current_func_ = mlir::func::FuncOp();
  for_stack_.clear();
  if_stack_.clear();
  mlir_value_table_.clear();
  ctx_.builder.setInsertionPointToEnd(ctx_.module->getBody());
}

void SunmmioMlirFunction::EmitReturn() {
  mlir::func::ReturnOp::create(ctx_.builder, Loc());
}

void SunmmioMlirFunction::BeginFor(const std::string &iv,
                                   const SunMMIOValue &lb,
                                   const SunMMIOValue &ub,
                                   const SunMMIOValue &step) {
  mlir::Value lb_v = EnsureIndex(
      ResolveValueOrCreatePlaceholder(lb, ctx_.builder.getIndexType()));
  mlir::Value ub_v = EnsureIndex(
      ResolveValueOrCreatePlaceholder(ub, ctx_.builder.getIndexType()));
  mlir::Value step_v = EnsureIndex(
      ResolveValueOrCreatePlaceholder(step, ctx_.builder.getIndexType()));
  mlir::scf::ForOp for_op =
      mlir::scf::ForOp::create(ctx_.builder, Loc(), lb_v, ub_v, step_v);
  for_stack_.push_back(for_op);
  mlir_value_table_[iv] = for_op.getInductionVar();
  ctx_.builder.setInsertionPointToStart(for_op.getBody());
  ctx_.insertion_point_stack.push_back("for");
}

void SunmmioMlirFunction::EndFor() {
  if (!for_stack_.empty()) {
    mlir::scf::ForOp for_op = for_stack_.back();
    for_stack_.pop_back();
    ctx_.builder.setInsertionPointAfter(for_op);
  }
  if (!ctx_.insertion_point_stack.empty()) {
    ctx_.insertion_point_stack.pop_back();
  }
}

void SunmmioMlirFunction::BeginIf(const SunMMIOValue &cond) {
  mlir::Value cond_v =
      EnsureI1(ResolveValueOrCreatePlaceholder(cond, ctx_.builder.getI1Type()));
  mlir::scf::IfOp if_op =
      mlir::scf::IfOp::create(ctx_.builder, Loc(), cond_v, true);
  if_stack_.push_back(IfFrame{if_op, false});
  ctx_.builder.setInsertionPointToStart(&if_op.getThenRegion().front());
  ctx_.insertion_point_stack.push_back("if.then");
}

void SunmmioMlirFunction::BeginElse() {
  if (!if_stack_.empty()) {
    IfFrame &frame = if_stack_.back();
    frame.in_else = true;
    ctx_.builder.setInsertionPointToStart(&frame.op.getElseRegion().front());
  }
  if (!ctx_.insertion_point_stack.empty()) {
    ctx_.insertion_point_stack.back() = "if.else";
  }
}

void SunmmioMlirFunction::EndIf() {
  if (!if_stack_.empty()) {
    IfFrame frame = if_stack_.back();
    if_stack_.pop_back();
    ctx_.builder.setInsertionPointAfter(frame.op);
  }
  if (!ctx_.insertion_point_stack.empty()) {
    ctx_.insertion_point_stack.pop_back();
  }
}

void SunmmioMlirFunction::EmitAssert(const SunMMIOValue &cond,
                                     const std::string &msg_text) {
  mlir::Value cond_v =
      EnsureI1(ResolveValueOrCreatePlaceholder(cond, ctx_.builder.getI1Type()));
  mlir::cf::AssertOp::create(ctx_.builder, Loc(), cond_v,
                             ctx_.builder.getStringAttr(msg_text));
}

} // namespace codegen
} // namespace tvm
