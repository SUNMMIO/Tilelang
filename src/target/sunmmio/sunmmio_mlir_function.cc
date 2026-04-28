#include "sunmmio_mlir_function.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "npuir/Dialect/SUVM/IR/Dialect.h"

namespace tvm {
namespace codegen {

SunmmioMlirFunction::SunmmioMlirFunction(SunmmioMlirContext &ctx)
    : ctx_(ctx), type_(ctx) {}

void SunmmioMlirFunction::BeginModule() {
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
  ctx_.for_stack.clear();
  if_stack_.clear();
  ctx_.ClearMLIRValueScopes();
}

void SunmmioMlirFunction::EndModule() {
  if (failed(mlir::verify(*ctx_.module))) {
    ctx_.module->emitError("Module verification failed");
  }
}

void SunmmioMlirFunction::BeginFunction(const std::string &name,
                                        const std::vector<BuilderArg> &args) {
  mlir::SmallVector<mlir::Type, 8> arg_types;
  arg_types.reserve(args.size());
  for (const BuilderArg &arg : args) {
    arg_types.push_back(type_.MapType(arg.type));
  }
  mlir::FunctionType func_type =
      ctx_.builder.getFunctionType(arg_types, mlir::TypeRange{});
  mlir::func::FuncOp func =
      mlir::func::FuncOp::create(type_.Loc(), name, func_type);
  ctx_.module->push_back(func);
  current_func_ = func;

  ctx_.ClearMLIRValueScopes();
  ctx_.PushMLIRValueScope();
  if_stack_.clear();
  ctx_.for_stack.clear();

  mlir::Block *entry = func.addEntryBlock();
  ctx_.builder.setInsertionPointToStart(entry);
  for (int i = 0, e = static_cast<int>(args.size()); i < e; ++i) {
    ctx_.BindMLIRValue(args[i].name, entry->getArgument(i));
  }
}

void SunmmioMlirFunction::EndFunction() {
  current_func_ = mlir::func::FuncOp();
  if_stack_.clear();
  ctx_.for_stack.clear();
  ctx_.ClearMLIRValueScopes();
  ctx_.builder.setInsertionPointToEnd(ctx_.module->getBody());
}

void SunmmioMlirFunction::EmitReturn() {
  mlir::func::ReturnOp::create(ctx_.builder, type_.Loc());
}

void SunmmioMlirFunction::BeginFor(
    const std::string &iv, const SunMMIOValue &lb, const SunMMIOValue &ub,
    const SunMMIOValue &step,
    const ffi::Map<ffi::String, ffi::Any> &annotations) {
  mlir::Value lb_v = type_.EnsureIndex(
      type_.ResolveValueOrCreatePlaceholder(lb, ctx_.builder.getIndexType()));
  mlir::Value ub_v = type_.EnsureIndex(
      type_.ResolveValueOrCreatePlaceholder(ub, ctx_.builder.getIndexType()));
  mlir::Value step_v = type_.EnsureIndex(
      type_.ResolveValueOrCreatePlaceholder(step, ctx_.builder.getIndexType()));
  mlir::scf::ForOp for_op =
      mlir::scf::ForOp::create(ctx_.builder, type_.Loc(), lb_v, ub_v, step_v);
  ctx_.for_stack.push_back(SunmmioMlirContext::ForFrame{for_op, annotations});
  ctx_.PushMLIRValueScope();
  ctx_.BindMLIRValue(iv, for_op.getInductionVar());
  ctx_.builder.setInsertionPointToStart(for_op.getBody());
}

void SunmmioMlirFunction::EndFor() {
  if (!ctx_.for_stack.empty()) {
    mlir::scf::ForOp for_op = ctx_.for_stack.back().op;
    ctx_.for_stack.pop_back();
    ctx_.builder.setInsertionPointAfter(for_op);
  }
  ctx_.PopMLIRValueScope();
}

void SunmmioMlirFunction::BeginIf(const SunMMIOValue &cond) {
  mlir::Value cond_v = type_.EnsureI1(
      type_.ResolveValueOrCreatePlaceholder(cond, ctx_.builder.getI1Type()));
  mlir::scf::IfOp if_op =
      mlir::scf::IfOp::create(ctx_.builder, type_.Loc(), cond_v, true);
  if_stack_.push_back(IfFrame{if_op, false});
  ctx_.PushMLIRValueScope();
  ctx_.builder.setInsertionPointToStart(&if_op.getThenRegion().front());
}

void SunmmioMlirFunction::BeginElse() {
  if (!if_stack_.empty()) {
    IfFrame &frame = if_stack_.back();
    frame.in_else = true;
    ctx_.PopMLIRValueScope();
    ctx_.PushMLIRValueScope();
    ctx_.builder.setInsertionPointToStart(&frame.op.getElseRegion().front());
  }
}

void SunmmioMlirFunction::EndIf() {
  if (!if_stack_.empty()) {
    IfFrame frame = if_stack_.back();
    if_stack_.pop_back();
    ctx_.builder.setInsertionPointAfter(frame.op);
  }
  ctx_.PopMLIRValueScope();
}

void SunmmioMlirFunction::EmitAssert(const SunMMIOValue &cond,
                                     const std::string &msg_text) {
  mlir::Value cond_v = type_.EnsureI1(
      type_.ResolveValueOrCreatePlaceholder(cond, ctx_.builder.getI1Type()));
  mlir::cf::AssertOp::create(ctx_.builder, type_.Loc(), cond_v,
                             ctx_.builder.getStringAttr(msg_text));
}

} // namespace codegen
} // namespace tvm
