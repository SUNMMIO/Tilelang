#include "sunmmio_mlir_function.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "npuir/Dialect/SUVM/IR/Attributes.h"
#include "npuir/Dialect/SUVM/IR/Dialect.h"
#include "npuir/Dialect/SUVM/IR/Types.h"

#include <algorithm>

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
  ctx_.module->getOperation()->setAttr(
      "suvm.device_arch", mlir::suvm::DeviceArchAttr::get(
                              &ctx_.mlir_ctx, mlir::suvm::DeviceArch::a4e));
  ctx_.builder.setInsertionPointToEnd(ctx_.module->getBody());
  current_func_ = mlir::func::FuncOp();
  ctx_.for_stack.clear();
  ctx_.if_stack.clear();
  ctx_.control_flow_stack.clear();
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
  ctx_.if_stack.clear();
  ctx_.for_stack.clear();
  ctx_.control_flow_stack.clear();

  mlir::Block *entry = func.addEntryBlock();
  ctx_.builder.setInsertionPointToStart(entry);
  for (int i = 0, e = static_cast<int>(args.size()); i < e; ++i) {
    ctx_.BindMLIRValue(args[i].name, entry->getArgument(i));
  }
}

void SunmmioMlirFunction::EndFunction() {
  current_func_ = mlir::func::FuncOp();
  ctx_.if_stack.clear();
  ctx_.for_stack.clear();
  ctx_.control_flow_stack.clear();
  ctx_.ClearMLIRValueScopes();
  ctx_.builder.setInsertionPointToEnd(ctx_.module->getBody());
}

void SunmmioMlirFunction::EmitReturn() {
  mlir::func::ReturnOp::create(ctx_.builder, type_.Loc());
}

void SunmmioMlirFunction::BeginFor(
    const std::string &iv, const SunMMIOValue &lb, const SunMMIOValue &ub,
    const SunMMIOValue &step,
    const ffi::Map<ffi::String, ffi::Any> &annotations,
    const std::vector<int64_t> &live_out_token_ids) {
  mlir::Value lb_v = type_.EnsureIndex(
      type_.ResolveValueOrCreatePlaceholder(lb, ctx_.builder.getIndexType()));
  mlir::Value ub_v = type_.EnsureIndex(
      type_.ResolveValueOrCreatePlaceholder(ub, ctx_.builder.getIndexType()));
  mlir::Value step_v = type_.EnsureIndex(
      type_.ResolveValueOrCreatePlaceholder(step, ctx_.builder.getIndexType()));

  mlir::SmallVector<mlir::Value, 8> init_args;
  init_args.reserve(live_out_token_ids.size());
  for (int64_t token_id : live_out_token_ids) {
    auto it = ctx_.token_by_id.find(token_id);
    // If this token already exists at the current insertion point, use it as
    // the initial iter_arg for scf.for.
    if (it != ctx_.token_by_id.end() && it->second) {
      init_args.push_back(it->second);
      continue;
    }
    // Otherwise, create a null !suvm.token as the initial iter_arg so the loop
    // always has a well-typed token to carry.
    mlir::Type token_ty = mlir::suvm::TokenType::get(&ctx_.mlir_ctx);
    mlir::OperationState st(type_.Loc(), "suvm.null_token");
    st.addTypes(token_ty);
    mlir::Operation *op = ctx_.builder.create(st);
    init_args.push_back(op->getResult(0));
  }

  mlir::scf::ForOp for_op = mlir::scf::ForOp::create(
      ctx_.builder, type_.Loc(), lb_v, ub_v, step_v, init_args);

  SunmmioMlirContext::ForFrame frame;
  frame.op = for_op;
  frame.annotations = annotations;
  frame.live_out_token_ids = live_out_token_ids;
  frame.iter_tokens.assign(for_op.getRegionIterArgs().begin(),
                           for_op.getRegionIterArgs().end());
  frame.produced_tokens.assign(frame.iter_tokens.size(), mlir::Value());
  for (int i = 0, e = static_cast<int>(live_out_token_ids.size()); i < e; ++i) {
    frame.token_id_to_index[frame.live_out_token_ids[i]] = i;
  }
  ctx_.for_stack.push_back(std::move(frame));
  ctx_.control_flow_stack.push_back(SunmmioMlirContext::ControlNode{
      SunmmioMlirContext::ControlKind::kFor,
      static_cast<int>(ctx_.for_stack.size()) - 1});
  ctx_.PushMLIRValueScope();
  ctx_.BindMLIRValue(iv, for_op.getInductionVar());
  ctx_.builder.setInsertionPointToStart(for_op.getBody());
}

void SunmmioMlirFunction::BeginFor(
    const std::string &iv, const SunMMIOValue &lb, const SunMMIOValue &ub,
    const SunMMIOValue &step,
    const ffi::Map<ffi::String, ffi::Any> &annotations,
    const std::vector<SunMMIOValue> &live_out_values) {
  mlir::Value lb_v = type_.EnsureIndex(
      type_.ResolveValueOrCreatePlaceholder(lb, ctx_.builder.getIndexType()));
  mlir::Value ub_v = type_.EnsureIndex(
      type_.ResolveValueOrCreatePlaceholder(ub, ctx_.builder.getIndexType()));
  mlir::Value step_v = type_.EnsureIndex(
      type_.ResolveValueOrCreatePlaceholder(step, ctx_.builder.getIndexType()));

  mlir::SmallVector<mlir::Value, 8> init_args;
  init_args.reserve(live_out_values.size());
  for (const SunMMIOValue &value : live_out_values) {
    mlir::Type expected_type = type_.MapType(value.type);
    mlir::Value init = ctx_.LookupMLIRValue(value.value);
    if (!init) {
      init = type_.ResolveValueOrCreatePlaceholder(value, expected_type);
    }
    init_args.push_back(init);
  }

  mlir::scf::ForOp for_op = mlir::scf::ForOp::create(
      ctx_.builder, type_.Loc(), lb_v, ub_v, step_v, init_args);

  SunmmioMlirContext::ForFrame frame;
  frame.op = for_op;
  frame.annotations = annotations;
  frame.live_out_value_names.reserve(live_out_values.size());
  for (const SunMMIOValue &value : live_out_values) {
    frame.live_out_value_names.push_back(value.value);
  }
  frame.iter_tokens.assign(for_op.getRegionIterArgs().begin(),
                           for_op.getRegionIterArgs().end());
  frame.produced_tokens.assign(frame.iter_tokens.size(), mlir::Value());
  ctx_.for_stack.push_back(std::move(frame));
  ctx_.control_flow_stack.push_back(SunmmioMlirContext::ControlNode{
      SunmmioMlirContext::ControlKind::kFor,
      static_cast<int>(ctx_.for_stack.size()) - 1});
  ctx_.PushMLIRValueScope();
  ctx_.BindMLIRValue(iv, for_op.getInductionVar());
  SunmmioMlirContext::ForFrame &active_frame = ctx_.for_stack.back();
  for (int i = 0,
           e = static_cast<int>(active_frame.live_out_value_names.size());
       i < e; ++i) {
    ctx_.BindMLIRValue(active_frame.live_out_value_names[i],
                       active_frame.iter_tokens[i]);
  }
  ctx_.builder.setInsertionPointToStart(for_op.getBody());
}

void SunmmioMlirFunction::EndFor() {
  if (ctx_.for_stack.empty()) {
    if (ctx_.module) {
      ctx_.module->emitError("EndFor called without a matching BeginFor");
    }
    ctx_.PopMLIRValueScope();
    return;
  }

  if (ctx_.control_flow_stack.empty() ||
      ctx_.control_flow_stack.back().kind !=
          SunmmioMlirContext::ControlKind::kFor ||
      ctx_.control_flow_stack.back().index !=
          static_cast<int>(ctx_.for_stack.size()) - 1) {
    if (ctx_.module) {
      ctx_.module->emitError("EndFor control_flow_stack mismatch");
    }
  } else {
    ctx_.control_flow_stack.pop_back();
  }

  SunmmioMlirContext::ForFrame frame = std::move(ctx_.for_stack.back());
  ctx_.for_stack.pop_back();
  mlir::scf::ForOp for_op = frame.op;

  // Build the values to yield from the loop body:
  // - Prefer tokens produced inside the body.
  // - Otherwise, forward the incoming iter_args unchanged.
  mlir::SmallVector<mlir::Value, 8> yielded;
  yielded.reserve(frame.iter_tokens.size());
  for (int i = 0, e = static_cast<int>(frame.iter_tokens.size()); i < e; ++i) {
    mlir::Value v = frame.produced_tokens[i] ? frame.produced_tokens[i]
                                             : frame.iter_tokens[i];
    yielded.push_back(v);
  }

  // Ensure the loop body terminator is an scf.yield with the computed operands.
  mlir::Block *body = for_op.getBody();
  mlir::Operation *terminator = body->getTerminator();
  mlir::scf::YieldOp yield_op =
      terminator ? mlir::dyn_cast<mlir::scf::YieldOp>(terminator)
                 : mlir::scf::YieldOp();
  if (yield_op) {
    yield_op.getOperation()->setOperands(yielded);
  } else {
    ctx_.builder.setInsertionPointToEnd(body);
    mlir::scf::YieldOp::create(ctx_.builder, type_.Loc(), yielded);
  }

  ctx_.builder.setInsertionPointAfter(for_op);

  // Restore token_by_id mappings that were locally overridden inside this loop.
  for (const auto &kv : frame.saved_token_by_id) {
    int64_t token_id = kv.first;
    const SunmmioMlirContext::SavedToken &saved = kv.second;
    if (saved.existed) {
      ctx_.token_by_id[token_id] = saved.value;
    } else {
      ctx_.token_by_id.erase(token_id);
    }
  }
  for (int i = 0, e = static_cast<int>(frame.live_out_token_ids.size()); i < e;
       ++i) {
    int64_t token_id = frame.live_out_token_ids[i];
    mlir::Value v = for_op.getResult(i);
    // Publish the final loop result for this live-out token_id to the outer
    // scope.
    ctx_.token_by_id[token_id] = v;
    for (auto nit = ctx_.control_flow_stack.rbegin();
         nit != ctx_.control_flow_stack.rend(); ++nit) {
      if (nit->kind == SunmmioMlirContext::ControlKind::kFor) {
        SunmmioMlirContext::ForFrame &outer = ctx_.for_stack[nit->index];
        auto tit = outer.token_id_to_index.find(token_id);
        if (tit != outer.token_id_to_index.end()) {
          int idx = tit->second;
          if (idx >= 0 &&
              idx < static_cast<int>(outer.produced_tokens.size())) {
            outer.produced_tokens[idx] = v;
          }
          break;
        }
      } else {
        SunmmioMlirContext::IfFrame &outer = ctx_.if_stack[nit->index];
        auto tit = outer.token_id_to_index.find(token_id);
        if (tit != outer.token_id_to_index.end()) {
          int idx = tit->second;
          if (idx >= 0 &&
              idx < static_cast<int>(outer.produced_tokens.size())) {
            outer.produced_tokens[idx] = v;
          }
          break;
        }
      }
    }
  }
  ctx_.PopMLIRValueScope();
  for (int i = 0, e = static_cast<int>(frame.live_out_value_names.size());
       i < e; ++i) {
    ctx_.BindMLIRValue(frame.live_out_value_names[i], for_op.getResult(i));
  }
}

void SunmmioMlirFunction::BeginIf(
    const SunMMIOValue &cond, const std::vector<int64_t> &live_out_token_ids) {
  mlir::Value cond_v = type_.EnsureI1(
      type_.ResolveValueOrCreatePlaceholder(cond, ctx_.builder.getI1Type()));
  mlir::Type token_ty = mlir::suvm::TokenType::get(&ctx_.mlir_ctx);
  mlir::SmallVector<mlir::Type, 8> result_types;
  result_types.assign(live_out_token_ids.size(), token_ty);

  mlir::scf::IfOp if_op = mlir::scf::IfOp::create(ctx_.builder, type_.Loc(),
                                                  result_types, cond_v, true);

  ctx_.if_stack.emplace_back();
  SunmmioMlirContext::IfFrame &frame = ctx_.if_stack.back();
  frame.op = if_op;
  frame.in_else = false;
  frame.live_out_token_ids = live_out_token_ids;
  frame.base_tokens.reserve(live_out_token_ids.size());
  frame.produced_tokens.reserve(live_out_token_ids.size());
  // For each live-out token id, capture the token value visible at the if
  // entry.
  for (int i = 0, e = static_cast<int>(live_out_token_ids.size()); i < e; ++i) {
    int64_t token_id = live_out_token_ids[i];
    frame.token_id_to_index[token_id] = i;
    mlir::Value base;
    // Prefer the nearest enclosing scf.for iter_arg if this token is
    // loop-carried.
    for (auto fit = ctx_.for_stack.rbegin(); fit != ctx_.for_stack.rend();
         ++fit) {
      auto tit = fit->token_id_to_index.find(token_id);
      if (tit != fit->token_id_to_index.end()) {
        int idx = tit->second;
        if (idx >= 0 && idx < static_cast<int>(fit->iter_tokens.size())) {
          base = fit->iter_tokens[idx];
          break;
        }
      }
    }
    // Otherwise fall back to the surrounding scope mapping.
    if (!base) {
      auto it = ctx_.token_by_id.find(token_id);
      if (it != ctx_.token_by_id.end() && it->second) {
        base = it->second;
      }
    }
    frame.base_tokens.push_back(base);
  }

  ctx_.control_flow_stack.push_back(SunmmioMlirContext::ControlNode{
      SunmmioMlirContext::ControlKind::kIf,
      static_cast<int>(ctx_.if_stack.size()) - 1});
  ctx_.PushMLIRValueScope();
  ctx_.builder.setInsertionPointToStart(&if_op.getThenRegion().front());

  // Initialize then-branch tokens from base_tokens; materialize suvm.null_token
  // when missing so the branch can use/yield well-typed tokens.
  for (int i = 0, e = static_cast<int>(frame.live_out_token_ids.size()); i < e;
       ++i) {
    int64_t token_id = frame.live_out_token_ids[i];
    mlir::Value tok = frame.base_tokens[i];
    if (!tok) {
      mlir::OperationState st(type_.Loc(), "suvm.null_token");
      st.addTypes(token_ty);
      mlir::Operation *op = ctx_.builder.create(st);
      tok = op->getResult(0);
    }
    frame.produced_tokens.push_back(tok);
    if (token_id >= 0) {
      ctx_.token_by_id[token_id] = tok;
    }
  }
}

void SunmmioMlirFunction::BeginIf(
    const SunMMIOValue &cond,
    const std::vector<SunMMIOValue> &live_out_values) {
  mlir::Value cond_v = type_.EnsureI1(
      type_.ResolveValueOrCreatePlaceholder(cond, ctx_.builder.getI1Type()));
  mlir::SmallVector<mlir::Type, 8> result_types;
  result_types.reserve(live_out_values.size());
  for (const SunMMIOValue &value : live_out_values) {
    result_types.push_back(type_.MapType(value.type));
  }

  mlir::scf::IfOp if_op = mlir::scf::IfOp::create(ctx_.builder, type_.Loc(),
                                                  result_types, cond_v, true);

  ctx_.if_stack.emplace_back();
  SunmmioMlirContext::IfFrame &frame = ctx_.if_stack.back();
  frame.op = if_op;
  frame.in_else = false;
  frame.live_out_value_names.reserve(live_out_values.size());
  frame.base_tokens.reserve(live_out_values.size());
  frame.produced_tokens.reserve(live_out_values.size());
  for (const SunMMIOValue &value : live_out_values) {
    frame.live_out_value_names.push_back(value.value);
    mlir::Value base = ctx_.LookupMLIRValue(value.value);
    if (!base) {
      base = type_.ResolveValueOrCreatePlaceholder(value,
                                                   type_.MapType(value.type));
    }
    frame.base_tokens.push_back(base);
  }

  ctx_.control_flow_stack.push_back(SunmmioMlirContext::ControlNode{
      SunmmioMlirContext::ControlKind::kIf,
      static_cast<int>(ctx_.if_stack.size()) - 1});
  ctx_.PushMLIRValueScope();
  ctx_.builder.setInsertionPointToStart(&if_op.getThenRegion().front());

  for (int i = 0, e = static_cast<int>(frame.live_out_value_names.size());
       i < e; ++i) {
    frame.produced_tokens.push_back(frame.base_tokens[i]);
    ctx_.BindMLIRValue(frame.live_out_value_names[i], frame.base_tokens[i]);
  }
}

void SunmmioMlirFunction::BeginElse() {
  if (ctx_.if_stack.empty()) {
    if (ctx_.module) {
      ctx_.module->emitError("BeginElse called without a matching BeginIf");
    }
    return;
  }
  SunmmioMlirContext::IfFrame &frame = ctx_.if_stack.back();
  if (frame.in_else) {
    if (ctx_.module) {
      ctx_.module->emitError("BeginElse called twice for the same scf.if");
    }
    return;
  }
  frame.then_yield_tokens = frame.produced_tokens;
  // Restore ctx_.token_by_id overrides made in the then-branch before entering
  // else.
  for (const auto &kv : frame.saved_token_by_id) {
    int64_t token_id = kv.first;
    const SunmmioMlirContext::SavedToken &saved = kv.second;
    if (saved.existed) {
      ctx_.token_by_id[token_id] = saved.value;
    } else {
      ctx_.token_by_id.erase(token_id);
    }
  }
  frame.saved_token_by_id.clear();
  frame.in_else = true;
  ctx_.PopMLIRValueScope();
  ctx_.PushMLIRValueScope();
  ctx_.builder.setInsertionPointToStart(&frame.op.getElseRegion().front());
  mlir::Type token_ty = mlir::suvm::TokenType::get(&ctx_.mlir_ctx);
  frame.produced_tokens.clear();
  frame.produced_tokens.reserve(frame.live_out_token_ids.size());
  if (!frame.live_out_value_names.empty()) {
    frame.produced_tokens.reserve(frame.live_out_value_names.size());
    for (int i = 0, e = static_cast<int>(frame.live_out_value_names.size());
         i < e; ++i) {
      ctx_.BindMLIRValue(frame.live_out_value_names[i], frame.base_tokens[i]);
      frame.produced_tokens.push_back(frame.base_tokens[i]);
    }
    return;
  }
  // Seed the else-branch scope from base_tokens (create suvm.null_token when
  // missing) and publish them into ctx_.token_by_id for in-branch lookups.
  for (int i = 0, e = static_cast<int>(frame.live_out_token_ids.size()); i < e;
       ++i) {
    int64_t token_id = frame.live_out_token_ids[i];
    mlir::Value tok = frame.base_tokens[i];
    if (!tok) {
      mlir::OperationState st(type_.Loc(), "suvm.null_token");
      st.addTypes(token_ty);
      mlir::Operation *op = ctx_.builder.create(st);
      tok = op->getResult(0);
    }
    ctx_.token_by_id[token_id] = tok;
    frame.produced_tokens.push_back(tok);
  }
}

void SunmmioMlirFunction::EndIf() {
  if (ctx_.if_stack.empty()) {
    if (ctx_.module) {
      ctx_.module->emitError("EndIf called without a matching BeginIf");
    }
    ctx_.PopMLIRValueScope();
    return;
  }

  if (ctx_.control_flow_stack.empty() ||
      ctx_.control_flow_stack.back().kind !=
          SunmmioMlirContext::ControlKind::kIf ||
      ctx_.control_flow_stack.back().index !=
          static_cast<int>(ctx_.if_stack.size()) - 1) {
    if (ctx_.module) {
      ctx_.module->emitError("EndIf control_flow_stack mismatch");
    }
  } else {
    ctx_.control_flow_stack.pop_back();
  }

  SunmmioMlirContext::IfFrame frame = std::move(ctx_.if_stack.back());
  ctx_.if_stack.pop_back();
  mlir::scf::IfOp if_op = frame.op;

  mlir::SmallVector<mlir::Value, 8> then_yield;
  mlir::SmallVector<mlir::Value, 8> else_yield;
  if (frame.in_else) {
    then_yield.assign(frame.then_yield_tokens.begin(),
                      frame.then_yield_tokens.end());
    else_yield.assign(frame.produced_tokens.begin(),
                      frame.produced_tokens.end());
  } else {
    then_yield.assign(frame.produced_tokens.begin(),
                      frame.produced_tokens.end());
    // No explicit else-branch emitted: yield the incoming (base) tokens.
    else_yield.reserve(frame.base_tokens.size());
    mlir::Type token_ty = mlir::suvm::TokenType::get(&ctx_.mlir_ctx);
    mlir::Block &else_body = if_op.getElseRegion().front();
    for (mlir::Value tok : frame.base_tokens) {
      if (!tok) {
        // Fill missing tokens with an explicit null token value.
        ctx_.builder.setInsertionPointToEnd(&else_body);
        mlir::OperationState st(type_.Loc(), "suvm.null_token");
        st.addTypes(token_ty);
        mlir::Operation *op = ctx_.builder.create(st);
        tok = op->getResult(0);
      }
      else_yield.push_back(tok);
    }
  }

  auto set_yield = [&](mlir::Region &region,
                       const mlir::SmallVector<mlir::Value, 8> &yielded) {
    mlir::Block &body = region.front();
    mlir::scf::YieldOp yield_op;
    if (!body.empty()) {
      // Check whether the last operation is already an scf.yield.
      yield_op = mlir::dyn_cast<mlir::scf::YieldOp>(body.back());
    }
    if (yield_op) {
      yield_op.getOperation()->setOperands(yielded);
    } else {
      ctx_.builder.setInsertionPointToEnd(&body);
      mlir::scf::YieldOp::create(ctx_.builder, type_.Loc(), yielded);
    }
  };

  set_yield(if_op.getThenRegion(), then_yield);
  set_yield(if_op.getElseRegion(), else_yield);

  ctx_.builder.setInsertionPointAfter(if_op);
  // Restore token_by_id to the state captured before entering this if.
  for (const auto &kv : frame.saved_token_by_id) {
    int64_t token_id = kv.first;
    const SunmmioMlirContext::SavedToken &saved = kv.second;
    if (saved.existed) {
      ctx_.token_by_id[token_id] = saved.value;
    } else {
      ctx_.token_by_id.erase(token_id);
    }
  }
  for (int i = 0, e = static_cast<int>(frame.live_out_token_ids.size()); i < e;
       ++i) {
    int64_t token_id = frame.live_out_token_ids[i];
    mlir::Value v = if_op.getResult(i);
    ctx_.token_by_id[token_id] = v;
    for (auto nit = ctx_.control_flow_stack.rbegin();
         nit != ctx_.control_flow_stack.rend(); ++nit) {
      if (nit->kind == SunmmioMlirContext::ControlKind::kFor) {
        SunmmioMlirContext::ForFrame &outer = ctx_.for_stack[nit->index];
        auto tit = outer.token_id_to_index.find(token_id);
        if (tit != outer.token_id_to_index.end()) {
          int idx = tit->second;
          if (idx >= 0 &&
              idx < static_cast<int>(outer.produced_tokens.size())) {
            outer.produced_tokens[idx] = v;
          }
          break;
        }
      } else {
        SunmmioMlirContext::IfFrame &outer = ctx_.if_stack[nit->index];
        auto tit = outer.token_id_to_index.find(token_id);
        if (tit != outer.token_id_to_index.end()) {
          int idx = tit->second;
          if (idx >= 0 &&
              idx < static_cast<int>(outer.produced_tokens.size())) {
            outer.produced_tokens[idx] = v;
          }
          break;
        }
      }
    }
  }
  ctx_.PopMLIRValueScope();
  for (int i = 0, e = static_cast<int>(frame.live_out_value_names.size());
       i < e; ++i) {
    ctx_.BindMLIRValue(frame.live_out_value_names[i], if_op.getResult(i));
  }
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
