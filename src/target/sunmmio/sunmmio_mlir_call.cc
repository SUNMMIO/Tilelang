#include "sunmmio_mlir_call.h"

#include "sunmmio_mlir_type.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Operation.h"
#include "npuir/Dialect/SUVM/IR/Attributes.h"
#include "npuir/Dialect/SUVM/IR/Types.h"

#include <tvm/runtime/logging.h>

#include <cstdint>

namespace tvm {
namespace codegen {

SunmmioMlirCall::SunmmioMlirCall(SunmmioMlirContext &ctx) : ctx_(ctx) {}

SunMMIOValue SunmmioMlirCall::Call(const std::string &result_name,
                                   const std::string &callee,
                                   const std::vector<SunMMIOValue> &operands,
                                   const std::vector<std::string> &string_args,
                                   const std::string &category,
                                   DataType ret_dtype,
                                   const SunMMIOType &ret_type) {
  SunmmioMlirType type(ctx_);
  auto parse_token_id = [&]() -> int64_t {
    for (const std::string &s : string_args) {
      if (s.rfind("token_id=", 0) == 0) {
        return std::stoll(s.substr(9));
      }
    }
    return -1;
  };
  auto record_token_by_id = [&](int64_t token_id, mlir::Value produced) {
    if (token_id < 0 || !produced) {
      return;
    }
    bool recorded = false;
    for (auto nit = ctx_.control_flow_stack.rbegin();
         nit != ctx_.control_flow_stack.rend(); ++nit) {
      if (nit->kind == SunmmioMlirContext::ControlKind::kFor) {
        SunmmioMlirContext::ForFrame &frame = ctx_.for_stack[nit->index];
        auto tit = frame.token_id_to_index.find(token_id);
        if (tit != frame.token_id_to_index.end()) {
          int idx = tit->second;
          if (idx >= 0 &&
              idx < static_cast<int>(frame.produced_tokens.size())) {
            frame.produced_tokens[idx] = produced;
          }
          recorded = true;
          break;
        }
      } else {
        SunmmioMlirContext::IfFrame &frame = ctx_.if_stack[nit->index];
        auto tit = frame.token_id_to_index.find(token_id);
        if (tit != frame.token_id_to_index.end()) {
          int idx = tit->second;
          if (idx >= 0 &&
              idx < static_cast<int>(frame.produced_tokens.size())) {
            frame.produced_tokens[idx] = produced;
          }
          recorded = true;
          break;
        }
      }
    }

    if (!recorded && !ctx_.control_flow_stack.empty()) {
      SunmmioMlirContext::ControlNode &node = ctx_.control_flow_stack.back();
      auto snapshot_if_needed = [&](auto &frame) {
        if (frame.saved_token_by_id.find(token_id) ==
            frame.saved_token_by_id.end()) {
          SunmmioMlirContext::SavedToken saved;
          auto it = ctx_.token_by_id.find(token_id);
          if (it != ctx_.token_by_id.end()) {
            saved.existed = true;
            saved.value = it->second;
          }
          frame.saved_token_by_id[token_id] = saved;
        }
      };
      if (node.kind == SunmmioMlirContext::ControlKind::kFor) {
        snapshot_if_needed(ctx_.for_stack[node.index]);
      } else {
        snapshot_if_needed(ctx_.if_stack[node.index]);
      }
    }

    ctx_.token_by_id[token_id] = produced;
  };

  if (callee == "tl.sync_null_token") {
    int64_t token_id = parse_token_id();
    if (token_id < 0) {
      LOG(FATAL) << "tl.sync_null_token requires token_id";
      TVM_FFI_UNREACHABLE();
    }
    mlir::Type token_ty = mlir::suvm::TokenType::get(&ctx_.mlir_ctx);
    mlir::OperationState st(type.Loc(), "suvm.null_token");
    st.addTypes(token_ty);
    mlir::Operation *op = ctx_.builder.create(st);
    if (!result_name.empty() && op && op->getNumResults() == 1) {
      ctx_.BindMLIRValue(result_name, op->getResult(0));
    }
    if (op && op->getNumResults() == 1) {
      record_token_by_id(token_id, op->getResult(0));
    }
    return SunMMIOValue{ret_dtype, result_name, ret_type};
  }

  if (callee == "tl.wait_token") {
    int64_t token_id = parse_token_id();
    if (token_id < 0) {
      LOG(FATAL) << "tl.wait_token requires token_id";
      TVM_FFI_UNREACHABLE();
    }
    mlir::Value tok;
    for (auto nit = ctx_.control_flow_stack.rbegin();
         nit != ctx_.control_flow_stack.rend(); ++nit) {
      if (nit->kind != SunmmioMlirContext::ControlKind::kFor) {
        continue;
      }
      SunmmioMlirContext::ForFrame &frame = ctx_.for_stack[nit->index];
      auto tit = frame.token_id_to_index.find(token_id);
      if (tit == frame.token_id_to_index.end()) {
        continue;
      }
      int idx = tit->second;
      if (idx >= 0 && idx < static_cast<int>(frame.iter_tokens.size())) {
        tok = frame.iter_tokens[idx];
        break;
      }
    }
    if (!tok) {
      auto it = ctx_.token_by_id.find(token_id);
      if (it == ctx_.token_by_id.end()) {
        LOG(FATAL) << "tl.wait_token token_id=" << token_id
                   << " has no corresponding sync_token.";
        TVM_FFI_UNREACHABLE();
      }
      tok = it->second;
    }
    mlir::OperationState st(type.Loc(), "suvm.wait_token");
    st.addOperands(tok);
    (void)ctx_.builder.create(st);
    return SunMMIOValue{ret_dtype, result_name, ret_type};
  }

  if (callee == "tl.dma_copy") {
    // (1) Build types (token/layout/memory spaces/MemTensor/TileView)
    mlir::Type token_ty = mlir::suvm::TokenType::get(&ctx_.mlir_ctx);
    mlir::Type elem_ty = ctx_.builder.getIntegerType(8);
    mlir::SmallVector<int64_t, 2> shape = {1, 1};
    mlir::SmallVector<int64_t, 2> stride = {1, 1};
    mlir::SmallVector<uint8_t, 2> dim_levels = {1, 1};
    auto layout =
        mlir::suvm::LayoutAttr::get(&ctx_.mlir_ctx, shape, stride, dim_levels);
    auto ms_global = mlir::suvm::MemorySpaceAttr::get(
        &ctx_.mlir_ctx, mlir::suvm::MemorySpace::global);
    auto ms_rsram = mlir::suvm::MemorySpaceAttr::get(
        &ctx_.mlir_ctx, mlir::suvm::MemorySpace::rsram);
    mlir::Type mem_src_ty =
        mlir::suvm::MemTensorType::get(shape, elem_ty, layout, ms_global, 0);
    mlir::Type mem_dst_ty =
        mlir::suvm::MemTensorType::get(shape, elem_ty, layout, ms_rsram, 0);
    mlir::Type tv_ty =
        mlir::suvm::TileViewType::get(&ctx_.mlir_ctx, shape, elem_ty);

    // (2) Emit IR: alloc -> get_partitioned_tile_view -> copy_async
    mlir::OperationState alloc_src_st(type.Loc(), "suvm.alloc");
    alloc_src_st.addTypes(mem_src_ty);
    mlir::Operation *alloc_src_op = ctx_.builder.create(alloc_src_st);

    mlir::OperationState alloc_dst_st(type.Loc(), "suvm.alloc");
    alloc_dst_st.addTypes(mem_dst_ty);
    mlir::Operation *alloc_dst_op = ctx_.builder.create(alloc_dst_st);

    mlir::Value c0 =
        mlir::arith::ConstantIndexOp::create(ctx_.builder, type.Loc(), 0);
    auto tiled_dims = ctx_.builder.getDenseI64ArrayAttr({0, 1});

    mlir::OperationState view_src_st(type.Loc(),
                                     "suvm.get_partitioned_tile_view");
    view_src_st.addOperands({alloc_src_op->getResult(0), c0, c0});
    view_src_st.addAttribute("tiled_dims", tiled_dims);
    view_src_st.addTypes(tv_ty);
    mlir::Operation *view_src_op = ctx_.builder.create(view_src_st);

    mlir::OperationState view_dst_st(type.Loc(),
                                     "suvm.get_partitioned_tile_view");
    view_dst_st.addOperands({alloc_dst_op->getResult(0), c0, c0});
    view_dst_st.addAttribute("tiled_dims", tiled_dims);
    view_dst_st.addTypes(tv_ty);
    mlir::Operation *view_dst_op = ctx_.builder.create(view_dst_st);

    mlir::OperationState copy_st(type.Loc(), "suvm.copy_async");
    mlir::Value view_src = view_src_op->getResult(0);
    mlir::Value view_dst = view_dst_op->getResult(0);
    copy_st.addOperands({view_src, view_dst});
    copy_st.addTypes(token_ty);
    mlir::Operation *copy_op = ctx_.builder.create(copy_st);

    // (3) Record the token produced by copy_async under token_id (scope-aware).
    int64_t token_id = parse_token_id();
    if (token_id >= 0 && copy_op && copy_op->getNumResults() == 1) {
      record_token_by_id(token_id, copy_op->getResult(0));
    }

    return SunMMIOValue{ret_dtype, result_name, ret_type};
  }

  (void)operands;
  (void)string_args;
  (void)category;
  ICHECK(ctx_.module)
      << "MLIR module must be initialized before lowering Sunmmio calls";
  mlir::TypedAttr value_attr = ctx_.builder.getIntegerAttr(
      mlir::Type::getFromOpaquePointer(
          ctx_.builder.getF32Type().getAsOpaquePointer()),
      0);
  auto fake_op = mlir::arith::ConstantOp::create(
      ctx_.builder, SunmmioMlirType(ctx_).MakeDebugLoc("fake_call"),
      value_attr);
  fake_op->setAttr("sunmmio.fake", ctx_.builder.getStringAttr("call"));
  mlir::Value call_value = fake_op.getResult();
  ctx_.BindMLIRValue(result_name, call_value);
  return SunMMIOValue{ret_dtype, result_name, ret_type};
}

} // namespace codegen
} // namespace tvm
