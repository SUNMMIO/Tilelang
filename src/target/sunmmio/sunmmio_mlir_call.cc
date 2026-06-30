#include "sunmmio_mlir_call.h"

#include "sunmmio_mlir_type.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "npuir/Dialect/SUVM/IR/Attributes.h"
#include "npuir/Dialect/SUVM/IR/Ops.h"
#include "npuir/Dialect/SUVM/IR/Types.h"

#include <tvm/runtime/logging.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace tvm {
namespace codegen {

SunmmioMlirCall::SunmmioMlirCall(SunmmioMlirContext &ctx) : ctx_(ctx) {}

SunMMIOValue SunmmioMlirCall::RegionCall(
    const std::string &result_name, const std::string &buffer_handle,
    const std::vector<SunMMIOValue> &mins, const std::vector<int64_t> &extents,
    DataType ret_dtype, const SunMMIOType &ret_type, int64_t byte_offset) {
  SunmmioMlirType type(ctx_);

  mlir::Value source = ctx_.LookupMLIRValue(buffer_handle);
  ICHECK(source) << "Missing MLIR source buffer for tl.tileop.region `"
                 << buffer_handle << "`";

  auto memtensor_ty =
      mlir::dyn_cast<mlir::suvm::MemTensorType>(source.getType());
  ICHECK(memtensor_ty)
      << "tl.tileop.region expects source buffer to be a suvm.memtensor";

  mlir::Value source_for_view = source;
  if (byte_offset != 0) {
    ICHECK_GE(byte_offset, 0)
        << "tl.tileop.region byte_offset must be non-negative";
    int64_t shifted_offset = memtensor_ty.getByteOffset() + byte_offset;
    auto shifted_ty = mlir::suvm::MemTensorType::get(
        memtensor_ty.getShape(), memtensor_ty.getElementType(),
        memtensor_ty.getLayout(), memtensor_ty.getMemorySpace(),
        shifted_offset);
    auto shift_op = mlir::suvm::ShiftMemTensorOp::create(
        ctx_.builder, type.MakeDebugLoc("shift_memtensor"), shifted_ty, source,
        static_cast<uint64_t>(byte_offset));
    source_for_view = shift_op.getResult();
  }

  mlir::SmallVector<mlir::Value, 4> indices;
  indices.reserve(mins.size());
  for (int64_t i = 0; i < static_cast<int64_t>(mins.size()); ++i) {
    const auto &min = mins[i];
    mlir::Value index = ctx_.LookupMLIRValue(min.value);
    ICHECK(index) << "Missing MLIR min value in tl.tileop.region for `"
                  << min.value << "`";
    index = type.EnsureIndex(index);
    indices.push_back(index);
  }

  mlir::SmallVector<int64_t, 4> shape;
  mlir::SmallVector<int64_t, 4> tiled_dims;

  shape.reserve(extents.size());
  for (int64_t i = 0; i < static_cast<int64_t>(extents.size()); ++i) {
    if (extents[i] != 1) {
      shape.push_back(extents[i]);
      tiled_dims.push_back(i);
    }
  }
  if (shape.empty()) {
    ICHECK(!extents.empty())
        << "tl.tileop.region expects at least one region dimension";
    shape.push_back(1);
    tiled_dims.push_back(0);
  }
  ICHECK(shape.size() == 1 || shape.size() == 2)
      << "tl.tileop.region expects one or two tiled dims with extent != 1, "
         "but got "
      << shape.size();

  mlir::Type elem_ty = memtensor_ty.getElementType();
  mlir::Type tile_view_ty =
      mlir::suvm::TileViewType::get(&ctx_.mlir_ctx, shape, elem_ty);
  auto tiled_dims_attr = ctx_.builder.getDenseI64ArrayAttr(tiled_dims);

  auto view_op = mlir::suvm::GetPartitionedTileViewOp::create(
      ctx_.builder, type.MakeDebugLoc("region"), tile_view_ty, source_for_view,
      indices, tiled_dims_attr);
  ctx_.BindMLIRValue(result_name, view_op->getResult(0));
  return SunMMIOValue{ret_dtype, result_name, ret_type};
}

SunMMIOValue SunmmioMlirCall::Call(const std::string &result_name,
                                   const std::string &callee,
                                   const std::vector<SunMMIOValue> &operands,
                                   const SunMMIOCallAttrs &attrs,
                                   const std::string &category,
                                   DataType ret_dtype,
                                   const SunMMIOType &ret_type) {
  SunmmioMlirType type(ctx_);
  auto get_int_attr = [&](const char *key) -> std::optional<int64_t> {
    auto it = attrs.find(key);
    if (it == attrs.end()) {
      return std::nullopt;
    }
    const int64_t *value = std::get_if<int64_t>(&it->second);
    ICHECK(value) << "SunMMIO call attr `" << key << "` must be int64_t";
    return *value;
  };
  auto get_bool_attr = [&](const char *key) -> std::optional<bool> {
    auto it = attrs.find(key);
    if (it == attrs.end()) {
      return std::nullopt;
    }
    const bool *value = std::get_if<bool>(&it->second);
    ICHECK(value) << "SunMMIO call attr `" << key << "` must be bool";
    return *value;
  };
  auto get_string_attr = [&](const char *key) -> std::optional<std::string> {
    auto it = attrs.find(key);
    if (it == attrs.end()) {
      return std::nullopt;
    }
    const std::string *value = std::get_if<std::string>(&it->second);
    ICHECK(value) << "SunMMIO call attr `" << key << "` must be string";
    return *value;
  };
  auto get_int_vector_attr = [&](const char *key) -> std::vector<int64_t> {
    auto it = attrs.find(key);
    if (it == attrs.end()) {
      return {};
    }
    const std::vector<int64_t> *value =
        std::get_if<std::vector<int64_t>>(&it->second);
    ICHECK(value) << "SunMMIO call attr `" << key
                  << "` must be vector<int64_t>";
    std::vector<int64_t> masks;
    for (int64_t mask : *value) {
      if (std::find(masks.begin(), masks.end(), mask) == masks.end()) {
        masks.push_back(mask);
      }
    }
    return masks;
  };
  auto parse_token_id = [&]() -> int64_t {
    return get_int_attr(SunMMIOCallAttrKey::kTokenId).value_or(-1);
  };
  auto parse_participant_mask = [&]() -> int64_t {
    return get_int_attr(SunMMIOCallAttrKey::kParticipantMask).value_or(-1);
  };
  auto parse_candidate_masks = [&]() -> std::vector<int64_t> {
    return get_int_vector_attr(SunMMIOCallAttrKey::kCandidateMasks);
  };
  auto ensure_i64 = [&](mlir::Value value,
                        const char *arg_name) -> mlir::Value {
    ICHECK(value) << arg_name << " is missing";
    mlir::Type ty = value.getType();
    mlir::Type i64_ty = ctx_.builder.getI64Type();
    if (ty == i64_ty) {
      return value;
    }
    if (ty.isIndex()) {
      return mlir::arith::IndexCastOp::create(ctx_.builder, type.Loc(), i64_ty,
                                              value)
          .getResult();
    }
    auto int_ty = mlir::dyn_cast<mlir::IntegerType>(ty);
    ICHECK(int_ty) << arg_name << " must be an integer or index value";
    unsigned width = int_ty.getWidth();
    ICHECK_NE(width, 64U) << arg_name << " has unsupported integer type";
    if (width < 64) {
      return mlir::arith::ExtUIOp::create(ctx_.builder, type.Loc(), i64_ty,
                                          value)
          .getResult();
    }
    return mlir::arith::TruncIOp::create(ctx_.builder, type.Loc(), i64_ty,
                                         value)
        .getResult();
  };
  auto record_token_by_id = [&](int64_t token_id, mlir::Value produced) {
    if (token_id < 0 || !produced) {
      return;
    }
    bool recorded = false;
    if (!ctx_.control_flow_stack.empty()) {
      SunmmioMlirContext::ControlNode &node = ctx_.control_flow_stack.back();
      if (node.kind == SunmmioMlirContext::ControlKind::kFor) {
        SunmmioMlirContext::ForFrame &frame = ctx_.for_stack[node.index];
        auto tit = frame.token_id_to_index.find(token_id);
        if (tit != frame.token_id_to_index.end()) {
          int idx = tit->second;
          if (idx >= 0 &&
              idx < static_cast<int>(frame.produced_tokens.size())) {
            frame.produced_tokens[idx] = produced;
          }
          recorded = true;
        }
      } else if (node.kind == SunmmioMlirContext::ControlKind::kIf) {
        SunmmioMlirContext::IfFrame &frame = ctx_.if_stack[node.index];
        auto tit = frame.token_id_to_index.find(token_id);
        if (tit != frame.token_id_to_index.end()) {
          int idx = tit->second;
          if (idx >= 0 &&
              idx < static_cast<int>(frame.produced_tokens.size())) {
            frame.produced_tokens[idx] = produced;
          }
          recorded = true;
        }
      } else {
        SunmmioMlirContext::WhileFrame &frame = ctx_.while_stack[node.index];
        auto tit = frame.token_id_to_index.find(token_id);
        if (tit != frame.token_id_to_index.end()) {
          int idx = tit->second;
          if (idx >= 0 &&
              idx < static_cast<int>(frame.produced_tokens.size())) {
            frame.produced_tokens[idx] = produced;
          }
          recorded = true;
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
      } else if (node.kind == SunmmioMlirContext::ControlKind::kIf) {
        snapshot_if_needed(ctx_.if_stack[node.index]);
      } else {
        snapshot_if_needed(ctx_.while_stack[node.index]);
      }
    }

    ctx_.token_by_id[token_id] = produced;
  };
  auto require_bool_attr = [&](const char *key, const char *arg_name) -> bool {
    std::optional<bool> value = get_bool_attr(key);
    if (value.has_value()) {
      return *value;
    }
    LOG(FATAL) << arg_name << " is missing from attrs";
    TVM_FFI_UNREACHABLE();
  };
  auto ensure_barrier_for_mask = [&](int64_t mask) -> mlir::Value {
    ICHECK_GE(mask, 0) << "barrier participant_mask must be non-negative";
    auto it = ctx_.barrier_by_mask.find(mask);
    if (it != ctx_.barrier_by_mask.end() && it->second) {
      return it->second;
    }
    mlir::IntegerAttr mask_attr = ctx_.builder.getI64IntegerAttr(mask);
    auto barrier_op = mlir::suvm::BarrierInitOp::create(
        ctx_.builder, type.MakeDebugLoc("barrier_init"), mlir::Value{},
        mask_attr);
    ctx_.barrier_by_mask[mask] = barrier_op.getBarrier();
    return barrier_op.getBarrier();
  };
  auto emit_barrier_arrive_and_wait = [&](mlir::Value barrier) {
    (void)mlir::suvm::BarrierArriveAndWaitOp::create(
        ctx_.builder, type.MakeDebugLoc("barrier_arrive_and_wait"), barrier,
        mlir::IntegerAttr{});
  };
  auto emit_dynamic_barrier_wait = [&](mlir::Value dynamic_mask,
                                       const std::vector<int64_t> &candidates) {
    ICHECK(dynamic_mask) << "dynamic barrier mask is missing";
    ICHECK(!candidates.empty())
        << "dynamic barrier wait requires static candidate masks";

    std::function<void(size_t)> emit_case = [&](size_t index) {
      ICHECK_LT(index, candidates.size());
      int64_t candidate = candidates[index];
      mlir::Value cst = mlir::arith::ConstantIntOp::create(
                            ctx_.builder, type.Loc(), candidate, 64)
                            .getResult();
      mlir::Value is_match = mlir::arith::CmpIOp::create(
          ctx_.builder, type.Loc(), mlir::arith::CmpIPredicate::eq,
          dynamic_mask, cst);
      auto if_op = mlir::scf::IfOp::create(ctx_.builder, type.Loc(), is_match,
                                           /*withElseRegion=*/true);

      mlir::Block &then_block = if_op.getThenRegion().front();
      ctx_.builder.setInsertionPointToStart(&then_block);
      emit_barrier_arrive_and_wait(ensure_barrier_for_mask(candidate));

      mlir::Block &else_block = if_op.getElseRegion().front();
      ctx_.builder.setInsertionPointToStart(&else_block);
      if (index + 1 < candidates.size()) {
        emit_case(index + 1);
      } else {
        mlir::Value always_false =
            mlir::arith::ConstantIntOp::create(ctx_.builder, type.Loc(), 0, 1)
                .getResult();
        mlir::cf::AssertOp::create(
            ctx_.builder, type.Loc(), always_false,
            ctx_.builder.getStringAttr(
                "dynamic barrier mask is not in candidate set"));
      }

      ctx_.builder.setInsertionPointAfter(if_op);
    };

    emit_case(0);
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
  } else if (callee == "tl.wait_token") {
    int64_t token_id = parse_token_id();
    if (token_id < 0) {
      LOG(FATAL) << "tl.wait_token requires token_id";
      TVM_FFI_UNREACHABLE();
    }
    mlir::Value tok;
    auto current_it = ctx_.token_by_id.find(token_id);
    if (current_it != ctx_.token_by_id.end()) {
      tok = current_it->second;
    }
    for (auto nit = ctx_.control_flow_stack.rbegin();
         !tok && nit != ctx_.control_flow_stack.rend(); ++nit) {
      if (nit->kind == SunmmioMlirContext::ControlKind::kFor) {
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
        continue;
      }
      if (nit->kind != SunmmioMlirContext::ControlKind::kWhile) {
        continue;
      }
      SunmmioMlirContext::WhileFrame &frame = ctx_.while_stack[nit->index];
      if (!frame.in_body) {
        continue;
      }
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
  } else if (callee == "tl.barrier_init") {
    int64_t participant_mask = parse_participant_mask();
    mlir::Value result_barrier;
    if (participant_mask >= 0) {
      result_barrier = ensure_barrier_for_mask(participant_mask);
    } else {
      ICHECK_LE(operands.size(), 1U)
          << "tl.barrier_init dynamic mask expects at most one mask operand";
      std::vector<int64_t> candidates = parse_candidate_masks();
      ICHECK(!candidates.empty())
          << "tl.barrier_init dynamic mask requires candidate_mask entries";
      for (int64_t mask : candidates) {
        (void)ensure_barrier_for_mask(mask);
      }
    }
    if (!result_name.empty() && result_barrier) {
      ctx_.BindMLIRValue(result_name, result_barrier);
    }
    return SunMMIOValue{ret_dtype, result_name, ret_type};
  } else if (callee == "tl.barrier_arrive_and_wait") {
    int64_t participant_mask = parse_participant_mask();
    if (participant_mask >= 0) {
      auto barrier_it = ctx_.barrier_by_mask.find(participant_mask);
      ICHECK(barrier_it != ctx_.barrier_by_mask.end() && barrier_it->second)
          << "tl.barrier_arrive_and_wait participant_mask=" << participant_mask
          << " has no corresponding tl.barrier_init.";
      emit_barrier_arrive_and_wait(barrier_it->second);
    } else {
      ICHECK_EQ(operands.size(), 1U)
          << "tl.barrier_arrive_and_wait dynamic mask expects one mask operand";
      mlir::Value dynamic_mask = ctx_.LookupMLIRValue(operands[0].value);
      if (!dynamic_mask) {
        dynamic_mask =
            type.ResolveValue(operands[0], ctx_.builder.getI64Type());
      }
      dynamic_mask =
          ensure_i64(dynamic_mask, "tl.barrier_arrive_and_wait mask");
      emit_dynamic_barrier_wait(dynamic_mask, parse_candidate_masks());
    }
    return SunMMIOValue{ret_dtype, result_name, ret_type};
  } else if (callee == "tl.dma_copy") {
    ICHECK_GE(operands.size(), 2)
        << "tl.dma_copy expects src and dst tile views";

    mlir::Value src = ctx_.LookupMLIRValue(operands[0].value);
    ICHECK(src) << "Missing MLIR source tile view for tl.dma_copy `"
                << operands[0].value << "`";
    auto src_ty = mlir::dyn_cast<mlir::suvm::TileViewType>(src.getType());
    ICHECK(src_ty) << "tl.dma_copy expects source to be a suvm.tile_view";

    mlir::Value dst = ctx_.LookupMLIRValue(operands[1].value);
    ICHECK(dst) << "Missing MLIR destination tile view for tl.dma_copy `"
                << operands[1].value << "`";
    auto dst_ty = mlir::dyn_cast<mlir::suvm::TileViewType>(dst.getType());
    ICHECK(dst_ty) << "tl.dma_copy expects destination to be a suvm.tile_view";

    auto copy_op = mlir::suvm::CopyAsyncOp::create(
        ctx_.builder, type.MakeDebugLoc("dma_copy"), src, dst,
        mlir::suvm::PadModeAttr{}, mlir::suvm::OdmaChannelAttr{});

    ICHECK(!result_name.empty()) << "tl.dma_copy expects a token result";
    ICHECK(copy_op && copy_op->getNumResults() == 1)
        << "tl.dma_copy lowering expects one token result";
    mlir::Value produced = copy_op->getResult(0);
    ctx_.BindMLIRValue(result_name, produced);

    int64_t token_id = parse_token_id();
    ICHECK_GE(token_id, 0) << "tl.dma_copy requires sync_token_id";
    record_token_by_id(token_id, produced);

    return SunMMIOValue{ret_dtype, result_name, ret_type};
  } else if (callee == "tl.sunmmio_layout_transform") {
    ICHECK_GE(operands.size(), 2)
        << "tl.sunmmio_layout_transform expects src and dst tile views";

    mlir::Value src = ctx_.LookupMLIRValue(operands[0].value);
    ICHECK(src)
        << "Missing MLIR source tile view for tl.sunmmio_layout_transform `"
        << operands[0].value << "`";
    auto src_ty = mlir::dyn_cast<mlir::suvm::TileViewType>(src.getType());
    ICHECK(src_ty)
        << "tl.sunmmio_layout_transform expects source to be a suvm.tile_view";

    mlir::Value dst = ctx_.LookupMLIRValue(operands[1].value);
    ICHECK(dst) << "Missing MLIR destination tile view for "
                   "tl.sunmmio_layout_transform `"
                << operands[1].value << "`";
    auto dst_ty = mlir::dyn_cast<mlir::suvm::TileViewType>(dst.getType());
    ICHECK(dst_ty) << "tl.sunmmio_layout_transform expects destination to be a "
                      "suvm.tile_view";

    auto transform_op = mlir::suvm::TransformAsyncOp::create(
        ctx_.builder, type.MakeDebugLoc("sunmmio_layout_transform"), src, dst,
        mlir::suvm::PadModeAttr{}, mlir::suvm::OdmaChannelAttr{});

    ICHECK(!result_name.empty())
        << "tl.sunmmio_layout_transform expects a token result";
    ICHECK(transform_op && transform_op->getNumResults() == 1)
        << "tl.sunmmio_layout_transform lowering expects one token result";
    mlir::Value produced = transform_op->getResult(0);
    ctx_.BindMLIRValue(result_name, produced);

    int64_t token_id = parse_token_id();
    ICHECK_GE(token_id, 0)
        << "tl.sunmmio_layout_transform requires sync_token_id";
    record_token_by_id(token_id, produced);

    return SunMMIOValue{ret_dtype, result_name, ret_type};
  } else if (callee == "tl.broadcast_") {
    ICHECK(operands.size() == 3 || operands.size() == 4)
        << "tl.broadcast_ expects src, dst, mask, and optional src_core "
           "operands";

    mlir::Value src = ctx_.LookupMLIRValue(operands[0].value);
    ICHECK(src) << "Missing MLIR source tile view for tl.broadcast_ `"
                << operands[0].value << "`";
    auto src_ty = mlir::dyn_cast<mlir::suvm::TileViewType>(src.getType());
    ICHECK(src_ty) << "tl.broadcast_ expects source to be a suvm.tile_view";

    mlir::Value dst = ctx_.LookupMLIRValue(operands[1].value);
    ICHECK(dst) << "Missing MLIR destination tile view for tl.broadcast_ `"
                << operands[1].value << "`";
    auto dst_ty = mlir::dyn_cast<mlir::suvm::TileViewType>(dst.getType());
    ICHECK(dst_ty)
        << "tl.broadcast_ expects destination to be a suvm.tile_view";

    mlir::Value mask = ctx_.LookupMLIRValue(operands[2].value);
    if (!mask) {
      mask = type.ResolveValue(operands[2], ctx_.builder.getI64Type());
    }
    mask = ensure_i64(mask, "tl.broadcast_ mask");

    std::string direction_name =
        get_string_attr(SunMMIOCallAttrKey::kDirection).value_or("");
    ICHECK(direction_name == "row" || direction_name == "col")
        << "tl.broadcast_ direction must be encoded as row or col";
    auto direction = direction_name == "row" ? mlir::suvm::McastDirection::row
                                             : mlir::suvm::McastDirection::col;

    auto create_mcast = [&]() -> mlir::Value {
      auto mcast_op = mlir::suvm::MulticastTokOp::create(
          ctx_.builder, type.MakeDebugLoc("broadcast"), src, dst, mask,
          direction, mlir::suvm::OdmaChannelAttr{});
      return mcast_op->getResult(0);
    };

    mlir::Value produced;
    if (operands.size() == 4) {
      mlir::Value src_core = ctx_.LookupMLIRValue(operands[3].value);
      if (!src_core) {
        src_core = type.ResolveValue(operands[3], ctx_.builder.getI64Type());
      }
      src_core = ensure_i64(src_core, "tl.broadcast_ src_core");

      mlir::Value core_id = mlir::suvm::GetCoreIdOp::create(
                                ctx_.builder, type.MakeDebugLoc("get_core_id"))
                                .getResult();
      core_id = ensure_i64(core_id, "suvm.get_core_id result");
      mlir::Value is_src = mlir::arith::CmpIOp::create(
          ctx_.builder, type.Loc(), mlir::arith::CmpIPredicate::eq, core_id,
          src_core);

      mlir::Type token_ty = mlir::suvm::TokenType::get(&ctx_.mlir_ctx);
      auto if_op = mlir::scf::IfOp::create(ctx_.builder, type.Loc(), token_ty,
                                           is_src, /*withElseRegion=*/true);

      mlir::Block &then_block = if_op.getThenRegion().front();
      ctx_.builder.setInsertionPointToStart(&then_block);
      mlir::Value mcast_token = create_mcast();
      mlir::scf::YieldOp::create(ctx_.builder, type.Loc(), mcast_token);

      mlir::Block &else_block = if_op.getElseRegion().front();
      ctx_.builder.setInsertionPointToStart(&else_block);
      mlir::Value null_token =
          mlir::suvm::NullTokenOp::create(ctx_.builder,
                                          type.MakeDebugLoc("broadcast_null"))
              .getResult();
      mlir::scf::YieldOp::create(ctx_.builder, type.Loc(), null_token);

      ctx_.builder.setInsertionPointAfter(if_op);
      produced = if_op.getResult(0);
    } else {
      produced = create_mcast();
    }

    ICHECK(!result_name.empty()) << "tl.broadcast_ expects a token result";
    ICHECK(produced) << "tl.broadcast_ lowering expects one token result";
    ctx_.BindMLIRValue(result_name, produced);
    int64_t token_id = parse_token_id();
    ICHECK_GE(token_id, 0) << "tl.broadcast_ requires sync_token_id";
    record_token_by_id(token_id, produced);

    return SunMMIOValue{ret_dtype, result_name, ret_type};
  } else if (callee == "tl.mma_sunmmio") {
    ICHECK_EQ(operands.size(), 3)
        << "tl.mma_sunmmio expects A/B/C tile views as operands";

    mlir::Value a = ctx_.LookupMLIRValue(operands[0].value);
    ICHECK(a) << "Missing MLIR activation tile view for tl.mma_sunmmio `"
              << operands[0].value << "`";
    auto a_ty = mlir::dyn_cast<mlir::suvm::TileViewType>(a.getType());
    ICHECK(a_ty) << "tl.mma_sunmmio expects activation to be a suvm.tile_view";

    mlir::Value w = ctx_.LookupMLIRValue(operands[1].value);
    ICHECK(w) << "Missing MLIR weight tile view for tl.mma_sunmmio `"
              << operands[1].value << "`";
    auto w_ty = mlir::dyn_cast<mlir::suvm::TileViewType>(w.getType());
    ICHECK(w_ty) << "tl.mma_sunmmio expects weight to be a suvm.tile_view";

    mlir::Value c = ctx_.LookupMLIRValue(operands[2].value);
    ICHECK(c) << "Missing MLIR accumulator tile view for tl.mma_sunmmio `"
              << operands[2].value << "`";
    auto c_ty = mlir::dyn_cast<mlir::suvm::TileViewType>(c.getType());
    ICHECK(c_ty) << "tl.mma_sunmmio expects accumulator to be a suvm.tile_view";

    bool trans_a =
        require_bool_attr(SunMMIOCallAttrKey::kTransA, "tl.mma_sunmmio transA");
    ICHECK(!trans_a)
        << "tl.mma_sunmmio lowering to suvm.tc.mma does not support transA";
    bool trans_b =
        require_bool_attr(SunMMIOCallAttrKey::kTransB, "tl.mma_sunmmio transB");
    bool clear_accum = require_bool_attr(SunMMIOCallAttrKey::kClearAccum,
                                         "tl.mma_sunmmio clearAccum");

    mlir::UnitAttr acc_attr =
        clear_accum ? mlir::UnitAttr() : ctx_.builder.getUnitAttr();
    mlir::UnitAttr trans_attr =
        trans_b ? ctx_.builder.getUnitAttr() : mlir::UnitAttr();

    auto mma_op = mlir::suvm::TcMmaOp::create(ctx_.builder,
                                              type.MakeDebugLoc("mma_sunmmio"),
                                              c, a, w, c, acc_attr, trans_attr);

    ICHECK(!result_name.empty()) << "tl.mma_sunmmio expects a token result";
    ICHECK(mma_op && mma_op->getNumResults() == 1)
        << "tl.mma_sunmmio lowering expects one token result";
    mlir::Value produced = mma_op->getResult(0);
    ctx_.BindMLIRValue(result_name, produced);

    int64_t token_id = parse_token_id();
    ICHECK_GE(token_id, 0) << "tl.mma_sunmmio requires sync_token_id";
    record_token_by_id(token_id, produced);

    return SunMMIOValue{ret_dtype, result_name, ret_type};
  } else if (callee == "tir.ret") {
    ICHECK_EQ(operands.size(), 1) << "tir.ret expects one operand";

    const SunMMIOValue &ret = operands[0];
    bool is_zero = ret.value == "0";
    if (!is_zero) {
      mlir::Value ret_value = ctx_.LookupMLIRValue(ret.value);
      if (ret_value) {
        if (auto const_op =
                ret_value.getDefiningOp<mlir::arith::ConstantOp>()) {
          if (auto int_attr =
                  mlir::dyn_cast<mlir::IntegerAttr>(const_op.getValue())) {
            is_zero = int_attr.getInt() == 0;
          }
        }
      }
    }

    ICHECK(is_zero) << "SunMMIO device kernel only supports T.ret(0); got "
                    << ret.value;
    return SunMMIOValue{
        DataType::Void(), "",
        SunMMIOType{SunMMIOType::Kind::kUnknown, DataType::Void(), 1, {}}};
  } else {
    LOG(FATAL) << "Unsupported SunMMIO call lowering for `" << callee
               << "` (category=" << category << ", operands=" << operands.size()
               << ")";
    TVM_FFI_UNREACHABLE();
  }
}

} // namespace codegen
} // namespace tvm
