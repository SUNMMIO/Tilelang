#include "sunmmio_mlir_call.h"

#include "sunmmio_mlir_type.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Operation.h"
#include "npuir/Dialect/SUVM/IR/Attributes.h"
#include "npuir/Dialect/SUVM/IR/Ops.h"
#include "npuir/Dialect/SUVM/IR/Types.h"

#include <tvm/runtime/logging.h>

#include <cstdint>

namespace tvm {
namespace codegen {

SunmmioMlirCall::SunmmioMlirCall(SunmmioMlirContext &ctx) : ctx_(ctx) {}

SunMMIOValue SunmmioMlirCall::RegionCall(const std::string &result_name,
                                         const std::string &buffer_handle,
                                         const std::vector<SunMMIOValue> &mins,
                                         const std::vector<int64_t> &extents,
                                         DataType ret_dtype,
                                         const SunMMIOType &ret_type) {
  SunmmioMlirType type(ctx_);

  mlir::Value source = ctx_.LookupMLIRValue(buffer_handle);
  ICHECK(source) << "Missing MLIR source buffer for tl.tileop.region `"
                 << buffer_handle << "`";

  auto memtensor_ty =
      mlir::dyn_cast<mlir::suvm::MemTensorType>(source.getType());
  ICHECK(memtensor_ty)
      << "tl.tileop.region expects source buffer to be a suvm.memtensor";

  mlir::SmallVector<mlir::Value, 4> indices;
  indices.reserve(mins.size());
  for (const auto &min : mins) {
    mlir::Value index = ctx_.LookupMLIRValue(min.value);
    ICHECK(index) << "Missing MLIR min value in tl.tileop.region for `"
                  << min.value << "`";
    indices.push_back(type.EnsureIndex(index));
  }

  mlir::SmallVector<int64_t, 4> shape;
  mlir::SmallVector<int64_t, 4> tiled_dims;

  shape.reserve(extents.size());

  if (extents.size() >= 2) {
    for (int64_t i = 0; i < static_cast<int64_t>(extents.size()); ++i) {
      if (extents[i] != 1) {
        shape.push_back(extents[i]);
        tiled_dims.push_back(i);
      }
    }
    ICHECK_EQ(tiled_dims.size(), 2)
        << "tl.tileop.region expects exactly 2 tiled "
           "dims with extent != 1, but got "
        << tiled_dims.size();
  } else {
    shape.push_back(extents[0]);
    tiled_dims.push_back(0);
  }

  mlir::Type elem_ty = memtensor_ty.getElementType();
  mlir::Type tile_view_ty =
      mlir::suvm::TileViewType::get(&ctx_.mlir_ctx, shape, elem_ty);
  auto tiled_dims_attr = ctx_.builder.getDenseI64ArrayAttr(tiled_dims);

  auto view_op = mlir::suvm::GetPartitionedTileViewOp::create(
      ctx_.builder, type.MakeDebugLoc("region"), tile_view_ty, source, indices,
      tiled_dims_attr);
  ctx_.BindMLIRValue(result_name, view_op->getResult(0));
  return SunMMIOValue{ret_dtype, result_name, ret_type};
}

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
  auto parse_named_bool = [&](const char *key, const char *arg_name) -> bool {
    std::string prefix = std::string(key) + "=";
    for (const std::string &s : string_args) {
      if (s.rfind(prefix, 0) == 0) {
        std::string value = s.substr(prefix.size());
        ICHECK(value == "0" || value == "1")
            << arg_name << " must be encoded as 0 or 1";
        return value == "1";
      }
    }
    LOG(FATAL) << arg_name << " is missing from string_args";
    TVM_FFI_UNREACHABLE();
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
        ctx_.builder, type.MakeDebugLoc("dma_copy"), src, dst, nullptr);

    ctx_.BindMLIRValue(result_name, copy_op->getResult(0));

    int64_t token_id = parse_token_id();
    if (token_id >= 0 && copy_op && copy_op->getNumResults() == 1) {
      record_token_by_id(token_id, copy_op->getResult(0));
    }

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

    bool trans_a = parse_named_bool("trans_a", "tl.mma_sunmmio transA");
    ICHECK(!trans_a)
        << "tl.mma_sunmmio lowering to suvm.tc.mma does not support transA";
    bool trans_b = parse_named_bool("trans_b", "tl.mma_sunmmio transB");
    bool clear_accum =
        parse_named_bool("clear_accum", "tl.mma_sunmmio clearAccum");

    mlir::UnitAttr acc_attr =
        clear_accum ? mlir::UnitAttr() : ctx_.builder.getUnitAttr();
    mlir::UnitAttr trans_attr =
        trans_b ? ctx_.builder.getUnitAttr() : mlir::UnitAttr();

    auto mma_op = mlir::suvm::TcMmaOp::create(ctx_.builder,
                                              type.MakeDebugLoc("mma_sunmmio"),
                                              c, a, w, c, acc_attr, trans_attr);

    ctx_.BindMLIRValue(result_name, mma_op->getResult(0));

    int64_t token_id = parse_token_id();
    if (token_id >= 0 && mma_op && mma_op->getNumResults() == 1) {
      record_token_by_id(token_id, mma_op->getResult(0));
    }

    return SunMMIOValue{ret_dtype, result_name, ret_type};
  } else {
    (void)operands;
    (void)string_args;
    (void)category;
    LOG(INFO) << "Calling " << callee << " with operands ";
    for (const auto &op : operands) {
      LOG(INFO) << op.value << " ";
    }
    ICHECK(ctx_.module)
        << "MLIR module must be initialized before lowering Sunmmio calls";
    mlir::Type result_type = SunmmioMlirType(ctx_).MapType(ret_type);
    mlir::TypedAttr value_attr;
    if (auto float_ty = mlir::dyn_cast<mlir::FloatType>(result_type)) {
      value_attr = ctx_.builder.getFloatAttr(float_ty, 0.0);
    } else if (result_type.isIndex()) {
      value_attr = ctx_.builder.getIndexAttr(0);
    } else if (auto int_ty = mlir::dyn_cast<mlir::IntegerType>(result_type)) {
      value_attr = ctx_.builder.getIntegerAttr(int_ty, 0);
    } else {
      result_type = ctx_.builder.getI32Type();
      value_attr = ctx_.builder.getIntegerAttr(result_type, 0);
    }
    auto fake_op = mlir::arith::ConstantOp::create(
        ctx_.builder, SunmmioMlirType(ctx_).MakeDebugLoc("fake_call"),
        value_attr);
    fake_op->setAttr("sunmmio.fake", ctx_.builder.getStringAttr("call"));
    mlir::Value call_value = fake_op.getResult();
    ctx_.BindMLIRValue(result_name, call_value);
    return SunMMIOValue{ret_dtype, result_name, ret_type};
  }
}

} // namespace codegen
} // namespace tvm
