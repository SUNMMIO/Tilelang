/*!
 * \file tl/op/comm.cc
 * \brief Implementation of Inter-core Communication Operators
 */

#include "comm.h"

#include <algorithm>
#include <tvm/tir/op.h>
#include <vector>

#include "../target/utils.h"
#include "copy.h"
#include "reduce.h"
#include "utils.h"

namespace tvm {
namespace tl {

#define TIR_DEFINE_TL_BUILTIN(OpName)                                          \
  const Op &OpName() {                                                         \
    static const Op &op = Op::Get("tl." #OpName);                              \
    return op;                                                                 \
  }                                                                            \
  TVM_REGISTER_OP("tl." #OpName)                                               \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", #OpName)
TIR_DEFINE_TL_BUILTIN(comm_barrier)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(comm_fence)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(CoreId).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(comm_current_core)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(comm_is_current_core)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(broadcast_)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
// src_buffer, dst_buffer, size(IntImm), src_core(IntImm)
// direction(0: horizontal, 1: vertical),
// *mask(optional: IntImm list of core ids to exclude)

using namespace tir;

Broadcast::Broadcast(Array<PrimExpr> args) {
  ObjectPtr<BroadcastNode> node = tvm::ffi::make_object<BroadcastNode>();
  node->src_expr = args[0];
  node->dst_expr = args[1];
  Array<Range> rgs[2];
  Buffer bf[2];
  for (int i = 0; i < 2; i++) {
    auto region = NormalizeToBufferRegion(args[i]);

    rgs[i] = region->region;
    bf[i] = region->buffer;
  }
  std::tie(node->src, node->dst) = std::tie(bf[0], bf[1]);
  std::tie(node->src_range, node->dst_range) = std::tie(rgs[0], rgs[1]);
  node->size = Downcast<IntImm>(args[2]);
  node->dst_offset = Downcast<IntImm>(args[3]);
  node->src_core = Downcast<IntImm>(args[4]);
  node->direction = Downcast<IntImm>(args[5])->value;
  data_ = std::move(node);
}

TileOperator BroadcastNode::Clone() const {
  auto op = tvm::ffi::make_object<BroadcastNode>(*this);
  return Broadcast(op);
}

LayoutMap BroadcastNode::InferLayout(const LayoutInferArgs &T,
                                     InferLevel level) const {
  Array<PrimExpr> args;
  args.push_back(src_expr);
  args.push_back(dst_expr);
  Copy copy_op = Copy(args);
  LayoutMap out_layout = copy_op->InferLayout(T, level);
  return out_layout;                                    
}

// int get_target_mesh_nrows(Target target) {
//   auto mattr = target->GetAttr<Array<String>>("mattr").value();
//   int x = 0;
//   for (size_t i = 0; i < mattr.size(); i++) {
//     std::string m = mattr[i];
//     if (m.find("device_mesh_nrow_") != std::string::npos) {
//       std::string s = m.substr(m.find_last_of('_') + 1);;
//       try {
//         x = std::stoi(s);
//       } catch (const std::invalid_argument& e) {
//         x = -1;
//       } catch (const std::out_of_range& e) {
//         x = -1;
//       }
//     }
//   }
//   ICHECK(x != 0) << "Device mesh row number not found.";
//   ICHECK(x > 0) << "Invalid device mesh row number: ";
//   return x;
// }

int get_target_mesh(Target target, int axis) {
  auto mattr = target->GetAttr<Array<String>>("mattr").value();
  int x = 0;
  std::string axis_str;
  if (axis == 0) {
    axis_str = "device_mesh_ncol_";
  } else if (axis == 1) {
    axis_str = "device_mesh_nrow_";
  } else {
    LOG(FATAL) << "Invalid axis " << axis << " for getting mesh dimension.";
  }
  for (size_t i = 0; i < mattr.size(); i++) {
    std::string m = mattr[i];
    if (m.find(axis_str) != std::string::npos) {
      std::string s = m.substr(m.find_last_of('_') + 1);;
      try {
        x = std::stoi(s);
      } catch (const std::invalid_argument& e) {
        x = -1;
      } catch (const std::out_of_range& e) {
        x = -1;
      }
    }
  }
  ICHECK(x != 0) << axis_str << " not found.";
  ICHECK(x > 0) << "Invalid " << axis_str;
  return x;
}


Stmt BroadcastNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  Target target = T.target;
  ICHECK(TargetIsSunmmio(target)) << "Broadcast only supports SUNMMIO targets.";
  int mesh_x = get_target_mesh(target, 0);
  int mesh_y = get_target_mesh(target, 1);

  // check for valid core id
  ICHECK(src_core->value >= 0 and src_core->value < mesh_x * mesh_y)
      << "Source core id " << src_core->value << " out of range [0, "
      << mesh_x * mesh_y << ")";

  // check for src and dst buffer sizes
  PrimExpr src_elements = 1;
  for (size_t i = 0; i < src_range.size(); i++) {
    src_elements *= src_range[i]->extent;
  }
  src_elements = analyzer->Simplify(src_elements);
  PrimExpr dst_elements = 1;
  for (size_t i = 0; i < dst_range.size(); i++) {
    dst_elements *= dst_range[i]->extent;
  }
  dst_elements = analyzer->Simplify(dst_elements);
  ICHECK(Downcast<IntImm>(src_elements)->value <=
         Downcast<IntImm>(dst_elements)->value)
      << "Source buffer size larger than destination buffer size: "
      << src_elements << " vs " << dst_elements;
  ICHECK(size->value <= Downcast<IntImm>(src_elements)->value)
      << "Broadcast size larger than data size: " << size->value << " vs "
      << Downcast<IntImm>(src_elements)->value;

  // check for size and dst_offset
  PrimExpr broadcast_elements;
  if (size->value < 0) {
    broadcast_elements = src_elements;
  } else {
    broadcast_elements = size;
  }
  ICHECK((Downcast<IntImm>(broadcast_elements)->value) <=
         Downcast<IntImm>(src_elements)->value)
      << "Broadcast size Larger than source buffer size: "
      << (Downcast<IntImm>(broadcast_elements)->value) << " vs "
      << Downcast<IntImm>(src_elements)->value;
  ICHECK((Downcast<IntImm>(broadcast_elements)->value + dst_offset->value) <=
         Downcast<IntImm>(dst_elements)->value)
      << "Broadcast size + dst_offset larger than destination buffer size: "
      << (Downcast<IntImm>(broadcast_elements)->value + dst_offset->value)
      << " vs " << Downcast<IntImm>(dst_elements)->value;

  // check for valid direction
  if (direction != 0 and direction != 1 and direction != 2) {
    LOG(FATAL) << "Invalid broadcast direction " << direction
               << ", must be 0 (horizontal) or 1 (vertical) or 2 (all).";
  }

  // all checks passed, generate the call
  PrimExpr src_addr = src.access_ptr(1, DataType::Handle(), 1, 0, src_elements);
  PrimExpr dst_addr =
      dst.access_ptr(2, DataType::Handle(), 1,
                     Downcast<IntImm>(dst_offset->value), src_elements);
  int src_core_y = src_core->value % mesh_y;

  if (direction == 0 or direction == 1) {
    // 1D broadcast
    Array<PrimExpr> args;
    args.push_back(src_addr);
    args.push_back(dst_addr);
    args.push_back(Downcast<IntImm>(broadcast_elements));
    args.push_back(src_core);
    args.push_back(direction);
    Stmt broadcast = Evaluate(Call(DataType::Handle(), broadcast_(), args));
    return broadcast;
  } else {
    // 2D broadcast
    Array<Stmt> seq;
    // vertical broadcast
    Array<PrimExpr> args;
    args.push_back(src_addr);
    args.push_back(dst_addr);
    args.push_back(Downcast<IntImm>(broadcast_elements));
    args.push_back(src_core);
    args.push_back(1); // direction: vertical
    Stmt broadcast = Evaluate(Call(DataType::Handle(), broadcast_(), args));
    seq.push_back(broadcast);
    // herizontal broadcast
    for (int i = 0; i < mesh_x; i++) {
      Array<PrimExpr> args;
      args.push_back(dst.access_ptr(1, DataType::Handle(), 1, 0, dst_elements));
      args.push_back(dst.access_ptr(2, DataType::Handle(), 1, 0, dst_elements));
      args.push_back(Downcast<IntImm>(broadcast_elements));
      args.push_back(int(i * mesh_y) + src_core_y);
      args.push_back(0); // direction: horeizontal
      args.push_back(
          IntImm(DataType::Int(32), src_core_y)); // mask: current core only
      Stmt broadcast = Evaluate(Call(DataType::Handle(), broadcast_(), args));
      seq.push_back(broadcast);
    }
    return SeqStmt::Flatten(seq);
  }
}

TIR_REGISTER_TL_TILE_OP(Broadcast, comm_broadcast)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

Put::Put(Array<PrimExpr> args) {
  ObjectPtr<PutNode> node = tvm::ffi::make_object<PutNode>();
  node->src_expr = args[0];
  node->dst_expr = args[1];
  Array<Range> rgs[2];
  Buffer bf[2];
  for (int i = 0; i < 2; i++) {
    auto region = NormalizeToBufferRegion(args[i]);
    rgs[i] = region->region;
    bf[i] = region->buffer;
  }
  std::tie(node->src, node->dst) = std::tie(bf[0], bf[1]);
  std::tie(node->src_range, node->dst_range) = std::tie(rgs[0], rgs[1]);
  node->size = Downcast<IntImm>(args[2]);
  node->src_core = Downcast<IntImm>(args[3]);
  node->dst_core = Downcast<IntImm>(args[4]);
  data_ = std::move(node);
}

TileOperator PutNode::Clone() const {
  auto op = tvm::ffi::make_object<PutNode>(*this);
  return Put(op);
}

LayoutMap PutNode::InferLayout(const LayoutInferArgs &T,
                               InferLevel level) const {
  Array<PrimExpr> args;
  args.push_back(src_expr);
  args.push_back(dst_expr);
  Copy copy_op = Copy(args);
  LayoutMap out_layout = copy_op->InferLayout(T, level);
  return out_layout;
}

Stmt PutNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  Target target = T.target;
  ICHECK(TargetIsSunmmio(target)) << "Put only supports SUNMMIO targets.";
  int mesh_x = get_target_mesh(target, 0);
  int mesh_y = get_target_mesh(target, 1);

  // check for valid core id
  ICHECK(src_core->value >= 0 and src_core->value < mesh_x * mesh_y)
      << "Source core id " << src_core->value << " out of range [0, "
      << mesh_x * mesh_y << ")";
  ICHECK(dst_core->value >= 0 and dst_core->value < mesh_x * mesh_y)
      << "Destination core id " << dst_core->value << " out of range [0, "
      << mesh_x * mesh_y << ")";

  // check for src and dst buffer sizes
  PrimExpr src_elements = 1;
  for (size_t i = 0; i < src_range.size(); i++) {
    src_elements *= src_range[i]->extent;
  }
  src_elements = analyzer->Simplify(src_elements);
  PrimExpr dst_elements = 1;
  for (size_t i = 0; i < dst_range.size(); i++) {
    dst_elements *= dst_range[i]->extent;
  }
  dst_elements = analyzer->Simplify(dst_elements);
  ICHECK(Downcast<IntImm>(src_elements)->value <=
         Downcast<IntImm>(dst_elements)->value)
      << "Source buffer size larger than destination buffer size: "
      << src_elements << " vs " << dst_elements;
  ICHECK(size->value <= Downcast<IntImm>(src_elements)->value)
      << "Broadcast size larger than data size: " << size->value << " vs "
      << Downcast<IntImm>(src_elements)->value;

  // check for size
  PrimExpr broadcast_elements;
  if (size->value < 0) {
    broadcast_elements = src_elements;
  } else {
    broadcast_elements = size;
  }
  ICHECK((Downcast<IntImm>(broadcast_elements)->value) <=
         Downcast<IntImm>(src_elements)->value)
      << "Broadcast size Larger than source buffer size: "
      << (Downcast<IntImm>(broadcast_elements)->value) << " vs "
      << Downcast<IntImm>(src_elements)->value;
  ICHECK((Downcast<IntImm>(broadcast_elements)->value) <=
         Downcast<IntImm>(dst_elements)->value)
      << "Broadcast size larger than destination buffer size: "
      << (Downcast<IntImm>(broadcast_elements)->value) << " vs "
      << Downcast<IntImm>(dst_elements)->value;

  // all checks passed, generate the call
  PrimExpr src_addr = src.access_ptr(1, DataType::Handle(), 1, 0, src_elements);
  PrimExpr dst_addr = dst.access_ptr(2, DataType::Handle(), 1, 0, dst_elements);
  int src_core_x = src_core->value / mesh_y;
  int src_core_y = src_core->value % mesh_y;
  int dst_core_x = dst_core->value / mesh_y;
  int dst_core_y = dst_core->value % mesh_y;

  if (src_core_x == dst_core_x) {
    // 1D put via horizontal communication
    Array<PrimExpr> args;
    args.push_back(src_addr);
    args.push_back(dst_addr);
    args.push_back(Downcast<IntImm>(broadcast_elements));
    args.push_back(src_core);
    args.push_back(0); // direction: horizontal
    for (int j = 0; j < mesh_y; j++) {
      if (j != dst_core_y) {
        args.push_back(
            IntImm(DataType::Int(32), j)); // mask: all cores except dst_core_y
      }
    }
    Stmt put = Evaluate(Call(DataType::Handle(), broadcast_(), args));
    return put;
  } else if (src_core_y == dst_core_y) {
    // 1D put via vertical communication
    Array<PrimExpr> args;
    args.push_back(src_addr);
    args.push_back(dst_addr);
    args.push_back(Downcast<IntImm>(broadcast_elements));
    args.push_back(src_core);
    args.push_back(1); // direction: vertical
    for (int i = 0; i < mesh_x; i++) {
      if (i != dst_core_x) {
        args.push_back(
            IntImm(DataType::Int(32), i)); // mask: all cores except dst_core_x
      }
    }
    Stmt put = Evaluate(Call(DataType::Handle(), broadcast_(), args));
    return put;
  } else {
    Array<Stmt> seq;
    // vertical transfer from src core to intermediate core
    int intermediate_core_id = src_core_x * mesh_y + dst_core_y;
    Array<PrimExpr> args1;
    args1.push_back(src_addr);
    args1.push_back(dst_addr);
    args1.push_back(Downcast<IntImm>(broadcast_elements));
    args1.push_back(src_core);
    args1.push_back(1); // direction: vertical
    for (int i = 0; i < mesh_x; i++) {
      if (i != dst_core_x) {
        args1.push_back(
            IntImm(DataType::Int(32), i)); // mask: all cores except dst_core_x
      }
    }
    Stmt put1 = Evaluate(Call(DataType::Handle(), broadcast_(), args1));
    seq.push_back(put1);
    // horizontal transfer from intermediate core to dst core
    Array<PrimExpr> args2;
    args2.push_back(dst.access_ptr(1, DataType::Handle(), 1, 0, src_elements));
    args2.push_back(dst.access_ptr(2, DataType::Handle(), 1, 0, dst_elements));
    args2.push_back(Downcast<IntImm>(broadcast_elements));
    args2.push_back(IntImm(DataType::Int(32), intermediate_core_id));
    args2.push_back(0); // direction: horizontal
    for (int j = 0; j < mesh_y; j++) {
      if (j != dst_core_y) {
        args2.push_back(
            IntImm(DataType::Int(32), j)); // mask: all cores except dst_core_y
      }
    }
    Stmt put2 = Evaluate(Call(DataType::Handle(), broadcast_(), args2));
    seq.push_back(put2);
    return SeqStmt::Flatten(seq);
  }
}

TIR_REGISTER_TL_TILE_OP(Put, comm_put)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

Allgather::Allgather(Array<PrimExpr> args) {
  ObjectPtr<AllgatherNode> node = tvm::ffi::make_object<AllgatherNode>();
  node->send = args[0];
  node->recv = args[1];
  node->direction = Downcast<IntImm>(args[2])->value;
  node->size = Downcast<IntImm>(args[3]);
  data_ = std::move(node);
}

TileOperator AllgatherNode::Clone() const {
  auto op = tvm::ffi::make_object<AllgatherNode>(*this);
  return Allgather(op);
}

Layout AllgatherNode::ComputeLinearLayout(const Buffer &shared_tensor) const {
  Array<PrimExpr> input_size = shared_tensor->shape;
  Array<PrimExpr> forward_vars;
  for (size_t i = 0; i < input_size.size(); i++) {
    forward_vars.push_back(InputPlaceholder(i));
  }
  // [i, j] -> [i // 256, j // 256, i % 256, j % 256]
  Array<PrimExpr> forward_index;
  for (size_t i = 0; i < input_size.size(); i++) {
    forward_index.push_back(FloorDiv(forward_vars[i], 256));
  }
  for (size_t i = 0; i < input_size.size(); i++) {
    forward_index.push_back(FloorMod(forward_vars[i], 256));
  }
  return Layout(input_size, forward_index);
}

LayoutMap AllgatherNode::InferLayout(const LayoutInferArgs &T,
                                     InferLevel level) const {
  Buffer recv_buffer = NormalizeToBufferRegion(recv)->buffer;                    
  Layout linear_layout = ComputeLinearLayout(recv_buffer);
  return Map<Buffer, Layout>({{recv_buffer, linear_layout}});
}

Stmt AllgatherNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  Target target = T.target;
  ICHECK(TargetIsSunmmio(target)) << "Allgather only supports SUNMMIO targets.";
  int mesh_x = get_target_mesh(target, 0);
  int mesh_y = get_target_mesh(target, 1);

  Array<Range> send_range, recv_range;
  auto send_region = NormalizeToBufferRegion(send);
  auto recv_region = NormalizeToBufferRegion(recv);
  send_range = send_region->region;
  recv_range = recv_region->region;

  int recv_num = 1;
  if (direction == 0) { // horizontal
    recv_num = mesh_y;
  } else if (direction == 1) { // vertical
    recv_num = mesh_x;
  } else if (direction == 2) { // all
    recv_num = mesh_x * mesh_y;
  } else {
    // invalid direction
    ICHECK(false) << "Invalid direction value for allgather: " << direction;
  }

  PrimExpr send_elements = 1;
  for (size_t i = 0; i < send_range.size(); i++) {
    send_elements *= send_range[i]->extent;
  }
  send_elements = analyzer->Simplify(send_elements);
  PrimExpr recv_elements = 1;
  for (size_t i = 0; i < recv_range.size(); i++) {
    recv_elements *= recv_range[i]->extent;
  }
  recv_elements = analyzer->Simplify(recv_elements);
  // check for buffer sizes
  ICHECK(Downcast<IntImm>(send_elements)->value * recv_num <=
         Downcast<IntImm>(recv_elements)->value)
      << "Receive buffer size not enough for allgather: required "
      << (Downcast<IntImm>(send_elements)->value * recv_num) << ", but got "
      << Downcast<IntImm>(recv_elements)->value;

  // all checks passed, generate the calls
  Array<Stmt> bcast_stmts;

  if (direction == 0) { // horizontal
    for (int i = 0; i < mesh_x; i++) {
      for (size_t j = 0; j < mesh_y; j++) {
        Array<PrimExpr> args;
        args.push_back(send);
        args.push_back(recv);
        args.push_back(size);
        args.push_back(IntImm(DataType::Int(32), j) * send_elements); // offset
        args.push_back(IntImm(DataType::Int(32), i * mesh_y + j)); // src_core
        args.push_back(0); // direction: horizontal
        Broadcast bcast = Broadcast(args);
        Stmt bcast_stmt = bcast->Lower(T, analyzer);
        bcast_stmts.push_back(bcast_stmt);
      }
    }
  } else if (direction == 1) { // vertical
    for (int j = 0; j < mesh_y; j++) {
      for (size_t i = 0; i < mesh_x; i++) {
        Array<PrimExpr> args;
        args.push_back(send);
        args.push_back(recv);
        args.push_back(size);
        args.push_back(IntImm(DataType::Int(32), i) * send_elements); // offset
        args.push_back(IntImm(DataType::Int(32), i * mesh_y + j)); // src_core
        args.push_back(1); // direction: vertical
        Broadcast bcast = Broadcast(args);
        Stmt bcast_stmt = bcast->Lower(T, analyzer);
        bcast_stmts.push_back(bcast_stmt);
      }
    }
  } else if (direction == 2) { // all
    // first do horizontal allgather
    for (int i = 0; i < mesh_x; i++) {
      for (size_t j = 0; j < mesh_y; j++) {
        Array<PrimExpr> args;
        args.push_back(send);
        args.push_back(recv);
        args.push_back(size);
        args.push_back(IntImm(DataType::Int(32), i * mesh_y + j) *
                       send_elements);                             // offset
        args.push_back(IntImm(DataType::Int(32), i * mesh_y + j)); // src_core
        args.push_back(0); // direction: horizontal
        Broadcast bcast = Broadcast(args);
        Stmt bcast_stmt = bcast->Lower(T, analyzer);
        bcast_stmts.push_back(bcast_stmt);
      }
    }
    // then do vertical allgather
    Buffer recv_buffer = recv_region->buffer;
    int allgather_size = (size->value < 0)
                             ? Downcast<IntImm>(send_elements)->value * mesh_y
                             : size->value * mesh_y;

    for (int j = 0; j < mesh_y; j++) {
      for (size_t i = 0; i < mesh_x; i++) {
        Array<PrimExpr> args;
        args.push_back(recv_buffer.access_ptr(
            1, DataType::Handle(), 1,
            IntImm(DataType::Int(32), i * mesh_y) * send_elements,
            IntImm(DataType::Int(32), mesh_y) * send_elements));
        args.push_back(recv_buffer.access_ptr(
            2, DataType::Handle(), 1,
            IntImm(DataType::Int(32), i * mesh_y) * send_elements,
            IntImm(DataType::Int(32), mesh_y) * send_elements));
        args.push_back(IntImm(DataType::Int(32), allgather_size)); // size
        args.push_back(IntImm(DataType::Int(32), i * mesh_y + j)); // src_core
        args.push_back(1);                            // direction: vertical
        args.push_back(IntImm(DataType::Int(32), i)); // mask: current row only
        Stmt bcast_stmt =
            Evaluate(Call(DataType::Handle(), broadcast_(), args));
        bcast_stmts.push_back(bcast_stmt);
      }
    }
  }
  return SeqStmt::Flatten(bcast_stmts);
}

TIR_REGISTER_TL_TILE_OP(Allgather, comm_allgather)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  PutNode::RegisterReflection();
  BroadcastNode::RegisterReflection();
  AllgatherNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
