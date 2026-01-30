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

BroadcastOp::BroadcastOp(Array<PrimExpr> args) {
  ObjectPtr<BroadcastOpNode> node = tvm::ffi::make_object<BroadcastOpNode>();
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

TileOperator BroadcastOpNode::Clone() const {
  auto op = tvm::ffi::make_object<BroadcastOpNode>(*this);
  return BroadcastOp(op);
}

LayoutMap BroadcastOpNode::InferLayout(const LayoutInferArgs &T,
                                       InferLevel level) const {
  Array<PrimExpr> args;
  args.push_back(src_expr);
  args.push_back(dst_expr);
  Copy copy_op = Copy(args);
  LayoutMap out_layout = copy_op->InferLayout(T, level);
  return out_layout;
}

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
      std::string s = m.substr(m.find_last_of('_') + 1);
      ;
      try {
        x = std::stoi(s);
      } catch (const std::invalid_argument &e) {
        x = -1;
      } catch (const std::out_of_range &e) {
        x = -1;
      }
    }
  }
  ICHECK(x != 0) << axis_str << " not found.";
  ICHECK(x > 0) << "Invalid " << axis_str;
  return x;
}

Stmt BroadcastOpNode::Lower(const LowerArgs &T,
                            arith::Analyzer *analyzer) const {
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
    // horizontal broadcast
    for (int i = 0; i < mesh_x; i++) {
      Array<PrimExpr> args;
      args.push_back(dst.access_ptr(1, DataType::Handle(), 1, 0, dst_elements));
      args.push_back(dst.access_ptr(2, DataType::Handle(), 1, 0, dst_elements));
      args.push_back(Downcast<IntImm>(broadcast_elements));
      args.push_back(int(i * mesh_y) + src_core_y);
      args.push_back(0); // direction: horizontal
      Stmt broadcast = Evaluate(Call(DataType::Handle(), broadcast_(), args));
      seq.push_back(broadcast);
    }
    return SeqStmt::Flatten(seq);
  }
}

TIR_REGISTER_TL_TILE_OP(BroadcastOp, comm_broadcast)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

PutOp::PutOp(Array<PrimExpr> args) {
  ObjectPtr<PutOpNode> node = tvm::ffi::make_object<PutOpNode>();
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

TileOperator PutOpNode::Clone() const {
  auto op = tvm::ffi::make_object<PutOpNode>(*this);
  return PutOp(op);
}

LayoutMap PutOpNode::InferLayout(const LayoutInferArgs &T,
                                 InferLevel level) const {
  Array<PrimExpr> args;
  args.push_back(src_expr);
  args.push_back(dst_expr);
  Copy copy_op = Copy(args);
  LayoutMap out_layout = copy_op->InferLayout(T, level);
  return out_layout;
}

Stmt PutOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
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
      << "Put size larger than data size: " << size->value << " vs "
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
      << "Put size Larger than source buffer size: "
      << (Downcast<IntImm>(broadcast_elements)->value) << " vs "
      << Downcast<IntImm>(src_elements)->value;
  ICHECK((Downcast<IntImm>(broadcast_elements)->value) <=
         Downcast<IntImm>(dst_elements)->value)
      << "Put size larger than destination buffer size: "
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

TIR_REGISTER_TL_TILE_OP(PutOp, comm_put)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

AllgatherOp::AllgatherOp(Array<PrimExpr> args) {
  ObjectPtr<AllgatherOpNode> node = tvm::ffi::make_object<AllgatherOpNode>();
  node->send = args[0];
  node->recv = args[1];
  node->direction = Downcast<IntImm>(args[2])->value;
  node->size = Downcast<IntImm>(args[3]);
  data_ = std::move(node);
}

TileOperator AllgatherOpNode::Clone() const {
  auto op = tvm::ffi::make_object<AllgatherOpNode>(*this);
  return AllgatherOp(op);
}

// Not yet complete; it will be further refined later
LayoutMap AllgatherOpNode::ComputeLayout(const LayoutInferArgs &T,
                                         InferLevel level, Buffer src,
                                         Buffer dst) const {
  if (src.scope() == "local.fragment" && dst.scope() == "local.fragment" &&
      T.layout_map.count(src)) {
    auto src_layout = T.layout_map[src].as<Fragment>().value();

    PrimExpr src_rep_extent = src_layout->ReplicateExtent();

    Array<PrimExpr> fwd;
    fwd.push_back(InputPlaceholder(0));
    for (int i = 0; i < static_cast<int>(src->shape.size()); i++) {
      fwd.push_back(InputPlaceholder(i + 1));
    }
    auto thd = src_layout->ForwardThread(fwd, std::nullopt);

    Fragment dst_layout =
        Fragment(dst->shape, {}, thd, src_rep_extent, std::nullopt)
            ->CondenseReplicateVar()
            ->BindThreadRange(T.thread_bounds);

    if (!T.layout_map.count(dst))
      return {{dst, dst_layout}};
    else {
      // Check if computed layout is compatible with existing: the existing one
      // must strictly contains the computed layout
      auto orig_dst_layout =
          T.layout_map.Get(dst).value().as<Fragment>().value();
      ICHECK(dst_layout->InputDim() == orig_dst_layout->InputDim());
      Array<PrimExpr> indices;
      indices.reserve(dst_layout->InputDim());
      arith::Analyzer inner_analyzer;
      for (int i = 0; i < dst_layout->InputDim(); ++i) {
        auto x = InputPlaceholder(i);
        indices.push_back(x);
        // should be literal - literal = 0, any analyzer will work
        ICHECK(is_zero(inner_analyzer.Simplify(
            dst_layout->InputShape()[i] - orig_dst_layout->InputShape()[i])));
        inner_analyzer.Bind(x, Range(0, dst_layout->InputShape()[i]));
      }

      ICHECK(as_const_int(dst_layout->ReplicateExtent()));
      ICHECK(as_const_int(src_layout->ReplicateExtent()));
      auto dst_rep = *as_const_int(dst_layout->ReplicateExtent());
      auto src_rep = *as_const_int(src_layout->ReplicateExtent());
      if (dst_rep < src_rep ||
          !ProveFragmentContains(orig_dst_layout, dst_layout, indices, indices,
                                 inner_analyzer)) {
        std::ostringstream oss;
        oss << "Layout may conflict with ReduceOp for buffer " << dst << " vs. "
            << src << "\nLHS = " << src_layout->DebugOutput()
            << "\nRHS = " << orig_dst_layout->DebugOutput()
            << "\nYou may need to use a shared memory to transform the "
               "layout";
        throw LayoutConflictException(oss.str());
      }

      if (dst_rep > src_rep) {
        return {{dst, dst_layout}};
      }
    }
  }
  return {};
}

LayoutMap AllgatherOpNode::InferLayout(const LayoutInferArgs &T,
                                       InferLevel level) const {
  Buffer src_buffer = NormalizeToBufferRegion(send)->buffer;
  Buffer recv_buffer = NormalizeToBufferRegion(recv)->buffer;
  return ComputeLayout(T, level, src_buffer, recv_buffer);
}

Stmt AllgatherOpNode::Lower(const LowerArgs &T,
                            arith::Analyzer *analyzer) const {
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
        BroadcastOp bcast = BroadcastOp(args);
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
        BroadcastOp bcast = BroadcastOp(args);
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
        BroadcastOp bcast = BroadcastOp(args);
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
        args.push_back(1); // direction: vertical
        Stmt bcast_stmt =
            Evaluate(Call(DataType::Handle(), broadcast_(), args));
        bcast_stmts.push_back(bcast_stmt);
      }
    }
  }
  return SeqStmt::Flatten(bcast_stmts);
}

TIR_REGISTER_TL_TILE_OP(AllgatherOp, comm_allgather)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

AllreduceOp::AllreduceOp(Array<PrimExpr> args) {
  ObjectPtr<AllreduceOpNode> node = tvm::ffi::make_object<AllreduceOpNode>();
  node->src = args[0];
  node->dst = args[1];
  node->row_allgather = args[2];
  node->col_allgather = args[3];

  node->type = Downcast<StringImm>(args[4]);
  node->direction = Downcast<IntImm>(args[5])->value;
  node->dim = Downcast<IntImm>(args[6]);
  node->clear = Downcast<IntImm>(args[7]);
  if (args.size() > 8) {
    node->dst_copy = args[8];
  }
  data_ = std::move(node);
}

TileOperator AllreduceOpNode::Clone() const {
  auto op = tvm::ffi::make_object<AllreduceOpNode>(*this);
  return AllreduceOp(op);
}

// Not yet complete; it will be further refined later
LayoutMap AllreduceOpNode::ComputeLayout(const LayoutInferArgs &T,
                                         InferLevel level, Buffer src,
                                         Buffer dst, int dim) const {
  if (level >= InferLevel::kStrict)
    return {};

  if (src.scope() == "local.fragment" && dst.scope() == "local.fragment" &&
      T.layout_map.count(src)) {
    auto src_layout = T.layout_map[src].as<Fragment>().value();

    PrimExpr indice_rep_extent = src->shape[dim];
    PrimExpr src_rep_extent = src_layout->ReplicateExtent();
    PrimExpr dest_buffer_rep_extent = indice_rep_extent * src_rep_extent;

    Array<PrimExpr> fwd;
    fwd.push_back(InputPlaceholder(0));
    for (int i = 0; i < static_cast<int>(src->shape.size()); i++) {
      if (i == dim) {
        ;
      } else if (i < dim) {
        fwd.push_back(InputPlaceholder(i + 1));
      } else if (i > dim) {
        fwd.push_back(InputPlaceholder(i - 1 + 1));
      }
    }
    auto thd = src_layout->ForwardThread(
        fwd, FloorDiv(ReplicationPlaceholder(), indice_rep_extent));

    // Ensure the thread count is divisible by the replicate extent.
    // Otherwise, we cannot infer a valid fragment<->fragment layout.
    {
      arith::Analyzer analyzer;
      PrimExpr num_threads = T.thread_bounds->extent;
      // Though the dest_buffer_rep_extent will be compressed at
      // CondenseReplicateVar, we need to check the divisibility here to avoid
      // the issue that the thread count is not divisible by the replicate
      // extent.
      if (!analyzer.CanProve(FloorMod(num_threads, dest_buffer_rep_extent) ==
                             0) &&
          !analyzer.CanProve(FloorMod(dest_buffer_rep_extent, num_threads) ==
                             0)) {
        ICHECK(false) << "ReduceOp fragment layout inference failed: "
                         "num_threads % replicate_extent != 0. "
                      << "This mapping requires the block's thread count to be "
                         "divisible by the "
                      << "replicate extent. "
                      << "Try one of: (1) choose a thread block size divisible "
                         "by replicate_extent; "
                      << "(2) pick a different reduce dimension or adjust the "
                         "source fragment layout; "
                      << "Details: num_threads=" << num_threads
                      << ", replicate_extent=" << indice_rep_extent
                      << ", src=" << src << ", dst=" << dst;
      }
    }

    Fragment dst_layout =
        Fragment(dst->shape, {}, thd, dest_buffer_rep_extent, std::nullopt)
            ->CondenseReplicateVar()
            ->BindThreadRange(T.thread_bounds);

    if (!T.layout_map.count(dst))
      return {{dst, dst_layout}};
    else {
      // Check if computed layout is compatible with existing: the existing one
      // must strictly contains the computed layout
      auto orig_dst_layout =
          T.layout_map.Get(dst).value().as<Fragment>().value();
      ICHECK(dst_layout->InputDim() == orig_dst_layout->InputDim());
      Array<PrimExpr> indices;
      indices.reserve(dst_layout->InputDim());
      arith::Analyzer inner_analyzer;
      for (int i = 0; i < dst_layout->InputDim(); ++i) {
        auto x = InputPlaceholder(i);
        indices.push_back(x);
        // should be literal - literal = 0, any analyzer will work
        ICHECK(is_zero(inner_analyzer.Simplify(
            dst_layout->InputShape()[i] - orig_dst_layout->InputShape()[i])));
        inner_analyzer.Bind(x, Range(0, dst_layout->InputShape()[i]));
      }

      ICHECK(as_const_int(dst_layout->ReplicateExtent()));
      ICHECK(as_const_int(src_layout->ReplicateExtent()));
      auto dst_rep = *as_const_int(dst_layout->ReplicateExtent());
      auto src_rep = *as_const_int(src_layout->ReplicateExtent());
      if (dst_rep < src_rep ||
          !ProveFragmentContains(orig_dst_layout, dst_layout, indices, indices,
                                 inner_analyzer)) {
        std::ostringstream oss;
        oss << "Layout may conflict with ReduceOp for buffer " << dst << " vs. "
            << src << "\nLHS = " << src_layout->DebugOutput()
            << "\nRHS = " << orig_dst_layout->DebugOutput()
            << "\nYou may need to use a shared memory to transform the "
               "layout";
        throw LayoutConflictException(oss.str());
      }

      if (dst_rep > src_rep) {
        return {{dst, dst_layout}};
      }
    }
  }
  return {};
}

LayoutMap AllreduceOpNode::InferLayout(const LayoutInferArgs &T,
                                       InferLevel level) const {
  LayoutMap lm;

  Array<PrimExpr> dst_layout_args;
  dst_layout_args.push_back(src);
  dst_layout_args.push_back(dst);
  dst_layout_args.push_back(type);
  dst_layout_args.push_back(dim);
  dst_layout_args.push_back(clear);
  ReduceOp dst_layout_op = ReduceOp(dst_layout_args);
  LayoutMap dst_layout_map = dst_layout_op->InferLayout(T, InferLevel::kFree);
  for (const auto &kv : dst_layout_map) {
    lm.Set(kv.first, kv.second);
  }

  if (dst_copy.defined()) {
    Array<PrimExpr> dst_copy_layout_args;
    dst_copy_layout_args.push_back(src);
    dst_copy_layout_args.push_back(dst_copy);
    dst_copy_layout_args.push_back(type);
    dst_copy_layout_args.push_back(dim);
    dst_copy_layout_args.push_back(clear);
    ReduceOp dst_copy_layout_op = ReduceOp(dst_copy_layout_args);
    LayoutMap dst_copy_layout_map =
        dst_copy_layout_op->InferLayout(T, InferLevel::kFree);
    for (const auto &kv : dst_copy_layout_map) {
      lm.Set(kv.first, kv.second);
    }
  }

  Buffer row_allgather_buffer = NormalizeToBufferRegion(row_allgather)->buffer;
  LayoutMap row_allgather_layout =
      ComputeLayout(T, InferLevel::kFree, NormalizeToBufferRegion(src)->buffer,
                    row_allgather_buffer, dim->value);
  for (const auto &kv : row_allgather_layout) {
    lm.Set(kv.first, kv.second);
  }

  Buffer col_allgather_buffer = NormalizeToBufferRegion(col_allgather)->buffer;
  LayoutMap col_allgather_layout =
      ComputeLayout(T, InferLevel::kFree, NormalizeToBufferRegion(src)->buffer,
                    col_allgather_buffer, dim->value);
  for (const auto &kv : col_allgather_layout) {
    lm.Set(kv.first, kv.second);
  }

  return lm;
}

Stmt AllreduceOpNode::Lower(const LowerArgs &T,
                            arith::Analyzer *analyzer) const {
  Target target = T.target;
  ICHECK(TargetIsSunmmio(target)) << "Allreduce only supports SUNMMIO targets.";
  int mesh_x = get_target_mesh(target, 0);
  int mesh_y = get_target_mesh(target, 1);

  ICHECK(direction == 0 || direction == 1 || direction == 2)
      << "Invalid allreduce direction " << direction
      << ", must be 0 (row-wise) or 1 (column-wise) or 2 (all).";

  Array<Stmt> stmts;

  if (clear.as<Bool>().value() == true) {
    // Local reduce to dst
    Array<PrimExpr> local_reduce_args;
    local_reduce_args.push_back(src);
    local_reduce_args.push_back(dst);
    local_reduce_args.push_back(type);
    local_reduce_args.push_back(dim);
    local_reduce_args.push_back(IntImm(DataType::Int(32), 1)); // clear = true
    ReduceOp local_reduce_op = ReduceOp(local_reduce_args);
    Stmt local_reduce_stmt = local_reduce_op->Lower(T, analyzer);
    stmts.push_back(local_reduce_stmt);

    if (direction == 0 or direction == 2) { // row-wise
      // Allgather dst in rows to row_allgather
      Array<PrimExpr> row_allgather_args;
      row_allgather_args.push_back(dst);
      row_allgather_args.push_back(row_allgather);
      row_allgather_args.push_back(
          IntImm(DataType::Int(32), 0)); // direction = horizontal
      row_allgather_args.push_back(IntImm(DataType::Int(32), -1)); // size
      AllgatherOp row_allgather_op = AllgatherOp(row_allgather_args);
      Stmt row_allgather_stmt = row_allgather_op->Lower(T, analyzer);
      stmts.push_back(row_allgather_stmt);

      // Local reduce from row_allgather to dst
      Array<PrimExpr> row_reduce_args;
      row_reduce_args.push_back(row_allgather);
      row_reduce_args.push_back(dst);
      row_reduce_args.push_back(type);
      row_reduce_args.push_back(IntImm(DataType::Int(32), 0)); // dim
      row_reduce_args.push_back(IntImm(DataType::Int(32), 1)); // clear = true
      ReduceOp row_reduce_op = ReduceOp(row_reduce_args);
      Stmt row_reduce_stmt = row_reduce_op->Lower(T, analyzer);
      stmts.push_back(row_reduce_stmt);
    }

    if (direction == 1 or direction == 2) { // column-wise
      // Allgather dst in columns to col_allgather
      Array<PrimExpr> col_allgather_args;
      col_allgather_args.push_back(dst);
      col_allgather_args.push_back(col_allgather);
      col_allgather_args.push_back(
          IntImm(DataType::Int(32), 1)); // direction = vertical
      col_allgather_args.push_back(IntImm(DataType::Int(32), -1)); // size
      AllgatherOp col_allgather_op = AllgatherOp(col_allgather_args);
      Stmt col_allgather_stmt = col_allgather_op->Lower(T, analyzer);
      stmts.push_back(col_allgather_stmt);

      // Local reduce from col_allgather to dst
      Array<PrimExpr> col_reduce_args;
      col_reduce_args.push_back(col_allgather);
      col_reduce_args.push_back(dst);
      col_reduce_args.push_back(type);
      col_reduce_args.push_back(IntImm(DataType::Int(32), 0)); // dim
      col_reduce_args.push_back(IntImm(DataType::Int(32), 1)); // clear = true
      ReduceOp col_reduce_op = ReduceOp(col_reduce_args);
      Stmt col_reduce_stmt = col_reduce_op->Lower(T, analyzer);
      stmts.push_back(col_reduce_stmt);
    }
  } else {
    // Local reduce to dst_copy
    Array<PrimExpr> local_reduce_args;
    local_reduce_args.push_back(src);
    local_reduce_args.push_back(dst_copy);
    local_reduce_args.push_back(type);
    local_reduce_args.push_back(dim);
    local_reduce_args.push_back(IntImm(DataType::Int(32), 1)); // clear = true
    ReduceOp local_reduce_op = ReduceOp(local_reduce_args);
    Stmt local_reduce_stmt = local_reduce_op->Lower(T, analyzer);
    stmts.push_back(local_reduce_stmt);

    if (direction == 0 or direction == 2) { // row-wise
      // Allgather dst in rows to row_allgather
      Array<PrimExpr> row_allgather_args;
      row_allgather_args.push_back(dst_copy);
      row_allgather_args.push_back(row_allgather);
      row_allgather_args.push_back(
          IntImm(DataType::Int(32), 0)); // direction = horizontal
      row_allgather_args.push_back(IntImm(DataType::Int(32), -1)); // size
      AllgatherOp row_allgather_op = AllgatherOp(row_allgather_args);
      Stmt row_allgather_stmt = row_allgather_op->Lower(T, analyzer);
      stmts.push_back(row_allgather_stmt);

      // Local reduce from row_allgather to dst
      Array<PrimExpr> row_reduce_args;
      row_reduce_args.push_back(row_allgather);
      row_reduce_args.push_back(direction == 0 ? dst : dst_copy);
      row_reduce_args.push_back(type);
      row_reduce_args.push_back(IntImm(DataType::Int(32), 0)); // dim
      row_reduce_args.push_back(IntImm(
          DataType::Int(32),
          direction == 0 ? 0 : 1)); // clear = direction == 0 ? false : true
      ReduceOp row_reduce_op = ReduceOp(row_reduce_args);
      Stmt row_reduce_stmt = row_reduce_op->Lower(T, analyzer);
      stmts.push_back(row_reduce_stmt);
    }

    if (direction == 1 or direction == 2) { // column-wise
      // Allgather dst in columns to col_allgather
      Array<PrimExpr> col_allgather_args;
      col_allgather_args.push_back(dst_copy);
      col_allgather_args.push_back(col_allgather);
      col_allgather_args.push_back(
          IntImm(DataType::Int(32), 1)); // direction = vertical
      col_allgather_args.push_back(IntImm(DataType::Int(32), -1)); // size
      AllgatherOp col_allgather_op = AllgatherOp(col_allgather_args);
      Stmt col_allgather_stmt = col_allgather_op->Lower(T, analyzer);
      stmts.push_back(col_allgather_stmt);

      // Local reduce from col_allgather to dst
      Array<PrimExpr> col_reduce_args;
      col_reduce_args.push_back(col_allgather);
      col_reduce_args.push_back(dst);
      col_reduce_args.push_back(type);
      col_reduce_args.push_back(IntImm(DataType::Int(32), 0)); // dim
      col_reduce_args.push_back(IntImm(DataType::Int(32), 0)); // clear = false
      ReduceOp col_reduce_op = ReduceOp(col_reduce_args);
      Stmt col_reduce_stmt = col_reduce_op->Lower(T, analyzer);
      stmts.push_back(col_reduce_stmt);
    }
  }

  return SeqStmt::Flatten(stmts);
}

TIR_REGISTER_TL_TILE_OP(AllreduceOp, comm_allreduce)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  PutOpNode::RegisterReflection();
  BroadcastOpNode::RegisterReflection();
  AllgatherOpNode::RegisterReflection();
  AllreduceOpNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
