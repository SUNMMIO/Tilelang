/*!
 * \file tl/op/comm.cc
 * \brief Implementation of Inter-core Communication Operators
 */

#include "comm.h"

#include <algorithm>
#include <tvm/tir/op.h>
#include <vector>

#include "../layout/cute_layout.h"
#include "../target/sunmmio_utils.h"
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

/*!
 * \brief Sunmmio SRAM layout inference for symmetric comm ops.
 *
 * Propagates layout between src and dst via DeriveLayoutLike.
 * Always proposes — TryAssign handles priority and conflict detection.
 *
 * No validation here: the level-based priority system in TryAssign
 * determines whether proposals are accepted or rejected.  When a buffer
 * is immutable (T.annotate_layout / DRAM metadata), TryAssign catches
 * any kStrict conflict.  kCommon proposals against immutable entries
 * are silently rejected — they arise from BFS noise (e.g., derived from
 * kFree defaults) and are not real constraints.
 */
static LayoutMap SunmmioCommInferLayout(const LayoutInferArgs &T,
                                        const Buffer &src, const Buffer &dst,
                                        InferLevel level) {
  // Comm ops are propagation-only — no hard constraints at kStrict.
  if (level >= InferLevel::kStrict)
    return {};

  LayoutMap result;
  bool src_has = T.layout_map.count(src);
  bool dst_has = T.layout_map.count(dst);

  // ZN WSRAM dst: its src is staged as ZZ (the transfer does ZZ->ZN), so infer
  // ZZ for src. dst is already determined (ZN from GEMM); propose nothing for
  // it.
  if (dst_has && dst.scope() == kSunmmioScopeWSRAM &&
      !sunmmio::IsZZLike(T.layout_map[dst])) {
    int rank = static_cast<int>(src->shape.size());
    if (IsSunmmioSramScope(src.scope()) && rank >= 2) {
      Array<Integer> axes{Integer(rank - 2), Integer(rank - 1)};
      result.Set(src, sunmmio::MakeZZ(
                          src->shape, axes,
                          GetSunmmioLayoutBlockShape(T.target, src->dtype)));
    }
    return result;
  }

  // Propagate: derive layout for each side from the other.
  if (src_has) {
    auto derived = DeriveLayoutLike(T.layout_map[src], dst->shape);
    if (derived.defined()) {
      result.Set(dst, derived.value());
    }
  }
  if (dst_has) {
    auto derived = DeriveLayoutLike(T.layout_map[dst], src->shape);
    if (derived.defined()) {
      result.Set(src, derived.value());
    }
  }

  return result;
}

BroadcastOp::BroadcastOp(Array<PrimExpr> args,
                         Map<String, ObjectRef> annotations) {
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
  node->src_core = args[3];
  node->direction = Downcast<IntImm>(args[4])->value;
  // Optional trailing positional arg; default 0.
  if (args.size() > 5) {
    node->srcOffsetByte_ = Downcast<IntImm>(args[5])->value;
  }
  data_ = std::move(node);
}

TileOperator BroadcastOpNode::Clone() const {
  auto op = tvm::ffi::make_object<BroadcastOpNode>(*this);
  return BroadcastOp(op);
}

LayoutMap BroadcastOpNode::InferLayout(const LayoutInferArgs &T,
                                       InferLevel level) const {
  if (IsSunmmioSramScope(src.scope()) || IsSunmmioSramScope(dst.scope())) {
    return SunmmioCommInferLayout(T, src, dst, level);
  }

  // Non-Sunmmio: delegate to Copy.
  Array<PrimExpr> args;
  args.push_back(src_expr);
  args.push_back(dst_expr);
  Copy copy_op = Copy(args);
  return copy_op->InferLayout(T, level);
}

// Builds the fixed positional prefix of a broadcast_() call, in the layout
// declared by BroadcastArg (comm.h). Callers append trailing core-mask
// indices (kBroadcastArgMaskBegin onward) as needed.
Array<PrimExpr> MakeBroadcastArgs(PrimExpr src_region, PrimExpr dst_region,
                                  PrimExpr size, PrimExpr src_core,
                                  PrimExpr direction,
                                  PrimExpr src_offset_byte) {
  return {src_region, dst_region, size, src_core, direction, src_offset_byte};
}

Stmt BroadcastOpNode::Lower(const LowerArgs &T,
                            arith::Analyzer *analyzer) const {
  Target target = T.target;
  ICHECK(TargetIsSunmmio(target)) << "Broadcast only supports SUNMMIO targets.";
  auto mesh = GetSunmmioMeshConfig(target);
  int mesh_nrow = mesh.nrow;
  int mesh_ncol = mesh.ncol;

  // check for valid core id
  if (src_core.as<IntImmNode>()) {
    int src_core_val = src_core.as<IntImmNode>()->value;
    ICHECK(src_core_val >= 0 and src_core_val < mesh_nrow * mesh_ncol)
        << "Source core id " << src_core_val << " out of range [0, "
        << mesh_nrow * mesh_ncol << ")";
  }

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

  // check for valid direction
  if (direction != 0 and direction != 1 and direction != 2) {
    LOG(FATAL) << "Invalid broadcast direction " << direction
               << ", must be 0 (horizontal) or 1 (vertical) or 2 (all).";
  }

  // all checks passed, generate the call. srcOffsetByte_ is appended as the
  // final positional arg of every broadcast_() emitted by this BroadcastOp,
  // so codegen reads it from a stable call-arg slot — no AttrStmt wrapper.
  PrimExpr src_offset_imm = IntImm(DataType::Int(32), srcOffsetByte_);
  if (direction == 0 or direction == 1) {
    // 1D broadcast
    Array<PrimExpr> args =
        MakeBroadcastArgs(MakeRegionExpr(src, src_range, /*access_mask=*/1),
                          MakeRegionExpr(dst, dst_range, /*access_mask=*/2),
                          Downcast<IntImm>(broadcast_elements), src_core,
                          direction, src_offset_imm);
    Stmt broadcast = Evaluate(Call(DataType::Handle(), broadcast_(), args));
    return broadcast;
  } else {
    // 2D broadcast
    ICHECK(src_core.as<IntImmNode>())
        << "2D broadcast only supports constant source core id.";
    int src_core_val = src_core.as<IntImmNode>()->value;
    int src_core_col = src_core_val % mesh_ncol;

    Array<Stmt> seq;
    // vertical broadcast
    Array<PrimExpr> args =
        MakeBroadcastArgs(MakeRegionExpr(src, src_range, /*access_mask=*/1),
                          MakeRegionExpr(dst, dst_range, /*access_mask=*/2),
                          Downcast<IntImm>(broadcast_elements), src_core,
                          /*direction=*/1, src_offset_imm);
    Stmt broadcast = Evaluate(Call(DataType::Handle(), broadcast_(), args));
    seq.push_back(broadcast);
    // horizontal broadcast
    for (int i = 0; i < mesh_nrow; i++) {
      Array<PrimExpr> args = MakeBroadcastArgs(
          MakeRegionExpr(dst, dst_range, /*access_mask=*/1),
          MakeRegionExpr(dst, dst_range, /*access_mask=*/2),
          Downcast<IntImm>(broadcast_elements),
          int(i * mesh_ncol) + src_core_col, /*direction=*/0, src_offset_imm);
      Stmt broadcast = Evaluate(Call(DataType::Handle(), broadcast_(), args));
      seq.push_back(broadcast);
    }
    return SeqStmt::Flatten(seq);
  }
}

TIR_REGISTER_TL_TILE_OP(BroadcastOp, comm_broadcast)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

PutOp::PutOp(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
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
  node->src_core = args[3];
  node->dst_core = args[4];
  data_ = std::move(node);
}

TileOperator PutOpNode::Clone() const {
  auto op = tvm::ffi::make_object<PutOpNode>(*this);
  return PutOp(op);
}

LayoutMap PutOpNode::InferLayout(const LayoutInferArgs &T,
                                 InferLevel level) const {
  if (IsSunmmioSramScope(src.scope()) || IsSunmmioSramScope(dst.scope())) {
    return SunmmioCommInferLayout(T, src, dst, level);
  }

  // Non-Sunmmio: delegate to Copy.
  Array<PrimExpr> args;
  args.push_back(src_expr);
  args.push_back(dst_expr);
  Copy copy_op = Copy(args);
  return copy_op->InferLayout(T, level);
}

Stmt PutOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  Target target = T.target;
  ICHECK(TargetIsSunmmio(target)) << "Put only supports SUNMMIO targets.";
  auto mesh = GetSunmmioMeshConfig(target);
  int mesh_nrow = mesh.nrow;
  int mesh_ncol = mesh.ncol;

  // check for valid core id
  if (src_core.as<IntImmNode>()) {
    int src_core_val = src_core.as<IntImmNode>()->value;
    ICHECK(src_core_val >= 0 and src_core_val < mesh_nrow * mesh_ncol)
        << "Source core id " << src_core_val << " out of range [0, "
        << mesh_nrow * mesh_ncol << ")";
  }
  if (dst_core.as<IntImmNode>()) {
    int dst_core_val = dst_core.as<IntImmNode>()->value;
    ICHECK(dst_core_val >= 0 and dst_core_val < mesh_nrow * mesh_ncol)
        << "Destination core id " << dst_core_val << " out of range [0, "
        << mesh_nrow * mesh_ncol << ")";
  }

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
  ICHECK(src_core.as<IntImmNode>())
      << "Put only supports constant source core id.";
  ICHECK(dst_core.as<IntImmNode>())
      << "Put only supports constant destination core id.";
  int src_core_val = src_core.as<IntImmNode>()->value;
  int dst_core_val = dst_core.as<IntImmNode>()->value;
  int src_core_row = src_core_val / mesh_ncol;
  int src_core_col = src_core_val % mesh_ncol;
  int dst_core_row = dst_core_val / mesh_ncol;
  int dst_core_col = dst_core_val % mesh_ncol;

  if (src_core_row == dst_core_row) {
    // 1D put via horizontal communication. PutOp does not use
    // src_offset_byte, so it passes 0 in that slot; masks follow.
    Array<PrimExpr> args = MakeBroadcastArgs(
        MakeRegionExpr(src, src_range, /*access_mask=*/1),
        MakeRegionExpr(dst, dst_range, /*access_mask=*/2),
        Downcast<IntImm>(broadcast_elements), src_core, /*direction=*/0,
        /*src_offset_byte=*/IntImm(DataType::Int(32), 0));
    for (int j = 0; j < mesh_ncol; j++) {
      if (j != dst_core_col) {
        args.push_back(IntImm(DataType::Int(32),
                              j)); // mask: all cores except dst_core_col
      }
    }
    Stmt put = Evaluate(Call(DataType::Handle(), broadcast_(), args));
    return put;
  } else if (src_core_col == dst_core_col) {
    // 1D put via vertical communication.
    Array<PrimExpr> args = MakeBroadcastArgs(
        MakeRegionExpr(src, src_range, /*access_mask=*/1),
        MakeRegionExpr(dst, dst_range, /*access_mask=*/2),
        Downcast<IntImm>(broadcast_elements), src_core, /*direction=*/1,
        /*src_offset_byte=*/IntImm(DataType::Int(32), 0));
    for (int i = 0; i < mesh_nrow; i++) {
      if (i != dst_core_row) {
        args.push_back(IntImm(DataType::Int(32),
                              i)); // mask: all cores except dst_core_row
      }
    }
    Stmt put = Evaluate(Call(DataType::Handle(), broadcast_(), args));
    return put;
  } else {
    Array<Stmt> seq;
    // vertical transfer from src core to intermediate core
    int intermediate_core_id = dst_core_row * mesh_ncol + src_core_col;
    Array<PrimExpr> args1 = MakeBroadcastArgs(
        MakeRegionExpr(src, src_range, /*access_mask=*/1),
        MakeRegionExpr(dst, dst_range, /*access_mask=*/2),
        Downcast<IntImm>(broadcast_elements), src_core, /*direction=*/1,
        /*src_offset_byte=*/IntImm(DataType::Int(32), 0));
    for (int i = 0; i < mesh_nrow; i++) {
      if (i != dst_core_row) {
        args1.push_back(IntImm(DataType::Int(32),
                               i)); // mask: all cores except dst_core_row
      }
    }
    Stmt put1 = Evaluate(Call(DataType::Handle(), broadcast_(), args1));
    seq.push_back(put1);
    // horizontal transfer from intermediate core to dst core
    Array<PrimExpr> args2 = MakeBroadcastArgs(
        MakeRegionExpr(dst, dst_range, /*access_mask=*/1),
        MakeRegionExpr(dst, dst_range, /*access_mask=*/2),
        Downcast<IntImm>(broadcast_elements),
        IntImm(DataType::Int(32), intermediate_core_id), /*direction=*/0,
        /*src_offset_byte=*/IntImm(DataType::Int(32), 0));
    for (int j = 0; j < mesh_ncol; j++) {
      if (j != dst_core_col) {
        args2.push_back(IntImm(DataType::Int(32),
                               j)); // mask: all cores except dst_core_col
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

AllgatherOp::AllgatherOp(Array<PrimExpr> args,
                         Map<String, ObjectRef> annotations) {
  ObjectPtr<AllgatherOpNode> node = tvm::ffi::make_object<AllgatherOpNode>();
  node->send = args[0];
  node->recv = args[1];
  node->direction = Downcast<IntImm>(args[2])->value;
  node->size = Downcast<IntImm>(args[3]);
  // axis is optional for backwards compatibility: -1 = legacy mode.
  node->axis = (args.size() > 4) ? Downcast<IntImm>(args[4])->value : -1;
  node->annotations = annotations;
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
  if (IsSunmmioSramScope(src.scope()) || IsSunmmioSramScope(dst.scope())) {
    return SunmmioCommInferLayout(T, src, dst, level);
  }

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
  auto mesh = GetSunmmioMeshConfig(target);
  int mesh_nrow = mesh.nrow;
  int mesh_ncol = mesh.ncol;

  Array<Range> send_range, recv_range;
  auto send_region = NormalizeToBufferRegion(send);
  auto recv_region = NormalizeToBufferRegion(recv);
  send_range = send_region->region;
  recv_range = recv_region->region;

  int recv_num = 1;
  if (direction == 0) { // horizontal
    recv_num = mesh_ncol;
  } else if (direction == 1) { // vertical
    recv_num = mesh_nrow;
  } else if (direction == 2) { // all
    recv_num = mesh_nrow * mesh_ncol;
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

  // Unified lowering for all axis modes. Every per-core contribution lands
  // in an equal-size slot along a single axis of recv ("slice_axis"). Slot
  // offset is encoded in the region's `min`, not in a separate offset arg.
  //
  //   axis == -1 (sentinel, "no axis") or axis == 0 -> slice_axis = 0
  //     legacy recv [K, d0, ...]:        slot_extent = 1
  //     axis=0 recv [K*d0, d1, ...]:     slot_extent = d0
  //   axis > 0 (last dim of send)        -> slice_axis = recv_rank - 1
  //     axis=-1 recv [d0, ..., K*dn-1]:  slot_extent = dn-1
  Buffer send_buffer = send_region->buffer;
  Buffer recv_buffer = recv_region->buffer;
  int recv_rank = static_cast<int>(recv_range.size());
  ICHECK_GT(recv_rank, 0) << "Allgather recv must have at least one dim.";
  if (axis > 0) {
    ICHECK_EQ(axis, static_cast<int>(send_range.size()) - 1)
        << "Only axis = last dim of send is supported; got axis=" << axis
        << " for send rank=" << send_range.size();
    ICHECK_EQ(recv_range.size(), send_range.size())
        << "In axis mode, recv and send must have the same rank.";
  }
  int slice_axis = (axis > 0) ? (recv_rank - 1) : 0;
  ICHECK(analyzer->CanProve(FloorMod(recv_range[slice_axis]->extent,
                                     IntImm(DataType::Int(32), recv_num)) == 0))
      << "Recv extent along slice axis " << slice_axis << " ("
      << recv_range[slice_axis]->extent << ") must be divisible by recv_num ("
      << recv_num << ").";
  PrimExpr slot_extent_p1 = analyzer->Simplify(FloorDiv(
      recv_range[slice_axis]->extent, IntImm(DataType::Int(32), recv_num)));
  IntImm bcast_size_imm =
      (size->value < 0) ? Downcast<IntImm>(send_elements) : size;

  // Build a sub-region of recv spanning a single slab of width `slot_extent`
  // along `slice_axis`, starting at slot index `slot_start`. All other dims
  // pass through `recv_range` unchanged.
  auto make_slab = [&](int slot_start, PrimExpr slot_extent) {
    Array<Range> ranges;
    for (int d = 0; d < recv_rank; d++) {
      if (d == slice_axis) {
        PrimExpr base = analyzer->Simplify(
            recv_range[d]->min +
            IntImm(DataType::Int(32), slot_start) * slot_extent);
        ranges.push_back(Range::FromMinExtent(base, slot_extent));
      } else {
        ranges.push_back(recv_range[d]);
      }
    }
    return ranges;
  };

  Array<Stmt> bcast_stmts;

  // Propagate src_offset_byte (read from this op's annotations) into each
  // BroadcastOp we construct, so it lands as the final positional arg of
  // every emitted broadcast_() call. No AttrStmt wrapping needed.
  int src_offset_byte = GetSrcOffsetByte();
  PrimExpr src_offset_imm = IntImm(DataType::Int(32), src_offset_byte);

  auto emit = [&](Array<Range> dst_ranges, IntImm bcast_elems, int src_core_id,
                  int bcast_dir, bool src_from_send) {
    Array<PrimExpr> args;
    if (src_from_send) {
      args.push_back(MakeRegionExpr(send_buffer, send_range, /*mask=*/1));
    } else {
      args.push_back(MakeRegionExpr(recv_buffer, dst_ranges, /*mask=*/1));
    }
    args.push_back(MakeRegionExpr(recv_buffer, dst_ranges, /*mask=*/2));
    args.push_back(bcast_elems);
    args.push_back(IntImm(DataType::Int(32), src_core_id));
    args.push_back(IntImm(DataType::Int(32), bcast_dir));
    args.push_back(src_offset_imm);
    BroadcastOp bcast(args);
    bcast_stmts.push_back(bcast->Lower(T, analyzer));
  };

  if (direction == 0) { // horizontal
    for (int i = 0; i < mesh_nrow; i++) {
      for (int j = 0; j < mesh_ncol; j++) {
        emit(make_slab(/*slot_start=*/j, slot_extent_p1), bcast_size_imm,
             /*src_core=*/i * mesh_ncol + j, /*bcast_dir=*/0,
             /*src_from_send=*/true);
      }
    }
  } else if (direction == 1) { // vertical
    for (int j = 0; j < mesh_ncol; j++) {
      for (int i = 0; i < mesh_nrow; i++) {
        emit(make_slab(/*slot_start=*/i, slot_extent_p1), bcast_size_imm,
             /*src_core=*/i * mesh_ncol + j, /*bcast_dir=*/1,
             /*src_from_send=*/true);
      }
    }
  } else { // direction == 2 ("all")
    // Phase 1: horizontal. Core (i,j) lands at slot (i*ncol + j).
    for (int i = 0; i < mesh_nrow; i++) {
      for (int j = 0; j < mesh_ncol; j++) {
        emit(make_slab(/*slot_start=*/i * mesh_ncol + j, slot_extent_p1),
             bcast_size_imm,
             /*src_core=*/i * mesh_ncol + j, /*bcast_dir=*/0,
             /*src_from_send=*/true);
      }
    }
    // Phase 2: vertical. Each row's mesh_ncol-wide gathered slab (rows
    // i*ncol .. (i+1)*ncol of phase-1 slots) is broadcast to every row in
    // each column.
    PrimExpr row_extent = analyzer->Simplify(
        IntImm(DataType::Int(32), mesh_ncol) * slot_extent_p1);
    IntImm row_size_imm =
        IntImm(DataType::Int(32), bcast_size_imm->value * mesh_ncol);
    for (int j = 0; j < mesh_ncol; j++) {
      for (int i = 0; i < mesh_nrow; i++) {
        // slab index `i` covers slots [i*ncol .. (i+1)*ncol) of phase 1;
        // offset along slice_axis is i * row_extent = i*ncol * slot_extent_p1.
        emit(make_slab(/*slot_index=*/i, row_extent), row_size_imm,
             /*src_core=*/i * mesh_ncol + j, /*bcast_dir=*/1,
             /*src_from_send=*/false);
      }
    }
  }

  return SeqStmt::Flatten(bcast_stmts);
}

TIR_REGISTER_TL_TILE_OP(AllgatherOp, comm_allgather)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

AllreduceOp::AllreduceOp(Array<PrimExpr> args,
                         Map<String, ObjectRef> annotations) {
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

  if (IsSunmmioSramScope(src.scope()) || IsSunmmioSramScope(dst.scope())) {
    return SunmmioCommInferLayout(T, src, dst, level);
  }

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
  auto mesh = GetSunmmioMeshConfig(target);
  int mesh_nrow = mesh.nrow;
  int mesh_ncol = mesh.ncol;

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
