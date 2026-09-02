/*!
 * \file tl/op/dist_comm.cc
 * \brief Rank-level communication TIR operations.
 */

#include "dist_comm.h"

#include <tvm/tir/op_attr_types.h>

#include "../layout/cute_layout.h"
#include "../target/sunmmio_utils.h"
#include "../target/utils.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tir;

#define TIR_DEFINE_DIST_BUILTIN(FuncName, OpName, PrinterName, NumInputs,      \
                                EffectKind)                                    \
  const Op &FuncName() {                                                       \
    static const Op &op = Op::Get(OpName);                                     \
    return op;                                                                 \
  }                                                                            \
  TVM_REGISTER_OP(OpName)                                                      \
      .set_num_inputs(NumInputs)                                               \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", PrinterName)         \
      .set_attr<TCallEffectKind>("TCallEffectKind",                            \
                                 Integer(CallEffectKind::EffectKind))

TIR_DEFINE_DIST_BUILTIN(dist_signal_decl, "tl.dist_signal_decl",
                        "dist_signal_decl", 2, kPure);
TIR_DEFINE_DIST_BUILTIN(dist_signal, "tl.dist_signal", "dist_signal", 2, kPure);
TIR_DEFINE_DIST_BUILTIN(dist_put_, "tl.dist_put_", "dist_put_", 6, kOpaque);
TIR_DEFINE_DIST_BUILTIN(dist_wait_signal_, "tl.dist_wait_signal_",
                        "dist_wait_signal_", 4, kOpaque);
TIR_DEFINE_DIST_BUILTIN(dist_wait_all, "tl.dist_wait_all", "dist_wait_all", -1,
                        kOpaque);
TIR_DEFINE_DIST_BUILTIN(dist_wait_send, "tl.dist_wait_send", "dist_wait_send",
                        0, kOpaque);
TIR_DEFINE_DIST_BUILTIN(dist_expect, "tl.dist_expect", "dist_expect", 2,
                        kOpaque);
TIR_DEFINE_DIST_BUILTIN(dist_expect_, "tl.dist_expect_", "dist_expect_", 2,
                        kOpaque);
TIR_DEFINE_DIST_BUILTIN(dist_route, "tl.dist_route", "dist_route", 3, kPure);
TIR_DEFINE_DIST_BUILTIN(dist_route_table, "tl.dist_route_table",
                        "dist_route_table", -1, kPure);
TIR_DEFINE_DIST_BUILTIN(dist_peer_route, "tl.dist_peer_route",
                        "dist_peer_route", 2, kPure);
TIR_DEFINE_DIST_BUILTIN(dist_peer_route_table, "tl.dist_peer_route_table",
                        "dist_peer_route_table", -1, kPure);
TIR_DEFINE_DIST_BUILTIN(dist_rank_routed_put, "tl.dist_rank_routed_put",
                        "dist_rank_routed_put", 6, kOpaque);
TIR_DEFINE_DIST_BUILTIN(dist_routed_put, "tl.dist_routed_put",
                        "dist_routed_put", 5, kOpaque);

#undef TIR_DEFINE_DIST_BUILTIN

namespace {

LayoutMap InferMatchingLayouts(const LayoutInferArgs &T, const Buffer &src,
                               const Buffer &dst, InferLevel level) {
  if (level >= InferLevel::kStrict) {
    return {};
  }

  LayoutMap result;
  if (T.layout_map.count(src) && IsSunmmioSramScope(dst.scope())) {
    auto derived =
        DeriveLayoutLikeForDType(T.layout_map[src], dst->shape, dst->dtype);
    if (derived.defined()) {
      result.Set(dst, derived.value());
    }
  }
  if (T.layout_map.count(dst) && IsSunmmioSramScope(src.scope())) {
    auto derived =
        DeriveLayoutLikeForDType(T.layout_map[dst], src->shape, src->dtype);
    if (derived.defined()) {
      result.Set(src, derived.value());
    }
  }
  return result;
}

bool IsDistDramBuffer(const Buffer &buffer) {
  return buffer.scope() == "global" || buffer.scope().empty();
}

void ValidateDistSource(const Buffer &buffer, const char *op_name) {
  ICHECK(buffer.scope() == kSunmmioScopeRSRAM || IsDistDramBuffer(buffer))
      << op_name << " source must use shared.rsram or global/DRAM, got "
      << buffer.scope();
}

void ValidateDistDestination(const Buffer &buffer, const char *op_name) {
  ICHECK(buffer.scope() == kSunmmioScopeRSRAM || buffer.scope() == "global" ||
         buffer.scope().empty())
      << op_name << " destination must use shared.rsram or global/DRAM, got "
      << buffer.scope();
}

void ValidateDistTransfer(const Buffer &src, const Buffer &dst,
                          const char *op_name) {
  ValidateDistSource(src, op_name);
  ValidateDistDestination(dst, op_name);
}

} // namespace

DistPutOp::DistPutOp(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  (void)annotations;
  ICHECK_EQ(args.size(), 6U)
      << "T.dist.put expects src, dst, dst_rank, dst_row, signal, and "
         "current_core";
  ObjectPtr<DistPutOpNode> node = tvm::ffi::make_object<DistPutOpNode>();
  BufferRegion src_region = NormalizeToBufferRegion(args[0]);
  BufferRegion dst_region = NormalizeToBufferRegion(args[1]);
  node->src = src_region->buffer;
  node->dst = dst_region->buffer;
  node->src_range = src_region->region;
  node->dst_range = dst_region->region;
  node->dst_rank = args[2];
  node->dst_row = args[3];
  node->signal = args[4];
  node->current_core = args[5];
  data_ = std::move(node);
}

TileOperator DistPutOpNode::Clone() const {
  return DistPutOp(tvm::ffi::make_object<DistPutOpNode>(*this));
}

LayoutMap DistPutOpNode::InferLayout(const LayoutInferArgs &T,
                                     InferLevel level) const {
  ICHECK(TargetIsSunmmio(T.target))
      << "T.dist.put is currently supported only on the Sunmmio target";
  ValidateDistTransfer(src, dst, "T.dist.put");
  ICHECK(src->dtype == dst->dtype)
      << "T.dist.put source and destination dtypes must match";
  return InferMatchingLayouts(T, src, dst, level);
}

Stmt DistPutOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  (void)T;
  (void)analyzer;
  ICHECK(false) << "Logical T.dist.put must be processed by LowerDistRouting "
                   "before LowerTileOp";
  return Evaluate(0);
}

TIR_REGISTER_TL_TILE_OP(DistPutOp, dist_put)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

DistPeerPutOp::DistPeerPutOp(Array<PrimExpr> args,
                             Map<String, ObjectRef> annotations) {
  (void)annotations;
  ICHECK_EQ(args.size(), 5U)
      << "Physical T.dist peer put expects src, dst, dst_rank, signal, and "
         "current_core";
  ObjectPtr<DistPeerPutOpNode> node =
      tvm::ffi::make_object<DistPeerPutOpNode>();
  BufferRegion src_region = NormalizeToBufferRegion(args[0]);
  BufferRegion dst_region = NormalizeToBufferRegion(args[1]);
  node->src = src_region->buffer;
  node->dst = dst_region->buffer;
  node->src_range = src_region->region;
  node->dst_range = dst_region->region;
  node->dst_rank = args[2];
  node->signal = args[3];
  node->current_core = args[4];
  data_ = std::move(node);
}

TileOperator DistPeerPutOpNode::Clone() const {
  return DistPeerPutOp(tvm::ffi::make_object<DistPeerPutOpNode>(*this));
}

LayoutMap DistPeerPutOpNode::InferLayout(const LayoutInferArgs &T,
                                         InferLevel level) const {
  ICHECK(TargetIsSunmmio(T.target))
      << "Physical T.dist peer put only supports the Sunmmio target";
  ValidateDistTransfer(src, dst, "T.dist.peer_put");
  ICHECK(src->dtype == dst->dtype)
      << "T.dist.peer_put source and destination dtypes must match";
  return InferMatchingLayouts(T, src, dst, level);
}

Stmt DistPeerPutOpNode::Lower(const LowerArgs &T,
                              arith::Analyzer *analyzer) const {
  (void)T;
  (void)analyzer;
  ICHECK(false) << "Physical T.dist peer put must be processed by "
                   "LowerDistCommunication before LowerTileOp";
  return Evaluate(0);
}

TIR_REGISTER_TL_TILE_OP(DistPeerPutOp, dist_peer_put)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

DistRoutedPeerPutOp::DistRoutedPeerPutOp(Array<PrimExpr> args,
                                         Map<String, ObjectRef> annotations) {
  (void)annotations;
  ICHECK_GE(args.size(), 5U);
  ICHECK_EQ((args.size() - 3U) % 2U, 0U)
      << "T.dist.routed_peer_put expects routes, signal, current_core, then "
         "aligned src/dst region pairs";
  ObjectPtr<DistRoutedPeerPutOpNode> node =
      tvm::ffi::make_object<DistRoutedPeerPutOpNode>();
  node->routes = args[0];
  node->signal = args[1];
  node->current_core = args[2];
  for (size_t index = 3; index < args.size(); index += 2) {
    BufferRegion src_region = NormalizeToBufferRegion(args[index]);
    BufferRegion dst_region = NormalizeToBufferRegion(args[index + 1]);
    node->src.push_back(src_region->buffer);
    node->dst.push_back(dst_region->buffer);
    node->src_ranges.push_back(src_region->region);
    node->dst_ranges.push_back(dst_region->region);
  }
  data_ = std::move(node);
}

TileOperator DistRoutedPeerPutOpNode::Clone() const {
  return DistRoutedPeerPutOp(
      tvm::ffi::make_object<DistRoutedPeerPutOpNode>(*this));
}

LayoutMap DistRoutedPeerPutOpNode::InferLayout(const LayoutInferArgs &T,
                                               InferLevel level) const {
  ICHECK(TargetIsSunmmio(T.target));
  LayoutMap result;
  for (size_t index = 0; index < src.size(); ++index) {
    ValidateDistTransfer(src[index], dst[index], "T.dist.routed_peer_put");
    ICHECK(src[index]->dtype == dst[index]->dtype);
    LayoutMap inferred = InferMatchingLayouts(T, src[index], dst[index], level);
    for (const auto &entry : inferred) {
      result.Set(entry.first, entry.second);
    }
  }
  return result;
}

Stmt DistRoutedPeerPutOpNode::Lower(const LowerArgs &T,
                                    arith::Analyzer *analyzer) const {
  (void)T;
  (void)analyzer;
  ICHECK(false) << "T.dist.routed_peer_put must be processed by "
                   "LowerDistCommunication before LowerTileOp";
  return Evaluate(0);
}

TIR_REGISTER_TL_TILE_OP(DistRoutedPeerPutOp, dist_routed_peer_put)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

DistWaitSignalOp::DistWaitSignalOp(Array<PrimExpr> args,
                                   Map<String, ObjectRef> annotations) {
  (void)annotations;
  ICHECK_EQ(args.size(), 2U) << "T.dist.wait_signal expects signal and dst";
  ObjectPtr<DistWaitSignalOpNode> node =
      tvm::ffi::make_object<DistWaitSignalOpNode>();
  node->signal = args[0];
  BufferRegion dst_region = NormalizeToBufferRegion(args[1]);
  node->dst = dst_region->buffer;
  node->dst_range = dst_region->region;
  data_ = std::move(node);
}

TileOperator DistWaitSignalOpNode::Clone() const {
  return DistWaitSignalOp(tvm::ffi::make_object<DistWaitSignalOpNode>(*this));
}

LayoutMap DistWaitSignalOpNode::InferLayout(const LayoutInferArgs &T,
                                            InferLevel level) const {
  (void)level;
  ICHECK(TargetIsSunmmio(T.target))
      << "T.dist.wait_signal is currently supported only on the Sunmmio target";
  ValidateDistDestination(dst, "T.dist.wait_signal");
  return {};
}

Stmt DistWaitSignalOpNode::Lower(const LowerArgs &T,
                                 arith::Analyzer *analyzer) const {
  (void)analyzer;
  ICHECK(TargetIsSunmmio(T.target))
      << "T.dist.wait_signal is currently supported only on the Sunmmio target";
  ValidateDistDestination(dst, "T.dist.wait_signal");
  return Evaluate(
      Call(DataType::Handle(), dist_wait_signal_(),
           {signal, MakeRegionExpr(dst, dst_range, /*access_mask=*/2)}));
}

TIR_REGISTER_TL_TILE_OP(DistWaitSignalOp, dist_wait_signal)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  DistPutOpNode::RegisterReflection();
  DistPeerPutOpNode::RegisterReflection();
  DistRoutedPeerPutOpNode::RegisterReflection();
  DistWaitSignalOpNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
