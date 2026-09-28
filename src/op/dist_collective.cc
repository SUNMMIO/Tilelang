/*!
 * \file tl/op/dist_collective.cc
 * \brief High-level Rank collective TIR operations.
 */

#include "dist_collective.h"

#include <tvm/tir/op_attr_types.h>

#include "../target/utils.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tir;

DistAllgatherOp::DistAllgatherOp(Array<PrimExpr> args,
                                 Map<String, ObjectRef> annotations) {
  (void)annotations;
  ICHECK_EQ(args.size(), 5U)
      << "T.dist.all_gather expects src, dst, axis, signal, and current_core";
  BufferRegion src_region = NormalizeToBufferRegion(args[0]);
  BufferRegion dst_region = NormalizeToBufferRegion(args[1]);
  const auto *axis = args[2].as<IntImmNode>();
  ICHECK(axis) << "T.dist.all_gather axis must be a compile-time integer";

  ObjectPtr<DistAllgatherOpNode> node =
      tvm::ffi::make_object<DistAllgatherOpNode>();
  node->src = src_region->buffer;
  node->dst = dst_region->buffer;
  node->src_range = src_region->region;
  node->dst_range = dst_region->region;
  node->axis = axis->value;
  node->signal = args[3];
  node->current_core = args[4];
  data_ = std::move(node);
}

TileOperator DistAllgatherOpNode::Clone() const {
  return DistAllgatherOp(tvm::ffi::make_object<DistAllgatherOpNode>(*this));
}

LayoutMap DistAllgatherOpNode::InferLayout(const LayoutInferArgs &T,
                                           InferLevel level) const {
  (void)level;
  ICHECK(TargetIsSunmmio(T.target))
      << "T.dist.all_gather is currently supported only on the Sunmmio target";
  return {};
}

Stmt DistAllgatherOpNode::Lower(const LowerArgs &T,
                                arith::Analyzer *analyzer) const {
  (void)T;
  (void)analyzer;
  ICHECK(false) << "T.dist.all_gather must be processed by "
                   "LowerDistCollectives before LowerTileOp";
  return Evaluate(0);
}

TIR_REGISTER_TL_TILE_OP(DistAllgatherOp, dist_allgather)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

DistAlltoallOp::DistAlltoallOp(Array<PrimExpr> args,
                               Map<String, ObjectRef> annotations) {
  (void)annotations;
  ICHECK_EQ(args.size(), 5U)
      << "T.dist.all_to_all expects src, dst, domain, signal, and current_core";
  BufferRegion src_region = NormalizeToBufferRegion(args[0]);
  BufferRegion dst_region = NormalizeToBufferRegion(args[1]);
  const auto *domain = args[2].as<StringImmNode>();
  ICHECK(domain) << "T.dist.all_to_all domain must be a compile-time string";

  ObjectPtr<DistAlltoallOpNode> node =
      tvm::ffi::make_object<DistAlltoallOpNode>();
  node->src = src_region->buffer;
  node->dst = dst_region->buffer;
  node->src_range = src_region->region;
  node->dst_range = dst_region->region;
  node->domain = domain->value;
  node->signal = args[3];
  node->current_core = args[4];
  data_ = std::move(node);
}

TileOperator DistAlltoallOpNode::Clone() const {
  return DistAlltoallOp(tvm::ffi::make_object<DistAlltoallOpNode>(*this));
}

LayoutMap DistAlltoallOpNode::InferLayout(const LayoutInferArgs &T,
                                          InferLevel level) const {
  (void)level;
  ICHECK(TargetIsSunmmio(T.target))
      << "T.dist.all_to_all is currently supported only on the Sunmmio target";
  return {};
}

Stmt DistAlltoallOpNode::Lower(const LowerArgs &T,
                               arith::Analyzer *analyzer) const {
  (void)T;
  (void)analyzer;
  ICHECK(false) << "T.dist.all_to_all must be processed by "
                   "LowerDistCollectives before LowerTileOp";
  return Evaluate(0);
}

TIR_REGISTER_TL_TILE_OP(DistAlltoallOp, dist_alltoall)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

DistAllreduceOp::DistAllreduceOp(Array<PrimExpr> args,
                                 Map<String, ObjectRef> annotations) {
  (void)annotations;
  ICHECK_EQ(args.size(), 5U)
      << "T.dist.all_reduce expects src, dst, reduce_type, internal signal, "
         "and current_core";
  BufferRegion src_region = NormalizeToBufferRegion(args[0]);
  BufferRegion dst_region = NormalizeToBufferRegion(args[1]);
  const auto *reduce_type = args[2].as<StringImmNode>();
  ICHECK(reduce_type)
      << "T.dist.all_reduce reduce_type must be a compile-time string";

  ObjectPtr<DistAllreduceOpNode> node =
      tvm::ffi::make_object<DistAllreduceOpNode>();
  node->src = src_region->buffer;
  node->dst = dst_region->buffer;
  node->src_range = src_region->region;
  node->dst_range = dst_region->region;
  node->reduce_type = reduce_type->value;
  node->signal = args[3];
  node->current_core = args[4];
  data_ = std::move(node);
}

TileOperator DistAllreduceOpNode::Clone() const {
  return DistAllreduceOp(tvm::ffi::make_object<DistAllreduceOpNode>(*this));
}

LayoutMap DistAllreduceOpNode::InferLayout(const LayoutInferArgs &T,
                                           InferLevel level) const {
  (void)level;
  ICHECK(TargetIsSunmmio(T.target))
      << "T.dist.all_reduce is currently supported only on the Sunmmio "
         "target";
  return {};
}

Stmt DistAllreduceOpNode::Lower(const LowerArgs &T,
                                arith::Analyzer *analyzer) const {
  (void)T;
  (void)analyzer;
  ICHECK(false) << "T.dist.all_reduce must be processed by "
                   "LowerDistCollectives before LowerTileOp";
  return Evaluate(0);
}

TIR_REGISTER_TL_TILE_OP(DistAllreduceOp, dist_allreduce)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

DistAlltoallvOp::DistAlltoallvOp(Array<PrimExpr> args,
                                 Map<String, ObjectRef> annotations) {
  (void)annotations;
  ICHECK_EQ(args.size(), 7U)
      << "T.dist.all_to_allv expects src, dst, send_counts, recv_counts, "
         "domain, signal resource, and current_core";
  BufferRegion src_region = NormalizeToBufferRegion(args[0]);
  BufferRegion dst_region = NormalizeToBufferRegion(args[1]);
  BufferRegion send_counts_region = NormalizeToBufferRegion(args[2]);
  BufferRegion recv_counts_region = NormalizeToBufferRegion(args[3]);
  const auto *domain = args[4].as<StringImmNode>();
  ICHECK(domain) << "T.dist.all_to_allv domain must be a compile-time string";

  ObjectPtr<DistAlltoallvOpNode> node =
      tvm::ffi::make_object<DistAlltoallvOpNode>();
  node->src = src_region->buffer;
  node->dst = dst_region->buffer;
  node->send_counts = send_counts_region->buffer;
  node->recv_counts = recv_counts_region->buffer;
  node->src_range = src_region->region;
  node->dst_range = dst_region->region;
  node->send_counts_range = send_counts_region->region;
  node->recv_counts_range = recv_counts_region->region;
  node->domain = domain->value;
  node->signal_resource = args[5];
  node->current_core = args[6];
  data_ = std::move(node);
}

TileOperator DistAlltoallvOpNode::Clone() const {
  return DistAlltoallvOp(tvm::ffi::make_object<DistAlltoallvOpNode>(*this));
}

LayoutMap DistAlltoallvOpNode::InferLayout(const LayoutInferArgs &T,
                                           InferLevel level) const {
  (void)level;
  ICHECK(TargetIsSunmmio(T.target))
      << "T.dist.all_to_allv is currently supported only on the Sunmmio "
         "target";
  return {};
}

Stmt DistAlltoallvOpNode::Lower(const LowerArgs &T,
                                arith::Analyzer *analyzer) const {
  (void)T;
  (void)analyzer;
  ICHECK(false) << "T.dist.all_to_allv must be processed by "
                   "LowerDistCollectives before LowerTileOp";
  return Evaluate(0);
}

TIR_REGISTER_TL_TILE_OP(DistAlltoallvOp, dist_alltoallv)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  DistAllgatherOpNode::RegisterReflection();
  DistAlltoallOpNode::RegisterReflection();
  DistAllreduceOpNode::RegisterReflection();
  DistAlltoallvOpNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
