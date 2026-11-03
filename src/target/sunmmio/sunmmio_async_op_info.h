/*!
 * \file sunmmio_async_op_info.h
 * \brief Shared parser for lowered asynchronous SunMMIO ODMA operations.
 *
 * Keep the argument-layout knowledge for dma/copy/transform/transpose/
 * broadcast in one place.  The channel is a planner hint only; current SUVM
 * lowering does not yet carry a concrete ODMA channel attribute.
 */
#ifndef TVM_TL_TARGET_SUNMMIO_ASYNC_OP_INFO_H_
#define TVM_TL_TARGET_SUNMMIO_ASYNC_OP_INFO_H_

#include "../../op/builtin.h"
#include "../../op/comm.h"
#include "../../op/utils.h"
#include "../sunmmio_utils.h"

#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace tl {

enum class SunmmioAsyncOpKind {
  kDmaCopy,
  kLayoutTransform,
  kTranspose,
  kBroadcast,
};

enum class SunmmioODMAPathKind {
  kLocal,
  kHLink,
  kVLink,
  kUnknown,
};

struct SunmmioAsyncOpInfo {
  SunmmioAsyncOpKind kind{SunmmioAsyncOpKind::kDmaCopy};
  tir::BufferRegion src;
  tir::BufferRegion dst;
  SunmmioODMAPathKind path_kind{SunmmioODMAPathKind::kUnknown};
  int odma_channel_hint{-1};
  bool channel_ambiguous{false};
  bool requires_transform_engine{false};
  PrimExpr direction;
  PrimExpr mask;
  PrimExpr src_core;
};

inline void PopulateSunmmioODMAChannelHint(SunmmioAsyncOpInfo *info) {
  const ffi::String &dst_scope = info->dst->buffer.scope();
  if (info->requires_transform_engine || dst_scope == "shared.asram") {
    info->odma_channel_hint = 1;
  } else if (dst_scope == "shared.wsram") {
    info->odma_channel_hint = 0;
  } else if (dst_scope == "shared.rsram" || dst_scope == "local") {
    // Both engines can serve RSRAM on A4E.  Keep the established conservative
    // ODMA0 mapping while surfacing that the choice is not architecturally
    // unique.
    info->odma_channel_hint = 0;
    info->channel_ambiguous = true;
  }
}

inline bool TryParseSunmmioAsyncODMAOp(const tir::CallNode *call,
                                       SunmmioAsyncOpInfo *info) {
  if (!call || !info) {
    return false;
  }

  if (call->op.same_as(dma_copy())) {
    if (call->args.size() < 2) {
      return false;
    }
    info->kind = SunmmioAsyncOpKind::kDmaCopy;
    info->src = NormalizeToBufferRegion(call->args[0]);
    info->dst = NormalizeToBufferRegion(call->args[1]);
    info->path_kind = SunmmioODMAPathKind::kLocal;
  } else if (call->op.same_as(sunmmio_layout_transform())) {
    if (call->args.size() < 2) {
      return false;
    }
    info->kind = SunmmioAsyncOpKind::kLayoutTransform;
    info->src = NormalizeToBufferRegion(call->args[0]);
    info->dst = NormalizeToBufferRegion(call->args[1]);
    info->path_kind = SunmmioODMAPathKind::kLocal;
    info->requires_transform_engine = true;
  } else if (call->op.same_as(sunmmio_transpose())) {
    if (call->args.size() < 2) {
      return false;
    }
    info->kind = SunmmioAsyncOpKind::kTranspose;
    info->src = NormalizeToBufferRegion(call->args[0]);
    info->dst = NormalizeToBufferRegion(call->args[1]);
    info->path_kind = SunmmioODMAPathKind::kLocal;
    info->requires_transform_engine = true;
  } else if (call->op.same_as(broadcast_())) {
    size_t semantic_args = call->args.size();
    if (semantic_args > 0 &&
        ParseSunmmioOdmaUnitExpr(call->args[semantic_args - 1])) {
      --semantic_args;
    }
    if (semantic_args != static_cast<size_t>(kBroadcastArgCount) &&
        semantic_args != static_cast<size_t>(kBroadcastArgCount + 1)) {
      return false;
    }
    info->kind = SunmmioAsyncOpKind::kBroadcast;
    info->src = NormalizeToBufferRegion(call->args[kBroadcastArgSrc]);
    info->dst = NormalizeToBufferRegion(call->args[kBroadcastArgDst]);
    info->direction = call->args[kBroadcastArgDirection];
    info->mask = call->args[kBroadcastArgMask];
    if (const auto *direction = info->direction.as<IntImmNode>()) {
      info->path_kind = direction->value == 0   ? SunmmioODMAPathKind::kHLink
                        : direction->value == 1 ? SunmmioODMAPathKind::kVLink
                                                : SunmmioODMAPathKind::kUnknown;
    }
    if (semantic_args == static_cast<size_t>(kBroadcastArgCount + 1)) {
      info->src_core = call->args[kBroadcastArgSrcCore];
    }
  } else {
    return false;
  }

  PopulateSunmmioODMAChannelHint(info);
  return true;
}

inline bool TryParseSunmmioAsyncODMAStmt(const tir::Stmt &stmt,
                                         SunmmioAsyncOpInfo *info) {
  const auto *eval = stmt.as<tir::EvaluateNode>();
  return eval &&
         TryParseSunmmioAsyncODMAOp(eval->value.as<tir::CallNode>(), info);
}

inline bool IsSunmmioAsyncODMAOp(const tir::CallNode *call) {
  SunmmioAsyncOpInfo info;
  return TryParseSunmmioAsyncODMAOp(call, &info);
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TARGET_SUNMMIO_ASYNC_OP_INFO_H_
