/*!
 * \file sunmmio_async_op_info.h
 * \brief Shared parser for lowered asynchronous SunMMIO ODMA operations.
 *
 * Keep the argument-layout knowledge for dma/copy/transform/transpose/
 * broadcast in one place. Prefer the concrete ODMA unit selected by
 * ResolveSunmmioUnit. Scope-derived hints support pre-resolution estimates.
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
  int odma_channel_mask{0};
  bool channel_ambiguous{false};
  bool requires_transform_engine{false};
  PrimExpr direction;
  PrimExpr mask;
  PrimExpr src_core;
};

inline void PopulateSunmmioODMAChannelHint(SunmmioAsyncOpInfo *info) {
  // Intersect the read path, write path, and transform capability masks from
  // NPU-IR DeviceArchitecture.td. bit 0 = ODMA0, bit 1 = ODMA1.
  int read_mask = 0;
  if (IsGlobalBuffer(info->src->buffer)) {
    read_mask = 0b01;
  } else {
    const ffi::String &src_scope = info->src->buffer.scope();
    if (src_scope == "shared.rsram" || src_scope == "local") {
      read_mask = 0b11;
    }
  }

  int write_mask = 0;
  if (info->path_kind == SunmmioODMAPathKind::kHLink) {
    const ffi::String &dst_scope = info->dst->buffer.scope();
    if (dst_scope == "shared.asram" || dst_scope == "shared.rsram" ||
        dst_scope == "local") {
      write_mask = 0b10;
    }
  } else if (info->path_kind == SunmmioODMAPathKind::kVLink) {
    const ffi::String &dst_scope = info->dst->buffer.scope();
    if (dst_scope == "shared.wsram" || dst_scope == "shared.rsram" ||
        dst_scope == "local") {
      write_mask = 0b01;
    }
  } else {
    const ffi::String &dst_scope = info->dst->buffer.scope();
    if (IsGlobalBuffer(info->dst->buffer)) {
      write_mask = 0b01;
    } else if (info->requires_transform_engine || dst_scope == "shared.asram") {
      write_mask = 0b10;
    } else if (dst_scope == "shared.wsram") {
      write_mask = 0b01;
    } else if (dst_scope == "shared.rsram" || dst_scope == "local") {
      write_mask = 0b11;
    }
  }

  info->odma_channel_mask = read_mask & write_mask;
  if (info->odma_channel_mask == 0b01) {
    info->odma_channel_hint = 0;
  } else if (info->odma_channel_mask == 0b10) {
    info->odma_channel_hint = 1;
  } else if (info->odma_channel_mask == 0b11) {
    // The current ILP representation needs one deterministic resource. Keep
    // ODMA0 as the compatibility hint, but expose the legal alternative.
    info->odma_channel_hint = 0;
    info->channel_ambiguous = true;
  } else {
    info->odma_channel_hint = -1;
  }
}

inline bool IsSunmmioOdmaUnitExpr(const PrimExpr &expr) {
  return ParseSunmmioOdmaUnitExpr(expr).has_value();
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
    size_t non_metadata_args = call->args.size();
    if (non_metadata_args > 0 &&
        IsSunmmioOdmaUnitExpr(call->args[non_metadata_args - 1])) {
      --non_metadata_args;
    }
    if (non_metadata_args != static_cast<size_t>(kBroadcastArgCount) &&
        non_metadata_args != static_cast<size_t>(kBroadcastArgCount + 1)) {
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
    if (non_metadata_args == static_cast<size_t>(kBroadcastArgCount + 1)) {
      info->src_core = call->args[kBroadcastArgSrcCore];
    }
  } else {
    return false;
  }

  PopulateSunmmioODMAChannelHint(info);
  if (auto unit = GetSunmmioOdmaUnit(call)) {
    int channel = *unit == SunmmioOdmaUnit::kOdma0 ? 0 : 1;
    ICHECK(info->odma_channel_mask & (1 << channel))
        << "Resolved ODMA unit does not support this A4E transfer path";
    // Keep the legal-route mask for diagnostics, but the resolved unit owns
    // the actual resource assignment even when both units support the route.
    info->odma_channel_hint = channel;
    info->channel_ambiguous = false;
  }
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
