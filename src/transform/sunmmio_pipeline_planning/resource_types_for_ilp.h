/*!
 * \file resource_types_for_ilp.h
 * \brief ILP-specific resource typing and resource extraction helpers.
 */
#ifndef TVM_TL_TRANSFORM_SUNMMIO_PIPELINE_PLANNING_RESOURCE_TYPES_FOR_ILP_H_
#define TVM_TL_TRANSFORM_SUNMMIO_PIPELINE_PLANNING_RESOURCE_TYPES_FOR_ILP_H_

#include "../../op/utils.h"
#include "../../target/sunmmio/hardware_types.h"

#include <algorithm>
#include <tvm/runtime/logging.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace tl {

using namespace tir;

enum class IlpResourceType : int {
  kTensorCore = 0,
  kVectorCore = 1,
  kODMA0 = 2,
  kODMA1 = 3,
  kWsramIn = 6,
  kWsramOut = 7,
  kAsramIn = 8,
  kAsramOut = 9,
  // kRsram = 10,
};

template <typename AccessInfoLike>
std::vector<int> BuildIlpResources(const Stmt &stmt, DeviceType type,
                                   const std::vector<AccessInfoLike> &accesses) {
  std::vector<int> resources;
  auto add_resource = [&](int resource) {
    if (std::find(resources.begin(), resources.end(), resource) ==
        resources.end()) {
      resources.push_back(resource);
    }
  };

  auto has_scope_read = [&](const char *scope) {
    for (const AccessInfoLike &access : accesses) {
      if (!access.is_write && access.buffer().scope() == scope) {
        return true;
      }
    }
    return false;
  };

  if (type == DeviceType::TensorCore) {
    add_resource(static_cast<int>(IlpResourceType::kTensorCore));
    if (has_scope_read("shared.wsram")) {
      add_resource(static_cast<int>(IlpResourceType::kWsramOut));
    }
    if (has_scope_read("shared.asram")) {
      add_resource(static_cast<int>(IlpResourceType::kAsramOut));
    }
    return resources;
  }

  if (type == DeviceType::VectorCore) {
    add_resource(static_cast<int>(IlpResourceType::kVectorCore));
    return resources;
  }

  if (const auto *eval = stmt.as<EvaluateNode>()) {
    if (const auto *call = eval->value.as<CallNode>()) {
      if (call->op.same_as(Op::Get("tl.dma_copy")) ||
          call->op.same_as(Op::Get("tl.broadcast_")) ||
          call->op.same_as(Op::Get("tl.sunmmio_layout_transform"))) {
        BufferRegion src_region = NormalizeToBufferRegion(call->args[0]);
        BufferRegion dst_region = call->op.same_as(Op::Get("tl.broadcast_"))
                                      ? NormalizeToBufferRegion(call->args[2])
                                      : NormalizeToBufferRegion(call->args[1]);
        if (IsGlobalBuffer(src_region->buffer)) {
          if (dst_region->buffer.scope() == "shared.asram") {
            LOG(FATAL)
                << "ILP graph does not model DRAM -> ASRAM dma path yet.";
          }
          if (dst_region->buffer.scope() == "shared.wsram") {
            add_resource(static_cast<int>(IlpResourceType::kODMA0));
            add_resource(static_cast<int>(IlpResourceType::kWsramIn));
            return resources;
          }
          if (dst_region->buffer.scope() == "shared.rsram" ||
              dst_region->buffer.scope() == "local") {
            add_resource(static_cast<int>(IlpResourceType::kODMA0));
            // add_resource(static_cast<int>(IlpResourceType::kRsram));
            return resources;
          }
        }
        if ((src_region->buffer.scope() == "shared.rsram" ||
             src_region->buffer.scope() == "local") &&
            dst_region->buffer.scope() == "shared.asram") {
          add_resource(static_cast<int>(IlpResourceType::kODMA1));
          add_resource(static_cast<int>(IlpResourceType::kAsramIn));
          // add_resource(static_cast<int>(IlpResourceType::kRsram));
          return resources;
        }
        add_resource(static_cast<int>(IlpResourceType::kODMA0));
        if (src_region->buffer.scope() == "shared.rsram" ||
            src_region->buffer.scope() == "local" ||
            dst_region->buffer.scope() == "shared.rsram" ||
            dst_region->buffer.scope() == "local") {
          // add_resource(static_cast<int>(IlpResourceType::kRsram));
        }
        if (dst_region->buffer.scope() == "shared.wsram") {
          add_resource(static_cast<int>(IlpResourceType::kWsramIn));
        }
        if (dst_region->buffer.scope() == "shared.asram") {
          add_resource(static_cast<int>(IlpResourceType::kAsramIn));
        }
        if (src_region->buffer.scope() == "shared.wsram") {
          add_resource(static_cast<int>(IlpResourceType::kWsramOut));
        }
        if (src_region->buffer.scope() == "shared.asram") {
          add_resource(static_cast<int>(IlpResourceType::kAsramOut));
        }
        return resources;
      }
    }
  }
  return resources;
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_SUNMMIO_PIPELINE_PLANNING_RESOURCE_TYPES_FOR_ILP_H_
