/*!
 * \file resource_types_for_ilp.h
 * \brief ILP-specific resource typing and resource extraction helpers.
 */
#ifndef TVM_TL_TRANSFORM_SUNMMIO_PIPELINE_PLANNING_RESOURCE_TYPES_FOR_ILP_H_
#define TVM_TL_TRANSFORM_SUNMMIO_PIPELINE_PLANNING_RESOURCE_TYPES_FOR_ILP_H_

#include "../../op/utils.h"
#include "../../target/sunmmio/hardware_types.h"
#include "../../target/sunmmio/sunmmio_async_op_info.h"
#include "../../target/sunmmio_utils.h"

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
  kVLink = 4,
  kHLink = 5,
  kWsramIn = 6,
  kWsramOut = 7,
  kAsramIn = 8,
  kAsramOut = 9,
  // kRsram = 10,
};

template <typename AccessInfoLike>
std::vector<int>
BuildIlpResources(const Stmt &stmt, DeviceType type,
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
      SunmmioAsyncOpInfo info;
      if (TryParseSunmmioAsyncODMAOp(call, &info)) {
        std::optional<SunmmioOdmaUnit> unit = GetSunmmioOdmaUnit(call);
        ICHECK(unit.has_value())
            << "Sunmmio pipeline planning requires resolved ODMA units";
        add_resource(static_cast<int>(*unit == SunmmioOdmaUnit::kOdma0
                                          ? IlpResourceType::kODMA0
                                          : IlpResourceType::kODMA1));
        if (info.path_kind == SunmmioODMAPathKind::kVLink) {
          add_resource(static_cast<int>(IlpResourceType::kVLink));
        } else if (info.path_kind == SunmmioODMAPathKind::kHLink) {
          add_resource(static_cast<int>(IlpResourceType::kHLink));
        }
        if (info.dst->buffer.scope() == "shared.wsram") {
          add_resource(static_cast<int>(IlpResourceType::kWsramIn));
        }
        if (info.dst->buffer.scope() == "shared.asram") {
          add_resource(static_cast<int>(IlpResourceType::kAsramIn));
        }
        if (info.src->buffer.scope() == "shared.wsram") {
          add_resource(static_cast<int>(IlpResourceType::kWsramOut));
        }
        if (info.src->buffer.scope() == "shared.asram") {
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
