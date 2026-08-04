/*!
 * \file cost_model.h
 * \brief CostModel to estimate the execution delay of pipeline instructions.
 */
#ifndef TVM_TL_TRANSFORM_SUNMMIO_PIPELINE_PLANNING_COST_MODEL_H_
#define TVM_TL_TRANSFORM_SUNMMIO_PIPELINE_PLANNING_COST_MODEL_H_

#include "hardware_types.h"

#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace tl {

// A minimal interface for CostModel
class CostModel {
public:
  static float EstimateDelay(DeviceType device_type, const tir::Stmt &stmt);

private:
  static float EstimateTensorCoreDelay(const tir::Stmt &stmt);
  static float EstimateODMADelay(const tir::Stmt &stmt);
  static float EstimateVectorCoreDelay(const tir::Stmt &stmt);
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_SUNMMIO_PIPELINE_PLANNING_COST_MODEL_H_
