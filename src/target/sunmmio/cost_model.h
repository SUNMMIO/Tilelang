/*!
 * \file cost_model.h
 * \brief Gem5-aligned execution-delay model for Sunmmio pipeline commands.
 */
#ifndef TVM_TL_TRANSFORM_SUNMMIO_PIPELINE_PLANNING_COST_MODEL_H_
#define TVM_TL_TRANSFORM_SUNMMIO_PIPELINE_PLANNING_COST_MODEL_H_

#include "hardware_types.h"

#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace tl {

// The public interface remains device-agnostic; device-specific gem5 timing
// parameters and TIR decoding are kept in the implementation.
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
