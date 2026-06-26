/*!
 * \file hardware_types.h
 * \brief Hardware specific enums and mappers for pipeline planning.
 */
#ifndef TVM_TL_TRANSFORM_SUNMMIO_PIPELINE_PLANNING_HARDWARE_TYPES_H_
#define TVM_TL_TRANSFORM_SUNMMIO_PIPELINE_PLANNING_HARDWARE_TYPES_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace tl {

using namespace tir;

/**
 * \brief The hardware execution unit.
 */
enum class DeviceType : int {
  ODMA,
  TensorCore,
  VectorCore,
  Unspecified,
};

/**
 * \brief HardwareMapper maps an AST statement to a specific DeviceType.
 */
class HardwareMapper {
public:
  static DeviceType Map(const Stmt &stmt) {
    if (const auto *block = stmt.as<BlockRealizeNode>()) {
      // Check for TensorCore (mma_sunmmio)
      auto body = block->block->body;
      if (const auto *eval = body.as<EvaluateNode>()) {
        if (const auto *call = eval->value.as<CallNode>()) {
          if (call->op.same_as(Op::Get("tl.mma_sunmmio"))) {
            return DeviceType::TensorCore;
          }
        }
      } else if (block->block->name_hint == "reduce_tile_op") {
        return DeviceType::VectorCore;
      }
      return DeviceType::VectorCore; // Default to VectorCore if not recognized
    } else if (const auto *eval = stmt.as<EvaluateNode>()) {
      // Check for ODMA (dma_copy, broadcast_, etc.)
      if (const auto *call = eval->value.as<CallNode>()) {
        if (call->op.same_as(Op::Get("tl.dma_copy")) ||
            call->op.same_as(Op::Get("tl.broadcast_")) ||
            call->op.same_as(Op::Get("tl.sunmmio_layout_transform"))) {
          return DeviceType::ODMA;
        }
      }
      return DeviceType::VectorCore;
    }
    // Set VectorCore for all other general computation loops/statements
    return DeviceType::VectorCore;
  }
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_SUNMMIO_PIPELINE_PLANNING_HARDWARE_TYPES_H_
