/*!
 * \file hardware_types.h
 * \brief Hardware specific enums and mappers for pipeline planning.
 */
#ifndef TVM_TL_TRANSFORM_SUNMMIO_PIPELINE_PLANNING_HARDWARE_TYPES_H_
#define TVM_TL_TRANSFORM_SUNMMIO_PIPELINE_PLANNING_HARDWARE_TYPES_H_

#include "sunmmio_async_op_info.h"
#include "tvm/runtime/logging.h"
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

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
  static bool HasNestedHardwareCommands(const Stmt &stmt) {
    class NestedHardwareCommandFinder : public StmtVisitor {
    public:
      bool found{false};

    private:
      int loop_depth_{0};

      void VisitStmt_(const ForNode *op) final {
        ++loop_depth_;
        StmtVisitor::VisitStmt_(op);
        --loop_depth_;
      }

      void VisitStmt_(const EvaluateNode *op) final {
        if (loop_depth_ == 0)
          return;
        const auto *call = op->value.as<CallNode>();
        if (!call ||
            !call->op.same_as(Op::Get("tl.vector_core_in_tile_reduce"))) {
          found = true;
        }
      }
    };

    // A loop containing a hardware intrinsic cannot be scheduled as one
    // VectorCore command. Walk through Block/BlockRealize wrappers because
    // lowering may preserve a named T.block around that loop.
    NestedHardwareCommandFinder finder;
    finder(stmt);
    return finder.found;
  }

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
      if (const auto *call = eval->value.as<CallNode>()) {
        if (IsSunmmioAsyncODMAOp(call)) {
          return DeviceType::ODMA;
        }
      }
      return DeviceType::VectorCore;
    } else if (const auto *for_stmt = stmt.as<ForNode>()) {
      // Set VectorCore for tilesloop
      return DeviceType::VectorCore;
    }
    LOG(FATAL) << "Can not identify the hardware type for a "
               << stmt->GetTypeKey() << " statement.";
  }
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_SUNMMIO_PIPELINE_PLANNING_HARDWARE_TYPES_H_
