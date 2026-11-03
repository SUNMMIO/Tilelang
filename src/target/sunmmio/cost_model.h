/*!
 * \file cost_model.h
 * \brief CostModel to estimate the execution delay of pipeline instructions.
 */
#ifndef TVM_TL_TRANSFORM_SUNMMIO_PIPELINE_PLANNING_COST_MODEL_H_
#define TVM_TL_TRANSFORM_SUNMMIO_PIPELINE_PLANNING_COST_MODEL_H_

#include "../../layout/layout.h"
#include "hardware_types.h"

#include <cstdint>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/string.h>
#include <tvm/target/target.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace tl {

enum class CostModelProfile {
  kHeuristic,
  kGem5A4E2026,
};

struct CostModelOptions {
  CostModelProfile profile{CostModelProfile::kHeuristic};
  bool fallback_to_heuristic{true};
  bool preserve_absolute_cycles{false};
};

struct CostModelContext {
  CostModelOptions options;
  ffi::Optional<Target> target;
  ffi::Map<tir::Buffer, Layout> layout_map;
  ffi::Map<tir::Buffer, Layout> global_layout_map;
  ffi::Optional<Integer> explicit_solver_scale;
};

/**
 * \brief Structured cost result used by planners and diagnostics.
 *
 * exact_static_submodel means the selected static source formula was
 * reproduced.  It never implies that dynamic queue/endpoint/retry timing was
 * modeled.
 */
struct CostEstimate {
  int64_t latency_cycles{0};
  int64_t issue_start_cycles{0};
  int64_t issue_finish_cycles{0};
  bool exact_static_submodel{false};
  bool used_fallback{false};
  ffi::String profile{"heuristic"};
  ffi::String model_version{"heuristic_v1"};
  ffi::String latency_source{"heuristic"};
  ffi::String fallback_reason{""};
  ffi::String op_kind{""};
  ffi::String transfer_path{""};
  ffi::String src_scope{""};
  ffi::String dst_scope{""};
  ffi::String input_dtype{""};
  ffi::String output_dtype{""};
  int64_t bytes{0};
  int64_t packet_count{0};
  int64_t path_start_delay{0};
  int64_t row_size{0};
  int64_t col_size{0};
  int64_t reduction_size{0};
  int64_t vector_chunks{0};
  int64_t load_ops{0};
  int64_t store_ops{0};
  int64_t arithmetic_ops{0};
  int64_t compare_convert_ops{0};
  int64_t nonlinear_ops{0};
  int64_t reduce_ops{0};
  int64_t galu_ops{0};
  int64_t solver_scale_factor{1};
  int64_t solver_latency_cycles{0};
  bool solver_scaled{false};
  bool endpoint_timing_modeled{false};
  bool resource_overlap_modeled{false};
};

class CostModel {
public:
  // Compatibility entry point.  It must stay heuristic unless a context is
  // explicitly supplied by a planner.
  // heristic兼容接口
  static float EstimateDelay(DeviceType device_type, const tir::Stmt &stmt);
  // 估算cycle数，但是没有target,layout等上下文，cost model通过options指定
  static float EstimateDelay(DeviceType device_type, const tir::Stmt &stmt,
                             const CostModelOptions &options);
  // 估算cost，但是返回cycle数
  static float EstimateDelay(DeviceType device_type, const tir::Stmt &stmt,
                             const CostModelContext &context);
  // 根据profile和设备类型调用不同硬件的cost
  // model，返回包含cost信息的CostEstimate结构
  static CostEstimate Estimate(DeviceType device_type, const tir::Stmt &stmt,
                               const CostModelContext &context);
  static CostModelContext BuildContext(const tir::PrimFunc &func);
  static const char *ProfileName(CostModelProfile profile);

private:
  // tensorcore cost: 使用旧经验估计
  static float EstimateTensorCoreDelayHeuristic(const tir::Stmt &stmt);
  // ODMA： 使用旧经验估计
  static float EstimateODMADelayHeuristic(const tir::Stmt &stmt);
  // Vector Core: 使用旧经验估计
  static float EstimateVectorCoreDelayHeuristic(const tir::Stmt &stmt);
  // 旧经验下的cost 估算入口
  static float EstimateDelayHeuristic(DeviceType device_type,
                                      const tir::Stmt &stmt);
  // Tensor Core: gem5版本的
  static CostEstimate
  EstimateTensorCoreDelayGem5A4E(const tir::Stmt &stmt,
                                 const CostModelContext &context);
  // ODMA: gem5版本
  static CostEstimate EstimateODMADelayGem5A4E(const tir::Stmt &stmt,
                                               const CostModelContext &context);
  // Vector Core: gem5版本
  static CostEstimate
  EstimateVectorCoreDelayGem5A4E(const tir::Stmt &stmt,
                                 const CostModelContext &context);
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_SUNMMIO_PIPELINE_PLANNING_COST_MODEL_H_
