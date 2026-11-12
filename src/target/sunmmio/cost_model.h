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

/**
 * \brief Semantic outcome of a cost-model query.
 *
 * Do not infer support from latency_cycles.  A positive latency may come from
 * a fallback, while an unsupported query intentionally carries no latency.
 */
enum class CostEstimateStatus {
  kUnsupported,
  kExactStatic,
  kApproximate,
  kFallback,
};

struct CostModelOptions {
  CostModelProfile profile{CostModelProfile::kGem5A4E2026};
  bool fallback_to_heuristic{true};
  // Calibration mode accepts only an exact static Gem5 submodel and forbids
  // heuristic fallback. Planners must separately reject any non-unit solver
  // scale while this mode is active.
  bool strict_calibration{false};
};

struct CostModelContext {
  CostModelOptions options;
  ffi::Optional<Target> target;
  ffi::Map<tir::Buffer, Layout> layout_map;
  ffi::Map<tir::Buffer, Layout> global_layout_map;
};

/**
 * \brief Structured cost result used by planners and diagnostics.
 *
 * status is the canonical outcome.  exact_static_submodel and used_fallback
 * are compatibility projections derived from it before the result is returned.
 * Exact-static never implies that dynamic queue/endpoint/retry timing was
 * modeled.
 */
struct CostEstimate {
  CostEstimateStatus status{CostEstimateStatus::kUnsupported};
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
  ffi::String weight_dtype{""};
  ffi::String output_dtype{""};
  ffi::String transform_kind{""};
  ffi::String timing_origin{"unspecified"};
  ffi::String timing_end{"completion"};
  int64_t bytes{0};
  int64_t packet_count{0};
  int64_t descriptor_count{0};
  int64_t descriptor_fetch_cycles{0};
  int64_t command_frontend_cycles{0};
  int64_t path_start_delay{0};
  int64_t endpoint_completion_cycles{0};
  int64_t link_completion_cycles{0};
  int64_t issue_interval_cycles{0};
  int64_t odma_channel{-1};
  int64_t odma_channel_mask{0};
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
  // Planner-owned compatibility diagnostics.  CostModel initializes these to
  // the raw hardware latency at scale 1; a planner may overwrite them after
  // applying its own solver-unit policy.
  int64_t solver_scale_factor{1};
  int64_t solver_latency_cycles{0};
  bool solver_scaled{false};
  bool endpoint_timing_modeled{false};
  bool resource_overlap_modeled{false};
  bool command_frontend_modeled{false};
  bool descriptor_plan_modeled{false};
  bool link_completion_modeled{false};
};

class CostModel {
public:
  // Compatibility entry point.  It stays heuristic unless a profile-bearing
  // context or options object is explicitly supplied.
  static float EstimateDelay(DeviceType device_type, const tir::Stmt &stmt);
  // Profile-aware query without target or layout context.
  static float EstimateDelay(DeviceType device_type, const tir::Stmt &stmt,
                             const CostModelOptions &options);
  // Profile-aware latency-only query with full context.
  static float EstimateDelay(DeviceType device_type, const tir::Stmt &stmt,
                             const CostModelContext &context);
  // Canonical structured query used by planners and diagnostics.
  static CostEstimate Estimate(DeviceType device_type, const tir::Stmt &stmt,
                               const CostModelContext &context);
  static CostModelContext BuildContext(const tir::PrimFunc &func);
  static const char *ProfileName(CostModelProfile profile);
  static const char *ModelVersion(CostModelProfile profile);
  static const char *StatusName(CostEstimateStatus status);

private:
  // Legacy heuristic submodels.
  static float EstimateTensorCoreDelayHeuristic(const tir::Stmt &stmt);
  static float EstimateODMADelayHeuristic(const tir::Stmt &stmt);
  static float EstimateVectorCoreDelayHeuristic(const tir::Stmt &stmt);
  static float EstimateDelayHeuristic(DeviceType device_type,
                                      const tir::Stmt &stmt);
  // Gem5 A4E static submodels.
  static CostEstimate
  EstimateTensorCoreDelayGem5A4E(const tir::Stmt &stmt,
                                 const CostModelContext &context);
  static CostEstimate EstimateODMADelayGem5A4E(const tir::Stmt &stmt,
                                               const CostModelContext &context);
  static CostEstimate
  EstimateVectorCoreDelayGem5A4E(const tir::Stmt &stmt,
                                 const CostModelContext &context);
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_SUNMMIO_PIPELINE_PLANNING_COST_MODEL_H_
