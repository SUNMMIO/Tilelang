#ifndef TILELANG_TRANSFORM_SUNMMIO_PIPELINE_DIAGNOSTIC_H_
#define TILELANG_TRANSFORM_SUNMMIO_PIPELINE_DIAGNOSTIC_H_

#include <string>
#include <unordered_set>
#include <utility>

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tl {

using namespace tir;

constexpr const char *kPipelineRequested = "tl.sunmmio.pipeline.requested";
constexpr const char *kPipelineApplied = "tl.sunmmio.pipeline.applied";
constexpr const char *kPipelineMode = "tl.sunmmio.pipeline.mode";
constexpr const char *kPipelineFallbackStage =
    "tl.sunmmio.pipeline.fallback_stage";
constexpr const char *kPipelineFallbackReason =
    "tl.sunmmio.pipeline.fallback_reason";
constexpr const char *kPipelineFallbackDetail =
    "tl.sunmmio.pipeline.fallback_detail";

struct PipelineDiagnostic {
  bool applied{false};
  std::string mode;
  std::string stage;
  std::string reason;
  std::string detail;
};

inline void SetPipelineAppliedAnnotations(Map<String, Any> *annotations,
                                          const std::string &mode) {
  annotations->Set(kPipelineRequested, Bool(true));
  annotations->Set(kPipelineApplied, Bool(true));
  annotations->Set(kPipelineMode, String(mode));
  annotations->erase(kPipelineFallbackStage);
  annotations->erase(kPipelineFallbackReason);
  annotations->erase(kPipelineFallbackDetail);
}

inline For MakePipelineFallback(const For &loop,
                                const PipelineDiagnostic &diagnostic,
                                bool emit_warning = true) {
  Map<String, Any> annotations = loop->annotations;
  annotations.Set(kPipelineRequested, Bool(true));
  annotations.Set(kPipelineApplied, Bool(diagnostic.applied));
  annotations.Set(kPipelineMode, String(diagnostic.mode));
  annotations.Set(kPipelineFallbackStage, String(diagnostic.stage));
  annotations.Set(kPipelineFallbackReason, String(diagnostic.reason));
  if (!diagnostic.detail.empty()) {
    annotations.Set(kPipelineFallbackDetail, String(diagnostic.detail));
  }
  if (emit_warning) {
    LOG(WARNING) << "[SunmmioPipeline][" << diagnostic.mode
                 << "][Fallback] stage=" << diagnostic.stage
                 << " reason=" << diagnostic.reason
                 << " detail=" << diagnostic.detail
                 << " extent=" << loop->extent;
  }
  For fallback = loop;
  fallback.CopyOnWrite()->annotations = annotations;
  return fallback;
}

inline For MakePipelineFallback(const For &loop, const std::string &mode,
                                const std::string &stage,
                                const std::string &reason,
                                bool emit_warning = true) {
  return MakePipelineFallback(
      loop, PipelineDiagnostic{false, mode, stage, reason, ""}, emit_warning);
}

inline bool IsPipelineScheduleAnnotation(const String &key) {
  static const std::unordered_set<std::string> keys = {
      "num_stages",
      "iterations",
      "ii",
      "makespan",
      "stage_count",
      "steady_state_max_iter_offset",
      "prologue_orders",
      "body_orders",
      "epilogue_orders",
      "dynamic_epilogue_orders",
      "used_buffers",
      "versioned_buffers",
      "bank_peer_buffers",
      "version_axis_buffers",
      "runtime_multiversion_buffers",
      "runtime_banked_buffers",
      "runtime_resident_banked_buffers",
      "runtime_bank_peer_buffers",
      "runtime_bank_start_phases",
      "runtime_bank_read_delta_parities",
      "runtime_bank_writer_phases",
      "runtime_bank_reader_phases",
  };
  return keys.count(std::string(key)) != 0;
}

class PipelineAtomicFallbackRewriter : public StmtMutator {
public:
  PipelineAtomicFallbackRewriter(std::string mode, std::string stage,
                                 std::string reason, std::string detail)
      : diagnostic_{false, std::move(mode), std::move(stage), std::move(reason),
                    std::move(detail)} {}

  bool changed() const { return changed_; }

private:
  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtMutator::VisitStmt_(op));
    if (!op->annotations.count(kPipelineRequested) &&
        !op->annotations.count("prologue_orders") &&
        !op->annotations.count("body_orders")) {
      return loop;
    }
    Map<String, Any> annotations;
    for (const auto &kv : loop->annotations) {
      if (!IsPipelineScheduleAnnotation(kv.first) &&
          kv.first != kPipelineApplied && kv.first != kPipelineFallbackStage &&
          kv.first != kPipelineFallbackReason &&
          kv.first != kPipelineFallbackDetail) {
        annotations.Set(kv.first, kv.second);
      }
    }
    loop.CopyOnWrite()->annotations = annotations;
    changed_ = true;
    return MakePipelineFallback(loop, diagnostic_, false);
  }

  PipelineDiagnostic diagnostic_;
  bool changed_{false};
};

inline PrimFunc
MakePipelineFunctionFallback(const PrimFunc &original,
                             const PipelineDiagnostic &diagnostic) {
  PipelineAtomicFallbackRewriter rewriter(diagnostic.mode, diagnostic.stage,
                                          diagnostic.reason, diagnostic.detail);
  PrimFunc fallback = original;
  auto *fptr = fallback.CopyOnWrite();
  fptr->body = rewriter(original->body);
  LOG(WARNING) << "[SunmmioPipeline][" << diagnostic.mode
               << "][AtomicFallback] stage=" << diagnostic.stage
               << " reason=" << diagnostic.reason
               << " detail=" << diagnostic.detail;
  return fallback;
}

class PipelineFallbackValidator : public StmtVisitor {
public:
  static Optional<String> FindDisallowed(const Stmt &body) {
    PipelineFallbackValidator validator;
    validator(body);
    return validator.reason_;
  }

private:
  void VisitStmt_(const ForNode *op) final {
    auto reason = op->annotations.Get(kPipelineFallbackReason);
    if (reason && !reason_) {
      String value = Downcast<String>(reason.value());
      if (value != "runtime_short_extent") {
        reason_ = value;
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

  Optional<String> reason_;
};

} // namespace tl
} // namespace tvm

#endif // TILELANG_TRANSFORM_SUNMMIO_PIPELINE_DIAGNOSTIC_H_
