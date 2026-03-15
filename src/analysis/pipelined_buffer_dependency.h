/*!
 * \file pipelined_buffer_dependency.h
 * \brief Structured results for pipelined-loop buffer dependency analysis.
 */

#ifndef TVM_TL_TRANSFORM_BUFFER_DEPENDENCY_ANALYSIS_H_
#define TVM_TL_TRANSFORM_BUFFER_DEPENDENCY_ANALYSIS_H_

#include <tvm/ffi/object.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/stmt.h>

#include "../support/ffi_aliases.h"

namespace tvm {
namespace tl {

using namespace tir;

class BufferDependencyEdge;
class BufferDependencyPattern;
class BufferDependencyInfo;
class BufferDependencyAnalysis;

class BufferDependencyEdgeNode : public Object {
public:
  String dep_kind;
  Integer distance;
  Buffer buffer;
  BufferRegion src_region;
  BufferRegion dst_region;
  Optional<Integer> src_effect_id;
  Optional<Integer> dst_effect_id;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.BufferDependencyEdge",
                                    BufferDependencyEdgeNode, Object);
};

class BufferDependencyEdge : public ObjectRef {
public:
  TVM_DLL
  BufferDependencyEdge(String dep_kind, Integer distance, Buffer buffer,
                       BufferRegion src_region, BufferRegion dst_region,
                       Optional<Integer> src_effect_id = Optional<Integer>(),
                       Optional<Integer> dst_effect_id = Optional<Integer>());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(BufferDependencyEdge, ObjectRef,
                                             BufferDependencyEdgeNode);
};

class BufferDependencyPatternNode : public Object {
public:
  String kind;
  Buffer buffer;
  Array<BufferRegion> regions;
  Array<Integer> effect_ids;
  String detail;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.BufferDependencyPattern",
                                    BufferDependencyPatternNode, Object);
};

class BufferDependencyPattern : public ObjectRef {
public:
  TVM_DLL BufferDependencyPattern(String kind, Buffer buffer = Buffer(),
                                  Array<BufferRegion> regions = {},
                                  Array<Integer> effect_ids = {},
                                  String detail = "");

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(BufferDependencyPattern, ObjectRef,
                                             BufferDependencyPatternNode);
};

class BufferDependencyInfoNode : public Object {
public:
  Buffer buffer;
  Array<BufferRegion> state_regions;
  Array<BufferRegion> channel_regions;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.BufferDependencyInfo",
                                    BufferDependencyInfoNode, Object);
};

class BufferDependencyInfo : public ObjectRef {
public:
  TVM_DLL BufferDependencyInfo(Buffer buffer, Array<BufferRegion> state_regions,
                               Array<BufferRegion> channel_regions);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(BufferDependencyInfo, ObjectRef,
                                             BufferDependencyInfoNode);
};

class BufferDependencyAnalysisNode : public Object {
public:
  Array<BufferDependencyInfo> buffers;
  Array<BufferDependencyEdge> edges;
  Array<BufferDependencyPattern> patterns;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.BufferDependencyAnalysis",
                                    BufferDependencyAnalysisNode, Object);
};

class BufferDependencyAnalysis : public ObjectRef {
public:
  TVM_DLL BufferDependencyAnalysis(Array<BufferDependencyInfo> buffers,
                                   Array<BufferDependencyEdge> edges,
                                   Array<BufferDependencyPattern> patterns);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(BufferDependencyAnalysis,
                                             ObjectRef,
                                             BufferDependencyAnalysisNode);
};

namespace attr {
constexpr const char *kBufferDependencyAnalysis =
    "tl_buffer_dependency_analysis";
} // namespace attr

namespace buffer_dependency {
constexpr const char *kDepKindRAW = "RAW";

// A current-iteration write is proven to fully cover a loop-carried reaching
// definition of the same buffer region. This is a structural proof that the
// previous iteration's value is completely rewritten before any unsupported
// "untouched remainder" can be observed.
constexpr const char *kPatternCoveredRewrite = "covered_rewrite";
// The same physical buffer has disjoint regions that play different semantic
// roles: one region is loop-carried state, while another region is only a
// current-iteration channel/write region. This is a factual region partition,
// not a support-policy decision.
constexpr const char *kPatternMixedRoleRegions = "mixed_role_regions";
// A loop-carried region is only partially overwritten, and a later read touches
// the untouched remainder rather than the overwritten part. This is the
// concrete witness that the carried value survives in a subregion after the
// overwrite.
constexpr const char *kPatternPartialOverwriteRemainderRead =
    "partial_overwrite_remainder_read";
// A statement-level opaque/update-state call was seen in the pipelined loop,
// but the analysis has no region-level decoder for its memory effects. This is
// a fail-closed fact that downstream passes can use to reject or route the
// loop to a more conservative lowering path.
constexpr const char *kPatternUnknownEffect = "unknown_effect";
} // namespace buffer_dependency

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_BUFFER_DEPENDENCY_ANALYSIS_H_
