/*!
 * \file pipelined_buffer_dependency.cc
 * \brief Structured results for pipelined-loop buffer dependency analysis.
 */

#include "pipelined_buffer_dependency.h"

#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace tl {

BufferDependencyEdge::BufferDependencyEdge(String dep_kind, Integer distance,
                                           Buffer buffer,
                                           BufferRegion src_region,
                                           BufferRegion dst_region,
                                           Optional<Integer> src_effect_id,
                                           Optional<Integer> dst_effect_id) {
  auto node = tvm::ffi::make_object<BufferDependencyEdgeNode>();
  node->dep_kind = std::move(dep_kind);
  node->distance = std::move(distance);
  node->buffer = std::move(buffer);
  node->src_region = std::move(src_region);
  node->dst_region = std::move(dst_region);
  node->src_effect_id = std::move(src_effect_id);
  node->dst_effect_id = std::move(dst_effect_id);
  data_ = std::move(node);
}

BufferDependencyPattern::BufferDependencyPattern(String kind, Buffer buffer,
                                                 Array<BufferRegion> regions,
                                                 Array<Integer> effect_ids,
                                                 String detail) {
  auto node = tvm::ffi::make_object<BufferDependencyPatternNode>();
  node->kind = std::move(kind);
  node->buffer = std::move(buffer);
  node->regions = std::move(regions);
  node->effect_ids = std::move(effect_ids);
  node->detail = std::move(detail);
  data_ = std::move(node);
}

BufferDependencyInfo::BufferDependencyInfo(Buffer buffer,
                                           Array<BufferRegion> state_regions,
                                           Array<BufferRegion> channel_regions) {
  auto node = tvm::ffi::make_object<BufferDependencyInfoNode>();
  node->buffer = std::move(buffer);
  node->state_regions = std::move(state_regions);
  node->channel_regions = std::move(channel_regions);
  data_ = std::move(node);
}

BufferDependencyAnalysis::BufferDependencyAnalysis(
    Array<BufferDependencyInfo> buffers, Array<BufferDependencyEdge> edges,
    Array<BufferDependencyPattern> patterns) {
  auto node = tvm::ffi::make_object<BufferDependencyAnalysisNode>();
  node->buffers = std::move(buffers);
  node->edges = std::move(edges);
  node->patterns = std::move(patterns);
  data_ = std::move(node);
}

void BufferDependencyEdgeNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<BufferDependencyEdgeNode>()
      .def_ro("dep_kind", &BufferDependencyEdgeNode::dep_kind)
      .def_ro("distance", &BufferDependencyEdgeNode::distance)
      .def_ro("buffer", &BufferDependencyEdgeNode::buffer)
      .def_ro("src_region", &BufferDependencyEdgeNode::src_region)
      .def_ro("dst_region", &BufferDependencyEdgeNode::dst_region)
      .def_ro("src_effect_id", &BufferDependencyEdgeNode::src_effect_id)
      .def_ro("dst_effect_id", &BufferDependencyEdgeNode::dst_effect_id);
}

void BufferDependencyPatternNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<BufferDependencyPatternNode>()
      .def_ro("kind", &BufferDependencyPatternNode::kind)
      .def_ro("buffer", &BufferDependencyPatternNode::buffer)
      .def_ro("regions", &BufferDependencyPatternNode::regions)
      .def_ro("effect_ids", &BufferDependencyPatternNode::effect_ids)
      .def_ro("detail", &BufferDependencyPatternNode::detail);
}

void BufferDependencyInfoNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<BufferDependencyInfoNode>()
      .def_ro("buffer", &BufferDependencyInfoNode::buffer)
      .def_ro("state_regions", &BufferDependencyInfoNode::state_regions)
      .def_ro("channel_regions", &BufferDependencyInfoNode::channel_regions);
}

void BufferDependencyAnalysisNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<BufferDependencyAnalysisNode>()
      .def_ro("buffers", &BufferDependencyAnalysisNode::buffers)
      .def_ro("edges", &BufferDependencyAnalysisNode::edges)
      .def_ro("patterns", &BufferDependencyAnalysisNode::patterns);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  BufferDependencyEdgeNode::RegisterReflection();
  BufferDependencyPatternNode::RegisterReflection();
  BufferDependencyInfoNode::RegisterReflection();
  BufferDependencyAnalysisNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm