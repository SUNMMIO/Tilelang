/*!
 * \file global_layout_utils.cc
 * \brief Implementation of utility functions to extract global buffer layouts
 *        from tensor_meta attributes for Sunmmio target.
 */

#include "common/global_layout_utils.h"

#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../layout/cute_layout.h"
#include "../layout/layout.h"
#include "../target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

Optional<Layout>
ParseGlobalBufferLayout(const Map<String, ObjectRef> &meta_entry,
                        const Buffer &buffer) {
  // Extract the CuteLayout object stored directly by MeshTensor.
  auto layout_obj = meta_entry.Get("sharded_layout");
  if (layout_obj) {
    if (auto layout = layout_obj.value().as<Layout>()) {
      return layout.value();
    }
  }
  return Optional<Layout>();
}

bool PopulateGlobalBufferLayouts(const PrimFunc &f, Target target,
                                 LayoutMap *layout_map) {
  if (!TargetIsSunmmio(target)) {
    return false;
  }

  auto tensor_meta_opt = f->GetAttr<Map<String, ObjectRef>>("tensor_meta");
  if (!tensor_meta_opt) {
    return false;
  }

  auto tensor_meta = tensor_meta_opt.value();
  bool any_added = false;

  for (const auto &kv : f->buffer_map) {
    const Var &var = kv.first;
    const Buffer &buffer = kv.second;

    if (buffer.scope() != "global") {
      continue;
    }

    String buffer_name = buffer->name;
    if (!tensor_meta.count(buffer_name)) {
      continue;
    }

    auto meta_entry_obj = tensor_meta[buffer_name];
    auto meta_entry = meta_entry_obj.as<Map<String, ObjectRef>>();
    if (!meta_entry.has_value()) {
      continue;
    }

    auto layout_opt = ParseGlobalBufferLayout(meta_entry.value(), buffer);

    if (layout_opt) {
      layout_map->Set(buffer, layout_opt.value());
      any_added = true;
    }
  }

  return any_added;
}

} // namespace tl
} // namespace tvm
