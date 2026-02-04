/*!
 * \file global_layout_utils.cc
 * \brief Implementation of utility functions to extract global buffer layouts
 *        from tensor_meta attributes for Sunmmio target.
 */

#include "common/global_layout_utils.h"

#include <tvm/tir/stmt_functor.h>

#include "../layout/layout.h"
#include "../target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

Optional<Layout>
ParseGlobalBufferLayout(const Map<String, ObjectRef> &meta_entry,
                        const Buffer &buffer) {
  // Extract sharded layout info
  auto hdims_obj = meta_entry.Get("sharded_hdims");
  auto hstrides_obj = meta_entry.Get("sharded_hstrides");
  auto hgroups_obj = meta_entry.Get("sharded_hgroups");

  if (!hdims_obj || !hstrides_obj || !hgroups_obj) {
    return Optional<Layout>();
  }

  // Convert to arrays for makeHierarchicalLayout
  Array<Integer> hdims_arr, hstrides_arr, logical_shape_arr;
  Array<Array<Integer>> groups_arr;

  // Parse hdims - it's an Array<Integer> from Python
  Array<Integer> hdims = Downcast<Array<Integer>>(hdims_obj.value());
  for (const auto &dim : hdims) {
    hdims_arr.push_back(dim);
  }

  // Parse hstrides
  Array<Integer> hstrides = Downcast<Array<Integer>>(hstrides_obj.value());
  for (const auto &stride : hstrides) {
    hstrides_arr.push_back(stride);
  }

  // Parse hgroups - Array<Array<Integer>>
  Array<Array<Integer>> hgroups =
      Downcast<Array<Array<Integer>>>(hgroups_obj.value());
  for (const auto &group : hgroups) {
    groups_arr.push_back(group);
  }

  // Use buffer shape as logical shape
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    if (auto *imm = buffer->shape[i].as<IntImmNode>()) {
      logical_shape_arr.push_back(Integer(imm->value));
    } else {
      return Optional<Layout>(); // Dynamic shape not supported
    }
  }

  // Verify that groups_arr matches logical shape dimensions
  if (groups_arr.size() != logical_shape_arr.size()) {
    return Optional<Layout>();
  }

  return makeHierarchicalLayout(hdims_arr, hstrides_arr, groups_arr,
                                logical_shape_arr);
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
