/*!
 * \file global_layout_utils.h
 * \brief Utility functions to extract global buffer layouts from tensor_meta
 *        attributes for Sunmmio target.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_GLOBAL_LAYOUT_UTILS_H_
#define TVM_TL_TRANSFORM_COMMON_GLOBAL_LAYOUT_UTILS_H_

#include <tvm/target/target.h>
#include <tvm/tir/function.h>

#include "../../layout/layout.h"
#include "../../target/utils.h"

namespace tvm {
namespace tl {

using LayoutMap = Map<tir::Buffer, Layout>;

/*!
 * \brief Populate layout_map with global buffer layouts from tensor_meta
 *        attribute. Only applies when target is Sunmmio.
 *
 * \param f The PrimFunc containing tensor_meta attribute
 * \param target The compilation target
 * \param layout_map The layout map to update (in-place)
 * \return true if any layouts were added, false otherwise
 */
bool PopulateGlobalBufferLayouts(const tir::PrimFunc &f, Target target,
                                 LayoutMap *layout_map);

/*!
 * \brief Parse a single buffer's hierarchical layout from tensor_meta entry
 *
 * \param meta_entry The metadata dict for one buffer
 * \param buffer The buffer to create layout for
 * \return Layout object, or nullopt if parsing fails
 */
Optional<Layout>
ParseGlobalBufferLayout(const Map<String, ObjectRef> &meta_entry,
                        const tir::Buffer &buffer);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_GLOBAL_LAYOUT_UTILS_H_
