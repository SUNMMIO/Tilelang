/*!
 * \file tileview/tileview.cc
 * \brief Implementation of TileView abstraction.
 */

#include "tileview.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace tl {

using namespace tir;

TileViewNode::TileViewNode(Array<PrimExpr> buffer_shape, Array<PrimExpr> tile_shape,
                           Array<PrimExpr> index_map) {
  ICHECK_EQ(tile_shape.size(), index_map.size())
      << "tile_shape and index_map must have the same size";

  buffer_shape_ = std::move(buffer_shape);
  tile_shape_ = std::move(tile_shape);
  index_map_ = std::move(index_map);

  int ndim = static_cast<int>(buffer_shape_.size());
  arith::Analyzer analyzer;

  // Build a map from original dim index to tile_dim_index
  std::vector<int> dim_to_tile_idx(ndim, -1);
  for (size_t i = 0; i < tile_shape_.size(); ++i) {
    const auto* idx_int = index_map_[i].as<IntImmNode>();
    ICHECK(idx_int) << "index_map must contain integer constants";
    int idx = static_cast<int>(idx_int->value);
    if (idx < 0) {
      idx = ndim + idx;
    }
    ICHECK(idx >= 0 && idx < ndim)
        << "index_map[" << i << "]=" << index_map_[i]
        << " is out of bounds for buffer with " << ndim << " dimensions";
    dim_to_tile_idx[idx] = static_cast<int>(i);
  }

  // Compute tiled_buffer_shape
  // For each original dimension:
  // - If tiled: replace with num_tiles (buffer_dim / tile_dim)
  // - If not tiled: keep as is
  // Then append all tile dimensions at the end

  Array<PrimExpr> tiled_shape;
  for (int d = 0; d < ndim; ++d) {
    if (dim_to_tile_idx[d] >= 0) {
      int tile_idx = dim_to_tile_idx[d];
      PrimExpr buf_dim = buffer_shape_[d];
      PrimExpr tile_dim = tile_shape_[tile_idx];

      // Check divisibility
      PrimExpr remainder = floormod(buf_dim, tile_dim);
      ICHECK(analyzer.CanProve(remainder == 0))
          << "Buffer dimension " << d << " (size=" << buf_dim
          << ") must be divisible by tile dimension " << tile_idx
          << " (size=" << tile_dim << ")";

      // num_tiles = buf_dim / tile_dim
      PrimExpr num_tiles = analyzer.Simplify(floordiv(buf_dim, tile_dim));
      tiled_shape.push_back(num_tiles);
    } else {
      tiled_shape.push_back(buffer_shape_[d]);
    }
  }

  // Append tile shape dimensions at the end
  for (const auto& ts : tile_shape_) {
    tiled_shape.push_back(ts);
  }

  tiled_buffer_shape_ = std::move(tiled_shape);
}

PrimExpr TileViewNode::VectorLanes() const {
  PrimExpr lanes = Integer(1);
  for (const auto& dim : tile_shape_) {
    lanes = lanes * dim;
  }
  return lanes;
}

bool TileViewNode::IsEqual(const TileViewNode* other) const {
  if (other == nullptr) return false;
  bool ret = StructuralEqual()(tile_shape_, other->tile_shape_);
  ret &= StructuralEqual()(index_map_, other->index_map_);
  ret &= StructuralEqual()(buffer_shape_, other->buffer_shape_);
  return ret;
}

void TileViewNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TileViewNode>()
      .def_ro("tile_shape", &TileViewNode::tile_shape_)
      .def_ro("index_map", &TileViewNode::index_map_)
      .def_ro("buffer_shape", &TileViewNode::buffer_shape_)
      .def_ro("tiled_buffer_shape", &TileViewNode::tiled_buffer_shape_);
}

TileView::TileView(Array<PrimExpr> buffer_shape, Array<PrimExpr> tile_shape,
                   Array<PrimExpr> index_map) {
  auto n = tvm::ffi::make_object<TileViewNode>(
      std::move(buffer_shape), std::move(tile_shape), std::move(index_map));
  data_ = std::move(n);
}

TileView makeTileView(Array<PrimExpr> buffer_shape, Array<PrimExpr> tile_shape,
                      Array<PrimExpr> index_map) {
  return TileView(std::move(buffer_shape), std::move(tile_shape),
                  std::move(index_map));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("tl.TileView",
                  [](PackedArgs args, Any* rv) {
                    *rv = TileView(args[0].cast<Array<PrimExpr>>(),
                                   args[1].cast<Array<PrimExpr>>(),
                                   args[2].cast<Array<PrimExpr>>());
                  })
      .def("tl.TileView_tile_shape",
           [](TileView tv) { return tv->TileShape(); })
      .def("tl.TileView_index_map",
           [](TileView tv) { return tv->IndexMap(); })
      .def("tl.TileView_buffer_shape",
           [](TileView tv) { return tv->BufferShape(); })
      .def("tl.TileView_tiled_buffer_shape",
           [](TileView tv) { return tv->TiledBufferShape(); })
      .def("tl.TileView_vector_lanes",
           [](TileView tv) { return tv->VectorLanes(); })
      .def("tl.TileView_is_equal",
           [](TileView tv, TileView other) {
             return tv->IsEqual(other.as<TileViewNode>());
           })
      .def("tl.make_tileview",
           [](Array<PrimExpr> buffer_shape, Array<PrimExpr> tile_shape,
              Array<PrimExpr> index_map) {
             return makeTileView(buffer_shape, tile_shape, index_map);
           });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  TileViewNode::RegisterReflection();
}

}  // namespace tl
}  // namespace tvm
