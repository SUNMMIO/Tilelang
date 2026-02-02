/*!
 * \file tileview.h
 * \brief TileView abstraction for buffer tiling annotation.
 *
 * TileView represents how a buffer is partitioned into tiles for processing
 * by the Tile unit. It is orthogonal to Layout:
 *   - TileView: How a buffer is partitioned into tiles (logical)
 *   - Layout: How elements are arranged in physical memory
 */

#ifndef TVM_TL_TILEVIEW_TILEVIEW_H_
#define TVM_TL_TILEVIEW_TILEVIEW_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/object.h>
#include <tvm/tir/buffer.h>

#include "../support/ffi_aliases.h"

namespace tvm {
namespace tl {

using namespace tir;

class TileView;

/*!
 * \brief TileViewNode represents how a buffer should be tiled for processing.
 *
 * A TileView captures:
 * - tile_shape: The shape of each tile (e.g., (16, 32) for a 2D tile)
 * - index_map: Which dimensions of the buffer are tiled
 * - tiled_buffer_shape: The shape of tiled buffer
 *
 * For Sunmmio target, the Tile unit processes fixed-size 2D tiles with
 * constraints: width must be 32, height can be 8/16/32.
 */
class TileViewNode : public Object {
public:
  TileViewNode() = default;

  /*!
   * \brief Construct a TileViewNode with tile shape and index map.
   * \param buffer_shape The shape of buffer being tiled
   * \param tile_shape The shape of each tile.
   * \param index_map Which dimensions of the buffer are tiled.
   *                  Negative indices are supported (e.g., -1 for last dim).
   */
  TileViewNode(Array<PrimExpr> buffer_shape, Array<PrimExpr> tile_shape,
               Array<PrimExpr> index_map);

  /*! \brief Number of dimensions being tiled. */
  size_t TileDim() const { return tile_shape_.size(); }

  /*! \brief Get the tile shape. */
  Array<PrimExpr> TileShape() const { return tile_shape_; }

  /*! \brief Get the index map (which dimensions are tiled). */
  Array<PrimExpr> IndexMap() const { return index_map_; }

  /*! \brief Get the tiled buffer shape. */
  Array<PrimExpr> TiledBufferShape() const { return tiled_buffer_shape_; }

  /*! \brief Get the original buffer shape. */
  Array<PrimExpr> BufferShape() const { return buffer_shape_; }

  /*!
   * \brief Compute the vector lanes (total elements per tile).
   * \return Product of tile_shape dimensions.
   */
  PrimExpr VectorLanes() const;

  /*! \brief Check equality with another TileViewNode. */
  bool IsEqual(const TileViewNode *other) const;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO("tl.TileView", TileViewNode, Object);
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;

protected:
  /*! \brief The shape of each tile. */
  Array<PrimExpr> tile_shape_;

  /*! \brief Which dimensions of the buffer are tiled.
   *         Supports negative indices (e.g., -1 for last dim). */
  Array<PrimExpr> index_map_;

  /*! \brief The shape of buffer before tiling. */
  Array<PrimExpr> buffer_shape_;

  /*! \brief The shape of tiled buffer. */
  Array<PrimExpr> tiled_buffer_shape_;
};

/*!
 * \brief TileView reference class.
 */
class TileView : public ObjectRef {
public:
  /*!
   * \brief Construct a TileView with tile shape and index map.
   * \param buffer_shape The shape of buffer being tiled
   * \param tile_shape The shape of each tile.
   * \param index_map Which dimensions of the buffer are tiled.
   */
  TVM_DLL TileView(Array<PrimExpr> buffer_shape, Array<PrimExpr> tile_shape,
                   Array<PrimExpr> index_map);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TileView, ObjectRef, TileViewNode);
};

/*!
 * \brief Create a TileView from buffer shape, tile shape, and index map.
 * \param buffer_shape The shape of the buffer being tiled.
 * \param tile_shape The shape of each tile.
 * \param index_map Which dimensions of the buffer are tiled.
 *                  Supports negative indices (e.g., -1 for last dim).
 * \return A TileView object.
 */
TileView makeTileView(Array<PrimExpr> buffer_shape, Array<PrimExpr> tile_shape,
                      Array<PrimExpr> index_map);

namespace attr {
/*! \brief Block attribute containing the TileView map for buffers. */
constexpr const char *kTileViewMap = "tileview_map";
} // namespace attr

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TILEVIEW_TILEVIEW_H_
