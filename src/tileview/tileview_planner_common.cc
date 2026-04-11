/*!
 * \file tileview/tileview_planner_common.cc
 * \brief Shared legality and enumeration helpers for TileView planning.
 */

#include "tileview_planner_common.h"

#include <algorithm>

#include <tvm/runtime/logging.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

TrailingTilePattern
MakeTrailingTilePatternImpl(const Array<PrimExpr> &buffer_shape,
                            const std::vector<int> &tile_shape) {
  int buffer_rank = static_cast<int>(buffer_shape.size());
  int tile_rank = static_cast<int>(tile_shape.size());

  TrailingTilePattern pattern;
  pattern.mapped_dims.reserve(tile_rank);
  pattern.tile_shape = tile_shape;
  for (int i = 0; i < tile_rank; ++i) {
    pattern.mapped_dims.push_back(buffer_rank - tile_rank + i);
  }
  return pattern;
}

void AppendRank1Pattern(std::vector<TrailingTilePattern> *patterns,
                        const Buffer &buffer, int tile_width,
                        arith::Analyzer *analyzer) {
  int buffer_rank = static_cast<int>(buffer->shape.size());
  if (buffer_rank < 1) {
    return;
  }

  int width_dim = buffer_rank - 1;
  if (!CanProveDivisible(analyzer, buffer->shape[width_dim], tile_width)) {
    return;
  }
  patterns->push_back(MakeTrailingTilePatternImpl(buffer->shape, {tile_width}));
}

void AppendRank2Pattern(std::vector<TrailingTilePattern> *patterns,
                        const Buffer &buffer, int tile_height, int tile_width,
                        arith::Analyzer *analyzer) {
  int buffer_rank = static_cast<int>(buffer->shape.size());
  if (buffer_rank < 2) {
    return;
  }

  int height_dim = buffer_rank - 2;
  int width_dim = buffer_rank - 1;
  if (!CanProveDivisible(analyzer, buffer->shape[width_dim], tile_width) ||
      !CanProveDivisible(analyzer, buffer->shape[height_dim], tile_height)) {
    return;
  }
  patterns->push_back(
      MakeTrailingTilePatternImpl(buffer->shape, {tile_height, tile_width}));
}

} // namespace

int64_t GetStaticIntValue(const PrimExpr &expr, int64_t fallback) {
  if (const auto *imm = expr.as<IntImmNode>()) {
    return imm->value;
  }
  return fallback;
}

LayoutClass GetLayoutClass(const Buffer &buffer,
                           const Map<Buffer, Layout> &layout_map) {
  return layout_map.count(buffer) ? LayoutClass::kBlockwise32x32
                                  : LayoutClass::kRowMajor;
}

int GetElementBits(const Buffer &buffer) {
  ICHECK_EQ(buffer->dtype.lanes(), 1)
      << "TileView planning expects scalar element dtypes, but buffer "
      << buffer->name << " uses lanes=" << buffer->dtype.lanes() << ".";
  return buffer->dtype.bits();
}

int GetCapacityElems(const Buffer &buffer,
                     const SunmmioTileProcessorConfig &config) {
  int element_bits = GetElementBits(buffer);
  ICHECK_GT(element_bits, 0)
      << "TileView planning requires a positive element bit-width for buffer "
      << buffer->name << ".";
  ICHECK_EQ(config.register_bits % element_bits, 0)
      << "Sunmmio tile register size " << config.register_bits
      << " is not divisible by element bit-width " << element_bits
      << " for buffer " << buffer->name << ".";
  return config.register_bits / element_bits;
}

bool CanProveDivisible(arith::Analyzer *analyzer, const PrimExpr &value,
                       int factor) {
  PrimExpr remainder = analyzer->Simplify(floormod(value, Integer(factor)));
  return analyzer->CanProve(remainder == make_zero(remainder.dtype()));
}

void RequireDivisible(arith::Analyzer *analyzer, const PrimExpr &value,
                      int factor, const PrimExpr &index, const Buffer &buffer) {
  ICHECK(CanProveDivisible(analyzer, value, factor))
      << "Tile access offset " << value << " is not divisible by tile size "
      << factor << " in index " << index << " for buffer " << buffer->name
      << ".";
}

int NormalizeMappedDim(const PrimExpr &expr, int ndim) {
  const auto *imm = expr.as<IntImmNode>();
  ICHECK(imm) << "TileView index_map entries must be IntImm, but got " << expr;
  int mapped_dim = static_cast<int>(imm->value);
  if (mapped_dim < 0) {
    mapped_dim += ndim;
  }
  ICHECK(mapped_dim >= 0 && mapped_dim < ndim)
      << "TileView index_map entry " << expr << " is out of bounds for rank "
      << ndim << ".";
  return mapped_dim;
}

bool HasTrailingIndexMap(const TileView &tv, int exec_rank) {
  if (static_cast<int>(tv->TileDim()) != exec_rank) {
    return false;
  }

  int buf_ndim = static_cast<int>(tv->BufferShape().size());
  int first_exec_dim = buf_ndim - exec_rank;
  for (int i = 0; i < exec_rank; ++i) {
    int mapped_dim = NormalizeMappedDim(tv->IndexMap()[i], buf_ndim);
    if (mapped_dim != first_exec_dim + i) {
      return false;
    }
  }
  return true;
}

int TileElements(const std::vector<int> &tile_shape) {
  int elems = 1;
  for (int extent : tile_shape) {
    elems *= extent;
  }
  return elems;
}

int TileElements(const Array<PrimExpr> &tile_shape) {
  int elems = 1;
  for (const PrimExpr &extent : tile_shape) {
    int64_t value = GetStaticIntValue(extent);
    ICHECK_GT(value, 0)
        << "TileView planning requires a positive static tile extent, but got "
        << extent << ".";
    elems *= static_cast<int>(value);
  }
  return elems;
}

Array<PrimExpr> MakeTileShapeExpr(const std::vector<int> &tile_shape) {
  Array<PrimExpr> tile_shape_expr;
  tile_shape_expr.reserve(tile_shape.size());
  for (int extent : tile_shape) {
    tile_shape_expr.push_back(Integer(extent));
  }
  return tile_shape_expr;
}

Array<PrimExpr> MakeCanonicalIndexMap(int buffer_rank, int tile_rank) {
  Array<PrimExpr> index_map;
  for (int i = 0; i < tile_rank; ++i) {
    index_map.push_back(Integer(i - tile_rank));
  }
  return index_map;
}

TileView MakeTrailingTileView(const Array<PrimExpr> &buffer_shape,
                              const TrailingTilePattern &pattern) {
  return makeTileView(
      buffer_shape, MakeTileShapeExpr(pattern.tile_shape),
      MakeCanonicalIndexMap(static_cast<int>(buffer_shape.size()),
                            static_cast<int>(pattern.tile_shape.size())));
}

std::vector<TrailingTilePattern> EnumerateInferredTrailingTilePatterns(
    const Buffer &buffer, int exec_rank, const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &config, arith::Analyzer *analyzer) {
  ICHECK(exec_rank == 1 || exec_rank == 2)
      << "TileView planning expects execution rank 1 or 2, but got "
      << exec_rank << ".";

  std::vector<TrailingTilePattern> patterns;
  int buffer_rank = static_cast<int>(buffer->shape.size());
  if (buffer_rank < 1) {
    return patterns;
  }

  int capacity_elems = GetCapacityElems(buffer, config);
  LayoutClass layout_class = GetLayoutClass(buffer, layout_map);

  if (layout_class == LayoutClass::kBlockwise32x32) {
    AppendRank1Pattern(&patterns, buffer, config.block_width, analyzer);
  } else {
    for (int tile_width = 1; tile_width <= capacity_elems; ++tile_width) {
      AppendRank1Pattern(&patterns, buffer, tile_width, analyzer);
    }
  }

  if (exec_rank != 2 || buffer_rank < 2) {
    return patterns;
  }

  int64_t row_width = GetStaticIntValue(buffer->shape[buffer_rank - 1]);
  if (layout_class == LayoutClass::kBlockwise32x32) {
    int max_height =
        std::min(config.block_height, capacity_elems / config.block_width);
    for (int tile_height = 1; tile_height <= max_height; ++tile_height) {
      AppendRank2Pattern(&patterns, buffer, tile_height, config.block_width,
                         analyzer);
    }
    return patterns;
  }

  if (row_width <= 0) {
    return patterns;
  }

  int max_single_row_width =
      std::min(capacity_elems, static_cast<int>(row_width));
  for (int tile_width = 1; tile_width <= max_single_row_width; ++tile_width) {
    AppendRank2Pattern(&patterns, buffer, /*tile_height=*/1, tile_width,
                       analyzer);
  }

  if (row_width > capacity_elems) {
    return patterns;
  }

  int max_height = capacity_elems / static_cast<int>(row_width);
  for (int tile_height = 2; tile_height <= max_height; ++tile_height) {
    AppendRank2Pattern(&patterns, buffer, tile_height,
                       static_cast<int>(row_width), analyzer);
  }
  return patterns;
}

TrailingTilePattern ValidateManualTrailingTileView(
    const Buffer &buffer, const TileView &manual_tv, int exec_rank,
    const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &config, arith::Analyzer *analyzer,
    const char *usage, bool enforce_blockwise_width_for_rank1) {
  int tv_rank = static_cast<int>(manual_tv->TileDim());
  int buffer_rank = static_cast<int>(buffer->shape.size());
  ICHECK_EQ(static_cast<int>(manual_tv->BufferShape().size()), buffer_rank)
      << usage << " rank mismatch for buffer " << buffer->name << ".";

  if (exec_rank == 1) {
    ICHECK_EQ(tv_rank, 1) << usage << " is incompatible with buffer "
                          << buffer->name << ", which requires a " << tv_rank
                          << "D TileView.";
  } else {
    ICHECK(tv_rank == 1 || tv_rank == 2)
        << usage << " currently supports only 1D or trailing-2D execution "
        << "TileViews for buffer " << buffer->name << ", but got rank "
        << tv_rank << ".";
  }

  ICHECK(HasTrailingIndexMap(manual_tv, tv_rank))
      << usage << " must target trailing " << tv_rank << " buffer dims for "
      << buffer->name << ".";

  int capacity_elems = GetCapacityElems(buffer, config);
  LayoutClass layout_class = GetLayoutClass(buffer, layout_map);
  int width_dim = buffer_rank - 1;
  int width = GetStaticIntValue(manual_tv->TileShape()[tv_rank - 1]);
  ICHECK_GT(width, 0) << usage
                      << " must use a positive static tile width for buffer "
                      << buffer->name << ".";
  ICHECK(CanProveDivisible(analyzer, buffer->shape[width_dim], width))
      << usage << " width " << width << " must divide trailing buffer dim "
      << buffer->shape[width_dim] << " for buffer " << buffer->name << ".";

  if (layout_class == LayoutClass::kBlockwise32x32 &&
      (tv_rank == 2 || enforce_blockwise_width_for_rank1)) {
    ICHECK_EQ(width, config.block_width)
        << usage << " must use trailing width " << config.block_width
        << " for buffer " << buffer->name << ".";
  } else {
    ICHECK_LE(width, capacity_elems)
        << usage << " width " << width
        << " exceeds the Sunmmio register capacity of " << capacity_elems
        << " elements for buffer " << buffer->name << ".";
  }

  if (tv_rank == 1) {
    return MakeTrailingTilePatternImpl(buffer->shape, {width});
  }

  int height_dim = buffer_rank - 2;
  int height = GetStaticIntValue(manual_tv->TileShape()[0]);
  ICHECK_GT(height, 0) << usage
                       << " must use a positive static tile height for buffer "
                       << buffer->name << ".";
  ICHECK(CanProveDivisible(analyzer, buffer->shape[height_dim], height))
      << usage << " height " << height << " must divide buffer dim "
      << height_dim << " of buffer " << buffer->name << ".";

  if (layout_class == LayoutClass::kBlockwise32x32) {
    ICHECK_LE(height * width, capacity_elems)
        << usage << " shape (" << height << ", " << width
        << ") exceeds the Sunmmio register capacity of " << capacity_elems
        << " elements for buffer " << buffer->name << ".";
    ICHECK_LE(height, config.block_height)
        << usage << " height " << height << " exceeds the modeled block "
        << "height " << config.block_height << " for buffer " << buffer->name
        << ".";
    return MakeTrailingTilePatternImpl(buffer->shape, {height, width});
  }

  int64_t row_width = GetStaticIntValue(buffer->shape[width_dim]);
  ICHECK_GT(row_width, 0) << usage
                          << " requires a static trailing row width for buffer "
                          << buffer->name << ".";
  ICHECK_LE(width, row_width)
      << usage << " width " << width << " exceeds trailing buffer dimension "
      << row_width << " for buffer " << buffer->name << ".";
  if (height > 1) {
    ICHECK_EQ(width, row_width)
        << usage << " multi-row width must match trailing buffer dimension "
        << row_width << " for buffer " << buffer->name << ".";
  }
  ICHECK_LE(height * width, capacity_elems)
      << usage << " shape (" << height << ", " << width
      << ") exceeds the Sunmmio register capacity of " << capacity_elems
      << " elements for buffer " << buffer->name << ".";
  return MakeTrailingTilePatternImpl(buffer->shape, {height, width});
}

} // namespace tl
} // namespace tvm
