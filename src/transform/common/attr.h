/*!
 * \file attr.h
 * \brief Check attributes of the IR
 */

namespace tvm {
namespace tl {

constexpr const char *MainBlockName = "tilelang_root";

constexpr const char *tilelang_is_cpu_kernel_frame =
    "tilelang.is_cpu_kernel_frame";

namespace attr {
// Attributes to mark CUDA sync calls
constexpr const char *kHasTriggerLaunch = "has_cuda_pdl_trigger";
constexpr const char *kHasGridSync = "has_cuda_pdl_sync";

// ---- annotations to indicate the loop is a tile-level loop ----
/*
 * When tile_level_loop is True  -> no loop-carried dependency,
 * when False -> loop carries dependency (e.g. reduction)
 */
constexpr const char *tile_level_loop = "tile.loop_parallel";
constexpr const char *tiled_buffer = "tile.tiled_buffer";
constexpr const char *kTileLoopStage = "tile.loop_stage";

// ---- these attrs will be added / normalized by legalize_tiles_loop pass ----
// Mark the loops corresponding to the index map(index_map=(-2, -1)) for
// subsequent passes
constexpr const char *tile_execution_loop = "tile.execution";
constexpr const char *tile_new_shape = "tile.buffer_new_shape";
constexpr const char *tile_tile_size = "tile.tile_size";
constexpr const char *tile_dim_map = "tile.dim_map";

} // namespace attr

enum class TileLoopStage : int {
  kInitial = 0,
  kLegalized = 1,
  kTiled = 2,
  kConsumed = 3,
};

} // namespace tl
} // namespace tvm
