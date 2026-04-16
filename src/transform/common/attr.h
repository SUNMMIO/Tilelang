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
// Marks the outermost raw loop of a T.Tiles scope and stores the logical
// iteration domain before legalization.
constexpr const char *kTileDomain = "tile.domain";
constexpr const char *kTileLoopStage = "tile.loop_stage";

// ---- these attrs will be added / normalized by legalize_tiles_loop pass ----
// tile.tile_size stores the chosen execution tile shape on the scope root.
// tile.tile_size[k] is the extent of execution axis k.
constexpr const char *tile_tile_size = "tile.tile_size";
// Explicit execution-axis mapping for a planned T.Tiles scope:
//   tile.execution_domain_axes[k] gives the logical domain axis bound to
//   execution axis k. Together with tile.tile_size, this means:
//     tile.tile_size[k] applies to domain axis tile.execution_domain_axes[k].
//   tile.execution_axis on a legalized loop marks that loop as the carrier of
//   execution axis k.
// Example:
//   for j, i in T.Tiles([N, M]) with accesses A[i, j] may infer
//     tile.execution_domain_axes = [1, 0]
//   meaning execution axis 0 uses domain axis 1 (i), and execution axis 1
//   uses domain axis 0 (j).
constexpr const char *tile_execution_domain_axes = "tile.execution_domain_axes";
constexpr const char *tile_execution_axis = "tile.execution_axis";

// ---- these attrs are added by the tiles_loop pass ----
// Marks the outermost tile.execution loop — codegen entry point for a tile
// scope
constexpr const char *tile_scope_entry = "tile.scope_entry";
// Marks generated inner loops (ki, kj) that iterate within a single tile
constexpr const char *tile_interior = "tile.interior";
// Which axis of the tile shape this interior loop corresponds to (0, 1, ...)
constexpr const char *tile_interior_axis = "tile.interior_axis";

} // namespace attr

enum class TileLoopStage : int {
  kInitial = 0,
  kLegalized = 1,
  kTiled = 2,
  kConsumed = 3,
};

} // namespace tl
} // namespace tvm
