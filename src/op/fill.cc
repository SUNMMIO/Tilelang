/*!
 * \file tl/op/fill.cc
 *
 * Define elment-wise operators.
 */

#include "fill.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "../layout/tcgen05_layout.h"
#include "../target/utils.h"
#include "../tileview/tileview.h"
#include "../transform/common/attr.h"
#include "../transform/common/loop_fusion_utils.h"
#include "../transform/loop_partition.h"
#include "../transform/loop_vectorize.h"
#include "builtin.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tir;

/**
 * @brief Construct a Fill operator node from call arguments and a buffer map.
 *
 * This constructor builds a FillNode describing an element-wise fill of a
 * destination buffer region with a scalar/vector value and stores it in
 * `data_`.
 *
 * Detailed behavior:
 * - If `args[0]` is a `BufferLoad`, the loaded buffer becomes the destination
 * and the load indices are converted to per-dimension ranges:
 *   - `Ramp(base, lanes, stride)` is converted to `Range(base, lanes)`. Only
 * stride == 1 and constant `lanes` are supported.
 *   - Non-ramp indices become `Range(index, 1)`.
 * - Otherwise `args[0]` is treated as an access pointer; the destination buffer
 * is resolved via `vmap[GetVarFromAccessPtr(args[0])]` and the region is the
 * full buffer shape for each dimension.
 * - `args[1]` is used as the fill value; it is cast to the destination buffer's
 * dtype if necessary.
 * - Performs validation:
 *   - Region dimensionality must match destination rank.
 *   - For statically-known region mins and extents, checks that mins >= 0 and
 * extents do not exceed the corresponding destination shape extents.
 *
 * Parameters:
 * @param args Call arguments: expected layout is [dst_access_or_bufferload,
 * value].
 *             - args[0]: destination access (BufferLoad or pointer expression).
 *             - args[1]: value to fill (scalar or vector).
 *
 * Notes:
 * - The constructor enforces constraints (e.g., stride == 1 ramps, constant
 * lanes) and will terminate (via CHECK/ICHECK) if inputs are unsupported or out
 * of bounds.
 */
Fill::Fill(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ObjectPtr<FillNode> node = tvm::ffi::make_object<FillNode>();

  BufferRegion region = NormalizeToBufferRegion(args[0]);
  node->dst = region->buffer;
  node->region = region->region;

  if (args[1]->dtype != node->dst->dtype) {
    node->value = Cast(node->dst->dtype, args[1]);
  } else {
    node->value = args[1];
  }

  ICHECK(node->region.size() == node->dst->shape.size())
      << "region size = " << node->region.size()
      << " != " << node->dst->shape.size();
  for (int i = 0; i < node->region.size(); i++) {
    // bound check if region is static
    if (const auto *min_imm = node->region[i]->min.as<IntImmNode>()) {
      int64_t min = min_imm->value;
      ICHECK_GE(min, 0) << "region[" << i << "] = " << min << " < 0";
    }
    if (const auto *extent_imm = node->region[i]->extent.as<IntImmNode>()) {
      // Only perform the upper-bound check when the destination shape
      // extent is also statically known. If the shape is symbolic (e.g., Var),
      // skip this static check to avoid invalid downcasts.
      if (const auto *shape_imm = node->dst->shape[i].as<IntImmNode>()) {
        ICHECK_LE(extent_imm->value, shape_imm->value)
            << "region[" << i << "] = " << extent_imm->value << " > "
            << node->dst->shape[i];
      }
    }
  }
  data_ = std::move(node);
}

/**
 * @brief Create a copy of this FillNode and return it as a TileOperator.
 *
 * Constructs a new FillNode by copying the current node and wraps the copy in a
 * Fill TileOperator.
 *
 * @return TileOperator A TileOperator that owns the copied FillNode.
 */
TileOperator FillNode::Clone() const {
  auto op = tvm::ffi::make_object<FillNode>(*this);
  return Fill(op);
}

/**
 * @brief Build a SIMT-style nested parallel loop that fills the destination
 * buffer.
 *
 * Constructs per-dimension data-parallel loop iterators matching this node's
 * region extents, emits a BufferStore that writes the node's `value` into `dst`
 * at the loop indices, and nests the loops (innermost to outermost) as parallel
 * `For` nodes. Returns the outermost `For` loop representing the complete
 * multi-dimensional fill kernel.
 *
 * @return For Outermost parallel `For` loop of the generated nested SIMT loop.
 */
For FillNode::MakeSIMTLoop(arith::Analyzer *analyzer) const {
  int ndim = dst->shape.size();
  Array<IterVar> loop_vars;
  Array<PrimExpr> dst_indices;
  for (int i = 0; i < ndim; i++) {
    Var var = Var(std::string{char('i' + i)}, region[i]->extent->dtype);
    loop_vars.push_back({region[i], var, IterVarType::kDataPar});
    // Offset the loop induction variable by region min to honor sliced regions
    dst_indices.push_back(region[i]->min + var);
  }
  Stmt body = BufferStore(dst, value, dst_indices);
  for (int i = ndim - 1; i >= 0; i--) {
    body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
               ForKind::kParallel, body);
  }
  return Downcast<For>(body);
}

/**
 * @brief Lower this Fill operator to a TIR statement for the target.
 *
 * Lowers the FillNode into a Stmt according to the destination buffer scope:
 * - "local.fragment" and shared ("shared", "shared.dyn"): create a parallel
 *   operation from a SIMT loop, infer its layout, partition the root loop by
 *   the thread variable, vectorize the resulting thread loop, and, if a
 *   per-thread predicate exists, guard the vectorized loop with that
 *   predicate.
 * - "local": build a SIMT loop and return its vectorized form.
 * - other scopes: fatal error.
 *
 * The lowering may query layout and thread information from @p T and uses the
 * provided analyzer for any required arithmetic/layout analysis.
 *
 * @param T Lowering arguments (target, thread bounds, thread var, layout map).
 * @return Stmt The lowered TIR statement implementing the fill.
 */
Stmt FillNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  if (TargetIsSunmmio(T.target)) {
    // For Sunmmio target, we strictly require the destination buffer to be in
    // "shared.rsram" scope. This is because the vector core on Sunmmio only
    // operates on data residing in RSRAM.
    ICHECK(dst.scope() == "shared.rsram")
        << "For Sunmmio target, Fill operator destination must be in "
           "shared.rsram scope, but got "
        << dst.scope();

    auto it = T.tileview_map.find(dst->data);
    if (it != T.tileview_map.end()) {
      TileView tv = (*it).second;
      // Build map from buffer dim index to tile size
      // This map is essential for handling tiled loops, as we need to know
      // which dimensions are tiled and what their tile sizes are.
      std::unordered_map<int, PrimExpr> dim_to_tile_size;
      int ndim = dst->shape.size();
      for (size_t i = 0; i < tv->IndexMap().size(); i++) {
        const auto *idx_ptr = tv->IndexMap()[i].as<IntImmNode>();
        ICHECK(idx_ptr);
        int dim = idx_ptr->value;
        if (dim < 0)
          dim += ndim;
        dim_to_tile_size[dim] = tv->TileShape()[i];
      }

      Array<IterVar> loop_vars;
      Array<PrimExpr> dst_indices;
      Map<String, ObjectRef> annotations;

      // Add common annotations required for subsequent passes (e.g. TilesLoop)
      // to correctly process and expand the loops.
      // - tile_new_shape: The shape of the buffer after tiling.
      // - tile_dim_map: Mapping between original and tiled dimensions.
      // - tile_tile_size: The size of each tile.
      // - kTileLoopStage: Set to kLegalized to indicate that we have handled
      // the tiling logic.
      // - tiled_buffer: The underlying buffer being tiled.
      // - tile_level_loop: Indicates that this loop iterates over tiles.
      annotations.Set(attr::tile_new_shape, tv->TiledBufferShape());
      annotations.Set(attr::tile_dim_map, tv->IndexMap());
      annotations.Set(attr::tile_tile_size, tv->TileShape());
      annotations.Set(attr::kTileLoopStage,
                      Integer(static_cast<int>(TileLoopStage::kLegalized)));
      annotations.Set(attr::tiled_buffer, dst->data);
      annotations.Set(attr::tile_level_loop, Integer(1));

      for (int i = 0; i < ndim; i++) {
        PrimExpr extent = region[i]->extent;
        bool is_tiled = dim_to_tile_size.count(i);
        if (is_tiled) {
          PrimExpr tile_size = dim_to_tile_size[i];
          // Ensure that the region start and extent are aligned with the tile
          // size. This is a hardware constraint for vector core operations on
          // tiles.
          ICHECK(analyzer->CanProve(truncmod(region[i]->min, tile_size) == 0))
              << "Fill region min " << region[i]->min
              << " not divisible by tile size " << tile_size;
          ICHECK(analyzer->CanProve(truncmod(extent, tile_size) == 0))
              << "Fill region extent " << extent
              << " not divisible by tile size " << tile_size;
          extent = truncdiv(extent, tile_size);
        }
        Var var = Var(std::string{char('i' + i)}, extent->dtype);
        loop_vars.push_back({Range(0, extent), var, IterVarType::kDataPar});
        // We use the original region min + var as the index.
        // The TilesLoop pass will later handle the substitution of var to
        // (var * tile_size + inner_var) for tiled dimensions.
        dst_indices.push_back(region[i]->min + var);
      }

      Stmt body = BufferStore(dst, value, dst_indices);

      for (int i = ndim - 1; i >= 0; i--) {
        Map<String, ObjectRef> loop_anno = annotations;
        // Mark the innermost loop of a tiled dimension as the execution loop
        // for that tile.
        if (dim_to_tile_size.count(i)) {
          loop_anno.Set(attr::tile_execution_loop, Integer(1));
        }
        body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
                   ForKind::kSerial, body, std::nullopt, loop_anno);
      }
      return body;
    }
  }

  if (IsFragmentBuffer(dst)) {
    auto par_op = ParallelOp(MakeSIMTLoop(analyzer));
    par_op->InferLayout({T.target,
                         T.thread_bounds,
                         T.layout_map,
                         analyzer,
                         false,
                         T.buffer_remap,
                         {},
                         {}},
                        InferLevel::kFree);
    auto thread_loop = PartitionLoop(par_op->GetRoot(), T.thread_var, analyzer,
                                     par_op->GetLoopLayout());
    auto vectorized_loop = VectorizeLoop(thread_loop, analyzer, T.layout_map);
    auto unrolled_loop = PragmaUnrollLoop(vectorized_loop);

    if (par_op->GetPredicate(T.thread_var).defined()) {
      return IfThenElse(par_op->GetPredicate(T.thread_var).value(),
                        unrolled_loop);
    }
    return unrolled_loop;
  } else if (IsLocalBuffer(dst) || IsLocalVarBuffer(dst)) {
    auto init_loop = MakeSIMTLoop(analyzer);
    auto vectorized_loop = VectorizeLoop(init_loop, analyzer, T.layout_map);
    auto unrolled_loop = PragmaUnrollLoop(vectorized_loop);
    return unrolled_loop;
  } else if (IsSharedBuffer(dst) || IsGlobalBuffer(dst)) {
    auto par_op = ParallelOp(MakeSIMTLoop(analyzer));
    par_op->InferLayout({T.target,
                         T.thread_bounds,
                         T.layout_map,
                         analyzer,
                         false,
                         T.buffer_remap,
                         {},
                         {}},
                        InferLevel::kFree);
    auto thread_loop = PartitionLoop(par_op->GetRoot(), T.thread_var, analyzer,
                                     par_op->GetLoopLayout());
    auto vectorized_loop = VectorizeLoop(thread_loop, analyzer, T.layout_map);
    auto unrolled_loop = PragmaUnrollLoop(vectorized_loop);
    if (par_op->GetPredicate(T.thread_var).defined()) {
      return IfThenElse(par_op->GetPredicate(T.thread_var).value(),
                        unrolled_loop);
    }
    return unrolled_loop;
  } else {
    LOG(FATAL) << "Unsupported scope " << dst.scope();
    return Stmt();
  }
}

/**
 * @brief Infer memory/layout mapping for the Fill operator.
 *
 * Returns the layout mapping produced by layout inference for this FillNode.
 * Currently no layout inference is performed for Fill and the function returns
 * an empty LayoutMap.
 *
 * @param T Context required for layout inference (unused).
 * @param level The inference level requested (unused).
 * @return LayoutMap Empty map indicating no inferred layouts for this operator.
 */
LayoutMap FillNode::InferLayout(const LayoutInferArgs &T,
                                InferLevel level) const {
  return {};
}

TIR_REGISTER_TL_TILE_OP(Fill, fill)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() { FillNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
