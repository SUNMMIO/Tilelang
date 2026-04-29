/*!
 * \file transform/sunmmio_layout_inference.h
 * \brief Standalone layout inference pass for the Sunmmio target.
 *
 * Assigns CuteLayout to every buffer and validates layout consistency.
 * Does NOT rewrite IR — attaches layout_map / global_layout_map as
 * block annotations for downstream consumption by LowerTileOp.
 */

#ifndef TVM_TL_TRANSFORM_SUNMMIO_LAYOUT_INFERENCE_H_
#define TVM_TL_TRANSFORM_SUNMMIO_LAYOUT_INFERENCE_H_

#include <tvm/target/target.h>
#include <tvm/tir/function.h>
#include <tvm/tir/transform.h>

#include "../layout/cute_layout.h"
#include "../op/operator.h"

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Per-buffer layout assignment with provenance tracking.
 */
struct SunmmioLayoutEntry {
  Layout layout;
  InferLevel level;  // kStrict > kCommon > kFree
  int source_op_idx; // -1 for pre-seeded DRAM or pass-assigned defaults
  bool is_immutable; // true for DRAM metadata or T.annotate_layout
};

/*!
 * \brief Sunmmio layout inference pass.
 *
 * Algorithm phases:
 *   1. Collect operators, buffers, annotations, aliases
 *   2. Pre-seed immutable layouts (DRAM metadata + T.annotate_layout)
 *   3. kFree — scope-dependent defaults (baseline for all SRAM)
 *   4. kStrict — seed hard constraints (Gemm overrides defaults)
 *   5. kCommon — BFS propagation (ops derive and override defaults)
 *  5A. Alias propagation
 *   6. Apply — attach layout_map / global_layout_map annotations
 *
 * Validation is not a separate phase — each op validates hard constraints
 * inside its InferLayout when all operands have layouts.
 */
class SunmmioLayoutInferencePass {
public:
  static PrimFunc Run(PrimFunc f);

private:
  // Phase 1: Collect
  void Collect(const PrimFunc &f);

  // Phase 2: Pre-seed immutable layouts (DRAM + T.annotate_layout)
  void SeedImmutableLayouts(const PrimFunc &f);

  // Phase 3: kFree — scope-dependent defaults (baseline for all SRAM)
  void AssignDefaults();

  // Phase 4: kStrict — run all ops at kStrict (sequential)
  void SeedStrictLayouts();

  // Phase 5: kCommon — BFS propagation
  void PropagateBFS();

  // Phase 5A: Alias propagation
  void PropagateAliases();

  // Phase 6: Apply to IR (attach layout_map, no IR rewriting)
  PrimFunc ApplyToIR(PrimFunc f);

  // --- Helpers ---

  bool TryAssign(const Buffer &buffer, const Layout &layout, InferLevel level,
                 int op_idx, bool immutable = false);

  LayoutInferArgs BuildInferArgs() const;

  // --- State ---

  Target target_;
  arith::Analyzer analyzer_;

  // Layout assignments (both SRAM and DRAM during processing)
  std::unordered_map<Buffer, SunmmioLayoutEntry, ObjectPtrHash, ObjectPtrEqual>
      layout_entries_;

  // Operator tracking
  std::vector<TileOperator> op_list_;
  std::unordered_map<Buffer, std::vector<int>, ObjectPtrHash, ObjectPtrEqual>
      buffer_to_ops_;

  // Buffer sets
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> sram_buffers_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> dram_buffers_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> tiles_buffers_;

  // Alias groups: multiple Buffer objects sharing the same data Var
  Map<Var, Array<Buffer>> buffer_data_to_buffers_;

  // From T.annotate_layout block attribute
  LayoutMap annotated_layout_map_;

  // DRAM layouts (separate from SRAM layout_entries)
  LayoutMap global_layout_map_;

  // LetStmt bindings for resolving buffer access through let vars
  Map<Var, PrimExpr> let_var_to_expr_;
};

tvm::transform::Pass SunmmioLayoutInference();

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_SUNMMIO_LAYOUT_INFERENCE_H_
