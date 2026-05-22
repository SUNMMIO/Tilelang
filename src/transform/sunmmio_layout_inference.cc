/*!
 * \file transform/sunmmio_layout_inference.cc
 * \brief Standalone layout inference pass for the Sunmmio target.
 *
 * Assigns CuteLayout to every buffer and validates layout consistency
 * across operators. Does NOT rewrite IR — attaches layout_map and
 * global_layout_map as block annotations for downstream LowerTileOp.
 *
 * See docs/design/sunmmio_layout_inference.md for the full design.
 */

#include "sunmmio_layout_inference.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include <deque>
#include <unordered_set>

#include "../layout/cute_layout.h"
#include "../layout/layout.h"
#include "../op/builtin.h"
#include "../op/operator.h"
#include "../op/region.h"
#include "../target/sunmmio_utils.h"
#include "../target/utils.h"
#include "common/global_layout_utils.h"

namespace tvm {
namespace tl {

using namespace tir;

// ---------------------------------------------------------------------------
// SunmmioLayoutInferencePass — static entry point
// ---------------------------------------------------------------------------

PrimFunc SunmmioLayoutInferencePass::Run(PrimFunc f) {
  SunmmioLayoutInferencePass pass;

  // Extract target
  auto target_opt = f->GetAttr<Target>(tvm::attr::kTarget);
  ICHECK(target_opt.defined()) << "SunmmioLayoutInference: PrimFunc has no "
                                  "target attribute.";
  pass.target_ = target_opt.value();
  ICHECK(TargetIsSunmmio(pass.target_))
      << "SunmmioLayoutInference: target is not Sunmmio: "
      << pass.target_->str();

  // Phase 1: Collect operators, buffers, annotations, aliases
  pass.Collect(f);

  // Phase 2: Pre-seed immutable layouts (DRAM metadata + T.annotate_layout)
  pass.SeedImmutableLayouts(f);

  // Phase 3: kFree — scope-dependent defaults (baseline for all SRAM)
  pass.AssignDefaults();

  // Phase 4: kStrict — seed hard constraints (Gemm overrides defaults)
  pass.SeedStrictLayouts();

  // Phase 5: kCommon — BFS propagation (ops derive and override defaults)
  pass.PropagateBFS();

  // Phase 5A: Alias propagation
  pass.PropagateAliases();

  // Phase 6: Apply annotations to IR
  return pass.ApplyToIR(std::move(f));
}

// ---------------------------------------------------------------------------
// Phase 1: Collect
// ---------------------------------------------------------------------------

namespace {

/*!
 * \brief IR visitor that collects TileOps, T.Tiles buffers,
 *        annotations, buffer aliases, and LetStmt bindings.
 */
class SunmmioIRCollector : public StmtExprVisitor {
public:
  SunmmioIRCollector(
      std::vector<TileOperator> *op_list,
      std::unordered_map<Buffer, std::vector<int>, ObjectPtrHash,
                         ObjectPtrEqual> *buffer_to_ops,
      std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> *sram_buffers,
      std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> *dram_buffers,
      std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> *tiles_buffers,
      Map<Var, Array<Buffer>> *buffer_data_to_buffers,
      LayoutMap *annotated_layout_map, Map<Var, PrimExpr> *let_var_to_expr)
      : op_list_(op_list), buffer_to_ops_(buffer_to_ops),
        sram_buffers_(sram_buffers), dram_buffers_(dram_buffers),
        tiles_buffers_(tiles_buffers),
        buffer_data_to_buffers_(buffer_data_to_buffers),
        annotated_layout_map_(annotated_layout_map),
        let_var_to_expr_(let_var_to_expr) {}

  void VisitStmt_(const BlockNode *op) final {
    // 1d: Collect allocated buffers for alias tracking (before reading
    // annotations so buffer_data_to_buffers_ is populated)
    for (const auto &buf : op->alloc_buffers) {
      RegisterBuffer(buf);
    }

    // 1c: Read T.annotate_layout from block annotations.
    // The annotation is stored as Map<Var, Layout> keyed by buffer->data.
    if (auto it = op->annotations.find(attr::kLayoutMap);
        it != op->annotations.end()) {
      if (auto lm = (*it).second.as<Map<Var, Layout>>()) {
        for (const auto &[var, layout] : lm.value()) {
          // Resolve Var → Buffer via buffer_data_to_buffers_
          if (buffer_data_to_buffers_->count(var)) {
            for (const auto &buf : (*buffer_data_to_buffers_)[var]) {
              annotated_layout_map_->Set(buf, layout);
            }
          }
        }
      }
    }

    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (auto call = op->value.as<Call>()) {
      auto tile_op = ParseOperator(call.value());
      if (tile_op.defined()) {
        int op_idx = static_cast<int>(op_list_->size());
        op_list_->push_back(tile_op);
        // Track all buffers this op accesses via call arg encodings
        // (tvm_access_ptr, tl.region, BufferLoad, etc.).
        // All buffers are allocated in the enclosing Block; the call args
        // are the single source of truth for buffer→op association —
        // consistent with the original layout_inference.cc approach.
        for (const auto &arg : call.value()->args) {
          auto buf = GetBufferFromArg(arg);
          if (buf.defined()) {
            auto &ops = (*buffer_to_ops_)[buf.value()];
            if (ops.empty() || ops.back() != op_idx) {
              ops.push_back(op_idx);
            }
          }
        }
        inside_tileop_ = true;
        StmtExprVisitor::VisitStmt_(op);
        inside_tileop_ = false;
        return;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const LetStmtNode *op) final {
    let_var_to_expr_->Set(op->var, op->value);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    RegisterBuffer(op->buffer);
    if (!inside_tileop_) {
      // 1b: T.Tiles buffer — element-wise access outside a TileOp
      if (IsSunmmioSramScope(op->buffer.scope())) {
        tiles_buffers_->insert(op->buffer);
        // Register with sentinel op index -1
        if (buffer_to_ops_->find(op->buffer) == buffer_to_ops_->end()) {
          (*buffer_to_ops_)[op->buffer] = {-1};
        }
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    RegisterBuffer(op->buffer);
    if (!inside_tileop_) {
      if (IsSunmmioSramScope(op->buffer.scope())) {
        tiles_buffers_->insert(op->buffer);
        if (buffer_to_ops_->find(op->buffer) == buffer_to_ops_->end()) {
          (*buffer_to_ops_)[op->buffer] = {-1};
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  // Collect buffers from PrimFunc::buffer_map
  void CollectFuncBuffers(const PrimFunc &f) {
    for (const auto &[var, buf] : f->buffer_map) {
      RegisterBuffer(buf);
    }
  }

private:
  /*!
   * \brief Extract a Buffer from a TileOp call argument.
   *
   * Handles tvm_access_ptr, tl.access_ptr, tl.region, and BufferLoad.
   */
  Optional<Buffer> GetBufferFromArg(const PrimExpr &expr) {
    // BufferLoad — direct buffer reference
    if (auto bl = expr.as<BufferLoadNode>()) {
      RegisterBuffer(bl->buffer);
      return bl->buffer;
    }
    auto call = expr.as<CallNode>();
    if (!call)
      return std::nullopt;
    // tvm_access_ptr or tl.access_ptr — extract data Var, look up buffer
    if (call->op.same_as(builtin::tvm_access_ptr())) {
      auto var_opt = call->args[1].as<Var>();
      if (var_opt.has_value()) {
        return LookupBufferByVar(var_opt.value());
      }
      return std::nullopt;
    }
    if (call->op.same_as(tl::access_ptr())) {
      if (!call->args.empty()) {
        if (auto load = call->args[0].as<BufferLoadNode>()) {
          RegisterBuffer(load->buffer);
          return load->buffer;
        }
      }
      return std::nullopt;
    }
    // tl.region — extract buffer from region descriptor
    if (call->op.same_as(RegionOp::Get())) {
      if (!call->args.empty()) {
        if (auto load = call->args[0].as<BufferLoadNode>()) {
          RegisterBuffer(load->buffer);
          return load->buffer;
        }
      }
      return std::nullopt;
    }
    return std::nullopt;
  }

  Optional<Buffer> LookupBufferByVar(const Var &var) {
    if (buffer_data_to_buffers_->count(var)) {
      const auto &buffers = (*buffer_data_to_buffers_)[var];
      if (!buffers.empty()) {
        return buffers[0];
      }
    }
    return std::nullopt;
  }

  void RegisterBuffer(const Buffer &buf) {
    if (all_buffers_.count(buf))
      return;
    all_buffers_.insert(buf);

    // Classify by scope
    if (IsSunmmioSramScope(buf.scope())) {
      sram_buffers_->insert(buf);
    } else if (buf.scope() == "global" || buf.scope().empty()) {
      dram_buffers_->insert(buf);
    }

    // 1d: Build alias groups
    if (buffer_data_to_buffers_->count(buf->data)) {
      auto arr = (*buffer_data_to_buffers_)[buf->data];
      arr.push_back(buf);
      buffer_data_to_buffers_->Set(buf->data, arr);
    } else {
      buffer_data_to_buffers_->Set(buf->data, {buf});
    }
  }

  std::vector<TileOperator> *op_list_;
  std::unordered_map<Buffer, std::vector<int>, ObjectPtrHash, ObjectPtrEqual>
      *buffer_to_ops_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> *sram_buffers_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> *dram_buffers_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> *tiles_buffers_;
  Map<Var, Array<Buffer>> *buffer_data_to_buffers_;
  LayoutMap *annotated_layout_map_;
  Map<Var, PrimExpr> *let_var_to_expr_;

  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> all_buffers_;
  bool inside_tileop_ = false;
};

} // namespace

void SunmmioLayoutInferencePass::Collect(const PrimFunc &f) {
  SunmmioIRCollector collector(&op_list_, &buffer_to_ops_, &sram_buffers_,
                               &dram_buffers_, &tiles_buffers_,
                               &buffer_data_to_buffers_, &annotated_layout_map_,
                               &let_var_to_expr_);
  collector.CollectFuncBuffers(f);
  collector(f->body);
}

// ---------------------------------------------------------------------------
// Phase 2: Pre-seed immutable layouts
// ---------------------------------------------------------------------------

void SunmmioLayoutInferencePass::SeedImmutableLayouts(const PrimFunc &f) {
  // 2a: DRAM layouts from tensor_meta
  PopulateGlobalBufferLayouts(f, target_, &global_layout_map_);

  for (const auto &buf : dram_buffers_) {
    if (global_layout_map_.count(buf)) {
      TryAssign(buf, global_layout_map_[buf], InferLevel::kStrict, -1,
                /*immutable=*/true);
    } else {
      // No metadata — default to row-major
      auto row_major = sunmmio::MakeRowMajor(buf->shape);
      global_layout_map_.Set(buf, row_major);
      TryAssign(buf, row_major, InferLevel::kStrict, -1, /*immutable=*/true);
    }
  }

  // 2b: User-annotated layouts from T.annotate_layout
  for (const auto &[buf, layout] : annotated_layout_map_) {
    TryAssign(buf, layout, InferLevel::kStrict, -1, /*immutable=*/true);
  }
}

// ---------------------------------------------------------------------------
// Phase 3: kFree — scope-dependent defaults (baseline for all SRAM)
// ---------------------------------------------------------------------------

void SunmmioLayoutInferencePass::AssignDefaults() {
  // Assign scope-dependent defaults as a kFree baseline for every SRAM
  // buffer that doesn't already have a layout (from Phase 2 immutables).
  // These can be overridden by kStrict (Phase 4) or kCommon (Phase 5).
  for (const auto &buf : sram_buffers_) {
    if (layout_entries_.count(buf))
      continue;

    String scope = buf.scope();
    int rank = static_cast<int>(buf->shape.size());

    if (rank >= 2) {
      Array<Integer> axes{Integer(rank - 2), Integer(rank - 1)};
      auto block_shape = GetSunmmioLayoutBlockShape(target_, buf->dtype);
      TryAssign(buf, sunmmio::MakeZZ(buf->shape, axes, block_shape),
                InferLevel::kFree, -1);
    } else {
      TryAssign(buf, sunmmio::MakeRowMajor(buf->shape), InferLevel::kFree, -1);
    }
  }

  // Final check: every SRAM buffer must have a layout
  for (const auto &buf : sram_buffers_) {
    ICHECK(layout_entries_.count(buf))
        << "SunmmioLayoutInference: Buffer " << buf->name
        << " (scope=" << buf.scope() << ") has no layout after inference.";
  }
}

// ---------------------------------------------------------------------------
// Phase 4: kStrict — seed hard constraints
// ---------------------------------------------------------------------------

void SunmmioLayoutInferencePass::SeedStrictLayouts() {
  for (int i = 0; i < static_cast<int>(op_list_.size()); ++i) {
    auto args = BuildInferArgs();
    auto updates = op_list_[i]->InferLayout(args, InferLevel::kStrict);
    for (const auto &[buf, layout] : updates) {
      TryAssign(buf, layout, InferLevel::kStrict, i);
    }
  }
}

// ---------------------------------------------------------------------------
// Phase 5: kCommon — BFS propagation
// ---------------------------------------------------------------------------

void SunmmioLayoutInferencePass::PropagateBFS() {
  int num_ops = static_cast<int>(op_list_.size());
  std::deque<int> queue;
  std::vector<bool> in_queue(num_ops, false);

  // Seed the queue: any op touching a buffer with a known layout
  for (int i = 0; i < num_ops; ++i) {
    for (const auto &[buf, ops] : buffer_to_ops_) {
      for (int op_idx : ops) {
        if (op_idx == i && layout_entries_.count(buf)) {
          if (!in_queue[i]) {
            queue.push_back(i);
            in_queue[i] = true;
          }
        }
      }
    }
  }

  // Helper: enqueue all ops touching a buffer (except the current one)
  auto enqueue_neighbors = [&](const Buffer &buf, int exclude_op) {
    if (buffer_to_ops_.count(buf)) {
      for (int j : buffer_to_ops_[buf]) {
        if (j >= 0 && j != exclude_op && !in_queue[j]) {
          queue.push_back(j);
          in_queue[j] = true;
        }
      }
    }
  };

  while (!queue.empty()) {
    int op_idx = queue.front();
    queue.pop_front();
    in_queue[op_idx] = false;

    // Generic kCommon via operator InferLayout.
    // Each op returns layout assignments for its operands (or {} if it has
    // nothing to propose).  Copy returns {} — it does not propagate layouts;
    // a DRAM<->RSRAM mismatch is bridged later in CopyNode::Lower by
    // splitting into a dma_copy plus a sunmmio_layout_transform.
    auto args = BuildInferArgs();
    auto updates = op_list_[op_idx]->InferLayout(args, InferLevel::kCommon);

    for (const auto &[buf, layout] : updates) {
      bool accepted = TryAssign(buf, layout, InferLevel::kCommon, op_idx);
      if (accepted) {
        enqueue_neighbors(buf, op_idx);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Phase 5A: Alias propagation (after BFS)
// ---------------------------------------------------------------------------

void SunmmioLayoutInferencePass::PropagateAliases() {
  for (const auto &[var, buffers] : buffer_data_to_buffers_) {
    // Find the representative with the highest-level layout in the alias
    // group.  This ensures we propagate from kStrict/kCommon entries
    // (real constraints) rather than kFree defaults.
    Layout rep_layout;
    InferLevel rep_level = InferLevel::kFree;
    for (const auto &buf : buffers) {
      auto it = layout_entries_.find(buf);
      if (it != layout_entries_.end() && it->second.level >= rep_level) {
        rep_layout = it->second.layout;
        rep_level = it->second.level;
      }
    }
    if (!rep_layout.defined())
      continue;

    // Propagate to siblings that only have kFree defaults.
    // Buffers with kCommon or kStrict layouts already have real constraints
    // from ops — don't override them.
    for (const auto &buf : buffers) {
      auto it = layout_entries_.find(buf);
      if (it != layout_entries_.end() && it->second.level > InferLevel::kFree)
        continue;

      // Try DeriveLayoutLike for shape adaptation
      auto derived = DeriveLayoutLike(rep_layout, buf->shape,
                                      Optional<Array<Integer>>(), &analyzer_);
      if (derived.defined()) {
        TryAssign(buf, derived.value(), InferLevel::kCommon, -1);
      } else if (!layout_entries_.count(buf)) {
        // Only warn for buffers with no entry at all (shouldn't happen
        // after Phase 3 defaults, but guard for safety)
        LOG(WARNING)
            << "SunmmioLayoutInference: Cannot derive layout for alias "
            << buf->name << " from representative layout.";
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Phase 6: Apply to IR
// ---------------------------------------------------------------------------

PrimFunc SunmmioLayoutInferencePass::ApplyToIR(PrimFunc f) {
  // Build SRAM layout_map and DRAM global_layout_map
  LayoutMap layout_map;
  LayoutMap final_global_layout_map;

  for (const auto &[buf, entry] : layout_entries_) {
    if (IsSunmmioSramScope(buf.scope())) {
      layout_map.Set(buf, entry.layout);
    } else if (buf.scope() == "global" || buf.scope().empty()) {
      final_global_layout_map.Set(buf, entry.layout);
    }
  }

  // Also include DRAM buffers from global_layout_map_ that may not be
  // in layout_entries_ (e.g., buffers without any TileOp usage)
  for (const auto &[buf, layout] : global_layout_map_) {
    if (!final_global_layout_map.count(buf)) {
      final_global_layout_map.Set(buf, layout);
    }
  }

  // Attach layout_map and global_layout_map to EVERY block in the IR.
  // This matches LayoutInference behavior: LowerTileOp looks up
  // layout_map from the nearest enclosing block, so inner blocks
  // (e.g., the kernel frame with T.annotate_layout) must also carry
  // the final inferred layouts. Without this, a stale annotate_layout
  // remnant on the inner block would shadow the outer block's map.
  class BlockAnnotator : public StmtExprMutator {
  public:
    BlockAnnotator(const LayoutMap &lm, const LayoutMap &glm)
        : layout_map_(lm), global_layout_map_(glm) {}

    Stmt VisitStmt_(const BlockNode *op) final {
      Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
      auto new_annotations = block->annotations;
      new_annotations.Set(attr::kLayoutMap, layout_map_);
      if (!global_layout_map_.empty()) {
        new_annotations.Set(attr::kGlobalLayoutMap, global_layout_map_);
      }
      if (!new_annotations.same_as(block->annotations)) {
        auto block_ptr = block.CopyOnWrite();
        block_ptr->annotations = new_annotations;
      }
      return block;
    }

  private:
    const LayoutMap &layout_map_;
    const LayoutMap &global_layout_map_;
  };

  BlockAnnotator annotator(layout_map, final_global_layout_map);
  auto new_body = annotator(f->body);
  auto new_func = PrimFunc(f->params, new_body, f->ret_type, f->buffer_map,
                           f->attrs, f->span);
  return new_func;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

bool SunmmioLayoutInferencePass::TryAssign(const Buffer &buffer,
                                           const Layout &layout,
                                           InferLevel level, int op_idx,
                                           bool immutable) {
  auto it = layout_entries_.find(buffer);
  if (it != layout_entries_.end()) {
    auto &existing = it->second;
    if (existing.is_immutable) {
      // Same layout is fine — silently deduplicate.
      if (IsSameLayout(existing.layout, layout, &analyzer_)) {
        return false;
      }
      // Different layout proposed for an immutable buffer.
      // At kStrict this is a hard conflict — the immutable source (DRAM
      // metadata or T.annotate_layout) contradicts what the operator
      // requires (e.g., Gemm hardware constraint).
      if (level >= InferLevel::kStrict) {
        ICHECK(false) << "SunmmioLayoutInference: buffer \"" << buffer->name
                      << "\" has immutable layout\n  "
                      << existing.layout->DebugOutput() << "\nbut op " << op_idx
                      << " requires\n  " << layout->DebugOutput();
      }
      // At kCommon/kFree: silently reject.  This typically arises from
      // BFS propagation deriving a layout from a kFree default on the
      // other side — it is not a real constraint.
      return false;
    }
    if (level > existing.level) {
      existing = {layout, level, op_idx, immutable};
      return true;
    }
    if (level == existing.level) {
      if (IsSameLayout(existing.layout, layout, &analyzer_)) {
        return false; // same layout, no change
      }
      // Same level, different layout — irreconcilable conflict.
      LOG(FATAL) << "SunmmioLayoutInference: layout conflict on buffer \""
                 << buffer->name << "\" at level " << static_cast<int>(level)
                 << "\n  existing (from op " << existing.source_op_idx
                 << "): " << existing.layout->DebugOutput()
                 << "\n  proposed (from op " << op_idx
                 << "): " << layout->DebugOutput();
      return false;
    }
    return false; // lower level, skip
  } else {
    layout_entries_[buffer] = {layout, level, op_idx, immutable};
    return true;
  }
}

LayoutInferArgs SunmmioLayoutInferencePass::BuildInferArgs() const {
  // Build current SRAM layout_map from layout_entries_
  LayoutMap current_layout_map;
  for (const auto &[buf, entry] : layout_entries_) {
    if (IsSunmmioSramScope(buf.scope())) {
      current_layout_map.Set(buf, entry.layout);
    }
  }

  return LayoutInferArgs{
      target_,
      Range(Integer(0), Integer(1)), // thread_bounds: 1 thread per core
      current_layout_map,
      const_cast<arith::Analyzer *>(&analyzer_),
      false,              // buffer_oob
      {},                 // buffer_remap
      let_var_to_expr_,   // let var bindings
      global_layout_map_, // DRAM layouts
  };
}

// ---------------------------------------------------------------------------
// Pass registration
// ---------------------------------------------------------------------------

tvm::transform::Pass SunmmioLayoutInference() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return SunmmioLayoutInferencePass::Run(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.SunmmioLayoutInference", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.SunmmioLayoutInference",
                        SunmmioLayoutInference);
}

} // namespace tl
} // namespace tvm
