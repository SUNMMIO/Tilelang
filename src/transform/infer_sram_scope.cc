/*!
 * \file infer_sram_scope.cc
 * \brief Infer shared memory SRAM scope
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include <algorithm>
#include <deque>
#include <memory>
#include <queue>
#include <unordered_map>

#include "../layout/utils.h"
#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/parallel.h"
#include "../op/region.h"
#include "../op/utils.h"
#include "../target/sunmmio_utils.h"
#include "../target/utils.h"
#include "../tileview/tileview.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "arith/ir_visitor_with_analyzer.h"
#include "common/attr.h"
#include "common/loop_fusion_utils.h"
#include "common/remap_buffer_rewriter.h"
#include "common/union_find.h"
#include "layout_reducer.h"
#include "loop_partition.h"
#include "loop_vectorize.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"
#include "tvm/ffi/container/array.h"
#include "tvm/ir/expr.h"
#include "tvm/node/cast.h"
#include "tvm/runtime/data_type.h"
#include "tvm/runtime/logging.h"
#include "tvm/tir/buffer.h"
#include "tvm/tir/stmt.h"
#include "tvm/tir/var.h"

namespace tvm {
namespace tl {

using namespace tir;

/** Creates a compact 0-based region that preserves the original extents. */
Array<Range> MakeCompactRegion(const Array<Range> &region) {
  Array<Range> compact_region;
  compact_region.reserve(region.size());
  for (const Range &range : region) {
    compact_region.push_back(Range::FromMinExtent(0, range->extent));
  }
  return compact_region;
}

/** Creates a temporary buffer that keeps the original buffer shape. */
Buffer makeNewBufferWithScope(const Buffer &buffer, std::string scope) {
  const auto *ptr_type =
      TVM_TYPE_AS(buffer->data->type_annotation, PointerTypeNode);
  Type new_type = PointerType(ptr_type->element_type, scope);
  Var new_var = Var(buffer->data->name_hint, new_type);
  return Buffer(new_var, buffer->dtype, buffer->shape, {}, buffer->elem_offset,
                buffer->name, buffer->data_alignment, buffer->offset_factor,
                buffer->buffer_type);
}

/** Creates a compact temporary buffer whose shape matches the region extents.
 */
Buffer makeNewCompactBufferWithScope(const Buffer &buffer,
                                     const Array<Range> &region,
                                     std::string scope) {
  const auto *ptr_type =
      TVM_TYPE_AS(buffer->data->type_annotation, PointerTypeNode);
  Type new_type = PointerType(ptr_type->element_type, scope);
  Var new_var = Var(buffer->data->name_hint, new_type);
  Array<PrimExpr> shape;
  shape.reserve(region.size());
  for (const Range &range : region) {
    shape.push_back(range->extent);
  }
  return Buffer(new_var, buffer->dtype, shape, {}, Integer(0), buffer->name,
                buffer->data_alignment, buffer->offset_factor,
                buffer->buffer_type);
}

struct BufferSourceInfo {
  Buffer src_buffer;
  Array<Range> src_region;
  Array<Range> dst_region;
};

class InferSramScopePass : public arith::IRMutatorWithAnalyzer {
public:
  /**
   * @brief Infer Sunmmio SRAM scopes and rewrite buffers in three explicit
   * phases.
   *
   * Phase 1 collects scope information from GEMM sites and records which shared
   * buffers still need a default shared.rsram scope.
   * Phase 2 resolves GEMM scope conflicts by inserting compact temporary copies
   * where a buffer's assigned scope does not match the scope required by a
   * specific GEMM operand.
   * Phase 3 applies the finalized remap to block
   * metadata and remaining buffer uses so the whole IR becomes internally
   * consistent.
   */
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    InferSramScopePass substituter(&analyzer);

    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined())
        << "InferSramScopePass: Require the target attribute";

    // Sunmmio specified pass
    if (!TargetIsSunmmio(target.value()))
      return f;

    auto *fptr = f.CopyOnWrite();

    // Phase 1 collects buffer usage info and initial scope decisions from the
    // original IR without rewriting it.
    substituter.phase_ = Phase::kCollectInfo;
    fptr->body = substituter.VisitStmt(f->body);

    substituter.InferUnspecifiedBuffer();

    // Phase 2 inserts GEMM-local compact copies on the original IR. This must
    // happen before the global remap because the compact regions are derived
    // from the original operand coordinates.
    substituter.phase_ = Phase::kResolveConflicts;
    fptr->body = substituter.VisitStmt(f->body);

    fptr->body =
        RemapBufferRewriter::Substitute(fptr->body, substituter.buffer_remap_);

    // Phase 3 rewrites block metadata and any remaining buffer references to
    // match the remapped buffers chosen in phases 1 and 2.
    substituter.phase_ = Phase::kRewriteScope;
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;
  enum class Phase {
    kCollectInfo,
    kResolveConflicts,
    kRewriteScope,
  };

  class LocalBufferUseCollector : public StmtExprVisitor {
  public:
    /** @brief Collect buffers referenced directly in the current block body. */
    static std::unordered_set<const VarNode *> Collect(const Stmt &stmt) {
      LocalBufferUseCollector collector;
      collector.VisitStmt(stmt);
      return collector.used_buffer_vars_;
    }

  private:
    void VisitStmt_(const BufferStoreNode *op) final {
      used_buffer_vars_.insert(op->buffer->data.get());
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitExpr_(const BufferLoadNode *op) final {
      used_buffer_vars_.insert(op->buffer->data.get());
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const VarNode *op) final { used_buffer_vars_.insert(op); }

    void VisitExpr_(const CallNode *op) final {
      if (op->op.same_as(RegionOp::Get())) {
        BufferRegion region =
            NormalizeToBufferRegion(tvm::ffi::GetRef<Call>(op));
        used_buffer_vars_.insert(region->buffer->data.get());
      }
      StmtExprVisitor::VisitExpr_(op);
    }

    std::unordered_set<const VarNode *> used_buffer_vars_;
  };

  static bool CanCopyDirectly(const std::string &src_scope,
                              const std::string &dst_scope) {
    return src_scope == kSunmmioScopeRSRAM &&
           (dst_scope == kSunmmioScopeASRAM ||
            dst_scope == kSunmmioScopeWSRAM || dst_scope == kSunmmioScopeRSRAM);
  }

  bool InInfoCollectionPhase() const { return phase_ == Phase::kCollectInfo; }
  bool InConflictResolutionPhase() const {
    return phase_ == Phase::kResolveConflicts;
  }
  bool InScopeRewritePhase() const { return phase_ == Phase::kRewriteScope; }

  Buffer RemapBuffer(const Buffer &buffer) {
    if (buffer_remap_.count(buffer)) {
      return buffer_remap_[buffer];
    }
    if (var_remap_.count(buffer->data)) {
      return Buffer(var_remap_[buffer->data], buffer->dtype, buffer->shape,
                    buffer->strides, buffer->elem_offset, buffer->name,
                    buffer->data_alignment, buffer->offset_factor,
                    buffer->buffer_type);
    }
    return buffer;
  }

  BufferRegion RemapBufferRegion(const BufferRegion &region) {
    return BufferRegion(RemapBuffer(region->buffer), region->region);
  }

  MatchBufferRegion RemapMatchBufferRegion(const MatchBufferRegion &match) {
    return MatchBufferRegion(RemapBuffer(match->buffer),
                             RemapBufferRegion(match->source));
  }

  void CollectOriginalSharedAllocations(const Block &block) {
    for (const Buffer &buffer : block->alloc_buffers) {
      if ((buffer.scope() == "shared") || (buffer.scope() == "shared.dyn")) {
        buffers_to_infer.insert(buffer);
        original_alloc_buffers_.insert(buffer);
      } else if (!IsSunmmioSramScope(buffer.scope())) {
        // SRAM scopes should already have been validated at the GEMM sites.
        ICHECK(0) << "Invalid scope " << buffer.scope() << " of " << buffer
                  << " in Sunmmio.";
      }
    }
  }

  void RewriteBlockAnnotations(Block *block_ptr) {
    if ((*block_ptr)->annotations.count(attr::kLayoutMap)) {
      auto map = (*block_ptr)
                     ->annotations.Get(attr::kLayoutMap)
                     ->as<Map<Var, Layout>>()
                     .value();
      Map<Var, Layout> new_map;
      for (const auto &[var, layout] : map) {
        if (var_remap_.count(var)) {
          new_map.Set(var_remap_[var], layout);
        } else {
          new_map.Set(var, layout);
        }
      }
      (*block_ptr).CopyOnWrite()->annotations.Set(attr::kLayoutMap, new_map);
    }

    if ((*block_ptr)->annotations.count(attr::kTileViewMap)) {
      auto map = (*block_ptr)
                     ->annotations.Get(attr::kTileViewMap)
                     ->as<Map<Var, TileView>>()
                     .value();
      Map<Var, TileView> new_map;
      for (const auto &[var, tileView] : map) {
        if (var_remap_.count(var)) {
          new_map.Set(var_remap_[var], tileView);
        } else {
          new_map.Set(var, tileView);
        }
      }
      (*block_ptr).CopyOnWrite()->annotations.Set(attr::kTileViewMap, new_map);
    }
  }

  void RewriteBlockBuffers(Block *block_ptr) {
    Array<BufferRegion> reads;
    reads.reserve((*block_ptr)->reads.size());
    for (const BufferRegion &region : (*block_ptr)->reads) {
      reads.push_back(RemapBufferRegion(region));
    }

    Array<BufferRegion> writes;
    writes.reserve((*block_ptr)->writes.size());
    for (const BufferRegion &region : (*block_ptr)->writes) {
      writes.push_back(RemapBufferRegion(region));
    }

    Array<MatchBufferRegion> match_buffers;
    match_buffers.reserve((*block_ptr)->match_buffers.size());
    for (const MatchBufferRegion &match : (*block_ptr)->match_buffers) {
      match_buffers.push_back(RemapMatchBufferRegion(match));
    }

    /** @brief Remap block-local allocations and add only block-local
     * temporaries. */
    Array<Buffer> alloc_buffers;
    for (const Buffer &buf : (*block_ptr)->alloc_buffers) {
      if (buffer_remap_.find(buf) != buffer_remap_.end()) {
        alloc_buffers.push_back(buffer_remap_.at(buf));
      } else {
        alloc_buffers.push_back(buf);
      }
    }

    std::unordered_map<const VarNode *, Buffer> remapped_buffers_by_var;
    for (const auto &it : buffer_remap_) {
      if (original_alloc_buffers_.count(it.first)) {
        continue;
      }
      remapped_buffers_by_var[it.second->data.get()] = it.second;
    }

    std::unordered_set<const VarNode *> used_buffer_vars =
        LocalBufferUseCollector::Collect((*block_ptr)->body);
    for (const VarNode *buf_var : used_buffer_vars) {
      auto it = remapped_buffers_by_var.find(buf_var);
      if (it == remapped_buffers_by_var.end()) {
        continue;
      }
      bool exists = std::find(alloc_buffers.begin(), alloc_buffers.end(),
                              it->second) != alloc_buffers.end();
      if (!exists) {
        alloc_buffers.push_back(it->second);
      }
    }

    if (!alloc_buffers.same_as((*block_ptr)->alloc_buffers)) {
      (*block_ptr).CopyOnWrite()->alloc_buffers = alloc_buffers;
    }
    if (!reads.same_as((*block_ptr)->reads)) {
      (*block_ptr).CopyOnWrite()->reads = reads;
    }
    if (!writes.same_as((*block_ptr)->writes)) {
      (*block_ptr).CopyOnWrite()->writes = writes;
    }
    if (!match_buffers.same_as((*block_ptr)->match_buffers)) {
      (*block_ptr).CopyOnWrite()->match_buffers = match_buffers;
    }
  }

  void CollectOperandScope(const BufferRegion &region,
                           const std::string &expected_scope) {
    Buffer buffer = region->buffer;
    if ((buffer.scope() == "shared") || (buffer.scope() == "shared.dyn")) {
      if (!buffer_remap_.count(buffer)) {
        buffer_remap_.Set(buffer, makeBufferWithScope(buffer, expected_scope));
      }
      return;
    }

    if (buffer.scope() != expected_scope) {
      ICHECK(0) << "Specify invalid scope " << buffer.scope() << " of "
                << buffer << " in GEMM Sunmmio.";
    }
  }

  PrimExpr MakeConflictCopySource(const Buffer &buffer,
                                  const Array<Range> &requested_region,
                                  const std::string &current_scope,
                                  const std::string &expected_scope,
                                  const char *operand_name) {
    if (CanCopyDirectly(current_scope, expected_scope)) {
      return MakeRegionExpr(buffer, requested_region, /*access_mask=*/1);
    }

    auto maybe_source_region =
        TryTranslateSourceRegion(buffer, requested_region);
    ICHECK(maybe_source_region.defined())
        << "Cannot resolve source for conflicting " << operand_name
        << " operand " << buffer << ".";
    return MakeRegionExpr(maybe_source_region.value()->buffer,
                          maybe_source_region.value()->region,
                          /*access_mask=*/1);
  }

  void
  ResolveOperandConflict(const BufferRegion &region, const char *operand_name,
                         const char *operand_key,
                         const std::string &expected_scope, Array<Stmt> *seq,
                         std::unordered_map<std::string, PrimExpr> *new_args) {
    Buffer buffer = region->buffer;
    if ((buffer.scope() == "shared") || (buffer.scope() == "shared.dyn")) {
      if (!buffer_remap_.count(buffer)) {
        return;
      }
      std::string current_scope = buffer_remap_[buffer].scope();
      if (current_scope != expected_scope) {
        auto transfer_region = MakeCompactRegion(region->region);
        auto transfer_buffer = makeNewCompactBufferWithScope(
            buffer, transfer_region, current_scope);
        PrimExpr src_region =
            MakeConflictCopySource(buffer, region->region, current_scope,
                                   expected_scope, operand_name);
        PrimExpr dst_region =
            MakeRegionExpr(transfer_buffer, transfer_region, /*access_mask=*/2);
        Call copy_call =
            Call(DataType::Handle(), Copy::Get(), {src_region, dst_region}, {});
        seq->push_back(Evaluate(copy_call));
        buffer_remap_.Set(transfer_buffer,
                          makeBufferWithScope(transfer_buffer, expected_scope));
        new_args->insert(
            {operand_key, MakeRegionExpr(transfer_buffer, transfer_region, 1)});
      }
      return;
    }

    if (buffer.scope() != expected_scope) {
      ICHECK(0) << "Specify invalid scope " << buffer.scope() << " of "
                << buffer << " in GEMM Sunmmio.";
    }
  }

  void CollectGemmScopes(const CallNode *call) {
    CollectOperandScope(NormalizeToBufferRegion(call->args[0]),
                        kSunmmioScopeASRAM);
    CollectOperandScope(NormalizeToBufferRegion(call->args[1]),
                        kSunmmioScopeWSRAM);
    CollectOperandScope(NormalizeToBufferRegion(call->args[2]),
                        kSunmmioScopeRSRAM);
  }

  Stmt ResolveGemmConflicts(const CallNode *call) {
    auto a_region = NormalizeToBufferRegion(call->args[0]);
    auto b_region = NormalizeToBufferRegion(call->args[1]);
    auto c_region = NormalizeToBufferRegion(call->args[2]);

    Array<Stmt> seq;
    std::unordered_map<std::string, PrimExpr> new_args;

    ResolveOperandConflict(a_region, "A", "A", kSunmmioScopeASRAM, &seq,
                           &new_args);
    ResolveOperandConflict(b_region, "B", "B", kSunmmioScopeWSRAM, &seq,
                           &new_args);
    ResolveOperandConflict(c_region, "C", "C", kSunmmioScopeRSRAM, &seq,
                           &new_args);

    if (seq.empty() || new_args.empty()) {
      return Evaluate(tvm::ffi::GetRef<Call>(call));
    }

    ICHECK_EQ(call->args.size(), 19U) << "GEMM should have 19 parameters.";
    Array<PrimExpr> new_gemm_args;
    new_gemm_args.push_back(new_args.count("A") ? new_args.at("A")
                                                : call->args[0]);
    new_gemm_args.push_back(new_args.count("B") ? new_args.at("B")
                                                : call->args[1]);
    new_gemm_args.push_back(new_args.count("C") ? new_args.at("C")
                                                : call->args[2]);
    for (size_t i = 3; i < call->args.size(); ++i) {
      new_gemm_args.push_back(call->args[i]);
    }
    seq.push_back(Evaluate(Call(DataType::Handle(), call->op, new_gemm_args)));
    return SeqStmt::Flatten(seq);
  }

  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    BlockRealize block_realize =
        Downcast<BlockRealize>(StmtExprMutator::VisitStmt_(op));
    Block block = block_realize->block;
    if (InInfoCollectionPhase()) {
      // Phase 1 only records which original block allocations still need a
      // default shared.rsram scope if GEMM did not assign them explicitly.
      CollectOriginalSharedAllocations(block);
      return block_realize;
    }

    if (InConflictResolutionPhase()) {
      return block_realize;
    }

    if (buffer_remap_.empty()) {
      return block_realize;
    }

    RewriteBlockAnnotations(&block);
    RewriteBlockBuffers(&block);
    block_realize.CopyOnWrite()->block = block;

    return block_realize;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (InInfoCollectionPhase()) {
      if (const auto *call = op->value.as<CallNode>()) {
        if (call->op.same_as(Op::Get("tl.tileop.gemm")) ||
            call->op.same_as(Op::Get("tl.tileop.gemm_py"))) {
          CollectGemmScopes(call);
          return tvm::ffi::GetRef<Stmt>(op);
        }
      }
    }
    if (InConflictResolutionPhase()) {
      if (const auto *call = op->value.as<CallNode>()) {
        if (call->op.same_as(Op::Get("tl.tileop.gemm")) ||
            call->op.same_as(Op::Get("tl.tileop.gemm_py"))) {
          /**
           * @brief Phase 2 only resolves GEMM scope conflicts.
           *
           * The compact copy is inserted here, before the global remap, so the
           * copy source region can still be derived from the original buffer
           * coordinates collected in phase 1.
           */
          return ResolveGemmConflicts(call);
        }
      }
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto load = Downcast<BufferLoad>(IRMutatorWithAnalyzer::VisitExpr_(op));
    if (InInfoCollectionPhase()) {
      auto buffer = load->buffer;
      if ((buffer.scope() == "shared") || (buffer.scope() == "shared.dyn")) {
        if (buffer_remap_.count(buffer)) {
          ICHECK(buffer_remap_[buffer].scope() == kSunmmioScopeRSRAM)
              << "Buffer " << buffer
              << " used in BufferLoad should be shared.rsram.";
        } else {
          buffers_to_infer.insert(buffer);
        }
      } else if ((buffer.scope() != kSunmmioScopeRSRAM) &&
                 (buffer.scope() != "global")) {
        ICHECK(0) << "Invalid scope " << buffer.scope() << " of " << buffer
                  << " in Sunmmio.";
      }
      return load;
    }
    auto buffer = load->buffer;
    if (buffer_remap_.count(buffer)) {
      return BufferLoad(buffer_remap_[load->buffer], load->indices);
    } else if (var_remap_.count(buffer->data)) {
      return BufferLoad(RemapBuffer(buffer), load->indices);
    }
    auto expr = StmtExprMutator::VisitExpr_(op);
    return expr;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto store = Downcast<BufferStore>(IRMutatorWithAnalyzer::VisitStmt_(op));
    if (InInfoCollectionPhase()) {
      // A plain BufferStore to a shared buffer is a Tile-/Scalar-unit output.
      // Both units operate on RSRAM, so the destination must be RSRAM -- the
      // write-side counterpart of the BufferLoad rule below. The scope is
      // bound eagerly (not deferred to InferUnspecifiedBuffer) so it wins
      // over a later GEMM operand claim: a buffer that is both a Tile-unit
      // output and a GEMM A/B operand is left in RSRAM, and the resulting
      // buffer-vs-GEMM scope mismatch is resolved by a staging copy in
      // Phase 2 (ResolveGemmConflicts).
      auto buffer = store->buffer;
      if ((buffer.scope() == "shared") || (buffer.scope() == "shared.dyn")) {
        if (buffer_remap_.count(buffer)) {
          // Already bound: RSRAM is consistent; ASRAM/WSRAM means a GEMM
          // operand use was collected first, i.e. the consuming GEMM
          // precedes the Tile-unit write that produces its operand.
          ICHECK(buffer_remap_[buffer].scope() == kSunmmioScopeRSRAM)
              << "InferSramScope: buffer " << buffer
              << " is written by a Tile-unit op and must be "
              << kSunmmioScopeRSRAM << ", but it was already bound to "
              << buffer_remap_[buffer].scope()
              << " by an earlier GEMM operand use. The Tile-unit write that "
                 "produces a GEMM operand must precede that GEMM.";
        } else {
          buffer_remap_.Set(buffer,
                            makeBufferWithScope(buffer, kSunmmioScopeRSRAM));
        }
      }
      return store;
    }
    if (!InScopeRewritePhase()) {
      return store;
    }
    auto buffer = store->buffer;
    if (buffer_remap_.count(buffer)) {
      return BufferStore(buffer_remap_[store->buffer], store->value,
                         store->indices);
    } else if (var_remap_.count(buffer->data)) {
      return BufferStore(RemapBuffer(buffer), store->value, store->indices);
    }
    return store;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (InInfoCollectionPhase()) {
      auto tile_op = ParseOperator(tvm::ffi::GetRef<Call>(op));
      if (auto *copy = tile_op.as<CopyNode>()) {
        if (copy->dst.defined() && copy->src.defined() &&
            copy->dst.scope() != "global") {
          buffer_source_map_[copy->dst] =
              BufferSourceInfo{copy->src, copy->src_range, copy->dst_range};
        }
        return tvm::ffi::GetRef<Call>(op);
      }
    }
    auto expr = StmtExprMutator::VisitExpr_(op);
    return expr;
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    Var var = tvm::ffi::GetRef<Var>(op);
    if (!InScopeRewritePhase()) {
      return std::move(var);
    }
    if (var_remap_.count(var)) {
      auto new_var = var_remap_[var];
      return std::move(new_var);
    }
    return std::move(var);
  }

  Buffer makeBufferWithScope(const Buffer &buffer, std::string scope) {
    const auto *ptr_type =
        TVM_TYPE_AS(buffer->data->type_annotation, PointerTypeNode);
    Var new_var;
    if (var_remap_.count(buffer->data)) {
      new_var = var_remap_[buffer->data];
    } else {
      Type new_type = PointerType(ptr_type->element_type, scope);
      new_var = Var(buffer->data->name_hint, new_type);
      var_remap_.Set(buffer->data, new_var);
    }
    return Buffer(new_var, buffer->dtype, buffer->shape, {},
                  buffer->elem_offset, buffer->name, buffer->data_alignment,
                  buffer->offset_factor, buffer->buffer_type);
  }

  void InferUnspecifiedBuffer() {
    for (const auto &buffer : buffers_to_infer) {
      if (!buffer_remap_.count(buffer)) {
        auto remap_buffer = makeBufferWithScope(buffer, kSunmmioScopeRSRAM);
        buffer_remap_.Set(buffer, remap_buffer);
      }
    }
  }

  Map<Buffer, Buffer> buffer_remap_;
  Map<Var, Var> var_remap_;
  std::unordered_map<Buffer, BufferSourceInfo, ObjectPtrHash, ObjectPtrEqual>
      buffer_source_map_;

  std::set<Buffer> buffers_to_infer;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>
      original_alloc_buffers_;

  Phase phase_ = Phase::kCollectInfo;

  Optional<BufferRegion>
  TryTranslateSourceRegion(const Buffer &buffer,
                           const Array<Range> &requested_region) {
    auto it = buffer_source_map_.find(buffer);
    if (it == buffer_source_map_.end()) {
      return Optional<BufferRegion>();
    }
    const BufferSourceInfo &info = it->second;
    if (info.dst_region.size() != requested_region.size() ||
        info.src_region.size() != requested_region.size()) {
      return Optional<BufferRegion>();
    }

    Array<Range> translated_region;
    translated_region.reserve(requested_region.size());
    for (size_t i = 0; i < requested_region.size(); ++i) {
      PrimExpr delta = analyzer_->Simplify(requested_region[i]->min -
                                           info.dst_region[i]->min);
      PrimExpr upper = analyzer_->Simplify(delta + requested_region[i]->extent);
      if (!analyzer_->CanProve(delta >= 0) ||
          !analyzer_->CanProve(upper <= info.dst_region[i]->extent)) {
        return Optional<BufferRegion>();
      }
      PrimExpr source_min =
          analyzer_->Simplify(info.src_region[i]->min + delta);
      translated_region.push_back(
          Range::FromMinExtent(source_min, requested_region[i]->extent));
    }
    return BufferRegion(info.src_buffer, translated_region);
  }
};

tvm::transform::Pass InferSramScope() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return InferSramScopePass::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InferSramScope", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InferSramScope", InferSramScope);
}

} // namespace tl
} // namespace tvm
