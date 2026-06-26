/*!
 * \file legalize_sunmmio_datapath.cc
 * \brief Legalize unsupported Sunmmio data-transfer paths.
 *
 * Rewrites transfers that have no direct datapath by inserting an RSRAM staging
 * step: global -> shared.asram, and casting RSRAM -> shared.asram/wsram (the
 * cast runs on the tile unit in RSRAM, then a same-dtype DMA reaches the
 * operand SRAM).  Works for any data-transfer tileop (copy, broadcast, put,
 * allgather).
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/node/cast.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <string>
#include <utility>
#include <vector>

#include "../op/comm.h"
#include "../op/copy.h"
#include "../op/utils.h"
#include "../target/sunmmio_utils.h"
#include "../target/utils.h"
#include "arith/ir_mutator_with_analyzer.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

/** Builds a compact 0-based region that preserves the original extents. */
Array<Range> MakeCompactRegionForStage(const Array<Range> &region) {
  Array<Range> compact_region;
  compact_region.reserve(region.size());
  for (const Range &range : region) {
    compact_region.push_back(Range::FromMinExtent(0, range->extent));
  }
  return compact_region;
}

/** Creates a compact temporary buffer whose shape matches the staged region. */
Buffer MakeCompactBufferWithScope(const Buffer &buffer,
                                  const Array<Range> &region,
                                  const std::string &scope,
                                  const std::string &name) {
  const auto *ptr_type = buffer->data->type_annotation.as<PointerTypeNode>();
  ICHECK(ptr_type != nullptr);
  Type new_type = PointerType(ptr_type->element_type, scope);
  Var new_var = Var(name, new_type);
  Array<PrimExpr> shape;
  shape.reserve(region.size());
  for (const Range &range : region) {
    shape.push_back(range->extent);
  }
  return Buffer(new_var, buffer->dtype, shape, {}, Integer(0), name,
                buffer->data_alignment, buffer->offset_factor,
                buffer->buffer_type);
}

class LegalizeSunmmioDataPathPass : public arith::IRMutatorWithAnalyzer {
public:
  explicit LegalizeSunmmioDataPathPass(arith::Analyzer *analyzer)
      : arith::IRMutatorWithAnalyzer(analyzer) {}

  /**
   * @brief Legalize unsupported Sunmmio data-transfer paths before lowering.
   *
   * For a data-transfer op (copy, broadcast, put, allgather) whose transfer has
   * no direct datapath -- global -> shared.asram, or a casting RSRAM ->
   * shared.asram/wsram -- the pass stages through a compact shared.rsram
   * buffer:
   * 1. Allocates the staging buffer (dst dtype on the cast path, else src
   * dtype).
   * 2. Inserts a copy: src -> staging (the cast, when dtypes differ).
   * 3. Rewrites the original op's source to use the staging buffer.
   */
  static PrimFunc Substitute(PrimFunc f) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsSunmmio(target.value())) {
      return f;
    }

    arith::Analyzer analyzer;
    LegalizeSunmmioDataPathPass rewriter(&analyzer);
    auto *fptr = f.CopyOnWrite();
    fptr->body = rewriter.VisitStmt(f->body);
    return f;
  }

private:
  /** Returns true for tileops that transfer data between src and dst regions.
   */
  static bool IsDataTransferOp(const CallNode *call) {
    return call->op.same_as(Copy::Get()) ||
           call->op.same_as(BroadcastOp::Get()) ||
           call->op.same_as(PutOp::Get()) ||
           call->op.same_as(AllgatherOp::Get());
  }

  // `proto` supplies the staging buffer's dtype, element type and name;
  // `region` its shape. They are the same buffer for a plain scope-route stage,
  // but differ for a cast stage (proto = dst, so the staging buffer takes the
  // dst dtype and the prepended copy performs the cast).
  Buffer CreateStageBuffer(const Buffer &proto, const Array<Range> &region) {
    ICHECK(!alloc_buffer_stack_.empty())
        << "LegalizeSunmmioDataPath expects data-transfer ops to appear "
           "inside a block.";
    Array<Range> compact_range = MakeCompactRegionForStage(region);
    std::string name = proto->name + "_rsram_stage";
    if (temp_buffer_counter_ != 0) {
      name += "_" + std::to_string(temp_buffer_counter_);
    }
    ++temp_buffer_counter_;
    Buffer temp = MakeCompactBufferWithScope(proto, compact_range,
                                             kSunmmioScopeRSRAM, name);
    alloc_buffer_stack_.back().push_back(temp);
    return temp;
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    alloc_buffer_stack_.emplace_back();
    Block block = Downcast<Block>(IRMutatorWithAnalyzer::VisitStmt_(op));
    Array<Buffer> new_alloc_buffers = alloc_buffer_stack_.back();
    alloc_buffer_stack_.pop_back();

    if (new_alloc_buffers.empty()) {
      return block;
    }

    Array<Buffer> alloc_buffers = block->alloc_buffers;
    for (const Buffer &buffer : new_alloc_buffers) {
      alloc_buffers.push_back(buffer);
    }

    auto block_ptr = block.CopyOnWrite();
    block_ptr->alloc_buffers = std::move(alloc_buffers);
    return block;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call == nullptr || !IsDataTransferOp(call)) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    // All data-transfer ops have src region in args[0] and dst region in
    // args[1].
    BufferRegion src_br = NormalizeToBufferRegion(call->args[0]);
    BufferRegion dst_br = NormalizeToBufferRegion(call->args[1]);

    const std::string src_scope = src_br->buffer.scope();
    const std::string dst_scope = dst_br->buffer.scope();
    bool dst_is_operand =
        dst_scope == kSunmmioScopeASRAM || dst_scope == kSunmmioScopeWSRAM;

    // global -> ASRAM has no direct DMA path; stage through RSRAM (same dtype).
    bool stage_global =
        src_scope == "global" && dst_scope == kSunmmioScopeASRAM;
    // Casting RSRAM -> ASRAM/WSRAM: the cast must run on the tile unit
    // (RSRAM -> RSRAM), so stage through an RSRAM buffer of the dst dtype.
    bool stage_cast = src_scope == kSunmmioScopeRSRAM && dst_is_operand &&
                      src_br->buffer->dtype != dst_br->buffer->dtype;
    if (!stage_global && !stage_cast) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    // Cast path stages with the dst dtype (so the prepended copy casts);
    // scope-route path stages with the src dtype.
    const Buffer &proto = stage_cast ? dst_br->buffer : src_br->buffer;
    Buffer staging = CreateStageBuffer(proto, src_br->region);
    Array<Range> staging_range = MakeCompactRegionForStage(src_br->region);

    // 1. Copy: src -> RSRAM staging buffer (casts when src/dst dtypes differ).
    PrimExpr src_region = MakeRegionExpr(src_br->buffer, src_br->region, 1);
    PrimExpr staging_write = MakeRegionExpr(staging, staging_range, 2);
    Map<String, ObjectRef> copy_annotations = call->op.same_as(Copy::Get())
                                                  ? call->annotations
                                                  : Map<String, ObjectRef>();
    Stmt copy_stmt =
        Evaluate(Call(DataType::Handle(), Copy::Get(),
                      {src_region, staging_write}, copy_annotations));

    // 2. Rewrite original op: replace src (args[0]) with staging region.
    PrimExpr staging_read = MakeRegionExpr(staging, staging_range, 1);
    Array<PrimExpr> new_args;
    new_args.push_back(staging_read);
    for (size_t i = 1; i < call->args.size(); i++) {
      new_args.push_back(call->args[i]);
    }
    Stmt rewritten = Evaluate(
        Call(call->dtype, Downcast<Op>(call->op), new_args, call->annotations));

    Array<Stmt> seq{copy_stmt, rewritten};
    return SeqStmt::Flatten(seq);
  }

  std::vector<Array<Buffer>> alloc_buffer_stack_;
  int temp_buffer_counter_ = 0;
};

Pass LegalizeSunmmioDataPath() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LegalizeSunmmioDataPathPass::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LegalizeSunmmioDataPath", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LegalizeSunmmioDataPath",
                        LegalizeSunmmioDataPath);
}

} // namespace tl
} // namespace tvm
