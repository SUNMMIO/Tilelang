/*!
 * \file legalize_sunmmio_datapath.cc
 * \brief Legalize unsupported Sunmmio data-transfer paths.
 *
 * Rewrites global -> shared.asram transfers by inserting an RSRAM staging
 * step.  Works uniformly for any data-transfer tileop (copy, broadcast,
 * put, allgather).
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
   * For any data-transfer op (copy, broadcast, put, allgather) with a
   * global -> shared.asram path, the pass:
   * 1. Allocates a compact shared.rsram staging buffer.
   * 2. Inserts a copy: global -> shared.rsram.
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

  Buffer CreateStageBuffer(const Buffer &src, const Array<Range> &src_range) {
    ICHECK(!alloc_buffer_stack_.empty())
        << "LegalizeSunmmioDataPath expects data-transfer ops to appear "
           "inside a block.";
    Array<Range> compact_range = MakeCompactRegionForStage(src_range);
    std::string name = src->name + "_rsram_stage";
    if (temp_buffer_counter_ != 0) {
      name += "_" + std::to_string(temp_buffer_counter_);
    }
    ++temp_buffer_counter_;
    Buffer temp = MakeCompactBufferWithScope(src, compact_range,
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

    if (src_br->buffer.scope() != "global" ||
        dst_br->buffer.scope() != kSunmmioScopeASRAM) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    // Create an RSRAM staging buffer matching the source region shape.
    Buffer staging = CreateStageBuffer(src_br->buffer, src_br->region);
    Array<Range> staging_range = MakeCompactRegionForStage(src_br->region);

    // 1. Copy: global src -> RSRAM staging buffer.
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
