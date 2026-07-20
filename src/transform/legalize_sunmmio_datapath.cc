/*!
 * \file legalize_sunmmio_datapath.cc
 * \brief Legalize unsupported Sunmmio data-transfer paths.
 *
 * Stages unsupported copy and communication paths through RSRAM.
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

class LegalizeSunmmioDataPathPass : public arith::IRMutatorWithAnalyzer {
public:
  LegalizeSunmmioDataPathPass(arith::Analyzer *analyzer, Target target)
      : arith::IRMutatorWithAnalyzer(analyzer), target_(std::move(target)) {}

  /** Legalize unsupported Sunmmio data paths before tile-op lowering. */
  static PrimFunc Substitute(PrimFunc f) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsSunmmio(target.value())) {
      return f;
    }

    arith::Analyzer analyzer;
    LegalizeSunmmioDataPathPass rewriter(&analyzer, target.value());
    auto *fptr = f.CopyOnWrite();
    fptr->body = rewriter.VisitStmt(f->body);
    return f;
  }

private:
  BufferRegion CreateStageRegion(const Buffer &stage_template,
                                 const Array<Range> &region) {
    ICHECK(!alloc_buffer_stack_.empty())
        << "LegalizeSunmmioDataPath expects data-transfer ops to appear "
           "inside a block.";
    std::string name = stage_template->name + "_rsram_stage";
    if (temp_buffer_counter_ != 0) {
      name += "_" + std::to_string(temp_buffer_counter_);
    }
    ++temp_buffer_counter_;
    Buffer stage =
        MakeCompactBufferLike(stage_template, region, kSunmmioScopeRSRAM, name);
    alloc_buffer_stack_.back().push_back(stage);
    return BufferRegion(stage, MakeCompactRegion(region));
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
    const auto *call_node = op->value.as<CallNode>();
    if (call_node == nullptr) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    Call call = tvm::ffi::GetRef<Call>(call_node);
    bool is_copy = call->op.same_as(Copy::Get());
    bool is_communication = IsCommunicationOp(call);
    if (!is_copy && !is_communication) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    // All data-transfer ops have src region in args[0] and dst region in
    // args[1].
    BufferRegion src_br = NormalizeToBufferRegion(call->args[0]);
    BufferRegion dst_br = NormalizeToBufferRegion(call->args[1]);

    const Buffer &src = src_br->buffer;
    const Buffer &dst = dst_br->buffer;
    CommunicationDirections communication_directions =
        CommunicationDirections::kNone;
    if (is_copy && SupportsSunmmioDirectCopy(target_, src.scope(), src->dtype,
                                             dst.scope(), dst->dtype)) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    if (!is_copy) {
      communication_directions =
          GetCommunicationSourceDirections(call, target_);
      if (SupportsSunmmioDirectCommunication(target_, communication_directions,
                                             src.scope(), src->dtype,
                                             dst.scope(), dst->dtype)) {
        return IRMutatorWithAnalyzer::VisitStmt_(op);
      }
    }

    // Staging preserves the source dtype unless an RSRAM copy needs a cast.
    Buffer stage_template = src;
    if (is_copy && src.scope() == kSunmmioScopeRSRAM) {
      ICHECK(src->dtype != dst->dtype)
          << "Unsupported copy from " << src.scope() << " to " << dst.scope()
          << " of Sunmmio target.";
      stage_template = dst;
    } else if (!is_copy) {
      ICHECK(src.scope() != kSunmmioScopeRSRAM)
          << "Unsupported communication path from " << src.scope() << " to "
          << dst.scope() << " of Sunmmio target.";
    }

    BufferRegion staging = CreateStageRegion(stage_template, src_br->region);
    ICHECK(SupportsSunmmioDirectCopy(target_, src.scope(), src->dtype,
                                     staging->buffer.scope(),
                                     staging->buffer->dtype))
        << "Unsupported copy from " << src.scope() << " to " << dst.scope()
        << " of Sunmmio target.";
    if (!is_copy) {
      ICHECK(SupportsSunmmioDirectCommunication(
          target_, communication_directions, staging->buffer.scope(),
          staging->buffer->dtype, dst.scope(), dst->dtype))
          << "Unsupported communication path from " << src.scope() << " to "
          << dst.scope() << " of Sunmmio target.";
    }

    PrimExpr src_region = MakeRegionExpr(src, src_br->region, 1);
    PrimExpr staging_write =
        MakeRegionExpr(staging->buffer, staging->region, 2);
    Map<String, ObjectRef> copy_annotations =
        is_copy ? call->annotations : Map<String, ObjectRef>();
    Stmt copy_stmt =
        Evaluate(Call(DataType::Handle(), Copy::Get(),
                      {src_region, staging_write}, copy_annotations));

    Array<PrimExpr> new_args;
    new_args.push_back(MakeRegionExpr(staging->buffer, staging->region, 1));
    for (size_t i = 1; i < call->args.size(); i++) {
      new_args.push_back(call->args[i]);
    }
    Stmt rewritten = Evaluate(
        Call(call->dtype, Downcast<Op>(call->op), new_args, call->annotations));
    if (is_copy) {
      rewritten = VisitStmt(rewritten);
    }

    return SeqStmt::Flatten(Array<Stmt>{copy_stmt, rewritten});
  }

  std::vector<Array<Buffer>> alloc_buffer_stack_;
  int temp_buffer_counter_ = 0;
  Target target_;
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
