/*!
 * \file legalize_sunmmio_batch_gemm_views.cc
 * \brief Materialize partial Batch GEMM operands before SRAM scope inference.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/node/cast.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <array>
#include <string>
#include <utility>
#include <vector>

#include "../op/copy.h"
#include "../op/gemm.h"
#include "../op/gemm_py.h"
#include "../op/utils.h"
#include "../target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

bool NeedsCompactBatchView(const BufferRegion &region,
                           arith::Analyzer *analyzer) {
  if (region->region.size() != 3)
    return false;
  const Range &batch = region->region[0];
  return !analyzer->CanProveEqual(batch->min, make_zero(batch->min.dtype())) ||
         !analyzer->CanProveEqual(batch->extent, region->buffer->shape[0]);
}

// A4E cannot copy out of ASRAM; compact while operands still have shared scope.
class CompactBatchViewRewriter : public StmtExprMutator {
public:
  explicit CompactBatchViewRewriter(arith::Analyzer *analyzer)
      : analyzer_(analyzer) {}

private:
  arith::Analyzer *analyzer_;
  int scratch_index_{0};
  std::vector<std::vector<Buffer>> block_scratch_;

  Stmt VisitStmt_(const BlockNode *op) final {
    block_scratch_.emplace_back();
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    std::vector<Buffer> scratch = std::move(block_scratch_.back());
    block_scratch_.pop_back();
    if (scratch.empty())
      return block;

    Array<Buffer> alloc_buffers = block->alloc_buffers;
    for (const Buffer &buffer : scratch)
      alloc_buffers.push_back(buffer);
    block.CopyOnWrite()->alloc_buffers = std::move(alloc_buffers);
    return block;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (!call ||
        (!call->op.same_as(Gemm::Get()) && !call->op.same_as(GemmPy::Get())))
      return StmtExprMutator::VisitStmt_(op);
    ICHECK_GE(call->args.size(), 10U);

    std::array<BufferRegion, 3> regions = {
        NormalizeToBufferRegion(call->args[0]),
        NormalizeToBufferRegion(call->args[1]),
        NormalizeToBufferRegion(call->args[2]),
    };
    bool needs_fallback = false;
    for (const BufferRegion &region : regions)
      needs_fallback |= NeedsCompactBatchView(region, analyzer_);
    if (!needs_fallback)
      return tvm::ffi::GetRef<Stmt>(op);

    ICHECK(!block_scratch_.empty())
        << "Sunmmio partial Batch GEMM must be nested in a block";
    Array<PrimExpr> new_args = call->args;
    Array<Stmt> before;
    Array<Stmt> after;
    static constexpr const char *kNames[] = {"a", "b", "c"};

    for (size_t i = 0; i < regions.size(); ++i) {
      const BufferRegion &region = regions[i];
      if (!NeedsCompactBatchView(region, analyzer_))
        continue;

      const Range &batch = region->region[0];
      const auto *batch_min = batch->min.as<IntImmNode>();
      const auto *batch_extent = batch->extent.as<IntImmNode>();
      ICHECK(batch_min && batch_min->value >= 0 && batch_extent &&
             batch_extent->value > 0)
          << "Sunmmio Batch GEMM partial view requires a non-negative static "
             "batch min and positive static extent, got min "
          << batch->min << ", extent " << batch->extent;
      ICHECK(analyzer_->CanProve(batch->min + batch->extent <=
                                 region->buffer->shape[0]))
          << "Sunmmio Batch GEMM partial view exceeds buffer axis 0: min "
          << batch->min << ", extent " << batch->extent << ", shape "
          << region->buffer->shape;

      std::string name = region->buffer->name + "_batch_" + kNames[i] +
                         "_compact_" + std::to_string(scratch_index_++);
      Buffer compact = MakeCompactBufferLike(region->buffer, region->region,
                                             region->buffer.scope(), name);
      Array<Range> compact_ranges = MakeCompactRegion(region->region);
      block_scratch_.back().push_back(compact);
      new_args.Set(i, MakeRegionExpr(compact, compact_ranges,
                                     i == 2 ? /*rw=*/3 : /*read=*/1));

      PrimExpr parent_read =
          MakeRegionExpr(region->buffer, region->region, /*read=*/1);
      PrimExpr compact_write =
          MakeRegionExpr(compact, compact_ranges, /*write=*/2);
      PrimExpr compact_read =
          MakeRegionExpr(compact, compact_ranges, /*read=*/1);
      PrimExpr parent_write =
          MakeRegionExpr(region->buffer, region->region, /*write=*/2);

      if (i < 2 || !is_one(call->args[9])) {
        before.push_back(Evaluate(Call(DataType::Handle(), Copy::Get(),
                                       {parent_read, compact_write}, {})));
      }
      if (i == 2) {
        after.push_back(Evaluate(Call(DataType::Handle(), Copy::Get(),
                                      {compact_read, parent_write}, {})));
      }
    }

    Call compact_gemm(call->dtype, Downcast<Op>(call->op), new_args,
                      call->annotations);
    Array<Stmt> sequence;
    for (const Stmt &stmt : before)
      sequence.push_back(stmt);
    sequence.push_back(Evaluate(compact_gemm));
    for (const Stmt &stmt : after)
      sequence.push_back(stmt);
    return SeqStmt::Flatten(sequence);
  }
};

PrimFunc Run(PrimFunc f) {
  auto target = f->GetAttr<Target>(tvm::attr::kTarget);
  if (!target.defined() || !TargetIsSunmmio(target.value()))
    return f;

  arith::Analyzer analyzer;
  auto *fptr = f.CopyOnWrite();
  fptr->body = CompactBatchViewRewriter(&analyzer)(f->body);
  return f;
}

} // namespace

tvm::transform::Pass LegalizeSunmmioBatchGemmViews() {
  auto pass_func = [](PrimFunc f, IRModule, tvm::transform::PassContext) {
    return Run(std::move(f));
  };
  return tir::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.LegalizeSunmmioBatchGemmViews", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LegalizeSunmmioBatchGemmViews",
                        LegalizeSunmmioBatchGemmViews);
}

} // namespace tl
} // namespace tvm
