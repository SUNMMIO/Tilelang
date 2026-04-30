/*!
 * \file legalize_sunmmio_copy_path.cc
 * \brief Legalize unsupported Sunmmio copy paths.
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

#include "../op/copy.h"
#include "../op/utils.h"
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

class LegalizeSunmmioCopyPathPass : public arith::IRMutatorWithAnalyzer {
public:
  explicit LegalizeSunmmioCopyPathPass(arith::Analyzer *analyzer)
      : arith::IRMutatorWithAnalyzer(analyzer) {}

  /**
   * @brief Legalize unsupported Sunmmio copy paths before lowering.
   *
   * The pass inserts a compact shared.rsram temporary buffer into the nearest
   * enclosing block allocation list, then rewrites one `tl.tileop.copy`
   * statement into two copies:
   * 1. global -> shared.rsram
   * 2. shared.rsram -> shared.asram
   */
  static PrimFunc Substitute(PrimFunc f) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsSunmmio(target.value())) {
      return f;
    }

    arith::Analyzer analyzer;
    LegalizeSunmmioCopyPathPass rewriter(&analyzer);
    auto *fptr = f.CopyOnWrite();
    fptr->body = rewriter.VisitStmt(f->body);
    return f;
  }

private:
  bool ShouldSplit(const Copy &copy) const {
    return copy->src.scope() == "global" && copy->dst.scope() == "shared.asram";
  }

  void CheckMatchingLogicalShape(const Copy &copy) {
    size_t src_dim = 0;
    size_t dst_dim = 0;

    while (src_dim < copy->src_range.size() &&
           dst_dim < copy->dst_range.size()) {
      while (src_dim < copy->src_range.size() &&
             is_one(copy->src_range[src_dim]->extent)) {
        ++src_dim;
      }
      while (dst_dim < copy->dst_range.size() &&
             is_one(copy->dst_range[dst_dim]->extent)) {
        ++dst_dim;
      }

      if (src_dim >= copy->src_range.size() ||
          dst_dim >= copy->dst_range.size()) {
        break;
      }

      ICHECK(analyzer_->CanProveEqual(copy->src_range[src_dim]->extent,
                                      copy->dst_range[dst_dim]->extent))
          << "Sunmmio staged copy requires matching logical src/dst extents, "
          << "but got " << copy->src_range[src_dim]->extent << " vs. "
          << copy->dst_range[dst_dim]->extent << ".";
      ++src_dim;
      ++dst_dim;
    }

    while (src_dim < copy->src_range.size()) {
      ICHECK(is_one(copy->src_range[src_dim]->extent))
          << "Sunmmio staged copy found an unmatched non-unit src extent at "
             "dim "
          << src_dim << ": " << copy->src_range[src_dim]->extent;
      ++src_dim;
    }
    while (dst_dim < copy->dst_range.size()) {
      ICHECK(is_one(copy->dst_range[dst_dim]->extent))
          << "Sunmmio staged copy found an unmatched non-unit dst extent at "
             "dim "
          << dst_dim << ": " << copy->dst_range[dst_dim]->extent;
      ++dst_dim;
    }
  }

  Buffer CreateStageBuffer(const Copy &copy) {
    ICHECK(!alloc_buffer_stack_.empty())
        << "LegalizeSunmmioCopyPath expects copies to appear inside a block.";
    Array<Range> temp_region = MakeCompactRegionForStage(copy->dst_range);
    std::string name = copy->dst->name + "_rsram_stage";
    if (temp_buffer_counter_ != 0) {
      name += "_" + std::to_string(temp_buffer_counter_);
    }
    ++temp_buffer_counter_;
    Buffer temp = MakeCompactBufferWithScope(copy->dst, temp_region,
                                             "shared.rsram", name);
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
    if (call == nullptr || !call->op.same_as(Copy::Get())) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    Copy copy = Downcast<Copy>(ParseOperator(tvm::ffi::GetRef<Stmt>(op)));
    if (!ShouldSplit(copy)) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    CheckMatchingLogicalShape(copy);
    Buffer temp = CreateStageBuffer(copy);
    Array<Range> temp_range = MakeCompactRegionForStage(copy->dst_range);

    PrimExpr src_region = MakeRegionExpr(copy->src, copy->src_range, 1);
    PrimExpr temp_write_region = MakeRegionExpr(temp, temp_range, 2);
    PrimExpr temp_read_region = MakeRegionExpr(temp, temp_range, 1);
    PrimExpr dst_region = MakeRegionExpr(copy->dst, copy->dst_range, 2);

    Array<Stmt> seq{
        Evaluate(Call(DataType::Handle(), Copy::Get(),
                      {src_region, temp_write_region}, copy->annotations)),
        Evaluate(Call(DataType::Handle(), Copy::Get(),
                      {temp_read_region, dst_region}, copy->annotations))};
    return SeqStmt::Flatten(seq);
  }

  std::vector<Array<Buffer>> alloc_buffer_stack_;
  int temp_buffer_counter_ = 0;
};

Pass LegalizeSunmmioCopyPath() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LegalizeSunmmioCopyPathPass::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LegalizeSunmmioCopyPath", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LegalizeSunmmioCopyPath",
                        LegalizeSunmmioCopyPath);
}

} // namespace tl
} // namespace tvm
