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

#include "../layout/utils.h"
#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/parallel.h"
#include "../op/region.h"
#include "../op/utils.h"
#include "../target/utils.h"
#include "../tileview/tileview.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "arith/ir_visitor_with_analyzer.h"
#include "common/loop_fusion_utils.h"
#include "common/remap_buffer_rewriter.h"
#include "common/union_find.h"
#include "layout_reducer.h"
#include "loop_partition.h"
#include "loop_vectorize.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"
#include "tvm/node/cast.h"
#include "tvm/tir/buffer.h"
#include "tvm/tir/stmt.h"
#include "tvm/tir/var.h"

namespace tvm {
namespace tl {

using namespace tir;

class InferSramScopePass : public arith::IRMutatorWithAnalyzer {
public:
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

    // collect remap info when replace_flag = false
    substituter.replace_flag = false;
    fptr->body = substituter.VisitStmt(f->body);

    substituter.InferUnspecifiedBuffer();

    fptr->body =
        RemapBufferRewriter::Substitute(fptr->body, substituter.buffer_remap_);

    // do remap when replace_flag = true
    substituter.replace_flag = true;
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;
  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    BlockRealize block_realize =
        Downcast<BlockRealize>(StmtExprMutator::VisitStmt_(op));
    Block block = block_realize->block;
    // remap shared buffers to rsram buffers by default
    if (!replace_flag) {
      Array<Buffer> alloc_buffers = block->alloc_buffers;
      for (auto buffer : alloc_buffers) {
        if ((buffer.scope() == "shared") || (buffer.scope() == "shared.dyn")) {
          buffers_to_infer.insert(buffer);
        } else if ((buffer.scope() != "shared.asram") &&
                   (buffer.scope() != "shared.wsram") &&
                   (buffer.scope() != "shared.rsram")) {
          // sram type has validated in GEMM node
          ICHECK(0) << "Invalid scope " << buffer.scope() << " of " << buffer
                    << " in Sunmmio.";
        }
      }
      return block_realize;
    }

    if (buffer_remap_.empty()) {
      return block_realize;
    }

    // do block attributes remap
    if (block->annotations.count(attr::kLayoutMap)) {
      auto map = block->annotations.Get(attr::kLayoutMap)
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
      block.CopyOnWrite()->annotations.Set(attr::kLayoutMap, new_map);
    }

    if (block->annotations.count(attr::kTileViewMap)) {
      auto map = block->annotations.Get(attr::kTileViewMap)
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
      block.CopyOnWrite()->annotations.Set(attr::kTileViewMap, new_map);
    }

    // do block->alloc_buffers remap
    Array<Buffer> alloc_buffers = block->alloc_buffers;

    // remove the buffers
    alloc_buffers.MutateByApply([this](Buffer buf) {
      if (buffer_remap_.find(buf) != buffer_remap_.end()) {
        return buffer_remap_.at(buf);
      }
      return buf;
    });

    if (!alloc_buffers.same_as(block->alloc_buffers)) {
      block.CopyOnWrite()->alloc_buffers = alloc_buffers;
    }
    block_realize.CopyOnWrite()->block = block;

    return block_realize;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    // collect remap info in gemm nodes
    if (!replace_flag) {
      if (const auto *call = op->value.as<CallNode>()) {
        if (call->op.same_as(Op::Get("tl.tileop.gemm")) ||
            call->op.same_as(Op::Get("tl.tileop.gemm_py"))) {
          auto aRegion_ = NormalizeToBufferRegion(call->args[0]);
          auto bRegion_ = NormalizeToBufferRegion(call->args[1]);
          auto cRegion_ = NormalizeToBufferRegion(call->args[2]);

          auto buffer = aRegion_->buffer;
          if ((buffer.scope() == "shared") ||
              (buffer.scope() == "shared.dyn")) {
            if (buffer_remap_.count(buffer)) {
              ICHECK(buffer_remap_[buffer].scope() == "shared.asram")
                  << "Infer scope shared.asram of " << buffer
                  << " in GEMM Sunmmio, but scope "
                  << buffer_remap_[buffer].scope() << " has been inferred for "
                  << buffer << ".";
            } else {
              auto remap_buffer =
                  makeBufferWithScope(aRegion_->buffer, "shared.asram");
              buffer_remap_.Set(buffer, remap_buffer);
            }
          } else if (buffer.scope() != "shared.asram") {
            // incorrect specification
            ICHECK(0) << "Specify invalid scope " << buffer.scope() << " of "
                      << buffer << " in GEMM Sunmmio.";
          }

          buffer = bRegion_->buffer;
          if ((buffer.scope() == "shared") ||
              (buffer.scope() == "shared.dyn")) {
            if (buffer_remap_.count(buffer)) {
              ICHECK(buffer_remap_[buffer].scope() == "shared.wsram")
                  << "Infer scope shared.wsram of " << buffer
                  << " in GEMM Sunmmio, but scope "
                  << buffer_remap_[buffer].scope() << " has been inferred for "
                  << buffer << ".";
            } else {
              auto remap_buffer =
                  makeBufferWithScope(bRegion_->buffer, "shared.wsram");
              buffer_remap_.Set(buffer, remap_buffer);
            }
          } else if (buffer.scope() != "shared.wsram") {
            // incorrect specification
            ICHECK(0) << "Specify invalid scope " << buffer.scope() << " of "
                      << buffer << " in GEMM Sunmmio.";
          }

          buffer = cRegion_->buffer;
          if ((buffer.scope() == "shared") ||
              (buffer.scope() == "shared.dyn")) {
            if (buffer_remap_.count(buffer)) {
              ICHECK(buffer_remap_[buffer].scope() == "shared.rsram")
                  << "Infer scope shared.rsram of " << buffer
                  << " in GEMM Sunmmio, but scope "
                  << buffer_remap_[buffer].scope() << " has been inferred for "
                  << buffer << ".";
            } else {
              auto remap_buffer =
                  makeBufferWithScope(cRegion_->buffer, "shared.rsram");
              buffer_remap_.Set(buffer, remap_buffer);
            }
          } else if (buffer.scope() != "shared.rsram") {
            // incorrect specification
            ICHECK(0) << "Specify invalid scope " << buffer.scope() << " of "
                      << buffer << " in GEMM Sunmmio.";
          }
        }
      }
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto load = Downcast<BufferLoad>(IRMutatorWithAnalyzer::VisitExpr_(op));
    if (!replace_flag) {
      return load;
    }
    auto buffer = load->buffer;
    if (buffer_remap_.count(buffer)) {
      auto new_buffer = buffer_remap_[load->buffer];
      std::string type_key = op->GetTypeKey();
      return BufferLoad(new_buffer, load->indices);
    } else if (var_remap_.count(buffer->data)) {
      auto new_buffer = Buffer(
          var_remap_[buffer->data], buffer->dtype, buffer->shape,
          buffer->strides, buffer->elem_offset, buffer->name,
          buffer->data_alignment, buffer->offset_factor, buffer->buffer_type);
      return BufferLoad(new_buffer, load->indices);
    }
    auto expr = StmtExprMutator::VisitExpr_(op);
    return expr;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto store = Downcast<BufferStore>(IRMutatorWithAnalyzer::VisitStmt_(op));
    if (!replace_flag) {
      return store;
    }
    auto buffer = store->buffer;
    if (buffer_remap_.count(buffer)) {
      auto new_buffer = buffer_remap_[store->buffer];
      return BufferStore(new_buffer, store->value, store->indices);
    } else if (var_remap_.count(buffer->data)) {
      auto new_buffer = Buffer(
          var_remap_[buffer->data], buffer->dtype, buffer->shape,
          buffer->strides, buffer->elem_offset, buffer->name,
          buffer->data_alignment, buffer->offset_factor, buffer->buffer_type);
      return BufferStore(new_buffer, store->value, store->indices);
    }
    return store;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (!replace_flag)
      return StmtExprMutator::VisitExpr_(op);
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_EQ(op->args.size(), 5U);
      Var buffer_data = Downcast<Var>(op->args[1]);
      if (!var_remap_.count(buffer_data)) {
        return StmtExprMutator::VisitExpr_(op);
      }
      Var new_data = var_remap_[buffer_data];
      return Call(
          op->dtype, op->op,
          {op->args[0], new_data, op->args[2], op->args[3], op->args[4]});
    }
    auto expr = StmtExprMutator::VisitExpr_(op);
    return expr;
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    Var var = tvm::ffi::GetRef<Var>(op);
    if (!replace_flag) {
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
        auto remap_buffer = makeBufferWithScope(buffer, "shared.rsram");
        buffer_remap_.Set(buffer, remap_buffer);
      }
    }
  }

  Map<Buffer, Buffer> buffer_remap_;
  Map<Var, Var> var_remap_;

  std::set<Buffer> buffers_to_infer;

  bool replace_flag = false;
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
