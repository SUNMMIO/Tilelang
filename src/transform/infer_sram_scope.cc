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

#include "arith/ir_mutator_with_analyzer.h"
#include "arith/ir_visitor_with_analyzer.h"
#include "common/loop_fusion_utils.h"
#include "common/loop_parallel_transform_utils.h"
#include "common/union_find.h"
#include "layout_reducer.h"
#include "loop_partition.h"
#include "loop_vectorize.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;

static Buffer makeBufferWithScope(const Buffer &buffer, std::string scope) {
  const auto *ptr_type =
      TVM_TYPE_AS(buffer->data->type_annotation, PointerTypeNode);
  Type new_type = PointerType(ptr_type->element_type, scope);
  Var new_var = Var(buffer->data->name_hint, new_type);
  return Buffer(new_var, buffer->dtype, buffer->shape, {}, buffer->elem_offset,
                buffer->name, buffer->data_alignment, buffer->offset_factor,
                buffer->buffer_type);
}

/*!
 * \brief A class that rewrites buffer references in a statement based on a
 * given buffer remapping.
 *
 * This class is used to update buffer references in a statement after buffer
 * transformations have been applied. It specifically handles the remapping of
 * padding annotations.
 */
class RemapBufferRewriter : public arith::IRMutatorWithAnalyzer {
public:
  /*!
   * \brief Substitute buffer references in a statement based on a given buffer
   * remapping. \param stmt The statement to rewrite. \param buffer_remap A map
   * from old buffers to new buffers. \return The rewritten statement.
   */
  static Stmt Substitute(const Stmt &stmt, Map<Buffer, Buffer> buffer_remap) {
    arith::Analyzer analyzer;
    RemapBufferRewriter substituter(&analyzer);
    substituter.buffer_remap_ = std::move(buffer_remap);
    return substituter.VisitStmt(stmt);
  }

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  Stmt VisitStmt_(const BlockNode *op) final {
    if (op->annotations.count(attr::kSafeValueMap)) {
      return RewritePaddingMap(op);
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  /*!
   * \brief Rewrite the padding map annotation of a block.
   * \param op The block node to rewrite.
   * \return The rewritten block.
   */
  Stmt RewritePaddingMap(const BlockNode *op) {
    auto safe_value_map = op->annotations.Get(attr::kSafeValueMap);
    if (!safe_value_map) {
      LOG(FATAL) << "Padding map annotation is missing";
    }

    Map<Var, Var> var_remap = CreateVarRemap();
    Map<Var, PrimExpr> new_safe_value_map = RemapPaddingMap(
        Downcast<Map<Var, PrimExpr>>(safe_value_map.value()), var_remap);

    LOG(INFO) << var_remap;
    LOG(INFO) << new_safe_value_map;
    auto block = Downcast<Block>(IRMutatorWithAnalyzer::VisitStmt_(op));
    auto block_ptr = block.CopyOnWrite();
    block_ptr->annotations.Set(attr::kSafeValueMap, new_safe_value_map);
    return block;
  }

  /*!
   * \brief Create a mapping from old variables to new variables based on buffer
   * remapping. \return A map from old variables to new variables.
   */
  Map<Var, Var> CreateVarRemap() const {
    Map<Var, Var> var_remap;
    for (const auto &[buffer, buffer_remap] : buffer_remap_) {
      var_remap.Set(buffer->data, buffer_remap->data);
    }
    return var_remap;
  }

  /*!
   * \brief Remap the padding map using the variable remapping.
   * \param safe_value_map The original padding map.
   * \param var_remap The variable remapping.
   * \return The remapped padding map.
   */
  Map<Var, PrimExpr> RemapPaddingMap(const Map<Var, PrimExpr> &safe_value_map,
                                     const Map<Var, Var> &var_remap) const {
    Map<Var, PrimExpr> new_safe_value_map;
    for (const auto &[var, padding] : safe_value_map) {
      if (var_remap.count(var)) {
        new_safe_value_map.Set(var_remap.at(var), padding);
      } else {
        new_safe_value_map.Set(var, padding);
      }
    }
    return new_safe_value_map;
  }

  Map<Buffer, Buffer> buffer_remap_;
};

class InferSramScopePass : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    InferSramScopePass substituter(&analyzer);

    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined())
        << "InferSramScopePass: Require the target attribute";
    if (!TargetIsSunmmio(target.value()))
      return f;
    // Sunmmio specified pass

    auto *fptr = f.CopyOnWrite();

    // collect remap info when replace_flag = false
    substituter.replace_flag = false;
    fptr->body = substituter.VisitStmt(f->body);

    fptr->body =
        RemapBufferRewriter::Substitute(fptr->body, substituter.buffer_remap_);

    // do remap when replace_flag = true
    substituter.replace_flag = true;
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;
  Stmt VisitStmt_(const BlockNode *op) final {
    // remap shared buffers to rsram buffers by default
    if (!replace_flag) {
      Block block = tvm::ffi::GetRef<Block>(op);
      Array<Buffer> alloc_buffers = op->alloc_buffers;
      for (auto buffer : alloc_buffers) {
        buffer_remap_.Set(buffer, makeBufferWithScope(buffer, "shared.rsram"));
        const auto *ptr_type =
            TVM_TYPE_AS(buffer->data->type_annotation, PointerTypeNode);
        Type new_type = PointerType(ptr_type->element_type, "shared.rsram");
        Var new_var = Var(buffer->data->name_hint, new_type);
        var_remap_.Set(buffer->data, new_var);
      }
      return StmtExprMutator::VisitStmt_(op);
    }

    // do op->alloc_buffers remap
    Block block = tvm::ffi::GetRef<Block>(op);
    Array<Buffer> alloc_buffers = op->alloc_buffers;

    if (buffer_remap_.empty()) {
      return StmtExprMutator::VisitStmt_(op);
    }

    // remove the buffers
    alloc_buffers.MutateByApply([this](Buffer buf) {
      if (buffer_remap_.find(buf) != buffer_remap_.end()) {
        return buffer_remap_.at(buf);
      }
      return buf;
    });

    if (!alloc_buffers.same_as(op->alloc_buffers)) {
      block.CopyOnWrite()->alloc_buffers = alloc_buffers;
      return StmtExprMutator::VisitStmt_(block.get());
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
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
          {
            auto remap_buffer =
                makeBufferWithScope(aRegion_->buffer, "shared.asram");
            const auto *ptr_type =
                TVM_TYPE_AS(buffer->data->type_annotation, PointerTypeNode);
            Type new_type = PointerType(ptr_type->element_type, "shared.asram");
            Var new_var = Var(buffer->data->name_hint, new_type);
            buffer_remap_.Set(buffer, remap_buffer);
            var_remap_.Set(buffer->data, new_var);
          }

          buffer = bRegion_->buffer;
          {
            auto remap_buffer =
                makeBufferWithScope(bRegion_->buffer, "shared.wsram");
            const auto *ptr_type =
                TVM_TYPE_AS(buffer->data->type_annotation, PointerTypeNode);
            Type new_type = PointerType(ptr_type->element_type, "shared.wsram");
            Var new_var = Var(buffer->data->name_hint, new_type);
            buffer_remap_.Set(buffer, remap_buffer);
            var_remap_.Set(buffer->data, new_var);
          }

          buffer = cRegion_->buffer;
          {
            auto remap_buffer =
                makeBufferWithScope(cRegion_->buffer, "shared.rsram");
            const auto *ptr_type =
                TVM_TYPE_AS(buffer->data->type_annotation, PointerTypeNode);
            Type new_type = PointerType(ptr_type->element_type, "shared.rsram");
            Var new_var = Var(buffer->data->name_hint, new_type);
            buffer_remap_.Set(buffer, remap_buffer);
            var_remap_.Set(buffer->data, new_var);
          }
        }
      }
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto load = Downcast<BufferLoad>(IRMutatorWithAnalyzer::VisitExpr_(op));
    if (!replace_flag)
      return load;
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
      LOG(INFO) << load;
      return BufferLoad(new_buffer, load->indices);
    }
    auto expr = StmtExprMutator::VisitExpr_(op);
    return expr;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto store = Downcast<BufferStore>(IRMutatorWithAnalyzer::VisitStmt_(op));
    if (!replace_flag)
      return store;
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
    if (!replace_flag)
      return var;
    if (var_remap_.count(var)) {
      return var_remap_[var];
    }
    return var;
  }

  Map<Buffer, Buffer> buffer_remap_;
  Map<Var, Var> var_remap_;
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map_;
  Map<Var, Buffer> buffer_data_to_buffer_;
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
