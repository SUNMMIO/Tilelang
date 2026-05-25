#include "../layout/utils.h"
#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/parallel.h"
#include "../op/region.h"
#include "../op/utils.h"
#include "../target/utils.h"
#include "../tileview/tileview.h"
#include "common/ast_traverser.h"
#include "common/loop_fusion_utils.h"
#include "common/remap_buffer_rewriter.h"
#include "common/sunmmio_pipeline_utils.h"
#include "tir/transforms/ir_utils.h"
#include "tvm/ir/expr.h"
#include "tvm/node/cast.h"
#include "tvm/node/structural_equal.h"
#include "tvm/runtime/logging.h"
#include "tvm/tir/function.h"
#include "tvm/tir/stmt.h"
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

using namespace tir;

struct LetWrapper {
  Var var;
  PrimExpr value;
};

class SunmmioMultiVersionBufferRewriter : public StmtExprMutator {
public:
  SunmmioMultiVersionBufferRewriter(const PrimFunc &f) {
    for (const auto &kv : f->buffer_map) {
      const Buffer &buffer = kv.second;
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
  }

  static Stmt Substitute(PrimFunc &f) {
    SunmmioMultiVersionBufferRewriter substituter(f);
    // collect used_buffers and iterations
    substituter.VisitStmt(f->body);
    substituter.replace_flag = true;

    for (auto &buffer : substituter.versioned_buffers_) {
      substituter.buffer_remap_.Set(
          buffer,
          substituter.makeMultiVersionBuffer(buffer, substituter.iterations_));
    }

    f.CopyOnWrite()->body =
        RemapBufferRewriter::Substitute(f->body, substituter.buffer_remap_);

    return substituter.VisitStmt(f->body);
  }

private:
  Buffer makeMultiVersionBuffer(const Buffer &buffer, int num_version) {
    const auto *ptr_type =
        TVM_TYPE_AS(buffer->data->type_annotation, PointerTypeNode);
    Var new_var;
    if (var_remap_.count(buffer->data)) {
      new_var = var_remap_[buffer->data];
    } else {
      Type new_type =
          PointerType(ptr_type->element_type, ptr_type->storage_scope);
      new_var = Var(buffer->data->name_hint, new_type);
      var_remap_.Set(buffer->data, new_var);
    }
    auto shape = buffer->shape;
    shape.insert(shape.begin(), num_version);
    return Buffer(new_var, buffer->dtype, shape, {}, buffer->elem_offset,
                  buffer->name, buffer->data_alignment, buffer->offset_factor,
                  buffer->buffer_type);
  }

  BufferRegion
  RewritePipelineBufferRegion(const BufferRegion &buffer_region) const {
    auto it = buffer_remap_.find(buffer_region->buffer);
    if (it != buffer_remap_.end()) {
      Region new_region = buffer_region->region;
      const Buffer &new_buffer = (*it).second;
      Range accessed_version = Range::FromMinExtent(0, 1);
      new_region.insert(new_region.begin(), accessed_version);
      return BufferRegion(new_buffer, new_region);
    }
    return buffer_region;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    auto versioned_buffers_anno = op->annotations.Get("versioned_buffers");
    auto used_buffers_anno = op->annotations.Get("used_buffers");
    auto iterations_anno = op->annotations.Get("iterations");
    if (versioned_buffers_anno && used_buffers_anno && iterations_anno) {
      Array<Buffer> versioned_buffers =
          Downcast<Array<Buffer>>(versioned_buffers_anno.value());
      int iterations = Downcast<int>(iterations_anno.value());
      if (!replace_flag) {
        versioned_buffers_ = versioned_buffers;
        iterations_ = iterations;
      } else {
        Array<Buffer> new_versioned_buffers;
        for (const Buffer &buffer : versioned_buffers) {
          if (buffer_remap_.count(buffer)) {
            new_versioned_buffers.push_back(buffer_remap_[buffer]);
          } else {
            new_versioned_buffers.push_back(buffer);
          }
        }
        loop.CopyOnWrite()->annotations.Set("versioned_buffers",
                                            new_versioned_buffers);
        Array<Buffer> used_buffers =
            Downcast<Array<Buffer>>(used_buffers_anno.value());
        Array<Buffer> new_used_buffers;
        for (const Buffer &buffer : used_buffers) {
          if (buffer_remap_.count(buffer)) {
            new_used_buffers.push_back(buffer_remap_[buffer]);
          } else {
            new_used_buffers.push_back(buffer);
          }
        }
        loop.CopyOnWrite()->annotations.Set("used_buffers", new_used_buffers);
      }
    }
    return loop;
  }

  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    BlockRealize block_realize =
        Downcast<BlockRealize>(StmtExprMutator::VisitStmt_(op));
    Block block = block_realize->block;
    if (!replace_flag) {
      for (const Buffer &alloc_buffer : block->alloc_buffers) {
        buffer_data_to_buffer_.Set(alloc_buffer->data, alloc_buffer);
      }
      return block_realize;
    }

    // do block attributes remap
    if (block->annotations.count(attr::kLayoutMap)) {
      auto map_anno = block->annotations.Get(attr::kLayoutMap);
      Map<Buffer, Layout> map = Downcast<Map<Buffer, Layout>>(map_anno.value());
      Map<Buffer, Layout> new_map;
      for (const auto &[buffer, layout] : map) {
        if (buffer_remap_.count(buffer)) {
          new_map.Set(buffer_remap_[buffer], layout);
        } else {
          new_map.Set(buffer, layout);
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

    block.CopyOnWrite()->reads.MutateByApply(
        [this](const BufferRegion &buffer_region) {
          return RewritePipelineBufferRegion(buffer_region);
        });
    block.CopyOnWrite()->writes.MutateByApply(
        [this](const BufferRegion &buffer_region) {
          return RewritePipelineBufferRegion(buffer_region);
        });

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

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    if (!replace_flag) {
      return load;
    }
    auto buffer = load->buffer;
    if (buffer_remap_.count(buffer)) {
      auto new_buffer = buffer_remap_[load->buffer];
      auto indices = load->indices;
      indices.insert(indices.begin(), 0);
      return BufferLoad(new_buffer, indices);
    }
    auto expr = StmtExprMutator::VisitExpr_(op);
    return expr;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    if (!replace_flag) {
      return store;
    }
    auto buffer = store->buffer;
    if (buffer_remap_.count(buffer)) {
      auto new_buffer = buffer_remap_[store->buffer];
      auto indices = store->indices;
      indices.insert(indices.begin(), 0);
      return BufferStore(new_buffer, store->value, indices);
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
    } else if (op->op.same_as(RegionOp::Get())) {
      RegionOp original_region(op->args);
      Buffer original_buffer = original_region->GetBuffer();

      if (!buffer_remap_.count(original_buffer)) {
        return StmtExprMutator::VisitExpr_(op);
      }

      Buffer new_buffer = buffer_remap_[original_buffer];
      Array<Range> new_ranges = original_region->GetRanges();
      new_ranges.insert(new_ranges.begin(), Range(0, 1));

      Array<PrimExpr> new_args;
      new_args.push_back(BufferLoad(new_buffer, [new_ranges]() {
        Array<PrimExpr> mins;
        for (auto r : new_ranges) {
          mins.push_back(r->min);
        }
        return mins;
      }()));
      new_args.push_back(original_region->GetAccessMask());
      for (auto r : new_ranges) {
        new_args.push_back(r->extent);
      }

      return Call(DataType::Handle(), RegionOp::Get(), new_args);
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

  Array<Buffer> versioned_buffers_;
  int iterations_ = -1;
  bool replace_flag = false;
  Map<Buffer, Buffer> buffer_remap_;
  Map<Var, Var> var_remap_;
  Map<Var, Buffer> buffer_data_to_buffer_;
};

class PipelineBodyRewriter : public StmtExprMutator {
public:
  PipelineBodyRewriter(Array<Buffer> used_buffers, For pipeline_loop) {
    used_buffers_ = used_buffers;
    pipeline_loop_ = std::move(pipeline_loop);
    for (auto it : used_buffers) {
      buffer_data_to_buffer_.Set(it->data, it);
    }
  }

  void set_current_version(int v) { current_version_ = v; }

  void set_loop_var_replacement(PrimExpr p) { replaced_loop_var_ = p; }

private:
  PrimExpr RewriteBufferAccess(const Call &call,
                               const std::vector<int> &arg_indices) {
    auto product = [](const Array<PrimExpr> &input) {
      return foldl(
          [](PrimExpr a, PrimExpr b, Span span) {
            return mul(std::move(a), std::move(b), std::move(span));
          },
          make_const(DataType::Int(32), 1), input);
    };
    Array<PrimExpr> new_args = call->args;
    for (int i : arg_indices) {
      // const Buffer &buffer =
      //     buffer_data_to_buffer_.at(Downcast<Var>(call->args[i]));
      // auto it = buffer_remap_.find(buffer);
      // if (it != buffer_remap_.end()) {
      //   const Buffer &new_buffer = (*it).second;
      //   const PrimExpr &old_index = call->args[i + 1];
      //   LOG(INFO) << old_index;
      //   PrimExpr offset;
      //   if (new_buffer->strides.empty()) {
      //     offset = product(buffer->shape);
      //   } else {
      //     offset = new_buffer->strides[0];
      //   }
      //   PrimExpr new_index =
      //       old_index +
      //       floormod(pipeline_loop_->loop_var, new_buffer->shape[0]) *
      //       offset;
      //   LOG(INFO) << new_index;
      //   new_args.Set(i + 1, new_index);
    }
    return Call(call->dtype, call->op, new_args, call->annotations, call->span);
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    BlockNode *n = block.CopyOnWrite();
    // n->reads.MutateByApply([this](const BufferRegion &buffer_region) {
    //   return RewritePipelineBufferRegion(buffer_region);
    // });
    // n->writes.MutateByApply([this](const BufferRegion &buffer_region) {
    //   return RewritePipelineBufferRegion(buffer_region);
    // });
    return block;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    bool count = false;
    for (auto it : used_buffers_) {
      if (StructuralEqual()(it, store->buffer))
        count = true;
    }
    if (!count) {
      return store;
    }
    auto *n = store.CopyOnWrite();
    n->indices.Set(0, current_version_);
    return store;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    bool count = false;
    for (auto it : used_buffers_) {
      if (StructuralEqual()(it, load->buffer))
        count = true;
    }
    if (!count) {
      return load;
    }
    auto *n = load.CopyOnWrite();
    n->indices.Set(0, current_version_);
    return load;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (call->op.same_as(builtin::tvm_access_ptr())) {
      return RewriteBufferAccess(call, {1});
    }
    return call;
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    Var var = Downcast<Var>(StmtExprMutator::VisitExpr_(op));
    if (ExprDeepEqual()(var, pipeline_loop_->loop_var)) {
      return replaced_loop_var_;
    }
    return var;
  }

  Array<Buffer> used_buffers_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  For pipeline_loop_;
  int current_version_ = 0;
  PrimExpr replaced_loop_var_;
};

class SunmmioPipelineInjector : public StmtExprMutator {
public:
  static Stmt Inject(const PrimFunc &func) {
    auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    SunmmioPipelineInjector injector(global_symbol, func);
    for (const auto &kv : func->buffer_map) {
      const Buffer &buffer = kv.second;
      injector.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    return injector(func->body);
  }

private:
  explicit SunmmioPipelineInjector(Optional<String> global_symbol,
                                   const PrimFunc &f)
      : global_symbol_(std::move(global_symbol)), traverser_(f) {
    traverser_.clear();
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // Step 1: Recursively rewrite the children first.
    For for_node = Downcast<For>(StmtExprMutator::VisitStmt_(op));

    auto iterations_anno = op->annotations.Get("iterations");
    auto used_buffers_anno = op->annotations.Get("used_buffers");
    auto versioned_buffers_anno = op->annotations.Get("versioned_buffers");
    auto prologue_orders_anno = op->annotations.Get("prologue_orders");
    auto body_orders_anno = op->annotations.Get("body_orders");
    auto epilogue_orders_anno = op->annotations.Get("epilogue_orders");

    if (!iterations_anno || !used_buffers_anno || !versioned_buffers_anno ||
        !prologue_orders_anno || !body_orders_anno || !epilogue_orders_anno) {
      return for_node;
    }

    // Step 2: Find the body and buffer allocations of the pipeline. The body
    // can be direct child of the for-loop. If the for-loop has BlockRealize as
    // its child, the pipeline body will be the child of the block.
    Stmt pipeline_body_root{nullptr};
    bool pipeline_body_from_block = false;
    Array<Buffer> pipeline_allocs;
    if (const auto *realize = for_node->body.as<BlockRealizeNode>()) {
      const auto &block = realize->block;
      for (const auto &buffer : block->alloc_buffers) {
        ICHECK(buffer->IsInstance<BufferNode>());
        buffer_data_to_buffer_.Set(buffer->data, buffer);
      }
      pipeline_body_root = block->body;
      pipeline_allocs = block->alloc_buffers;
      pipeline_body_from_block = true;
    } else {
      pipeline_body_root = for_node->body;
    }

    const SeqStmtNode *pipeline_body_seq = nullptr;
    std::vector<std::function<Stmt(Stmt)>> rewrap_fns;
    std::vector<LetWrapper> loop_var_let_wrappers;
    auto append_attr_wrapper = [&rewrap_fns](const AttrStmtNode *attr) {
      Any node = attr->node;
      String attr_key = attr->attr_key;
      PrimExpr value = attr->value;
      Span span = attr->span;
      rewrap_fns.emplace_back(
          [node = std::move(node), attr_key = std::move(attr_key),
           value = std::move(value), span](Stmt body) -> Stmt {
            return AttrStmt(node, attr_key, value, body, span);
          });
    };
    {
      Stmt current = pipeline_body_root;
      while (true) {
        if (const auto *seq_stmt = current.as<SeqStmtNode>()) {
          pipeline_body_seq = seq_stmt;
          break;
        }
        if (const auto *if_then_else = current.as<IfThenElseNode>()) {
          ICHECK(!if_then_else->else_case.defined())
              << "InjectSoftwarePipeline: Can't handle the body of the loop "
                 "because the IfThenElse node has an else branch";
          PrimExpr condition = if_then_else->condition;
          Span span = if_then_else->span;
          rewrap_fns.emplace_back(
              [condition = std::move(condition), span](Stmt body) -> Stmt {
                return IfThenElse(condition, body, Stmt(), span);
              });
          current = if_then_else->then_case;
          continue;
        }
        if (const auto *let_stmt = current.as<LetStmtNode>()) {
          // If this Let value uses the pipeline loop var, record it and push
          // inside each rewritten block later so the loop var can be
          // substituted with the correct per-iteration index. Otherwise, keep
          // it as a normal wrapper.
          bool uses_loop_var = UsesVar(
              let_stmt->value,
              [v = op->loop_var.get()](const VarNode *vn) { return vn == v; });
          if (uses_loop_var) {
            loop_var_let_wrappers.push_back({let_stmt->var, let_stmt->value});
          } else {
            Var var = let_stmt->var;
            PrimExpr value = let_stmt->value;
            Span span = let_stmt->span;
            rewrap_fns.emplace_back([var = std::move(var),
                                     value = std::move(value),
                                     span](Stmt body) -> Stmt {
              return LetStmt(var, value, body, span);
            });
          }
          current = let_stmt->body;
          continue;
        }
        if (const auto *attr = current.as<AttrStmtNode>()) {
          append_attr_wrapper(attr);
          current = attr->body;
          continue;
        }
        LOG(FATAL) << "ValueError: The body of the software pipeline should be "
                   << "SeqStmt, got " << current->GetTypeKey();
      }
    }
    ICHECK(pipeline_body_seq != nullptr);

    // Step 3: Rewrite the body of loop.
    int iterations = Downcast<IntImm>(iterations_anno.value())->value;
    Array<String> prologue_orders =
        Downcast<Array<String>>(prologue_orders_anno.value());
    Array<String> body_orders =
        Downcast<Array<String>>(body_orders_anno.value());
    Array<String> epilogue_orders =
        Downcast<Array<String>>(epilogue_orders_anno.value());
    Array<Buffer> versioned_buffers =
        Downcast<Array<Buffer>>(versioned_buffers_anno.value());
    Array<Buffer> used_buffers =
        Downcast<Array<Buffer>>(used_buffers_anno.value());
    for (auto it : used_buffers) {
      pipeline_allocs.push_back(it);
    }

    auto rewriter = PipelineBodyRewriter(versioned_buffers, for_node);
    Array<Stmt> for_body;
    // Step 3.1: Rewrite prologue
    for (const auto &order_str : prologue_orders) {
      int iter = name2iter(order_str);
      int id = name2id(order_str);
      Stmt stmt = pipeline_body_seq->seq[id];
      rewriter.set_current_version(iter);
      PrimExpr replaced_loop_var = 0 + iter + for_node->min;
      rewriter.set_loop_var_replacement(replaced_loop_var);
      stmt = rewriter(stmt);
      for_body.push_back(stmt);
    }

    // Step 3.2: Rewrite the for body of loop.
    Array<Stmt> body;
    for (const auto &order_str : body_orders) {
      int iter = name2iter(order_str);
      PrimExpr replaced_loop_var =
          iterations * for_node->loop_var + iter + for_node->min;
      if (iter == iterations) {
        iter = 0;
      }
      int id = name2id(order_str);
      Stmt stmt = pipeline_body_seq->seq[id];
      rewriter.set_current_version(iter);
      rewriter.set_loop_var_replacement(replaced_loop_var);
      stmt = rewriter(stmt);
      body.push_back(stmt);
    }

    auto extent = floordiv(for_node->extent, iterations);
    PrimExpr epilogue_iterations_expr = floormod(for_node->extent, iterations);
    int epilogue_iterations = -1;
    if (const auto *mod_int = epilogue_iterations_expr.as<IntImmNode>()) {
      epilogue_iterations = mod_int->value;
    }
    ICHECK(epilogue_iterations != -1)
        << "Can't calculate the epilogue iterations.";

    if (epilogue_iterations == 0) {
      extent = extent - 1;
    }
    For new_for_stmt =
        For(for_node->loop_var, PrimExpr(0), extent, ForKind::kSerial,
            SeqStmt::Flatten(body), for_node->thread_binding, {});
    for_body.push_back(new_for_stmt);

    // Step 3.3: Rewrite the epilogue.
    for (const auto &order_str : epilogue_orders) {
      int iter = name2iter(order_str);
      int id = name2id(order_str);
      Stmt stmt = pipeline_body_seq->seq[id];
      rewriter.set_current_version(iter);
      PrimExpr replaced_loop_var = extent * iterations + iter + for_node->min;
      rewriter.set_loop_var_replacement(replaced_loop_var);
      stmt = rewriter(stmt);
      for_body.push_back(stmt);
    }
    return SeqStmt::Flatten(for_body);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  Optional<String> global_symbol_;
  ASTTraverser traverser_;
};

tvm::transform::Pass InjectSunmmioPipeline() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    auto *fptr = f.CopyOnWrite();
    fptr->body = SunmmioMultiVersionBufferRewriter::Substitute(f);
    fptr->body = SunmmioPipelineInjector::Inject(f);
    fptr->body = ConvertSSA(std::move(fptr->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectSunmmioPipeline", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectSunmmioPipeline",
                        InjectSunmmioPipeline);
}

} // namespace tl
} // namespace tvm
