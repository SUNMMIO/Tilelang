#include "../layout/cute_layout.h"
#include "../layout/utils.h"
#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/parallel.h"
#include "../op/region.h"
#include "../op/utils.h"
#include "../target/sunmmio_utils.h"
#include "../target/utils.h"
#include "../tileview/tileview.h"
#include "common/loop_fusion_utils.h"
#include "common/remap_buffer_rewriter.h"
#include "sunmmio_pipeline_planning/stmt_read_write_collector.h"
#include "sunmmio_pipeline_planning/sunmmio_pipeline_utils.h"
#include "tir/transforms/ir_utils.h"
#include "tvm/ir/attrs.h"
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

#include <string>
#include <unordered_map>
#include <unordered_set>

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
      if (substituter.IsBankedBuffer(buffer)) {
        int ping_versions = (substituter.iterations_ + 1) / 2;
        int pong_versions = substituter.iterations_ / 2;
        Buffer ping = substituter.makeMultiVersionBuffer(buffer, ping_versions,
                                                         "_ping", true);
        Buffer pong = substituter.makeMultiVersionBuffer(buffer, pong_versions,
                                                         "_pong", false);
        substituter.buffer_remap_.Set(buffer, ping);
        substituter.bank_peer_buffers_[buffer.get()] = pong;
      } else {
        substituter.buffer_remap_.Set(buffer,
                                      substituter.makeMultiVersionBuffer(
                                          buffer, substituter.iterations_));
      }
    }

    substituter.RewriteFunctionLayoutAttrs(f);
    substituter.RecordDefaultPingPongAttrs(f);

    f.CopyOnWrite()->body =
        RemapBufferRewriter::Substitute(f->body, substituter.buffer_remap_);

    return substituter.VisitStmt(f->body);
  }

private:
  void RewriteFunctionLayoutAttrs(PrimFunc &f) {
    auto layout_map_opt = f->GetAttr<Map<Buffer, Layout>>(attr::kLayoutMap);
    if (!layout_map_opt) {
      return;
    }

    arith::Analyzer analyzer;
    Map<Buffer, Layout> new_layout_map;
    for (const auto &[buffer, layout] : layout_map_opt.value()) {
      auto it = buffer_remap_.find(buffer);
      if (it == buffer_remap_.end()) {
        new_layout_map.Set(buffer, layout);
        continue;
      }

      const Buffer &new_buffer = (*it).second;
      Optional<Layout> derived_layout =
          DeriveLayoutLikeForDType(layout, new_buffer->shape, new_buffer->dtype,
                                   Optional<Array<Integer>>(), &analyzer);
      ICHECK(derived_layout.defined())
          << "Failed to derive multiversioned layout for buffer "
          << buffer->name << " with shape " << new_buffer->shape;
      new_layout_map.Set(new_buffer, derived_layout.value());
      auto peer_it = bank_peer_buffers_.find(buffer.get());
      if (peer_it != bank_peer_buffers_.end()) {
        const Buffer &peer_buffer = peer_it->second;
        Optional<Layout> peer_layout = DeriveLayoutLikeForDType(
            layout, peer_buffer->shape, peer_buffer->dtype,
            Optional<Array<Integer>>(), &analyzer);
        ICHECK(peer_layout.defined())
            << "Failed to derive ping/pong layout for buffer " << buffer->name
            << " with shape " << peer_buffer->shape;
        new_layout_map.Set(peer_buffer, peer_layout.value());
      }
    }
    f = WithAttr(std::move(f), attr::kLayoutMap, new_layout_map);
  }

  void RecordDefaultPingPongAttrs(PrimFunc &f) {
    if (buffer_remap_.empty()) {
      return;
    }

    Map<Var, String> alloc_ping_pong;
    for (const auto &kv : bank_peer_buffers_) {
      alloc_ping_pong.Set(kv.second->data, String("pong"));
    }

    if (alloc_ping_pong.empty()) {
      return;
    }

    f = WithAttr(std::move(f), tl::attr::kSunmmioAllocPingPong,
                 alloc_ping_pong);
  }

  bool IsBankedBuffer(const Buffer &buffer) const {
    return buffer.scope() == kSunmmioScopeASRAM ||
           buffer.scope() == kSunmmioScopeWSRAM;
  }

  Buffer makeMultiVersionBuffer(const Buffer &buffer, int num_version,
                                const std::string &name_suffix = "",
                                bool reuse_primary_var = true) {
    const auto *ptr_type =
        TVM_TYPE_AS(buffer->data->type_annotation, PointerTypeNode);
    Var new_var;
    std::string data_name = std::string(buffer->data->name_hint) + name_suffix;
    std::string buffer_name = std::string(buffer->name) + name_suffix;
    if (reuse_primary_var && var_remap_.count(buffer->data)) {
      new_var = var_remap_[buffer->data];
    } else {
      Type new_type =
          PointerType(ptr_type->element_type, ptr_type->storage_scope);
      new_var = Var(data_name, new_type);
      if (reuse_primary_var) {
        var_remap_.Set(buffer->data, new_var);
      }
    }
    auto shape = buffer->shape;
    if (num_version > 1) {
      shape.insert(shape.begin(), num_version);
    }
    buffer_has_version_axis_[new_var.get()] = num_version > 1;
    return Buffer(new_var, buffer->dtype, shape, {}, buffer->elem_offset,
                  String(buffer_name), buffer->data_alignment,
                  buffer->offset_factor, buffer->buffer_type);
  }

  BufferRegion
  RewritePipelineBufferRegion(const BufferRegion &buffer_region) const {
    auto it = buffer_remap_.find(buffer_region->buffer);
    if (it != buffer_remap_.end()) {
      Region new_region = buffer_region->region;
      const Buffer &new_buffer = (*it).second;
      if (HasVersionAxis(new_buffer)) {
        Range accessed_version = Range::FromMinExtent(0, 1);
        new_region.insert(new_region.begin(), accessed_version);
      }
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
        Map<Buffer, Buffer> bank_peer_buffers;
        for (const Buffer &buffer : versioned_buffers) {
          auto remap_it = buffer_remap_.find(buffer);
          auto peer_it = bank_peer_buffers_.find(buffer.get());
          if (remap_it != buffer_remap_.end() &&
              peer_it != bank_peer_buffers_.end()) {
            bank_peer_buffers.Set((*remap_it).second, peer_it->second);
          }
        }
        if (!bank_peer_buffers.empty()) {
          loop.CopyOnWrite()->annotations.Set("bank_peer_buffers",
                                              bank_peer_buffers);
        }
        Array<Buffer> version_axis_buffers;
        for (const auto &kv : buffer_remap_) {
          if (HasVersionAxis(kv.second)) {
            version_axis_buffers.push_back(kv.second);
          }
          auto peer_it = bank_peer_buffers_.find(kv.first.get());
          if (peer_it != bank_peer_buffers_.end() &&
              HasVersionAxis(peer_it->second)) {
            version_axis_buffers.push_back(peer_it->second);
          }
        }
        loop.CopyOnWrite()->annotations.Set("version_axis_buffers",
                                            version_axis_buffers);
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
          auto peer_it = bank_peer_buffers_.find(buffer.get());
          if (peer_it != bank_peer_buffers_.end()) {
            new_map.Set(peer_it->second, layout);
          }
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
    Array<Buffer> alloc_buffers;
    for (const Buffer &buf : block->alloc_buffers) {
      auto remap_it = buffer_remap_.find(buf);
      if (remap_it == buffer_remap_.end()) {
        alloc_buffers.push_back(buf);
        continue;
      }
      alloc_buffers.push_back((*remap_it).second);
      auto peer_it = bank_peer_buffers_.find(buf.get());
      if (peer_it != bank_peer_buffers_.end()) {
        alloc_buffers.push_back(peer_it->second);
      }
    }

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
      if (HasVersionAxis(new_buffer)) {
        indices.insert(indices.begin(), 0);
      }
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
      if (HasVersionAxis(new_buffer)) {
        indices.insert(indices.begin(), 0);
      }
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
      if (HasVersionAxis(new_buffer)) {
        new_ranges.insert(new_ranges.begin(), Range(0, 1));
      }

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
  bool HasVersionAxis(const Buffer &buffer) const {
    auto it = buffer_has_version_axis_.find(buffer->data.get());
    return it != buffer_has_version_axis_.end() && it->second;
  }

  int iterations_ = -1;
  bool replace_flag = false;
  Map<Buffer, Buffer> buffer_remap_;
  Map<Var, Var> var_remap_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  std::unordered_map<const BufferNode *, Buffer> bank_peer_buffers_;
  std::unordered_map<const VarNode *, bool> buffer_has_version_axis_;
};

class PipelineBodyRewriter : public StmtExprMutator {
public:
  PipelineBodyRewriter(Array<Buffer> used_buffers,
                       Map<Buffer, Buffer> bank_peer_buffers,
                       Array<Buffer> version_axis_buffers, For pipeline_loop) {
    used_buffers_ = used_buffers;
    bank_peer_buffers_ = std::move(bank_peer_buffers);
    pipeline_loop_ = std::move(pipeline_loop);
    for (const Buffer &buffer : version_axis_buffers) {
      version_axis_buffers_.insert(buffer.get());
    }
    for (auto it : used_buffers) {
      buffer_data_to_buffer_.Set(it->data, it);
      if (bank_peer_buffers_.count(it)) {
        const Buffer &peer = bank_peer_buffers_[it];
        buffer_data_to_buffer_.Set(peer->data, peer);
      }
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
      Var data = Downcast<Var>(call->args[i]);
      if (!buffer_data_to_buffer_.count(data)) {
        continue;
      }
      const Buffer &buffer = buffer_data_to_buffer_[data];
      if (!IsVersionedBuffer(buffer)) {
        continue;
      }
      Buffer target = ResolveTargetBuffer(buffer);
      if (!HasVersionAxis(target)) {
        new_args.Set(i, target->data);
        continue;
      }
      PrimExpr offset;
      if (!target->strides.empty()) {
        offset = target->strides[0];
      } else {
        Array<PrimExpr> inner_shape;
        for (size_t axis = 1; axis < target->shape.size(); ++axis) {
          inner_shape.push_back(target->shape[axis]);
        }
        offset = product(inner_shape);
      }
      new_args.Set(i, target->data);
      new_args.Set(i + 1, call->args[i + 1] +
                              Integer(CurrentVersionSlot(buffer)) * offset);
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
    Buffer target = ResolveTargetBuffer(store->buffer);
    Array<PrimExpr> indices = store->indices;
    RewriteIndices(store->buffer, target, &indices);
    return BufferStore(target, store->value, indices);
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
    Buffer target = ResolveTargetBuffer(load->buffer);
    Array<PrimExpr> indices = load->indices;
    RewriteIndices(load->buffer, target, &indices);
    return BufferLoad(target, indices);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(RegionOp::Get())) {
      RegionOp original_region(op->args);
      Buffer source = original_region->GetBuffer();
      if (IsVersionedBuffer(source)) {
        Buffer target = ResolveTargetBuffer(source);
        Array<Range> ranges = original_region->GetRanges();
        if (HasVersionAxis(target)) {
          ICHECK(HasVersionAxis(source));
          ranges.Set(0, Range::FromMinExtent(CurrentVersionSlot(source), 1));
        } else if (HasVersionAxis(source)) {
          Array<Range> squeezed;
          for (size_t i = 1; i < ranges.size(); ++i) {
            squeezed.push_back(ranges[i]);
          }
          ranges = squeezed;
        }

        Array<PrimExpr> args;
        Array<PrimExpr> mins;
        for (const Range &range : ranges) {
          mins.push_back(VisitExpr(range->min));
        }
        args.push_back(BufferLoad(target, mins));
        args.push_back(VisitExpr(original_region->GetAccessMask()));
        for (const Range &range : ranges) {
          args.push_back(VisitExpr(range->extent));
        }
        return Call(DataType::Handle(), RegionOp::Get(), args);
      }
    }
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

  bool IsVersionedBuffer(const Buffer &buffer) const {
    for (const Buffer &candidate : used_buffers_) {
      if (candidate.same_as(buffer)) {
        return true;
      }
    }
    return false;
  }

  bool IsBankedBuffer(const Buffer &buffer) const {
    return bank_peer_buffers_.count(buffer) != 0;
  }

  Buffer ResolveTargetBuffer(const Buffer &buffer) const {
    if (!IsBankedBuffer(buffer) || current_version_ % 2 == 0) {
      return buffer;
    }
    return bank_peer_buffers_[buffer];
  }

  int CurrentVersionSlot(const Buffer &buffer) const {
    return IsBankedBuffer(buffer) ? current_version_ / 2 : current_version_;
  }

  bool HasVersionAxis(const Buffer &buffer) const {
    return version_axis_buffers_.count(buffer.get()) != 0;
  }

  void RewriteIndices(const Buffer &source, const Buffer &target,
                      Array<PrimExpr> *indices) const {
    if (HasVersionAxis(target)) {
      ICHECK(HasVersionAxis(source));
      indices->Set(0, CurrentVersionSlot(source));
    } else if (HasVersionAxis(source)) {
      Array<PrimExpr> squeezed;
      for (size_t i = 1; i < indices->size(); ++i) {
        squeezed.push_back((*indices)[i]);
      }
      *indices = squeezed;
    }
  }

  Array<Buffer> used_buffers_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, Buffer> bank_peer_buffers_;
  std::unordered_set<const BufferNode *> version_axis_buffers_;
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
      : global_symbol_(std::move(global_symbol)), stmt_rw_collector(f) {
    stmt_rw_collector.clear();
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // Step 1: Recursively rewrite the children first.
    For for_node = Downcast<For>(StmtExprMutator::VisitStmt_(op));

    auto iterations_anno = op->annotations.Get("iterations");
    auto used_buffers_anno = op->annotations.Get("used_buffers");
    auto versioned_buffers_anno = op->annotations.Get("versioned_buffers");
    auto bank_peer_buffers_anno = op->annotations.Get("bank_peer_buffers");
    auto version_axis_buffers_anno =
        op->annotations.Get("version_axis_buffers");
    auto prologue_orders_anno = op->annotations.Get("prologue_orders");
    auto body_orders_anno = op->annotations.Get("body_orders");
    auto epilogue_orders_anno = op->annotations.Get("epilogue_orders");

    if (!iterations_anno || !used_buffers_anno || !versioned_buffers_anno ||
        !prologue_orders_anno || !body_orders_anno) {
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
    Array<String> epilogue_orders;
    if (epilogue_orders_anno) {
      epilogue_orders = Downcast<Array<String>>(epilogue_orders_anno.value());
    }
    Array<Buffer> versioned_buffers =
        Downcast<Array<Buffer>>(versioned_buffers_anno.value());
    Array<Buffer> used_buffers =
        Downcast<Array<Buffer>>(used_buffers_anno.value());
    Map<Buffer, Buffer> bank_peer_buffers;
    if (bank_peer_buffers_anno) {
      bank_peer_buffers =
          Downcast<Map<Buffer, Buffer>>(bank_peer_buffers_anno.value());
    }
    Array<Buffer> version_axis_buffers;
    if (version_axis_buffers_anno) {
      version_axis_buffers =
          Downcast<Array<Buffer>>(version_axis_buffers_anno.value());
    }
    for (auto it : used_buffers) {
      pipeline_allocs.push_back(it);
    }

    auto rewriter = PipelineBodyRewriter(versioned_buffers, bank_peer_buffers,
                                         version_axis_buffers, for_node);
    auto version_slot = [iterations](int iter) {
      ICHECK_GT(iterations, 0);
      int slot = iter % iterations;
      return slot < 0 ? slot + iterations : slot;
    };
    Array<Stmt> for_body;
    // Step 3.1: Rewrite prologue
    for (const auto &order_str : prologue_orders) {
      int iter = name2iter(order_str);
      int id = name2id(order_str);
      Stmt stmt = pipeline_body_seq->seq[id];
      rewriter.set_current_version(version_slot(iter));
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
      int id = name2id(order_str);
      Stmt stmt = pipeline_body_seq->seq[id];
      rewriter.set_current_version(version_slot(iter));
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

    if (epilogue_iterations == 0) {
      extent = extent - 1;
    }
    For new_for_stmt =
        For(for_node->loop_var, PrimExpr(0), extent, ForKind::kSerial,
            SeqStmt::Flatten(body), for_node->thread_binding, {});
    for_body.push_back(new_for_stmt);

    // Step 3.3: Rewrite the epilogue.
    if (epilogue_iterations != -1) {
      for (const auto &order_str : epilogue_orders) {
        int iter = name2iter(order_str);
        int id = name2id(order_str);
        Stmt stmt = pipeline_body_seq->seq[id];
        rewriter.set_current_version(version_slot(iter));
        PrimExpr replaced_loop_var = extent * iterations + iter + for_node->min;
        rewriter.set_loop_var_replacement(replaced_loop_var);
        stmt = rewriter(stmt);
        for_body.push_back(stmt);
      }
    } else {
      // Dynamic epilogue loop for non-constant iterations
      Var epilogue_loop_var("epilogue_i", for_node->loop_var->dtype);
      Array<Stmt> epilogue_body;
      for (int slot = 0; slot < iterations; ++slot) {
        Array<Stmt> slot_body;
        for (size_t id = 0; id < pipeline_body_seq->size(); ++id) {
          Stmt stmt = pipeline_body_seq->seq[id];
          rewriter.set_current_version(slot);
          PrimExpr replaced_loop_var =
              extent * iterations + epilogue_loop_var + for_node->min;
          rewriter.set_loop_var_replacement(replaced_loop_var);
          slot_body.push_back(rewriter(stmt));
        }
        epilogue_body.push_back(IfThenElse(EQ(epilogue_loop_var, Integer(slot)),
                                           SeqStmt::Flatten(slot_body)));
      }
      For dynamic_epilogue_for = For(
          epilogue_loop_var, PrimExpr(0), epilogue_iterations_expr,
          ForKind::kSerial, SeqStmt::Flatten(epilogue_body), std::nullopt, {});
      for_body.push_back(dynamic_epilogue_for);
    }
    return SeqStmt::Flatten(for_body);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  Optional<String> global_symbol_;
  StmtReadWriteCollector stmt_rw_collector;
};

tvm::transform::Pass InjectSunmmioPipeline() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    Stmt multiversioned_body = SunmmioMultiVersionBufferRewriter::Substitute(f);
    auto *fptr = f.CopyOnWrite();
    fptr->body = multiversioned_body;
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
