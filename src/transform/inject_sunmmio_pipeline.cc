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
#include "sunmmio_pipeline_planning/pipeline_diagnostic.h"
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

#include <cstdint>
#include <exception>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
      int versions = substituter.version_counts_.at(buffer.get());
      if (substituter.IsBankedBuffer(buffer)) {
        int ping_versions = (versions + 1) / 2;
        int pong_versions = ping_versions;
        Buffer ping = substituter.makeMultiVersionBuffer(buffer, ping_versions,
                                                         "_ping", true);
        Buffer pong = substituter.makeMultiVersionBuffer(buffer, pong_versions,
                                                         "_pong", false);
        substituter.buffer_remap_.Set(buffer, ping);
        substituter.bank_peer_buffers_[buffer.get()] = pong;
      } else {
        substituter.buffer_remap_.Set(
            buffer, substituter.makeMultiVersionBuffer(buffer, versions));
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
    auto writer_phases_anno = op->annotations.Get("runtime_bank_writer_phases");
    auto reader_phases_anno = op->annotations.Get("runtime_bank_reader_phases");
    if (versioned_buffers_anno && used_buffers_anno && iterations_anno) {
      Array<Buffer> versioned_buffers =
          Downcast<Array<Buffer>>(versioned_buffers_anno.value());
      int iterations = Downcast<int>(iterations_anno.value());
      if (!replace_flag) {
        for (const Buffer &buffer : versioned_buffers) {
          auto [it, inserted] =
              version_counts_.try_emplace(buffer.get(), iterations);
          if (inserted) {
            versioned_buffers_.push_back(buffer);
          } else {
            it->second = std::max(it->second, iterations);
          }
        }
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
        for (const Buffer &buffer : versioned_buffers) {
          auto remap_it = buffer_remap_.find(buffer);
          if (remap_it == buffer_remap_.end()) {
            continue;
          }
          const Buffer &remapped = (*remap_it).second;
          if (HasVersionAxis(remapped)) {
            version_axis_buffers.push_back(remapped);
          }
          auto peer_it = bank_peer_buffers_.find(buffer.get());
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
        auto remap_buffer_array_annotation = [&](const char *name) {
          auto annotation = op->annotations.Get(name);
          if (!annotation) {
            return;
          }
          Array<Buffer> buffers = Downcast<Array<Buffer>>(annotation.value());
          Array<Buffer> remapped;
          for (const Buffer &buffer : buffers) {
            auto it = buffer_remap_.find(buffer);
            remapped.push_back(it == buffer_remap_.end() ? buffer
                                                         : (*it).second);
          }
          loop.CopyOnWrite()->annotations.Set(name, remapped);
        };
        remap_buffer_array_annotation(kRegionCrossOuterCoreVersionedBuffers);
        remap_buffer_array_annotation(kRegionCrossOuterRegionVersionedBuffers);
        auto remap_phase_annotation = [&](const Optional<Any> &annotation,
                                          const char *name) {
          if (!annotation)
            return;
          Map<Buffer, Map<Integer, PrimExpr>> phases =
              Downcast<Map<Buffer, Map<Integer, PrimExpr>>>(annotation.value());
          Map<Buffer, Map<Integer, PrimExpr>> remapped;
          for (const auto &[buffer, per_command] : phases) {
            auto it = buffer_remap_.find(buffer);
            remapped.Set(it == buffer_remap_.end() ? buffer : (*it).second,
                         per_command);
          }
          loop.CopyOnWrite()->annotations.Set(name, remapped);
        };
        remap_phase_annotation(writer_phases_anno,
                               "runtime_bank_writer_phases");
        remap_phase_annotation(reader_phases_anno,
                               "runtime_bank_reader_phases");
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

  std::unordered_map<const BufferNode *, int> version_counts_;
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
                       Array<Buffer> version_axis_buffers,
                       Map<Buffer, Map<Integer, PrimExpr>> writer_phases,
                       Map<Buffer, Map<Integer, PrimExpr>> reader_phases,
                       For pipeline_loop) {
    used_buffers_ = used_buffers;
    bank_peer_buffers_ = std::move(bank_peer_buffers);
    pipeline_loop_ = std::move(pipeline_loop);
    for (const Buffer &buffer : version_axis_buffers) {
      version_axis_buffers_.insert(buffer.get());
    }
    auto import_phases = [](const Map<Buffer, Map<Integer, PrimExpr>> &source,
                            auto *destination) {
      for (const auto &[buffer, phases] : source) {
        auto &per_command = (*destination)[buffer.get()];
        for (const auto &[command_id, phase] : phases) {
          per_command[command_id->value] = Downcast<IntImm>(phase)->value;
        }
      }
    };
    import_phases(writer_phases, &writer_phases_);
    import_phases(reader_phases, &reader_phases_);
    for (auto it : used_buffers) {
      buffer_data_to_buffer_.Set(it->data, it);
      if (bank_peer_buffers_.count(it)) {
        const Buffer &peer = bank_peer_buffers_[it];
        buffer_data_to_buffer_.Set(peer->data, peer);
      }
    }
  }

  void set_current_version(int v) { current_version_ = v; }
  void set_current_command(int id) { current_command_id_ = id; }

  void set_loop_var_replacement(PrimExpr p) {
    set_var_replacement(pipeline_loop_->loop_var, std::move(p));
  }

  void set_var_replacement(const Var &var, PrimExpr replacement) {
    loop_var_replacements_.Set(var, std::move(replacement));
  }

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
      Buffer target = ResolveTargetBuffer(buffer, true);
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
    Buffer target = ResolveTargetBuffer(store->buffer, true);
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
    Buffer target = ResolveTargetBuffer(load->buffer, false);
    Array<PrimExpr> indices = load->indices;
    RewriteIndices(load->buffer, target, &indices);
    return BufferLoad(target, indices);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(RegionOp::Get())) {
      RegionOp original_region(op->args);
      Buffer source = original_region->GetBuffer();
      if (IsVersionedBuffer(source)) {
        bool is_write = HasCommandPhase(writer_phases_, source);
        Buffer target = ResolveTargetBuffer(source, is_write);
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
    if (auto replacement = loop_var_replacements_.Get(var)) {
      return replacement.value();
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

  bool HasCommandPhase(
      const std::unordered_map<const BufferNode *, std::unordered_map<int, int>>
          &phases,
      const Buffer &buffer) const {
    auto it_buffer = phases.find(buffer.get());
    return it_buffer != phases.end() &&
           it_buffer->second.count(current_command_id_) != 0;
  }

  int CurrentPhase(const Buffer &buffer, bool is_write) const {
    const auto &phases = is_write ? writer_phases_ : reader_phases_;
    auto it_buffer = phases.find(buffer.get());
    if (it_buffer == phases.end())
      return 0;
    auto it_command = it_buffer->second.find(current_command_id_);
    return it_command == it_buffer->second.end() ? 0 : it_command->second;
  }

  Buffer ResolveTargetBuffer(const Buffer &buffer, bool is_write) const {
    if (!IsBankedBuffer(buffer) ||
        (current_version_ + CurrentPhase(buffer, is_write)) % 2 == 0) {
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
  int current_command_id_ = -1;
  std::unordered_map<const BufferNode *, std::unordered_map<int, int>>
      writer_phases_;
  std::unordered_map<const BufferNode *, std::unordered_map<int, int>>
      reader_phases_;
  Map<Var, PrimExpr> loop_var_replacements_;
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
    auto dynamic_epilogue_orders_anno =
        op->annotations.Get("dynamic_epilogue_orders");
    auto writer_phases_anno = op->annotations.Get("runtime_bank_writer_phases");
    auto reader_phases_anno = op->annotations.Get("runtime_bank_reader_phases");

    if (!iterations_anno || !used_buffers_anno || !versioned_buffers_anno ||
        !prologue_orders_anno || !body_orders_anno) {
      return for_node;
    }

    arith::Analyzer extent_analyzer;
    PrimExpr simplified_extent = extent_analyzer.Simplify(for_node->extent);
    const auto *static_extent = simplified_extent.as<IntImmNode>();
    auto make_sequential_fallback = [&](const std::string &reason,
                                        bool emit_warning = true) {
      Map<String, Any> annotations;
      for (const auto &kv : for_node->annotations) {
        if (kv.first != "num_stages" && kv.first != "iterations" &&
            kv.first != "prologue_orders" && kv.first != "body_orders" &&
            kv.first != "epilogue_orders") {
          annotations.Set(kv.first, kv.second);
        }
      }
      For sequential = for_node;
      sequential.CopyOnWrite()->annotations = annotations;
      return MakePipelineFallback(sequential, "greedy", "inject", reason,
                                  emit_warning);
    };
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
    int max_body_iter_offset = 0;
    for (const String &order : body_orders) {
      max_body_iter_offset = std::max(max_body_iter_offset, name2iter(order));
    }
    if (static_extent != nullptr &&
        static_extent->value <= max_body_iter_offset) {
      return make_sequential_fallback("short_extent_unsupported");
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
    Map<Buffer, Map<Integer, PrimExpr>> writer_phases;
    if (writer_phases_anno) {
      writer_phases = Downcast<Map<Buffer, Map<Integer, PrimExpr>>>(
          writer_phases_anno.value());
    }
    Map<Buffer, Map<Integer, PrimExpr>> reader_phases;
    if (reader_phases_anno) {
      reader_phases = Downcast<Map<Buffer, Map<Integer, PrimExpr>>>(
          reader_phases_anno.value());
    }
    for (auto it : used_buffers) {
      pipeline_allocs.push_back(it);
    }

    auto rewriter = PipelineBodyRewriter(versioned_buffers, bank_peer_buffers,
                                         version_axis_buffers, writer_phases,
                                         reader_phases, for_node);
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
      rewriter.set_current_command(id);
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
      rewriter.set_current_command(id);
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
    } else if (epilogue_iterations == -1) {
      extent = floordiv(max(0, for_node->extent - 1), iterations);
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
        rewriter.set_current_command(id);
        rewriter.set_current_version(version_slot(iter));
        PrimExpr replaced_loop_var = extent * iterations + iter + for_node->min;
        rewriter.set_loop_var_replacement(replaced_loop_var);
        stmt = rewriter(stmt);
        for_body.push_back(stmt);
      }
    } else {
      ICHECK(dynamic_epilogue_orders_anno)
          << "Dynamic pipeline requires remainder-specific epilogue orders";
      Map<Integer, Array<String>> dynamic_orders =
          Downcast<Map<Integer, Array<String>>>(
              dynamic_epilogue_orders_anno.value());
      auto build_dynamic_epilogue = [&](const Array<String> &schedule) {
        Array<Stmt> epilogue_body;
        for (const String &order_str : schedule) {
          int iter = name2iter(order_str);
          int id = name2id(order_str);
          PrimExpr logical_iter = extent * iterations + iter;
          PrimExpr replaced_loop_var = logical_iter + for_node->min;
          Stmt stmt = pipeline_body_seq->seq[id];
          rewriter.set_current_command(id);
          rewriter.set_current_version(version_slot(iter));
          rewriter.set_loop_var_replacement(replaced_loop_var);
          stmt = rewriter(stmt);
          PrimExpr valid = And(GE(logical_iter, Integer(0)),
                               LT(logical_iter, for_node->extent));
          epilogue_body.push_back(IfThenElse(valid, stmt));
        }
        return SeqStmt::Flatten(epilogue_body);
      };

      PrimExpr runtime_remainder =
          floormod(for_node->extent, Integer(iterations));
      Stmt dispatched_epilogue{nullptr};
      for (int remainder = iterations - 1; remainder >= 0; --remainder) {
        Optional<Array<String>> schedule;
        for (const auto &kv : dynamic_orders) {
          if (kv.first->value == remainder) {
            schedule = kv.second;
            break;
          }
        }
        ICHECK(schedule.defined())
            << "Missing dynamic epilogue schedule for remainder " << remainder;
        Stmt branch = build_dynamic_epilogue(schedule.value());
        if (!dispatched_epilogue.defined()) {
          dispatched_epilogue = std::move(branch);
        } else {
          dispatched_epilogue =
              IfThenElse(EQ(runtime_remainder, Integer(remainder)),
                         std::move(branch), std::move(dispatched_epilogue));
        }
      }
      ICHECK(dispatched_epilogue.defined());
      for_body.push_back(dispatched_epilogue);
    }
    return SeqStmt::Flatten(for_body);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  Optional<String> global_symbol_;
  StmtReadWriteCollector stmt_rw_collector;
};

static bool VerifyCrossOuterCoverage(int64_t total_extent, int iterations,
                                     int command_count,
                                     const Array<String> &prologue_orders,
                                     const Array<String> &body_orders,
                                     const Array<String> &epilogue_orders,
                                     std::string *detail) {
  if (total_extent <= 0 || iterations <= 0 || command_count <= 0 ||
      total_extent % iterations != 0) {
    *detail = "invalid verifier dimensions";
    return false;
  }
  if (static_cast<uint64_t>(total_extent) >
      std::numeric_limits<uint64_t>::max() /
          static_cast<uint64_t>(command_count)) {
    *detail = "coverage key space overflows uint64";
    return false;
  }

  std::unordered_set<uint64_t> covered;
  auto record = [&](const String &order, int64_t base) {
    int iter = name2iter(order);
    int id = name2id(order);
    int64_t logical = base + iter;
    if (iter < 0 || id < 0 || id >= command_count || logical < 0 ||
        logical >= total_extent) {
      std::ostringstream os;
      os << "out-of-range order=" << order << " logical=" << logical;
      *detail = os.str();
      return false;
    }
    int version = iter % iterations;
    if (version < 0) {
      version += iterations;
    }
    if (version < 0 || version >= iterations) {
      *detail = "version slot out of range";
      return false;
    }
    uint64_t key =
        static_cast<uint64_t>(logical) * static_cast<uint64_t>(command_count) +
        static_cast<uint64_t>(id);
    if (!covered.insert(key).second) {
      std::ostringstream os;
      os << "duplicate order=" << order << " logical=" << logical;
      *detail = os.str();
      return false;
    }
    return true;
  };

  for (const String &order : prologue_orders) {
    if (!record(order, 0)) {
      return false;
    }
  }
  int64_t total_chunks = total_extent / iterations;
  for (int64_t chunk = 0; chunk + 1 < total_chunks; ++chunk) {
    for (const String &order : body_orders) {
      if (!record(order, chunk * iterations)) {
        return false;
      }
    }
  }
  for (const String &order : epilogue_orders) {
    if (!record(order, (total_chunks - 1) * iterations)) {
      return false;
    }
  }

  uint64_t expected = static_cast<uint64_t>(total_extent) *
                      static_cast<uint64_t>(command_count);
  if (covered.size() != expected) {
    std::ostringstream os;
    os << "coverage size=" << covered.size() << " expected=" << expected;
    *detail = os.str();
    return false;
  }
  return true;
}

class CrossOuterPlanningCleanup : public StmtMutator {
public:
  CrossOuterPlanningCleanup(std::string reason, std::string detail)
      : reason_(std::move(reason)), detail_(std::move(detail)) {}

private:
  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtMutator::VisitStmt_(op));
    if (!op->annotations.count(kCrossOuterMarker)) {
      return loop;
    }
    Map<String, Any> annotations;
    for (const auto &[key, value] : loop->annotations) {
      if (!IsPipelineScheduleAnnotation(key) && key != kPipelineRequested &&
          key != kPipelineApplied && key != kPipelineMode &&
          key != kPipelineFallbackStage && key != kPipelineFallbackReason &&
          key != kPipelineFallbackDetail) {
        annotations.Set(key, value);
      }
    }
    SetCrossOuterFallbackAnnotations(&annotations, "inject", reason_, detail_);
    loop.CopyOnWrite()->annotations = annotations;
    return loop;
  }

  std::string reason_;
  std::string detail_;
};

static PrimFunc MakeCrossOuterFunctionFallback(const PrimFunc &original,
                                               const std::string &reason,
                                               const std::string &detail) {
  PrimFunc fallback = original;
  CrossOuterPlanningCleanup cleanup(reason, detail);
  fallback.CopyOnWrite()->body = cleanup(original->body);
  return fallback;
}

class SunmmioCrossOuterPipelineInjector : public StmtExprMutator {
public:
  static Stmt Inject(const PrimFunc &func, int *injected_count) {
    SunmmioCrossOuterPipelineInjector injector;
    Stmt result = injector(func->body);
    *injected_count = injector.injected_count_;
    return result;
  }

private:
  Stmt VisitStmt_(const ForNode *outer) final {
    if (!outer->annotations.count(kCrossOuterMarker)) {
      return StmtExprMutator::VisitStmt_(outer);
    }

    const auto *inner = outer->body.as<ForNode>();
    ICHECK(inner != nullptr)
        << "cross_outer planned loop must directly contain the inner loop";
    const auto *pipeline_body = inner->body.as<SeqStmtNode>();
    ICHECK(pipeline_body != nullptr)
        << "cross_outer inner body must remain a direct SeqStmt";

    auto get_int_annotation = [&](const char *key) {
      auto value = outer->annotations.Get(key);
      ICHECK(value.has_value()) << "missing cross_outer annotation " << key;
      return Downcast<IntImm>(value.value())->value;
    };
    int iterations = static_cast<int>(get_int_annotation("iterations"));
    int64_t outer_extent = get_int_annotation(kCrossOuterOuterExtent);
    int64_t inner_extent = get_int_annotation(kCrossOuterInnerExtent);
    arith::Analyzer extent_analyzer;
    auto get_static_extent = [&](const PrimExpr &extent,
                                 const char *loop_name) {
      PrimExpr simplified = extent_analyzer.Simplify(extent);
      const auto *value = simplified.as<IntImmNode>();
      ICHECK(value != nullptr)
          << "cross_outer " << loop_name
          << " extent must simplify to a compile-time integer";
      return value->value;
    };
    ICHECK_GT(iterations, 0);
    ICHECK_EQ(outer_extent, get_static_extent(outer->extent, "outer"));
    ICHECK_EQ(inner_extent, get_static_extent(inner->extent, "inner"));
    ICHECK_EQ(inner_extent % iterations, 0);
    int64_t total_extent = outer_extent * inner_extent;

    Array<String> prologue_orders =
        Downcast<Array<String>>(outer->annotations.at("prologue_orders"));
    Array<String> body_orders =
        Downcast<Array<String>>(outer->annotations.at("body_orders"));
    Array<String> epilogue_orders =
        Downcast<Array<String>>(outer->annotations.at("epilogue_orders"));
    std::string coverage_detail;
    ICHECK(VerifyCrossOuterCoverage(
        total_extent, iterations, static_cast<int>(pipeline_body->seq.size()),
        prologue_orders, body_orders, epilogue_orders, &coverage_detail))
        << "cross_outer_schedule_coverage_failed: " << coverage_detail;

    Array<Buffer> versioned_buffers =
        Downcast<Array<Buffer>>(outer->annotations.at("versioned_buffers"));
    Map<Buffer, Buffer> bank_peer_buffers;
    if (auto ann = outer->annotations.Get("bank_peer_buffers")) {
      bank_peer_buffers = Downcast<Map<Buffer, Buffer>>(ann.value());
    }
    Array<Buffer> version_axis_buffers;
    if (auto ann = outer->annotations.Get("version_axis_buffers")) {
      version_axis_buffers = Downcast<Array<Buffer>>(ann.value());
    }
    Map<Buffer, Map<Integer, PrimExpr>> writer_phases;
    if (auto ann = outer->annotations.Get("runtime_bank_writer_phases")) {
      writer_phases =
          Downcast<Map<Buffer, Map<Integer, PrimExpr>>>(ann.value());
    }
    Map<Buffer, Map<Integer, PrimExpr>> reader_phases;
    if (auto ann = outer->annotations.Get("runtime_bank_reader_phases")) {
      reader_phases =
          Downcast<Map<Buffer, Map<Integer, PrimExpr>>>(ann.value());
    }

    For outer_ref = tvm::ffi::GetRef<For>(outer);
    auto rewriter = PipelineBodyRewriter(versioned_buffers, bank_peer_buffers,
                                         version_axis_buffers, writer_phases,
                                         reader_phases, outer_ref);
    auto version_slot = [iterations](int iter) {
      int slot = iter % iterations;
      return slot < 0 ? slot + iterations : slot;
    };
    auto rewrite_command = [&](const String &order, const PrimExpr &logical) {
      int iter = name2iter(order);
      int id = name2id(order);
      ICHECK_GE(id, 0);
      ICHECK_LT(id, static_cast<int>(pipeline_body->seq.size()));
      rewriter.set_current_command(id);
      rewriter.set_current_version(version_slot(iter));
      rewriter.set_var_replacement(
          outer->loop_var,
          outer->min + floordiv(logical, Integer(inner_extent)));
      rewriter.set_var_replacement(
          inner->loop_var,
          inner->min + floormod(logical, Integer(inner_extent)));
      return rewriter(pipeline_body->seq[id]);
    };

    Array<Stmt> result;
    for (const String &order : prologue_orders) {
      result.push_back(rewrite_command(order, Integer(name2iter(order))));
    }

    Var outer_chunk(std::string(outer->loop_var->name_hint) + "_cross_outer",
                    outer->loop_var.dtype());
    Var inner_chunk(std::string(inner->loop_var->name_hint) + "_cross_chunk",
                    inner->loop_var.dtype());
    int64_t chunks_per_outer = inner_extent / iterations;
    int64_t total_chunks = total_extent / iterations;
    PrimExpr global_chunk =
        outer_chunk * Integer(chunks_per_outer) + inner_chunk;
    Array<Stmt> steady_commands;
    for (const String &order : body_orders) {
      PrimExpr logical =
          global_chunk * Integer(iterations) + Integer(name2iter(order));
      steady_commands.push_back(rewrite_command(order, logical));
    }
    Stmt steady_body = SeqStmt::Flatten(steady_commands);
    steady_body =
        IfThenElse(LT(global_chunk, Integer(total_chunks - 1)), steady_body);
    For steady_inner(inner_chunk, Integer(0), Integer(chunks_per_outer),
                     ForKind::kSerial, steady_body);
    Map<String, Any> steady_annotations;
    SetPipelineAppliedAnnotations(&steady_annotations, "cross_outer");
    steady_annotations.Set(kCrossOuterSteady, Bool(true));
    For steady_outer(outer_chunk, Integer(0), Integer(outer_extent),
                     ForKind::kSerial, steady_inner, outer->thread_binding,
                     steady_annotations);
    result.push_back(steady_outer);

    PrimExpr epilogue_base = Integer((total_chunks - 1) * iterations);
    for (const String &order : epilogue_orders) {
      result.push_back(
          rewrite_command(order, epilogue_base + Integer(name2iter(order))));
    }
    ++injected_count_;
    return SeqStmt::Flatten(result);
  }

  int injected_count_{0};
};

struct RegionCrossOuterEvent {
  String order;
  int64_t logical{0};
  int64_t outer{0};
};

static bool BuildRegionCrossOuterEvents(
    int64_t outer_extent, int64_t inner_extent, int iterations,
    int command_count, const Array<String> &prologue_orders,
    const Array<String> &body_orders, const Array<String> &epilogue_orders,
    std::vector<RegionCrossOuterEvent> *events, std::vector<int64_t> *first,
    std::vector<int64_t> *last, bool *boundary_overlap, std::string *detail) {
  int64_t total_extent = outer_extent * inner_extent;
  if (!VerifyCrossOuterCoverage(total_extent, iterations, command_count,
                                prologue_orders, body_orders, epilogue_orders,
                                detail)) {
    return false;
  }
  first->assign(outer_extent, -1);
  last->assign(outer_extent, -1);
  auto append = [&](const String &order, int64_t base) {
    int64_t logical = base + name2iter(order);
    int64_t outer = logical / inner_extent;
    int64_t position = static_cast<int64_t>(events->size());
    events->push_back(RegionCrossOuterEvent{order, logical, outer});
    if ((*first)[outer] < 0) {
      (*first)[outer] = position;
    }
    (*last)[outer] = position;
  };
  for (const String &order : prologue_orders) {
    append(order, 0);
  }
  int64_t total_chunks = total_extent / iterations;
  for (int64_t chunk = 0; chunk + 1 < total_chunks; ++chunk) {
    for (const String &order : body_orders) {
      append(order, chunk * iterations);
    }
  }
  for (const String &order : epilogue_orders) {
    append(order, (total_chunks - 1) * iterations);
  }
  *boundary_overlap = false;
  for (int64_t outer = 0; outer < outer_extent; ++outer) {
    if ((*first)[outer] < 0 || (*last)[outer] < (*first)[outer]) {
      *detail = "outer iteration has no complete Core interval";
      return false;
    }
    if (outer + 1 < outer_extent && (*first)[outer + 1] < (*last)[outer]) {
      *boundary_overlap = true;
    }
  }
  return true;
}

class RegionCrossOuterPlanningCleanup : public StmtMutator {
public:
  RegionCrossOuterPlanningCleanup(std::string reason, std::string detail)
      : reason_(std::move(reason)), detail_(std::move(detail)) {}

private:
  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtMutator::VisitStmt_(op));
    if (!op->annotations.count(kRegionCrossOuterMarker)) {
      return loop;
    }
    Map<String, Any> annotations;
    for (const auto &[key, value] : loop->annotations) {
      if (!IsPipelineScheduleAnnotation(key) && key != kPipelineRequested &&
          key != kPipelineApplied && key != kPipelineMode &&
          key != kPipelineFallbackStage && key != kPipelineFallbackReason &&
          key != kPipelineFallbackDetail && key != kRegionCrossOuterRequested &&
          key != kRegionCrossOuterFallbackStage &&
          key != kRegionCrossOuterFallbackReason &&
          key != kRegionCrossOuterFallbackDetail) {
        annotations.Set(key, value);
      }
    }
    SetRegionCrossOuterFallbackAnnotations(&annotations, "inject", reason_,
                                           detail_);
    loop.CopyOnWrite()->annotations = annotations;
    return loop;
  }

  std::string reason_;
  std::string detail_;
};

static PrimFunc
MakeRegionCrossOuterFunctionFallback(const PrimFunc &original,
                                     const std::string &reason,
                                     const std::string &detail) {
  PrimFunc fallback = original;
  RegionCrossOuterPlanningCleanup cleanup(reason, detail);
  fallback.CopyOnWrite()->body = cleanup(original->body);
  return fallback;
}

class SunmmioRegionCrossOuterPipelineInjector : public StmtExprMutator {
public:
  static Stmt Inject(const PrimFunc &func, int *injected_count,
                     bool *boundary_overlap) {
    SunmmioRegionCrossOuterPipelineInjector injector;
    Stmt result = injector(func->body);
    *injected_count = injector.injected_count_;
    *boundary_overlap = injector.boundary_overlap_;
    return result;
  }

private:
  Stmt VisitStmt_(const ForNode *outer) final {
    if (!outer->annotations.count(kRegionCrossOuterMarker)) {
      return StmtExprMutator::VisitStmt_(outer);
    }
    String strategy =
        Downcast<String>(outer->annotations.at(kRegionCrossOuterStrategy));
    if (strategy != "continuous") {
      return StmtExprMutator::VisitStmt_(outer);
    }
    const auto *region_body = outer->body.as<SeqStmtNode>();
    ICHECK(region_body != nullptr)
        << "region_cross_outer planned body must remain a SeqStmt";
    int core_index = static_cast<int>(
        Downcast<IntImm>(outer->annotations.at(kRegionCrossOuterCoreIndex))
            ->value);
    int prefix_count = static_cast<int>(
        Downcast<IntImm>(outer->annotations.at(kRegionCrossOuterPrefixCount))
            ->value);
    int suffix_count = static_cast<int>(
        Downcast<IntImm>(outer->annotations.at(kRegionCrossOuterSuffixCount))
            ->value);
    ICHECK_EQ(core_index, prefix_count);
    ICHECK_EQ(static_cast<int>(region_body->seq.size()),
              prefix_count + 1 + suffix_count);
    const auto *inner = region_body->seq[core_index].as<ForNode>();
    ICHECK(inner != nullptr) << "region_cross_outer Core must remain a For";
    const auto *pipeline_body = inner->body.as<SeqStmtNode>();
    ICHECK(pipeline_body != nullptr)
        << "region_cross_outer Core body must remain a SeqStmt";

    auto get_int_annotation = [&](const char *key) {
      return Downcast<IntImm>(outer->annotations.at(key))->value;
    };
    int iterations = static_cast<int>(get_int_annotation("iterations"));
    int64_t outer_extent = get_int_annotation(kRegionCrossOuterOuterExtent);
    int64_t inner_extent = get_int_annotation(kRegionCrossOuterInnerExtent);
    arith::Analyzer analyzer;
    auto static_extent = [&](const PrimExpr &extent) {
      PrimExpr simplified = analyzer.Simplify(extent);
      const auto *value = simplified.as<IntImmNode>();
      ICHECK(value != nullptr)
          << "region_cross_outer extent changed after planning";
      return value->value;
    };
    ICHECK_EQ(outer_extent, static_extent(outer->extent));
    ICHECK_EQ(inner_extent, static_extent(inner->extent));

    Array<String> prologue_orders =
        Downcast<Array<String>>(outer->annotations.at("prologue_orders"));
    Array<String> body_orders =
        Downcast<Array<String>>(outer->annotations.at("body_orders"));
    Array<String> epilogue_orders =
        Downcast<Array<String>>(outer->annotations.at("epilogue_orders"));
    std::vector<RegionCrossOuterEvent> events;
    std::vector<int64_t> first;
    std::vector<int64_t> last;
    bool boundary_overlap = false;
    std::string coverage_detail;
    ICHECK(BuildRegionCrossOuterEvents(
        outer_extent, inner_extent, iterations,
        static_cast<int>(pipeline_body->seq.size()), prologue_orders,
        body_orders, epilogue_orders, &events, &first, &last, &boundary_overlap,
        &coverage_detail))
        << "region_cross_outer_schedule_coverage_failed: " << coverage_detail;
    ICHECK(boundary_overlap)
        << "region_cross_outer continuous schedule has no boundary overlap";

    Array<Buffer> core_versioned_buffers = Downcast<Array<Buffer>>(
        outer->annotations.at(kRegionCrossOuterCoreVersionedBuffers));
    Array<Buffer> region_versioned_buffers = Downcast<Array<Buffer>>(
        outer->annotations.at(kRegionCrossOuterRegionVersionedBuffers));
    Map<Buffer, Buffer> bank_peer_buffers;
    if (auto ann = outer->annotations.Get("bank_peer_buffers")) {
      bank_peer_buffers = Downcast<Map<Buffer, Buffer>>(ann.value());
    }
    Array<Buffer> version_axis_buffers;
    if (auto ann = outer->annotations.Get("version_axis_buffers")) {
      version_axis_buffers = Downcast<Array<Buffer>>(ann.value());
    }
    Map<Buffer, Map<Integer, PrimExpr>> writer_phases;
    if (auto ann = outer->annotations.Get("runtime_bank_writer_phases")) {
      writer_phases =
          Downcast<Map<Buffer, Map<Integer, PrimExpr>>>(ann.value());
    }
    Map<Buffer, Map<Integer, PrimExpr>> reader_phases;
    if (auto ann = outer->annotations.Get("runtime_bank_reader_phases")) {
      reader_phases =
          Downcast<Map<Buffer, Map<Integer, PrimExpr>>>(ann.value());
    }
    Map<Buffer, Map<Integer, PrimExpr>> empty_phases;
    For outer_ref = tvm::ffi::GetRef<For>(outer);
    PipelineBodyRewriter core_rewriter(core_versioned_buffers,
                                       bank_peer_buffers, version_axis_buffers,
                                       writer_phases, reader_phases, outer_ref);
    PipelineBodyRewriter region_rewriter(
        region_versioned_buffers, bank_peer_buffers, version_axis_buffers,
        empty_phases, empty_phases, outer_ref);
    auto positive_mod = [iterations](int64_t value) {
      int slot = static_cast<int>(value % iterations);
      return slot < 0 ? slot + iterations : slot;
    };
    auto prepare_region_rewriter = [&](int64_t outer_index) {
      region_rewriter.set_current_command(-1);
      region_rewriter.set_current_version(positive_mod(outer_index));
      region_rewriter.set_var_replacement(outer->loop_var,
                                          outer->min + Integer(outer_index));
    };
    auto rewrite_region_statement = [&](const Stmt &stmt, int64_t outer_index) {
      prepare_region_rewriter(outer_index);
      return region_rewriter(stmt);
    };
    auto rewrite_core_event = [&](const RegionCrossOuterEvent &event) {
      int iter = name2iter(event.order);
      int id = name2id(event.order);
      core_rewriter.set_current_command(id);
      core_rewriter.set_current_version(positive_mod(iter));
      core_rewriter.set_var_replacement(outer->loop_var,
                                        outer->min + Integer(event.outer));
      core_rewriter.set_var_replacement(
          inner->loop_var, inner->min + Integer(event.logical % inner_extent));
      Stmt stmt = core_rewriter(pipeline_body->seq[id]);
      prepare_region_rewriter(event.outer);
      return region_rewriter(stmt);
    };

    std::vector<int> prefix_coverage(outer_extent, 0);
    std::vector<int> suffix_coverage(outer_extent, 0);
    Array<Stmt> result;
    for (size_t position = 0; position < events.size(); ++position) {
      const RegionCrossOuterEvent &event = events[position];
      if (static_cast<int64_t>(position) == first[event.outer]) {
        for (int index = 0; index < prefix_count; ++index) {
          result.push_back(
              rewrite_region_statement(region_body->seq[index], event.outer));
        }
        ++prefix_coverage[event.outer];
      }
      result.push_back(rewrite_core_event(event));
      if (static_cast<int64_t>(position) == last[event.outer]) {
        for (int index = 0; index < suffix_count; ++index) {
          result.push_back(rewrite_region_statement(
              region_body->seq[core_index + 1 + index], event.outer));
        }
        ++suffix_coverage[event.outer];
      }
    }
    for (int64_t outer_index = 0; outer_index < outer_extent; ++outer_index) {
      ICHECK_EQ(prefix_coverage[outer_index], 1)
          << "Prefix coverage failed for outer " << outer_index;
      ICHECK_EQ(suffix_coverage[outer_index], 1)
          << "Suffix coverage failed for outer " << outer_index;
    }
    ++injected_count_;
    boundary_overlap_ = true;
    return SeqStmt::Flatten(result);
  }

  int injected_count_{0};
  bool boundary_overlap_{false};
};

tvm::transform::Pass InjectSunmmioPipeline() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    const PrimFunc &original = f;
    try {
      PrimFunc candidate = f;
      auto *fptr = candidate.CopyOnWrite();
      fptr->body = SunmmioMultiVersionBufferRewriter::Substitute(candidate);
      fptr->body = SunmmioPipelineInjector::Inject(candidate);
      fptr->body = ConvertSSA(std::move(fptr->body));
      Optional<String> disallowed =
          PipelineFallbackValidator::FindDisallowed(fptr->body);
      if (disallowed) {
        return MakePipelineFunctionFallback(
            original, PipelineDiagnostic{false, "greedy", "inject_validation",
                                         "candidate_fallback",
                                         std::string(disallowed.value())});
      }
      return candidate;
    } catch (const std::exception &error) {
      return MakePipelineFunctionFallback(
          original,
          PipelineDiagnostic{false, "greedy", "inject_exception",
                             "candidate_rewrite_failed", error.what()});
    } catch (...) {
      return MakePipelineFunctionFallback(
          original,
          PipelineDiagnostic{false, "greedy", "inject_exception",
                             "candidate_rewrite_failed", "unknown exception"});
    }
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectSunmmioPipeline", {});
}

static tvm::transform::Pass
InjectSunmmioCrossOuterPipeline(String policy = "auto") {
  using namespace tir::transform;
  std::string policy_value = policy;
  ICHECK(policy_value == "auto" || policy_value == "forced")
      << "cross_outer policy must be 'auto' or 'forced', got " << policy_value;
  bool forced = policy_value == "forced";
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    const PrimFunc &original = f;
    try {
      PrimFunc candidate = f;
      PrimFuncNode *fptr = candidate.CopyOnWrite();
      fptr->body = SunmmioMultiVersionBufferRewriter::Substitute(candidate);
      int injected_count = 0;
      fptr->body =
          SunmmioCrossOuterPipelineInjector::Inject(candidate, &injected_count);
      if (injected_count == 0) {
        return original;
      }
      fptr->body = ConvertSSA(std::move(fptr->body));
      return candidate;
    } catch (const std::exception &error) {
      if (forced) {
        LOG(FATAL) << "[SunmmioPipeline][cross_outer][ForcedReject] "
                      "reason=candidate_rewrite_failed detail="
                   << error.what();
      }
      return MakeCrossOuterFunctionFallback(
          original, "candidate_rewrite_failed", error.what());
    } catch (...) {
      if (forced) {
        LOG(FATAL)
            << "[SunmmioPipeline][cross_outer][ForcedReject] "
               "reason=candidate_rewrite_failed detail=unknown_exception";
      }
      return MakeCrossOuterFunctionFallback(
          original, "candidate_rewrite_failed", "unknown exception");
    }
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectSunmmioCrossOuterPipeline",
                            {});
}

static tvm::transform::Pass
InjectSunmmioRegionCrossOuterPipeline(String policy = "auto",
                                      String strategy = "continuous") {
  using namespace tir::transform;
  std::string policy_value = policy;
  std::string strategy_value = strategy;
  ICHECK(policy_value == "auto" || policy_value == "forced")
      << "region_cross_outer policy must be 'auto' or 'forced', got "
      << policy_value;
  ICHECK(strategy_value == "conservative" || strategy_value == "continuous")
      << "region_cross_outer strategy must be 'conservative' or "
         "'continuous', got "
      << strategy_value;
  bool forced = policy_value == "forced";
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    const PrimFunc &original = f;
    bool planned = false;
    PostOrderVisit(f->body, [&](const ObjectRef &obj) {
      if (const auto *loop = obj.as<ForNode>()) {
        planned = planned || loop->annotations.count(kRegionCrossOuterMarker);
      }
    });
    if (!planned) {
      return original;
    }
    try {
      PrimFunc candidate = f;
      PrimFuncNode *fptr = candidate.CopyOnWrite();
      fptr->body = SunmmioMultiVersionBufferRewriter::Substitute(candidate);
      bool boundary_overlap = false;
      if (strategy_value == "conservative") {
        fptr->body = SunmmioPipelineInjector::Inject(candidate);
      } else {
        int injected_count = 0;
        fptr->body = SunmmioRegionCrossOuterPipelineInjector::Inject(
            candidate, &injected_count, &boundary_overlap);
        ICHECK_GT(injected_count, 0)
            << "region_cross_outer marker was not injected";
      }
      fptr->body = ConvertSSA(std::move(fptr->body));
      Optional<String> disallowed =
          PipelineFallbackValidator::FindDisallowed(fptr->body);
      if (disallowed) {
        throw std::runtime_error("candidate_fallback: " +
                                 std::string(disallowed.value()));
      }
      candidate = WithAttr(std::move(candidate),
                           "tl.sunmmio.region_cross_outer.strategy",
                           String(strategy_value));
      candidate = WithAttr(std::move(candidate),
                           "tl.sunmmio.region_cross_outer.coverage_verified",
                           Bool(true));
      candidate = WithAttr(std::move(candidate),
                           "tl.sunmmio.region_cross_outer.boundary_overlap",
                           Bool(boundary_overlap));
      return candidate;
    } catch (const std::exception &error) {
      if (forced) {
        LOG(FATAL) << "[SunmmioPipeline][region_cross_outer][ForcedReject] "
                      "reason=candidate_rewrite_failed detail="
                   << error.what();
      }
      return MakeRegionCrossOuterFunctionFallback(
          original, "candidate_rewrite_failed", error.what());
    } catch (...) {
      if (forced) {
        LOG(FATAL)
            << "[SunmmioPipeline][region_cross_outer][ForcedReject] "
               "reason=candidate_rewrite_failed detail=unknown_exception";
      }
      return MakeRegionCrossOuterFunctionFallback(
          original, "candidate_rewrite_failed", "unknown exception");
    }
  };
  return CreatePrimFuncPass(pass_func, 0,
                            "tl.InjectSunmmioRegionCrossOuterPipeline", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectSunmmioPipeline",
                        InjectSunmmioPipeline);
  refl::GlobalDef().def("tl.transform.InjectSunmmioCrossOuterPipeline",
                        InjectSunmmioCrossOuterPipeline);
  refl::GlobalDef().def("tl.transform.InjectSunmmioRegionCrossOuterPipeline",
                        InjectSunmmioRegionCrossOuterPipeline);
}

} // namespace tl
} // namespace tvm
