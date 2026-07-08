#include "../layout/cute_layout.h"
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

namespace tvm {
namespace tl {

using namespace tir;

struct LetWrapper {
  Var var;
  PrimExpr value;
};

int CeilDiv(int a, int b) {
  ICHECK_GT(b, 0);
  return (a + b - 1) / b;
}

void AppendUniqueBuffer(Array<Buffer> *buffers, const Buffer &buffer) {
  if (std::find(buffers->begin(), buffers->end(), buffer) == buffers->end()) {
    buffers->push_back(buffer);
  }
}

Array<Buffer> DeriveRuntimeMultiversionBuffers(
    const Optional<Any> &runtime_buffers_anno,
    const Optional<Any> &versioned_buffers_anno,
    const Array<Buffer> &banked_buffers, int iterations) {
  bool enable_banked_multiversion = iterations > 2;
  if (runtime_buffers_anno) {
    return Downcast<Array<Buffer>>(runtime_buffers_anno.value());
  }
  if (!versioned_buffers_anno) {
    Array<Buffer> runtime_buffers;
    if (enable_banked_multiversion) {
      for (const Buffer& buffer : banked_buffers) {
        AppendUniqueBuffer(&runtime_buffers, buffer);
      }
    }
    return runtime_buffers;
  }

  std::unordered_set<const BufferNode *> banked;
  for (const Buffer &buffer : banked_buffers) {
    banked.insert(buffer.get());
  }

  Array<Buffer> runtime_buffers;
  for (const Buffer &buffer :
       Downcast<Array<Buffer>>(versioned_buffers_anno.value())) {
    runtime_buffers.push_back(buffer);
  }
  if (enable_banked_multiversion) {
    for (const Buffer& buffer : banked_buffers) {
      if (!banked.count(buffer.get())) {
        continue;
      }
      AppendUniqueBuffer(&runtime_buffers, buffer);
    }
  }
  return runtime_buffers;
}

class SunmmioILPMultiVersionBufferRewriter : public StmtExprMutator {
public:
  SunmmioILPMultiVersionBufferRewriter(const PrimFunc &f) {
    for (const auto &kv : f->buffer_map) {
      const Buffer &buffer = kv.second;
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
  }

  static Stmt Substitute(PrimFunc &f) {
    SunmmioILPMultiVersionBufferRewriter substituter(f);
    // collect used_buffers and iterations
    substituter.VisitStmt(f->body);
    substituter.replace_flag = true;

    for (auto &buffer : substituter.versioned_buffers_) {
      int num_versions = substituter.GetVersionCount(buffer);
      if (substituter.IsBankedBuffer(buffer)) {
        Buffer ping_buffer =
            substituter.makeRuntimeBuffer(buffer, num_versions, "_ping", true);
        Buffer pong_buffer =
            substituter.makeRuntimeBuffer(buffer, num_versions, "_pong", false);
        substituter.buffer_remap_.Set(buffer, ping_buffer);
        substituter.bank_peer_buffers_[buffer.get()] = pong_buffer;
      } else {
        substituter.buffer_remap_.Set(
            buffer, substituter.makeRuntimeBuffer(buffer, num_versions));
      }
    }

    substituter.RewriteFunctionLayoutAttrs(f);
    substituter.RecordDefaultPingPongAttrs(f);

    f.CopyOnWrite()->body =
        RemapBufferRewriter::Substitute(f->body, substituter.buffer_remap_);

    return substituter.VisitStmt(f->body);
  }

private:
  void RecordDefaultPingPongAttrs(PrimFunc &f) {
    if (buffer_remap_.empty()) {
      return;
    }

    Map<Var, String> alloc_ping_pong;
    for (const auto &kv : bank_peer_buffers_) {
      const Buffer &peer_buffer = kv.second;
      alloc_ping_pong.Set(peer_buffer->data, String("pong"));
    }

    if (alloc_ping_pong.empty()) {
      return;
    }

    f = WithAttr(std::move(f), tl::attr::kSunmmioAllocPingPong,
                 alloc_ping_pong);
  }

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
      Optional<Layout> derived_layout = DeriveLayoutLike(
          layout, new_buffer->shape, Optional<Array<Integer>>(), &analyzer);
      ICHECK(derived_layout.defined())
          << "Failed to derive ILP multiversioned layout for buffer "
          << buffer->name << " with shape " << new_buffer->shape;
      new_layout_map.Set(new_buffer, derived_layout.value());
      auto peer_it = bank_peer_buffers_.find(buffer.get());
      if (peer_it != bank_peer_buffers_.end()) {
        const Buffer &peer_buffer = peer_it->second;
        Optional<Layout> peer_layout = DeriveLayoutLike(
            layout, peer_buffer->shape, Optional<Array<Integer>>(), &analyzer);
        ICHECK(peer_layout.defined())
            << "Failed to derive ILP ping/pong layout for buffer "
            << buffer->name << " with shape " << peer_buffer->shape;
        new_layout_map.Set(peer_buffer, peer_layout.value());
      }
    }
    f = WithAttr(std::move(f), attr::kLayoutMap, new_layout_map);
  }

  bool HasVersionAxis(const Buffer &buffer) const {
    return version_axis_buffers_.count(buffer.get()) != 0;
  }

  bool IsBankedBuffer(const Buffer &buffer) const {
    return banked_buffers_.count(buffer.get()) != 0;
  }

  int GetVersionCount(const Buffer &buffer) const {
    auto it = buffer_versions_.find(buffer.get());
    int num_versions = it == buffer_versions_.end() ? 1 : it->second;
    if (HasVersionAxis(buffer) && IsBankedBuffer(buffer)) {
      return CeilDiv(num_versions, 2);
    }
    return num_versions;
  }

  Array<PrimExpr> AddDefaultRuntimeAxes(const Buffer &buffer,
                                        Array<PrimExpr> indices) const {
    if (HasVersionAxis(buffer)) {
      indices.insert(indices.begin(), Integer(0));
    }
    return indices;
  }

  Array<Range> AddDefaultRuntimeAxes(const Buffer &buffer,
                                     Array<Range> ranges) const {
    if (HasVersionAxis(buffer)) {
      ranges.insert(ranges.begin(), Range::FromMinExtent(0, 1));
    }
    return ranges;
  }

  Buffer makeRuntimeBuffer(const Buffer &buffer, int num_version,
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
    if (HasVersionAxis(buffer)) {
      shape.insert(shape.begin(), num_version);
    }
    return Buffer(new_var, buffer->dtype, shape, {}, buffer->elem_offset,
                  String(buffer_name), buffer->data_alignment,
                  buffer->offset_factor, buffer->buffer_type);
  }

  BufferRegion
  RewritePipelineBufferRegion(const BufferRegion &buffer_region) const {
    auto it = buffer_remap_.find(buffer_region->buffer);
    if (it != buffer_remap_.end()) {
      Region new_region = buffer_region->region;
      if (HasVersionAxis(buffer_region->buffer)) {
        new_region.insert(new_region.begin(), Range::FromMinExtent(0, 1));
      }
      const Buffer &new_buffer = (*it).second;
      return BufferRegion(new_buffer, new_region);
    }
    return buffer_region;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    auto runtime_buffers_anno =
        op->annotations.Get("runtime_multiversion_buffers");
    auto versioned_buffers_anno = op->annotations.Get("versioned_buffers");
    auto banked_buffers_anno = op->annotations.Get("runtime_banked_buffers");
    auto used_buffers_anno = op->annotations.Get("used_buffers");
    auto iterations_anno = op->annotations.Get("iterations");
    auto bank_start_phases_anno =
        op->annotations.Get("runtime_bank_start_phases");
    auto bank_read_delta_parities_anno =
        op->annotations.Get("runtime_bank_read_delta_parities");
    auto bank_writer_phases_anno =
        op->annotations.Get("runtime_bank_writer_phases");
    auto bank_reader_phases_anno =
        op->annotations.Get("runtime_bank_reader_phases");
    if (used_buffers_anno && iterations_anno &&
        (runtime_buffers_anno || versioned_buffers_anno)) {
      Array<Buffer> banked_buffers;
      if (banked_buffers_anno) {
        banked_buffers = Downcast<Array<Buffer>>(banked_buffers_anno.value());
      }
      int iterations = Downcast<IntImm>(iterations_anno.value())->value;
      Array<Buffer> runtime_buffers = DeriveRuntimeMultiversionBuffers(
          runtime_buffers_anno, versioned_buffers_anno, banked_buffers,
          iterations);
      if (!replace_flag) {
        for (const Buffer &buffer : runtime_buffers) {
          AppendUniqueBuffer(&versioned_buffers_, buffer);
          version_axis_buffers_.insert(buffer.get());
          int &num_versions = buffer_versions_[buffer.get()];
          num_versions = std::max(num_versions, iterations);
        }
        for (const Buffer &buffer : banked_buffers) {
          AppendUniqueBuffer(&versioned_buffers_, buffer);
          banked_buffers_.insert(buffer.get());
        }
      } else {
        Array<Buffer> new_runtime_buffers;
        for (const Buffer &buffer : runtime_buffers) {
          if (buffer_remap_.count(buffer)) {
            new_runtime_buffers.push_back(buffer_remap_[buffer]);
          } else {
            new_runtime_buffers.push_back(buffer);
          }
        }
        loop.CopyOnWrite()->annotations.Set("runtime_multiversion_buffers",
                                            new_runtime_buffers);
        if (versioned_buffers_anno) {
          Array<Buffer> versioned_buffers =
              Downcast<Array<Buffer>>(versioned_buffers_anno.value());
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
        }
        if (banked_buffers_anno) {
          Array<Buffer> banked_buffers =
              Downcast<Array<Buffer>>(banked_buffers_anno.value());
          Array<Buffer> new_banked_buffers;
          for (const Buffer &buffer : banked_buffers) {
            if (buffer_remap_.count(buffer)) {
              new_banked_buffers.push_back(buffer_remap_[buffer]);
            } else {
              new_banked_buffers.push_back(buffer);
            }
          }
          loop.CopyOnWrite()->annotations.Set("runtime_banked_buffers",
                                              new_banked_buffers);
        }
        if (bank_start_phases_anno) {
          Map<Buffer, PrimExpr> bank_start_phases =
              Downcast<Map<Buffer, PrimExpr>>(bank_start_phases_anno.value());
          Map<Buffer, PrimExpr> new_bank_start_phases;
          for (const auto &[buffer, phase] : bank_start_phases) {
            if (buffer_remap_.count(buffer)) {
              new_bank_start_phases.Set(buffer_remap_[buffer], phase);
            } else {
              new_bank_start_phases.Set(buffer, phase);
            }
          }
          loop.CopyOnWrite()->annotations.Set("runtime_bank_start_phases",
                                              new_bank_start_phases);
        }
        if (bank_read_delta_parities_anno) {
          Map<Buffer, PrimExpr> bank_read_delta_parities =
              Downcast<Map<Buffer, PrimExpr>>(bank_read_delta_parities_anno.value());
          Map<Buffer, PrimExpr> new_bank_read_delta_parities;
          for (const auto &[buffer, parity] : bank_read_delta_parities) {
            if (buffer_remap_.count(buffer)) {
              new_bank_read_delta_parities.Set(buffer_remap_[buffer], parity);
            } else {
              new_bank_read_delta_parities.Set(buffer, parity);
            }
          }
          loop.CopyOnWrite()->annotations.Set("runtime_bank_read_delta_parities",
                                              new_bank_read_delta_parities);
        }
        if (bank_writer_phases_anno) {
          Map<Buffer, Map<Integer, PrimExpr>> bank_writer_phases =
              Downcast<Map<Buffer, Map<Integer, PrimExpr>>>(
                  bank_writer_phases_anno.value());
          Map<Buffer, Map<Integer, PrimExpr>> new_bank_writer_phases;
          for (const auto &[buffer, per_op] : bank_writer_phases) {
            if (buffer_remap_.count(buffer)) {
              new_bank_writer_phases.Set(buffer_remap_[buffer], per_op);
            } else {
              new_bank_writer_phases.Set(buffer, per_op);
            }
          }
          loop.CopyOnWrite()->annotations.Set("runtime_bank_writer_phases",
                                              new_bank_writer_phases);
        }
        if (bank_reader_phases_anno) {
          Map<Buffer, Map<Integer, PrimExpr>> bank_reader_phases =
              Downcast<Map<Buffer, Map<Integer, PrimExpr>>>(
                  bank_reader_phases_anno.value());
          Map<Buffer, Map<Integer, PrimExpr>> new_bank_reader_phases;
          for (const auto &[buffer, per_op] : bank_reader_phases) {
            if (buffer_remap_.count(buffer)) {
              new_bank_reader_phases.Set(buffer_remap_[buffer], per_op);
            } else {
              new_bank_reader_phases.Set(buffer, per_op);
            }
          }
          loop.CopyOnWrite()->annotations.Set("runtime_bank_reader_phases",
                                              new_bank_reader_phases);
        }
        if (banked_buffers_anno) {
          Array<Buffer> banked_buffers =
              Downcast<Array<Buffer>>(banked_buffers_anno.value());
          Map<Buffer, Buffer> runtime_bank_peer_buffers;
          for (const Buffer &buffer : banked_buffers) {
            auto remap_it = buffer_remap_.find(buffer);
            auto peer_it = bank_peer_buffers_.find(buffer.get());
            if (remap_it != buffer_remap_.end() &&
                peer_it != bank_peer_buffers_.end()) {
              runtime_bank_peer_buffers.Set((*remap_it).second,
                                            peer_it->second);
            }
          }
          loop.CopyOnWrite()->annotations.Set("runtime_bank_peer_buffers",
                                              runtime_bank_peer_buffers);
        }
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
      if (remap_it != buffer_remap_.end()) {
        alloc_buffers.push_back((*remap_it).second);
        auto peer_it = bank_peer_buffers_.find(buf.get());
        if (peer_it != bank_peer_buffers_.end()) {
          alloc_buffers.push_back(peer_it->second);
        }
      } else {
        alloc_buffers.push_back(buf);
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
      auto indices = AddDefaultRuntimeAxes(buffer, load->indices);
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
      auto indices = AddDefaultRuntimeAxes(buffer, store->indices);
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
      Array<Range> new_ranges =
          AddDefaultRuntimeAxes(original_buffer, original_region->GetRanges());

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
  std::unordered_map<const BufferNode *, int> buffer_versions_;
  std::unordered_set<const BufferNode *> version_axis_buffers_;
  std::unordered_set<const BufferNode *> banked_buffers_;
  std::unordered_map<const BufferNode *, Buffer> bank_peer_buffers_;
};

class SunmmioILPPipelineBodyRewriter : public StmtExprMutator {
public:
  SunmmioILPPipelineBodyRewriter(Array<Buffer> runtime_buffers,
                                 Array<Buffer> version_axis_buffers,
                                 Array<Buffer> banked_buffers,
                                 Map<Buffer, Buffer> bank_peer_buffers,
                                 Map<Buffer, PrimExpr> bank_start_phases,
                                 Map<Buffer, PrimExpr> bank_read_delta_parities,
                                 Map<Buffer, Map<Integer, PrimExpr>> bank_writer_phases,
                                 Map<Buffer, Map<Integer, PrimExpr>> bank_reader_phases,
                                 For pipeline_loop,
                                 int iterations) {
    pipeline_loop_ = std::move(pipeline_loop);
    iterations_ = iterations;
    for (const Buffer &it : runtime_buffers) {
      rewritten_buffers_.insert(it.get());
      buffer_data_to_buffer_.Set(it->data, it);
    }
    for (const Buffer &it : version_axis_buffers) {
      version_axis_buffers_.insert(it.get());
    }
    for (const Buffer &it : banked_buffers) {
      banked_buffers_.insert(it.get());
    }
    for (const auto &[buffer, peer] : bank_peer_buffers) {
      bank_peer_buffers_[buffer.get()] = peer;
      buffer_data_to_buffer_.Set(peer->data, peer);
    }
    for (const auto &[buffer, phase] : bank_start_phases) {
      if (const auto* imm = phase.as<IntImmNode>()) {
        buffer_bank_start_phase_[buffer.get()] = imm->value;
      }
    }
    for (const auto &[buffer, parity] : bank_read_delta_parities) {
      if (const auto* imm = parity.as<IntImmNode>()) {
        buffer_read_delta_parity_[buffer.get()] = imm->value & 1;
      }
    }
    for (const auto &[buffer, per_op] : bank_writer_phases) {
      auto& dst = buffer_writer_phase_[buffer.get()];
      for (const auto &[op_id, phase] : per_op) {
        if (const auto* op_imm = op_id.as<IntImmNode>()) {
          if (const auto* phase_imm = phase.as<IntImmNode>()) {
            dst[op_imm->value] = phase_imm->value;
          }
        }
      }
    }
    for (const auto &[buffer, per_op] : bank_reader_phases) {
      auto& dst = buffer_reader_phase_[buffer.get()];
      for (const auto &[op_id, phase] : per_op) {
        if (const auto* op_imm = op_id.as<IntImmNode>()) {
          if (const auto* phase_imm = phase.as<IntImmNode>()) {
            dst[op_imm->value] = phase_imm->value;
          }
        }
      }
    }
  }

  void set_current_stmt_id(int stmt_id) { current_stmt_id_ = stmt_id; }

  void set_loop_var_replacement(PrimExpr p) { replaced_loop_var_ = p; }
  void set_logical_iter_parity_override(int parity) {
    logical_iter_parity_override_ = parity;
  }
  void clear_logical_iter_parity_override() {
    logical_iter_parity_override_ = -1;
  }
  void set_pipeline_loop_parity_override(int parity) {
    pipeline_loop_parity_override_ = parity;
  }
  void clear_pipeline_loop_parity_override() {
    pipeline_loop_parity_override_ = -1;
  }
  void clear_parity_overrides() {
    clear_logical_iter_parity_override();
    clear_pipeline_loop_parity_override();
  }

private:
  int VersionAxis(const Buffer& buffer) const {
    return version_axis_buffers_.count(buffer.get()) ? 0 : -1;
  }

  PrimExpr LogicalIterExpr() const {
    return replaced_loop_var_ - pipeline_loop_->min;
  }

  PrimExpr EffectiveVersionExpr(const Buffer& buffer) const {
    if (!version_axis_buffers_.count(buffer.get())) {
      return Integer(0);
    }
    PrimExpr logical_iter = LogicalIterExpr();
    if (banked_buffers_.count(buffer.get())) {
      int num_versions = CeilDiv(iterations_, 2);
      return floormod(floordiv(logical_iter, Integer(2)),
                      Integer(num_versions));
    }
    return floormod(logical_iter, Integer(iterations_));
  }

  int ResolveLogicalIterParity() const {
    if (logical_iter_parity_override_ >= 0) {
      return logical_iter_parity_override_;
    }
    if (pipeline_loop_parity_override_ >= 0) {
      arith::Analyzer analyzer;
      PrimExpr iter_offset =
          analyzer.Simplify(LogicalIterExpr() - pipeline_loop_->loop_var);
      if (const auto* imm = iter_offset.as<IntImmNode>()) {
        int parity = (pipeline_loop_parity_override_ + imm->value) % 2;
        return parity < 0 ? parity + 2 : parity;
      }
    }
    arith::Analyzer analyzer;
    PrimExpr simplified = analyzer.Simplify(LogicalIterExpr());
    if (const auto* imm = simplified.as<IntImmNode>()) {
      int parity = imm->value % 2;
      return parity < 0 ? parity + 2 : parity;
    }
    return -1;
  }

  Buffer ResolveTargetBuffer(const Buffer& buffer, bool is_read = false) const {
    if (!banked_buffers_.count(buffer.get())) {
      return buffer;
    }
    int logical_iter_parity = ResolveLogicalIterParity();
    ICHECK_GE(logical_iter_parity, 0)
        << "Dynamic bank parity must resolve before selecting ping/pong for "
        << buffer->name;
    int bank = -1;
    if (current_stmt_id_ >= 0) {
      if (is_read) {
        auto it_buf = buffer_reader_phase_.find(buffer.get());
        if (it_buf != buffer_reader_phase_.end()) {
          auto it_stmt = it_buf->second.find(current_stmt_id_);
          if (it_stmt != it_buf->second.end()) {
            bank = (logical_iter_parity + it_stmt->second) % 2;
          }
        }
      } else {
        auto it_buf = buffer_writer_phase_.find(buffer.get());
        if (it_buf != buffer_writer_phase_.end()) {
          auto it_stmt = it_buf->second.find(current_stmt_id_);
          if (it_stmt != it_buf->second.end()) {
            bank = (logical_iter_parity + it_stmt->second) % 2;
          }
        }
      }
    }
    if (bank < 0) {
      int start_phase = 0;
      auto it = buffer_bank_start_phase_.find(buffer.get());
      if (it != buffer_bank_start_phase_.end()) {
        start_phase = it->second;
      }
      int read_delta_parity = 0;
      if (is_read) {
        auto it_delta = buffer_read_delta_parity_.find(buffer.get());
        if (it_delta != buffer_read_delta_parity_.end()) {
          read_delta_parity = it_delta->second;
        }
      }
      bank = (logical_iter_parity + start_phase + read_delta_parity) % 2;
    }
    if (bank < 0) {
      bank += 2;
    }
    if (bank == 0) {
      return buffer;
    }
    auto peer_it = bank_peer_buffers_.find(buffer.get());
    ICHECK(peer_it != bank_peer_buffers_.end())
        << "Missing peer buffer for banked runtime buffer " << buffer->name;
    return peer_it->second;
  }

  PrimExpr RewriteBufferAccess(const Call &call,
                               const std::vector<int> &arg_indices) {
    auto product = [](const Array<PrimExpr> &input) {
      return foldl(
          [](PrimExpr a, PrimExpr b, Span span) {
            return mul(std::move(a), std::move(b), std::move(span));
          },
          make_const(DataType::Int(32), 1), input);
    };
    auto axis_stride = [&](const Buffer& buffer, int axis) {
      if (!buffer->strides.empty()) {
        return buffer->strides[axis];
      }
      Array<PrimExpr> suffix;
      for (size_t j = axis + 1; j < buffer->shape.size(); ++j) {
        suffix.push_back(buffer->shape[j]);
      }
      return product(suffix);
    };
    Array<PrimExpr> new_args = call->args;
    for (int i : arg_indices) {
      Var buffer_data = Downcast<Var>(call->args[i]);
      if (!buffer_data_to_buffer_.count(buffer_data)) {
        continue;
      }
      const Buffer &buffer = buffer_data_to_buffer_[buffer_data];
      if (!rewritten_buffers_.count(buffer.get())) {
        continue;
      }
      Buffer target_buffer = ResolveTargetBuffer(buffer, /*is_read=*/false);
      PrimExpr new_index = call->args[i + 1];
      int version_axis = VersionAxis(buffer);
      if (version_axis >= 0) {
        PrimExpr offset = axis_stride(target_buffer, version_axis);
        new_index = new_index + EffectiveVersionExpr(buffer) * offset;
      }
      new_args.Set(i, target_buffer->data);
      new_args.Set(i + 1, new_index);
    }
    return Call(call->dtype, call->op, new_args, call->annotations, call->span);
  }

  PrimExpr RewriteRegionExpr(const Call& call) {
    RegionOp original_region(call->args);
    Buffer original_buffer = original_region->GetBuffer();
    if (!rewritten_buffers_.count(original_buffer.get())) {
      return call;
    }

    Buffer target_buffer = ResolveTargetBuffer(original_buffer, /*is_read=*/true);
    Array<Range> new_ranges = original_region->GetRanges();
    int version_axis = VersionAxis(original_buffer);
    if (version_axis >= 0) {
      new_ranges.Set(version_axis,
                     Range::FromMinExtent(EffectiveVersionExpr(original_buffer), 1));
    }
    return MakeRegionExpr(target_buffer, new_ranges,
                          original_region->GetAccessMask());
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
    int prev_stmt_id = current_stmt_id_;
    current_stmt_id_ = logical_stmt_cursor_;
    logical_stmt_cursor_ += 1;
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    current_stmt_id_ = prev_stmt_id;
    if (!rewritten_buffers_.count(store->buffer.get())) {
      return store;
    }
    Buffer target_buffer = ResolveTargetBuffer(store->buffer, /*is_read=*/false);
    Array<PrimExpr> indices = store->indices;
    int version_axis = VersionAxis(store->buffer);
    if (version_axis >= 0) {
      indices.Set(version_axis, EffectiveVersionExpr(store->buffer));
    }
    return BufferStore(target_buffer, store->value, indices);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    int prev_stmt_id = current_stmt_id_;
    current_stmt_id_ = logical_stmt_cursor_;
    logical_stmt_cursor_ += 1;
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    current_stmt_id_ = prev_stmt_id;
    if (!rewritten_buffers_.count(load->buffer.get())) {
      return load;
    }
    Buffer target_buffer = ResolveTargetBuffer(load->buffer, /*is_read=*/true);
    Array<PrimExpr> indices = load->indices;
    int version_axis = VersionAxis(load->buffer);
    if (version_axis >= 0) {
      indices.Set(version_axis, EffectiveVersionExpr(load->buffer));
    }
    return BufferLoad(target_buffer, indices);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (call->op.same_as(builtin::tvm_access_ptr())) {
      return RewriteBufferAccess(call, {1});
    }
    if (call->op.same_as(RegionOp::Get())) {
      return RewriteRegionExpr(call);
    }
    return call;
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    Var var = Downcast<Var>(StmtExprMutator::VisitExpr_(op));
    if (var.same_as(pipeline_loop_->loop_var)) {
      return replaced_loop_var_;
    }
    return var;
  }

  std::unordered_set<const BufferNode *> rewritten_buffers_;
  std::unordered_set<const BufferNode *> version_axis_buffers_;
  std::unordered_set<const BufferNode *> banked_buffers_;
  std::unordered_map<const BufferNode *, Buffer> bank_peer_buffers_;
  std::unordered_map<const BufferNode *, int> buffer_bank_start_phase_;
  std::unordered_map<const BufferNode *, int> buffer_read_delta_parity_;
  std::unordered_map<const BufferNode *, std::unordered_map<int, int>> buffer_writer_phase_;
  std::unordered_map<const BufferNode *, std::unordered_map<int, int>> buffer_reader_phase_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  For pipeline_loop_;
  int iterations_ = 1;
  PrimExpr replaced_loop_var_;
  int logical_iter_parity_override_ = -1;
  int pipeline_loop_parity_override_ = -1;
  int current_stmt_id_ = -1;
  int logical_stmt_cursor_ = 0;
};

class SunmmioILPPipelineInjector : public StmtExprMutator {
public:
  static Stmt Inject(const PrimFunc &func) {
    auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    SunmmioILPPipelineInjector injector(global_symbol, func);
    for (const auto &kv : func->buffer_map) {
      const Buffer &buffer = kv.second;
      injector.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    return injector(func->body);
  }

private:
  explicit SunmmioILPPipelineInjector(Optional<String> global_symbol,
                                      const PrimFunc &f)
      : global_symbol_(std::move(global_symbol)), traverser_(f) {
    traverser_.clear();
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // Step 1: Recursively rewrite the children first.
    For for_node = Downcast<For>(StmtExprMutator::VisitStmt_(op));

    auto iterations_anno = op->annotations.Get("iterations");
    auto ii_anno = op->annotations.Get("ii");
    auto makespan_anno = op->annotations.Get("makespan");
    auto stage_count_anno = op->annotations.Get("stage_count");
    auto steady_state_max_iter_offset_anno =
        op->annotations.Get("steady_state_max_iter_offset");
    auto used_buffers_anno = op->annotations.Get("used_buffers");
    auto runtime_buffers_anno =
        op->annotations.Get("runtime_multiversion_buffers");
    auto versioned_buffers_anno = op->annotations.Get("versioned_buffers");
    auto banked_buffers_anno =
        op->annotations.Get("runtime_banked_buffers");
    auto bank_start_phases_anno =
        op->annotations.Get("runtime_bank_start_phases");
    auto bank_read_delta_parities_anno =
        op->annotations.Get("runtime_bank_read_delta_parities");
    auto bank_writer_phases_anno =
        op->annotations.Get("runtime_bank_writer_phases");
    auto bank_reader_phases_anno =
        op->annotations.Get("runtime_bank_reader_phases");
    auto bank_peer_buffers_anno =
        op->annotations.Get("runtime_bank_peer_buffers");
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
    int ii = ii_anno ? Downcast<IntImm>(ii_anno.value())->value : iterations;
    int makespan =
        makespan_anno ? Downcast<IntImm>(makespan_anno.value())->value : -1;
    int stage_count =
        stage_count_anno ? Downcast<IntImm>(stage_count_anno.value())->value : -1;
    int steady_state_max_iter_offset =
        steady_state_max_iter_offset_anno
            ? Downcast<IntImm>(steady_state_max_iter_offset_anno.value())->value
            : 0;
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
    Array<Buffer> banked_buffers;
    if (banked_buffers_anno) {
      banked_buffers = Downcast<Array<Buffer>>(banked_buffers_anno.value());
    }
    Array<Buffer> runtime_buffers = DeriveRuntimeMultiversionBuffers(
        runtime_buffers_anno, versioned_buffers_anno, banked_buffers,
        iterations);
    Array<Buffer> rewritten_buffers = runtime_buffers;
    for (const Buffer& buffer : banked_buffers) {
      AppendUniqueBuffer(&rewritten_buffers, buffer);
    }
    Map<Buffer, PrimExpr> bank_start_phases;
    if (bank_start_phases_anno) {
      bank_start_phases =
          Downcast<Map<Buffer, PrimExpr>>(bank_start_phases_anno.value());
    }
    Map<Buffer, PrimExpr> bank_read_delta_parities;
    if (bank_read_delta_parities_anno) {
      bank_read_delta_parities =
          Downcast<Map<Buffer, PrimExpr>>(bank_read_delta_parities_anno.value());
    }
    Map<Buffer, Map<Integer, PrimExpr>> bank_writer_phases;
    if (bank_writer_phases_anno) {
      bank_writer_phases =
          Downcast<Map<Buffer, Map<Integer, PrimExpr>>>(
              bank_writer_phases_anno.value());
    }
    Map<Buffer, Map<Integer, PrimExpr>> bank_reader_phases;
    if (bank_reader_phases_anno) {
      bank_reader_phases =
          Downcast<Map<Buffer, Map<Integer, PrimExpr>>>(
              bank_reader_phases_anno.value());
    }
    Map<Buffer, Buffer> bank_peer_buffers;
    if (bank_peer_buffers_anno) {
      bank_peer_buffers =
          Downcast<Map<Buffer, Buffer>>(bank_peer_buffers_anno.value());
    }

    auto rewriter = SunmmioILPPipelineBodyRewriter(
        rewritten_buffers, runtime_buffers, banked_buffers, bank_peer_buffers,
        bank_start_phases, bank_read_delta_parities,
        bank_writer_phases, bank_reader_phases, for_node, iterations);
    arith::Analyzer analyzer;
    auto rewrite_stmt_with_logical_iter_parity =
        [&](const Stmt& stmt, const PrimExpr& replaced_loop_var,
            int parity) -> Stmt {
      rewriter.set_loop_var_replacement(replaced_loop_var);
      rewriter.clear_pipeline_loop_parity_override();
      if (parity >= 0) {
        rewriter.set_logical_iter_parity_override(parity);
      } else {
        rewriter.clear_logical_iter_parity_override();
      }
      Stmt rewritten = rewriter(stmt);
      rewriter.clear_parity_overrides();
      return rewritten;
    };
    auto rewrite_stmt_with_pipeline_loop_parity =
        [&](const Stmt& stmt, const PrimExpr& replaced_loop_var,
            int parity) -> Stmt {
      rewriter.set_loop_var_replacement(replaced_loop_var);
      rewriter.clear_logical_iter_parity_override();
      if (parity >= 0) {
        rewriter.set_pipeline_loop_parity_override(parity);
      } else {
        rewriter.clear_pipeline_loop_parity_override();
      }
      Stmt rewritten = rewriter(stmt);
      rewriter.clear_parity_overrides();
      return rewritten;
    };
    auto rewrite_stmt = [&](const Stmt& stmt,
                            const PrimExpr& replaced_loop_var) -> Stmt {
      if (banked_buffers.size() == 0) {
        return rewrite_stmt_with_logical_iter_parity(stmt, replaced_loop_var, -1);
      }

      PrimExpr logical_iter =
          analyzer.Simplify(replaced_loop_var - for_node->min);
      if (logical_iter.as<IntImmNode>()) {
        return rewrite_stmt_with_logical_iter_parity(stmt, replaced_loop_var, -1);
      }

      Stmt even_stmt =
          rewrite_stmt_with_logical_iter_parity(stmt, replaced_loop_var, 0);
      Stmt odd_stmt =
          rewrite_stmt_with_logical_iter_parity(stmt, replaced_loop_var, 1);

      PrimExpr is_even =
          EQ(floormod(replaced_loop_var - for_node->min, Integer(2)),
             Integer(0));
      return IfThenElse(is_even, even_stmt, odd_stmt);
    };
    Array<Stmt> for_body;
    // Step 3.1: Rewrite prologue
    for (const auto &order_str : prologue_orders) {
      int iter = name2iter(order_str);
      int id = name2id(order_str);
      Stmt stmt = pipeline_body_seq->seq[id];
      PrimExpr replaced_loop_var = 0 + iter + for_node->min;
      stmt = rewrite_stmt(stmt, replaced_loop_var);
      for_body.push_back(stmt);
    }

    // Step 3.2: Rewrite the for body of loop.
    Array<Stmt> body;
    for (const auto &order_str : body_orders) {
      int iter_offset = name2iter(order_str);
      PrimExpr replaced_loop_var =
          for_node->loop_var + iter_offset + for_node->min;
      int id = name2id(order_str);
      Stmt stmt = pipeline_body_seq->seq[id];
      body.push_back(stmt);
    }

    if (banked_buffers.size() != 0) {
      bool body_has_dynamic_parity = false;
      for (const auto &order_str : body_orders) {
        int iter_offset = name2iter(order_str);
        PrimExpr replaced_loop_var =
            for_node->loop_var + iter_offset + for_node->min;
        PrimExpr logical_iter =
            analyzer.Simplify(replaced_loop_var - for_node->min);
        if (!logical_iter.as<IntImmNode>()) {
          body_has_dynamic_parity = true;
          break;
        }
      }
      if (body_has_dynamic_parity) {
        Array<Stmt> even_body;
        Array<Stmt> odd_body;
        for (const auto &order_str : body_orders) {
          int iter_offset = name2iter(order_str);
          PrimExpr replaced_loop_var =
              for_node->loop_var + iter_offset + for_node->min;
          int id = name2id(order_str);
          Stmt stmt = pipeline_body_seq->seq[id];
          even_body.push_back(
              rewrite_stmt_with_pipeline_loop_parity(stmt, replaced_loop_var, 0));
          odd_body.push_back(
              rewrite_stmt_with_pipeline_loop_parity(stmt, replaced_loop_var, 1));
        }
        PrimExpr steady_state_is_even = EQ(floormod(for_node->loop_var, Integer(2)),
                                           Integer(0));
        body = {IfThenElse(steady_state_is_even, SeqStmt::Flatten(even_body),
                           SeqStmt::Flatten(odd_body))};
      } else {
        Array<Stmt> rewritten_body;
        for (const auto &order_str : body_orders) {
          int iter_offset = name2iter(order_str);
          PrimExpr replaced_loop_var =
              for_node->loop_var + iter_offset + for_node->min;
          int id = name2id(order_str);
          Stmt stmt = pipeline_body_seq->seq[id];
          rewritten_body.push_back(rewrite_stmt(stmt, replaced_loop_var));
        }
        body = rewritten_body;
      }
    } else {
      Array<Stmt> rewritten_body;
      for (const auto &order_str : body_orders) {
        int iter_offset = name2iter(order_str);
        PrimExpr replaced_loop_var =
            for_node->loop_var + iter_offset + for_node->min;
        int id = name2id(order_str);
        Stmt stmt = pipeline_body_seq->seq[id];
        rewritten_body.push_back(rewrite_stmt(stmt, replaced_loop_var));
      }
      body = rewritten_body;
    }

    PrimExpr extent =
        stage_count > 0 ? max(0, for_node->extent + 1 - stage_count)
                        : max(0, for_node->extent - steady_state_max_iter_offset);
    For new_for_stmt =
        For(for_node->loop_var, PrimExpr(0), extent, ForKind::kSerial,
            SeqStmt::Flatten(body), for_node->thread_binding, {});
    for_body.push_back(new_for_stmt);

    // Step 3.3: Rewrite the epilogue.
    PrimExpr epilogue_base =
        max(0, extent - (iterations - 1)) + for_node->min;
    for (const auto &order_str : epilogue_orders) {
      int iter_offset = name2iter(order_str);
      int id = name2id(order_str);
      Stmt stmt = pipeline_body_seq->seq[id];
      PrimExpr replaced_loop_var = epilogue_base + iter_offset;
      stmt = rewrite_stmt(stmt, replaced_loop_var);
      for_body.push_back(stmt);
    }
    return SeqStmt::Flatten(for_body);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  Optional<String> global_symbol_;
  ASTTraverser traverser_;
};

tvm::transform::Pass InjectSunmmioPipelineILP() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    auto *fptr = f.CopyOnWrite();
    fptr->body = SunmmioILPMultiVersionBufferRewriter::Substitute(f);
    fptr->body = SunmmioILPPipelineInjector::Inject(f);
    fptr->body = ConvertSSA(std::move(fptr->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectSunmmioPipelineILP", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectSunmmioPipelineILP",
                        InjectSunmmioPipelineILP);
}

} // namespace tl
} // namespace tvm
