/*!
 * \file resolve_sunmmio_mesh_symbols.cc
 * \brief Resolve symbolic Sunmmio mesh dimensions from the bound target.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>

#include "../layout/cute_layout.h"
#include "../layout/layout.h"
#include "../target/sunmmio_utils.h"
#include "../target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

PrimExpr I32Imm(int64_t value) { return IntImm(DataType::Int(32), value); }

class SunmmioMeshSymbolResolver : public StmtExprMutator {
public:
  explicit SunmmioMeshSymbolResolver(Target target) {
    auto mesh = GetSunmmioMeshConfig(target);
    nrows_ = mesh.nrow;
    ncols_ = mesh.ncol;
  }

  Stmt ResolveStmt(const Stmt &stmt) { return VisitStmt(stmt); }

  PrimExpr VisitExpr(const PrimExpr &expr) final {
    return analyzer_.Simplify(StmtExprMutator::VisitExpr(expr));
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(Op::Get("tl.mesh_nrows"))) {
      return I32Imm(nrows_);
    }
    if (op->op.same_as(Op::Get("tl.mesh_ncols"))) {
      return I32Imm(ncols_);
    }
    if (op->op.same_as(Op::Get("tl.mesh_ncores"))) {
      return I32Imm(nrows_ * ncols_);
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Buffer ResolveBuffer(const Buffer &buffer) {
    auto it = buffer_cache_.find(buffer.get());
    if (it != buffer_cache_.end()) {
      return it->second;
    }

    Array<PrimExpr> shape;
    shape.reserve(buffer->shape.size());
    for (const auto &extent : buffer->shape) {
      shape.push_back(VisitExpr(extent));
    }

    Array<PrimExpr> strides;
    strides.reserve(buffer->strides.size());
    for (const auto &stride : buffer->strides) {
      strides.push_back(VisitExpr(stride));
    }

    PrimExpr elem_offset = VisitExpr(buffer->elem_offset);

    Buffer resolved(buffer->data, buffer->dtype, shape, strides, elem_offset,
                    buffer->name, buffer->data_alignment, buffer->offset_factor,
                    buffer->buffer_type, buffer->axis_separators, buffer->span);
    buffer_cache_[buffer.get()] = resolved;
    return resolved;
  }

  Layout ResolveLayout(const Layout &layout) {
    if (auto *cute = layout.as<CuteLayoutNode>()) {
      Array<PrimExpr> logical_shape;
      logical_shape.reserve(cute->GetLogicalShape().size());
      for (const auto &extent : cute->GetLogicalShape()) {
        logical_shape.push_back(VisitExpr(extent));
      }

      Array<PrimExpr> mode_shape;
      mode_shape.reserve(cute->GetModeShape().size());
      for (const auto &extent : cute->GetModeShape()) {
        mode_shape.push_back(VisitExpr(extent));
      }

      Array<PrimExpr> mode_stride;
      mode_stride.reserve(cute->GetModeStride().size());
      for (const auto &stride : cute->GetModeStride()) {
        mode_stride.push_back(VisitExpr(stride));
      }

      return CuteLayout(logical_shape, mode_shape, mode_stride,
                        cute->GetDimLevels());
    }
    return layout;
  }

  Map<String, ObjectRef> ResolveTensorMeta(const Map<String, ObjectRef> &meta) {
    Map<String, ObjectRef> resolved_meta;
    for (const auto &[tensor_name, entry_obj] : meta) {
      auto entry = entry_obj.as<Map<String, ObjectRef>>();
      if (!entry.has_value()) {
        resolved_meta.Set(tensor_name, entry_obj);
        continue;
      }

      Map<String, ObjectRef> resolved_entry;
      for (const auto &[key, value] : entry.value()) {
        if (auto layout = value.as<Layout>()) {
          resolved_entry.Set(key, ResolveLayout(layout.value()));
        } else {
          resolved_entry.Set(key, value);
        }
      }
      resolved_meta.Set(tensor_name, resolved_entry);
    }
    return resolved_meta;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    Buffer buffer = ResolveBuffer(op->buffer);
    Array<PrimExpr> indices;
    indices.reserve(op->indices.size());
    for (const auto &index : op->indices) {
      indices.push_back(VisitExpr(index));
    }
    return BufferLoad(buffer, indices, std::nullopt, op->span);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    Buffer buffer = ResolveBuffer(op->buffer);
    PrimExpr value = VisitExpr(op->value);
    Array<PrimExpr> indices;
    indices.reserve(op->indices.size());
    for (const auto &index : op->indices) {
      indices.push_back(VisitExpr(index));
    }
    return BufferStore(buffer, value, indices, std::nullopt, op->span);
  }

  Stmt VisitStmt_(const DeclBufferNode *op) final {
    Buffer buffer = ResolveBuffer(op->buffer);
    Stmt body = VisitStmt(op->body);
    return DeclBuffer(buffer, body, op->span);
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    auto *node = block.CopyOnWrite();

    Array<Buffer> alloc_buffers;
    alloc_buffers.reserve(node->alloc_buffers.size());
    for (const auto &buffer : node->alloc_buffers) {
      alloc_buffers.push_back(ResolveBuffer(buffer));
    }
    node->alloc_buffers = std::move(alloc_buffers);

    Array<MatchBufferRegion> match_buffers;
    match_buffers.reserve(node->match_buffers.size());
    for (const auto &match_buffer : node->match_buffers) {
      Buffer buffer = ResolveBuffer(match_buffer->buffer);
      match_buffers.push_back(MatchBufferRegion(buffer, match_buffer->source));
    }
    node->match_buffers = std::move(match_buffers);

    return block;
  }

private:
  int nrows_;
  int ncols_;
  arith::Analyzer analyzer_;
  std::unordered_map<const BufferNode *, Buffer> buffer_cache_;
};

PrimFunc ResolvePrimFunc(PrimFunc func) {
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  ICHECK(target.defined())
      << "ResolveSunmmioMeshSymbols expects a bound target.";
  ICHECK(TargetIsSunmmio(target.value()))
      << "ResolveSunmmioMeshSymbols only supports Sunmmio targets.";

  SunmmioMeshSymbolResolver resolver(target.value());

  Map<Var, Buffer> buffer_map;
  for (const auto &[var, buffer] : func->buffer_map) {
    buffer_map.Set(var, resolver.ResolveBuffer(buffer));
  }

  DictAttrs attrs = func->attrs;
  if (auto tensor_meta = func->GetAttr<Map<String, ObjectRef>>("tensor_meta")) {
    attrs = WithAttr(std::move(attrs), ffi::String("tensor_meta"),
                     ffi::Any(resolver.ResolveTensorMeta(tensor_meta.value())));
  }

  Stmt body = resolver.ResolveStmt(func->body);
  return PrimFunc(func->params, body, func->ret_type, buffer_map, attrs,
                  func->span);
}

} // namespace

namespace transform {

tvm::transform::Pass ResolveSunmmioMeshSymbols() {
  auto pass_func = [=](PrimFunc func, const IRModule &mod,
                       const tvm::transform::PassContext &ctx) {
    return ResolvePrimFunc(std::move(func));
  };
  return tir::transform::CreatePrimFuncPass(pass_func, 0,
                                            "tl.ResolveSunmmioMeshSymbols", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ResolveSunmmioMeshSymbols",
                        ResolveSunmmioMeshSymbols);
}

} // namespace transform

} // namespace tl
} // namespace tvm
