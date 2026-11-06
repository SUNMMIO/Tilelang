#include "mx_view.h"

#include "../../layout/cute_layout.h"
#include "../../op/region.h"
#include "../../op/utils.h"
#include "../../target/sunmmio_utils.h"
#include "attr.h"
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/transform.h>
#include <unordered_set>

namespace tvm::tl {
using namespace tir;

bool IsMXFieldHandle(const PrimExpr &expr) {
  const auto *call = expr.as<CallNode>();
  return call && call->op.same_as(mx_field_handle());
}

bool HasMXShapePlaceholder(const PrimExpr &expr) {
  bool found = false;
  PostOrderVisit(expr, [&](const ObjectRef &obj) {
    const auto *call = obj.as<CallNode>();
    found |= call && call->op.same_as(mx_scale_extent());
  });
  return found;
}

MXViewMap CollectMXViews(const Stmt &body) {
  MXViewMap result;
  PostOrderVisit(body, [&](const ObjectRef &obj) {
    const auto *let = obj.as<LetStmtNode>();
    if (!let || !IsMXFieldHandle(let->value))
      return;
    auto call = Downcast<Call>(let->value);
    ICHECK_EQ(call->args.size(), 2U);
    auto parent = Downcast<Buffer>(call->annotations.at(kMXParent));
    ICHECK(parent->data.same_as(call->args[0]))
        << "MX view parent metadata is stale; normalize bindings after "
           "remapping";
    int field = static_cast<int>(Downcast<IntImm>(call->args[1])->value);
    ICHECK(field == 0 || field == 1) << "Invalid MX field role";
    result.emplace(let->var.get(), MXViewInfo{parent, field, call});
  });
  return result;
}

Var MXStorageRoot(const Buffer &buffer, const MXViewMap &views) {
  auto it = views.find(buffer->data.get());
  return it == views.end() ? buffer->data : it->second.parent->data;
}

Map<Var, PrimExpr> CollectMXViewLets(const Stmt &body) {
  Map<Var, PrimExpr> lets;
  PostOrderVisit(body, [&](const ObjectRef &obj) {
    if (const auto *let = obj.as<LetStmtNode>())
      lets.Set(let->var, let->value);
  });
  return lets;
}

bool ReadsMXField(const PrimExpr &expr, const MXViewMap &views,
                  const Map<Var, PrimExpr> &lets) {
  bool found = false;
  std::unordered_set<const VarNode *> visiting;
  std::function<void(const PrimExpr &)> inspect = [&](const PrimExpr &value) {
    PostOrderVisit(value, [&](const ObjectRef &obj) {
      if (const auto *load = obj.as<BufferLoadNode>())
        found |= views.count(load->buffer->data.get()) != 0;
      if (const auto *var = obj.as<VarNode>(); var && !visiting.count(var)) {
        Var v = ffi::GetRef<Var>(var);
        if (lets.count(v)) {
          visiting.insert(var);
          inspect(lets[v]);
        }
      }
    });
  };
  inspect(expr);
  return found;
}

Call RewriteMXCopyAnnotations(
    Call call, const std::function<PrimExpr(const PrimExpr &)> &rewrite) {
  if (!call->op.same_as(Op::Get(kMXDeferredCopy)))
    return call;
  std::function<ffi::Any(const ffi::Any &)> visit =
      [&](const ffi::Any &value) -> ffi::Any {
    if (auto expr = value.as<PrimExpr>())
      return rewrite(expr.value());
    if (auto array = value.as<Array<ffi::Any>>()) {
      Array<ffi::Any> result;
      for (const auto &item : array.value())
        result.push_back(visit(item));
      return result;
    }
    if (auto map = value.as<Map<ffi::Any, ffi::Any>>()) {
      Map<ffi::Any, ffi::Any> result;
      for (const auto &[key, item] : map.value())
        result.Set(visit(key), visit(item));
      return result;
    }
    return value;
  };
  Map<String, ObjectRef> annotations;
  for (const auto &[key, value] : call->annotations)
    annotations.Set(key, visit(value).cast<ObjectRef>());
  call.CopyOnWrite()->annotations = annotations;
  return call;
}

namespace {

class MXBindingRewriter : public StmtExprMutator {
public:
  Map<Var, Array<PrimExpr>> scale_shapes;
  Map<Var, Layout> field_layouts;
  Map<Var, Integer> field_offsets;
  Map<Var, Buffer> buffers;
  Map<Var, Var> shape_parents;

  PrimExpr VisitExpr(const PrimExpr &expr) final {
    PrimExpr result = StmtExprMutator::VisitExpr(expr);
    if (!scale_shapes.empty() &&
        (result.dtype().is_int() || result.dtype().is_uint()))
      result = shape_analyzer_.Simplify(result);
    return result;
  }

  Buffer RemapBuffer(const Buffer &buffer) {
    auto it = remapped_.find(buffer);
    if (it != remapped_.end())
      return it->second;
    Buffer result = buffer;
    auto shape =
        buffer->shape.Map([&](const PrimExpr &x) { return VisitExpr(x); });
    auto strides =
        buffer->strides.Map([&](const PrimExpr &x) { return VisitExpr(x); });
    PrimExpr offset = VisitExpr(buffer->elem_offset);
    if (!shape.same_as(buffer->shape) || !strides.same_as(buffer->strides) ||
        !offset.same_as(buffer->elem_offset)) {
      auto *n = result.CopyOnWrite();
      n->shape = shape;
      n->strides = strides;
      n->elem_offset = offset;
    }
    if (field_handles_.count(buffer->data.get()) &&
        buffers.count(buffer->data) && !buffers[buffer->data].same_as(buffer)) {
      Buffer canonical = RemapBuffer(buffers[buffer->data]);
      arith::Analyzer analyzer;
      auto equal = [&](const Array<PrimExpr> &lhs, const Array<PrimExpr> &rhs) {
        if (lhs.size() != rhs.size())
          return false;
        for (size_t i = 0; i < lhs.size(); ++i)
          if (!analyzer.CanProveEqual(lhs[i], rhs[i]))
            return false;
        return true;
      };
      ICHECK(
          result->dtype == canonical->dtype &&
          equal(result->shape, canonical->shape) &&
          equal(result->strides, canonical->strides) &&
          analyzer.CanProveEqual(result->elem_offset, canonical->elem_offset))
          << "MX fields do not support reshaped or reinterpreted storage "
             "aliases";
      result = canonical;
    }
    remapped_[buffer] = result;
    return result;
  }

  ffi::Any RemapAny(const ffi::Any &value) {
    if (auto b = value.as<Buffer>())
      return RemapBuffer(b.value());
    if (auto r = value.as<BufferRegion>()) {
      return BufferRegion(RemapBuffer(r.value()->buffer),
                          r.value()->region.Map([&](const Range &x) {
                            return Range::FromMinExtent(VisitExpr(x->min),
                                                        VisitExpr(x->extent));
                          }));
    }
    if (auto r = value.as<Range>())
      return Range::FromMinExtent(VisitExpr(r.value()->min),
                                  VisitExpr(r.value()->extent));
    if (auto layout = value.as<CuteLayout>()) {
      auto *n = layout.value().operator->();
      auto map = [&](const Array<PrimExpr> &xs) {
        return xs.Map([&](const PrimExpr &x) { return VisitExpr(x); });
      };
      return CuteLayout(map(n->GetLogicalShape()), map(n->GetModeShape()),
                        map(n->GetModeStride()), n->GetDimLevels());
    }
    if (auto e = value.as<PrimExpr>())
      return VisitExpr(e.value());
    if (auto a = value.as<Array<ffi::Any>>()) {
      Array<ffi::Any> out;
      for (const auto &x : a.value())
        out.push_back(RemapAny(x));
      return out;
    }
    if (auto m = value.as<Map<ffi::Any, ffi::Any>>()) {
      Map<ffi::Any, ffi::Any> out;
      for (const auto &[k, v] : m.value())
        out.Set(RemapAny(k), RemapAny(v));
      return out;
    }
    return value;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (call->op.same_as(mx_scale_extent())) {
      Var parent = Downcast<Var>(call->args[0]);
      if (shape_parents.count(parent)) {
        auto args = call->args;
        args.Set(0, shape_parents[parent]);
        call.CopyOnWrite()->args = args;
      }
    }
    if (call->op.same_as(mx_scale_extent()) && !scale_shapes.empty()) {
      Var parent = Downcast<Var>(call->args[0]);
      ICHECK(scale_shapes.count(parent))
          << "Unresolved MX scale shape for " << parent;
      int axis = Downcast<IntImm>(call->args[1])->value;
      ICHECK(axis == 0 || axis == 1);
      return scale_shapes[parent][axis];
    }
    Map<String, ObjectRef> annotations;
    for (const auto &[key, value] : call->annotations)
      annotations.Set(key, RemapAny(value).cast<ObjectRef>());
    if (call->op.same_as(mx_field_handle())) {
      Buffer parent = Downcast<Buffer>(annotations.at(kMXParent));
      Var actual = Downcast<Var>(call->args[0]);
      ICHECK(is_zero(parent->elem_offset))
          << "T.mx_view requires a complete MX allocation with zero offset";
      if (buffers.count(actual)) {
        arith::Analyzer analyzer;
        bool same_shape = parent->shape.size() == buffers[actual]->shape.size();
        for (size_t i = 0; same_shape && i < parent->shape.size(); ++i)
          same_shape = analyzer.CanProveEqual(parent->shape[i],
                                              buffers[actual]->shape[i]);
        ICHECK(same_shape && parent->dtype == buffers[actual]->dtype)
            << "T.mx_view cannot unpack a reshaped parent alias";
        parent = RemapBuffer(buffers[actual]);
      } else if (!parent->data.same_as(actual)) {
        parent.CopyOnWrite()->data = actual;
      }
      buffers.Set(actual, parent);
      annotations.Set(kMXParent, parent);
    }
    call.CopyOnWrite()->annotations = annotations;
    return call;
  }

  Stmt VisitStmt_(const LetStmtNode *op) final {
    bool shape_value =
        HasMXShapePlaceholder(op->value) ||
        UsesVar(op->value, [&](const VarNode *var) {
          return resolved_shape_vars_.count(ffi::GetRef<Var>(var)) != 0;
        });
    if (!scale_shapes.empty() && shape_value &&
        SideEffect(op->value) <= CallEffectKind::kPure) {
      arith::Analyzer analyzer;
      resolved_shape_vars_.Set(op->var,
                               analyzer.Simplify(VisitExpr(op->value)));
      return VisitStmt(op->body);
    }
    LetStmt stmt = Downcast<LetStmt>(StmtExprMutator::VisitStmt_(op));
    if (IsMXFieldHandle(stmt->value) && field_layouts.count(stmt->var)) {
      Call call = Downcast<Call>(stmt->value);
      auto *n = call.CopyOnWrite();
      n->annotations.Set(kMXFieldLayout, field_layouts[stmt->var]);
      n->annotations.Set(kMXFieldOffset, field_offsets[stmt->var]);
      Var parent = Downcast<Var>(call->args[0]);
      n->annotations.Set(kMXParentLayout, field_layouts[parent]);
      stmt.CopyOnWrite()->value = call;
    }
    return stmt;
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    Var var = ffi::GetRef<Var>(op);
    auto value = resolved_shape_vars_.find(var);
    return value == resolved_shape_vars_.end() ? PrimExpr(var)
                                               : (*value).second;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto *n = load.CopyOnWrite();
    n->buffer = RemapBuffer(n->buffer);
    if (n->predicate.defined())
      n->predicate = VisitExpr(n->predicate.value());
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto *n = store.CopyOnWrite();
    n->buffer = RemapBuffer(n->buffer);
    if (n->predicate.defined())
      n->predicate = VisitExpr(n->predicate.value());
    return store;
  }

  Stmt VisitStmt_(const DeclBufferNode *op) final {
    DeclBuffer decl = Downcast<DeclBuffer>(StmtExprMutator::VisitStmt_(op));
    decl.CopyOnWrite()->buffer = RemapBuffer(decl->buffer);
    return decl;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    ObjectRef node = RemapAny(op->node).cast<ObjectRef>();
    AttrStmt attr = Downcast<AttrStmt>(StmtExprMutator::VisitStmt_(op));
    attr.CopyOnWrite()->node = node;
    return attr;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    if (!scale_shapes.empty() &&
        shape_analyzer_.CanProve(VisitExpr(op->extent) <= 0))
      return Evaluate(0);
    For loop = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    Map<String, ffi::Any> annotations;
    for (const auto &[key, value] : loop->annotations)
      annotations.Set(key, RemapAny(value));
    loop.CopyOnWrite()->annotations = annotations;
    return loop;
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    if (!scale_shapes.empty()) {
      PrimExpr condition = VisitExpr(op->condition);
      if (is_one(condition))
        return VisitStmt(op->then_case);
      if (is_zero(condition))
        return op->else_case.defined() ? VisitStmt(op->else_case.value())
                                       : Stmt(Evaluate(0));
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    auto *n = block.CopyOnWrite();
    n->alloc_buffers =
        n->alloc_buffers.Map([&](const Buffer &b) { return RemapBuffer(b); });
    auto region = [&](const BufferRegion &r) {
      return BufferRegion(RemapBuffer(r->buffer),
                          r->region.Map([&](const Range &x) {
                            return Range::FromMinExtent(VisitExpr(x->min),
                                                        VisitExpr(x->extent));
                          }));
    };
    n->reads = n->reads.Map(region);
    n->writes = n->writes.Map(region);
    n->match_buffers = n->match_buffers.Map([&](const MatchBufferRegion &m) {
      return MatchBufferRegion(RemapBuffer(m->buffer), region(m->source));
    });
    Map<String, ffi::Any> annotations;
    for (const auto &[key, value] : n->annotations)
      annotations.Set(key, RemapAny(value));
    n->annotations = annotations;
    return block;
  }

  PrimFunc Run(PrimFunc func) {
    PostOrderVisit(func->body, [&](const ObjectRef &obj) {
      if (const auto *let = obj.as<LetStmtNode>();
          let && IsMXFieldHandle(let->value))
        field_handles_.insert(let->var.get());
    });
    PostOrderVisit(func->body, [&](const ObjectRef &obj) {
      if (const auto *b = obj.as<DeclBufferNode>())
        buffers.Set(b->buffer->data, b->buffer);
      if (const auto *b = obj.as<BlockNode>())
        for (const Buffer &buffer : b->alloc_buffers)
          buffers.Set(buffer->data, buffer);
      if (const auto *call = obj.as<CallNode>();
          call && call->op.same_as(mx_field_handle())) {
        Buffer parent = Downcast<Buffer>(call->annotations.at(kMXParent));
        Var actual = Downcast<Var>(call->args[0]);
        if (!parent->data.same_as(actual))
          shape_parents.Set(parent->data, actual);
      }
    });
    auto *n = func.CopyOnWrite();
    n->body = VisitStmt(n->body);
    Map<Var, Buffer> buffer_map;
    for (const auto &[var, buffer] : n->buffer_map)
      buffer_map.Set(var, RemapBuffer(buffer));
    n->buffer_map = buffer_map;
    if (n->attrs.defined()) {
      auto attrs = n->attrs->dict;
      for (const auto &[key, value] : n->attrs->dict)
        attrs.Set(key, RemapAny(value));
      n->attrs = DictAttrs(attrs);
    }
    Map<Buffer, Layout> resolved;
    for (const auto &[handle, info] : CollectMXViews(n->body)) {
      if (!info.binding->annotations.count(kMXFieldLayout))
        continue;
      resolved.Set(info.parent, Downcast<Layout>(info.binding->annotations.at(
                                    kMXParentLayout)));
      Var var = ffi::GetRef<Var>(handle);
      if (buffers.count(var))
        resolved.Set(
            RemapBuffer(buffers[var]),
            Downcast<Layout>(info.binding->annotations.at(kMXFieldLayout)));
    }
    if (!resolved.empty()) {
      class RestoreLayouts : public StmtMutator {
      public:
        explicit RestoreLayouts(const Map<Buffer, Layout> &layouts)
            : layouts_(layouts) {}
        Stmt VisitStmt_(const BlockNode *op) final {
          Block block = Downcast<Block>(StmtMutator::VisitStmt_(op));
          if (auto value = block->annotations.Get(attr::kLayoutMap)) {
            if (auto map = value.value().as<Map<Buffer, Layout>>()) {
              auto updated = map.value();
              for (const auto &[buffer, layout] : layouts_) {
                if (updated.count(buffer))
                  ICHECK(IsSameLayout(updated[buffer], layout))
                      << "MX view storage layout changed after field "
                         "resolution";
                updated.Set(buffer, layout);
              }
              block.CopyOnWrite()->annotations.Set(attr::kLayoutMap, updated);
            }
          }
          return block;
        }

      private:
        const Map<Buffer, Layout> &layouts_;
      };
      n->body = RestoreLayouts(resolved)(n->body);
      if (auto map = func->GetAttr<Map<Buffer, Layout>>(attr::kLayoutMap)) {
        auto updated = map.value();
        for (const auto &[buffer, layout] : resolved) {
          if (updated.count(buffer))
            ICHECK(IsSameLayout(updated[buffer], layout))
                << "MX view storage layout changed after field resolution";
          updated.Set(buffer, layout);
        }
        auto attrs = n->attrs->dict;
        attrs.Set(attr::kLayoutMap, updated);
        n->attrs = DictAttrs(attrs);
      }
    }
    return func;
  }

private:
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> remapped_;
  Map<Var, PrimExpr> resolved_shape_vars_;
  std::unordered_set<const VarNode *> field_handles_;
  arith::Analyzer shape_analyzer_;
};

class MXViewValidator : public StmtExprVisitor {
public:
  explicit MXViewValidator(const PrimFunc &f)
      : views_(CollectMXViews(f->body)) {}
  void Run(const Stmt &body) { VisitStmt(body); }

private:
  void VisitStmt_(const AllocateNode *op) final {
    allocated_.insert(op->buffer_var.get());
    StmtExprVisitor::VisitStmt_(op);
    allocated_.erase(op->buffer_var.get());
  }
  void VisitStmt_(const BlockNode *op) final {
    for (const Buffer &b : op->alloc_buffers)
      allocated_.insert(b->data.get());
    for (const auto &m : op->match_buffers)
      ICHECK(!views_.count(m->source->buffer->data.get()))
          << "MX fields do not support derived storage aliases";
    StmtExprVisitor::VisitStmt_(op);
    for (const Buffer &b : op->alloc_buffers)
      allocated_.erase(b->data.get());
  }
  void VisitStmt_(const LetStmtNode *op) final {
    if (IsMXFieldHandle(op->value)) {
      const MXViewInfo &info = views_.at(op->var.get());
      ICHECK(allocated_.count(info.parent->data.get()))
          << "T.mx_view expects an independently allocated local MX buffer";
      const ObjectRef &id = info.binding->annotations.at(kMXDeclaration);
      auto [it, inserted] = declarations_.emplace(info.parent->data.get(), id);
      ICHECK(inserted || it->second.same_as(id))
          << "T.mx_view permits only one declaration per parent allocation";
      VisitExpr(op->value);
      active_fields_.insert(op->var.get());
      VisitStmt(op->body);
      active_fields_.erase(op->var.get());
      return;
    }
    if (const auto *var = op->value.as<VarNode>())
      ICHECK(!views_.count(var))
          << "MX field handles cannot be exported as raw pointers";
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const DeclBufferNode *op) final {
    if (views_.count(op->buffer->data.get())) {
      auto [it, inserted] = fields_.emplace(op->buffer->data.get(), op->buffer);
      ICHECK(inserted || it->second.same_as(op->buffer))
          << "MX fields do not support derived storage aliases";
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const ForNode *op) final {
    bool tiles = op->annotations.count(attr::kTileDomain);
    tile_depth_ += tiles;
    StmtExprVisitor::VisitStmt_(op);
    tile_depth_ -= tiles;
  }
  void VisitExpr_(const BufferLoadNode *op) final {
    ICHECK(!views_.count(op->buffer->data.get()) ||
           active_fields_.count(op->buffer->data.get()))
        << "MX field access must be dominated by its view declaration";
    ICHECK(tile_depth_ || region_load_ == op ||
           !views_.count(op->buffer->data.get()))
        << "MX fields do not support standalone scalar reads; use T.Tiles or a "
           "region copy";
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    ICHECK(!views_.count(op->buffer->data.get()) ||
           active_fields_.count(op->buffer->data.get()))
        << "MX field access must be dominated by its view declaration";
    ICHECK(tile_depth_ || !views_.count(op->buffer->data.get()))
        << "MX fields do not support standalone scalar writes; use T.Tiles or "
           "a region copy";
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(RegionOp::Get()) || op->op.same_as(mx_copy_operand())) {
      auto previous = region_load_;
      region_load_ = op->args[op->op.same_as(mx_copy_operand()) ? 1 : 0]
                         .as<BufferLoadNode>();
      StmtExprVisitor::VisitExpr_(op);
      region_load_ = previous;
      return;
    }
    if (op->op.same_as(Op::Get("tl.tileop.copy"))) {
      auto src = NormalizeToBufferRegion(op->args[0])->buffer;
      auto dst = NormalizeToBufferRegion(op->args[1])->buffer;
      if (views_.count(src->data.get()) || views_.count(dst->data.get())) {
        ICHECK(src->dtype == dst->dtype)
            << "MX field copy requires matching dtypes; use an explicit Tiles "
               "cast";
        auto rsram = [](const Buffer &b) {
          return b.scope() == "shared.rsram" || b.scope() == "shared" ||
                 b.scope() == "shared.dyn";
        };
        ICHECK(rsram(src) && rsram(dst))
            << "MX field copy supports only RSRAM peers";
      }
    } else {
      for (const PrimExpr &arg : op->args) {
        const auto *region = arg.as<CallNode>();
        if (region && region->op.same_as(RegionOp::Get())) {
          Buffer buffer = NormalizeToBufferRegion(arg)->buffer;
          ICHECK(!views_.count(buffer->data.get()))
              << "MX fields support only Tiles accesses and RSRAM region "
                 "copies";
        }
      }
    }
    for (const PrimExpr &arg : op->args) {
      if (const auto *v = arg.as<VarNode>())
        ICHECK(!views_.count(v))
            << "MX field handles cannot be exported as raw pointers";
    }
    StmtExprVisitor::VisitExpr_(op);
  }
  MXViewMap views_;
  std::unordered_set<const VarNode *> allocated_;
  std::unordered_set<const VarNode *> active_fields_;
  std::unordered_map<const VarNode *, ObjectRef> declarations_;
  std::unordered_map<const VarNode *, Buffer> fields_;
  int tile_depth_{0};
  const BufferLoadNode *region_load_{nullptr};
};

} // namespace

PrimFunc NormalizeMXViewBindings(PrimFunc func) {
  bool has_views = false;
  PostOrderVisit(func->body, [&](const ObjectRef &obj) {
    if (const auto *call = obj.as<CallNode>())
      has_views |= call->op.same_as(mx_field_handle());
  });
  return has_views ? MXBindingRewriter().Run(std::move(func)) : func;
}

PrimFunc ResolveMXViewShapes(PrimFunc func,
                             const Map<Var, Array<PrimExpr>> &shapes,
                             const Map<Var, Layout> &layouts,
                             const Map<Var, Integer> &offsets) {
  MXBindingRewriter rewriter;
  rewriter.scale_shapes = shapes;
  rewriter.field_layouts = layouts;
  rewriter.field_offsets = offsets;
  func = rewriter.Run(std::move(func));
  func = NormalizeMXCopyIndices(std::move(func));
  class ReplayCopies : public StmtExprMutator {
    PrimExpr VisitExpr_(const CallNode *op) final {
      Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
      if (!call->op.same_as(Op::Get(kMXDeferredCopy)))
        return call;
      auto replay =
          ffi::Function::GetGlobalRequired("tl.mx.ResolveDeferredCopy");
      return replay(call->args[0], call->args[1], call->args[2],
                    call->annotations)
          .cast<PrimExpr>();
    }
  } replay;
  func.CopyOnWrite()->body = replay(func->body);
  return func;
}

void ValidateMXViewDeclarations(const PrimFunc &func) {
  MXViewValidator(func).Run(func->body);
}

void CheckMXViewResolved(const PrimFunc &func) {
  std::unordered_set<const Object *> visited;
  std::function<void(const ffi::Any &)> inspect = [&](const ffi::Any &value) {
    auto object = value.as<ObjectRef>();
    if (!object || !object.value().defined() ||
        !visited.insert(object.value().get()).second)
      return;
    if (auto call = value.as<Call>()) {
      ICHECK(!call.value()->op.same_as(mx_scale_extent()) &&
             !call.value()->op.same_as(mx_copy_operand()) &&
             !call.value()->op.same_as(Op::Get(kMXDeferredCopy)))
          << "Unresolved MX operation after layout inference: "
          << call.value()->op;
    }
    if (auto array = value.as<Array<ffi::Any>>()) {
      for (const auto &item : array.value())
        inspect(item);
    } else if (auto map = value.as<Map<ffi::Any, ffi::Any>>()) {
      for (const auto &[key, item] : map.value()) {
        inspect(key);
        inspect(item);
      }
    } else {
      ffi::reflection::ForEachFieldInfo(
          TVMFFIGetTypeInfo(object.value()->type_index()),
          [&](const TVMFFIFieldInfo *field) {
            inspect(ffi::reflection::FieldGetter(field)(object.value()));
          });
    }
  };
  inspect(func);
}

void CheckMXViewPipeline(const PrimFunc &func) {
  auto views = CollectMXViews(func->body);
  if (views.empty())
    return;
  std::unordered_set<const VarNode *> storage;
  for (const auto &[var, view] : views) {
    storage.insert(var);
    storage.insert(view.parent->data.get());
  }
  PostOrderVisit(func->body, [&](const ObjectRef &obj) {
    const auto *loop = obj.as<ForNode>();
    if (!loop || !loop->annotations.count("num_stages"))
      return;
    if (auto stages = loop->annotations.at("num_stages").as<IntImm>();
        stages && stages.value()->value <= 0)
      return;
    bool touches = false;
    PostOrderVisit(loop->body, [&](const ObjectRef &node) {
      if (const auto *v = node.as<VarNode>())
        touches |= storage.count(v) != 0;
      if (const auto *l = node.as<BufferLoadNode>())
        touches |= storage.count(l->buffer->data.get()) != 0;
      if (const auto *s = node.as<BufferStoreNode>())
        touches |= storage.count(s->buffer->data.get()) != 0;
    });
    ICHECK(!touches) << "MX views and their parent storage do not support "
                        "automatic pipelining";
  });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.sunmmio.CheckMXViewResolved", CheckMXViewResolved);
  refl::GlobalDef().def("tl.transform.NormalizeMXViews", [](bool validate) {
    auto transform = [=](PrimFunc f, const IRModule &,
                         const tvm::transform::PassContext &) {
      if (validate) {
        std::unordered_map<const VarNode *, int> handles;
        PostOrderVisit(f->body, [&](const ObjectRef &obj) {
          if (const auto *let = obj.as<LetStmtNode>();
              let && IsMXFieldHandle(let->value)) {
            auto call = Downcast<Call>(let->value);
            handles.emplace(let->var.get(),
                            Downcast<IntImm>(call->args[1])->value);
          }
        });
        // User annotations are keyed by handles and may precede the fields'
        // DeclBuffers. Check them before rewriting or dropping metadata.
        // Compiler-derived layout maps use Buffer keys and remain valid.
        auto check_annotation = [&](const String &key, const ffi::Any &value) {
          if (key == "safe_value_map" && !handles.empty()) {
            TVM_FFI_THROW(ValueError)
                << "T.annotate_safe_value is not supported in functions "
                   "using T.mx_view, including annotations on ordinary "
                   "buffers. Remove the annotation from this function.";
          }
          if (key != "layout_map" && key != "tileview_map")
            return;
          if (auto map = value.as<Map<ffi::Any, ffi::Any>>()) {
            for (const auto &[target, _] : map.value()) {
              auto var = target.as<Var>();
              if (!var || !handles.count(var.value().get()))
                continue;
              int role = handles.at(var.value().get());
              if (key == "layout_map") {
                TVM_FFI_THROW(ValueError)
                    << "T.annotate_layout cannot target mx_view "
                    << (role == 0 ? "data" : "scale") << " field '"
                    << var.value()->name_hint
                    << "': its layout is derived from the parent MX buffer. "
                       "Annotate the parent MX buffer instead.";
              }
              if (role == 1) {
                TVM_FFI_THROW(ValueError)
                    << "T.annotate_tileview is not yet supported for mx_view "
                       "scale field '"
                    << var.value()->name_hint
                    << "'. Omit the annotation and use automatic TileView "
                       "inference.";
              }
            }
          }
        };
        if (f->attrs.defined())
          for (const auto &[key, value] : f->attrs->dict)
            check_annotation(key, value);
        std::unordered_map<const VarNode *, Buffer> declarations;
        PostOrderVisit(f->body, [&](const ObjectRef &obj) {
          if (const auto *block = obj.as<BlockNode>())
            for (const auto &[key, value] : block->annotations)
              check_annotation(key, value);
          if (const auto *decl = obj.as<DeclBufferNode>();
              decl && handles.count(decl->buffer->data.get())) {
            auto [it, inserted] =
                declarations.emplace(decl->buffer->data.get(), decl->buffer);
            ICHECK(inserted || it->second.same_as(decl->buffer))
                << "MX fields do not support derived storage aliases";
          }
        });
      }
      f = NormalizeMXViewBindings(std::move(f));
      if (validate)
        ValidateMXViewDeclarations(f);
      return f;
    };
    return tir::transform::CreatePrimFuncPass(transform, 0,
                                              "tl.NormalizeMXViews", {});
  });
  refl::GlobalDef().def("tl.transform.CheckMXViewPipeline", []() {
    auto transform = [](PrimFunc f, const IRModule &,
                        const tvm::transform::PassContext &) {
      CheckMXViewPipeline(f);
      return f;
    };
    return tir::transform::CreatePrimFuncPass(transform, 0,
                                              "tl.CheckMXViewPipeline", {});
  });
}

} // namespace tvm::tl
