/*!
 * \file resolve_sunmmio_mesh_symbols.cc
 * \brief Resolve symbolic Sunmmio mesh dimensions from the bound target.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../layout/cute_layout.h"
#include "../layout/layout.h"
#include "../target/sunmmio_utils.h"
#include "../target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

constexpr const char *kMeshNRowsAttr = "tl.sunmmio.mesh_nrows_var";
constexpr const char *kMeshNColsAttr = "tl.sunmmio.mesh_ncols_var";

PrimExpr I32Imm(int64_t value) { return IntImm(DataType::Int(32), value); }

PrimExpr SubstituteAndSimplify(const PrimExpr &expr,
                               const ffi::Map<Var, PrimExpr> &vmap,
                               arith::Analyzer *analyzer) {
  return analyzer->Simplify(tir::Substitute(expr, vmap));
}

Layout SubstituteLayout(const Layout &layout,
                        const ffi::Map<Var, PrimExpr> &vmap,
                        arith::Analyzer *analyzer) {
  if (auto *cute = layout.as<CuteLayoutNode>()) {
    Array<PrimExpr> logical_shape;
    logical_shape.reserve(cute->GetLogicalShape().size());
    for (const auto &extent : cute->GetLogicalShape()) {
      logical_shape.push_back(SubstituteAndSimplify(extent, vmap, analyzer));
    }

    Array<PrimExpr> mode_shape;
    mode_shape.reserve(cute->GetModeShape().size());
    for (const auto &extent : cute->GetModeShape()) {
      mode_shape.push_back(SubstituteAndSimplify(extent, vmap, analyzer));
    }

    Array<PrimExpr> mode_stride;
    mode_stride.reserve(cute->GetModeStride().size());
    for (const auto &stride : cute->GetModeStride()) {
      mode_stride.push_back(SubstituteAndSimplify(stride, vmap, analyzer));
    }

    return CuteLayout(logical_shape, mode_shape, mode_stride,
                      cute->GetDimLevels());
  }
  return layout;
}

ObjectRef SubstituteObject(const ObjectRef &obj,
                           const ffi::Map<Var, PrimExpr> &vmap,
                           arith::Analyzer *analyzer) {
  if (auto expr = obj.as<PrimExpr>()) {
    return SubstituteAndSimplify(expr.value(), vmap, analyzer);
  }
  if (auto layout = obj.as<Layout>()) {
    return SubstituteLayout(layout.value(), vmap, analyzer);
  }
  if (auto arr = obj.as<Array<ObjectRef>>()) {
    Array<ObjectRef> result;
    result.reserve(arr.value().size());
    for (const auto &item : arr.value()) {
      result.push_back(SubstituteObject(item, vmap, analyzer));
    }
    return result;
  }
  if (auto map = obj.as<Map<String, ObjectRef>>()) {
    Map<String, ObjectRef> result;
    for (const auto &[key, value] : map.value()) {
      result.Set(key, SubstituteObject(value, vmap, analyzer));
    }
    return result;
  }
  return obj;
}

DictAttrs SubstituteAttrs(DictAttrs attrs, const ffi::Map<Var, PrimExpr> &vmap,
                          arith::Analyzer *analyzer) {
  DictAttrs result = attrs;

  if (auto tensor_meta = attrs.GetAttr<Map<String, ObjectRef>>("tensor_meta")) {
    Map<String, ObjectRef> new_meta;
    for (const auto &[tensor_name, entry_obj] : tensor_meta.value()) {
      new_meta.Set(tensor_name, SubstituteObject(entry_obj, vmap, analyzer));
    }
    result = WithAttr(std::move(result), ffi::String("tensor_meta"),
                      ffi::Any(new_meta));
  }

  result = WithoutAttr(std::move(result), ffi::String(kMeshNRowsAttr));
  result = WithoutAttr(std::move(result), ffi::String(kMeshNColsAttr));
  return result;
}

// Loop/block annotations (e.g. `tile.domain`) can embed mesh symbols but are
// not rewritten by PrimFunc specialization, so substitute PrimExprs inside
// annotation values separately.
ffi::Any SubstituteAnnotationValue(const ffi::Any &value,
                                   const ffi::Map<Var, PrimExpr> &vmap,
                                   arith::Analyzer *analyzer) {
  if (auto arr = value.try_cast<Array<PrimExpr>>()) {
    Array<PrimExpr> result;
    result.reserve(arr->size());
    for (const auto &elem : arr.value()) {
      result.push_back(SubstituteAndSimplify(elem, vmap, analyzer));
    }
    return ffi::Any(result);
  }

  if (auto expr = value.as<PrimExpr>()) {
    return ffi::Any(SubstituteAndSimplify(expr.value(), vmap, analyzer));
  }

  if (auto obj = value.as<ObjectRef>()) {
    return ffi::Any(SubstituteObject(obj.value(), vmap, analyzer));
  }

  return value;
}

Map<String, ffi::Any>
SubstituteAnnotations(const Map<String, ffi::Any> &annotations,
                      const ffi::Map<Var, PrimExpr> &vmap,
                      arith::Analyzer *analyzer) {
  Map<String, ffi::Any> result;
  for (const auto &[key, value] : annotations) {
    result.Set(key, SubstituteAnnotationValue(value, vmap, analyzer));
  }
  return result;
}

class MeshSymbolAnnotationSubstituter : public StmtExprMutator {
public:
  explicit MeshSymbolAnnotationSubstituter(ffi::Map<Var, PrimExpr> vmap,
                                           arith::Analyzer *analyzer)
      : vmap_(std::move(vmap)), analyzer_(analyzer) {}

  Stmt Substitute(const Stmt &stmt) { return VisitStmt(stmt); }

  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    if (!loop->annotations.empty()) {
      loop.CopyOnWrite()->annotations =
          SubstituteAnnotations(loop->annotations, vmap_, analyzer_);
    }
    return loop;
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    if (!block->annotations.empty()) {
      block.CopyOnWrite()->annotations =
          SubstituteAnnotations(block->annotations, vmap_, analyzer_);
    }
    return block;
  }

private:
  ffi::Map<Var, PrimExpr> vmap_;
  arith::Analyzer *analyzer_;
};

PrimFunc ResolvePrimFunc(PrimFunc func) {
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  ICHECK(target.defined())
      << "ResolveSunmmioMeshSymbols expects a bound target.";
  ICHECK(TargetIsSunmmio(target.value()))
      << "ResolveSunmmioMeshSymbols only supports Sunmmio targets.";

  auto nrows_var = func->GetAttr<Var>(kMeshNRowsAttr);
  auto ncols_var = func->GetAttr<Var>(kMeshNColsAttr);
  if (!nrows_var.defined() || !ncols_var.defined()) {
    return func;
  }

  auto mesh = GetSunmmioMeshConfig(target.value());
  PrimExpr nrows = I32Imm(mesh.nrow);
  PrimExpr ncols = I32Imm(mesh.ncol);
  ffi::Map<Var, PrimExpr> vmap = {{nrows_var.value(), nrows},
                                  {ncols_var.value(), ncols}};

  // TVM's public Specialize API expects specialized PrimExpr vars to be
  // PrimFunc params. Mesh vars are frontend-only symbols, so add them
  // temporarily and let Specialize remove them while rewriting standard TIR.
  Array<Var> params = func->params;
  params.push_back(nrows_var.value());
  params.push_back(ncols_var.value());
  func = PrimFunc(params, func->body, func->ret_type, func->buffer_map,
                  func->attrs, func->span);

  ffi::Map<Var, ffi::Variant<Buffer, PrimExpr>> param_map = {
      {nrows_var.value(), nrows},
      {ncols_var.value(), ncols},
  };
  func = Specialize(std::move(func), param_map);

  arith::Analyzer analyzer;
  Stmt body =
      MeshSymbolAnnotationSubstituter(vmap, &analyzer).Substitute(func->body);
  DictAttrs attrs = SubstituteAttrs(func->attrs, vmap, &analyzer);
  return PrimFunc(func->params, body, func->ret_type, func->buffer_map, attrs,
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
