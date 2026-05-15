/*
 * Hoist selected block annotations to PrimFunc attributes.
 *
 * This pass is intentionally small: it preserves metadata that later block
 * lowering would otherwise drop, without changing block annotations in-place.
 */
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <string>
#include <utility>

#include "../layout/layout.h"
#include "../op/builtin.h"

namespace tvm {
namespace tl {
using namespace tvm::tir;

namespace {

// 需要将block注释提升到函数属性的annotation key列表
constexpr const char *kHoistedAnnotationKeys[] = {
    attr::kLayoutMap,
    attr::kGlobalLayoutMap,
};
// 需要保存到device_func_attr_keys列表中的属性key
constexpr const char *kDeviceFuncPropagatedAttrKeys[] = {
    attr::kLayoutMap,
    attr::kGlobalLayoutMap,
};

void CopyMapEntries(const ffi::Map<ffi::Any, ffi::Any> &map,
                    ffi::Map<ffi::Any, ffi::Any> *out) {
  for (const auto &kv : map) {
    out->Set(kv.first, kv.second);
  }
}

ffi::Any MergeAnnotationValue(const ffi::Any &existing,
                              const ffi::Any &incoming) {
  auto existing_map = existing.try_cast<ffi::Map<ffi::Any, ffi::Any>>();
  auto incoming_map = incoming.try_cast<ffi::Map<ffi::Any, ffi::Any>>();
  if (existing_map && incoming_map) {
    ffi::Map<ffi::Any, ffi::Any> merged;
    CopyMapEntries(existing_map.value(), &merged);
    CopyMapEntries(incoming_map.value(), &merged);
    return merged;
  }
  return existing;
}

void AppendUniqueString(ffi::Array<ffi::String> *array,
                        const ffi::String &key) {
  for (const ffi::String &existing : *array) {
    if (static_cast<std::string>(existing) == static_cast<std::string>(key)) {
      return;
    }
  }
  array->push_back(key);
}

ffi::Array<ffi::String> MergeDeviceFuncAttrKeys(const PrimFunc &f) {
  ffi::Array<ffi::String> result;
  if (f->attrs.defined()) {
    if (auto existing = f->GetAttr<ffi::Array<ffi::String>>(
            tl::attr::kDeviceFuncAttrKeys)) {
      for (const ffi::String &key : existing.value()) {
        AppendUniqueString(&result, key);
      }
    }
  }
  for (const char *key : kDeviceFuncPropagatedAttrKeys) {
    AppendUniqueString(&result, key);
  }
  return result;
}

class BlockAnnotationCollector : public StmtVisitor {
public:
  void Collect(const Stmt &stmt) { VisitStmt(stmt); }

  const ffi::Map<ffi::String, ffi::Any> &Result() const { return collected_; }

private:
  void VisitStmt_(const BlockNode *op) final {
    for (const char *key : kHoistedAnnotationKeys) {
      auto it = op->annotations.find(key);
      if (it == op->annotations.end()) {
        continue;
      }

      if (auto existing = collected_.Get(key)) {
        collected_.Set(key,
                       MergeAnnotationValue(existing.value(), (*it).second));
      } else {
        collected_.Set(key, (*it).second);
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

  ffi::Map<ffi::String, ffi::Any> collected_;
};

PrimFunc HoistBlockAnnotationsToFuncAttrs(PrimFunc f) {
  if (!f.defined()) {
    return f;
  }

  BlockAnnotationCollector collector;
  collector.Collect(f->body);

  for (const auto &kv : collector.Result()) {
    std::string key = kv.first;
    ffi::Any value = kv.second;
    if (f->attrs.defined()) {
      auto it = f->attrs->dict.find(key);
      if (it != f->attrs->dict.end()) {
        value = MergeAnnotationValue((*it).second, kv.second);
      }
    }
    f = WithAttr(std::move(f), key, value);
  }
  f = WithAttr(std::move(f), tl::attr::kDeviceFuncAttrKeys,
               MergeDeviceFuncAttrKeys(f));
  return f;
}

} // namespace

namespace transform {

tvm::transform::Pass HoistBlockAnnotationsToFuncAttrs() {
  auto pass_func = [](PrimFunc f, const IRModule &,
                      const tvm::transform::PassContext &) {
    return tvm::tl::HoistBlockAnnotationsToFuncAttrs(std::move(f));
  };
  return tvm::tir::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.HoistBlockAnnotationsToFuncAttrs", {});
}

} // namespace transform

} // namespace tl
} // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.HoistBlockAnnotationsToFuncAttrs",
                        tvm::tl::transform::HoistBlockAnnotationsToFuncAttrs);
}
