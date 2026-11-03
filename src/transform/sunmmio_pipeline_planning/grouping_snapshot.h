/*!
 * \file grouping_snapshot.h
 * \brief Whole-function grouping rollback with executable layout metadata.
 */
#ifndef TVM_TL_TRANSFORM_GROUPING_SNAPSHOT_H_
#define TVM_TL_TRANSFORM_GROUPING_SNAPSHOT_H_

#include "../../layout/cute_layout.h"

#include <tvm/ir/attrs.h>
#include <tvm/node/serialization.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace tl {
namespace residency {
namespace grouping_snapshot_detail {

// JSON preserves identities within its graph, but does not intern deserialized
// Vars into the process-wide InputPlaceholder cache.  Only these two layout
// types have a complete reconstruction contract here.  In particular, do not
// silently erase Fragment or SwizzledLayout semantics by rebuilding a Layout.
inline bool SupportedLayout(const Layout &layout) {
  return layout.as<CuteLayoutNode>() != nullptr ||
         std::string(layout->GetTypeKey()) == "tl.Layout";
}

class MetadataCheck : public tir::StmtVisitor {
public:
  bool supported{true};
  size_t input_dimensions{0};

  void CheckValue(const ffi::Any &value, bool map_key = false) {
    if (auto layout = value.as<Layout>()) {
      supported = supported && !map_key && SupportedLayout(layout.value());
      input_dimensions = std::max(input_dimensions, layout.value()->InputDim());
    } else if (auto attrs = value.as<DictAttrs>()) {
      CheckValue(attrs.value()->dict, map_key);
    } else if (auto array = value.as<ffi::Array<ffi::Any>>()) {
      for (const ffi::Any &item : array.value())
        CheckValue(item, map_key);
    } else if (auto map = value.as<ffi::Map<ffi::Any, ffi::Any>>()) {
      for (const auto &[key, item] : map.value()) {
        // Buffer/Var keys belong to the restored graph and remain untouched.
        // A Layout key would itself require replacement, outside this contract.
        CheckValue(key, true);
        CheckValue(item, map_key);
      }
    }
  }

  void CheckFunction(const tir::PrimFunc &func) {
    if (func->attrs.defined())
      CheckValue(func->attrs);
    operator()(func->body);
  }

private:
  void VisitStmt_(const tir::BlockNode *op) final {
    CheckValue(op->annotations);
    tir::StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const tir::ForNode *op) final {
    CheckValue(op->annotations);
    tir::StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const tir::AttrStmtNode *op) final {
    CheckValue(op->node);
    tir::StmtVisitor::VisitStmt_(op);
  }
};

class LayoutRestorer : public tir::StmtMutator {
public:
  explicit LayoutRestorer(const ffi::Array<tir::Var> &saved_placeholders) {
    for (size_t i = 0; i < saved_placeholders.size(); ++i) {
      placeholders_.Set(saved_placeholders[i], InputPlaceholder(i));
    }
  }

  ffi::Any RestoreValue(const ffi::Any &value) {
    if (auto layout = value.as<Layout>())
      return RestoreLayout(layout.value());
    if (auto attrs = value.as<DictAttrs>()) {
      return DictAttrs(RestoreAnnotations(attrs.value()->dict));
    }
    if (auto array = value.as<ffi::Array<ffi::Any>>()) {
      ffi::Array<ffi::Any> result;
      result.reserve(array.value().size());
      for (const ffi::Any &item : array.value()) {
        result.push_back(RestoreValue(item));
      }
      return result;
    }
    if (auto map = value.as<ffi::Map<ffi::Any, ffi::Any>>()) {
      ffi::Map<ffi::Any, ffi::Any> result;
      for (const auto &[key, item] : map.value()) {
        result.Set(key, RestoreValue(item));
      }
      return result;
    }
    return value;
  }

  ffi::Map<ffi::String, ffi::Any>
  RestoreAnnotations(const ffi::Map<ffi::String, ffi::Any> &annotations) {
    ffi::Map<ffi::String, ffi::Any> result;
    for (const auto &[key, value] : annotations) {
      result.Set(key, RestoreValue(value));
    }
    return result;
  }

private:
  Layout RestoreLayout(const Layout &layout) {
    auto found = layouts_.find(layout.get());
    if (found != layouts_.end())
      return found->second;
    ICHECK(SupportedLayout(layout)) << "unsupported grouping snapshot layout";

    Layout restored;
    if (const auto *cute = layout.as<CuteLayoutNode>()) {
      // These structural fields are the source of truth.  The constructor
      // rebuilds forward_index with the current canonical placeholders, while
      // retaining dynamic dimensions' Vars from the deserialized function.
      restored = CuteLayout(cute->GetLogicalShape(), cute->GetModeShape(),
                            cute->GetModeStride(), cute->GetDimLevels());
    } else {
      auto forward = layout->GetForwardIndex().Map([&](const PrimExpr &expr) {
        return tir::Substitute(expr, placeholders_);
      });
      restored = Layout(layout->InputShape(), forward);
    }
    layouts_.emplace(layout.get(), restored);
    return restored;
  }

  tir::Stmt VisitStmt_(const tir::BlockNode *op) final {
    tir::Block block = Downcast<tir::Block>(tir::StmtMutator::VisitStmt_(op));
    block.CopyOnWrite()->annotations = RestoreAnnotations(block->annotations);
    return block;
  }

  tir::Stmt VisitStmt_(const tir::ForNode *op) final {
    tir::For loop = Downcast<tir::For>(tir::StmtMutator::VisitStmt_(op));
    loop.CopyOnWrite()->annotations = RestoreAnnotations(loop->annotations);
    return loop;
  }

  tir::Stmt VisitStmt_(const tir::AttrStmtNode *op) final {
    tir::AttrStmt attr =
        Downcast<tir::AttrStmt>(tir::StmtMutator::VisitStmt_(op));
    attr.CopyOnWrite()->node = RestoreValue(attr->node);
    return attr;
  }

  ffi::Map<tir::Var, PrimExpr> placeholders_;
  std::unordered_map<const LayoutNode *, Layout> layouts_;
};

} // namespace grouping_snapshot_detail

/*! \brief Check layout support before making any grouping changes. */
inline bool CanSnapshotGroupingFunction(const tir::PrimFunc &func) {
  grouping_snapshot_detail::MetadataCheck check;
  check.CheckFunction(func);
  return check.supported;
}

/*! \brief Save the function and canonical placeholder witnesses in one graph.
 */
inline String SnapshotGroupingFunction(const tir::PrimFunc &func) {
  grouping_snapshot_detail::MetadataCheck check;
  check.CheckFunction(func);
  ICHECK(check.supported) << "unsupported grouping snapshot layout";
  ffi::Array<tir::Var> placeholders;
  placeholders.reserve(check.input_dimensions);
  for (size_t i = 0; i < check.input_dimensions; ++i) {
    placeholders.push_back(InputPlaceholder(i));
  }
  // A separate SaveJSON call for the witnesses would lose their identity
  // relationship to the layout expressions in the saved function.
  return String(SaveJSON(ffi::Array<ffi::Any>{func, placeholders}));
}

/*! \brief Restore all allocations and metadata, then rebind layout caches. */
inline tir::PrimFunc RestoreGroupingFunction(const String &snapshot) {
  auto envelope = LoadJSON(std::string(snapshot)).cast<ffi::Array<ffi::Any>>();
  ICHECK_EQ(envelope.size(), 2U) << "invalid grouping snapshot envelope";
  tir::PrimFunc func = envelope[0].cast<tir::PrimFunc>();
  auto placeholders = envelope[1].cast<ffi::Array<tir::Var>>();
  grouping_snapshot_detail::MetadataCheck check;
  check.CheckFunction(func);
  ICHECK(check.supported) << "unsupported grouping snapshot layout";
  ICHECK_EQ(placeholders.size(), check.input_dimensions)
      << "incomplete grouping snapshot placeholder witnesses";

  grouping_snapshot_detail::LayoutRestorer restore(placeholders);
  if (func->attrs.defined()) {
    func.CopyOnWrite()->attrs =
        DictAttrs(restore.RestoreAnnotations(func->attrs->dict));
  }
  func.CopyOnWrite()->body = restore(func->body);
  return func;
}

} // namespace residency
} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_GROUPING_SNAPSHOT_H_
