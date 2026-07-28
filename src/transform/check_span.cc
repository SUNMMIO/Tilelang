/*!
 * \file check_span.cc
 * \brief Check final TIR nodes for missing source Span metadata.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <map>
#include <sstream>
#include <string>
#include <unordered_set>

#include "../op/builtin.h"
#include "common/sunmmio_logging.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

namespace {

class MissingSpanCollector final : public StmtExprVisitor {
public:
  void CheckPrimFunc(const PrimFunc &func) {
    Record(func.get(), func->span);
    VisitStmt(func->body);
  }

  size_t missing_count() const { return missing_count_; }

  std::string Summary() const {
    std::ostringstream os;
    bool first = true;
    for (const auto &[type_key, count] : missing_types_) {
      if (!first) {
        os << ", ";
      }
      first = false;
      os << type_key << "=" << count;
    }
    return os.str();
  }

  void VisitStmt(const Stmt &stmt) final {
    if (stmt.defined()) {
      Record(stmt.get(), stmt->span);
    }
    StmtExprVisitor::VisitStmt(stmt);
  }

protected:
  void VisitExpr(const PrimExpr &expr) final {
    if (expr.defined()) {
      Record(expr.get(), expr->span);
    }
    StmtExprVisitor::VisitExpr(expr);
  }

private:
  void Record(const Object *node, const Span &span) {
    if (node == nullptr || span.defined() || !visited_.insert(node).second) {
      return;
    }
    ++missing_count_;
    ++missing_types_[node->GetTypeKey()];
  }

  size_t missing_count_{0};
  std::map<std::string, size_t> missing_types_;
  std::unordered_set<const Object *> visited_;
};

std::string GetGlobalSymbol(const PrimFunc &func) {
  return static_cast<std::string>(
      func->GetAttr<String>(tvm::attr::kGlobalSymbol).value_or("<anonymous>"));
}

} // namespace

Pass CheckSpan() {
  auto pass_func = [](PrimFunc func, IRModule mod, PassContext ctx) {
    (void)mod;
    const String configured_level =
        ctx->GetConfig<String>(kCheckSpanLogLevel, String("WARNING")).value();
    const std::string level = static_cast<std::string>(configured_level);
    if (level != "FATAL" && level != "WARNING") {
      SUNMMIO_LOG(FATAL, func.get())
          << "Invalid " << kCheckSpanLogLevel << " value `" << level
          << "`; expected `FATAL` or `WARNING`";
    }

    MissingSpanCollector collector;
    collector.CheckPrimFunc(func);
    if (collector.missing_count() == 0) {
      return func;
    }

    std::ostringstream message;
    message << "CheckSpan found " << collector.missing_count()
            << " TIR node(s) without Span in PrimFunc `"
            << GetGlobalSymbol(func)
            << "` after target lowering. Missing node types: "
            << collector.Summary();

    if (level == "FATAL") {
      SUNMMIO_LOG(FATAL, func.get()) << message.str();
    } else {
      SUNMMIO_LOG(WARNING, func.get()) << message.str();
    }
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.CheckSpan", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.CheckSpan", CheckSpan);
}

} // namespace tl
} // namespace tvm
