/*!
 * \file attach_dsl_span.cc
 * \brief Convert TileLang frontend source-location markers into TVM Span.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/ir/source_map.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <string>
#include <vector>

#include "common/attr.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

namespace {

std::vector<std::string> SplitFields(const std::string &value) {
  std::vector<std::string> fields;
  std::string current;
  for (char ch : value) {
    if (ch == '|') {
      fields.push_back(current);
      current.clear();
    } else {
      current.push_back(ch);
    }
  }
  fields.push_back(current);
  return fields;
}

std::string UnescapeField(const std::string &value) {
  std::string result;
  result.reserve(value.size());
  for (size_t i = 0; i < value.size(); ++i) {
    if (value[i] == '%' && i + 2 < value.size()) {
      const std::string code = value.substr(i, 3);
      if (code == "%25") {
        result.push_back('%');
        i += 2;
        continue;
      }
      if (code == "%7C") {
        result.push_back('|');
        i += 2;
        continue;
      }
    }
    result.push_back(value[i]);
  }
  return result;
}

bool ParseInt(const std::string &value, int *out) {
  try {
    size_t parsed = 0;
    int parsed_value = std::stoi(value, &parsed);
    if (parsed != value.size()) {
      return false;
    }
    *out = parsed_value;
    return true;
  } catch (...) {
    return false;
  }
}

Span DecodeDslSpan(const PrimExpr &value) {
  const auto *str = value.as<StringImmNode>();
  if (str == nullptr) {
    return Span();
  }
  std::vector<std::string> fields =
      SplitFields(static_cast<std::string>(str->value));
  if (fields.size() != 3) {
    return Span();
  }

  int line = 0;
  if (!ParseInt(fields[1], &line)) {
    return Span();
  }
  if (line <= 0) {
    return Span();
  }

  return Span(SourceName::Get(UnescapeField(fields[0])), line, line, 0, 0);
}

class AttachDslSpanMutator : public StmtExprMutator {
public:
  Stmt VisitStmt(const Stmt &stmt) override {
    Stmt ret = StmtExprMutator::VisitStmt(stmt);
    return AttachSpan(ret);
  }

protected:
  PrimExpr VisitExpr(const PrimExpr &expr) override {
    PrimExpr ret = StmtExprMutator::VisitExpr(expr);
    return AttachSpan(ret);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) override {
    if (op->attr_key == attr::kDslSpan) {
      Span span = DecodeDslSpan(op->value);
      const bool pushed = span.defined();
      if (pushed) {
        span_stack_.push_back(span);
      }
      Stmt body = VisitStmt(op->body);
      if (pushed) {
        span_stack_.pop_back();
      }
      return body;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

private:
  Span CurrentSpan() const {
    if (span_stack_.empty()) {
      return Span();
    }
    return span_stack_.back();
  }

  Stmt AttachSpan(const Stmt &stmt) const {
    Span span = CurrentSpan();
    if (stmt.defined() && stmt.as<BlockNode>() == nullptr &&
        stmt.as<BlockRealizeNode>() == nullptr && span.defined() &&
        !stmt->span.defined()) {
      stmt->span = span;
    }
    return stmt;
  }

  PrimExpr AttachSpan(const PrimExpr &expr) const {
    Span span = CurrentSpan();
    if (expr.defined() && span.defined() && !expr->span.defined()) {
      expr->span = span;
    }
    return expr;
  }

  std::vector<Span> span_stack_;
};

} // namespace

Pass AttachDslSpan() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    (void)m;
    (void)ctx;
    Stmt body = AttachDslSpanMutator()(f->body);
    return PrimFunc(f->params, body, f->ret_type, f->buffer_map, f->attrs,
                    f->span);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AttachDslSpan", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AttachDslSpan", AttachDslSpan);
}

} // namespace tl
} // namespace tvm
