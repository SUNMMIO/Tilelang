#ifndef TVM_TL_TRANSFORM_COMMON_SUNMMIO_LOGGING_H_
#define TVM_TL_TRANSFORM_COMMON_SUNMMIO_LOGGING_H_

#include <tvm/ir/expr.h>
#include <tvm/ir/source_map.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/object.h>
#include <tvm/tir/stmt.h>

#include <ostream>
#include <sstream>
#include <string>

namespace tvm {
namespace tl {
namespace log {

inline std::string FormatSpanForLog(const Span &span) {
  if (!span.defined()) {
    return "";
  }
  if (const auto *seq = span.as<SequentialSpanNode>()) {
    if (!seq->spans.empty()) {
      return FormatSpanForLog(seq->spans[0]);
    }
    return "";
  }
  const auto *node = span.as<SpanNode>();
  if (!node || !node->source_name.defined()) {
    return "";
  }

  std::ostringstream os;
  os << static_cast<std::string>(node->source_name->name) << ":" << node->line;
  if (node->column > 0) {
    os << ":" << node->column;
  }
  return os.str();
}

inline std::string FormatObjectSpanForLog(const Object *op) {
  if (op == nullptr) {
    return "";
  }
  if (op->IsInstance<tir::StmtNode>()) {
    const auto *stmt = static_cast<const tir::StmtNode *>(op);
    return FormatSpanForLog(stmt->span);
  }
  if (op->IsInstance<BaseExprNode>()) {
    const auto *expr = static_cast<const BaseExprNode *>(op);
    return FormatSpanForLog(expr->span);
  }
  return "";
}

inline void AppendObjectSpanForLog(std::ostream *os, const Object *op) {
  const std::string loc = FormatObjectSpanForLog(op);
  if (!loc.empty()) {
    *os << "\n  at TileLang DSL: " << loc;
  }
}

class LOGFatal {
public:
  LOGFatal(const char *file, int line, const Object *op)
      : log_(file, line), op_(op) {}

  ~LOGFatal() TVM_THROW_EXCEPTION {
    AppendObjectSpanForLog(&log_.stream(), op_);
  }

  std::ostream &stream() { return log_.stream(); }

private:
  runtime::detail::LogFatal log_;
  const Object *op_;
};

} // namespace log
} // namespace tl
} // namespace tvm

#define TL_FATAL(op) ::tvm::tl::log::LOGFatal(__FILE__, __LINE__, (op)).stream()

#define TL_CHECK(cond, op)                                                     \
  (cond) ? (void)0                                                             \
         : ::tvm::runtime::detail::LogMessageVoidify() &                       \
               ::tvm::tl::log::LOGFatal(__FILE__, __LINE__, (op)).stream()     \
                   << "Check failed: (" #cond ") is false: "

#define SUNMMIO_FATAL(op) TL_FATAL(op)

#define SUNMMIO_CHECK(cond, op) TL_CHECK((cond), (op))

#endif // TVM_TL_TRANSFORM_COMMON_SUNMMIO_LOGGING_H_
