#include "codegen_sunmmio.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace codegen {

CodeGenTileLangSunMMIO::CodeGenTileLangSunMMIO() = default;

void CodeGenTileLangSunMMIO::Init() {
  stream_.str("");
  stream_.clear();
  indent_ = 0;
  value_counter_ = 0;
  var_ids_.clear();
  initialized_ = true;
  EmitLine("sunmmio.module {");
  EnterScope();
}

void CodeGenTileLangSunMMIO::Clear() {
  stream_.str("");
  stream_.clear();
  indent_ = 0;
  value_counter_ = 0;
  var_ids_.clear();
  initialized_ = false;
}

void CodeGenTileLangSunMMIO::AddFunction(const GlobalVar& gvar, const tir::PrimFunc& f) {
  if (!initialized_) {
    Init();
  }

  std::string func_name = gvar->name_hint;
  std::ostringstream sig;
  sig << "sunmmio.func @" << func_name << "(";
  for (size_t i = 0; i < f->params.size(); ++i) {
    const tir::Var& v = f->params[i];
    std::string value_name = NewValue();
    var_ids_[v.get()] = value_name;
    if (i != 0) {
      sig << ", ";
    }
    sig << value_name << ": " << PrintType(v.dtype());
  }
  sig << ")";

  EmitLine(sig.str() + " {");
  EnterScope();
  VisitStmt(f->body);
  EmitLine("sunmmio.return");
  ExitScope();
  EmitLine("}");
}

std::string CodeGenTileLangSunMMIO::Finish() {
  if (!initialized_) {
    Init();
  }
  ExitScope();
  EmitLine("}");
  initialized_ = false;
  return stream_.str();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::ForNode* op) {
  std::string min = VisitExpr(op->min);
  std::string extent = VisitExpr(op->extent);
  std::string iv = NewValue();
  var_ids_[op->loop_var.get()] = iv;
  EmitLine("sunmmio.for " + iv + " = " + min + " to " + extent + " {");
  EnterScope();
  VisitStmt(op->body);
  ExitScope();
  EmitLine("}");
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::BufferStoreNode* op) {
  std::string value = VisitExpr(op->value);
  std::string index = op->indices.empty() ? "0" : VisitExpr(op->indices[0]);
  auto it = var_ids_.find(op->buffer->data.get());
  std::string buffer = it == var_ids_.end() ? static_cast<std::string>(op->buffer->name) : it->second;
  EmitLine("sunmmio.store " + value + ", " + buffer + "[" + index + "]");
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::AllocateNode* op) {
  std::string alloc_name = NewValue();
  var_ids_[op->buffer_var.get()] = alloc_name;
  std::string extent = op->extents.empty() ? "1" : VisitExpr(op->extents[0]);
  EmitLine(alloc_name + " = sunmmio.alloc " + extent + " : " + PrintType(op->dtype));
  VisitStmt(op->body);
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::AttrStmtNode* op) {
  VisitStmt(op->body);
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::IfThenElseNode* op) {
  std::string cond = VisitExpr(op->condition);
  EmitLine("sunmmio.if " + cond + " {");
  EnterScope();
  VisitStmt(op->then_case);
  ExitScope();
  if (op->else_case.defined()) {
    EmitLine("} else {");
    EnterScope();
    VisitStmt(op->else_case.value());
    ExitScope();
  }
  EmitLine("}");
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::EvaluateNode* op) {
  std::string value = VisitExpr(op->value);
  EmitLine("sunmmio.eval " + value);
}

std::string CodeGenTileLangSunMMIO::VisitExpr_(const tir::VarNode* op) {
  auto it = var_ids_.find(op);
  if (it != var_ids_.end()) {
    return it->second;
  }
  std::string name = NewValue();
  var_ids_[op] = name;
  return name;
}

std::string CodeGenTileLangSunMMIO::VisitExpr_(const tir::BufferLoadNode* op) {
  auto it = var_ids_.find(op->buffer->data.get());
  std::string buffer = it == var_ids_.end() ? static_cast<std::string>(op->buffer->name) : it->second;
  std::string index = op->indices.empty() ? "0" : VisitExpr(op->indices[0]);
  std::string dst = NewValue();
  EmitLine(dst + " = sunmmio.load " + buffer + "[" + index + "] : " + PrintType(op->dtype));
  return dst;
}

std::string CodeGenTileLangSunMMIO::VisitExpr_(const tir::IntImmNode* op) {
  return std::to_string(op->value);
}

std::string CodeGenTileLangSunMMIO::VisitExpr_(const tir::FloatImmNode* op) {
  std::ostringstream os;
  os << op->value;
  return os.str();
}

std::string CodeGenTileLangSunMMIO::VisitExpr_(const tir::AddNode* op) {
  return EmitBinary("sunmmio.add", op->a, op->b, op->dtype);
}

std::string CodeGenTileLangSunMMIO::VisitExpr_(const tir::SubNode* op) {
  return EmitBinary("sunmmio.sub", op->a, op->b, op->dtype);
}

std::string CodeGenTileLangSunMMIO::VisitExpr_(const tir::MulNode* op) {
  return EmitBinary("sunmmio.mul", op->a, op->b, op->dtype);
}

std::string CodeGenTileLangSunMMIO::VisitExpr_(const tir::DivNode* op) {
  return EmitBinary("sunmmio.div", op->a, op->b, op->dtype);
}

std::string CodeGenTileLangSunMMIO::VisitExpr_(const tir::CastNode* op) {
  std::string src = VisitExpr(op->value);
  std::string dst = NewValue();
  EmitLine(dst + " = sunmmio.cast " + src + " : " + PrintType(op->dtype));
  return dst;
}

std::string CodeGenTileLangSunMMIO::VisitExpr_(const tir::CallNode* op) {
  std::string dst = NewValue();
  std::ostringstream call;
  std::string call_target = "unknown";
  if (const auto* op_node = op->op.as<OpNode>()) {
    call_target = op_node->name;
  }
  call << dst << " = sunmmio.call @" << call_target << "(";
  for (size_t i = 0; i < op->args.size(); ++i) {
    if (i != 0) {
      call << ", ";
    }
    call << VisitExpr(op->args[i]);
  }
  call << ") : " << PrintType(op->dtype);
  EmitLine(call.str());
  return dst;
}

std::string CodeGenTileLangSunMMIO::VisitExprDefault_(const Object* op) {
  std::string dst = NewValue();
  EmitLine(dst + " = sunmmio.unhandled_expr");
  return dst;
}

std::string CodeGenTileLangSunMMIO::NewValue() {
  return "%v" + std::to_string(value_counter_++);
}

std::string CodeGenTileLangSunMMIO::EmitBinary(const char* op_name, const tvm::PrimExpr& lhs,
                                               const tvm::PrimExpr& rhs, tvm::DataType dtype) {
  std::string lhs_value = VisitExpr(lhs);
  std::string rhs_value = VisitExpr(rhs);
  std::string dst = NewValue();
  EmitLine(dst + " = " + op_name + " " + lhs_value + ", " + rhs_value + " : " + PrintType(dtype));
  return dst;
}

std::string CodeGenTileLangSunMMIO::PrintType(tvm::DataType dtype) const {
  if (dtype.is_float()) {
    return "f" + std::to_string(dtype.bits());
  }
  if (dtype.is_int()) {
    return "i" + std::to_string(dtype.bits());
  }
  if (dtype.is_uint()) {
    return "u" + std::to_string(dtype.bits());
  }
  if (dtype.is_bool()) {
    return "i1";
  }
  return "opaque";
}

void CodeGenTileLangSunMMIO::EmitLine(const std::string& line) {
  for (int i = 0; i < indent_; ++i) {
    stream_ << "  ";
  }
  stream_ << line << "\n";
}

void CodeGenTileLangSunMMIO::EnterScope() { ++indent_; }

void CodeGenTileLangSunMMIO::ExitScope() {
  if (indent_ > 0) {
    --indent_;
  }
}

} // namespace codegen
} // namespace tvm
