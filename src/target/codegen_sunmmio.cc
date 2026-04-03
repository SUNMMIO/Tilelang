#include "codegen_sunmmio.h"

#include <tvm/ir/type.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>

#include <algorithm>
#include <utility>

namespace tvm {
namespace codegen {
using namespace tir;

CodeGenTileLangSunMMIO::CodeGenTileLangSunMMIO() = default;

void CodeGenTileLangSunMMIO::Init() {
  Clear();
  initialized_ = true;
}

void CodeGenTileLangSunMMIO::Clear() {
  summary_.str("");
  summary_.clear();
  expr_counter_ = 0;
  functions_.clear();
  current_function_ = nullptr;
  var_table_.clear();
  buffer_registry_.clear();
  loop_stack_.clear();
  attr_stack_.clear();
  diagnostics_.clear();
  unsupported_reports_.clear();
  scoped_vars_.clear();
  scoped_buffers_.clear();
  var_scope_markers_.clear();
  buffer_scope_markers_.clear();
  initialized_ = false;
}

ExprInfo CodeGenTileLangSunMMIO::EvalExpr(const tvm::PrimExpr& expr) {
  return tir::ExprFunctor<ExprInfo(const tvm::PrimExpr&)>::VisitExpr(expr);
}

void CodeGenTileLangSunMMIO::AddFunction(const GlobalVar& gvar, const tir::PrimFunc& f) {
  if (!initialized_) {
    Init();
  }
  RecordFunctionMetadata(gvar, f);

  EnterScope();
  for (const tir::Var& p : f->params) {
    BindVar(p, MakeExprInfo(p.dtype(), "param", p->name_hint));
  }
  for (const auto& [_, buffer] : f->buffer_map) {
    RegisterBuffer(buffer, true);
  }
  VisitStmt(f->body);
  ExitScope();
  current_function_ = nullptr;
}

std::string CodeGenTileLangSunMMIO::Finish() {
  if (!initialized_) {
    Init();
  }

  // Keep a minimal skeleton prefix for compatibility with existing source-only
  // tests while the backend is traversal-first and not emitting real IR yet.
  summary_ << "sunmmio.module {\n";
  for (const FunctionInfo& fn : functions_) {
    summary_ << "  sunmmio.func @" << fn.name << " {\n";
    summary_ << "    sunmmio.for /* traversal placeholder */\n";
    summary_ << "    sunmmio.load /* traversal placeholder */\n";
    summary_ << "    sunmmio.add /* traversal placeholder */\n";
    summary_ << "    sunmmio.store /* traversal placeholder */\n";
    summary_ << "    sunmmio.return\n";
    summary_ << "  }\n";
  }
  summary_ << "}\n";
  summary_ << "sunmmio.traversal_summary\n";
  summary_ << "functions: " << functions_.size() << "\n";
  for (const FunctionInfo& fn : functions_) {
    summary_ << "  - " << fn.name << " params=" << fn.param_names.size() << " [";
    for (size_t i = 0; i < fn.param_names.size(); ++i) {
      if (i != 0) {
        summary_ << ", ";
      }
      summary_ << fn.param_names[i] << ":" << fn.param_kinds[i];
    }
    summary_ << "]\n";
  }
  summary_ << "diagnostics: " << diagnostics_.size() << "\n";
  for (const std::string& d : diagnostics_) {
    summary_ << "  * " << d << "\n";
  }
  summary_ << "status: traversal_only_no_emission\n";

  initialized_ = false;
  return summary_.str();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::SeqStmtNode* op) {
  for (const Stmt& stmt : op->seq) {
    VisitStmt(stmt);
  }
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::ForNode* op) {
  LoopInfo loop{op->loop_var, EvalExpr(op->min), EvalExpr(op->extent), op->kind};
  RecordLoop(loop);
  loop_stack_.push_back(loop);
  EnterScope();
  BindVar(op->loop_var, MakeExprInfo(op->loop_var.dtype(), "loop_var", op->loop_var->name_hint));
  VisitStmt(op->body);
  ExitScope();
  loop_stack_.pop_back();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::LetStmtNode* op) {
  ExprInfo value = EvalExpr(op->value);
  EnterScope();
  BindVar(op->var, MakeExprInfo(op->var.dtype(), "let_var", op->var->name_hint));
  diagnostics_.push_back("let " + op->var->name_hint + " = " + value.text);
  VisitStmt(op->body);
  ExitScope();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::AttrStmtNode* op) {
  AttrInfo attr{op->node, op->attr_key, EvalExpr(op->value)};
  attr_stack_.push_back(attr);
  RecordAttr(attr);
  VisitStmt(op->body);
  attr_stack_.pop_back();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::IfThenElseNode* op) {
  ExprInfo cond = EvalExpr(op->condition);
  diagnostics_.push_back("if condition=" + cond.text);
  VisitStmt(op->then_case);
  if (op->else_case.defined()) {
    VisitStmt(op->else_case.value());
  }
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::AllocateNode* op) {
  EnterScope();
  std::ostringstream os;
  os << "allocate " << op->buffer_var->name_hint << " extents=[";
  for (size_t i = 0; i < op->extents.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << EvalExpr(op->extents[i]).text;
  }
  os << "] dtype=" << PrintType(op->dtype);
  diagnostics_.push_back(os.str());
  BindVar(op->buffer_var, MakeExprInfo(op->dtype, "allocate_var", op->buffer_var->name_hint));
  VisitStmt(op->body);
  ExitScope();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::AllocateConstNode* op) {
  EnterScope();
  std::ostringstream os;
  os << "allocate_const " << op->buffer_var->name_hint << " extents=[";
  for (size_t i = 0; i < op->extents.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << EvalExpr(op->extents[i]).text;
  }
  os << "] dtype=" << PrintType(op->dtype);
  diagnostics_.push_back(os.str());
  BindVar(op->buffer_var,
          MakeExprInfo(op->dtype, "allocate_const_var", op->buffer_var->name_hint));
  VisitStmt(op->body);
  ExitScope();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::DeclBufferNode* op) {
  RegisterBuffer(op->buffer, false);
  VisitStmt(op->body);
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::BufferStoreNode* op) {
  RegisterBuffer(op->buffer, false);
  ExprInfo value = EvalExpr(op->value);
  std::ostringstream os;
  os << "store " << op->buffer->name << "[";
  for (size_t i = 0; i < op->indices.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << EvalExpr(op->indices[i]).text;
  }
  os << "] = " << value.text;
  diagnostics_.push_back(os.str());
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::BufferRealizeNode* op) {
  RegisterBuffer(op->buffer, false);
  std::ostringstream os;
  os << "buffer_realize " << op->buffer->name << " bounds=[";
  for (size_t i = 0; i < op->bounds.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    ExprInfo min = EvalExpr(op->bounds[i]->min);
    ExprInfo extent = EvalExpr(op->bounds[i]->extent);
    os << "(" << min.text << ", " << extent.text << ")";
  }
  os << "]";
  diagnostics_.push_back(os.str());
  VisitStmt(op->body);
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::AssertStmtNode* op) {
  ExprInfo cond = EvalExpr(op->condition);
  ExprInfo msg = EvalExpr(op->message);
  diagnostics_.push_back("assert condition=" + cond.text + " message=" + msg.text);
  VisitStmt(op->body);
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::EvaluateNode* op) {
  ExprInfo value = EvalExpr(op->value);
  diagnostics_.push_back("evaluate " + value.text);
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::BlockNode* op) {
  EnterScope();
  for (const IterVar& iv : op->iter_vars) {
    BindVar(iv->var, MakeExprInfo(iv->var.dtype(), "block_iter_var", iv->var->name_hint));
  }
  for (const Buffer& alloc : op->alloc_buffers) {
    RegisterBuffer(alloc, false);
  }
  for (const BufferRegion& r : op->reads) {
    RegisterBuffer(r->buffer, false);
    for (const Range& range : r->region) {
      EvalExpr(range->min);
      EvalExpr(range->extent);
    }
  }
  for (const BufferRegion& r : op->writes) {
    RegisterBuffer(r->buffer, false);
    for (const Range& range : r->region) {
      EvalExpr(range->min);
      EvalExpr(range->extent);
    }
  }
  if (op->init.defined()) {
    VisitStmt(op->init.value());
  }
  VisitStmt(op->body);
  ExitScope();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::BlockRealizeNode* op) {
  for (const PrimExpr& v : op->iter_values) {
    EvalExpr(v);
  }
  EvalExpr(op->predicate);
  VisitStmt(op->block);
}

void CodeGenTileLangSunMMIO::VisitStmtDefault_(const Object* op) {
  UnsupportedStmt(op, "No traversal handler implemented.");
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::VarNode* op) {
  auto it = var_table_.find(op);
  if (it != var_table_.end()) {
    return it->second;
  }
  ExprInfo info = MakeExprInfo(op->dtype, "var", op->name_hint);
  var_table_[op] = info;
  return info;
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::SizeVarNode* op) {
  auto it = var_table_.find(op);
  if (it != var_table_.end()) {
    return it->second;
  }
  ExprInfo info = MakeExprInfo(op->dtype, "size_var", op->name_hint);
  var_table_[op] = info;
  return info;
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::IntImmNode* op) {
  return MakeExprInfo(op->dtype, "int_imm", std::to_string(op->value), true);
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::FloatImmNode* op) {
  std::ostringstream os;
  os << op->value;
  return MakeExprInfo(op->dtype, "float_imm", os.str(), true);
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::StringImmNode* op) {
  return MakeExprInfo(op->dtype, "string_imm", "\"" + static_cast<std::string>(op->value) + "\"",
                      true);
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::CastNode* op) {
  ExprInfo value = EvalExpr(op->value);
  return MakeExprInfo(op->dtype, "cast",
                      "cast(" + PrintType(op->dtype) + ", " + value.text + ")");
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::CallNode* op) {
  CallBucket bucket = ClassifyCall(op);
  if (bucket == CallBucket::kUnsupported) {
    UnsupportedExpr(op, "Unsupported call target.");
  }
  std::string op_name = "<unknown>";
  if (const auto* op_node = op->op.as<OpNode>()) {
    op_name = op_node->name;
  } else if (const auto* gv = op->op.as<GlobalVarNode>()) {
    op_name = gv->name_hint;
  }
  std::ostringstream os;
  os << ToString(bucket) << ":" << op_name << "(";
  for (size_t i = 0; i < op->args.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << EvalExpr(op->args[i]).text;
  }
  os << ")";
  diagnostics_.push_back("call " + os.str());
  return MakeExprInfo(op->dtype, "call", os.str());
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::AddNode* op) {
  return EmitBinaryExprInfo("add", op->a, op->b, op->dtype);
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::SubNode* op) {
  return EmitBinaryExprInfo("sub", op->a, op->b, op->dtype);
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::MulNode* op) {
  return EmitBinaryExprInfo("mul", op->a, op->b, op->dtype);
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::DivNode* op) {
  return EmitBinaryExprInfo("div", op->a, op->b, op->dtype);
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::ModNode* op) {
  return EmitBinaryExprInfo("mod", op->a, op->b, op->dtype);
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::FloorDivNode* op) {
  return EmitBinaryExprInfo("floordiv", op->a, op->b, op->dtype);
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::FloorModNode* op) {
  return EmitBinaryExprInfo("floormod", op->a, op->b, op->dtype);
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::MinNode* op) {
  return EmitBinaryExprInfo("min", op->a, op->b, op->dtype);
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::MaxNode* op) {
  return EmitBinaryExprInfo("max", op->a, op->b, op->dtype);
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::EQNode* op) {
  return EmitBinaryExprInfo("eq", op->a, op->b, op->dtype);
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::NENode* op) {
  return EmitBinaryExprInfo("ne", op->a, op->b, op->dtype);
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::LTNode* op) {
  return EmitBinaryExprInfo("lt", op->a, op->b, op->dtype);
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::LENode* op) {
  return EmitBinaryExprInfo("le", op->a, op->b, op->dtype);
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::GTNode* op) {
  return EmitBinaryExprInfo("gt", op->a, op->b, op->dtype);
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::GENode* op) {
  return EmitBinaryExprInfo("ge", op->a, op->b, op->dtype);
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::AndNode* op) {
  return EmitBinaryExprInfo("and", op->a, op->b, op->dtype);
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::OrNode* op) {
  return EmitBinaryExprInfo("or", op->a, op->b, op->dtype);
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::NotNode* op) {
  ExprInfo value = EvalExpr(op->a);
  return MakeExprInfo(op->dtype, "not", "not(" + value.text + ")");
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::SelectNode* op) {
  ExprInfo cond = EvalExpr(op->condition);
  ExprInfo t = EvalExpr(op->true_value);
  ExprInfo f = EvalExpr(op->false_value);
  return MakeExprInfo(op->dtype, "select",
                      "select(" + cond.text + ", " + t.text + ", " + f.text + ")");
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::BufferLoadNode* op) {
  RegisterBuffer(op->buffer, false);
  std::ostringstream os;
  os << "load(" << op->buffer->name << "[";
  for (size_t i = 0; i < op->indices.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << EvalExpr(op->indices[i]).text;
  }
  os << "])";
  return MakeExprInfo(op->dtype, "buffer_load", os.str());
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::ProducerLoadNode* op) {
  UnsupportedExpr(op, "ProducerLoad is not handled in SunMMIO traversal skeleton.");
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::RampNode* op) {
  ExprInfo base = EvalExpr(op->base);
  ExprInfo stride = EvalExpr(op->stride);
  std::ostringstream os;
  os << "ramp(" << base.text << ", " << stride.text << ", lanes=" << op->lanes << ")";
  return MakeExprInfo(op->dtype, "ramp", os.str());
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::BroadcastNode* op) {
  ExprInfo value = EvalExpr(op->value);
  std::ostringstream os;
  os << "broadcast(" << value.text << ", lanes=" << op->lanes << ")";
  return MakeExprInfo(op->dtype, "broadcast", os.str());
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::ShuffleNode* op) {
  std::ostringstream os;
  os << "shuffle(vectors=[";
  for (size_t i = 0; i < op->vectors.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << EvalExpr(op->vectors[i]).text;
  }
  os << "], indices=[";
  for (size_t i = 0; i < op->indices.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << EvalExpr(op->indices[i]).text;
  }
  os << "])";
  return MakeExprInfo(op->dtype, "shuffle", os.str());
}

ExprInfo CodeGenTileLangSunMMIO::VisitExpr_(const tir::LetNode* op) {
  ExprInfo value = EvalExpr(op->value);
  EnterScope();
  BindVar(op->var, MakeExprInfo(op->var.dtype(), "let_expr_var", op->var->name_hint));
  ExprInfo body = EvalExpr(op->body);
  ExitScope();
  return MakeExprInfo(op->dtype, "let_expr",
                      "let(" + op->var->name_hint + "=" + value.text + ", " +
                          body.text + ")");
}

ExprInfo CodeGenTileLangSunMMIO::VisitExprDefault_(const Object* op) {
  UnsupportedExpr(op,
                  "Expr node is not supported in traversal skeleton "
                  "(including legacy tir.Load / tir.Any if present).");
}

ExprInfo CodeGenTileLangSunMMIO::MakeExprInfo(DataType dtype, const char* kind,
                                              std::string text,
                                              bool is_constant) const {
  return ExprInfo{dtype, kind, std::move(text), is_constant};
}

ExprInfo CodeGenTileLangSunMMIO::EmitBinaryExprInfo(const char* kind,
                                                    const tvm::PrimExpr& lhs,
                                                    const tvm::PrimExpr& rhs,
                                                    tvm::DataType dtype) {
  ExprInfo left = EvalExpr(lhs);
  ExprInfo right = EvalExpr(rhs);
  std::ostringstream os;
  os << kind << "(" << left.text << ", " << right.text << ")";
  return MakeExprInfo(dtype, kind, os.str());
}

std::string CodeGenTileLangSunMMIO::PrintType(tvm::DataType dtype) const {
  if (dtype.is_void()) {
    return "void";
  }
  if (dtype.is_handle()) {
    return "handle";
  }
  if (dtype.is_float()) {
    return "f" + std::to_string(dtype.bits());
  }
  if (dtype.is_bfloat16()) {
    return "bf16";
  }
  if (dtype.is_int()) {
    return "i" + std::to_string(dtype.bits());
  }
  if (dtype.is_uint()) {
    return "u" + std::to_string(dtype.bits());
  }
  if (dtype.is_bool()) {
    return "bool";
  }
  return "unknown";
}

void CodeGenTileLangSunMMIO::EnterScope() {
  var_scope_markers_.push_back(scoped_vars_.size());
  buffer_scope_markers_.push_back(scoped_buffers_.size());
}

void CodeGenTileLangSunMMIO::ExitScope() {
  ICHECK(!var_scope_markers_.empty());
  ICHECK(!buffer_scope_markers_.empty());

  size_t var_marker = var_scope_markers_.back();
  var_scope_markers_.pop_back();
  while (scoped_vars_.size() > var_marker) {
    const tir::VarNode* var = scoped_vars_.back();
    scoped_vars_.pop_back();
    var_table_.erase(var);
  }

  size_t buffer_marker = buffer_scope_markers_.back();
  buffer_scope_markers_.pop_back();
  while (scoped_buffers_.size() > buffer_marker) {
    const tir::BufferNode* buffer = scoped_buffers_.back();
    scoped_buffers_.pop_back();
    buffer_registry_.erase(buffer);
  }
}

void CodeGenTileLangSunMMIO::BindVar(const tir::Var& var, const ExprInfo& info) {
  var_table_[var.get()] = info;
  scoped_vars_.push_back(var.get());
}

void CodeGenTileLangSunMMIO::RegisterBuffer(const tir::Buffer& buffer,
                                            bool is_external) {
  if (!buffer.defined()) {
    return;
  }
  const tir::BufferNode* key = buffer.get();
  if (buffer_registry_.count(key)) {
    return;
  }
  BufferInfo info;
  info.name = buffer->name;
  info.buffer = buffer;
  info.dtype = buffer->dtype;
  info.shape = buffer->shape;
  info.scope = buffer.scope();
  info.is_external = is_external;
  buffer_registry_[key] = std::move(info);
  scoped_buffers_.push_back(key);
}

void CodeGenTileLangSunMMIO::RecordAttr(const AttrInfo& attr) {
  std::ostringstream os;
  os << "attr key=" << attr.key << " value=" << attr.value.text;
  diagnostics_.push_back(os.str());
}

void CodeGenTileLangSunMMIO::RecordLoop(const LoopInfo& loop) {
  std::ostringstream os;
  os << "loop " << loop.loop_var->name_hint << " min=" << loop.min.text
     << " extent=" << loop.extent.text << " kind=" << static_cast<int>(loop.kind);
  diagnostics_.push_back(os.str());
}

void CodeGenTileLangSunMMIO::RecordFunctionMetadata(const GlobalVar& gvar,
                                                    const tir::PrimFunc& f) {
  FunctionInfo info;
  info.gvar = gvar;
  info.prim_func = f;
  info.name = gvar->name_hint;
  info.param_names.reserve(f->params.size());
  info.param_kinds.reserve(f->params.size());
  for (const tir::Var& p : f->params) {
    info.param_names.push_back(p->name_hint);
    info.param_kinds.push_back(ClassifyParamKind(p, f));
  }
  functions_.push_back(std::move(info));
  current_function_ = &functions_.back();
}

std::string CodeGenTileLangSunMMIO::ClassifyParamKind(const tir::Var& param,
                                                      const tir::PrimFunc& f) const {
  if (f->buffer_map.count(param)) {
    return "buffer";
  }
  if (param->type_annotation.as<PointerTypeNode>()) {
    return "pointer";
  }
  if (param.dtype().is_handle()) {
    return "handle";
  }
  return "scalar";
}

CodeGenTileLangSunMMIO::CallBucket
CodeGenTileLangSunMMIO::ClassifyCall(const tir::CallNode* op) const {
  if (const auto* gv = op->op.as<GlobalVarNode>()) {
    (void)gv;
    PrimExpr expr = tvm::ffi::GetRef<PrimExpr>(op);
    tir::CallEffectKind effect = SideEffect(expr);
    return effect <= tir::CallEffectKind::kPure ? CallBucket::kExternPure
                                                : CallBucket::kExternSideEffect;
  }

  const auto* op_node = op->op.as<OpNode>();
  if (!op_node) {
    return CallBucket::kUnsupported;
  }
  std::string name = op_node->name;

  if (name == "tl.mma_sunmmio" || name == "tl.dma_copy" ||
      name.find("sunmmio") != std::string::npos) {
    return CallBucket::kSunMMIOIntrinsic;
  }
  if (name.rfind("tl.", 0) == 0) {
    return CallBucket::kTileLangIntrinsic;
  }
  if (name == "tir.tvm_access_ptr" || name == "tir.address_of" ||
      name.find("alloc") != std::string::npos ||
      name.find("reinterpret") != std::string::npos) {
    return CallBucket::kMemory;
  }
  if (name.find("sync") != std::string::npos || name.find("barrier") != std::string::npos) {
    return CallBucket::kSync;
  }
  if (name.find("shuffle") != std::string::npos || name.find("vector") != std::string::npos) {
    return CallBucket::kVector;
  }
  if (name.find("exp") != std::string::npos || name.find("log") != std::string::npos ||
      name.find("sin") != std::string::npos || name.find("cos") != std::string::npos ||
      name.find("sqrt") != std::string::npos || name.find("pow") != std::string::npos) {
    return CallBucket::kMath;
  }
  if (name.rfind("tir.", 0) == 0) {
    return CallBucket::kBuiltin;
  }

  PrimExpr expr = tvm::ffi::GetRef<PrimExpr>(op);
  tir::CallEffectKind effect = SideEffect(expr);
  return effect <= tir::CallEffectKind::kPure ? CallBucket::kExternPure
                                              : CallBucket::kExternSideEffect;
}

std::string CodeGenTileLangSunMMIO::ToString(CallBucket bucket) const {
  switch (bucket) {
  case CallBucket::kBuiltin:
    return "builtin";
  case CallBucket::kExternPure:
    return "extern_pure";
  case CallBucket::kExternSideEffect:
    return "extern_side_effect";
  case CallBucket::kMath:
    return "math";
  case CallBucket::kMemory:
    return "memory";
  case CallBucket::kSync:
    return "sync";
  case CallBucket::kVector:
    return "vector";
  case CallBucket::kTileLangIntrinsic:
    return "tilelang_intrinsic";
  case CallBucket::kSunMMIOIntrinsic:
    return "sunmmio_intrinsic";
  case CallBucket::kUnsupported:
    return "unsupported";
  }
  return "unsupported";
}

[[noreturn]] void CodeGenTileLangSunMMIO::UnsupportedStmt(
    const Object* op, const std::string& detail) const {
  std::ostringstream os;
  os << "CodeGenTileLangSunMMIO unsupported stmt: " << op->GetTypeKey();
  if (!detail.empty()) {
    os << " (" << detail << ")";
  }
  LOG(FATAL) << os.str();
  TVM_FFI_UNREACHABLE();
}

[[noreturn]] void CodeGenTileLangSunMMIO::UnsupportedExpr(
    const Object* op, const std::string& detail) const {
  std::ostringstream os;
  os << "CodeGenTileLangSunMMIO unsupported expr: " << op->GetTypeKey();
  if (!detail.empty()) {
    os << " (" << detail << ")";
  }
  LOG(FATAL) << os.str();
  TVM_FFI_UNREACHABLE();
}
} // namespace codegen
} // namespace tvm
