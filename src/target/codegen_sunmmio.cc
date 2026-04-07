#include "codegen_sunmmio.h"

#include <tvm/ir/type.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>

#include <algorithm>
#include <cctype>
#include <utility>

namespace tvm {
namespace codegen {
using namespace tir;

CodeGenTileLangSunMMIO::CodeGenTileLangSunMMIO() = default;

void CodeGenTileLangSunMMIO::Init() {
  Clear();
  mlir_ << "module {\n";
  module_open_ = true;
  indent_ = 1;
  initialized_ = true;
}

void CodeGenTileLangSunMMIO::Clear() {
  mlir_.str("");
  mlir_.clear();
  ssa_counter_ = 0;
  label_counter_ = 0;
  indent_ = 0;
  module_open_ = false;
  function_open_ = false;
  var_table_.clear();
  buffer_registry_.clear();
  attr_stack_.clear();
  scoped_vars_.clear();
  scoped_buffers_.clear();
  var_scope_markers_.clear();
  buffer_scope_markers_.clear();
  initialized_ = false;
}

SunMMIOValue CodeGenTileLangSunMMIO::EvalExpr(const tvm::PrimExpr& expr) {
  return tir::ExprFunctor<SunMMIOValue(const tvm::PrimExpr&)>::VisitExpr(expr);
}

void CodeGenTileLangSunMMIO::AddFunction(const GlobalVar& gvar, const tir::PrimFunc& f) {
  if (!initialized_) {
    Init();
  }
  EnterScope();
  std::vector<std::string> arg_defs;
  int arg_index = 0;
  for (const tir::Var& p : f->params) {
    std::string arg_name = "%arg" + std::to_string(arg_index++);
    if (f->buffer_map.count(p)) {
      const tir::Buffer& buffer = f->buffer_map.at(p);
      arg_defs.push_back(arg_name + ": " + MapBufferType(buffer));
      BindVar(p, SunMMIOValue{p.dtype(), arg_name, MapBufferType(buffer)});
      RegisterBuffer(buffer, true, arg_name);
    } else {
      arg_defs.push_back(arg_name + ": " + MapType(p.dtype()));
      BindVar(p, SunMMIOValue{p.dtype(), arg_name, MapType(p.dtype())});
    }
  }

  std::ostringstream sig;
  for (size_t i = 0; i < arg_defs.size(); ++i) {
    if (i != 0) {
      sig << ", ";
    }
    sig << arg_defs[i];
  }
  EmitLine("func.func @" + gvar->name_hint + "(" + sig.str() + ") {");
  indent_++;
  function_open_ = true;
  VisitStmt(f->body);
  EmitLine("return");
  indent_--;
  EmitLine("}");
  function_open_ = false;
  ExitScope();
}

std::string CodeGenTileLangSunMMIO::Finish() {
  if (!initialized_) {
    Init();
  }
  if (module_open_) {
    indent_ = std::max(0, indent_ - 1);
    EmitLine("}");
    module_open_ = false;
  }
  initialized_ = false;
  return mlir_.str();
}

void CodeGenTileLangSunMMIO::EmitLine(const std::string& line) {
  for (int i = 0; i < indent_; ++i) {
    mlir_ << "  ";
  }
  mlir_ << line << "\n";
}

std::string CodeGenTileLangSunMMIO::NewValueName() {
  return "%v" + std::to_string(ssa_counter_++);
}

std::string CodeGenTileLangSunMMIO::NewLabel(const std::string& prefix) {
  return prefix + std::to_string(label_counter_++);
}

std::string CodeGenTileLangSunMMIO::MapType(tvm::DataType dtype) const {
  if (dtype.is_void()) {
    return "none";
  }
  if (dtype.is_bool()) {
    return "i1";
  }
  if (dtype.is_bfloat16()) {
    return "bf16";
  }
  if (dtype.is_float()) {
    return "f" + std::to_string(dtype.bits());
  }
  if (dtype.is_int() || dtype.is_uint()) {
    return "i" + std::to_string(dtype.bits());
  }
  if (dtype.is_handle()) {
    return "!sunmmio.handle";
  }
  return "!sunmmio.unknown";
}

std::string CodeGenTileLangSunMMIO::MapStorageScope(const std::string& scope) const {
  if (scope.empty()) {
    return "global";
  }
  std::string out = scope;
  std::replace(out.begin(), out.end(), '.', '_');
  return out;
}

std::string CodeGenTileLangSunMMIO::MapBufferType(const tir::Buffer& buffer) const {
  std::ostringstream os;
  os << "memref<";
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    if (const auto* imm = buffer->shape[i].as<IntImmNode>()) {
      os << imm->value;
    } else {
      os << "?";
    }
    os << "x";
  }
  os << MapType(buffer->dtype) << ">";
  return os.str();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::SeqStmtNode* op) {
  for (const Stmt& stmt : op->seq) {
    VisitStmt(stmt);
  }
}

SunMMIOValue CodeGenTileLangSunMMIO::EmitConstIndex(int64_t v) {
  std::string name = NewValueName();
  EmitLine(name + " = arith.constant " + std::to_string(v) + " : index");
  return SunMMIOValue{DataType::Int(32), name, "index"};
}

SunMMIOValue CodeGenTileLangSunMMIO::EnsureIndex(const SunMMIOValue& v) {
  if (v.mlir_type == "index") {
    return v;
  }
  std::string name = NewValueName();
  EmitLine(name + " = arith.index_cast " + v.value + " : " + v.mlir_type + " to index");
  return SunMMIOValue{DataType::Int(32), name, "index"};
}

SunMMIOValue CodeGenTileLangSunMMIO::EnsureType(const SunMMIOValue& v,
                                                const std::string& mlir_type,
                                                DataType dtype) {
  if (v.mlir_type == mlir_type) {
    return v;
  }
  return EmitCast(v, dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::BindVar(const tir::Var& var,
                                             const SunMMIOValue& value) {
  var_table_[var.get()] = value;
  scoped_vars_.push_back(var.get());
  return value;
}

void CodeGenTileLangSunMMIO::RegisterBuffer(const tir::Buffer& buffer, bool is_external,
                                            const std::string& handle_hint) {
  if (!buffer.defined()) {
    return;
  }
  if (buffer_registry_.count(buffer.get())) {
    return;
  }
  BufferBinding binding;
  binding.buffer = buffer;
  binding.scope = buffer.scope();
  binding.memref_type = MapBufferType(buffer);
  binding.is_external = is_external;
  if (!handle_hint.empty()) {
    binding.handle = handle_hint;
  } else if (var_table_.count(buffer->data.get())) {
    binding.handle = var_table_.at(buffer->data.get()).value;
  } else {
    binding.handle = NewValueName();
  }
  buffer_registry_[buffer.get()] = std::move(binding);
  scoped_buffers_.push_back(buffer.get());
}

const BufferBinding& CodeGenTileLangSunMMIO::LookupBuffer(const tir::Buffer& buffer) const {
  auto it = buffer_registry_.find(buffer.get());
  ICHECK(it != buffer_registry_.end()) << "CodeGenTileLangSunMMIO: unknown buffer " << buffer;
  return it->second;
}

void CodeGenTileLangSunMMIO::EmitAlloc(const tir::Var& buffer_var, DataType dtype,
                                       const ffi::Array<PrimExpr>& extents,
                                       const std::string& scope_hint) {
  std::vector<SunMMIOValue> dyn_extents;
  std::ostringstream memref;
  memref << "memref<";
  for (size_t i = 0; i < extents.size(); ++i) {
    if (const auto* imm = extents[i].as<IntImmNode>()) {
      memref << imm->value;
    } else {
      memref << "?";
      dyn_extents.push_back(EnsureIndex(EvalExpr(extents[i])));
    }
    memref << "x";
  }
  memref << MapType(dtype) << ">";
  std::string alloc_name = NewValueName();
  std::ostringstream dyn_sig;
  for (size_t i = 0; i < dyn_extents.size(); ++i) {
    if (i != 0) {
      dyn_sig << ", ";
    }
    dyn_sig << dyn_extents[i].value;
  }
  std::string scope_attr = " {sunmmio.scope = \"" + MapStorageScope(scope_hint) + "\"}";
  if (dyn_extents.empty()) {
    EmitLine(alloc_name + " = memref.alloc()" + scope_attr + " : " + memref.str());
  } else {
    EmitLine(alloc_name + " = memref.alloc(" + dyn_sig.str() + ")" + scope_attr + " : " +
             memref.str());
  }
  BindVar(buffer_var, SunMMIOValue{dtype, alloc_name, memref.str()});
}

void CodeGenTileLangSunMMIO::EmitFor(const tir::ForNode* op) {
  SunMMIOValue min = EnsureIndex(EvalExpr(op->min));
  SunMMIOValue extent = EnsureIndex(EvalExpr(op->extent));
  SunMMIOValue step = EmitConstIndex(1);
  std::string upper = NewValueName();
  EmitLine(upper + " = arith.addi " + min.value + ", " + extent.value + " : index");
  std::string iv = "%" + op->loop_var->name_hint;
  EmitLine("scf.for " + iv + " = " + min.value + " to " + upper + " step " + step.value + " {");
  indent_++;
  EnterScope();
  BindVar(op->loop_var, SunMMIOValue{op->loop_var.dtype(), iv, "index"});
  VisitStmt(op->body);
  ExitScope();
  indent_--;
  EmitLine("}");
}

void CodeGenTileLangSunMMIO::EmitIf(const tir::IfThenElseNode* op) {
  SunMMIOValue cond = EnsureType(EvalExpr(op->condition), "i1", DataType::Bool());
  EmitLine("scf.if " + cond.value + " {");
  indent_++;
  VisitStmt(op->then_case);
  indent_--;
  if (op->else_case.defined()) {
    EmitLine("} else {");
    indent_++;
    VisitStmt(op->else_case.value());
    indent_--;
  }
  EmitLine("}");
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::ForNode* op) { EmitFor(op); }

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::LetStmtNode* op) {
  SunMMIOValue value = EvalExpr(op->value);
  EnterScope();
  BindVar(op->var, value);
  VisitStmt(op->body);
  ExitScope();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::AttrStmtNode* op) {
  ScopedAttr attr{op->node, op->attr_key, EvalExpr(op->value)};
  attr_stack_.push_back(attr);
  VisitStmt(op->body);
  attr_stack_.pop_back();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::IfThenElseNode* op) { EmitIf(op); }

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::WhileNode* op) {
  (void)op;
  UnsupportedStmt(op, "WhileNode is not supported by SunMMIO direct MLIR lowering.");
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::AllocateNode* op) {
  EnterScope();
  EmitAlloc(op->buffer_var, op->dtype, op->extents, "local");
  VisitStmt(op->body);
  ExitScope();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::AllocateConstNode* op) {
  EnterScope();
  EmitAlloc(op->buffer_var, op->dtype, op->extents, "const");
  VisitStmt(op->body);
  ExitScope();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::DeclBufferNode* op) {
  RegisterBuffer(op->buffer, false, NewValueName());
  const BufferBinding& binding = LookupBuffer(op->buffer);
  EmitLine(binding.handle + " = memref.alloc() {sunmmio.scope = \"" +
           MapStorageScope(op->buffer.scope()) + "\"} : " + binding.memref_type);
  VisitStmt(op->body);
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::BufferStoreNode* op) {
  if (!buffer_registry_.count(op->buffer.get())) {
    RegisterBuffer(op->buffer, false, NewValueName());
    const BufferBinding& binding = LookupBuffer(op->buffer);
    EmitLine(binding.handle + " = memref.alloc() {sunmmio.scope = \"" +
             MapStorageScope(op->buffer.scope()) + "\"} : " + binding.memref_type);
  }
  EmitStore(op->buffer, op->indices, EvalExpr(op->value));
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::BufferRealizeNode* op) {
  EnterScope();
  RegisterBuffer(op->buffer, false, NewValueName());
  const BufferBinding& binding = LookupBuffer(op->buffer);
  std::vector<SunMMIOValue> dyn_bounds;
  for (const Range& range : op->bounds) {
    EvalExpr(range->min);
    if (!range->extent.as<IntImmNode>()) {
      dyn_bounds.push_back(EnsureIndex(EvalExpr(range->extent)));
    }
  }
  std::ostringstream dyn_sig;
  for (size_t i = 0; i < dyn_bounds.size(); ++i) {
    if (i != 0) {
      dyn_sig << ", ";
    }
    dyn_sig << dyn_bounds[i].value;
  }
  if (dyn_bounds.empty()) {
    EmitLine(binding.handle + " = memref.alloc() {sunmmio.scope = \"" +
             MapStorageScope(op->buffer.scope()) + "\"} : " + binding.memref_type);
  } else {
    EmitLine(binding.handle + " = memref.alloc(" + dyn_sig.str() +
             ") {sunmmio.scope = \"" + MapStorageScope(op->buffer.scope()) + "\"} : " +
             binding.memref_type);
  }
  VisitStmt(op->body);
  ExitScope();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::AssertStmtNode* op) {
  SunMMIOValue cond = EnsureType(EvalExpr(op->condition), "i1", DataType::Bool());
  SunMMIOValue msg = EvalExpr(op->message);
  std::string text = msg.value.empty() ? "\"assertion failed\"" : msg.value;
  EmitLine("cf.assert " + cond.value + ", " + text);
  VisitStmt(op->body);
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::EvaluateNode* op) {
  (void)EvalExpr(op->value);
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::BlockNode* op) {
  EnterScope();
  for (const IterVar& iv : op->iter_vars) {
    BindVar(iv->var, EvalExpr(iv->var));
  }
  for (const Buffer& alloc : op->alloc_buffers) {
    RegisterBuffer(alloc, false, NewValueName());
    const BufferBinding& binding = LookupBuffer(alloc);
    EmitLine(binding.handle + " = memref.alloc() {sunmmio.scope = \"" +
             MapStorageScope(alloc.scope()) + "\"} : " + binding.memref_type);
  }
  for (const BufferRegion& r : op->reads) {
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
  EnterScope();
  for (size_t i = 0; i < op->iter_values.size() && i < op->block->iter_vars.size(); ++i) {
    BindVar(op->block->iter_vars[i]->var, EvalExpr(op->iter_values[i]));
  }
  (void)EnsureType(EvalExpr(op->predicate), "i1", DataType::Bool());
  VisitStmt(op->block);
  ExitScope();
}

void CodeGenTileLangSunMMIO::VisitStmtDefault_(const Object* op) {
  UnsupportedStmt(op, "No direct MLIR lowering handler implemented.");
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::VarNode* op) {
  auto it = var_table_.find(op);
  if (it != var_table_.end()) {
    return it->second;
  }
  SunMMIOValue info{op->dtype, "%" + op->name_hint, MapType(op->dtype)};
  var_table_[op] = info;
  return info;
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::SizeVarNode* op) {
  auto it = var_table_.find(op);
  if (it != var_table_.end()) {
    return it->second;
  }
  SunMMIOValue info{op->dtype, "%" + op->name_hint, MapType(op->dtype)};
  var_table_[op] = info;
  return info;
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::IntImmNode* op) {
  std::string name = NewValueName();
  std::string ty = MapType(op->dtype);
  EmitLine(name + " = arith.constant " + std::to_string(op->value) + " : " + ty);
  return SunMMIOValue{op->dtype, name, ty};
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::FloatImmNode* op) {
  std::string name = NewValueName();
  std::string ty = MapType(op->dtype);
  std::ostringstream os;
  os << op->value;
  EmitLine(name + " = arith.constant " + os.str() + " : " + ty);
  return SunMMIOValue{op->dtype, name, ty};
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::StringImmNode* op) {
  return SunMMIOValue{op->dtype, "\"" + static_cast<std::string>(op->value) + "\"",
                      "!sunmmio.string"};
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::CastNode* op) {
  return EmitCast(EvalExpr(op->value), op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::CallNode* op) {
  return EmitCall(op);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::AddNode* op) {
  return EmitBinary("add", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::SubNode* op) {
  return EmitBinary("sub", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::MulNode* op) {
  return EmitBinary("mul", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::DivNode* op) {
  return EmitBinary("div", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::ModNode* op) {
  return EmitBinary("mod", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::FloorDivNode* op) {
  return EmitBinary("floordiv", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::FloorModNode* op) {
  return EmitBinary("floormod", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::MinNode* op) {
  return EmitBinary("min", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::MaxNode* op) {
  return EmitBinary("max", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::EQNode* op) {
  return EmitCmp("eq", op->a, op->b);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::NENode* op) {
  return EmitCmp("ne", op->a, op->b);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::LTNode* op) {
  return EmitCmp("lt", op->a, op->b);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::LENode* op) {
  return EmitCmp("le", op->a, op->b);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::GTNode* op) {
  return EmitCmp("gt", op->a, op->b);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::GENode* op) {
  return EmitCmp("ge", op->a, op->b);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::AndNode* op) {
  return EmitBinary("and", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::OrNode* op) {
  return EmitBinary("or", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::NotNode* op) {
  SunMMIOValue v = EnsureType(EvalExpr(op->a), "i1", DataType::Bool());
  std::string one = NewValueName();
  EmitLine(one + " = arith.constant 1 : i1");
  std::string out = NewValueName();
  EmitLine(out + " = arith.xori " + v.value + ", " + one + " : i1");
  return SunMMIOValue{DataType::Bool(), out, "i1"};
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::SelectNode* op) {
  SunMMIOValue cond = EnsureType(EvalExpr(op->condition), "i1", DataType::Bool());
  SunMMIOValue tv = EvalExpr(op->true_value);
  SunMMIOValue fv = EvalExpr(op->false_value);
  fv = EnsureType(fv, tv.mlir_type, tv.dtype);
  std::string out = NewValueName();
  EmitLine(out + " = arith.select " + cond.value + ", " + tv.value + ", " + fv.value + " : " +
           tv.mlir_type);
  return SunMMIOValue{op->dtype, out, tv.mlir_type};
}

SunMMIOValue CodeGenTileLangSunMMIO::EmitLoad(const tir::Buffer& buffer,
                                              const ffi::Array<PrimExpr>& indices) {
  const BufferBinding& binding = LookupBuffer(buffer);
  std::vector<SunMMIOValue> idx_vals;
  for (const PrimExpr& idx : indices) {
    idx_vals.push_back(EnsureIndex(EvalExpr(idx)));
  }
  std::ostringstream idx;
  for (size_t i = 0; i < idx_vals.size(); ++i) {
    if (i != 0) {
      idx << ", ";
    }
    idx << idx_vals[i].value;
  }
  std::string out = NewValueName();
  EmitLine(out + " = memref.load " + binding.handle + "[" + idx.str() + "] : " +
           binding.memref_type);
  return SunMMIOValue{buffer->dtype, out, MapType(buffer->dtype)};
}

void CodeGenTileLangSunMMIO::EmitStore(const tir::Buffer& buffer,
                                       const ffi::Array<PrimExpr>& indices,
                                       const SunMMIOValue& value) {
  const BufferBinding& binding = LookupBuffer(buffer);
  std::vector<SunMMIOValue> idx_vals;
  for (const PrimExpr& idx : indices) {
    idx_vals.push_back(EnsureIndex(EvalExpr(idx)));
  }
  std::ostringstream idx;
  for (size_t i = 0; i < idx_vals.size(); ++i) {
    if (i != 0) {
      idx << ", ";
    }
    idx << idx_vals[i].value;
  }
  SunMMIOValue casted = EnsureType(value, MapType(buffer->dtype), buffer->dtype);
  EmitLine("memref.store " + casted.value + ", " + binding.handle + "[" + idx.str() + "] : " +
           binding.memref_type);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::BufferLoadNode* op) {
  if (!buffer_registry_.count(op->buffer.get())) {
    RegisterBuffer(op->buffer, false, NewValueName());
    const BufferBinding& b = LookupBuffer(op->buffer);
    EmitLine(b.handle + " = memref.alloc() {sunmmio.scope = \"" +
             MapStorageScope(op->buffer.scope()) + "\"} : " + b.memref_type);
  }
  return EmitLoad(op->buffer, op->indices);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::ProducerLoadNode* op) {
  UnsupportedExpr(op, "ProducerLoadNode is not supported.");
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::RampNode* op) {
  UnsupportedExpr(op, "RampNode lowering is not implemented yet.");
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::BroadcastNode* op) {
  UnsupportedExpr(op, "BroadcastNode lowering is not implemented yet.");
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::ShuffleNode* op) {
  UnsupportedExpr(op, "ShuffleNode lowering is not implemented yet.");
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::LetNode* op) {
  SunMMIOValue value = EvalExpr(op->value);
  EnterScope();
  BindVar(op->var, value);
  SunMMIOValue body = EvalExpr(op->body);
  ExitScope();
  return body;
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExprDefault_(const Object* op) {
  UnsupportedExpr(op, "Expr node is not supported in SunMMIO direct lowering.");
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

SunMMIOValue CodeGenTileLangSunMMIO::EmitBinary(const char* op_name,
                                                const tvm::PrimExpr& lhs,
                                                const tvm::PrimExpr& rhs,
                                                tvm::DataType dtype) {
  SunMMIOValue a = EvalExpr(lhs);
  SunMMIOValue b = EvalExpr(rhs);
  std::string ty = MapType(dtype);
  a = EnsureType(a, ty, dtype);
  b = EnsureType(b, ty, dtype);
  std::string out = NewValueName();
  std::string opcode;
  const std::string op(op_name);
  if (dtype.is_float() || dtype.is_bfloat16()) {
    if (op == "add")
      opcode = "arith.addf";
    else if (op == "sub")
      opcode = "arith.subf";
    else if (op == "mul")
      opcode = "arith.mulf";
    else if (op == "div" || op == "floordiv")
      opcode = "arith.divf";
    else if (op == "mod" || op == "floormod")
      opcode = "arith.remf";
    else if (op == "min")
      opcode = "arith.minf";
    else if (op == "max")
      opcode = "arith.maxf";
  } else {
    if (op == "add")
      opcode = "arith.addi";
    else if (op == "sub")
      opcode = "arith.subi";
    else if (op == "mul")
      opcode = "arith.muli";
    else if (op == "div" || op == "floordiv")
      opcode = "arith.divsi";
    else if (op == "mod" || op == "floormod")
      opcode = "arith.remsi";
    else if (op == "min")
      opcode = "arith.minsi";
    else if (op == "max")
      opcode = "arith.maxsi";
    else if (op == "and")
      opcode = "arith.andi";
    else if (op == "or")
      opcode = "arith.ori";
  }
  if (opcode.empty()) {
    UnsupportedExpr(lhs.get(), "Unsupported binary op in EmitBinary: " + op);
  }
  EmitLine(out + " = " + opcode + " " + a.value + ", " + b.value + " : " + ty);
  return SunMMIOValue{dtype, out, ty};
}

SunMMIOValue CodeGenTileLangSunMMIO::EmitCmp(const char* pred,
                                             const tvm::PrimExpr& lhs,
                                             const tvm::PrimExpr& rhs) {
  SunMMIOValue a = EvalExpr(lhs);
  SunMMIOValue b = EvalExpr(rhs);
  std::string ty = a.mlir_type;
  b = EnsureType(b, ty, a.dtype);
  std::string out = NewValueName();
  if (a.dtype.is_float() || a.dtype.is_bfloat16()) {
    std::string p = std::string("o") + pred;
    EmitLine(out + " = arith.cmpf " + p + ", " + a.value + ", " + b.value + " : " + ty);
  } else {
    EmitLine(out + " = arith.cmpi " + std::string(pred) + ", " + a.value + ", " + b.value +
             " : " + ty);
  }
  return SunMMIOValue{DataType::Bool(), out, "i1"};
}

SunMMIOValue CodeGenTileLangSunMMIO::EmitCast(const SunMMIOValue& v,
                                              tvm::DataType target_dtype) {
  std::string dst = MapType(target_dtype);
  if (v.mlir_type == dst) {
    return v;
  }
  std::string out = NewValueName();
  if (dst == "index" || v.mlir_type == "index") {
    EmitLine(out + " = arith.index_cast " + v.value + " : " + v.mlir_type + " to " + dst);
    return SunMMIOValue{target_dtype, out, dst};
  }
  std::string op = "builtin.unrealized_conversion_cast";
  EmitLine(out + " = " + op + " " + v.value + " : " + v.mlir_type + " to " + dst);
  return SunMMIOValue{target_dtype, out, dst};
}

CodeGenTileLangSunMMIO::CallBucket
CodeGenTileLangSunMMIO::ClassifyCall(const tir::CallNode* op) const {
  if (op->op.as<GlobalVarNode>()) {
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
      name.find("alloc") != std::string::npos || name.find("reinterpret") != std::string::npos) {
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

const char* CodeGenTileLangSunMMIO::CallBucketName(CallBucket bucket) const {
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

SunMMIOValue CodeGenTileLangSunMMIO::EmitCall(const tir::CallNode* op) {
  CallBucket bucket = ClassifyCall(op);
  if (bucket == CallBucket::kUnsupported) {
    UnsupportedExpr(op, "Unsupported call target.");
  }
  std::string callee = "unknown";
  if (const auto* op_node = op->op.as<OpNode>()) {
    callee = op_node->name;
  } else if (const auto* gv = op->op.as<GlobalVarNode>()) {
    callee = gv->name_hint;
  }
  std::ostringstream operands;
  std::ostringstream arg_types;
  std::ostringstream str_attrs;
  bool first_operand = true;
  bool first_type = true;
  bool first_str = true;
  for (const PrimExpr& arg : op->args) {
    if (const auto* s = arg.as<StringImmNode>()) {
      if (!first_str) {
        str_attrs << ", ";
      }
      str_attrs << "\"" << static_cast<std::string>(s->value) << "\"";
      first_str = false;
      continue;
    }
    SunMMIOValue v = EvalExpr(arg);
    if (!first_operand) {
      operands << ", ";
    }
    if (!first_type) {
      arg_types << ", ";
    }
    operands << v.value;
    arg_types << v.mlir_type;
    first_operand = false;
    first_type = false;
  }
  std::string attr = " {category = \"" + std::string(CallBucketName(bucket)) + "\"";
  if (!first_str) {
    attr += ", string_args = [" + str_attrs.str() + "]";
  }
  attr += "}";
  std::string call_head = "sunmmio.call @\"" + callee + "\"(" + operands.str() + ")" + attr +
                          " : (" + arg_types.str() + ")";
  if (op->dtype.is_void()) {
    EmitLine(call_head + " -> ()");
    return SunMMIOValue{op->dtype, "", "none"};
  }
  std::string out = NewValueName();
  std::string ret_ty = MapType(op->dtype);
  EmitLine(out + " = " + call_head + " -> " + ret_ty);
  return SunMMIOValue{op->dtype, out, ret_ty};
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
