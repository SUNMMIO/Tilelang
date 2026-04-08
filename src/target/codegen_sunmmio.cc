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

class TextSunMMIOBuilder final : public SunMMIOBuilder {
public:
  void Init() final { Clear(); }

  void Clear() final {
    os_.str("");
    os_.clear();
    indent_ = 0;
    module_open_ = false;
  }

  std::string Finish() final { return os_.str(); }

  void BeginModule() final {
    EmitLine("module {");
    indent_++;
    module_open_ = true;
  }

  void EndModule() final {
    if (module_open_) {
      indent_ = std::max(0, indent_ - 1);
      EmitLine("}");
      module_open_ = false;
    }
  }

  void BeginFunction(const std::string& name,
                     const std::vector<std::string>& arg_defs) final {
    std::ostringstream sig;
    for (size_t i = 0; i < arg_defs.size(); ++i) {
      if (i != 0)
        sig << ", ";
      sig << arg_defs[i];
    }
    EmitLine("func.func @" + name + "(" + sig.str() + ") {");
    indent_++;
  }

  void EndFunction() final {
    indent_ = std::max(0, indent_ - 1);
    EmitLine("}");
  }

  void EmitReturn() final { EmitLine("return"); }

  SunMMIOValue ConstantInt(const std::string& result_name, int64_t v,
                           const std::string& ty, DataType dtype) final {
    EmitLine(result_name + " = arith.constant " + std::to_string(v) + " : " + ty);
    return SunMMIOValue{dtype, result_name, ty};
  }

  SunMMIOValue ConstantFloat(const std::string& result_name,
                             const std::string& literal,
                             const std::string& ty,
                             DataType dtype) final {
    EmitLine(result_name + " = arith.constant " + literal + " : " + ty);
    return SunMMIOValue{dtype, result_name, ty};
  }

  SunMMIOValue Cast(const std::string& result_name, const SunMMIOValue& v,
                    const std::string& dst_ty, DataType dst_dtype) final {
    if (dst_ty == "index" || v.mlir_type == "index") {
      EmitLine(result_name + " = arith.index_cast " + v.value + " : " +
               v.mlir_type + " to " + dst_ty);
    } else {
      EmitLine(result_name + " = builtin.unrealized_conversion_cast " + v.value +
               " : " + v.mlir_type + " to " + dst_ty);
    }
    return SunMMIOValue{dst_dtype, result_name, dst_ty};
  }

  SunMMIOValue Binary(const std::string& result_name, const std::string& opcode,
                      const SunMMIOValue& a, const SunMMIOValue& b,
                      const std::string& ty, DataType dtype) final {
    EmitLine(result_name + " = " + opcode + " " + a.value + ", " + b.value +
             " : " + ty);
    return SunMMIOValue{dtype, result_name, ty};
  }

  SunMMIOValue Compare(const std::string& result_name, const std::string& opcode,
                       const SunMMIOValue& a, const SunMMIOValue& b,
                       const std::string& ty) final {
    EmitLine(result_name + " = " + opcode + " " + a.value + ", " + b.value +
             " : " + ty);
    return SunMMIOValue{DataType::Bool(), result_name, "i1"};
  }

  SunMMIOValue Select(const std::string& result_name, const SunMMIOValue& cond,
                      const SunMMIOValue& tv, const SunMMIOValue& fv,
                      const std::string& ty, DataType dtype) final {
    EmitLine(result_name + " = arith.select " + cond.value + ", " + tv.value +
             ", " + fv.value + " : " + ty);
    return SunMMIOValue{dtype, result_name, ty};
  }

  SunMMIOValue Alloc(const std::string& result_name, const std::string& memref_type,
                     const std::vector<SunMMIOValue>& dyn_extents,
                     const std::string& scope_attr, DataType dtype) final {
    std::ostringstream dyn_sig;
    for (size_t i = 0; i < dyn_extents.size(); ++i) {
      if (i != 0)
        dyn_sig << ", ";
      dyn_sig << dyn_extents[i].value;
    }
    if (dyn_extents.empty()) {
      EmitLine(result_name + " = memref.alloc()" + scope_attr + " : " + memref_type);
    } else {
      EmitLine(result_name + " = memref.alloc(" + dyn_sig.str() + ")" + scope_attr +
               " : " + memref_type);
    }
    return SunMMIOValue{dtype, result_name, memref_type};
  }

  SunMMIOValue Load(const std::string& result_name, const std::string& buffer_handle,
                    const std::vector<SunMMIOValue>& indices,
                    const std::string& memref_type, DataType dtype,
                    const std::string& result_ty) final {
    std::ostringstream idx;
    for (size_t i = 0; i < indices.size(); ++i) {
      if (i != 0)
        idx << ", ";
      idx << indices[i].value;
    }
    EmitLine(result_name + " = memref.load " + buffer_handle + "[" + idx.str() +
             "] : " + memref_type);
    return SunMMIOValue{dtype, result_name, result_ty};
  }

  void Store(const SunMMIOValue& value, const std::string& buffer_handle,
             const std::vector<SunMMIOValue>& indices,
             const std::string& memref_type) final {
    std::ostringstream idx;
    for (size_t i = 0; i < indices.size(); ++i) {
      if (i != 0)
        idx << ", ";
      idx << indices[i].value;
    }
    EmitLine("memref.store " + value.value + ", " + buffer_handle + "[" +
             idx.str() + "] : " + memref_type);
  }

  SunMMIOValue Call(const std::string& result_name, const std::string& callee,
                    const std::vector<SunMMIOValue>& operands,
                    const std::vector<std::string>& string_args,
                    const std::string& category, DataType ret_dtype,
                    const std::string& ret_ty) final {
    std::ostringstream op_sig;
    std::ostringstream ty_sig;
    for (size_t i = 0; i < operands.size(); ++i) {
      if (i != 0) {
        op_sig << ", ";
        ty_sig << ", ";
      }
      op_sig << operands[i].value;
      ty_sig << operands[i].mlir_type;
    }
    std::ostringstream attr;
    attr << " {category = \"" << category << "\"";
    if (!string_args.empty()) {
      attr << ", string_args = [";
      for (size_t i = 0; i < string_args.size(); ++i) {
        if (i != 0)
          attr << ", ";
        attr << "\"" << string_args[i] << "\"";
      }
      attr << "]";
    }
    attr << "}";
    std::string head = "sunmmio.call @\"" + callee + "\"(" + op_sig.str() + ")" +
                       attr.str() + " : (" + ty_sig.str() + ")";
    if (ret_dtype.is_void()) {
      EmitLine(head + " -> ()");
      return SunMMIOValue{ret_dtype, "", "none"};
    }
    EmitLine(result_name + " = " + head + " -> " + ret_ty);
    return SunMMIOValue{ret_dtype, result_name, ret_ty};
  }

  void BeginFor(const std::string& iv, const SunMMIOValue& lb,
                const SunMMIOValue& ub, const SunMMIOValue& step) final {
    EmitLine("scf.for " + iv + " = " + lb.value + " to " + ub.value + " step " +
             step.value + " {");
    indent_++;
  }

  void EndFor() final {
    indent_ = std::max(0, indent_ - 1);
    EmitLine("}");
  }

  void BeginIf(const SunMMIOValue& cond) final {
    EmitLine("scf.if " + cond.value + " {");
    indent_++;
  }

  void BeginElse() final {
    indent_ = std::max(0, indent_ - 1);
    EmitLine("} else {");
    indent_++;
  }

  void EndIf() final {
    indent_ = std::max(0, indent_ - 1);
    EmitLine("}");
  }

  void EmitAssert(const SunMMIOValue& cond, const std::string& msg_text) final {
    EmitLine("cf.assert " + cond.value + ", " + msg_text);
  }

  void EmitRawLine(const std::string& line) final { EmitLine(line); }

private:
  void EmitLine(const std::string& line) {
    for (int i = 0; i < indent_; ++i) {
      os_ << "  ";
    }
    os_ << line << "\n";
  }

  std::ostringstream os_;
  int indent_{0};
  bool module_open_{false};
};

CodeGenTileLangSunMMIO::CodeGenTileLangSunMMIO() = default;

void CodeGenTileLangSunMMIO::Init() {
  Clear();
  builder_ = std::make_unique<TextSunMMIOBuilder>();
  builder_->Init();
  builder_->BeginModule();
  initialized_ = true;
}

void CodeGenTileLangSunMMIO::Clear() {
  if (builder_) {
    builder_->Clear();
  }
  builder_.reset();
  ssa_counter_ = 0;
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
  ICHECK(builder_) << "CodeGenTileLangSunMMIO builder is not initialized";
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

  builder_->BeginFunction(gvar->name_hint, arg_defs);
  VisitStmt(f->body);
  builder_->EmitReturn();
  builder_->EndFunction();
  ExitScope();
}

std::string CodeGenTileLangSunMMIO::Finish() {
  if (!initialized_) {
    Init();
  }
  ICHECK(builder_) << "CodeGenTileLangSunMMIO builder is not initialized";
  builder_->EndModule();
  std::string out = builder_->Finish();
  initialized_ = false;
  return out;
}

std::string CodeGenTileLangSunMMIO::NewValueName() {
  return "%v" + std::to_string(ssa_counter_++);
}

std::string CodeGenTileLangSunMMIO::MapType(tvm::DataType dtype) const {
  if (dtype.lanes() > 1) {
    DataType scalar = dtype.with_lanes(1);
    return "vector<" + std::to_string(dtype.lanes()) + "x" + MapType(scalar) + ">";
  }
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
  return builder_->ConstantInt(NewValueName(), v, "index", DataType::Int(32));
}

SunMMIOValue CodeGenTileLangSunMMIO::EnsureIndex(const SunMMIOValue& v) {
  if (v.mlir_type == "index") {
    return v;
  }
  return builder_->Cast(NewValueName(), v, "index", DataType::Int(32));
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
  std::string scope_attr = " {sunmmio.scope = \"" + MapStorageScope(scope_hint) + "\"}";
  SunMMIOValue alloc =
      builder_->Alloc(NewValueName(), memref.str(), dyn_extents, scope_attr, dtype);
  BindVar(buffer_var, alloc);
}

void CodeGenTileLangSunMMIO::EmitFor(const tir::ForNode* op) {
  SunMMIOValue min = EnsureIndex(EvalExpr(op->min));
  SunMMIOValue extent = EnsureIndex(EvalExpr(op->extent));
  SunMMIOValue step = EmitConstIndex(1);
  SunMMIOValue upper =
      builder_->Binary(NewValueName(), "arith.addi", min, extent, "index", DataType::Int(32));
  std::string iv = "%" + op->loop_var->name_hint;
  builder_->BeginFor(iv, min, upper, step);
  EnterScope();
  BindVar(op->loop_var, SunMMIOValue{op->loop_var.dtype(), iv, "index"});
  VisitStmt(op->body);
  ExitScope();
  builder_->EndFor();
}

void CodeGenTileLangSunMMIO::EmitIf(const tir::IfThenElseNode* op) {
  SunMMIOValue cond = EnsureType(EvalExpr(op->condition), "i1", DataType::Bool());
  builder_->BeginIf(cond);
  VisitStmt(op->then_case);
  if (op->else_case.defined()) {
    builder_->BeginElse();
    VisitStmt(op->else_case.value());
  }
  builder_->EndIf();
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
  builder_->Alloc(binding.handle, binding.memref_type, {},
                  " {sunmmio.scope = \"" + MapStorageScope(op->buffer.scope()) + "\"}",
                  op->buffer->dtype);
  VisitStmt(op->body);
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::BufferStoreNode* op) {
  if (!buffer_registry_.count(op->buffer.get())) {
    RegisterBuffer(op->buffer, false, NewValueName());
    const BufferBinding& binding = LookupBuffer(op->buffer);
    builder_->Alloc(binding.handle, binding.memref_type, {},
                    " {sunmmio.scope = \"" + MapStorageScope(op->buffer.scope()) + "\"}",
                    op->buffer->dtype);
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
  builder_->Alloc(binding.handle, binding.memref_type, dyn_bounds,
                  " {sunmmio.scope = \"" + MapStorageScope(op->buffer.scope()) + "\"}",
                  op->buffer->dtype);
  VisitStmt(op->body);
  ExitScope();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::AssertStmtNode* op) {
  SunMMIOValue cond = EnsureType(EvalExpr(op->condition), "i1", DataType::Bool());
  SunMMIOValue msg = EvalExpr(op->message);
  std::string text = msg.value.empty() ? "\"assertion failed\"" : msg.value;
  builder_->EmitAssert(cond, text);
  VisitStmt(op->body);
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::EvaluateNode* op) {
  (void)EvalExpr(op->value);
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::BlockNode* op) {
  auto traverse_range = [this](const Range& range) {
    (void)EvalExpr(range->min);
    (void)EvalExpr(range->extent);
  };
  auto traverse_buffer_region = [&traverse_range](const BufferRegion& region) {
    for (const Range& r : region->region) {
      traverse_range(r);
    }
  };
  auto traverse_annotation_obj =
      [this](const ffi::Any& value,
             const auto& self_ref) -> void {
    if (auto maybe_expr = value.as<PrimExpr>()) {
      (void)EvalExpr(maybe_expr.value());
      return;
    }
    if (auto maybe_arr_expr = value.as<ffi::Array<PrimExpr>>()) {
      for (const PrimExpr& item : maybe_arr_expr.value()) {
        (void)EvalExpr(item);
      }
      return;
    }
    if (auto maybe_arr_any = value.as<ffi::Array<ffi::Any>>()) {
      for (const ffi::Any& item : maybe_arr_any.value()) {
        self_ref(item, self_ref);
      }
      return;
    }
    if (auto maybe_map_expr = value.as<ffi::Map<ffi::String, PrimExpr>>()) {
      for (const auto& kv : maybe_map_expr.value()) {
        (void)EvalExpr(kv.second);
      }
      return;
    }
    if (auto maybe_map_any = value.as<ffi::Map<ffi::String, ffi::Any>>()) {
      for (const auto& kv : maybe_map_any.value()) {
        self_ref(kv.second, self_ref);
      }
      return;
    }
    if (auto maybe_map_any_any = value.as<ffi::Map<ffi::Any, ffi::Any>>()) {
      for (const auto& kv : maybe_map_any_any.value()) {
        self_ref(kv.first, self_ref);
        self_ref(kv.second, self_ref);
      }
      return;
    }
  };

  EnterScope();
  for (const IterVar& iv : op->iter_vars) {
    if (!var_table_.count(iv->var.get())) {
      BindVar(iv->var, EvalExpr(iv->var));
    }
  }
  for (const Buffer& alloc : op->alloc_buffers) {
    RegisterBuffer(alloc, false, NewValueName());
    const BufferBinding& binding = LookupBuffer(alloc);
    builder_->Alloc(binding.handle, binding.memref_type, {},
                    " {sunmmio.scope = \"" + MapStorageScope(alloc.scope()) + "\"}",
                    alloc->dtype);
  }
  for (const MatchBufferRegion& match : op->match_buffers) {
    if (match->source.defined()) {
      RegisterBuffer(match->source->buffer, false);
      traverse_buffer_region(match->source);
    }
    if (!buffer_registry_.count(match->buffer.get())) {
      if (match->source.defined() &&
          buffer_registry_.count(match->source->buffer.get())) {
        const BufferBinding& src = LookupBuffer(match->source->buffer);
        RegisterBuffer(match->buffer, false, src.handle);
      } else {
        RegisterBuffer(match->buffer, false, NewValueName());
      }
    }
  }
  for (const BufferRegion& r : op->reads) {
    RegisterBuffer(r->buffer, false);
    traverse_buffer_region(r);
  }
  for (const BufferRegion& r : op->writes) {
    RegisterBuffer(r->buffer, false);
    traverse_buffer_region(r);
  }
  for (const auto& kv : op->annotations) {
    traverse_annotation_obj(kv.second, traverse_annotation_obj);
  }
  if (op->init.defined()) {
    VisitStmt(op->init.value());
  }
  VisitStmt(op->body);
  ExitScope();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::BlockRealizeNode* op) {
  auto is_trivially_true = [](const PrimExpr& expr) {
    if (is_one(expr)) {
      return true;
    }
    if (const auto* imm = expr.as<IntImmNode>()) {
      return imm->value != 0;
    }
    return false;
  };

  EnterScope();
  for (size_t i = 0; i < op->iter_values.size() && i < op->block->iter_vars.size(); ++i) {
    BindVar(op->block->iter_vars[i]->var, EvalExpr(op->iter_values[i]));
  }
  if (is_trivially_true(op->predicate)) {
    VisitStmt(op->block);
  } else {
    SunMMIOValue pred = EnsureType(EvalExpr(op->predicate), "i1", DataType::Bool());
    builder_->BeginIf(pred);
    VisitStmt(op->block);
    builder_->EndIf();
  }
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
  std::string ty = MapType(op->dtype);
  return builder_->ConstantInt(NewValueName(), op->value, ty, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::FloatImmNode* op) {
  std::ostringstream os;
  os << op->value;
  std::string ty = MapType(op->dtype);
  return builder_->ConstantFloat(NewValueName(), os.str(), ty, op->dtype);
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
  SunMMIOValue one = builder_->ConstantInt(NewValueName(), 1, "i1", DataType::Bool());
  return builder_->Binary(NewValueName(), "arith.xori", v, one, "i1",
                          DataType::Bool());
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::SelectNode* op) {
  SunMMIOValue cond = EnsureType(EvalExpr(op->condition), "i1", DataType::Bool());
  SunMMIOValue tv = EvalExpr(op->true_value);
  SunMMIOValue fv = EvalExpr(op->false_value);
  fv = EnsureType(fv, tv.mlir_type, tv.dtype);
  return builder_->Select(NewValueName(), cond, tv, fv, tv.mlir_type, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::EmitLoad(const tir::Buffer& buffer,
                                              const ffi::Array<PrimExpr>& indices) {
  const BufferBinding& binding = LookupBuffer(buffer);
  std::vector<SunMMIOValue> idx_vals;
  for (const PrimExpr& idx : indices) {
    idx_vals.push_back(EnsureIndex(EvalExpr(idx)));
  }
  return builder_->Load(NewValueName(), binding.handle, idx_vals, binding.memref_type,
                        buffer->dtype, MapType(buffer->dtype));
}

void CodeGenTileLangSunMMIO::EmitStore(const tir::Buffer& buffer,
                                       const ffi::Array<PrimExpr>& indices,
                                       const SunMMIOValue& value) {
  const BufferBinding& binding = LookupBuffer(buffer);
  std::vector<SunMMIOValue> idx_vals;
  for (const PrimExpr& idx : indices) {
    idx_vals.push_back(EnsureIndex(EvalExpr(idx)));
  }
  SunMMIOValue casted = EnsureType(value, MapType(buffer->dtype), buffer->dtype);
  builder_->Store(casted, binding.handle, idx_vals, binding.memref_type);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::BufferLoadNode* op) {
  if (!buffer_registry_.count(op->buffer.get())) {
    RegisterBuffer(op->buffer, false, NewValueName());
    const BufferBinding& b = LookupBuffer(op->buffer);
    builder_->Alloc(b.handle, b.memref_type, {},
                    " {sunmmio.scope = \"" + MapStorageScope(op->buffer.scope()) + "\"}",
                    op->buffer->dtype);
  }
  return EmitLoad(op->buffer, op->indices);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::ProducerLoadNode* op) {
  UnsupportedExpr(op, "ProducerLoadNode is not supported.");
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::RampNode* op) {
  DataType elem_dtype = op->dtype.with_lanes(1);
  std::string elem_ty = MapType(elem_dtype);
  std::string vec_ty = MapType(op->dtype);

  SunMMIOValue base = EvalExpr(op->base);
  SunMMIOValue stride = EvalExpr(op->stride);
  base = EnsureType(base, elem_ty, elem_dtype);
  stride = EnsureType(stride, elem_ty, elem_dtype);

  std::string out = NewValueName();
  builder_->EmitRawLine(out + " = sunmmio.ramp " + base.value + ", " + stride.value +
                        " {lanes = " + std::to_string(op->dtype.lanes()) + "} : " +
                        elem_ty + " -> " + vec_ty);
  return SunMMIOValue{op->dtype, out, vec_ty};
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::BroadcastNode* op) {
  SunMMIOValue scalar = EvalExpr(op->value);
  DataType scalar_dtype = op->dtype.with_lanes(1);
  std::string scalar_ty = MapType(scalar_dtype);
  std::string vec_ty = MapType(op->dtype);
  scalar = EnsureType(scalar, scalar_ty, scalar_dtype);

  std::string out = NewValueName();
  builder_->EmitRawLine(out + " = vector.broadcast " + scalar.value + " : " +
                        scalar_ty + " to " + vec_ty);
  return SunMMIOValue{op->dtype, out, vec_ty};
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
  return builder_->Binary(out, opcode, a, b, ty, dtype);
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
    return builder_->Compare(out, "arith.cmpf " + p + ",", a, b, ty);
  } else {
    return builder_->Compare(out, "arith.cmpi " + std::string(pred) + ",", a, b, ty);
  }
}

SunMMIOValue CodeGenTileLangSunMMIO::EmitCast(const SunMMIOValue& v,
                                              tvm::DataType target_dtype) {
  std::string dst = MapType(target_dtype);
  if (v.mlir_type == dst) {
    return v;
  }
  return builder_->Cast(NewValueName(), v, dst, target_dtype);
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
  std::vector<SunMMIOValue> operands;
  std::vector<std::string> string_args;
  for (const PrimExpr& arg : op->args) {
    if (const auto* s = arg.as<StringImmNode>()) {
      string_args.push_back(static_cast<std::string>(s->value));
      continue;
    }
    operands.push_back(EvalExpr(arg));
  }
  std::string ret_ty = MapType(op->dtype);
  std::string result_name = op->dtype.is_void() ? "" : NewValueName();
  return builder_->Call(result_name, callee, operands, string_args,
                        CallBucketName(bucket), op->dtype, ret_ty);
}

[[noreturn]] void CodeGenTileLangSunMMIO::UnsupportedStmt(
    const Object* op, const std::string& detail) const {
  // Intentional unsupported contract for this backend:
  // - tir::WhileNode
  // - legacy tir::LoadNode
  // - tir::AnyNode
  // - tir::ShuffleNode
  // These forms must be eliminated before SunMMIO codegen; reaching here is
  // a pipeline bug and should fail loudly.
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
  // Intentional unsupported contract for this backend:
  // - tir::WhileNode
  // - legacy tir::LoadNode
  // - tir::AnyNode
  // - tir::ShuffleNode
  // These forms must be eliminated before SunMMIO codegen; reaching here is
  // a pipeline bug and should fail loudly.
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
