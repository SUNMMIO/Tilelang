#include "codegen_sunmmio.h"
#include "sunmmio_mlir_builder.h"

#include <tvm/ir/type.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <unordered_set>
#include <utility>

#include <tvm/runtime/logging.h>

namespace tvm {
namespace codegen {
using namespace tir;

namespace {

std::string GetAllocateStorageScope(const tir::Var &buffer_var) {
  if (const auto *ptr = buffer_var->type_annotation.as<PointerTypeNode>()) {
    const std::string &scope = ptr->storage_scope;
    if (scope == "shared.asram" || scope == "shared.wsram" ||
        scope == "shared.rsram") {
      return scope;
    } else {
      LOG(FATAL) << "get Allocate StorageScope error:  " << scope;
    }
  }
  LOG(FATAL) << "SunMMIO SUVM allocate expects PointerType buffer_var";
  TVM_FFI_UNREACHABLE();
}

} // namespace

CodeGenTileLangSunMMIO::CodeGenTileLangSunMMIO() = default;

void CodeGenTileLangSunMMIO::Init() {
  Clear();
  builder_ = std::make_unique<SuvmSunmmioBuilder>();
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
  expected_node_types_.clear();
  visited_node_types_.clear();
  expected_call_ops_.clear();
  visited_call_ops_.clear();
  initialized_ = false;
}

SunMMIOValue CodeGenTileLangSunMMIO::EvalExpr(const tvm::PrimExpr &expr) {
  if (expr.defined()) {
    MarkVisitedNodeType(expr->GetTypeKey());
    MarkVisitedCallOpFromExpr(expr);
  }
  return tir::ExprFunctor<SunMMIOValue(const tvm::PrimExpr &)>::VisitExpr(expr);
}

void CodeGenTileLangSunMMIO::VisitStmtTracked(const tir::Stmt &stmt) {
  if (stmt.defined()) {
    MarkVisitedNodeType(stmt->GetTypeKey());
  }
  tir::StmtVisitor::VisitStmt(stmt);
}

void CodeGenTileLangSunMMIO::MarkVisitedNodeType(const std::string &type_key) {
  visited_node_types_.insert(type_key);
}

void CodeGenTileLangSunMMIO::MarkVisitedCallOpFromExpr(
    const tvm::PrimExpr &expr) {
  const auto *call = expr.as<tir::CallNode>();
  if (!call) {
    return;
  }
  if (const auto *op_node = call->op.as<OpNode>()) {
    visited_call_ops_.insert(op_node->name);
  } else if (const auto *gv = call->op.as<GlobalVarNode>()) {
    visited_call_ops_.insert(std::string("global::") + gv->name_hint);
  } else {
    visited_call_ops_.insert("unknown_call_target");
  }
}

void CodeGenTileLangSunMMIO::CollectExpectedCoverage(const tir::PrimFunc &f) {
  tir::PostOrderVisit(f->body, [&](const ObjectRef &obj) {
    if (!obj.defined()) {
      return;
    }
    expected_node_types_.insert(obj->GetTypeKey());
    if (const auto *call = obj.as<tir::CallNode>()) {
      if (const auto *op_node = call->op.as<OpNode>()) {
        expected_call_ops_.insert(op_node->name);
      } else if (const auto *gv = call->op.as<GlobalVarNode>()) {
        expected_call_ops_.insert(std::string("global::") + gv->name_hint);
      } else {
        expected_call_ops_.insert("unknown_call_target");
      }
    }
  });
}

void CodeGenTileLangSunMMIO::WriteCoverageReport() const {
  const char *path = std::getenv("TL_SUNMMIO_CODEGEN_COVERAGE_PATH");
  if (path == nullptr || std::string(path).empty()) {
    return;
  }
  std::ofstream os(path, std::ios::out | std::ios::trunc);
  if (!os.is_open()) {
    LOG(WARNING) << "CodeGenTileLangSunMMIO: failed to open coverage path: "
                 << path;
    return;
  }
  auto write_list = [&os](const char *key,
                          const std::set<std::string> &values) {
    os << "  \"" << key << "\": [";
    bool first = true;
    for (const auto &item : values) {
      if (!first) {
        os << ", ";
      }
      first = false;
      os << "\"" << item << "\"";
    }
    os << "]";
  };
  auto diff = [](const std::set<std::string> &a,
                 const std::set<std::string> &b) {
    std::set<std::string> out;
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                        std::inserter(out, out.begin()));
    return out;
  };
  std::set<std::string> missing_nodes =
      diff(expected_node_types_, visited_node_types_);
  std::set<std::string> missing_calls =
      diff(expected_call_ops_, visited_call_ops_);

  os << "{\n";
  write_list("expected_node_types", expected_node_types_);
  os << ",\n";
  write_list("visited_node_types", visited_node_types_);
  os << ",\n";
  write_list("missing_node_types", missing_nodes);
  os << ",\n";
  write_list("expected_call_ops", expected_call_ops_);
  os << ",\n";
  write_list("visited_call_ops", visited_call_ops_);
  os << ",\n";
  write_list("missing_call_ops", missing_calls);
  os << "\n}\n";
}

void CodeGenTileLangSunMMIO::CheckCoverageOrFail() const {
  auto diff = [](const std::set<std::string> &a,
                 const std::set<std::string> &b) {
    std::set<std::string> out;
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                        std::inserter(out, out.begin()));
    return out;
  };
  std::set<std::string> missing_nodes =
      diff(expected_node_types_, visited_node_types_);
  std::set<std::string> missing_calls =
      diff(expected_call_ops_, visited_call_ops_);

  const char *strict_env = std::getenv("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT");
  bool strict = strict_env != nullptr && std::string(strict_env) == "1";

  if (!missing_nodes.empty() || !missing_calls.empty()) {
    LOG(WARNING) << "CodeGenTileLangSunMMIO coverage gaps: missing_nodes="
                 << missing_nodes.size()
                 << ", missing_call_ops=" << missing_calls.size();
    if (strict) {
      std::ostringstream err;
      err << "SunMMIO codegen traversal incomplete. Missing node types: ";
      for (const auto &s : missing_nodes) {
        err << s << "; ";
      }
      err << "Missing call ops: ";
      for (const auto &s : missing_calls) {
        err << s << "; ";
      }
      LOG(FATAL) << err.str();
    }
  }
}

void CodeGenTileLangSunMMIO::AddFunction(const GlobalVar &gvar,
                                         const tir::PrimFunc &f) {
  if (!initialized_) {
    Init();
  }
  ICHECK(builder_) << "CodeGenTileLangSunMMIO builder is not initialized";
  CollectExpectedCoverage(f);
  EnterScope();
  std::vector<BuilderArg> args;
  int arg_index = 0;
  for (const tir::Var &p : f->params) {
    std::string arg_name = "%arg" + std::to_string(arg_index++);
    if (f->buffer_map.count(p)) {
      const tir::Buffer &buffer = f->buffer_map.at(p);
      SunMMIOType buf_ty = MapBufferType(buffer);
      args.push_back({arg_name, buf_ty});
      BindVar(p, SunMMIOValue{p.dtype(), arg_name, buf_ty});
      RegisterBuffer(buffer, true, arg_name);
    } else {
      SunMMIOType arg_ty = MapType(p.dtype());
      args.push_back({arg_name, arg_ty});
      BindVar(p, SunMMIOValue{p.dtype(), arg_name, arg_ty});
    }
  }

  builder_->BeginFunction(gvar->name_hint, args);
  VisitStmtTracked(f->body);
  builder_->EmitReturn();
  builder_->EndFunction();
  ExitScope();
}

std::string CodeGenTileLangSunMMIO::Finish() {
  if (!initialized_) {
    Init();
  }
  ICHECK(builder_) << "CodeGenTileLangSunMMIO builder is not initialized";
  WriteCoverageReport();
  CheckCoverageOrFail();
  builder_->EndModule();
  std::string out = builder_->Finish();
  initialized_ = false;
  return out;
}

std::string CodeGenTileLangSunMMIO::NewValueName() {
  return "%v" + std::to_string(ssa_counter_++);
}

SunMMIOType CodeGenTileLangSunMMIO::MapType(tvm::DataType dtype) const {
  if (dtype.lanes() > 1) {
    return SunMMIOType{
        SunMMIOType::Kind::kVector, dtype.with_lanes(1), dtype.lanes(), {}};
  }
  if (dtype.is_handle()) {
    return SunMMIOType{SunMMIOType::Kind::kHandle, dtype, 1, {}};
  }
  if (dtype.is_void()) {
    return SunMMIOType{SunMMIOType::Kind::kUnknown, dtype, 1, {}};
  }
  return SunMMIOType{SunMMIOType::Kind::kScalar, dtype, 1, {}};
}

std::string
CodeGenTileLangSunMMIO::MapStorageScope(const std::string &scope) const {
  if (scope.empty()) {
    return "global";
  }
  std::string out = scope;
  std::replace(out.begin(), out.end(), '.', '_');
  return out;
}

SunMMIOType
CodeGenTileLangSunMMIO::MapBufferType(const tir::Buffer &buffer) const {
  std::vector<PrimExpr> shape;
  shape.reserve(buffer->shape.size());
  for (const PrimExpr &dim : buffer->shape) {
    shape.push_back(dim);
  }
  return SunMMIOType{SunMMIOType::Kind::kMemRef, buffer->dtype, 1,
                     std::move(shape)};
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::SeqStmtNode *op) {
  for (const Stmt &stmt : op->seq) {
    VisitStmtTracked(stmt);
  }
}

SunMMIOValue CodeGenTileLangSunMMIO::EmitConstIndex(int64_t v) {
  return builder_->ConstantInt(
      NewValueName(), v,
      SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
      DataType::Int(32));
}

SunMMIOValue CodeGenTileLangSunMMIO::EnsureIndex(const SunMMIOValue &v) {
  if (v.type.kind == SunMMIOType::Kind::kIndex) {
    return v;
  }
  return builder_->Cast(
      NewValueName(), v,
      SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
      DataType::Int(32));
}

SunMMIOValue CodeGenTileLangSunMMIO::EnsureType(const SunMMIOValue &v,
                                                const SunMMIOType &target_type,
                                                DataType dtype) {
  if (v.type.kind == target_type.kind && v.type.dtype == target_type.dtype &&
      v.type.lanes == target_type.lanes &&
      v.type.shape.size() == target_type.shape.size()) {
    return v;
  }
  return builder_->Cast(NewValueName(), v, target_type, dtype);
}

ArithmeticFlavor
CodeGenTileLangSunMMIO::GetArithmeticFlavor(DataType dtype) const {
  if (dtype.is_float() || dtype.is_bfloat16()) {
    return ArithmeticFlavor::kFloat;
  }
  if (dtype.is_uint()) {
    return ArithmeticFlavor::kUnsignedInt;
  }
  if (dtype.is_bool()) {
    return ArithmeticFlavor::kBool;
  }
  return ArithmeticFlavor::kSignedInt;
}

CompareDomain CodeGenTileLangSunMMIO::GetCompareDomain(DataType dtype) const {
  if (dtype.is_float() || dtype.is_bfloat16()) {
    return CompareDomain::kFloat;
  }
  if (dtype.is_uint()) {
    return CompareDomain::kUnsignedInt;
  }
  if (dtype.is_bool()) {
    return CompareDomain::kBool;
  }
  return CompareDomain::kSignedInt;
}

SunMMIOValue CodeGenTileLangSunMMIO::BindVar(const tir::Var &var,
                                             const SunMMIOValue &value) {
  var_table_[var.get()] = value;
  scoped_vars_.push_back(var.get());
  return value;
}

void CodeGenTileLangSunMMIO::RegisterBuffer(const tir::Buffer &buffer,
                                            bool is_external,
                                            const std::string &handle_hint) {
  if (!buffer.defined()) {
    return;
  }
  if (buffer_registry_.count(buffer.get())) {
    return;
  }
  BufferBinding binding;
  binding.buffer = buffer;
  binding.scope = buffer.scope();
  binding.buffer_type = MapBufferType(buffer);
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

const BufferBinding &
CodeGenTileLangSunMMIO::LookupBuffer(const tir::Buffer &buffer) const {
  auto it = buffer_registry_.find(buffer.get());
  ICHECK(it != buffer_registry_.end())
      << "CodeGenTileLangSunMMIO: unknown buffer " << buffer;
  return it->second;
}

void CodeGenTileLangSunMMIO::EmitAlloc(const tir::Var &buffer_var,
                                       DataType dtype,
                                       const ffi::Array<PrimExpr> &extents,
                                       const std::string &scope_hint) {
  std::vector<SunMMIOValue> dyn_extents;
  std::vector<PrimExpr> shape;
  shape.reserve(extents.size());
  for (size_t i = 0; i < extents.size(); ++i) {
    shape.push_back(extents[i]);
    if (!extents[i].as<IntImmNode>()) {
      dyn_extents.push_back(EnsureIndex(EvalExpr(extents[i])));
    }
  }
  SunMMIOType memtensor_type;
  memtensor_type.kind = SunMMIOType::Kind::kMemTensor;
  memtensor_type.dtype = dtype.with_lanes(1);
  memtensor_type.lanes = 1;
  memtensor_type.shape = std::move(shape);
  memtensor_type.memory_scope = scope_hint;
  memtensor_type.byte_offset = 0;

  SunMMIOValue alloc = builder_->Alloc(NewValueName(), memtensor_type,
                                       dyn_extents, scope_hint, dtype);
  BindVar(buffer_var, alloc);
}

namespace {
struct TokenSummary {
  std::vector<int64_t> live_out;
};

struct IterState {
  std::unordered_set<int64_t> avail_tokens;

  std::vector<int64_t> produced_order;
  std::unordered_set<int64_t> produced_seen;

  void MarkProduced(int64_t token_id) {
    if (token_id < 0) {
      return;
    }
    avail_tokens.insert(token_id);
    if (produced_seen.insert(token_id).second) {
      produced_order.push_back(token_id);
    }
  }
};

struct TokenAnalyzer {
  static int64_t ParseTokenIdFromArgs(const ffi::Array<PrimExpr> &args) {
    for (const PrimExpr &arg : args) {
      if (const auto *call = arg.as<CallNode>()) {
        if (const auto *op_node = call->op.as<OpNode>()) {
          if (op_node->name == "tl.sync_token_id" && call->args.size() == 1) {
            if (const auto *imm = call->args[0].as<IntImmNode>()) {
              return static_cast<int64_t>(imm->value);
            }
          }
        }
      }
    }
    for (const PrimExpr &arg : args) {
      if (const auto *imm = arg.as<IntImmNode>()) {
        return static_cast<int64_t>(imm->value);
      }
    }
    return -1;
  }

  static void MergeProducedOrder(IterState &dst,
                                 const std::vector<int64_t> &order) {
    for (int64_t t : order) {
      if (t < 0) {
        continue;
      }
      if (dst.produced_seen.insert(t).second) {
        dst.produced_order.push_back(t);
      }
    }
  }

  TokenSummary AnalyzeFor(const tir::ForNode *for_op) {
    IterState st;
    AnalyzeStmt(for_op->body, st);

    std::vector<int64_t> live_out_order;
    live_out_order.reserve(st.produced_order.size());
    for (int64_t t : st.produced_order) {
      if (t >= 0 && st.avail_tokens.count(t) != 0) {
        live_out_order.push_back(t);
      }
    }

    TokenSummary summary;
    summary.live_out = std::move(live_out_order);
    return summary;
  }

  TokenSummary AnalyzeIf(const tir::IfThenElseNode *if_op) {
    IterState then_st;
    AnalyzeStmt(if_op->then_case, then_st);
    IterState else_st;
    if (if_op->else_case.defined()) {
      AnalyzeStmt(if_op->else_case.value(), else_st);
    }

    std::vector<int64_t> live_out_order;
    std::unordered_set<int64_t> live_out_set;
    for (int64_t t : then_st.produced_order) {
      if (t >= 0 && then_st.avail_tokens.count(t) != 0) {
        if (live_out_set.insert(t).second) {
          live_out_order.push_back(t);
        }
      }
    }
    for (int64_t t : else_st.produced_order) {
      if (t >= 0 && else_st.avail_tokens.count(t) != 0) {
        if (live_out_set.insert(t).second) {
          live_out_order.push_back(t);
        }
      }
    }

    TokenSummary summary;
    summary.live_out = std::move(live_out_order);
    return summary;
  }

  void AnalyzeStmt(const Stmt &stmt, IterState &st) {
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &s : seq->seq) {
        AnalyzeStmt(s, st);
      }
      return;
    }

    if (const auto *inner_for = stmt.as<ForNode>()) {
      TokenSummary inner = AnalyzeFor(inner_for);
      for (int64_t t : inner.live_out) {
        st.MarkProduced(t);
      }

      return;
    }

    if (const auto *block = stmt.as<BlockNode>()) {
      if (block->init.defined()) {
        AnalyzeStmt(block->init.value(), st);
      }
      AnalyzeStmt(block->body, st);
      return;
    }

    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      AnalyzeStmt(realize->block, st);
      return;
    }

    if (const auto *alloc = stmt.as<AllocateNode>()) {
      AnalyzeStmt(alloc->body, st);
      return;
    }

    if (const auto *alloc_const = stmt.as<AllocateConstNode>()) {
      AnalyzeStmt(alloc_const->body, st);
      return;
    }

    if (const auto *buf_realize = stmt.as<BufferRealizeNode>()) {
      AnalyzeStmt(buf_realize->body, st);
      return;
    }

    if (const auto *decl_buf = stmt.as<DeclBufferNode>()) {
      AnalyzeStmt(decl_buf->body, st);
      return;
    }

    if (const auto *asserts = stmt.as<AssertStmtNode>()) {
      AnalyzeStmt(asserts->body, st);
      return;
    }

    if (const auto *eval = stmt.as<EvaluateNode>()) {
      if (const auto *call = eval->value.as<CallNode>()) {
        if (const auto *op_node = call->op.as<OpNode>()) {
          if (op_node->name == "tl.dma_copy") {
            int64_t token_id = ParseTokenIdFromArgs(call->args);
            st.MarkProduced(token_id);
            return;
          }
          if (op_node->name == "tl.sync_null_token") {
            int64_t token_id = ParseTokenIdFromArgs(call->args);
            st.MarkProduced(token_id);
            return;
          }
          if (op_node->name == "tl.wait_token") {
            int64_t token_id = ParseTokenIdFromArgs(call->args);
            if (token_id >= 0) {
              st.avail_tokens.erase(token_id);
            }
            return;
          }
        }
      }
      return;
    }

    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      AnalyzeStmt(attr->body, st);
      return;
    }

    if (const auto *let = stmt.as<LetStmtNode>()) {
      AnalyzeStmt(let->body, st);
      return;
    }

    if (const auto *ifs = stmt.as<IfThenElseNode>()) {
      IterState then_st = st;
      AnalyzeStmt(ifs->then_case, then_st);
      IterState else_st = st;
      if (ifs->else_case.defined()) {
        AnalyzeStmt(ifs->else_case.value(), else_st);
      }

      st.avail_tokens = then_st.avail_tokens;
      st.avail_tokens.insert(else_st.avail_tokens.begin(),
                             else_st.avail_tokens.end());

      MergeProducedOrder(st, then_st.produced_order);
      MergeProducedOrder(st, else_st.produced_order);
      return;
    }
  }
};
} // namespace

void CodeGenTileLangSunMMIO::EmitFor(const tir::ForNode *op) {
  TokenAnalyzer analyzer;
  TokenSummary summary = analyzer.AnalyzeFor(op);

  SunMMIOValue min = EnsureIndex(EvalExpr(op->min));
  SunMMIOValue extent = EnsureIndex(EvalExpr(op->extent));
  SunMMIOValue step = EmitConstIndex(1);
  SunMMIOValue upper = builder_->Binary(
      NewValueName(), BinaryOp::kAdd, ArithmeticFlavor::kIndex, min, extent,
      SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
      DataType::Int(32));
  std::string iv = "%" + op->loop_var->name_hint;
  builder_->BeginFor(iv, min, upper, step, op->annotations, summary.live_out);
  EnterScope();
  BindVar(
      op->loop_var,
      SunMMIOValue{
          op->loop_var.dtype(), iv,
          SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}}});
  VisitStmtTracked(op->body);
  ExitScope();
  builder_->EndFor();
}

void CodeGenTileLangSunMMIO::EmitIf(const tir::IfThenElseNode *op) {
  SunMMIOValue cond = EnsureType(
      EvalExpr(op->condition),
      SunMMIOType{SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}},
      DataType::Bool());
  TokenAnalyzer analyzer;
  TokenSummary summary = analyzer.AnalyzeIf(op);
  builder_->BeginIf(cond, summary.live_out);
  VisitStmtTracked(op->then_case);
  if (op->else_case.defined()) {
    builder_->BeginElse();
    VisitStmtTracked(op->else_case.value());
  }
  builder_->EndIf();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::ForNode *op) { EmitFor(op); }

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::LetStmtNode *op) {
  SunMMIOValue value = EvalExpr(op->value);
  EnterScope();
  BindVar(op->var, value);
  VisitStmtTracked(op->body);
  ExitScope();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::AttrStmtNode *op) {
  ScopedAttr attr{op->node, op->attr_key, EvalExpr(op->value)};
  attr_stack_.push_back(attr);
  VisitStmtTracked(op->body);
  attr_stack_.pop_back();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::IfThenElseNode *op) {
  EmitIf(op);
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::WhileNode *op) {
  (void)op;
  UnsupportedStmt(
      op, "WhileNode is not supported by SunMMIO direct MLIR lowering.");
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::AllocateNode *op) {
  std::string scope = GetAllocateStorageScope(op->buffer_var);
  EnterScope();
  EmitAlloc(op->buffer_var, op->dtype, op->extents, scope);
  VisitStmtTracked(op->body);
  ExitScope();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::AllocateConstNode *op) {
  UnsupportedStmt(
      op, "AllocateConstNode should be lowered/eliminated before SunMMIO "
          "codegen");
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::DeclBufferNode *op) {
  // DeclBuffer may still survive the current pipeline. For this backend path it
  // is treated as a benign declaration wrapper with no direct codegen effect.
  // Dedicated view/alias lowering can be added later if required.
  VisitStmtTracked(op->body);
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::BufferStoreNode *op) {
  UnsupportedStmt(
      op, "generic BufferStoreNode should not reach SunMMIO codegen; "
          "tiled buffer stores must be lowered through tile-aware paths");
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::BufferRealizeNode *op) {
  UnsupportedStmt(
      op, "BufferRealizeNode should be lowered into a concrete view/alias "
          "representation before SunMMIO codegen");
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::AssertStmtNode *op) {
  SunMMIOValue cond = EnsureType(
      EvalExpr(op->condition),
      SunMMIOType{SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}},
      DataType::Bool());
  SunMMIOValue msg = EvalExpr(op->message);
  std::string text = msg.value.empty() ? "\"assertion failed\"" : msg.value;
  builder_->EmitAssert(cond, text);
  VisitStmtTracked(op->body);
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::EvaluateNode *op) {
  (void)EvalExpr(op->value);
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::BlockNode *op) {
  auto traverse_range = [this](const Range &range) {
    (void)EvalExpr(range->min);
    (void)EvalExpr(range->extent);
  };
  auto traverse_buffer_region = [&traverse_range](const BufferRegion &region) {
    for (const Range &r : region->region) {
      traverse_range(r);
    }
  };
  auto traverse_annotation_obj = [this](const ffi::Any &value,
                                        const auto &self_ref) -> void {
    if (auto maybe_expr = value.as<PrimExpr>()) {
      (void)EvalExpr(maybe_expr.value());
      return;
    }
    if (auto maybe_arr_expr = value.as<ffi::Array<PrimExpr>>()) {
      for (const PrimExpr &item : maybe_arr_expr.value()) {
        (void)EvalExpr(item);
      }
      return;
    }
    if (auto maybe_arr_any = value.as<ffi::Array<ffi::Any>>()) {
      for (const ffi::Any &item : maybe_arr_any.value()) {
        self_ref(item, self_ref);
      }
      return;
    }
    if (auto maybe_map_expr = value.as<ffi::Map<ffi::String, PrimExpr>>()) {
      for (const auto &kv : maybe_map_expr.value()) {
        (void)EvalExpr(kv.second);
      }
      return;
    }
    if (auto maybe_map_any = value.as<ffi::Map<ffi::String, ffi::Any>>()) {
      for (const auto &kv : maybe_map_any.value()) {
        self_ref(kv.second, self_ref);
      }
      return;
    }
    if (auto maybe_map_any_any = value.as<ffi::Map<ffi::Any, ffi::Any>>()) {
      for (const auto &kv : maybe_map_any_any.value()) {
        self_ref(kv.first, self_ref);
        self_ref(kv.second, self_ref);
      }
      return;
    }
  };

  EnterScope();
  for (const IterVar &iv : op->iter_vars) {
    if (!var_table_.count(iv->var.get())) {
      BindVar(iv->var, EvalExpr(iv->var));
    }
  }
  for (const Buffer &alloc : op->alloc_buffers) {
    EmitAlloc(alloc->data, alloc->dtype, alloc->shape,
              alloc.scope().empty() ? "global" : alloc.scope());
    RegisterBuffer(alloc, false);
  }
  for (const MatchBufferRegion &match : op->match_buffers) {
    if (match->source.defined()) {
      RegisterBuffer(match->source->buffer, false);
      traverse_buffer_region(match->source);
    }
    if (!buffer_registry_.count(match->buffer.get())) {
      if (match->source.defined() &&
          buffer_registry_.count(match->source->buffer.get())) {
        const BufferBinding &src = LookupBuffer(match->source->buffer);
        RegisterBuffer(match->buffer, false, src.handle);
      } else {
        RegisterBuffer(match->buffer, false, NewValueName());
      }
    }
  }
  for (const BufferRegion &r : op->reads) {
    RegisterBuffer(r->buffer, false);
    traverse_buffer_region(r);
  }
  for (const BufferRegion &r : op->writes) {
    RegisterBuffer(r->buffer, false);
    traverse_buffer_region(r);
  }
  for (const auto &kv : op->annotations) {
    traverse_annotation_obj(kv.second, traverse_annotation_obj);
  }
  if (op->init.defined()) {
    VisitStmtTracked(op->init.value());
  }
  VisitStmtTracked(op->body);
  ExitScope();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::BlockRealizeNode *op) {
  UnsupportedStmt(
      op, "BlockRealizeNode should be eliminated by LowerOpaqueBlock before "
          "SunMMIO codegen");
}

void CodeGenTileLangSunMMIO::VisitStmtDefault_(const Object *op) {
  UnsupportedStmt(op, "No direct MLIR lowering handler implemented.");
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::VarNode *op) {
  auto it = var_table_.find(op);
  if (it != var_table_.end()) {
    return it->second;
  }
  SunMMIOValue info{op->dtype, "%" + op->name_hint, MapType(op->dtype)};
  var_table_[op] = info;
  return info;
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::SizeVarNode *op) {
  auto it = var_table_.find(op);
  if (it != var_table_.end()) {
    return it->second;
  }
  SunMMIOValue info{op->dtype, "%" + op->name_hint, MapType(op->dtype)};
  var_table_[op] = info;
  return info;
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::IntImmNode *op) {
  SunMMIOType ty = MapType(op->dtype);
  return builder_->ConstantInt(NewValueName(), op->value, ty, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::FloatImmNode *op) {
  std::ostringstream os;
  os << op->value;
  SunMMIOType ty = MapType(op->dtype);
  return builder_->ConstantFloat(NewValueName(), os.str(), ty, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::StringImmNode *op) {
  return SunMMIOValue{
      op->dtype, "\"" + static_cast<std::string>(op->value) + "\"",
      SunMMIOType{SunMMIOType::Kind::kUnknown, op->dtype, 1, {}}};
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::CastNode *op) {
  return EmitCast(EvalExpr(op->value), op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::CallNode *op) {
  return EmitCall(op);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::AddNode *op) {
  return EmitBinary("add", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::SubNode *op) {
  return EmitBinary("sub", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::MulNode *op) {
  return EmitBinary("mul", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::DivNode *op) {
  return EmitBinary("div", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::ModNode *op) {
  return EmitBinary("mod", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::FloorDivNode *op) {
  return EmitBinary("floordiv", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::FloorModNode *op) {
  return EmitBinary("floormod", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::MinNode *op) {
  return EmitBinary("min", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::MaxNode *op) {
  return EmitBinary("max", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::EQNode *op) {
  return EmitCmp("eq", op->a, op->b);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::NENode *op) {
  return EmitCmp("ne", op->a, op->b);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::LTNode *op) {
  return EmitCmp("lt", op->a, op->b);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::LENode *op) {
  return EmitCmp("le", op->a, op->b);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::GTNode *op) {
  return EmitCmp("gt", op->a, op->b);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::GENode *op) {
  return EmitCmp("ge", op->a, op->b);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::AndNode *op) {
  return EmitBinary("and", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::OrNode *op) {
  return EmitBinary("or", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::NotNode *op) {
  SunMMIOType bool_ty{SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}};
  SunMMIOValue v = EnsureType(EvalExpr(op->a), bool_ty, DataType::Bool());
  SunMMIOValue one =
      builder_->ConstantInt(NewValueName(), 1, bool_ty, DataType::Bool());
  return builder_->Binary(NewValueName(), BinaryOp::kXor,
                          ArithmeticFlavor::kBool, v, one, bool_ty,
                          DataType::Bool());
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::SelectNode *op) {
  SunMMIOType bool_ty{SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}};
  SunMMIOValue cond =
      EnsureType(EvalExpr(op->condition), bool_ty, DataType::Bool());
  SunMMIOValue tv = EvalExpr(op->true_value);
  SunMMIOValue fv = EvalExpr(op->false_value);
  fv = EnsureType(fv, tv.type, tv.dtype);
  return builder_->Select(NewValueName(), cond, tv, fv, tv.type, op->dtype);
}

SunMMIOValue
CodeGenTileLangSunMMIO::EmitLoad(const tir::Buffer &buffer,
                                 const ffi::Array<PrimExpr> &indices) {
  const BufferBinding &binding = LookupBuffer(buffer);
  std::vector<SunMMIOValue> idx_vals;
  for (const PrimExpr &idx : indices) {
    idx_vals.push_back(EnsureIndex(EvalExpr(idx)));
  }
  return builder_->Load(NewValueName(), binding.handle, idx_vals,
                        binding.buffer_type, buffer->dtype,
                        MapType(buffer->dtype));
}

void CodeGenTileLangSunMMIO::EmitStore(const tir::Buffer &buffer,
                                       const ffi::Array<PrimExpr> &indices,
                                       const SunMMIOValue &value) {
  const BufferBinding &binding = LookupBuffer(buffer);
  std::vector<SunMMIOValue> idx_vals;
  for (const PrimExpr &idx : indices) {
    idx_vals.push_back(EnsureIndex(EvalExpr(idx)));
  }
  SunMMIOValue casted =
      EnsureType(value, MapType(buffer->dtype), buffer->dtype);
  builder_->Store(casted, binding.handle, idx_vals, binding.buffer_type);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::BufferLoadNode *op) {
  UnsupportedExpr(
      op, "generic BufferLoadNode should not reach SunMMIO codegen; tiled "
          "buffer accesses must be lowered through tile-aware paths");
}

SunMMIOValue
CodeGenTileLangSunMMIO::VisitExpr_(const tir::ProducerLoadNode *op) {
  UnsupportedExpr(op, "ProducerLoadNode is not supported.");
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::RampNode *op) {
  DataType elem_dtype = op->dtype.with_lanes(1);
  SunMMIOType elem_ty = MapType(elem_dtype);
  SunMMIOType vec_ty = MapType(op->dtype);

  SunMMIOValue base = EvalExpr(op->base);
  SunMMIOValue stride = EvalExpr(op->stride);
  base = EnsureType(base, elem_ty, elem_dtype);
  stride = EnsureType(stride, elem_ty, elem_dtype);

  return builder_->Ramp(NewValueName(), base, stride, op->dtype.lanes(),
                        elem_ty, vec_ty, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::BroadcastNode *op) {
  SunMMIOValue scalar = EvalExpr(op->value);
  DataType scalar_dtype = op->dtype.with_lanes(1);
  SunMMIOType scalar_ty = MapType(scalar_dtype);
  SunMMIOType vec_ty = MapType(op->dtype);
  scalar = EnsureType(scalar, scalar_ty, scalar_dtype);

  return builder_->Broadcast(NewValueName(), scalar, op->dtype.lanes(),
                             scalar_ty, vec_ty, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::ShuffleNode *op) {
  UnsupportedExpr(op, "ShuffleNode lowering is not implemented yet.");
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::LetNode *op) {
  SunMMIOValue value = EvalExpr(op->value);
  EnterScope();
  BindVar(op->var, value);
  SunMMIOValue body = EvalExpr(op->body);
  ExitScope();
  return body;
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExprDefault_(const Object *op) {
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
    const tir::VarNode *var = scoped_vars_.back();
    scoped_vars_.pop_back();
    var_table_.erase(var);
  }

  size_t buffer_marker = buffer_scope_markers_.back();
  buffer_scope_markers_.pop_back();
  while (scoped_buffers_.size() > buffer_marker) {
    const tir::BufferNode *buffer = scoped_buffers_.back();
    scoped_buffers_.pop_back();
    buffer_registry_.erase(buffer);
  }
}

SunMMIOValue CodeGenTileLangSunMMIO::EmitBinary(const char *op_name,
                                                const tvm::PrimExpr &lhs,
                                                const tvm::PrimExpr &rhs,
                                                tvm::DataType dtype) {
  SunMMIOValue a = EvalExpr(lhs);
  SunMMIOValue b = EvalExpr(rhs);
  SunMMIOType result_type = MapType(dtype);
  a = EnsureType(a, result_type, dtype);
  b = EnsureType(b, result_type, dtype);
  std::string out = NewValueName();
  const std::string op(op_name);
  BinaryOp bin_op;
  if (op == "add")
    bin_op = BinaryOp::kAdd;
  else if (op == "sub")
    bin_op = BinaryOp::kSub;
  else if (op == "mul")
    bin_op = BinaryOp::kMul;
  else if (op == "div" || op == "floordiv")
    bin_op = BinaryOp::kDiv;
  else if (op == "mod" || op == "floormod")
    bin_op = BinaryOp::kMod;
  else if (op == "min")
    bin_op = BinaryOp::kMin;
  else if (op == "max")
    bin_op = BinaryOp::kMax;
  else if (op == "and")
    bin_op = BinaryOp::kAnd;
  else if (op == "or")
    bin_op = BinaryOp::kOr;
  else
    UnsupportedExpr(lhs.get(), "Unsupported binary op in EmitBinary: " + op);

  return builder_->Binary(out, bin_op, GetArithmeticFlavor(dtype), a, b,
                          result_type, dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::EmitCmp(const char *pred,
                                             const tvm::PrimExpr &lhs,
                                             const tvm::PrimExpr &rhs) {
  SunMMIOValue a = EvalExpr(lhs);
  SunMMIOValue b = EvalExpr(rhs);
  SunMMIOType ty = a.type;
  b = EnsureType(b, ty, a.dtype);
  std::string out = NewValueName();
  CompareOp cmp_op;
  std::string p(pred);
  if (p == "eq")
    cmp_op = CompareOp::kEQ;
  else if (p == "ne")
    cmp_op = CompareOp::kNE;
  else if (p == "lt")
    cmp_op = CompareOp::kLT;
  else if (p == "le")
    cmp_op = CompareOp::kLE;
  else if (p == "gt")
    cmp_op = CompareOp::kGT;
  else if (p == "ge")
    cmp_op = CompareOp::kGE;
  else
    UnsupportedExpr(lhs.get(), "Unsupported compare op in EmitCmp: " + p);
  return builder_->Compare(out, cmp_op, GetCompareDomain(a.dtype), a, b, ty);
}

SunMMIOValue CodeGenTileLangSunMMIO::EmitCast(const SunMMIOValue &v,
                                              tvm::DataType target_dtype) {
  SunMMIOType dst = MapType(target_dtype);
  if (v.type.kind == dst.kind && v.type.dtype == dst.dtype &&
      v.type.lanes == dst.lanes && v.type.shape.size() == dst.shape.size()) {
    return v;
  }
  return builder_->Cast(NewValueName(), v, dst, target_dtype);
}

CodeGenTileLangSunMMIO::CallBucket
CodeGenTileLangSunMMIO::ClassifyCall(const tir::CallNode *op) const {
  if (op->op.as<GlobalVarNode>()) {
    PrimExpr expr = tvm::ffi::GetRef<PrimExpr>(op);
    tir::CallEffectKind effect = SideEffect(expr);
    return effect <= tir::CallEffectKind::kPure ? CallBucket::kExternPure
                                                : CallBucket::kExternSideEffect;
  }
  const auto *op_node = op->op.as<OpNode>();
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
  if (name.find("sync") != std::string::npos ||
      name.find("barrier") != std::string::npos) {
    return CallBucket::kSync;
  }
  if (name.find("shuffle") != std::string::npos ||
      name.find("vector") != std::string::npos) {
    return CallBucket::kVector;
  }
  if (name.find("exp") != std::string::npos ||
      name.find("log") != std::string::npos ||
      name.find("sin") != std::string::npos ||
      name.find("cos") != std::string::npos ||
      name.find("sqrt") != std::string::npos ||
      name.find("pow") != std::string::npos) {
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

const char *CodeGenTileLangSunMMIO::CallBucketName(CallBucket bucket) const {
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

SunMMIOValue CodeGenTileLangSunMMIO::EmitCall(const tir::CallNode *op) {
  CallBucket bucket = ClassifyCall(op);
  if (bucket == CallBucket::kUnsupported) {
    UnsupportedExpr(op, "Unsupported call target.");
  }
  std::string callee = "unknown";
  if (const auto *op_node = op->op.as<OpNode>()) {
    callee = op_node->name;
  } else if (const auto *gv = op->op.as<GlobalVarNode>()) {
    callee = gv->name_hint;
  }
  bool is_token_intrin =
      (callee == "tl.sync_null_token" || callee == "tl.wait_token");
  std::vector<SunMMIOValue> operands;
  std::vector<std::string> string_args;
  for (int i = 0, e = static_cast<int>(op->args.size()); i < e; ++i) {
    const PrimExpr &arg = op->args[i];
    if (const auto *call = arg.as<CallNode>()) {
      if (const auto *op_node = call->op.as<OpNode>()) {
        if (op_node->name == "tl.sync_token_id" && call->args.size() == 1) {
          if (const auto *imm = call->args[0].as<IntImmNode>()) {
            MarkVisitedNodeType("tir.IntImm");
            string_args.push_back(
                "token_id=" + std::to_string(static_cast<int64_t>(imm->value)));
            continue;
          }
        }
      }
    }
    if (is_token_intrin && i == 0) {
      if (const auto *imm = arg.as<IntImmNode>()) {
        MarkVisitedNodeType("tir.IntImm");
        string_args.push_back("token_id=" +
                              std::to_string(static_cast<int64_t>(imm->value)));
        continue;
      }
    }
    if (const auto *s = arg.as<StringImmNode>()) {
      MarkVisitedNodeType("tir.StringImm");
      string_args.push_back(static_cast<std::string>(s->value));
      continue;
    }
    operands.push_back(EvalExpr(arg));
  }
  SunMMIOType ret_ty = MapType(op->dtype);
  std::string result_name = op->dtype.is_void() ? "" : NewValueName();
  return builder_->Call(result_name, callee, operands, string_args,
                        CallBucketName(bucket), op->dtype, ret_ty);
}

/*!
 * \brief Backend input invariants for generic SunMMIO codegen.
 *
 * This backend is not a generic TIR code generator. It expects the input to
 * have already been lowered into a SunMMIO-oriented form where tiled buffers
 * are accessed through tile-aware paths rather than generic element-wise
 * BufferLoad/BufferStore. Nodes such as BlockRealize, BufferRealize, and
 * DeclBuffer should normally have been eliminated or lowered before reaching
 * this layer. Likewise, generic BufferLoad/BufferStore on tiled buffers are
 * treated as pipeline violations unless intercepted by a dedicated tile-based
 * lowering path earlier in the pipeline.
 *
 * The generic SunMMIO codegen path is expected to handle control flow, scalar
 * and index expressions, loop structure, target intrinsics, and tile-aware
 * operations that have already been normalized into backend-expected forms.
 * Reaching unsupported structural nodes here should fail loudly so pipeline
 * regressions are caught early.
 */

[[noreturn]] void
CodeGenTileLangSunMMIO::UnsupportedStmt(const Object *op,
                                        const std::string &detail) const {
  // Generic SunMMIO codegen intentionally rejects pre-lowered structural forms.
  // Reaching unsupported nodes here indicates a pipeline invariant violation.
  std::ostringstream os;
  os << "CodeGenTileLangSunMMIO unsupported stmt: " << op->GetTypeKey();
  if (!detail.empty()) {
    os << " (" << detail << ")";
  }
  LOG(FATAL) << os.str();
  TVM_FFI_UNREACHABLE();
}

[[noreturn]] void
CodeGenTileLangSunMMIO::UnsupportedExpr(const Object *op,
                                        const std::string &detail) const {
  // Generic SunMMIO codegen intentionally rejects pre-lowered structural forms.
  // Reaching unsupported nodes here indicates a pipeline invariant violation.
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
