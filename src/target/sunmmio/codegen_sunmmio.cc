#include "codegen_sunmmio.h"
#include "sunmmio_mlir_builder.h"

#include "../../layout/layout.h"
#include "../../op/builtin.h"
#include "../../op/comm.h"
#include "../../op/region.h"
#include "../../op/utils.h"
#include "../sunmmio_utils.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ir/type.h>
#include <tvm/node/script_printer.h>
#include <tvm/node/structural_equal.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <unordered_set>
#include <utility>

#include <tvm/runtime/logging.h>

namespace tvm {
namespace codegen {
using namespace tir;

namespace {

class DeclBufferCollector final : public tir::StmtVisitor {
public:
  std::unordered_map<const tir::VarNode *, tir::Buffer> buffer_data_to_buffer;

private:
  void VisitStmt_(const tir::DeclBufferNode *op) final {
    Record(op->buffer);
    tir::StmtVisitor::VisitStmt_(op);
  }

  void Record(const tir::Buffer &buffer) {
    if (!buffer.defined() || !buffer->data.defined()) {
      return;
    }
    const tir::VarNode *data = buffer->data.get();
    auto it = buffer_data_to_buffer.find(data);
    if (it == buffer_data_to_buffer.end()) {
      buffer_data_to_buffer.emplace(data, buffer);
      // return;
    } else {
      LOG(WARNING) << "Found duplicate DeclBuffer for data var " << data;
    }
  }
};

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

bool IsSunmmioReduceRegisterTempBuffer(const tir::Buffer &buffer) {
  if (!buffer.defined()) {
    return false;
  }
  const std::string scope = buffer.scope();
  if (scope != "shared.rsram" && scope != "rsram") {
    return false;
  }
  const std::string name = buffer->name;
  return name.size() >= 4 && (name.rfind("_acc") == name.size() - 4 ||
                              name.rfind("_res") == name.size() - 4);
}

bool IsSunmmioLocalVarBuffer(const tir::Buffer &buffer) {
  if (!buffer.defined()) {
    return false;
  }
  const std::string scope = buffer.scope();
  return scope == "local.var";
}

bool IsSunmmioRsramScope(const std::string &scope) {
  return scope == tl::kSunmmioScopeRSRAM || scope == "rsram";
}

bool IsSunmmioAsramScope(const std::string &scope) {
  return scope == tl::kSunmmioScopeASRAM || scope == "asram";
}

bool IsSunmmioWsramScope(const std::string &scope) {
  return scope == tl::kSunmmioScopeWSRAM || scope == "wsram";
}

bool IsSuvmTilePickDType(DataType dtype) {
  dtype = CanonicalizeSuvmDType(dtype).with_lanes(1);
  if (dtype.bits() != 16 && dtype.bits() != 32) {
    return false;
  }
  return dtype.is_int() || dtype.is_uint() || dtype.is_float() ||
         dtype.is_bfloat16();
}

SunMMIOType MakeScalarType(DataType dtype) {
  dtype = CanonicalizeSuvmDType(dtype).with_lanes(1);
  return SunMMIOType{SunMMIOType::Kind::kScalar, dtype, 1, {}};
}

SunMMIOType MakeTileTypeForShape(DataType dtype,
                                 const std::vector<int64_t> &shape,
                                 SunMMIOType::Kind kind) {
  dtype = CanonicalizeSuvmDType(dtype).with_lanes(1);
  SunMMIOType type;
  type.kind = kind;
  type.dtype = dtype;
  type.lanes = 1;
  for (int64_t extent : shape) {
    type.shape.push_back(IntImm(DataType::Int(32), extent));
  }
  return type;
}

SunMMIOType MakeRank1TileType(DataType dtype, int64_t extent) {
  return MakeTileTypeForShape(dtype, {extent}, SunMMIOType::Kind::kTile);
}

SunMMIOType MakeRank1TileViewType(DataType dtype, int64_t extent) {
  return MakeTileTypeForShape(dtype, {extent}, SunMMIOType::Kind::kTileView);
}

SunMMIOType MakeRank2TileType(DataType dtype, int64_t rows, int64_t cols) {
  return MakeTileTypeForShape(dtype, {rows, cols}, SunMMIOType::Kind::kTile);
}

SunMMIOType MakeRank2TileViewType(DataType dtype, int64_t rows, int64_t cols) {
  return MakeTileTypeForShape(dtype, {rows, cols},
                              SunMMIOType::Kind::kTileView);
}

std::vector<int64_t> StaticBufferShape(const tir::Buffer &buffer) {
  std::vector<int64_t> shape;
  shape.reserve(buffer->shape.size());
  for (const PrimExpr &extent : buffer->shape) {
    const auto *imm = extent.as<IntImmNode>();
    if (!imm || imm->value <= 0) {
      return {};
    }
    shape.push_back(imm->value);
  }
  return shape;
}

bool SameTypeShape(const SunMMIOType &lhs, const SunMMIOType &rhs) {
  if (lhs.shape.size() != rhs.shape.size()) {
    return false;
  }
  StructuralEqual equal;
  for (size_t i = 0; i < lhs.shape.size(); ++i) {
    if (!equal(lhs.shape[i], rhs.shape[i])) {
      return false;
    }
  }
  return true;
}

std::string LocalVarValueName(const tir::VarNode *var) {
  ICHECK(var != nullptr);
  return "%local_var_" + var->name_hint;
}

} // namespace

CodeGenTileLangSunMMIO::CodeGenTileLangSunMMIO() = default;

void CodeGenTileLangSunMMIO::SetTarget(tvm::Target target) { target_ = target; }

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
  local_var_table_.clear();
  buffer_registry_.clear();
  buffer_data_to_buffer_.clear();
  attr_stack_.clear();
  scoped_vars_.clear();
  scoped_local_vars_.clear();
  scoped_buffers_.clear();
  var_scope_markers_.clear();
  local_var_scope_markers_.clear();
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

bool CodeGenTileLangSunMMIO::TryConsumeSyncTokenId(const tvm::PrimExpr &expr,
                                                   SunMMIOCallAttrs *attrs) {
  const auto *call = expr.as<tir::CallNode>();
  if (!call) {
    return false;
  }
  const auto *op_node = call->op.as<OpNode>();
  if (!op_node || op_node->name != "tl.sync_token_id") {
    return false;
  }

  MarkVisitedNodeType(call->GetTypeKey());
  MarkVisitedCallOpFromExpr(expr);
  ICHECK_EQ(call->args.size(), 1)
      << "tl.sync_token_id expects exactly one argument";
  const auto *imm = call->args[0].as<IntImmNode>();
  ICHECK(imm) << "tl.sync_token_id expects an IntImm token id";
  MarkVisitedNodeType(imm->GetTypeKey());
  (*attrs)[SunMMIOCallAttrKey::kTokenId] = static_cast<int64_t>(imm->value);
  return true;
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

void CodeGenTileLangSunMMIO::CollectDeclBuffers(const tir::Stmt &stmt) {
  DeclBufferCollector collector;
  collector(stmt);
  buffer_data_to_buffer_ = std::move(collector.buffer_data_to_buffer);
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
  CollectDeclBuffers(f->body);

  SunMMIOBuilder::TirLayoutMap layout_map;
  SunMMIOBuilder::TirLayoutMap global_layout_map;
  if (auto opt =
          f->GetAttr<SunMMIOBuilder::TirLayoutMap>(tl::attr::kLayoutMap)) {
    layout_map = opt.value();
  }
  if (auto opt = f->GetAttr<SunMMIOBuilder::TirLayoutMap>(
          tl::attr::kGlobalLayoutMap)) {
    global_layout_map = opt.value();
  }
  builder_->PushLayoutScope(layout_map, global_layout_map);

  EnterScope();
  std::vector<BuilderArg> args;
  std::vector<PendingExternalBuffer> pending_external_buffers;
  int arg_index = 0;
  for (const tir::Var &p : f->params) {
    std::string arg_name = "%arg" + std::to_string(arg_index++);
    auto buffer_it = buffer_data_to_buffer_.find(p.get());
    if (buffer_it != buffer_data_to_buffer_.end()) {
      const tir::Buffer &buffer = buffer_it->second;
      SunMMIOType buf_ty = MapBufferType(buffer);
      args.push_back({arg_name, buf_ty});
      BindVar(p, SunMMIOValue{p.dtype(), arg_name, buf_ty});
      RegisterBuffer(buffer, true, arg_name);
      pending_external_buffers.push_back({buffer, arg_name, buf_ty});
    } else {
      SunMMIOType arg_ty = MapType(p.dtype());
      args.push_back({arg_name, arg_ty});
      BindVar(p, SunMMIOValue{p.dtype(), arg_name, arg_ty});
    }
  }

  builder_->BeginFunction(gvar->name_hint, args);
  for (const PendingExternalBuffer &pending : pending_external_buffers) {
    BindExternalBufferLayout(pending);
  }
  VisitStmtTracked(f->body);
  builder_->EmitReturn();
  builder_->EndFunction();
  ExitScope();
  builder_->PopLayoutScope();
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
  dtype = CanonicalizeSuvmDType(dtype);
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
  SunMMIOType type;
  type.kind = SunMMIOType::Kind::kMemTensor;
  type.dtype = buffer->dtype.with_lanes(1);
  type.shape = std::move(shape);
  type.memory_scope = buffer.scope();
  type.byte_offset = 0;
  if (builder_) {
    builder_->ApplyLayoutToType(buffer, &type);
  }
  return type;
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
      v.type.lanes == target_type.lanes && SameTypeShape(v.type, target_type)) {
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

const SunMMIOValue &
CodeGenTileLangSunMMIO::LookupVar(const tir::VarNode *var) const {
  ICHECK(var != nullptr);
  auto it = var_table_.find(var);
  if (it != var_table_.end()) {
    return it->second;
  }
  LOG(FATAL) << "CodeGenTileLangSunMMIO: unbound TIR var `" << var->name_hint
             << "` reached SunMMIO codegen without a parameter, loop, let, "
                "allocation, or thread binding";
  TVM_FFI_UNREACHABLE();
}

const SunMMIOValue &
CodeGenTileLangSunMMIO::LookupLocalVar(const tir::VarNode *var) const {
  ICHECK(var != nullptr);
  auto it = local_var_table_.find(var);
  ICHECK(it != local_var_table_.end())
      << "CodeGenTileLangSunMMIO: unknown local.var buffer data "
      << var->name_hint;
  return it->second;
}

SunMMIOValue CodeGenTileLangSunMMIO::MaterializeDynamicLayoutExpr(
    const tvm::PrimExpr &expr) {
  arith::Analyzer analyzer;
  PrimExpr simplified = analyzer.Simplify(expr);
  tir::PostOrderVisit(simplified, [&](const ObjectRef &node) {
    if (!node.defined()) {
      return;
    }
    if (const auto *var = node.as<VarNode>()) {
      ICHECK(var_table_.count(var))
          << "SunMMIO dynamic layout expression depends on unbound runtime "
             "variable "
          << var->name_hint << " in " << simplified;
      return;
    }
    ICHECK(!node.as<BufferLoadNode>() && !node.as<ProducerLoadNode>() &&
           !node.as<CallNode>() && !node.as<RampNode>() &&
           !node.as<BroadcastNode>() && !node.as<ShuffleNode>() &&
           !node.as<LetNode>())
        << "SunMMIO dynamic layout expression must be scalar arithmetic over "
           "runtime parameters, got unsupported node "
        << node->GetTypeKey() << " in " << simplified;
  });
  return EnsureIndex(EvalExpr(simplified));
}

std::vector<SunMMIOValue> CodeGenTileLangSunMMIO::CollectDynamicLayoutValues(
    const std::vector<PrimExpr> &exprs) {
  std::vector<SunMMIOValue> values;
  values.reserve(exprs.size());
  for (const PrimExpr &expr : exprs) {
    arith::Analyzer analyzer;
    PrimExpr simplified = analyzer.Simplify(expr);
    if (simplified.as<IntImmNode>()) {
      continue;
    }
    values.push_back(MaterializeDynamicLayoutExpr(simplified));
  }
  return values;
}

void CodeGenTileLangSunMMIO::BindExternalBufferLayout(
    const PendingExternalBuffer &pending) {
  std::vector<SunMMIOValue> dynamic_shapes =
      CollectDynamicLayoutValues(pending.type.layout_hshape);
  std::vector<SunMMIOValue> dynamic_strides =
      CollectDynamicLayoutValues(pending.type.layout_hstride);
  if (dynamic_shapes.empty() && dynamic_strides.empty()) {
    return;
  }

  SunMMIOValue source{pending.buffer->dtype, pending.handle, pending.type};
  std::string bound_handle = pending.handle + "_layout";
  SunMMIOValue bound = builder_->BindLayout(bound_handle, source,
                                            dynamic_shapes, dynamic_strides);

  var_table_[pending.buffer->data.get()] = bound;
  auto it = buffer_registry_.find(pending.buffer.get());
  ICHECK(it != buffer_registry_.end())
      << "Missing registered external buffer for dynamic layout binding: "
      << pending.buffer->name;
  it->second.handle = bound.value;
  it->second.buffer_type = bound.type;
}

void CodeGenTileLangSunMMIO::RegisterBuffer(const tir::Buffer &buffer,
                                            bool is_external,
                                            const std::string &handle_hint) {
  if (!buffer.defined()) {
    return;
  }
  if (IsSunmmioLocalVarBuffer(buffer)) {
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
  if (IsSunmmioReduceRegisterTempBuffer(buffer)) {
    buffer_registry_[buffer.get()] = std::move(binding);
    scoped_buffers_.push_back(buffer.get());
    return;
  }
  if (!handle_hint.empty()) {
    binding.handle = handle_hint;
    auto storage_it = var_table_.find(buffer->data.get());
    if (storage_it != var_table_.end() &&
        storage_it->second.type.kind == SunMMIOType::Kind::kMemTensor) {
      binding.buffer_type = storage_it->second.type;
    }
  } else {
    const SunMMIOValue &storage = LookupVar(buffer->data.get());
    binding.handle = storage.value;
    if (storage.type.kind == SunMMIOType::Kind::kMemTensor) {
      binding.buffer_type = storage.type;
    }
  }
  buffer_registry_[buffer.get()] = std::move(binding);
  scoped_buffers_.push_back(buffer.get());
}

const BufferBinding &
CodeGenTileLangSunMMIO::LookupBuffer(const tir::Buffer &buffer) const {
  auto it = buffer_registry_.find(buffer.get());
  if (it == buffer_registry_.end()) {
    auto data_it = buffer_data_to_buffer_.find(buffer->data.get());
    ICHECK(data_it != buffer_data_to_buffer_.end())
        << "SunMMIO LookupBuffer: missing buffer registration for name="
        << buffer->name << ", buffer_ptr=" << buffer.get()
        << ", data_name=" << buffer->data->name_hint
        << ", data_ptr=" << buffer->data.get();
    auto fallback_it = buffer_registry_.find(data_it->second.get());
    ICHECK(fallback_it != buffer_registry_.end())
        << "SunMMIO LookupBuffer: fallback buffer for name=" << buffer->name
        << " is not registered; fallback_name=" << data_it->second->name
        << ", fallback_ptr=" << data_it->second.get();
    return fallback_it->second;
  }
  return it->second;
}

void CodeGenTileLangSunMMIO::EmitAlloc(
    const tir::Buffer &buffer, const std::string &scope_hint,
    const ffi::Map<ffi::String, ffi::Any> &annotations) {
  std::vector<SunMMIOValue> dyn_extents;
  for (const PrimExpr &dim : buffer->shape) {
    if (!dim.as<IntImmNode>()) {
      dyn_extents.push_back(EnsureIndex(EvalExpr(dim)));
    }
  }

  SunMMIOType memtensor_type = MapBufferType(buffer);
  if (memtensor_type.memory_scope.empty()) {
    memtensor_type.memory_scope = scope_hint;
  }

  std::optional<std::string> ping_pong;
  auto ping_pong_it = annotations.find(tl::attr::kSunmmioAllocPingPong);
  if (ping_pong_it != annotations.end()) {
    ping_pong =
        static_cast<std::string>(Downcast<ffi::String>((*ping_pong_it).second));
  }

  SunMMIOValue alloc =
      builder_->Alloc(NewValueName(), memtensor_type, dyn_extents, scope_hint,
                      buffer->dtype, std::move(ping_pong));
  BindVar(buffer->data, alloc);

  auto it = buffer_registry_.find(buffer.get());
  if (it != buffer_registry_.end()) {
    it->second.handle = alloc.value;
    it->second.buffer_type = alloc.type;
  }
}

void CodeGenTileLangSunMMIO::EmitLocalVarAlloc(const tir::AllocateNode *op,
                                               const tir::Buffer &buffer) {
  ICHECK(IsSunmmioLocalVarBuffer(buffer));
  ICHECK_EQ(op->ConstantAllocationSize(), 1)
      << "local.var allocation must be scalar-sized";

  PrimExpr init = tir::make_const(op->dtype, 0);
  auto init_it = op->annotations.find(tl::attr::kLocalVarInit);
  if (init_it != op->annotations.end()) {
    PrimExpr user_init = Downcast<PrimExpr>((*init_it).second);
    if (!user_init.dtype().is_void() && user_init.dtype() != op->dtype) {
      user_init = tir::Cast(op->dtype, user_init);
    }
    init = user_init;
  }

  DataType dtype = CanonicalizeSuvmDType(buffer->dtype);
  SunMMIOValue init_value = EnsureType(EvalExpr(init), MapType(dtype), dtype);
  SunMMIOValue state = builder_->BindValueAlias(
      LocalVarValueName(buffer->data.get()), init_value);
  local_var_table_[buffer->data.get()] = state;
  scoped_local_vars_.push_back(buffer->data.get());
}

SunMMIOValue
CodeGenTileLangSunMMIO::EmitLocalVarLoad(const tir::Buffer &buffer,
                                         const ffi::Array<PrimExpr> &indices) {
  ICHECK(IsSunmmioLocalVarBuffer(buffer));
  ICHECK_EQ(indices.size(), 1)
      << "local.var buffer loads must use exactly one scalar index";
  arith::Analyzer analyzer;
  PrimExpr index = analyzer.Simplify(indices[0]);
  const auto *index_imm = index.as<IntImmNode>();
  ICHECK(index_imm && index_imm->value == 0)
      << "local.var buffer loads only support index 0";
  return LookupLocalVar(buffer->data.get());
}

void CodeGenTileLangSunMMIO::EmitLocalVarStore(
    const tir::Buffer &buffer, const ffi::Array<PrimExpr> &indices,
    const SunMMIOValue &value) {
  ICHECK(IsSunmmioLocalVarBuffer(buffer));
  ICHECK_EQ(indices.size(), 1)
      << "local.var buffer stores must use exactly one scalar index";
  arith::Analyzer analyzer;
  PrimExpr index = analyzer.Simplify(indices[0]);
  const auto *index_imm = index.as<IntImmNode>();
  ICHECK(index_imm && index_imm->value == 0)
      << "local.var buffer stores only support index 0";

  const SunMMIOValue &current = LookupLocalVar(buffer->data.get());
  DataType dtype = CanonicalizeSuvmDType(buffer->dtype);
  SunMMIOValue casted = EnsureType(value, MapType(dtype), dtype);
  SunMMIOValue state = builder_->BindValueAlias(current.value, casted);
  local_var_table_[buffer->data.get()] = state;
}

std::vector<SunMMIOValue> CodeGenTileLangSunMMIO::CollectLocalVarLiveOutValues(
    const tir::Stmt &stmt) const {
  class Collector final : public tir::StmtVisitor {
  public:
    explicit Collector(const std::unordered_map<const tir::VarNode *,
                                                SunMMIOValue> &local_var_table)
        : local_var_table_(local_var_table) {}

    std::vector<SunMMIOValue> values;

  private:
    void VisitStmt_(const tir::BufferStoreNode *op) final {
      if (IsSunmmioLocalVarBuffer(op->buffer)) {
        const tir::VarNode *data = op->buffer->data.get();
        auto it = local_var_table_.find(data);
        if (it != local_var_table_.end() && seen_.insert(data).second) {
          values.push_back(it->second);
        }
      }
      tir::StmtVisitor::VisitStmt_(op);
    }

    const std::unordered_map<const tir::VarNode *, SunMMIOValue>
        &local_var_table_;
    std::unordered_set<const tir::VarNode *> seen_;
  };

  Collector collector(local_var_table_);
  collector(stmt);
  return std::move(collector.values);
}

static void
AppendUniqueLocalVarLiveOutValues(std::vector<SunMMIOValue> *dst,
                                  const std::vector<SunMMIOValue> &src) {
  std::unordered_set<std::string> seen;
  for (const SunMMIOValue &value : *dst) {
    seen.insert(value.value);
  }
  for (const SunMMIOValue &value : src) {
    if (seen.insert(value.value).second) {
      dst->push_back(value);
    }
  }
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

  TokenSummary AnalyzeWhile(const tir::WhileNode *while_op) {
    IterState st;
    AnalyzeStmt(while_op->body, st);

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

    if (const auto *inner_while = stmt.as<WhileNode>()) {
      TokenSummary inner = AnalyzeWhile(inner_while);
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
          if (op_node->name == "tl.sunmmio_layout_transform") {
            int64_t token_id = ParseTokenIdFromArgs(call->args);
            st.MarkProduced(token_id);
            return;
          }
          if (op_node->name == "tl.mma_sunmmio") {
            int64_t token_id = ParseTokenIdFromArgs(call->args);
            st.MarkProduced(token_id);
            return;
          }
          if (op_node->name == "tl.broadcast_") {
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
  std::vector<SunMMIOValue> local_live_out_values =
      CollectLocalVarLiveOutValues(op->body);

  SunMMIOValue min = EnsureIndex(EvalExpr(op->min));
  SunMMIOValue extent = EnsureIndex(EvalExpr(op->extent));
  SunMMIOValue step = EmitConstIndex(1);
  SunMMIOValue upper = builder_->Binary(
      NewValueName(), BinaryOp::kAdd, ArithmeticFlavor::kIndex, min, extent,
      SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
      DataType::Int(32));
  std::string iv = "%" + op->loop_var->name_hint;
  if (!local_live_out_values.empty()) {
    builder_->BeginFor(iv, min, upper, step, op->annotations, summary.live_out,
                       local_live_out_values);
  } else {
    builder_->BeginFor(iv, min, upper, step, op->annotations, summary.live_out);
  }
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
  std::vector<SunMMIOValue> local_live_out_values =
      CollectLocalVarLiveOutValues(op->then_case);
  if (op->else_case.defined()) {
    std::vector<SunMMIOValue> else_live_out_values =
        CollectLocalVarLiveOutValues(op->else_case.value());
    AppendUniqueLocalVarLiveOutValues(&local_live_out_values,
                                      else_live_out_values);
  }
  if (!local_live_out_values.empty()) {
    builder_->BeginIf(cond, summary.live_out, local_live_out_values);
  } else {
    builder_->BeginIf(cond, summary.live_out);
  }
  VisitStmtTracked(op->then_case);
  if (op->else_case.defined()) {
    builder_->BeginElse();
    VisitStmtTracked(op->else_case.value());
  }
  builder_->EndIf();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::ForNode *op) {
  if (TryLowerTilesScope(op)) {
    return;
  }
  EmitFor(op);
}

void CodeGenTileLangSunMMIO::EmitWhile(const tir::WhileNode *op) {
  TokenAnalyzer analyzer;
  TokenSummary summary = analyzer.AnalyzeWhile(op);
  std::vector<SunMMIOValue> local_live_out_values =
      CollectLocalVarLiveOutValues(op->body);
  if (!local_live_out_values.empty()) {
    builder_->BeginWhile(summary.live_out, local_live_out_values);
  } else {
    builder_->BeginWhile(summary.live_out);
  }
  SunMMIOValue cond = EnsureType(
      EvalExpr(op->condition),
      SunMMIOType{SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}},
      DataType::Bool());
  builder_->BeginWhileBody(cond);
  EnterScope();
  VisitStmtTracked(op->body);
  ExitScope();
  builder_->EndWhile();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::LetStmtNode *op) {
  SunMMIOValue value = EvalExpr(op->value);
  EnterScope();
  BindVar(op->var, value);
  VisitStmtTracked(op->body);
  ExitScope();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::AttrStmtNode *op) {
  if (op->attr_key == tir::attr::thread_extent) {
    IterVar iv = Downcast<IterVar>(op->node);
    if (iv->thread_tag == "blockIdx.x") {
      EnterScope();
      BindVar(iv->var, builder_->GetCoreId(NewValueName(), iv->var.dtype()));
      VisitStmtTracked(op->body);
      ExitScope();
    } else {
      VisitStmtTracked(op->body);
    }
    return;
  }
  ScopedAttr attr{op->node, op->attr_key, EvalExpr(op->value)};
  attr_stack_.push_back(attr);
  VisitStmtTracked(op->body);
  attr_stack_.pop_back();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::IfThenElseNode *op) {
  EmitIf(op);
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::WhileNode *op) {
  EmitWhile(op);
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::AllocateNode *op) {
  EnterScope();
  auto buffer_it = buffer_data_to_buffer_.find(op->buffer_var.get());
  if (buffer_it != buffer_data_to_buffer_.end()) {
    const tir::Buffer &buffer = buffer_it->second;
    if (IsSunmmioLocalVarBuffer(buffer)) {
      EmitLocalVarAlloc(op, buffer);
    } else if (IsSunmmioReduceRegisterTempBuffer(buffer)) {
      // TIR materializes reduce intermediates as alloc_buffer so the algorithm
      // can be expressed with BufferLoad/Store.  On SunMMIO these values live
      // in vector-core tile registers and are lowered inside the Tiles scope as
      // SSA tiles, not as rsram memtensors.
    } else {
      std::string scope = GetAllocateStorageScope(op->buffer_var);
      EmitAlloc(buffer_it->second, scope, op->annotations);
    }
  } else {
    LOG(FATAL) << "SunMMIO SUVM allocate cannot find buffer for variable "
               << op->buffer_var->name_hint;
    TVM_FFI_UNREACHABLE();
  }
  VisitStmtTracked(op->body);
  ExitScope();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::AllocateConstNode *op) {
  UnsupportedStmt(
      op, "AllocateConstNode should be lowered/eliminated before SunMMIO "
          "codegen");
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::DeclBufferNode *op) {
  EnterScope();
  if (!IsSunmmioLocalVarBuffer(op->buffer) &&
      !IsSunmmioReduceRegisterTempBuffer(op->buffer)) {
    auto data_it = var_table_.find(op->buffer->data.get());
    RegisterBuffer(op->buffer, false,
                   data_it != var_table_.end() ? data_it->second.value
                                               : NewValueName());
  }
  VisitStmtTracked(op->body);
  ExitScope();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::BufferStoreNode *op) {
  if (IsSunmmioLocalVarBuffer(op->buffer)) {
    EmitLocalVarStore(op->buffer, op->indices, EvalExpr(op->value));
    return;
  }
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
  if (const auto *call = op->value.as<tir::CallNode>()) {
    if (call->op.same_as(tir::builtin::ret())) {
      MarkVisitedCallOpFromExpr(op->value);
      ICHECK_EQ(call->args.size(), 1) << "tir.ret expects one argument";
      const auto *imm = call->args[0].as<tir::IntImmNode>();
      ICHECK(imm && imm->value == 0)
          << "SunMMIO device kernel only supports T.ret(0)";
      MarkVisitedNodeType("tir.IntImm");
      return;
    }
  }
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
      LOG(FATAL) << "CodeGenTileLangSunMMIO: unbound block iter var `"
                 << iv->var->name_hint
                 << "` reached SunMMIO codegen without a BlockRealize binding";
      TVM_FFI_UNREACHABLE();
    }
  }
  for (const Buffer &alloc : op->alloc_buffers) {
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
  return LookupVar(op);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::SizeVarNode *op) {
  return LookupVar(op);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::IntImmNode *op) {
  DataType dtype = CanonicalizeSuvmDType(op->dtype);
  SunMMIOType ty = MapType(dtype);
  return builder_->ConstantInt(NewValueName(), op->value, ty, dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::FloatImmNode *op) {
  std::ostringstream os;
  os << op->value;
  DataType dtype = CanonicalizeSuvmDType(op->dtype);
  SunMMIOType ty = MapType(dtype);
  return builder_->ConstantFloat(NewValueName(), os.str(), ty, dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::StringImmNode *op) {
  DataType dtype = CanonicalizeSuvmDType(op->dtype);
  return SunMMIOValue{dtype, "\"" + static_cast<std::string>(op->value) + "\"",
                      SunMMIOType{SunMMIOType::Kind::kUnknown, dtype, 1, {}}};
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
  DataType dtype = CanonicalizeSuvmDType(op->dtype);
  return builder_->Select(NewValueName(), cond, tv, fv, tv.type, dtype);
}

SunMMIOValue
CodeGenTileLangSunMMIO::EmitLoad(const tir::Buffer &buffer,
                                 const ffi::Array<PrimExpr> &indices) {
  if (IsSunmmioLocalVarBuffer(buffer)) {
    return EmitLocalVarLoad(buffer, indices);
  }
  const BufferBinding &binding = LookupBuffer(buffer);
  std::vector<SunMMIOValue> idx_vals;
  for (const PrimExpr &idx : indices) {
    idx_vals.push_back(EnsureIndex(EvalExpr(idx)));
  }
  DataType dtype = CanonicalizeSuvmDType(buffer->dtype);
  return builder_->Load(NewValueName(), binding.handle, idx_vals,
                        binding.buffer_type, dtype, MapType(dtype));
}

SunMMIOValue
CodeGenTileLangSunMMIO::EmitScalarTilePick(const tir::BufferLoadNode *op) {
  const tir::Buffer &buffer = op->buffer;
  const ffi::Array<PrimExpr> &indices = op->indices;
  const std::string scope = buffer.scope();
  DataType dtype = CanonicalizeSuvmDType(buffer->dtype).with_lanes(1);
  auto describe = [&]() {
    std::ostringstream os;
    os << "buffer=" << buffer->name << ", scope=" << scope
       << ", dtype=" << dtype << ", rank=" << indices.size();
    return os.str();
  };

  if (IsSunmmioAsramScope(scope) || IsSunmmioWsramScope(scope)) {
    UnsupportedExpr(
        op, "Sunmmio scalar BufferLoad cannot read from ASRAM/WSRAM; stage "
            "readable side data through RSRAM first. " +
                describe());
  }
  if (!IsSunmmioRsramScope(scope)) {
    UnsupportedExpr(
        op, "Sunmmio scalar BufferLoad from DRAM/global must be legalized by "
            "staging through RSRAM before codegen. " +
                describe());
  }

  const BufferBinding &binding = LookupBuffer(buffer);
  if (!IsSuvmTilePickDType(dtype)) {
    UnsupportedExpr(op, "Sunmmio scalar BufferLoad tile.pick supports only "
                        "i16/ui16/i32/ui32/bf16/f32. " +
                            describe());
  }
  if (indices.size() != 1 && indices.size() != 2) {
    UnsupportedExpr(
        op, "Sunmmio RSRAM scalar BufferLoad tile.pick currently "
            "supports rank-1 or rank-2 RSRAM buffers; stage higher-rank "
            "side data into an explicit 1D/2D RSRAM buffer before pick. " +
                describe());
  }

  const int64_t dtype_bits = static_cast<int64_t>(dtype.bits());
  ICHECK_GT(dtype_bits, 0) << "Unexpected zero-width dtype for " << describe();
  const tl::SunmmioTileProcessorConfig tile_processor_config =
      tl::GetSunmmioTileProcessorConfig(target_);
  ICHECK_GT(tile_processor_config.register_bits, 0)
      << "Sunmmio tile.pick requires a positive vector register width";
  ICHECK_EQ(tile_processor_config.register_bits % dtype_bits, 0)
      << "Sunmmio vector register width " << tile_processor_config.register_bits
      << " is not divisible by scalar dtype bit-width " << dtype_bits;
  const int64_t tile_elems = tile_processor_config.register_bits / dtype_bits;
  ICHECK_GT(tile_elems, 0) << "Sunmmio tile.pick produced empty tile";

  SunMMIOValue memtensor{dtype, binding.handle, binding.buffer_type};
  if (indices.size() == 1) {
    PrimExpr tile_elems_expr = IntImm(indices[0].dtype(), tile_elems);
    PrimExpr partition_expr = floordiv(indices[0], tile_elems_expr);
    PrimExpr local_expr = floormod(indices[0], tile_elems_expr);

    SunMMIOValue partition = EnsureIndex(EvalExpr(partition_expr));
    SunMMIOValue local_index = EnsureIndex(EvalExpr(local_expr));

    SunMMIOType tile_view_type = MakeRank1TileViewType(dtype, tile_elems);
    SunMMIOValue tile_view = builder_->GetPartitionedTileView(
        NewValueName(), memtensor, {partition}, {0}, tile_view_type, dtype);
    SunMMIOType tile_type = MakeRank1TileType(dtype, tile_elems);
    SunMMIOValue tile = builder_->TileLoad(NewValueName(), tile_view, tile_type,
                                           std::nullopt, std::nullopt, dtype);
    return builder_->TilePick(NewValueName(), tile, {local_index},
                              MakeScalarType(dtype), dtype);
  }

  std::vector<int64_t> static_shape = StaticBufferShape(buffer);
  if (static_shape.size() != 2) {
    UnsupportedExpr(op,
                    "Sunmmio rank-2 RSRAM scalar BufferLoad tile.pick requires "
                    "a statically-shaped 2D RSRAM staging buffer. " +
                        describe());
  }
  const int64_t rows = static_shape[0];
  const int64_t cols = static_shape[1];

  SunMMIOValue row_index = EnsureIndex(EvalExpr(indices[0]));
  SunMMIOValue col_index = EnsureIndex(EvalExpr(indices[1]));
  SunMMIOValue zero = EmitConstIndex(0);
  SunMMIOType tile_view_type = MakeRank2TileViewType(dtype, rows, cols);
  SunMMIOValue tile_view = builder_->GetPartitionedTileView(
      NewValueName(), memtensor, {zero, zero}, {0, 1}, tile_view_type, dtype);
  SunMMIOType tile_type = MakeRank2TileType(dtype, rows, cols);
  SunMMIOValue tile = builder_->TileLoad(NewValueName(), tile_view, tile_type,
                                         std::nullopt, std::nullopt, dtype);
  return builder_->TilePick(NewValueName(), tile, {row_index, col_index},
                            MakeScalarType(dtype), dtype);
}

void CodeGenTileLangSunMMIO::EmitStore(const tir::Buffer &buffer,
                                       const ffi::Array<PrimExpr> &indices,
                                       const SunMMIOValue &value) {
  if (IsSunmmioLocalVarBuffer(buffer)) {
    EmitLocalVarStore(buffer, indices, value);
    return;
  }
  const BufferBinding &binding = LookupBuffer(buffer);
  std::vector<SunMMIOValue> idx_vals;
  for (const PrimExpr &idx : indices) {
    idx_vals.push_back(EnsureIndex(EvalExpr(idx)));
  }
  DataType dtype = CanonicalizeSuvmDType(buffer->dtype);
  SunMMIOValue casted = EnsureType(value, MapType(dtype), dtype);
  builder_->Store(casted, binding.handle, idx_vals, binding.buffer_type);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::BufferLoadNode *op) {
  if (IsSunmmioLocalVarBuffer(op->buffer)) {
    return EmitLocalVarLoad(op->buffer, op->indices);
  }
  return EmitScalarTilePick(op);
}

SunMMIOValue
CodeGenTileLangSunMMIO::VisitExpr_(const tir::ProducerLoadNode *op) {
  UnsupportedExpr(op, "ProducerLoadNode is not supported.");
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::RampNode *op) {
  DataType vec_dtype = CanonicalizeSuvmDType(op->dtype);
  DataType elem_dtype = vec_dtype.with_lanes(1);
  SunMMIOType elem_ty = MapType(elem_dtype);
  SunMMIOType vec_ty = MapType(vec_dtype);

  SunMMIOValue base = EvalExpr(op->base);
  SunMMIOValue stride = EvalExpr(op->stride);
  base = EnsureType(base, elem_ty, elem_dtype);
  stride = EnsureType(stride, elem_ty, elem_dtype);

  return builder_->Ramp(NewValueName(), base, stride, vec_dtype.lanes(),
                        elem_ty, vec_ty, vec_dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::BroadcastNode *op) {
  SunMMIOValue scalar = EvalExpr(op->value);
  DataType vec_dtype = CanonicalizeSuvmDType(op->dtype);
  DataType scalar_dtype = vec_dtype.with_lanes(1);
  SunMMIOType scalar_ty = MapType(scalar_dtype);
  SunMMIOType vec_ty = MapType(vec_dtype);
  scalar = EnsureType(scalar, scalar_ty, scalar_dtype);

  return builder_->Broadcast(NewValueName(), scalar, vec_dtype.lanes(),
                             scalar_ty, vec_ty, vec_dtype);
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
  local_var_scope_markers_.push_back(scoped_local_vars_.size());
  buffer_scope_markers_.push_back(scoped_buffers_.size());
}

void CodeGenTileLangSunMMIO::ExitScope() {
  ICHECK(!var_scope_markers_.empty());
  ICHECK(!local_var_scope_markers_.empty());
  ICHECK(!buffer_scope_markers_.empty());

  size_t var_marker = var_scope_markers_.back();
  var_scope_markers_.pop_back();
  while (scoped_vars_.size() > var_marker) {
    const tir::VarNode *var = scoped_vars_.back();
    scoped_vars_.pop_back();
    var_table_.erase(var);
  }

  size_t local_var_marker = local_var_scope_markers_.back();
  local_var_scope_markers_.pop_back();
  while (scoped_local_vars_.size() > local_var_marker) {
    const tir::VarNode *var = scoped_local_vars_.back();
    scoped_local_vars_.pop_back();
    local_var_table_.erase(var);
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
  dtype = CanonicalizeSuvmDType(dtype);
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
  else if (op == "xor")
    bin_op = BinaryOp::kXor;
  else if (op == "shl")
    bin_op = BinaryOp::kShl;
  else if (op == "shr")
    bin_op = BinaryOp::kShr;
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
  target_dtype = CanonicalizeSuvmDType(target_dtype);
  SunMMIOType dst = MapType(target_dtype);
  if (v.type.kind == dst.kind && v.type.dtype == dst.dtype &&
      v.type.lanes == dst.lanes && SameTypeShape(v.type, dst)) {
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
      name == "tl.broadcast_" || name.find("sunmmio") != std::string::npos) {
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

SunMMIOValue
CodeGenTileLangSunMMIO::EmitRegionCall(const tvm::PrimExpr &region_expr,
                                       int64_t byte_offset) {
  if (region_expr.defined()) {
    MarkVisitedNodeType(region_expr->GetTypeKey());
    MarkVisitedCallOpFromExpr(region_expr);
  }
  if (const auto *region_call = region_expr.as<tir::CallNode>()) {
    if (!region_call->args.empty()) {
      if (const auto *load = region_call->args[0].as<tir::BufferLoadNode>()) {
        MarkVisitedNodeType(load->GetTypeKey());
      }
    }
  }

  BufferRegion region = tl::NormalizeToBufferRegion(region_expr);
  const BufferBinding &binding = LookupBuffer(region->buffer);
  std::vector<SunMMIOValue> mins;
  std::vector<int64_t> extents;
  mins.reserve(region->region.size());
  extents.reserve(region->region.size());
  arith::Analyzer analyzer;
  for (const Range &range : region->region) {
    const auto *extent_imm = range->extent.as<IntImmNode>();
    ICHECK(extent_imm) << "tl.tileop.region extent must be IntImm";
    MarkVisitedNodeType(range->extent->GetTypeKey());
    extents.push_back(static_cast<int64_t>(extent_imm->value));
    PrimExpr min = floordiv(range->min, range->extent);
    min = analyzer.Simplify(min);
    mins.push_back(EvalExpr(min));
  }
  SunMMIOType ret_ty = MapType(region_expr.dtype());
  std::string result_name = region_expr.dtype().is_void() ? "" : NewValueName();
  return builder_->RegionCall(result_name, binding.handle, mins, extents,
                              region_expr.dtype(), ret_ty, byte_offset);
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
  std::vector<SunMMIOValue> operands;
  SunMMIOCallAttrs attrs;
  if (callee == "tl.tileop.region") {
    return EmitRegionCall(tvm::ffi::GetRef<PrimExpr>(op));
  } else if (callee == "tl.sync_null_token" || callee == "tl.wait_token") {
    for (int i = 0, e = static_cast<int>(op->args.size()); i < e; ++i) {
      const PrimExpr &arg = op->args[i];
      if (i == 0) {
        if (const auto *imm = arg.as<IntImmNode>()) {
          MarkVisitedNodeType("tir.IntImm");
          attrs[SunMMIOCallAttrKey::kTokenId] =
              static_cast<int64_t>(imm->value);
          continue;
        }
      }
      if (const auto *s = arg.as<StringImmNode>()) {
        MarkVisitedNodeType("tir.StringImm");
        continue;
      }
      operands.push_back(EvalExpr(arg));
    }
  } else if (callee == "tl.barrier_init" ||
             callee == "tl.barrier_arrive_and_wait") {
    ICHECK_GE(op->args.size(), 1U)
        << callee << " expects participant_mask argument";
    const PrimExpr &mask = op->args[0];
    attrs[SunMMIOCallAttrKey::kBarrierMaskKey] =
        TVMScriptPrinter::Script(mask, std::nullopt);
    if (const auto *imm = mask.as<IntImmNode>()) {
      MarkVisitedNodeType("tir.IntImm");
      int64_t mask_value = static_cast<int64_t>(imm->value);
      if (mask_value >= 0) {
        attrs[SunMMIOCallAttrKey::kParticipantMask] = mask_value;
      }
    } else {
      operands.push_back(EvalExpr(mask));
    }
    std::vector<int64_t> candidate_masks;
    for (size_t i = 1; i < op->args.size(); ++i) {
      const auto *imm = op->args[i].as<IntImmNode>();
      ICHECK(imm) << callee << " candidate masks must be IntImm";
      MarkVisitedNodeType("tir.IntImm");
      int64_t candidate_mask = static_cast<int64_t>(imm->value);
      if (std::find(candidate_masks.begin(), candidate_masks.end(),
                    candidate_mask) == candidate_masks.end()) {
        candidate_masks.push_back(candidate_mask);
      }
    }
    if (!candidate_masks.empty()) {
      attrs[SunMMIOCallAttrKey::kCandidateMasks] = std::move(candidate_masks);
    }
  } else if (callee == "tl.dma_copy") {
    ICHECK_EQ(op->args.size(), 4)
        << "tl.dma_copy expects src region, dst region, src_offset_byte, "
           "and sync_token_id";
    auto count_tiled_dims = [](const PrimExpr &region_expr) -> int {
      BufferRegion region = tl::NormalizeToBufferRegion(region_expr);
      int count = 0;
      for (const Range &range : region->region) {
        const auto *extent_imm = range->extent.as<IntImmNode>();
        ICHECK(extent_imm) << "tl.dma_copy region extent must be IntImm";
        if (extent_imm->value != 1) {
          ++count;
        }
      }
      return count;
    };

    int src_tiled_dims = count_tiled_dims(op->args[0]);
    int dst_tiled_dims = count_tiled_dims(op->args[1]);

    const auto *src_offset_imm = op->args[2].as<IntImmNode>();
    ICHECK(src_offset_imm)
        << "tl.dma_copy src_offset_byte must be a constant IntImm";
    int64_t src_offset_byte = static_cast<int64_t>(src_offset_imm->value);
    ICHECK_GE(src_offset_byte, 0)
        << "tl.dma_copy src_offset_byte must be non-negative";
    MarkVisitedNodeType(src_offset_imm->GetTypeKey());

    operands.reserve(2);

    ICHECK(TryConsumeSyncTokenId(op->args[3], &attrs))
        << "tl.dma_copy expects fourth argument to be tl.sync_token_id";

    operands.push_back(EmitRegionCall(op->args[0], src_offset_byte));
    operands.push_back(EmitRegionCall(op->args[1]));
  } else if (callee == "tl.sunmmio_layout_transform") {
    ICHECK_EQ(op->args.size(), 3)
        << "tl.sunmmio_layout_transform expects src region, dst region, and "
           "sync_token_id";
    struct LayoutTransformRegionInfo {
      int rank{0};
      int tiled_dims{0};
    };
    auto get_region_info =
        [](const PrimExpr &region_expr) {
          BufferRegion region = tl::NormalizeToBufferRegion(region_expr);
          LayoutTransformRegionInfo info;
          info.rank = static_cast<int>(region->region.size());
          for (const Range &range : region->region) {
            const auto *extent_imm = range->extent.as<IntImmNode>();
            ICHECK(extent_imm)
                << "tl.sunmmio_layout_transform region extent must be IntImm";
            if (extent_imm->value != 1) {
              ++info.tiled_dims;
            }
          }
          return info;
        };

    LayoutTransformRegionInfo src_info = get_region_info(op->args[0]);
    LayoutTransformRegionInfo dst_info = get_region_info(op->args[1]);
    auto is_singleton_1d_region = [](const LayoutTransformRegionInfo &info) {
      return info.rank == 1 && info.tiled_dims == 0;
    };
    bool is_2d_transform = src_info.tiled_dims == 2 && dst_info.tiled_dims == 2;
    bool is_singleton_1d_transform =
        is_singleton_1d_region(src_info) && is_singleton_1d_region(dst_info);
    ICHECK(is_2d_transform || is_singleton_1d_transform)
        << "tl.sunmmio_layout_transform expects source and destination "
           "regions to both have exactly 2 tiled dims, or to both be rank-1 "
           "singleton regions; got source rank="
        << src_info.rank << ", tiled dims=" << src_info.tiled_dims
        << ", destination rank=" << dst_info.rank
        << ", tiled dims=" << dst_info.tiled_dims;

    operands.reserve(2);
    operands.push_back(EmitRegionCall(op->args[0]));
    operands.push_back(EmitRegionCall(op->args[1]));

    ICHECK(TryConsumeSyncTokenId(op->args[2], &attrs))
        << "tl.sunmmio_layout_transform expects third argument to be "
           "tl.sync_token_id";
  } else if (callee == "tir.bitwise_and" || callee == "tir.bitwise_or" ||
             callee == "tir.bitwise_xor" || callee == "tir.shift_left" ||
             callee == "tir.shift_right") {
    ICHECK_EQ(op->args.size(), 2) << callee << " expects exactly two arguments";
    if (callee == "tir.bitwise_and") {
      return EmitBinary("and", op->args[0], op->args[1], op->dtype);
    }
    if (callee == "tir.bitwise_or") {
      return EmitBinary("or", op->args[0], op->args[1], op->dtype);
    }
    if (callee == "tir.bitwise_xor") {
      return EmitBinary("xor", op->args[0], op->args[1], op->dtype);
    }
    if (callee == "tir.shift_left") {
      return EmitBinary("shl", op->args[0], op->args[1], op->dtype);
    }
    return EmitBinary("shr", op->args[0], op->args[1], op->dtype);
  } else if (callee == "tl.broadcast_") {
    size_t non_token_args = op->args.size();
    bool has_sync_token =
        non_token_args > 0 && TryConsumeSyncTokenId(op->args.back(), &attrs);
    ICHECK(has_sync_token)
        << "tl.broadcast_ expects last argument to be tl.sync_token_id";
    --non_token_args;
    ICHECK(non_token_args == static_cast<size_t>(tl::kBroadcastArgCount) ||
           non_token_args == static_cast<size_t>(tl::kBroadcastArgCount + 1))
        << "tl.broadcast_ expects src region, dst region, direction, mask, "
           "src_offset_byte, optional src_core, and sync_token_id";

    const auto *direction_imm =
        op->args[tl::kBroadcastArgDirection].as<IntImmNode>();
    ICHECK(direction_imm)
        << "tl.broadcast_ direction must be a constant IntImm";
    int64_t direction = static_cast<int64_t>(direction_imm->value);
    ICHECK(direction == 0 || direction == 1)
        << "tl.broadcast_ MLIR lowering only supports direction 0 or 1, got "
        << direction;
    MarkVisitedNodeType(direction_imm->GetTypeKey());

    int64_t src_offset_byte = 0;
    const auto *src_offset_imm =
        op->args[tl::kBroadcastArgSrcOffsetByte].as<IntImmNode>();
    ICHECK(src_offset_imm)
        << "tl.broadcast_ src_offset_byte must be a constant IntImm";
    src_offset_byte = static_cast<int64_t>(src_offset_imm->value);
    ICHECK_GE(src_offset_byte, 0)
        << "tl.broadcast_ src_offset_byte must be non-negative";
    MarkVisitedNodeType(src_offset_imm->GetTypeKey());

    operands.reserve(4);
    operands.push_back(
        EmitRegionCall(op->args[tl::kBroadcastArgSrc], src_offset_byte));
    operands.push_back(EmitRegionCall(op->args[tl::kBroadcastArgDst]));
    operands.push_back(EvalExpr(op->args[tl::kBroadcastArgMask]));
    if (non_token_args == static_cast<size_t>(tl::kBroadcastArgCount + 1)) {
      operands.push_back(EvalExpr(op->args[tl::kBroadcastArgSrcCore]));
    }

    attrs[SunMMIOCallAttrKey::kDirection] =
        std::string(direction == 0 ? "row" : "col");
  } else if (callee == "tl.mma_sunmmio") {
    ICHECK_EQ(op->args.size(), 8) << "tl.mma_sunmmio expects A/B/C regions, "
                                     "three flag operands, acc_offset_byte, "
                                     "and sync_token_id";
    auto parse_bool_arg = [&](const PrimExpr &arg,
                              const char *arg_name) -> bool {
      const auto *imm = arg.as<IntImmNode>();
      ICHECK(imm) << arg_name << " must be a constant bool";
      ICHECK(imm->dtype.is_bool()) << arg_name << " must have bool dtype";
      return imm->value != 0;
    };

    const auto *acc_offset_imm = op->args[6].as<IntImmNode>();
    ICHECK(acc_offset_imm)
        << "tl.mma_sunmmio acc_offset_byte must be a constant IntImm";
    int64_t acc_offset_byte = static_cast<int64_t>(acc_offset_imm->value);
    ICHECK_GE(acc_offset_byte, 0)
        << "tl.mma_sunmmio acc_offset_byte must be non-negative";
    MarkVisitedNodeType(acc_offset_imm->GetTypeKey());

    operands.reserve(3);
    operands.push_back(EmitRegionCall(op->args[0]));
    operands.push_back(EmitRegionCall(op->args[1]));
    operands.push_back(EmitRegionCall(op->args[2], acc_offset_byte));

    attrs[SunMMIOCallAttrKey::kTransA] =
        parse_bool_arg(op->args[3], "tl.mma_sunmmio transA");
    attrs[SunMMIOCallAttrKey::kTransB] =
        parse_bool_arg(op->args[4], "tl.mma_sunmmio transB");
    attrs[SunMMIOCallAttrKey::kClearAccum] =
        parse_bool_arg(op->args[5], "tl.mma_sunmmio clearAccum");

    ICHECK(TryConsumeSyncTokenId(op->args[7], &attrs))
        << "tl.mma_sunmmio expects last argument to be tl.sync_token_id";
  } else {
    for (int i = 0, e = static_cast<int>(op->args.size()); i < e; ++i) {
      const PrimExpr &arg = op->args[i];
      if (TryConsumeSyncTokenId(arg, &attrs)) {
        continue;
      }
      if (const auto *s = arg.as<StringImmNode>()) {
        MarkVisitedNodeType("tir.StringImm");
        continue;
      }
      operands.push_back(EvalExpr(arg));
    }
  }
  DataType ret_dtype = CanonicalizeSuvmDType(op->dtype);
  SunMMIOType ret_ty = MapType(ret_dtype);
  std::string result_name = op->dtype.is_void() ? "" : NewValueName();
  return builder_->Call(result_name, callee, operands, attrs,
                        CallBucketName(bucket), ret_dtype, ret_ty);
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
