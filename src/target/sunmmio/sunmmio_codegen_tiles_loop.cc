#include "codegen_sunmmio.h"

#include "../../op/utils.h"
#include "../../transform/common/attr.h"
#include "sunmmio_mlir_builder.h"
#include "sunmmio_mlir_context.h"

#include <algorithm>
#include <iomanip>
#include <optional>

#include <tvm/ir/op.h>
#include <tvm/runtime/logging.h>

namespace tvm {
namespace codegen {

namespace {

using namespace tir;

struct TilesScopeInfo {
  const ForNode *root{nullptr};
  ffi::Array<PrimExpr> domain_shape;
  std::vector<const ForNode *> domain_loops;
  std::vector<const ForNode *> execution_loops;
  const ForNode *interior_axis0_loop{nullptr};
  const ForNode *interior_axis1_loop{nullptr};
  std::vector<int> execution_domain_axes;
  std::vector<int64_t> tile_shape;
  Stmt tile_block_body;
  PrimExpr tail_predicate;
  Stmt full_tile_body;
  Stmt tail_tile_body;
  Stmt full_tile_block_body;
  Stmt tail_tile_block_body;
  const ForNode *tail_interior_axis0_loop{nullptr};
  const ForNode *tail_interior_axis1_loop{nullptr};
  bool is_reduce_scope{false};
};

struct TileBlockState {
  const TilesScopeInfo *scope{nullptr};
  SunmmioMlirContext *mlir_ctx{nullptr};
  std::unordered_map<const VarNode *, SunMMIOValue> let_values;
  std::unordered_map<const BufferNode *, SunMMIOValue> register_tile_values;
  std::unordered_map<const BufferNode *, SunMMIOType> register_tile_types;
  std::unordered_map<const BufferNode *, int64_t> register_unsqueeze_axes;
  std::unordered_map<const BufferNode *, SunMMIOValue> local_tile_values;
  std::unordered_map<const BufferNode *, int64_t> local_unit_tile_axes;
  std::unordered_map<const BufferNode *, SunMMIOValue> tile_view_cache;
  std::unordered_map<const BufferNode *, SunMMIOValue> current_tile_values;
  std::optional<SunMMIOValue> tile_mask;
  const ForNode *interior_axis0_loop{nullptr};
  const ForNode *interior_axis1_loop{nullptr};
};

struct TileAccessInfo {
  Buffer buffer;
  int tile_rank{0};
  std::vector<int64_t> tile_shape;
  std::vector<int> tile_axes;
  std::vector<SunMMIOValue> partition_indices;
  std::vector<int64_t> tiled_dims;
  bool promoted_unit_tile_view{false};
  int64_t unsqueeze_axis{-1};
  bool requires_aligned_1d_load{false};
  int64_t aligned_load_bytes{0};
  int64_t aligned_load_elems{0};
  int64_t aligned_load_axis{-1};
  std::vector<int64_t> aligned_load_shape;
};

struct TiledIndexMatch {
  int64_t offset{0};
  bool uses_execution_index{false};
};

struct TailMaskInfo {
  SunMMIOValue valid_rows;
  SunMMIOValue valid_cols;
  SunMMIOValue row_tail_cond;
  SunMMIOValue col_tail_cond;
  SunMMIOType mask_type;
};

std::vector<int64_t>
ParseStaticIntArray(const ffi::Map<ffi::String, ffi::Any> &annotations,
                    const char *key) {
  auto it = annotations.find(key);
  ICHECK(it != annotations.end()) << "Missing tile annotation `" << key << "`";
  ffi::Array<PrimExpr> values = Downcast<ffi::Array<PrimExpr>>((*it).second);
  std::vector<int64_t> result;
  result.reserve(values.size());
  for (const PrimExpr &value : values) {
    const auto *imm = value.as<IntImmNode>();
    ICHECK(imm) << "Tile annotation `" << key << "` must be static IntImm";
    result.push_back(static_cast<int64_t>(imm->value));
  }
  return result;
}

std::vector<const ForNode *> CollectLinearForChain(const ForNode *root) {
  std::vector<const ForNode *> loops;
  const ForNode *current = root;
  while (current != nullptr) {
    loops.push_back(current);
    current = current->body.as<ForNode>();
  }
  return loops;
}

SunmmioMlirContext *
TryGetMlirContext(std::unique_ptr<SunMMIOBuilder> &builder) {
  auto *suvm_builder = dynamic_cast<SuvmSunmmioBuilder *>(builder.get());
  if (!suvm_builder) {
    return nullptr;
  }
  return &suvm_builder->Context();
}

SunMMIOType MakeTileType(DataType dtype, const std::vector<int64_t> &shape) {
  SunMMIOType type;
  type.kind = SunMMIOType::Kind::kTile;
  type.dtype = CanonicalizeSuvmDType(dtype).with_lanes(1);
  type.lanes = 1;
  for (int64_t dim : shape) {
    type.shape.push_back(IntImm(DataType::Int(32), dim));
  }
  return type;
}

SunMMIOType MakeTileViewType(DataType dtype,
                             const std::vector<int64_t> &shape) {
  SunMMIOType type;
  type.kind = SunMMIOType::Kind::kTileView;
  type.dtype = CanonicalizeSuvmDType(dtype).with_lanes(1);
  type.lanes = 1;
  for (int64_t dim : shape) {
    type.shape.push_back(IntImm(DataType::Int(32), dim));
  }
  return type;
}

bool IsTokenLikeTileStmt(const Stmt &stmt) {
  const auto *eval = stmt.as<EvaluateNode>();
  if (!eval) {
    return false;
  }
  const auto *call = eval->value.as<CallNode>();
  if (!call) {
    return false;
  }
  const auto *op_node = call->op.as<tvm::OpNode>();
  if (!op_node) {
    return false;
  }
  return op_node->name == "tl.wait_token" ||
         op_node->name == "tl.sync_token_id";
}

std::pair<const ForNode *, const ForNode *>
FindInteriorLoops(const Stmt &stmt) {
  if (const auto *loop = stmt.as<ForNode>()) {
    auto axis_it = loop->annotations.find(tl::attr::tile_interior_axis);
    if (axis_it != loop->annotations.end()) {
      int axis = Downcast<Integer>((*axis_it).second)->value;
      if (axis == 0) {
        const ForNode *axis1 = nullptr;
        if (const auto *inner = loop->body.as<ForNode>()) {
          auto inner_axis_it =
              inner->annotations.find(tl::attr::tile_interior_axis);
          if (inner_axis_it != inner->annotations.end() &&
              Downcast<Integer>((*inner_axis_it).second)->value == 1) {
            axis1 = inner;
          }
        }
        return {loop, axis1};
      }
    }
  }

  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const Stmt &s : seq->seq) {
      auto found = FindInteriorLoops(s);
      if (found.first != nullptr) {
        return found;
      }
    }
    return {nullptr, nullptr};
  }

  if (const auto *ifs = stmt.as<IfThenElseNode>()) {
    auto found = FindInteriorLoops(ifs->then_case);
    if (found.first != nullptr) {
      return found;
    }
    if (ifs->else_case.defined()) {
      return FindInteriorLoops(ifs->else_case.value());
    }
  }

  return {nullptr, nullptr};
}

bool IsTileLike(const SunMMIOValue &value) {
  return value.type.kind == SunMMIOType::Kind::kTile;
}

bool IsScalarLike(const SunMMIOValue &value) {
  return value.type.kind == SunMMIOType::Kind::kScalar ||
         value.type.kind == SunMMIOType::Kind::kIndex;
}

bool IsRsramScope(const std::string &scope) {
  return scope == "shared.rsram" || scope == "rsram";
}

bool IsReduceRegisterTempBuffer(const Buffer &buffer) {
  if (!buffer.defined() || !IsRsramScope(buffer.scope())) {
    return false;
  }
  const std::string name = buffer->name;
  return name.size() >= 4 && (name.rfind("_acc") == name.size() - 4 ||
                              name.rfind("_res") == name.size() - 4);
}

bool IsReduceLoopCarriedTempBuffer(const Buffer &buffer) {
  if (!IsReduceRegisterTempBuffer(buffer)) {
    return false;
  }
  const std::string name = buffer->name;
  return name.size() >= 4 && name.rfind("_acc") == name.size() - 4;
}

bool IsReduceLocalTempBuffer(const Buffer &buffer) {
  if (!IsReduceRegisterTempBuffer(buffer)) {
    return false;
  }
  const std::string name = buffer->name;
  return name.size() >= 4 && name.rfind("_res") == name.size() - 4;
}

bool ContainsVectorCoreInTileReduce(const Stmt &stmt) {
  if (!stmt.defined()) {
    return false;
  }
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    if (const auto *call = eval->value.as<CallNode>()) {
      const auto *op_node = call->op.as<OpNode>();
      return op_node && op_node->name == "tl.vector_core_in_tile_reduce";
    }
    return false;
  }
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const Stmt &s : seq->seq) {
      if (ContainsVectorCoreInTileReduce(s)) {
        return true;
      }
    }
    return false;
  }
  if (const auto *ifs = stmt.as<IfThenElseNode>()) {
    if (ContainsVectorCoreInTileReduce(ifs->then_case)) {
      return true;
    }
    return ifs->else_case.defined() &&
           ContainsVectorCoreInTileReduce(ifs->else_case.value());
  }
  if (const auto *loop = stmt.as<ForNode>()) {
    return ContainsVectorCoreInTileReduce(loop->body);
  }
  return false;
}

bool IsReduceLikeTileBody(const Stmt &stmt) {
  if (ContainsVectorCoreInTileReduce(stmt)) {
    return true;
  }
  const auto *seq = stmt.as<SeqStmtNode>();
  if (!seq) {
    return false;
  }
  bool has_guard = false;
  bool has_interior = false;
  for (const Stmt &s : seq->seq) {
    has_guard = has_guard || s.as<IfThenElseNode>() != nullptr;
    has_interior = has_interior || FindInteriorLoops(s).first != nullptr;
  }
  return has_guard && has_interior;
}

std::vector<int64_t> ExtractStaticShape(const SunMMIOType &type) {
  std::vector<int64_t> shape;
  shape.reserve(type.shape.size());
  for (const PrimExpr &dim : type.shape) {
    const auto *imm = dim.as<IntImmNode>();
    ICHECK(imm)
        << "Tiles lowering currently requires static tile/memtensor shape";
    shape.push_back(static_cast<int64_t>(imm->value));
  }
  return shape;
}

bool StaticShapesEqual(const SunMMIOType &a, const SunMMIOType &b) {
  return ExtractStaticShape(a) == ExtractStaticShape(b);
}

std::optional<int64_t> GetStaticLoopExtent(const ForNode *loop) {
  if (loop == nullptr) {
    return std::nullopt;
  }
  const auto *imm = loop->extent.as<IntImmNode>();
  if (!imm) {
    return std::nullopt;
  }
  return static_cast<int64_t>(imm->value);
}

std::optional<int> GetInteriorAxisAnnotation(const ForNode *loop) {
  if (loop == nullptr) {
    return std::nullopt;
  }
  auto axis_it = loop->annotations.find(tl::attr::tile_interior_axis);
  if (axis_it == loop->annotations.end()) {
    return std::nullopt;
  }
  return static_cast<int>(Downcast<Integer>((*axis_it).second)->value);
}

std::optional<TiledIndexMatch>
MatchTiledIndex(const PrimExpr &index, const Var &exec, const Var &interior,
                int64_t tile_extent, bool allow_standalone_interior) {
  if (index.same_as(interior)) {
    if (!allow_standalone_interior) {
      return std::nullopt;
    }
    return TiledIndexMatch{0, false};
  }

  std::vector<PrimExpr> terms;
  std::function<void(const PrimExpr &)> flatten_add =
      [&](const PrimExpr &expr) {
        if (const auto *add = expr.as<AddNode>()) {
          flatten_add(add->a);
          flatten_add(add->b);
          return;
        }
        terms.push_back(expr);
      };
  flatten_add(index);

  bool seen_interior = false;
  bool seen_exec = false;
  int64_t const_offset = 0;

  auto match_exec_mul = [&](const PrimExpr &expr) -> bool {
    const auto *mul = expr.as<MulNode>();
    if (!mul) {
      return false;
    }
    auto matches = [&](const PrimExpr &var_term,
                       const PrimExpr &imm_term) -> bool {
      if (!var_term.same_as(exec)) {
        return false;
      }
      const auto *imm = imm_term.as<IntImmNode>();
      return imm && static_cast<int64_t>(imm->value) == tile_extent;
    };
    return matches(mul->a, mul->b) || matches(mul->b, mul->a);
  };

  for (const PrimExpr &term : terms) {
    if (term.same_as(interior)) {
      if (seen_interior) {
        return std::nullopt;
      }
      seen_interior = true;
      continue;
    }
    if (match_exec_mul(term)) {
      if (seen_exec) {
        return std::nullopt;
      }
      seen_exec = true;
      continue;
    }
    if (const auto *imm = term.as<IntImmNode>()) {
      const_offset += static_cast<int64_t>(imm->value);
      continue;
    }
    return std::nullopt;
  }

  if (seen_interior && seen_exec && const_offset % tile_extent == 0) {
    return TiledIndexMatch{const_offset / tile_extent, true};
  }
  if (allow_standalone_interior && seen_interior && !seen_exec &&
      const_offset % tile_extent == 0) {
    return TiledIndexMatch{const_offset / tile_extent, false};
  }
  return std::nullopt;
}

} // namespace

bool CodeGenTileLangSunMMIO::TryLowerTilesScope(const tir::ForNode *op) {
  if (!op->annotations.count(tl::attr::kTileDomain)) {
    return false;
  }

  TilesScopeInfo scope;
  scope.root = op;
  scope.domain_shape =
      Downcast<ffi::Array<PrimExpr>>(op->annotations.at(tl::attr::kTileDomain));
  {
    std::vector<int64_t> parsed_axes = ParseStaticIntArray(
        op->annotations, tl::attr::tile_execution_domain_axes);
    scope.execution_domain_axes.reserve(parsed_axes.size());
    for (int64_t axis : parsed_axes) {
      scope.execution_domain_axes.push_back(static_cast<int>(axis));
    }
  }
  scope.tile_shape =
      ParseStaticIntArray(op->annotations, tl::attr::tile_tile_size);
  ICHECK_EQ(scope.execution_domain_axes.size(), scope.tile_shape.size())
      << "tile.execution_domain_axes and tile.tile_size rank mismatch";

  std::vector<const ForNode *> chain = CollectLinearForChain(op);
  ICHECK_GE(chain.size(), scope.domain_shape.size())
      << "Tiles scope loop chain shorter than tile.domain rank";
  for (size_t i = 0; i < scope.domain_shape.size(); ++i) {
    scope.domain_loops.push_back(chain[i]);
  }
  scope.execution_loops.assign(scope.execution_domain_axes.size(), nullptr);
  for (const ForNode *loop : scope.domain_loops) {
    auto axis_it = loop->annotations.find(tl::attr::tile_execution_axis);
    if (axis_it == loop->annotations.end()) {
      continue;
    }
    int exec_axis = Downcast<Integer>((*axis_it).second)->value;
    ICHECK_GE(exec_axis, 0);
    ICHECK_LT(static_cast<size_t>(exec_axis), scope.execution_loops.size())
        << "tile.execution_axis is out of range";
    scope.execution_loops[static_cast<size_t>(exec_axis)] = loop;
  }
  for (const ForNode *loop : scope.execution_loops) {
    ICHECK(loop != nullptr)
        << "Tiles scope is missing an execution loop for one tile axis";
  }

  Stmt tile_scope_stmt = scope.domain_loops.back()->body;
  scope.is_reduce_scope = IsReduceLikeTileBody(tile_scope_stmt);
  if (scope.is_reduce_scope) {
    auto loops = FindInteriorLoops(tile_scope_stmt);
    scope.interior_axis0_loop = loops.first;
    scope.interior_axis1_loop = loops.second;
    ICHECK(scope.interior_axis0_loop != nullptr)
        << "Reduce tiles scope is missing interior axis 0 loop";
    scope.tile_block_body = tile_scope_stmt;
  } else if (const auto *ifs = tile_scope_stmt.as<IfThenElseNode>()) {
    scope.tail_predicate = ifs->condition;
    scope.full_tile_body = ifs->then_case;
    scope.tail_tile_body =
        ifs->else_case.defined() ? ifs->else_case.value() : Stmt();
    auto full_loops = FindInteriorLoops(scope.full_tile_body);
    scope.interior_axis0_loop = full_loops.first;
    scope.interior_axis1_loop = full_loops.second;
    ICHECK(scope.interior_axis0_loop != nullptr)
        << "Tiles full-tile branch is missing interior axis 0 loop";
    scope.full_tile_block_body = scope.interior_axis1_loop != nullptr
                                     ? scope.interior_axis1_loop->body
                                     : scope.interior_axis0_loop->body;
    auto tail_loops = FindInteriorLoops(scope.tail_tile_body);
    scope.tail_interior_axis0_loop = tail_loops.first;
    scope.tail_interior_axis1_loop = tail_loops.second;
    ICHECK(scope.tail_interior_axis0_loop != nullptr)
        << "Tiles tail-tile branch is missing interior axis 0 loop";
    scope.tail_tile_block_body = scope.tail_interior_axis1_loop != nullptr
                                     ? scope.tail_interior_axis1_loop->body
                                     : scope.tail_interior_axis0_loop->body;
    scope.tile_block_body = scope.full_tile_block_body;
  } else {
    auto loops = FindInteriorLoops(tile_scope_stmt);
    scope.interior_axis0_loop = loops.first;
    scope.interior_axis1_loop = loops.second;
    ICHECK(scope.interior_axis0_loop != nullptr)
        << "Tiles scope is missing interior axis 0 loop";
    scope.tile_block_body = scope.interior_axis1_loop != nullptr
                                ? scope.interior_axis1_loop->body
                                : scope.interior_axis0_loop->body;
  }

  auto warn_token_stmt = [&](const Stmt &body) {
    if (!body.defined()) {
      return;
    }
    if (const auto *seq = body.as<SeqStmtNode>()) {
      for (const Stmt &stmt : seq->seq) {
        if (IsTokenLikeTileStmt(stmt)) {
          LOG(WARNING) << "Ignoring token-related Evaluate inside T.Tiles body "
                          "per current integration contract";
        }
      }
    } else if (IsTokenLikeTileStmt(body)) {
      LOG(WARNING) << "Ignoring token-related Evaluate inside T.Tiles body per "
                      "current integration contract";
    }
  };
  warn_token_stmt(scope.tile_block_body);

  SunmmioMlirContext *mlir_ctx = TryGetMlirContext(builder_);
  ICHECK(mlir_ctx != nullptr)
      << "Tiles lowering currently expects SuvmSunmmioBuilder";

  auto analyze_access = [&](const Buffer &buffer,
                            const ffi::Array<PrimExpr> &indices,
                            TileBlockState *state) -> TileAccessInfo {
    TileAccessInfo access;
    access.buffer = buffer;
    const BufferBinding &binding = LookupBuffer(buffer);

    std::vector<int64_t> memtensor_shape =
        ExtractStaticShape(binding.buffer_type);
    access.partition_indices.reserve(memtensor_shape.size());
    access.tiled_dims.clear();

    std::vector<int> logical_tile_axes(indices.size(), -1);
    std::vector<int64_t> logical_offsets(indices.size(), 0);
    std::vector<bool> logical_uses_execution_index(indices.size(), false);
    for (int dim = 0; dim < static_cast<int>(indices.size()); ++dim) {
      for (int axis = 0; axis < static_cast<int>(scope.execution_loops.size());
           ++axis) {
        const ForNode *exec_loop = scope.execution_loops[axis];
        if (exec_loop == nullptr) {
          continue;
        }

        std::vector<const ForNode *> candidate_interior_loops;
        auto push_candidate = [&](const ForNode *loop) {
          if (loop == nullptr) {
            return;
          }
          if (std::find(candidate_interior_loops.begin(),
                        candidate_interior_loops.end(),
                        loop) == candidate_interior_loops.end()) {
            candidate_interior_loops.push_back(loop);
          }
        };
        for (const ForNode *loop :
             {state->interior_axis0_loop, state->interior_axis1_loop}) {
          auto extent = GetStaticLoopExtent(loop);
          if (extent && *extent == scope.tile_shape[axis]) {
            push_candidate(loop);
          }
        }
        const ForNode *primary_loop =
            axis == 0 ? state->interior_axis0_loop : state->interior_axis1_loop;
        if (primary_loop != nullptr) {
          auto extent = GetStaticLoopExtent(primary_loop);
          if (!extent) {
            push_candidate(primary_loop);
          }
        }

        for (const ForNode *interior_loop : candidate_interior_loops) {
          auto match =
              MatchTiledIndex(indices[dim], exec_loop->loop_var,
                              interior_loop->loop_var, scope.tile_shape[axis],
                              /*allow_standalone_interior=*/true);
          if (match) {
            logical_tile_axes[dim] = axis;
            logical_offsets[dim] = match->offset;
            logical_uses_execution_index[dim] = match->uses_execution_index;
            break;
          }
        }
        if (logical_tile_axes[dim] >= 0) {
          break;
        }
      }
    }

    for (int dim = 0; dim < static_cast<int>(memtensor_shape.size()); ++dim) {
      if (dim < static_cast<int>(logical_tile_axes.size()) &&
          logical_tile_axes[dim] >= 0) {
        int axis = logical_tile_axes[dim];
        access.tiled_dims.push_back(dim);
        access.tile_shape.push_back(scope.tile_shape[axis]);
        access.tile_axes.push_back(axis);
        SunMMIOValue exec_index =
            logical_uses_execution_index[dim]
                ? EvalExpr(scope.execution_loops[axis]->loop_var)
                : builder_->ConstantInt(
                      NewValueName(), 0,
                      SunMMIOType{
                          SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
                      DataType::Int(32));
        if (logical_offsets[dim] != 0) {
          SunMMIOValue offset = builder_->ConstantInt(
              NewValueName(), logical_offsets[dim],
              SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
              DataType::Int(32));
          exec_index = logical_uses_execution_index[dim]
                           ? builder_->Binary(
                                 NewValueName(), BinaryOp::kAdd,
                                 ArithmeticFlavor::kIndex, exec_index, offset,
                                 SunMMIOType{SunMMIOType::Kind::kIndex,
                                             DataType::Int(32),
                                             1,
                                             {}},
                                 DataType::Int(32))
                           : offset;
        }
        access.partition_indices.push_back(exec_index);
      } else {
        if (dim < static_cast<int>(indices.size())) {
          access.partition_indices.push_back(EvalExpr(indices[dim]));
        } else {
          access.partition_indices.push_back(builder_->ConstantInt(
              NewValueName(), 0,
              SunMMIOType{SunMMIOType::Kind::kScalar, DataType::Int(32), 1, {}},
              DataType::Int(32)));
        }
      }
    }

    auto force_it = state->local_unit_tile_axes.find(buffer.get());
    if (force_it != state->local_unit_tile_axes.end() &&
        access.tile_shape.size() == 1 && access.tile_axes.size() == 1) {
      // Final in-tile reduce writeback consumes a local 2D unit tile, e.g.
      // !suvm.tile<1x32>.  The TIR target access is often rank-1 after the
      // reduced dimension disappears, so promote the target view to a matching
      // unit tile_view and keep load/compute/store entirely in 2D tile form.
      int unit_axis = static_cast<int>(force_it->second);
      int existing_axis = access.tile_axes[0];
      if (unit_axis != existing_axis) {
        auto is_tiled_dim = [&](int64_t dim) {
          return std::find(access.tiled_dims.begin(), access.tiled_dims.end(),
                           dim) != access.tiled_dims.end();
        };
        std::optional<int64_t> unit_dim;
        for (int64_t dim = 0;
             dim < static_cast<int64_t>(memtensor_shape.size()); ++dim) {
          if (!is_tiled_dim(dim)) {
            unit_dim = dim;
            break;
          }
        }
        if (unit_dim.has_value()) {
          int64_t existing_dim = access.tiled_dims[0];
          // Keep tile_view dimensions in memtensor/layout order.  For reduce
          // axis 1 this intentionally creates a row-major 1xN view instead of
          // a Nx1 view with tiled_dims=[data_dim, unit_dim]; RHS unit-vector
          // tiles are re-oriented later with squeeze/unsqueeze when needed.
          if (*unit_dim < existing_dim) {
            access.tiled_dims.insert(access.tiled_dims.begin(), *unit_dim);
            access.tile_axes.insert(access.tile_axes.begin(), unit_axis);
            access.tile_shape.insert(access.tile_shape.begin(), 1);
          } else {
            access.tiled_dims.push_back(*unit_dim);
            access.tile_axes.push_back(unit_axis);
            access.tile_shape.push_back(1);
          }
          access.promoted_unit_tile_view = true;
        }
      }
    }

    access.tile_rank = static_cast<int>(access.tile_shape.size());
    ICHECK(access.tile_rank == 1 || access.tile_rank == 2)
        << "Clean v4 tiles lowering currently only supports 1D or 2D tile "
           "accesses inside T.Tiles";
    if (access.tile_rank == 1) {
      ICHECK_EQ(access.tile_axes.size(), 1U);
      access.unsqueeze_axis = access.tile_axes[0] == 0 ? 1 : 0;
      if (IsRsramScope(binding.buffer_type.memory_scope)) {
        int64_t dtype_bytes =
            static_cast<int64_t>(CanonicalizeSuvmDType(buffer->dtype).bytes());
        ICHECK_GT(dtype_bytes, 0)
            << "Unexpected zero-sized dtype in Tiles lowering";
        ICHECK_EQ(64 % dtype_bytes, 0)
            << "64B alignment path requires dtype byte width to divide 64";
        access.aligned_load_bytes = 64;
        access.aligned_load_elems = 64 / dtype_bytes;
        int64_t tile_bytes = access.tile_shape[0] * dtype_bytes;
        access.requires_aligned_1d_load =
            tile_bytes < access.aligned_load_bytes;
        access.aligned_load_axis = access.unsqueeze_axis == 1 ? 0 : 1;
        access.aligned_load_shape =
            access.unsqueeze_axis == 1
                ? std::vector<int64_t>{access.aligned_load_elems, 1}
                : std::vector<int64_t>{1, access.aligned_load_elems};
      }
    }
    return access;
  };

  auto get_or_create_tile_view = [&](const TileAccessInfo &access,
                                     TileBlockState *state) -> SunMMIOValue {
    bool bypass_cache = access.promoted_unit_tile_view;
    if (!bypass_cache) {
      auto it = state->tile_view_cache.find(access.buffer.get());
      if (it != state->tile_view_cache.end()) {
        return it->second;
      }
    }
    const BufferBinding &binding = LookupBuffer(access.buffer);
    SunMMIOValue memtensor{
        CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1),
        binding.handle, binding.buffer_type};
    SunMMIOType view_type =
        MakeTileViewType(access.buffer->dtype, access.tile_shape);
    SunMMIOValue view = builder_->GetPartitionedTileView(
        NewValueName(), memtensor, access.partition_indices, access.tiled_dims,
        view_type, CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1));
    if (!bypass_cache) {
      state->tile_view_cache.emplace(access.buffer.get(), view);
    }
    return view;
  };

  auto make_tile_view_from_region = [&](const BufferRegion &region,
                                        TileBlockState *state) -> SunMMIOValue {
    (void)state;
    const Buffer &buffer = region->buffer;
    const BufferBinding &binding = LookupBuffer(buffer);
    SunMMIOValue memtensor{CanonicalizeSuvmDType(buffer->dtype).with_lanes(1),
                           binding.handle, binding.buffer_type};

    std::vector<SunMMIOValue> indices;
    indices.reserve(region->region.size());
    std::vector<int64_t> tiled_dims;
    std::vector<int64_t> tile_shape;
    for (int64_t dim = 0; dim < static_cast<int64_t>(region->region.size());
         ++dim) {
      const Range &range = region->region[dim];
      indices.push_back(EnsureIndex(EvalExpr(range->min)));
      const auto *extent_imm = range->extent.as<IntImmNode>();
      ICHECK(extent_imm) << "Tile region extent must be IntImm";
      if (extent_imm->value != 1) {
        tiled_dims.push_back(dim);
        tile_shape.push_back(static_cast<int64_t>(extent_imm->value));
      }
    }
    if (tile_shape.empty()) {
      ICHECK(!region->region.empty())
          << "Tile region lowering expects at least one region dimension";
      // A fully scalar region is still represented as a one-element tile so it
      // can be the destination of tile.reduce from a 1D source tile.
      tiled_dims.push_back(0);
      tile_shape.push_back(1);
    }
    ICHECK(tile_shape.size() == 1 || tile_shape.size() == 2)
        << "Tile region lowering expects one or two non-unit extents";

    SunMMIOType view_type = MakeTileViewType(buffer->dtype, tile_shape);
    return builder_->GetPartitionedTileView(
        NewValueName(), memtensor, indices, tiled_dims, view_type,
        CanonicalizeSuvmDType(buffer->dtype).with_lanes(1));
  };

  auto make_tile_type_from_region = [&](const BufferRegion &region) {
    const Buffer &buffer = region->buffer;
    std::vector<int64_t> tile_shape;
    tile_shape.reserve(region->region.size());
    for (const Range &range : region->region) {
      const auto *extent_imm = range->extent.as<IntImmNode>();
      ICHECK(extent_imm) << "Register tile region extent must be IntImm";
      if (extent_imm->value != 1) {
        tile_shape.push_back(static_cast<int64_t>(extent_imm->value));
      }
    }
    if (tile_shape.empty()) {
      tile_shape.push_back(1);
    }
    ICHECK(tile_shape.size() == 1 || tile_shape.size() == 2)
        << "Register tile region expects one or two non-unit extents";
    return MakeTileType(buffer->dtype, tile_shape);
  };

  auto make_register_value_name = [&](const Buffer &buffer) {
    return "__tile_reg_" + buffer->name;
  };

  auto make_register_tile_value = [&](const Buffer &buffer,
                                      const SunMMIOType &type) {
    return SunMMIOValue{type.dtype, make_register_value_name(buffer), type};
  };

  auto make_local_value_name = [&](const Buffer &buffer) {
    return "__tile_local_" + buffer->name;
  };

  auto note_register_unsqueeze_axis = [&](TileBlockState *state,
                                          const Buffer &buffer, int64_t axis) {
    if (!IsReduceRegisterTempBuffer(buffer)) {
      return;
    }
    // vector_core_in_tile_reduce squeezes the reduced axis when its destination
    // is a 1D register tile. Later BufferLoad users must insert the inverse
    // unsqueeze on the same axis to recover the expected 2D tile shape.
    state->register_unsqueeze_axes[buffer.get()] = axis;
  };

  auto collect_register_live_out_values = [&](TileBlockState *state) {
    std::vector<SunMMIOValue> live_out_values;
    live_out_values.reserve(state->register_tile_values.size());
    for (const auto &kv : state->register_tile_values) {
      live_out_values.push_back(kv.second);
    }
    std::sort(live_out_values.begin(), live_out_values.end(),
              [](const SunMMIOValue &lhs, const SunMMIOValue &rhs) {
                return lhs.value < rhs.value;
              });
    return live_out_values;
  };

  auto discover_reduce_register_temps = [&](const Stmt &stmt,
                                            TileBlockState *state) {
    std::function<void(const Stmt &)> visit_stmt;
    std::function<void(const PrimExpr &)> visit_expr;
    auto register_buffer = [&](const Buffer &buffer,
                               const ffi::Array<PrimExpr> &indices) {
      if (!IsReduceLoopCarriedTempBuffer(buffer) ||
          state->register_tile_types.count(buffer.get())) {
        return;
      }
      TileAccessInfo access = analyze_access(buffer, indices, state);
      SunMMIOType tile_type = MakeTileType(buffer->dtype, access.tile_shape);
      state->register_tile_types[buffer.get()] = tile_type;
      state->register_tile_values[buffer.get()] =
          make_register_tile_value(buffer, tile_type);
      if (access.tile_rank == 1 && access.unsqueeze_axis >= 0 &&
          !state->register_unsqueeze_axes.count(buffer.get())) {
        state->register_unsqueeze_axes[buffer.get()] = access.unsqueeze_axis;
      }
    };

    visit_expr = [&](const PrimExpr &expr) {
      if (!expr.defined()) {
        return;
      }
      if (const auto *load = expr.as<BufferLoadNode>()) {
        register_buffer(load->buffer, load->indices);
        return;
      }
      if (const auto *call = expr.as<CallNode>()) {
        const auto *op_node = call->op.as<OpNode>();
        if (op_node && op_node->name == "tl.vector_core_in_tile_reduce" &&
            call->args.size() >= 3) {
          BufferRegion dst_region = tl::NormalizeToBufferRegion(call->args[1]);
          BufferRegion src_region = tl::NormalizeToBufferRegion(call->args[2]);
          if (call->args.size() >= 4) {
            const auto *axis_imm = call->args[3].as<IntImmNode>();
            ICHECK(axis_imm)
                << "tl.vector_core_in_tile_reduce axis must be IntImm";
            note_register_unsqueeze_axis(state, dst_region->buffer,
                                         static_cast<int64_t>(axis_imm->value));
          }
          if (IsReduceLoopCarriedTempBuffer(src_region->buffer) &&
              !state->register_tile_types.count(src_region->buffer.get())) {
            SunMMIOType src_type = make_tile_type_from_region(src_region);
            state->register_tile_types[src_region->buffer.get()] = src_type;
            state->register_tile_values[src_region->buffer.get()] =
                make_register_tile_value(src_region->buffer, src_type);
          }
          if (IsReduceLoopCarriedTempBuffer(dst_region->buffer) &&
              !state->register_tile_types.count(dst_region->buffer.get())) {
            SunMMIOType dst_type = make_tile_type_from_region(dst_region);
            state->register_tile_types[dst_region->buffer.get()] = dst_type;
            state->register_tile_values[dst_region->buffer.get()] =
                make_register_tile_value(dst_region->buffer, dst_type);
          }
          return;
        }
      }
      tir::PostOrderVisit(expr, [&](const ObjectRef &obj) {
        if (const auto *load = obj.as<BufferLoadNode>()) {
          register_buffer(load->buffer, load->indices);
        }
      });
    };

    visit_stmt = [&](const Stmt &s) {
      if (!s.defined()) {
        return;
      }
      if (const auto *seq = s.as<SeqStmtNode>()) {
        for (const Stmt &child : seq->seq) {
          visit_stmt(child);
        }
        return;
      }
      if (const auto *ifs = s.as<IfThenElseNode>()) {
        visit_expr(ifs->condition);
        visit_stmt(ifs->then_case);
        if (ifs->else_case.defined()) {
          visit_stmt(ifs->else_case.value());
        }
        return;
      }
      if (const auto *loop = s.as<ForNode>()) {
        visit_expr(loop->min);
        visit_expr(loop->extent);
        visit_stmt(loop->body);
        return;
      }
      if (const auto *store = s.as<BufferStoreNode>()) {
        register_buffer(store->buffer, store->indices);
        visit_expr(store->value);
        return;
      }
      if (const auto *eval = s.as<EvaluateNode>()) {
        visit_expr(eval->value);
      }
    };

    visit_stmt(stmt);
  };

  auto initialize_reduce_register_temps = [&](TileBlockState *state) {
    for (const auto &kv : state->register_tile_types) {
      const BufferNode *buffer_node = kv.first;
      const SunMMIOType &tile_type = kv.second;
      SunMMIOType scalar_type{
          SunMMIOType::Kind::kScalar, tile_type.dtype, 1, {}};
      SunMMIOValue zero =
          tile_type.dtype.is_float() || tile_type.dtype.is_bfloat16()
              ? builder_->ConstantFloat(NewValueName(), "0.0", scalar_type,
                                        tile_type.dtype)
              : builder_->ConstantInt(NewValueName(), 0, scalar_type,
                                      tile_type.dtype);
      SunMMIOValue filled =
          builder_->TileFill(NewValueName(), zero, tile_type, tile_type.dtype);
      state->register_tile_values[buffer_node] = builder_->BindValueAlias(
          make_register_value_name(ffi::GetRef<Buffer>(buffer_node)), filled);
    }
  };

  std::function<SunMMIOValue(const PrimExpr &, TileBlockState *,
                             std::optional<DataType>)>
      lower_expr;
  std::function<void(const Stmt &, TileBlockState *)> lower_stmt;
  std::function<void(const Stmt &, TileBlockState *)> lower_reduce_stmt;

  auto find_local_unit_axis_in_expr =
      [&](const PrimExpr &expr,
          TileBlockState *state) -> std::optional<int64_t> {
    std::optional<int64_t> axis;
    if (!expr.defined()) {
      return axis;
    }
    tir::PostOrderVisit(expr, [&](const ObjectRef &obj) {
      if (axis.has_value()) {
        return;
      }
      const auto *load = obj.as<BufferLoadNode>();
      if (!load) {
        return;
      }
      if (!state->local_tile_values.count(load->buffer.get())) {
        return;
      }
      auto axis_it = state->local_unit_tile_axes.find(load->buffer.get());
      if (axis_it != state->local_unit_tile_axes.end()) {
        axis = axis_it->second;
      }
    });
    return axis;
  };

  auto choose_result_dtype = [&](DataType expr_dtype,
                                 std::optional<DataType> preferred_dtype) {
    DataType dtype = preferred_dtype.value_or(expr_dtype);
    return CanonicalizeSuvmDType(dtype).with_lanes(1);
  };

  auto is_float_like_dtype = [](DataType dtype) {
    return dtype.is_float() || dtype.is_bfloat16();
  };

  auto arithmetic_flavor_for_dtype = [](DataType dtype) {
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
  };

  auto is_tile_compare_operand = [](const SunMMIOValue &lhs,
                                    const SunMMIOValue &rhs) {
    return IsTileLike(lhs) || IsTileLike(rhs);
  };

  auto supports_mixed_precision_binary = [](BinaryOp op) {
    // The SUVM dialect currently allows mixed element types for tile.mulf.
    // Other tile float binary ops carry AllElementTypesMatch and must be kept
    // type-homogeneous before hitting the verifier.
    return op == BinaryOp::kMul;
  };

  auto supports_mixed_precision_unary = [](TileUnaryOp op) {
    // These vfpwln-family ops may choose an output precision independently of
    // the input precision.  abs/neg and f32-only rounding ops do not.
    return op == TileUnaryOp::kExp || op == TileUnaryOp::kLn ||
           op == TileUnaryOp::kRecip || op == TileUnaryOp::kRsqrt;
  };

  auto cast_value_to_dtype = [&](const SunMMIOValue &value, DataType dtype) {
    DataType dst_dtype = CanonicalizeSuvmDType(dtype).with_lanes(1);
    if (value.dtype == dst_dtype) {
      return value;
    }
    if (IsTileLike(value)) {
      SunMMIOType dst_type =
          MakeTileType(dst_dtype, ExtractStaticShape(value.type));
      return builder_->Cast(NewValueName(), value, dst_type, dst_dtype);
    }
    SunMMIOType dst_type{SunMMIOType::Kind::kScalar, dst_dtype, 1, {}};
    return builder_->Cast(NewValueName(), value, dst_type, dst_dtype);
  };

  auto unit_axis_for_2d_shape =
      [](const std::vector<int64_t> &shape) -> std::optional<int64_t> {
    if (shape.size() != 2) {
      return std::nullopt;
    }
    if (shape[0] == 1) {
      return 0;
    }
    if (shape[1] == 1) {
      return 1;
    }
    return std::nullopt;
  };

  auto unit_vector_extent = [](const std::vector<int64_t> &shape,
                               int64_t unit_axis) -> int64_t {
    return unit_axis == 0 ? shape[1] : shape[0];
  };

  auto reorient_unit_tile_to_shape =
      [&](const SunMMIOValue &value,
          const std::vector<int64_t> &dst_shape) -> SunMMIOValue {
    if (!IsTileLike(value)) {
      return value;
    }
    std::vector<int64_t> src_shape = ExtractStaticShape(value.type);
    if (src_shape == dst_shape) {
      return value;
    }

    if (src_shape.size() == 1 && dst_shape.size() == 2) {
      std::optional<int64_t> dst_unit_axis = unit_axis_for_2d_shape(dst_shape);
      if (dst_unit_axis.has_value() &&
          src_shape[0] == unit_vector_extent(dst_shape, *dst_unit_axis)) {
        SunMMIOType dst_type = MakeTileType(value.dtype, dst_shape);
        return builder_->TileUnsqueeze(NewValueName(), value, dst_type,
                                       *dst_unit_axis, value.dtype);
      }
      return value;
    }

    if (src_shape.size() == 2 && dst_shape.size() == 1) {
      std::optional<int64_t> src_unit_axis = unit_axis_for_2d_shape(src_shape);
      if (src_unit_axis.has_value() &&
          unit_vector_extent(src_shape, *src_unit_axis) == dst_shape[0]) {
        SunMMIOType dst_type = MakeTileType(value.dtype, dst_shape);
        return builder_->TileSqueeze(NewValueName(), value, dst_type,
                                     *src_unit_axis, value.dtype);
      }
      return value;
    }

    if (src_shape.size() == 2 && dst_shape.size() == 2) {
      std::optional<int64_t> src_unit_axis = unit_axis_for_2d_shape(src_shape);
      std::optional<int64_t> dst_unit_axis = unit_axis_for_2d_shape(dst_shape);
      if (src_unit_axis.has_value()) {
        bool can_broadcast = true;
        for (size_t i = 0; i < src_shape.size(); ++i) {
          if (src_shape[i] != dst_shape[i] && src_shape[i] != 1) {
            can_broadcast = false;
            break;
          }
        }
        if (can_broadcast) {
          SunMMIOType dst_type = MakeTileType(value.dtype, dst_shape);
          return builder_->TileBroadcast(NewValueName(), value, dst_type,
                                         value.dtype);
        }
      }
      if (src_unit_axis.has_value() && dst_unit_axis.has_value() &&
          unit_vector_extent(src_shape, *src_unit_axis) ==
              unit_vector_extent(dst_shape, *dst_unit_axis)) {
        std::vector<int64_t> squeezed_shape{
            unit_vector_extent(src_shape, *src_unit_axis)};
        SunMMIOType squeezed_type = MakeTileType(value.dtype, squeezed_shape);
        SunMMIOValue squeezed = builder_->TileSqueeze(
            NewValueName(), value, squeezed_type, *src_unit_axis, value.dtype);
        SunMMIOType dst_type = MakeTileType(value.dtype, dst_shape);
        return builder_->TileUnsqueeze(NewValueName(), squeezed, dst_type,
                                       *dst_unit_axis, value.dtype);
      }
    }
    return value;
  };

  auto normalize_for_store = [&](const TileAccessInfo &access,
                                 const SunMMIOValue &value) -> SunMMIOValue {
    DataType dst_dtype =
        CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1);
    if (value.type.kind == SunMMIOType::Kind::kTile) {
      SunMMIOValue tile = value;
      if (access.tile_rank == 1 && value.type.shape.size() == 2) {
        SunMMIOType squeezed_type =
            MakeTileType(access.buffer->dtype, access.tile_shape);
        tile = builder_->TileSqueeze(NewValueName(), tile, squeezed_type,
                                     access.unsqueeze_axis, dst_dtype);
      }
      tile = reorient_unit_tile_to_shape(tile, access.tile_shape);
      SunMMIOType dst_tile_type =
          access.tile_rank == 1
              ? MakeTileType(access.buffer->dtype, access.tile_shape)
              : MakeTileType(access.buffer->dtype, access.tile_shape);
      if (tile.dtype == dst_dtype &&
          StaticShapesEqual(tile.type, dst_tile_type)) {
        return tile;
      }
      return builder_->Cast(NewValueName(), tile, dst_tile_type, dst_dtype);
    }
    ICHECK(IsScalarLike(value))
        << "Tiles store normalization only supports scalar or tile values";
    SunMMIOType dst_tile_type =
        MakeTileType(access.buffer->dtype, access.tile_shape);
    SunMMIOValue scalar = value;
    if (scalar.type.kind != SunMMIOType::Kind::kScalar ||
        scalar.dtype != dst_dtype) {
      SunMMIOType scalar_type{SunMMIOType::Kind::kScalar, dst_dtype, 1, {}};
      scalar = builder_->Cast(NewValueName(), scalar, scalar_type, dst_dtype);
    }
    return builder_->TileFill(NewValueName(), scalar, dst_tile_type, dst_dtype);
  };

  auto maybe_unsqueeze_tile = [&](const SunMMIOValue &value,
                                  const TileAccessInfo &access) {
    if (!IsTileLike(value) || access.tile_rank != 1) {
      return value;
    }
    if (value.type.shape.size() != 1) {
      return value;
    }
    if (access.unsqueeze_axis < 0) {
      return value;
    }
    ICHECK(access.unsqueeze_axis == 0 || access.unsqueeze_axis == 1)
        << "1D tile access is missing unsqueeze axis";
    // Unsqueeze only raises rank: it does not perform broadcast. So a 1D tile
    // of shape [N] becomes either [N, 1] or [1, N], depending on the chosen
    // axis.
    std::vector<int64_t> unsqueezed_shape =
        access.unsqueeze_axis == 1
            ? std::vector<int64_t>{access.tile_shape[0], 1}
            : std::vector<int64_t>{1, access.tile_shape[0]};
    SunMMIOType tile_type = MakeTileType(value.dtype, unsqueezed_shape);
    return builder_->TileUnsqueeze(NewValueName(), value, tile_type,
                                   access.unsqueeze_axis, tile_type.dtype);
  };

  auto make_index_const = [&](int64_t value) {
    return builder_->ConstantInt(
        NewValueName(), value,
        SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
        DataType::Int(32));
  };

  auto get_const_index_value =
      [&](const SunMMIOValue &value) -> std::optional<int64_t> {
    mlir::Value mlir_value = mlir_ctx->LookupMLIRValue(value.value);
    if (!mlir_value) {
      return std::nullopt;
    }
    if (auto cst = mlir::getConstantIntValue(mlir_value)) {
      return static_cast<int64_t>(*cst);
    }
    return std::nullopt;
  };

  auto add_index = [&](const SunMMIOValue &lhs, const SunMMIOValue &rhs) {
    auto lhs_cst = get_const_index_value(lhs);
    auto rhs_cst = get_const_index_value(rhs);
    if (lhs_cst.has_value() && rhs_cst.has_value()) {
      return make_index_const(*lhs_cst + *rhs_cst);
    }
    return builder_->Binary(
        NewValueName(), BinaryOp::kAdd, ArithmeticFlavor::kIndex, lhs, rhs,
        SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
        DataType::Int(32));
  };

  auto mul_index = [&](const SunMMIOValue &lhs, const SunMMIOValue &rhs) {
    auto lhs_cst = get_const_index_value(lhs);
    auto rhs_cst = get_const_index_value(rhs);
    if (lhs_cst.has_value() && rhs_cst.has_value()) {
      return make_index_const(*lhs_cst * *rhs_cst);
    }
    return builder_->Binary(
        NewValueName(), BinaryOp::kMul, ArithmeticFlavor::kIndex, lhs, rhs,
        SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
        DataType::Int(32));
  };

  auto div_index = [&](const SunMMIOValue &lhs, const SunMMIOValue &rhs) {
    auto lhs_cst = get_const_index_value(lhs);
    auto rhs_cst = get_const_index_value(rhs);
    if (lhs_cst.has_value() && rhs_cst.has_value()) {
      ICHECK_NE(*rhs_cst, 0) << "index division by zero in aligned 1D lowering";
      return make_index_const(*lhs_cst / *rhs_cst);
    }
    return builder_->Binary(
        NewValueName(), BinaryOp::kDiv, ArithmeticFlavor::kIndex, lhs, rhs,
        SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
        DataType::Int(32));
  };

  auto mod_index = [&](const SunMMIOValue &lhs, const SunMMIOValue &rhs) {
    auto lhs_cst = get_const_index_value(lhs);
    auto rhs_cst = get_const_index_value(rhs);
    if (lhs_cst.has_value() && rhs_cst.has_value()) {
      ICHECK_NE(*rhs_cst, 0) << "index modulo by zero in aligned 1D lowering";
      return make_index_const(*lhs_cst % *rhs_cst);
    }
    return builder_->Binary(
        NewValueName(), BinaryOp::kMod, ArithmeticFlavor::kIndex, lhs, rhs,
        SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
        DataType::Int(32));
  };

  auto load_aligned_1d_tile = [&](const TileAccessInfo &access,
                                  TileBlockState *state) -> SunMMIOValue {
    ICHECK(access.requires_aligned_1d_load);
    ICHECK_EQ(access.tiled_dims.size(), 1U)
        << "64B-aligned 1D tile load expects exactly one tiled dimension";

    int64_t dtype_bytes = static_cast<int64_t>(
        CanonicalizeSuvmDType(access.buffer->dtype).bytes());
    int64_t tiled_dim = access.tiled_dims[0];
    ICHECK_LT(tiled_dim, static_cast<int64_t>(access.partition_indices.size()));
    SunMMIOValue tile_index =
        EnsureIndex(access.partition_indices[static_cast<size_t>(tiled_dim)]);
    SunMMIOValue tile_extent = make_index_const(access.tile_shape[0]);
    SunMMIOValue elem_size = make_index_const(dtype_bytes);
    SunMMIOValue aligned_bytes = make_index_const(access.aligned_load_bytes);
    SunMMIOValue aligned_elems = make_index_const(access.aligned_load_elems);

    SunMMIOValue base_elem = mul_index(tile_index, tile_extent);
    SunMMIOValue base_bytes = mul_index(base_elem, elem_size);
    SunMMIOValue region_index = div_index(base_bytes, aligned_bytes);
    SunMMIOValue offset_bytes = mod_index(base_bytes, aligned_bytes);
    SunMMIOValue offset_elems = div_index(offset_bytes, elem_size);

    const BufferBinding &binding = LookupBuffer(access.buffer);
    SunMMIOValue memtensor{
        CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1),
        binding.handle, binding.buffer_type};

    SunMMIOType aligned_view_type =
        MakeTileViewType(access.buffer->dtype, {access.aligned_load_elems});
    std::vector<SunMMIOValue> aligned_partition_indices =
        access.partition_indices;
    aligned_partition_indices[static_cast<size_t>(tiled_dim)] = region_index;
    SunMMIOValue aligned_view = builder_->GetPartitionedTileView(
        NewValueName(), memtensor, aligned_partition_indices, {tiled_dim},
        aligned_view_type,
        CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1));
    SunMMIOType aligned_tile_type =
        MakeTileType(access.buffer->dtype, {access.aligned_load_elems});
    SunMMIOValue aligned_tile = builder_->TileLoad(
        NewValueName(), aligned_view, aligned_tile_type, std::nullopt,
        std::nullopt,
        CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1));
    SunMMIOValue aligned_2d_tile = builder_->TileUnsqueeze(
        NewValueName(), aligned_tile,
        MakeTileType(access.buffer->dtype, access.aligned_load_shape),
        access.unsqueeze_axis,
        CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1));

    std::vector<SunMMIOValue> slice_offsets;
    slice_offsets.reserve(2);
    if (access.aligned_load_axis == 0) {
      slice_offsets.push_back(offset_elems);
      slice_offsets.push_back(make_index_const(0));
    } else {
      slice_offsets.push_back(make_index_const(0));
      slice_offsets.push_back(offset_elems);
    }

    SunMMIOType sliced_tile_type =
        MakeTileType(access.buffer->dtype,
                     access.unsqueeze_axis == 1
                         ? std::vector<int64_t>{access.tile_shape[0], 1}
                         : std::vector<int64_t>{1, access.tile_shape[0]});
    SunMMIOValue sliced_tile = builder_->TileSlice(
        NewValueName(), aligned_2d_tile, slice_offsets, sliced_tile_type,
        CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1));
    return sliced_tile;
  };

  auto store_aligned_1d_tile = [&](const TileAccessInfo &access,
                                   const SunMMIOValue &value,
                                   TileBlockState *state) {
    ICHECK(access.requires_aligned_1d_load);
    ICHECK_EQ(access.tiled_dims.size(), 1U)
        << "64B-aligned 1D tile store expects exactly one tiled dimension";

    int64_t dtype_bytes = static_cast<int64_t>(
        CanonicalizeSuvmDType(access.buffer->dtype).bytes());
    int64_t tiled_dim = access.tiled_dims[0];
    ICHECK_LT(tiled_dim, static_cast<int64_t>(access.partition_indices.size()));
    SunMMIOValue tile_index =
        EnsureIndex(access.partition_indices[static_cast<size_t>(tiled_dim)]);
    SunMMIOValue tile_extent = make_index_const(access.tile_shape[0]);
    SunMMIOValue elem_size = make_index_const(dtype_bytes);
    SunMMIOValue aligned_bytes = make_index_const(access.aligned_load_bytes);

    SunMMIOValue base_elem = mul_index(tile_index, tile_extent);
    SunMMIOValue base_bytes = mul_index(base_elem, elem_size);
    SunMMIOValue region_index = div_index(base_bytes, aligned_bytes);
    SunMMIOValue offset_bytes = mod_index(base_bytes, aligned_bytes);
    SunMMIOValue offset_elems = div_index(offset_bytes, elem_size);

    const BufferBinding &binding = LookupBuffer(access.buffer);
    SunMMIOValue memtensor{
        CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1),
        binding.handle, binding.buffer_type};

    std::vector<SunMMIOValue> aligned_partition_indices =
        access.partition_indices;
    aligned_partition_indices[static_cast<size_t>(tiled_dim)] = region_index;

    SunMMIOType aligned_view_type =
        MakeTileViewType(access.buffer->dtype, {access.aligned_load_elems});
    SunMMIOValue aligned_view = builder_->GetPartitionedTileView(
        NewValueName(), memtensor, aligned_partition_indices, {tiled_dim},
        aligned_view_type,
        CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1));
    SunMMIOType aligned_tile_type =
        MakeTileType(access.buffer->dtype, {access.aligned_load_elems});
    SunMMIOValue aligned_tile = builder_->TileLoad(
        NewValueName(), aligned_view, aligned_tile_type, std::nullopt,
        std::nullopt,
        CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1));
    SunMMIOType aligned_2d_type =
        MakeTileType(access.buffer->dtype, access.aligned_load_shape);
    SunMMIOValue aligned_2d_tile = builder_->TileUnsqueeze(
        NewValueName(), aligned_tile, aligned_2d_type, access.unsqueeze_axis,
        CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1));

    SunMMIOValue src_slice = value;
    if (src_slice.type.shape.size() == 1) {
      SunMMIOType slice_2d_type =
          MakeTileType(access.buffer->dtype,
                       access.unsqueeze_axis == 1
                           ? std::vector<int64_t>{access.tile_shape[0], 1}
                           : std::vector<int64_t>{1, access.tile_shape[0]});
      src_slice =
          builder_->TileUnsqueeze(NewValueName(), src_slice, slice_2d_type,
                                  access.unsqueeze_axis, slice_2d_type.dtype);
    } else {
      src_slice = reorient_unit_tile_to_shape(
          src_slice, access.unsqueeze_axis == 1
                         ? std::vector<int64_t>{access.tile_shape[0], 1}
                         : std::vector<int64_t>{1, access.tile_shape[0]});
    }

    std::vector<SunMMIOValue> slice_offsets;
    slice_offsets.reserve(2);
    if (access.aligned_load_axis == 0) {
      slice_offsets.push_back(offset_elems);
      slice_offsets.push_back(make_index_const(0));
    } else {
      slice_offsets.push_back(make_index_const(0));
      slice_offsets.push_back(offset_elems);
    }

    SunMMIOValue merged_tile = builder_->TileInsertSlice(
        NewValueName(), aligned_2d_tile, src_slice, slice_offsets,
        aligned_2d_type,
        CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1));

    SunMMIOType aligned_store_view_type =
        MakeTileViewType(access.buffer->dtype, access.aligned_load_shape);
    SunMMIOValue aligned_store_view = builder_->GetPartitionedTileView(
        NewValueName(), memtensor, aligned_partition_indices, {tiled_dim},
        aligned_store_view_type,
        CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1));
    builder_->TileStore(merged_tile, aligned_store_view, std::nullopt);
  };

  auto build_tail_mask_info = [&](TileBlockState *state) -> TailMaskInfo {
    (void)state;
    ICHECK(scope.tail_predicate.defined());

    auto make_index_const = [&](int64_t value) {
      return builder_->ConstantInt(
          NewValueName(), value,
          SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
          DataType::Int(32));
    };

    auto sub_index = [&](const SunMMIOValue &lhs, const SunMMIOValue &rhs) {
      return builder_->Binary(
          NewValueName(), BinaryOp::kSub, ArithmeticFlavor::kIndex, lhs, rhs,
          SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
          DataType::Int(32));
    };

    auto min_index = [&](const SunMMIOValue &lhs, const SunMMIOValue &rhs) {
      return builder_->Binary(
          NewValueName(), BinaryOp::kMin, ArithmeticFlavor::kIndex, lhs, rhs,
          SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
          DataType::Int(32));
    };

    SunMMIOValue exec_i = EvalExpr(scope.execution_loops[0]->loop_var);
    SunMMIOValue exec_j = EvalExpr(scope.execution_loops[1]->loop_var);
    SunMMIOValue tile_m = make_index_const(scope.tile_shape[0]);
    SunMMIOValue tile_n = make_index_const(scope.tile_shape[1]);
    SunMMIOValue domain_m = EnsureIndex(
        EvalExpr(scope.domain_shape[scope.execution_domain_axes[0]]));
    SunMMIOValue domain_n = EnsureIndex(
        EvalExpr(scope.domain_shape[scope.execution_domain_axes[1]]));

    SunMMIOValue valid_rows = min_index(
        tile_m,
        sub_index(domain_m,
                  builder_->Binary(
                      NewValueName(), BinaryOp::kMul, ArithmeticFlavor::kIndex,
                      exec_i, tile_m,
                      SunMMIOType{
                          SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
                      DataType::Int(32))));
    SunMMIOValue valid_cols = min_index(
        tile_n,
        sub_index(domain_n,
                  builder_->Binary(
                      NewValueName(), BinaryOp::kMul, ArithmeticFlavor::kIndex,
                      exec_j, tile_n,
                      SunMMIOType{
                          SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
                      DataType::Int(32))));

    SunMMIOType bool_ty{SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}};
    SunMMIOValue row_tail_cond = builder_->Compare(
        NewValueName(), CompareOp::kLT, CompareDomain::kSignedInt, valid_rows,
        tile_m, valid_rows.type);
    row_tail_cond = EnsureType(row_tail_cond, bool_ty, DataType::Bool());
    SunMMIOValue col_tail_cond = builder_->Compare(
        NewValueName(), CompareOp::kLT, CompareDomain::kSignedInt, valid_cols,
        tile_n, valid_cols.type);
    col_tail_cond = EnsureType(col_tail_cond, bool_ty, DataType::Bool());

    SunMMIOType mask_type;
    mask_type.kind = SunMMIOType::Kind::kTile;
    mask_type.dtype = DataType::Bool();
    mask_type.lanes = 1;
    for (int64_t dim : scope.tile_shape) {
      mask_type.shape.push_back(IntImm(DataType::Int(32), dim));
    }
    return TailMaskInfo{valid_rows, valid_cols, row_tail_cond, col_tail_cond,
                        mask_type};
  };

  lower_expr = [&](const PrimExpr &expr, TileBlockState *state,
                   std::optional<DataType> preferred_dtype) -> SunMMIOValue {
    auto emit_select = [&](const PrimExpr &condition,
                           const PrimExpr &true_value_expr,
                           const PrimExpr &false_value_expr, DataType dtype) {
      SunMMIOValue cond = lower_expr(condition, state, std::nullopt);
      SunMMIOValue true_value =
          lower_expr(true_value_expr, state, preferred_dtype);
      SunMMIOValue false_value =
          lower_expr(false_value_expr, state, preferred_dtype);
      if (IsTileLike(cond) || IsTileLike(true_value) ||
          IsTileLike(false_value)) {
        if (!IsTileLike(true_value) || !IsTileLike(false_value)) {
          UnsupportedExpr(
              expr.get(),
              "Clean v4 tile select currently expects tile true and false "
              "values");
        }
        DataType result_dtype = choose_result_dtype(dtype, preferred_dtype);
        SunMMIOType result_type =
            MakeTileType(result_dtype, ExtractStaticShape(true_value.type));
        return builder_->TileSelect(NewValueName(), cond, true_value,
                                    false_value, result_type, result_dtype);
      }
      DataType result_dtype = choose_result_dtype(dtype, preferred_dtype);
      SunMMIOType result_type{SunMMIOType::Kind::kScalar, result_dtype, 1, {}};
      return builder_->Select(NewValueName(), cond, true_value, false_value,
                              result_type, result_dtype);
    };

    if (const auto *var = expr.as<VarNode>()) {
      auto it = state->let_values.find(var);
      if (it != state->let_values.end()) {
        return it->second;
      }
      return LookupVar(var);
    }
    if (const auto *let = expr.as<LetNode>()) {
      SunMMIOValue value = lower_expr(let->value, state, preferred_dtype);
      TileBlockState let_state = *state;
      let_state.let_values[let->var.get()] = value;
      return lower_expr(let->body, &let_state, preferred_dtype);
    }
    if (const auto *load = expr.as<BufferLoadNode>()) {
      auto local_it = state->local_tile_values.find(load->buffer.get());
      if (local_it != state->local_tile_values.end()) {
        return local_it->second;
      }
      auto reg_it = state->register_tile_values.find(load->buffer.get());
      if (reg_it != state->register_tile_values.end()) {
        SunMMIOValue value = reg_it->second;
        if (IsTileLike(value) && value.type.shape.size() == 1 &&
            state->interior_axis0_loop != nullptr &&
            state->interior_axis1_loop == nullptr) {
          int64_t unsqueeze_axis = 0;
          auto axis_it =
              state->register_unsqueeze_axes.find(load->buffer.get());
          if (axis_it != state->register_unsqueeze_axes.end()) {
            unsqueeze_axis = axis_it->second;
          }
          ICHECK(unsqueeze_axis == 0 || unsqueeze_axis == 1)
              << "1D register tile can only be unsqueezed back to a 2D tile";
          int64_t extent = ExtractStaticShape(value.type)[0];
          std::vector<int64_t> unsqueezed_shape =
              unsqueeze_axis == 0 ? std::vector<int64_t>{1, extent}
                                  : std::vector<int64_t>{extent, 1};
          SunMMIOType unsqueezed_type =
              MakeTileType(value.dtype, unsqueezed_shape);
          value =
              builder_->TileUnsqueeze(NewValueName(), value, unsqueezed_type,
                                      unsqueeze_axis, value.dtype);
        }
        return value;
      }
      TileAccessInfo access =
          analyze_access(load->buffer, load->indices, state);
      if (!access.promoted_unit_tile_view) {
        auto it = state->current_tile_values.find(load->buffer.get());
        if (it != state->current_tile_values.end()) {
          return it->second;
        }
      }
      SunMMIOValue tile;
      if (access.requires_aligned_1d_load) {
        tile = load_aligned_1d_tile(access, state);
      } else {
        SunMMIOValue view = get_or_create_tile_view(access, state);
        SunMMIOType tile_type =
            MakeTileType(load->buffer->dtype, access.tile_shape);
        // Tail masking is applied at store time with tile.select(new, old).
        // Loads intentionally fetch the whole padded tile; this avoids relying
        // on masked tile.load semantics and preserves the old destination
        // values explicitly before the final unmasked store.
        tile = builder_->TileLoad(
            NewValueName(), view, tile_type, std::nullopt, std::nullopt,
            CanonicalizeSuvmDType(load->buffer->dtype).with_lanes(1));
      }
      tile = maybe_unsqueeze_tile(tile, access);
      if (!access.promoted_unit_tile_view) {
        state->current_tile_values.emplace(load->buffer.get(), tile);
      }
      return tile;
    }
    if (const auto *imm = expr.as<IntImmNode>()) {
      DataType dtype = CanonicalizeSuvmDType(imm->dtype);
      SunMMIOType scalar_type{SunMMIOType::Kind::kScalar, dtype, 1, {}};
      return builder_->ConstantInt(NewValueName(), imm->value, scalar_type,
                                   dtype.with_lanes(1));
    }
    if (const auto *imm = expr.as<FloatImmNode>()) {
      DataType dtype = CanonicalizeSuvmDType(imm->dtype);
      SunMMIOType scalar_type{SunMMIOType::Kind::kScalar, dtype, 1, {}};
      std::ostringstream os;
      os << std::setprecision(17) << imm->value;
      return builder_->ConstantFloat(NewValueName(), os.str(), scalar_type,
                                     dtype.with_lanes(1));
    }
    auto emit_binary = [&](BinaryOp op, const PrimExpr &lhs_expr,
                           const PrimExpr &rhs_expr, DataType dtype) {
      bool supports_mixed_precision = supports_mixed_precision_binary(op);
      std::optional<DataType> operand_preferred_dtype =
          supports_mixed_precision ? preferred_dtype : std::nullopt;
      SunMMIOValue lhs = lower_expr(lhs_expr, state, operand_preferred_dtype);
      SunMMIOValue rhs = lower_expr(rhs_expr, state, operand_preferred_dtype);
      DataType result_dtype = supports_mixed_precision
                                  ? choose_result_dtype(dtype, preferred_dtype)
                                  : CanonicalizeSuvmDType(dtype).with_lanes(1);
      if (IsScalarLike(lhs) && IsScalarLike(rhs)) {
        SunMMIOType result_type{
            SunMMIOType::Kind::kScalar, result_dtype, 1, {}};
        lhs = EnsureType(lhs, result_type, result_dtype);
        rhs = EnsureType(rhs, result_type, result_dtype);
        return builder_->Binary(NewValueName(), op,
                                arithmetic_flavor_for_dtype(result_dtype), lhs,
                                rhs, result_type, result_dtype);
      }
      std::vector<int64_t> result_shape;
      if (IsTileLike(lhs)) {
        result_shape = ExtractStaticShape(lhs.type);
      } else if (IsTileLike(rhs)) {
        result_shape = ExtractStaticShape(rhs.type);
      } else {
        result_shape = scope.tile_shape;
      }
      lhs = reorient_unit_tile_to_shape(lhs, result_shape);
      rhs = reorient_unit_tile_to_shape(rhs, result_shape);
      SunMMIOType tile_type = MakeTileType(result_dtype, result_shape);
      if (supports_mixed_precision) {
        if (!IsTileLike(lhs) && !is_float_like_dtype(lhs.dtype)) {
          lhs = cast_value_to_dtype(lhs, result_dtype);
        }
        if (!IsTileLike(rhs) && !is_float_like_dtype(rhs.dtype)) {
          rhs = cast_value_to_dtype(rhs, result_dtype);
        }
      } else {
        lhs = cast_value_to_dtype(lhs, result_dtype);
        rhs = cast_value_to_dtype(rhs, result_dtype);
      }
      return builder_->Binary(NewValueName(), op, ArithmeticFlavor::kFloat, lhs,
                              rhs, tile_type, result_dtype);
    };
    auto emit_compare = [&](CompareOp op, const PrimExpr &lhs_expr,
                            const PrimExpr &rhs_expr) {
      SunMMIOValue lhs = lower_expr(lhs_expr, state, std::nullopt);
      SunMMIOValue rhs = lower_expr(rhs_expr, state, std::nullopt);
      if (!is_tile_compare_operand(lhs, rhs)) {
        SunMMIOType operand_type = lhs.type;
        rhs = EnsureType(rhs, operand_type, lhs.dtype);
        return builder_->Compare(NewValueName(), op,
                                 GetCompareDomain(lhs.dtype), lhs, rhs,
                                 operand_type);
      }
      std::vector<int64_t> result_shape;
      if (IsTileLike(lhs)) {
        result_shape = ExtractStaticShape(lhs.type);
      } else {
        result_shape = ExtractStaticShape(rhs.type);
      }
      SunMMIOType operand_type =
          MakeTileType(IsTileLike(lhs) ? lhs.dtype : rhs.dtype, result_shape);
      DataType operand_dtype = operand_type.dtype;
      if (!IsTileLike(lhs) && lhs.dtype != operand_dtype) {
        SunMMIOType scalar_type{
            SunMMIOType::Kind::kScalar, operand_dtype, 1, {}};
        lhs = builder_->Cast(NewValueName(), lhs, scalar_type, operand_dtype);
      }
      if (!IsTileLike(rhs) && rhs.dtype != operand_dtype) {
        SunMMIOType scalar_type{
            SunMMIOType::Kind::kScalar, operand_dtype, 1, {}};
        rhs = builder_->Cast(NewValueName(), rhs, scalar_type, operand_dtype);
      }
      return builder_->Compare(NewValueName(), op,
                               GetCompareDomain(operand_dtype), lhs, rhs,
                               operand_type);
    };
    auto emit_unary = [&](TileUnaryOp op, const PrimExpr &arg, DataType dtype,
                          bool force_f32 = false) {
      bool supports_mixed_precision =
          !force_f32 && supports_mixed_precision_unary(op);
      SunMMIOValue data =
          lower_expr(arg, state,
                     force_f32 ? std::optional<DataType>(DataType::Float(32))
                               : (supports_mixed_precision ? preferred_dtype
                                                           : std::nullopt));
      if (!IsTileLike(data)) {
        UnsupportedExpr(
            expr.get(),
            "Clean v4 tiles lowering currently only supports tile-valued unary "
            "math inside T.Tiles");
      }
      DataType result_dtype =
          force_f32 ? DataType::Float(32)
                    : (supports_mixed_precision
                           ? choose_result_dtype(dtype, preferred_dtype)
                           : CanonicalizeSuvmDType(dtype).with_lanes(1));
      if ((!supports_mixed_precision || force_f32) &&
          data.dtype != result_dtype) {
        SunMMIOType f32_type =
            MakeTileType(result_dtype, ExtractStaticShape(data.type));
        data = builder_->Cast(NewValueName(), data, f32_type, result_dtype);
      }
      SunMMIOType result_type =
          MakeTileType(result_dtype, ExtractStaticShape(data.type));
      return builder_->Unary(NewValueName(), op, data, result_type,
                             result_dtype);
    };
    if (const auto *add = expr.as<AddNode>()) {
      return emit_binary(BinaryOp::kAdd, add->a, add->b, add->dtype);
    }
    if (const auto *sub = expr.as<SubNode>()) {
      const auto *zero = sub->a.as<FloatImmNode>();
      if (zero && zero->value == 0.0) {
        return emit_unary(TileUnaryOp::kNeg, sub->b, sub->dtype);
      }
      return emit_binary(BinaryOp::kSub, sub->a, sub->b, sub->dtype);
    }
    if (const auto *mul = expr.as<MulNode>()) {
      return emit_binary(BinaryOp::kMul, mul->a, mul->b, mul->dtype);
    }
    if (const auto *div = expr.as<DivNode>()) {
      return emit_binary(BinaryOp::kDiv, div->a, div->b, div->dtype);
    }
    if (const auto *mod = expr.as<ModNode>()) {
      return emit_binary(BinaryOp::kMod, mod->a, mod->b, mod->dtype);
    }
    if (const auto *min = expr.as<MinNode>()) {
      return emit_binary(BinaryOp::kMin, min->a, min->b, min->dtype);
    }
    if (const auto *max = expr.as<MaxNode>()) {
      return emit_binary(BinaryOp::kMax, max->a, max->b, max->dtype);
    }
    if (const auto *eq = expr.as<EQNode>()) {
      return emit_compare(CompareOp::kEQ, eq->a, eq->b);
    }
    if (const auto *ne = expr.as<NENode>()) {
      return emit_compare(CompareOp::kNE, ne->a, ne->b);
    }
    if (const auto *lt = expr.as<LTNode>()) {
      return emit_compare(CompareOp::kLT, lt->a, lt->b);
    }
    if (const auto *le = expr.as<LENode>()) {
      return emit_compare(CompareOp::kLE, le->a, le->b);
    }
    if (const auto *gt = expr.as<GTNode>()) {
      return emit_compare(CompareOp::kGT, gt->a, gt->b);
    }
    if (const auto *ge = expr.as<GENode>()) {
      return emit_compare(CompareOp::kGE, ge->a, ge->b);
    }
    if (const auto *select = expr.as<SelectNode>()) {
      return emit_select(select->condition, select->true_value,
                         select->false_value, select->dtype);
    }
    if (const auto *cast = expr.as<CastNode>()) {
      SunMMIOValue value = lower_expr(cast->value, state, preferred_dtype);
      if (IsTileLike(value)) {
        DataType dst_dtype = CanonicalizeSuvmDType(cast->dtype).with_lanes(1);
        if (value.dtype == dst_dtype) {
          return value;
        }
        if (preferred_dtype.has_value() && is_float_like_dtype(value.dtype) &&
            is_float_like_dtype(dst_dtype) &&
            value.dtype ==
                CanonicalizeSuvmDType(preferred_dtype.value()).with_lanes(1)) {
          return value;
        }
        SunMMIOType dst_type = MakeTileType(CanonicalizeSuvmDType(cast->dtype),
                                            ExtractStaticShape(value.type));
        return builder_->Cast(NewValueName(), value, dst_type,
                              CanonicalizeSuvmDType(cast->dtype).with_lanes(1));
      }
      SunMMIOType scalar_type{SunMMIOType::Kind::kScalar,
                              CanonicalizeSuvmDType(cast->dtype),
                              1,
                              {}};
      return builder_->Cast(NewValueName(), value, scalar_type,
                            CanonicalizeSuvmDType(cast->dtype).with_lanes(1));
    }
    if (const auto *call = expr.as<CallNode>()) {
      const auto *op_node = call->op.as<OpNode>();
      if (op_node && op_node->name == "tl.infinity") {
        DataType dtype = CanonicalizeSuvmDType(call->dtype).with_lanes(1);
        SunMMIOType scalar_type{SunMMIOType::Kind::kScalar, dtype, 1, {}};
        return builder_->ConstantFloat(NewValueName(), "inf", scalar_type,
                                       dtype);
      }
      if (op_node && call->args.size() >= 3 &&
          op_node->name == "tir.if_then_else") {
        return emit_select(call->args[0], call->args[1], call->args[2],
                           call->dtype);
      }
      if (op_node && call->args.size() == 1 && op_node->name == "tir.exp") {
        return emit_unary(TileUnaryOp::kExp, call->args[0], call->dtype);
      }
      if (op_node && call->args.size() == 1 && op_node->name == "tir.exp2") {
        PrimExpr scaled_arg = call->args[0] * FloatImm(call->args[0].dtype(),
                                                       0.69314718055994530942);
        return emit_unary(TileUnaryOp::kExp, scaled_arg, call->dtype);
      }
      if (op_node && call->args.size() == 1 &&
          (op_node->name == "tir.fabs" || op_node->name == "tir.abs")) {
        return emit_unary(TileUnaryOp::kAbs, call->args[0], call->dtype);
      }
      if (op_node && call->args.size() == 1 && op_node->name == "tir.ceil") {
        return emit_unary(TileUnaryOp::kCeil, call->args[0], call->dtype, true);
      }
      if (op_node && call->args.size() == 1 && op_node->name == "tir.floor") {
        return emit_unary(TileUnaryOp::kFloor, call->args[0], call->dtype,
                          true);
      }
      if (op_node && call->args.size() == 1 && op_node->name == "tir.log") {
        return emit_unary(TileUnaryOp::kLn, call->args[0], call->dtype);
      }
      if (op_node && call->args.size() == 1 && op_node->name == "tir.round") {
        return emit_unary(TileUnaryOp::kRound, call->args[0], call->dtype,
                          true);
      }
      if (op_node && call->args.size() == 1 && op_node->name == "tir.rsqrt") {
        return emit_unary(TileUnaryOp::kRsqrt, call->args[0], call->dtype);
      }
      if (op_node && call->args.size() == 1 && op_node->name == "tir.trunc") {
        return emit_unary(TileUnaryOp::kTrunc, call->args[0], call->dtype,
                          true);
      }
      if (op_node && call->args.size() == 2 && op_node->name == "tir.fmod") {
        return emit_binary(BinaryOp::kMod, call->args[0], call->args[1],
                           call->dtype);
      }
      if (op_node && call->args.size() >= 1 &&
          op_node->name == "tl.ieee_frcp") {
        return emit_unary(TileUnaryOp::kRecip, call->args[0], call->dtype);
      }
      if (op_node && call->args.size() == 1 &&
          op_node->name == "tl.ieee_frsqrt") {
        return emit_unary(TileUnaryOp::kRsqrt, call->args[0], call->dtype);
      }
    }
    UnsupportedExpr(expr.get(),
                    "Clean v4 tiles lowering currently supports only "
                    "BufferLoad/add/sub/mul/div/mod/min/max/compare/select/"
                    "cast/constants and selected unary math calls");
  };

  lower_stmt = [&](const Stmt &stmt, TileBlockState *state) {
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &s : seq->seq) {
        lower_stmt(s, state);
      }
      return;
    }
    if (IsTokenLikeTileStmt(stmt)) {
      return;
    }
    if (const auto *let = stmt.as<LetStmtNode>()) {
      SunMMIOValue value = lower_expr(let->value, state, std::nullopt);
      TileBlockState let_state = *state;
      let_state.let_values[let->var.get()] = value;
      lower_stmt(let->body, &let_state);
      state->tile_view_cache = let_state.tile_view_cache;
      state->current_tile_values = let_state.current_tile_values;
      state->register_tile_values = let_state.register_tile_values;
      state->register_unsqueeze_axes = let_state.register_unsqueeze_axes;
      state->local_tile_values = let_state.local_tile_values;
      state->local_unit_tile_axes = let_state.local_unit_tile_axes;
      return;
    }
    if (const auto *store = stmt.as<BufferStoreNode>()) {
      auto local_it = state->local_tile_values.find(store->buffer.get());
      if (local_it != state->local_tile_values.end()) {
        SunMMIOType local_type = local_it->second.type;
        SunMMIOValue rhs = lower_expr(store->value, state, local_type.dtype);
        if (!IsTileLike(rhs)) {
          SunMMIOType scalar_type{
              SunMMIOType::Kind::kScalar, local_type.dtype, 1, {}};
          rhs = EnsureType(rhs, scalar_type, local_type.dtype);
          rhs = builder_->TileFill(NewValueName(), rhs, local_type,
                                   local_type.dtype);
        } else if (!StaticShapesEqual(rhs.type, local_type) ||
                   rhs.dtype != local_type.dtype) {
          rhs =
              builder_->Cast(NewValueName(), rhs, local_type, local_type.dtype);
        }
        state->local_tile_values[store->buffer.get()] =
            builder_->BindValueAlias(make_local_value_name(store->buffer), rhs);
        return;
      }
      auto reg_ty_it = state->register_tile_types.find(store->buffer.get());
      if (reg_ty_it != state->register_tile_types.end()) {
        SunMMIOType reg_type = reg_ty_it->second;
        SunMMIOValue rhs = lower_expr(store->value, state, reg_type.dtype);
        if (!IsTileLike(rhs)) {
          SunMMIOType scalar_type{
              SunMMIOType::Kind::kScalar, reg_type.dtype, 1, {}};
          rhs = EnsureType(rhs, scalar_type, reg_type.dtype);
          rhs =
              builder_->TileFill(NewValueName(), rhs, reg_type, reg_type.dtype);
        } else {
          std::vector<int64_t> rhs_shape = ExtractStaticShape(rhs.type);
          std::vector<int64_t> reg_shape = ExtractStaticShape(reg_type);
          if (rhs_shape != reg_shape &&
              rhs_shape.size() == reg_shape.size() + 1) {
            for (int64_t axis = 0;
                 axis < static_cast<int64_t>(rhs_shape.size()); ++axis) {
              if (rhs_shape[axis] != 1) {
                continue;
              }
              std::vector<int64_t> squeezed_shape = rhs_shape;
              squeezed_shape.erase(squeezed_shape.begin() + axis);
              if (squeezed_shape == reg_shape) {
                rhs = builder_->TileSqueeze(NewValueName(), rhs, reg_type, axis,
                                            reg_type.dtype);
                rhs_shape = reg_shape;
                break;
              }
            }
          }
          ICHECK(rhs_shape == reg_shape)
              << "Reduce register tile store cannot normalize RHS shape";
          if (rhs.dtype != reg_type.dtype) {
            rhs = builder_->Cast(NewValueName(), rhs, reg_type, reg_type.dtype);
          }
        }
        state->register_tile_values[store->buffer.get()] =
            builder_->BindValueAlias(make_register_value_name(store->buffer),
                                     rhs);
        return;
      }
      std::optional<int64_t> forced_unit_axis =
          find_local_unit_axis_in_expr(store->value, state);
      TileAccessInfo natural_access =
          analyze_access(store->buffer, store->indices, state);
      bool use_forced_unit_axis = forced_unit_axis.has_value() &&
                                  !natural_access.requires_aligned_1d_load;
      std::optional<int64_t> saved_forced_axis;
      bool had_saved_forced_axis = false;
      if (use_forced_unit_axis) {
        // Lower the target load/store in the same 2D unit-tile shape as the
        // local in-tile reduce result, avoiding fake 1D load/squeeze/store.
        auto saved_it = state->local_unit_tile_axes.find(store->buffer.get());
        if (saved_it != state->local_unit_tile_axes.end()) {
          saved_forced_axis = saved_it->second;
          had_saved_forced_axis = true;
        }
        state->local_unit_tile_axes[store->buffer.get()] = *forced_unit_axis;
      }
      TileAccessInfo access =
          use_forced_unit_axis
              ? analyze_access(store->buffer, store->indices, state)
              : natural_access;
      SunMMIOValue rhs = normalize_for_store(
          access,
          lower_expr(
              store->value, state,
              CanonicalizeSuvmDType(store->buffer->dtype).with_lanes(1)));
      SunMMIOValue dst_view = get_or_create_tile_view(access, state);
      std::optional<SunMMIOValue> mask =
          (access.tile_rank == 2) ? state->tile_mask : std::nullopt;
      if (mask.has_value()) {
        SunMMIOType dst_tile_type =
            MakeTileType(store->buffer->dtype, access.tile_shape);
        SunMMIOValue old_tile = builder_->TileLoad(
            NewValueName(), dst_view, dst_tile_type, std::nullopt, std::nullopt,
            CanonicalizeSuvmDType(store->buffer->dtype).with_lanes(1));
        rhs = builder_->TileSelect(
            NewValueName(), mask.value(), rhs, old_tile, dst_tile_type,
            CanonicalizeSuvmDType(store->buffer->dtype).with_lanes(1));
      }
      if (access.requires_aligned_1d_load) {
        store_aligned_1d_tile(access, rhs, state);
      } else {
        builder_->TileStore(rhs, dst_view, std::nullopt);
      }
      if (access.promoted_unit_tile_view) {
        state->current_tile_values.erase(store->buffer.get());
      } else {
        state->current_tile_values[store->buffer.get()] = rhs;
      }
      if (use_forced_unit_axis) {
        if (had_saved_forced_axis) {
          state->local_unit_tile_axes[store->buffer.get()] = *saved_forced_axis;
        } else {
          state->local_unit_tile_axes.erase(store->buffer.get());
        }
      }
      return;
    }
    UnsupportedStmt(stmt.get(),
                    "Clean v4 tiles lowering currently supports only "
                    "SeqStmt/token Evaluate/BufferStore");
  };

  auto lower_vector_core_in_tile_reduce = [&](const CallNode *call,
                                              TileBlockState *state) {
    ICHECK_EQ(call->args.size(), 4U)
        << "tl.vector_core_in_tile_reduce expects predicate, dst region, src "
           "region, and axis";
    const auto *predicate = call->args[0].as<StringImmNode>();
    ICHECK(predicate)
        << "tl.vector_core_in_tile_reduce predicate must be StringImm";
    BufferRegion dst_region = tl::NormalizeToBufferRegion(call->args[1]);
    BufferRegion src_region = tl::NormalizeToBufferRegion(call->args[2]);
    const auto *axis_imm = call->args[3].as<IntImmNode>();
    ICHECK(axis_imm) << "tl.vector_core_in_tile_reduce axis must be IntImm";
    int64_t axis = static_cast<int64_t>(axis_imm->value);
    note_register_unsqueeze_axis(state, dst_region->buffer, axis);

    SunMMIOValue src_tile;
    std::vector<int64_t> src_shape;
    auto reg_src_it =
        state->register_tile_values.find(src_region->buffer.get());
    if (reg_src_it != state->register_tile_values.end()) {
      src_tile = reg_src_it->second;
      src_shape = ExtractStaticShape(src_tile.type);
    } else {
      SunMMIOValue src_view = make_tile_view_from_region(src_region, state);
      src_shape = ExtractStaticShape(src_view.type);
      SunMMIOType src_tile_type =
          MakeTileType(src_region->buffer->dtype, src_shape);
      src_tile = builder_->TileLoad(
          NewValueName(), src_view, src_tile_type, std::nullopt, std::nullopt,
          CanonicalizeSuvmDType(src_region->buffer->dtype).with_lanes(1));
    }

    std::vector<int64_t> result_shape = src_shape;
    ICHECK_GE(axis, 0);
    ICHECK_LT(axis, static_cast<int64_t>(result_shape.size()));
    result_shape[axis] = 1;
    SunMMIOType result_tile_type =
        MakeTileType(src_region->buffer->dtype, result_shape);
    SunMMIOValue reduced = builder_->TileReduce(
        NewValueName(), static_cast<std::string>(predicate->value), src_tile,
        result_tile_type, axis,
        CanonicalizeSuvmDType(src_region->buffer->dtype).with_lanes(1));

    if (IsReduceLocalTempBuffer(dst_region->buffer)) {
      SunMMIOValue local = builder_->BindValueAlias(
          make_local_value_name(dst_region->buffer), reduced);
      state->local_tile_values[dst_region->buffer.get()] = local;
      state->local_unit_tile_axes[dst_region->buffer.get()] = axis;
      return;
    }

    SunMMIOType dst_tile_type;
    bool dst_is_register =
        IsReduceRegisterTempBuffer(dst_region->buffer) &&
        state->register_tile_types.count(dst_region->buffer.get());
    if (dst_is_register) {
      dst_tile_type = state->register_tile_types.at(dst_region->buffer.get());
    } else {
      // Type-only query: the actual store path below materializes the
      // tile_view that is consumed by tile.store.
      dst_tile_type = make_tile_type_from_region(dst_region);
    }
    SunMMIOValue rhs = reduced;
    std::vector<int64_t> rhs_shape = ExtractStaticShape(rhs.type);
    std::vector<int64_t> dst_shape = ExtractStaticShape(dst_tile_type);
    if (rhs_shape.size() == dst_shape.size() + 1) {
      for (int64_t axis_to_squeeze = 0;
           axis_to_squeeze < static_cast<int64_t>(rhs_shape.size());
           ++axis_to_squeeze) {
        if (rhs_shape[axis_to_squeeze] != 1) {
          continue;
        }
        std::vector<int64_t> squeezed_shape = rhs_shape;
        squeezed_shape.erase(squeezed_shape.begin() + axis_to_squeeze);
        if (squeezed_shape == dst_shape) {
          rhs = builder_->TileSqueeze(
              NewValueName(), rhs, dst_tile_type, axis_to_squeeze,
              CanonicalizeSuvmDType(dst_region->buffer->dtype).with_lanes(1));
          break;
        }
      }
    }
    if (!StaticShapesEqual(rhs.type, dst_tile_type) ||
        rhs.dtype !=
            CanonicalizeSuvmDType(dst_region->buffer->dtype).with_lanes(1)) {
      rhs = builder_->Cast(
          NewValueName(), rhs, dst_tile_type,
          CanonicalizeSuvmDType(dst_region->buffer->dtype).with_lanes(1));
    }
    if (dst_is_register) {
      state->register_tile_values[dst_region->buffer.get()] =
          builder_->BindValueAlias(make_register_value_name(dst_region->buffer),
                                   rhs);
    } else {
      SunMMIOValue dst_view = make_tile_view_from_region(dst_region, state);
      builder_->TileStore(rhs, dst_view, std::nullopt);
      state->current_tile_values[dst_region->buffer.get()] = rhs;
    }
  };

  lower_reduce_stmt = [&](const Stmt &stmt, TileBlockState *state) {
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &s : seq->seq) {
        // A reduce tile body has conditional init/finalize phases around the
        // straight-line accumulation phase.  Values stored inside one guarded
        // branch are not valid SSA values outside that branch, so do not let
        // tile-value cache entries leak across top-level phases.
        state->current_tile_values.clear();
        lower_reduce_stmt(s, state);
      }
      state->current_tile_values.clear();
      return;
    }
    if (const auto *ifs = stmt.as<IfThenElseNode>()) {
      SunMMIOType bool_ty{SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}};
      SunMMIOValue cond =
          EnsureType(EvalExpr(ifs->condition), bool_ty, DataType::Bool());
      auto saved_cache = state->current_tile_values;
      auto saved_registers = state->register_tile_values;
      auto saved_locals = state->local_tile_values;
      auto saved_local_axes = state->local_unit_tile_axes;
      std::vector<SunMMIOValue> live_out_values =
          collect_register_live_out_values(state);
      builder_->BeginIf(cond, live_out_values);
      TileBlockState then_state = *state;
      then_state.current_tile_values = saved_cache;
      then_state.register_tile_values = saved_registers;
      then_state.local_tile_values = saved_locals;
      then_state.local_unit_tile_axes = saved_local_axes;
      lower_reduce_stmt(ifs->then_case, &then_state);
      if (ifs->else_case.defined()) {
        builder_->BeginElse();
        TileBlockState else_state = *state;
        else_state.current_tile_values = saved_cache;
        else_state.register_tile_values = saved_registers;
        else_state.local_tile_values = saved_locals;
        else_state.local_unit_tile_axes = saved_local_axes;
        lower_reduce_stmt(ifs->else_case.value(), &else_state);
      }
      builder_->EndIf();
      state->current_tile_values = saved_cache;
      state->register_tile_values = then_state.register_tile_values;
      state->local_tile_values = saved_locals;
      state->local_unit_tile_axes = saved_local_axes;
      return;
    }
    if (const auto *loop = stmt.as<ForNode>()) {
      auto axis = GetInteriorAxisAnnotation(loop);
      if (!axis.has_value()) {
        SunMMIOValue min = EnsureIndex(EvalExpr(loop->min));
        SunMMIOValue extent = EnsureIndex(EvalExpr(loop->extent));
        SunMMIOValue step = EmitConstIndex(1);
        SunMMIOValue upper = builder_->Binary(
            NewValueName(), BinaryOp::kAdd, ArithmeticFlavor::kIndex, min,
            extent,
            SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
            DataType::Int(32));
        std::vector<SunMMIOValue> live_out_values =
            collect_register_live_out_values(state);
        std::string iv = "%" + loop->loop_var->name_hint;
        builder_->BeginFor(iv, min, upper, step, loop->annotations,
                           live_out_values);
        EnterScope();
        BindVar(loop->loop_var,
                SunMMIOValue{
                    loop->loop_var.dtype(), iv,
                    SunMMIOType{
                        SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}}});
        TileBlockState loop_state = *state;
        for (const auto &kv : state->register_tile_types) {
          const BufferNode *buffer = kv.first;
          auto value_it = state->register_tile_values.find(buffer);
          ICHECK(value_it != state->register_tile_values.end());
          loop_state.register_tile_values[buffer] = builder_->BindValueAlias(
              value_it->second.value,
              SunMMIOValue{value_it->second.dtype, value_it->second.value,
                           kv.second});
        }
        auto saved_locals = loop_state.local_tile_values;
        auto saved_local_axes = loop_state.local_unit_tile_axes;
        loop_state.local_tile_values.clear();
        loop_state.local_unit_tile_axes.clear();
        lower_reduce_stmt(loop->body, &loop_state);
        loop_state.local_tile_values = saved_locals;
        loop_state.local_unit_tile_axes = saved_local_axes;
        ExitScope();
        builder_->EndFor();
        state->register_tile_values = loop_state.register_tile_values;
        state->current_tile_values = loop_state.current_tile_values;
        return;
      }
      TileBlockState loop_state = *state;
      if (axis.value() == 0) {
        loop_state.interior_axis0_loop = loop;
        loop_state.interior_axis1_loop = nullptr;
      } else if (axis.value() == 1) {
        loop_state.interior_axis1_loop = loop;
      } else {
        UnsupportedStmt(loop,
                        "Reduce tiles lowering currently supports up to 2D "
                        "interior loops");
      }
      lower_reduce_stmt(loop->body, &loop_state);
      state->current_tile_values = loop_state.current_tile_values;
      return;
    }
    if (IsTokenLikeTileStmt(stmt)) {
      return;
    }
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      if (const auto *call = eval->value.as<CallNode>()) {
        const auto *op_node = call->op.as<OpNode>();
        if (op_node && op_node->name == "tl.vector_core_in_tile_reduce") {
          lower_vector_core_in_tile_reduce(call, state);
          return;
        }
      }
    }
    lower_stmt(stmt, state);
  };

  auto emit_tile_stmt = [&](TileBlockState *state) {
    if (scope.tail_predicate.defined() && scope.full_tile_body.defined() &&
        scope.tail_tile_body.defined()) {
      auto lower_tail_with_mask = [&](const SunMMIOValue &mask) {
        TileBlockState tail_state = *state;
        tail_state.tile_mask = mask;
        tail_state.interior_axis0_loop = scope.tail_interior_axis0_loop;
        tail_state.interior_axis1_loop = scope.tail_interior_axis1_loop;
        lower_stmt(scope.tail_tile_block_body, &tail_state);
      };

      SunMMIOType bool_ty{SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}};
      SunMMIOValue cond =
          EnsureType(EvalExpr(scope.tail_predicate), bool_ty, DataType::Bool());
      builder_->BeginIf(cond, std::vector<int64_t>{});
      TileBlockState full_state = *state;
      full_state.tile_mask.reset();
      full_state.interior_axis0_loop = scope.interior_axis0_loop;
      full_state.interior_axis1_loop = scope.interior_axis1_loop;
      lower_stmt(scope.full_tile_block_body, &full_state);
      builder_->BeginElse();
      TailMaskInfo mask_info = build_tail_mask_info(state);
      builder_->BeginIf(mask_info.row_tail_cond, std::vector<int64_t>{});
      builder_->BeginIf(mask_info.col_tail_cond, std::vector<int64_t>{});
      SunMMIOValue rect_mask =
          builder_->TileRectMask(NewValueName(), mask_info.valid_rows,
                                 mask_info.valid_cols, mask_info.mask_type);
      lower_tail_with_mask(rect_mask);
      builder_->BeginElse();
      SunMMIOValue row_mask = builder_->TileAxisMask(
          NewValueName(), 0, mask_info.valid_rows, mask_info.mask_type);
      lower_tail_with_mask(row_mask);
      builder_->EndIf();
      builder_->BeginElse();
      SunMMIOValue col_mask = builder_->TileAxisMask(
          NewValueName(), 1, mask_info.valid_cols, mask_info.mask_type);
      lower_tail_with_mask(col_mask);
      builder_->EndIf();
      builder_->EndIf();
      return;
    }
    if (scope.is_reduce_scope) {
      lower_reduce_stmt(scope.tile_block_body, state);
      return;
    }
    lower_stmt(scope.tile_block_body, state);
  };

  std::function<void(size_t, TileBlockState *)> emit_loop_nest;
  emit_loop_nest = [&](size_t loop_index, TileBlockState *state) {
    if (loop_index == scope.domain_loops.size()) {
      emit_tile_stmt(state);
      return;
    }
    const ForNode *loop = scope.domain_loops[loop_index];
    SunMMIOValue min = EnsureIndex(EvalExpr(loop->min));
    SunMMIOValue extent = EnsureIndex(EvalExpr(loop->extent));
    SunMMIOValue step = EmitConstIndex(1);
    SunMMIOValue upper = builder_->Binary(
        NewValueName(), BinaryOp::kAdd, ArithmeticFlavor::kIndex, min, extent,
        SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
        DataType::Int(32));
    std::string iv = "%" + loop->loop_var->name_hint;
    std::vector<SunMMIOValue> live_out_values;
    if (scope.is_reduce_scope) {
      live_out_values = collect_register_live_out_values(state);
      builder_->BeginFor(iv, min, upper, step, loop->annotations,
                         live_out_values);
    } else {
      builder_->BeginFor(iv, min, upper, step, loop->annotations,
                         std::vector<int64_t>{});
    }
    EnterScope();
    BindVar(
        loop->loop_var,
        SunMMIOValue{
            loop->loop_var.dtype(), iv,
            SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}}});
    emit_loop_nest(loop_index + 1, state);
    ExitScope();
    builder_->EndFor();
  };

  TileBlockState state;
  state.scope = &scope;
  state.mlir_ctx = mlir_ctx;
  state.interior_axis0_loop = scope.interior_axis0_loop;
  state.interior_axis1_loop = scope.interior_axis1_loop;
  if (scope.is_reduce_scope) {
    discover_reduce_register_temps(scope.tile_block_body, &state);
    initialize_reduce_register_temps(&state);
  }
  emit_loop_nest(0, &state);
  return true;
}

} // namespace codegen
} // namespace tvm
