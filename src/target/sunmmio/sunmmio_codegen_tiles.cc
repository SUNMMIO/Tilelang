#include "codegen_sunmmio.h"

#include "../../transform/common/attr.h"
#include "sunmmio_mlir_builder.h"
#include "sunmmio_mlir_context.h"

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
};

struct TileBlockState {
  const TilesScopeInfo *scope{nullptr};
  SunmmioMlirContext *mlir_ctx{nullptr};
  std::unordered_map<const BufferNode *, SunMMIOValue> tile_view_cache;
  std::unordered_map<const BufferNode *, SunMMIOValue> current_tile_values;
};

struct TileAccessInfo {
  Buffer buffer;
  int tile_rank{0};
  std::vector<int64_t> tile_shape;
  std::vector<int> tile_axes;
  std::vector<SunMMIOValue> partition_indices;
  std::vector<int64_t> tiled_dims;
  int64_t unsqueeze_axis{-1};
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
  type.dtype = dtype.with_lanes(1);
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
  type.dtype = dtype.with_lanes(1);
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

bool IsTileLike(const SunMMIOValue &value) {
  return value.type.kind == SunMMIOType::Kind::kTile;
}

bool IsScalarLike(const SunMMIOValue &value) {
  return value.type.kind == SunMMIOType::Kind::kScalar ||
         value.type.kind == SunMMIOType::Kind::kIndex;
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

const SunmmioMlirContext::MemTensorBinding *
ResolveMemTensorBinding(SunmmioMlirContext *ctx, const Buffer &buffer,
                        std::string fallback_ssa_name) {
  const auto *binding = ctx->LookupMemTensorBinding(buffer->data);
  if (binding != nullptr) {
    return binding;
  }

  LOG(WARNING) << "Missing memtensor binding for buffer `" << buffer->name
               << "`, using provisional fake binding";
  SunMMIOType memtensor_type;
  memtensor_type.kind = SunMMIOType::Kind::kMemTensor;
  memtensor_type.dtype = buffer->dtype.with_lanes(1);
  memtensor_type.lanes = 1;
  for (const PrimExpr &dim : buffer->shape) {
    memtensor_type.shape.push_back(dim);
  }
  memtensor_type.memory_scope = buffer.scope();
  memtensor_type.byte_offset = 0;
  ctx->BindMemTensor(buffer->data, memtensor_type, fallback_ssa_name,
                     /*is_fake=*/true);
  return ctx->LookupMemTensorBinding(buffer->data);
}

std::optional<int64_t> MatchTiledIndex(const PrimExpr &index, const Var &exec,
                                       const Var &interior,
                                       int64_t tile_extent) {
  if (index.same_as(interior)) {
    return int64_t{0};
  }
  if (const auto *add = index.as<AddNode>()) {
    PrimExpr lhs = add->a;
    PrimExpr rhs = add->b;
    if (rhs.same_as(interior)) {
      if (const auto *mul = lhs.as<MulNode>()) {
        if (mul->a.same_as(exec)) {
          if (const auto *imm = mul->b.as<IntImmNode>()) {
            if (static_cast<int64_t>(imm->value) == tile_extent) {
              return int64_t{0};
            }
          }
        }
      }
      if (const auto *add2 = lhs.as<AddNode>()) {
        PrimExpr tiled = add2->a;
        PrimExpr offset = add2->b;
        if (const auto *mul = tiled.as<MulNode>()) {
          if (mul->a.same_as(exec)) {
            if (const auto *imm = mul->b.as<IntImmNode>()) {
              if (static_cast<int64_t>(imm->value) == tile_extent) {
                if (const auto *off_imm = offset.as<IntImmNode>()) {
                  if (static_cast<int64_t>(off_imm->value) % tile_extent == 0) {
                    return static_cast<int64_t>(off_imm->value) / tile_extent;
                  }
                }
              }
            }
          }
        }
      }
    }
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

  for (size_t i = scope.domain_shape.size(); i < chain.size(); ++i) {
    const ForNode *loop = chain[i];
    auto axis_it = loop->annotations.find(tl::attr::tile_interior_axis);
    if (axis_it == loop->annotations.end()) {
      continue;
    }
    int axis = Downcast<Integer>((*axis_it).second)->value;
    if (axis == 0) {
      scope.interior_axis0_loop = loop;
    } else if (axis == 1) {
      scope.interior_axis1_loop = loop;
    }
  }

  ICHECK(scope.interior_axis0_loop != nullptr)
      << "Tiles scope is missing interior axis 0 loop";
  scope.tile_block_body = scope.interior_axis0_loop->body;
  if (scope.interior_axis1_loop != nullptr) {
    scope.tile_block_body = scope.interior_axis1_loop->body;
  }
  if (const auto *seq = scope.tile_block_body.as<SeqStmtNode>()) {
    for (const Stmt &stmt : seq->seq) {
      if (IsTokenLikeTileStmt(stmt)) {
        LOG(WARNING) << "Ignoring token-related Evaluate inside T.Tiles body "
                        "per current integration contract";
      }
    }
  } else if (IsTokenLikeTileStmt(scope.tile_block_body)) {
    LOG(WARNING) << "Ignoring token-related Evaluate inside T.Tiles body per "
                    "current integration contract";
  }

  SunmmioMlirContext *mlir_ctx = TryGetMlirContext(builder_);
  ICHECK(mlir_ctx != nullptr)
      << "Tiles lowering currently expects SuvmSunmmioBuilder";

  auto analyze_access =
      [&](const Buffer &buffer,
          const ffi::Array<PrimExpr> &indices) -> TileAccessInfo {
    TileAccessInfo access;
    access.buffer = buffer;
    const auto *binding = ResolveMemTensorBinding(
        mlir_ctx, buffer, "%fake_memtensor_" + buffer->name);
    ICHECK(binding != nullptr)
        << "Unable to resolve memtensor binding for buffer `" << buffer->name
        << "`";

    std::vector<int64_t> memtensor_shape =
        ExtractStaticShape(binding->memtensor_type);
    access.partition_indices.reserve(memtensor_shape.size());
    access.tiled_dims.clear();

    std::vector<int> logical_tile_axes(indices.size(), -1);
    std::vector<int64_t> logical_offsets(indices.size(), 0);
    for (int dim = 0; dim < static_cast<int>(indices.size()); ++dim) {
      for (int axis = 0; axis < static_cast<int>(scope.execution_loops.size());
           ++axis) {
        const ForNode *exec_loop = scope.execution_loops[axis];
        const ForNode *interior_loop =
            axis == 0 ? scope.interior_axis0_loop : scope.interior_axis1_loop;
        if (exec_loop == nullptr || interior_loop == nullptr) {
          continue;
        }
        auto offset =
            MatchTiledIndex(indices[dim], exec_loop->loop_var,
                            interior_loop->loop_var, scope.tile_shape[axis]);
        if (offset) {
          logical_tile_axes[dim] = axis;
          logical_offsets[dim] = *offset;
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
            EvalExpr(scope.execution_loops[axis]->loop_var);
        if (logical_offsets[dim] != 0) {
          SunMMIOValue offset = builder_->ConstantInt(
              NewValueName(), logical_offsets[dim],
              SunMMIOType{SunMMIOType::Kind::kScalar, DataType::Int(32), 1, {}},
              DataType::Int(32));
          exec_index = builder_->Binary(
              NewValueName(), BinaryOp::kAdd, ArithmeticFlavor::kIndex,
              exec_index, offset,
              SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
              DataType::Int(32));
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

    access.tile_rank = static_cast<int>(access.tile_shape.size());
    ICHECK(access.tile_rank == 1 || access.tile_rank == 2)
        << "Clean v4 tiles lowering currently only supports 1D or 2D tile "
           "accesses inside T.Tiles";
    if (access.tile_rank == 1) {
      ICHECK_EQ(access.tile_axes.size(), 1U);
      access.unsqueeze_axis = access.tile_axes[0] == 0 ? 1 : 0;
    }
    return access;
  };

  auto get_or_create_tile_view = [&](const TileAccessInfo &access,
                                     TileBlockState *state) -> SunMMIOValue {
    auto it = state->tile_view_cache.find(access.buffer.get());
    if (it != state->tile_view_cache.end()) {
      return it->second;
    }
    const auto *binding = mlir_ctx->LookupMemTensorBinding(access.buffer->data);
    ICHECK(binding != nullptr)
        << "Missing memtensor binding in SunmmioMlirContext for buffer `"
        << access.buffer->name << "`";
    SunMMIOValue memtensor{access.buffer->dtype.with_lanes(1),
                           binding->ssa_name, binding->memtensor_type};
    SunMMIOType view_type =
        MakeTileViewType(access.buffer->dtype, access.tile_shape);
    SunMMIOValue view = builder_->GetPartitionedTileView(
        NewValueName(), memtensor, access.partition_indices, access.tiled_dims,
        view_type, access.buffer->dtype.with_lanes(1));
    state->tile_view_cache.emplace(access.buffer.get(), view);
    return view;
  };

  std::function<SunMMIOValue(const PrimExpr &, TileBlockState *)> lower_expr;
  std::function<void(const Stmt &, TileBlockState *)> lower_stmt;

  auto normalize_for_store = [&](const TileAccessInfo &access,
                                 const SunMMIOValue &value) -> SunMMIOValue {
    DataType dst_dtype = access.buffer->dtype.with_lanes(1);
    if (value.type.kind == SunMMIOType::Kind::kTile) {
      SunMMIOType dst_tile_type =
          access.tile_rank == 1
              ? MakeTileType(access.buffer->dtype,
                             ExtractStaticShape(value.type))
              : MakeTileType(access.buffer->dtype, access.tile_shape);
      if (value.dtype == dst_dtype &&
          StaticShapesEqual(value.type, dst_tile_type)) {
        return value;
      }
      return builder_->Cast(NewValueName(), value, dst_tile_type, dst_dtype);
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
    if (access.unsqueeze_axis < 0) {
      return value;
    }
    ICHECK(access.unsqueeze_axis == 0 || access.unsqueeze_axis == 1)
        << "1D tile access is missing unsqueeze axis";
    // The clean v4 path keeps 1D tile accesses explicit. Once dialect-side
    // 1D tile/tileview + tile.unsqueeze land, the builder implementation can
    // materialize this as a real rank-raising op. For now, keep the semantic
    // axis and move into the enclosing 2D execution-tile shape.
    std::vector<int64_t> unsqueezed_shape =
        scope.tile_shape.size() == 2
            ? scope.tile_shape
            : (access.unsqueeze_axis == 1
                   ? std::vector<int64_t>{access.tile_shape[0], 1}
                   : std::vector<int64_t>{1, access.tile_shape[0]});
    SunMMIOType tile_type = MakeTileType(value.dtype, unsqueezed_shape);
    return builder_->TileUnsqueeze(NewValueName(), value, tile_type,
                                   access.unsqueeze_axis,
                                   value.dtype.with_lanes(1));
  };

  lower_expr = [&](const PrimExpr &expr,
                   TileBlockState *state) -> SunMMIOValue {
    if (const auto *load = expr.as<BufferLoadNode>()) {
      TileAccessInfo access = analyze_access(load->buffer, load->indices);
      auto it = state->current_tile_values.find(load->buffer.get());
      if (it != state->current_tile_values.end()) {
        return it->second;
      }
      SunMMIOValue view = get_or_create_tile_view(access, state);
      SunMMIOType tile_type =
          MakeTileType(load->buffer->dtype, access.tile_shape);
      SunMMIOValue tile = builder_->TileLoad(NewValueName(), view, tile_type,
                                             load->buffer->dtype.with_lanes(1));
      tile = maybe_unsqueeze_tile(tile, access);
      state->current_tile_values.emplace(load->buffer.get(), tile);
      return tile;
    }
    if (const auto *imm = expr.as<IntImmNode>()) {
      SunMMIOType scalar_type{SunMMIOType::Kind::kScalar, imm->dtype, 1, {}};
      return builder_->ConstantInt(NewValueName(), imm->value, scalar_type,
                                   imm->dtype.with_lanes(1));
    }
    if (const auto *imm = expr.as<FloatImmNode>()) {
      SunMMIOType scalar_type{SunMMIOType::Kind::kScalar, imm->dtype, 1, {}};
      std::ostringstream os;
      os << std::setprecision(17) << imm->value;
      return builder_->ConstantFloat(NewValueName(), os.str(), scalar_type,
                                     imm->dtype.with_lanes(1));
    }
    auto emit_binary = [&](BinaryOp op, const PrimExpr &lhs_expr,
                           const PrimExpr &rhs_expr, DataType dtype) {
      SunMMIOValue lhs = lower_expr(lhs_expr, state);
      SunMMIOValue rhs = lower_expr(rhs_expr, state);
      if (IsScalarLike(lhs) && IsScalarLike(rhs)) {
        SunMMIOType result_type{
            SunMMIOType::Kind::kScalar, dtype.with_lanes(1), 1, {}};
        return builder_->Binary(NewValueName(), op, ArithmeticFlavor::kFloat,
                                lhs, rhs, result_type, dtype.with_lanes(1));
      }
      std::vector<int64_t> result_shape;
      if (IsTileLike(lhs)) {
        result_shape = ExtractStaticShape(lhs.type);
      } else if (IsTileLike(rhs)) {
        result_shape = ExtractStaticShape(rhs.type);
      } else {
        result_shape = scope.tile_shape;
      }
      SunMMIOType tile_type = MakeTileType(dtype, result_shape);
      if (!IsTileLike(lhs)) {
        SunMMIOType scalar_type{
            SunMMIOType::Kind::kScalar, dtype.with_lanes(1), 1, {}};
        lhs = builder_->Cast(NewValueName(), lhs, scalar_type,
                             dtype.with_lanes(1));
      }
      if (!IsTileLike(rhs)) {
        SunMMIOType scalar_type{
            SunMMIOType::Kind::kScalar, dtype.with_lanes(1), 1, {}};
        rhs = builder_->Cast(NewValueName(), rhs, scalar_type,
                             dtype.with_lanes(1));
      }
      return builder_->Binary(NewValueName(), op, ArithmeticFlavor::kFloat, lhs,
                              rhs, tile_type, dtype.with_lanes(1));
    };
    if (const auto *add = expr.as<AddNode>()) {
      return emit_binary(BinaryOp::kAdd, add->a, add->b, add->dtype);
    }
    if (const auto *sub = expr.as<SubNode>()) {
      return emit_binary(BinaryOp::kSub, sub->a, sub->b, sub->dtype);
    }
    if (const auto *mul = expr.as<MulNode>()) {
      return emit_binary(BinaryOp::kMul, mul->a, mul->b, mul->dtype);
    }
    if (const auto *div = expr.as<DivNode>()) {
      return emit_binary(BinaryOp::kDiv, div->a, div->b, div->dtype);
    }
    if (const auto *cast = expr.as<CastNode>()) {
      SunMMIOValue value = lower_expr(cast->value, state);
      if (IsTileLike(value)) {
        SunMMIOType dst_type =
            MakeTileType(cast->dtype, ExtractStaticShape(value.type));
        return builder_->Cast(NewValueName(), value, dst_type,
                              cast->dtype.with_lanes(1));
      }
      SunMMIOType scalar_type{SunMMIOType::Kind::kScalar, cast->dtype, 1, {}};
      return builder_->Cast(NewValueName(), value, scalar_type,
                            cast->dtype.with_lanes(1));
    }
    if (const auto *call = expr.as<CallNode>()) {
      const auto *op_node = call->op.as<OpNode>();
      if (op_node && call->args.size() == 1 && op_node->name == "tir.exp") {
        SunMMIOValue data = lower_expr(call->args[0], state);
        if (!IsTileLike(data)) {
          UnsupportedExpr(
              expr.get(),
              "Clean v4 tiles lowering currently only supports tile-valued "
              "tir.exp inside T.Tiles");
        }
        SunMMIOType result_type =
            MakeTileType(call->dtype, ExtractStaticShape(data.type));
        return builder_->Unary(NewValueName(), TileUnaryOp::kExp, data,
                               result_type, call->dtype.with_lanes(1));
      }
    }
    UnsupportedExpr(expr.get(),
                    "Clean v4 tiles lowering currently supports only "
                    "BufferLoad/add/sub/mul/div/cast/constants/tir.exp");
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
    if (const auto *store = stmt.as<BufferStoreNode>()) {
      TileAccessInfo access = analyze_access(store->buffer, store->indices);
      SunMMIOValue rhs =
          normalize_for_store(access, lower_expr(store->value, state));
      SunMMIOValue dst_view = get_or_create_tile_view(access, state);
      builder_->TileStore(rhs, dst_view);
      state->current_tile_values[store->buffer.get()] = rhs;
      return;
    }
    UnsupportedStmt(stmt.get(),
                    "Clean v4 tiles lowering currently supports only "
                    "SeqStmt/token Evaluate/BufferStore");
  };

  std::function<void(size_t, TileBlockState *)> emit_loop_nest;
  emit_loop_nest = [&](size_t loop_index, TileBlockState *state) {
    if (loop_index == scope.domain_loops.size()) {
      lower_stmt(scope.tile_block_body, state);
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
    builder_->BeginFor(iv, min, upper, step, loop->annotations, {});
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
  emit_loop_nest(0, &state);
  return true;
}

} // namespace codegen
} // namespace tvm
