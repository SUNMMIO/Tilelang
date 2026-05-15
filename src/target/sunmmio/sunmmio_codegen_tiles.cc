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
  PrimExpr tail_predicate;
  Stmt full_tile_body;
  Stmt tail_tile_body;
  Stmt full_tile_block_body;
  Stmt tail_tile_block_body;
  const ForNode *tail_interior_axis0_loop{nullptr};
  const ForNode *tail_interior_axis1_loop{nullptr};
};

struct TileBlockState {
  const TilesScopeInfo *scope{nullptr};
  SunmmioMlirContext *mlir_ctx{nullptr};
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
  int64_t unsqueeze_axis{-1};
  bool requires_aligned_1d_load{false};
  int64_t aligned_load_bytes{0};
  int64_t aligned_load_elems{0};
  int64_t aligned_load_axis{-1};
  std::vector<int64_t> aligned_load_shape;
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

std::optional<int64_t> MatchTiledIndex(const PrimExpr &index, const Var &exec,
                                       const Var &interior,
                                       int64_t tile_extent) {
  if (index.same_as(interior)) {
    return int64_t{0};
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
    return const_offset / tile_extent;
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

  Stmt tile_scope_stmt = scope.execution_loops.back()->body;
  if (const auto *ifs = tile_scope_stmt.as<IfThenElseNode>()) {
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
    for (int dim = 0; dim < static_cast<int>(indices.size()); ++dim) {
      for (int axis = 0; axis < static_cast<int>(scope.execution_loops.size());
           ++axis) {
        const ForNode *exec_loop = scope.execution_loops[axis];
        const ForNode *interior_loop =
            axis == 0 ? state->interior_axis0_loop : state->interior_axis1_loop;
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
              SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
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
      if (IsRsramScope(binding.buffer_type.memory_scope)) {
        int64_t dtype_bytes =
            static_cast<int64_t>(CanonicalizeSuvmDType(buffer->dtype).bytes());
        ICHECK_GT(dtype_bytes, 0)
            << "Unexpected zero-sized dtype in Tiles lowering";
        ICHECK_EQ(64 % dtype_bytes, 0)
            << "64B alignment path requires dtype byte width to divide 64";
        access.requires_aligned_1d_load = true;
        access.aligned_load_bytes = 64;
        access.aligned_load_elems = 64 / dtype_bytes;
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
    auto it = state->tile_view_cache.find(access.buffer.get());
    if (it != state->tile_view_cache.end()) {
      return it->second;
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
    state->tile_view_cache.emplace(access.buffer.get(), view);
    return view;
  };

  std::function<SunMMIOValue(const PrimExpr &, TileBlockState *)> lower_expr;
  std::function<void(const Stmt &, TileBlockState *)> lower_stmt;

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

  auto add_index = [&](const SunMMIOValue &lhs, const SunMMIOValue &rhs) {
    return builder_->Binary(
        NewValueName(), BinaryOp::kAdd, ArithmeticFlavor::kIndex, lhs, rhs,
        SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
        DataType::Int(32));
  };

  auto mul_index = [&](const SunMMIOValue &lhs, const SunMMIOValue &rhs) {
    return builder_->Binary(
        NewValueName(), BinaryOp::kMul, ArithmeticFlavor::kIndex, lhs, rhs,
        SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
        DataType::Int(32));
  };

  auto div_index = [&](const SunMMIOValue &lhs, const SunMMIOValue &rhs) {
    return builder_->Binary(
        NewValueName(), BinaryOp::kDiv, ArithmeticFlavor::kIndex, lhs, rhs,
        SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
        DataType::Int(32));
  };

  auto mod_index = [&](const SunMMIOValue &lhs, const SunMMIOValue &rhs) {
    return builder_->Binary(
        NewValueName(), BinaryOp::kMod, ArithmeticFlavor::kIndex, lhs, rhs,
        SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
        DataType::Int(32));
  };

  auto load_aligned_1d_tile = [&](const TileAccessInfo &access,
                                  TileBlockState *state) -> SunMMIOValue {
    ICHECK(access.requires_aligned_1d_load);
    ICHECK_EQ(access.partition_indices.size(), 1U)
        << "64B-aligned 1D tile load currently expects rank-1 source memtensor";

    int64_t dtype_bytes = static_cast<int64_t>(
        CanonicalizeSuvmDType(access.buffer->dtype).bytes());
    SunMMIOValue tile_index = EnsureIndex(access.partition_indices[0]);
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
    SunMMIOValue aligned_view = builder_->GetPartitionedTileView(
        NewValueName(), memtensor, {region_index}, {0}, aligned_view_type,
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
    return builder_->TileSqueeze(
        NewValueName(), sliced_tile,
        MakeTileType(access.buffer->dtype, access.tile_shape),
        access.unsqueeze_axis,
        CanonicalizeSuvmDType(access.buffer->dtype).with_lanes(1));
  };

  auto build_zero_tile = [&](DataType dtype,
                             const std::vector<int64_t> &shape) {
    SunMMIOType scalar_type{SunMMIOType::Kind::kScalar,
                            CanonicalizeSuvmDType(dtype).with_lanes(1),
                            1,
                            {}};
    SunMMIOValue zero = builder_->ConstantInt(NewValueName(), 0, scalar_type,
                                              scalar_type.dtype);
    return builder_->TileFill(NewValueName(), zero, MakeTileType(dtype, shape),
                              CanonicalizeSuvmDType(dtype).with_lanes(1));
  };

  auto build_tail_mask = [&](TileBlockState *state) -> SunMMIOValue {
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

    SunMMIOValue next_m = builder_->Binary(
        NewValueName(), BinaryOp::kMul, ArithmeticFlavor::kIndex,
        builder_->Binary(
            NewValueName(), BinaryOp::kAdd, ArithmeticFlavor::kIndex, exec_i,
            make_index_const(1),
            SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
            DataType::Int(32)),
        tile_m,
        SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
        DataType::Int(32));
    SunMMIOValue next_n = builder_->Binary(
        NewValueName(), BinaryOp::kMul, ArithmeticFlavor::kIndex,
        builder_->Binary(
            NewValueName(), BinaryOp::kAdd, ArithmeticFlavor::kIndex, exec_j,
            make_index_const(1),
            SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
            DataType::Int(32)),
        tile_n,
        SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
        DataType::Int(32));

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

    SunMMIOType mask_type;
    mask_type.kind = SunMMIOType::Kind::kTile;
    mask_type.dtype = DataType::Bool();
    mask_type.lanes = 1;
    for (int64_t dim : scope.tile_shape) {
      mask_type.shape.push_back(IntImm(DataType::Int(32), dim));
    }
    return builder_->TileRectMask(NewValueName(), valid_rows, valid_cols,
                                  mask_type);
  };

  lower_expr = [&](const PrimExpr &expr,
                   TileBlockState *state) -> SunMMIOValue {
    if (const auto *load = expr.as<BufferLoadNode>()) {
      TileAccessInfo access =
          analyze_access(load->buffer, load->indices, state);
      auto it = state->current_tile_values.find(load->buffer.get());
      if (it != state->current_tile_values.end()) {
        return it->second;
      }
      SunMMIOValue tile;
      if (access.requires_aligned_1d_load) {
        tile = load_aligned_1d_tile(access, state);
      } else {
        SunMMIOValue view = get_or_create_tile_view(access, state);
        SunMMIOType tile_type =
            MakeTileType(load->buffer->dtype, access.tile_shape);
        std::optional<SunMMIOValue> mask =
            (access.tile_rank == 2) ? state->tile_mask : std::nullopt;
        std::optional<SunMMIOValue> maskedoff =
            (access.tile_rank == 2 && state->tile_mask.has_value())
                ? std::optional<SunMMIOValue>(
                      build_zero_tile(load->buffer->dtype, access.tile_shape))
                : std::nullopt;
        tile = builder_->TileLoad(
            NewValueName(), view, tile_type, mask, maskedoff,
            CanonicalizeSuvmDType(load->buffer->dtype).with_lanes(1));
      }
      tile = maybe_unsqueeze_tile(tile, access);
      state->current_tile_values.emplace(load->buffer.get(), tile);
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
      SunMMIOValue lhs = lower_expr(lhs_expr, state);
      SunMMIOValue rhs = lower_expr(rhs_expr, state);
      if (IsScalarLike(lhs) && IsScalarLike(rhs)) {
        SunMMIOType result_type{SunMMIOType::Kind::kScalar,
                                CanonicalizeSuvmDType(dtype).with_lanes(1),
                                1,
                                {}};
        return builder_->Binary(NewValueName(), op, ArithmeticFlavor::kFloat,
                                lhs, rhs, result_type,
                                CanonicalizeSuvmDType(dtype).with_lanes(1));
      }
      std::vector<int64_t> result_shape;
      if (IsTileLike(lhs)) {
        result_shape = ExtractStaticShape(lhs.type);
      } else if (IsTileLike(rhs)) {
        result_shape = ExtractStaticShape(rhs.type);
      } else {
        result_shape = scope.tile_shape;
      }
      SunMMIOType tile_type =
          MakeTileType(CanonicalizeSuvmDType(dtype), result_shape);
      if (!IsTileLike(lhs)) {
        SunMMIOType scalar_type{SunMMIOType::Kind::kScalar,
                                CanonicalizeSuvmDType(dtype).with_lanes(1),
                                1,
                                {}};
        lhs = builder_->Cast(NewValueName(), lhs, scalar_type,
                             CanonicalizeSuvmDType(dtype).with_lanes(1));
      }
      if (!IsTileLike(rhs)) {
        SunMMIOType scalar_type{SunMMIOType::Kind::kScalar,
                                CanonicalizeSuvmDType(dtype).with_lanes(1),
                                1,
                                {}};
        rhs = builder_->Cast(NewValueName(), rhs, scalar_type,
                             CanonicalizeSuvmDType(dtype).with_lanes(1));
      }
      return builder_->Binary(NewValueName(), op, ArithmeticFlavor::kFloat, lhs,
                              rhs, tile_type,
                              CanonicalizeSuvmDType(dtype).with_lanes(1));
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
      if (op_node && call->args.size() == 1 && op_node->name == "tir.exp") {
        SunMMIOValue data = lower_expr(call->args[0], state);
        if (!IsTileLike(data)) {
          UnsupportedExpr(
              expr.get(),
              "Clean v4 tiles lowering currently only supports tile-valued "
              "tir.exp inside T.Tiles");
        }
        SunMMIOType result_type = MakeTileType(
            CanonicalizeSuvmDType(call->dtype), ExtractStaticShape(data.type));
        return builder_->Unary(
            NewValueName(), TileUnaryOp::kExp, data, result_type,
            CanonicalizeSuvmDType(call->dtype).with_lanes(1));
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
      TileAccessInfo access =
          analyze_access(store->buffer, store->indices, state);
      SunMMIOValue rhs =
          normalize_for_store(access, lower_expr(store->value, state));
      SunMMIOValue dst_view = get_or_create_tile_view(access, state);
      std::optional<SunMMIOValue> mask =
          (access.tile_rank == 2) ? state->tile_mask : std::nullopt;
      builder_->TileStore(rhs, dst_view, mask);
      state->current_tile_values[store->buffer.get()] = rhs;
      return;
    }
    UnsupportedStmt(stmt.get(),
                    "Clean v4 tiles lowering currently supports only "
                    "SeqStmt/token Evaluate/BufferStore");
  };

  auto emit_tile_stmt = [&](TileBlockState *state) {
    if (scope.tail_predicate.defined() && scope.full_tile_body.defined() &&
        scope.tail_tile_body.defined()) {
      SunMMIOType bool_ty{SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}};
      SunMMIOValue cond =
          EnsureType(EvalExpr(scope.tail_predicate), bool_ty, DataType::Bool());
      builder_->BeginIf(cond, {});
      TileBlockState full_state = *state;
      full_state.tile_mask.reset();
      full_state.interior_axis0_loop = scope.interior_axis0_loop;
      full_state.interior_axis1_loop = scope.interior_axis1_loop;
      lower_stmt(scope.full_tile_block_body, &full_state);
      builder_->BeginElse();
      TileBlockState tail_state = *state;
      tail_state.tile_mask = build_tail_mask(state);
      tail_state.interior_axis0_loop = scope.tail_interior_axis0_loop;
      tail_state.interior_axis1_loop = scope.tail_interior_axis1_loop;
      lower_stmt(scope.tail_tile_block_body, &tail_state);
      builder_->EndIf();
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
  state.interior_axis0_loop = scope.interior_axis0_loop;
  state.interior_axis1_loop = scope.interior_axis1_loop;
  emit_loop_nest(0, &state);
  return true;
}

} // namespace codegen
} // namespace tvm
