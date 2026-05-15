#include "sunmmio_mlir_tile.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "npuir/Dialect/SUVM/IR/Ops.h"
#include "tvm/runtime/logging.h"

namespace tvm {
namespace codegen {

namespace {

mlir::Location MapMlirLoc(SunmmioMlirContext &ctx) {
  return SunmmioMlirType(ctx).Loc();
}

mlir::Type MapMlirType(SunmmioMlirContext &ctx, const SunMMIOType &type) {
  return SunmmioMlirType(ctx).MapType(type);
}

mlir::Value GetTileCastOp(SunmmioMlirContext &ctx, mlir::Value src_value,
                          const SunMMIOType &dst_type) {
  mlir::Location loc = MapMlirLoc(ctx);
  mlir::Type dst_mlir_type = MapMlirType(ctx, dst_type);
  auto tile_type = mlir::dyn_cast<mlir::suvm::TileType>(dst_mlir_type);
  ICHECK(tile_type) << "Expected SUVM tile type for tile.cast result";
  return ctx.builder.create<mlir::suvm::TileCastOp>(loc, tile_type, src_value)
      .getResult();
}

mlir::Value CreateTypedPlaceholder(SunmmioMlirContext &ctx,
                                   mlir::Type result_type,
                                   llvm::StringRef tag) {
  mlir::Location loc = SunmmioMlirType(ctx).MakeDebugLoc(tag.str());
  mlir::Value seed =
      mlir::arith::ConstantIntOp::create(ctx.builder, loc, 0, 32);
  mlir::OperationState st(loc, "builtin.unrealized_conversion_cast");
  st.addOperands(seed);
  st.addTypes(result_type);
  mlir::Operation *cast_op = ctx.builder.create(st);
  cast_op->setAttr("sunmmio.fake", ctx.builder.getStringAttr(tag));
  return cast_op->getResult(0);
}

mlir::Value CreateTypedPlaceholderWithOperands(
    SunmmioMlirContext &ctx, mlir::Type result_type,
    llvm::ArrayRef<mlir::Value> operands, llvm::StringRef tag) {
  mlir::Location loc = SunmmioMlirType(ctx).MakeDebugLoc(tag.str());
  mlir::OperationState st(loc, "builtin.unrealized_conversion_cast");
  st.addOperands(operands);
  st.addTypes(result_type);
  mlir::Operation *cast_op = ctx.builder.create(st);
  cast_op->setAttr("sunmmio.fake", ctx.builder.getStringAttr(tag));
  return cast_op->getResult(0);
}

} // namespace

SunmmioMlirTile::SunmmioMlirTile(SunmmioMlirContext &ctx)
    : ctx_(ctx), type_(ctx) {}

SunMMIOValue SunmmioMlirTile::GetPartitionedTileView(
    const std::string &result_name, const SunMMIOValue &memtensor,
    const std::vector<SunMMIOValue> &indices,
    const std::vector<int64_t> &tiled_dims, const SunMMIOType &view_type,
    DataType dtype) {
  mlir::Type result_type = MapMlirType(ctx_, view_type);
  mlir::Value view_value;
  if (view_type.shape.size() == 2) {
    mlir::Value memtensor_value =
        ctx_.LookupOrCreateFakeValue(memtensor, "fake_missing_memtensor");
    mlir::OperationState st(MapMlirLoc(ctx_), "suvm.get_partitioned_tile_view");
    st.addOperands(memtensor_value);
    SunmmioMlirType type(ctx_);
    for (const SunMMIOValue &idx : indices) {
      mlir::Value idx_value =
          type.EnsureIndex(type.ResolveValueOrCreatePlaceholder(
              idx, ctx_.builder.getIndexType()));
      st.addOperands(idx_value);
    }
    st.addAttribute("tiled_dims",
                    ctx_.builder.getDenseI64ArrayAttr(tiled_dims));
    st.addTypes(result_type);
    view_value = ctx_.builder.create(st)->getResult(0);
  } else {
    LOG(WARNING)
        << "Using provisional 1D tile_view placeholder for clean v4 Tiles "
           "lowering; replace with real SUVM 1D tile_view once dialect "
           "support lands";
    std::vector<mlir::Value> operands;
    operands.reserve(1 + indices.size());
    operands.push_back(
        ctx_.LookupOrCreateFakeValue(memtensor, "fake_missing_memtensor"));
    SunmmioMlirType type(ctx_);
    for (const SunMMIOValue &idx : indices) {
      operands.push_back(type.EnsureIndex(type.ResolveValueOrCreatePlaceholder(
          idx, ctx_.builder.getIndexType())));
    }
    view_value = CreateTypedPlaceholderWithOperands(
        ctx_, result_type, operands, "fake_partitioned_tile_view");
    if (mlir::Operation *def = view_value.getDefiningOp()) {
      def->setAttr("tiled_dims", ctx_.builder.getDenseI64ArrayAttr(tiled_dims));
    }
  }
  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, view_value);
  }
  return SunMMIOValue{dtype, result_name, view_type};
}

SunMMIOValue SunmmioMlirTile::TileLoad(
    const std::string &result_name, const SunMMIOValue &tile_view,
    const SunMMIOType &tile_type, const std::optional<SunMMIOValue> &mask,
    const std::optional<SunMMIOValue> &maskedoff, DataType dtype) {
  mlir::Type result_type = MapMlirType(ctx_, tile_type);
  mlir::Value tile_value;
  if (tile_type.shape.size() == 2) {
    mlir::Value base =
        ctx_.LookupOrCreateFakeValue(tile_view, "fake_missing_tile_view");
    mlir::Value mask_value;
    mlir::Value maskedoff_value;
    if (mask.has_value()) {
      mask_value =
          ctx_.LookupOrCreateFakeValue(mask.value(), "fake_missing_tile_mask");
    }
    if (maskedoff.has_value()) {
      maskedoff_value = ctx_.LookupOrCreateFakeValue(
          maskedoff.value(), "fake_missing_tile_maskedoff");
    }
    tile_value = mlir::suvm::TileLoadOp::create(ctx_.builder, MapMlirLoc(ctx_),
                                                result_type, base, mask_value,
                                                maskedoff_value)
                     .getResult();
  } else {
    LOG(WARNING)
        << "Using provisional 1D tile placeholder for clean v4 Tiles "
           "lowering; replace with real SUVM 1D tile.load once dialect "
           "support lands";
    mlir::Value base =
        ctx_.LookupOrCreateFakeValue(tile_view, "fake_missing_tile_view");
    tile_value = CreateTypedPlaceholderWithOperands(ctx_, result_type, {base},
                                                    "fake_tile_load");
  }
  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, tile_value);
  }
  return SunMMIOValue{dtype, result_name, tile_type};
}

SunMMIOValue SunmmioMlirTile::TileFill(const std::string &result_name,
                                       const SunMMIOValue &scalar,
                                       const SunMMIOType &tile_type,
                                       DataType dtype) {
  mlir::Type result_type = MapMlirType(ctx_, tile_type);
  mlir::Value tile_value;
  if (tile_type.shape.size() == 2) {
    mlir::Value scalar_value =
        ctx_.LookupOrCreateFakeValue(scalar, "fake_missing_tile_fill_scalar");
    mlir::OperationState st(MapMlirLoc(ctx_), "suvm.tile.fill");
    st.addOperands(scalar_value);
    st.addTypes(result_type);
    tile_value = ctx_.builder.create(st)->getResult(0);
  } else {
    LOG(WARNING)
        << "Using provisional 1D tile.fill placeholder for clean v4 Tiles "
           "lowering; replace with real SUVM 1D tile.fill once dialect "
           "support lands";
    tile_value = CreateTypedPlaceholder(ctx_, result_type, "fake_tile_fill");
  }
  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, tile_value);
  }
  return SunMMIOValue{dtype, result_name, tile_type};
}

SunMMIOValue SunmmioMlirTile::Cast(const std::string &result_name,
                                   const SunMMIOValue &v,
                                   const SunMMIOType &dst_type,
                                   DataType dst_dtype) {
  mlir::Value src_value =
      ctx_.LookupOrCreateFakeValue(v, "fake_missing_tile_cast_src");
  mlir::Value cast_value = GetTileCastOp(ctx_, src_value, dst_type);
  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, cast_value);
  }
  return SunMMIOValue{dst_dtype, result_name, dst_type};
}

SunMMIOValue SunmmioMlirTile::Binary(const std::string &result_name,
                                     BinaryOp op, ArithmeticFlavor flavor,
                                     const SunMMIOValue &a,
                                     const SunMMIOValue &b,
                                     const SunMMIOType &result_type,
                                     DataType dtype) {
  ICHECK(flavor == ArithmeticFlavor::kFloat)
      << "Clean v4 tile backend currently only supports floating-point tile "
         "binary ops";
  mlir::Value lhs =
      ctx_.LookupOrCreateFakeValue(a, "fake_missing_tile_binary_lhs");
  mlir::Value rhs =
      ctx_.LookupOrCreateFakeValue(b, "fake_missing_tile_binary_rhs");
  mlir::Location loc = MapMlirLoc(ctx_);
  mlir::Type result_mlir_type = MapMlirType(ctx_, result_type);
  auto tile_type = mlir::dyn_cast<mlir::suvm::TileType>(result_mlir_type);
  ICHECK(tile_type) << "Expected SUVM tile type for tile binary result";

  mlir::Value binary_value;
  switch (op) {
  case BinaryOp::kAdd:
    binary_value =
        ctx_.builder.create<mlir::suvm::TileAddFOp>(loc, tile_type, lhs, rhs)
            .getResult();
    break;
  case BinaryOp::kSub:
    binary_value =
        ctx_.builder.create<mlir::suvm::TileSubFOp>(loc, tile_type, lhs, rhs)
            .getResult();
    break;
  case BinaryOp::kMul:
    binary_value =
        ctx_.builder.create<mlir::suvm::TileMulFOp>(loc, tile_type, lhs, rhs)
            .getResult();
    break;
  case BinaryOp::kDiv:
    binary_value =
        ctx_.builder.create<mlir::suvm::TileDivFOp>(loc, tile_type, lhs, rhs)
            .getResult();
    break;
  default:
    LOG(FATAL) << "Unsupported clean v4 tile binary op";
  }
  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, binary_value);
  }
  return SunMMIOValue{dtype, result_name, result_type};
}

SunMMIOValue SunmmioMlirTile::Unary(const std::string &result_name,
                                    TileUnaryOp op, const SunMMIOValue &data,
                                    const SunMMIOType &result_type,
                                    DataType dtype) {
  mlir::Value input =
      ctx_.LookupOrCreateFakeValue(data, "fake_missing_tile_unary_data");
  mlir::Location loc = MapMlirLoc(ctx_);
  mlir::Type result_mlir_type = MapMlirType(ctx_, result_type);
  auto tile_type = mlir::dyn_cast<mlir::suvm::TileType>(result_mlir_type);
  ICHECK(tile_type) << "Expected SUVM tile type for tile unary result";

  mlir::Value unary_value;
  switch (op) {
  case TileUnaryOp::kExp:
    unary_value =
        mlir::suvm::TileExpOp::create(ctx_.builder, loc, result_mlir_type,
                                      input, mlir::Value(), mlir::Value())
            .getResult();
    break;
  default:
    LOG(FATAL) << "Unsupported clean v4 tile unary op";
  }

  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, unary_value);
  }
  return SunMMIOValue{dtype, result_name, result_type};
}

SunMMIOValue SunmmioMlirTile::TileUnsqueeze(const std::string &result_name,
                                            const SunMMIOValue &tile,
                                            const SunMMIOType &tile_type,
                                            int64_t axis, DataType dtype) {
  mlir::Type result_type = MapMlirType(ctx_, tile_type);
  LOG(WARNING)
      << "Using provisional tile.unsqueeze placeholder in clean v4 Tiles "
         "lowering; replace with real SUVM tile.unsqueeze once dialect "
         "support lands";
  mlir::Value input =
      ctx_.LookupOrCreateFakeValue(tile, "fake_missing_tile_unsqueeze_src");
  mlir::Value tile_value = CreateTypedPlaceholderWithOperands(
      ctx_, result_type, {input}, "fake_tile_unsqueeze");
  if (mlir::Operation *def = tile_value.getDefiningOp()) {
    def->setAttr("axis", ctx_.builder.getI64IntegerAttr(axis));
  }
  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, tile_value);
  }
  return SunMMIOValue{dtype, result_name, tile_type};
}

SunMMIOValue
SunmmioMlirTile::TileSlice(const std::string &result_name,
                           const SunMMIOValue &tile,
                           const std::vector<SunMMIOValue> &offsets,
                           const SunMMIOType &tile_type, DataType dtype) {
  mlir::Type result_type = MapMlirType(ctx_, tile_type);
  mlir::Value input =
      ctx_.LookupOrCreateFakeValue(tile, "fake_missing_tile_slice_src");
  std::vector<mlir::Value> operands;
  operands.reserve(1 + offsets.size());
  operands.push_back(input);
  SunmmioMlirType type(ctx_);
  for (const SunMMIOValue &offset : offsets) {
    operands.push_back(type.EnsureIndex(type.ResolveValueOrCreatePlaceholder(
        offset, ctx_.builder.getIndexType())));
  }

  mlir::Value tile_value;
  bool has_static_offsets = true;
  llvm::SmallVector<int64_t, 4> static_offsets;
  static_offsets.reserve(offsets.size());
  for (const SunMMIOValue &offset : offsets) {
    mlir::Value offset_value = ctx_.LookupMLIRValue(offset.value);
    if (!offset_value) {
      has_static_offsets = false;
      break;
    }
    if (auto cst = mlir::getConstantIntValue(offset_value)) {
      static_offsets.push_back(*cst);
      continue;
    }
    has_static_offsets = false;
    break;
  }

  if (has_static_offsets) {
    llvm::SmallVector<int64_t, 4> sizes;
    for (const PrimExpr &dim : tile_type.shape) {
      const auto *imm = dim.as<IntImmNode>();
      ICHECK(imm) << "tile.slice currently expects static result shape";
      sizes.push_back(static_cast<int64_t>(imm->value));
    }
    mlir::OperationState st(MapMlirLoc(ctx_), "suvm.tile.slice");
    st.addOperands(input);
    st.addAttribute("offsets",
                    ctx_.builder.getDenseI64ArrayAttr(static_offsets));
    st.addAttribute("sizes", ctx_.builder.getDenseI64ArrayAttr(sizes));
    st.addTypes(result_type);
    tile_value = ctx_.builder.create(st)->getResult(0);
  } else {
    LOG(WARNING)
        << "Using provisional tile.slice placeholder with dynamic offsets in "
           "clean v4 Tiles lowering; replace with real dynamic tile.slice "
           "once dialect support lands";
    tile_value = CreateTypedPlaceholderWithOperands(ctx_, result_type, operands,
                                                    "fake_tile_slice");
  }
  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, tile_value);
  }
  return SunMMIOValue{dtype, result_name, tile_type};
}

SunMMIOValue SunmmioMlirTile::TileRectMask(const std::string &result_name,
                                           const SunMMIOValue &valid_rows,
                                           const SunMMIOValue &valid_cols,
                                           const SunMMIOType &tile_type) {
  auto make_tile_type = [&](DataType dtype,
                            std::initializer_list<int64_t> shape) {
    SunMMIOType type;
    type.kind = SunMMIOType::Kind::kTile;
    type.dtype = dtype;
    type.lanes = 1;
    for (int64_t dim : shape) {
      type.shape.push_back(IntImm(DataType::Int(32), dim));
    }
    return type;
  };

  auto extract_static_dim = [&](size_t axis) -> int64_t {
    ICHECK_LT(axis, tile_type.shape.size());
    const auto *imm = tile_type.shape[axis].as<IntImmNode>();
    ICHECK(imm) << "TileRectMask expects static tile dimensions";
    return static_cast<int64_t>(imm->value);
  };

  auto to_i32_scalar = [&](const SunMMIOValue &value,
                           llvm::StringRef debug_tag) -> mlir::Value {
    mlir::Value raw = type_.ResolveValueOrCreatePlaceholder(
        value, ctx_.builder.getIndexType());
    if (raw.getType().isIndex()) {
      return mlir::arith::IndexCastOp::create(
          ctx_.builder, SunmmioMlirType(ctx_).MakeDebugLoc(debug_tag.str()),
          ctx_.builder.getI32Type(), raw);
    }
    if (raw.getType().isInteger(32)) {
      return raw;
    }
    if (auto int_ty = mlir::dyn_cast<mlir::IntegerType>(raw.getType())) {
      return mlir::arith::ExtSIOp::create(
          ctx_.builder, SunmmioMlirType(ctx_).MakeDebugLoc(debug_tag.str()),
          ctx_.builder.getI32Type(), raw);
    }
    LOG(FATAL) << "TileRectMask expects integer/index valid extent input";
    TVM_FFI_UNREACHABLE();
  };

  int64_t rows_dim = extract_static_dim(0);
  int64_t cols_dim = extract_static_dim(1);

  SunMMIOType row_range_type = make_tile_type(DataType::Int(32), {rows_dim, 1});
  SunMMIOType col_range_type = make_tile_type(DataType::Int(32), {1, cols_dim});
  SunMMIOType row_range_full_type =
      make_tile_type(DataType::Int(32), {rows_dim, cols_dim});
  SunMMIOType col_range_full_type =
      make_tile_type(DataType::Int(32), {rows_dim, cols_dim});
  SunMMIOType mask_full_type =
      make_tile_type(DataType::Bool(), {rows_dim, cols_dim});

  mlir::Type row_range_mlir_type = MapMlirType(ctx_, row_range_type);
  mlir::Value row_range =
      CreateTypedPlaceholder(ctx_, row_range_mlir_type, "fake_tile_range");
  if (mlir::Operation *def = row_range.getDefiningOp()) {
    def->setAttr("axis", ctx_.builder.getI64IntegerAttr(0));
  }

  mlir::Type col_range_mlir_type = MapMlirType(ctx_, col_range_type);
  mlir::Value col_range =
      mlir::suvm::TileRangeOp::create(ctx_.builder, MapMlirLoc(ctx_),
                                      col_range_mlir_type)
          .getResult();

  mlir::Value rows_i32 = to_i32_scalar(valid_rows, "tail_mask_rows");
  mlir::Value cols_i32 = to_i32_scalar(valid_cols, "tail_mask_cols");

  mlir::Value row_range_full =
      mlir::suvm::TileBroadcastOp::create(
          ctx_.builder, MapMlirLoc(ctx_),
          MapMlirType(ctx_, row_range_full_type), row_range)
          .getResult();
  mlir::Value col_range_full =
      mlir::suvm::TileBroadcastOp::create(
          ctx_.builder, MapMlirLoc(ctx_),
          MapMlirType(ctx_, col_range_full_type), col_range)
          .getResult();

  mlir::Value row_mask =
      mlir::suvm::TileCmpIOp::create(
          ctx_.builder, MapMlirLoc(ctx_), MapMlirType(ctx_, mask_full_type),
          mlir::suvm::VCmpIPredicate::slt, row_range_full, rows_i32)
          .getResult();
  mlir::Value col_mask =
      mlir::suvm::TileCmpIOp::create(
          ctx_.builder, MapMlirLoc(ctx_), MapMlirType(ctx_, mask_full_type),
          mlir::suvm::VCmpIPredicate::slt, col_range_full, cols_i32)
          .getResult();

  mlir::Value mask_value =
      mlir::suvm::TileAndIOp::create(ctx_.builder, MapMlirLoc(ctx_),
                                     MapMlirType(ctx_, mask_full_type),
                                     row_mask, col_mask)
          .getResult();
  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, mask_value);
  }
  return SunMMIOValue{DataType::Bool(), result_name, tile_type};
}

SunMMIOValue SunmmioMlirTile::TileSqueeze(const std::string &result_name,
                                          const SunMMIOValue &tile,
                                          const SunMMIOType &tile_type,
                                          int64_t axis, DataType dtype) {
  mlir::Type result_type = MapMlirType(ctx_, tile_type);
  LOG(WARNING)
      << "Using provisional tile.squeeze placeholder in clean v4 Tiles "
         "lowering; replace with real SUVM tile.squeeze once dialect "
         "support lands";
  mlir::Value input =
      ctx_.LookupOrCreateFakeValue(tile, "fake_missing_tile_squeeze_src");
  mlir::Value tile_value = CreateTypedPlaceholderWithOperands(
      ctx_, result_type, {input}, "fake_tile_squeeze");
  if (mlir::Operation *def = tile_value.getDefiningOp()) {
    def->setAttr("axis", ctx_.builder.getI64IntegerAttr(axis));
  }
  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, tile_value);
  }
  return SunMMIOValue{dtype, result_name, tile_type};
}

void SunmmioMlirTile::TileStore(const SunMMIOValue &value,
                                const SunMMIOValue &tile_view,
                                const std::optional<SunMMIOValue> &mask) {
  if (value.type.shape.size() != 2 || tile_view.type.shape.size() != 2) {
    LOG(WARNING)
        << "Using provisional 1D tile.store placeholder in clean v4 Tiles "
           "lowering; replace with real SUVM 1D tile.store once dialect "
           "support lands";
    mlir::Value data =
        ctx_.LookupOrCreateFakeValue(value, "fake_missing_tile_store_value");
    mlir::Value base =
        ctx_.LookupOrCreateFakeValue(tile_view, "fake_missing_tile_store_view");
    mlir::OperationState st(MapMlirLoc(ctx_),
                            "builtin.unrealized_conversion_cast");
    st.addOperands({data, base});
    st.addTypes(ctx_.builder.getI32Type());
    mlir::Operation *fake_store = ctx_.builder.create(st);
    fake_store->setAttr("sunmmio.fake",
                        ctx_.builder.getStringAttr("fake_tile_store"));
    return;
  }
  mlir::Value data =
      ctx_.LookupOrCreateFakeValue(value, "fake_missing_tile_store_value");
  mlir::Value base =
      ctx_.LookupOrCreateFakeValue(tile_view, "fake_missing_tile_store_view");
  mlir::Value mask_value;
  if (mask.has_value()) {
    mask_value = ctx_.LookupOrCreateFakeValue(mask.value(),
                                              "fake_missing_tile_store_mask");
  }
  (void)mlir::suvm::TileStoreOp::create(ctx_.builder, MapMlirLoc(ctx_), base,
                                        data, mask_value);
}

} // namespace codegen
} // namespace tvm
