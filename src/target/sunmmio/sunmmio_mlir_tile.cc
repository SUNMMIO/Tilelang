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
    view_value =
        CreateTypedPlaceholder(ctx_, result_type, "fake_partitioned_tile_view");
  }
  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, view_value);
  }
  return SunMMIOValue{dtype, result_name, view_type};
}

SunMMIOValue SunmmioMlirTile::TileLoad(const std::string &result_name,
                                       const SunMMIOValue &tile_view,
                                       const SunMMIOType &tile_type,
                                       DataType dtype) {
  mlir::Type result_type = MapMlirType(ctx_, tile_type);
  mlir::Value tile_value;
  if (tile_type.shape.size() == 2) {
    mlir::Value base =
        ctx_.LookupOrCreateFakeValue(tile_view, "fake_missing_tile_view");
    mlir::OperationState st(MapMlirLoc(ctx_), "suvm.tile.load");
    st.addOperands(base);
    st.addTypes(result_type);
    tile_value = ctx_.builder.create(st)->getResult(0);
  } else {
    LOG(WARNING)
        << "Using provisional 1D tile placeholder for clean v4 Tiles "
           "lowering; replace with real SUVM 1D tile.load once dialect "
           "support lands";
    tile_value = CreateTypedPlaceholder(ctx_, result_type, "fake_tile_load");
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
        ctx_.builder.create<mlir::suvm::TileExpOp>(loc, tile_type, input)
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
  (void)axis;
  mlir::Type result_type = MapMlirType(ctx_, tile_type);
  LOG(WARNING)
      << "Using provisional tile.unsqueeze placeholder in clean v4 Tiles "
         "lowering; replace with real SUVM tile.unsqueeze once dialect "
         "support lands";
  mlir::Value tile_value =
      CreateTypedPlaceholder(ctx_, result_type, "fake_tile_unsqueeze");
  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, tile_value);
  }
  return SunMMIOValue{dtype, result_name, tile_type};
}

void SunmmioMlirTile::TileStore(const SunMMIOValue &value,
                                const SunMMIOValue &tile_view) {
  if (value.type.shape.size() != 2 || tile_view.type.shape.size() != 2) {
    LOG(WARNING)
        << "Skipping provisional 1D tile.store in clean v4 Tiles lowering; "
           "emit real SUVM 1D tile.store once dialect support lands";
    return;
  }
  mlir::Value data =
      ctx_.LookupOrCreateFakeValue(value, "fake_missing_tile_store_value");
  mlir::Value base =
      ctx_.LookupOrCreateFakeValue(tile_view, "fake_missing_tile_store_view");
  mlir::OperationState st(MapMlirLoc(ctx_), "suvm.tile.store");
  st.addOperands({data, base});
  (void)ctx_.builder.create(st);
}

} // namespace codegen
} // namespace tvm
