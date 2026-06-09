#include "sunmmio_mlir_tile_op.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
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
  return mlir::suvm::TileCastOp::create(ctx.builder, loc, tile_type, src_value)
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

llvm::SmallVector<int64_t, 4> StaticShapeVector(const SunMMIOType &type,
                                                llvm::StringRef op_name) {
  llvm::SmallVector<int64_t, 4> shape;
  shape.reserve(type.shape.size());
  for (const PrimExpr &dim : type.shape) {
    const auto *imm = dim.as<IntImmNode>();
    ICHECK(imm) << op_name.str() << " expects static tile shape";
    shape.push_back(static_cast<int64_t>(imm->value));
  }
  return shape;
}

struct MixedIndexList {
  llvm::SmallVector<mlir::Value, 4> dynamic_values;
  llvm::SmallVector<int64_t, 4> static_values;
};

MixedIndexList BuildMixedIndexList(SunmmioMlirContext &ctx,
                                   llvm::ArrayRef<SunMMIOValue> values) {
  MixedIndexList result;
  result.static_values.reserve(values.size());
  SunmmioMlirType type(ctx);
  for (const SunMMIOValue &value : values) {
    mlir::Value mlir_value =
        value.value.empty() ? mlir::Value() : ctx.LookupMLIRValue(value.value);
    if (!mlir_value) {
      mlir_value = type.ResolveValueOrCreatePlaceholder(
          value, ctx.builder.getIndexType());
    }
    mlir_value = type.EnsureIndex(mlir_value);
    if (auto cst = mlir::getConstantIntValue(mlir_value)) {
      result.static_values.push_back(*cst);
    } else {
      result.static_values.push_back(mlir::ShapedType::kDynamic);
      result.dynamic_values.push_back(mlir_value);
    }
  }
  return result;
}

mlir::suvm::VCmpFPredicate GetTileCmpFloatPredicate(CompareOp op) {
  switch (op) {
  case CompareOp::kEQ:
    return mlir::suvm::VCmpFPredicate::eq;
  case CompareOp::kNE:
    return mlir::suvm::VCmpFPredicate::ne;
  case CompareOp::kLT:
    return mlir::suvm::VCmpFPredicate::lt;
  case CompareOp::kLE:
    return mlir::suvm::VCmpFPredicate::le;
  case CompareOp::kGT:
    return mlir::suvm::VCmpFPredicate::gt;
  case CompareOp::kGE:
    return mlir::suvm::VCmpFPredicate::ge;
  }
  LOG(FATAL) << "Unsupported tile float compare op";
  throw;
}

mlir::suvm::VCmpIPredicate GetTileCmpIntegerPredicate(CompareOp op,
                                                      CompareDomain domain) {
  if (op == CompareOp::kEQ) {
    return mlir::suvm::VCmpIPredicate::eq;
  }
  if (op == CompareOp::kNE) {
    return mlir::suvm::VCmpIPredicate::ne;
  }
  if (domain == CompareDomain::kUnsignedInt) {
    switch (op) {
    case CompareOp::kLT:
      return mlir::suvm::VCmpIPredicate::ult;
    case CompareOp::kLE:
      return mlir::suvm::VCmpIPredicate::ule;
    case CompareOp::kGT:
      return mlir::suvm::VCmpIPredicate::ugt;
    case CompareOp::kGE:
      return mlir::suvm::VCmpIPredicate::uge;
    default:
      break;
    }
  }
  switch (op) {
  case CompareOp::kLT:
    return mlir::suvm::VCmpIPredicate::slt;
  case CompareOp::kLE:
    return mlir::suvm::VCmpIPredicate::sle;
  case CompareOp::kGT:
    return mlir::suvm::VCmpIPredicate::sgt;
  case CompareOp::kGE:
    return mlir::suvm::VCmpIPredicate::sge;
  default:
    break;
  }
  LOG(FATAL) << "Unsupported tile integer compare op";
  throw;
}

} // namespace

SunmmioMlirTileOp::SunmmioMlirTileOp(SunmmioMlirContext &ctx)
    : ctx_(ctx), type_(ctx) {}

SunMMIOValue SunmmioMlirTileOp::GetPartitionedTileView(
    const std::string &result_name, const SunMMIOValue &memtensor,
    const std::vector<SunMMIOValue> &indices,
    const std::vector<int64_t> &tiled_dims, const SunMMIOType &view_type,
    DataType dtype) {
  mlir::Type result_type = MapMlirType(ctx_, view_type);
  mlir::Value view_value;
  bool can_emit_real_view = view_type.shape.size() == tiled_dims.size() &&
                            view_type.shape.size() <= 2 &&
                            memtensor.type.shape.size() >= tiled_dims.size();
  if (can_emit_real_view) {
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
    LOG(WARNING) << "Using provisional tile_view placeholder for unsupported "
                    "rank adaptation in clean v4 Tiles lowering";
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

SunMMIOValue SunmmioMlirTileOp::TileLoad(
    const std::string &result_name, const SunMMIOValue &tile_view,
    const SunMMIOType &tile_type, const std::optional<SunMMIOValue> &mask,
    const std::optional<SunMMIOValue> &maskedoff, DataType dtype) {
  mlir::Type result_type = MapMlirType(ctx_, tile_type);
  ICHECK(tile_type.shape.size() == 1 || tile_type.shape.size() == 2)
      << "suvm.tile.load supports rank-1 or rank-2 tiles";
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
  mlir::Value tile_value = mlir::suvm::TileLoadOp::create(
                               ctx_.builder, MapMlirLoc(ctx_), result_type,
                               base, mask_value, maskedoff_value)
                               .getResult();
  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, tile_value);
  }
  return SunMMIOValue{dtype, result_name, tile_type};
}

SunMMIOValue SunmmioMlirTileOp::TileFill(const std::string &result_name,
                                         const SunMMIOValue &scalar,
                                         const SunMMIOType &tile_type,
                                         DataType dtype) {
  mlir::Type result_type = MapMlirType(ctx_, tile_type);
  mlir::Value tile_value;
  mlir::Value scalar_value =
      ctx_.LookupOrCreateFakeValue(scalar, "fake_missing_tile_fill_scalar");
  mlir::OperationState st(MapMlirLoc(ctx_), "suvm.tile.fill");
  st.addOperands(scalar_value);
  st.addTypes(result_type);
  tile_value = ctx_.builder.create(st)->getResult(0);
  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, tile_value);
  }
  return SunMMIOValue{dtype, result_name, tile_type};
}

SunMMIOValue SunmmioMlirTileOp::TileRange(const std::string &result_name,
                                          const SunMMIOType &tile_type,
                                          DataType dtype) {
  ICHECK_EQ(tile_type.shape.size(), 1U)
      << "suvm.tile.range expects a rank-1 tile result";
  ICHECK(CanonicalizeSuvmDType(dtype).is_int() ||
         CanonicalizeSuvmDType(dtype).is_uint())
      << "suvm.tile.range expects an integer tile element type";
  mlir::Value tile_value =
      mlir::suvm::TileRangeOp::create(ctx_.builder, MapMlirLoc(ctx_),
                                      MapMlirType(ctx_, tile_type))
          .getResult();
  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, tile_value);
  }
  return SunMMIOValue{CanonicalizeSuvmDType(dtype).with_lanes(1), result_name,
                      tile_type};
}

SunMMIOValue SunmmioMlirTileOp::Cast(const std::string &result_name,
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

SunMMIOValue SunmmioMlirTileOp::Binary(const std::string &result_name,
                                       BinaryOp op, ArithmeticFlavor flavor,
                                       const SunMMIOValue &a,
                                       const SunMMIOValue &b,
                                       const SunMMIOType &result_type,
                                       DataType dtype) {
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
    if (flavor == ArithmeticFlavor::kFloat) {
      binary_value =
          mlir::suvm::TileAddFOp::create(ctx_.builder, loc, tile_type, lhs, rhs)
              .getResult();
    } else {
      binary_value =
          mlir::suvm::TileAddIOp::create(ctx_.builder, loc, tile_type, lhs, rhs)
              .getResult();
    }
    break;
  case BinaryOp::kSub:
    if (flavor == ArithmeticFlavor::kFloat) {
      binary_value =
          mlir::suvm::TileSubFOp::create(ctx_.builder, loc, tile_type, lhs, rhs)
              .getResult();
    } else {
      binary_value =
          mlir::suvm::TileSubIOp::create(ctx_.builder, loc, tile_type, lhs, rhs)
              .getResult();
    }
    break;
  case BinaryOp::kMul:
    ICHECK(flavor == ArithmeticFlavor::kFloat)
        << "SUVM tile integer multiply is not currently available";
    binary_value =
        mlir::suvm::TileMulFOp::create(ctx_.builder, loc, tile_type, lhs, rhs)
            .getResult();
    break;
  case BinaryOp::kDiv:
    ICHECK(flavor == ArithmeticFlavor::kFloat)
        << "SUVM tile integer division is not currently available";
    binary_value =
        mlir::suvm::TileDivFOp::create(ctx_.builder, loc, tile_type, lhs, rhs)
            .getResult();
    break;
  case BinaryOp::kMod:
    ICHECK(flavor == ArithmeticFlavor::kFloat)
        << "SUVM tile remainder is only available for floating-point tiles";
    binary_value =
        mlir::suvm::TileRemFOp::create(ctx_.builder, loc, tile_type, lhs, rhs)
            .getResult();
    break;
  case BinaryOp::kMin:
    ICHECK(flavor == ArithmeticFlavor::kFloat)
        << "SUVM tile min currently supports floating-point tiles only";
    binary_value =
        mlir::suvm::TileMinFOp::create(ctx_.builder, loc, tile_type, lhs, rhs)
            .getResult();
    break;
  case BinaryOp::kMax:
    ICHECK(flavor == ArithmeticFlavor::kFloat)
        << "SUVM tile max currently supports floating-point tiles only";
    binary_value =
        mlir::suvm::TileMaxFOp::create(ctx_.builder, loc, tile_type, lhs, rhs)
            .getResult();
    break;
  case BinaryOp::kAnd:
    ICHECK(flavor != ArithmeticFlavor::kFloat)
        << "SUVM tile.andi expects integer-like tiles";
    binary_value =
        mlir::suvm::TileAndIOp::create(ctx_.builder, loc, tile_type, lhs, rhs)
            .getResult();
    break;
  case BinaryOp::kOr:
    ICHECK(flavor != ArithmeticFlavor::kFloat)
        << "SUVM tile.ori expects integer-like tiles";
    binary_value =
        mlir::suvm::TileOrIOp::create(ctx_.builder, loc, tile_type, lhs, rhs)
            .getResult();
    break;
  case BinaryOp::kXor:
    ICHECK(flavor != ArithmeticFlavor::kFloat)
        << "SUVM tile.xori expects integer-like tiles";
    binary_value =
        mlir::suvm::TileXorIOp::create(ctx_.builder, loc, tile_type, lhs, rhs)
            .getResult();
    break;
  case BinaryOp::kShl:
    ICHECK(flavor != ArithmeticFlavor::kFloat)
        << "SUVM tile.shlli expects integer-like tiles";
    binary_value =
        mlir::suvm::TileShllIOp::create(ctx_.builder, loc, tile_type, lhs, rhs)
            .getResult();
    break;
  case BinaryOp::kShr:
    ICHECK(flavor != ArithmeticFlavor::kFloat)
        << "SUVM tile shift-right expects integer-like tiles";
    if (flavor == ArithmeticFlavor::kUnsignedInt) {
      binary_value = mlir::suvm::TileShrlIOp::create(ctx_.builder, loc,
                                                     tile_type, lhs, rhs)
                         .getResult();
    } else {
      binary_value = mlir::suvm::TileShraIOp::create(ctx_.builder, loc,
                                                     tile_type, lhs, rhs)
                         .getResult();
    }
    break;
  default:
    LOG(FATAL) << "Unsupported clean v4 tile binary op";
  }
  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, binary_value);
  }
  return SunMMIOValue{dtype, result_name, result_type};
}

SunMMIOValue SunmmioMlirTileOp::Unary(const std::string &result_name,
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
  case TileUnaryOp::kAbs:
    unary_value =
        mlir::suvm::TileAbsOp::create(ctx_.builder, loc, result_mlir_type,
                                      input, mlir::Value(), mlir::Value())
            .getResult();
    break;
  case TileUnaryOp::kCeil:
    unary_value =
        mlir::suvm::TileCeilOp::create(ctx_.builder, loc, result_mlir_type,
                                       input, mlir::Value(), mlir::Value())
            .getResult();
    break;
  case TileUnaryOp::kExp:
    unary_value =
        mlir::suvm::TileExpOp::create(ctx_.builder, loc, result_mlir_type,
                                      input, mlir::Value(), mlir::Value())
            .getResult();
    break;
  case TileUnaryOp::kFloor:
    unary_value =
        mlir::suvm::TileFloorOp::create(ctx_.builder, loc, result_mlir_type,
                                        input, mlir::Value(), mlir::Value())
            .getResult();
    break;
  case TileUnaryOp::kLn:
    unary_value =
        mlir::suvm::TileLnOp::create(ctx_.builder, loc, result_mlir_type, input,
                                     mlir::Value(), mlir::Value())
            .getResult();
    break;
  case TileUnaryOp::kNeg:
    unary_value =
        mlir::suvm::TileNegOp::create(ctx_.builder, loc, result_mlir_type,
                                      input, mlir::Value(), mlir::Value())
            .getResult();
    break;
  case TileUnaryOp::kRecip:
    unary_value =
        mlir::suvm::TileRecipOp::create(ctx_.builder, loc, result_mlir_type,
                                        input, mlir::Value(), mlir::Value())
            .getResult();
    break;
  case TileUnaryOp::kRound:
    unary_value =
        mlir::suvm::TileRoundOp::create(ctx_.builder, loc, result_mlir_type,
                                        input, mlir::Value(), mlir::Value())
            .getResult();
    break;
  case TileUnaryOp::kRsqrt:
    unary_value =
        mlir::suvm::TileRsqrtOp::create(ctx_.builder, loc, result_mlir_type,
                                        input, mlir::Value(), mlir::Value())
            .getResult();
    break;
  case TileUnaryOp::kTrunc:
    unary_value =
        mlir::suvm::TileTruncOp::create(ctx_.builder, loc, result_mlir_type,
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

SunMMIOValue SunmmioMlirTileOp::Compare(const std::string &result_name,
                                        CompareOp op, CompareDomain domain,
                                        const SunMMIOValue &a,
                                        const SunMMIOValue &b,
                                        const SunMMIOType &operand_type) {
  mlir::Value lhs =
      ctx_.LookupOrCreateFakeValue(a, "fake_missing_tile_compare_lhs");
  mlir::Value rhs =
      ctx_.LookupOrCreateFakeValue(b, "fake_missing_tile_compare_rhs");
  mlir::Location loc = MapMlirLoc(ctx_);

  SunMMIOType result_type{SunMMIOType::Kind::kTile, DataType::Bool(), 1,
                          operand_type.shape};
  mlir::Type result_mlir_type = MapMlirType(ctx_, result_type);
  auto tile_type = mlir::dyn_cast<mlir::suvm::TileType>(result_mlir_type);
  ICHECK(tile_type) << "Expected SUVM tile type for tile compare result";

  mlir::Value compare_value;
  if (domain == CompareDomain::kFloat) {
    compare_value =
        mlir::suvm::TileCmpFOp::create(ctx_.builder, loc, tile_type,
                                       GetTileCmpFloatPredicate(op), lhs, rhs)
            .getResult();
  } else {
    compare_value = mlir::suvm::TileCmpIOp::create(
                        ctx_.builder, loc, tile_type,
                        GetTileCmpIntegerPredicate(op, domain), lhs, rhs)
                        .getResult();
  }

  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, compare_value);
  }
  return SunMMIOValue{DataType::Bool(), result_name, result_type};
}

SunMMIOValue SunmmioMlirTileOp::TileUnsqueeze(const std::string &result_name,
                                              const SunMMIOValue &tile,
                                              const SunMMIOType &tile_type,
                                              int64_t axis, DataType dtype) {
  mlir::Type result_type = MapMlirType(ctx_, tile_type);
  mlir::Value input =
      ctx_.LookupOrCreateFakeValue(tile, "fake_missing_tile_unsqueeze_src");
  mlir::OperationState st(MapMlirLoc(ctx_), "suvm.tile.unsqueeze");
  st.addOperands(input);
  st.addAttribute("axes", ctx_.builder.getDenseI64ArrayAttr({axis}));
  st.addTypes(result_type);
  mlir::Value tile_value = ctx_.builder.create(st)->getResult(0);
  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, tile_value);
  }
  return SunMMIOValue{dtype, result_name, tile_type};
}

SunMMIOValue SunmmioMlirTileOp::TileBroadcast(const std::string &result_name,
                                              const SunMMIOValue &tile,
                                              const SunMMIOType &tile_type,
                                              DataType dtype) {
  mlir::Type result_type = MapMlirType(ctx_, tile_type);
  mlir::Value input =
      ctx_.LookupOrCreateFakeValue(tile, "fake_missing_tile_broadcast_src");
  mlir::Value tile_value =
      mlir::suvm::TileBroadcastOp::create(ctx_.builder, MapMlirLoc(ctx_),
                                          result_type, input)
          .getResult();
  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, tile_value);
  }
  return SunMMIOValue{dtype, result_name, tile_type};
}

SunMMIOValue
SunmmioMlirTileOp::TileSlice(const std::string &result_name,
                             const SunMMIOValue &tile,
                             const std::vector<SunMMIOValue> &offsets,
                             const SunMMIOType &tile_type, DataType dtype) {
  mlir::Type result_type = MapMlirType(ctx_, tile_type);
  mlir::Value input =
      ctx_.LookupOrCreateFakeValue(tile, "fake_missing_tile_slice_src");
  MixedIndexList mixed_offsets = BuildMixedIndexList(ctx_, offsets);
  llvm::SmallVector<int64_t, 4> static_sizes =
      StaticShapeVector(tile_type, "tile.extract_slice");
  mlir::Value tile_value =
      mlir::suvm::TileExtractSliceOp::create(
          ctx_.builder, MapMlirLoc(ctx_), result_type, input,
          mixed_offsets.dynamic_values, mlir::ValueRange{},
          ctx_.builder.getDenseI64ArrayAttr(mixed_offsets.static_values),
          ctx_.builder.getDenseI64ArrayAttr(static_sizes))
          .getResult();
  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, tile_value);
  }
  return SunMMIOValue{dtype, result_name, tile_type};
}

SunMMIOValue SunmmioMlirTileOp::TileInsertSlice(
    const std::string &result_name, const SunMMIOValue &base,
    const SunMMIOValue &slice, const std::vector<SunMMIOValue> &offsets,
    const SunMMIOType &result_type, DataType dtype) {
  mlir::Type result_mlir_type = MapMlirType(ctx_, result_type);
  mlir::Value base_value =
      ctx_.LookupOrCreateFakeValue(base, "fake_missing_tile_insert_slice_base");
  mlir::Value slice_value = ctx_.LookupOrCreateFakeValue(
      slice, "fake_missing_tile_insert_slice_slice");
  MixedIndexList mixed_offsets = BuildMixedIndexList(ctx_, offsets);
  llvm::SmallVector<int64_t, 4> static_sizes =
      StaticShapeVector(slice.type, "tile.insert_slice");
  mlir::Value tile_value =
      mlir::suvm::TileInsertSliceOp::create(
          ctx_.builder, MapMlirLoc(ctx_), result_mlir_type, slice_value,
          base_value, mixed_offsets.dynamic_values, mlir::ValueRange{},
          ctx_.builder.getDenseI64ArrayAttr(mixed_offsets.static_values),
          ctx_.builder.getDenseI64ArrayAttr(static_sizes))
          .getResult();
  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, tile_value);
  }
  return SunMMIOValue{dtype, result_name, result_type};
}

SunMMIOValue SunmmioMlirTileOp::TileAxisMask(const std::string &result_name,
                                             int64_t axis,
                                             const SunMMIOValue &valid_extent,
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
    LOG(FATAL) << "TileAxisMask expects integer/index valid extent input";
    TVM_FFI_UNREACHABLE();
  };

  int64_t rows_dim = extract_static_dim(0);
  int64_t cols_dim = extract_static_dim(1);
  ICHECK(axis == 0 || axis == 1) << "TileAxisMask expects axis 0 or 1";

  int64_t range_dim = axis == 0 ? rows_dim : cols_dim;
  SunMMIOType range_type = make_tile_type(DataType::Int(32), {range_dim});
  SunMMIOType range_2d_type =
      axis == 0 ? make_tile_type(DataType::Int(32), {rows_dim, 1})
                : make_tile_type(DataType::Int(32), {1, cols_dim});
  SunMMIOType range_full_type =
      make_tile_type(DataType::Int(32), {rows_dim, cols_dim});
  SunMMIOType mask_full_type =
      make_tile_type(DataType::Bool(), {rows_dim, cols_dim});

  mlir::Value range =
      mlir::suvm::TileRangeOp::create(ctx_.builder, MapMlirLoc(ctx_),
                                      MapMlirType(ctx_, range_type))
          .getResult();

  int64_t unsqueeze_axis = axis == 0 ? 1 : 0;
  mlir::OperationState unsqueeze_st(MapMlirLoc(ctx_), "suvm.tile.unsqueeze");
  unsqueeze_st.addOperands(range);
  unsqueeze_st.addAttribute(
      "axes", ctx_.builder.getDenseI64ArrayAttr({unsqueeze_axis}));
  unsqueeze_st.addTypes(MapMlirType(ctx_, range_2d_type));
  mlir::Value range_2d = ctx_.builder.create(unsqueeze_st)->getResult(0);

  mlir::Value valid_i32 = to_i32_scalar(
      valid_extent, axis == 0 ? "tail_mask_rows" : "tail_mask_cols");

  mlir::Value range_full = mlir::suvm::TileBroadcastOp::create(
                               ctx_.builder, MapMlirLoc(ctx_),
                               MapMlirType(ctx_, range_full_type), range_2d)
                               .getResult();

  mlir::Value mask_value =
      mlir::suvm::TileCmpIOp::create(
          ctx_.builder, MapMlirLoc(ctx_), MapMlirType(ctx_, mask_full_type),
          mlir::suvm::VCmpIPredicate::slt, range_full, valid_i32)
          .getResult();
  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, mask_value);
  }
  return SunMMIOValue{DataType::Bool(), result_name, tile_type};
}

SunMMIOValue SunmmioMlirTileOp::TileMaskAnd(const std::string &result_name,
                                            const SunMMIOValue &lhs,
                                            const SunMMIOValue &rhs,
                                            const SunMMIOType &tile_type) {
  mlir::Type result_type = MapMlirType(ctx_, tile_type);
  mlir::Value lhs_value =
      ctx_.LookupOrCreateFakeValue(lhs, "fake_missing_tile_mask_lhs");
  mlir::Value rhs_value =
      ctx_.LookupOrCreateFakeValue(rhs, "fake_missing_tile_mask_rhs");
  mlir::Value mask_value =
      mlir::suvm::TileAndIOp::create(ctx_.builder, MapMlirLoc(ctx_),
                                     result_type, lhs_value, rhs_value)
          .getResult();
  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, mask_value);
  }
  return SunMMIOValue{DataType::Bool(), result_name, tile_type};
}

SunMMIOValue SunmmioMlirTileOp::TileRectMask(const std::string &result_name,
                                             const SunMMIOValue &valid_rows,
                                             const SunMMIOValue &valid_cols,
                                             const SunMMIOType &tile_type) {
  ICHECK(!result_name.empty()) << "TileRectMask expects a result handle";
  SunMMIOValue row_mask =
      TileAxisMask(result_name + "_row", 0, valid_rows, tile_type);
  SunMMIOValue col_mask =
      TileAxisMask(result_name + "_col", 1, valid_cols, tile_type);
  return TileMaskAnd(result_name, row_mask, col_mask, tile_type);
}

SunMMIOValue SunmmioMlirTileOp::TileSelect(const std::string &result_name,
                                           const SunMMIOValue &mask,
                                           const SunMMIOValue &true_value,
                                           const SunMMIOValue &false_value,
                                           const SunMMIOType &result_type,
                                           DataType dtype) {
  mlir::Type result_mlir_type = MapMlirType(ctx_, result_type);
  mlir::Value mask_value =
      ctx_.LookupOrCreateFakeValue(mask, "fake_missing_tile_select_mask");
  mlir::Value true_mlir_value =
      ctx_.LookupOrCreateFakeValue(true_value, "fake_missing_tile_select_true");
  mlir::Value false_mlir_value = ctx_.LookupOrCreateFakeValue(
      false_value, "fake_missing_tile_select_false");
  mlir::Value select_value =
      mlir::suvm::TileSelectOp::create(ctx_.builder, MapMlirLoc(ctx_),
                                       result_mlir_type, mask_value,
                                       true_mlir_value, false_mlir_value)
          .getResult();
  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, select_value);
  }
  return SunMMIOValue{dtype, result_name, result_type};
}

SunMMIOValue SunmmioMlirTileOp::TileReduce(const std::string &result_name,
                                           const std::string &predicate,
                                           const SunMMIOValue &data,
                                           const SunMMIOType &result_type,
                                           int64_t axis, DataType dtype) {
  mlir::Value input =
      ctx_.LookupOrCreateFakeValue(data, "fake_missing_tile_reduce_data");
  mlir::Type result_mlir_type = MapMlirType(ctx_, result_type);
  auto tile_type = mlir::dyn_cast<mlir::suvm::TileType>(result_mlir_type);
  ICHECK(tile_type) << "Expected SUVM tile type for tile.reduce result";

  mlir::suvm::ReducePredicate reduce_predicate;
  if (predicate == "sum") {
    reduce_predicate = mlir::suvm::ReducePredicate::sum;
  } else if (predicate == "min") {
    reduce_predicate = mlir::suvm::ReducePredicate::min;
  } else if (predicate == "max") {
    reduce_predicate = mlir::suvm::ReducePredicate::max;
  } else {
    LOG(FATAL) << "Unsupported suvm.tile.reduce predicate: " << predicate;
  }

  mlir::Value reduce_value =
      mlir::suvm::TileReduceOp::create(ctx_.builder, MapMlirLoc(ctx_),
                                       result_mlir_type, reduce_predicate,
                                       input, static_cast<uint64_t>(axis))
          .getResult();
  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, reduce_value);
  }
  return SunMMIOValue{dtype, result_name, result_type};
}

SunMMIOValue SunmmioMlirTileOp::TileSqueeze(const std::string &result_name,
                                            const SunMMIOValue &tile,
                                            const SunMMIOType &tile_type,
                                            int64_t axis, DataType dtype) {
  mlir::Type result_type = MapMlirType(ctx_, tile_type);
  mlir::Value input =
      ctx_.LookupOrCreateFakeValue(tile, "fake_missing_tile_squeeze_src");
  mlir::OperationState st(MapMlirLoc(ctx_), "suvm.tile.squeeze");
  st.addOperands(input);
  st.addAttribute("axes", ctx_.builder.getDenseI64ArrayAttr({axis}));
  st.addTypes(result_type);
  mlir::Value tile_value = ctx_.builder.create(st)->getResult(0);
  if (!result_name.empty()) {
    ctx_.BindMLIRValue(result_name, tile_value);
  }
  return SunMMIOValue{dtype, result_name, tile_type};
}

void SunmmioMlirTileOp::TileStore(const SunMMIOValue &value,
                                  const SunMMIOValue &tile_view,
                                  const std::optional<SunMMIOValue> &mask) {
  mlir::Value base =
      ctx_.LookupOrCreateFakeValue(tile_view, "fake_missing_tile_store_view");
  bool fake_view_boundary =
      base.getDefiningOp() && base.getDefiningOp()->hasAttr("sunmmio.fake");
  ICHECK(value.type.shape.size() == 1 || value.type.shape.size() == 2)
      << "suvm.tile.store supports rank-1 or rank-2 tiles";
  ICHECK_EQ(value.type.shape.size(), tile_view.type.shape.size())
      << "suvm.tile.store expects data and tile_view ranks to match";
  if (fake_view_boundary) {
    LOG(WARNING)
        << "Using provisional tile.store placeholder in clean v4 Tiles "
           "lowering when the tile_view boundary is still fake";
    mlir::Value data =
        ctx_.LookupOrCreateFakeValue(value, "fake_missing_tile_store_value");
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
