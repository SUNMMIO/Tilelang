/*!
 * \file tl/op/transpose.cc
 * \brief Lower a TileLang matrix transpose to the Sunmmio ODMA intrinsic.
 */

#include "transpose.h"

#include "../layout/cute_layout.h"
#include "../target/sunmmio_utils.h"
#include "../target/utils.h"
#include "builtin.h"
#include "utils.h"

#include <tvm/tir/op_attr_types.h>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

void ValidateShapeAndType(const Buffer &src, const Buffer &dst,
                          arith::Analyzer *analyzer) {
  ICHECK_EQ(src->shape.size(), 2U)
      << "T.transpose requires a rank-2 source buffer, got "
      << src->shape.size() << " for " << src->name;
  ICHECK_EQ(dst->shape.size(), 2U)
      << "T.transpose requires a rank-2 destination buffer, got "
      << dst->shape.size() << " for " << dst->name;
  ICHECK(src->dtype == dst->dtype)
      << "T.transpose requires matching element types, got " << src->dtype
      << " and " << dst->dtype;
  ICHECK(src->dtype.is_bfloat16() ||
         (src->dtype.is_float() && src->dtype.bits() == 32))
      << "Sunmmio T.transpose supports only bfloat16 and float32, got "
      << src->dtype;
  ICHECK(analyzer->CanProveEqual(src->shape[0], dst->shape[1]) &&
         analyzer->CanProveEqual(src->shape[1], dst->shape[0]))
      << "T.transpose destination shape must be the transpose of source: src="
      << src->shape << ", dst=" << dst->shape;
  for (size_t i = 0; i < 2; ++i) {
    const auto *src_extent = src->shape[i].as<IntImmNode>();
    const auto *dst_extent = dst->shape[i].as<IntImmNode>();
    ICHECK(src_extent && dst_extent)
        << "Sunmmio T.transpose currently requires static shapes, got src="
        << src->shape << ", dst=" << dst->shape;
    ICHECK_EQ(src_extent->value % 32, 0)
        << "Sunmmio T.transpose requires each source dimension to be a "
           "multiple of the A4E 32x32 block shape, got src="
        << src->shape;
    ICHECK_EQ(dst_extent->value % 32, 0)
        << "Sunmmio T.transpose requires each destination dimension to be a "
           "multiple of the A4E 32x32 block shape, got dst="
        << dst->shape;
  }
}

Optional<Layout> CanonicalTransposeLayout(const Layout &src_layout,
                                          const Buffer &src, const Buffer &dst,
                                          const Target &target,
                                          arith::Analyzer *analyzer) {
  Array<Integer> axes{Integer(0), Integer(1)};
  Array<PrimExpr> block_shape = GetSunmmioLayoutBlockShape(target, src->dtype);
  Layout src_zz = sunmmio::MakeZZ(src->shape, axes, block_shape);
  if (IsSameLayout(src_layout, src_zz, analyzer))
    return sunmmio::MakeZZ(dst->shape, axes, block_shape);

  Layout src_zn = sunmmio::MakeZN(src->shape, axes, block_shape);
  if (IsSameLayout(src_layout, src_zn, analyzer))
    return sunmmio::MakeZN(dst->shape, axes, block_shape);

  return std::nullopt;
}

void ValidateFullRegion(const Buffer &buffer, const Array<Range> &region,
                        const char *operand, arith::Analyzer *analyzer) {
  ICHECK_EQ(region.size(), 2U)
      << "T.transpose " << operand << " region must be rank-2";
  for (size_t i = 0; i < 2; ++i) {
    ICHECK(analyzer->CanProveEqual(region[i]->min, 0) &&
           analyzer->CanProveEqual(region[i]->extent, buffer->shape[i]))
        << "Sunmmio T.transpose requires the " << operand
        << " region to cover the complete buffer; dim " << i
        << " has min=" << region[i]->min << ", extent=" << region[i]->extent
        << ", buffer extent=" << buffer->shape[i];
  }
}

} // namespace

Transpose::Transpose(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  (void)annotations;
  ICHECK_EQ(args.size(), 2U)
      << "T.transpose expects source and destination regions";
  ObjectPtr<TransposeNode> node = tvm::ffi::make_object<TransposeNode>();
  BufferRegion src_region = NormalizeToBufferRegion(args[0]);
  BufferRegion dst_region = NormalizeToBufferRegion(args[1]);
  node->src = src_region->buffer;
  node->dst = dst_region->buffer;
  node->src_range = src_region->region;
  node->dst_range = dst_region->region;
  data_ = std::move(node);
}

TileOperator TransposeNode::Clone() const {
  return Transpose(tvm::ffi::make_object<TransposeNode>(*this));
}

LayoutMap TransposeNode::InferLayout(const LayoutInferArgs &T,
                                     InferLevel level) const {
  (void)level;
  ICHECK(TargetIsSunmmio(T.target))
      << "T.transpose is currently supported only on the Sunmmio target";
  ValidateShapeAndType(src, dst, T.analyzer);
  ICHECK_EQ(src.scope(), kSunmmioScopeRSRAM)
      << "Sunmmio T.transpose source must use shared.rsram, got "
      << src.scope();
  ICHECK_EQ(dst.scope(), kSunmmioScopeRSRAM)
      << "Sunmmio T.transpose destination must use shared.rsram, got "
      << dst.scope();
  ICHECK(T.layout_map.count(src))
      << "Sunmmio T.transpose source has no inferred layout";

  Optional<Layout> expected = CanonicalTransposeLayout(
      T.layout_map[src], src, dst, T.target, T.analyzer);
  ICHECK(expected.defined())
      << "Sunmmio T.transpose source layout must be a two-level 32x32 ZZ or "
         "ZN layout";

  LayoutMap updates;
  updates.Set(dst, expected.value());
  return updates;
}

Stmt TransposeNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  ICHECK(TargetIsSunmmio(T.target))
      << "T.transpose is currently supported only on the Sunmmio target";
  ValidateShapeAndType(src, dst, analyzer);
  ICHECK_EQ(src.scope(), kSunmmioScopeRSRAM)
      << "Sunmmio T.transpose source must use shared.rsram, got "
      << src.scope();
  ICHECK_EQ(dst.scope(), kSunmmioScopeRSRAM)
      << "Sunmmio T.transpose destination must use shared.rsram, got "
      << dst.scope();
  ICHECK(!src->data.same_as(dst->data))
      << "Sunmmio T.transpose does not support in-place operation";
  ValidateFullRegion(src, src_range, "source", analyzer);
  ValidateFullRegion(dst, dst_range, "destination", analyzer);

  ICHECK(T.layout_map.count(src) && T.layout_map.count(dst))
      << "Sunmmio T.transpose requires inferred source and destination layouts";
  Optional<Layout> expected =
      CanonicalTransposeLayout(T.layout_map[src], src, dst, T.target, analyzer);
  ICHECK(expected.defined())
      << "Sunmmio T.transpose source layout must be a two-level 32x32 ZZ or "
         "ZN layout";
  ICHECK(IsSameLayout(T.layout_map[dst], expected.value(), analyzer))
      << "Sunmmio T.transpose destination layout is not the canonical "
         "transpose of the source layout";

  return Evaluate(Call(DataType::Handle(), sunmmio_transpose(),
                       {MakeRegionExpr(src, src_range, /*access_mask=*/1),
                        MakeRegionExpr(dst, dst_range, /*access_mask=*/2)}));
}

TIR_REGISTER_TL_TILE_OP(Transpose, transpose)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() { TransposeNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
