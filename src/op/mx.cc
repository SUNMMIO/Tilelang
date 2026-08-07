/*!
 * \file tl/op/mx.cc
 * \brief MX physical pack/unpack tile operators.
 */

#include "mx.h"

#include "../layout/cute_layout.h"
#include "../target/sunmmio_utils.h"
#include "../target/utils.h"
#include "builtin.h"
#include "utils.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tir/op_attr_types.h>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

DataType ExpectedMXDataDType(DataType mx_dtype) {
  ICHECK(sunmmio::IsMXDType(mx_dtype))
      << "MX pack/unpack expects mxfp8 or mxfp4 buffer dtype, got " << mx_dtype;
  if (mx_dtype.bits() == 8) {
    return DataType::Float8E4M3FN();
  }
  if (mx_dtype.bits() == 4) {
    return DataType::Float4E2M1FN();
  }
  LOG(FATAL) << "Unsupported MX dtype " << mx_dtype;
  TVM_FFI_UNREACHABLE();
}

DataType ExpectedMXScaleDType() { return DataType::Float8E8M0FNU(); }

void CheckRank2(const Buffer &buffer, const char *name) {
  ICHECK(buffer.defined()) << name << " buffer is undefined";
  ICHECK_EQ(buffer->shape.size(), 2U)
      << "T.mx_pack/T.mx_unpack only support static rank-2 buffers; " << name
      << " has rank " << buffer->shape.size();
}

void CheckStaticShape(const Buffer &buffer, const char *name) {
  for (const PrimExpr &dim : buffer->shape) {
    ICHECK(dim.as<IntImmNode>())
        << "T.mx_pack/T.mx_unpack only support static rank-2 shapes; " << name
        << " has dynamic dimension " << dim;
  }
}

void CheckRsram(const Buffer &buffer, const char *name) {
  ICHECK(buffer.scope() == kSunmmioScopeRSRAM)
      << "T.mx_pack/T.mx_unpack require " << name
      << " buffer in shared.rsram, got scope `" << buffer.scope() << "`";
}

void CheckSameShape(const Array<PrimExpr> &lhs, const Array<PrimExpr> &rhs,
                    const char *lhs_name, const char *rhs_name,
                    arith::Analyzer *analyzer) {
  ICHECK_EQ(lhs.size(), rhs.size())
      << lhs_name << " rank must match " << rhs_name << " rank";
  for (size_t i = 0; i < lhs.size(); ++i) {
    ICHECK(analyzer->CanProveEqual(lhs[i], rhs[i]))
        << lhs_name << ".shape[" << i << "] must equal " << rhs_name
        << ".shape[" << i << "], got " << lhs[i] << " vs " << rhs[i];
  }
}

void CheckFullRegion(const Buffer &buffer, const Array<Range> &ranges,
                     const char *name, arith::Analyzer *analyzer) {
  ICHECK_EQ(ranges.size(), buffer->shape.size())
      << name << " region rank must match buffer rank";
  for (size_t i = 0; i < ranges.size(); ++i) {
    PrimExpr zero = make_zero(ranges[i]->min.dtype());
    ICHECK(analyzer->CanProveEqual(ranges[i]->min, zero))
        << "T.mx_pack/T.mx_unpack only support full buffer regions; " << name
        << " region dim " << i << " has non-zero min " << ranges[i]->min;
    ICHECK(analyzer->CanProveEqual(ranges[i]->extent, buffer->shape[i]))
        << "T.mx_pack/T.mx_unpack only support full buffer regions; " << name
        << " region dim " << i << " extent " << ranges[i]->extent
        << " does not match buffer extent " << buffer->shape[i];
  }
}

Layout LookupOrDefaultMXLayout(const LayoutInferArgs &T, const Buffer &mx) {
  if (T.layout_map.count(mx)) {
    return T.layout_map[mx];
  }
  Array<Integer> axes{Integer(0), Integer(1)};
  return sunmmio::MakeMXZZ(mx->shape, axes, mx->dtype);
}

sunmmio::MXLayoutAnalysis
ValidateCommon(const Buffer &data, const Array<Range> &data_range,
               const Buffer &scale, const Array<Range> &scale_range,
               const Buffer &mx, const Array<Range> &mx_range,
               const Layout &mx_layout, arith::Analyzer *analyzer) {
  CheckRank2(data, "data");
  CheckRank2(scale, "scale");
  CheckRank2(mx, "mx");
  CheckStaticShape(data, "data");
  CheckStaticShape(scale, "scale");
  CheckStaticShape(mx, "mx");
  CheckRsram(data, "data");
  CheckRsram(scale, "scale");
  CheckRsram(mx, "mx");
  CheckFullRegion(data, data_range, "data", analyzer);
  CheckFullRegion(scale, scale_range, "scale", analyzer);
  CheckFullRegion(mx, mx_range, "mx", analyzer);

  ICHECK(sunmmio::IsMXDType(mx->dtype))
      << "mx buffer dtype must be mxfp8 or mxfp4, got " << mx->dtype;
  ICHECK(data->dtype == ExpectedMXDataDType(mx->dtype))
      << "data dtype must be " << ExpectedMXDataDType(mx->dtype)
      << " for mx dtype " << mx->dtype << ", got " << data->dtype;
  ICHECK(scale->dtype == ExpectedMXScaleDType())
      << "scale dtype must be " << ExpectedMXScaleDType() << ", got "
      << scale->dtype;
  CheckSameShape(data->shape, mx->shape, "data", "mx", analyzer);

  ICHECK(mx_layout.defined())
      << "T.mx_pack/T.mx_unpack require mx buffer to have an MX layout";
  auto analysis = sunmmio::AnalyzeMXLayout(mx_layout, mx->dtype, analyzer);
  ICHECK(analysis.has_value())
      << "T.mx_pack/T.mx_unpack support only MX row-major, MXZZ, and MXZNZ "
         "layouts for the mx buffer";
  ICHECK(analysis->kind != sunmmio::MXLayoutKind::kMXZNN)
      << "T.mx_pack/T.mx_unpack do not accept MXZNN as a user mx buffer "
         "layout; use MXZNZ for RSRAM data staged before WSRAM MXZNN";
  CheckSameShape(scale->shape, analysis->scale_shape, "scale",
                 "required MX scale", analyzer);
  return *analysis;
}

Layout ExpectedDataLayout(const Buffer &data,
                          const sunmmio::MXLayoutAnalysis &analysis,
                          int align_bytes) {
  Array<Integer> axes{Integer(0), Integer(1)};
  switch (analysis.kind) {
  case sunmmio::MXLayoutKind::kRowMajor:
    return sunmmio::MakeAlignedRowMajor(data->shape, data->dtype, align_bytes);
  case sunmmio::MXLayoutKind::kMXZZ:
  case sunmmio::MXLayoutKind::kMXZNZ:
    return sunmmio::MakeZZ(data->shape, axes,
                           Array<PrimExpr>{IntImm(DataType::Int(32), 32),
                                           IntImm(DataType::Int(32), 32)});
  case sunmmio::MXLayoutKind::kMXZNN:
    LOG(FATAL) << "MXZNN is an internal WSRAM layout and is not accepted by "
                  "T.mx_pack/T.mx_unpack";
  }
  LOG(FATAL) << "Unsupported MX layout kind";
  TVM_FFI_UNREACHABLE();
}

LayoutMap InferCommonLayout(const Buffer &data, const Array<Range> &data_range,
                            const Buffer &scale,
                            const Array<Range> &scale_range, const Buffer &mx,
                            const Array<Range> &mx_range,
                            const LayoutInferArgs &T) {
  Layout mx_layout = LookupOrDefaultMXLayout(T, mx);
  sunmmio::MXLayoutAnalysis analysis =
      ValidateCommon(data, data_range, scale, scale_range, mx, mx_range,
                     mx_layout, T.analyzer);

  int align_bytes = GetSunmmioTileProcessorConfig(T.target).rsram_align_bytes;
  LayoutMap out;
  out.Set(mx, mx_layout);
  out.Set(data, ExpectedDataLayout(data, analysis, align_bytes));
  out.Set(scale, sunmmio::MakeAlignedRowMajor(scale->shape, scale->dtype,
                                              align_bytes));
  return out;
}

} // namespace

MXPack::MXPack(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  (void)annotations;
  ICHECK_EQ(args.size(), 3U) << "T.mx_pack expects data, scale, and mx";

  BufferRegion data_region = NormalizeToBufferRegion(args[0]);
  BufferRegion scale_region = NormalizeToBufferRegion(args[1]);
  BufferRegion mx_region = NormalizeToBufferRegion(args[2]);

  ObjectPtr<MXPackNode> node = tvm::ffi::make_object<MXPackNode>();
  node->data = data_region->buffer;
  node->scale = scale_region->buffer;
  node->mx = mx_region->buffer;
  node->data_range = data_region->region;
  node->scale_range = scale_region->region;
  node->mx_range = mx_region->region;
  data_ = std::move(node);
}

TileOperator MXPackNode::Clone() const {
  auto op = tvm::ffi::make_object<MXPackNode>(*this);
  return MXPack(op);
}

Stmt MXPackNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  ICHECK(TargetIsSunmmio(T.target))
      << "T.mx_pack currently only supports SunMMIO targets";
  Layout mx_layout = T.layout_map.count(mx) ? T.layout_map[mx] : Layout();
  (void)ValidateCommon(data, data_range, scale, scale_range, mx, mx_range,
                       mx_layout, analyzer);
  return Evaluate(Call(DataType::Handle(), mx_pack(),
                       {MakeRegionExpr(data, data_range, /*access_mask=*/1),
                        MakeRegionExpr(scale, scale_range, /*access_mask=*/1),
                        MakeRegionExpr(mx, mx_range, /*access_mask=*/2)}));
}

LayoutMap MXPackNode::InferLayout(const LayoutInferArgs &T,
                                  InferLevel level) const {
  (void)level;
  return InferCommonLayout(data, data_range, scale, scale_range, mx, mx_range,
                           T);
}

MXUnpack::MXUnpack(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  (void)annotations;
  ICHECK_EQ(args.size(), 3U) << "T.mx_unpack expects mx, data, and scale";

  BufferRegion mx_region = NormalizeToBufferRegion(args[0]);
  BufferRegion data_region = NormalizeToBufferRegion(args[1]);
  BufferRegion scale_region = NormalizeToBufferRegion(args[2]);

  ObjectPtr<MXUnpackNode> node = tvm::ffi::make_object<MXUnpackNode>();
  node->mx = mx_region->buffer;
  node->data = data_region->buffer;
  node->scale = scale_region->buffer;
  node->mx_range = mx_region->region;
  node->data_range = data_region->region;
  node->scale_range = scale_region->region;
  data_ = std::move(node);
}

TileOperator MXUnpackNode::Clone() const {
  auto op = tvm::ffi::make_object<MXUnpackNode>(*this);
  return MXUnpack(op);
}

Stmt MXUnpackNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  ICHECK(TargetIsSunmmio(T.target))
      << "T.mx_unpack currently only supports SunMMIO targets";
  Layout mx_layout = T.layout_map.count(mx) ? T.layout_map[mx] : Layout();
  (void)ValidateCommon(data, data_range, scale, scale_range, mx, mx_range,
                       mx_layout, analyzer);
  return Evaluate(Call(DataType::Handle(), mx_unpack(),
                       {MakeRegionExpr(mx, mx_range, /*access_mask=*/1),
                        MakeRegionExpr(data, data_range, /*access_mask=*/2),
                        MakeRegionExpr(scale, scale_range,
                                       /*access_mask=*/2)}));
}

LayoutMap MXUnpackNode::InferLayout(const LayoutInferArgs &T,
                                    InferLevel level) const {
  (void)level;
  return InferCommonLayout(data, data_range, scale, scale_range, mx, mx_range,
                           T);
}

TIR_REGISTER_TL_TILE_OP(MXPack, mx_pack)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_TILE_OP(MXUnpack, mx_unpack)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  MXPackNode::RegisterReflection();
  MXUnpackNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
