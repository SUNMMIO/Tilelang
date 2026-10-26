/*!
 * \file tl/op/utils.cc
 * \brief Common utilities implementation for TL ops.
 */

#include "utils.h"
#include "../layout/cute_layout.h"
#include "../target/sunmmio_utils.h"
#include "tvm/tir/expr.h"

#include <tvm/tir/builtin.h>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

Layout LookupTransferLayout(const LayoutInferArgs &args, const Buffer &buffer) {
  const LayoutMap &layouts =
      IsGlobalBuffer(buffer) ? args.global_layout_map : args.layout_map;
  return layouts.count(buffer) ? layouts[buffer] : Layout();
}

bool HasEstablishedTransferLayout(const LayoutInferArgs &args,
                                  const Buffer &buffer) {
  if (!LookupTransferLayout(args, buffer).defined())
    return false;
  return IsGlobalBuffer(buffer) || !args.layout_levels.count(buffer) ||
         args.layout_levels[buffer].IntValue() >
             static_cast<int>(InferLevel::kFree);
}

// Require the canonical three-level form, even when an N32 MXZZ carrier has
// an equivalent byte map. Layout constructors enforce pack-size constraints.
bool IsMXWeightLayout(const Layout &layout, const Buffer &buffer,
                      bool transposed_block, arith::Analyzer *analyzer) {
  const auto *cute = layout.as<CuteLayoutNode>();
  int rank = buffer->shape.size();
  if (!cute || !sunmmio::IsMXDType(buffer->dtype) || rank < 2)
    return false;
  auto levels = cute->GetDimLevels();
  if (levels.size() != static_cast<size_t>(rank) ||
      levels[rank - 2].IntValue() != 3 || levels[rank - 1].IntValue() != 3)
    return false;
  Array<Integer> axes{Integer(rank - 2), Integer(rank - 1)};
  Layout expected =
      transposed_block ? sunmmio::MakeMXZNN(buffer->shape, axes, buffer->dtype)
                       : sunmmio::MakeMXZNZ(buffer->shape, axes, buffer->dtype);
  return IsSameLayout(layout, expected, analyzer);
}

bool IsMXWeightTransfer(const LayoutInferArgs &args, const Buffer &src,
                        const Buffer &dst) {
  if (!sunmmio::IsMXDType(src->dtype) || !sunmmio::IsMXDType(dst->dtype))
    return false;
  if (HasEstablishedTransferLayout(args, dst)) {
    Layout layout = LookupTransferLayout(args, dst);
    if (dst.scope() == kSunmmioScopeWSRAM &&
        IsMXWeightLayout(layout, dst, true, args.analyzer))
      return true;
    if ((dst.scope() == kSunmmioScopeRSRAM || IsGlobalBuffer(dst)) &&
        IsMXWeightLayout(layout, dst, false, args.analyzer))
      return true;
  }
  return IsSunmmioSramScope(dst.scope()) &&
         HasEstablishedTransferLayout(args, src) &&
         IsMXWeightLayout(LookupTransferLayout(args, src), src, false,
                          args.analyzer);
}

Layout MakeMXWeightSourceLayout(const Buffer &buffer) {
  int rank = buffer->shape.size();
  ICHECK_GE(rank, 2) << "MXZNZ weight staging requires matrix buffers";
  return sunmmio::MakeMXZNZ(
      buffer->shape, {Integer(rank - 2), Integer(rank - 1)}, buffer->dtype);
}

} // namespace

std::optional<LayoutMap>
InferSunmmioMXWeightTransferLayout(const LayoutInferArgs &args,
                                   const Buffer &src, const Buffer &dst,
                                   InferLevel level) {
  if (level != InferLevel::kCommon || !IsMXWeightTransfer(args, src, dst))
    return std::nullopt;
  LayoutMap updates;
  // Established and immutable layouts are checked after propagation. Only
  // weak defaults may be replaced; never reinterpret an input or annotation.
  for (const Buffer &buffer : {src, dst}) {
    if (buffer.scope() == kSunmmioScopeRSRAM &&
        !HasEstablishedTransferLayout(args, buffer))
      updates.Set(buffer, MakeMXWeightSourceLayout(buffer));
  }
  return updates;
}

void ValidateSunmmioMXWeightTransferLayout(const LayoutInferArgs &args,
                                           const BufferRegion &src,
                                           const BufferRegion &dst,
                                           const char *op_name) {
  if (!IsMXWeightTransfer(args, src->buffer, dst->buffer))
    return;
  ICHECK_EQ(src->buffer->dtype, dst->buffer->dtype)
      << op_name << " MX weight transfer requires matching MX dtypes";
  ICHECK(dst->buffer.scope() != kSunmmioScopeASRAM)
      << "Sunmmio MX weight layout conflict in " << op_name << " ("
      << src->buffer->name << " -> " << dst->buffer->name
      << "): MXZNZ weight storage cannot also satisfy the ASRAM MXZZ "
         "activation requirement.";
  for (const BufferRegion &region : {src, dst}) {
    const Buffer &buffer = region->buffer;
    bool is_wsram = buffer.scope() == kSunmmioScopeWSRAM;
    const char *required = is_wsram ? "MXZNN" : "MXZNZ";
    Layout actual = LookupTransferLayout(args, buffer);
    ICHECK(IsMXWeightLayout(actual, buffer, is_wsram, args.analyzer))
        << "Sunmmio MX weight layout conflict in " << op_name << " ("
        << src->buffer->name << " -> " << dst->buffer->name << "): buffer \""
        << buffer->name << "\" requires " << required
        << " for the non-transposed weight path, but its actual layout is "
        << (actual.defined() ? actual->DebugOutput() : "undefined")
        << ". Declare MXZNZ input/staging layouts; implicit MXZZ <-> MXZNZ "
           "conversion is not supported.";
    int rank = region->region.size();
    ICHECK_GE(rank, 2);
    for (int dim = rank - 2; dim < rank; ++dim) {
      const Range &range = region->region[dim];
      ICHECK(args.analyzer->CanProveEqual(floormod(range->min, 32), 0) &&
             args.analyzer->CanProveEqual(floormod(range->extent, 32), 0))
          << op_name << " MX weight region for buffer \"" << buffer->name
          << "\" requires whole aligned 32x32 blocks at dimension " << dim
          << ", got " << range;
    }
  }
}

bool IsBufferLikeExpr(const PrimExpr &expr) {
  if (expr.as<BufferLoadNode>() || expr.as<BufferRegionNode>()) {
    return true;
  }
  if (const auto *call = expr.as<CallNode>()) {
    return (call->op.same_as(RegionOp::Get()));
  }
  return false;
}

BufferRegion NormalizeToBufferRegion(const PrimExpr &arg) {
  return NormalizeToBufferRegion(arg, RegionExprObserver{});
}

BufferRegion
NormalizeToBufferRegion(const PrimExpr &arg,
                        const RegionExprObserver &observe_consumed_root) {
  auto observe = [&](const PrimExpr &expr) {
    if (observe_consumed_root) {
      observe_consumed_root(expr);
    }
  };
  auto observe_index = [&](const PrimExpr &index) {
    observe(index);
    if (const auto *ramp = index.as<RampNode>()) {
      observe(ramp->stride);
      observe(ramp->lanes);
    }
  };

  // Case 1: Already a BufferRegion
  if (arg->IsInstance<BufferRegionNode>()) {
    observe(arg);
    return Downcast<BufferRegion>(arg);
  }

  // Case 2: BufferLoad — convert indices to ranges (Ramp -> lanes, else
  // extent=1)
  if (const auto *load = arg.as<BufferLoadNode>()) {
    observe(arg);
    Array<Range> ranges;
    for (const PrimExpr &index : load->indices) {
      observe_index(index);
      if (const auto *ramp = index.as<RampNode>()) {
        ICHECK(ramp->stride.as<IntImmNode>()) << "Ramp stride must be IntImm";
        ICHECK_EQ(ramp->stride.as<IntImmNode>()->value, 1)
            << "Only stride-1 Ramp is supported in region conversion";
        ICHECK(ramp->lanes.as<IntImmNode>())
            << "Scalable vector lanes not supported in region conversion";
        ranges.push_back(Range::FromMinExtent(ramp->base, ramp->lanes));
      } else {
        ranges.push_back(Range::FromMinExtent(index, 1));
      }
    }
    return BufferRegion(load->buffer, ranges);
  }

  // Case 3: tl.region(...) — reconstruct via RegionOp (bridge)
  if (const auto *call = arg.as<CallNode>()) {
    if (call->op.same_as(RegionOp::Get())) {
      observe(arg);
      for (const PrimExpr &call_arg : call->args) {
        observe(call_arg);
      }
      if (!call->args.empty()) {
        if (const auto *load = call->args[0].as<BufferLoadNode>()) {
          for (const PrimExpr &index : load->indices) {
            observe_index(index);
          }
        }
      }
      RegionOp region(call->args);
      return BufferRegion(region->GetBuffer(), region->GetRanges());
    }
    LOG(FATAL) << "Unsupported argument for BufferRegion (expect "
                  "BufferLoad/BufferRegion/tl.region): "
               << arg;
  }

  LOG(FATAL) << "Unsupported argument for BufferRegion: " << arg;
  throw; // Unreachable
}

/*!
 * \brief Encode a Buffer + Array<Range> into a tl.tileop.region Call
 * expression.
 *
 * This is the inverse of NormalizeToBufferRegion: it packs buffer region
 * metadata into a PrimExpr so it can travel through Call arguments (where
 * BufferRegion cannot appear directly).
 *
 * Use this when emitting intrinsic calls (e.g. tl.dma_copy) that need to
 * carry full region semantics — buffer identity, per-axis min/extent, and
 * access mode — as opaque PrimExpr arguments for later codegen consumption.
 *
 * Encoding layout:
 *   args[0] = BufferLoad(buffer, {range[0].min, range[1].min, ...})
 *   args[1] = access_mask  (1=read, 2=write, 3=read-write)
 *   args[2+i] = range[i].extent
 *
 * \param buffer      The buffer this region refers to.
 * \param ranges      Per-axis [min, extent) ranges describing the tile.
 * \param access_mask 1=read, 2=write, 3=read-write.
 * \return A Call(tl.tileop.region, ...) expression.
 */
PrimExpr MakeRegionExpr(const Buffer &buffer, const Array<Range> &ranges,
                        int access_mask) {
  // Build BufferLoad with indices = per-axis minima
  Array<PrimExpr> indices;
  for (const auto &r : ranges) {
    indices.push_back(r->min);
  }
  BufferLoad load(buffer, indices);

  // Pack args: [load, access_mask, extent_0, extent_1, ...]
  Array<PrimExpr> args;
  args.push_back(load);
  args.push_back(IntImm(DataType::Int(32), access_mask));
  for (const auto &r : ranges) {
    args.push_back(r->extent);
  }
  return Call(DataType::Handle(), RegionOp::Get(), args);
}

Array<Range> MakeCompactRegion(const Array<Range> &region) {
  Array<Range> compact;
  compact.reserve(region.size());
  for (const Range &range : region) {
    compact.push_back(Range::FromMinExtent(0, range->extent));
  }
  return compact;
}

Buffer MakeCompactBufferLike(const Buffer &prototype,
                             const Array<Range> &region,
                             const ffi::String &scope,
                             const ffi::String &name) {
  const auto *ptr_type = prototype->data->type_annotation.as<PointerTypeNode>();
  ICHECK(ptr_type != nullptr);

  Array<PrimExpr> shape;
  shape.reserve(region.size());
  for (const Range &range : region) {
    shape.push_back(range->extent);
  }

  ffi::String data_name = name.empty() ? prototype->data->name_hint : name;
  ffi::String buffer_name = name.empty() ? prototype->name : name;
  Var data(data_name, PointerType(ptr_type->element_type, scope));
  return Buffer(data, prototype->dtype, shape, {}, Integer(0), buffer_name,
                prototype->data_alignment, prototype->offset_factor,
                prototype->buffer_type);
}

PrimExpr MakeAccessPtrFromRegion(const BufferRegion &region, int rw_mask,
                                 bool require_2d) {
  Buffer buf = region->buffer;
  int ndim = static_cast<int>(buf->shape.size());
  if (require_2d) {
    ICHECK(ndim >= 2) << "Expect buffers with at least 2 dims";
  }

  PrimExpr offset, extent;
  if (ndim == 1) {
    // 1D: straightforward
    auto axis = region->region[0];
    offset = axis->min;
    extent = axis->extent;
  } else {
    // Compute row-major strides
    std::vector<PrimExpr> strides(ndim);
    PrimExpr one = make_const(buf->shape[0].dtype(), 1);
    PrimExpr cur = one;
    for (int i = ndim - 1; i >= 0; --i) {
      strides[i] = cur;
      cur = cur * buf->shape[i];
    }
    // Offset: sum_{i in [0..ndim-3]} min_i * stride_i
    offset = make_const(buf->shape[0].dtype(), 0);
    for (int i = 0; i < ndim - 2; ++i) {
      offset = offset + region->region[i]->min * strides[i];
    }
    // Extent: last two extents product (elements)
    extent =
        region->region[ndim - 2]->extent * region->region[ndim - 1]->extent;
  }

  // ptype and return handle
  PrimExpr ptype = tir::TypeAnnotation(buf->dtype);
  Array<PrimExpr> acc_args{ptype, buf->data, offset, extent,
                           IntImm(DataType::Int(32), rw_mask)};
  return Call(DataType::Handle(), builtin::tvm_access_ptr(), acc_args);
}

PrimExpr MakeAccessPtrFromBufferLoad(const BufferLoad &load, int rw_mask) {
  Buffer buf = load->buffer;
  int ndim = static_cast<int>(buf->shape.size());

  // Compute offset using row-major layout (iterate in reverse)
  PrimExpr offset = 0;
  PrimExpr stride = 1;

  for (int i = ndim - 1; i >= 0; --i) {
    const PrimExpr &index = load->indices[i];
    if (const auto *ramp = index.as<RampNode>()) {
      // For Ramp, use the base
      offset = offset + ramp->base * stride;
    } else {
      // For scalar index (IntImm or other PrimExpr)
      offset = offset + index * stride;
    }
    stride = stride * buf->shape[i];
  }

  // Extent is 1 element for a single BufferLoad access
  PrimExpr extent = make_const(DataType::Int(32), 1);

  // Build access_ptr
  PrimExpr ptype = tir::TypeAnnotation(buf->dtype);
  Array<PrimExpr> acc_args{ptype, buf->data, offset, extent,
                           IntImm(DataType::Int(32), rw_mask)};
  return Call(DataType::Handle(), builtin::tvm_access_ptr(), acc_args);
}

// Maps TVM DataType to CUDA's CUtensorMapDataType enum value.
int to_CUtensorMapDataType(DataType dtype) {
  CUtensorMapDataType tp;
  if (dtype.is_float()) {
    switch (dtype.bits()) {
    case 64:
      tp = CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
      break;
    case 32:
      tp = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
      break;
    case 16:
      tp = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
      break;
    case 8:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT8;
      break;
    default:
      ICHECK(0) << dtype;
    }
  } else if (dtype.is_bfloat16()) {
    tp = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  } else if (dtype.is_float8()) {
    tp = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else if (dtype.is_int()) {
    switch (dtype.bits()) {
    case 64:
      tp = CU_TENSOR_MAP_DATA_TYPE_INT64;
      break;
    case 32:
      tp = CU_TENSOR_MAP_DATA_TYPE_INT32;
      break;
    case 16:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT16;
      break;
    case 8:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT8;
      break;
    default:
      ICHECK(0) << dtype;
    }
  } else if (dtype.is_uint()) {
    switch (dtype.bits()) {
    case 64:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT64;
      break;
    case 32:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT32;
      break;
    case 16:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT16;
      break;
    case 8:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT8;
      break;
    default:
      ICHECK(0) << dtype;
    }
  } else {
    ICHECK(0) << dtype;
  }
  return static_cast<int>(tp);
}

} // namespace tl
} // namespace tvm
