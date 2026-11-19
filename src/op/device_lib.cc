/*!
 * \file tl/op/device_lib.cc
 * \brief Sunmmio device library tile operators.
 */
#include "../layout/cute_layout.h"
#include "../target/sunmmio_utils.h"
#include "../target/utils.h"
#include "builtin.h"
#include "operator.h"
#include "utils.h"
#include <tvm/tir/op_attr_types.h>

namespace tvm {
namespace tl {
using namespace tir;

class DeviceLibNode : public TileOperatorNode {
public:
  Array<PrimExpr> args;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.DeviceLib", DeviceLibNode,
                                    TileOperatorNode);
  static void RegisterReflection() {
    tvm::ffi::reflection::ObjectDef<DeviceLibNode>().def_ro(
        "args", &DeviceLibNode::args);
  }
  TileOperator Clone() const override;
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override {
    ICHECK(TargetIsSunmmio(T.target))
        << "Device library operations require Sunmmio";
    if (args[0].as<StringImmNode>()->value == "topk") {
      auto input = NormalizeToBufferRegion(args[1]);
      ICHECK_EQ(input->region.size(), 2U) << "T.topk requires rank-two regions";
      for (int i : {2, 3}) {
        auto output = NormalizeToBufferRegion(args[i]);
        ICHECK_EQ(output->region.size(), 2U)
            << "T.topk requires rank-two regions";
        ICHECK(analyzer->CanProveEqual(input->region[0]->extent,
                                       output->region[0]->extent))
            << "T.topk operands must have the same row count";
        if (const auto *k = args[4].as<IntImmNode>()) {
          ICHECK_GE(k->value, 0) << "T.topk k must be non-negative";
          ICHECK(analyzer->CanProve(args[4] <= output->region[1]->extent))
              << "T.topk output width must be at least k";
        }
      }
    }
    return Evaluate(Call(DataType::Handle(), device_lib(), args));
  }
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override {
    LayoutMap result;
    const auto *name = args[0].as<StringImmNode>();
    bool topk = name->value == "topk";
    int count = topk ? 3 : 2;
    int alignment = GetSunmmioTileProcessorConfig(T.target).rsram_align_bytes;
    for (int i = 1; i <= count; ++i) {
      Buffer buffer = NormalizeToBufferRegion(args[i])->buffer;
      // Top-k outputs are dense rows. Other operands can retain inferred ZZ
      // layouts; unconstrained operands start with aligned row-major storage.
      if (topk && i > 1) {
        result.Set(buffer, sunmmio::MakeAlignedRowMajor(
                               buffer->shape, buffer->dtype, alignment));
      } else if (level == InferLevel::kStrict) {
        continue;
      } else if (!T.layout_map.count(buffer) ||
                 (T.layout_levels.count(buffer) &&
                  T.layout_levels[buffer].IntValue() ==
                      static_cast<int>(InferLevel::kFree))) {
        result.Set(buffer, sunmmio::MakeAlignedRowMajor(
                               buffer->shape, buffer->dtype, alignment));
      } else {
        result.Set(buffer, T.layout_map[buffer]);
      }
    }
    if (!topk && level != InferLevel::kStrict) {
      Buffer input = NormalizeToBufferRegion(args[1])->buffer;
      Buffer output = NormalizeToBufferRegion(args[2])->buffer;
      if (!input.same_as(output)) {
        if (auto block = sunmmio::GetZZBlockShape(result[input])) {
          result.Set(
              output,
              sunmmio::MakeZZ(output->shape, {Integer(0), Integer(1)},
                              {Integer(block->height), Integer(block->width)}));
        } else {
          result.Set(output, sunmmio::MakeAlignedRowMajor(
                                 output->shape, output->dtype, alignment));
        }
      }
    }
    return result;
  }
};

class DeviceLib : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DeviceLib, TileOperator,
                                             DeviceLibNode);
  DeviceLib(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
    ICHECK(!args.empty() && args[0].as<StringImmNode>());
    const auto &name = args[0].as<StringImmNode>()->value;
    ICHECK(name == "topk" || name == "exp_fast" || name == "ln_fast" ||
           name == "recip_fast" || name == "rsqrt_fast")
        << "Unknown device library function: " << name;
    ICHECK_EQ(args.size(), name == "topk" ? 5U : 3U);
    auto node = tvm::ffi::make_object<DeviceLibNode>();
    node->args = args;
    data_ = std::move(node);
  }
};
TileOperator DeviceLibNode::Clone() const {
  return DeviceLib(tvm::ffi::make_object<DeviceLibNode>(*this));
}
TVM_REGISTER_OP("tl.tileop.device_lib")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "_device_lib")
    .set_attr<OpBuilderFunc>("TLOpBuilder",
                             [](Array<PrimExpr> args,
                                Map<String, ObjectRef> annotations) {
                               return DeviceLib(args, annotations);
                             })
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TVM_FFI_STATIC_INIT_BLOCK() { DeviceLibNode::RegisterReflection(); }
} // namespace tl
} // namespace tvm
