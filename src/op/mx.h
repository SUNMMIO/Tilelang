/*!
 * \file tl/op/mx.h
 * \brief MX physical pack/unpack tile operators.
 */

#ifndef TVM_TL_OP_MX_H_
#define TVM_TL_OP_MX_H_

#include "operator.h"

namespace tvm {
namespace tl {

using namespace tir;

class MXPackNode : public TileOperatorNode {
public:
  Buffer data;
  Buffer scale;
  Buffer mx;
  Array<Range> data_range;
  Array<Range> scale_range;
  Array<Range> mx_range;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.MXPack", MXPackNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MXPackNode>()
        .def_ro("data", &MXPackNode::data)
        .def_ro("scale", &MXPackNode::scale)
        .def_ro("mx", &MXPackNode::mx)
        .def_ro("data_range", &MXPackNode::data_range)
        .def_ro("scale_range", &MXPackNode::scale_range)
        .def_ro("mx_range", &MXPackNode::mx_range);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  TileOperator Clone() const override;
  static const Op &Get();
};

class MXPack : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(MXPack, TileOperator, MXPackNode);
  TVM_DLL MXPack(Array<PrimExpr> args,
                 Map<String, ObjectRef> annotations = Map<String, ObjectRef>());
  static const Op &Get();
};

class MXUnpackNode : public TileOperatorNode {
public:
  Buffer mx;
  Buffer data;
  Buffer scale;
  Array<Range> mx_range;
  Array<Range> data_range;
  Array<Range> scale_range;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.MXUnpack", MXUnpackNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MXUnpackNode>()
        .def_ro("mx", &MXUnpackNode::mx)
        .def_ro("data", &MXUnpackNode::data)
        .def_ro("scale", &MXUnpackNode::scale)
        .def_ro("mx_range", &MXUnpackNode::mx_range)
        .def_ro("data_range", &MXUnpackNode::data_range)
        .def_ro("scale_range", &MXUnpackNode::scale_range);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  TileOperator Clone() const override;
  static const Op &Get();
};

class MXUnpack : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(MXUnpack, TileOperator,
                                             MXUnpackNode);
  TVM_DLL
  MXUnpack(Array<PrimExpr> args,
           Map<String, ObjectRef> annotations = Map<String, ObjectRef>());
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_MX_H_
