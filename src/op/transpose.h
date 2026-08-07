/*!
 * \file tl/op/transpose.h
 * \brief Sunmmio RSRAM matrix transpose operator.
 */

#ifndef TVM_TL_OP_TRANSPOSE_H_
#define TVM_TL_OP_TRANSPOSE_H_

#include "operator.h"

namespace tvm {
namespace tl {

using namespace tir;

class TransposeNode : public TileOperatorNode {
public:
  Buffer src, dst;
  Array<Range> src_range, dst_range;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.Transpose", TransposeNode,
                                    TileOperatorNode);

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  TileOperator Clone() const override;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TransposeNode>()
        .def_ro("src", &TransposeNode::src)
        .def_ro("dst", &TransposeNode::dst)
        .def_ro("src_range", &TransposeNode::src_range)
        .def_ro("dst_range", &TransposeNode::dst_range);
  }
};

class Transpose : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Transpose, TileOperator,
                                             TransposeNode);
  TVM_DLL Transpose(Array<PrimExpr> args,
                    Map<String, ObjectRef> annotations = {});
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_TRANSPOSE_H_
