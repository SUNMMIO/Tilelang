/*!
 * \file tl/op/dist_collective.h
 * \brief High-level Rank collective TIR operations.
 */

#ifndef TVM_TL_OP_DIST_COLLECTIVE_H_
#define TVM_TL_OP_DIST_COLLECTIVE_H_

#include "operator.h"

namespace tvm {
namespace tl {

class DistAllgatherOpNode : public TileOperatorNode {
public:
  Buffer src, dst;
  Array<Range> src_range, dst_range;
  int axis;
  PrimExpr signal, current_core;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.DistAllgatherOp", DistAllgatherOpNode,
                                    TileOperatorNode);

  TileOperator Clone() const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DistAllgatherOpNode>()
        .def_ro("src", &DistAllgatherOpNode::src)
        .def_ro("dst", &DistAllgatherOpNode::dst)
        .def_ro("src_range", &DistAllgatherOpNode::src_range)
        .def_ro("dst_range", &DistAllgatherOpNode::dst_range)
        .def_ro("axis", &DistAllgatherOpNode::axis)
        .def_ro("signal", &DistAllgatherOpNode::signal)
        .def_ro("current_core", &DistAllgatherOpNode::current_core);
  }
};

class DistAllgatherOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DistAllgatherOp, TileOperator,
                                             DistAllgatherOpNode);
  TVM_DLL DistAllgatherOp(Array<PrimExpr> args,
                          Map<String, ObjectRef> annotations = {});
  static const Op &Get();
};

class DistAlltoallOpNode : public TileOperatorNode {
public:
  Buffer src, dst;
  Array<Range> src_range, dst_range;
  ffi::String domain;
  PrimExpr signal, current_core;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.DistAlltoallOp", DistAlltoallOpNode,
                                    TileOperatorNode);

  TileOperator Clone() const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DistAlltoallOpNode>()
        .def_ro("src", &DistAlltoallOpNode::src)
        .def_ro("dst", &DistAlltoallOpNode::dst)
        .def_ro("src_range", &DistAlltoallOpNode::src_range)
        .def_ro("dst_range", &DistAlltoallOpNode::dst_range)
        .def_ro("domain", &DistAlltoallOpNode::domain)
        .def_ro("signal", &DistAlltoallOpNode::signal)
        .def_ro("current_core", &DistAlltoallOpNode::current_core);
  }
};

class DistAlltoallOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DistAlltoallOp, TileOperator,
                                             DistAlltoallOpNode);
  TVM_DLL DistAlltoallOp(Array<PrimExpr> args,
                         Map<String, ObjectRef> annotations = {});
  static const Op &Get();
};

class DistAllreduceOpNode : public TileOperatorNode {
public:
  Buffer src, dst;
  Array<Range> src_range, dst_range;
  ffi::String reduce_type;
  PrimExpr signal, current_core;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.DistAllreduceOp", DistAllreduceOpNode,
                                    TileOperatorNode);

  TileOperator Clone() const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DistAllreduceOpNode>()
        .def_ro("src", &DistAllreduceOpNode::src)
        .def_ro("dst", &DistAllreduceOpNode::dst)
        .def_ro("src_range", &DistAllreduceOpNode::src_range)
        .def_ro("dst_range", &DistAllreduceOpNode::dst_range)
        .def_ro("reduce_type", &DistAllreduceOpNode::reduce_type)
        .def_ro("signal", &DistAllreduceOpNode::signal)
        .def_ro("current_core", &DistAllreduceOpNode::current_core);
  }
};

class DistAllreduceOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DistAllreduceOp, TileOperator,
                                             DistAllreduceOpNode);
  TVM_DLL DistAllreduceOp(Array<PrimExpr> args,
                          Map<String, ObjectRef> annotations = {});
  static const Op &Get();
};

class DistAlltoallvOpNode : public TileOperatorNode {
public:
  Buffer src, dst, send_counts, recv_counts;
  Array<Range> src_range, dst_range, send_counts_range, recv_counts_range;
  ffi::String domain;
  PrimExpr signal_resource, current_core;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.DistAlltoallvOp", DistAlltoallvOpNode,
                                    TileOperatorNode);

  TileOperator Clone() const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DistAlltoallvOpNode>()
        .def_ro("src", &DistAlltoallvOpNode::src)
        .def_ro("dst", &DistAlltoallvOpNode::dst)
        .def_ro("send_counts", &DistAlltoallvOpNode::send_counts)
        .def_ro("recv_counts", &DistAlltoallvOpNode::recv_counts)
        .def_ro("src_range", &DistAlltoallvOpNode::src_range)
        .def_ro("dst_range", &DistAlltoallvOpNode::dst_range)
        .def_ro("send_counts_range", &DistAlltoallvOpNode::send_counts_range)
        .def_ro("recv_counts_range", &DistAlltoallvOpNode::recv_counts_range)
        .def_ro("domain", &DistAlltoallvOpNode::domain)
        .def_ro("signal_resource", &DistAlltoallvOpNode::signal_resource)
        .def_ro("current_core", &DistAlltoallvOpNode::current_core);
  }
};

class DistAlltoallvOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DistAlltoallvOp, TileOperator,
                                             DistAlltoallvOpNode);
  TVM_DLL DistAlltoallvOp(Array<PrimExpr> args,
                          Map<String, ObjectRef> annotations = {});
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_DIST_COLLECTIVE_H_
