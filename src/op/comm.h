/*!
 * \file tl/op/comm.h
 * \brief Implementation of Inter-core Communication Operators
 */

#ifndef TVM_TL_OP_COMM_H_
#define TVM_TL_OP_COMM_H_

#include "operator.h"

namespace tvm {
namespace tl {

TVM_DLL const Op &CoreId();
TVM_DLL const Op &comm_current_core();
TVM_DLL const Op &comm_is_current_core();
TVM_DLL const Op &comm_barrier();
TVM_DLL const Op &comm_fence();
TVM_DLL const Op &broadcast_();

using namespace tir;

class BroadcastOpNode : public TileOperatorNode {
public:
  Buffer src, dst;
  Array<Range> src_range, dst_range;
  PrimExpr src_expr, dst_expr;
  IntImm size;
  IntImm dst_offset;
  IntImm src_core;
  int direction;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.comm_broadcast", BroadcastOpNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BroadcastOpNode>()
        .def_ro("src", &BroadcastOpNode::src)
        .def_ro("dst", &BroadcastOpNode::dst)
        .def_ro("src_range", &BroadcastOpNode::src_range)
        .def_ro("dst_range", &BroadcastOpNode::dst_range)
        .def_ro("src_core", &BroadcastOpNode::src_core)
        .def_ro("direction", &BroadcastOpNode::direction)
        .def_ro("size", &BroadcastOpNode::size)
        .def_ro("dst_offset", &BroadcastOpNode::dst_offset);
  }

  TileOperator Clone() const;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
};

class BroadcastOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(BroadcastOp, TileOperator,
                                             BroadcastOpNode);
  TVM_DLL BroadcastOp(Array<PrimExpr> args);
  static const Op &Get();
};

class PutOpNode : public TileOperatorNode {
public:
  Buffer src, dst;
  Array<Range> src_range, dst_range;
  PrimExpr src_expr, dst_expr;
  IntImm src_core, dst_core;
  IntImm size;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.comm_put", PutOpNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PutOpNode>()
        .def_ro("src", &PutOpNode::src)
        .def_ro("dst", &PutOpNode::dst)
        .def_ro("src_range", &PutOpNode::src_range)
        .def_ro("dst_range", &PutOpNode::dst_range)
        .def_ro("src_core", &PutOpNode::src_core)
        .def_ro("dst_core", &PutOpNode::dst_core)
        .def_ro("size", &PutOpNode::size);
  }

  TileOperator Clone() const;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
};

class PutOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PutOp, TileOperator, PutOpNode);
  TVM_DLL PutOp(Array<PrimExpr> args);
  static const Op &Get();
};

class AllgatherOpNode : public TileOperatorNode {
public:
  PrimExpr send, recv;
  int direction;
  IntImm size;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.comm_allgather", AllgatherOpNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AllgatherOpNode>()
        .def_ro("send", &AllgatherOpNode::send)
        .def_ro("recv", &AllgatherOpNode::recv)
        .def_ro("direction", &AllgatherOpNode::direction)
        .def_ro("size", &AllgatherOpNode::size);
  }

  TileOperator Clone() const;
  LayoutMap ComputeLayout(const LayoutInferArgs &T, InferLevel level,
                          Buffer src, Buffer dst) const;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
};

class AllgatherOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(AllgatherOp, TileOperator,
                                             AllgatherOpNode);
  TVM_DLL AllgatherOp(Array<PrimExpr> args);
  static const Op &Get();
};

class AllreduceOpNode : public TileOperatorNode {
public:
  PrimExpr src, dst;
  PrimExpr row_allgather, col_allgather;
  PrimExpr dst_copy;
  StringImm type;
  int direction;
  IntImm dim;
  IntImm clear;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.comm_allreduce", AllreduceOpNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AllreduceOpNode>()
        .def_ro("src", &AllreduceOpNode::src)
        .def_ro("dst", &AllreduceOpNode::dst)
        .def_ro("row_allgather", &AllreduceOpNode::row_allgather)
        .def_ro("col_allgather", &AllreduceOpNode::col_allgather)
        .def_ro("type", &AllreduceOpNode::type)
        .def_ro("dim", &AllreduceOpNode::dim)
        .def_ro("clear", &AllreduceOpNode::clear)
        .def_ro("direction", &AllreduceOpNode::direction)
        .def_ro("dst_copy", &AllreduceOpNode::dst_copy);
  }

  TileOperator Clone() const;
  LayoutMap ComputeLayout(const LayoutInferArgs &T, InferLevel level,
                          Buffer src, Buffer dst, int dim) const;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
};

class AllreduceOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(AllreduceOp, TileOperator,
                                             AllreduceOpNode);
  TVM_DLL AllreduceOp(Array<PrimExpr> args);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_COMM_H_
