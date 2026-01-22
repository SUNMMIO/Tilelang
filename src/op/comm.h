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

class BroadcastNode : public TileOperatorNode {
public:
  Buffer src, dst;
  Array<Range> src_range, dst_range;
  PrimExpr src_expr, dst_expr;
  IntImm size;
  IntImm dst_offset;
  IntImm src_core;
  int direction;
  // Array<IntImm> group;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.comm_broadcast", BroadcastNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BroadcastNode>()
        .def_ro("src", &BroadcastNode::src)
        .def_ro("dst", &BroadcastNode::dst)
        .def_ro("src_range", &BroadcastNode::src_range)
        .def_ro("dst_range", &BroadcastNode::dst_range)
        .def_ro("src_core", &BroadcastNode::src_core)
        .def_ro("direction", &BroadcastNode::direction)
        .def_ro("size", &BroadcastNode::size)
        .def_ro("dst_offset", &BroadcastNode::dst_offset);
    // .def_ro("group", &BroadcastNode::group);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  TileOperator Clone() const;
};

class Broadcast : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Broadcast, TileOperator,
                                             BroadcastNode);
  TVM_DLL Broadcast(Array<PrimExpr> args);
  static const Op &Get();
};

class PutNode : public TileOperatorNode {
public:
  Buffer src, dst;
  Array<Range> src_range, dst_range;
  PrimExpr src_expr, dst_expr;
  IntImm src_core, dst_core;
  IntImm size;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.comm_put", PutNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PutNode>()
        .def_ro("src", &PutNode::src)
        .def_ro("dst", &PutNode::dst)
        .def_ro("src_range", &PutNode::src_range)
        .def_ro("dst_range", &PutNode::dst_range)
        .def_ro("src_core", &PutNode::src_core)
        .def_ro("dst_core", &PutNode::dst_core)
        .def_ro("size", &PutNode::size);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  TileOperator Clone() const;
};

class Put : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Put, TileOperator, PutNode);
  TVM_DLL Put(Array<PrimExpr> args);
  static const Op &Get();
};

class AllgatherNode : public TileOperatorNode {
public:
  PrimExpr send, recv;
  int direction;
  IntImm size;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.comm_allgather", AllgatherNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AllgatherNode>()
        .def_ro("send", &AllgatherNode::send)
        .def_ro("recv", &AllgatherNode::recv)
        .def_ro("direction", &AllgatherNode::direction)
        .def_ro("size", &AllgatherNode::size);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  Layout ComputeLinearLayout(const Buffer &shared_tensor) const;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  TileOperator Clone() const;
};

class Allgather : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Allgather, TileOperator,
                                             AllgatherNode);
  TVM_DLL Allgather(Array<PrimExpr> args);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_COMM_H_
