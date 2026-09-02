/*!
 * \file tl/op/dist_comm.h
 * \brief Rank-level communication TIR operations.
 */

#ifndef TVM_TL_OP_DIST_COMM_H_
#define TVM_TL_OP_DIST_COMM_H_

#include "operator.h"

namespace tvm {
namespace tl {

TVM_DLL const Op &dist_signal_decl();
TVM_DLL const Op &dist_signal();
TVM_DLL const Op &dist_put_();
TVM_DLL const Op &dist_wait_signal_();
TVM_DLL const Op &dist_wait_all();
TVM_DLL const Op &dist_wait_send();
TVM_DLL const Op &dist_expect();
TVM_DLL const Op &dist_expect_();
TVM_DLL const Op &dist_route();
TVM_DLL const Op &dist_route_table();
TVM_DLL const Op &dist_peer_route();
TVM_DLL const Op &dist_peer_route_table();
TVM_DLL const Op &dist_rank_routed_put();
TVM_DLL const Op &dist_routed_put();

class DistPutOpNode : public TileOperatorNode {
public:
  Buffer src, dst;
  Array<Range> src_range, dst_range;
  PrimExpr dst_rank, dst_row, signal, current_core;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.DistPutOp", DistPutOpNode,
                                    TileOperatorNode);

  TileOperator Clone() const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DistPutOpNode>()
        .def_ro("src", &DistPutOpNode::src)
        .def_ro("dst", &DistPutOpNode::dst)
        .def_ro("src_range", &DistPutOpNode::src_range)
        .def_ro("dst_range", &DistPutOpNode::dst_range)
        .def_ro("dst_rank", &DistPutOpNode::dst_rank)
        .def_ro("dst_row", &DistPutOpNode::dst_row)
        .def_ro("signal", &DistPutOpNode::signal)
        .def_ro("current_core", &DistPutOpNode::current_core);
  }
};

class DistPeerPutOpNode : public TileOperatorNode {
public:
  Buffer src, dst;
  Array<Range> src_range, dst_range;
  PrimExpr dst_rank, signal, current_core;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.DistPeerPutOp", DistPeerPutOpNode,
                                    TileOperatorNode);

  TileOperator Clone() const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DistPeerPutOpNode>()
        .def_ro("src", &DistPeerPutOpNode::src)
        .def_ro("dst", &DistPeerPutOpNode::dst)
        .def_ro("src_range", &DistPeerPutOpNode::src_range)
        .def_ro("dst_range", &DistPeerPutOpNode::dst_range)
        .def_ro("dst_rank", &DistPeerPutOpNode::dst_rank)
        .def_ro("signal", &DistPeerPutOpNode::signal)
        .def_ro("current_core", &DistPeerPutOpNode::current_core);
  }
};

class DistRoutedPeerPutOpNode : public TileOperatorNode {
public:
  PrimExpr routes, signal, current_core;
  Array<Buffer> src, dst;
  Array<Array<Range>> src_ranges, dst_ranges;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.DistRoutedPeerPutOp",
                                    DistRoutedPeerPutOpNode, TileOperatorNode);

  TileOperator Clone() const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DistRoutedPeerPutOpNode>()
        .def_ro("routes", &DistRoutedPeerPutOpNode::routes)
        .def_ro("signal", &DistRoutedPeerPutOpNode::signal)
        .def_ro("current_core", &DistRoutedPeerPutOpNode::current_core)
        .def_ro("src", &DistRoutedPeerPutOpNode::src)
        .def_ro("dst", &DistRoutedPeerPutOpNode::dst)
        .def_ro("src_ranges", &DistRoutedPeerPutOpNode::src_ranges)
        .def_ro("dst_ranges", &DistRoutedPeerPutOpNode::dst_ranges);
  }
};

class DistRoutedPeerPutOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DistRoutedPeerPutOp, TileOperator,
                                             DistRoutedPeerPutOpNode);
  TVM_DLL DistRoutedPeerPutOp(Array<PrimExpr> args,
                              Map<String, ObjectRef> annotations = {});
  static const Op &Get();
};

class DistPeerPutOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DistPeerPutOp, TileOperator,
                                             DistPeerPutOpNode);
  TVM_DLL DistPeerPutOp(Array<PrimExpr> args,
                        Map<String, ObjectRef> annotations = {});
  static const Op &Get();
};

class DistPutOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DistPutOp, TileOperator,
                                             DistPutOpNode);
  TVM_DLL DistPutOp(Array<PrimExpr> args,
                    Map<String, ObjectRef> annotations = {});
  static const Op &Get();
};

class DistWaitSignalOpNode : public TileOperatorNode {
public:
  PrimExpr signal;
  Buffer dst;
  Array<Range> dst_range;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.DistWaitSignalOp", DistWaitSignalOpNode,
                                    TileOperatorNode);

  TileOperator Clone() const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DistWaitSignalOpNode>()
        .def_ro("signal", &DistWaitSignalOpNode::signal)
        .def_ro("dst", &DistWaitSignalOpNode::dst)
        .def_ro("dst_range", &DistWaitSignalOpNode::dst_range);
  }
};

class DistWaitSignalOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DistWaitSignalOp, TileOperator,
                                             DistWaitSignalOpNode);
  TVM_DLL DistWaitSignalOp(Array<PrimExpr> args,
                           Map<String, ObjectRef> annotations = {});
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_DIST_COMM_H_
