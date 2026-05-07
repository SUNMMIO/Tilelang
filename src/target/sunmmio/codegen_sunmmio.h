#ifndef TVM_TL_TARGET_CODEGEN_SUNMMIO_H_
#define TVM_TL_TARGET_CODEGEN_SUNMMIO_H_

#include <tvm/ir/module.h>
#include <tvm/ir/type.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include "sunmmio_mlir_type.h"

#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace codegen {

class SunMMIOBuilder {
public:
  virtual ~SunMMIOBuilder() = default;

  virtual void Init() = 0;
  virtual void Clear() = 0;
  virtual std::string Finish() = 0;

  virtual void BeginModule() = 0;
  virtual void EndModule() = 0;

  virtual void BeginFunction(const std::string &name,
                             const std::vector<BuilderArg> &args) = 0;
  virtual void EndFunction() = 0;
  virtual void EmitReturn() = 0;

  virtual SunMMIOValue ConstantInt(const std::string &result_name, int64_t v,
                                   const SunMMIOType &type, DataType dtype) = 0;
  virtual SunMMIOValue ConstantFloat(const std::string &result_name,
                                     const std::string &literal,
                                     const SunMMIOType &type,
                                     DataType dtype) = 0;

  virtual SunMMIOValue Cast(const std::string &result_name,
                            const SunMMIOValue &v, const SunMMIOType &dst_type,
                            DataType dst_dtype) = 0;

  virtual SunMMIOValue Binary(const std::string &result_name, BinaryOp op,
                              ArithmeticFlavor flavor, const SunMMIOValue &a,
                              const SunMMIOValue &b,
                              const SunMMIOType &result_type,
                              DataType dtype) = 0;

  virtual SunMMIOValue Compare(const std::string &result_name, CompareOp op,
                               CompareDomain domain, const SunMMIOValue &a,
                               const SunMMIOValue &b,
                               const SunMMIOType &operand_type) = 0;

  virtual SunMMIOValue Select(const std::string &result_name,
                              const SunMMIOValue &cond, const SunMMIOValue &tv,
                              const SunMMIOValue &fv,
                              const SunMMIOType &result_type,
                              DataType dtype) = 0;

  virtual SunMMIOValue Alloc(const std::string &result_name,
                             const SunMMIOType &memref_type,
                             const std::vector<SunMMIOValue> &dyn_extents,
                             const std::string &scope_name, DataType dtype) = 0;

  virtual SunMMIOValue Load(const std::string &result_name,
                            const std::string &buffer_handle,
                            const std::vector<SunMMIOValue> &indices,
                            const SunMMIOType &memref_type, DataType dtype,
                            const SunMMIOType &result_type) = 0;

  virtual void Store(const SunMMIOValue &value,
                     const std::string &buffer_handle,
                     const std::vector<SunMMIOValue> &indices,
                     const SunMMIOType &memref_type) = 0;

  virtual SunMMIOValue Call(const std::string &result_name,
                            const std::string &callee,
                            const std::vector<SunMMIOValue> &operands,
                            const std::vector<std::string> &string_args,
                            const std::string &category, DataType ret_dtype,
                            const SunMMIOType &ret_type) = 0;

  virtual SunMMIOValue Ramp(const std::string &result_name,
                            const SunMMIOValue &base,
                            const SunMMIOValue &stride, int lanes,
                            const SunMMIOType &elem_type,
                            const SunMMIOType &vec_type, DataType dtype) = 0;

  virtual SunMMIOValue Broadcast(const std::string &result_name,
                                 const SunMMIOValue &scalar, int lanes,
                                 const SunMMIOType &scalar_type,
                                 const SunMMIOType &vec_type,
                                 DataType dtype) = 0;

  virtual void BeginFor(const std::string &iv, const SunMMIOValue &lb,
                        const SunMMIOValue &ub, const SunMMIOValue &step,
                        const ffi::Map<ffi::String, ffi::Any> &annotations) = 0;
  virtual void EndFor() = 0;

  virtual void BeginIf(const SunMMIOValue &cond) = 0;
  virtual void BeginElse() = 0;
  virtual void EndIf() = 0;

  virtual void EmitAssert(const SunMMIOValue &cond,
                          const std::string &msg_text) = 0;
};

class CodeGenTileLangSunMMIO final
    : public tir::StmtVisitor,
      public tir::ExprFunctor<SunMMIOValue(const tvm::PrimExpr &)> {
public:
  CodeGenTileLangSunMMIO();
  ~CodeGenTileLangSunMMIO() noexcept override = default;

  void Init();
  void Clear();
  void AddFunction(const GlobalVar &gvar, const tir::PrimFunc &f);
  std::string Finish();

protected:
  void VisitStmt_(const tir::SeqStmtNode *op) override;
  void VisitStmt_(const tir::ForNode *op) override;
  void VisitStmt_(const tir::LetStmtNode *op) override;
  void VisitStmt_(const tir::AttrStmtNode *op) override;
  void VisitStmt_(const tir::IfThenElseNode *op) override;
  void VisitStmt_(const tir::WhileNode *op) override;
  void VisitStmt_(const tir::AllocateNode *op) override;
  void VisitStmt_(const tir::AllocateConstNode *op) override;
  void VisitStmt_(const tir::DeclBufferNode *op) override;
  void VisitStmt_(const tir::BufferStoreNode *op) override;
  void VisitStmt_(const tir::BufferRealizeNode *op) override;
  void VisitStmt_(const tir::AssertStmtNode *op) override;
  void VisitStmt_(const tir::EvaluateNode *op) override;
  void VisitStmt_(const tir::BlockNode *op) override;
  void VisitStmt_(const tir::BlockRealizeNode *op) override;
  void VisitStmtDefault_(const Object *op) override;

  SunMMIOValue VisitExpr_(const tir::VarNode *op) override;
  SunMMIOValue VisitExpr_(const tir::SizeVarNode *op) override;
  SunMMIOValue VisitExpr_(const tir::IntImmNode *op) override;
  SunMMIOValue VisitExpr_(const tir::FloatImmNode *op) override;
  SunMMIOValue VisitExpr_(const tir::StringImmNode *op) override;
  SunMMIOValue VisitExpr_(const tir::CastNode *op) override;
  SunMMIOValue VisitExpr_(const tir::CallNode *op) override;
  SunMMIOValue VisitExpr_(const tir::AddNode *op) override;
  SunMMIOValue VisitExpr_(const tir::SubNode *op) override;
  SunMMIOValue VisitExpr_(const tir::MulNode *op) override;
  SunMMIOValue VisitExpr_(const tir::DivNode *op) override;
  SunMMIOValue VisitExpr_(const tir::ModNode *op) override;
  SunMMIOValue VisitExpr_(const tir::FloorDivNode *op) override;
  SunMMIOValue VisitExpr_(const tir::FloorModNode *op) override;
  SunMMIOValue VisitExpr_(const tir::MinNode *op) override;
  SunMMIOValue VisitExpr_(const tir::MaxNode *op) override;
  SunMMIOValue VisitExpr_(const tir::EQNode *op) override;
  SunMMIOValue VisitExpr_(const tir::NENode *op) override;
  SunMMIOValue VisitExpr_(const tir::LTNode *op) override;
  SunMMIOValue VisitExpr_(const tir::LENode *op) override;
  SunMMIOValue VisitExpr_(const tir::GTNode *op) override;
  SunMMIOValue VisitExpr_(const tir::GENode *op) override;
  SunMMIOValue VisitExpr_(const tir::AndNode *op) override;
  SunMMIOValue VisitExpr_(const tir::OrNode *op) override;
  SunMMIOValue VisitExpr_(const tir::NotNode *op) override;
  SunMMIOValue VisitExpr_(const tir::SelectNode *op) override;
  SunMMIOValue VisitExpr_(const tir::BufferLoadNode *op) override;
  SunMMIOValue VisitExpr_(const tir::ProducerLoadNode *op) override;
  SunMMIOValue VisitExpr_(const tir::RampNode *op) override;
  SunMMIOValue VisitExpr_(const tir::BroadcastNode *op) override;
  SunMMIOValue VisitExpr_(const tir::ShuffleNode *op) override;
  SunMMIOValue VisitExpr_(const tir::LetNode *op) override;
  SunMMIOValue VisitExprDefault_(const Object *op) override;

private:
  enum class CallBucket {
    kBuiltin,
    kExternPure,
    kExternSideEffect,
    kMath,
    kMemory,
    kSync,
    kVector,
    kTileLangIntrinsic,
    kSunMMIOIntrinsic,
    kUnsupported
  };

  struct ScopedAttr {
    ffi::Any node;
    ffi::String key;
    SunMMIOValue value;
  };

  SunMMIOValue EvalExpr(const tvm::PrimExpr &expr);
  void VisitStmtTracked(const tir::Stmt &stmt);
  void CollectExpectedCoverage(const tir::PrimFunc &f);
  void MarkVisitedNodeType(const std::string &type_key);
  void MarkVisitedCallOpFromExpr(const tvm::PrimExpr &expr);
  void WriteCoverageReport() const;
  void CheckCoverageOrFail() const;
  SunMMIOValue EmitBinary(const char *op_name, const tvm::PrimExpr &lhs,
                          const tvm::PrimExpr &rhs, tvm::DataType dtype);
  SunMMIOValue EmitCmp(const char *pred, const tvm::PrimExpr &lhs,
                       const tvm::PrimExpr &rhs);
  SunMMIOValue EmitCast(const SunMMIOValue &v, tvm::DataType target_dtype);
  SunMMIOValue EmitCall(const tir::CallNode *op);
  SunMMIOValue EmitLoad(const tir::Buffer &buffer,
                        const ffi::Array<PrimExpr> &indices);
  void EmitStore(const tir::Buffer &buffer, const ffi::Array<PrimExpr> &indices,
                 const SunMMIOValue &value);
  void EmitAlloc(const tir::Var &buffer_var, DataType dtype,
                 const ffi::Array<PrimExpr> &extents,
                 const std::string &scope_hint);
  void EmitFor(const tir::ForNode *op);
  void EmitIf(const tir::IfThenElseNode *op);

  SunMMIOType MapType(tvm::DataType dtype) const;
  SunMMIOType MapBufferType(const tir::Buffer &buffer) const;
  std::string MapStorageScope(const std::string &scope) const;
  std::string NewValueName();
  SunMMIOValue EmitConstIndex(int64_t v);
  SunMMIOValue EnsureIndex(const SunMMIOValue &v);
  SunMMIOValue EnsureType(const SunMMIOValue &v, const SunMMIOType &target_type,
                          DataType dtype);
  ArithmeticFlavor GetArithmeticFlavor(DataType dtype) const;
  CompareDomain GetCompareDomain(DataType dtype) const;
  SunMMIOValue BindVar(const tir::Var &var, const SunMMIOValue &value);
  void RegisterBuffer(const tir::Buffer &buffer, bool is_external,
                      const std::string &handle_hint = "");
  const BufferBinding &LookupBuffer(const tir::Buffer &buffer) const;
  void EnterScope();
  void ExitScope();

  CallBucket ClassifyCall(const tir::CallNode *op) const;
  const char *CallBucketName(CallBucket bucket) const;
  [[noreturn]] void UnsupportedStmt(const Object *op,
                                    const std::string &detail = "") const;
  [[noreturn]] void UnsupportedExpr(const Object *op,
                                    const std::string &detail = "") const;

  std::unique_ptr<SunMMIOBuilder> builder_;
  bool initialized_{false};
  int ssa_counter_{0};

  std::unordered_map<const tir::VarNode *, SunMMIOValue> var_table_;
  std::unordered_map<const tir::BufferNode *, BufferBinding> buffer_registry_;
  std::vector<ScopedAttr> attr_stack_;

  std::vector<const tir::VarNode *> scoped_vars_;
  std::vector<const tir::BufferNode *> scoped_buffers_;
  std::vector<size_t> var_scope_markers_;
  std::vector<size_t> buffer_scope_markers_;

  // Traversal coverage sets for codegen completeness checking.
  std::set<std::string> expected_node_types_;
  std::set<std::string> visited_node_types_;
  std::set<std::string> expected_call_ops_;
  std::set<std::string> visited_call_ops_;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_SUNMMIO_H_
